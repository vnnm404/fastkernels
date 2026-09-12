"""Execute the fastkernels validate CLI and verify its artifacts; no inference code."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def terminate_tree(pid):
    """Workers create new sessions, so killing only the parent group is insufficient."""
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    children = {}
    for row in rows.splitlines():
        child, parent = map(int, row.split())
        children.setdefault(parent, []).append(child)
    descendants = []
    def visit(parent):
        for child in children.get(parent, []):
            visit(child)
            descendants.append(child)
    visit(pid)
    for target in descendants + [pid]:
        try:
            os.kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_command(command, log, timeout, env=None, disk_root=None, minimum_free_bytes=0):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    heartbeat = started
    with log.open('w') as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            while True:
                now = time.monotonic()
                if now - heartbeat >= 60:
                    print(f"Still running ({now-started:.0f}s); log: {log}", flush=True)
                    heartbeat = now
                if disk_root and shutil.disk_usage(disk_root).free < minimum_free_bytes:
                    terminate_tree(proc.pid)
                    proc.wait()
                    return {'status': 'disk-full', 'elapsed_s': time.monotonic() - started}
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    rc = proc.wait(timeout=min(2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except subprocess.TimeoutExpired:
            terminate_tree(proc.pid)
            proc.wait()
            return {'status': 'timeout', 'elapsed_s': time.monotonic() - started}
        except BaseException:
            terminate_tree(proc.pid)
            proc.wait()
            raise
    return {'status': 'ok' if rc == 0 else 'failed', 'returncode': rc,
            'elapsed_s': time.monotonic() - started}


def cli_root(output):
    # The CLI keys compiler caches by output directory basename. Give each
    # validation a unique one, including across changed audit namespaces.
    output = Path(output)
    return output / ('cli-' + hashlib.sha256(str(output).encode()).hexdigest()[:20])


def command_for(args, model, workload, reference_python, output, frozen):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = output / 'scenario.yaml'
    workloads = [workload]
    if not args.skip_latency:
        workloads += ['single-request', 'fixed-batch-32']
    # JSON is also valid YAML. No YAML dependency is needed in the audit process.
    table.write_text(json.dumps({'scenarios': [{
        'model': model, 'tp': args.tp, 'dtype': 'bfloat16',
        'legacy_workloads': workloads, 'enforce_eager': args.enforce_eager,
    }]}, indent=2))
    command = [str(args.fk_python.parent / 'fastkernels'), 'validate', str(table),
               '--text-scenario', workload,
               '--vllm-python', str(reference_python),
               '--max-requests', str(args.num_seqs), '--seed', str(args.seed),
               '--latency-iters', str(args.latency_iters),
               '--reference-patches', args.reference_patches,
               '--timeout', str(max(1, int(args.run_timeout_s))),
               '--output-dir', str(cli_root(output)),
               '--inputs-json' if frozen.exists() else '--save-inputs-json', str(frozen)]
    if args.skip_latency:
        command.append('--skip-latency')
    if args.max_layers:
        command += ['--max-layers', str(args.max_layers)]
    return command


def collect_cli_results(output):
    """Require a successful CLI job, retaining originals and copying raw results."""
    output = Path(output)
    summaries = list(output.glob('cli-*/summary.json'))
    if len(summaries) != 1:
        raise ValueError('validate CLI summary is missing or ambiguous')
    cli = summaries[0].parent
    summary = json.loads((cli / 'summary.json').read_text())
    models = summary.get('models', [])
    if (summary.get('run', {}).get('status') != 'PASS' or len(models) != 1 or
            models[0].get('status') != 'PASS' or models[0].get('harness') != 'bench_vllm'):
        raise ValueError('validate CLI did not report exactly one passing vLLM validation')
    # Locate locally rather than relying on absolute paths in archived summaries.
    results = list(cli.glob('*/results.json'))
    if len(results) != 1:
        raise ValueError('validate CLI results are missing or ambiguous')
    for name in ('results.json', 'vllm_raw.json', 'fastkernels_raw.json'):
        shutil.copyfile(results[0].parent / name, output / name)


def validation_files(output):
    """Include CLI success evidence in completion stamps, not just copied results."""
    output = Path(output)
    paths = [output / n for n in ('results.json', 'vllm_raw.json', 'fastkernels_raw.json', 'scenario.yaml')]
    for cli in output.glob('cli-*'):
        paths += list(cli.rglob('*.json'))
        paths += list(cli.rglob('run.log'))
    return [str(p.relative_to(output)) for p in sorted(paths)]


def read_validation(output, frozen=None):
    output = Path(output)
    result = json.loads((output / 'results.json').read_text())
    # These are the unmodified raw outputs saved by the validation harness.
    for name in ('vllm', 'fastkernels'):
        envelope = json.loads((output / f'{name}_raw.json').read_text())
        result[name + '_raw'] = envelope['raw']
    if frozen is not None:
        inputs = json.loads(Path(frozen).read_text())
        if result['input_sha256'] != inputs['sha256']:
            raise ValueError('Validation result does not match the frozen input checksum')
        for engine in ('vllm', 'fastkernels'):
            actual = result[engine + '_raw']['throughput']
            expected = inputs['payload']['throughput']
            if len(actual) != len(expected):
                raise ValueError('Validation omitted a frozen scenario')
            for a, e in zip(actual, expected):
                if 'prompt_token_ids' not in e:
                    if a['name'] != e['name'] or not a['outputs']:
                        raise ValueError('Validation omitted frozen media requests')
                    if any(len(o['token_ids']) != e['output_len'] for o in a['outputs']):
                        raise ValueError('Validation did not honor media output budgets')
                    continue
                if a['name'] != e['name'] or len(a['outputs']) != len(e['prompt_token_ids']):
                    raise ValueError('Validation omitted frozen requests')
                if [len(o['token_ids']) for o in a['outputs']] != e['output_lens']:
                    raise ValueError('Validation did not honor frozen output budgets')
        expected_latency = inputs['payload'].get('latency', [])
        combined_latency = result.get('latency_scenarios', [])
        if [s['scenario'] for s in combined_latency] != [s['name'] for s in expected_latency]:
            raise ValueError('Validation omitted frozen latency scenarios')
        for actual, expected in zip(combined_latency, expected_latency):
            for engine in ('vllm', 'fastkernels'):
                values = actual[engine + '_latencies']
                if len(values) != expected['num_iters']:
                    raise ValueError('Validation omitted latency iterations')
    return result
