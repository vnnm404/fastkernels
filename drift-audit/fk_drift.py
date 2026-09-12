"""Audit a fixed FastKernels checkout against two isolated production references.

Invokes the actual fastkernels validate CLI (including Ray dispatch). Environments
must be provisioned explicitly; this script never upgrades the fixed FK install.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import fcntl
from drift_runner import command_for, read_validation, run_command, collect_cli_results, validation_files

SCHEMA = 3
HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def geomean(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Ratios must be finite and positive')
    return math.exp(statistics.mean(map(math.log, values)))


def interval(values):
    """Descriptive paired-repeat bootstrap; not an architecture-population CI."""
    if len(values) < 3:
        return None
    rng = random.Random(42)
    samples = sorted(geomean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [samples[49], samples[1949]]


def alignment(a, b):
    if not a or len(a) != len(b):
        raise ValueError('Missing or unequal output request counts')
    prefixes = []
    for x, y in zip(a, b):
        x, y = x['token_ids'], y['token_ids']
        if not x or not y or len(x) != len(y):
            raise ValueError('Missing or unequal output token budgets')
        count = 0
        for u, v in zip(x, y):
            if u != v:
                break
            count += 1
        prefixes.append(count)
    return statistics.mean(prefixes)


def compare(old, new, workload, floor):
    if old['input_sha256'] != new['input_sha256']:
        raise ValueError('Reference runs consumed different inputs')
    def scenario(result, engine):
        return next(s for s in result[engine + '_raw']['throughput'] if s['name'] == workload)
    ob, nb = scenario(old, 'vllm'), scenario(new, 'vllm')
    of, nf = scenario(old, 'fastkernels'), scenario(new, 'fastkernels')
    # Check both FK-to-reference relationships, cross-version output agreement,
    # and repeatability of the unchanged FK implementation.
    aligns = {'old_fk_reference': alignment(of['outputs'], ob['outputs']),
              'new_fk_reference': alignment(nf['outputs'], nb['outputs']),
              'old_new_reference': alignment(ob['outputs'], nb['outputs']),
              'fixed_fk': alignment(of['outputs'], nf['outputs'])}
    outputs = (ob, nb, of, nf)
    counts = [s['total_output_tokens'] for s in outputs]
    if len(set(counts)) != 1 or counts[0] <= 0:
        raise ValueError('Engines performed unequal output work')
    for s in outputs:
        if s['total_output_tokens'] != sum(len(o['token_ids']) for o in s['outputs']):
            raise ValueError('Reported token count differs from raw output')
        if not math.isfinite(s['elapsed']) or s['elapsed'] <= 0:
            raise ValueError('Invalid elapsed time')
    rates = [s['total_output_tokens'] / s['elapsed'] for s in outputs]
    old_ref, new_ref, old_fk, new_fk = rates
    result = {'throughput_drift': new_ref / old_ref,
              'fk_vs_old': old_fk / old_ref, 'fk_vs_new': new_fk / new_ref,
              'fk_stability': new_fk / old_fk, 'alignment': aligns,
              'valid': all(v >= floor for v in aligns.values()),
              'requests': len(ob['outputs']), 'latency': {}}
    old_lat = {s['scenario']: s for s in old.get('latency_scenarios', [])}
    new_lat = {s['scenario']: s for s in new.get('latency_scenarios', [])}
    if old_lat.keys() != new_lat.keys():
        raise ValueError('Latency scenario coverage differs')
    for name, a in old_lat.items():
        b = new_lat[name]
        vals = [a['vllm_median_s'], b['vllm_median_s'], a['fastkernels_median_s'], b['fastkernels_median_s']]
        geomean(vals)  # validate
        result['latency'][name] = {'drift': vals[0] / vals[1],
                                   'fk_vs_old': vals[0] / vals[2],
                                   'fk_vs_new': vals[1] / vals[3]}
    return result


def aggregate(comparisons, expected, args):
    groups = defaultdict(list)
    for row in comparisons:
        if row['valid']:
            groups[(row['family'], row['model'], row['workload'])].append(row)
    cells, families = [], defaultdict(list)
    issues = []
    if len(comparisons) != expected or any(not r['valid'] for r in comparisons):
        issues.append('Missing, failed, or alignment-excluded comparisons')
    for (family, model, workload), rows in sorted(groups.items()):
        ratios = [r['throughput_drift'] for r in rows]
        cell = {'family': family, 'model': model, 'workload': workload,
                'repeats': len(rows), 'throughput_drift': geomean(ratios),
                'throughput_ci95': interval(ratios)}
        for metric in ('fk_vs_old', 'fk_vs_new', 'fk_stability'):
            cell[metric] = geomean([r[metric] for r in rows])
        cell['latency'] = {}
        for name in rows[0]['latency']:
            vals = [r['latency'][name]['drift'] for r in rows]
            cell['latency'][name] = {'drift': geomean(vals), 'ci95': interval(vals)}
        if len(rows) < 3:
            issues.append('Fewer than three paired repeats')
        if min(r['requests'] for r in rows) < {'mixed': 1000, 'long-context': 64}[workload]:
            issues.append('Reduced request count (smoke test)')
        if not 1 / args.parity_tolerance <= cell['fk_vs_old'] <= args.parity_tolerance:
            issues.append('Fixed FastKernels is not within the declared old-reference parity band')
        if any(not 1 / args.stability_tolerance <= r['fk_stability'] <= args.stability_tolerance for r in rows):
            issues.append('Fixed FastKernels performance changed across reference runs; investigate noise')
        cells.append(cell)
        families[family].append(cell)
    family_results = {}
    for family, items in families.items():
        family_results[family] = {
            'throughput_drift': geomean([c['throughput_drift'] for c in items]),
            'fk_vs_old': geomean([c['fk_vs_old'] for c in items]),
            'fk_vs_new': geomean([c['fk_vs_new'] for c in items]),
            'cells': len(items),
        }
    def outside(x):
        return x > args.threshold or x < 1 / args.threshold
    signals = []
    uncertain = False
    for cell in cells:
        for name, point, ci in [('throughput', cell['throughput_drift'], cell['throughput_ci95'])] + [
                (f'latency:{name}', value['drift'], value['ci95']) for name, value in cell['latency'].items()]:
            if ci is None or not (ci[0] > args.threshold or ci[1] < 1 / args.threshold or
                                  (ci[0] >= 1 / args.threshold and ci[1] <= args.threshold)):
                uncertain = True
            if outside(point):
                signals.append({'family': cell['family'], 'model': cell['model'],
                                'workload': cell['workload'], 'metric': name, 'ratio': point})
    if args.max_layers:
        issues.append('Truncated model (smoke test)')
    if args.enforce_eager or args.force_v1_runner:
        issues.append('Non-default execution path; not a default-production audit')
    if getattr(args, 'inherited_execution_overrides', False):
        issues.append('Inherited execution overrides; not an unqualified default-production audit')
    if args.skip_latency:
        issues.append('Latency omitted; throughput-only evidence')
    if not cells:
        verdict = 'NO VALID COMPARISONS'
    elif issues or uncertain:
        verdict = 'INSUFFICIENT EVIDENCE'
    elif signals:
        verdict = 'RE-RELEASE REVIEW RECOMMENDED'
    else:
        verdict = 'WITHIN THRESHOLD FOR TESTED SCOPE'
    return {'verdict': verdict, 'issues': sorted(set(issues)), 'signals': signals,
            'uncertain_intervals': uncertain, 'cells': cells, 'families': family_results,
            'macro_throughput_drift': geomean([x['throughput_drift'] for x in family_results.values()]) if families else None,
            'coverage': {'valid_pairs': sum(r['valid'] for r in comparisons), 'completed_pairs': len(comparisons), 'expected_pairs': expected}}


def source_identity(repo):
    extensions = ('.py', '.toml', '.yaml', '.yml', '.json', '.cu', '.cuh', '.cpp', '.h', '.hpp')
    repo = Path(repo)
    listing = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], capture_output=True, text=True)
    # Generated reports now live inside the checkout. Honor gitignore so a
    # run does not invalidate its own source identity by writing results.
    candidates = (repo/p for p in set(listing.stdout.split('\0')) if p) if listing.returncode == 0 else repo.rglob('*')
    files = sorted(p for p in candidates if p.is_file() and p.suffix in extensions and '.git' not in p.parts)
    git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'sha256': digest([(str(p.relative_to(repo)), file_digest(p)) for p in files]),
            'git_commit': git.stdout.strip() if git.returncode == 0 else None}


def environment_identity(python, destination, timeout):
    code = '''import importlib.metadata as m, json, torch, resource, sys
from urllib.parse import urlsplit,urlunsplit
sources={}
for d in m.distributions():
 raw=d.read_text('direct_url.json')
 if raw:
  direct=json.loads(raw); u=urlsplit(direct.get('url',''))
  direct['url']=urlunsplit((u.scheme,u.hostname or '',u.path,'',''))
  sources[d.metadata['Name']]=direct
try:
 t=torch.zeros(1024,pin_memory=True); pin=bool(t.is_pinned()); error=None
except Exception as e:
 pin=False; error=str(e)
x={'python':sys.version,'executable':sys.executable,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions()),'torch':torch.__version__,'vllm':m.version('vllm'),'cuda':torch.version.cuda,'pin_memory':pin,'pin_error':error,'memlock':resource.getrlimit(resource.RLIMIT_MEMLOCK)[0],'direct_sources':sources,'devices':[{'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]}
json.dump(x,open(sys.argv[1],'w'))
'''
    result = run_command([str(python), '-c', code, str(destination)], destination.with_suffix('.log'), timeout)
    if result['status'] != 'ok':
        raise RuntimeError(f'Environment preflight failed: {destination.with_suffix(".log")}')
    return json.loads(destination.read_text())


def resolve_model(python, model, destination, timeout):
    # Resolve once to an immutable snapshot and use the identical weights and
    # tokenizer in FK and BOTH references. Download time is outside timings.
    code = '''from huggingface_hub import HfApi,snapshot_download
import json,sys
model_spec,output=sys.argv[1:]
model,_,revision=model_spec.partition('@')
info=HfApi().model_info(model,revision=revision or None)
path=snapshot_download(model,revision=info.sha,allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.txt','*.jinja'])
json.dump({'model':model,'revision':info.sha,'path':path},open(output,'w'))
'''
    result = run_command([str(python), '-c', code, model, str(destination)], destination.with_suffix('.log'), timeout)
    if result['status'] != 'ok':
        raise RuntimeError(f'Model snapshot failed: {destination.with_suffix(".log")}')
    return json.loads(destination.read_text())


def write_report(root, manifest):
    s = manifest.get('summary', {})
    lines = ['# FastKernels validation drift audit', '', f"Verdict: **{s.get('verdict', 'INCOMPLETE')}**", '',
             f"References: {manifest['config']['baseline']} → {manifest['config']['candidate']}",
             f"Scope: {manifest['config']['model']}; workloads {manifest['config']['workloads']}; TP={manifest['config']['tp']}",
             f"Coverage: {s.get('coverage', {})}", '',
             '| family | model | workload | repeats | production drift | 95% repeat CI | FK / old | FK / new |',
             '|---|---|---|---|---|---|---|---|']
    for c in s.get('cells', []):
        ci = c['throughput_ci95']
        ci_text = f'{ci[0]:.3f}–{ci[1]:.3f}' if ci else 'unavailable'
        lines.append(f"| {c['family']} | {c['model']} | {c['workload']} | {c['repeats']} | {c['throughput_drift']:.3f}× | {ci_text} | {c['fk_vs_old']:.3f}× | {c['fk_vs_new']:.3f}× |")
    lines += ['', 'Family aggregates (equal weight per model/workload cell):', '']
    for name, value in s.get('families', {}).items():
        lines.append(f"- {name}: throughput drift {value['throughput_drift']:.3f}×, FK / new {value['fk_vs_new']:.3f}×")
    lines += ['', 'Latency drift (old / new batch-completion latency; kept separate from throughput):', '']
    for c in s.get('cells', []):
        for name, value in c['latency'].items():
            lines.append(f"- {c['model']} / {c['workload']} / {name}: {value['drift']:.3f}×; 95% repeat CI {value['ci95']}")
    lines += ['', 'Evidence limitations:', '']
    lines += ['- ' + x for x in s.get('issues', [])]
    if s.get('uncertain_intervals'):
        lines.append('- Repeat intervals are missing or overlap a release boundary.')
    lines += ['- Output-prefix agreement is a screening heuristic, not a downstream quality guarantee.',
              '- Repeat intervals describe timing variability for these inputs, not architecture or workload generalization.',
              '- Reference workarounds and environment settings are recorded in manifest.json; inspect logs for applied patches.',
              '- Drift does not mechanically rescale isolated-kernel agent scores. A release needs baseline changes and revalidation.', '', 'Failures:', '']
    lines += ['- ' + str(x) for x in manifest.get('failures', [])]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def persist(workdir, manifest):
    """Keep historical configuration metadata alongside its raw measurements."""
    destinations = [workdir]
    if manifest.get('run_id'):
        destinations.append(workdir / manifest['run_id'])
    for destination in destinations:
        write_json(destination / 'manifest.json', manifest)
        write_report(destination, manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fk-repo', required=True, type=Path)
    ap.add_argument('--fk-python', required=True, type=Path)
    ap.add_argument('--baseline-python', required=True, type=Path)
    ap.add_argument('--candidate-python', required=True, type=Path)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--model', action='append', required=True, help='FAMILY=HF_MODEL_ID[@REVISION]; repeat for multiple models')
    ap.add_argument('--workloads', default='mixed,long-context')
    ap.add_argument('--num-seqs', type=int, default=1000)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--latency-iters', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=1.05)
    ap.add_argument('--parity-tolerance', type=float, default=1.05)
    ap.add_argument('--stability-tolerance', type=float, default=1.05)
    ap.add_argument('--alignment-floor', type=float, default=32)
    ap.add_argument('--budget-minutes', type=float, default=60)
    ap.add_argument('--run-timeout-s', type=float, default=900)
    ap.add_argument('--workdir', type=Path, default=Path.home() / '.fk-validate-drift')
    ap.add_argument('--reference-patches', choices=['auto', 'off'], default='auto')
    ap.add_argument('--force-v1-runner', action='store_true', help='Explicit non-default-path diagnostic; never automatic')
    ap.add_argument('--max-layers', type=int)
    ap.add_argument('--enforce-eager', action='store_true')
    ap.add_argument('--skip-latency', action='store_true')
    args = ap.parse_args()
    if min(args.num_seqs, args.repeats, args.tp, args.latency_iters, args.budget_minutes, args.run_timeout_s) <= 0 or min(args.threshold, args.parity_tolerance, args.stability_tolerance) <= 1 or args.alignment_floor < 0:
        ap.error('Counts and budgets must be positive; ratio thresholds must exceed one')
    models = []
    for item in args.model:
        family, separator, model = item.partition('=')
        if not separator or not family or not model:
            ap.error('--model must be FAMILY=HF_MODEL_ID')
        models.append((family, model))
    workloads = args.workloads.split(',')
    if not workloads or any(w not in ('mixed', 'long-context') for w in workloads):
        ap.error('Supported workloads: mixed,long-context')
    if len({model for _, model in models}) != len(models) or len(set(workloads)) != len(workloads):
        ap.error('Duplicate models or workloads')
    for field in ('fk_repo', 'fk_python', 'baseline_python', 'candidate_python', 'workdir'):
        # Do not resolve interpreter symlinks: venv identity lives in its path.
        setattr(args, field, Path(os.path.abspath(getattr(args, field).expanduser())))
    args.workdir.mkdir(parents=True, exist_ok=True)
    lock = (args.workdir / '.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        ap.error('Another audit is using this workdir')
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Audit wall-clock budget exhausted')
        return min(args.run_timeout_s, seconds)
    config = vars(args).copy()
    manifest = {'schema': SCHEMA, 'config': config, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'failures': [], 'runs': [], 'comparisons': []}
    expected = len(models) * len(workloads) * args.repeats
    try:
        manifest['source'] = source_identity(args.fk_repo)
        manifest['audit_source'] = {p.name: file_digest(p) for p in HERE.glob('*.py')}
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version,power.limit,clocks.max.sm', '--format=csv,noheader'], text=True, timeout=remaining())
        manifest['hardware'] = gpu.strip().splitlines()
        if len(manifest['hardware']) < args.tp:
            raise ValueError('Not enough visible GPUs for requested TP')
        envs = {}
        for label, py in [('fk', args.fk_python), ('baseline', args.baseline_python), ('candidate', args.candidate_python)]:
            envs[label] = environment_identity(py, args.workdir / f'{label}-environment.json', remaining())
        for label, version in [('baseline', args.baseline), ('candidate', args.candidate)]:
            if envs[label]['vllm'] != version:
                raise ValueError(f'{label} interpreter has vLLM {envs[label]["vllm"]}, expected {version}')
        manifest['environments'] = envs
        pyproject = args.fk_repo / 'pyproject.toml'
        if pyproject.exists():
            pins = dict(re.findall(r'^\s+"(torch|vllm|transformers)==([^";]+)"', pyproject.read_text(), re.M))
            installed = dict(envs['fk']['packages'])
            manifest['fk_declared_core_pins'] = pins
            for package, version in pins.items():
                actual = installed.get(package, '').split('+')[0]
                if actual != version:
                    raise ValueError(f'Fixed FK environment has {package}={actual}, checkout requires {version}')
        env = os.environ.copy()
        # Prevent user candidates from entering the fixed-reference engine.
        empty = args.workdir / 'empty-candidates'
        empty.mkdir(exist_ok=True)
        if any(empty.iterdir()):
            raise ValueError('Audit empty-candidates directory must be empty')
        env['FASTKERNELS_CANDIDATE_DIR'] = str(empty)
        env['PYTHONPATH'] = str(args.fk_repo)
        if args.force_v1_runner:
            env['VLLM_USE_V2_MODEL_RUNNER'] = '0'
        if any(not e['pin_memory'] for e in envs.values()) and not args.force_v1_runner:
            raise ValueError('Pinned-memory health failed; fix host or explicitly request diagnostic --force-v1-runner')
        prefixes = ('VLLM_', 'FASTKERNELS_', 'CUDA_', 'NCCL_', 'TORCH_', 'HF_HUB_', 'OMP_')
        manifest['execution_env'] = {k: v for k, v in env.items() if k.startswith(prefixes) and not any(s in k for s in ('TOKEN', 'KEY', 'SECRET'))}
        args.inherited_execution_overrides = any(
            k.startswith(('VLLM_', 'FASTKERNELS_')) and
            not any(s in k for s in ('CACHE', 'LOG', 'RESULT', 'CANDIDATE_DIR'))
            for k in os.environ
        )
        manifest['models'] = {}
        for family, model in models:
            manifest['models'][model] = resolve_model(args.fk_python, model, args.workdir / f'model-{digest(model)[:12]}.json', remaining())
        # Include all config, actual installed versions, source, hardware, and
        # immutable model revisions. A changed setting gets a new cache namespace.
        identity = {k: manifest[k] for k in ('schema', 'config', 'source', 'audit_source', 'hardware', 'environments', 'execution_env', 'models')}
        run_id = digest(identity)
        root = args.workdir / run_id
        root.mkdir(exist_ok=True)
        manifest['run_id'] = run_id
        for family, model in models:
            for workload in workloads:
                frozen = root / f'inputs-{digest([model, workload])[:16]}.json'
                for repeat in range(args.repeats):
                    pair = {}
                    order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
                    for label in order:
                        remaining()
                        output = root / f'{digest([model, workload])[:16]}-r{repeat}-{label}'
                        stamp = output / 'complete.json'
                        cached = False
                        if stamp.exists() and frozen.exists():
                            record = json.loads(stamp.read_text())
                            cached = record['input_file_sha256'] == file_digest(frozen) and all(
                                (output / name).exists() and file_digest(output / name) == value for name, value in record['files'].items())
                        if cached:
                            manifest['runs'].append({'status': 'cached', 'label': label,
                                                     'output': str(output), 'command': record['command']})
                        if not cached:
                            output.mkdir(parents=True, exist_ok=True)
                            for name in ('results.json', 'vllm_raw.json', 'fastkernels_raw.json', 'complete.json'):
                                (output / name).unlink(missing_ok=True)
                            cmd = command_for(args, manifest['models'][model]['path'], workload, getattr(args, label + '_python'), output, frozen)
                            print(f'[{family}/{model}/{workload} repeat {repeat + 1}] {label}', flush=True)
                            record = {'command': cmd, 'label': label, 'output': str(output), **run_command(cmd, output / 'run.log', remaining(), env)}
                            manifest['runs'].append(record)
                            if record['status'] != 'ok':
                                manifest['failures'].append(record)
                                break
                        try:
                            if not cached:
                                collect_cli_results(output)
                            result = read_validation(output, frozen)
                            if not cached:
                                write_json(stamp, {'command': cmd, 'input_file_sha256': file_digest(frozen), 'files': {n: file_digest(output / n) for n in validation_files(output)}})
                            pair[label] = result
                        except (OSError, ValueError, KeyError) as e:
                            manifest['failures'].append({'output': str(output), 'error': str(e)})
                            break
                    if len(pair) == 2:
                        try:
                            row = compare(pair['baseline'], pair['candidate'], workload, args.alignment_floor)
                            row.update(family=family, model=model, workload=workload, repeat=repeat)
                            manifest['comparisons'].append(row)
                        except (ValueError, KeyError, StopIteration) as e:
                            manifest['failures'].append({'model': model, 'workload': workload, 'error': str(e)})
                    manifest['summary'] = aggregate(manifest['comparisons'], expected, args)
                    persist(args.workdir, manifest)
                    if len(pair) != 2:
                        break  # don't pay to repeat the same setup failure
        if source_identity(args.fk_repo) != manifest['source']:
            raise RuntimeError('FastKernels source changed during the audit')
    except (Exception, KeyboardInterrupt) as e:
        manifest['failures'].append({'error': str(e) or 'Interrupted'})
    finally:
        manifest['summary'] = aggregate(manifest['comparisons'], expected, args)
        if manifest['failures'] and manifest['summary']['verdict'] not in ('NO VALID COMPARISONS', 'INSUFFICIENT EVIDENCE'):
            manifest['summary']['verdict'] = 'INSUFFICIENT EVIDENCE'
        manifest['elapsed_minutes'] = (time.monotonic() - started) / 60
        persist(args.workdir, manifest)
    lock.close()
    print(manifest['summary']['verdict'])
    print(args.workdir / 'report.md')
    return 2 if manifest['summary']['verdict'] in ('INSUFFICIENT EVIDENCE', 'NO VALID COMPARISONS') else 0


if __name__ == '__main__':
    sys.exit(main())
