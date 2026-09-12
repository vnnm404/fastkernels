#!/usr/bin/env python3
"""Serial, resumable standard-vLLM version audit driven by a scenario table.

Run with the fixed FastKernels interpreter. No packages are installed or upgraded.
Only the script's private per-row model/download cache is eligible for deletion.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from drift_runner import run_command, collect_cli_results, read_validation, validation_files
from fk_drift import digest, file_digest, write_json, environment_identity, source_identity, compare, interval
from compatibility import inherit_hf_auth, hub_retry, access_problem

SCHEMA = 1


def plan_scenarios(repo, selection):
    """Resolve workload enums locally; planning never downloads model configs."""
    sys.path.insert(0, str(repo))
    import yaml
    from fastkernels.workloads import _module_from_name, _resolve_workload_token
    from fastkernels.validate import _MODULE_TO_HARNESS
    path = Path(selection).expanduser()
    if not path.is_file():
        path = repo / 'fastkernels/scenarios' / (selection if selection.endswith('.yaml') else selection + '.yaml')
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('scenarios'), list):
        raise ValueError('Expected a scenarios list')
    rows = []
    for index, raw in enumerate(data['scenarios']):
        if not isinstance(raw, dict) or not isinstance(raw.get('model'), str):
            raise ValueError(f'Invalid scenario row {index}')
        row = dict(raw)
        model = row['model']
        module = _module_from_name(model)
        harness = _MODULE_TO_HARNESS.get(module)
        if model.startswith('fla-hub/'):
            harness = 'bench_fla'
        if 'bitnet' in model.lower():
            harness = 'bench_microsoft_bitnet'
        if row.get('draft_model'):
            harness = 'bench_sglang'
        # Dense Qwen2 uses the shared Llama engine; distinguish VL/Omni rows.
        if module is None and model.startswith(('Qwen/Qwen2-', 'Qwen/Qwen2.5-')) and not any(x in model for x in ('VL', 'Omni')):
            harness = 'bench_vllm'
        workloads = row.get('workloads', row.get('legacy_workloads'))
        if not isinstance(workloads, list) or not workloads or len(workloads) != len(set(workloads)):
            raise ValueError(f'{model}: workloads must be a nonempty list without duplicates')
        names = [str(_resolve_workload_token(w).value) for w in workloads] if 'workloads' in row else workloads
        if type(row.get('tp')) is not int or row['tp'] < 1 or not row.get('dtype'):
            raise ValueError(f'{model}: positive tp and dtype required')
        rows.append({'id': f'{index:03d}-{digest(row)[:12]}', 'scenario': row,
                     'workloads': names, 'harness': harness,
                     'status': 'planned' if harness == 'bench_vllm' else 'skipped',
                     'reason': None if harness == 'bench_vllm' else f'Reference is {harness or "unrecognized"}, not standard vLLM'})
    return {'scenario_file': str(path.resolve()), 'scenario_sha256': file_digest(path), 'rows': rows}


def completed(output, frozen):
    try:
        stamp = json.loads((output / 'complete.json').read_text())
        if stamp['input_file_sha256'] != file_digest(frozen):
            return False
        if not all(file_digest(output / name) == sha for name, sha in stamp['files'].items()):
            return False
        collect_cli_results(output)
        read_validation(output, frozen)
        return True
    except (OSError, ValueError, KeyError):
        return False


def pair_metrics(old, new, floor, expected):
    if old.get('max_num_seqs') != new.get('max_num_seqs'):
        raise ValueError('Concurrent sequence limits differ between reference runs')
    if old['input_sha256'] != new['input_sha256'] or old.get('media_inputs', {}) != new.get('media_inputs', {}):
        raise ValueError('Inputs or decoded-media checksums differ between reference versions')
    throughput = [x['scenario'] for x in old.get('scenarios', [])]
    latency = [x['scenario'] for x in old.get('latency_scenarios', [])]
    if set(throughput + latency) != set(expected) or len(throughput + latency) != len(expected):
        raise ValueError('Requested workload coverage is incomplete or duplicated')
    if [x['scenario'] for x in new.get('scenarios', [])] != throughput:
        raise ValueError('Throughput coverage differs')
    if [x['scenario'] for x in new.get('latency_scenarios', [])] != latency:
        raise ValueError('Latency coverage differs')
    metrics = []
    for name in throughput:
        for result in (old, new):
            for engine in ('vllm', 'fastkernels'):
                raw = next(x for x in result[engine + '_raw']['throughput'] if x['name'] == name)
                if raw.get('warmup_iters', 0) < 1:
                    raise ValueError('Missing full-workload warmup evidence')
                combined = next(x for x in result.get('scenarios', []) if x['scenario'] == name)
                elapsed = raw['elapsed']
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError('Invalid elapsed time')
                if not math.isclose(combined[engine + '_tok_per_s'], raw['total_output_tokens']/elapsed, rel_tol=1e-6):
                    raise ValueError('Throughput disagrees with raw measurement')
        check = compare(old, new, name, floor)
        if not check['valid']:
            raise ValueError(f'{name}: output-prefix agreement failed: {check["alignment"]}')
        a = next(x for x in old.get('scenarios', []) if x['scenario'] == name)
        b = next(x for x in new.get('scenarios', []) if x['scenario'] == name)
        metrics.append({'workload': name, 'kind': 'throughput', 'unit': 'tokens/s',
                        'fk_old': a['fastkernels_tok_per_s'], 'old': a['vllm_tok_per_s'],
                        'fk_new': b['fastkernels_tok_per_s'], 'new': b['vllm_tok_per_s'],
                        'drift': check['throughput_drift'], 'fk_stability': check['fk_stability'],
                        'alignment': check['alignment'], 'requests': check['requests']})
    for a, b in zip(old.get('latency_scenarios', []), new.get('latency_scenarios', [])):
        vals = [a['fastkernels_median_s'], a['vllm_median_s'], b['fastkernels_median_s'], b['vllm_median_s']]
        for result in (a, b):
            for engine in ('vllm', 'fastkernels'):
                samples = result[engine + '_latencies']
                if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
                    raise ValueError('Invalid latency samples')
                if not math.isclose(statistics.median(samples), result[engine + '_median_s'], rel_tol=1e-6):
                    raise ValueError('Latency median disagrees with samples')
        if any(not math.isfinite(v) or v <= 0 for v in vals):
            raise ValueError('Invalid latency')
        metrics.append({'workload': a['scenario'], 'kind': 'latency', 'unit': 'ms',
                        'fk_old': vals[0]*1000, 'old': vals[1]*1000,
                        'fk_new': vals[2]*1000, 'new': vals[3]*1000,
                        'drift': vals[1]/vals[3], 'fk_stability': vals[0]/vals[2]})
    return metrics


def persist(root, state):
    state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(root / 'status.json', state)
    lines = ['# Scenario-driven vLLM drift audit', '', f"Status: **{state['status']}**", '',
             f"References: {state['baseline']} → {state['candidate']}. One model at a time; full untimed warmup before throughput.", '',
             '| Model | Workload | Unit | FK / old run | Old vLLM | FK / new run | New vLLM | New/old speed | Repeat 95% CI |',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    for row in state['rows']:
        groups = {}
        for pair in row.get('pairs', []):
            for metric in pair['metrics']:
                groups.setdefault((metric['workload'], metric['kind']), []).append(metric)
        for (_, kind), metrics in groups.items():
            m = metrics[0]
            values = [statistics.median([x[k] for x in metrics]) for k in ('fk_old','old','fk_new','new')]
            ratio = math.exp(statistics.mean(math.log(x['drift']) for x in metrics))
            ci = interval([x['drift'] for x in metrics])
            ci_text = f'{ci[0]:.3f}–{ci[1]:.3f}' if ci else 'unavailable'
            lines.append(f"| {row['scenario']['model']} | {m['workload']} | {m['unit']} | " + ' | '.join(f'{v:,.2f}' for v in values) + f' | {ratio:.3f}× | {ci_text} |')
    lines += ['', 'Throughput: higher is better. Latency: lower is better; its speed ratio is old/new latency.',
              'Values are medians across paired runs; intervals describe repeat variability, not generalization.',
              'A PASS means requested work completed and checks passed, not that a release decision is justified.', '', '## Coverage and failures', '']
    for row in state['rows']:
        lines.append(f"- {row['id']} {row['scenario']['model']}: **{row['status']}**" + (f" — {row.get('reason')}" if row.get('reason') else ''))
        if row.get('max_num_seqs') is not None:
            lines.append(f"  Concurrent sequence limit: {row['max_num_seqs']} in both engines and both reference runs.")
        for warning in sorted(set(row.get('warnings', []))):
            lines.append(f"  Warning: {warning}")
    if state.get('error'):
        lines += ['', 'Error: ' + state['error']]
    (root / 'report.md.tmp').write_text('\n'.join(lines)+'\n')
    (root / 'report.md.tmp').replace(root / 'report.md')


def prepare_model(model, metadata, data_root, reserve, refresh=False, interpreters=None):
    """Runs in a bounded child. Download only one pinned checkpoint at a time."""
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    repo, _, revision = model.partition('@')
    if metadata.exists() and not refresh:
        info = json.loads(metadata.read_text())
    else:
        try:
            record = hub_retry(lambda: HfApi().model_info(repo, revision=revision or None, files_metadata=True, timeout=30))
        except Exception as exc:
            problem = access_problem(exc)
            if problem:
                write_json(metadata, dict(problem, repo=repo));return
            raise
        weights = sum(x.size or 0 for x in record.siblings if x.rfilename.endswith('.safetensors'))
        if not weights:
            raise ValueError('No safetensors checkpoint found; refusing an unbounded download')
        info = {'repo': repo, 'revision': record.sha, 'weight_bytes': weights}
        write_json(metadata, info)
    # A previous access failure has no revision to resume.
    if 'revision' not in info:
        return prepare_model(model, metadata, data_root, reserve, True, interpreters)
    info.pop('status', None);info.pop('reason', None)
    try:
        config_file = hub_retry(lambda: hf_hub_download(info['repo'], 'config.json', revision=info['revision']))
    except Exception as exc:
        problem = access_problem(exc)
        if problem:
            write_json(metadata, dict(info, **problem));return
        raise
    if interpreters:
        config_dir = metadata.parent/'config-preflight'
        config_dir.mkdir(exist_ok=True)
        shutil.copyfile(config_file, config_dir/'config.json')
        info['compatibility'] = {}
        for label, python in interpreters.items():
            output = metadata.parent/(label+'-compatibility.json')
            output.unlink(missing_ok=True)
            # Stay in the preparation process group so its outer timeout also
            # kills the probe. These checks do not launch engine workers.
            with (metadata.parent/(label+'-compatibility.log')).open('w') as log:
                try:
                    result = subprocess.run([python, str(Path(__file__).with_name('compatibility.py')),
                                             str(config_dir), str(output)],
                                            stdout=log, stderr=subprocess.STDOUT, timeout=120)
                    ok = result.returncode == 0 and output.exists()
                except subprocess.TimeoutExpired:
                    ok = False
            check = json.loads(output.read_text()) if ok else {
                'status':'environment-error', 'reason':f'{label} preflight failed/timed out; see {label}-compatibility.log'}
            info['compatibility'][label] = check
        problems = [(label, check) for label, check in info['compatibility'].items() if check['status']!='ok']
        if problems:
            info.update(status='unsupported' if all(c['status']=='unsupported' for _,c in problems) else 'blocked',
                        reason='; '.join(label+': '+c['reason'] for label,c in problems))
            write_json(metadata, info);return
    cached = sum(p.stat().st_size for p in {p.resolve() for p in data_root.glob('hf/hub/models--*/snapshots/*/*.safetensors')} if p.is_file())
    if shutil.disk_usage(data_root).free < max(0, info['weight_bytes'] * 1.1 - cached) + reserve:
        raise OSError('Insufficient disk for checkpoint plus reserve: ' + str(info))
    path = hub_retry(lambda: snapshot_download(info['repo'], revision=info['revision'], allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.txt','*.jinja']))
    info['path'] = path
    write_json(metadata, info)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenarios', default='full')
    ap.add_argument('--fk-repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--baseline-python', type=Path)
    ap.add_argument('--candidate-python', type=Path)
    ap.add_argument('--baseline', default='0.18.0')
    ap.add_argument('--candidate', default='0.26.0')
    ap.add_argument('--workdir', type=Path, default=Path(__file__).resolve().parent/'results')
    ap.add_argument('--data-root', type=Path, help='Scratch filesystem for private model/media caches')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--warmup-iters', type=int, default=1)
    ap.add_argument('--latency-iters', type=int, default=5)
    ap.add_argument('--max-requests', type=int, help='Diagnostic cap; omit for canonical workload sizes')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alignment-floor', type=float, default=32)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=3600, help='Seconds per CLI attempt or model preparation')
    ap.add_argument('--budget-hours', type=float, default=12)
    ap.add_argument('--min-free-gb', type=float, default=10)
    ap.add_argument('--keep-models', action='store_true', help='Retain private download caches after each row')
    ap.add_argument('--skip-insufficient-gpus', action='store_true', help='Report rows exceeding visible GPU count as skipped')
    ap.add_argument('--dry-run', action='store_true', help='Plan locally; no GPU checks, environment setup or downloads')
    args = ap.parse_args(argv)
    for field in ('fk_repo','workdir','data_root','baseline_python','candidate_python'):
        val = getattr(args, field)
        if val is not None:
            setattr(args, field, Path(os.path.abspath(val.expanduser())))
    if min(args.repeats,args.warmup_iters,args.latency_iters,args.timeout,args.budget_hours) <= 0 or args.retries < 0 or args.min_free_gb < 0 or (args.max_requests is not None and args.max_requests < 1):
        ap.error('Counts and time limits must be positive; retries and reserve nonnegative')
    plan = plan_scenarios(args.fk_repo, args.scenarios)
    if args.dry_run:
        print(json.dumps(plan, indent=2));return 0
    if not args.baseline_python or not args.candidate_python or not args.data_root:
        ap.error('--baseline-python, --candidate-python and --data-root are required to run')
    inherit_hf_auth()
    args.workdir.mkdir(parents=True, exist_ok=True)
    lock = (args.workdir/'.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('Another audit is using this workdir')
    state = {'schema':SCHEMA, 'status':'running', 'baseline':args.baseline, 'candidate':args.candidate,
             'plan':{k:v for k,v in plan.items() if k!='rows'}, 'rows':plan['rows'], 'config':vars(args)}
    started = time.monotonic();deadline=started+args.budget_hours*3600
    reserve = int(args.min_free_gb*2**30)
    def timeout():
        left=deadline-time.monotonic()
        if left<=0:raise TimeoutError('Total audit budget exhausted')
        return min(left,args.timeout)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    persist(args.workdir,state)
    private = None
    try:
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True,timeout=20).strip().splitlines()
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        gpu_count=len(visible.split(',')) if visible else len(hardware)
        state['hardware']=hardware
        if args.skip_insufficient_gpus:
            for row in state['rows']:
                if row['status']=='planned' and row['scenario']['tp']>gpu_count:
                    row.update(status='skipped', reason=f"Requires TP={row['scenario']['tp']}; only {gpu_count} GPUs visible")
        source=source_identity(args.fk_repo);state['source']=source
        envs={}
        for label,py in [('fk',Path(sys.executable)),('baseline',args.baseline_python),('candidate',args.candidate_python)]:
            envs[label]=environment_identity(py,args.workdir/(label+'-environment.json'),timeout())
            if not envs[label]['pin_memory']:raise ValueError(label+': pinned memory unavailable')
        for label,version in [('baseline',args.baseline),('candidate',args.candidate)]:
            if envs[label]['vllm']!=version:raise ValueError(label+': wrong vLLM version')
        state['environments']=envs
        import re
        pins=dict(re.findall(r'^\s+"(torch|vllm|transformers)==([^";]+)"',(args.fk_repo/'pyproject.toml').read_text(),re.M))
        packages=dict(envs['fk']['packages'])
        if any(packages.get(k,'').split('+')[0]!=v for k,v in pins.items()):raise ValueError('Fixed FK environment does not match checkout pins')
        identity=digest([SCHEMA,plan,source,envs,hardware,visible,args.repeats,args.warmup_iters,args.latency_iters,args.max_requests,args.seed,args.alignment_floor,str(args.data_root),
                         {k:v for k,v in os.environ.items() if k.startswith(('VLLM_','FASTKERNELS_','CUDA_','NCCL_','OMP_','TORCH_')) and not any(secret in k for secret in ('TOKEN','KEY','SECRET'))},
                         {p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')}])
        root=args.workdir/identity;root.mkdir(exist_ok=True)
        args.data_root.mkdir(parents=True,exist_ok=True)
        private=args.data_root/('fk-drift-'+digest(str(args.workdir))[:16])
        if private.is_symlink():raise ValueError('Private scratch must not be a symlink')
        private.mkdir(mode=0o700,exist_ok=True)
        if private.stat().st_uid != os.getuid():raise ValueError('Private scratch belongs to another user')
        private.chmod(0o700)
        owner=private/'.owner.json'
        if owner.exists():
            if json.loads(owner.read_text())!={'workdir':str(args.workdir)}:raise ValueError('Scratch ownership mismatch')
        elif any(private.iterdir()):raise ValueError('Refusing to use unowned scratch directory')
        else:write_json(owner,{'workdir':str(args.workdir)})
        state['run_id']=identity
        for row in state['rows']:
            if row['status']=='skipped':continue
            rowroot=root/row['id'];rowroot.mkdir(exist_ok=True)
            cache=private/row['id']
            if cache.is_symlink():raise ValueError('Row cache must not be a symlink')
            cache.mkdir(exist_ok=True,mode=0o700)
            row['pairs']=[]
            row['status']='running';persist(args.workdir,state)
            try:
                if row['scenario']['tp']>gpu_count:raise ValueError(f'Requires TP={row["scenario"]["tp"]}, only {gpu_count} GPUs visible')
                busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
                if busy:raise RuntimeError('GPU compute processes already running; refusing a contended measurement')
                env=os.environ.copy();env.update({'PYTHONPATH':str(args.fk_repo),'HF_HOME':str(cache/'hf'),
                    'HF_HUB_CACHE':str(cache/'hf/hub'),'HF_DATASETS_CACHE':str(cache/'hf/datasets'),
                    'FASTKERNELS_MEDIA_CACHE':str(cache/'media'),'FASTKERNELS_CANDIDATE_DIR':str(cache/'empty-candidates')})
                (cache/'empty-candidates').mkdir(exist_ok=True)
                if any((cache/'empty-candidates').iterdir()):raise ValueError('Candidate directory is not empty')
                frozen=rowroot/'inputs.json'
                all_done=all(completed(rowroot/f'r{rep}-{label}',frozen) for rep in range(args.repeats) for label in ('baseline','candidate'))
                meta=rowroot/'model.json'
                if not all_done:
                    code='from pathlib import Path; from run_scenarios import prepare_model; import sys,json; prepare_model(sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]),interpreters=json.loads(sys.argv[5]))'
                    prep_env=env.copy();prep_env['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+str(args.fk_repo)
                    interpreters={'fk':sys.executable,'baseline':str(args.baseline_python),'candidate':str(args.candidate_python)}
                    result=run_command([sys.executable,'-c',code,row['scenario']['model'],str(meta),str(cache),str(reserve),json.dumps(interpreters)],rowroot/'prepare.log',timeout(),prep_env,cache,reserve)
                    if result['status']!='ok':raise RuntimeError('Model preparation '+result['status']+'; see '+str(rowroot/'prepare.log'))
                model=json.loads(meta.read_text())
                row['compatibility']=model.get('compatibility',{})
                for label, check in row['compatibility'].items():
                    if check.get('warning'):
                        row.setdefault('warnings',[]).append(label+': '+check['warning'])
                if model.get('status') in ('blocked','unsupported'):
                    row.update(status=model['status'],reason=model['reason']);continue
                row['model_revision']=model['revision']
                for rep in range(args.repeats):
                    pair={}
                    for label in (('baseline','candidate') if rep%2==0 else ('candidate','baseline')):
                        output=rowroot/f'r{rep}-{label}'
                        if not completed(output,frozen):
                            for attempt in range(args.retries+1):
                                timeout()
                                if output.exists():
                                    archive=rowroot/f'r{rep}-{label}-failed-{time.time_ns()}'
                                    output.rename(archive)
                                output.mkdir()
                                scenario=dict(row['scenario']);scenario['model']=model['path']
                                write_json(output/'scenario.yaml',{'scenarios':[scenario]})
                                command=[str(Path(sys.executable).parent/'fastkernels'),'validate',str(output/'scenario.yaml'),
                                    '--vllm-python',str(getattr(args,label+'_python')),'--seed',str(args.seed),
                                    '--warmup-iters',str(args.warmup_iters),'--latency-iters',str(args.latency_iters),
                                    '--reference-patches','auto','--timeout',str(int(timeout())),
                                    '--output-dir',str(output/('cli-'+digest(str(output))[:20])),
                                    '--inputs-json' if frozen.exists() else '--save-inputs-json',str(frozen)]
                                if args.max_requests is not None:command+=['--max-requests',str(args.max_requests)]
                                row['active']={'repeat':rep,'reference':label,'attempt':attempt,'command':command}
                                persist(args.workdir,state)
                                print(f"[{row['id']}] {label} repeat {rep+1}/{args.repeats}, attempt {attempt+1}",flush=True)
                                result=run_command(command,output/'run.log',timeout(),env,cache,reserve)
                                try:
                                    if result['status']!='ok':raise RuntimeError('CLI '+result['status'])
                                    collect_cli_results(output);read_validation(output,frozen)
                                    write_json(output/'complete.json',{'input_file_sha256':file_digest(frozen),'files':{n:file_digest(output/n) for n in validation_files(output)}})
                                    break
                                except (OSError,ValueError,KeyError,RuntimeError) as exc:
                                    row.setdefault('attempt_failures',[]).append({'repeat':rep,'reference':label,'attempt':attempt,'error':str(exc),'log':str(output/'run.log')})
                                    persist(args.workdir,state)
                                    if attempt==args.retries:raise
                        pair[label]=read_validation(output,frozen)
                    metrics=pair_metrics(pair['baseline'],pair['candidate'],args.alignment_floor,row['workloads'])
                    row['max_num_seqs']=pair['baseline'].get('max_num_seqs')
                    if any(max(m['fk_stability'],1/m['fk_stability'])>1.05 for m in metrics):
                        row.setdefault('warnings',[]).append('Fixed FK timing varied by over 5%; investigate contention/noise')
                    row['pairs'].append({'repeat':rep,'metrics':metrics});persist(args.workdir,state)
                row['status']='passed';row.pop('active',None)
            except KeyboardInterrupt as exc:
                row['status']='interrupted';row['reason']=str(exc)
                raise
            except Exception as exc:
                row['status']='failed';row['reason']=str(exc)
            finally:
                if not args.keep_models and cache.exists():
                    if row['status']=='passed':
                        shutil.rmtree(cache)
                    elif (cache/'hf').exists():
                        # Preserve frozen media for a failed row's later resume.
                        shutil.rmtree(cache/'hf')
                persist(args.workdir,state)
        if source_identity(args.fk_repo)!=source:raise RuntimeError('FK source changed during audit')
        selected=[r for r in state['rows'] if r['status']!='skipped']
        state['status']='PASS' if selected and all(r['status']=='passed' for r in selected) else 'INCOMPLETE'
    except (Exception,KeyboardInterrupt) as exc:
        state['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'INCOMPLETE';state['error']=str(exc)
    finally:
        state['elapsed_seconds']=time.monotonic()-started;persist(args.workdir,state);lock.close()
    print(state['status'],args.workdir/'report.md',flush=True)
    return 0 if state['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
