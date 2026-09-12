import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fk_drift import aggregate, compare, digest
import fk_drift
from drift_runner import run_command, command_for, collect_cli_results
spec = importlib.util.spec_from_file_location('frozen_inputs', ROOT.parent / 'fastkernels/validate/frozen_inputs.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def validation(ref_rate, fk_rate=100):
    def raw(rate):
        return {'throughput': [{'name': 'mixed', 'outputs': [{'token_ids': list(range(64))}] * 4,
                                'total_output_tokens': 256, 'elapsed': 256 / rate}]}
    return {'input_sha256': 'same', 'vllm_raw': raw(ref_rate), 'fastkernels_raw': raw(fk_rate),
            'latency_scenarios': [{'scenario': 'fixed-batch-32', 'vllm_median_s': 100 / ref_rate,
                                   'fastkernels_median_s': 100 / fk_rate}]}


class AuditTests(unittest.TestCase):
    def test_drift_and_direct_gap(self):
        c = compare(validation(100), validation(120), 'mixed', 32)
        self.assertAlmostEqual(c['throughput_drift'], 1.2)
        self.assertAlmostEqual(c['fk_vs_new'], 1 / 1.2)
        self.assertAlmostEqual(c['latency']['fixed-batch-32']['drift'], 1.2)
        self.assertEqual(c['fk_stability'], 1)
        self.assertTrue(c['valid'])

    def test_reject_unequal_inputs_counts_and_work(self):
        for change in ('hash', 'requests', 'tokens', 'elapsed'):
            b = validation(120)
            raw = b['vllm_raw']['throughput'][0]
            if change == 'hash': b['input_sha256'] = 'other'
            if change == 'requests': raw['outputs'] = raw['outputs'][:-1]
            if change == 'tokens': raw['total_output_tokens'] = 1
            if change == 'elapsed': raw['elapsed'] = float('nan')
            with self.assertRaises(ValueError): compare(validation(100), b, 'mixed', 32)

    def test_alignment_checks_fk_as_well_as_references(self):
        b = validation(120)
        b['fastkernels_raw']['throughput'][0]['outputs'] = [{'token_ids': [999] * 64}] * 4
        self.assertFalse(compare(validation(100), b, 'mixed', 32)['valid'])

    def args(self):
        return SimpleNamespace(threshold=1.05, parity_tolerance=1.05, stability_tolerance=1.05,
                               max_layers=None, enforce_eager=False, force_v1_runner=False, skip_latency=False)

    def rows(self, ratio, family='dense', model='a'):
        rows = []
        for i in range(3):
            r = compare(validation(100), validation(100 * ratio), 'mixed', 32)
            r.update(family=family, model=model, workload='mixed', repeat=i, requests=1000)
            rows.append(r)
        return rows

    def test_complete_threshold_and_regression(self):
        for ratio in (1.2, 0.8):
            self.assertEqual(aggregate(self.rows(ratio), 3, self.args())['verdict'], 'RE-RELEASE REVIEW RECOMMENDED')
        self.assertEqual(aggregate(self.rows(1.01), 3, self.args())['verdict'], 'WITHIN THRESHOLD FOR TESTED SCOPE')

    def test_missing_and_smoke_never_within(self):
        rows = self.rows(1)
        self.assertEqual(aggregate(rows, 6, self.args())['verdict'], 'INSUFFICIENT EVIDENCE')
        rows[0]['requests'] = 4
        self.assertEqual(aggregate(rows, 3, self.args())['verdict'], 'INSUFFICIENT EVIDENCE')

    def test_family_cancellation_cannot_hide_signal(self):
        rows = self.rows(1.2) + self.rows(1 / 1.2, family='moe', model='b')
        result = aggregate(rows, 6, self.args())
        self.assertAlmostEqual(result['macro_throughput_drift'], 1)
        self.assertEqual(len(result['families']), 2)
        self.assertEqual(result['verdict'], 'RE-RELEASE REVIEW RECOMMENDED')

    def test_latency_can_trigger_with_flat_throughput(self):
        rows = self.rows(1)
        for r in rows: r['latency']['fixed-batch-32']['drift'] = 1.2
        self.assertEqual(aggregate(rows, 3, self.args())['verdict'], 'RE-RELEASE REVIEW RECOMMENDED')

    def test_frozen_roundtrip_tampering_and_config(self):
        payload = {'schema': 1, 'identity': {'model': 'm', 'seed': 42}, 'max_model_len': 10,
                   'throughput': [{'prompt_token_ids': [[1, 2]], 'output_lens': [3]}], 'latency': []}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'inputs.json'
            frozen.save_inputs(p, payload)
            self.assertEqual(frozen.load_inputs(p, payload['identity']), payload)
            with self.assertRaises(ValueError): frozen.load_inputs(p, {'model': 'm', 'seed': 43})
            obj = json.loads(p.read_text()); obj['payload']['throughput'][0]['output_lens'] = [4]
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError): frozen.load_inputs(p, payload['identity'])

    def test_timeout_returns_failure_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'run.log'
            result = run_command([sys.executable, '-u', '-c', 'import time; print("started"); time.sleep(20)'], p, .2)
            self.assertEqual(result['status'], 'timeout')
            self.assertIn('started', p.read_text())

    def test_noisy_repeats_do_not_claim_release_or_parity(self):
        rows = self.rows(1)
        for row, ratio in zip(rows, (0.8, 1.0, 1.2)):
            row['throughput_drift'] = ratio
        self.assertEqual(aggregate(rows, 3, self.args())['verdict'], 'INSUFFICIENT EVIDENCE')

    def test_native_kernel_edit_invalidates_source_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kernel = root / 'kernel.cu'
            kernel.write_text('old implementation')
            a = fk_drift.source_identity(root)
            kernel.write_text('new implementation')
            self.assertNotEqual(a['sha256'], fk_drift.source_identity(root)['sha256'])

    def test_ignored_results_do_not_invalidate_source_inside_clone(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'.gitignore').write_text('results/\n')
            kernel = root/'kernel.cu'; kernel.write_text('source')
            before = fk_drift.source_identity(root)
            (root/'results').mkdir()
            (root/'results/status.json').write_text('{"status":"running"}')
            self.assertEqual(before, fk_drift.source_identity(root))
            kernel.write_text('changed source')
            self.assertNotEqual(before['sha256'], fk_drift.source_identity(root)['sha256'])

    def test_cli_failure_cannot_be_accepted_as_harness_success(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            cli = output / 'cli-test'; cli.mkdir()
            (cli / 'summary.json').write_text(json.dumps({
                'run': {'status': 'FAIL'},
                'models': [{'status': 'PASS', 'harness': 'bench_vllm'}],
            }))
            with self.assertRaisesRegex(ValueError, 'exactly one passing'):
                collect_cli_results(output)

    def test_orchestrator_cold_run_resume_and_changed_request_count(self):
        # A fake CLI process exercises the scenario/summary/subprocess boundary
        # without CUDA. It deliberately knows nothing about audit cache logic.
        worker = '''#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('subcommand',choices=['validate'])
p.add_argument('table')
for n in ('text-scenario','max-requests','vllm-python','output-dir','inputs-json','save-inputs-json'):
 p.add_argument('--'+n)
a,_=p.parse_known_args()
assert json.loads(Path(a.table).read_text())['scenarios'][0]['model']=='/snapshot'
a.scenario=a.text_scenario; a.num_seqs=a.max_requests
cli=Path(a.output_dir);out=cli/'000_bench_vllm';out.mkdir(parents=True,exist_ok=True)
marker=out.parents[3]/'launches.txt'
with marker.open('a') as f:f.write('launch\\n')
if a.inputs_json:
 envelope=json.loads(Path(a.inputs_json).read_text())
else:
 payload={'throughput':[{'name':a.scenario,'prompt_token_ids':[[1]]*int(a.num_seqs),'output_lens':[64]*int(a.num_seqs)}]}
 sha=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 envelope={'payload':payload,'sha256':sha}
 Path(a.save_inputs_json).write_text(json.dumps(envelope))
raw={'throughput':[{'name':a.scenario,'elapsed':1.0,'total_output_tokens':64*int(a.num_seqs),'outputs':[{'token_ids':list(range(64))}]*int(a.num_seqs)}]}
for engine in ('vllm','fastkernels'):
 (out/(engine+'_raw.json')).write_text(json.dumps({'raw':raw}))
(out/'results.json').write_text(json.dumps({'input_sha256':envelope['sha256']}))
(cli/'summary.json').write_text(json.dumps({'run':{'status':'PASS'},'models':[{'status':'PASS','harness':'bench_vllm'}]}))
'''
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); repo=root/'repo'; harness=root/'bin/fastkernels'
            harness.parent.mkdir(parents=True); harness.write_text(worker); harness.chmod(0o755)
            argv=['fk_drift.py','--fk-repo',str(repo),'--fk-python',str(root/'bin/python'),
                  '--baseline-python','/tmp/old/bin/python','--candidate-python','/tmp/new/bin/python',
                  '--baseline','old','--candidate','new','--model','dense=test/model',
                  '--workloads','mixed','--num-seqs','8','--repeats','1','--workdir',str(root/'audit')]
            def environment(py,*unused):
                return {'vllm': 'old' if 'old' in str(py) else 'new', 'pin_memory':True}
            with patch.object(fk_drift,'source_identity',return_value={'sha256':'fixed'}), \
                 patch.object(fk_drift,'environment_identity',side_effect=environment), \
                 patch.object(fk_drift,'resolve_model',return_value={'path':'/snapshot','revision':'fixed'}), \
                 patch.object(fk_drift.subprocess,'check_output',return_value='GPU-test'), \
                 patch.object(sys,'argv',argv):
                self.assertEqual(fk_drift.main(),2)
                manifest=json.loads((root/'audit/manifest.json').read_text())
                self.assertEqual(len(manifest['comparisons']),1)
                self.assertEqual(manifest['runs'][0]['command'][1], 'validate')
                self.assertNotIn('bench_vllm.py', ' '.join(manifest['runs'][0]['command']))
                historical = root/'audit'/manifest['run_id']/'manifest.json'
                self.assertTrue(historical.exists())
                self.assertIn('--save-inputs-json',manifest['runs'][0]['command'])
                self.assertIn('--inputs-json',manifest['runs'][1]['command'])
                self.assertEqual(fk_drift.main(),2)
                self.assertEqual((root/'audit/launches.txt').read_text().count('launch'),2)
                cached = json.loads((root/'audit/manifest.json').read_text())
                self.assertTrue(all(r['status'] == 'cached' and r['command'] for r in cached['runs']))
                argv[argv.index('--num-seqs')+1]='9'
                self.assertEqual(fk_drift.main(),2)
                self.assertEqual((root/'audit/launches.txt').read_text().count('launch'),4)
                self.assertEqual(json.loads(historical.read_text())['config']['num_seqs'],8)


if __name__ == '__main__': unittest.main()
