import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_scenarios as suite
REPO=ROOT.parent


class ScenarioTests(unittest.TestCase):
    def test_full_plan_is_offline_and_routes_all_standard_vllm(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')):
            plan=suite.plan_scenarios(REPO,'full')
        selected=[r for r in plan['rows'] if r['status']=='planned']
        self.assertEqual(len(selected),15)
        mixtral=next(r for r in selected if 'Mixtral' in r['scenario']['model'])
        self.assertEqual(mixtral['scenario']['tp'],2)
        self.assertEqual(mixtral['workloads'],['mixed','long-context','single-request','fixed-batch-32'])
        glm=next(r for r in selected if 'GLM' in r['scenario']['model'])
        self.assertEqual(glm['scenario']['kv_cache_dtype'],'fp8_e4m3')
        self.assertEqual(len(next(r for r in selected if 'Omni' in r['scenario']['model'])['workloads']),8)
        self.assertTrue(all(r['reason'] for r in plan['rows'] if r['status']=='skipped'))

    def test_bad_plan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.yaml';p.write_text(json.dumps({'scenarios':[{'model':'Qwen/Qwen2.5-0.5B-Instruct','tp':1,'dtype':'bfloat16','workloads':['LLM.mixed','LLM.mixed']}]}))
            with self.assertRaises(ValueError):suite.plan_scenarios(REPO,str(p))

    def test_disk_guard_kills_process(self):
        from drift_runner import run_command
        with tempfile.TemporaryDirectory() as tmp:
            result=run_command([sys.executable,'-c','import time;time.sleep(60)'],Path(tmp)/'log',30,disk_root=tmp,minimum_free_bytes=10**30)
            self.assertEqual(result['status'],'disk-full')

    def test_media_freezes_and_rejects_corruption(self):
        sys.path.insert(0,str(REPO))
        from fastkernels.validate.media_inputs import frozen_media,media_manifest
        calls=[]
        @frozen_media
        def load(name,n,seed):
            calls.append(1);return [{'payload':[1,2,3]}]
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'FASTKERNELS_MEDIA_CACHE':tmp}):
            self.assertEqual(load('data',1,42),load('data',1,42))
            self.assertEqual(len(calls),1)
            self.assertEqual(len(media_manifest()),1)
            next(Path(tmp).glob('*.pickle')).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):load('data',1,42)

    def fake_run(self,command,log,timeout,env=None,*rest):
        if '-c' in command:
            if getattr(self,'prep_status',None):
                suite.write_json(Path(command[-4]),{'status':self.prep_status,'reason':'preflight evidence'})
                return {'status':'ok'}
            meta=Path(command[-4]);suite.write_json(meta,{'revision':'pinned-sha','path':'/fake/model','weight_bytes':10});return {'status':'ok'}
        scenario=json.loads(Path(command[2]).read_text())['scenarios'][0]
        ref='candidate' if '/candidate' in command[command.index('--vllm-python')+1] else 'baseline'
        self.calls.append(ref)
        if self.failures:
            self.failures-=1;return {'status':'failed'}
        names=['mixed','long-context'];output=Path(command[command.index('--output-dir')+1]);job=output/'000_bench_vllm';job.mkdir(parents=True)
        flag='--inputs-json' if '--inputs-json' in command else '--save-inputs-json';frozen=Path(command[command.index(flag)+1])
        if not frozen.exists():suite.write_json(frozen,{'sha256':'same','payload':{'throughput':[{'name':n,'prompt_token_ids':[[1]],'output_lens':[64]} for n in names],'latency':[{'name':'single-request','num_iters':2}]}})
        raw={'throughput':[{'name':n,'outputs':[{'token_ids':list(range(64))}],'total_output_tokens':64,'elapsed':1,'warmup_iters':1} for n in names]}
        for engine in ['vllm','fastkernels']:suite.write_json(job/(engine+'_raw.json'),{'raw':raw})
        result={'input_sha256':'same','scenarios':[{'scenario':n,'fastkernels_tok_per_s':64,'vllm_tok_per_s':64} for n in names],
                'latency_scenarios':[{'scenario':'single-request','vllm_median_s':1,'fastkernels_median_s':1,'vllm_latencies':[1,1],'fastkernels_latencies':[1,1]}]}
        suite.write_json(job/'results.json',result);(job/'run.log').write_text('fake evidence')
        suite.write_json(output/'summary.json',{'run':{'status':'PASS'},'models':[{'status':'PASS','harness':'bench_vllm'}]})
        return {'status':'ok'}

    def exercise(self,tmp,rows=1,extra=()):
        p=Path(tmp);scenario=p/'scenario.yaml'
        scenario.write_text(json.dumps({'scenarios':[{'model':f'Qwen/Qwen2.5-{i+1}B-Instruct','tp':1,'dtype':'bfloat16','workloads':['LLM.mixed','LLM.long_context','LLM.single_request']} for i in range(rows)]}))
        args=['--scenarios',str(scenario),'--fk-repo',str(REPO),'--baseline-python','/baseline','--candidate-python','/candidate','--workdir',str(p/'results'),'--data-root',str(p/'data'),'--repeats','1','--latency-iters','2','--min-free-gb','0']
        def hardware(command,**kw):return '' if '--query-compute-apps=pid' in command else 'GPU-test, H100, 80000, driver\n'
        def environment(py,*a):return {'vllm':'0.28.0' if str(py)=='/candidate' else '0.26.0','pin_memory':True,'packages':[('torch','2.11.0'),('vllm','0.26.0'),('transformers','5.14.1')]}
        with patch.object(suite,'inherit_hf_auth'),patch.object(suite,'run_command',side_effect=self.fake_run),patch.object(suite,'environment_identity',side_effect=environment),patch.object(suite,'source_identity',return_value={'sha256':'fixed'}),patch.object(suite.subprocess,'check_output',side_effect=hardware):
            rc=suite.main(args+list(extra))
        return rc,json.loads((p/'results/status.json').read_text())

    def test_success_resume_and_corrupt_evidence_rerun(self):
        self.calls=[];self.failures=0
        with tempfile.TemporaryDirectory() as tmp:
            rc,state=self.exercise(tmp);self.assertEqual(rc,0);self.assertEqual(len(self.calls),2)
            rc,state=self.exercise(tmp);self.assertEqual(rc,0);self.assertEqual(len(self.calls),2)
            raw=next((Path(tmp)/'results').glob('*/000-*/r0-baseline/vllm_raw.json'));raw.write_text('{}')
            rc,state=self.exercise(tmp);self.assertEqual(rc,0);self.assertEqual(len(self.calls),3)

    def test_retry_then_continue_after_failed_model(self):
        self.calls=[];self.failures=2
        with tempfile.TemporaryDirectory() as tmp:
            rc,state=self.exercise(tmp,rows=2)
            self.assertEqual(rc,2);self.assertEqual([r['status'] for r in state['rows']],['failed','passed'])
            self.assertEqual(len(self.calls),4)
            self.assertEqual(len(state['rows'][0]['attempt_failures']),2)

    def test_preflight_failure_is_incomplete_without_cli_attempts(self):
        self.calls=[];self.failures=0
        for status in ('blocked','unsupported'):
            self.prep_status=status
            with tempfile.TemporaryDirectory() as tmp:
                rc,state=self.exercise(tmp)
                self.assertEqual(rc,2)
                self.assertEqual(state['status'],'INCOMPLETE')
                self.assertEqual(state['rows'][0]['status'],status)
                self.assertEqual(self.calls,[])

    def test_latency_only_and_invalid_samples(self):
        import copy
        old={'input_sha256':'same','latency_scenarios':[{'scenario':'single-request',
             'vllm_latencies':[1,2,3], 'fastkernels_latencies':[2,3,4],
             'vllm_median_s':2, 'fastkernels_median_s':3}]}
        new=copy.deepcopy(old)
        self.assertEqual(len(suite.pair_metrics(old,new,32,['single-request'])),1)
        new['latency_scenarios'][0]['vllm_latencies'][0]=float('nan')
        with self.assertRaises(ValueError):suite.pair_metrics(old,new,32,['single-request'])

    def test_explicit_gpu_skip(self):
        self.calls=[];self.failures=0
        original=suite.plan_scenarios
        def multi(*args):
            plan=original(*args)
            plan['rows'][0]['scenario']['tp']=2
            return plan
        with tempfile.TemporaryDirectory() as tmp,patch.object(suite,'plan_scenarios',side_effect=multi):
            rc,state=self.exercise(tmp,rows=2,extra=['--skip-insufficient-gpus'])
            self.assertEqual(rc,0)
            self.assertEqual([r['status'] for r in state['rows']],['skipped','passed'])
            self.assertIn('only 1 GPUs',state['rows'][0]['reason'])
            self.assertEqual(len(self.calls),2)

    def test_media_preparation_uses_exact_worker_cache_keys(self):
        from fastkernels.validate.media_inputs import preload_media
        from unittest.mock import Mock
        loader=Mock()
        throughput=[{'dataset':'d','dataset_split':'test','num_seqs':8},
                    {'prompt_token_ids':[[1]],'dataset':'text'}]
        latency=[{'dataset':'d','dataset_split':'test','batch_size':32}]
        preload_media(throughput,latency,42,loader)
        self.assertEqual(loader.call_args_list, [unittest.mock.call('d','test',8,42), unittest.mock.call('d','test',1,42)])
        loader.reset_mock()
        preload_media(throughput,latency,42,loader,whisper=True)
        self.assertEqual(loader.call_args_list, [unittest.mock.call('d','test',8,42), unittest.mock.call('d','test',32,242)])
