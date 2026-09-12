import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compatibility
import run_scenarios as suite


class HubError(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status, headers={'Retry-After':'0'})


class CompatibilityTests(unittest.TestCase):
    def test_different_concurrency_limits_cannot_be_compared(self):
        with self.assertRaisesRegex(ValueError, 'Concurrent sequence limits'):
            suite.pair_metrics({'max_num_seqs':128},{'max_num_seqs':512},32,[])

    def test_token_survives_private_cache_without_being_written(self):
        hub = SimpleNamespace(get_token=lambda:'test-secret')
        with patch.dict(sys.modules, {'huggingface_hub':hub}), patch.dict(os.environ, {}, clear=True):
            compatibility.inherit_hf_auth()
            os.environ['HF_HOME']='/different/private/cache'
            self.assertEqual(os.environ['HF_TOKEN'], 'test-secret')

    def test_retry_is_bounded_and_does_not_retry_denied_access(self):
        with patch.object(compatibility.time, 'sleep') as sleep:
            call=Mock(side_effect=[HubError(429), HubError(503), 'ok'])
            self.assertEqual(compatibility.hub_retry(call), 'ok')
            self.assertEqual(sleep.call_count, 2)
            call=Mock(side_effect=[TimeoutError('network timeout'),ConnectionError('reset'),'ok'])
            self.assertEqual(compatibility.hub_retry(call),'ok')
            call=Mock(side_effect=ValueError('bad config'))
            with self.assertRaises(ValueError):compatibility.hub_retry(call)
            self.assertEqual(call.call_count,1)
            for status, calls in [(403,1),(500,4)]:
                call=Mock(side_effect=HubError(status))
                with self.assertRaises(HubError):compatibility.hub_retry(call)
                self.assertEqual(call.call_count, calls)

    def test_preflight_blocks_download_and_can_resume_after_access_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'config.json'; config.write_text('{"architectures":["Gemma4ForConditionalGeneration"]}')
            metadata=root/'model.json'
            hub=SimpleNamespace(
                HfApi=lambda:SimpleNamespace(model_info=lambda *a,**kw:SimpleNamespace(sha='frozen',siblings=[SimpleNamespace(size=10,rfilename='model.safetensors')])),
                snapshot_download=Mock(return_value='/snapshot'),
                hf_hub_download=Mock(side_effect=HubError(403)))
            with patch.dict(sys.modules, {'huggingface_hub':hub}):
                suite.prepare_model('repo/model',metadata,root,0)
                self.assertEqual(json.loads(metadata.read_text())['status'],'blocked')
                hub.snapshot_download.assert_not_called()
                hub.hf_hub_download.side_effect=None;hub.hf_hub_download.return_value=str(config)
                def unsupported(command,*args,**kwargs):
                    Path(command[-1]).write_text(json.dumps({'status':'unsupported','reason':'Architecture absent'}))
                    return SimpleNamespace(returncode=0)
                with patch.object(suite.subprocess,'run',side_effect=unsupported):
                    suite.prepare_model('repo/model',metadata,root,0,interpreters={'baseline':'/python'})
                self.assertEqual(json.loads(metadata.read_text())['status'],'unsupported')
                hub.snapshot_download.assert_not_called()
                def supported(command,*args,**kwargs):
                    Path(command[-1]).write_text(json.dumps({'status':'ok'}));return SimpleNamespace(returncode=0)
                with patch.object(suite.subprocess,'run',side_effect=supported):
                    suite.prepare_model('repo/model',metadata,root,0,interpreters={'baseline':'/python'})
                self.assertEqual(json.loads(metadata.read_text())['path'],'/snapshot')
                self.assertNotIn('status',json.loads(metadata.read_text()))
                self.assertEqual(hub.snapshot_download.call_args.kwargs['revision'],'frozen')


if __name__=='__main__':
    unittest.main()
