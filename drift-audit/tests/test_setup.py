import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('setup_envs', ROOT/'setup_envs.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_recipe_keeps_reference_separate_and_uses_checkout_pins(self):
        plan = setup.commands(ROOT.parent,Path('/scratch/envs with spaces'))
        fk = plan[1]
        self.assertIn('vllm==0.26.0',fk)
        self.assertIn('torch==2.11.0',fk)
        self.assertIn('transformers==5.14.1',fk)
        self.assertTrue(any('cu130torch2.11' in x for x in fk))
        old = plan[5]
        self.assertIn('vllm==0.18.0',old)
        self.assertIn('transformers==4.57.6',old)
        self.assertNotEqual(fk[4],old[4])
        self.assertNotIn('-e',old)

    def test_setup_dry_run_does_not_create_environment(self):
        with tempfile.TemporaryDirectory() as td:
            envs = Path(td)/'new environments'
            result = subprocess.run([sys.executable,str(ROOT/'setup_envs.py'),
                                     '--env-root',str(envs),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('drift.env.sh',result.stdout)
            self.assertFalse(envs.exists())

    def test_default_scenario_path_works_from_other_directory(self):
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run([sys.executable,str(ROOT/'run_scenarios.py'),'--dry-run'],
                                    cwd=td,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('bench_vllm',result.stdout)

    def test_launcher_dry_run_from_other_directory(self):
        with tempfile.TemporaryDirectory() as td:
            env = {**os.environ, 'FK_PYTHON':sys.executable,
                   'OLD_VLLM_PYTHON':sys.executable,'NEW_VLLM_PYTHON':sys.executable,
                   'DRIFT_SCRATCH':str(Path(td)/'scratch')}
            result = subprocess.run(['bash',str(ROOT/'run_overnight.sh'),'--dry-run'],
                                    cwd=td,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('bench_vllm',result.stdout)
            self.assertFalse((Path(td)/'scratch').exists())
