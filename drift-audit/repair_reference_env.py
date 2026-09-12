#!/usr/bin/env python3
"""Repair known data-loader dependencies without changing benchmark core pins.

Run explicitly during provisioning, before collecting any timing results.
Usage: python3 repair_reference_env.py /opt/old-env/bin/python
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CORE = ('vllm', 'torch', 'transformers')
EXTRAS = ('datasets==3.6.0', 'av==18.1.0')


def core_versions(python):
    code = 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in '+repr(CORE)+'}))'
    return json.loads(subprocess.check_output([python, '-c', code], text=True))


def repair(python):
    before = core_versions(python)
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('Install uv before repairing reference dependencies')
    with tempfile.TemporaryDirectory(prefix='fk-reference-pins-') as tmp:
        pins = Path(tmp)/'constraints.txt'
        pins.write_text(''.join(f'{name}=={version}\n' for name,version in before.items()))
        subprocess.run([uv, 'pip', 'install', '--python', python, '--constraint', str(pins), *EXTRAS], check=True)
    if core_versions(python) != before:
        raise RuntimeError('Core versions changed; do not benchmark this environment')
    subprocess.run([python, '-c', 'import datasets,pyarrow,av; print("Data-loader imports passed")'], check=True)
    print(json.dumps({'core_versions_preserved': before, 'extras': EXTRAS}, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    repair(str(Path(sys.argv[1]).expanduser().absolute()))
