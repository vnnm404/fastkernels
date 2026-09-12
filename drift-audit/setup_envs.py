#!/usr/bin/env python3
"""Provision the tested 0.18 → 0.26 audit environments without model downloads.

Requires Python 3.11+, uv, and a Linux CUDA build environment (nvcc, compiler,
Python development headers, git). Environments belong on large scratch storage.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tomllib


def commands(repo, root):
    deps = tomllib.loads((repo/'pyproject.toml').read_text())['project']['dependencies']
    # Only the standard-vLLM harness and its FK model engines. Other full.yaml
    # reference engines (OpenFold, SAM, SGLang, etc.) are outside this audit.
    names = {'torch','torchvision','torchaudio','vllm','numpy','pyyaml','transformers',
             'huggingface_hub','safetensors','tqdm','blobfile','halo','quack-kernels',
             'ray','decord','diffusers','peft','einops','rotary-embedding-torch',
             'fastsafetensors','flash-linear-attention','flash-attn'}
    selected = [d for d in deps if re.match(r'[\w-]+',d).group() in names]
    if 'vllm==0.26.0' not in selected:
        raise ValueError('This recipe is validated for FK vLLM 0.26.0; update the recipe for new core pins')
    fk = root/'fk-env/bin/python'; old = root/'vllm-0.18-env/bin/python'
    build = [d for d in deps if d.startswith('deep-gemm ')]
    return [
        ['uv','venv','--allow-existing','--python','3.12',str(fk.parents[1])],
        ['uv','pip','install','--python',str(fk),*selected,'datasets==3.6.0','av==18.1.0','setuptools','wheel','ninja','packaging'],
        ['uv','pip','install','--python',str(fk),'--no-build-isolation',*build],
        ['uv','pip','install','--python',str(fk),'--no-deps','--no-build-isolation','-e',str(repo)],
        ['uv','venv','--allow-existing','--python','3.12',str(old.parents[1])],
        ['uv','pip','install','--python',str(old),'vllm==0.18.0','transformers==4.57.6','fastsafetensors','datasets==3.6.0','av==18.1.0'],
        [str(fk),'-c','import vllm, torch, transformers, datasets, pyarrow, av, flash_attn, deep_gemm, fastsafetensors; from fastkernels.tasks.baseline.L2.vision_attention import VisionAttention; print("FK imports passed")'],
        [str(old),'-c','import vllm, torch, transformers, datasets, pyarrow, av, fastsafetensors; print("Reference imports passed")'],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-root',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true',help='Print commands only; no installs or downloads')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.env_root.expanduser().resolve()
    if root == repo or repo in root.parents:
        parser.error('--env-root must be outside the source checkout')
    plan = commands(repo,root)
    if not args.dry_run:
        if os.uname().sysname != 'Linux':
            parser.error('Provisioning requires Linux; use --dry-run to inspect elsewhere')
        root.mkdir(parents=True,exist_ok=True)
    for command in plan:
        print(shlex.join(command),flush=True)
        if not args.dry_run:
            subprocess.run(command,check=True,env={**os.environ,'UV_NO_CACHE':'1'})
    exports = {'FK_PYTHON':root/'fk-env/bin/python',
               'OLD_VLLM_PYTHON':root/'vllm-0.18-env/bin/python',
               'NEW_VLLM_PYTHON':root/'fk-env/bin/python'}
    env_file = root/'drift.env.sh'
    if not args.dry_run:
        env_file.write_text(''.join(f'export {k}={shlex.quote(str(v))}\n' for k,v in exports.items()))
    print('After successful setup: source '+shlex.quote(str(env_file)))


if __name__ == '__main__':
    main()
