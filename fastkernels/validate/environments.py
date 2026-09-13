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
    deps = tomllib.loads((repo / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    # Only the standard-vLLM harness and its FK model engines. Other full.yaml
    # reference engines (OpenFold, SAM, SGLang, etc.) are outside this audit.
    names = {
        "torch",
        "torchvision",
        "torchaudio",
        "vllm",
        "numpy",
        "pyyaml",
        "transformers",
        "huggingface_hub",
        "safetensors",
        "tqdm",
        "blobfile",
        "halo",
        "quack-kernels",
        "ray",
        "decord",
        "diffusers",
        "peft",
        "einops",
        "rotary-embedding-torch",
        "fastsafetensors",
        "flash-linear-attention",
        "flash-attn",
    }
    selected = [d for d in deps if re.match(r"[\w-]+", d).group() in names]
    if "vllm==0.26.0" not in selected:
        raise ValueError(
            "This recipe is validated for FK vLLM 0.26.0; update the recipe for new core pins"
        )
    fk = root / "fk-env/bin/python"
    old = root / "vllm-0.18.0/bin/python"
    build = [d for d in deps if d.startswith("deep-gemm ")]
    return [
        ["uv", "venv", "--allow-existing", "--python", "3.12", str(fk.parents[1])],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(fk),
            *selected,
            "datasets==3.6.0",
            "av==18.1.0",
            "setuptools",
            "wheel",
            "ninja",
            "packaging",
        ],
        ["uv", "pip", "install", "--python", str(fk), "--no-build-isolation", *build],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(fk),
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(repo),
        ],
        ["uv", "venv", "--allow-existing", "--python", "3.12", str(old.parents[1])],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(old),
            "vllm==0.18.0",
            "transformers==4.57.6",
            "fastsafetensors",
            "datasets==3.6.0",
            "av==18.1.0",
        ],
        [
            str(fk),
            "-c",
            'import vllm, torch, transformers, datasets, pyarrow, av, flash_attn, deep_gemm, fastsafetensors; from fastkernels.tasks.baseline.L2.vision_attention import VisionAttention; print("FK imports passed")',
        ],
        [
            str(old),
            "-c",
            'import vllm, torch, transformers, datasets, pyarrow, av, fastsafetensors; print("Reference imports passed")',
        ],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; no installs or downloads",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.env_root.expanduser().resolve()
    if root == repo or repo in root.parents:
        parser.error("--env-root must be outside the source checkout")
    plan = commands(repo, root)
    if not args.dry_run:
        if os.uname().sysname != "Linux":
            parser.error(
                "Provisioning requires Linux; use --dry-run to inspect elsewhere"
            )
        root.mkdir(parents=True, exist_ok=True)
    for command in plan:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, env={**os.environ, "UV_NO_CACHE": "1"})
    if not args.dry_run:
        (root / "vllm-0.18.0/.ready").touch()
    print(
        "After successful setup: source "
        + shlex.quote(str(root / "fk-env/bin/activate"))
    )


def environment_identity(python):
    """Record installed dependencies without creating CUDA contexts in the driver."""
    import json

    code = """import importlib.metadata as m, json, sys
from urllib.parse import urlsplit, urlunsplit
sources = {}
for d in m.distributions():
    raw = d.read_text('direct_url.json')
    if raw:
        info = json.loads(raw); u = urlsplit(info.get('url', ''))
        info['url'] = urlunsplit((u.scheme, u.hostname or '', u.path, '', ''))
        sources[d.metadata['Name']] = info
print(json.dumps({'python': sys.version, 'executable': sys.executable,
 'packages': sorted((d.metadata['Name'].lower().replace('_','-'), d.version) for d in m.distributions()),
 'vllm': m.version('vllm'), 'sources': sources}))"""
    result = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def reference_python(version, explicit, root):
    """Never install into the host. Unknown releases require a supplied venv."""
    import fcntl
    import sys

    if explicit:
        python = Path(explicit).expanduser().absolute()
    elif environment_identity(sys.executable)["vllm"] == version:
        python = Path(sys.executable)
    else:
        if version not in ("0.18.0", "0.26.0"):
            raise ValueError(
                f"No setup recipe for {version}; supply its --baseline-python or --candidate-python"
            )
        root.mkdir(parents=True, exist_ok=True)
        python = root / ("vllm-" + version) / "bin/python"
        with (root / (version + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = python.parents[1] / ".ready"
            if not marker.exists():
                subprocess.run(
                    [
                        "uv",
                        "venv",
                        "--allow-existing",
                        "--python",
                        "3.12",
                        str(python.parents[1]),
                    ],
                    check=True,
                )
                transformer = "4.57.6" if version == "0.18.0" else "5.14.1"
                subprocess.run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "vllm==" + version,
                        "transformers==" + transformer,
                        "fastsafetensors",
                        "datasets==3.6.0",
                        "av==18.1.0",
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        str(python),
                        "-c",
                        "import vllm, torch, transformers, datasets, pyarrow, av, fastsafetensors",
                    ],
                    check=True,
                    timeout=120,
                )
                marker.touch()
    identity = environment_identity(python)
    if identity["vllm"] != version:
        raise ValueError(f"{python}: expected vLLM {version}, found {identity['vllm']}")
    return str(python), identity


if __name__ == "__main__":
    main()
