"""Paired release validation, scheduled by validate's shared Ray runner."""

from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from fastkernels import CACHE_DIR
from . import (
    _parser,
    _resolve_validate_scenarios,
    _resolve_run_root,
    _detect_gpus,
    _REPO_ROOT,
)
from .drift_results import (
    digest,
    file_digest,
    write_json,
    source_identity,
    read_validation,
    pair_metrics,
)


def check_existing_gpu_memory(output, limit):
    """Require idle GPUs by default; optionally tolerate a bounded shared load."""
    totals = {}
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        uuid, pid, memory = (item.strip() for item in row)
        if limit == 0:
            raise ValueError("Selected GPUs already have compute processes; use an exclusive allocation")
        try:
            used = int(memory)
            if used < 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"Cannot determine existing GPU memory for PID {pid}: {memory}") from None
        totals[uuid] = totals.get(uuid, 0) + used
    for uuid, used in totals.items():
        if used > limit:
            raise ValueError(f"Existing compute processes on {uuid} use {used} MiB > {limit} MiB")
    return totals


def configure_job(job, args, root):
    # Drift verifies phase artifacts itself; ordinary harness resume is redundant.
    if "--resume" in job["cmd"]:
        job["cmd"].remove("--resume")
    job["drift"] = dict(args.drift_config)
    job["cache_root"] = str(
        Path(args.drift_config["data_root"])
        / "compiler"
        / digest(str(root))[:16]
        / str(job["index"])
    )


def _phase(
    job,
    name,
    command,
    artifact,
    env,
    timeout,
    stall_timeout,
    repo_root,
    gpus,
    numactl_mode,
):
    from .ray_runner import _run_job_subprocess

    root = Path(job["run_dir"]) / name
    phase = dict(
        job,
        cmd=command,
        run_dir=str(root),
        log_path=str(root / "run.log"),
        artifact=artifact,
        env=env,
        cache_root=str(Path(job["cache_root"]) / name),
        disk_root=job["drift"]["data_root"],
        min_free_bytes=int(job["drift"]["min_free_gb"] * 2**30),
    )
    phase.pop("drift", None)
    result = _run_job_subprocess(
        phase, timeout, stall_timeout, repo_root, gpus, numactl_mode
    )
    if result["status"] != "PASS":
        from .ray_runner import _tail

        raise RuntimeError(
            f"{name}: {result['status']}; {result['log_path']}\n"
            + _tail(Path(result["log_path"]), 12)
        )
    return root


def _completed(output, frozen, excluded, identity):
    try:
        stamp = json.loads((output / "complete.json").read_text())
        if stamp["identity"] != identity or stamp["inputs"] != file_digest(frozen):
            return False
        if not all(file_digest(output / n) == sha for n, sha in stamp["files"].items()):
            return False
        read_validation(output, frozen, excluded)
        return True
    except (OSError, ValueError, KeyError):
        return False


def run_job(job, timeout, stall_timeout, repo_root, gpus, numactl_mode):
    """One Ray allocation owns both references for every repeat of a model."""
    config = job["drift"]
    root = Path(job["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "inputs.json"
    excluded = config["exclude_fastkernels"]
    identity = digest(
        [config, job["cmd"], job["num_cpus"], os.environ.get("CUDA_VISIBLE_DEVICES")]
    )
    scratch = (
        Path(config["data_root"])
        / ("run-" + config["identity"][:16])
        / str(job["index"])
    )
    if scratch.is_symlink():
        raise ValueError("Private scratch must not be a symlink")
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = {
        "HF_HOME": str(scratch / "hf"),
        "HF_HUB_CACHE": str(scratch / "hf/hub"),
        "HF_DATASETS_CACHE": str(scratch / "hf/datasets"),
        "FASTKERNELS_MEDIA_CACHE": str(scratch / "media"),
        "FASTKERNELS_CANDIDATE_DIR": str(scratch / "empty-candidates"),
    }
    Path(env["FASTKERNELS_CANDIDATE_DIR"]).mkdir(exist_ok=True)
    if any(Path(env["FASTKERNELS_CANDIDATE_DIR"]).iterdir()):
        raise ValueError("Candidate directory is not empty")
    state = dict(
        index=job["index"],
        name=job["name"],
        status="RUNNING",
        pairs=[],
        warnings=[],
        attempt_failures=[],
        physical_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        log_path=job["log_path"],
        run_dir=str(root),
        num_cpus=job["num_cpus"],
    )
    started = time.monotonic()

    def persist():
        write_json(root / "results.json", state)

    def phase(name, command, artifact="results.json"):
        return _phase(
            job,
            name,
            command,
            artifact,
            env,
            timeout,
            stall_timeout,
            repo_root,
            gpus,
            numactl_mode,
        )

    persist()
    try:
        metadata = root / "prepare/model.json"
        preparation = root / "prepare.json"
        write_json(
            preparation,
            dict(
                model=job["name"],
                metadata=str(metadata),
                data_root=str(scratch),
                reserve=int(config["min_free_gb"] * 2**30),
                interpreters=config["interpreters"],
                lock=str(Path(config["data_root"]) / ".download.lock"),
            ),
        )
        all_done = all(
            _completed(root / f"r{rep}-{label}", frozen, excluded, identity)
            for rep in range(config["repeats"])
            for label in ("baseline", "candidate")
        )
        if not all_done:
            for attempt in range(config["retries"] + 1):
                try:
                    phase(
                        "prepare",
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "fastkernels.validate.drift_prepare",
                            str(preparation),
                        ],
                        "model.json",
                    )
                    break
                except RuntimeError as exc:
                    state["attempt_failures"].append(str(exc))
                    persist()
                    if attempt == config["retries"]:
                        raise
        model = json.loads(metadata.read_text())
        state["model_revision"] = model.get("revision")
        state["compatibility"] = model.get("compatibility", {})
        if model.get("status") in ("blocked", "unsupported"):
            raise ValueError(model["reason"])
        for label, check in state["compatibility"].items():
            if check.get("warning"):
                state["warnings"].append(label + ": " + check["warning"])
        for rep in range(config["repeats"]):
            pair = {}
            order = (
                ("baseline", "candidate") if rep % 2 == 0 else ("candidate", "baseline")
            )
            for label in order:
                name = f"r{rep}-{label}"
                output = root / name
                if not _completed(output, frozen, excluded, identity):
                    for attempt in range(config["retries"] + 1):
                        if output.exists():
                            output.rename(root / (name + f"-failed-{time.time_ns()}"))
                        command = list(job["cmd"])
                        command[command.index("--model") + 1] = model["path"]
                        command[command.index("--output-dir") + 1] = str(output)
                        command += [
                            "--vllm-python",
                            config["interpreters"][label],
                            "--inputs-json"
                            if frozen.exists()
                            else "--save-inputs-json",
                            str(frozen),
                        ]
                        with Path(job["log_path"]).open("a") as log:
                            log.write(
                                f"{name}, attempt {attempt + 1}: {output}/run.log\n"
                            )
                        try:
                            phase(name, command)
                            read_validation(output, frozen, excluded)
                            files = ["results.json", "vllm_raw.json"]
                            if not excluded:
                                files.append("fastkernels_raw.json")
                            write_json(
                                output / "complete.json",
                                dict(
                                    identity=identity,
                                    inputs=file_digest(frozen),
                                    files={n: file_digest(output / n) for n in files},
                                ),
                            )
                            break
                        except (RuntimeError, ValueError, OSError, KeyError) as exc:
                            state["attempt_failures"].append(str(exc))
                            persist()
                            if attempt == config["retries"]:
                                raise
                pair[label] = read_validation(output, frozen, excluded)
            metrics = pair_metrics(
                pair["baseline"],
                pair["candidate"],
                config["alignment_floor"],
                job["workloads"],
                excluded,
            )
            state["pairs"].append(metrics)
            if any(
                max(m.get("fk_stability", 1), 1 / m.get("fk_stability", 1)) > 1.05
                for m in metrics
            ):
                state["warnings"].append(
                    "Fixed FastKernels timing varied by over 5%; investigate contention/noise."
                )
            persist()
        state["status"] = "PASS"
    except Exception as exc:
        state.update(status="FAIL", reason=str(exc))
        with Path(job["log_path"]).open("a") as log:
            log.write(str(exc) + "\n")
    finally:
        state["elapsed_s"] = time.monotonic() - started
        persist()
        # Only delete this audit's private model cache. Preserve failed media for replay.
        if not config["keep_models"]:
            shutil.rmtree(scratch / "hf", ignore_errors=True)
        if state["status"] == "PASS" and not config["keep_models"]:
            shutil.rmtree(scratch / "media", ignore_errors=True)
    return state


def main(argv=None):
    parser = _parser(drift=True)
    args = parser.parse_args(argv)
    if (
        min(args.repeats, args.warmup_iters, args.latency_iters, args.timeout) < 1
        or args.retries < 0
    ):
        parser.error("Counts and timeout must be positive; retries must be nonnegative")
    if not 0 <= args.min_free_gb < float(
        "inf"
    ) or not 0 <= args.alignment_floor < float("inf"):
        parser.error("Disk reserve and alignment floor must be finite and nonnegative")
    if args.max_existing_gpu_memory_mib < 0:
        parser.error("--max-existing-gpu-memory-mib must be nonnegative")
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests must be positive")
    if not all(
        re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)", v)
        for v in (args.baseline, args.candidate)
    ):
        parser.error("Expected explicit release versions, for example 0.18.0 0.26.0")
    try:
        scenarios = _resolve_validate_scenarios(args.scenarios)
        if not args.run_id and not args.output_dir:
            args.run_id = f"drift-{args.baseline}-{args.candidate}"
        _, root = _resolve_run_root(args)
        root = root.resolve()
        args.data_root = (
            (
                args.data_root
                or Path(
                    os.environ.get("DRIFT_SCRATCH", str(CACHE_DIR / "drift-models"))
                )
            )
            .expanduser()
            .absolute()
        )
        args.env_root = (
            (args.env_root or CACHE_DIR / "reference-envs").expanduser().absolute()
        )
        args.drift_config = {"data_root": str(args.data_root)}
        if args.dry_run:
            from . import _harness_for, _scenario_workloads

            for scenario in scenarios:
                h = _harness_for(
                    scenario.hf_name, getattr(scenario, "draft_model", None)
                )
                print(
                    f"{'RUN' if h == 'bench_vllm' else 'SKIP'} {scenario.hf_name}: tp={scenario.tp}; {h}; "
                    + ", ".join(_scenario_workloads(scenario))
                )
            print(
                f"References: {args.baseline} → {args.candidate}; FastKernels: {not args.exclude_fastkernels}"
            )
            print(
                f"Report: {root / 'report.md'}; environments: {args.env_root}; scratch: {args.data_root}"
            )
            return 0
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from .compatibility import inherit_hf_auth
            from .environments import reference_python, environment_identity
            from .ray_runner import run_validation

            inherit_hf_auth()
            if (root / "drift.json").exists() and not args.resume:
                raise ValueError(
                    f"{root} already exists; pass --resume or choose a new --output-dir"
                )
            interpreters, environments = {}, {}
            for label, version in (
                ("baseline", args.baseline),
                ("candidate", args.candidate),
            ):
                interpreters[label], environments[label] = reference_python(
                    version, getattr(args, label + "_python"), args.env_root
                )
            environments["fk"] = environment_identity(sys.executable)
            # The host always prepares workloads; included FK must match its checkout.
            if not args.exclude_fastkernels:
                import tomllib

                deps = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())[
                    "project"
                ]["dependencies"]
                packages = dict(environments["fk"]["packages"])
                for dep in deps:
                    match = re.fullmatch(r"(torch|vllm|transformers)==(.+)", dep)
                    if match and packages.get(match[1], "").split("+")[0] != match[2]:
                        raise ValueError(
                            "Fixed FastKernels environment does not match " + dep
                        )
                interpreters["fk"] = sys.executable
            gpus = _detect_gpus(args.gpus)
            if not gpus:
                raise ValueError("No GPUs selected")
            busy = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    ",".join(gpus),
                    "--query-compute-apps=gpu_uuid,pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=20,
            ).strip()
            existing = check_existing_gpu_memory(busy, args.max_existing_gpu_memory_mib)
            if args.max_existing_gpu_memory_mib:
                print(f"Shared-GPU startup limit: {args.max_existing_gpu_memory_mib} MiB per GPU; existing usage: {existing}. Timings may be affected.", flush=True)
            write_json(root / "gpu-startup.json", {"existing_memory_mib": existing, "limit_mib": args.max_existing_gpu_memory_mib})
            hardware = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    ",".join(gpus),
                    "--query-gpu=uuid,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=20,
            )
            config = {
                k: getattr(args, k)
                for k in (
                    "baseline",
                    "candidate",
                    "exclude_fastkernels",
                    "repeats",
                    "retries",
                    "alignment_floor",
                    "min_free_gb",
                    "max_existing_gpu_memory_mib",
                    "keep_models",
                    "warmup_iters",
                    "latency_iters",
                    "seed",
                    "max_requests",
                    "max_layers",
                    "reference_patches",
                    "numactl_mode",
                )
            }
            config.update(
                data_root=str(args.data_root),
                interpreters=interpreters,
                environments=environments,
                source=source_identity(_REPO_ROOT),
                hardware=hardware,
                gpus=gpus,
                scenarios=[str(s) for s in scenarios],
                output_dir=str(root),
                controls={
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(
                        ("FASTKERNELS_", "VLLM_", "CUDA_", "NCCL_", "OMP_", "TORCH_")
                    )
                    and not any(secret in k for secret in ("TOKEN", "KEY", "SECRET"))
                },
            )
            config["identity"] = digest(config)
            if (root / "drift.json").exists() and digest(
                json.loads((root / "drift.json").read_text())
            ) != digest(config):
                raise ValueError(
                    "Resume configuration/source/environments/hardware changed; choose a new --output-dir"
                )
            args.drift_config = config
            args.data_root.mkdir(parents=True, exist_ok=True)
            write_json(root / "drift.json", config)
            print(f"Drift report: {root / 'report.md'}", flush=True)

            def interrupt(signum, frame):
                raise KeyboardInterrupt

            previous = signal.signal(signal.SIGTERM, interrupt)
            try:
                return run_validation(scenarios, args, gpus, root)
            finally:
                signal.signal(signal.SIGTERM, previous)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
