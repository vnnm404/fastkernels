from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fastkernels.validate import (
    main,
    _parser,
    _resolve_validate_scenarios,
    ValidateScenario,
)
from fastkernels.validate import drift, ray_runner
from fastkernels.validate.drift_results import (
    read_validation,
    pair_metrics,
    write_json,
    write_summary,
)


def args(tmp_path, excluded=False):
    a = _parser(drift=True).parse_args(
        ["0.18.0", "0.26.0"] + (["--exclude-fastkernels"] if excluded else [])
    )
    a.drift_config = dict(
        baseline=a.baseline,
        candidate=a.candidate,
        exclude_fastkernels=excluded,
        repeats=2,
        retries=1,
        alignment_floor=32,
        min_free_gb=0,
        keep_models=False,
        identity="test",
        data_root=str(tmp_path / "scratch"),
        interpreters={"baseline": "/old/python", "candidate": "/new/python"},
    )
    return a


def job(tmp_path, excluded=False):
    a = args(tmp_path, excluded)
    scenario = ValidateScenario(
        "meta-llama/Llama-3.1-8B-Instruct", 1, "bfloat16", ("mixed", "single-request")
    )
    j = ray_runner._make_job(0, scenario, "bench_vllm", a, tmp_path / "results")
    j["num_cpus"] = 4
    Path(a.drift_config["data_root"]).mkdir()
    return j, scenario


def artifact(output, frozen, excluded=False, rate=64):
    output.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        write_json(
            frozen,
            {
                "sha256": "same",
                "payload": {
                    "throughput": [
                        {
                            "name": "mixed",
                            "prompt_token_ids": [[1]],
                            "output_lens": [64],
                        }
                    ],
                    "latency": [{"name": "single-request", "num_iters": 2}],
                },
            },
        )
    from fastkernels.validate.frozen_inputs import digest

    envelope = json.loads(frozen.read_text())
    envelope["sha256"] = digest(envelope["payload"])
    write_json(frozen, envelope)
    result = dict(
        input_sha256=envelope["sha256"],
        scenarios=[{"scenario": "mixed"}],
        latency_scenarios=[{"scenario": "single-request"}],
    )
    for engine in ("vllm",) if excluded else ("vllm", "fastkernels"):
        raw = {
            "throughput": [
                {
                    "name": "mixed",
                    "outputs": [{"token_ids": list(range(64))}],
                    "total_output_tokens": 64,
                    "elapsed": 64 / rate,
                    "warmup_iters": 1,
                }
            ],
            "latency": [
                {"name": "single-request", "latencies": [1.0, 1.0], "num_warmup": 3}
            ],
        }
        write_json(output / (engine + "_raw.json"), {"raw": raw})
        result["scenarios"][0][engine + "_tok_per_s"] = rate
        result["latency_scenarios"][0].update(
            {engine + "_latencies": [1.0, 1.0], engine + "_median_s": 1.0}
        )
    write_json(output / "results.json", result)


def fake_runner(calls, failures=None):
    def run(j, *rest):
        output = Path(j["run_dir"])
        output.mkdir(parents=True, exist_ok=True)
        cmd = j["cmd"]
        if "fastkernels.validate.drift_prepare" in cmd:
            write_json(
                output / "model.json",
                {"revision": "pinned", "path": "/model", "compatibility": {}},
            )
        else:
            calls.append(
                (
                    cmd[cmd.index("--vllm-python") + 1],
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                )
            )
            if failures and failures.pop():
                return {"status": "FAIL(timeout)", "log_path": j["log_path"]}
            flag = "--inputs-json" if "--inputs-json" in cmd else "--save-inputs-json"
            artifact(
                output, Path(cmd[cmd.index(flag) + 1]), "--exclude-fastkernels" in cmd
            )
        return {"status": "PASS"}

    return run


def test_cli_drift_defaults_and_offline_plan(monkeypatch, capsys):
    def forbidden(*a, **kw):
        raise AssertionError("dry run must not query GPUs, install, or download")

    monkeypatch.setattr(drift, "_detect_gpus", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    assert main(["drift", "0.18.0", "0.26.0", "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len([line for line in lines if line.startswith("RUN ")]) == 15
    assert len([line for line in lines if line.startswith("SKIP ")]) == 32
    assert "FastKernels: True" in "\n".join(lines)


def test_gpu_planning_and_shared_ray_dispatch(tmp_path, monkeypatch):
    a = args(tmp_path)
    jobs, results, cached = ray_runner._plan_jobs(
        _resolve_validate_scenarios("full"), a, 1, tmp_path, hopper=False
    )
    assert len(jobs) == 9
    assert len([v for v in results.values() if v == "SKIP(tp>gpus)"]) == 6
    assert not cached
    j = jobs[0]
    assert "fastkernels.validate.bench_vllm" in j["cmd"]
    assert "validate" not in j["cmd"]  # No nested CLI / Ray cluster.
    monkeypatch.setattr(drift, "run_job", lambda *a: {"status": "PASS", "job": a[0]})
    assert ray_runner._ray_run_job(j, 1, 1, ".", ["0"], "off")["job"] is j


@pytest.mark.parametrize("excluded", [False, True])
def test_pairs_reuse_allocation_resume_and_detect_corruption(
    tmp_path, monkeypatch, excluded
):
    j, scenario = job(tmp_path, excluded)
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(ray_runner, "_run_job_subprocess", fake_runner(calls))
    result = drift.run_job(j, 60, 60, ".", ["0", "1", "2", "3"], "off")
    assert result["status"] == "PASS", result
    assert calls == [
        ("/old/python", "3"),
        ("/new/python", "3"),
        ("/new/python", "3"),
        ("/old/python", "3"),
    ]
    assert len(result["pairs"]) == 2
    assert ("fk_old" in result["pairs"][0][0]) is not excluded
    assert drift.run_job(j, 60, 60, ".", ["3"], "off")["status"] == "PASS"
    assert len(calls) == 4
    (Path(j["run_dir"]) / "r0-baseline/vllm_raw.json").write_text("{}")
    assert drift.run_job(j, 60, 60, ".", ["3"], "off")["status"] == "PASS"
    assert len(calls) == 5
    assert list(Path(j["run_dir"]).glob("r0-baseline-failed-*"))


def test_retry_and_failed_model_evidence(tmp_path, monkeypatch):
    j, _ = job(tmp_path)
    monkeypatch.setattr(
        ray_runner, "_run_job_subprocess", fake_runner([], [True, True])
    )
    result = drift.run_job(j, 1, 1, ".", ["0"], "off")
    assert result["status"] == "FAIL"
    assert len(result["attempt_failures"]) == 2
    assert "timeout" in result["reason"]
    assert (
        json.loads((Path(j["run_dir"]) / "results.json").read_text())["status"]
        == "FAIL"
    )


def test_resume_flag_does_not_change_phase_identity(tmp_path):
    j, scenario = job(tmp_path)
    a = args(tmp_path)
    a.resume = True
    resumed = ray_runner._make_job(0, scenario, "bench_vllm", a, tmp_path / "results")
    assert resumed["cmd"] == j["cmd"]


@pytest.mark.parametrize("excluded", [False, True])
def test_result_checks_and_report(tmp_path, excluded):
    frozen = tmp_path / "inputs.json"
    artifact(tmp_path / "old", frozen, excluded)
    artifact(tmp_path / "new", frozen, excluded, rate=128)
    old = read_validation(tmp_path / "old", frozen, excluded)
    new = read_validation(tmp_path / "new", frozen, excluded)
    metrics = pair_metrics(old, new, 32, ["mixed", "single-request"], excluded)
    assert metrics[0]["drift"] == 2
    broken = copy.deepcopy(new)
    broken["vllm_raw"]["throughput"][0]["outputs"][0]["token_ids"][0] = -1
    with pytest.raises(ValueError, match="agreement"):
        pair_metrics(old, broken, 32, ["mixed", "single-request"], excluded)
    broken = copy.deepcopy(new)
    broken["vllm_raw"]["throughput"][0]["warmup_iters"] = 0
    with pytest.raises(ValueError, match="warmup"):
        pair_metrics(old, broken, 32, ["mixed", "single-request"], excluded)
    with pytest.raises(ValueError, match="coverage"):
        pair_metrics(old, new, 32, ["mixed"], excluded)
    j, scenario = job(tmp_path, excluded)
    root = Path(j["run_dir"]).parent
    write_json(root / "drift.json", j["drift"])
    write_json(
        Path(j["run_dir"]) / "results.json",
        dict(pairs=[metrics], warnings=[], status="PASS"),
    )
    assert write_summary(root, [scenario], {0: "PASS"}) == 0
    text = (root / "report.md").read_text()
    assert "2.000×" in text
    assert ("FK / old run" in text) is not excluded


def test_shared_watchdog_checks_disk_and_custom_artifact(tmp_path, monkeypatch):
    j, _ = job(tmp_path)
    j.pop("drift")
    j.update(
        cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
        disk_root=str(tmp_path),
        min_free_bytes=10**30,
    )
    monkeypatch.setattr(ray_runner, "_reclaim_gpus", lambda *a: None)
    result = ray_runner._run_job_subprocess(j, 20, 20, str(tmp_path), ["0"], "off")
    assert "scratch disk reserve" in result["status"]
    j.update(
        cmd=[
            sys.executable,
            "-c",
            f'from pathlib import Path; Path({str(Path(j["run_dir"]) / "model.json")!r}).write_text("{{}}")',
        ],
        artifact="model.json",
        min_free_bytes=0,
    )
    assert (
        ray_runner._run_job_subprocess(j, 20, 20, str(tmp_path), ["0"], "off")["status"]
        == "PASS"
    )


def test_unknown_reference_recipe_does_not_install(tmp_path, monkeypatch):
    from fastkernels.validate import environments

    monkeypatch.setattr(
        environments, "environment_identity", lambda p: {"vllm": "0.26.0"}
    )
    with pytest.raises(ValueError, match="No setup recipe"):
        environments.reference_python("0.99.0", None, tmp_path)


def test_source_identity_ignores_reports_and_tracks_native_code(tmp_path):
    from fastkernels.validate.drift_results import source_identity

    package = tmp_path / "fastkernels"
    package.mkdir()
    source = package / "kernel.cu"
    source.write_text("original")
    before = source_identity(tmp_path)
    (tmp_path / "results.json").write_text('{"status":"RUNNING"}')
    assert source_identity(tmp_path) == before
    source.write_text("changed")
    assert source_identity(tmp_path) != before


def test_setup_uses_project_pins_without_installing_fk_in_old_env(tmp_path):
    from fastkernels.validate.environments import commands
    from fastkernels.validate import _REPO_ROOT

    plan = commands(_REPO_ROOT, tmp_path)
    host = next(c for c in plan if "vllm==0.26.0" in c)
    old = next(c for c in plan if "vllm==0.18.0" in c)
    assert "torch==2.11.0" in host
    assert "transformers==5.14.1" in host
    assert "transformers==4.57.6" in old
    assert str(tmp_path / "vllm-0.18.0/bin/python") in old
    installs = [c for c in plan if "-e" in c]
    assert len(installs) == 1 and "--no-deps" in installs[0]
    assert str(tmp_path / "fk-env/bin/python") in installs[0]


def test_failed_job_still_writes_report(tmp_path, monkeypatch):
    a = args(tmp_path)
    scenario = ValidateScenario(
        "meta-llama/Llama-3.1-8B-Instruct", 1, "bfloat16", ("mixed",)
    )
    write_json(tmp_path / "drift.json", a.drift_config)
    monkeypatch.setattr(
        ray_runner, "_plan_jobs", lambda *a: ([], {0: "FAIL(test failure)"}, [])
    )
    assert ray_runner.run_validation([scenario], a, ["0"], tmp_path) == 1
    assert "FAIL(test failure)" in (tmp_path / "report.md").read_text()


def test_shared_environment_defaults_do_not_create_cuda_contexts(monkeypatch):
    from fastkernels.validate import environments

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='{"vllm":"0.26.0"}')

    monkeypatch.setattr(environments.subprocess, "run", run)
    assert environments.environment_identity("/python")["vllm"] == "0.26.0"
    assert "torch.cuda" not in calls[0][-1]
    assert "direct_url.json" in calls[0][-1]


@pytest.mark.parametrize("override", [False, True])
def test_benchmark_helpers_respect_temp_directory(tmp_path, monkeypatch, override):
    import importlib.util
    import tempfile
    from fastkernels.validate import _VALIDATE_DIR

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object))
    spec = importlib.util.spec_from_file_location("test_bench_paths", _VALIDATE_DIR / "bench_vllm.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("FASTKERNELS_BENCH_PORT_LOCK_DIR", raising=False)
    monkeypatch.delenv("FASTKERNELS_FLASHINFER_SITECUSTOMIZE_DIR", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    locks = tmp_path / "fastkernels_bench_ports"
    site = tmp_path / "fastkernels_flashinfer_sitecustomize"
    if override:
        locks, site = tmp_path / "custom-locks", tmp_path / "custom-site"
        monkeypatch.setenv("FASTKERNELS_BENCH_PORT_LOCK_DIR", str(locks))
        monkeypatch.setenv("FASTKERNELS_FLASHINFER_SITECUSTOMIZE_DIR", str(site))
    port, lock = bench._reserve_tcp_port()
    try:
        assert (locks / f"{port}.lock").is_file()
        other_port, other_lock = bench._reserve_tcp_port(preferred=port)
        try:
            assert other_port != port
        finally:
            other_lock.close()
        bench._install_bench_sitecustomize()
        assert (site / "sitecustomize.py").is_file()
        assert os.environ["PYTHONPATH"] == str(site)
    finally:
        lock.close()


@pytest.mark.parametrize("excluded", [False, True])
def test_actual_harness_dispatch_excludes_only_requested_engine(
    tmp_path, monkeypatch, excluded
):
    """Run the actual host harness on replayed inputs, replacing only GPU workers."""
    import importlib.util
    from unittest.mock import patch
    from fastkernels.validate import _VALIDATE_DIR
    from fastkernels.validate.frozen_inputs import save_inputs

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object)
    )
    spec = importlib.util.spec_from_file_location(
        "test_bench_host", _VALIDATE_DIR / "bench_vllm.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    model = "meta-llama/Llama-3.1-8B-Instruct"
    frozen = tmp_path / "inputs.json"
    payload = {
        "schema": 1,
        "max_model_len": 128,
        "identity": {
            "model": model,
            "scenario": None,
            "num_seqs": 1000,
            "seed": 42,
            "skip_throughput": False,
            "skip_latency": False,
            "latency_iters": 5,
            "workloads": "mixed,single-request",
            "warmup_iters": 1,
            "dtype": "bfloat16",
        },
        "throughput": [
            {"name": "mixed", "prompt_token_ids": [[1]], "output_lens": [64]}
        ],
        "latency": [
            {
                "name": "single-request",
                "prompt_token_ids": [[1]],
                "output_lens": [64],
                "num_iters": 5,
                "num_warmup": 3,
            }
        ],
    }
    save_inputs(frozen, payload)
    calls = []

    def worker(source, config, label, **kwargs):
        calls.append((label, kwargs.get("python_executable")))
        return {
            "throughput": [
                {
                    "name": "mixed",
                    "outputs": [{"token_ids": list(range(64))}],
                    "total_output_tokens": 64,
                    "elapsed": 1,
                    "warmup_iters": 1,
                }
            ],
            "latency": [
                {
                    "name": "single-request",
                    "latencies": [1.0] * 5,
                    "batch_size": 1,
                    "output_len": 64,
                    "num_iters": 5,
                    "num_warmup": 3,
                }
            ],
        }

    monkeypatch.setattr(bench, "run_worker", worker)
    monkeypatch.setattr(bench, "_detect_gpu_name", lambda: "B200")
    monkeypatch.setattr(bench, "_cuda_cc_major", lambda: 10)
    monkeypatch.setattr(bench, "_get_model_max_context_len", lambda m: 128)
    monkeypatch.setattr(bench, "_reserve_tcp_port", lambda **kw: (12345, None))
    monkeypatch.setattr(bench, "_install_bench_sitecustomize", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_vllm",
            "--model",
            model,
            "--dtype",
            "bfloat16",
            "--workloads",
            "mixed,single-request",
            "--inputs-json",
            str(frozen),
            "--vllm-python",
            "/reference/python",
            "--output-dir",
            str(tmp_path),
        ]
        + (["--exclude-fastkernels"] if excluded else []),
    )
    with patch.dict(os.environ):
        bench.main()
    assert len(calls) == (1 if excluded else 2)
    assert calls[0][1] == "/reference/python"
    result = read_validation(tmp_path, frozen, excluded)
    assert ("fastkernels_raw" in result) is not excluded
    assert result["scenarios"][0]["vllm_tok_per_s"] == 64


def test_driver_resume_rejects_changed_configuration(tmp_path, monkeypatch):
    from fastkernels.validate import environments, compatibility

    monkeypatch.setattr(compatibility, "inherit_hf_auth", lambda: None)
    monkeypatch.setattr(drift, "_detect_gpus", lambda g: ["0"])
    monkeypatch.setattr(drift, "source_identity", lambda r: {"sha256": "fixed"})
    monkeypatch.setattr(
        environments,
        "reference_python",
        lambda v, p, r: ("/" + v + "/python", {"vllm": v}),
    )
    monkeypatch.setattr(
        environments,
        "environment_identity",
        lambda p: {
            "vllm": "0.26.0",
            "packages": [
                ("vllm", "0.26.0"),
                ("torch", "2.11.0"),
                ("transformers", "5.14.1"),
            ],
        },
    )
    monkeypatch.setattr(
        drift.subprocess,
        "check_output",
        lambda cmd, **kw: (
            "" if "--query-compute-apps=gpu_uuid,pid,used_memory" in cmd else "GPU-test, B200, 180GB, driver"
        ),
    )
    calls = []
    monkeypatch.setattr(ray_runner, "run_validation", lambda *a: calls.append(a) or 0)
    argv = [
        "0.18.0",
        "0.26.0",
        "--output-dir",
        str(tmp_path / "out"),
        "--data-root",
        str(tmp_path / "scratch"),
    ]
    assert drift.main(argv) == 0
    assert drift.main(argv + ["--resume"]) == 0
    assert len(calls) == 2
    assert drift.main(argv + ["--resume", "--max-requests", "8"]) == 2
    assert len(calls) == 2


def test_existing_gpu_memory_limit_is_per_gpu_and_sums_processes():
    sample = "GPU-a, 1, 6000\nGPU-a, 2, 4000\nGPU-b, 3, 9000\n"
    assert drift.check_existing_gpu_memory(sample, 10000) == {"GPU-a": 10000, "GPU-b": 9000}
    with pytest.raises(ValueError, match="10000 MiB > 9999 MiB"):
        drift.check_existing_gpu_memory(sample, 9999)
    with pytest.raises(ValueError, match="exclusive allocation"):
        drift.check_existing_gpu_memory("GPU-a, 1, 0", 0)
    with pytest.raises(ValueError, match="Cannot determine"):
        drift.check_existing_gpu_memory("GPU-a, 1, [N/A]", 10000)
    assert drift.check_existing_gpu_memory("", 0) == {}
