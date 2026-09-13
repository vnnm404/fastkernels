"""Optional real-Ray integration test using CPU workers and logical GPUs only.

RUN_RAY_TESTS=1 python -m pytest tests/test_validate_drift_ray.py -q
"""

import json
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RAY_TESTS") != "1", reason="opt-in local Ray runtime"
)


def test_parallel_pairs_and_multi_gpu_queue(tmp_path, monkeypatch):
    ray = pytest.importorskip("ray")
    from fastkernels.validate import ray_runner, ValidateScenario, _parser
    from fastkernels.validate.drift_results import write_json

    bench = tmp_path / "cpu_bench.py"
    bench.write_text("""import argparse, json, os, time
from pathlib import Path
from fastkernels.validate.frozen_inputs import digest
p=argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--output-dir'); p.add_argument('--vllm-python')
p.add_argument('--save-inputs-json'); p.add_argument('--inputs-json')
a=p.parse_args()
root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=True)
frozen=Path(a.inputs_json or a.save_inputs_json)
payload={'throughput':[{'name':'mixed','prompt_token_ids':[[1]],'output_lens':[64]}], 'latency':[]}
sha=digest(payload)
if not frozen.exists(): frozen.write_text(json.dumps({'payload':payload,'sha256':sha}))
start=time.time(); time.sleep(1); end=time.time()
row={'name':'mixed','outputs':[{'token_ids':list(range(64))}],'total_output_tokens':64,'elapsed':1,'warmup_iters':1}
for engine in ['vllm','fastkernels']:
 (root/(engine+'_raw.json')).write_text(json.dumps({'raw':{'throughput':[row],'latency':[]}}))
(root/'results.json').write_text(json.dumps({'input_sha256':sha,'scenarios':[{'scenario':'mixed','vllm_tok_per_s':64,'fastkernels_tok_per_s':64}], 'start':start,'end':end,'gpus':os.environ.get('CUDA_VISIBLE_DEVICES')}))
""")
    a = _parser(drift=True).parse_args(["0.18.0", "0.26.0"])
    a.drift_config = dict(
        baseline="0.18.0",
        candidate="0.26.0",
        exclude_fastkernels=False,
        repeats=1,
        retries=0,
        alignment_floor=32,
        min_free_gb=0,
        keep_models=False,
        identity="ray-test",
        data_root=str(tmp_path / "scratch"),
        interpreters={"baseline": "old", "candidate": "new"},
    )
    Path(a.drift_config["data_root"]).mkdir()
    root = tmp_path / "results"
    root.mkdir()
    write_json(root / "drift.json", a.drift_config)
    scenarios = [
        ValidateScenario(
            f"meta-llama/Llama-3.1-{i}B-Instruct", tp, "bfloat16", ("mixed",)
        )
        for i, tp in enumerate([1, 1, 2])
    ]
    monkeypatch.setattr(
        ray_runner,
        "_build_cmd",
        lambda s, h, a, out: [
            sys.executable,
            str(bench),
            "--model",
            s.hf_name,
            "--output-dir",
            str(out),
        ],
    )
    monkeypatch.setattr(ray_runner, "_cuda_cc_major", lambda: None)
    monkeypatch.setattr(
        ray_runner,
        "_init_ray",
        lambda r, a: r.init(
            num_cpus=4,
            num_gpus=2,
            include_dashboard=False,
            log_to_driver=False,
            runtime_env={"env_vars": {"PYTHONPATH": str(Path.cwd())}},
        ),
    )
    # Stub only Hub preparation. All measurement subprocesses, Ray allocation,
    # pair sequencing, artifact checks, and driver reporting are real.
    original_remote = ray.remote

    def execute(job, timeout, stall, repo, gpus, numa):
        from fastkernels.validate import drift as d
        from fastkernels.validate.drift_results import write_json
        from pathlib import Path
        import os
        import ray

        # macOS has no CUDA accelerator manager; expose Ray's logical allocation
        # in the same environment variable Linux workers receive automatically.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(int(i)) for i in ray.get_gpu_ids()
        )
        original_phase = d._phase

        def phase(job, name, *args):
            if name == "prepare":
                output = Path(job["run_dir"]) / name
                output.mkdir(parents=True, exist_ok=True)
                write_json(
                    output / "model.json",
                    {"revision": "fixed", "path": "/model", "compatibility": {}},
                )
                return output
            return original_phase(job, name, *args)

        d._phase = phase
        try:
            return d.run_job(job, timeout, stall, repo, gpus, numa)
        finally:
            d._phase = original_phase

    def remote(*args, **kwargs):
        if len(args) == 1 and args[0] is ray_runner._ray_run_job:
            return original_remote(execute)
        return original_remote(*args, **kwargs)

    monkeypatch.setattr(ray, "remote", remote)
    assert ray_runner.run_validation(scenarios, a, ["0", "1"], root) == 0
    phases = []
    for i, s in enumerate(scenarios):
        path, _ = ray_runner._job_paths(root, i, s, "bench_vllm")
        old = json.loads((path / "r0-baseline/results.json").read_text())
        new = json.loads((path / "r0-candidate/results.json").read_text())
        assert old["gpus"] == new["gpus"]
        assert old["end"] <= new["start"]
        phases.append(old)
    assert phases[0]["gpus"] != phases[1]["gpus"]
    assert max(phases[0]["start"], phases[1]["start"]) < min(
        phases[0]["end"], phases[1]["end"]
    )
    assert len(phases[2]["gpus"].split(",")) == 2
    assert json.loads((root / "status.json").read_text())["status"] == "PASS"
