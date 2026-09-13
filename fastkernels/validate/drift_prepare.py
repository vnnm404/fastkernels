"""Pin and preflight a checkpoint before downloading it into private scratch."""

from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys
from .compatibility import hub_retry, access_problem
from .drift_results import write_json


def prepare_model(
    model, metadata, data_root, reserve, refresh=False, interpreters=None
):
    """Runs in a bounded child. Download only one pinned checkpoint at a time."""
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download

    repo, _, revision = model.partition("@")
    if metadata.exists() and not refresh:
        info = json.loads(metadata.read_text())
    else:
        try:
            record = hub_retry(
                lambda: HfApi().model_info(
                    repo, revision=revision or None, files_metadata=True, timeout=30
                )
            )
        except Exception as exc:
            problem = access_problem(exc)
            if problem:
                write_json(metadata, dict(problem, repo=repo))
                return
            raise
        weights = sum(
            x.size or 0 for x in record.siblings if x.rfilename.endswith(".safetensors")
        )
        if not weights:
            raise ValueError(
                "No safetensors checkpoint found; refusing an unbounded download"
            )
        info = {"repo": repo, "revision": record.sha, "weight_bytes": weights}
        write_json(metadata, info)
    # A previous access failure has no revision to resume.
    if "revision" not in info:
        return prepare_model(model, metadata, data_root, reserve, True, interpreters)
    info.pop("status", None)
    info.pop("reason", None)
    try:
        config_file = hub_retry(
            lambda: hf_hub_download(
                info["repo"], "config.json", revision=info["revision"]
            )
        )
    except Exception as exc:
        problem = access_problem(exc)
        if problem:
            write_json(metadata, dict(info, **problem))
            return
        raise
    if interpreters:
        config_dir = metadata.parent / "config-preflight"
        config_dir.mkdir(exist_ok=True)
        shutil.copyfile(config_file, config_dir / "config.json")
        info["compatibility"] = {}
        for label, python in interpreters.items():
            output = metadata.parent / (label + "-compatibility.json")
            output.unlink(missing_ok=True)
            # Stay in the preparation process group so its outer timeout also
            # kills the probe. These checks do not launch engine workers.
            with (metadata.parent / (label + "-compatibility.log")).open("w") as log:
                try:
                    result = subprocess.run(
                        [
                            python,
                            str(Path(__file__).with_name("compatibility.py")),
                            str(config_dir),
                            str(output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    )
                    ok = result.returncode == 0 and output.exists()
                except subprocess.TimeoutExpired:
                    ok = False
            check = (
                json.loads(output.read_text())
                if ok
                else {
                    "status": "environment-error",
                    "reason": f"{label} preflight failed/timed out; see {label}-compatibility.log",
                }
            )
            info["compatibility"][label] = check
        problems = [
            (label, check)
            for label, check in info["compatibility"].items()
            if check["status"] != "ok"
        ]
        if problems:
            info.update(
                status="unsupported"
                if all(c["status"] == "unsupported" for _, c in problems)
                else "blocked",
                reason="; ".join(label + ": " + c["reason"] for label, c in problems),
            )
            write_json(metadata, info)
            return
    cached = sum(
        p.stat().st_size
        for p in {
            p.resolve()
            for p in data_root.glob("hf/hub/models--*/snapshots/*/*.safetensors")
        }
        if p.is_file()
    )
    if (
        shutil.disk_usage(data_root).free
        < max(0, info["weight_bytes"] * 1.1 - cached) + reserve
    ):
        raise OSError("Insufficient disk for checkpoint plus reserve: " + str(info))
    path = hub_retry(
        lambda: snapshot_download(
            info["repo"],
            revision=info["revision"],
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.tiktoken",
                "*.txt",
                "*.jinja",
            ],
        )
    )
    info["path"] = path
    write_json(metadata, info)


if __name__ == "__main__":
    config = json.loads(Path(sys.argv[1]).read_text())
    import fcntl
    import time

    # Serialize checkpoint downloads across this node's drift jobs. The free-space
    # check then accounts for weights already held by concurrently running models.
    with Path(config["lock"]).open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print("Waiting for checkpoint download slot", flush=True)
                time.sleep(10)
        prepare_model(
            config["model"],
            Path(config["metadata"]),
            Path(config["data_root"]),
            config["reserve"],
            interpreters=config["interpreters"],
        )
