"""Evidence validation and reporting for paired vLLM release measurements."""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def geomean(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Ratios must be finite and positive")
    return math.exp(statistics.mean(map(math.log, values)))


def interval(values):
    """Descriptive paired-repeat bootstrap; not an architecture-population CI."""
    if len(values) < 3:
        return None
    rng = random.Random(42)
    samples = sorted(geomean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [samples[49], samples[1949]]


def alignment(a, b):
    from .comparison import alignment_from_token_ids

    if not a or len(a) != len(b):
        raise ValueError("Missing or unequal output request counts")
    left, right = [r["token_ids"] for r in a], [r["token_ids"] for r in b]
    if any(not x or len(x) != len(y) for x, y in zip(left, right)):
        raise ValueError("Missing or unequal output token budgets")
    return alignment_from_token_ids(left, right)["avg_matching_tokens_per_request"]


def source_identity(repo):
    extensions = (
        ".py",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".cu",
        ".cuh",
        ".cpp",
        ".h",
        ".hpp",
    )
    repo = Path(repo)
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        text=True,
    )
    # Generated reports now live inside the checkout. Honor gitignore so a
    # run does not invalidate its own source identity by writing results.
    candidates = (
        (repo / p for p in set(listing.stdout.split("\0")) if p)
        if listing.returncode == 0
        else repo.rglob("*")
    )
    files = sorted(
        p
        for p in candidates
        if p.is_file()
        and p.suffix in extensions
        and (
            p.relative_to(repo).parts[0] == "fastkernels" or p.name == "pyproject.toml"
        )
    )
    git = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return {
        "sha256": digest([(str(p.relative_to(repo)), file_digest(p)) for p in files]),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
    }


def read_validation(output, frozen=None, exclude_fastkernels=False):
    output = Path(output)
    result = json.loads((output / "results.json").read_text())
    # These are the unmodified raw outputs saved by the validation harness.
    engines = ("vllm",) if exclude_fastkernels else ("vllm", "fastkernels")
    for name in engines:
        envelope = json.loads((output / f"{name}_raw.json").read_text())
        result[name + "_raw"] = envelope["raw"]
    if frozen is not None:
        inputs = json.loads(Path(frozen).read_text())
        from .frozen_inputs import digest as input_digest

        if input_digest(inputs["payload"]) != inputs["sha256"]:
            raise ValueError("Frozen input checksum mismatch")
        if result["input_sha256"] != inputs["sha256"]:
            raise ValueError(
                "Validation result does not match the frozen input checksum"
            )
        for engine in engines:
            actual = result[engine + "_raw"]["throughput"]
            expected = inputs["payload"]["throughput"]
            if len(actual) != len(expected):
                raise ValueError("Validation omitted a frozen scenario")
            for a, e in zip(actual, expected):
                if "prompt_token_ids" not in e:
                    if a["name"] != e["name"] or not a["outputs"]:
                        raise ValueError("Validation omitted frozen media requests")
                    if any(
                        len(o["token_ids"]) != e["output_len"] for o in a["outputs"]
                    ):
                        raise ValueError(
                            "Validation did not honor media output budgets"
                        )
                    continue
                if a["name"] != e["name"] or len(a["outputs"]) != len(
                    e["prompt_token_ids"]
                ):
                    raise ValueError("Validation omitted frozen requests")
                if [len(o["token_ids"]) for o in a["outputs"]] != e["output_lens"]:
                    raise ValueError("Validation did not honor frozen output budgets")
        expected_latency = inputs["payload"].get("latency", [])
        for engine in engines:
            raw_latency = result[engine + "_raw"].get("latency", [])
            if [r["name"] for r in raw_latency] != [
                r["name"] for r in expected_latency
            ]:
                raise ValueError("Raw latency workload coverage differs")
        combined_latency = result.get("latency_scenarios", [])
        if [s["scenario"] for s in combined_latency] != [
            s["name"] for s in expected_latency
        ]:
            raise ValueError("Validation omitted frozen latency scenarios")
        for actual, expected in zip(combined_latency, expected_latency):
            for engine in engines:
                values = actual[engine + "_latencies"]
                if len(values) != expected["num_iters"]:
                    raise ValueError("Validation omitted latency iterations")
                raw = next(
                    r
                    for r in result[engine + "_raw"]["latency"]
                    if r["name"] == expected["name"]
                )
                if raw["latencies"] != values:
                    raise ValueError("Latency samples disagree with raw measurement")
                if raw.get("num_warmup", 0) < expected.get("num_warmup", 3):
                    raise ValueError("Missing latency warmup evidence")
    return result


def pair_metrics(old, new, floor, expected, exclude_fastkernels=False):
    """Accept timings only when coverage, replay, work, and agreement match."""
    engines = ("vllm",) if exclude_fastkernels else ("vllm", "fastkernels")
    for key in (
        "input_sha256",
        "media_inputs",
        "max_num_seqs",
        "max_model_len",
        "gpu_memory_utilization",
        "dtype",
        "tp",
        "seed",
        "temperature",
        "kv_cache_dtype",
        "enforce_eager",
    ):
        if old.get(key) != new.get(key):
            raise ValueError(f"Reference runs differ in {key}")
    names = [
        r["scenario"]
        for field in ("scenarios", "latency_scenarios")
        for r in old.get(field, [])
    ]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("Requested workload coverage is incomplete or duplicated")
    metrics = []
    for field, kind in (("scenarios", "throughput"), ("latency_scenarios", "latency")):
        arows, brows = old.get(field, []), new.get(field, [])
        if [r["scenario"] for r in arows] != [r["scenario"] for r in brows]:
            raise ValueError("Reference workload coverage differs")
        for a, b in zip(arows, brows):
            name = a["scenario"]
            values, outputs = {}, {}
            for label, result, row in (("old", old, a), ("new", new, b)):
                for engine in engines:
                    key = label if engine == "vllm" else "fk_" + label
                    if kind == "throughput":
                        raw = next(
                            r
                            for r in result[engine + "_raw"]["throughput"]
                            if r["name"] == name
                        )
                        if raw.get("warmup_iters", 0) < 1:
                            raise ValueError("Missing full-workload warmup evidence")
                        if not math.isfinite(raw["elapsed"]) or raw["elapsed"] <= 0:
                            raise ValueError("Invalid elapsed time")
                        count = sum(len(o["token_ids"]) for o in raw["outputs"])
                        if count <= 0 or count != raw["total_output_tokens"]:
                            raise ValueError(
                                "Reported token count differs from raw output"
                            )
                        value = count / raw["elapsed"]
                        if not math.isclose(
                            value, row[engine + "_tok_per_s"], rel_tol=1e-6
                        ):
                            raise ValueError(
                                "Throughput disagrees with raw measurement"
                            )
                        outputs[key] = raw["outputs"]
                    else:
                        samples = row[engine + "_latencies"]
                        geomean(samples)
                        value = statistics.median(samples)
                        if not math.isclose(
                            value, row[engine + "_median_s"], rel_tol=1e-6
                        ):
                            raise ValueError("Latency median disagrees with samples")
                        value *= 1000
                    values[key] = value
            metric = dict(
                values,
                workload=name,
                kind=kind,
                unit="tokens/s" if kind == "throughput" else "ms",
            )
            metric["drift"] = (
                values["new"] / values["old"]
                if kind == "throughput"
                else values["old"] / values["new"]
            )
            if not exclude_fastkernels:
                metric["fk_stability"] = values["fk_new"] / values["fk_old"]
            if outputs:
                checks = {
                    "old_new_reference": alignment(outputs["old"], outputs["new"])
                }
                if not exclude_fastkernels:
                    checks.update(
                        old_fk_reference=alignment(outputs["old"], outputs["fk_old"]),
                        new_fk_reference=alignment(outputs["new"], outputs["fk_new"]),
                        fixed_fk=alignment(outputs["fk_old"], outputs["fk_new"]),
                    )
                if any(v < floor for v in checks.values()):
                    raise ValueError(
                        f"{name}: output-prefix agreement failed: {checks}"
                    )
                metric["alignment"] = checks
            metrics.append(metric)
    return metrics


def write_summary(root, scenarios, statuses):
    """Called only by the Ray driver; workers publish atomic per-model files."""
    from .ray_runner import _job_paths

    config = json.loads((root / "drift.json").read_text())
    excluded = config["exclude_fastkernels"]
    request_cap = config.get("max_requests")
    workload_size = (
        f"diagnostic cap of {request_cap} throughput requests per workload"
        if request_cap is not None
        else "canonical workload sizes"
    )
    lines = [
        "# vLLM version drift",
        "",
        f"vLLM {config['baseline']} → {config['candidate']}; FastKernels {'excluded' if excluded else 'included'}.",
        "",
        f"Settings: {workload_size}; {config.get('repeats', 3)} paired repeat(s); "
        f"{config.get('warmup_iters', 1)} full throughput warmup(s); "
        f"3 latency warmups + {config.get('latency_iters', 5)} measured iterations.",
        "",
        "Ray schedules models on disjoint GPUs when capacity permits. Each old/new pair uses the same GPUs,",
        "frozen inputs, and full throughput warmup. CPU, storage, and host bandwidth are shared.",
        "These compare release environments, including their dependency differences.",
        "",
    ]
    headers = ["Model", "Workload", "Unit"]
    keys = ["old", "new"] if excluded else ["fk_old", "old", "fk_new", "new"]
    headers += (
        ["Old vLLM", "New vLLM"]
        if excluded
        else ["FK / old run", "Old vLLM", "FK / new run", "New vLLM"]
    )
    headers += ["New/old speed", "Repeat 95% CI"]
    lines += [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    rows = []
    for index, scenario in enumerate(scenarios):
        path, _ = _job_paths(root, index, scenario, "bench_vllm")
        result_path = path / "results.json"
        try:
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            status = statuses.get(index, result.get("status", "PENDING"))
        except (OSError, ValueError) as exc:
            result = {"reason": f"Cannot read model results: {exc}"}
            status = "FAIL(results unreadable)"
        row = dict(
            result,
            model=scenario.hf_name,
            status=status,
            log_path=str(path / "run.log"),
        )
        rows.append(row)
        groups = {}
        for pair in result.get("pairs", []):
            for metric in pair:
                groups.setdefault(metric["workload"], []).append(metric)
        for workload, group in groups.items():
            values = [statistics.median(m[k] for m in group) for k in keys]
            ci = interval([m["drift"] for m in group])
            cells = [scenario.hf_name, workload, group[0]["unit"]]
            cells += [f"{v:,.2f}" for v in values]
            cells += [
                f"{geomean([m['drift'] for m in group]):.3f}×",
                f"{ci[0]:.3f}–{ci[1]:.3f}" if ci else "unavailable",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Higher speed ratios favor new vLLM; latency uses old/new latency.",
        "Values are medians; intervals describe paired-repeat variability, not generalization.",
        "Partial completed pairs may appear for failed models; check coverage below.",
        "",
        "## Coverage and failures",
        "",
    ]
    for row in rows:
        reason = str(row.get("reason", "")).replace("\n", " ")
        lines.append(f"- {row['model']}: **{row['status']}** {reason}")
        for warning in row.get("warnings", []):
            lines.append(f"  Warning: {warning}")
        if row.get("attempt_failures"):
            lines.append(
                f"  Failed attempts: {len(row['attempt_failures'])}; see {row['log_path']} and phase logs."
            )
    incomplete = any(not r["status"].startswith(("PASS", "SKIP")) for r in rows)
    write_json(
        root / "status.json",
        dict(config=config, status="INCOMPLETE" if incomplete else "PASS", rows=rows),
    )
    temporary = root / "report.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "report.md")
    return int(incomplete)
