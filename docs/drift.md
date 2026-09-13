# Run a vLLM version comparison

Use Bash on an **exclusive Linux GPU node**, with `uv`, Python 3.12, Git, a C++
compiler, Python development headers, and CUDA 13 installed. Replace
`/large-scratch` with your scratch mount.

```bash
# Clone and prepare the environments once (no models downloaded yet).
git clone --branch drift-audit https://github.com/vnnm404/fastkernels.git
cd fastkernels
export FASTKERNELS_CACHE_DIR=/large-scratch/fastkernels-cache
export TMPDIR="$FASTKERNELS_CACHE_DIR/tmp"
mkdir -p "$TMPDIR"
python3.12 -m fastkernels.validate.environments \
  --env-root "$FASTKERNELS_CACHE_DIR/reference-envs"
source "$FASTKERNELS_CACHE_DIR/reference-envs/fk-env/bin/activate"

# Enter a read token with access to the gated models.
read -rsp 'HF token: ' HF_TOKEN; echo
export HF_TOKEN

# Run all standard-vLLM workloads from full.yaml on the available GPUs.
nohup fastkernels validate drift 0.18.0 0.26.0 > drift.log 2>&1 < /dev/null &
tail -f drift.log
```

**Share:** `~/.fastkernels/results/validate/drift-0.18.0-0.26.0/report.md`.
The report includes comparisons, failures, and skipped models. Raw phase logs,
inputs, environment versions, and results remain beside it. Ctrl-C exits `tail`;
the background run continues.

```bash
# Resume the same run after an interruption.
fastkernels validate drift 0.18.0 0.26.0 --resume

# vLLM-only measurements, in a separate output directory.
fastkernels validate drift 0.18.0 0.26.0 --exclude-fastkernels \
  --output-dir /large-scratch/vllm-only

# Inspect selection without GPU queries, installs, or downloads.
fastkernels validate drift 0.18.0 0.26.0 --dry-run
```

<details>
<summary>Scheduling, defaults, compatibility, and storage</summary>

FastKernels is included by default. Ray runs independent models concurrently on
separate GPUs, reserving each model's TP degree and proportional CPU/memory
resources. Both vLLM versions run sequentially on that model's assigned GPUs;
FastKernels stays fixed and is measured alongside each reference. Rows needing
more GPUs than available and nonstandard reference engines are reported as
skipped. There are 15 standard-vLLM rows in `full.yaml`.

Defaults: three paired repeats, alternating reference order, one full throughput
warmup, three latency warmups and five measured latency iterations, one retry,
a one-hour timeout per preparation/validation attempt, and a 15-minute log-stall
timeout. There is **no suite-wide 12-hour cutoff**. Individual timeouts remain
configurable with `--timeout` and `--stall-timeout`. CPU, disk, and host bandwidth
are shared; warmup does not eliminate interference. Serial-versus-concurrent GPU
performance still needs validation before treating the parallel results as final.

The runner pins checkpoint revisions, replays frozen inputs and decoded media,
checks output budgets and throughput prefix agreement, and verifies artifacts
before resuming. The prefix check is a heuristic, not proof of correctness.
Gemma 4 is unsupported by the pinned 0.18 environment; an earlier Qwen3-VL video
comparison failed cross-version agreement. These remain failures with reasons.
Jamba requires access to `ai21labs/AI21-Jamba-Mini-1.7`.

Reference environments are reused or created automatically for 0.18.0/0.26.0;
the host environment is never upgraded by `validate drift`. Other versions
require `--baseline-python` / `--candidate-python` pointing to prepared
environments. This compares release environments, including dependency changes.
The setup command above installs only the standard-vLLM subset of FastKernels'
dependencies and targets the checkout's 0.26.0 pins.

Use `--scenarios PATH`, `--gpus 0,1`, `--data-root PATH`, `--env-root PATH`, or
`--output-dir PATH` to override paths/selection. Scratch must hold the active
models and decoded media plus a 10 GiB reserve. Downloads are serialized and
checked against available space; insufficient space is reported as a failure.
Private weights are removed after each model unless `--keep-models` is set;
failed media and measurement evidence are retained. Run on one local node;
attaching to an existing remote Ray cluster is not supported for drift.

</details>

[H100 diagnostic results: vLLM 0.18.0 vs 0.26.0](results/drift-h100-2026-09-13.md).

[Two-H100 parallelism check](results/drift-2h100-parallel-2026-09-13.md).
