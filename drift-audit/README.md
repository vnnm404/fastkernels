# Version-drift audit

Compare a fixed FastKernels checkout with vLLM 0.18.0 and 0.26.0 through the
**installed `fastkernels validate` CLI**. The runner selects standard `bench_vllm`
rows from a scenario file, executes one model at a time, and writes one Markdown
report. In `full.yaml`, 15 of the 47 rows use this harness. Other reference engines
are explicitly skipped.

## Clone and prepare

```bash
git clone --branch drift-audit https://github.com/vnnm404/fastkernels.git
cd fastkernels
```

Use a Linux CUDA host with `uv`, Python 3.12, `git`, a C++ compiler, Python
development headers, and a CUDA toolkit compatible with the checkout's Torch
pin (currently CUDA 13). Setup downloads packages and builds extensions, but
**does not download model weights or run inference**. Allow space for two Python
environments, CUDA libraries, and compiler caches on a large scratch filesystem.
Do setup before the benchmark; do not change packages or source during a run.

```bash
# Replace /large-scratch with your allocated scratch mount, outside the clone.
export DRIFT_SCRATCH=/large-scratch/drift-models
export DRIFT_ENV_ROOT=/large-scratch/drift-envs
python3 drift-audit/setup_envs.py --env-root "$DRIFT_ENV_ROOT"
source "$DRIFT_ENV_ROOT/drift.env.sh"
```

Use `--dry-run` to inspect installation commands without downloading anything.
The recipe installs the standard-vLLM subset of the checkout's dependencies,
including its exact FlashAttention wheel and DeepGEMM commit. It preserves FK's
vLLM/Torch/Transformers pins and puts vLLM 0.18 in a separate environment. It does
not install unrelated reference frameworks from the other scenario rows.
The final import checks must pass before launching validation. This recipe
packages the H100-tested installation steps; the full 8×B200 sweep has not yet
been verified.

Provide a read token with access to the gated checkpoints, using `HF_TOKEN` or
`HF_TOKEN_PATH` (a private file outside the clone, mode 600). The runner resolves
credentials before switching to its disposable model caches. Never put a token
in a scenario, command argument, or tracked file. Jamba's checkpoint is
`ai21labs/AI21-Jamba-Mini-1.7`; the runner checks the token owner's access.

If environments already exist, set `FK_PYTHON`, `OLD_VLLM_PYTHON`, and
`NEW_VLLM_PYTHON` to their absolute `bin/python` paths instead of running setup.
For 0.18 → 0.26, `NEW_VLLM_PYTHON` can equal `FK_PYTHON`. The optional
`repair_reference_env.py /path/to/reference/bin/python` repairs the known
Datasets/PyArrow mismatch while constraining existing core versions.

## Run on an exclusive 8×B200 allocation

```bash
export HF_TOKEN_PATH=/path/to/private/hf-token
# On a directly allocated node; under a scheduler, retain its GPU visibility.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DRIFT_RESULTS="$PWD/drift-audit/overnight-v018-v026"
nohup bash drift-audit/run_overnight.sh \
  > drift-audit/overnight-v018-v026.log 2>&1 < /dev/null &
```

The launcher defaults to 0.18.0 → 0.26.0, `full.yaml`, three paired repeats, one
full throughput warmup, five measured latency iterations, one retry, a one-hour
limit per command, and a 12-hour total limit. It uses canonical workload sizes;
there is no diagnostic request cap. A 12-hour limit does not guarantee that the
suite finishes. Supply trailing flags to override defaults, for example:

```bash
# Offline scenario planning: no GPU checks, model downloads, or setup.
"$FK_PYTHON" drift-audit/run_scenarios.py --scenarios full --dry-run

# Small diagnostic on an exclusive single GPU, in a separate results directory.
DRIFT_RESULTS="$PWD/drift-audit/overnight-smoke" \
  bash drift-audit/run_overnight.sh --skip-insufficient-gpus \
  --max-requests 8 --repeats 1 --latency-iters 3 --budget-hours 2
```

For another release pair, prepare the matching interpreters and pass
`--baseline VERSION --candidate VERSION`. The fixed FK environment must still
match the checkout pins. The setup recipe itself targets 0.18 → 0.26 only.

## Results and resume

- `$DRIFT_RESULTS/report.md`: comparison table, coverage, failures, and warnings.
- `$DRIFT_RESULTS/status.json`: progress, resolved commands, versions, and hashes.
- `$DRIFT_RESULTS/<run-id>/<row-id>/`: CLI logs, frozen inputs, raw results, and
  completion stamps. Keep this evidence with the report.

Run the same command with the same environment, source, GPUs, and output directory
to resume. Verified completed CLI runs are reused; failed attempts remain in the
archive. Changed source/settings/environments select a new run namespace. Output
directories are locked against concurrent writers. Exit 0 means all selected
rows passed; exit 2 means incomplete or interrupted. A report can contain valid
comparisons even when another model fails.

Keep downloads on scratch with room for the largest selected checkpoint, decoded
media, and at least the default 10 GiB reserve. Weights are fetched one model at
a time. Only the runner's private caches are cleaned; failed rows retain frozen
media. `--keep-models` retains weights too. Results, experiment archives, logs,
and credentials are excluded from this branch.

## Measurement and compatibility rules

- Both references replay the same pinned checkpoint revision, frozen text token
  IDs, and checksummed decoded media. Media is prepared in the fixed FK environment.
- Throughput gets a full untimed workload warmup; latency gets three untimed
  iterations. Reference order alternates across paired repeats. FK remains fixed
  within each pair; timing-control changes above 5% produce warnings.
- Throughput checks equal output-token budgets and an average matching output
  prefix of at least 32 tokens across FK/reference and cross-version comparisons.
  This heuristic is not proof of model correctness. Latency has separate coverage,
  sample-count, and timing checks, but no generated-output agreement gate.
- Different vLLM releases may require different Torch/Transformers versions;
  these are comparisons of release environments, not isolated vLLM source changes.
- Preflight checks access, imports, and config/architecture support before weight
  downloads. Transient network/Hub errors get bounded retries. Unsupported or
  access-blocked rows remain `INCOMPLETE` rather than being counted as passes.
- Gemma 4 is unsupported by the pinned 0.18 environment. Earlier Qwen3-VL video
  measurements failed cross-version output agreement; the threshold stays in place.
- Codestral uses `max_num_seqs=64` for both engines by default: its old 512-slot
  state allocation exhausted an H100 during vLLM 0.18 graph profiling. This
  constrains concurrency, not request count or the batch-32 latency workload.
  Explicit scenario limits override the default and are recorded in results.
- Exclusive allocation is required. The runner checks for existing GPU processes,
  but cannot prevent another user from starting work later. The eight-GPU node
  is not used to run independent models concurrently.

## Files and local checks

`run_overnight.sh` supplies defaults; `run_scenarios.py` plans and runs the audit.
`compatibility.py` handles credentials, retries, and preflight checks.
`drift_runner.py` launches commands and verifies CLI artifacts. `fk_drift.py`
provides hashing, comparison/statistics helpers, and an optional explicit-model
CLI. `setup_envs.py` prepares the two environments; `repair_reference_env.py`
repairs data-loader dependencies in an existing reference environment.
The replay and warmup protocol lives in `fastkernels/validate/` in this same clone.

CPU-only regression checks (Python 3.12, `torch`, `numpy`, `pyyaml`, and `pytest`):

```bash
PYTHONPATH=. python3 -m unittest discover -s drift-audit/tests -v
PYTHONPATH=. python3 -m pytest tests/test_validate.py -q
bash -n drift-audit/run_overnight.sh
```
