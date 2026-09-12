# Run the version-drift audit

Run these commands in **Bash on an exclusive 8×B200 Linux allocation**, with
`uv`, Python 3.12, Git, a C++ compiler, Python development headers, and CUDA 13 installed.
Replace `/large-scratch` below with your scratch directory (outside the clone).

```bash
# 1. Clone the branch.
git clone --branch drift-audit https://github.com/vnnm404/fastkernels.git
cd fastkernels

# 2. Set storage paths and install the environments (once).
export DRIFT_SCRATCH=/large-scratch/drift-models
export DRIFT_ENV_ROOT=/large-scratch/drift-envs
export DRIFT_RESULTS="$PWD/drift-audit/overnight-v018-v026"
python3 drift-audit/setup_envs.py --env-root "$DRIFT_ENV_ROOT"
source "$DRIFT_ENV_ROOT/drift.env.sh"

# 3. Enter your Hugging Face read token with gated-model access.
read -rsp 'HF token: ' HF_TOKEN; echo
export HF_TOKEN

# 4. Start the overnight run.
nohup bash drift-audit/run_overnight.sh \
  > "$DRIFT_RESULTS.log" 2>&1 < /dev/null &

# 5. Watch progress (Ctrl-C stops watching; the run continues).
tail -f "$DRIFT_RESULTS.log"
```

The run compares **vLLM 0.18.0 vs 0.26.0 using `fastkernels validate`**, covering
all 15 standard-vLLM rows in `full.yaml`, one model at a time, with warmup and
three paired repeats. It stops after 12 hours if unfinished. Keep the GPU
allocation active; `nohup` does not extend scheduler allocations.

**Share this single report:** `drift-audit/overnight-v018-v026/report.md`.
It includes comparisons and failure/skip reasons; raw logs remain alongside it.
To resume, repeat step 4 with the same environment and output directory.
On a smaller node, append `--skip-insufficient-gpus` to the launch command.

[Compatibility notes, defaults, and troubleshooting](DETAILS.md).
