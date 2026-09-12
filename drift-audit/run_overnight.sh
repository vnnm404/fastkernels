#!/usr/bin/env bash
# Run inside an exclusive GPU allocation. Does not install packages.
set -euo pipefail
: "${FK_PYTHON:?Set FK_PYTHON to the prepared FastKernels environment absolute bin/python path}"
: "${OLD_VLLM_PYTHON:?Set OLD_VLLM_PYTHON to the prepared baseline environment absolute bin/python path}"
: "${NEW_VLLM_PYTHON:?Set NEW_VLLM_PYTHON to the prepared candidate vLLM environment absolute bin/python path}"
: "${DRIFT_SCRATCH:?Set DRIFT_SCRATCH to a large scratch filesystem, not the small home filesystem}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$FK_PYTHON" "$SCRIPT_DIR/run_scenarios.py" \
  --baseline "${DRIFT_BASELINE:-0.18.0}" --candidate "${DRIFT_CANDIDATE:-0.26.0}" \
  --scenarios full --fk-repo "$SCRIPT_DIR/.." \
  --baseline-python "$OLD_VLLM_PYTHON" \
  --candidate-python "$NEW_VLLM_PYTHON" \
  --data-root "$DRIFT_SCRATCH" \
  --workdir "${DRIFT_RESULTS:-$SCRIPT_DIR/overnight-results}" \
  --repeats 3 --warmup-iters 1 --latency-iters 5 \
  --timeout 3600 --budget-hours 12 --min-free-gb 10 --retries 1 "$@"
