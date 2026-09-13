# Two-H100 validation parallelism check

**Confirmed:** single-GPU model pairs run concurrently on distinct GPUs, and a queued model starts when one GPU is released.

Tested commit: `fead6cc`. Hardware: two NVIDIA H100 PCIe 80 GB GPUs, driver 580.159.04. Three unchanged TP=1 rows from `full.yaml`; FastKernels included. All three models passed, producing 12 accepted workload comparisons.

## Scheduling evidence

| Model | GPU | First observed active (UTC) | First observed passed (UTC) | Result |
| --- | --- | --- | --- | --- |
| meta-llama/Llama-3.1-8B-Instruct | 0 | 05:31:06 | 05:41:31 | PASS |
| state-spaces/mamba-2.8b-hf | 1 | 05:31:06 | 05:41:09 | PASS |
| mistralai/Mamba-Codestral-7B-v0.1 | 1 | 05:41:09 | 05:53:02 | PASS |

Times are sampled on 2026-09-13, approximately every two seconds, and include preparation and warmup.

- 63 samples showed both GPUs above 20% utilization; 37 showed both at 100%. Example: 05:33:49 UTC, with separate GPU process IDs.
- At 05:41:09 UTC, Mamba had passed, Codestral was running on its freed GPU 1, and Llama was still running on GPU 0.
- All six baseline/candidate phase markers retained their model’s assigned GPU. No sampled active model assignments overlapped on the same GPU.
- Each task received 21 CPUs. Model downloads were serialized, but inference ran concurrently. No scheduler code change was needed.

## Command and limitations

After environment setup:

```bash
fastkernels validate drift 0.18.0 0.26.0 \
  --env-root /workspace/envs --scenarios /workspace/parallel.yaml \
  --output-dir /workspace/parallel-results \
  --max-requests 8 --repeats 1 --latency-iters 3 --timeout 1200
```

`parallel.yaml` selects Llama 3.1 8B Instruct, Mamba 2.8B and Mamba-Codestral 7B with their four standard workloads. One full throughput warmup and three latency warmups were retained.

**This confirms scheduling and GPU isolation, not absence of CPU/storage interference or a speedup over serial execution.** No serial control run was performed on this node. It is a capped diagnostic with one repeat, not canonical-size paper measurements. GPU idle periods during compilation, downloads, engine changes, and the final lone task are expected.

## Measurement report

vLLM 0.18.0 → 0.26.0; FastKernels included.

Settings: diagnostic cap of 8 throughput requests per workload; 1 paired repeat(s); 1 full throughput warmup(s); 3 latency warmups + 3 measured iterations.

Ray schedules models on disjoint GPUs when capacity permits. Each old/new pair uses the same GPUs,
frozen inputs, and full throughput warmup. CPU, storage, and host bandwidth are shared.
These compare release environments, including their dependency differences.

| Model | Workload | Unit | FK / old run | Old vLLM | FK / new run | New vLLM | New/old speed | Repeat 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.1-8B-Instruct | mixed | tokens/s | 411.93 | 410.72 | 411.84 | 409.96 | 0.998× | unavailable |
| meta-llama/Llama-3.1-8B-Instruct | long-context | tokens/s | 208.80 | 209.28 | 208.89 | 209.10 | 0.999× | unavailable |
| meta-llama/Llama-3.1-8B-Instruct | single-request | ms | 1,226.92 | 1,229.92 | 1,226.51 | 1,226.35 | 1.003× | unavailable |
| meta-llama/Llama-3.1-8B-Instruct | fixed-batch-32 | ms | 1,762.05 | 1,773.30 | 1,761.47 | 1,777.49 | 0.998× | unavailable |
| state-spaces/mamba-2.8b-hf | mixed | tokens/s | 712.85 | 722.60 | 714.12 | 711.33 | 0.984× | unavailable |
| state-spaces/mamba-2.8b-hf | long-context | tokens/s | 318.84 | 274.76 | 318.14 | 280.84 | 1.022× | unavailable |
| state-spaces/mamba-2.8b-hf | single-request | ms | 700.03 | 723.62 | 694.31 | 716.53 | 1.010× | unavailable |
| state-spaces/mamba-2.8b-hf | fixed-batch-32 | ms | 2,242.73 | 2,332.54 | 2,250.75 | 2,323.00 | 1.004× | unavailable |
| mistralai/Mamba-Codestral-7B-v0.1 | mixed | tokens/s | 365.58 | 391.84 | 378.74 | 382.97 | 0.977× | unavailable |
| mistralai/Mamba-Codestral-7B-v0.1 | long-context | tokens/s | 219.52 | 222.13 | 219.33 | 219.81 | 0.990× | unavailable |
| mistralai/Mamba-Codestral-7B-v0.1 | single-request | ms | 1,251.43 | 1,217.32 | 1,251.07 | 1,245.38 | 0.977× | unavailable |
| mistralai/Mamba-Codestral-7B-v0.1 | fixed-batch-32 | ms | 2,824.48 | 2,726.03 | 2,827.56 | 2,784.35 | 0.979× | unavailable |

Higher speed ratios favor new vLLM; latency uses old/new latency.
Values are medians; intervals describe paired-repeat variability, not generalization.
Partial completed pairs may appear for failed models; check coverage below.

## Coverage and failures

- meta-llama/Llama-3.1-8B-Instruct: **PASS** 
- state-spaces/mamba-2.8b-hf: **PASS** 
- mistralai/Mamba-Codestral-7B-v0.1: **PASS** 

## Saved evidence and cost

Raw evidence is retained locally outside the repository in `drift-results/audit/experiments/2026-09-13-parallel-2h100/`: `gpu-monitor.jsonl`, `parallelism-evidence.json`, `parallel-results/`, the scenario YAML, commands, provisioning log and billing records. All six completed phase artifact checksums were verified. No model weights were copied back.

Reported cost: **$3.01**, including the first inaccessible rental and transfers (billing may settle later).
Both rented instances were destroyed and their absence verified.
