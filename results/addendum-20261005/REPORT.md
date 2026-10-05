# Addendum 2026-10-05: results

Post-hoc addendum to "Bytes Are Not Latency" (preregistered amendment `docs/amendment-20261005.md`, committed before any addendum timing; decisions after gate results are in its Section 10). No frozen root or table was changed. Same-work speedups and fixed-memory throughput are reported separately. All latencies are host wall-clock, CUDA Graph, fixed L, single decode step: 64 warmup replays, then 5 batches x 256 replays per process; process median = median of the 5 batch values; point = median of 3 process medians; CV = sample SD / mean of the process medians; CV > 3 % = unstable (none were rerun). BF16 was remeasured in the same run for every ratio.

## Identity

- Measurement container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`; execution repository `ec534d9958d5616eef981c628b87a92c7c809872` (Full Scan timing commit) mounted read-only; addendum code on branch `addendum-20261005` (per-row `addendum_git_sha`).
- Model: meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f249a16976918f6564b8830bc894c89659.
- GPU GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b; driver/persistence/clocks at start: `GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 595.71.05, Enabled, 180 MHz, 405 MHz, 3090 MHz, 14001 MHz, [Requested functionality has been deprecated], [Requested functionality has been deprecated], 600.00 W, 30, 119 MiB, Default, P8`.
- Clocks at driver defaults (not locked), as in the Full Scan; per-process SM clock ranges are in the raw records.

## Task status

| Task | Status |
|---|---|
| Task 1 TurboQuant 32 splits | 36 completed processes |
| Task 1 split sweep | 12 completed processes |
| Task 2 KIVI grouped residual (gate FAILED; diagnostic timing) | 54 completed processes |
| Task 3 retry, prefix caching off (stopped; incomplete attempt) | 4 completed processes |
| Task 3 vLLM FP8 positive control (prefix caching on) | 36 completed processes |
| Task 4 synthetic-cache gate | 12 completed processes |
| Task 4 fixed-memory throughput | 6 completed processes |
| Task 5 KVQuant identity pass-through (optional) | not run: implementation estimate 2-3 h > 1 h condition |

Failures, caps and decisions: see `FAILURES.md` (copied at the end).

## Task 1: TurboQuant with 32 KV splits

| Configuration | Variant | B | L | T_BF16 (ms) | CV_BF16 | T_method (ms) | CV_method | S = T_BF16/T | Status | Frozen S (reference) |
|---|---|---|---|---|---|---|---|---|---|---|
| TQ-3bit | tq_splits=32 | 1 | 32K | 14.098 | 0.01% | 23.438 | 0.09% | 0.601 | stable | 0.139 |
| TQ-3bit | tq_splits=32 | 1 | 128K | 22.261 | 0.01% | 57.883 | 0.03% | 0.385 | stable | 0.060 |
| TQ-3bit | tq_splits=32 | 8 | 32K | 32.628 | 0.00% | 64.087 | 0.33% | 0.509 | stable | 0.317 |
| TQ-4bit | tq_splits=32 | 1 | 32K | 14.098 | 0.01% | 26.023 | 0.10% | 0.542 | stable | 0.114 |
| TQ-4bit | tq_splits=32 | 1 | 128K | 22.261 | 0.01% | 68.200 | 0.07% | 0.326 | stable | 0.048 |
| TQ-4bit | tq_splits=32 | 8 | 32K | 32.628 | 0.00% | 56.775 | 0.90% | 0.575 | stable | 0.266 |
| TQ-k3v4 | tq_splits=32 | 1 | 32K | 14.098 | 0.01% | 22.087 | 0.06% | 0.638 | stable | 0.163 |
| TQ-k3v4 | tq_splits=32 | 1 | 128K | 22.261 | 0.01% | 52.468 | 0.03% | 0.424 | stable | 0.072 |
| TQ-k3v4 | tq_splits=32 | 8 | 32K | 32.628 | 0.00% | 59.831 | 0.27% | 0.545 | stable | 0.359 |

Frozen S: frozen Full Scan medians (as-ported 4 splits; existing KIVI adapter) over frozen BF16, from inputs/frozen_reference.json; context only, never combined with addendum timings.

Split sweep, TQ-k3v4, B = 1, 128K (same-run BF16):

| Configuration | Variant | B | L | T_BF16 (ms) | CV_BF16 | T_method (ms) | CV_method | S = T_BF16/T | Status | Frozen S (reference) |
|---|---|---|---|---|---|---|---|---|---|---|
| TQ-k3v4 | tq_splits=8 | 1 | 128K | 22.259 | 0.02% | 159.609 | 0.05% | 0.139 | stable | 0.072 |
| TQ-k3v4 | tq_splits=16 | 1 | 128K | 22.259 | 0.02% | 85.937 | 0.12% | 0.259 | stable | 0.072 |
| TQ-k3v4 | tq_splits=64 | 1 | 128K | 22.259 | 0.02% | 37.002 | 0.06% | 0.602 | stable | 0.072 |

Effective cache-path bandwidth (fit T = c0 + D_cache / BW_eff over the addendum contexts at fixed B):

| Configuration | Variant | B | Contexts | c0 (ms) | BW_eff (GB/s) | eta = BW_eff / BW_eff,BF16 |
|---|---|---|---|---|---|---|
| TQ-3bit | tq_splits=32 | 1 | 32K, 128K | 11.95 | 112.9 | 0.071 |
| TQ-4bit | tq_splits=32 | 1 | 32K, 128K | 11.95 | 111.2 | 0.070 |
| TQ-k3v4 | tq_splits=32 | 1 | 32K, 128K | 11.95 | 142.0 | 0.089 |
| BF16 | — | 1 | 32K, 128K | 11.38 | 1596.4 | 1 |

D_cache = alpha_v2 x allocated cache bytes (Part A traffic model; alpha measured on the as-ported 4-split kernels). Modeled, not measured, HBM traffic; two contexts per fit.

Correctness at (B = 1, 4K), frozen Q0 core-l4096 probe, eager growing cache, 100 greedy tokens:

| Reference | Candidate | Step-0 max abs logit diff | Max abs diff before divergence | Agreeing positions | First divergence |
|---|---|---|---|---|---|
| t1-greedy-s4 | t1-greedy-s32 | 0.2500 | 7.9844 | 69/100 | 38 |
| t1-greedy-bf16 | t1-greedy-s4 | 0.8350 | 2.0938 | 6/100 | 4 |
| t1-greedy-bf16 | t1-greedy-s32 | 1.0000 | 2.1562 | 6/100 | 4 |

(BF16 rows are supplementary context, not preregistered.) Compute Sanitizer memcheck on the split-32 path: 0 errors, filtered to the TurboQuant decode kernels and unfiltered.

## Task 2: KIVI with KV-head-grouped residual (gate FAILED; timing is diagnostic only)

Gate 1 (100 identical greedy tokens versus the existing adapter, B = 1, 4K): **FAIL**.

| Reference | Candidate | Step-0 max abs logit diff | Agreeing positions | First divergence |
|---|---|---|---|---|
| t2-greedy-k4v4-orig | t2-greedy-k4v4-grouped | 0.1250 | 32/100 | 31 |
| t2-greedy-k2v2-orig | t2-greedy-k2v2-grouped | 0.1328 | 81/100 | 81 |
| t2-greedy-k4v4-orig | t2-greedy-k4v4-orig-repeat | 0.0000 | 100/100 | — |
| t2-greedy-k4v4-orig | t2-greedy-k4v4-grouped-norrr | 0.1250 | 32/100 | 31 |
| t2-greedy-k2v2-orig | t2-greedy-k2v2-grouped-norrr | 0.1328 | 81/100 | 81 |
| t2-greedy-k4v4-orig | t2-greedy-k4v4-m1x4 | 0.1250 | 32/100 | 31 |
| t2-greedy-k2v2-orig | t2-greedy-k2v2-m1x4 | 0.1250 | 86/100 | 86 |

Other gates (grouped adapter; `-control` = existing adapter through the same harness):

| Check | G3/G4 + G1 + graph tests (semantic native_gqa) | Literal native_gqa |
|---|---|---|
| t2-gate-k4v4 | True | False |
| t2-gate-k2v2 | True | False |
| t2-gate-k4v4-control | True | True |
| t2-gate-k2v2-control | True | True |

Diagnostic timing (label `gate_failed_diagnostic_only`; not an admitted result):

| Configuration | Variant | B | L | T_BF16 (ms) | CV_BF16 | T_method (ms) | CV_method | S = T_BF16/T | Status | Frozen S (reference) |
|---|---|---|---|---|---|---|---|---|---|---|
| KIVI-k2v2 | kivi_grouped_residual=True | 1 | 4K | 11.746 | 0.04% | 12.795 | 0.03% | 0.918 | stable; diagnostic only | 0.616 |
| KIVI-k2v2 | kivi_grouped_residual=True | 1 | 32K | 14.091 | 0.21% | 18.619 | 0.39% | 0.757 | stable; diagnostic only | 0.565 |
| KIVI-k2v2 | kivi_grouped_residual=True | 1 | 128K | 22.256 | 0.31% | 39.025 | 0.11% | 0.570 | stable; diagnostic only | 0.471 |
| KIVI-k2v2 | kivi_grouped_residual=True | 8 | 4K | 14.261 | 0.00% | 14.723 | 0.05% | 0.969 | stable; diagnostic only | 0.690 |
| KIVI-k2v2 | kivi_grouped_residual=True | 8 | 24K | 27.456 | 0.00% | 27.383 | 0.12% | 1.003 | stable; diagnostic only | 0.819 |
| KIVI-k2v2 | kivi_grouped_residual=True | 8 | 32K | 32.625 | 0.24% | 32.471 | 0.07% | 1.005 | stable; diagnostic only | 0.841 |
| KIVI-k4v4 | kivi_grouped_residual=True | 1 | 4K | 11.746 | 0.04% | 12.702 | 0.09% | 0.925 | stable; diagnostic only | 0.618 |
| KIVI-k4v4 | kivi_grouped_residual=True | 1 | 32K | 14.091 | 0.21% | 18.183 | 0.02% | 0.775 | stable; diagnostic only | 0.575 |
| KIVI-k4v4 | kivi_grouped_residual=True | 1 | 128K | 22.256 | 0.31% | 37.465 | 0.09% | 0.594 | stable; diagnostic only | 0.486 |
| KIVI-k4v4 | kivi_grouped_residual=True | 8 | 4K | 14.261 | 0.00% | 15.109 | 0.03% | 0.944 | stable; diagnostic only | 0.678 |
| KIVI-k4v4 | kivi_grouped_residual=True | 8 | 24K | 27.456 | 0.00% | 30.407 | 0.06% | 0.903 | stable; diagnostic only | 0.748 |
| KIVI-k4v4 | kivi_grouped_residual=True | 8 | 32K | 32.625 | 0.24% | 37.138 | 0.12% | 0.878 | stable; diagnostic only | 0.756 |

Effective cache-path bandwidth (diagnostic):

| Configuration | Variant | B | Contexts | c0 (ms) | BW_eff (GB/s) | eta = BW_eff / BW_eff,BF16 |
|---|---|---|---|---|---|---|
| KIVI-k2v2 | kivi_grouped_residual=True | 1 | 4K, 32K, 128K | 11.85 | 148.3 | 0.093 |
| KIVI-k2v2 | kivi_grouped_residual=True | 8 | 4K, 24K, 32K | 12.03 | 389.6 | 0.236 |
| KIVI-k4v4 | kivi_grouped_residual=True | 1 | 4K, 32K, 128K | 11.82 | 241.0 | 0.151 |
| KIVI-k4v4 | kivi_grouped_residual=True | 8 | 4K, 24K, 32K | 11.79 | 487.5 | 0.295 |
| BF16 | — | 1 | 4K, 32K, 128K | 11.39 | 1600.1 | 1 |
| BF16 | — | 8 | 4K, 24K, 32K | 11.65 | 1654.0 | 1 |

D_cache from the Part A traffic model (alpha_v2 of the existing KIVI kernels; the quantized-history kernels are unchanged). Three contexts per fit.

Nsight Systems at (B = 1, 4K), 8 graph replays (profiler kernel time, not wall-clock):

| Trace | Kernels per step | Kernel time per step (ms) |
|---|---|---|
| t2-nsys-k4v4-orig | 6826 | 18.292 |
| t2-nsys-k4v4-grouped | 1866 | 12.495 |
| t2-nsys-k2v2-orig | 6826 | 18.342 |
| t2-nsys-k2v2-grouped | 1866 | 12.593 |

## Task 3: vLLM FP8 KV-cache positive control

Measured as task3b (amendment Section 10): the first installation attempt reached the 3 h cap; the retry's first pass failed at engine start (missing curand headers, fixed via CPATH); its second pass (prefix caching off, the benchmark default) was stopped after 2 of 36 processes; task3b enables prefix caching. vLLM numbers compare only with vLLM BF16, never with kvbench timings.

vLLM/torch/CUDA: `0.25.1 2.11.0+cu130 13.0`; engine settings: vLLM defaults except --kv-cache-dtype and --enable-prefix-caching; CUDA Graphs on (no --enforce-eager).

| KV dtype | B | L (input_len) | Step (ms) | CV | n | Attention backend(s) | FP8 / BF16 step | BF16 / FP8 |
|---|---|---|---|---|---|---|---|---|
| bf16 (auto) | 1 | 32K (32768) | 12.890 | 0.07% | 3 | FLASH_ATTN | — | — |
| bf16 (auto) | 1 | 128K (131007) | 20.531 | 0.11% | 3 | FLASH_ATTN | — | — |
| bf16 (auto) | 8 | 32K (32768) | 30.181 | 0.07% | 3 | FLASH_ATTN | — | — |
| fp8 | 1 | 32K (32768) | 11.697 | 0.19% | 3 | FLASHINFER | 0.907 | 1.102 |
| fp8 | 1 | 128K (131007) | 15.175 | 0.05% | 3 | FLASHINFER | 0.739 | 1.353 |
| fp8 | 8 | 32K (32768) | 20.008 | 0.12% | 3 | FLASHINFER | 0.663 | 1.508 |

## Task 4: fixed-memory throughput (synthetic cache, timing only)

Synthetic-cache validity gate at (B = 1, 32K): **PASS** (|T_syn - T_real| / T_real <= max(CV_real, CV_syn, 0.01)).

| Configuration | T_real (ms) | T_synthetic (ms) | Relative difference | Threshold | Pass |
|---|---|---|---|---|---|
| BF16 | 14.089 | 14.081 | 0.05% | 1.00% | True |
| KIVI-k4v4 | 24.488 | 24.474 | 0.05% | 1.00% | True |

B_max from the steady-state formula (L = 128K):

| Configuration | B | Steady-state bytes | Headroom to 89,733,904,465 B | Fits |
|---|---|---|---|---|
| BF16 | 2 | 54,720,799,049 | 35,013,105,416 | True |
| BF16 | 3 | 74,050,838,249 | 15,683,066,216 | True |
| BF16 | 4 | 93,380,877,449 | -3,646,972,984 | False |
| KIVI-k4v4 | 8 | 76,227,974,268 | 13,505,930,197 | True |
| KIVI-k4v4 | 9 | 83,740,511,818 | 5,993,392,647 | True |
| KIVI-k4v4 | 10 | 91,253,049,369 | -1,519,144,904 | False |

L = 128K at B_max (steady-state formula, limit 89,733,904,465 bytes; addendum/feasibility.py). **Synthetic cache, timing only.**

| Configuration | B_max | Step (ms) | CV | tokens/s = B / step | Status |
|---|---|---|---|---|---|
| BF16 | 3 | 43.238 | 0.10% | 69.4 | stable |
| KIVI-k4v4 | 9 | 154.093 | 0.03% | 58.4 | stable |

## FAILURES.md (verbatim)


## 2026-10-05T10:37:45.822689Z smoke

process `smoke-r0-o000-tq_k3v4_nc-s32-b1-l4096` attempt 0: failed (worker_failed) RuntimeError: split scratch was not allocated at cache construction

## 2026-10-05T10:39:59.410335Z smoke

process `smoke-r0-o000-tq_k3v4_nc-s32-b1-l4096` attempt 1: failed (worker_failed) RuntimeError: split scratch was not allocated at cache construction: cache id 136244955519056 type weakref.ProxyType init object.__init__ class is patched class False recorded ids [136245011205200] records 1

## 2026-10-05T10:41:10.966404Z task1

check `t1-greedy-s4` failed: returncode 3 No module named 'scripts.q0_cache_sensitive_correctness'

## 2026-10-05T10:41:24.784297Z task1

check `t1-greedy-s32` failed: returncode 3 No module named 'scripts.q0_cache_sensitive_correctness'

## 2026-10-05T13:07:20Z task3

vLLM 0.25.1 install attempt 1 (pypi.org, started 20:05:23 CST) stopped after 60 min by the session background-task limit while still downloading dependencies (files.pythonhosted.org throughput near zero); no GPU run affected. Attempt 2 uses the PyPI mirror https://mirrors.aliyun.com/pypi/simple/ (same packages). Task 3 cap (3 h including installation) counts from 20:05:23 CST: deadline 23:05:23 CST.

## 2026-10-05T13:25:00Z task2 (gate)

Task 2 gate 1 FAILED: 100 greedy tokens at B=1, 4K (frozen Q0 core-l4096 probe) are not identical between the existing adapter and the KV-head-grouped adapter. Machine-readable reason: `gate_failed_greedy_tokens_not_identical`.

| comparison | step-0 max abs logit diff | agreeing positions | first divergence |
|---|---|---|---|
| k4v4 existing vs grouped (m4) | 0.125 | 32/100 | 31 |
| k2v2 existing vs grouped (m4) | 0.133 | 81/100 | 81 |
| k4v4 existing vs grouped (m4, FP16 reduced-precision reduction off) | 0.125 | 32/100 | 31 |
| k2v2 existing vs grouped (m4, FP16 reduced-precision reduction off) | 0.133 | 81/100 | 81 |
| k4v4 existing vs grouped (m1x4: 4 bmm calls of M=1 over B*8 KV heads) | 0.125 | 32/100 | 31 |
| k2v2 existing vs grouped (m1x4) | 0.125 | 86/100 | 86 |
| k4v4 existing vs existing (repeat, determinism control) | 0.0 | 100/100 | none |

All other gates pass for the grouped adapter (m4): G3 items with the semantic replacement for the literal `native_gqa` source-string check (literal check fails by construction; the frozen-adapter control reproduces the literal pass), G4 (capture/replay, zero replay allocation, no fallback), G1 fixture conformance (2/2) and graph harness (1/1) unit tests, per-head output equivalence within 0.02 (FP16 max abs 6.1e-5, not bitwise), Compute Sanitizer memcheck 0 errors (k4v4, k2v2). Cause: batching the four query heads of a KV group (or batching over KV heads) changes cuBLAS kernel selection and FP16 rounding order; the existing adapter is deterministic. No Task 2 timing has run. Decision on how to proceed requested from the author (2026-10-05 ~20:20 CST).

## 2026-10-05T15:13:27Z task3 (cap reached)

Task 3 stopped at its 3 h cap (20:05:23 to 23:05:23 CST, installation included) before any vLLM run. Machine-readable reason: `cap_reached_during_environment_installation`. Completed: an isolated virtual environment (/home/rockrock/addendum-vllm-env, outside the measurement container). Attempt 1 of `pip install vllm==0.25.1` from pypi.org downloaded dependencies for 60 min at near-zero throughput and was stopped by the session's background-task limit. Attempt 2 from the mirror mirrors.aliyun.com (21:07 CST) was still downloading dependencies at 23:12 CST (last: triton 3.8.0 wheel, 248 MB; pip cache 916 MB in total) and was stopped so that no installation runs during the next timing task. vLLM was never imported: there is no sm_120 compatibility result and no FP8 or BF16 vLLM measurement. Logs: logs/task3-install-vllm-0.25.1.log, logs/task3-install-vllm-0.25.1-attempt2-aliyun.log.

## 2026-10-05T20:16:43.244582Z task3

`task3-r0-o000-auto-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:17:15.142168Z task3

`task3-r0-o001-auto-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:17:43.390449Z task3

`task3-r0-o002-auto-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:18:11.704208Z task3

`task3-r0-o003-auto-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:18:40.090522Z task3

`task3-r0-o004-auto-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:19:08.477212Z task3

`task3-r0-o005-auto-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:20:13.036682Z task3

`task3-r0-o006-fp8-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:20:42.162652Z task3

`task3-r0-o007-fp8-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:21:11.005677Z task3

`task3-r0-o008-fp8-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:21:39.916479Z task3

`task3-r0-o009-fp8-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:22:08.789391Z task3

`task3-r0-o010-fp8-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:22:38.101819Z task3

`task3-r0-o011-fp8-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:23:06.465237Z task3

`task3-r1-o000-auto-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:23:34.743481Z task3

`task3-r1-o001-auto-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:24:03.170395Z task3

`task3-r1-o002-auto-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:24:31.629257Z task3

`task3-r1-o003-auto-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:25:00.048394Z task3

`task3-r1-o004-auto-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:25:28.493022Z task3

`task3-r1-o005-auto-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:25:57.321896Z task3

`task3-r1-o006-fp8-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:26:26.193012Z task3

`task3-r1-o007-fp8-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:26:55.032620Z task3

`task3-r1-o008-fp8-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:27:23.938364Z task3

`task3-r1-o009-fp8-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:27:52.754623Z task3

`task3-r1-o010-fp8-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:28:21.601711Z task3

`task3-r1-o011-fp8-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:28:50.387741Z task3

`task3-r2-o000-fp8-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:29:19.202612Z task3

`task3-r2-o001-fp8-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:29:48.117977Z task3

`task3-r2-o002-fp8-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:30:16.940279Z task3

`task3-r2-o003-fp8-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:30:45.797836Z task3

`task3-r2-o004-fp8-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:31:14.552045Z task3

`task3-r2-o005-fp8-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:31:42.939596Z task3

`task3-r2-o006-auto-b1-l131072-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:32:11.300148Z task3

`task3-r2-o007-auto-b8-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:32:39.653049Z task3

`task3-r2-o008-auto-b1-l32768-out1` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:33:08.027128Z task3

`task3-r2-o009-auto-b1-l131072-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:33:36.355581Z task3

`task3-r2-o010-auto-b8-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:34:04.810940Z task3

`task3-r2-o011-auto-b1-l32768-out65` attempt 0: failed vllm_failed

```
s
    return cls(
           ^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 105, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 786, in __init__
    super().__init__(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 573, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1213, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rockrock/addendum-vllm-env/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1272, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

```

## 2026-10-05T20:36:01Z task3 (retry, first pass)

All 36 Task 3 processes of the retry's first pass (04:14-04:34 CST) failed before measuring: the vLLM engine did not start because FlashInfer 0.6.13 JIT-compiles its sampling ops for sm_120 with the host nvcc (/usr/local/cuda-13.0), whose toolkit has no curand headers (`sampling.cuh: fatal error: curand.h: No such file or directory`). Machine-readable reason: `engine_start_failed_missing_curand_headers`. These are environment (infrastructure) failures: no latency was measured. Fix: a directory with symlinks to the curand headers of the venv's own nvidia-curand (cu13) package, passed as CPATH to the vLLM processes (nothing in /usr/local/cuda or the measurement container changed; FlashInfer sampler left at the vLLM default). Each process is rerun once (attempt1) within the retry's cap (deadline 06:35:35 CST). The attention backend vLLM selected for the BF16 runs was FLASH_ATTN.

## 2026-10-05T20:58:13Z task3 (retry, second pass stopped)

The retry's second pass (prefix caching off, the `vllm bench latency` default) completed 2 of 36 processes (BF16, B=1, 128K, output_len 1 and 65; 22.85 s and 24.23 s per iteration, full prefill every iteration) and was stopped at 04:57 CST by author decision because the remaining processes could not finish before the cap. Machine-readable reason: `stopped_protocol_change_prefix_caching`. The process in flight (task3-r0-o002) was killed and has no result. Task 3 continues as task3b with --enable-prefix-caching (amendment Section 10).

## 2026-10-05T21:39:09Z task5 (optional; not run)

Optional Task 5 (KVQuant identity pass-through diagnostic) was not started. Machine-readable reason: `implementation_estimate_exceeds_1h`. Read-only assessment: an exact-codebook identity is impossible (per-channel Key tables and per-token Value tables of 2-16 levels; the deterministic Value kernels require exactly 12 outliers); the most faithful feasible design (a proxy around the KVQuant extension that forwards all store kernels and replaces only the Key-RoPE and Value decode kernels with exact FP32 computation from a BF16 side cache) needs about 150-200 lines plus validation and a GPU run, estimated at 2-3 h end to end, above the preregistered 1 h condition.


## Files and checksums

SHA-256 of every file under results/addendum-20261005/ except REPORT.md, report-data.json and SHA256SUMS are in `SHA256SUMS` (generated with this report).

