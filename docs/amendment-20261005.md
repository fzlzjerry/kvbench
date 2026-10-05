# Amendment 2026-10-05: post-hoc addendum experiments

Status: COMMITTED before any addendum timing (author approval 2026-10-05,
"use the defaults": D1 = 5 x 256 as in the Full Scan; D2 = semantic check in
place of G3's literal `native_gqa` string check, reported as such; D3 = raw
data synced to the R2 bucket `kvbench-artifacts` under `addendum-20261005/`
plus an off-repository tarball; D4 = split sweep inside Task 1's cap).

## 0. Declaration

These experiments were designed after the main results of "Bytes Are Not
Latency" were known. They are a post-hoc addendum. They do not replace,
rerun, or reinterpret any frozen result:

- performance freeze tag `perf-freeze-20260917-83536c37-r1` and its roots;
- the roots listed in the paper's Appendix A (full-scan wall-clock
  `5605558b...`, profiler `641fc02d...`, modeling `05d5c4e8...`, joint
  results `2d609efb...`) and the post-hoc roots (`3dc84c64...`,
  `c347e20a...`, `4cdbb3ae...`, `39efeb1c...`, `47296f7c...`,
  `b0969338...`, `1998be3e...`).

No file under `artifacts/`, `docs/evidence/`, or any frozen root is
written. Addendum code lives on branch `addendum-20261005`; addendum outputs
live only under `results/addendum-20261005/`. Same-work ratios and
fixed-memory throughput (Task 4) are reported separately and never pooled.

## 1. Identity

| Item | Value |
|---|---|
| Branch base | `121c24d` (main). `src/` and `scripts/phase13_pilot.py` are byte-identical to the Full Scan timing commit `ec534d99` and to the results commit `0641de4b`. |
| Measurement container | `kvbench-measurement:phase6a`, image `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e` (Tasks 1, 2, 4, 5) |
| Model | `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`, BF16 weights |
| GPU | RTX PRO 6000 Blackwell Workstation Edition, `GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b`, driver 595.71.05 |
| GPU state | Persistence mode set to enabled before any GPU run (as recorded in E00); clocks left at driver defaults (not locked), as in the Full Scan. `nvidia-smi -q` (clocks, power, performance, processes) is recorded before each task. |

## 2. Common timing protocol (Tasks 1, 2, 4)

Unchanged from the Full Scan except the replicate count:

1. One fresh worker process per (configuration, B, L, replicate), launched
   with the Full Scan's `docker run` mounts and environment, `--network none`.
2. Session construction is the Full Scan's: BF16 restores its frozen prefix
   snapshot where one exists; compressed configurations rebuild the cache from
   the frozen logical prefix by prefill. Task 4 synthetic points are the only
   exception (Section 6).
3. CUDA Graph, fixed L, single-step decode. 64 warmup replays, then the
   Full Scan measurement: 5 batches of 256 replays, host wall-clock
   (`perf_counter_ns`) per batch (decision D1).
4. Process median = median of the per-batch host ms/op. Point latency =
   median of the process medians. CV = sample SD / mean of the process
   medians. CV > 3 % marks the point `unstable`; unstable points are kept and
   never rerun.
5. 3 processes per point. Replicates run as 3 rounds; in each round the
   configuration blocks (BF16 is one block) are shuffled, then the points within
   each block, with `random.Random(seed)`, seed = 2026100500 + 10 * task + round
   (written to the run manifest before the first launch).
6. BF16 is remeasured in the same run for every compared (B, L). Ratios use
   only same-run BF16: S = T_BF16 / T_method.
7. Only infrastructure failures (container start, OOM unrelated to the
   preregistered formula, driver/Xid errors, host crash) may be rerun, once
   per process, and each rerun is logged with a machine-readable reason.
8. No allocation inside the timed region: the Full Scan worker's graph-replay
   allocation audit must report `allocation_stable`; any extra scratch is
   allocated at construction time, before warmup and capture.
9. GPU exclusivity: `nvidia-smi --query-compute-apps` must be empty before
   each launch (display-server graphics contexts are recorded, as in the Full
   Scan). No other GPU job and no package installation runs during timing.
10. After each point completes, one JSONL row is appended to
    `results/addendum-20261005/<task>/points.jsonl` (configuration, B, L,
    per-process medians, point median, CV, stability, git commit, container
    digest, UTC timestamp, worker result paths and SHA-256). Per-process
    records are appended to `processes.jsonl` as they finish. The driver skips
    processes already recorded as completed, so it can resume after
    interruption.
11. Hard caps: the driver does not launch a process if its predicted duration
    (Full Scan per-process duration x 1.2) would cross the task cap. Points
    left with fewer than 3 processes are reported as `incomplete`. Every
    failure or timeout is written to `results/addendum-20261005/FAILURES.md`.
12. No GPU run starts in the last hour of machine access.

## 3. Task 1: TurboQuant with 32 KV splits (cap 3 h GPU)

- Configurations: TQ-4bit (`tq_4bit_nc`), TQ-k3v4 (`tq_k3v4_nc`), TQ-3bit
  (`tq_3bit_nc`) with 32 KV splits; BF16 control.
- Points: (B=1, 32K), (B=1, 128K), (B=8, 32K): 9 method points + 3 BF16.
- Mechanism: the stage-1/stage-2 launch uses 32 splits and a caller-owned
  FP32 scratch of shape (B, Hq, 32, D+1), allocated once when the cache is
  constructed (as upstream preallocates for CUDA Graphs), before warmup and
  capture. The cache's own tensors, byte accounting and fingerprints remain
  those of the as-ported configuration; the scratch bytes are recorded
  separately. No frozen source file is edited.
- Correctness (before timing, not timed): at (B=1, 4K), split 4 vs 32, eager
  growing-cache decode on the frozen Q0 probe `core-l4096` (WikiText-2 test
  prefix of 4,096 tokens plus its burn-in token): maximum absolute logit
  difference at the first decode step and over the steps before the first
  divergence, and the number of agreeing positions in a 100-token greedy
  continuation (plus the first divergence).
- AGENTS.md rule 6 checks for the split-32 launch: Compute Sanitizer
  (memcheck, restricted to the TurboQuant decode kernels) on a small shape;
  CUDA Graph capture/replay and zero-allocation replay are the Full Scan
  worker's own audits in every timed process (`allocation_stable`,
  `kernel_path_stable`, `finite_output`).
- Split sweep {8, 16, 64} for TQ-k3v4 at (B=1, 128K), with BF16 at the same
  point remeasured in the same run, 3 processes each, runs after the main
  Task 1 run within the same 3 h cap (predicted 1.06 h + 0.63 h). Its results
  are reported separately from the 9 main points.

## 4. Task 2: KIVI with KV-head-grouped residual (engineering cap 5 h)

- Change: a new adapter class replaces the per-query-head residual loop
  (`src/kvbench/adapters/kivi.py` `_decode_compressed`, residual QK and PV)
  with batched products per KV head: query `[B*8, 4, 128]` against residual K
  `[B*8, R, 128]` and pending K, probabilities `[B*8, 4, R]` against residual V,
  using views of the existing preallocated buffers. Quantization, packing,
  metadata, softmax, ring unrolling and rollover are unchanged. The existing
  adapter and its tests are not modified.
- Gates, all required before timing:
  1. 100 greedy tokens at a fixed shape identical to the existing adapter;
     logit differences reported.
  2. Zero allocation in graph replay after capture.
  3. The existing G3 and G4 checks, run unchanged on the new adapter
     (decision D2: G3's `native_gqa` item is a literal source-string check for
     `kv_head = query_head // KIVI_GQA_GROUP_SIZE`, which the grouped form
     cannot contain; it is reported as failing-by-construction / not
     applicable and replaced by a semantic check: per-head output equivalence
     with the existing adapter, no K/V temporary with a 32-head dimension, no
     `repeat_kv`/`expand`/`torch.cat`).
  4. AGENTS.md rule 6: G1 fixture conformance at the existing tolerance (0.02),
     Compute Sanitizer memcheck, graph capture/replay, allocation audit.
- Configurations: KIVI-k4v4, KIVI-k2v2 (new adapter); BF16 control.
- Points: B=1 x {4K, 32K, 128K}, B=8 x {4K, 24K, 32K}: 12 method points + 6 BF16.
- One Nsight Systems trace at (B=1, 4K) per configuration of the new adapter
  (kernel counts and durations of bmm/add); diagnostic only, never timing.

## 5. Task 3: FP8 KV-cache positive control in vLLM (cap 3 h including install)

- Separate virtual environment outside the measurement container (the
  container is not modified). vLLM v0.25.1 (the TurboQuant port's upstream
  release); if it cannot be installed or run on sm_120, the error is recorded
  and v0.31.0 is tried once within the same cap.
- Same model and revision, `vllm bench latency`, CUDA Graphs on (no
  `--enforce-eager`), all other engine settings at vLLM defaults (recorded),
  `--kv-cache-dtype fp8` vs default (BF16).
- Per-step decode time by differencing: output_len 1 and 65,
  step = (T65 - T1) / 64. Each (dtype, point, output_len) runs as 3 separate
  processes; T = median iteration latency within a process; replicate i pairs
  T65_i with T1_i; point step = median of the 3 differences, CV over them.
- Points: B=1 x {32K, 128K}, B=8 x 32K. Because input + output must not exceed
  the model's 131,072 positions, the 128K point uses input_len 131,007.
- From the logs: the attention backend selected for each dtype, the KV-cache
  dtype in effect, the vLLM version, and absence of fallback. A run whose log
  shows a different KV dtype or a fallback is reported as such, not as FP8.

## 6. Task 4: fixed-memory throughput (remaining time)

- Synthetic cache, timing only: K/V drawn from a seeded normal distribution
  are written through each method's normal store/quantization path instead of
  model prefill; everything after cache construction is the Section 2 protocol.
- Validity gate at (B=1, 32K), BF16 and KIVI-k4v4: |T_syn - T_real| / T_real
  <= max(CV_real, CV_syn, 1 %), 3 processes each. If either fails, Task 4 stops.
- At L = 128K, B_max per configuration is the largest B whose steady-state
  memory (weights + allocated cache + endpoint workspace + graph reserve +
  fixed decode inputs; the post-hoc steady-state formula of Appendix
  "capacity") is at most 89,733,904,465 bytes. B_max is computed by code and
  written to the run manifest before any Task 4 timing.
- BF16 and KIVI-k4v4 (new adapter if Task 2 passed its gates, else the
  existing one) at their B_max, 3 processes each; tokens/s = B / step time.
  If a point fails with OOM, the failure is recorded and B_max - 1 is tried once.

## 7. Task 5 (optional): KVQuant identity pass-through diagnostic

Only if Tasks 1-4 are done and the implementation is estimated under 1 h.
Quantization replaced by identity, integration path otherwise unchanged; at
(B=1, 4K) report logit differences against BF16. No kernel fix, no timing.

## 8. Derived quantities

- S = T_BF16 / T_method (same run, same B and L).
- eta = BW_eff,method / BW_eff,BF16 at the same B, where BW_eff is the slope
  of T = c0 + D_cache / BW_eff fitted by least squares over the addendum's
  contexts at that B, with D_cache from the unchanged Part A traffic model.
  Computed only where at least two contexts exist at that B (Task 1: B=1;
  Task 2: B=1 and B=8). D_cache is modeled traffic, not measured HBM traffic.

## 9. Out of scope

Full rescans, KVQuant kernel rewrites, any quality evaluation, changes to the
original gates or timing protocol, and any other hyperparameter change.

## 10. Decision log (appended; earlier sections unchanged)

- 2026-10-05 ~21:30 CST, after the Task 2 gate results and before any Task 2
  timing: Task 2 gate 1 (100 identical greedy tokens versus the existing
  adapter at B=1, 4K) failed for the grouped adapter (k4v4: first divergence
  at token 31; k2v2: 81), also with FP16 reduced-precision reduction off and
  with a 4 x (M=1) batched form; the existing adapter is deterministic (100/100
  against itself). All other gates passed. Author decision (option 3): the gate
  status is recorded as FAIL (reason `gate_failed_greedy_tokens_not_identical`,
  results/addendum-20261005/FAILURES.md) and is not changed; Task 2 is timed
  with the grouped adapter in its preregistered form (Section 4, one bmm per
  residual operand over [B*8, 4, ...]) as a diagnostic only, labeled
  `gate_failed_diagnostic_only`, never as an admitted result. Points, ratios
  and eta from Task 2 carry that label wherever reported.
- Consequently Task 4 uses the existing KIVI adapter (Section 6: "new adapter
  if Task 2 passed its gates, else the existing one").
- Task 3 installation: the first attempt from pypi.org (started 20:05:23 CST)
  was stopped after 60 min while downloading; the second uses the PyPI mirror
  mirrors.aliyun.com. The 3 h cap still counts from 20:05:23 CST.
- 2026-10-05 23:12 CST: Task 3 reached its 3 h cap during installation (no
  vLLM run; FAILURES.md, reason `cap_reached_during_environment_installation`).
  Author decision (~23:20 CST): retry Task 3 once after the Task 4 / Task 2
  timing chain has finished, with a new 3 h cap counted from the start of the
  retry installation; installation with `uv` from the same mirror, never
  during a timing run. Task 3's protocol (Section 5) is otherwise unchanged.
  The first attempt's failure record stays.
- 2026-10-06 ~04:55 CST, Task 3 retry: `vllm bench latency` disables prefix
  caching by default (its source: "V1 enables prefix caching by default which
  skews the latency numbers"), so every iteration re-ran the full prefill
  (22.8 s per iteration at B=1, 128K) and the 36 processes could not finish
  within the retry's cap. Author decision (option 1): stop that run (its two
  completed BF16 processes stay under results/addendum-20261005/task3/ as an
  incomplete attempt, not reported as results) and rerun Task 3 under
  results/addendum-20261005/task3b/ with `--enable-prefix-caching` (the vLLM
  engine default that the benchmark script overrides). The warmup iterations
  build the cache; T1 and T65 then differ by exactly the 64 decode steps.
  Everything else in Section 5 is unchanged; the deadline is still the retry's
  (06:35:35 CST).
