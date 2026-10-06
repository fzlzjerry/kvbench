# Addendum 2026-10-05: supplementary CPU calculations

No new measurement. Inputs (SHA-256 in supplement.json) are the addendum records, checked against
results/addendum-20261005/SHA256SUMS (= sealed R2 root 57f8ff27...), and two sealed post-hoc files
(Part E S_adj, Part B split diagnostic) checked against their ledgers.

- Task 1: wall-clock T_TQ/T_BF16 with 32 splits 1.567--3.064
  (traced kernel time, Part B: 1.603--3.161).
- Task 2 (gate failed; diagnostic): best S = 1.005 (k2v2, B=8,
  32K); the nine pairwise ratios of process medians span
  1.004--1.009. Max |S - S_adj| = 0.026;
  S_adj from Part E (2026-10-05T07:46:53.352420Z), before the first Task 2 process
  (2026-10-05T16:34:09.149239Z); not refitted. Max |c0 - c0,BF16| = 0.46 ms.
- Task 3: T_pred,FP8(B,L) = T_BF16(B,L) - D_BF16(B,L) / (2 * BW_BF16); BW_BF16 = 1685 GB/s (nominal,
  fitted); measured/predicted FP8 step = 1.007, 0.983, 1.002
  (measured/predicted > 1: FP8 slower than the cache-halving prediction).
- Task 4: KIVI-k4v4 (B=9) / BF16 (B=3) tokens/s = 0.842 (pairwise 0.841--0.843);
  scope: full Llama-3.1-8B decode step with BF16 weights resident, CUDA Graph replay of one decode step at fixed L; cache filled with synthetic K/V through the method's store path; no prefill, no request-level serving.
- GPU state: persistence mode restored to Disabled on 2026-10-06 (~09:10 CST), its state before the addendum.
