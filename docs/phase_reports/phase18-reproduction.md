# Phase 18 CPU Reproduction Package

Status: **PASS**

- Starting HEAD: `1aac29e408383b3ac265e64b89c7a13abc553814`.
- Package execution HEAD: `a866fa6859e8128cc3ea21d340be39b66d034efe`.
- Phase 17 source root:
  `05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb`.
- Package: `phase18-20260917t024901884906z-a866fa68-6cfb61`.

## Reporting audit

The frozen selected-model values are labeled
`macro_mean_of_cell_median_relative_errors=0.0791985632276268` and
`macro_mean_of_cell_p95_relative_errors=1.645283992575687`. Each is the
equal-weight mean of 11 applicable geometry-holdout cell statistics; neither
is a pooled quantile. BF16 leave-one-config-out is not applicable and session
holdout is separate.

The BF16 batch-holdout tail is arithmetically and referentially consistent.
The largest logical-point error is held-out B=1/L=4096: observed
`11.773624859375` ms and predicted `668.6230370648544` ms, absolute relative
error `55.789904982614516`. The 20 largest rows are preserved. Units remain
milliseconds per full-batch decode step; no per-token division occurred; D has
direct positive output rather than a missing inverse transform; actual
historical-L scaling, fold joins, point-dependent BF16 `r_alloc`, and finite
outputs pass. Interior-batch BF16 P95 is `0.060193662118041176`; edge-batch P95
is `40.16989986942676`. The short-context B=1 edge extrapolation is a real
predictive failure, not clipped or refitted. Phase 17 did not persist separate
fold parameter vectors, so their identity remains unavailable; the OOF table,
candidate specification, and execution code checksum-bind the predictions.

## Frozen outcome and reproduction

Model D remains the scientific selection and deployment default. The 5%
median, 10% P95, 95% same-work sign, and 90% ranking targets remain missed;
knee error remains not evaluable. The selected outer score is not an unbiased
evaluation of an additional model-selection procedure. No model family,
threshold, fold, or prediction was changed.

The pure-standard-library default reproduction passed in a new temporary
directory. It recomputed the audit, regenerated six SVG figures, and produced
four predictor examples with `gpu_launched=false`, `network_accessed=false`,
and `quality_evaluation_executed=false`. The standalone predictor exactly
matches the Phase 17 serialized predictor on BF16, TurboQuant, KIVI, and
KVQuant examples. Focused tests pass 6/6.

## Durable publication and handoff

- Local/R2 root:
  `cd6ee2324d0163088102e29e45b4ad124b5d524fb1900717faddf997da6805e2`.
- R2 URI:
  `r2://kvbench-artifacts/kvbench/sha256/cd6ee2324d0163088102e29e45b4ad124b5d524fb1900717faddf997da6805e2/`.
- Objects: 54; COMPLETE-last PASS; one clean retrieval PASS; Bucket Lock PASS.

Quality remains **LOCKED** and `PERFORMANCE_DATA_FROZEN` remains absent. The
release manifest is a draft only. The next separately authorized task is the
existing QP-0 performance-freeze procedure: verify no active performance
process, create/checksum the complete performance inventory, record Git,
container, and hardware identities, create the freeze tag and locked hot-path
list, populate the quality contract from the frozen manifest, and stop at its
human-approval gate. No quality work started.

