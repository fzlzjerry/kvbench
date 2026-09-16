# Phase 17 Modeling Report

Status: **PASS**

The CPU-only analysis used the immutable Phase 16R host-wall dataset: 2205 accepted process observations at 441 feasible logical points. The 38 infrastructure replacements contribute exactly one accepted observation per logical replicate slot; infeasible records remain coverage evidence and were not assigned latency.

## Predictive result

The preregistered candidates were E (scalar byte law), RQ2 (B/L/r surface), D (positive knee surface), F_shape (D plus static byte-shape features), and F_diagnostic (plus observed kernel count, diagnostic only). Scientific selection chose **D**; the offline deployment default chose **D**. Selection used the declared outer results, so its selected score is not an unbiased estimate of a further model-selection procedure.

| Holdout | Method family | Scored rows | Median relative error | P95 relative error | Domain |
|---|---:|---:|---:|---:|---|
| leave_one_batch_out | bf16 | 42 | 3.885% | 1625.690% | ['edge_extrapolation', 'interior_interpolation'] |
| leave_one_batch_out | turboquant | 135 | 7.774% | 23.102% | ['edge_extrapolation', 'interior_interpolation'] |
| leave_one_batch_out | kivi | 144 | 8.212% | 20.684% | ['edge_extrapolation', 'interior_interpolation'] |
| leave_one_batch_out | kvquant | 120 | 6.153% | 11.744% | ['edge_extrapolation', 'interior_interpolation'] |
| leave_one_config_out | bf16 | 0 | not_applicable | not_applicable | [] |
| leave_one_config_out | turboquant | 135 | 29.850% | 44.554% | ['configuration_extrapolation'] |
| leave_one_config_out | kivi | 144 | 2.915% | 14.117% | ['configuration_extrapolation'] |
| leave_one_config_out | kvquant | 120 | 4.651% | 16.210% | ['configuration_extrapolation'] |
| leave_context_band_out | bf16 | 11 | 1.754% | 2.411% | ['interior_band_interpolation'] |
| leave_context_band_out | turboquant | 39 | 9.731% | 21.859% | ['interior_band_interpolation'] |
| leave_context_band_out | kivi | 39 | 4.927% | 16.181% | ['interior_band_interpolation'] |
| leave_context_band_out | kvquant | 39 | 7.267% | 13.259% | ['interior_band_interpolation'] |
| session_holdout | bf16 | 210 | 1.171% | 3.879% | ['repeat_session_same_geometry'] |
| session_holdout | turboquant | 675 | 7.892% | 21.979% | ['repeat_session_same_geometry'] |
| session_holdout | kivi | 720 | 3.488% | 11.780% | ['repeat_session_same_geometry'] |
| session_holdout | kvquant | 600 | 4.484% | 9.980% | ['repeat_session_same_geometry'] |

Session holdout is process-level repeat-session prediction; the three geometry protocols are logical-point-level. BF16 has no leave-one-compressed-configuration fold, so that cell is not applicable.

- Median relative-error target: **missed** (0.079199).
- P95 relative-error target: **missed** (1.645284).
- Same-work speedup-sign target: **missed** (0.932018 (425/456)).
- Pairwise method-ranking target: **missed** (0.674464 (346/513)).
- Knee-relative-error target: **not_evaluable**; no independent knee reference exists.

The scalar E law was insufficient. RQ2 improved on E, and D improved further under the frozen macro rule. F_shape did not materially improve on D; F_diagnostic scored better on the median but is not deployable because kernel count is an observed held-out-run feature. Thus the evidence supports method-conditioned B/L/r structure, not a scalar allocated-byte law, while the predeclared structural byte fractions did not add a robust further improvement. Phase 15 traffic features remain restricted to their one common profiler point and normal timing rows retain `r_hbm=null`.

## Knee uncertainty

Across 50 configuration/batch curves, 23 were identified in range and 27 were non-identifiable or preferred linear/constant descriptions. Intervals use 1,000 shared replicate-segment bootstrap draws and are conditional on the five sampled sessions. Pilot knees were not treated as truth.

## Scope

Predictions are host-wall milliseconds per full-batch decode step for this model, GPU, container, graph path, and admitted configurations. They are not quality-preserving performance claims and do not validate arbitrary ratios, configurations, checkpoints, GPUs, or serving systems. The Phase 14 negative pure-launch-floor result remains unchanged. Quality is **LOCKED**.

## Execution and durable publication

- Phase starting HEAD: `39b800a5063767951c7f3e3e87d797c18658d2fe`.
- Modeling execution HEAD: `9e20208a5eab4f14263334436878492654756ce9`.
- Final bundle: `phase17-20260916t164055992658z-9e20208a-c944c5`.
- Local/R2 root: `05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb`.
- R2 URI: `r2://kvbench-artifacts/kvbench/sha256/05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb/`.
- Publication: 44 conditional content-addressed objects, `COMPLETE` last.
- Clean retrieval: PASS; inventory, checksum ledger, COMPLETE, and root all verified once.
- Phase 18: not started.
