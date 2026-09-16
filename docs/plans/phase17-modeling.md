# Phase 17 modeling plan

Phase 17 is CPU-only analysis of the completed Phase 16R Full Scan. It does
not authorize CUDA, admission, timing, profiling, quality evaluation, new
experimental points, or Phase 18.

## Frozen inputs and endpoint

The source family is
`phase16-20260831t123029614620z-ec534d99-de80ac`. The primary authority is the
host-wall closure root
`5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e`;
the original outer root
`d74587675dd59b464d81c6e82885d3c9706c681a9da216ad1a7fe4c6ccd88daa`
is secondary CUDA-event evidence. Phase 15 root
`641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8`
is interpretation-only and is never expanded across B or L.

The response is host-observed milliseconds per full-batch fixed-L decode
step. It is not divided by batch. Exactly one published effective process
record is used for each logical replicate slot; 38 replaced attempts remain
provenance, not samples. Capacity-infeasible records remain coverage evidence
and never receive fabricated latency. The top configured label 131072 maps to
historical L=131071 and total attended length 131072.

## Frozen candidates and splits

Local per-configuration/per-batch descriptions are constant, A (linear), B
(free segmented), and C (nonnegative floor/hinge). Predictive method-family
candidates are E (quadratic log byte law), RQ2 (small quadratic B,L,r surface),
D (12-parameter knee-aware surface), F_shape (D plus shape-derived metadata,
full-precision, outlier, and workspace fractions), and diagnostic-only
F_diagnostic (F_shape plus observed kernel count). Preprocessing, knots, and
feature scaling are fitted on training rows only. Configuration IDs are not
predictors for leave-one-config-out.

The four frozen protocols are leave-one-batch-out for B={1,2,4,8,16},
leave-one-compressed-config-out within each method family, historical-context
band [24576,49152] inclusive, and leave-one-replicate-segment-out. Geometry
protocols group at least `(method_config,B,actual_L)`; the session protocol is
reported separately as repeat-session prediction. Base and adaptive errors
are reported separately.

All predictive models minimize log-latency loss with the fixed complexity in
`configs/plans/phase17_modeling.yaml`. Final deployable selection uses macro
median relative error across the three geometry protocols, then macro P95,
then lower complexity. That selected outer score is explicitly not presented
as an unbiased evaluation of the selection procedure. The offline default is
selected separately from E/RQ2/D so that method+B+L+config or explicit
`r_alloc` is sufficient; F_shape remains a scientific candidate and requires
its explicit shape features rather than guessing them from scalar r.

## Uncertainty and targets

Local knee intervals use 1,000 shared replicate-segment bootstrap draws with
seed 20260918. A claim-bearing in-range knee requires at least two unique
contexts in each regime, BIC improvement of at least 2 over linear, and at
least 500 valid bootstrap draws. Failed/non-identifiable draws remain counted.
New-process prediction intervals use geometric out-of-fold log residuals and
are distinct from fitted-mean uncertainty.

Targets remain median relative error <=5%, P95 <=10%, speedup-sign accuracy
>=95%, ranking accuracy >=90%, and independent-reference knee error <=10%.
Targets are reported met, missed, or not_evaluable without threshold changes.
Near ties are <=1% relative latency difference, remain in the primary ranking
denominator, and are also reported separately.

## Outputs and custody

The compact append-only Phase 17 bundle contains the canonical process table,
split manifest, candidate specification, out-of-fold predictions, comparison
table, local knee estimates, uncertainty/target summaries, final model files,
offline predictor example, diagnostics, report, inventory, checksums, and
COMPLETE. It references rather than copies Phase 16/15 bundles. Publication is
content-addressed, COMPLETE-last, and verified once by clean retrieval; the
receipt remains outside the bundle. Quality stays LOCKED and Phase 18 is
explicitly deferred.
