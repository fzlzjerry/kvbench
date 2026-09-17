# Method-conditioned KV-cache decode performance study

## Research question and scope

The study tested whether full-model fixed-L decode latency on one NVIDIA RTX
PRO 6000 Blackwell can be predicted from method-conditioned batch B,
historical context L, and allocated compression ratio `r_alloc`. The scalar
byte law and knee structure were tested rather than assumed. The endpoint is
host-wall milliseconds per full-batch CUDA Graph decode step for the frozen
Llama-3.1-8B-Instruct revision and ten admitted BF16, TurboQuant-vLLM,
project-patched KIVI, and project-patched KVQuant configurations.

## Dataset and custody

Phase 16R supplies 2,205 accepted process observations at 441 feasible logical
points, five independent replicate slots per point, plus 465 explicit
capacity-infeasible records. Thirty-eight infrastructure replacements retain
one accepted observation per slot. No Pilot, admission, profiler, or stopped
campaign timing enters the Phase 17 response table.

## Models and validation

The frozen candidates were E (scalar byte law), RQ2 (B/L/r surface), D
(positive knee response surface), F_shape (D plus static byte-shape features),
and diagnostic-only F_diagnostic (observed kernel count). Four distinct tests
were retained: leave-one-batch, leave-one-compressed-config, leave-context-band,
and repeat-session holdout. D is the scientific selection and deployment
default, but its outer selection score is not an unbiased evaluation of a
further model-selection procedure.

The primary labels are equal-cell macro means over 11 applicable geometry
cells: median-relative-error `0.079198563227627` and P95-relative-
error `1.645283992575687`. They are not pooled quantiles. The frozen
5% and 10% targets are missed. Same-work sign accuracy remains 425/456
(93.2018%, missed), pairwise ranking remains 346/513 (67.4464%, missed), and
knee error is not evaluable without independent truth. TurboQuant
leave-one-config transfer remains 29.850% median and 44.554% P95; KIVI is
2.915%/14.117% and KVQuant 4.651%/16.210%. The approximately 67.45% ranking
accuracy is not a reliable automatic method selector.

E to RQ2 to D is a relative improvement among tested candidates under the
declared rule, not universal proof that B/L/r is sufficient. F_shape's lack of
robust improvement does not show metadata or workspace costs are irrelevant.
F_diagnostic is not deployable because held-out-run kernel count is required.

## Error tail and knees

The BF16 batch-holdout median is small but its P95 is 1625.690%. The audit
finds correct full-batch units, joins, scaling, point-dependent BF16 ratio, and
finite positive outputs. The tail is a real D-model predictive failure at
short-context B=1 edge extrapolation, retained in every primary metric and
figure. Twenty-three of 50 local curves have identifiable in-range knees;
27 are weak/non-identifiable, insufficient-span, or prefer linear/constant
descriptions. The 1,000 shared-session bootstrap is conditional on five sampled
sessions, and no independent knee truth exists.

## Mechanism and claim boundaries

Phase 14 found 0/14 fully identifiable comparisons supporting the complete
pure launch-floor-only criterion. Phase 15 directly found fewer CPU submissions
and lower GPU idle in 16/16 anchor pairs, while retaining the
`method_specific_mixed` interpretation. Its HBM/L2 data remain scoped only to
B=1, historical L=131071, Graph mode. They are not filled into Full Scan rows
and do not establish critical-path causality.

Quality has not been evaluated. All predicted ratios are performance-only and
quality-unvalidated; capacity feasibility is separate. The next quality work
must follow the existing performance-freeze protocol and human approval gate.
