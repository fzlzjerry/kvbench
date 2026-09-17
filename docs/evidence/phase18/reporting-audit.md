# Phase 18 reporting audit

The frozen selected-model labels are explicitly:

- `macro_mean_of_cell_median_relative_errors` = `0.079198563227627`;
- `macro_mean_of_cell_p95_relative_errors` = `1.645283992575687`.

They are equal-weight means of 11 applicable geometry-holdout cell statistics,
not pooled quantiles. BF16 leave-one-config-out is not applicable and session
holdout remains separate. No target status changed.

The BF16 leave-one-batch logical-point median error is
`3.885%` and its P95 is
`1625.690%`. The largest existing error is held-out
B=1, historical L=4096:
observed `11.773624859` ms and predicted
`668.623037065` ms, absolute relative error
`5578.990%`.

- edge_extrapolation: n=15, median=21.509%, P95=4016.990%, max=5578.990%
- interior_interpolation: n=27, median=2.459%, P95=6.019%, max=10.445%

The stored rows use milliseconds per full-batch decode step, without division
by B. Fold/config joins, historical-L scaling, point-dependent BF16 `r_alloc`,
and finite positive outputs pass. D emits positive latency directly, so no
missing inverse transform exists. The failure is concentrated in short-context
B=1 edge extrapolation and is retained without clipping, exclusion, or refit.
Phase 17 did not persist each fold's fitted parameter vector separately; this
audit therefore binds OOF values to the immutable prediction table, candidate
specification, and execution code rather than misidentifying the final
full-data BF16-D parameters as fold parameters.
