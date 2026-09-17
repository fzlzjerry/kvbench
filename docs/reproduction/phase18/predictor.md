# Offline predictor

The default model is frozen model D, evaluated separately by method family:

`T = tau(B,r) + s(B,r) * max(L/131071 - lambda(B,r), 0)`.

For each of tau, s, and lambda, standardized `log(B)` and `log(r_alloc)` form
the basis `[1, z_B, z_r, z_B*z_r]`. Tau and slope use exponentiated bounded
linear predictors; lambda uses `0.02 + 0.96*sigmoid(.)`. Exact scaler means,
scales, coefficients, bounds, residual intervals, training domains, and model
hashes are in `models/*-D.json`. Output units are milliseconds per full-batch
decode step.

For a supported configuration, `r_alloc` is calculated at the requested B/L
from the frozen allocated-byte formula, including metadata, residual/sink,
sparse, and workspace storage. It is point-dependent. An explicit numeric r
requires a method family and does not establish validity for an unseen
quantizer. Only the three measured compressed configurations per family are
within configuration support.

The interval is the stored empirical new-process OOF log-residual interval,
not fitted-mean uncertainty. The BF16 latency and ratio are fully predicted;
no measured baseline is substituted. Domain status does not assert memory
feasibility. Extrapolations are never clipped. All outputs retain
`quality_status=unvalidated` and `performance_claim_eligible=false`.
