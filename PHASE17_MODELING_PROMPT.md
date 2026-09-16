# Phase 17 — Knee and Method-Conditioned Performance Modeling

This task implements and runs Phase 17 on the completed Phase 16R dataset.
It is CPU-only analysis. Do not launch CUDA, regenerate prefixes, rerun timing,
run profilers, run PPL/LongBench, or unlock quality evaluation.

## 1. Read the relevant inputs; start without another admission campaign

Read:

- AGENTS.md.
- CODEX_WORKFLOW.md: research questions, mathematical modeling/statistics,
  mechanism-derived quantities, and the modeling/reproduction sequence.
- docs/measurement_protocol.md: endpoint definitions only.
- docs/phase_reports/phase16r-full-scan.md.
- The Full Scan family manifest, five segment indexes, outer-bundle index,
  replacement-run map, and host-wall supplement manifest/receipt.
- The current method configurations and byte-accounting definitions.
- Phase 14 analysis closure and Phase 15 mechanism/traffic summaries.
- The existing analysis/ implementation and any already accepted modeling plan.

Run only git status --short and git rev-parse HEAD at entry.
Expected reported HEAD prefix: 39b800a. Resolve the full SHA from Git; do not
invent it or reset a documented descendant.

Accept existing admission and publication receipts. Do not rerun package-lock,
GPU checks, complete tests, admission, or historical R2 download/verification.
Load local compact analysis inputs; fetch only missing referenced files.
Do not download model weights, prefix catalogs, complete raw trace archives,
or entire historical campaigns.

Reported source coverage:

- 534 logical design points: 450 base + 84 adaptive.
- 2,670 terminal process records: 2,205 completed + 465 capacity-infeasible.
- 441 feasible logical points, five accepted process replicates each.
- Zero unstable logical points; maximum host-wall CV 1.305277%.
- 38 infrastructure replacements, with original attempts preserved.
- Five segments, outer bundle, and host-wall supplement published and verified.

Checking these counts while constructing the analysis table is sufficient.
A missing file should cause a targeted fetch, not a full historical audit.

## 2. Freeze a small analysis plan before comparing model scores

Create docs/plans/phase17-modeling.md and one analysis configuration.
Reuse existing code rather than introducing a model registry, workflow engine,
experiment database, dashboard, or general validation framework.

Record the endpoint, input identities, replacement-selection rule, candidate
models, model complexity, loss, split membership, random seeds, bootstrap
scheme, metrics, and model-selection rule. This is an analysis plan made after
Pilot/design inspection, not a claim that the whole study was previously unseen.

Retain already accepted statistical settings. Where no implementation detail
is fixed, choose a small deterministic default and record it before scoring.
Do not expand model complexity repeatedly in response to outer-test errors.

No new experimental points are authorized in this task.

## 3. Construct one canonical process-level analysis table

Use Phase 16R normal timing records only for primary fitting and validation.
Do not mix Pilot, densification timing, admission timing, Graph A/B, profiler
observations, or the stopped Full Scan attempts into the primary dataset.
The 84 adaptive locations measured anew in Phase 16R ARE Full Scan data;
retain their adaptive-origin labels.

Resolve the 38 infrastructure replacements using the published replacement
map. Each logical replicate slot contributes exactly one accepted process
observation. Retain original attempt IDs as provenance, not extra samples.

Join the host-wall supplement according to its documented authority and exact
run/replicate keys. A supplement is not another run. Do not silently use an
older GPU-event summary when the primary host-wall result is in the supplement.
Do not average an original and corrected field or overwrite source artifacts.

Primary response: the frozen host-observed milliseconds per completed
full-batch decode step, using the existing process-summary statistic.
Do not divide by B and relabel batch latency as token latency.
CUDA-event time is a separate secondary endpoint, never a replacement label.

Keep method family, configuration, source fingerprint, B, configured L label,
actual historical L, total attended length, process/run ID, replicate/segment,
input identity, base/adaptive origin, response, and recorded byte features.
For the top label preserve actual historical L=131071 and attended L=131072.
Use the documented L convention consistently in fitting and prediction.

Fit with equal logical-point weight; five replicates must not become five
independent geometry discoveries. Report metrics separately for base-grid and
adaptive points, as well as combined and per-method/per-batch summaries.

Keep all 465 infeasible process records in a coverage table, but do not assign
them zero/infinite latency or include them in latency regression. They encode
the configured feasibility policy, not observed hardware OOM boundaries.

## 4. Define what the B,L,r relationship actually tests

The research target is method-conditioned latency:

    T_m = F_m(B, L, r_alloc)
    S_m = T_BF16(B,L) / T_m(B,L,r_alloc)

Use canonical ratios:

    r_alloc = BF16 allocated bytes / method allocated bytes
    rho_alloc = 1 / r_alloc

Keep the frozen allocation scope. Do not silently substitute nominal bits,
active payload only, or a workspace-excluding denominator for r_alloc.

For these implementations r_alloc is a known function of B, L, and method
configuration, not an independently randomized variable. Evaluate it at every
row. Do not assign one constant r to a KIVI or KVQuant context curve.

The main test is whether scalar r_alloc suffices within a fixed method, versus
needing metadata, residual/sink/sparse, workspace, or K/V-path information.
Keep key_bits and value_bits distinct in the config metadata. Do not use config
IDs as a disguised substitute for r in the leave-one-config-out scalar model.

Only three main compressed configurations exist per family. Report the
supported configuration/range domain and distinguish interpolation from
extrapolation. Do not claim validation for arbitrary compression ratios,
unmeasured KIVI k4v2, TurboQuant k8v4, other checkpoints, GPUs, or serving stacks.

## 5. Fit the workflow's model candidates without forcing a knee

Reuse analysis/fit_linear.py, fit_segmented.py, fit_knee.py and related code
where present. Preserve workflow model labels.

Local descriptive candidates, fitted per configuration and batch:

    A: T = alpha + beta*L
    B: T = alpha + beta1*L + beta2*max(L-k, 0)
    C: T = max(tau, a + c*L)

Also retain a constant baseline. These local fits describe curves; by
themselves they cannot predict an entirely missing batch or configuration.
Mark such CV comparisons not_applicable rather than fabricating parameters.

Predictive candidates, fitted separately by method family:

    E: T = G_m(B*L/r_alloc)                       [naive byte law]
    RQ2 control: T = F_m(B,L,r_alloc)             [small non-knee surface]
    D: T = tau_m(B,r) + s_m(B,r)*max(L-lambda_m(B,r), 0)
    F: the same small predictive architecture augmented by method features.

Use low-dimensional regression/splines and a small fixed complexity budget.
For D, small smooth parameterizations in log(B) and log(r) are sufficient;
constrain predicted latency positive and the floor model's post-floor slope
nonnegative. Do not force Model B's pre-knee slope to zero.
Do not run a GP/boosting/neural-network model zoo just to lower error.

Fit D jointly against process observations using each observation's actual r.
Do not estimate its parameter surfaces from full-data local knees before CV.
For a fixed configuration, its operational transition must be evaluated along
r_alloc(B,L,config); lambda(B,r) at an arbitrary fixed r is not automatically
the observable knee of that configuration's curve.

For F, use a small predeclared set from the workflow: metadata bytes,
full-precision fraction, outlier bytes, and kernel count; include workspace
bytes if the recorded scope requires them. Use the same folds as D/control.
Distinguish features computable from config/shape without measurement from
observed runtime features. Observed kernel counts from a held-out run cannot
be used in a deployable predictor; report that version as diagnostic-only.

No latency, fitted residual, phase/attempt ID, or test-derived knee may be a
predictor. Fit scaling, spline knots, feature selection and tuning on training
folds only. If tuning is needed, use a small inner grouped CV. Report all fixed
candidate scores; do not present the winning outer score as unbiased evaluation
of a model-selection procedure that used those same scores.

## 6. Run the four required holdout protocols

Freeze membership before evaluating scores.

A. Leave-one-batch-out: five folds, holding out B=1,2,4,8,16 in turn across all
methods and contexts. All five process replicates of held-out conditions remain
outside training. Separate held-out interior batches from edge extrapolation.

B. Leave-one-config-out: within each compressed method, hold out one complete
main configuration at a time across all B,L and replicates. BF16 has no
compressed-config holdout. Do not train on admission/Pilot samples of the
held-out configuration. Do not invent an unseen categorical coefficient.

C. Leave-context-band-out: use the already frozen band definition; otherwise
freeze historical L in [24576,49152] inclusive as the primary held-out band.
Hold out all methods, batches, configurations and replicates in that band,
including adaptive locations inside it. Do not use Pilot knees to expose its
held-out Full Scan response. Note the grid itself was Pilot-informed.

D. Session/replicate holdout: leave one of the five replicate segments out at a
time, preserving actual process IDs and replacement lineage. Training may see
the same B,L in other segments; label this repeat-session prediction, NOT
prediction of unseen geometry.

For A/B/C, the group key is at least (method_config,B,actual_L): no replicate
of that condition may occur on both sides. Never split individual decode
operations randomly. Keep Session holdout results separate from A/B/C.

## 7. Report prediction errors, ratios, and rankings

For every applicable model/method/holdout report:

- MAE in milliseconds, MAPE, median absolute relative error, P95 relative error.
- Fold counts, scored logical-point counts, missing-prediction counts, and
  domain/extrapolation status.
- Residuals by B, L, config and base/adaptive origin.
- AIC/BIC only for comparable likelihoods, response scales and observations;
  otherwise not_applicable. Training R-squared is descriptive only.

Preserve the workflow's targets:

    median relative error <= 5%
    P95 relative error <= 10%
    speedup sign accuracy >= 95%
    method ranking accuracy >= 90%
    knee relative error <= 10%, only when a valid independent reference exists

For predicted same-work ratios, fit a separate BF16 baseline using the
appropriate training folds. Use exact B,L, mode and work matches for observed
ratio targets. Distinguish fully predicted ratios from predictions conditional
on an observed BF16 denominator; never silently mix the two.

For ranking, compare configurations only when their held-out predictions and
observed same-work outcomes are jointly available. Define the pairwise ranking
metric before scoring, include denominators, and separate uncertain near-ties
without removing their primary errors. Do not compare in-sample predictions
against out-of-fold predictions as one held-out ranking result.

No observed BF16 counterpart means no observed speedup target. A modeled BF16
extrapolation is not a measured baseline. Keep capacity-feasibility advantages
separate from latency ratios.

## 8. Quantify knee identifiability and uncertainty

Report local A/B/C fits for up to 10 configs x 5 batches, with actual coverage.
A converged segmented optimizer does not prove an observed knee.

Use explicit statuses such as:

    identified_in_range
    below_observed_range
    above_observed_range
    linear_or_constant_preferred
    weakly_identified
    insufficient_feasible_span
    fit_failed

Only report a finite claim-bearing knee interval when observations support
both regimes and the fitted transition is identifiable. Do not recycle the
Pilot's knee label as ground truth or force every curve into a kink.

Use process/session-level bootstrap, not decode-step resampling. Preserve
shared replicate-block structure for cross-configuration comparisons; do not
count an infrastructure replacement as an additional session. Reuse accepted
bootstrap settings; otherwise freeze 500 resamples and a fixed seed. Report
failed or non-identifiable resamples instead of dropping them to narrow CIs.
With only five replicate segments, describe intervals as conditional on this
sampled machine/session design.

Distinguish uncertainty in the fitted mean from prediction uncertainty for a
new process. Report held-out interval coverage only for the type actually
constructed. Knee-error targets are not evaluable without a separately
supported reference; a full-data fitted knee is not exact truth.

Do not launch more measurements to resolve unidentifiable knees in this task.

## 9. Test the scientific claims rather than assume them

Answer from held-out results:

- Does BL/r suffice?
- Does full B,L,r improve prediction?
- Does a knee-aware structure improve it further?
- Do structural features materially improve on scalar compression ratio?
- Which findings vary by method and by interpolation/extrapolation split?

For identified knees only, evaluate B*L_star and
B*rho_alloc(B,L_star,config)*L_star as hypotheses. Do not impose either scaling
law during fitting and then describe it as a discovery.

Any derivative with respect to r is model-based sensitivity, not a measured
causal critical-path elasticity: bitwidth changes also change kernels and
implementation costs. Do not claim zero benefit below a fitted knee by fiat.

Use Phase 14/15 for interpretation only. The single common-point r_hbm and
A_traffic values must not be filled across the Full Scan or used as universal
measured covariates. Normal timing r_hbm remains null. Retain the existing
negative pure-launch-floor result; do not restore the original narrative by
choosing a favorable fit.

## 10. Export a usable offline relationship, not just fit scores

After evaluation, fit final candidate models on all eligible Full Scan data
for deployment. Keep their in-sample fit statistics separate from CV scores.

Provide a small offline predictor accepting:

    method, B, L, and either a supported method_config or an explicit r_alloc

Return:

- predicted host-wall decode latency and units;
- predicted BF16 latency and same-work ratio when supported;
- model ID, coefficients/basis parameters, feature requirements;
- training domain, interpolation/extrapolation flag, model/input fingerprints;
- uncertainty interval when estimable;
- quality_status=unvalidated.

Derive config-based r_alloc and byte features from frozen CPU-side formulas,
not a CUDA allocation or timing run. A feature-augmented prediction needs its
actual additional inputs; do not silently guess them from scalar r.

Document the formula/basis and parameter values so the exported model can be
reproduced without the GPU. Do not build a web service or dashboard.

## 11. Deliver compact analysis artifacts and focused tests

Use existing analysis modules. Add only missing preparation, fitting, split,
metric and prediction code. A small CPU environment with recorded dependencies
is sufficient; do not modify or recertify the Measurement Container.

Create one compact model bundle containing:

    input_manifest.json
    analysis_frame.parquet
    split_manifest.json
    candidate_spec.json
    out_of_fold_predictions.parquet
    model_comparison.parquet
    knee_estimates.parquet
    prediction_metrics.json
    model_target_status.json
    models/
    prediction_example.json
    plots/
    phase17_report.md
    inventory/checksums/COMPLETE using the existing writer

Make simple diagnostics: predicted-vs-measured, held-out errors by protocol,
residuals, A/B/C curve comparisons, identified-knee intervals, D/E/F comparison,
and same-work predictions with unavailable regions masked.

Focused tests only: host-wall units/supplement join, unique replacement mapping,
infeasible exclusion, no grouped leakage, train-only preprocessing, deterministic
fits, metric formulas, non-identifiable knees, exact same-work matching, and
exported predictor agreement. Reuse make fit if appropriate; add one narrow
validation target. Do not run full tests, GPU smoke, admission or historical
evidence checks.

Publish this small derived bundle through the existing R2 tool and verify its
new objects once. Reference Full Scan/Phase 15 roots rather than copying them.
Keep the publication receipt outside its own content-addressed bundle; do not
create recursive inner/outer bundles to include a receipt containing its own
root. Upload retry never requires model refitting. Do not touch prefix catalogs
or expose credentials.

## 12. Completion criteria and concise report

Execution status and scientific model targets are separate.

Phase 17 can PASS when the prescribed analysis, holdouts, uncertainty handling,
export, tests and publication are complete even if every candidate misses a
prediction target or many knees remain unidentified. Record target results as
met / missed / not_evaluable; never relax targets or rerun experiments to force
PASS. A negative answer to B,L,r sufficiency is a legitimate research outcome.

Use PARTIAL for unfinished applicable analyses; BLOCKED only for a concrete
missing/corrupt input or execution problem that prevents the analysis.
Keep Quality LOCKED. Do not create PERFORMANCE_DATA_FROZEN or start later
validation merely because model fitting finished; follow the existing final
performance-release protocol in its own authorized task.

Return PHASE 17 MODELING REPORT with:

- Status; starting/final full HEAD.
- Source Full Scan roots and host-wall supplement used.
- Accepted process/logical-point counts and 38-replacement treatment.
- Candidate model definitions and scoped feature availability.
- Four holdout results per method, coverage and extrapolation flags.
- Median/P95 errors, speedup-sign and ranking accuracy, target statuses.
- Identified/non-identifiable knees and uncertainty limitations.
- B,L,r sufficiency versus augmented-feature conclusion.
- Exported predictor path and one reproducible prediction example.
- R2 root/URI; focused test results.
- Quality state and next action.

After completion, Phase 18 reproducibility packaging may be proposed separately.
Do not begin it automatically. Prefer a few reviewable commits; do not push or
tag without explicit authorization.
