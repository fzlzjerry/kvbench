# Phase 18 — CPU Reproduction Package and Evidence-Bounded Research Report

Execute Phase 18 only. Package the completed performance study and its offline predictor. This is CPU-side work, not another GPU validation campaign or an attempt to improve Phase 17 scores.

## 1. Start with the existing artifacts, not a new entry gate

Read only:
- AGENTS.md and the current status/task entry.
- CODEX_WORKFLOW.md: final deliverables, modeling interpretation, and reproduction requirements.
- The completed Phase 17 report and frozen analysis plan.
- The Phase 17 input manifest, candidate specification, split manifest, out-of-fold predictions, model-comparison table, target-status JSON, knee estimates, and exported models/predictor.
- The Phase 16R compact host-wall source index and replacement mapping referenced by Phase 17, only as needed to document lineage.
- The Phase 14 analysis closure and Phase 15 mechanism/traffic summary.
- The repository's existing post-performance quality protocol, only for the release-to-quality handoff. Do not run that protocol here.

At entry, run only `git status --short` and `git rev-parse HEAD`. Record the actual current full SHA. The Phase 17 modeling execution SHA below is not necessarily the final reporting commit; do not reset to it or invent a final SHA.

Use existing local compact inputs and accepted publication receipts. Fetch only a specific missing file needed for packaging. Do not download historical raw campaigns, model weights, prefix catalogs, trace archives, or calibration payloads. Do not rerun package-lock checks, full tests, admission, CUDA tests, or R2 verification of historical bundles.

Known Phase 17 identity:
- Modeling execution HEAD: `9e20208a5eab4f14263334436878492654756ce9`.
- Bundle: `phase17-20260916t164055992658z-9e20208a-c944c5`.
- Root: `05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb`.
- R2 locator: `r2://kvbench-artifacts/kvbench/sha256/05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb/`.
- Published objects: 44; existing clean retrieval passed.
- Data: 2,205 accepted process observations, 441 feasible logical points, five replicate slots per logical point.
- Infrastructure replacements: 38, each replacing a slot rather than adding statistical weight.
- Quality: LOCKED.

## 2. Preserve the actual scientific outcome

Phase 17 execution passed; its numerical prediction targets were missed. These are different statuses.

Retain the frozen candidate labels:
- E: scalar byte law.
- RQ2: B/L/r response surface.
- D: positive knee response surface.
- F_shape: D plus predeclared static byte-shape features.
- F_diagnostic: additional observed kernel count; diagnostic-only.

Scientific selection and deployment default are D. Do not replace D, refit a new model family, tune complexity, alter thresholds, or remove poor folds to obtain better results in Phase 18.

Preserve these reported outcomes, using full precision from saved artifacts:
- Median relative-error target missed: reported aggregate 0.079199.
- P95 relative-error target missed: reported aggregate 1.645284.
- Same-work speedup-sign accuracy: 425/456, approximately 93.2018%; target missed.
- Pairwise ranking accuracy: 346/513, approximately 67.4464%; target missed.
- Knee-relative-error target: not evaluable without an independent knee reference.
- Identified in-range curves: 23 of 50; remaining 27 non-identifiable or better described by linear/constant alternatives.
- Uncertainty: 1,000 shared replicate-segment bootstrap draws, conditional on the five sampled sessions.

Describe E → RQ2 → D as relative improvement among tested candidates under the declared selection rule, not proof that B/L/r is universally sufficient. The absence of a robust F_shape improvement does not establish that metadata or workspace costs are irrelevant. F_diagnostic is not a deployable improvement when its required kernel count comes from the held-out run.

Selection used outer results. Preserve the disclosure that the selected outer score is not an unbiased evaluation of an additional model-selection procedure. Do not call the result nested-CV-validated unless the stored experiment actually performed that evaluation. Do not add another selection search in this task.

## 3. Add one small reporting audit, using existing predictions only

This is a reporting task, not a new modeling phase. Use one short script or the existing analysis code. No new framework, database, approval stage, or GPU run.

### Aggregate metric labels

Read the exact frozen macro rule. From the supplied rounded table, the average of the 11 applicable geometry-holdout cell medians is approximately 7.919909%, and the average of their P95 values is approximately 164.528273%. These match the reported aggregate values to rounding.

Confirm the actual rule from saved code/configuration. Label these quantities explicitly, for example:
- macro_mean_of_cell_median_relative_errors;
- macro_mean_of_cell_p95_relative_errors.

Use the actual frozen formula if it differs. Document the cells, weights, and exclusion of not-applicable BF16 config holdout and separately reported session holdout where applicable. Do not describe an average of group quantiles as a pooled quantile. Do not change the scoring rule or target statuses.

### BF16 leave-one-batch-out tail

The reported BF16 P95 error is 1625.690%, while its median is 3.885%. Export the 20 largest existing BF16 batch-holdout errors, or all rows if fewer are available, with:
- fold and held-out B;
- actual L and feasibility/support domain;
- observed and predicted host-wall milliseconds;
- signed and absolute relative error;
- training B/L support;
- interpolation/extrapolation classification;
- saved model/parameter identity.

Check only concrete implementation/reporting possibilities: units, full-batch versus per-token normalization, inverse transform, coordinate scaling, wrong fold/model joins, baseline r=1, and non-finite or impossible output handling. Do not assume an error before examining the records.

If arithmetic and joins are correct, retain these rows as real predictive failures and document where they occur. Separate interior interpolation from edge extrapolation in a supplementary summary without replacing the primary metric. Do not clip predictions, exclude the worst rows, or refit to conceal failure.

If an actual arithmetic/export/join bug is demonstrated, preserve the original Phase 17 artifact; make the smallest correction and rerun only the affected CPU analysis under the same frozen specification into a new version. Report old and corrected values. Do not rerun timing or change candidate families. A large error alone is not a bug.

### Interpretation discipline

Report the TurboQuant leave-one-config-out median 29.850% and P95 44.554% as a transfer limitation of the tested model/data. Do not attribute it to one kernel or K/V bit asymmetry without supporting analysis. The same applies to better KIVI/KVQuant config-holdout medians: show their P95 errors too.

Do not turn approximately 67.45% pairwise ranking accuracy into a reliable automatic method selector. Keep near-tie conventions and scored-pair denominators unchanged.

Save this audit as a small table and note included in the reproduction package, not another gate report.

## 4. Package the frozen offline predictor

Reuse the Phase 17 predictor. Do not create a service, web app, plugin system, generalized model registry, or new dependency stack.

Expose the existing CPU interface for:
- method family;
- B;
- historical L under the frozen convention;
- supported method configuration, or explicit r_alloc where the model supports it.

Return the existing model's:
- predicted host-wall milliseconds per full-batch decode step;
- predicted BF16 latency and fully predicted same-work ratio where supported;
- exact model ID and parameters;
- input/configuration domain and interpolation/extrapolation annotation;
- uncertainty only of the type actually computed;
- quality_status=unvalidated.

Document the exact D parameterization, basis, transforms, normalization constants, coefficients, units, and allocation-ratio definition from the implementation. Do not invent a closed form in place of the fitted representation.

Keep r_alloc point-dependent and configuration-derived where appropriate. It is not an independently randomized variable in this study. Only three main compressed configurations per family were measured. An explicit numeric r does not demonstrate validity for arbitrary quantizers or unseen configurations.

Preserve the poor extrapolation results in the model documentation. Do not silently clip extrapolated predictions or substitute measured BF16 values for predicted BF16 values. Distinguish a fully predicted ratio from a ratio conditional on an observed baseline. A prediction outside measured feasibility support is not evidence that the workload fits in memory.

Add a few small offline examples using supported inputs already covered by the stored model. Verify that the serialized predictor reproduces the existing exported outputs. Do not score those examples as a new held-out validation set.

## 5. Produce the research report and data-driven figures

Create or update a concise README, a performance-study research report, and predictor documentation. Treat this as the completed performance component, not a completed quality-preserving systems claim.

The report must contain:
1. The original research question: predicting full-model decode from method-conditioned B/L/r, with scalar-byte and knee hypotheses tested rather than assumed.
2. The actual scope: frozen model, RTX PRO 6000, one GPU, admitted source/configurations, fixed-L batch decode, CUDA Graph path, host-wall endpoint.
3. The measurement dataset and feasibility/replacement treatment.
4. The E/RQ2/D/F candidate comparisons and four distinct holdout protocols.
5. All missed targets, explicit aggregation formulas, error tails, and deployment limitations.
6. The 23 identified versus 27 non-identified/linear/constant curves; bootstrap limitations; no independent knee truth.
7. Phase 14's negative pure-launch-floor result and Phase 15's qualified mechanism evidence.
8. Reproduction instructions and exact input/model/result provenance.
9. Quality not yet evaluated; next work governed by the already existing quality protocol.

Use all methods' actual implementation labels, including pinned TurboQuant-vLLM and project-patched KIVI/KVQuant. Do not revert to the original draft's universal 4x GQA-materialization claim or zero theoretical compression benefit below a knee; later evidence did not establish those claims for this measurement path.

Phase 15 traffic features remain tied to their exact common profiler point. Do not fill r_hbm into Full Scan timing rows or infer critical-path causality from allocated bytes. Missing or inseparable profiler quantities remain unavailable.

Reuse figure scripts and saved tables. Essential figures are:
- held-out predicted versus measured latency;
- errors separated by holdout protocol, method, and interpolation/extrapolation;
- all candidate comparisons with the original macro rule;
- identified-knee intervals plus counts of non-identifiable cases;
- measured versus predicted same-work ratios on exact matched points;
- Phase 14/15 mechanism summaries with the profiler point labeled.

Keep large-error points visible. A log scale or separate panel is acceptable with explicit labels; removing them is not. Do not redraw unavailable raw traces or fabricate missing figures. No need to create dozens of new plots.

## 6. Provide a small CPU-only reproduction path

The default reproduction command must not launch CUDA, preflight, admission, model download, prefix generation, Full Scan, profiler collection, or quality evaluation.

Reuse existing commands where possible and document their real names and arguments. Provide two clear scopes:
- Default: load the compact offline package, recompute summary tables from frozen predictions, regenerate figures, and run predictor examples.
- Optional explicit command: rerun the original Phase 17 fitting/CV on the compact 2,205-row analysis dataset with the frozen specification. Do not execute this costly option merely to finish Phase 18 unless missing outputs require it.

Document GPU experiment commands separately as future operator-invoked reproduction instructions. Do not execute them in this phase.

Use a temporary CPU working directory to exercise the default path once. It must not import CUDA-dependent runtime modules just to calculate bytes or predict latency. Do not modify the Measurement Container.

Include only what the offline package needs: compact analysis input/predictions/splits, model parameters, relevant CPU code and dependencies, figures, report, and a source/result manifest. Small purposeful copies for an offline package are fine; do not duplicate large campaign, Fisher, prefix, model, or profiler archives.

## 7. Quality handoff without changing the rules

Quality remains LOCKED throughout this task. Do not install quality-only dependencies or run PPL/LongBench.

Read the existing post-performance protocol to identify the exact performance-freeze prerequisites and the next authorized quality task. Report their status using existing receipts rather than rechecking all history.

Do not create PERFORMANCE_DATA_FROZEN or change quality authorization solely because packaging finished or because Phase 17 says PASS. Prepare the release manifest that the existing freeze procedure requires and state the next action under that procedure. Do not invent another multi-stage governance framework or new quality thresholds.

Do not hide failed configurations or tune quantization after looking at future quality results without treating that as a separate implementation/version.

## 8. Tests, publication, and completion

Run focused CPU tests only:
- metric aggregation/percentage units;
- preserved outlier and not-applicable statuses;
- predictor serialization/example agreement;
- ratio baseline/scope labeling;
- default reproduction does not launch GPU work;
- source references and figure generation.

Do not rerun full tests, historical admission, package checks, raw-data downloads, or model selection. Update the exact scope allowlist only for genuinely needed release files. Ordinary documentation changes do not require a new decision per file.

Publish one compact new reproduction bundle using the existing R2 publisher. Keep large source artifacts referenced by their existing roots. Verify the new bundle once. Keep its publication receipt outside the bundle that it names; do not create recursive inner/outer bundles. Upload retries never require refitting or rerunning experiments. Do not expose .env or credentials, delete prefix catalogs, or push/tag Git without authorization.

Phase 18 completion means the CPU reproduction path, honest research report, predictor documentation, focused tests, and small release publication are complete. It does not mean the missed scientific targets became met. Real input/code problems must be described; poor model accuracy alone does not block packaging.

Return a concise PHASE 18 REPORT with:
- execution status and actual starting/final HEAD;
- Phase 17 source root;
- macro metric definition and BF16 tail diagnosis, including unknowns;
- frozen model/default and unchanged scientific target statuses;
- CPU reproduction command/result and predictor example;
- research report and figure paths;
- new compact R2 root/URI;
- Quality status and the exact existing freeze/quality handoff next action.

Prefer a few reviewable commits. Do not start quality evaluation automatically.
