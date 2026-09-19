# Q0 — Diagnose the Shared Cross-Batch Failure and Repair Only a Proven Harness Defect

This is a targeted continuation of Q0, not a new admission campaign or a request to make all configurations pass. Diagnose the shared batch-invariance failure, starting with BF16. A demonstrated defect confined to non-protected quality code may be repaired in this task. Preserve the approved contract and frozen performance implementation.

Do not rerun the full 110-unit Q0 suite, performance experiments, QP-0/QP-1, model fitting, historical admission, or historical bulk R2 verification. Do not start Fast/Full PPL or LongBench scoring. Do not change thresholds or automatically waive the failed gate.

## 1. Use the present state and only relevant evidence

Reported current HEAD:
`5dd0142a317e4d7046ef84b9daf1ac531ea91870`

Approved contract ID:
`quality-qp1-20260918t170812117127z-170b638c-e8f4a2`

Approved contract SHA-256:
`b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`

Corrected performance tag:
`perf-freeze-20260917-83536c37-r1`

Quality image:
`sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`

Q0 result root:
`2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0`

The report states 110 completed units: 100 passing non-batch units and 10 failed batch-invariance units. Treat the duplicated report in the conversation as one execution, not two.

Read only:
- AGENTS.md, current Q0 status, and `docs/phase_reports/q0-cache-sensitive-correctness.md`.
- The approved contract's actual batch-invariance definitions, sample IDs, modes, output/answer requirements, tolerances, and their referenced authority.
- Q0's batch-invariance worker, input builder, comparison/aggregation code, and the quality lifecycle wrapper they use.
- Existing per-sample results for the ten failed batch units, and saved logits/input receipts where present.
- Relevant frozen model/session entry points, read-only, if needed to understand shapes and state handling.

At entry only record `git status --short` and `git rev-parse HEAD`. Use the already accepted image and receipts. Fetch a missing compact Q0 result only; do not download unrelated campaigns or build images.

## 2. Audit the comparison on CPU before any new inference

Produce one small per-sample table, using existing outputs as far as possible. Include configuration, sample ID, B=1 versus B=4/B=8, row, effective length, decode step, mode, input identity, exact failing assertion, and available error metrics.

Determine whether each failed unit means:
- a logit closeness failure;
- a selected-token/answer failure;
- both;
- or an input/state/shape mismatch.

Record the denominator for each agreement rate. Separate first conditioned-token agreement, teacher-forced per-step agreement, generated sequence agreement, and parsed task-answer agreement. Do not treat these as the same metric.

Verify that each comparison is the SAME configuration against itself at different batches, not a quantized configuration against the BF16 baseline.

Check these concrete possibilities:
- Same sample and exact logical token sequence, not merely the same random seed for differently shaped random generation.
- Same prompt wrapper, positions, valid length, conditioning split, decode token and effective history.
- Correct row mapping after batching, sorting, padding and unpacking; no broadcasting in the comparison.
- Correct valid-token/logit extraction. A padded physical position is not necessarily the logical final token.
- Actual support for per-row masks/lengths, rather than silently sending ragged inputs through an equal-length-only API.
- Equivalent fresh state; no reused mutated cache, duplicated append, stale lengths or shared cache rows.
- Comparison outputs copied into independent storage before another decode overwrites a reusable buffer. A detached view alone does not guarantee independent storage.
- Identical teacher-forced inputs. After independent greedy trajectories diverge, later logits no longer have the same conditioning history; keep trajectory divergence as a separate diagnostic.
- Same model/eval state, inference dtype and recorded backend policy.

Trace the exact tolerance values to the approved contract and referenced code. Earlier admission records distinguish attention/Graph tolerances from full-model-logit tolerances; this is an audit clue, NOT permission to substitute an older, looser tolerance. If the approved contract explicitly specifies 0.02/0.02 for these logits, it remains authoritative in this task.

If saved data already proves a comparator or input-mapping error, do not launch exploratory GPU work just to rediscover it.

## 3. Expand the numerical diagnostic, without changing the gate

For aligned finite logit vectors, use B=1 as the reference and retain the exact contract closeness predicate. For the reported tolerance it is:

    abs(z_B - z_1) <= 0.02 + 0.02 * abs(z_1)

Calculate diagnostics in FP32 or FP64 after copying the existing outputs; that does not change inference precision.

Report:
- maximum, median and P95 absolute logit difference;
- count/fraction of elements violating the ORIGINAL predicate;
- normalized discrepancy `abs(z_B-z_1)/(atol+rtol*abs(z_1))`;
- count of failed samples, separate from count of failed logit elements;
- top-1/top-2 scores, candidate IDs, top-two margin, and selected-token changes;
- an explicitly directed log-softmax/KL diagnostic where logits are available;
- first divergent decode position and whether histories were still identical.

Define selected-token semantics from actual code: unrestricted vocabulary argmax, restricted option scores, or parsed generated answer. Do not assume A/B/C/D are single tokens or replace the approved parser.

For an unrestricted unique argmax, the elementary diagnostic is:

    gamma = largest(z_1) - second_largest(z_1)
    epsilon = max(abs(z_B-z_1))

`gamma > 2*epsilon` guarantees preservation of that argmax for the aligned vectors. `gamma <= 2*epsilon` only makes a flip possible; it does NOT establish harmless numerical noise or justify discarding the sample. Apply the bound to the actual score space only.

Small per-element differences, near ties, or a similar BF16 failure do not automatically satisfy the approved gate.

## 4. If inference is needed, use a bounded BF16 reproducer first

Select at most three BF16 failing inputs from the frozen sample set, deterministically by sample ID while covering distinct recorded failure types where available. These are diagnosis inputs, not an unbiased new evaluation sample. Preserve original lengths unless an explicitly labeled reduced reproduction is needed.

Start in eager mode to avoid adding Graph as another variable. Reuse the existing model/session, not a second full-model implementation.

Use the smallest useful controls:
1. Two independent B=1 reconstructions with identical input and teacher-forced continuation.
2. The same input duplicated intentionally into B=4 and B=8, with identical lengths and no padding. Inspect every row. Label duplication as a diagnostic control, not formal batch-coverage evidence.
3. For one reproducible failure, put the input beside different equal-length companions, then permute rows with correctly corresponding inputs/state and restore logical row order.
4. Exercise the original mixed-length/padded case only if it was in the approved test and the frozen API actually supports it.

Do not run every combination on all 100 samples during diagnosis. Stop expanding controls once a mechanism is localized. Same-batch repeats and companion/permutation probes diagnose state leakage or alignment; do not demand new bitwise guarantees beyond the frozen numerical criteria.

Compare identical histories first. Independently growing greedy histories are unsuitable for attributing later logit differences solely to batching.

If unequal-length operation is unsupported, report that precise limitation. Do not truncate inputs, silently serialize a B=8 request as eight B=1 calls, replace the backend, or claim the required padded case passed using an equal-length control.

## 5. Localize a surviving BF16 difference before blaming quantizers

If tokens, positions, masks, state, output ownership and comparison code are correct, inspect one failing BF16 input at a few existing boundaries: embedding, early/middle/final hidden state and final logits. Add a read-only quality diagnostic hook only if needed, and keep it out of normal execution.

Determine whether discrepancy is introduced in prefill/cache construction, at a particular decode operation, or gradually across common model computation.

Record GEMM/attention precision and dispatch settings for the reproducer. PyTorch does not guarantee batched operations and sliced operations to be bitwise identical. This is a possible mechanism, not a blanket explanation of the observed magnitude.

Only when an operation is localized, a small higher-precision or otherwise controlled reference on captured inputs may be used to test that mechanism. Label it diagnostic-only, record the changed arithmetic, and never use it as the approved method result. Do not globally switch the model to FP32, toggle deterministic flags as a supposed universal cure, or install batch-invariant kernels to make Q0 pass.

No broad Nsight or new sanitizer campaign is required for a read-only quality-harness diagnosis. A concrete indication of memory corruption must be isolated in the affected process and reported.

## 6. Act on the demonstrated cause

### A. Defect confined to quality input/comparison/lifecycle code

Repair the smallest non-protected function, add a focused regression reproducing the defect, and continue in this same task. Do not create a separate approval stage for an ordinary harness fix.

If retained raw tensors suffice to correct a wrong comparison, recompute on CPU without new inference. If the old execution used wrong inputs/state or lost the needed output, run new affected batch units with correct semantics and new IDs.

For a shared batching fix, rerun the ten affected batch-invariance units over the original approved 100-sample set and B={1,4,8}. Do not rerun the other 100 passing units unless a dependency review shows that the same defect invalidates them too. Do not claim all previous passes survive automatically.

Preserve the original failed outcomes as superseded-by-a-demonstrated-harness-fix evidence, not as selectively discarded low-agreement results. Show before/after results.

### B. Genuine cross-batch arithmetic sensitivity, with no demonstrated semantic defect

Keep the original batch gate FAILED. Report the aligned error distribution, margin diagnostics, same-batch repeat behavior, localized operation and remaining uncertainty.

Do not raise tolerances, replace raw-logit checks with KL/top-k, excuse quantized failures by subtracting BF16 error, or turn all ten configurations into PASS. A proposed tolerance/metric or B=1-only execution amendment is a separate, explicit human decision and must preserve the old failed result. Do not implement that amendment here.

Explain whether evidence supports continued investigation under B=1, but do not start PPL. Such evidence would not by itself validate B>1 quality or the full B-dependent performance/quality join.

### C. Frozen shared-model/runtime/adapter defect

Produce the minimal reproducer and identify the first incorrect operation, affected configurations, batches, lengths, modes and whether that path was used in the completed fixed-L performance study.

Do not edit protected paths or automatically invalidate/rerun every performance record. Also do not declare them unaffected solely because source hashes are unchanged. Assess the actual shared execution path and report the bounded impact for a separately authorized fix.

### D. Evidence remains inconclusive

State the precise unresolved observation and the smallest missing diagnostic. Do not turn a hypothesis into a root-cause statement, and do not expand into an unbounded ten-method experiment search.

## 7. Close with only new targeted evidence

Keep the approved contract bytes/hash, freeze tags, 68 locked hot paths, method binaries, calibration, runtime image, old Q0 bundle, and performance data unchanged.

Use one small diagnostic script or existing helper plus a short report and per-sample table. Save full-vocabulary logits only for the few necessary reproductions. No new validation registry, release hierarchy, retry framework or materialized KV-prefix catalog.

Run only tests for the changed comparator/input/wrapper logic and the reproduced defect. Use the existing GPU lock for actual probes. Do not run whole-repository tests, repeat entry admission, reverify historical R2 bundles or rebuild images.

Publish one compact new evidence bundle with existing tooling. Reference old Q0 results instead of duplicating them. A publication/receipt failure never requires new GPU inference. Do not print or publish credentials, weights or caches.

Return a concise Q0 BATCH DIAGNOSIS REPORT:
- Actual starting/execution/final HEAD.
- Exact failed assertion and tolerance provenance.
- Root cause: demonstrated / supported hypothesis / unresolved.
- Input/state/alignment findings and first divergent operation.
- BF16 per-sample error and margin summary; actual agreement denominators.
- Changed files and whether any protected implementation changed.
- Original valid units reused, invalidated units, CPU reanalyses and GPU reexecutions.
- Per-configuration current batch-gate status under the UNCHANGED contract.
- Evidence-supported impact on frozen performance, including unknowns.
- New compact evidence root and next action.

Diagnostic execution can complete even when the scientific batch gate remains failed. Do not equate a completed diagnostic task with Q0 PASS. Fast PPL remains unstarted until the approved eligibility conditions are actually met or explicitly amended by the operator.
