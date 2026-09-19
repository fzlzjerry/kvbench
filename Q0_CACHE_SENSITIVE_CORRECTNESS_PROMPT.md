# Q0 — Execute Cache-Sensitive Correctness Under the Approved QP-1 Contract

Use this task after the operator explicitly approves the contract ID and SHA-256 below. Merely uploading this document or forwarding the QP-1 report is not approval.

This task records that approval and implements/executes Q0 only. Do not run Fast PPL, Full PPL, LongBench benchmark scoring, performance timing, model fitting, or another preparation campaign. Short Q0 diagnostic generation is authorized; it is not a benchmark quality-pass result.

## 1. Bind the approved version without changing its bytes

Reported QP-1 identities:

```text
QP-1 final HEAD:
825c14db15590da29ce19351f563829646eec581

Performance freeze tag:
perf-freeze-20260917-83536c37-r1

Performance source commit:
83536c37433875cda98c36e2848e05692e9407d0

Contract ID:
quality-qp1-20260918t170812117127z-170b638c-e8f4a2

Contract SHA-256:
b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1

QP-1 contract/fixture bundle root:
6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62

Quality image:
sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32

Chat-template SHA-256:
e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65
```

Read the finalized contract at its actual snapshot path and confirm the contract digest once. Use its existing byte/canonicalization definition, not a newly invented serialization. Record the operator's explicit approval in a small separate receipt referencing the exact ID/hash, freeze tag, actual approval time, and authorization scope: Q0 only.

Do not edit the frozen contract to change `requires_human_approval` into `approved`: that would change the approved bytes. The approval receipt supplies effective approval. If the existing non-hot-path launcher only understands an embedded approval field, make the smallest change to consume the receipt. Do not create a new contract version, tag, image, or approval framework for this.

No approval present means no GPU execution. After explicit approval is present, do not ask for it again for each Q0 subtest. Do not backdate approval or claim it existed before the performance results were known.

## 2. Read only the inputs needed for this work

Read:
- AGENTS.md and the current task/status entry.
- `CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md`: Q0, cache-sensitive PPL/LongBench, invariance, failure classification, and recovery sections.
- `docs/phase_reports/qp1-quality-contract.md` and the finalized quality contract/snapshot.
- The contract's Q0 settings, dependency lock, fixture indexes, frozen numerical tolerances, and exact configuration identities.
- QP-0's corrected tag receipt, locked-path list, and external-kernel identity map.
- Existing quality helpers, frozen model/session entry points, method reference comparison helpers, and the existing artifact writer.

At entry run only `git status --short` and `git rev-parse HEAD`. A clean, documented descendant is acceptable; do not reset to the reported SHA. Use existing local fixtures and publication receipts. Fetch only a needed missing input, not entire historical bundles.

Check the protected-file diff once before execution and once after implementation. Bind the actual loaded method/extension identities through the existing helper; do not redownload or recertify the runtime. Use the existing GPU lock when launching Q0. No package audit, complete repository tests, historical admission, full R2 retrieval, or new monitoring service.

## 3. Scope: one small quality harness around the existing implementation

Evaluate all ten configurations:

```text
bf16
tq_4bit_nc
tq_k3v4_nc
tq_3bit_nc
k4v4
k2v4
k2v2
kvq4
kvq3
kvq2
```

Use the exact model, tokenizer, quantizers, skip-layer rules, residual/sink/outlier policy, backend, and method identity from the contract. No speed-based selection.

Place new evaluation code outside the 68 locked hot-path files. Reuse the current adapters and full-model forward/session entry points. Do not copy and maintain a second implementation of quantization or model forward. A small quality-only wrapper plus focused tests is sufficient; no new server, scheduler, generic runner registry, or artifact framework.

Use the pinned Quality image and evaluation dependency lock. Do not upgrade the measured runtime, rebuild method kernels, alter calibration, replace weights, or change cache layouts. Follow any runtime overlay already bound by the contract rather than silently installing packages.

If a frozen API genuinely cannot express required growing-context decoding, show the exact failing call and minimal reproduction. Do not silently fall back to a different model/backend, patch protected code, or declare all performance data invalid merely because the new harness is incomplete.

## 4. Freeze one bounded Q0 run plan before producing results

Use Q0 sample IDs, lengths, generation budgets, modes, and tolerances already specified by the approved contract. Do not substitute the entire PPL/LongBench corpus for Q0.

The existing protocol requires diagnostics across short, medium, and long prefixes: 512, 4096, 16384, 24576, 32768, 65536, and a safe near-maximum length. Read the actual near-maximum budget from the contract; reserve space for conditioning and diagnostic generation. Never use a full maximum-length prefix and then append beyond the model limit.

Use the contract's fixed graph/batch-invariance sample set and budget; the parent protocol specifies a 100-sample batch-invariance set at B={1,4,8}. Do not silently reduce an approved budget. These checks are Q0 scope, not another performance admission grid.

For purely mechanical Q0 details not fixed by the contract, use a small deterministic rule and record it once in `docs/plans/q0-cache-sensitive-correctness.md` before inference. For example, select the first eligible frozen input ID for a core length probe. Do not add research margins or choose cases based on model outputs. A missing contractual threshold stays missing; never invent a BF16-agreement threshold to reject quantization quality.

Record concrete unit count and maximum diagnostic token count, not a runtime estimate. Deduplicate identical tests/evidence across sections and keep already valid Q0 units reusable.

## 5. Establish the correct decode semantics

Quality decoding advances through real positions. It must not repeatedly overwrite the fixed-L benchmark scratch slot.

For each independent input:
1. Load the frozen logical tokens/positions.
2. Allocate capacity for prefix, conditioning, and diagnostic continuation.
3. Build the method cache through the existing prefill path.
4. Advance token-by-token using the existing append/decode operations.
5. Respect sequence lengths, attention masks, and per-sequence positions.
6. Consume the output logits before any reusable output buffer is overwritten.
7. Reset state between independent cases; no stale cache, omitted tokens, or duplicated appends.

Preserve source-faithful semantics: KIVI residual rollover, KVQuant pre-RoPE K and sink handling, fixed sparse policy, and TurboQuant compressed versus skipped layers. The contract and frozen implementation, not remembered defaults, define the layout.

Construct caches in memory from shared logical inputs. Do not create another disk catalog of materialized KV caches. Do not request TB-scale prefix storage.

## 6. Prove the PPL execution split uses compressed cache

Use small Q0 control inputs, not the complete PPL anchor list. Preserve the approved alignment:

```text
prefix = ids[:L]
burn_in = ids[L]
targets = ids[L+1 : L+1+H]

prefill(prefix)
logits = decode_one(burn_in)
# These logits predict targets[0]; burn_in itself is unscored.
```

For later targets, teacher-force the preceding true token and use the resulting logits. Follow the contract's scoring masks, document-boundary handling, and trailing-decode convention. Never score a target from logits generated after feeding that target, and never score stale prefill logits.

In Q0, test alignment with labeled tiny controls and collect logit diagnostics. Do not calculate aggregate Fast/Full PPL or inspect their pass/fail scores. The complete PPL runners execute later.

Use the identical prefill/decode split for BF16. Ordinary full-sequence forward may be a clearly labeled reference control where valid, not the main quantized scoring path.

## 7. Prove the LongBench answer depends on compressed cache

Take the exact already-tokenized effective prompt from QP-1, including its frozen task/chat formatting. Use:

```text
prefill_ids = prompt_ids[:-16]
conditioning_ids = prompt_ids[-16:]
```

Decode all 16 conditioning tokens in order. Only logits produced after the final conditioning token may initiate answer generation. Verify concatenation identity. Do not insert dummy text, retokenize with a different wrapper, or silently switch to native-prefill answer logits.

Exercise this as a small Q0 diagnostic control. Do not run the 3,668-example LongBench-E suite, score the 321 eligible v2 primary examples, or run the CoT benchmark in Q0. Fixed invariant samples may produce diagnostic answers, but they are not benchmark admission scores.

## 8. Demonstrate actual cache dependence, not only function calls

For every compressed configuration, demonstrate both executed-path evidence and output dependence:

- Identify the method-specific decode operation/kernel after conditioning and before the scored/answer logits. Reuse existing call counters or one small quality-only trace; no broad Nsight campaign.
- In a disposable debug cache, change valid encoded values in active compressed historical storage and show a reproducible change in the logits. Do not perturb only the uncompressed residual, sink, skipped layers, padding, or inactive slots.
- Preserve valid metadata, sparse index bounds, shapes, and lengths during the controlled perturbation. Cache corruption producing NaN is not successful dependence evidence.
- Use a quality-only negative control that disables/intercepts the compressed decode read and makes the test fail explicitly. It must not succeed by returning stale logits or falling back to dense attention.
- Repeat the unperturbed control enough to distinguish deterministic perturbation effects from the contract's normal numerical tolerance. Use the frozen criterion; do not tune a threshold after seeing the effect.

All perturbation/disable controls operate on disposable test state. They never touch calibration, model weights, finalized fixtures, or production evaluation inputs. Record these as fault-injection controls, not ordinary method outputs.

Existing performance traces support implementation lineage; they do not replace proving that the new quality path actually consumes the cache.

## 9. Separate implementation correctness from quantization differences

Use the frozen per-method reference as the correctness oracle for lossy representations. Quantized round-trip values need not equal the original BF16 values. Do not require packed payloads to be lossless relative to BF16.

Collect the protocol's cache/attention and logit diagnostics: K/V MSE and cosine, attention-output error, attention-score divergence, next-token KL, logit cosine, top-1 agreement, top-5 overlap, and the BF16 top-1 rank. Define direction/units where needed, and reuse existing metrics rather than making a large analysis stack.

Run short teacher-forced and greedy controls. Report first-divergence position, edit/repetition diagnostics, invalid text, and termination behavior with the defined denominator/budget.

Quantized versus BF16 divergence alone is not an implementation hard failure. In particular, a changed greedy token is not automatically a bug or a quality-gate failure. Record algorithmic degradation for later PPL/LongBench assessment.

Q0 hard failures are the actual protocol conditions: NaN/Inf, illegal tokens or cache corruption, wrong GQA/position/mask behavior, source-reference mismatch beyond its frozen tolerance, materially inconsistent execution modes/batches, absent compressed-cache dependence, or mismatched method identity.

## 10. Test the quality path's eager/Graph and batch invariance

Compare the same exact configuration against itself, not different quantization methods against BF16 with an equality requirement.

For eager/Graph:
- Start from equivalent cache state and identical tokens/positions.
- Advance through multiple distinct tokens and lengths.
- Check logits under the frozen tolerance, selected-token/answer consistency according to the contract, and compressed-cache reads in both modes.
- Do not reuse a fixed-L graph with stale captured lengths/positions as if it were a growing-context graph.
- Use the existing supported graph strategy. If the approved quality lane is eager-primary, retain that label and validate the required relationship to the frozen Graph performance path. Do not pretend its execution fingerprint is identical merely because the algorithm name matches.

For B={1,4,8}:
- Preserve each sample's logical tokens, valid length, positions, and scoring mask.
- Compare its batched result to its B=1 result.
- Exercise the approved padding/masking cases; do not claim padding support from an equal-length-only test.
- Batch unequal-length samples only through an API that actually supports their masks/lengths. Do not add mask semantics by silently altering the frozen model.

Reuse known admission evidence for untouched low-level kernels. Add only the new quality-execution checks. Use focused memcheck/initcheck coverage of genuinely new multi-step append/masking paths per distinct executed kernel family where required by Q0; do not rerun every historical sanitizer case. An unavailable tool run is not a passed check.

## 11. Resume units, preserve outcomes, and isolate failures

Reuse the existing atomic writer and resumable Q0 units. Preserve completed valid units and original failed attempts. Do not restart every configuration for one I/O error.

Apply the approved retry policy: only infrastructure/transient-I/O/incomplete-output failures may be retried, with a new attempt ID and the original retained. Do not select the best of multiple valid outputs, retry low-agreement results, or discard unexpected but valid quantization behavior.

A failure in new scoring alignment or wrapper plumbing is a quality-harness problem until evidence ties it to a frozen implementation. Fix only non-protected quality code; rerun affected diagnostic units and report which earlier units became obsolete. Do not invalidate all old performance data by default.

A demonstrated frozen-adapter/kernel correctness defect requires isolating the affected configuration/shape/source identity. Preserve evidence, stop its downstream eligibility, and report the affected performance scope. Do not repair protected code or launch performance reruns in this task.

An upload, receipt, or telemetry-query failure does not prove model failure and does not require rerunning valid GPU outputs. True CUDA corruption must terminate the affected process before any further results are trusted.

## 12. Outputs and focused completion checks

Produce one compact Q0 bundle containing:
- Approval receipt reference and contract ID/hash.
- Q0 execution plan, input IDs, execution/quality-image identity, and exact method mappings.
- Per-configuration/per-test results and failure reasons.
- Teacher-forced/greedy diagnostics.
- Cache-dependence controls.
- Eager/Graph and batch-invariance results.
- Focused sanitizer evidence where required.
- Raw diagnostic outputs sufficient to reproduce reported metrics, stored in a safe format; keep full-vocabulary logits only where needed, not every case by default.
- Existing inventory/checksum/COMPLETE structure and a short Q0 report.

Use focused tests only: approval binding without contract mutation; teacher-forcing alignment; prompt-token identity; advancing cache/positions; real cache-dependence controls; invariance; per-method reference comparisons; resume/retry semantics; and unchanged protected paths. No full repository suite, new performance admission, or re-freezing.

Publish only the new approval/Q0 evidence through the existing R2 tool. Reference QP-0/QP-1 roots without duplicating them. Verify new objects once; keep publication receipts outside the bundle they identify. Use credentials through the existing host-side path without printing or including them in artifacts. Upload retry never requires rerunning Q0. Do not delete historical prefix catalogs or raw evidence.

## 13. Finish Q0, not the whole quality study

Report preparation/execution completeness separately from each configuration's Q0 result. A completed suite with a demonstrated implementation-invalid configuration is a real result, not a reason to conceal it or force every configuration to pass. Unfinished required tests remain explicit; do not promote them to PASS.

Use the repository's existing authorization/status fields. Human approval authorizes this Q0 task; it does not make any configuration `quality_pass`. Keep all frozen performance outcomes quality-unvalidated until the prescribed later quality join.

Stop after Q0. Report which configurations are eligible for the subsequent Fast PPL stage under the approved protocol, without launching it. Do not skip Q0 failures, change thresholds, or run LongBench directly.

Return a concise Q0 REPORT:
- Status; actual starting/execution/final HEAD.
- Approval receipt and unchanged approved contract ID/hash.
- Exact image/configuration scope; planned/completed Q0 units.
- Per-configuration correctness verdicts.
- PPL alignment and LongBench suffix/decode-cache dependence.
- Teacher-forced/greedy diagnostics, with BF16 differences distinguished from bugs.
- Eager/Graph and B=1/4/8 invariance.
- Sanitizer/new-path coverage and any untested conditions.
- Protected-path comparison; concrete affected source scope if a defect exists.
- Preserved/retried attempts and compact R2 root/URI.
- Fast PPL eligibility; confirmation no PPL/LongBench benchmark scores or performance reruns occurred.

Prefer a few reviewable changes. Do not push branches/tags or create another approval stage for already authorized Q0 work.
