# Explicit B=1 Quality-Scope Amendment and Q1A Fast PPL

Execute this task only after the operator explicitly authorizes the scope amendment described below and Q1A execution. Merely forwarding the diagnosis report or adding this file is not approval.

The task is one continuous piece of work: record the precise scope amendment, reuse eligible B=1 Q0 evidence, then execute Fast PPL. Do not create another approval campaign, batch-invariant runtime project, performance rerun, or full-Q0 rerun.

## 1. Starting point and bounded interpretation

Reported current HEAD:
`a89ddc183744d87889c4e0fc25cb8b62889dadf1`

Original approved contract:
- ID: `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`
- SHA-256: `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`

Performance freeze tag:
`perf-freeze-20260917-83536c37-r1`

Quality image:
`sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`

Existing evidence roots:
- QP-1 contract/inputs: `6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62`
- Original Q0: `2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0`
- Batch diagnosis: `eedee1596bb36692da8557fc001c45593f50f930e62c27227a7151ab6aa88ba6`

The diagnosis reports BF16 differences first at layer-0 q_proj/v_proj on three bounded inputs, before the first attention cache read. Same-B repeats, duplicate rows, companion changes, permutations and input/state ownership controls passed on the diagnostic cases. The comparator orientation was corrected without changing the original failures. All ten original cross-batch units remain FAILED.

Do not generalize this BF16 diagnosis into proof that every quantized cross-batch difference is harmless. It does not establish cross-batch task-quality equivalence. It is the disclosed motivation for restricting the next quality study to B=1.

## 2. Read only what is needed

Read:
- AGENTS.md and the current quality status/task entry.
- The finalized approved contract and its Fast-PPL, sampling, margin, statistics, execution-mode and retry fields.
- `docs/phase_reports/q0-cache-sensitive-correctness.md`.
- `docs/phase_reports/q0-batch-diagnosis.md` and `docs/evidence/q0/batch-diagnosis.json`.
- Existing Q0 result indexes for the B=1 core, cache-dependence, alignment and eager/Graph units.
- QP-1 Fast-PPL stream/anchor/mask manifests and dependency/runtime records.
- Existing quality harness, teacher-forcing helpers, artifact writer and publisher.

At entry record only `git status --short` and `git rev-parse HEAD`. Accept a clean documented descendant. Use accepted local receipts; retrieve only an actually missing compact input. No full repository tests, package audit, admission, historical R2 retrieval, model refit, image build, profiler or new GPU-entry campaign.

Use the existing protected-path check once before inference. Preserve the 68 locked hot-path files and external method binaries. The one-line quality comparator correction is not a new performance implementation.

## 3. Record the exact authorized scope amendment

Preserve the original contract snapshot, SHA, approval, all Q0 results, diagnosis and performance freeze unchanged.

Create one small versioned amendment referencing the original contract ID/SHA and the diagnosis root. Use the existing contract mechanism. Prefer a base-contract-plus-amendment representation; if the launcher requires a complete effective snapshot, write a new successor snapshot without overwriting the original. Do not create a generic policy engine.

The ONLY scientific changes authorized by this task are:

1. Formal quality inference in this study track is restricted to physical model batch B=1.
2. The original cross-batch raw-logit/selected-token requirements remain FAILED and disclosed. They are not converted into passes, erased, or reinterpreted as evidence of equivalence.
3. Cross-batch raw-logit agreement is no longer a prerequisite for executing this explicitly B=1-only track. It remains a barrier to transferring B=1 quality qualification to other batches.
4. B=1 numerical correctness, source/reference fidelity, finite outputs, state progression, real compressed-cache dependence, approved prompt/scoring alignment, and required same-B execution-mode consistency remain mandatory.
5. All PPL/LongBench quality margins, sample IDs, tokenization, document masks, bootstrap rules, stage sequencing and finalist policies remain unchanged.
6. No method-specific exemption is granted: the same B=1-only rule applies to BF16 and all nine compressed configurations.
7. B=2/4/8/16 performance observations remain performance-only and are not automatically quality-qualified by results from this track. Further quality evidence at those batches would require a separate defined study, not a name-only join.

Record chronology truthfully: this amendment is specified after seeing Q0 batch failures and the BF16 diagnosis, but before Fast PPL or LongBench benchmark scores. Do not describe the amended whole protocol as unchanged preregistration.

Record the operator's actual approval text/time and authorized scope in an external receipt. Finalize/hash the amendment before producing benchmark scores. The operator's explicit approval of these exact rule changes also authorizes their deterministic materialization; do not ask for approval again merely because the new file's computed hash was not known in advance. Do not claim that the new hash was approved before it existed. Any additional substantive change is outside this authorization.

Use explicit scope fields or their existing equivalents:

```text
original_cross_batch_gate = failed
quality_evaluated_batch_sizes = [1]
quality_transfer_to_other_batches = not_established
amendment_reason = scope_restriction_after_Q0_batch_failure
quality_status = unvalidated
```

Do not set global Q0=PASS. Produce only per-config B=1 eligibility under the amended contract, referencing the existing required successful B=1 units. No new inference is needed merely to aggregate this eligibility. A missing or failed B=1 requirement stays blocking for that configuration; the amendment does not waive it.

## 4. Reuse the ten configurations and frozen Fast-PPL inputs

Configurations:

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

Use every configuration with complete B=1-scoped Q0 eligibility. Do not select by speed, prior predictions, compression benefit or observed PPL. BF16 is the paired baseline, not a quantized candidate to score against itself as evidence of non-inferiority.

Use the actual approved Fast-PPL fixtures. Expected settings inherited from the protocol are:

```text
batch_size = 1
prefix_lengths = [4096, 24576, 32768, 65536]
anchors_per_length = 16
scored_horizon = 128
burn_in_decode_tokens = 1
datasets = WikiText-2 raw test + frozen C4 validation subset
```

The materialized approved anchor IDs, masks and actual scored-token counts are authoritative. Reuse them, without resampling, retokenizing, changing document-safe boundaries or altering budgets. Print the concrete planned anchor/scored-token totals from the manifests, not an estimated runtime.

Reuse the pinned image, dependency overlay/lock, model revision, tokenizer, calibration, backend, method configuration and approved primary execution mode. B=1 applies to prefill and decode model invocation, not merely to a loader argument or a final summary. Do not silently pack separate documents into a larger physical model batch or pad them to B=8.

Do not change BF16/FP16 inference precision, GEMM flags, accumulation settings, attention backend, kernels or quantizer settings to make results agree across batch. Do not serialize a claimed B=8 execution as B=1 calls. This track explicitly runs B=1.

## 5. Implement only the missing Fast-PPL loop around the validated Q0 path

Reuse the existing growing-context model/session and the Q0-proven compressed-cache path. Add only non-protected quality/scoring code. No second model-forward implementation, generic evaluator framework or new cache implementation.

For each approved anchor use:

```text
prefix = ids[:L]
burn_in = ids[L]
targets = ids[L+1 : L+1+H]
```

Execution:
1. Build the method-specific cache in memory from the frozen logical prefix.
2. Decode `burn_in`; this token is not scored.
3. Use the resulting logits to score `targets[0]`.
4. Feed each true preceding target to predict the next target, advancing positions and cache state.
5. Apply the frozen scoring masks/document-boundary rule; divide by the actual scored-token count, not padded length.
6. Follow the approved trailing-decode convention and context budget.

BF16 uses exactly the same split, input IDs, positions and masks. Never score stale prefill logits, feed a target before scoring that same target, use free-running generation for teacher-forced PPL, or substitute an ordinary full-sequence forward for cache-sensitive inference.

Use stable FP32 log-softmax/cross-entropy calculations on copied/consumed output logits and adequate-precision NLL accumulation. This is evaluation arithmetic, not a change to model-forward precision. Preserve any exact scoring implementation already approved. Consume reusable logits before the next decode overwrites them.

Store per-anchor NLL sums/counts, required paired scoring data, source IDs/cluster IDs, method identity, execution scope, terminal status and checksums. Do not retain full-vocabulary logits for every scored token or serialize model KV caches. Stream results through the existing atomic writer. No materialized prefix catalog.

Reuse valid BF16 anchor results for all quantized comparisons under the same effective contract and execution identity. Compute the baseline once, not once per quantizer.

## 6. Run only focused new tests, then the actual Fast-PPL units

Focused CPU/tiny-control tests:
- First scored target uses the logits after burn-in; no target leakage or off-by-one shift.
- Scoring masks/counts and NLL-to-PPL conversion match the frozen definitions.
- Pairing uses exact dataset/anchor/scoring IDs and B=1 on both sides.
- Effective scope amendment leaves the original contract and failed batch results intact.
- Existing B=1 eligibility is reused without rerunning Q0.
- A resumed completed anchor/unit is not scored twice.
- B>1 quality qualification is rejected by this B=1 track.

No full suite or independent GPU smoke campaign. The first scheduled real anchor may serve as a plumbing check and remains its one accepted observation when valid; do not perform and discard a duplicate scored run merely to call it a smoke test.

Use the existing GPU lock and resumable atomic units. No profiler, batch diagnostics, admission, Full Scan or additional correctness matrix.

If a genuinely new quality wrapper defect is found, repair the smallest non-protected code, preserve the affected attempts and rerun only units affected by that defect under a new implementation version. Do not mix obsolete and corrected scores. A demonstrated protected-runtime correctness defect is not authorized for repair here: isolate the affected scope and stop its downstream eligibility.

## 7. Apply the original Fast-PPL statistics without tuning

Read exact aggregation weights, dataset/length rules, anchor/document clusters, bootstrap seed, 10,000 paired draws, confidence interval definition, margins and Fast-PPL status logic from the approved contract.

For matched observations report:

```text
delta_nll = nll_quantized - nll_bf16
relative_ppl_change = exp(delta_nll) - 1
```

Report per-dataset, per-length and approved aggregate statistics. Do not pool datasets or lengths under new weights to obtain a pass. Resample the approved paired anchor/document clusters, not highly correlated individual tokens. Preserve overlap/cluster identities.

Preserve inherited margins, including the global 1% relative-PPL margin, length-review 2%, length hard-fail 5%, and the full approved pass/fail/inconclusive logic. Do not subtract a measured BF16 cross-batch error as a quality allowance. Do not replace these thresholds with a looser margin, KL or token agreement.

Use pass/fail/inconclusive per quantized configuration. BF16 needs valid baseline coverage; its self-difference being zero is not a scientific quality-pass result. Missing paired baseline or insufficient output must remain missing/inconclusive or implementation-invalid as prescribed, never imputed as zero degradation.

Keep Fast-stage evidence separate from Full-PPL admission. A Fast pass grants eligibility for Q1B only; it does not qualify performance as quality-preserving. Do not automatically execute extra anchors on inconclusive results in this task unless the approved Fast-stage contract explicitly requires that exact extension; otherwise report the next eligible action.

## 8. Recover ordinary failures without restarting completed work

Use the approved bounded retry rule for genuine infrastructure, transient I/O or incomplete-output failures. Preserve original attempt IDs, replacement linkage and exactly one accepted score per logical unit. Never retry low scores, valid model outputs or a failed quality margin, and never retain the best score from multiple attempts.

One configuration's quality failure does not stop other eligible configurations' scheduled Fast PPL. CUDA corruption terminates the affected process. A baseline validity failure prevents comparisons that require that baseline but is not a license to invent it.

Upload, receipt, log or finalization failures do not require rerunning completed valid inference. Continue/recover the existing atomic units. Do not redownload historical campaigns or delete prefix catalogs.

## 9. Publish compact outputs and report the actual scope

Reuse existing output conventions. Produce:
- The new scope amendment, effective contract identity and operator authorization receipt.
- Per-configuration B=1 Q0 eligibility referencing existing units.
- Fast-PPL plan/input references and completed/incomplete attempt index.
- Paired NLL/PPL tables, confidence intervals and Fast-stage status.
- A concise Q1A report retaining original cross-batch failures.

Publish only new amendment/Fast-PPL evidence through the existing R2 tool; reference original roots. Finalize result bytes before computing hashes. Keep publication/approval receipts outside the immutable bundles they reference. Verify new objects once. No new tag, performance freeze, image hierarchy or nested receipt recursion.

Every new result must retain the effective contract, exact method/fingerprint mapping, B=1 scope and `quality_status` appropriate to this incomplete quality sequence. Do not mutate historical performance rows.

Future joins must include quality scope, not only method name or fingerprint. B=1 evidence cannot by itself authorize B=2/4/8/16 performance as quality-preserving. Even B=1 performance qualification must wait for the required Full PPL and LongBench stages and their dataset/length/mode coverage.

Return a concise report with:
- Actual starting/execution/final HEAD.
- Original contract identity and new amendment identity/hash; approval provenance.
- Original batch gate: FAILED, unchanged.
- B=1-scoped Q0 eligibility by configuration.
- Dataset/length/anchor/scored-token coverage.
- BF16 baseline validity.
- Per-configuration delta NLL, relative PPL change, paired CI and Fast-stage status.
- Incomplete units and preserved infrastructure replacements.
- Q1B eligibility, without starting Q1B.
- Protected runtime unchanged; no new performance execution.
- R2 root and receipt.

Stop after Q1A. Do not start Full PPL, LongBench, batch-invariant kernel work or the performance-quality join.
