# Q2A — B=1 Cache-Sensitive LongBench-E for BF16 and k4v4

Execute Q2A only when the operator authorizes this task. Q1B is complete: k4v4 passed Full PPL and BF16 is its valid paired baseline. Implement only missing LongBench-E generation/scoring glue, run the frozen task suite, and report its actual quality verdict.

Do not repeat performance freezing, QP-1 preparation, Q0, batch invariance, Fast/Full PPL, admission, profiler collection, or performance experiments. Do not start LongBench v2, CoT stress, native-prefill secondary evaluation, KVQuant diagnosis, or the final quality/performance join. No additional preparation stage is required.

## 1. Use the completed handoff and the existing approved contract

Read only:
- AGENTS.md and the current quality task/status entry.
- `docs/phase_reports/q1b-full-ppl.md` and `docs/evidence/q1b/full-ppl.json`.
- The approved quality contract, B=1 amendment, approval records, and their LongBench-E generation, task, metric, statistical and recovery settings.
- The QP-1 LongBench-E input manifest: frozen sample IDs/order, tokenized prompts, length eligibility, generation budgets, task-specific formatting, stop rules and metric/parser source identities.
- The existing Q0/Q1A/Q1B growing-context quality session, LongBench suffix helper, generation utilities, statistical utilities and artifact publisher, as needed.

At entry run only `git status --short` and `git rev-parse HEAD`. The reported Q1B final HEAD is abbreviated as `d1b1b81d`; record the actual full SHA, not an invented expansion. A clean documented quality-only descendant is acceptable. Do not reset Git or discard unrelated work.

Known identities:

```text
Original contract ID:
quality-qp1-20260918t170812117127z-170b638c-e8f4a2
Original contract SHA-256:
b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1

Accepted B=1 amendment ID:
quality-q1a-b1-20260919t135650246825z-a89ddc18
Amendment SHA-256:
f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441

Corrected performance-freeze tag:
perf-freeze-20260917-83536c37-r1

Quality image:
sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32

Q1B campaign:
q1b-20260921t130810654798z-e5961237-fe8adff2
Q1B root:
725f28a2b5cfa00fc68642b5661755233f20c10b0c7272998abfa3200bf39dd8

QP-1 contract/input root:
6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62
```

Use the existing local inputs and accepted publication receipts. Fetch only a specifically missing input; do not retrieve entire historical campaigns or rematerialize the datasets. Record this task's actual authorization using the existing mechanism without changing the frozen contract/amendment bytes or creating another approval framework.

Use the existing protected-path identity comparison once before inference. Preserve the 68 locked paths, external kernels, model precision, inference backend, calibration, KIVI group/residual settings and method fingerprint mappings. Do not repeat broad admission, package audits, image builds, full repository tests or historical R2 verification.

## 2. Fix the eligible scope; do not reopen selection

Execute exactly:

```text
bf16   — paired baseline
k4v4   — sole Full-PPL-passing compressed configuration
physical model batch_size = 1
```

B=1 applies to prefill and decode, not merely to a data-loader setting. Preserve the contract's approved execution mode and its mapping to the performance implementation. Do not silently move the quality path to another backend, precision or Graph strategy.

The original cross-batch Q0 gate remains FAILED. Do not rerun it, change its tolerance, or promote B=2/4/8/16 based on B=1 evidence. Keep `tq_4bit_nc` Fast-PPL INCONCLUSIVE and the other seven compressed Fast-PPL failures unchanged and out of this task.

Full PPL passage is eligibility for Q2A, not proof of LongBench quality or a quality-preserving speedup. A Q2A result must retain its exact configuration, checkpoint, B=1, task/sample, length, execution-mode and protocol scope.

## 3. Reuse the frozen LongBench-E suite and inputs

Use the actual official E-split identities and evaluator pinned in QP-1, not a fresh download of the latest benchmark and not ordinary LongBench substituted for LongBench-E.

The frozen task set is:

```text
qasper
multifieldqa_en
hotpotqa
2wikimqa
gov_report
multi_news
trec
triviaqa
samsum
passage_count
passage_retrieval_en
lcc
repobench-p
```

The QP-1 report records 3,668 eligible sample IDs across these 13 tasks. Read the exact per-task counts and sample order from the frozen input manifest. Expected primary execution is 3,668 outputs per configuration: 3,668 paired samples and 7,336 configuration/sample outputs, not 7,336 independent pairs. Record any discrepancy in the source indexes instead of resampling or inventing missing records.

Do not change sample IDs/order, task coverage, prompt tokens, chat wrappers, output budgets, stop tokens, scoring references, metrics or parsers. Use the already tokenized effective prompts, including any official task-specific chat-template exceptions; do not apply a second chat template. Preserve official handling of multiple reference answers and task-specific output postprocessing.

Use the QP-1 eligibility/budget records for `prompt_tokens + generation_budget + margin <= model_limit`. A runtime OOM is an execution failure, not permission to reclassify a sample as originally excluded, shorten its prompt, lower its generation budget, or drop it only for k4v4. No task or sample may be removed based on a score or method behavior.

Create one short execution plan using the existing task mechanism, containing the two configurations, source manifest IDs, exact counts, frozen order, settings and recovery rule. This is an execution record, not a new quality contract or approval checkpoint.

## 4. Generate through the already validated cache-sensitive path

Reuse the growing-context session and LongBench suffix path established in Q0. Add only non-protected quality generation/scoring glue; do not copy the model forward or quantization implementation.

For each sample, use its frozen complete prompt IDs `p`:

```text
prefill_ids      = p[:-16]
conditioning_ids = p[-16:]

prefill(prefill_ids)
for token in conditioning_ids:
    logits = decode_one(token)
# These logits predict the first answer token after the complete prompt.
```

Concatenation must reproduce the frozen prompt exactly. Use the same split for BF16 and k4v4. Do not generate an answer from prefill logits, add dummy text, feed the final prompt token twice, or count the 16 conditioning tokens as generated answer tokens.

Generate autoregressively with the frozen deterministic policy (`do_sample=false` and the accepted generation settings). Choose the first answer token from the logits after the final conditioning token; consume/copy reusable output before it is overwritten. Feed generated tokens, not ground-truth reference answers, for subsequent answer generation. Teacher forcing belongs to PPL, not LongBench answer generation.

Advance positions and active cache length on every append. Allocate capacity from the actual prompt and frozen task generation budget before generation. Use the existing KIVI residual rollover and compressed-history behavior; do not substitute fixed-L scratch-slot overwrite. Terminate using the exact task-specific EOS/stop/max-new-token rules. Preserve stop reason, generated token count and budget exhaustion; do not force every answer to the maximum budget or invent an early-stop heuristic.

Build caches in memory from shared logical inputs. Do not create materialized KV-prefix catalogs, disk snapshots, or full-vocabulary logits for every generated token. Reuse Q0 cache-read/invariance evidence and the existing lightweight assertions; no new Nsight, sanitizer or batch-invariance campaign is required for ordinary quality-only generation glue.

## 5. Execute once per configuration/sample and preserve outputs

Use the existing runner/atomic writer and one accepted output per configuration/sample. Valid formal Q2A results may be reused on resume only when sample, input, generation settings, method and effective contract identities match. PPL scores and short Q0 diagnostics are not a substitute for this LongBench baseline.

Compute BF16 once for each sample. Store compact records sufficient for rescoring:
- task and sample ID; effective contract/amendment and input-manifest identity;
- method/configuration, source/image identities, physical B and execution mode;
- prompt token hash and length, conditioning split, requested generation budget;
- generated token IDs and text, output hash, parsed/postprocessed text;
- stop reason, generated length, task-specific invalid-output flag/reason;
- metric/parser identity, score, reference-ID linkage, terminal status and attempt ID.

Keep benchmark answers in the scoring side only; do not use them to select generation, stop decoding or choose an attempt. Incorrect-but-valid output is still valid output and must be scored. An empty answer or budget-limited answer must receive the already specified scoring/invalid-output treatment, not an automatic retry. A missing output from a process failure is not the same as a scored wrong answer.

Use the existing GPU lock. Do not run concurrent performance or diagnostic workloads on the GPU. Do not turn generation durations into new performance claims.

## 6. Score with the frozen evaluator and paired statistics

Use the exact pinned official task metrics, parser/normalization rules, per-task aggregation and E length-bucket convention bound by the approved contract. Do not quietly substitute sample averaging for bucket averaging, or vice versa. Keep raw metric scales explicit and convert to the approved common score-point scale before cross-task macro aggregation.

Report each of the 13 tasks, the fixed task categories, the frozen length buckets, invalid-output rates and generation-length/budget-exhaustion summaries. Primary aggregation is the frozen equally weighted task macro, not a sample-micro average:

```text
Q_bf16 = (1/13) * sum(task_score_bf16[t])
Q_k4v4 = (1/13) * sum(task_score_k4v4[t])
D = Q_bf16 - Q_k4v4
```

Use the approved task-stratified paired bootstrap: resample paired sample IDs within each task, apply the same draws to BF16/k4v4, calculate each task score using the frozen rule, then calculate the task macro. Retain any additional stratification required by the effective contract. Use its 10,000 draws, seed and interval convention; do not invent new bootstrap or multiple-testing rules.

Preserve the full effective Q2A decision logic. The inherited primary margins are:

```text
macro score-drop margin:             2.0 score points
category review threshold:          3.0 score points
category hard-fail threshold:        5.0 score points
invalid-output increase limit:       1.0 percentage point
```

These are score points/percentage points, not relative percent changes. Use the contract's actual category memberships, invalid-output definitions, denominators and equality rules.

For complete valid evidence, apply the frozen logic: primary PASS requires the upper 95% paired-CI bound for D to be within the 2.0-point margin and all hard guardrails to pass; a lower bound above the margin or a hard guardrail breach gives FAIL; a CI crossing the margin or inadequate evidence yields INCONCLUSIVE under the specified rules. Do not interpret a nonsignificant difference as demonstrated non-inferiority.

Do not treat all task-specific differences as new independent pass gates unless the approved contract says so. Report task/length effects even when the macro passes. Define invalid outputs only as the contract does; an ordinary incorrect answer is not automatically a format-invalid answer.

Do not silently compute the primary result on a changing complete-pair subset. Missing outputs, unusable parser results or missing baseline partners remain explicit. Recover eligible infrastructure failures; otherwise report incomplete coverage and the applicable non-PASS status rather than imputing zero degradation or dropping a task from the 13-task denominator.

## 7. Recover interruptions without rerunning valid inference

Reuse the approved bounded retry policy (one retry for the already authorized infrastructure/transient-I/O/incomplete-output classes, where applicable). Retain every attempt and explicit replacement link. Never retry a low score, valid wrong answer, early valid EOS, quality failure or inconvenient task result. Never choose the best attempt.

A log, snapshot query, report write or R2 outage does not require rerunning valid generated outputs. Resume incomplete sample units; do not restart the suite. Recompute CPU scores from saved outputs when only metric/export code failed.

A demonstrated new quality-wrapper defect may be minimally repaired outside protected paths; preserve affected attempts and rerun only outputs shown to be invalidated by that defect. Do not mix old and corrected generation semantics without explicit versioning. A protected runtime/kernel defect is outside this task: isolate its affected scope, stop the affected process, and report it without changing kernels or automatically invalidating unrelated historical evidence. True CUDA corruption terminates that process before additional outputs are trusted.

## 8. Focused tests only, followed by real scheduled samples

Run only tests for newly added or affected code:
- Q1B eligibility and B=1 scope, with original batch FAIL preserved;
- exact prompt/suffix concatenation and first-answer-token alignment;
- correct autoregressive append, output-buffer consumption, task stopping and length accounting;
- pinned task metric/parser behavior, multi-reference handling and score units;
- paired task-macro bootstrap and frozen gate boundaries;
- resume/replacement indexing and no duplicated accepted output.

Reuse existing tests/results when the implementation is unchanged. No full repository suite, package-lock audit, new image, fresh Q0, admission or GPU smoke campaign. The first scheduled real sample may be used as the plumbing check and retained as its single valid observation; do not run and discard duplicate benchmark generations merely to call them smoke tests.

## 9. Publish a compact result, then stop

Reuse current output paths and writers. Produce generated-output records, input/attempt index, paired per-sample scores, per-task/category/length summaries, task-macro CI, invalid-output and length summaries, gate JSON, a short report and existing checksum/inventory/COMPLETE structure. Do not build a new dataset registry, benchmark framework or nested receipt hierarchy.

Publish only the new Q2A evidence with existing host-side R2 tooling. Keep old roots as references. Finalize bytes before calculating digests, upload COMPLETE last, verify the new bundle once, and keep its receipt outside the bundle it names. Failed publication does not invalidate inference. Never publish credentials, `.env`, weights, KV snapshots or duplicated historical campaigns.

Report execution completeness independently from the scientific gate. A completed experiment with FAIL or INCONCLUSIVE is a finished result, not a reason to relax a margin or rerun until PASS. Incomplete required evidence remains incomplete.

If k4v4 passes, report Q2A PASS and its eligibility for the separately authorized Q2B LongBench-v2 finalist stage under the existing rule. Do not start Q2B, select extra methods, run CoT/native-prefill tests, execute the final quality-performance join or change frozen performance rows to quality-qualified. Preserve the original cross-batch FAIL; no B=1 conclusion transfers to B=2/4/8/16.

Return a concise Q2A REPORT with:
- actual starting/execution/final HEAD, contract/amendment, image and configuration scope;
- planned/completed/missing/failed configuration-sample outputs and accepted pair count;
- per-task BF16/k4v4 scores and paired differences for all 13 tasks;
- primary task-macro scores, D, paired 95% CI and gate verdict;
- category/length summaries, review flags, hard-guardrail results and denominators;
- invalid-output, generated-length, stop/budget-exhaustion summaries;
- original attempts/retries/replacements and any corrected wrapper scope;
- protected-path preservation, focused tests and new R2 root/receipt;
- Q2B eligibility, B=1-only limitation and confirmation later stages were not started.
