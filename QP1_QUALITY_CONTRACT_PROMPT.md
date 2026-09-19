# QP-1 — Complete the Quality Contract and Materialize Evaluation Inputs

Execute QP-1 preparation only. Finish the six technical pending items from QP-0: dependency lock, chat-template hash, dataset revisions, sample IDs, prompt hashes, and fixture materialization. Deliver one complete, versioned contract for human approval.

Do not start Q0, compute any BF16 or quantized quality scores, run PPL/LongBench, rerun performance, refit models, or repeat admission. Quality execution remains LOCKED. Approval of this preparation task is not approval of the final quality contract.

## 1. Read the current contract; do not reopen performance freezing

Read only:
- AGENTS.md and the current status/task entry.
- CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md: precedence, QP-1, dataset/prompt preparation, cache-sensitive execution, quality gates/statistics, and recovery rules.
- CODEX_QUALITY_EVALUATION_ADDENDUM.md only where the accepted protocol refers to it; do not restore superseded requirements.
- docs/phase_reports/qp0-tag-correction-closure.md.
- docs/evidence/qp0/freeze-tag-correction.json and its publication receipt.
- The finalized QP-0 manifest, locked-path list, image records, and current configs/quality/quality_contract.yaml.
- Existing quality data-preparation code, dependency locks, sample manifests, prompt files, and tokenizer snapshot metadata.

At entry run only `git status --short` and `git rev-parse HEAD`. Use the accepted local freeze marker and correction receipt. Do not rerun full tests, package audits, performance inventory assembly, reproduction, GPU checks, historical R2 retrieval, or admission.

Known handoff identities:

```text
QP-0 closure HEAD:
170b638cc0213ca67238681de5bb0c78f950b1d7

performance source commit:
83536c37433875cda98c36e2848e05692e9407d0

corrected freeze tag:
perf-freeze-20260917-83536c37-r1

corrected tag object:
773b12c32ae312bfae9b782d772364f6bbbc008e

final freeze manifest SHA-256:
91db28a33940e9bbdda6c723a2678ae9459e73f813b20bfecb6c97222ba38fc5

original freeze root:
9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f

correction root:
44c9594079e372d708d509449589a428f4b27c0392e2f1b1e491acc3d17e2559
```

Use the corrected tag binding. Preserve the original tag, freeze bundle, correction bundle, and PERFORMANCE_DATA_FROZEN. Do not retag or recreate the freeze. A documentation-only descendant of the reported HEAD is acceptable; do not reset the repository.

## 2. Bind the actual measured configurations

Populate the contract from the performance inventory, not from remembered defaults:

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

Preserve the exact checkpoint/tokenizer revision, model limit, RoPE, weight dtype, K/V bit widths, residual/sink/outlier policies, skipped layers, quantizers, cache layouts, backend, and source/kernel identities.

All ten configurations remain scheduled for Q0 and Fast PPL under the existing eligibility rules. Do not use latency, Phase 17 predictions, or performance rankings to select them. Do not add held-out configurations.

Keep shape/mode-specific fingerprint mappings where they exist. Quality has growing-context evaluation, while performance used fixed-L timing; record that distinction and the required Q0 equivalence checks rather than inventing a universal fingerprint or claiming equivalence has already been tested.

## 3. Complete the dependency lock without changing the measured runtime

Read the complete Measurement and QP-0 Quality image digests from their records. Reuse the existing Quality derivative whenever its packages suffice.

Lock only the CPU dataset/tokenizer, official metric/parser, and statistical dependencies actually needed by the accepted quality protocol. Record exact versions and available artifact hashes, plus the pinned benchmark source commits.

Preserve the measured CUDA, PyTorch, Triton, model-forward implementation, and method extensions. A dependency resolver must not silently upgrade them. Use metadata/import checks that do not initialize CUDA; do not load model weights.

If additional evaluation dependencies are necessary, one thin, explicitly locked Quality-image successor or the existing supported CPU preprocessing environment is allowed in this task. Record the final image and runtime dependency delta. Do not mutate the Measurement image or rebuild method kernels.

An actual dependency conflict is a specific unfinished item, not a reason to repeat performance freezing. Do not create an environment-management framework.

## 4. Freeze the tokenizer and effective prompts

Load only the tokenizer/configuration assets from the frozen model revision. Do not download model weights merely to tokenize text.

Record the raw chat-template hash and the effective formatting policy for each benchmark/task. Preserve any official task-specific chat-wrapper exceptions. Where the pinned benchmark already applies chat formatting, do not wrap it a second time. Freeze BOS/EOS handling, generation-prompt behavior, special-token insertion, and relevant tokenizer versions.

If the performance experiment did not use chat prompts, record that fact. A template resolved now from its frozen tokenizer is a QP-1 evaluation-input choice, not a newly discovered historical performance input.

Pin official benchmark prompt/config/parser/metric files to exact commits. For selected examples save the rendered prompt or its reproducible inputs, prompt-text hash, final token-ID hash and length, generation budget, and stop/parser settings.

Preserve the cache-sensitive LongBench split:

```text
original prompt IDs = prefill IDs + final 16 conditioning IDs
answer generation starts only after all conditioning IDs have been decoded
```

Use the same tokens and split for BF16 and every quantized method. CPU tests can prove concatenation identity now; proving actual compressed-cache reads belongs to Q0, not QP-1. Do not insert dummy text or change prompts to fit this split.

## 5. Resolve dataset revisions and sample selection once

Use the datasets named by the accepted protocol:
- WikiText-2 raw test split.
- A fixed C4 validation subset.
- The official LongBench-E 13-task suite.
- LongBench v2, with its primary no-CoT and separate finalist/CoT policies.

Preserve already pinned revisions. For genuinely unpinned sources, resolve an official repository revision once and record the exact commit, split/configuration, file identities, and content hashes. `main`, `latest`, or `PINNED` is not a finalized identity. Do not replace a benchmark or split silently.

Use existing accepted sample-selection rules and seeds. If a mechanical detail has never been specified, choose one deterministic, bounded implementation and document it in a short `resolved_choices` section for the forthcoming approval. Do not pretend newly resolved choices were made before performance results were known. Preserve the actual earlier protocol history.

No model outputs, quality scores, performance rankings, or answer correctness may influence sample selection. Every configuration must receive the same evaluation IDs and order for an applicable stage.

For C4, download/stream only the declared validation shards and selected subset needed by the frozen anchor rule. Record the sampling frame, stable document IDs, shard order, selection seed, and stopping rule before sampling. Do not download the full corpus or describe a bounded-shard sample as a uniform sample of all C4. Never read training text into the quality test stream.

For LongBench-E, retain these exact tasks unless an accepted repository amendment already states otherwise:

```text
qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report, multi_news,
trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p
```

Use the actual official E split/configuration naming in the pinned release. Do not substitute ordinary LongBench for LongBench-E. Preserve the protocol's suite coverage; an access problem must be explicit, not silently resolved by dropping a task.

For LongBench v2, freeze sample eligibility before choosing future method finalists. Finalists are selected later by the existing quality/allocated-ratio rule, not speed. Resolve any still-unspecified finalist tie rule, comparison geometry for context-dependent r_alloc, and CoT budget/subset rule now as proposed contract choices, without looking at quality results.

## 6. Materialize compact, method-independent fixtures

Create safe-format selected text/token inputs and manifests only. Do not create GPU KV caches, per-method prefix catalogs, logits, or model outputs. Share each logical input across configurations. Use one token stream plus anchor offsets rather than copying a long prefix for every anchor.

### PPL

Carry forward the accepted parameters. The saved protocol specifies:

```text
Fast:
  prefix lengths = [4096, 24576, 32768, 65536]
  anchors per length = 16
  scored horizon = 128

Full:
  prefix lengths = [4096, 16384, 24576, 28672, 32768, 65536, 98304, 130560]
  anchors per length = 64
  scored horizon = 256

burn-in decode tokens = 1
```

Apply the repository's final accepted version if amended; make any difference visible in the review summary. Do not silently reduce these budgets or move lengths to fit a newly estimated knee.

For each dataset/stage/length, record selected anchor IDs, source document IDs, stream offsets, scoring positions/masks, and actual scored-token counts. Keep the protocol's document-boundary EOS and 32-token ignored-start policy. Specify how a scoring horizon crossing a boundary is handled; do not silently count ignored tokens as scored tokens.

Record anchor/document overlap and cluster IDs for the prescribed paired bootstrap. Do not impose an invented rule that all long prefixes must be disjoint, and do not later treat overlapping tokens as independent samples.

The future runner must implement the frozen alignment:

```text
prefix = ids[:L]
burn_in = ids[L]
targets = ids[L+1 : L+1+H]
```

The burn-in token itself is not scored; the logits after decoding it score the first target. Reserve context for the accepted burn-in/scoring sequence and any documented trailing decode. Use the protocol's length budget and safety margin. Do not assign PPL to an ordinary full-sequence forward.

### LongBench

Tokenize the effective complete prompt once with the frozen tokenizer and formatting policy. Record `N`, task generation budget `G`, safety margin `M`, eligibility under `N + G + M <= model_max_length`, length bucket, and every exclusion reason.

Use the protocol's method-independent feasible-set rule; do not shorten or select prompts per method. Keep model-length eligibility distinct from an observed GPU OOM. Do not run GPU feasibility probes in QP-1 or label a CPU budget calculation as measured allocation evidence.

The saved LongBench-v2 primary contract uses no-CoT, max_new_tokens=8, safety margin=128, 16 conditioning tokens, and A/B/C/D parsing. Keep no-CoT, CoT stress, and native-prefill secondary protocols separate, with their own exact budgets and eligibility tables. Record boundary inclusion rules for length buckets so a boundary sample is not counted twice.

Materialize all selected inputs needed to make the contract approval-ready, including the selection pool for later finalists. Process excluded long records in a bounded streaming pass instead of retaining every unselected corpus file. Download/tokenization interruptions should resume completed file/sample units, not regenerate everything.

## 7. Preserve the actual quality margins and statistical rules

Populate the accepted margin file and contract from the authoritative protocol. Do not change thresholds because the performance study is finished or a method looks fast.

The saved protocol's main margins are:
- PPL: paired NLL comparisons; global relative-PPL margin 1%, length review 2%, length hard fail 5%; report NLL, delta NLL and exp(delta NLL)-1.
- LongBench-E: 2.0 score-point macro-drop margin, category review 3.0 points, category hard fail 5.0 points, invalid-output increase limit 1 percentage point; official metrics and task-macro aggregation.
- LongBench v2: 2 percentage-point accuracy-drop margin, length/category hard limits 5 percentage points, invalid-output increase limit 1 percentage point, BF16-correct retention point estimate at least 95% with its CI reported.

Carry forward the full pass/fail/inconclusive logic, not just these numbers. Distinguish a relative percentage change from an absolute percentage-point change and a benchmark score point. Freeze internal score scales and conversions.

Use the existing paired 95% CI procedures and anchor/document or task-stratified sample resampling, not independent-token bootstrap. Record exact aggregation weights, bootstrap seed/count, insufficient-sample handling, and any predeclared inconclusive-extension budget. If these mechanical choices are unresolved, propose deterministic values for review now; do not adapt them after scores appear.

Implementation invalidity and algorithmic quality loss remain distinct. Do not claim that nonsignificance establishes equivalent quality. Do not automatically relabel performance data based on a quality result before the prescribed fingerprint-based join.

## 8. Write one complete contract and one human-readable summary

Use the repository's current paths/schema. Expected outputs include:

```text
configs/quality/quality_contract.yaml
configs/quality/datasets/...
configs/quality/fixtures/...
configs/quality/prompts/...
configs/quality/gates/quality_margins.yaml
artifacts/quality_contract/<contract_id>/contract_snapshot.yaml
artifacts/quality_contract/<contract_id>/input_manifest.json
artifacts/quality_contract/<contract_id>/selected_inputs/...
docs/phase_reports/qp1-quality-contract.md
```

Create only needed files; do not build a generic contract engine or a new gate framework. Keep all protected performance paths unchanged. One targeted diff against the corrected freeze tag at completion is sufficient; do not rehash historical performance archives.

The review summary must show:
- Corrected freeze tag, performance manifest SHA, freeze root, and correction receipt.
- Exact ten configurations and evaluation mode/equivalence requirements.
- Quality image and dependency lock, with unchanged measured runtime.
- Tokenizer/chat-template and benchmark prompt/parser identities.
- Dataset revisions, selected counts by stage/task/length, exclusions and reason counts.
- PPL alignment, LongBench split, context/output budgets, and total planned scoring work (token/sample counts, not invented runtime estimates).
- Quality margins, CI/aggregation rules, stage eligibility, and finalist rules.
- Which settings were inherited versus resolved now, with chronology.
- Final contract ID, SHA-256, and the only remaining action: human approval of that exact version.

Use an acyclic publication order: finalize fixtures and their identities, write the final contract referencing them, hash its final bytes, finalize its bundle, then write publication/approval receipts outside the bundle they identify. Do not mutate the contract after recording its hash. Later approval references its digest in a separate record; it is not a self-referential field to rewrite inside the finalized contract.

## 9. Test and publish only the new preparation work

Run focused CPU checks only:
- Selected IDs/token hashes reconstruct and match the pinned sources.
- PPL prefix/burn-in/target alignment and document-boundary masks.
- LongBench token split concatenates exactly to the effective prompt.
- Length/generation budgets and exclusion accounting.
- All ten configurations, corrected freeze binding, units/margins, and pending approval state.
- Dependency imports without model loading, CUDA execution, or replacing the frozen runtime.
- Resume of one interrupted preparation unit without rewriting completed inputs.

Do not run a full repository suite, performance freeze, admission, reproduction, Q0 or a benchmark smoke that produces scores.

Publish the new compact contract/fixture bundle using the existing R2 tool; retain historical roots as references. Verify new objects once. Reuse successfully materialized inputs and retry only failed I/O/publication operations. Never rerun data preparation merely because a receipt write failed.

Use existing host-side credentials without printing or including them in artifacts. Do not upload .env, model weights, caches, calibration tensors, or historical campaign copies. Do not delete prefix catalogs. Do not push Git branches or tags.

## 10. Stop at one approval point

QP-1 preparation is PASS when the six technical items are complete, the contract and selected fixtures are reproducible, focused CPU checks pass, and the new bundle is published. Human approval can remain pending without making preparation PARTIAL.

Report PARTIAL only for an actually unfinished technical item, such as an inaccessible source, unresolved dependency conflict, or missing required input. Preserve completed preparation rather than restarting unrelated work. Do not label unresolved technical fields as approval-ready.

Final state:

```text
QP-0: PASS, unchanged
PERFORMANCE_DATA_FROZEN: present, unchanged
QP-1 preparation: PASS
quality_contract: requires_human_approval
Quality execution: LOCKED
Q0 / PPL / LongBench: NOT RUN
```

Return a concise QP-1 REPORT with actual starting/final HEAD, the six technical-item statuses, dataset/task/length counts, dependency/image identity, contract ID/hash, R2 root, resolved-choice summary, and pending human approval. Do not grant approval, start Q0, implement GPU evaluation runners, or add another preparation phase automatically.
