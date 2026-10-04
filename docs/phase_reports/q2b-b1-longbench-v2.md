# Q2B physical-B=1 LongBench-v2 primary no-CoT

Execution **COMPLETE**; scientific gate **FAIL**; R2 publication and one clean retrieval **PASS**. Closed 2026-10-04 SGT. This is a completed negative quality result, not missing execution. The original cross-batch Q0 gate remains **FAILED**.

## Scope and identity

Only `bf16` and the single deduplicated `k4v4` finalist ran, at physical B=1 for both prefill and decode. The frozen primary set has 503 IDs: 321 eligible pairs, 182 predeclared length exclusions, and **642/642 completed outputs**. Missing/failed outputs, inference retries, and replacements are all **zero**. Each configuration used one successful worker attempt; all accepted outputs remain unchanged.

Starting HEAD: `df09bbf272dd9ef40611b57d1141a6739eb27d35`. Execution HEAD: `9c25c5ca2020bd447a20ee8a2b99458111bba73a`. The final documentation-closure HEAD is supplied in the operator handoff; it cannot be self-embedded in its own commit. Campaign: `q2b-20261002t094109291012z-9c25c5ca-2aa1f724`.

Approved contract: `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`, SHA-256 `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`. Accepted B=1 amendment: `quality-q1a-b1-20260919t135650246825z-a89ddc18`, SHA-256 `f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441`. Performance freeze: `perf-freeze-20260917-83536c37-r1`.

Quality image: `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`. Model/tokenizer: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`, BF16 model weights. Contract method fingerprints: BF16 `81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b`; k4v4 `97289ed9c875e27013ddcf7659fc6e849b3d438c58d0d86bd3dcac5d82eefb09`. Per-output growing-context/shape-specific runtime and cache-layout fingerprints are retained separately, not equated to the contract fingerprint.

Inputs are the frozen QP-1 LongBench-v2 primary no-CoT tokens at dataset revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. Eligible prompt lengths are 10,110–130,925 tokens; no under-8K sample is present. The 182 exclusions keep their original labels and do not enter the accuracy denominator. No retokenization, prompt truncation, CoT subset, extra finalist, native-prefill secondary run, or new sample was used.

The unchanged Q2A growing-context eager generator prefills all but the final 16 prompt tokens, decodes those 16, and generates from the final-conditioning logits. Generation is greedy, seed 20260722, with the unchanged eight-token budget and EOS rule. No answer mask or parser-based early stop was introduced. The exact pinned parser SHA-256 is `ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138`; invalid generated text remains an incorrect outcome.

## Paired results and unchanged gate

Accuracy is sample-weighted, not LongBench-E task macro. Positive drop means BF16 is higher; drops and their intervals are **percentage points**. CIs use the frozen paired bootstrap: 10,000 draws, seed 20260722, joint token-length-bucket × source-category strata and the existing percentile convention.

| Overall / category | Pairs | BF16 correct / accuracy | k4v4 correct / accuracy | Drop (pp) | Paired 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| Overall | 321 | 94 / 29.283% | 89 / 27.726% | 1.558 | [-0.312, 3.427] |
| Code Repository Understanding | 15 | 7 / 46.667% | 6 / 40.000% | 6.667 | [0.000, 20.000] |
| Long In-context Learning | 41 | 10 / 24.390% | 10 / 24.390% | 0.000 | [-9.756, 9.756] |
| Long Structured Data Understanding | 10 | 1 / 10.000% | 1 / 10.000% | 0.000 | [0.000, 0.000] |
| Long-dialogue History Understanding | 39 | 4 / 10.256% | 4 / 10.256% | 0.000 | [0.000, 0.000] |
| Multi-Document QA | 90 | 29 / 32.222% | 28 / 31.111% | 1.111 | [0.000, 3.333] |
| Single-Document QA | 126 | 43 / 34.127% | 40 / 31.746% | 2.381 | [-0.794, 6.349] |

| Frozen token-length bucket | Pairs | BF16 correct / accuracy | k4v4 correct / accuracy | Drop (pp) | Paired 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| 8-16k | 28 | 8 / 28.571% | 7 / 25.000% | 3.571 | [0.000, 10.714] |
| 16-32k | 90 | 34 / 37.778% | 33 / 36.667% | 1.111 | [-2.222, 4.444] |
| 32-64k | 72 | 23 / 31.944% | 21 / 29.167% | 2.778 | [0.000, 6.944] |
| 64-128k | 131 | 29 / 22.137% | 28 / 21.374% | 0.763 | [-2.290, 3.817] |

Overall accuracy CIs are BF16 **[24.611, 33.956]%** and k4v4 **[23.053, 32.399]%**. Contingency `n11/n10/n01/n00 = 86/8/3/224`: eight correct-to-wrong and three wrong-to-correct transitions. BF16-correct retention is **86/94 = 91.489%**, paired 95% CI **[85.577%, 96.629%]**. The frozen retention gate applies to its point estimate, not the interval lower bound. The discordant table is the prescribed McNemar diagnostic; no additional p-value gate was introduced.

| Criterion | Observed | Frozen rule | Result |
|---|---|---|---|
| Overall accuracy non-inferiority | Drop 1.5576 pp; CI [-0.3115, 3.4268] pp | CI upper bound ≤2 pp | INCONCLUSIVE by itself |
| Category guardrail | Code Repository Understanding: 6.6667 pp drop, N=15 | Every category drop ≤5 pp | FAIL; other five pass |
| Length guardrail | Maximum drop 3.5714 pp, 8–16k | Every bucket drop ≤5 pp | PASS |
| Invalid-output increase | 0.3115 pp | Increase ≤1 pp | PASS |
| BF16-correct retention | 91.4894%, denominator 94 | Point estimate ≥95% | FAIL |

The two hard failures determine the overall **FAIL**, despite the primary interval itself being inconclusive. No threshold, denominator, parser, or budget was changed. Invalid rates are BF16 **24/321 = 7.477%** and k4v4 **25/321 = 7.788%**; invalid answers remain in the 321-pair denominator. In the long-dialogue category each method has 20 invalid outputs out of 39; this limitation is retained, not repaired by favorable parsing or retries.

Both methods generated a mean/median of eight tokens, including EOS when it is the eighth token. Budget-exhausted/EOS counts are BF16 **253/68** and k4v4 **256/65**. All budget-limited outputs were scored. Full category/length accuracy and retention CIs are in `docs/evidence/q2b/longbench-v2.json`; three small-category retention CIs are null because at least one bootstrap draw has no BF16-correct sample, with the counts explicitly retained. The primary retention CI has zero undefined draws.

## Validation, preservation, and publication

Focused Q2B tests passed **9/9**; the synthetic CPU-only finalization/publisher-compatibility check passed. A read-only closure audit checked all 642 accepted result hashes, single-attempt pointers, conditioning/position accounting, generated IDs/text hashes, pinned parses, saved paired scores, and source identities. It independently checked the contingency counts and subgroup point estimates without rerunning inference or bootstrap. Already-retrieved controls match the local sealed bundle.

The pre-inference comparison passed for all **68 protected paths**, aggregate SHA-256 `f05c6986d3b6b9f0a6b986b385ecd8e2840e215e6eff94bed33e2d7833d70bc9`. Execution-source changes were only the new Q2B glue, focused tests, and short plan. Q2A generation, protected runtime/kernels, model precision, backend, calibration, cache layouts, contracts, fixtures, margins, and performance data were unchanged. No historical bulk validation, whole-repository suite, admission, image build, or performance campaign was repeated.

Local immutable root: `artifacts/q2b/q2b-20261002t094109291012z-9c25c5ca-2aa1f724`. Content root: `6191a72b0390b9119c2532604ba1021f53a7c81a279a2cafef62651cd4de8c4e`. The bundle contains **1,949 objects**, including all generation records and CPU-rescorable results; its internal report SHA-256 is `d22954f6952c7369dddf1fe6e7276c757baf502ac975a2d19c851627ef9edcf4`.

R2 URI: `r2://kvbench-artifacts/kvbench/sha256/6191a72b0390b9119c2532604ba1021f53a7c81a279a2cafef62651cd4de8c4e/`. Publication conditionally created 1,948 objects and verified one existing object; **COMPLETE last**. The single clean retrieval verified every object, inventory, ledger, COMPLETE, and root at **2026-10-04 03:11:22 SGT**. Private Bucket Lock: `kvbench-evidence-indefinite`, exact `kvbench/sha256/` prefix, indefinite retention. External receipt: `docs/evidence/q2b/r2-publication.json`. Host-side transfer retries did not rerun inference. No credentials, model weights, KV caches, or historical bundles were published.

The final closure changes only compact evidence/receipt, this report, and current status/task/risk records. Unrelated operator monitoring files and the operator prompt remain untracked and untouched. No push or tag was made.

Q1B Full-PPL and Q2A LongBench-E remain **PASS**; `tq_4bit_nc` remains Fast-PPL **INCONCLUSIVE** and the other seven compressed configurations remain **FAIL**. Q2B does **not** qualify k4v4 for Q2C. Stop for human review: no automatic extension, answer retry, Q2C, native-prefill secondary evaluation, KVQuant diagnosis, Q3, or Q4. No B=1 result transfers to B=2/4/8/16, and no quality/performance join or quality-preserving speedup is claimed.
