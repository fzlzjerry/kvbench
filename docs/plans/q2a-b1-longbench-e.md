# Q2A physical-B=1 LongBench-E execution

Authority: approved QP-1 contract `quality-qp1-20260918t170812117127z-170b638c-e8f4a2` (`b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`), accepted B=1 amendment `quality-q1a-b1-20260919t135650246825z-a89ddc18` (`f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441`), and Q1B PASS root `725f28a2b5cfa00fc68642b5661755233f20c10b0c7272998abfa3200bf39dd8`. Operator authorization is the Q2A request; its receipt is written into the new campaign. The original Q0 cross-batch gate remains FAILED.

Only `bf16`, then `k4v4`, execute in the exact Quality image `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`, with physical batch 1 and the unchanged eager growing-cache path. Within each configuration, use the frozen task order below and each input index's sample order. The QP-1 input manifest, token hashes, official metric sources, budgets, and 16-token conditioning split are verified before execution.

| Task | Eligible samples |
|---|---:|
| qasper | 224 |
| multifieldqa_en | 150 |
| hotpotqa | 300 |
| 2wikimqa | 300 |
| gov_report | 300 |
| multi_news | 294 |
| trec | 300 |
| triviaqa | 300 |
| samsum | 300 |
| passage_count | 300 |
| passage_retrieval_en | 300 |
| lcc | 300 |
| repobench-p | 300 |

Total: 3,668 paired samples and 7,336 configuration/sample outputs. Prefix construction and 16-token conditioning precede greedy answer generation; no benchmark answer enters generation. The frozen budgets are task-specific (32, 64, 128, or 512 tokens); EOS, the official samsum newline stop, and budget exhaustion are recorded. The Quality image runs one GPU worker at a time under the existing GPU lock. Every accepted output is atomic and immutable. Resume skips accepted outputs; only incomplete or classified infrastructure attempts may have one linked replacement. Low scores and valid early stops are never retried.

The pinned official metric is scored on the frozen E buckets (`0-4k`, `4-8k`, `8k+`); task score is the equal mean of those three official bucket scores, and the primary macro equally weights 13 tasks. The frozen paired task-stratified bootstrap uses 10,000 draws, seed 20260722. Primary score-drop margin is 2 points; category review/hard thresholds are 3/5 points, and invalid-output increase hard limit is 1 percentage point. Scores, per-task/category/bucket summaries, output validity, lengths, and stop reasons are generated from immutable outputs. A completed FAIL or INCONCLUSIVE is retained as the actual scientific result.

The new Q2A bundle is finalized append-only, published with COMPLETE last, and checked through one clean R2 retrieval. Q2B, CoT, native-prefill evaluation, and the quality-performance join are deferred.
