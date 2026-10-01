# Q2A physical-B=1 LongBench-E

Status: execution **COMPLETE**; frozen scientific gate **PASS**; durable R2 publication and one clean retrieval **PASS**. Closed 2026-10-02 SGT. The original cross-batch Q0 gate remains **FAILED**.

The approved contract is `quality-qp1-20260918t170812117127z-170b638c-e8f4a2` (SHA-256 `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`), with B=1 amendment `quality-q1a-b1-20260919t135650246825z-a89ddc18` (SHA-256 `f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441`). Starting HEAD was `d1b1b81dadd1a596bd109a38d3e43bb42992c0cc`; inference used HEAD `47151cce5704be0a67fcbe1a884cab1da6afd316` and Quality image `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`. The checkpoint was `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`. Only `bf16` and `k4v4` ran at physical B=1 through the frozen growing-context eager path.

All 3,668 frozen sample IDs completed once per configuration: 7,336/7,336 accepted outputs, 3,668 complete pairs, zero missing/failed outputs, zero inference retries or replacements. The original coordinator exited while the k4v4 Docker worker continued; a host-only observer retained the GPU lock and recorded its exit code 0. The existing worker and all accepted outputs were left unchanged. A missing worker-level exit receipt was appended after checking the observer journal and all accepted outputs. The campaign has one worker attempt per configuration.

Scores are official LongBench-E task scores in points; each task equally averages its three frozen E length buckets. These buckets use the official dataset source-length field (`<4000`, `4000–7999`, and `>=8000`), which differs from the effective model prompt-token count. Positive difference means BF16 scored higher.

| Task | Pairs | BF16 | k4v4 | BF16 − k4v4 |
|---|---:|---:|---:|---:|
| qasper | 224 | 42.217 | 43.133 | -0.917 |
| multifieldqa_en | 150 | 55.300 | 55.847 | -0.547 |
| hotpotqa | 300 | 60.887 | 60.230 | 0.657 |
| 2wikimqa | 300 | 47.070 | 47.137 | -0.067 |
| gov_report | 300 | 34.357 | 34.340 | 0.017 |
| multi_news | 294 | 25.727 | 25.617 | 0.110 |
| trec | 300 | 71.333 | 70.667 | 0.667 |
| triviaqa | 300 | 92.047 | 92.123 | -0.077 |
| samsum | 300 | 42.687 | 43.133 | -0.447 |
| passage_count | 300 | 16.957 | 15.407 | 1.550 |
| passage_retrieval_en | 300 | 99.667 | 99.667 | 0.000 |
| lcc | 300 | 66.443 | 66.303 | 0.140 |
| repobench-p | 300 | 52.263 | 51.633 | 0.630 |

The equally weighted 13-task macro is BF16 **54.3810**, k4v4 **54.2490**: drop **0.1321** points, task-stratified paired 95% CI **[-0.2908, 0.5449]** from the frozen 10,000 draws and seed 20260722. The upper bound is below the unchanged 2-point margin. All six category hard guardrails pass; no category crosses the 3-point review threshold or 5-point hard-fail threshold. Category drops (single-document QA, multi-document QA, summarization, few-shot, synthetic, code) are respectively -0.732, 0.295, -0.107, 0.295, 0.775, and 0.385 points, with 374/600/894/600/600/600 paired samples. Length-bucket drops for 0–4k (1,267 pairs), 4–8k (1,270), and 8k+ (1,131) are 0.158, 0.223, and 0.015 points. The full per-task bucket scores and category/length scores are in `longbench_e_summary.json`.

Under the frozen scorer's top-level invalid-output definition, `parsed.strip() == ""` after the official first-line handling for trec/triviaqa/samsum, BF16 had 0/3,668 invalid outputs and k4v4 had 0/3,668 (increase 0.0 percentage points; unchanged hard limit 1.0). The code metric additionally filters comment/fence lines internally: four BF16 and four k4v4 outputs have empty or whitespace-only metric inputs after that filtering, although their top-level parsed text is nonempty. This caveat does not change the recorded scores, invalid-output counts, or gate. Mean/median generated tokens: BF16 109.411/42 and k4v4 109.332/42. Stop reasons (budget, EOS, samsum newline): BF16 1,802/1,824/42; k4v4 1,806/1,816/46. Budget-limited answers were scored rather than retried. No task was omitted or reweighted based on score.

The official evaluator source SHA-256 is `e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323`. CPU scoring used the QP-1 locked direct dependency versions and verified artifact hashes; the `rouge` transitive `six` dependency was 1.16.0. The focused Q2A tests passed 9/9. All 68 protected runtime paths still match the lock (aggregate SHA-256 `f05c6986d3b6b9f0a6b986b385ecd8e2840e215e6eff94bed33e2d7833d70bc9`); no protected wrapper, KIVI prefill, kernel, contract, prompt, budget, metric, margin, or performance data changed.

The immutable campaign is `q2a-20260927t030705670149z-47151cce-7f4a2b`, locally under `artifacts/q2a/`. Its 22,028 objects validate at root `9b7c4f5a3631afb634a0d9fd50232c3aba3418a518bc86ebeedd1ccbabdeedfa`; its internal report SHA-256 is `fa39f73c4c4b0194d96ad8fd3ba7f663ed4296ab0e9f9f65e9d57c1f583328c2`. Host-side publication verified 6,579 already present objects and conditionally created 15,449 missing objects, with COMPLETE last. One clean retrieval verified every object, inventory, checksum ledger, COMPLETE, and root at 2026-10-01 23:05:15 SGT. The private Bucket Lock is `kvbench-evidence-indefinite`, exact prefix `kvbench/sha256/`, indefinite retention. URI: `r2://kvbench-artifacts/kvbench/sha256/9b7c4f5a3631afb634a0d9fd50232c3aba3418a518bc86ebeedd1ccbabdeedfa/`. The external receipt is `docs/evidence/q2a/r2-publication.json`. Host-only bounded transport retry checks passed 9/9; publication recovery did not rerun inference or alter finalized evidence.

Repository closure changes only this report, compact evidence/receipt, and current status/task/risk records. The actual final closure HEAD is supplied in the operator handoff (it cannot be self-embedded in its own commit). Unrelated operator-created watch/probe files remain untracked and untouched; no push or tag was made.

The scientific Q2A gate PASS makes k4v4 eligible for a separately authorized Q2B at physical B=1 only. This does not waive the original cross-batch FAIL, qualify B=2/4/8/16 performance, or constitute a quality-preserving performance claim. Q2B, LongBench-v2, CoT stress, native-prefill secondary evaluation, KVQuant diagnosis, and the quality-performance join were not started.
