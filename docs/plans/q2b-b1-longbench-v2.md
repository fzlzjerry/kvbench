# Q2B — physical-B=1 LongBench-v2 primary no-CoT

Operator-authorized Q2B only; starting HEAD
`df09bbf272dd9ef40611b57d1141a6739eb27d35`. Reuse the approved QP-1 contract,
accepted B=1 amendment and Q2A PASS receipt. Execute exactly BF16 and the single
deduplicated Full-PPL/LongBench-E finalist k4v4, in frozen primary sample order.
Preserve unrelated untracked monitoring files.

503 frozen primary IDs comprise 321 eligible pairs (642 generated outputs) and
182 predeclared length exclusions. Read the existing tokenized primary inputs;
no CoT subset, retokenization, extra chat wrapping, truncation or reselection.
The same Q2A `generate_sample` performs full prompt = prefill + 16 conditioning
tokens, then greedy generation with EOS or eight-token budget; no letter masks,
parser-driven stopping or reference-answer conditioning. The exact Quality image
and protected runtime paths remain unchanged. The first real sample is the
plumbing check and is retained. No separate GPU smoke or admission is run.

New code is Q2B plan/parser/scoring/record glue only. Compile only the exact
`extract_answer` function from QP-1-pinned `pred.py`, after its file-hash check;
strip output as its caller does. Invalid generated answers remain incorrect
outcomes; missing executions are separate. Reuse Q2A generation and immutable
unit writer. Repair only missing completion pointers from fully saved validated
outputs, never regenerate them for report/upload failures. At most one bounded
infrastructure/incomplete-unit retry; never retry a valid wrong/invalid answer.

Overall aggregation is sample accuracy, not LongBench-E macro. Paired bootstrap:
10,000 draws, seed 20260722, sample jointly within frozen token-length bucket ×
source domain, fixed stratum counts. Use the existing quality order-statistic
percentile endpoints. Report overall/category/length CIs and contingency table;
retention is n11/(n11+n10), not an alternate accuracy denominator. The contract
specifies no extra zero-denominator resampling rule: report undefined retention
and a null interval if any such draw occurs, with its count; do not fabricate or
redraw. The prescribed McNemar diagnostic is its discordant table, not a substitute
gate or added p-value requirement.

Unchanged gate: drop CI95 upper bound ≤2 pp; category/length point drops ≤5 pp;
invalid increase ≤1 pp; retention point estimate ≥0.95 with CI reported (no
additional lower-CI threshold). A lower primary CI above margin or hard breach
fails; crossing the margin is inconclusive. Preserve observed denominators and
frozen bucket labels, including any under-8K cases. Scientific verdict and
execution completeness are separate.

Save authorization, plan, 503-row eligibility index, 642 compact generation
records/attempts, paired scores, summaries, gate, report and control checksums
under one fresh `artifacts/q2b/<campaign_id>/`. No materialized KV snapshots or
full logits. Finalize COMPLETE last, publish via the existing host-side R2 client,
then verify one clean retrieval and store its receipt outside the sealed bundle.
No repeated historical validation or duplicated historical publication.

Run only focused Q2B tests and one targeted protected-path comparison before
inference. Original batch gate stays FAILED; physical B=1 evidence does not
qualify B>1. Report Q2C readiness only. Q2C, native-prefill secondary evaluation,
KVQuant diagnosis, Q3 and Q4 remain unstarted.
