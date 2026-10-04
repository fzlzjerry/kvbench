# Q3/Q4 Joint Admission and Scoped Results

Execution **COMPLETE**, CPU-only; publication and clean retrieval **PASS**.
Scientific outcome: **zero fully qualified compressed configurations**, not an
experimental PASS.

- Starting HEAD: `21251cd56a94cbed46925ef84d3ecdc44896d797`.
- Execution HEAD: `f406314c0f48085fb7f338c53b7627d6e8e32407`.
- Final closure HEAD is supplied in the operator handoff; it cannot be embedded
  in its own commit.
- Bundle: `artifacts/joint_results/q3-q4-20261004t034803222550z-f406314c-cb983f`.
- Root: `2d609efb39c5e50d617d3c03affa22795076415364249a51bf4b1087e64c7704`.

## Scope and stage outcomes

The unchanged contract is `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`,
SHA-256 `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`.
The accepted B=1 amendment is `quality-q1a-b1-20260919t135650246825z-a89ddc18`,
SHA-256 `f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441`.
Freeze tag `perf-freeze-20260917-83536c37-r1` is unchanged. The protocol preceded
performance; exact manifest binding followed known performance results, and
the B=1 amendment followed Q0 batch failures, before Fast-PPL scores.

All ten retain accepted B=1 Q0 eligibility; the original cross-batch gate stays
**FAILED**. `NR` below means not run due to stage eligibility, not failed execution.

| Configuration | Fast PPL | Full PPL | LongBench-E | v2 | Joint B=1 outcome |
|---|---|---|---|---|---|
| bf16 | baseline | baseline | baseline | baseline | reference only |
| tq_4bit_nc | INCONCLUSIVE | NR | NR | NR | quality_inconclusive |
| tq_k3v4_nc | FAIL | NR | NR | NR | quality_fail, Q1A |
| tq_3bit_nc | FAIL | NR | NR | NR | quality_fail, Q1A |
| k4v4 | PASS | PASS | PASS | FAIL | quality_fail, Q2B |
| k2v4 | FAIL | NR | NR | NR | quality_fail, Q1A |
| k2v2 | FAIL | NR | NR | NR | quality_fail, Q1A |
| kvq4 | FAIL | NR | NR | NR | quality_fail, Q1A |
| kvq3 | FAIL | NR | NR | NR | quality_fail, Q1A |
| kvq2 | FAIL | NR | NR | NR | quality_fail, Q1A |

Nine compressed candidates: **8 FAIL, 1 INCONCLUSIVE, 0 full qualifiers**.
The qualified table has a valid empty schema; best configuration and speedup
are null, not 0x/1x. BF16 is not a compressed winner. Q2C is
`not_run_no_eligible_finalist`. Native-prefill remains
`not_run_conditional_finalists_only` under the accepted contract, not a passed
test. No downstream inference is authorized or started.

## Positive findings and the Q2B failure

k4v4 retains Full-PPL PASS (WikiText-2 +0.173736%, C4 +0.181803% relative PPL)
and LongBench-E PASS (task-macro drop 0.1321 points, CI [-0.2908, 0.5449]).
Q2B completed all 642 outputs/321 pairs, with 182 frozen exclusions. Accuracy
is BF16 94/321 = 29.2835%, k4v4 89/321 = 27.7259%; drop 1.5576 pp,
CI [-0.3115, 3.4268]. The overall non-inferiority result stays **INCONCLUSIVE**;
the combined gate stays **FAIL**:

- Code Repository Understanding: N=15, 7/15 versus 6/15, drop 6.6667 pp,
  CI [0,20] pp, exceeding the 5 pp hard guard.
- BF16-correct retention: 86/94 = 91.4894%, CI [85.577%,96.629%], below the
  unchanged 95% point-estimate guard. No retention-CI gate was introduced.
- Contingency n11/n10/n01/n00 = 86/8/3/224. Eleven saved discordant pairs
  retain IDs, category/length, answers, invalid flags, token counts and stops.
- Invalid outputs 24/321 versus 25/321; increase 0.311526 pp passes. Length
  guards pass. All category/length intervals remain in the compact inputs.
- B=1, no-CoT, 16 conditioning tokens and eight generated tokens maximum
  remain fixed. Budget stops 253/321 versus 256/321 are descriptive, not an
  established cause of failure. Small subgroup sizes and undefined subgroup
  retention intervals remain disclosed.

## Exact-identity performance sidecar

The left join retains **2,670 slots = 2,205 accepted + 465 infeasible**, with
38 replacement links and no extra statistical weight. There are **0 unmatched
identities and 0 duplicate slots**. Contract fingerprints, frozen adapter and
kernel identities, model/tokenizer, container and accepted run/shape keys
agree; stored protected-path receipts are reused without historical revalidation.

The 600 B=1 slots receive configuration metadata only; B=1 is not a universal
length/domain/growing-to-fixed qualification. All 2,070 B>1 slots are
`outside_evaluated_quality_scope`, not failed quality measurements. Q0's stored
eager/Graph relationship is explicitly limited to its exact fixed-L shape.
Historical `quality_status=unvalidated` is untouched. All new timing rows keep
`r_hbm=null`; Phase15 features remain at their single profiler point.

The 357 existing measured same-work ratios are retained, with 27 separate
capacity-amplification points. No null is filled by a predictor or generation
duration. No quality-preserving optimum/headline is supported. A quality
failure alone does not invalidate frozen timing. KVQuant's severe Fast-PPL
anomaly remains unresolved, not a diagnosed upstream-algorithm defect.
Phase17's missed targets, selected-outer-score limitation, BF16 edge
extrapolation failure and 27 non-identifiable knees remain unchanged.

## Reproduction and custody

The bundle contains the stage registry, scoped Parquet join, empty qualified
table, discordant appendix, summary, `QUALITY_VALIDATION_REPORT.md`, and 53
compact source files with identity/ledger references. It is 2,251,831 bytes and
70 objects. Twelve focused tests pass; fresh-process CPU reproduction matches
all seven products byte-for-byte. All 53 selected sources remain unchanged.

```sh
cd artifacts/joint_results/q3-q4-20261004t034803222550z-f406314c-cb983f
/home/rockrock/cmu_paper/.phase17-venv/bin/python reproduce.py \
  --reproduce . --output /new/nonexistent/q3-q4-results
```

Python 3.12.3 and pyarrow 25.0.0 were used. The command also runs on another
CPU-only machine with pyarrow, using the compact bundle alone. No models,
network, inference, rescoring, bootstrap, fitting or historical raw data are
needed. The output directory must not already exist.

Q1A/Q1B/Q2A/Q2B source roots respectively:

- `23d11321522bc9d997f2d26f9f6c4110e3332d0aedc47e3f53b8e1f7f547a5d0`
- `725f28a2b5cfa00fc68642b5661755233f20c10b0c7272998abfa3200bf39dd8`
- `9b7c4f5a3631afb634a0d9fd50232c3aba3418a518bc86ebeedd1ccbabdeedfa`
- `6191a72b0390b9119c2532604ba1021f53a7c81a279a2cafef62651cd4de8c4e`

Performance host-wall root:
`5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e`.
These are referenced, not bulk copied or remotely reverified. New-bundle URI:
`r2://kvbench-artifacts/kvbench/sha256/2d609efb39c5e50d617d3c03affa22795076415364249a51bf4b1087e64c7704/`.
All 70 objects were published with COMPLETE last and verified in one clean
retrieval at `2026-10-04T03:53:56.733405Z`. Inventory, checksum ledger and root
passed; no unexpected objects. Private R2 Bucket Lock
`kvbench-evidence-indefinite` covers `kvbench/sha256/` indefinitely. The receipt
is `docs/evidence/q3-q4/r2-publication.json`, outside the sealed bundle.
No protected source, historical evidence, contract,
threshold, model, or performance data changed. Unrelated monitoring files and
operator prompts remain untracked and untouched; no push, tag or deletion.

Stop for human review. Additional studies require separate authorization.
