# Q1B B=1 Full PPL

Status: **execution COMPLETE; k4v4 Full PPL PASS**. BF16 is a valid paired
baseline. The original cross-batch Q0 gate remains **FAILED**.

## Authority and coverage

- Starting HEAD: `4d4b269c45fd4c792535a04792591171b81eb32b`;
  execution HEAD: `e59612371077ebc350e26eec537eb9cb5201cb18`.
- Original contract: `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`,
  SHA-256 `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`.
- Accepted physical-B=1 amendment:
  `quality-q1a-b1-20260919t135650246825z-a89ddc18`, SHA-256
  `f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441`.
- Quality image:
  `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`.
- Only `bf16` and `k4v4` ran. WikiText-2 test and frozen C4 validation each
  contributed 64 anchors at each of eight prefix lengths. The 256-token
  scoring horizon and one-token burn-in yielded 1,024/1,024 completed anchors
  and 262,144 scored tokens per configuration: 2,048 anchors and 524,288
  configuration-token scores total. There were no failed or missing anchors,
  retries, or replacements; each configuration has only `attempt-00`.
- Paired document-cluster bootstrap: 10,000 draws, seed 20260722; the frozen
  1% dataset margin, 2% length review threshold, and 5% length hard-fail
  threshold were unchanged.

## Dataset results

NLL is mean negative log likelihood per scored token. Intervals are paired
95% confidence intervals for `k4v4 − bf16` delta NLL; relative PPL is
`exp(delta NLL) − 1`.

| Dataset | Clusters | BF16 NLL / PPL | k4v4 NLL / PPL | Delta NLL [95% CI] | Relative PPL |
|---|---:|---:|---:|---:|---:|
| WikiText-2 test | 124 | 1.920072 / 6.821448 | 1.921808 / 6.833300 | 0.001736 [0.001228, 0.002242] | +0.173736% |
| C4 validation | 304 | 2.351262 / 10.498813 | 2.353079 / 10.517900 | 0.001816 [0.001145, 0.002571] | +0.181803% |

Both dataset upper bounds are below the frozen `log(1.01)` margin. The
descriptive pooled estimate is delta NLL 0.001776 [0.001356, 0.002223] and
relative PPL +0.177769%; it does not replace the two dataset gates.

## Length results

Each row has 64 paired anchors and 16,384 scored tokens. Cluster counts and
full-precision values are retained in the immutable `full_ppl_summary.json`.
No length crossed either frozen review or hard-fail threshold.

| Dataset | Prefix L | BF16 NLL / PPL | k4v4 NLL / PPL | Delta NLL [95% CI] | Relative PPL |
|---|---:|---:|---:|---:|---:|
| WikiText-2 | 4096 | 1.882419 / 6.569374 | 1.884262 / 6.581498 | 0.001844 [0.000902, 0.002785] | +0.184557% |
| WikiText-2 | 16384 | 1.878813 / 6.545728 | 1.880527 / 6.556961 | 0.001715 [0.000636, 0.002856] | +0.171621% |
| WikiText-2 | 24576 | 1.917283 / 6.802451 | 1.918738 / 6.812354 | 0.001455 [0.000349, 0.002553] | +0.145580% |
| WikiText-2 | 28672 | 1.937726 / 6.942941 | 1.938264 / 6.946680 | 0.000538 [-0.000889, 0.001969] | +0.053849% |
| WikiText-2 | 32768 | 1.933504 / 6.913693 | 1.935369 / 6.926598 | 0.001865 [0.000590, 0.003102] | +0.186670% |
| WikiText-2 | 65536 | 1.888989 / 6.612682 | 1.890923 / 6.625479 | 0.001933 [0.000498, 0.003442] | +0.193531% |
| WikiText-2 | 98304 | 1.918292 / 6.809320 | 1.921082 / 6.828344 | 0.002790 [0.001729, 0.003875] | +0.279370% |
| WikiText-2 | 130560 | 2.003549 / 7.415329 | 2.005296 / 7.428295 | 0.001747 [0.000587, 0.002921] | +0.174847% |
| C4 | 4096 | 2.324732 / 10.223937 | 2.325247 / 10.229204 | 0.000515 [-0.000902, 0.002220] | +0.051519% |
| C4 | 16384 | 2.404524 / 11.073155 | 2.406163 / 11.091321 | 0.001639 [-0.000066, 0.003424] | +0.164057% |
| C4 | 24576 | 2.316103 / 10.136092 | 2.318793 / 10.163398 | 0.002690 [0.001495, 0.003841] | +0.269387% |
| C4 | 28672 | 2.314648 / 10.121358 | 2.318165 / 10.157015 | 0.003517 [0.000293, 0.007866] | +0.352286% |
| C4 | 32768 | 2.330579 / 10.283890 | 2.332696 / 10.305687 | 0.002117 [0.000810, 0.003574] | +0.211952% |
| C4 | 65536 | 2.409364 / 11.126886 | 2.410975 / 11.144826 | 0.001611 [0.000311, 0.002960] | +0.161233% |
| C4 | 98304 | 2.259144 / 9.574893 | 2.260914 / 9.591851 | 0.001769 [0.000530, 0.003071] | +0.177104% |
| C4 | 130560 | 2.451004 / 11.599989 | 2.451676 / 11.607787 | 0.000672 [-0.000312, 0.001680] | +0.067223% |

## Decision and custody

`k4v4` passes Full PPL for physical B=1 and becomes eligible for a separately
authorized Q2A LongBench-E evaluation with its paired BF16 baseline. Q2A was
not started. `tq_4bit_nc` retains its Fast-PPL `INCONCLUSIVE` status; the seven
Fast-failed compressed configurations retain their failures. All eight have
Q1B status `not_run_due_to_fast_stage_status`. The original cross-batch Q0
gate remains FAILED, and this B=1 result does not qualify B=2/4/8/16 or
establish a quality-preserving performance claim.

Fast and Full PPL share zero exact Full-compatible observations; two anchors
share a prefix but have only the shorter Fast horizon. No Fast scores were
used as Full scores. Protected runtime paths, calibration, fixtures,
performance evidence, and the approved contract/amendment bytes were
unchanged. No LongBench, KVQuant diagnosis, or quality-performance join ran.

The finalized campaign is
`q1b-20260921t130810654798z-e5961237-fe8adff2`; its report SHA-256 is
`21a1074dd0f9892b3cf99ed4ecf9cf4552715b2e9cb6ea85907905f7b6806e7c`.
The 4,115-object root
`725f28a2b5cfa00fc68642b5661755233f20c10b0c7272998abfa3200bf39dd8`
was published at
`r2://kvbench-artifacts/kvbench/sha256/725f28a2b5cfa00fc68642b5661755233f20c10b0c7272998abfa3200bf39dd8/`
with `COMPLETE` last. Clean retrieval validated all objects, inventory,
checksum ledger, root, and the private indefinite Bucket Lock. The compact
evidence and receipt are `docs/evidence/q1b/full-ppl.json` and
`docs/evidence/q1b/r2-publication.json`.
