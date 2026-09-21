# Q1A B=1 Scope Amendment and Fast PPL

Status: **COMPLETE — one compressed configuration passes Fast PPL**

The operator-authorized amendment restricts this quality track to physical
model batch B=1. The original B=1 versus B={4,8} Q0 gate remains `FAILED`, its
tolerances and evidence remain unchanged, and no B=1 quality result transfers
to performance at B=2/4/8/16.

## Authority and coverage

- Starting HEAD: `a89ddc183744d87889c4e0fc25cb8b62889dadf1`.
- Execution HEAD: `6e0c78033ea7885bf71c9a5c8097839b1b71a100`.
- Original contract:
  `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`, SHA-256
  `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`.
- Amendment: `quality-q1a-b1-20260919t135650246825z-a89ddc18`, SHA-256
  `f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441`.
- Quality image:
  `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`.
- All ten configurations were B=1-scoped Q0 eligible from existing evidence;
  Q0 and its batch-invariance units were not rerun.
- WikiText-2 test and frozen C4 validation; L={4096,24576,32768,65536};
  16 anchors per dataset/length, 128 anchors and 16,384 scored tokens per
  configuration.
- 1,280/1,280 anchor units completed, totaling 163,840 scored tokens; no unit
  was incomplete or failed. BF16 baseline coverage is valid.
- Statistics use paired document-cluster bootstrap, 10,000 draws, seed
  20260722, and the unchanged 1% global relative-PPL margin.

## Fast PPL result

Values are the paired aggregate delta NLL and relative PPL change versus BF16;
the interval is the paired 95% delta-NLL interval.

| Configuration | Delta NLL | Relative PPL | 95% delta-NLL CI | Fast stage | Q1B eligible |
|---|---:|---:|---:|---|---|
| `bf16` | baseline | baseline | baseline | baseline valid | no self-comparison |
| `tq_4bit_nc` | 0.010289 | +1.034% | [0.007791, 0.012803] | INCONCLUSIVE | no |
| `tq_k3v4_nc` | 0.033913 | +3.449% | [0.028862, 0.039200] | FAIL | no |
| `tq_3bit_nc` | 0.038915 | +3.968% | [0.033723, 0.044242] | FAIL | no |
| `k4v4` | 0.001650 | +0.165% | [0.000520, 0.002769] | PASS | yes |
| `k2v4` | 0.036746 | +3.743% | [0.031583, 0.042133] | FAIL | no |
| `k2v2` | 0.043371 | +4.433% | [0.037622, 0.049425] | FAIL | no |
| `kvq4` | 3.397227 | +2888.113% | [3.297271, 3.500041] | FAIL | no |
| `kvq3` | 3.387448 | +2859.034% | [3.288357, 3.489320] | FAIL | no |
| `kvq2` | 3.411054 | +2929.715% | [3.319086, 3.506032] | FAIL | no |

The unusually large KVQuant degradations are present in the paired per-anchor
NLL evidence across both datasets and are retained as observed; no result was
retried or reweighted because of its value.

## Preservation and publication

Two pre-campaign wrapper attempts are preserved as non-claim staging evidence:
one Docker bind failure and one success-persistence defect. Neither contains an
accepted score and neither contributed to the final campaign. The corrected
campaign used exactly one accepted result per logical anchor. Protected runtime,
kernels, model precision, GEMM settings, calibration, Q0 evidence, and
performance data are unchanged. Full PPL and LongBench were not started.

Campaign: `q1a-20260919t141635000000z-6e0c7803-4c6a18f2`.
Root: `23d11321522bc9d997f2d26f9f6c4110e3332d0aedc47e3f53b8e1f7f547a5d0`.
R2 URI:
`r2://kvbench-artifacts/kvbench/sha256/23d11321522bc9d997f2d26f9f6c4110e3332d0aedc47e3f53b8e1f7f547a5d0/`.
All 2,603 objects were uploaded with `COMPLETE` last. One clean retrieval
validated the root, inventory, checksum ledger, marker, and private indefinite
Bucket Lock. No credential material was uploaded.

## Decision

Q1A is complete. Only `k4v4` is eligible for Q1B Full PPL. Q1B is not started;
it requires separate authorization. `tq_4bit_nc` remains inconclusive and the
other seven compressed configurations fail Fast PPL under the frozen margins.
