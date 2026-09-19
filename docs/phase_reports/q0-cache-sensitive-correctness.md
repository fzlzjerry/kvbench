# Q0 Cache-Sensitive Correctness

Status: **PARTIAL — execution complete; all ten configurations fail frozen batch invariance**

The approved Q0 suite completed 110/110 selected diagnostic units. Every
configuration passes all seven growing-context core probes, the 16-token
LongBench suffix diagnostic, exact eager/Graph invariance, finite-output
checks, and cache-dependence where applicable. Every configuration fails the
unchanged B=1 versus B={4,8} logit-tolerance check, so no configuration is
eligible for Fast PPL. No PPL or LongBench benchmark score was computed.

## Authority and execution

- Starting HEAD: `825c14db15590da29ce19351f563829646eec581`.
- Primary execution HEAD: `b85111f1325a66d580bcdeeae940b37041114a9c`.
- KVQuant continuation HEADs: `4d8de2124c219f6fb19e7a00cdadd29135a3907e`
  and `d120657d24c7b6dcfa3c3a30b0b1bc632527f8a4`.
- Finalization HEAD: `5b143f16f7f248a52d4c729d705adc331faa2f55`.
- Contract: `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`,
  SHA-256 `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`.
- Approval receipt SHA-256:
  `7b6cf83bf422be49f1c3d1474031b7c57aefd5911f23bb66aac1342bb8aed324`.
- Performance freeze tag: `perf-freeze-20260917-83536c37-r1`.
- Quality image:
  `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`.
- Locked hot paths: 68/68 unchanged; aggregate SHA-256
  `f05c6986d3b6b9f0a6b986b385ecd8e2840e215e6eff94bed33e2d7833d70bc9`.

## Per-configuration result

| Configuration | Q0 units | Failed stage | Q0 | Fast PPL eligible |
|---|---:|---|---|---|
| `bf16` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `tq_4bit_nc` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `tq_k3v4_nc` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `tq_3bit_nc` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `k4v4` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `k2v4` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `k2v2` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `kvq4` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `kvq3` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |
| `kvq2` | 10 PASS / 1 FAIL | batch invariance | FAIL | no |

The B=4/B=8 maximum absolute logit errors respectively were: BF16
0.34375/0.375; TurboQuant 4-bit 2.34375/0.9140625, k3v4
1.5625/1.75, 3-bit 1.734375/2.0625; KIVI k4v4 0.65625/0.625,
k2v4 2.46875/1.5234375, k2v2 3.03125/1.34375; KVQuant q4
1.1875/1.533203125, q3 1.4296875/1.7939453125, q2
2.078125/2.0546875. The frozen tolerance was not changed. Selected-token
agreement ranged from 89/100 to 97/100. Unequal-length padding is not claimed
because the frozen endpoint exposes no approved per-row mask contract.

## Other Q0 controls

- Core probes at L=512, 4096, 16384, 24576, 32768, 65536, and 130560:
  70/70 PASS with finite outputs and advancing growing-context state.
- Eager/Graph: 10/10 PASS, 100/100 selected tokens per configuration and zero
  observed maximum absolute error.
- Cache dependence: 9/9 compressed configurations PASS repeat/restoration,
  active encoded-storage perturbation, finite changed output, and fail-closed
  decode interception. BF16 is correctly recorded not applicable.
- LongBench: 10/10 suffix diagnostics PASS after all 16 frozen conditioning
  tokens. This is not LongBench scoring.
- Four-token greedy diagnostic: no invalid tokens. First BF16 divergence was
  at token 2 for `tq_3bit_nc`, `k2v4`, and `k2v2`; this lossy-method
  difference is recorded and is not itself a Q0 hard failure.
- No protected adapter/CUDA/cache/config file changed. No new CUDA or masking
  kernel path was introduced, so historical admitted sanitizer evidence was
  retained and no broad sanitizer rerun was performed.

## Preservation and publication

The original 33 KVQuant loader-routing failures and four intermediate
double-commit failures remain append-only and are excluded only through
explicit replacement references. No valid BF16, TurboQuant, or KIVI unit was
rerun. A first root finalization attempt omitted nested unit `COMPLETE` files
from its ledger and remains preserved; the closure copied the same result
bytes without GPU reexecution and corrected only finalization controls.

Final campaign:
`q0-20260919t054207209666z-b85111f1-27c21357-closure-5b143f1`.
Root:
`2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0`.
R2 URI:
`r2://kvbench-artifacts/kvbench/sha256/2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0/`.
All 356 objects were uploaded with `COMPLETE` last and one clean retrieval
validated the inventory, checksum ledger, marker, root, and indefinite private
Bucket Lock. No credential material was uploaded.

## Decision

Q0 preparation and execution are complete, but all ten configurations are
ineligible for Fast PPL because batch invariance is a frozen Q0 hard gate.
Fast PPL, Full PPL, LongBench benchmark scoring, and performance reruns remain
not started. The next action is a separately authorized diagnosis of the
cross-batch numerical inconsistency; this Q0 task does not modify protected
implementations or relax the approved contract.
