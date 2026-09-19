# Q0 Batch Diagnosis Report

Status: **COMPLETE — diagnosis demonstrated; original batch gate remains FAILED**

## Result

The Q0 batch failure is genuine batch-shape numerical sensitivity in the
frozen BF16 full-model path, not a row-mapping, input, cache-lifecycle, output-
ownership, padding, or quantizer defect. For three deterministic BF16 samples,
two independent B=1 reconstructions are exact. B=4 and B=8 duplicate rows are
exact within each batch, and mixed-companion outputs remain exact after row
permutation and inverse mapping. Nevertheless, all six aligned B=1-versus-
batch comparisons fail the unchanged reference-directed 0.02/0.02 logit
predicate.

Embedding and first-layer inputs are exact. The first non-exact operation is
layer-0 `q_proj` for sample 2 and layer-0 `v_proj` for samples 0 and 1, before
cache attention consumes the prefix. Differences then grow through the common
BF16 model path to maximum logit errors of 0.12890625–0.1875. This demonstrates
shape-dependent BF16 projection arithmetic; it does not establish a cache or
method-kernel correctness defect.

## Comparator and evidence

The approved predicate is
`abs(z_batch-z_b1) <= atol + rtol*abs(z_b1)`. The quality comparator had used
`torch.allclose(left, right)`, whose relative term is scaled by `right`. The
small non-protected repair implements the approved B=1-reference formula
explicitly. It does not explain or remove the failures: all ten affected
batch-invariance units were rerun with fresh IDs and still fail, with the same
maximum errors and selected-token counts as the original evidence. The other
100 valid Q0 units were reused unchanged.

| Sample | B | max / median / P95 absolute error | Violating logits | top-1 | First divergence |
|---:|---:|---|---:|---|---|
| 0 | 4 | 0.1875 / 0.03125 / 0.078125 | 22,184 / 128,256 | same | `layer00_v_proj` |
| 0 | 8 | 0.1875 / 0.03125 / 0.0703125 | 19,684 / 128,256 | same | `layer00_v_proj` |
| 1 | 4 | 0.12890625 / 0.01953125 / 0.0625 | 14,298 / 128,256 | same | `layer00_v_proj` |
| 1 | 8 | 0.1640625 / 0.0234375 / 0.0625 | 18,265 / 128,256 | same | `layer00_v_proj` |
| 2 | 4 | 0.1875 / 0.046875 / 0.09375 | 50,334 / 128,256 | same | `layer00_q_proj` |
| 2 | 8 | 0.15625 / 0.015625 / 0.0625 | 14,720 / 128,256 | same | `layer00_q_proj` |

Normalized maximum discrepancies are 5.80–7.43 times the allowed elementwise
bound. Directed KL(B=1 || batch) is 3.24e-5–8.38e-4. All three diagnostic
top-1 tokens agree at both B=4 and B=8, but top-1 agreement is not the gate.
In the original 100-sample BF16 unit it is 97/100 at B=4 and 95/100 at B=8.
Teacher-forced step, generated-sequence, and parsed-answer agreement are not
part of this one-step equal-length batch stage and are not conflated with it.

## Scope and disposition

- Starting HEAD: `5dd0142a317e4d7046ef84b9daf1ac531ea91870`.
- Bounded BF16 diagnosis HEAD: `5c14189b30fbb99eeb58fc53642ff8e2a932e727`.
- Ten-unit execution HEAD: `e0783369864a24e95a4bbd6cae2c173532f0155d`.
- Changed implementation: one quality comparator line plus a standalone
  quality-only diagnosis/continuation helper and focused tests.
- Protected adapters, CUDA, model precision, caches, frozen inputs, contract,
  tolerances, 68 locked hot paths, and performance evidence: unchanged.
- Original valid units reused: 100/100. Original failed batch units remain
  immutable. New affected units: 10/10 completed, 0 PASS, 10 FAIL.
- Fast-PPL eligibility: 0/10 (`bf16`, all TurboQuant, KIVI, and KVQuant
  configurations remain ineligible). Fast/Full PPL and LongBench scoring were
  not started.

Compact evidence root:
`eedee1596bb36692da8557fc001c45593f50f930e62c27227a7151ab6aa88ba6`.
R2 URI:
`r2://kvbench-artifacts/kvbench/sha256/eedee1596bb36692da8557fc001c45593f50f930e62c27227a7151ab6aa88ba6/`.
All 62 objects were uploaded COMPLETE-last and passed one clean retrieval
under the private indefinite Bucket Lock.

The frozen performance study used the same shared BF16 projection path, so
batch-shape arithmetic dependence is relevant to numerical outputs at B>1.
This diagnosis does not show a performance-timing defect and does not alter or
invalidate performance samples; the quality consequence remains unresolved by
the approved gate. Any B=1-only quality amendment or new metric/tolerance is a
separate human decision. No waiver is made here.

Next action: retain Q0 and all ten batch gates as failed. If the operator wants
quality work to continue, separately decide whether to investigate a protected
batch-invariant model/runtime path or amend the quality contract. Do not start
PPL under the current contract.
