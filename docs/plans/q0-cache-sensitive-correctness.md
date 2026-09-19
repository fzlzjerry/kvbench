# Q0 cache-sensitive correctness plan

Authority is the separately approved contract
`quality-qp1-20260918t170812117127z-170b638c-e8f4a2` with SHA-256
`b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`
and performance freeze tag `perf-freeze-20260917-83536c37-r1`.  The frozen
contract is read-only.  This plan authorizes Q0 diagnostics only.

## Frozen scope

- Configurations: `bf16`, `tq_4bit_nc`, `tq_k3v4_nc`, `tq_3bit_nc`,
  `k4v4`, `k2v4`, `k2v2`, `kvq4`, `kvq3`, and `kvq2`.
- Quality image:
  `sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32`.
- Core historical-prefix lengths: 512, 4096, 16384, 24576, 32768, 65536,
  and 130560.  The last value is the contract's largest Full-PPL prefix and
  leaves 512 positions below the 131072-token model limit.
- Each core probe uses one unscored burn-in and four teacher-forced targets.
  The first eligible frozen WikiText-2 Full-PPL anchor at each available
  length is selected before inference.  L=512 is the first 512 tokens of the
  first L=4096 anchor.  No model output influences selection.
- The LongBench control is the first eligible frozen Qasper record.  Its last
  16 prompt tokens are decoded in order after prefill; four greedy diagnostic
  tokens follow.  No benchmark score is computed.
- Eager/Graph invariance uses 100 deterministic fixed-L logical samples at
  B=1/L=512.  The Graph member uses the existing fixed-shape capture strategy;
  it is not represented as growing-context Graph execution.
- Batch invariance uses the same 100 deterministic equal-length logical
  samples at B=1,4,8 and L=512.  The final B=8 carrier repeats four already
  measured samples, and only the 100 unique sample IDs are aggregated.
  Unequal-length padding/masking is not claimed because the frozen endpoint
  exposes no approved per-row mask contract.
- Growing-context correctness is exercised by every core probe and the
  16-token LongBench suffix using the existing append/decode path.
- Compressed-cache dependence is tested once per compressed configuration at
  B=1/L=512 using an active packed payload perturbation plus an intercepting
  negative control.  BF16 records this stage as not applicable.

## Units and retry

The append-only unit is `configuration x stage`: seven core-length stages,
one LongBench suffix stage, one eager/Graph stage, one batch-invariance stage,
and one cache-dependence stage.  This is 110 terminal units.  A completed unit
is never rerun.  Only infrastructure, transient-I/O, or incomplete-output
failure may receive one replacement attempt with the original attempt
preserved.  Valid unexpected numerical or quantization behavior is not
retried.

The maximum diagnostic continuation is 20 tokens (16 conditioning plus four
greedy tokens).  Full PPL, Fast PPL, LongBench scoring, performance timing,
and performance reruns are excluded.

## Decision rules and outputs

Hard failures are limited to the approved protocol conditions: non-finite
output, illegal token/cache state, frozen-reference or same-configuration
invariance failure under existing tolerances, missing compressed-cache
dependence, forbidden fallback, or identity drift.  Quantized/BF16 logit or
greedy-token divergence is diagnostic and does not by itself fail Q0.

Each unit stores compact checksums, top-k/logit diagnostics, path and identity
evidence, and its terminal decision.  Existing method-admission evidence is
referenced for unchanged low-level numerical and sanitizer coverage.  The
campaign is finalized with inventory, checksum ledger, and `COMPLETE`, then
published content-addressed to R2 with `COMPLETE` last and one clean retrieval.
Fast-PPL eligibility is reported but Fast PPL is not started.
