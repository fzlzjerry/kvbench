# Q1B — B=1 Full Cache-Sensitive PPL for BF16 and KIVI k4v4

Execute Q1B only when the operator authorizes this task. Continue the approved quality protocol and its B=1 scope amendment. This is the next quality experiment, not another preparation, admission, or performance campaign.

Run the paired BF16 baseline and `k4v4` only. Do not promote Fast-inconclusive or Fast-failed configurations. Do not run LongBench, add anchors, change quality margins, investigate or repair KVQuant in this task, or modify the measured implementation.

## 1. Use the existing handoff

Read only:
- AGENTS.md and the current quality task/status.
- The approved contract's Full-PPL fixtures, alignment, masks, execution mode, aggregation, gates, bootstrap, and retry rules.
- The accepted B=1 scope amendment and its approval receipt.
- The Q1A result/eligibility index and the existing Fast-PPL runner/scorer.
- QP-1's materialized Full-PPL token streams, anchor IDs, masks, and cluster IDs.
- The current quality image/dependency record and the existing artifact/publisher helpers.

At entry record only `git status --short` and `git rev-parse HEAD`. Record the actual current full SHA; the Q1A execution SHA below is not necessarily its final reporting HEAD. Do not reset Git.

Use local fixtures and accepted receipts. Retrieve only an individually missing required input. Do not rerun QP-0/QP-1, Q0, cross-batch invariance, Fast PPL, historical admission, package audits, Full Scan, model fitting, image builds, or historical bulk R2 verification. Do not run the full repository test suite.

Known handoff:

```text
Original contract ID:
quality-qp1-20260918t170812117127z-170b638c-e8f4a2

Original contract SHA-256:
b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1

B=1 amendment ID:
quality-q1a-b1-20260919t135650246825z-a89ddc18

Amendment SHA-256:
f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441

Performance freeze tag:
perf-freeze-20260917-83536c37-r1

Quality image:
sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32

Q1A execution SHA:
6e0c78033ea7885bf71c9a5c8097839b1b71a100

Q1A campaign:
q1a-20260919t141635000000z-6e0c7803-4c6a18f2

Q1A evidence root:
23d11321522bc9d997f2d26f9f6c4110e3332d0aedc47e3f53b8e1f7f547a5d0
```

Preserve original contract/amendment bytes and approvals. No new contract version, freeze tag, waiver, or approval framework is needed for this already specified stage. Record the operator's Q1B execution authorization with the existing mechanism.

## 2. Carry forward the actual eligibility

Use the existing per-configuration Fast-stage verdicts:

```text
bf16: valid paired baseline; required for Full PPL, not a self-comparison candidate
k4v4: PASS; the only compressed Q1B candidate

tq_4bit_nc: INCONCLUSIVE; excluded from this Q1B task

tq_k3v4_nc, tq_3bit_nc, k2v4, k2v2, kvq4, kvq3, kvq2:
FAIL; excluded from this Q1B task
```

No speed-based selection. Preserve all nine compressed configurations in the overall quality-status table, including failed and inconclusive results. The excluded configurations have Q1B status `not_run_due_to_fast_stage_status`, not new Q1B failures.

The original cross-batch Q0 gate remains FAILED. Both prefill and decode use physical model batch B=1. Existing successful B=1 Q0 checks are reused, not rerun. Do not transfer future B=1 qualification to B=2/4/8/16.

## 3. Run exactly the frozen Full-PPL workload

Load the actual QP-1 Full-stage fixture manifest. Expected settings are:

```text
Configurations: bf16, k4v4
Datasets: WikiText-2 raw test, frozen C4 validation subset
Physical model batch: 1
Prefix lengths: [4096, 16384, 24576, 28672, 32768, 65536, 98304, 130560]
Anchors per dataset/length/configuration: 64
Scored horizon: 256
Burn-in decode tokens: 1
```

Expected logical coverage, if the finalized masks retain all planned targets:

```text
Per configuration:
  2 datasets × 8 lengths × 64 anchors = 1,024 anchor evaluations
  1,024 × 256 = 262,144 scored tokens

Paired experiment:
  1,024 matched BF16/k4v4 anchor pairs
  2,048 configuration-anchor evaluations
  524,288 scored-token evaluations across both configurations
```

Derive exact expected and actual scored-token counts from the frozen masks. An ignored token is not a scored token. Do not silently change masks or fixture counts to match these arithmetic expectations. Use the already materialized stage-specific IDs, boundaries, EOS/BOS rules, and source/cluster identities; do not sample new inputs.

BF16's Fast-stage baseline is not automatically a complete Full-stage baseline. Full PPL changes the anchor set, lengths and horizon. Reuse only genuinely identical, Full-stage-compatible saved observations with exact logical prefix, targets, masks, execution semantics and adequate coverage. Do not stretch 128-token scores into 256-token scores, duplicate results, or treat a Fast summary as a Full baseline.

Record Fast/Full scoring overlap from existing fixture metadata where available. Report Full PPL as the prespecified stage-gated follow-up; do not claim it is an entirely independent confirmation set unless the actual fixtures establish that.

## 4. Reuse the validated cache-sensitive scoring path

Use the existing Q0/Q1A growing-context session and scorer. Prefer selecting `stage=ppl_full` and filtering the two configurations over creating another runner.

For an anchor:

```text
prefix  = ids[:L]
burn_in = ids[L]
targets = ids[L+1 : L+1+H]

cache = prefill(prefix)
logits = decode_one(burn_in, cache)
# These logits score targets[0]; burn_in itself is unscored.
```

Consume each prediction before feeding the token it predicts. Then teacher-force the true token when the next prediction is needed, advancing cache positions/lengths. Follow the exact approved trailing-decode convention and masks.

For BF16 and k4v4 preserve identical logical inputs, positions, scoring intervals and execution split. Do not use free-running generated tokens or ordinary full-sequence forward as the primary PPL path. Do not truncate vocabulary, score padding, use stale prefill logits, or reuse overwritten output views.

Keep stable existing FP32 scoring and sufficiently precise NLL accumulation; do not change model-forward precision, GEMM settings, quantizers, KIVI group/residual policy, backend, kernel binary, or cache layout. B=1 applies to actual model invocation, not just a dataloader label.

Construct the cache in memory from logical inputs. No new materialized KV-prefix catalog, full-vocabulary-logit archive, model download, or calibration run.

Near the model limit, retain the approved bound for prefix, burn-in, scoring, safety margin and any trailing decode. Never shorten a frozen horizon to make a point fit. Report a real unsupported/capacity failure for the affected unit instead of substituting another input.

## 5. Execute and resume ordinary units

Use the existing GPU lock, quality image, dependency overlay and resumable writer. Record the current execution identity and frozen-method binding once with the campaign; do not rebuild or recertify the runtime.

Use the protocol's atomic units and preserve completed anchor results. BF16 needs to be evaluated once for each required anchor, not repeatedly for comparisons. Run in the declared deterministic order, without score-dependent stopping or choosing better attempts.

The first scheduled real anchor may serve as the plumbing check and is retained when valid. No separate admission/smoke campaign is required.

Preserve all attempts. Retry only genuine infrastructure/transient-I/O/incomplete-output cases under the existing bounded policy, linking one accepted result to its logical slot. Never retry a valid poor NLL, failed gate or inconclusive result. Publication/receipt failure is not an inference failure; retry that operation without rerunning scoring.

A demonstrated scorer/lifecycle defect permits the smallest non-protected quality-code fix and new results for its affected units; keep obsolete outputs and do not mix scoring versions. A demonstrated protected implementation defect is outside this task: preserve its evidence and report the affected scope rather than altering kernels or rerunning performance.

## 6. Apply the unchanged Full-PPL decision

Read the finalized contract rather than rebuilding statistics from this prompt. Retain:
- paired document/anchor-cluster bootstrap;
- 10,000 draws and seed 20260722;
- actual fixed aggregation weights and confidence interval definition;
- separate dataset and length results;
- the global 1% relative-PPL margin;
- length review threshold 2% and hard-fail threshold 5%;
- the approved pass/fail/inconclusive rules.

For each applicable aggregate:

```text
delta_nll = nll_k4v4 - nll_bf16
relative_ppl_change = exp(delta_nll) - 1
global_nll_margin = log(1.01)
```

Use absolute NLL/PPL for BOTH configurations as well as their paired difference. Do not label a pooled two-dataset interval as proof of the separate dataset requirements. A Full pass requires the required WikiText-2 and C4 bounds and all guardrails, not just an attractive overall average.

Retain real cluster IDs and report effective document/cluster counts. More bootstrap draws are not more independent data. Do not resample individual correlated tokens as independent observations or use the number of anchor rows as an invented effective sample size.

Apply the actual frozen bucket rules without substituting a newly chosen point-estimate/CI rule. If a guardrail fails, report which dataset/length and why. Never impute missing baseline or method scores as zero degradation.

Do not pool Fast and Full stage results to obtain a pass. Do not alter weights, quality margins, masks, inference precision, or the eligible configuration set. An inconclusive Full result stays inconclusive; no extra anchors are authorized in this task unless an exact approved stage rule already mandates them.

## 7. Focused tests, compact outputs, and stop

Run only tests for genuinely changed Full-stage code:
- Full-stage fixture selection and exact paired coverage;
- 256-target alignment, masks/counts, and near-limit accounting;
- baseline reuse eligibility without duplicate observations;
- inherited cluster bootstrap, units and gate aggregation;
- resumable slot accounting;
- B=1 scope and unchanged failed cross-batch gate.

Reuse existing Q0/Q1A tests/results when the scorer is unchanged. Do not run full tests, all-ten-config Q0, batch-invariant diagnostics, historical admission or performance checks as entry gates.

Produce a compact result bundle using existing paths/writers, containing the plan and source references, per-anchor NLL sums/counts and pairing data, attempt index, per-dataset/per-length summaries/CIs, gate status/reasons and a short report. Keep enough paired sufficient data to reproduce the statistics; do not store every full-vocabulary logit.

Publish only new Q1B evidence through the existing R2 tool, with COMPLETE last and one verification of the new bundle. Reference old roots; do not duplicate historical bundles. Keep receipts outside the bundles they name. Protect credentials and do not delete catalogs or old raw outputs.

Execution completion and quality outcome are separate:

```text
Q1B execution: COMPLETE/PARTIAL, based on real coverage and artifact completion
BF16: baseline valid/invalid
k4v4 Full PPL: PASS/FAIL/INCONCLUSIVE
Original cross-batch gate: FAILED, unchanged
Quality evaluated batch: B=1 only
```

Only a Full-PPL PASS gives k4v4 eligibility for Q2A LongBench-E, accompanied by its required BF16 baseline. Do not run Q2A, v2, CoT, a performance-quality join, Fast-PPL expansion, or KVQuant diagnosis automatically. A Fast or Full pass alone is not a final quality-preserving performance claim.

Return a concise Q1B REPORT with actual starting/execution/final HEAD, contract/amendment identities, the two configurations, planned/completed/scored coverage, BF16 validity, dataset/length NLL/PPL and paired CIs, k4v4 gate reasons, overlap disclosure, preserved failures/replacements, Q2A eligibility, B=1 limits and the new R2 root.
