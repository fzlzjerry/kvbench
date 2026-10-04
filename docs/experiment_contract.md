# Experiment contract

Status: Phase 2 contract; no performance endpoint or quality endpoint has been
executed or validated.

Authority: CODEX_WORKFLOW.md for active performance work and
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md for all quality scheduling and
execution.

## 1. Research objects

The primary research object is an exact method configuration evaluated on an
exact model, hardware, software environment, runner, graph mode, backend,
batch size, and effective context length.

The project distinguishes:

- a configuration template, which may contain explicit unresolved identities;
- a resolved experiment configuration, whose required identities and content
  hashes are complete;
- a process replicate, which is one fresh operating-system process for one
  preregistered point or block;
- a raw sample, which is an observation retained inside that replicate;
- a run, which is one append-only artifact unit with a manifest, inventory,
  checksum ledger, and completion state;
- a comparison, which joins compatible finalized runs only after provenance,
  status, and comparison-key validation;
- a claim class, which keeps same-work latency, capacity amplification, and
  mechanism evidence separate.

Reference-lane outputs establish algorithmic fixtures and correctness. They do
not supply cross-method performance numbers. Only a future unified Measurement
Lane may generate comparable performance data after all applicable admission
gates pass.

## 2. Identity requirements

### Model identity

A resolved model identity requires:

- repository or registry model ID;
- immutable model revision;
- model configuration SHA-256;
- tokenizer ID and immutable revision where tokenization is relevant;
- weight dtype;
- verified model geometry and maximum context;
- hashes for any local model/config artifacts used by the run.

The Phase 2 primary-model file is a blocked template. An unresolved identity is
never silently replaced by a guessed model or floating revision.

### Method identity

A resolved method identity requires:

- method name and versioned configuration schema;
- exact semantic fields such as bit widths, group size, residual window, sink
  policy, outlier cap, skipped layers, layout, backend, and graph support where
  applicable;
- exact source revision and adapter/kernel hashes once an implementation
  exists;
- calibration artifact hashes where the method requires calibration;
- a deterministic method-configuration fingerprint.

TurboQuant, KIVI, and KVQuant templates record only preregistered fields.
Templates do not assert backend support, source equivalence, calibration
availability, or implementation readiness.

### Hardware identity

A claim-bearing run requires an immutable hardware-manifest reference and
fingerprint covering at least full GPU SKU, UUID, PCI identity, compute
capability, VRAM, ECC state, driver-visible power/clock state, and relevant host
identity. Native-host E00 evidence is a certification input, not a substitute
for the future measurement-container hardware/software manifest.

### Software identity

A claim-bearing run requires:

- Git SHA;
- explicit dirty-tree state;
- digest-pinned measurement container;
- driver, CUDA runtime/toolkit, Python, PyTorch, Triton, and backend versions;
- dependency-lock and relevant executable or binary hashes;
- artifact-schema and experiment-contract versions.

B-010 remains open: no measurement-container digest or container-parity G0 is
currently available. Decision 0007 permits only bounded, explicitly non-claim
Phase 3 `native_host_admission` engineering evidence on the certified host;
claim-bearing and ordinary timing still require the container identity above.

## 3. Fingerprints and canonical configuration

Schema-bearing data includes a schema version. Canonical serialization is
compact UTF-8 JSON with sorted keys, deterministic separators, no NaN or
infinity, and exact null semantics. SHA-256 over those canonical bytes is the
fingerprint.

A resolved run retains, rather than recomputes from mutable global state:

- experiment/config fingerprint;
- method-configuration fingerprint;
- model fingerprint;
- hardware-manifest fingerprint;
- software-environment fingerprint;
- experiment-contract fingerprint;
- artifact-schema version.

Paths alone are not identities. Referenced artifacts require content hashes.
Quality and performance may later join only on the exact fingerprint required
by the post-performance quality protocol; joining on a broad method name is
forbidden.

## 4. Comparison rules

### Same-work latency

A same-work comparison is valid only when both sides match exactly on:

- model checkpoint/revision and weight dtype;
- runner kind;
- batch size and effective context length;
- output work and sampling mode;
- graph mode;
- attention backend and relevant compile mode;
- cache lifetime and timing boundary;
- tensor-parallel size and hardware identity;
- compatible software/contract versions;
- run kind equal to normal timing;
- preregistered randomization and replicate protocol.

Compressed and BF16 runs with unequal work are not a speedup pair.

### Graph mode

Graph-on results are compared only with graph-on results. Graph-off results are
compared only with graph-off results. A future Graph A/B mechanism experiment
may vary graph mode only while holding method, cache, backend, runner, and shape
fixed. It is not a cross-method speedup shortcut.

### Capacity amplification

Capacity amplification asks which additional batch/context points fit under a
fixed memory policy. It is not a same-work latency speedup. Capacity-infeasible
BF16 points may support a separately labelled capacity claim but cannot provide
a same-work latency denominator.

### Performance, profiler, and quality lanes

Normal timing, Nsight Systems, and Nsight Compute are distinct run kinds.
Profiler-instrumented duration never enters ordinary latency analysis.
Profiler runs may explain mechanisms only for their exact configuration.

Quality is a separate future run family governed by
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md. Quality runtime is not performance
runtime, and quality results do not modify completed performance artifacts.

## 5. Metric semantics

Metrics not collected are explicitly null or absent according to their schema;
zero is never used to mean missing.

The project distinguishes:

- nominal compression from configured bit width;
- allocated compression from actual storage allocation;
- HBM compression from directly measured physical traffic.

r_hbm is null outside a profiler-supported record unless a documented direct
measurement source exists. Estimated nominal bytes, allocated bytes, latency,
or a model-derived traffic estimate can never populate r_hbm. Every byte
breakdown must be internally checkable.

No Phase 2 template or synthetic fixture is a scientific measurement.

## 6. Provenance required for a run

A resolved run manifest records at least:

- run ID, run kind, lifecycle/status, timestamps, and schema version;
- intended command and normalized plan source;
- exact config snapshot or canonical inline config;
- Git SHA and dirty-tree state;
- container/environment identity;
- hardware-manifest identity;
- method and model identities and fingerprints;
- runner kind, graph mode, backend, seed, and process replicate;
- admission/precondition outcomes;
- raw artifact inventory and SHA-256 checksums;
- completion marker and final manifest digest;
- explicit failure or exclusion reasons.

No performance claim is valid without Git SHA, container digest, hardware
manifest, complete config, raw samples, and independent process replicates.

## 7. Dirty-tree policy

Future claim-bearing and admission runs refuse a dirty tree before creating a
run directory. A validation dry run may report a dirty tree without producing
scientific artifacts. If an explicitly authorized diagnostic run permits a
dirty tree, it records the dirty state and a diff fingerprint and is ineligible
for a claim. A clean later run receives a new run ID; the diagnostic is never
rewritten.

E00 retains its separately certified clean-tree refusal semantics.

## 8. Run lifecycle and terminal states

The supported lifecycle is:

    created -> running -> finalizing -> completed

The supported terminal failure statuses are:

- build_failed;
- runtime_failed;
- numerical_failed;
- graph_capture_failed;
- profiler_failed;
- capacity_infeasible;
- unstable;
- backend_fallback;
- unsupported_geometry;
- aborted.

A completion marker means the artifact unit was finalized, not that the
scientific gate passed. Successful and terminal-failure runs are both preserved.
An interrupted staging directory remains distinguishable and is never silently
reused. A fix creates a new run ID.

## 9. Append-only raw-data policy

Raw data and finalized manifests are append-only:

- write only inside a unique staging directory;
- reject any existing final, reservation, or incomplete run ID;
- use exclusive file creation and reject unsafe paths/symlinks;
- validate schemas before finalization;
- generate a complete inventory and SHA-256 ledger;
- write an authenticated completion marker last;
- atomically promote without replacing an existing target;
- expose no supported operation that edits a finalized run;
- retain failed and interrupted evidence.

Local chmod is defense in depth, not a durable-retention guarantee. B-009
remains open until the project selects and demonstrates a durable append-only
store, immutable locator/publication scheme, and retention policy.

Completed E00 evidence under docs/evidence/e00 is a separate Decision-0002
boundary and must never be used as a mutable test root.

## 10. Exclusions, infeasibility, and failure

Every planned but unanalyzed point has a machine-readable ExclusionRecord with
the original plan key, status, reason code, evidence reference, and timestamp.
Exclusions are never implemented by deleting rows or omitting a planned point
silently.

Memory feasibility is assessed before launch using the frozen safety margin.
A predicted infeasible point is recorded as capacity_infeasible. An unexpected
OOM preserves logs and the attempted configuration; it is not rerun with a
smaller batch, context, workspace, or different method unless the
preregistered policy applies uniformly and creates a new run.

Backend or cache-dtype fallback is forbidden. A detected mismatch finalizes as
backend_fallback with direct evidence. It is never relabelled as the requested
method and never silently rerun under a substitute backend.

Unsupported geometry, instability, numerical failure, Graph failure, and
profiler failure remain visible in the planned grid. No failure authorizes
selective reruns or cherry-picking.

## 11. Quality governance

Quality execution is LOCKED. PERFORMANCE_DATA_FROZEN is absent. PPL, NLL,
LongBench, LongBench-E, LongBench v2, RULER, NIAH, lm-evaluation-harness, and
all other quality benchmarks are prohibited in this phase, as are quality-only
dependency installations.

Performance data created before post-performance quality validation carries:

    quality_status: unvalidated
    claim_eligibility: performance_only

Those fields do not claim that performance data currently exists. They define
the required metadata for future data. Quality unlock, performance freeze,
exact-fingerprint joining, and rerun rules come only from
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md.

## 12. Admission and current claim boundary

Native-host G0 remains PASS for its exact recorded fingerprint. G1 through G5
were NOT EVALUATED at Phase 2, and the full scan remains CLOSED. Decision 0007
permits only a Phase 3 BF16 engineering G1 verdict from bounded, locally
finalized non-claim evidence; it is not formal/unified admission and does not
close B-009 or B-010. No method enters a pilot or full scan until all applicable
gates pass.

Phase 2 evidence can support only claims that schemas, validation, dry-run
reconstruction, and local append-only lifecycle controls behave as tested. It
supports no latency, throughput, capacity, HBM-traffic, method-ranking,
deployment, model-quality, or paper-result claim.
