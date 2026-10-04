# Measurement protocol

Status: future behavior frozen during Phase 2. This document does not authorize
or report any measurement.

CODEX_WORKFLOW.md remains authoritative if this summary is incomplete.
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md remains authoritative for quality
scheduling. Quality execution is currently LOCKED and
PERFORMANCE_DATA_FROZEN is absent.

## 1. Preconditions for future measurement

No normal measurement begins until:

- the exact plan and all referenced identities validate;
- the applicable admission gates pass;
- a digest-pinned measurement container exists;
- identical E00 preflight passes inside that container;
- the requested model/method/backend/graph mode is explicitly supported;
- the working tree is clean;
- the GPU is exclusively available under the process-audit policy;
- memory feasibility is evaluated for the complete planned grid;
- the artifact root passes safety and append-only checks.

G1 through G5 were NOT EVALUATED at Phase 2 and Full Scan remains CLOSED.
Decision 0007 narrowly makes the protocol executable for bounded BF16 Phase 3
`native_host_admission` engineering evidence only. B-009 and B-010 remain open;
ordinary timing, formal/unified admission, later methods, and performance
claims remain closed.

## 2. Runner semantics

### Fixed-L decode

The primary response-surface runner preconstructs an effective cache of length
L and repeats a one-token decode at an unchanged shape and unchanged effective
L. Cache construction is outside timing. It measures fixed work and is the only
runner used to fit the primary latency floor, knee, and slope.

### Growing-context validation

The secondary request runner starts at L and generates a fixed, preregistered
number of tokens while context grows by one per step. It validates request-level
reconstruction and is stored separately from fixed-L samples. Its observations
never enter the fixed-L fit as if they were constant-L work.

## 3. Preparation outside timing

The following complete before warmup or measured timing:

- process startup and environment validation;
- model/tokenizer load;
- method construction and static cache allocation;
- input and cache construction;
- CUDA extension or Triton compilation;
- JIT specialization and autotuning;
- workspace allocation;
- CUDA Graph capture and capture validation;
- telemetry collector initialization;
- logging/config serialization;
- output/checksum fixture preparation.

Any preparation or allocation event that unexpectedly recurs during
measurement is an execution-path failure, even if a caching allocator serves
the request from an already reserved block and frees it before the end
snapshot. Before/after equality is necessary but not proof of zero allocation.

## 4. Warmup semantics

Warmup uses the exact method, backend, graph mode, batch, context shape, cache
layout, and decode operation of the measured point. The configured warmup count
is executed in full and never contributes timing samples.

Warmup is used to:

- complete lazy initialization;
- prove output and pointer stability;
- expose unexpected compilation or autotuning;
- reach the configured thermal/clock readiness policy;
- verify stable kernel path and absence of allocation/fallback.

Warmup count is plan-controlled and cannot be shortened after observing a
result. A failed warmup produces a recorded failure; it does not authorize
discarding the point.

## 5. Measured-step semantics and boundaries

The primary endpoint is host-observed wall time per fixed-work decode step.
CUDA-event device time is recorded as a secondary diagnostic when supported.
Raw observations are retained; an aggregate never replaces them.

The timing boundary includes only the repeated decode operation and the host
submission/device-completion cost specified by the runner. It excludes:

- cache construction or growth outside the defined one-token update;
- model/tokenizer load;
- JIT, compilation, or autotune;
- Graph capture;
- input generation and sampling;
- profiler startup;
- telemetry setup;
- serialization and log flushing;
- summary/checksum generation.

A synchronization needed to establish the start boundary occurs before the
measured region. A synchronization needed to establish completion occurs after
the measured launch/replay sequence and before reading event results. There is
no per-step tensor-to-host conversion, logging, dynamic allocation, or
unplanned CPU synchronization inside the decode region.

For Graph mode, measured work is replay only. Capture, instantiation, warmup
replays, and capture correctness checks are excluded. Eager and Graph results
remain separate lanes.

The plan freezes warmup steps, measured steps, output work, and stopping rules
before execution. A formal replicate runs the entire configured count even if
intermediate observations appear slow.

## 6. Independent process replicates

A process replicate is a fresh operating-system process with fresh model and
runtime initialization. Multiple measured steps inside one process are not
independent process replicates.

Every point retains:

- replicate index and process/session identity;
- random seed;
- full raw samples;
- start/end timestamps;
- output checksum;
- telemetry/QC records;
- status and failure reason.

Claim-bearing comparisons require the configured number of independent
replicates, never a selected subset. The workflow default is at least three;
the preregistered plan controls the exact count for each stage.

## 7. Randomization

Execution uses blocked randomization:

1. method/configuration is the block;
2. batch/context points are randomized within a block;
3. block order rotates across independent replicates;
4. the seed and fully materialized order are saved before launch.

Pilot and full-scan order is never adjusted after viewing latency. Every
planned point remains in the inventory as completed, failed, unstable, or
capacity-infeasible.

## 8. Telemetry and environment state

Future timing runs record the hardware/software provenance required by the
experiment contract plus relevant GPU state before, during, and after the
measurement window:

- GPU UUID/full SKU and active-process audit;
- power draw/limit and performance state;
- SM and memory clocks;
- temperature and ECC state;
- driver/runtime/toolkit versions;
- allocator/cache/workspace byte accounting;
- backend, kernel-path, output, and Graph-mode identity.

Telemetry sampling must not introduce tensor-to-host conversion or
synchronization into the decode region. Collection overhead and cadence are
fixed in the plan. A telemetry gap is recorded; values are not interpolated to
manufacture compliance.

Thermal, clock, power, process, allocation, output, and kernel-path checks are
quality-control evidence. A threshold violation marks the point according to
the preregistered rule, often unstable, and never causes selective deletion.

## 9. Profiler separation

Nsight Systems and Nsight Compute execute only as explicit nsys or ncu run
kinds on the preregistered subset. They use separate run IDs and artifacts.
Profiler- or audit-instrumented duration cannot enter normal timing summaries
or speedup calculations. Phase 3 may use a separate allocation/operator audit
control, but its duration is never a timing sample.

Physical HBM traffic and r_hbm require direct supported counter evidence with
the metric map, profiler version, kernel scope, and aggregation recorded.
Nominal bytes, allocated bytes, or latency cannot be converted into purported
measured HBM traffic.

## 10. Failures, exclusions, and retries

The complete grid is preregistered before results. Predicted memory-infeasible
points receive capacity_infeasible records before launch. Runtime OOM,
unsupported geometry, numerical error, Graph capture failure, backend fallback,
profiler failure, and instability retain configuration, stdout/stderr, and
machine-readable reasons.

A code or environment fix creates a new run ID and preserves the failed run.
No implementation is changed mid-run. No backend, cache dtype, graph mode,
batch, context, or safety margin silently changes to make a point succeed.

Retries follow only a preregistered, configuration-independent infrastructure
policy. Low or slow observations are never retry triggers. The first formal
replicate is never discarded because later replicates look faster.

## 11. Run finalization

Each future run follows:

    create unique staging
    -> write initial lifecycle/manifest state
    -> write raw artifacts exclusively
    -> close and fsync files
    -> validate all schemas and provenance
    -> enter finalizing
    -> generate complete artifact inventory
    -> generate SHA-256 ledger
    -> write completion marker last
    -> verify staged bytes
    -> atomically promote without replacement

Terminal-failure runs use the same finalization discipline. Unexpected
interruption leaves a distinguishable staging directory and reservation.
Neither is silently resumed or reused.

The checksum ledger covers every payload file according to the versioned
artifact schema. The completion marker authenticates the manifest and ledger.
Validation rechecks inventory membership, sizes, digests, lifecycle status,
completion state, and required provenance independently of mutable global
state.

## 12. Analysis admission

Before analysis, every run must:

- be finalized or be handled explicitly as incomplete;
- validate against its exact schema version;
- pass its checksum ledger;
- retain its provenance and exact fingerprints;
- match the requested run kind;
- satisfy the comparison key;
- have no silent fallback or undeclared exclusion.

Only normal timing runs enter ordinary latency analysis. Same-work speedup and
capacity amplification are computed and reported separately. Graph modes never
cross-compare.

No selective rerun, fastest-run selection, post-result grid edit, or raw-sample
deletion is permitted. Statistical resampling uses process/session units where
required by the workflow, not individual correlated decode steps.

## 13. Quality boundary

This protocol does not run or schedule quality evaluation. Future performance
records created before quality validation carry:

    quality_status: unvalidated
    claim_eligibility: performance_only

Quality work starts only after the complete performance plan is closed,
checksummed, explicitly marked PERFORMANCE_DATA_FROZEN, and the quality
contract receives its required approval. Until then, quality execution remains
LOCKED and no quality-only dependency may be installed.

## 14. Current interpretation

No performance or profiler data was created by Phase 2. Decision 0007 permits
only bounded native-host Phase 3 engineering timing with explicit non-claim
fields; it does not validate an ordinary timing, memory-capacity, r_hbm, method
comparison, or quality endpoint. Native-host G0 remains PASS. Phase 3 may
evaluate only its BF16 engineering G1 verdict; G2-G5 and formal/unified
admission remain NOT EVALUATED.
