# Blockers

Last updated: 2026-09-19.

## Current Q0 disposition

Q0 executed all 110 required units, but every one of the ten configurations
fails the unchanged B=1 versus B={4,8} batch-invariance tolerance. The other
100 selected units pass. This blocks Fast PPL eligibility for all ten
configurations; it is not repaired or reclassified in Q0. The next permitted
action is a separate diagnosis of the shared cross-batch numerical behavior.
Fast PPL, Full PPL, and LongBench scoring have not started.

## Current Phase 16R disposition

Phase 16R Full Scan is PASS: all 2670 planned records are terminal, five
segments and both analysis bundles are remotely verified, and all 441 feasible
logical points are stable under the primary host-wall endpoint. The original
CUDA-event outer remains immutable; a separately validated host-wall closure
corrects only the derived endpoint. No active Phase 16R blocker remains.
See `docs/phase_reports/phase16r-full-scan.md`. Phase 17 is READY, not started;
Quality LOCKED. Earlier dispositions below remain historical evidence.

## Current disposition

Phase 2 remains PASS, native-host Phase 3 BF16 G1 remains PASS, the Phase 4
common BF16 adapter is PASS, and the Phase 5 TurboQuant Reference Lane is PASS.
The immutable Phase 3 report
`phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`, SHA-256
`c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9`.
The report independently replays all 80 consolidated raw B-011/B-012 operation
bundles from the unchanged 20-run campaigns. Every earlier failed report,
campaign, and run remains immutable. B-011 through B-017 are resolved without
a silent fallback or weakened scientific gate. Phase 5 resolves B-003 only
for the explicitly bounded vLLM reference authority. It does not close or
weaken B-009 or B-010. Phase 6A remediation separately resolves both: the exact
Decision 0016 image passed container G0 and both BF16 parity smokes, and the
private indefinite-locked R2 path passed synthetic plus container-G0 clean
retrieval. The TurboQuant Measurement Adapter is implemented. Its original
final run
`phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager`
remains immutable failed evidence. The narrow B-018 remediation produced one
new sanitizer-only artifact for each mandatory configuration; all three probes
pass with exit code 0, `ERROR SUMMARY: 0 errors`, and
`LEAK SUMMARY: 0 bytes leaked`. B-018 is RESOLVED. The later approved admission
at clean execution HEAD `0df5bb4d445d48e6cba17e30723733f8de35cb14`
reran all three sanitizers successfully and completed the frozen bounded grid
9/9. Its 167-object root
`f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d`
was published COMPLETE-last and passed clean R2 retrieval. The first
publication invocation recorded `TransportError`; an explicit rerun used the
existing conditional-write path to verify identical existing objects and
create only missing objects. G2-TQ is PASS; global G2-G5 remain NOT EVALUATED.
Phase 7 then selected the official KIVI default-branch commit under Decision
0017. Its advertised Llama GQA decode path physically expands recent K/V from
8 to 32 heads through `repeat_kv`. Decision 0018 now binds one project patch
that replaces only those residual contractions with native-eight-head grouped
BMM operations. Exact CPU and SM120 BF16 equivalence and operand-shape checks
pass, so B-019 is RESOLVED under patched-source authority. Phase 7 subsequently
completed its locked environment, official extension, fixtures, sanitizer,
trace, byte accounting, and R2 publication.

Phase 8 then completed at execution SHA
`462325e9df809d3bcf24a06361bf004bc7383d73`. Its ten-point grid and all 17
strict admission checks pass. Inner root
`f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515`
and report-bearing outer root
`de7d41f151af9fe1e716f27ae0f1fc24d2ef0a4b16e8e5c3ecf45d5f9983e132`
are COMPLETE-last and cleanly retrieved. G2-KIVI is PASS; no Phase 8 blocker
remains. Global G2-G5 remain NOT EVALUATED, Full Scan is CLOSED, quality
execution is LOCKED.

Phase 9 completed the narrowly authorized KVQuant calibration and resolves
B-005. Final calibration `kvqcal-cdb724c806d64d095c040d2673a987a3`,
root `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`,
is COMPLETE-last, checksum-valid, conditionally published, and cleanly
retrieved. B-006 remains OPEN; Phase 10/E10 and E11 have not started.
G2-KVQ and global G2-G5 remain NOT EVALUATED, Full Scan remains CLOSED, and
quality execution remains LOCKED.

| ID | Blocking condition | Blocks | Evidence / next action | Status |
|---|---|---|---|---|
| B-001 | No Git commit or code SHA existed after Phase 0. | E00 durable gate evidence and every later run | Resolved by reviewed root commit 9569d938d9023a3e71d98f12234efa1897004533. | resolved 2026-07-22 |
| B-002 | Formal G0 failed because required SASS inspection could not find `nvdisasm`; runtime and sanitizer lanes were not admitted in that run. | E01 and all non-E00 CUDA or timing work | Resolved without changing E00 semantics: the exact `cuda-nvdisasm-13-0=13.0.85-1` package/tool identity is locked at 6442ba1f7554ea0ebf0b3bb1a920c94567cab689, and new immutable run `e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` passed every G0 lane. The original failed run remains unchanged. | resolved 2026-07-22 |
| B-003 | TurboQuant paper has no identified author-owned code repository; a bounded implementation authority was required for E05. | E05 golden fixtures | Resolved by selecting official vLLM release v0.25.1 at exact commit `752a3a504485790a2e8491cacbb35c137339ad34` as authority for this lane only. Source inspection found the preregistered names unchanged, so no decision record was needed. Fixtures claim conformance only to this pinned vLLM implementation, not every paper variant. | resolved 2026-07-24 |
| B-004 | Primary model ID, immutable revision, config hash, geometry, and context limit were unset. | E02 and method fixtures | Resolved by Decision 0007 and `docs/evidence/phase3/model-identity.md`: all 11 exact-revision artifacts match their frozen SHA-256 values and the full checkpoint/tokenizer load offline with the required BF16 GQA geometry. | resolved 2026-07-22 |
| B-005 | KVQuant calibration dataset/revision, preprocessing, seed, cap, and artifact did not exist. | E09-E11 | Phase 9 froze the exact WikiText-2 train identity, 16 x 2048 token tensors, seed 20260721, all-layer K/V Fisher artifacts, three quantizer families, sink/cap/dtypes, reproducibility evidence, and final root `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`; publication and clean retrieval pass. | resolved 2026-07-28 |
| B-006 | KVQuant root licensing and embedded/adapted-source lineage are incomplete. | E10 reference execution, copying, or redistribution | Map recorded embedded trees and attributed files to exact upstream commits/patch deltas; resolve repository-wide license authority. | open |
| B-007 | Archive acquisition URL and pre-workspace provenance were not supplied. | Exact external reacquisition of literature bundle | Ask the operator for origin metadata if available; retain the local archive/file hashes meanwhile. | open, non-gating for local audit |
| B-008 | qpdf is not installed for an additional PDF structural scan. | Optional defense-in-depth literature check | Install only in a reviewed environment or use an equivalent static scanner; current pdfinfo/pdfdetach checks are recorded. | open, non-gating |
| B-009 | Durable acceptance required verified private Cloudflare R2 management state, an active indefinite lock covering `kvbench/sha256/`, synthetic verification, and container-G0 publication plus clean retrieval. | E01 durable-store closure and any future claim-bearing artifact publication. | Read-only REST verification confirms bucket `kvbench-artifacts`, disabled r2.dev, zero custom domains, and exact enabled indefinite rule `kvbench-evidence-indefinite`. Synthetic root `bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e` cleanly reverified. Container-G0 root `85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c` published 222 conditional objects with COMPLETE last and cleanly retrieved with every checksum and no unexpected object. | resolved 2026-07-25 |
| B-010 | Measurement Lane execution required one exact built/scanned image, full container G0, both bounded BF16 parity smokes, and digest-bound authority. | Measurement Lane CUDA environment authority; method-specific and global admission remain separate gates. | Docker image ID / OCI image-index digest `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e` passed exact package/tool/secret/model-weight verification. G0 run `e00-20260724T195014.679255Z-a6025ae023e1-23dbe853` and both eager/graph parity runs pass. Decision 0016 authorizes only that digest, disallows floating tags, and requires full recertification after a digest change. | resolved 2026-07-25 |
| B-011 | The immutable original report failed because only high-level SDPA names were visible and found no positive materialization evidence. Both complete campaigns at execution SHA `9def265ab613cde7a06b0e51850f066d0564d635` preserve direct CUDA traces for all 80 operations. | BF16 G1, Phase 4, every later method baseline comparison. | Reporting commit `7f72c95f9932c608f9bd68f1971d6e86378596a2` independently replays the checksum-bound raw traces, source/shape evidence, allocation join, and exact operation keys. New immutable report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` derives `gqa_nonmaterialization_verified` for all 80 operations, identifies `pytorch_flash::flash_fwd_splitkv` for both controls, and passes `gqa_not_materialized`, `no_torch_cat_growth`, and `no_backend_fallback`. Preserve the corrected taxonomy and direct device-kernel contract. | resolved 2026-07-23 |
| B-012 | The immutable original eager audits had unexplained transient traffic. Decisions 0013/0014 froze the source-backed eager criterion and geometry-specific split control. | BF16 G1 eager/graph lanes, Phase 4 common baseline. | The new report independently replays every allocator event: all 72 eager operations pass `phase3_eager_attributed_ephemeral_v1` with 1,066 fully attributed events each and no failure reason; all 8 graph operations pass `phase3_graph_zero_allocation_v1` with zero events. `no_unexplained_measured_region_allocation` and `graph_replay_no_allocation` both PASS. Retain Decisions 0013/0014 and strict graph-zero-allocation. | resolved 2026-07-23 |
| B-013 | The original fixed-L terminal query race and three first-remediation recurrences remain immutable. The exact live race recurred in eight snapshots in the latest complete campaigns; every row joined to the registered GPU/PID/start time and passed as `owned_only`. Targeted tests keep unregistered foreign processes and PID reuse as hard failures. | Complete BF16 G1 process ownership. | Resolved by commit `eb908f6e372d6b232e6079e9344c2103bc90cdea`; retain the exact registered-row rule and all foreign/PID-reuse controls. | resolved 2026-07-23 |
| B-014 | The first remediation campaign exposed audit/measured checksum mismatch. Untimed exact-endpoint diagnostics showed stable eager and retained-graph outputs; the defect was inconsistent checksum serialization, not a model-output difference. Five completed latest-campaign runs preserved exact audit/measured checksum equality under the unchanged tolerances and measured region. | BF16 G1 fixed-L eager/graph lanes and every later baseline comparison. | Resolved by canonicalizing the audit witness to the frozen tensor checksum in commit `fe28f5e`; retain full-repeat eager/graph regression controls. | resolved 2026-07-23 |
| B-015 | Fifteen second-remediation workers aborted before measurement because raw-audit production did not complete; two exposed the old 1 GiB run bound and thirteen hid their lower producer cause. Untimed worst-case diagnostics preserved `WorkerProtocolError: paired allocator controls did not verify`, proved geometry-specific GQA=11/MHA=5 split counts, measured the 16-step bundle, and supported Decision 0014's 1,152 MiB envelope. The new fixed-L campaign had no size-bound or generic-wrapper failure, and every new producer failure retained its exact decode step and `ChromeTraceValidationError` cause. | Complete B-011/B-012 coverage, BF16 G1, Phase 4, every later baseline comparison. | Resolved by commit `52f41ce9d9be4edc07a833e00fe3404fbfa80b89`; retain the hard-bound/+1-byte, geometry-specific split, and exception-preservation tests. | resolved 2026-07-23 |
| B-016 | Three B-015 fixed-L graph controls (`B1/L16384`, `B4/L4096`, and `B4/L16384`) aborted before measurement with preserved `ChromeTraceValidationError: graph GPU marker is not contained by its host marker`. Decision 0015's untimed `B1/L16384` diagnostic reproduced the exact error: the host contained the unique asynchronous `cudaGraphLaunch`, while the valid MHA GPU range extended 150.023 microseconds past host return. In-memory removal of only that invalid relation recovered the exact two correlated Flash split-K nodes. | Complete B-011/B-012 graph coverage, BF16 G1, the growing campaign, Phase 4, and every later baseline comparison. | Resolved by commit `e7219e0dd714149e3eea783ce7a8602c4bf9bc54`: allow asynchronous GPU completion while retaining and strengthening launch-correlation, stream, External-ID, graph/node, unknown-category, kernel-family, and materialization checks. Deterministic parser tests, the real `B1/L16385` CUDA control, `make checks`, `make test`, `make test-cuda` (14/14), and `make test-graph` (3/3) pass. Retain these controls in both new complete campaigns. | resolved 2026-07-23 |
| B-017 | Immutable report `phase3-g1-20260723t123322160580z-9def265a-08dc69` remains valid FAIL evidence for the former legacy-summary/raw-bundle disconnect. | BF16 G1, B-011/B-012 closure, Phase 4, every later method baseline comparison. | Commit `7f72c95f9932c608f9bd68f1971d6e86378596a2` binds report behavior to its recorded generator SHA, reuses coordinator raw replay, rejects missing/tampered/mismatched evidence, ignores serialized worker `passed` booleans, and derives the five criteria from raw bytes. `make checks`, `make test`, `make test-cuda` (14/14), and `make test-graph` (3/3) pass. New no-replace report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` independently validates PASS with no errors; the two campaigns and 20 source runs were not rerun or modified. | resolved 2026-07-23 |

| B-018 | The exact pinned TurboQuant store-append-decode probe completed functionally, but Compute Sanitizer memcheck reported 2,093,260 leaked bytes in 28 allocations and exited 99 in final run `phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager`. | G2-TQ and the bounded admission grid | Resolved without changing the algorithm, fixtures, container, runner, timing, or sanitizer criteria. Commit `aac794c21b01b3e43ff93e317286285d21dbcd47` explicitly releases sanitizer-only CUDA storages and library resources. New runs `phase6-b018-20260725t141636785z-aac794c2-6444b4-4bit_nc-sanitizer`, `phase6-b018-20260725t141640440z-aac794c2-eb5675-k3v4_nc-sanitizer`, and `phase6-b018-20260725t141643545z-aac794c2-4d7433-3bit_nc-sanitizer` each pass the probe, exit 0, and record zero errors and zero leaked bytes with valid COMPLETE/inventory/checksum ledgers. Current-HEAD admission at `0df5bb4d445d48e6cba17e30723733f8de35cb14` independently reran all three probes successfully before the frozen grid passed 9/9. | resolved 2026-07-25; current-HEAD confirmation 2026-07-26 |
| B-019 | The selected official KIVI `main` implementation advertises Llama 3/GQA support but calls Transformers `repeat_kv` for recent key/value regions. At H_Q=32/H_KV=8 the exact expand/reshape helper creates a distinct contiguous 32-head tensor with four times the eight-head storage. | Phase 7 KIVI Reference Lane, E07, E08, and G2-KIVI | Decision 0018 authorizes one exact project patch on the Decision 0017 commit. The patched tree passes exact BF16 formula, native-eight-head operand, forbidden-operation, head-mapping, and unsupported-geometry checks on CPU and SM120. Phase 7 reference evidence and Phase 8 static-adapter admission both preserve that exact patched authority. | resolved under patched-source authority 2026-07-26; confirmed by G2-KIVI 2026-07-27 |
| B-020 | A Decision 0031-feasible kvq2 B=1/L=131072 Pilot worker exceeded the frozen 7,200-second supervision deadline and was terminated with return code -15 before finalizing a timing run. | Phase 13 completion and Phase 14 readiness | Decision 0032 and Phase 13T prove normal forward progress, bind ordered finite geometry-scaled stage deadlines, preserve the stopped campaign, and publish cleanly retrieved root `5e98c8103c6e15ca0877e39f7b9363a208ba8722bb6993ef67a04a06c9f795cb`. The later wholly new successor Pilot completes all 684 feasible records. | resolved 2026-08-03; successor Pilot PASS 2026-08-25 |
| B-021 | The q4 Value-decode caller workspace was hard-coded to 32 tiles, so blocked Pilot `phase13-20260804t111810342595z-a127b0d1-8649c3` failed at kvq4 B=8/L=16384 before Graph capture because 128 tiles were required. | Phase 13 completion and successor Pilot | Decision 0036 derives immutable preallocated workspace geometry from declared total-attended capacity. All nine fixtures, targeted B=4/8 long-context controls, Graph/allocation/stream/sanitizer, successor q4 admission, q4 G5 refresh, R2 publication, and clean retrieval pass at root `9f027d64424844d0d62311daad5740e2b76960d1d5e7b8be26aa1ccba100a8db`. The blocked campaign remains immutable and cannot be resumed. | resolved 2026-08-20 |

## Phase 4 disposition

The adapter boundary adds no new blocker and closes none. Its report and three
functional smoke records remain local, checksum-bound, non-claim evidence. At
Phase 4 completion, B-009 still required durable append-only publication and
B-010 still required a digest-pinned Measurement Container with parity G0.
The separate Phase 5 reference environment did not satisfy container parity;
Phase 6A remediation later resolved both blockers independently.

## Phase 5 disposition

B-003 is resolved for the narrowly selected vLLM authority. The exact source,
environment, fixtures, and checksum ledgers reproduce on SM120; no source or
configuration substitution occurred. Direct CUDA Graph smoke was not exercised
in the minimal official API and remains Phase 6 work, but is not a Phase 5
blocker under the frozen acceptance criteria. At Phase 5 completion, B-009 and
B-010 remained OPEN; Phase 6A remediation later resolved them.

## Phase 6 entry-blocked disposition

Phase 6 was attempted and BLOCKED at entry because B-009 and B-010 were both
unresolved. At that entry, B-009 was OPEN without durable append-only
publication and clean retrieval verification, and B-010 was OPEN without an
authorized digest-pinned Measurement Container and container G0 parity.
Neither blocker is narrowed or resolved by the retrospective record. No Phase
6 implementation is present. The initial Phase 6A prerequisite attempt did not
alter this retrospective record and left both blockers open.

## Phase 6A remediation disposition

Phase 6A remediation reused the existing implementation and directly resolved
B-009 and B-010. Decision 0016 binds Measurement Lane CUDA execution to exact
Docker image ID / OCI image-index digest
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
Container G0 and both bounded BF16 parity smokes pass. Cloudflare management
state is private, and exact enabled rule `kvbench-evidence-indefinite` retains
`kvbench/sha256/` indefinitely. The synthetic root and container-G0 root
`85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`
both cleanly retrieve and validate.

At Phase 6A completion, blocker closure supplied environment and durable-store
authority only; Phase 6 had not yet been restarted and G2-TQ was
`NOT EVALUATED / READY`. The later Phase 6 disposition below supersedes that
historical entry state.


## Phase 6 measurement-adapter disposition

The minimal adapter and all three frozen fixture audits pass, including exact
packed bytes, byte accounting, execution path, eager allocation, and fixture
Graph replay. The mandatory sanitizer gate fails in finalized run `phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager`.
The stop rule prohibited the remaining sanitizer configurations and the entire
bounded grid. The failed artifact is immutable, independently valid, and
durably published at `r2://kvbench-artifacts/kvbench/sha256/f319c4b05054ee2f31bdcbfe15fa67850ea784c17718192b82e527567c5cf343/`. This durable failure does not admit any
configuration.

The subsequent B-018-only remediation preserved that evidence and used new
run IDs under the same authorized image. All three mandatory sanitizer probes
now pass with zero errors and zero leaked bytes, and each finalized
sanitizer-only bundle independently validates. Their `aborted` lifecycle is
intentional: it records that the authorized B-018-only scope completed without
entering the bounded grid. B-018 is RESOLVED. A later clean-HEAD admission
reran all three sanitizers, passed the frozen grid 9/9, and durably published
and cleanly retrieved the 167-object admission bundle. G2-TQ is PASS; global
G2-G5 remain NOT EVALUATED, Full Scan remains CLOSED, quality execution remains
LOCKED, and all original failures remain immutable.

## Phase 7 KIVI source-audit disposition

The full entry suite passed at clean HEAD
`0974bbc98f8f941b09800786591108292dc4e0dd`, including independent clean
retrieval of both the 176-object outer Phase 6 root and unchanged 167-object
original root. Exact source audit then selected official KIVI `main` commit
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` under Decision 0017.

The source and an exact non-timing BF16 storage-semantics audit prove that the
advertised primary-model GQA path materializes recent K/V at H_Q=32. B-019 is
therefore OPEN. The hard stop occurred before any reference container, CUDA
build, fixture, sanitizer, trace, graph smoke, byte accounting, or Phase 7 R2
publication. At that stopped source-audit attempt, Phase 8 remained unstarted
and the factory remained fail-closed.

### B-019 remediation

A fresh entry and remote-ref audit started at clean commit
`755c1bdb87af3e7becda792bd5d300ab877fee7e`; no new author-maintained KIVI
revision existed. Decision 0018 therefore authorized exactly one patch on the
same official base commit. The patch and resulting tree are checksum-bound and
replace only residual GQA contractions. Quantization, packing, metadata, cache
layout, rollover, and CUDA extension source are unchanged.

The patched source matches the original repeat formula exactly in BF16 at
H_Q=32/H_KV=8 for contexts 17 and 33 on CPU and SM120. Its BMM K/V operands
retain leading dimension `batch * 8`, the head map is `query_head // 4`, and
prohibited expansion operations are absent. B-019 is therefore RESOLVED under
patched-source authority. No reference environment, official extension build,
fixture, sanitizer, trace, graph smoke, byte accounting, R2 fixture bundle, or
Phase 8 work was started.

### Phase 7 completion after B-019

The continuation entered at clean commit
`3417ea0e7f322369eed21bb787a9a9a19b0a69bd` and revalidated every prerequisite.
The exact patched tree then built the unchanged official CUDA extension in the
isolated SM120 reference image. Native cubin, PTX/JIT, minimal sanitizer,
four fixture configurations, rollover, byte ownership, native eight-head GQA,
and non-performance trace controls all pass.

B-019 remains RESOLVED under the same Decision 0018 authority. The final
reference bundle is COMPLETE-last and cleanly retrieved from content root
`abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302`.
No Phase 7 blocker remained. At that historical phase boundary, this did not
admit G2-KIVI or authorize Phase 8; the KIVI Measurement Adapter was still
absent and fail-closed.

## Phase 8 KIVI measurement-adapter disposition

Phase 8 reused the existing runners, graph harness, allocation/path audits,
artifact lifecycle, process supervision, and R2 publisher. The exact
Decision 0018 source and extension, all four frozen fixtures, static rollover,
canonical byte ratios, native eight-head GQA, eager attribution, zero-allocation
Graph replay, minimal sanitizer matrix, and bounded grid pass.

The first append-only `96cd5248` attempt stopped at `bounded_grid_8` with
`EndpointSessionError: session audits did not pass`. Its artifacts remain
immutable and were neither overwritten nor promoted. The successful clean-HEAD
execution at `462325e9df809d3bcf24a06361bf004bc7383d73` used new run IDs and
passed 10/10 points. The 331-object inner and 341-object report-bearing outer
bundles are COMPLETE-last and cleanly retrieved.

G2-KIVI is PASS and no Phase 8 blocker remains. At that historical Phase 8
boundary, B-005 and B-006 remained open for separately authorized Phase 9
KVQuant work; Phase 8 did not narrow or resolve them. Global G2-G5 remained
NOT EVALUATED, Full Scan remained CLOSED, quality execution remained LOCKED,
and Phase 9 had not started. The subsequent Phase 9 disposition follows.

## Phase 9 KVQuant calibration disposition

Phase 9 reconstructed the exact Decision 0021 patched source without changing
the patch. One separate calibration image froze the exact model/tokenizer,
WikiText-2 train data, 16 x 2048 token tensors, seed and precision contract.
One full collection produced all 32 pre-RoPE K and all 32 V finite Fisher
artifacts. Exactly `kvq4`, `kvq3`, and `kvq2` were generated from that shared
Fisher root with sink tokens 5 and shared K/V cap 12.

The first local token-probe failure and the first complete calibration whose
safe-format regeneration lacked an explicit tensor-value acceptance rule are
preserved under their original finalized failed IDs. They were not overwritten.
The final fresh ID `kvqcal-cdb724c806d64d095c040d2673a987a3` passes token
reconstruction, representative K/V replay, all-family fresh-process tensor
regeneration, policy replay, inventory, checksums, and COMPLETE-last
validation. Its 68-object root is
`8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`.
The first R2 transport failure stopped before `COMPLETE`; the same conditional
publisher then verified the six identical existing objects, uploaded the
remaining objects, wrote `COMPLETE` last, and passed clean retrieval.

B-005 is RESOLVED for this exact calibration root. B-006 remains OPEN and
continues to gate Phase 10/E10 reference work and any applicable copying or
redistribution. No Phase 10 fixture or Measurement Adapter exists, so G2-KVQ
remains NOT EVALUATED. Global G2-G5 remain NOT EVALUATED, Full Scan remains
CLOSED, quality execution remains LOCKED, and `PERFORMANCE_DATA_FROZEN`
remains absent.

## Phase 13B compressed-batch disposition

The stopped Phase 13 campaign remains immutable and durably retrieved. Decision
0030 separately authorizes only static B=1/4/8 geometry for the nine compressed
main configurations. Phase 13B passed 27/27 bounded container points, exact B=1
preservation, byte/path/GQA/allocation, eager/Graph, pointer, stream, and
sanitizer controls without changing CUDA or method semantics. Its successor
reports are checksum-bound, and the 52-object root
`f1c96eaacbbace1c23b249d1afe8d892aa26c3f6b8d04e07f373a2becafba1fe`
is COMPLETE-last and cleanly retrieved. No Phase 13B blocker remains. Any Pilot
must start as a completely new campaign; the stopped campaign cannot resume.

## Phase 13F peak-feasibility disposition

Decision 0031 corrects only the pre-launch feasibility contract. All 810
records were deterministically recomputed with model weights, cache and
persistent workspace, endpoint controls/workspace, source-derived prefix
attention/MLP peak, Graph reserve, and the frozen 0.88 limit. The former
`tq_3bit_nc` B=8/L=98304 failure is now source-faithfully
`capacity_infeasible`. The 9-object root
`5090b193c046637cb3836f7d5a3ee5ebbad95d9a459672ab4aa3ff0ddb756589`
is COMPLETE-last and cleanly retrieved. No Phase 13F blocker remains. Both
stopped campaigns remain immutable; Phase 13 still requires a wholly new
Pilot campaign and neither stopped campaign may resume.

## Phase 13R2 supervisor-timeout disposition

Decision 0031 correctly retained 684 feasible and 126 capacity-infeasible
records, including tq_3bit_nc B=8/L=98304 as capacity-infeasible. Fresh
campaign phase13-20260802t045837322693z-3127f1d1-486dcb stopped fail-closed
when feasible kvq2 B=1/L=131072 exceeded the frozen 7,200-second supervisor
deadline. It preserves 23 completed, one runtime-failed, 660 aborted, and 126
capacity-infeasible records with zero selective retries. Root
581b02a6ca1a09c976a899b2b5d7eeb7897c0ad8f7ed8ad9fb11be5f6475f327 is
COMPLETE-last and cleanly retrieved. This campaign cannot be resumed or
reclassified.

## Phase 13T supervisor-timeout disposition

The exact-container kvq2 B=1/L=131072 diagnostic completed all 32 prefix
layers, Graph capture, warmup, measurement, and finalization with normal
forward progress. Decision 0032 replaces only the single global deadline with
ordered, finite, geometry-scaled stage deadlines; heartbeats cannot extend
them. The unchanged feasibility matrix remains 810/684/126. The 16-object root
5e98c8103c6e15ca0877e39f7b9363a208ba8722bb6993ef67a04a06c9f795cb is
COMPLETE-last and passed one clean retrieval. B-020 is resolved for
supervision; Phase 13 still requires a wholly new Pilot campaign.

## Phase 13 successor Pilot disposition

Fresh campaign `phase13-20260822t150835736582z-4ddd7b17-3a8fb3` completes
684/684 feasible runs, records all 126 capacity-infeasible cases, and has zero
failed, aborted, unstable, fallback, output-mismatch, kernel-drift,
allocation-drift, or selective-rerun results. Its 17,384-object root
`feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531`
is COMPLETE-last and passed one clean retrieval. B-020 and B-021 remain
resolved; no concrete execution blocker remains for Phase 13. Phase 14 remains
gated because 25/30 provisional knees lack sufficient density, requiring a
separately preregistered densification task rather than alteration or selective
extension of this immutable campaign.
