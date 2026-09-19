# Research task breakdown

Status values in this research ledger are planning states, not admission
results. No remote issue tracker is configured, so this file is the
authoritative local task index until issues are created elsewhere.

| ID | Scope | Depends on | Required evidence / gate | Status |
|---|---|---|---|---|
| E00 | Hardware preflight | Phase 0 PASS | hardware manifest; native extension; PTX/JIT; Compute Sanitizer; G0 | complete: native-host G0 PASS in `e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` |
| E01 | Repository scaffold and schemas | G0 | strict schemas; append-only writer; durable artifact policy; digest-pinned container; parity preflight | complete: local contracts/writer PASS; exact Decision 0016 image built/scanned; container G0 plus BF16 eager/graph parity PASS; private indefinite-locked R2 state verified; synthetic and container-G0 roots cleanly retrieved. B-009/B-010 RESOLVED. |
| E02 | BF16 static-cache baseline | E01 and container-parity G0 for formal closure; Decision 0007 for Phase 3 engineering scope | reference numerical match; static allocation; GQA audit | Native-host engineering G1 PASS in independently validated immutable report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`. All 20 criteria pass from unchanged campaigns/raw audits. Phase 6A eager/graph runs establish container parity only and create no new unified or claim-bearing G1 result. |
| E03 | Fixed-L benchmark | E02; Decision 0007 permits only bounded Phase 3 admission runner | fixed-L and growing-context runners; timing-boundary tests | Execution SHA `9def265ab613cde7a06b0e51850f066d0564d635` completed all 16 fixed-L and all 4 growing-context runs with new IDs, no abort/failure, no unattempted point, and no selective rerun. The results remain non-claim admission evidence. |
| E04 | Common cache-method and CUDA Graph harness | E02-E03 | stable method protocol; correctness/allocation/path facades; capture/replay correctness; no replay allocation | complete for Phase 4 BF16 at `0cf160caa532c7cac23275c8a14fd8694789a86f`; `docs/evidence/phase4/method-admission.json` validates. TurboQuant reference work is complete, but later Measurement Lane adapters still require their own G2 evidence. |
| E05 | TurboQuant reference lane | E00-E04 | authoritative pinned source; isolated container; golden fixtures | complete: official vLLM v0.25.1 commit `752a3a504485790a2e8491cacbb35c137339ad34`; isolated SM120 environment; 3 mandatory MSE+NC and 1 held-out deterministic fixture; exact regeneration and validation PASS |
| E06 | TurboQuant measurement adapter | E05 | numerical, byte, graph, path, sanitizer, smoke evidence; G2-TQ | complete: execution HEAD `0df5bb4d445d48e6cba17e30723733f8de35cb14`; current-HEAD sanitizer 3/3 PASS; frozen bounded grid 9/9 PASS; 167-object root `f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d` COMPLETE-last and cleanly retrieved; G2-TQ PASS |
| E07 | KIVI reference lane | G2-TQ | exact patched-source authority; isolated environment; rollover and K/V asymmetry fixtures; native eight-head GQA storage | complete: exact source plus Decision 0018 patch; locked reference image; SM120/PTX/JIT and zero-error sanitizer; four deterministic fixture configurations; rollover, actual bytes, native eight-head GQA, non-performance trace; COMPLETE-last R2 publication and clean retrieval PASS |
| E08 | KIVI measurement adapter | E07 | static buffers; canonical rho_alloc/r_alloc; GQA indexing; G2-KIVI | complete: execution SHA `462325e9df809d3bcf24a06361bf004bc7383d73`; four configurations conform; sanitizer, path/allocation, rollover, Graph, and frozen grid 10/10 PASS; 331-object inner root `f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515` and 341-object report-bearing outer root `de7d41f151af9fe1e716f27ae0f1fc24d2ef0a4b16e8e5c3ecf45d5f9983e132` are COMPLETE-last and cleanly retrieved; G2-KIVI PASS |
| E09 | KVQuant calibration | G2-KIVI | frozen dataset/revision/seed/cap/artifacts/checksums | complete: Decision 0021 patched source, isolated calibration image, exact model/tokenizer, WikiText-2 train 16 x 2048 tokens, all 32 K plus 32 V Fisher artifacts, `kvq4`/`kvq3`/`kvq2`, sink 5, K/V cap 12, reproducibility, 68-object root `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf` COMPLETE-last and cleanly retrieved; B-005 RESOLVED; G2-KVQ remains NOT EVALUATED |
| E10 | KVQuant reference lane | E09 | dense/sparse/sink fixtures for 4/3/2-bit and source-faithful Key occupancy cases | complete: Decisions 0021/0023; nine deterministic source-authoritative fixtures; Key counts 0/6/12; non-sink Value fixed-extrema count 12 and sink count 0; native SM120/PTX/JIT and sanitizer PASS; fixture ID `kvqref-a50af6511c314b6394e58a7f81ceefb8`, 113-object root `32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab` COMPLETE-last and cleanly retrieved; G2-KVQ remains NOT EVALUATED |
| E11 | KVQuant measurement adapter | E10 | fixed sparse buffers; byte breakdown; graph/path tests; G2-KVQ | complete: Decision 0027 deterministic q4 decode binding and Decision 0029 current-source re-admission remain preserved. Decision 0036 adds only capacity-derived preallocated q4 workspace geometry; all nine fixtures, long-context admission, Graph/allocation/stream/sanitizer checks pass. Successor q4 report SHA-256 `75605637f460a309081e1e0a4065e90e8e14194d365250cb513092616ef89ec7` is bound to cleanly retrieved root `9f027d64424844d0d62311daad5740e2b76960d1d5e7b8be26aa1ccba100a8db`; q3/q2 and historical evidence remain unchanged; G2-KVQ remains PASS. |
| E12 | Admission gates | E02-E11 | machine-readable G1-G5 report for every main configuration | complete: original Phase 12R campaign `phase12-20260731t062914664948z-6165f78d-c78b9a` remains immutable. Phase 13R reuses nine unchanged records and refreshes only q4 with three independent standardized processes (CV 0.162733%); refreshed report SHA-256 `13553823a68f34a0a538674caeb5abf2f9e638d5086b90d21fbdb94aa66be05a`; G0-G5 PASS and no speedup/comparative claim. |
| E13 | Pilot scan | E12 PASS; Phase 13B PASS; Phase 13F PASS; Phase 13T PASS; Phase 13R q4 workspace PASS | immutable randomized samples; QC; provisional knees; pilot gate | complete: fresh campaign `phase13-20260822t150835736582z-4ddd7b17-3a8fb3` preserves 810/810 records and cleanly retrieved root `feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531`. Phase 13D preserves its first 51 valid runs and failed-finalization record, then completes one replacement plus the frozen remaining 200 in a separate continuation segment. All 252 logical records pass QC, all 25 targets have resolved density/boundary statuses, and root `a8559a5e01edaad949df1e128c4bddff37638801cfc89f2eb8d4894c31ef82d2` is COMPLETE-last and cleanly retrieved. Phase 14 used this evidence as immutable entry authority. |
| E14 | Nsight Systems integration | E13 and M14-GRAPH-AB PASS | nsys-only runs around knees; launch/sync/kernel evidence | complete: 32/32 selected traces and 16 eager/Graph pairs; Graph reduces CPU CUDA-call count and GPU idle in 16/16 while kernel count/order and observed overlap remain unchanged; profiler durations remain mechanism-only |
| E15 | Nsight Compute integration | E13 | current-SM metric discovery; measured traffic; ncu-only runs | complete: 22/22 selected profiles, live SM120 metric discovery, all-ten B=1/L=131072 same-work traffic, exact kernel classification, and measured cache-path `r_hbm`; root `641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8` cleanly retrieved |
| E16 | Full scan | pilot gate, M14-GRAPH-AB, E14-E15 | preregistered grid; feasibility/exclusion records; immutable samples | complete: Phase 16R PASS; 2670 terminal records, 2205 completed, 465 capacity-infeasible; five remote-verified segments and host-wall closure root `5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e`; 441 stable points, zero selective reruns |
| E17 | Knee and response-surface fitting | E16 | candidate models; strict holdouts; session bootstrap CIs | complete: Phase 17 PASS; 53 grouped folds and 265 fixed candidate fits over the immutable Phase 16R host-wall data; D selected, all applicable accuracy targets missed without threshold changes, knee-error target not evaluable, 23/50 local knees identified; 44-object root `05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb` cleanly retrieved; Quality LOCKED and Phase 18 not started |
| E18 | Reproducibility package | E17 and all gates | pinned containers; reproduction commands; figures; final report | complete: Phase 18 PASS; pure-stdlib CPU reproduction, standalone frozen-D predictor, explicit macro-label/BF16-tail audit, six data-driven figures, research report, quality-freeze handoff draft, and 54-object cleanly retrieved root `cd6ee2324d0163088102e29e45b4ad124b5d524fb1900717faddf997da6805e2`; missed Phase 17 targets unchanged; Quality LOCKED and QP-0 not started |
| QP-0 | Freeze completed performance release and prepare quality handoff | E18 | complete inventory; local annotated tag; hot-path lock; Quality image; manifest-bound pending contract; R2 publication | complete: the original tag is preserved; corrected local tag `perf-freeze-20260917-83536c37-r1` binds the ledger-verified finalized manifest and unchanged 14-object root `9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f`. Nine-object correction root `44c9594079e372d708d509449589a428f4b27c0392e2f1b1e491acc3d17e2559` is COMPLETE-last and cleanly retrieved. Performance freeze marker present; QP-1 and its later Q0-only approval reuse this binding without recreating it. |
| QP-1 | Complete quality contract and materialize evaluation inputs | QP-0 | dependency lock; tokenizer/prompt hashes; dataset revisions; sample IDs; compact fixtures; human approval handoff | preparation complete: six technical items PASS under contract `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`, SHA-256 `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`; 62-object root `6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62` is COMPLETE-last and cleanly retrieved. Exact operator approval is recorded separately and authorizes Q0 only; PPL/LongBench remain unauthorized and not run. |
| Q0 | Cache-sensitive correctness under the approved quality contract | QP-1 plus exact operator approval | core growing probes; suffix/decode-cache dependence; eager/Graph and batch invariance; per-configuration Fast-PPL eligibility | execution complete, overall PARTIAL: 110/110 original units terminal; each configuration has 10 PASS and one batch-invariance FAIL. Bounded diagnosis plus ten comparator-affected reruns demonstrate genuine BF16 batch-shape arithmetic sensitivity and preserve all ten FAIL decisions under unchanged thresholds. Diagnostic root `eedee1596bb36692da8557fc001c45593f50f930e62c27227a7151ab6aa88ba6` is cleanly retrieved. Fast-PPL eligibility remains 0/10; Fast/Full PPL and LongBench scoring NOT STARTED. |

## Required post-pilot milestone

| ID | Scope | Owner and schedule | Required evidence | Status |
|---|---|---|---|---|
| M14-GRAPH-AB | Phase 14 CUDA Graph A/B mechanism experiment | E04 harness owner; execute after E13 pilot admission and before E16 | same method/cache/backend/shape with only Graph mode changed; output/cache identity; floor, slope, knee, launch-gap proxy, and backend evidence | complete through Phase 14C: source campaign preserves 660/660 feasible runs and five eager `unstable` groups. Among 14 fully identifiable comparisons, 0 support the complete launch-floor-only criterion; 4 are inconclusive from eager instability and 2 have no positive eager slope. The negative heterogeneous-effect result makes Phase 15 READY without rerunning timing; seven-object closure root `4cd29ea1b94201f493db8cef9ebd01933b4c4f573185eff317ec5af81e9fb000` is cleanly retrieved. |

M14-GRAPH-AB is a named milestone, not a renumbering of the contract's E00-E18
task list. E16 is now complete under the separately authorized Phase 16R contract.

## Cross-cutting subtasks

- Phase 2 implements the local portion of docs/artifact_policy.md. Phase 6A
  resolves B-009 with verified private, indefinite-locked Cloudflare R2 plus
  clean synthetic and container-G0 retrieval. Every future publication still
  repeats the management and clean-retrieval checks.
- E01 is complete for exact Decision 0016 Docker image ID / OCI image-index
  digest
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
  Tags are non-authoritative; a changed digest requires full recertification.
  Native-host admission remains non-claim-bearing.
- E03 includes growing-context request validation but does not mix those
  samples with fixed-L fitting.
- E04 owns the capture/replay harness used by M14-GRAPH-AB; that milestone runs
  only after pilot admission and keeps method/cache/backend/shape fixed.
- Phase 4 adds only the BF16 adapter/factory and reusable admission facades.
  Phase 5 adds only the TurboQuant reference lane. Phase 6A remediation is
  PASS. Phase 6 has a minimal implementation; current-HEAD sanitizer passed
  3/3, the frozen bounded grid passed 9/9, durable publication and clean
  retrieval passed, and G2-TQ is PASS. Decision 0018 authorizes exactly one
  checksum-bound GQA patch on the official KIVI commit after no newer official
  revision was found. E07 is complete under that patched-source authority:
  the locked SM120 reference, four fixtures, rollover/byte/GQA evidence,
  sanitizer, trace, publication, and clean retrieval pass. E08 is now complete:
  one static KIVI adapter passes all 17 strict checks, the ten-point grid, and
  inner plus report-bearing outer durable publication. G2-KIVI is PASS.
  E09 is complete under Decision 0021 with one immutable, durably retrieved
  calibration root. E10 is complete with the immutable Phase 10 numerical
  fixture root. E11 is complete under Decision 0027: the static Adapter binds
  caller-owned deterministic q4 decode workspace, all corrected fixtures and
  nine bounded points pass, and the checksum-bound admission evidence is
  durably retrieved. G2-KVQ is PASS. Decision 0029's successor current-source
  re-admission preserves that history and the unchanged Adapter/cache/session;
  all corrected fixtures, nine fresh bounded points, and required audits pass,
  and successor inner root
  `8ea533b9544e99140aec04b4cb9b1ad26f271273206d170e7abefa195c0581aa`
  is durably retrieved. E12 is complete: its exact 30-run common-point campaign
  passes G0-G5 and its 391-object root is COMPLETE-last and cleanly retrieved.
  E13's three stopped campaigns remain immutable and durably retrieved.
  Phase 13B separately admits B=1/4/8 static geometry for all nine compressed
  configurations. Phase 13F corrects the 810-record end-to-end peak-memory
  feasibility contract under Decision 0031. Phase 13R2 confirms that matrix
  but exposes B-020. Decision 0032 and Phase 13T prove normal progress and bind
  finite stage-aware supervision without changing timing or grid semantics.
  The remediation root is cleanly retrieved. The wholly new successor Pilot
  then completed all 684 feasible runs with no failure or selective rerun and
  published root
  `feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531`.
  Phase 13D completes the separately preregistered densification with 252/252
  logical records and resolves all 25 targets as one `density_sufficient` and
  24 `insufficient_feasible_span`. Phase 14C closes the frozen A/B campaign as
  a valid heterogeneous negative mechanism result while retaining five eager
  unstable groups. Phase 15 then completes 32 Nsys and 22 NCU selected
  profiles, all-ten common-point physical traffic, durable publication, and
  clean retrieval. Phase 16R subsequently completed the separately authorized
  Full Scan with five independently published and verified segments. Phase 17
  is READY but not started; quality execution remains LOCKED.
- E12 includes an operator-level MHA control with identical head dimension and
  no GQA repetition.
- B-011 through B-017 are resolved for native-host BF16 G1. The reporting-only
  repair reused the same immutable campaigns and did not rerun any point.
  Preserve every failed and passing run/report.
- E13 and E16 use blocked randomization and retain every failed, unstable, and
  capacity-infeasible point with a machine-readable reason.
- E14/E15 must set run_kind to nsys or ncu; only run_kind=timing enters latency
  analysis.
- Every issue that changes experimental semantics must link a decision record,
  and the semantics change must not share a PR with method implementation.
