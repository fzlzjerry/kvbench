# Project status

Last updated: 2026-09-19
Authoritative contracts: CODEX_WORKFLOW.md for active performance engineering;
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md for post-performance quality
scheduling; CODEX_QUALITY_EVALUATION_ADDENDUM.md for non-conflicting quality
requirements; and AGENTS.md. Decision 0005 records precedence.

## Current state

- Latest scoped task: Q0 batch diagnosis `COMPLETE`, while the approved Q0
  batch gate remains `FAILED`. A bounded three-sample BF16 reproducer rules out
  input/row/state/output-ownership errors and localizes the first cross-batch
  difference to layer-0 Q/V projection output under identical inputs. A small
  quality-only comparator repair now applies the frozen tolerance relative to
  the B=1 reference explicitly; all ten affected batch units were rerun and
  remain FAIL with unchanged aggregate maxima and selected-token counts. The
  other 100 Q0 units, all 68 locked hot paths, and all performance evidence are
  unchanged. Compact root
  `eedee1596bb36692da8557fc001c45593f50f930e62c27227a7151ab6aa88ba6`
  is COMPLETE-last and cleanly retrieved. Fast-PPL eligibility remains 0/10;
  Fast/Full PPL and LongBench scoring remain not started.
- Latest scoped task: Q0 cache-sensitive correctness `PARTIAL`. Operator
  approval is bound separately to contract
  `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`, SHA-256
  `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`.
  All 110 required units completed. Every configuration passes its seven core
  probes, suffix diagnostic, eager/Graph invariance, and applicable cache-
  dependence control, but all ten fail the frozen B=1 versus B={4,8} batch-
  invariance gate. Fast PPL eligibility is therefore 0/10 and Fast/Full PPL
  plus LongBench scoring remain not started. The 356-object root
  `2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0`
  is COMPLETE-last and passed one clean R2 retrieval. Performance evidence and
  all 68 locked hot paths remain unchanged; no configuration is quality-pass.
- Latest scoped task: QP-1 quality-contract preparation `PASS`. All six
  technical pending items are complete under contract
  `quality-qp1-20260918t170812117127z-170b638c-e8f4a2`, SHA-256
  `b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1`.
  The 62-object root
  `6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62`
  is COMPLETE-last and passed one clean R2 retrieval. The separate approval
  receipt now authorizes Q0 only; Q0 completed PARTIAL as recorded above, while
  PPL and LongBench benchmark scoring remain not started.
- QP-0 performance freeze remains `PASS`, unchanged. The compact
  2,670-slot release inventory, 68-path hot-path lock, external-kernel
  identities, Quality derivative image, and pending manifest-bound quality
  contract are complete. The 14-object root
  `9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f`
  is COMPLETE-last and passed one clean R2 retrieval. The original local tag
  remains immutable historical evidence; corrected tag
  `perf-freeze-20260917-83536c37-r1` targets the same performance source and
  binds the ledger-verified finalized manifest SHA. `PERFORMANCE_DATA_FROZEN`
  records performance-freeze completion only. The nine-object correction root
  `44c9594079e372d708d509449589a428f4b27c0392e2f1b1e491acc3d17e2559`
  is COMPLETE-last and passed one clean R2 retrieval. Quality remains LOCKED;
  QP-1 used this binding without recreating either tag or freeze bundle.
- Latest scoped phase: Phase 18 CPU Reproduction Package `PASS`. The compact
  package preserves Phase 17 model D, all missed target states, 23/50
  identified knees, and the outer-selection limitation. Its reporting audit
  labels the two primary aggregates as equal-cell macro means, not pooled
  quantiles, and retains the BF16 B=1 short-context edge-extrapolation tail.
  A pure-standard-library fresh-directory reproduction regenerated six figures
  and four predictor examples without GPU or network work. The 54-object root
  `cd6ee2324d0163088102e29e45b4ad124b5d524fb1900717faddf997da6805e2`
  is COMPLETE-last and passed one clean R2 retrieval. Quality remains LOCKED,
  `PERFORMANCE_DATA_FROZEN` remains absent, and QP-0 was not started.
- Latest scoped phase: Phase 17 Modeling `PASS`. CPU-only analysis used the
  immutable Phase 16R host-wall closure: 2205 accepted process observations,
  441 feasible logical points, and 465 explicit capacity-infeasible process
  records. All 38 infrastructure replacements contribute exactly one accepted
  observation per logical replicate slot. The frozen E/RQ2/D/F candidates were
  evaluated under 53 grouped folds with no fit failure. Model D was selected,
  but the median-error, P95-error, same-work sign, and pairwise-ranking targets
  were honestly missed; knee error is not evaluable without an independent
  reference. Twenty-three of 50 local curves have identifiable in-range knees;
  the remaining 27 retain explicit non-identifiable statuses. The 44-object
  root `05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb`
  is COMPLETE-last and passed one clean R2 retrieval. Quality remains LOCKED,
  `PERFORMANCE_DATA_FROZEN` remains absent; Phase 18 packaged this evidence
  without changing the scientific outcome.
- Latest scoped phase: Phase 16R Full Scan `PASS`. Family
  `phase16-20260831t123029614620z-ec534d99-de80ac` preserves all 2670
  terminal records: 2205 completed and 465 capacity-infeasible, with 38
  explicitly linked infrastructure replacements and zero selective reruns.
  All 441 feasible logical points are host-wall stable; maximum CV is
  1.305276915%. Five segment roots, the original CUDA-event outer analysis,
  and the append-only primary host-wall closure are COMPLETE-last and cleanly
  retrieved. Primary host-wall root:
  `5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e`.
  See `docs/phase_reports/phase16r-full-scan.md` and
  `docs/evidence/phase16r/full-scan-publication.json`.
  Full Scan is COMPLETE; G0-G5 remain PASS; Phase 17 used this immutable input;
  Quality remains LOCKED and `PERFORMANCE_DATA_FROZEN` absent. Historical
  prefix catalogs and stopped campaigns remain unchanged. No reboot occurred.
- Historical Phase 15 Profiler Subset `PASS`. Campaign
  `phase15-20260828t144810363697z-446b334e-90460f` completed 32/32 selected
  Nsight Systems profiles and 22/22 selected Nsight Compute profiles with zero
  selected profiler failures. The common B=1/L=131072 Graph point covers all
  ten configurations and populates measured cache-path `r_hbm` for 10/10.
  Across 16 anchor eager/Graph pairs, Graph reduces CPU CUDA submission-call
  count and GPU inter-kernel idle in 16/16 while preserving kernel count,
  ordering, and observed overlap. Combined with Phase 14, the supported
  mechanism classification is `method_specific_mixed`, not a pure
  launch-floor-only effect. Profiler durations remain excluded from normal
  timing and fit data. The 893-object root
  `641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8`
  is COMPLETE-last and passed one clean R2 retrieval. Phase 16 is `READY` but
  not started; G0-G5 remain PASS, Full Scan remains CLOSED, quality remains
  LOCKED, and `PERFORMANCE_DATA_FROZEN` remains absent.
- Latest scoped phase: Phase 14C analysis closure `PASS`. The immutable Phase
  14 campaign `phase14-20260826t115110887808z-47ba4220-42fc95` preserves all
  720 planned mode records: 660 feasible runs completed, 60 records belong to
  30 predeclared capacity-infeasible pairs, and there were no runtime failures
  or selective reruns. Five eager mode points remain `unstable` under the
  unchanged 3% CV threshold and are not used for stable fitted mechanism
  comparisons. The other 105 A/B condition groups are stable with zero output,
  backend, cache-identity, kernel-path, or allocation mismatch. Fourteen
  floor+slope+knee comparisons are fully identifiable; 0 of 14 satisfy the
  complete launch-floor-only criterion, four comparisons are inconclusive
  because eager data are unstable, and two have no positive eager slope. The
  host-minus-device proxy is lower under Graph in only 23 of 105 stable pairs.
  The resulting negative mechanism conclusion is that the observed Graph
  effect is heterogeneous and is not explained by a pure launch-floor
  reduction. The 18,628-object source root
  `22a613b07c1ee6d3e9a0a7fc81df6065ccc8a10bf783b1a69aded3c2eb8068f0`
  and original BLOCKED report remain immutable. The separate seven-object
  closure root
  `4cd29ea1b94201f493db8cef9ebd01933b4c4f573185eff317ec5af81e9fb000`
  is COMPLETE-last and passed one clean R2 retrieval without reuploading source
  objects. Phase 15 is `READY`; G0-G5 remain PASS from unified admission, Full
  Scan remains CLOSED, quality remains LOCKED, and
  `PERFORMANCE_DATA_FROZEN` remains absent.
- Latest scoped phase: Phase 13D knee densification continuation PASS. The successful
  Phase 13 successor Pilot remains PASS. Campaign
  `phase13-20260822t150835736582z-4ddd7b17-3a8fb3` preserves all 810 planned
  records: 684/684 feasible runs completed, 126 were predeclared
  capacity-infeasible, no run failed or was selectively rerun, and all 228
  evaluated points are stable with maximum three-process CV
  `0.002218507701406616`. Output checksums, kernel paths, allocations, and
  finite-output controls agree. All 30 provisional fit records have status
  `knee_observed`, but only 5/30 have sufficient below/near/above density;
  Phase 13D preregistered 25 targets and 84 new contexts. A post-measurement
  snapshot finalization failure preserved 51 valid runs and one failed record;
  the minimal continuation fix now persists every raw snapshot before assigning
  exactly `clean`, `foreign_process_detected`, or `query_failed`. Segment A's
  51 runs were not changed or rerun, Segment B completed the replacement plus
  the frozen remaining 200 records, and the 84 prefix states were reused
  checksum-verified and read-only. The combined 252/252 records form 84 stable
  points with maximum CV `0.0019765939935609987`; all 25 targets resolve as one
  `density_sufficient` and 24 `insufficient_feasible_span`. The 7,381-object
  root `a8559a5e01edaad949df1e128c4bddff37638801cfc89f2eb8d4894c31ef82d2`
  is COMPLETE-last and passed one clean R2 retrieval. This evidence supplied
  the immutable entry authority for Phase 14. The successful Pilot's
  17,384-object root
  `feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531`
  is COMPLETE-last and passed one clean R2 retrieval. The blocked Pilot
  `phase13-20260804t111810342595z-a127b0d1-8649c3` remains immutable and none
  of its timing data was reused. Full Scan is CLOSED, quality is LOCKED, and
  PERFORMANCE_DATA_FROZEN is absent.
- Phase 6 status: PASS for method-specific G2-TQ at execution commit
  `0df5bb4d445d48e6cba17e30723733f8de35cb14`. The approved admission driver
  reran all three mandatory Compute Sanitizer probes against that clean HEAD
  before executing the unchanged frozen bounded grid; sanitizer passed 3/3
  and the grid passed 9/9. Final bundle
  `phase6-20260726t035257468z-0df5bb4d-4139a6-4bit_nc-fixed-l128-eager`
  contains the complete grid and remains admission-only evidence. Its
  167-object root
  `f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d`
  was published to the existing R2 content-addressed prefix with `COMPLETE`
  last and passed clean retrieval. The first publication invocation preserved
  a `TransportError`; an explicit rerun of the same conditional publisher
  verified already-present identical objects, created only missing objects,
  and completed without replacement. B-018 remains RESOLVED and all earlier
  failed evidence remains immutable.
  The append-only Phase 6 outer bundle preserves that root unchanged and adds
  all nine run references, the MethodAdmissionReport, original R2 publication
  record, final PASS report, inventory, ledger, and COMPLETE. Its 176-object
  root `8c4cf76f76bb17e648dfd911f11e268235ed827a9983814b774b9e95405496b0`
  is published at
  `r2://kvbench-artifacts/kvbench/sha256/8c4cf76f76bb17e648dfd911f11e268235ed827a9983814b774b9e95405496b0/`
  and passed independent clean retrieval; the original 167-object root remains
  unchanged.
- Phase 6A prerequisite status: PASS. B-010 is RESOLVED for the exact Decision
  0016 image after full image identity/layer verification, container G0, and
  separate BF16 eager and CUDA Graph parity runs.
- The synthetic R2 artifact
  `phase6a-r2-synthetic-20260724t135642z`, root SHA-256
  `bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e`,
  remains checksum-valid after a new clean retrieval. Cloudflare REST confirms
  bucket `kvbench-artifacts` is private and covered by enabled indefinite rule
  `kvbench-evidence-indefinite` at exact prefix `kvbench/sha256/`. Container-G0
  root `85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`
  was published COMPLETE-last and cleanly retrieved. B-009 is RESOLVED.
- Phase 0 status: PASS
- Phase 1 remediation status: PASS
- Phase 7 status: PASS under exact Decision 0018 patched-source authority.
  The isolated reference image, official extension, native SM120, PTX/JIT,
  zero-error sanitizer, four deterministic fixtures, rollover, actual bytes,
  native eight-head GQA, and non-performance trace pass. Final 30-object R2
  root `abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302`
  is COMPLETE-last and cleanly retrieved. At that phase boundary, Phase 8, Pilot,
  and all later execution remained unauthorized.
  KIVI Measurement Adapter remains fail-closed at that historical boundary.
- Phase 8 status: PASS for method-specific G2-KIVI at execution commit
  `462325e9df809d3bcf24a06361bf004bc7383d73`. The frozen ten-point grid
  passed 10/10, all 17 strictly derived admission checks pass, and the inner
  331-object root
  `f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515`
  is COMPLETE-last and cleanly retrieved. Report-bearing outer run
  `phase8-r2-outer-20260727t123744540656675z-7ff9f36`, built at
  `7ff9f3602c39eb910e13b1b2ca9be9b9a8bde142`, preserves that inner root,
  the MethodAdmissionReport, inner publication receipt, final Phase 8 report,
  inventory, ledger, and COMPLETE. Its 341-object root
  `de7d41f151af9fe1e716f27ae0f1fc24d2ef0a4b16e8e5c3ecf45d5f9983e132`
  was conditionally published COMPLETE-last and passed independent clean
  retrieval. G2-KIVI is method-specific only.
- Phase 9 status: PASS for offline KVQuant calibration under Decision 0021's
  exact patched-upstream authority. Final calibration
  `kvqcal-cdb724c806d64d095c040d2673a987a3` contains the frozen WikiText-2
  train tokens, all-layer K/V Fisher artifacts, `kvq4`/`kvq3`/`kvq2`, shared
  sink/cap policies, 192-row layer statistics, and reproducibility evidence.
  Its 68-object root
  `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`
  was published COMPLETE-last to the existing content-addressed R2 namespace
  and passed clean retrieval. G2-KVQ remains NOT EVALUATED.
- Phase 10 status: PASS for the KVQuant numerical Reference Lane under
  Decisions 0021 and 0023. The source-faithful matrix contains exactly nine
  deterministic fixtures: `kvq4`/`kvq3`/`kvq2` crossed with Key sparse counts
  0/6/12, while every non-sink Value row retains the frozen fixed-extrema count
  12 and every sink Value row has count 0. Native SM120, forced PTX/JIT,
  zero-error and zero-leak Compute Sanitizer coverage, native 32Q/8KV GQA,
  store/append/decode controls, exact bytes, deterministic regeneration, and
  clean validation pass. Final fixture ID
  `kvqref-a50af6511c314b6394e58a7f81ceefb8`, 113-object root
  `32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab`,
  was published COMPLETE-last and passed clean R2 retrieval. The KVQuant
  Measurement Adapter remained unimplemented and fail-closed at that phase
  boundary; G2-KVQ remained NOT EVALUATED.
- Phase 11R status: PASS. The static `kvq4`/`kvq3`/`kvq2` Adapter binds
  Decision 0027's deterministic q4 Value-decode API and caller-owned FP32
  workspace. All nine corrected fixtures, the nine-point bounded admission,
  exact path/byte/allocation/GQA controls, fixed-L Graph, and Compute
  Sanitizer pass. The 165-object admission root
  `0834410509ea7324a41715e0e84e09617bf9b188b10394a234f9a57e804dd1f2`
  is COMPLETE-last and cleanly retrieved; MethodAdmissionReport SHA-256 is
  `59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a`.
  The earlier BLOCKED report, failed 100-object bundle, and diagnostic remain
  immutable. G2-KVQ is PASS.
- Phase 11R-Q23 status: PASS for current Decision 0029 execution source
  `kvquant_gqa_longctx_deterministic_q23_v4`, commit/tree
  `34b0bdfa83082e1f30387d9ac5cca369006e089c` /
  `1f85af65fe03061583ffe8bd91e47d7ecffdd312`, aggregate patch SHA-256
  `7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a`,
  and extension SHA-256
  `b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d`.
  All nine corrected fixtures and the fresh bounded grid pass 9/9 together
  with path, byte, allocation, GQA, Graph, and sanitizer controls. Successor
  MethodAdmissionReport SHA-256 is
  `9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2`;
  its bound 212-object inner admission root
  `8ea533b9544e99140aec04b4cb9b1ad26f271273206d170e7abefa195c0581aa`
  is COMPLETE-last and cleanly retrieved. The historical Decision 0027
  MethodAdmissionReport remains unchanged. G2-KVQ remains PASS. At the
  Phase 11R-Q23 boundary, Global G2-G5 were NOT EVALUATED.
- Active admission gate: G0 PASS; G1 PASS; G2 PASS; G3 PASS; G4 PASS; G5 PASS;
  Phase 13 successor Pilot PASS; knee densification required before Phase 14;
  Full Scan CLOSED
- Benchmark implementation changes: exact BF16 static cache, fixed-L and
  growing-context runners, eager and CUDA Graph lanes, timing, allocation,
  telemetry, campaign lifecycle, and source-backed G1 reporting are
  implemented. Phase 4 adds only a thin method adapter, explicit BF16-only
  factory, shared audit facades, and a strict admission report schema. Phase 5
  adds only an isolated upstream TurboQuant reference lane and compact
  fixtures. Phase 6 adds one TurboQuant adapter and static cache through the
  same runners; it is admitted only at method-specific G2-TQ. Phase 8 adds one
  static KIVI adapter and one KIVI-specific cache state through those same
  runners; it is admitted only at method-specific G2-KIVI. Phase 9 adds only
  one isolated offline calibration container and narrow KVQuant calibration
  scripts. Phase 10 adds one isolated KVQuant reference runner and compact
  numerical fixtures only. Phase 11R admits one static KVQuant Adapter through
  the same common runners after binding Decision 0027's deterministic q4
  decode API. Phase 11R-Q23 re-admits that unchanged Adapter against Decision
  0029's current execution source through evidence binding only; no new
  framework or runner was introduced.
- CUDA builds or executions: the new formal E00 run passed extension build,
  native execution, forced PTX/JIT, numerical golden, CUDA Graph, allocation,
  SASS/PTX inspection, and all required Compute Sanitizer lanes
- Benchmark, performance-profiler, or quality data produced: all earlier Phase 3
  runs, campaigns, and reports remain immutable. The B-015 execution preserved
  one 16-run fixed-L campaign with 13 completed and 3 graph aborts; B-016 added
  only an untimed diagnostic. Execution SHA
  `9def265ab613cde7a06b0e51850f066d0564d635` then preserved two complete new
  campaigns with 20/20 completed runs and immutable FAIL report
  `phase3-g1-20260723t123322160580z-9def265a-08dc69`. Reporting-only commit
  `7f72c95f9932c608f9bd68f1971d6e86378596a2` then published immutable PASS
  report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` from those same
  source runs without executing timing. Phase 4 produced exactly three
  checksum-bound functional smoke records at B=1/L=128; they contain no
  latency, independent timing replicates, or formal performance data. No
  profiler campaign or quality evidence was produced. Phase 5 added only
  deterministic reference tensors and kernel-name traces; all profiler
  durations were discarded and no formal timing sample was created. Phase 6A
  added only untimed container certification and parity artifacts. Phase 6
  added only correctness/audit/sanitizer and bounded-admission evidence; the
  frozen grid completed 9/9 without opening Pilot or Full Scan. Phase 8 added
  only correctness, fixture, rollover, allocation, execution-path, sanitizer,
  Graph, and bounded-admission evidence; its grid completed 10/10. No formal
  performance sample, Nsight result, or quality result was created. Phase 9
  added only offline calibration inputs, Fisher tensors, quantizers, policy
  evidence, and publication receipts. It did not run or record benchmark
  timing, HBM traffic, capacity, Nsight, PPL, LongBench, or other quality
  results. Phase 10 added only source-authoritative fixtures, byte records,
  duration-free reference traces, CUDA correctness/sanitizer evidence, and
  publication receipts. It did not create performance, profiler, HBM,
  capacity, or quality data. Phase 11R added only correctness, allocation,
  Graph, sanitizer, and nine bounded-admission points. The earlier L=4096
  failure and untimed determinism control remain non-claim evidence; no
  speedup, comparative latency, profiler, HBM, capacity, or quality result was
  produced. Phase 11R-Q23 adds only current-source correctness, allocation,
  Graph, sanitizer, and bounded-admission evidence; it produces no speedup,
  comparative-latency, profiler, HBM, capacity, or quality result. Phase 12R
  adds only the preregistered common-point three-process reproducibility
  evidence and unified gate aggregation. It calculates no speedup and makes no
  comparative performance, HBM, knee, capacity, or quality claim.
- Scientific performance claims: none
- Quality protocol: preregistered by Decision 0005 before any performance or
  quality result
- Quality execution: LOCKED; `PERFORMANCE_DATA_FROZEN` is absent
- Quality runs or quality-only dependency installations: none
- Full-scan admission: CLOSED
- Gate state: G0 PASS; G1 PASS; G2 PASS; G3 PASS; G4 PASS; G5 PASS; Pilot
  READY but not started; Full Scan CLOSED

## Phase 7 KIVI reference lane

Phase 7 passed its repository and R2 entry checks at clean HEAD
`0974bbc98f8f941b09800786591108292dc4e0dd`. The official author repository
`https://github.com/jy-yuan/KIVI.git` is source-audit pinned at commit
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`, tree
`c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b`, under MIT. Exact relevant Git
blob and SHA-256 identities are recorded in `third_party/LOCK.json`.

Decision 0017 records that the official repository exposes incompatible
`main`, `develop`, and `lmeval` heads and selects `main` because it is the only
audited official head advertising the required Llama 3/GQA scope. That path
calls Transformers 4.43.1 `repeat_kv` for recent K/V. The dependency's
`expand(...).reshape(...)` produces a distinct contiguous H_Q=32 tensor from
H_KV=8 storage; the exact BF16 audit observed four times the eight-head storage.
This violated the mandatory native eight-head GQA acceptance criterion and
created B-019. The original attempt stopped before environment, CUDA, fixture,
trace, byte-layout, sanitizer, Graph, or R2 work; its evidence remains immutable.

The B-019 remediation restarted entry at clean commit
`755c1bdb87af3e7becda792bd5d300ab877fee7e`. A fresh remote-ref audit found no
new author-maintained revision. Decision 0018 therefore authorizes exactly one
checksum-bound project patch on the same official base commit. Patch, manifest,
patched files, and resulting tree
`b617493dea5aff1a754cd27ad6be12ac512b2aee` are bound in the source lock.

CPU and SM120 BF16 checks at contexts 17 and 33 match the original repeat
formula exactly. Both residual BMMs operate at `batch * H_KV=8`, preserve the
explicit `query_head // 4` mapping, and add no prohibited expansion operation.
B-019 is RESOLVED under patched-source authority. The continuation from clean
commit `3417ea0e7f322369eed21bb787a9a9a19b0a69bd` completes Phase 7: the locked
reference image, unchanged official extension, native SM120 and PTX/JIT,
sanitizer, four fixtures, rollover, byte ownership, native eight-head GQA,
duration-free trace, durable publication, and clean retrieval all pass. The
final 30-object root is
`abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302`.
At Phase 7 completion, Phase 8 remained unstarted and the KIVI Measurement
Adapter remained fail-closed. The subsequent Phase 8 result is recorded below.

## Phase 8 KIVI measurement adapter

Phase 8 entered from clean commit
`8d6d766a34a15bd40bd42cc47c5482b0dd052cc0` and executed only inside the
unchanged Decision 0016 Measurement Container. The implementation uses one
static KIVI cache state, one adapter, the existing fixed-L and growing-context
runners, and the exact checksum-bound patched official source, extension, and
four Phase 7 fixture configurations. Decision 0019 corrects only the legacy
Phase 7 allocation-ratio name; historical bytes and fixtures remain unchanged.

All four configurations conform, rollover at L=31/32/33/34 is exact, native
H_KV=8 GQA remains materialization-free, eager allocation is fully attributed,
fixed-L Graph replay has zero allocation, and the minimal two-bit/four-bit
sanitizer matrix reports zero errors and zero leaks. The bounded grid passed
10/10 without calculating speedup.

Strict MethodAdmissionReport
`docs/evidence/phase8/kivi-method-admission.json`, SHA-256
`3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a`,
derives 17/17 PASS checks from the immutable inner bundle and its durable
receipt. The inner 331-object root is
`f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515`.
The report-bearing 341-object outer root is
`de7d41f151af9fe1e716f27ae0f1fc24d2ef0a4b16e8e5c3ecf45d5f9983e132`;
its external receipt SHA-256 is
`9e9d8a650c0c1ed35eb4ecad32a34ede75cf45d4953e0baa32d2c0d561476db4`.
Both roots are COMPLETE-last and pass clean retrieval under exact indefinite
Bucket Lock rule `kvbench-evidence-indefinite`.

G2-KIVI is PASS. Global G2-G5 remain NOT EVALUATED, Full Scan remains CLOSED,
quality execution remains LOCKED, and `PERFORMANCE_DATA_FROZEN` remains
absent. At that Phase 8 boundary, Phase 9 had not started; the later Phase 9
result is recorded below. This admission makes no speedup, physical-HBM, knee,
capacity, performance, or quality claim.

## Phase 9 KVQuant calibration

Phase 9 entered at clean synchronized commit
`b4d253724717076188a38032d6d6204fdf15e191`. Decision 0021 binds method
identifier `kvquant_gqa_upstream_patch_v1` to pinned upstream commit
`57a238357f0ffe50084670fcd5781c9848f80ea2`, project patch SHA-256
`db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`,
and patched tree `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`. Reconstruction and all
recorded before/after file hashes pass. This is project-patched upstream
authority, not a claim that the upstream authors released native Llama-3.1
GQA support.

The separate digest-pinned calibration image loaded the exact
`meta-llama/Llama-3.1-8B-Instruct` model and tokenizer revision
`0e9e39f249a16976918f6564b8830bc894c89659` in native 32Q/8KV geometry.
WikiText-2 train data was frozen into exactly 16 ordered sequences of 2,048
tokens at seed `20260721`; the test split was not loaded. One full Fisher run
produced 32 pre-RoPE K and 32 V FP32 finite tensors. The same Fisher artifact
generated exactly the three safe-format quantizer families `kvq4`, `kvq3`,
and `kvq2` with five sink tokens and shared Key/Value cap 12.

Final calibration `kvqcal-cdb724c806d64d095c040d2673a987a3`, executed from
source HEAD `37ffdac439ff29df8606c2f61f57157278f321ad`, has 68-object root
`8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`.
Token reconstruction, representative K/V Fisher replay, all-family
fresh-process quantizer regeneration, deterministic equal-value ties, fixed
capacity, dtypes, and zero fill pass. Tensor values regenerate exactly; safe
serialization is accepted by the pre-frozen numerical rule because JSON
header ordering is not canonical.

The first R2 publication attempt stopped before `COMPLETE` on a transport
error and left six identical content-addressed objects. The retry verified
those objects, uploaded the remaining 62, wrote `COMPLETE` last, and a clean
retrieval verified all 68 objects and the root under indefinite Bucket Lock
rule `kvbench-evidence-indefinite`. Two earlier failed local calibration IDs
remain immutable, finalized, and terminally failed.

Phase 9 changes no adapter or Measurement Container and creates no KVQuant
reference fixture. G2-KVQ and global G2-G5 remain NOT EVALUATED, Full Scan
remains CLOSED, quality remains LOCKED, and `PERFORMANCE_DATA_FROZEN` remains
absent. No performance, HBM, capacity, knee, speedup, or quality claim follows
from calibration.

## Phase 5 TurboQuant reference lane

The official vLLM repository is pinned at release `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`, tree
`3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`, under Apache-2.0. Exact Git
blobs and SHA-256 values bind the preset, cache dtype, store, decode, backend,
centroid, and upstream-test sources. The installed wheel runtime files match
the pinned source bytes. No floating branch or local TurboQuant rewrite is
used.

The isolated reference environment records Python 3.12.3, PyTorch
2.11.0+cu130, CUDA 13.0, Triton 3.6.0, vLLM 0.25.1, driver 595.71.05, and the
SM120 GPU. The alternative official vLLM image is pinned by linux/amd64 digest
`sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`.
This environment is not the Measurement Lane and did not itself close B-010.

With batch 1, 32 query heads, 8 KV heads, head dimension 128, 17 stored
tokens, one append, block size 16, seed 20260724, and BF16 inputs, the official
store/append/decode functions produced three mandatory fixtures and the
same-path optional held-out k8v4 fixture. Actual cache files agree with the
source-derived 134, 118, 102, and 196-byte slots. Kernel-name traces identify
the official MSE/FP8 store and split-KV decode kernels, with no observed
full-prefix dequantization, GQA materialization, or backend fallback in this
minimal path. Direct graph smoke is deferred to Phase 6; upstream declares
`AttentionCGSupport.UNIFORM_BATCH`.

`make reference-turboquant` first published the no-replace set, then a second
identical run returned `verified_existing` without replacing it.
`make validate-reference-turboquant` validates all manifests, 34 root checksum
entries, layouts, actual storage sizes, and claim boundaries. At Phase 5
completion, TurboQuant remained rejected by the Measurement Lane adapter
factory and G2-TQ was `NOT EVALUATED / READY`. The current Phase 6 state below
supersedes that historical entry state.

## Phase 6 retrospective entry-blocked record

Phase 6 was attempted from
`7bccb3217e257d2dbc72deefe8653e9f3556d4f2` and stopped BLOCKED at entry.
Method-specific G2-TQ was BLOCKED because B-009 and B-010 were unresolved;
global G2-G5 were NOT EVALUATED. Provisional plan commit
`1f8e29a8da97e3ad56567c319ec817bec91593be` was completely reverted by
`a9cb4833bfba15a01426bf314c31add7e1c1c698`. No TurboQuant Measurement
Adapter implementation, formal run, performance data, profiler data, or
quality data was retained or created.

The retrospective governance record is
`docs/phase_reports/phase6-turboquant-measurement-blocked.md`. It was created
after the complete revert and is not CUDA, correctness, performance, profiler,
quality, or method-admission evidence. Native-host G0 and native-host BF16 G1
remain PASS, Full Scan remains
CLOSED, quality execution remains LOCKED, and `PERFORMANCE_DATA_FROZEN`
remains absent.

## Phase 6A initial blocked prerequisite attempt

Phase 6A added one digest-pinned Measurement Container definition, an explicit
container mode for the existing E00 implementation, and one Cloudflare
R2-specific publisher/verifier. Unit and repository validation passed, but the
execution host had no Docker/OCI runtime or NVIDIA Container Toolkit. No
Docker image ID or OCI image-index digest, container G0 run, BF16 eager/graph
parity run, or execution authority existed in that attempt.

The bounded synthetic R2 object-path acceptance test passed conditional
creation, exact-existing verification, conflicting-byte rejection, and clean
retrieval at the content-addressed root
`bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e`.
The read-only Cloudflare REST certification returned HTTP 403, so the bucket's
public state and active Bucket Lock rule remain NOT VERIFIED. The required
container-G0 bundle was unavailable for the second publication test. The
historical report is
`docs/phase_reports/phase6a-measurement-container-and-r2-blocked.md`.
At that attempt, B-009 and B-010 remained OPEN.

## Phase 6A remediation

The existing implementation was reused. Docker 29.6.1 and NVIDIA Container
Toolkit 1.19.1 built and verified the exact linux/amd64 Docker image ID / OCI
image-index digest
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
Its saved-layer scan found no model weights, operator environment files, or
configured credential bytes. Exact dpkg, Python, CUDA, compiler, profiler-tool,
and executable-hash identities validate against the reviewed container lock.

Container run `e00-20260724T195014.679255Z-a6025ae023e1-23dbe853` passed all
17 G0 checks, including native SASS, forced PTX/JIT, all four Compute Sanitizer
lanes, `sm_120`/`compute_120` inspection, Graph capture/replay, allocation, and
GPU exclusivity. Separate eager and CUDA Graph BF16 B=1/L=128 parity runs pass
with the frozen model, adapter, Flash backend, 32/8 GQA geometry, numerical,
non-materialization, eager-allocation, and zero graph-replay-allocation
controls. They are untimed, non-claim parity artifacts.

Read-only Cloudflare management verification confirms bucket
`kvbench-artifacts` is private and exact enabled rule
`kvbench-evidence-indefinite` covers `kvbench/sha256/` indefinitely. The
existing synthetic root cleanly reverified. Container-G0 root
`85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`
was conditionally published with COMPLETE last and cleanly retrieved with all
222 objects. Decision 0016 authorizes Measurement Lane CUDA only inside the
exact image digest. B-009 and B-010 are RESOLVED. The complete report is
`docs/phase_reports/phase6a-measurement-container-and-r2.md`.

At Phase 6A completion, Phase 6 had not yet been restarted and G2-TQ was
`NOT EVALUATED / READY`. The current Phase 6 state below supersedes that
historical entry state.

## Phase 4 common adapter

Clean implementation SHA `0cf160caa532c7cac23275c8a14fd8694789a86f`
places the existing BF16 static-cache and forced-Flash path behind the small
`KVCacheMethod` protocol. Fixed-L and growing-context runners now consume the
common session facade without changing timing boundaries or scientific
semantics. Quantized methods remain `phase_not_implemented`.

`make test`, `make test-cuda` (15/15), and `make test-graph` (4/4) passed.
The three bounded functional smokes passed with new run IDs and valid checksum
ledgers. The strict report at `docs/evidence/phase4/method-admission.json`
retains the historical native-host G0/G1 PASS, G2-G5 NOT EVALUATED, Full Scan
CLOSED, quality LOCKED, B-009/B-010 OPEN state at its publication, and
native-host non-claim status. No decision record was
needed because delegation was mechanical and changed no experiment semantics.

## Phase 3 remediation attempt

Clean execution SHA `7bd6dd48c1d88ac2b61684b02cc636f66b121054`
passed `make checks`, `make test`, `make test-cuda` (12/12), and
`make test-graph` (3/3). The prior 600-file Phase 3 evidence set remained
byte-identical before execution.

Fresh fixed-L campaign
`phase3-20260723t042422417332z-7bd6dd48-8a9cb6` preregistered and attempted
all 16 frozen points once. All 16 finalized `aborted`; no timing was retained.
Six operations completed raw B-011/B-012 audit, directly verified the
`pytorch_flash::flash_fwd_splitkv` GQA/MHA kernel family, found no
materialization/expanded-KV evidence, and passed the frozen eager or graph
allocation criterion. Those six then failed the retained-callable output
equivalence check. Seven runs recorded `owned_worker_failure`, and three
reproduced the registered-PID `[No data]`/missing-`pmon` race.

The campaign and all 16 runs independently validate, are read-only, and have
COMPLETE-last finalization. The stop condition prevented a growing-context
campaign and a new G1 report. No selective rerun occurred. See
`docs/phase_reports/phase3-remediation.md`.

## Phase 3 remediation execution 2

Clean execution SHA `eb908f6e372d6b232e6079e9344c2103bc90cdea` passed
`make checks`, `make test`, `make test-cuda` (13/13), and `make test-graph`
(3/3). Both prior Phase 3 evidence baselines remained byte-identical.

Fresh fixed-L campaign
`phase3-20260723t051939423712z-eb908f6e-b1039a` attempted all 16 frozen points
once and preserved 5 completed plus 11 aborted runs. Fresh growing campaign
`phase3-20260723t052647190745z-eb908f6e-5caf7f` attempted all 4 frozen points
once and preserved 4 aborts. No point was selectively rerun.

The five completed operations directly verified the
`pytorch_flash::flash_fwd_splitkv` GQA/MHA family with no materialization or
expanded-KV evidence. Two eager operations passed the source-backed 1,066-event
criterion and three graph operations retained strict zero allocation. All five
passed frozen numerical controls and exact audit/measured checksum equality.
The registered `compute_apps`/`pmon` race recurred eight times and correctly
joined to `owned_only`; foreign and PID-reuse controls remain fail-closed.
Thus B-013 and B-014 are resolved.

Fifteen workers aborted before measurement. Two preserve an explicit raw-audit
run hard-limit failure; thirteen preserve only the producer wrapper and omit
the lower-level cause. B-015 therefore keeps B-011/B-012 and G1 open.

Reporting-only SHA `3f2c365a5fd495cb3666b421e279b196b58dfb88` published
immutable report `phase3-g1-20260723t060636246041z-3f2c365a-26bf3c`, status
FAIL, SHA-256
`2bc0b4be6c1cc4a723b5b031e56b42520709de2d98cb35917bea857de70412c0`.
Independent validation passes with no errors and `COMPLETE` written last.
Quality remains locked, Full Scan remains closed, and Phase 4 did not begin.

## Phase 3 remediation execution 3 (B-015)

Untimed diagnostics at starting SHA
`8d64c673696ab3c8147310fa09b25217cac5104c` preserved the lower producer
exception, proved that the frozen Flash split heuristic selected 11 partitions
for GQA and 5 for the held-constant MHA geometry, and measured the exact
16-step worst-case raw bundle. Decision 0014 corrected only that disproven
cross-geometry equality and the source-backed raw transport envelope.

Clean execution SHA `52f41ce9d9be4edc07a833e00fe3404fbfa80b89`
passed `make checks`, `make test`, `make test-cuda` (13/13), and
`make test-graph` (3/3). The complete 1,978-file entry Phase 3 artifact baseline
and the original report SHA-256
`060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`
remained unchanged.

Fresh fixed-L campaign
`phase3-20260723t072710859854z-52f41ce9-88e3fd` preregistered and attempted all
16 frozen points once. It preserved 13 completed and 3 aborted runs, with no
unattempted point or selective rerun. All 13 completed operations independently
rederived `gqa_nonmaterialization_verified` for
`pytorch_flash::flash_fwd_splitkv`, found no replication/copy kernel or
expanded-KV allocation, passed frozen numerical/state/checksum controls, and
retained stable cache geometry. All 8 eager operations attributed 1,066 events
each with no forbidden/unknown event; all 5 graph operations had zero allocation
events and deltas.

The graph controls at `B1/L16384`, `B4/L4096`, and `B4/L16384` aborted before
measurement. Each immutable worker log preserves the same exact lower cause:
`ChromeTraceValidationError: graph GPU marker is not contained by its host
marker`. Each registered worker was correctly classified as an owned failure,
reaped with PID start-time protection, and never treated as foreign. This is
B-016; B-013 remains resolved.

Independent validation passed for the campaign and all 16 runs. The 17 new
directories contain 778 checksum-bound files, are read-only, and have
COMPLETE-last finalization; their aggregate manifest SHA-256 is
`7a584e456a253c4d583649a6c19ed538e6a8a1fb10e182ece3b5766467132dee`.
The stop condition prevented a growing campaign and new G1 report. G1 remains
FAIL; quality, Full Scan, pilot, and Phase 4 remained closed.

## Phase 3 B-016 remediation admission

At clean starting HEAD `7c4057c797230e21755812281bcfffe8e7319d5f`, an
untimed forced-Flash `B1/L16384` graph diagnostic reproduced the prior MHA
failure and preserved both raw traces plus the lower parser exception under
`/tmp/phase3-b016-7c4057c-b1-l16384-xbohqv1i`. The MHA GPU annotation extended
150.023 microseconds beyond host return, while the host still contained the
unique `cudaGraphLaunch` and the launch preceded the GPU range. Extending only
an in-memory copy of the host duration recovered the exact two correlated Flash
split-K graph nodes, proving host containment was the sole failed predicate.

Decision 0015 and commit `e7219e0dd714149e3eea783ce7a8602c4bf9bc54`
correct only that asynchronous parser boundary. The parser still requires the
host/GPU marker identity, unique launch, ordering, correlation, stream,
External-ID agreement, one graph ID, unique graph-node IDs, recognized Flash
kernel sequence, and no materialization activity. It now additionally rejects
launch-correlated device events outside the GPU marker and unknown device-like
categories across the union of host and GPU ranges.

The deterministic parser suite, pure replay of the preserved failing raw trace,
and actual long `B1/L16385` MHA graph control pass. `make checks`, `make test`,
`make test-cuda` (14/14), and `make test-graph` (3/3) pass, including strict
graph zero allocation, process ownership, and immutable-evidence checks. B-016
is resolved. B-011/B-012 and G1 remain open pending two entirely new complete
campaigns. No campaign, quality, Full Scan, pilot, or Phase 4 work ran during
this remediation admission.

## Phase 3 remediation execution 4 (complete post-B-016 campaigns)

Execution SHA `9def265ab613cde7a06b0e51850f066d0564d635`
passed `make checks`, `make test`, `make test-cuda` (14/14), and
`make test-graph` (3/3) before campaign admission. The complete 1,978-file
entry baseline, the B-015 778-file aggregate digest, and the original G1
report SHA-256 `060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`
remained unchanged.

Fresh fixed-L campaign
`phase3-20260723t112051327159z-9def265a-aa9c5e` preregistered and completed
all 16 frozen points. Fresh growing-context campaign
`phase3-20260723t121325332843z-9def265a-8fbf6a` preregistered and completed
all 4 frozen points. Each attempted exactly its expected process count, with
no failure, abort, capacity exclusion, unattempted point, or selective rerun.

All 20 source runs and both campaign records independently validate. The 22
campaign/run directories contain 1,368 checksum-ledger entries; every digest
matches, every directory is read-only, no unsafe link exists, and no file is
newer than its `COMPLETE` marker. Report generation revalidated the exact
source-run commitments without mutating them.

Campaign-side independent replay consumed all 80 checksum-bound raw audit
operation bundles rather than trusting worker verdict booleans. All 80 derive
`gqa_nonmaterialization_verified`; both held-constant controls identify the
`pytorch_flash::flash_fwd_splitkv` family and have no GQA failure reason. The
72 eager operations pass `phase3_eager_attributed_ephemeral_v1` with exactly
1,066 attributed events each and no allocation failure reason. All 8 graph
operations pass `phase3_graph_zero_allocation_v1` with zero events.

Append-only report `phase3-g1-20260723t123322160580z-9def265a-08dc69`
is valid, COMPLETE-last, immutable, source-checksum-valid, and SHA-256
`db044273f681bb66f5578c4c19327497302c903f1b4409a08b7b582a2d47ba07`.
It marks G1 FAIL on `no_torch_cat_growth`,
`no_unexplained_measured_region_allocation`, `gqa_not_materialized`,
`graph_replay_no_allocation`, and `no_backend_fallback`. The report derivation
does not consume the consolidated raw audit bundle and instead checks legacy
runtime summaries that are null in all 20 new runs.

This is B-017, a reporting-only raw-evidence join blocker. No campaign point
may be rerun to repair it. Quality execution remains LOCKED,
`PERFORMANCE_DATA_FROZEN` remains absent, Full Scan remains CLOSED, and pilot
and Phase 4 remain unstarted.

## Phase 3 B-017 reporting-only closure

Starting from clean HEAD `9c517ceeec1f9d0587be709166e62cdeca4d6831`, the
original report retained SHA-256
`060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`,
the B-017 FAIL report retained SHA-256
`db044273f681bb66f5578c4c19327497302c903f1b4409a08b7b582a2d47ba07`,
and every entry checksum remained unchanged.

Reporting-only commit `7f72c95f9932c608f9bd68f1971d6e86378596a2`
reuses the existing coordinator raw replay. It reconstructs the execution-SHA
source pin, binds each sidecar to its canonical index, binds every operation
key and declared raw file digest to the selected run, and derives report facts
from local raw replay rather than serialized worker `passed` booleans. The
legacy derivation remains generator-SHA-bound so every older immutable report
still validates under its original semantics. Targeted tests cover local
pass/worker-fail, local-fail/worker-pass, missing files, tampered files, and a
mismatched sidecar/index.

Before publication, `make checks`, `make test` (38 schema, 31 Phase 2, 226
Phase 3, and 167 remediation-control tests), `make test-cuda` (14/14), and
`make test-graph` (3/3) passed. The tree was clean at the report-generator SHA.
No campaign, performance timing, pilot, quality evaluation, Full Scan, or
Phase 4 work ran.

The append-only publisher reused exactly the complete fixed-L campaign
`phase3-20260723t112051327159z-9def265a-aa9c5e` and growing-context campaign
`phase3-20260723t121325332843z-9def265a-8fbf6a`. It created no new run and did
not modify any source run. New no-replace report
`phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` has SHA-256
`c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9`.
Independent validation returns `valid=true` with no errors; its ledger passes,
all 20 criteria are PASS, and no payload is newer than `COMPLETE`.

The resulting gate state at that report's publication was G0 PASS,
native-host BF16 G1 PASS, G2-G5 NOT EVALUATED, and Full Scan CLOSED. B-011,
B-012, and B-017 were resolved. At that point B-009 and B-010 still blocked
formal E02 closure, later method execution, ordinary timing, and every
performance claim. Quality execution remains LOCKED and
`PERFORMANCE_DATA_FROZEN` remains absent.

## Repository

The initial non-Git workspace contained three operator-provided inputs but no
implementation. The reviewed Phase 0 records are committed on branch main at
9569d938d9023a3e71d98f12234efa1897004533. The E00 collector and certification
tests are committed at 980eff7b6f5904c4828aa79d684c01a8dc45320d. Formal run
`e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d` remains immutable FAIL
evidence after `cuobjdump --dump-sass` could not find `nvdisasm`. The quality
protocol was preregistered at 6535a6f6a4e5caa53213e917e9fcf8fc9c0f0190,
and the exact `cuda-nvdisasm-13-0=13.0.85-1` package/tool identity was locked at
6442ba1f7554ea0ebf0b3bb1a920c94567cab689. New formal run
`e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` finalized as immutable PASS
evidence. At that point B-001, B-002, and B-004 were resolved while B-009 and
B-010 remained open. At that point, no remote was configured.

Phase 2 adds a dependency-free strict schema package, 11 versioned contract
templates, a fail-closed CLI, deterministic command reconstruction, and a
local append-only staging/finalization implementation. Its tests use temporary
roots only. The initial Phase 6A attempt later selected Cloudflare R2 and
exercised one synthetic content-addressed object path, but at that attempt
control-plane lock/public-state certification and the container-G0 publication
test remained incomplete, so B-009 remained open. No built and certified
digest-pinned Measurement Container or container-parity G0 existed then, so
B-010 remained open. The Phase 6A remediation recorded above subsequently
resolved both blockers.

Phase 3 execution used clean SHA
`457123b12220aa4a724968c1b4dd04340cf34a54`. The fixed-L campaign
`phase3-20260722t112917207390z-457123b1-36731e` attempted all 16 frozen
processes; the growing-context campaign
`phase3-20260722t113532869819z-457123b1-694228` attempted all four. Nineteen
runs finalized as `gqa_materialization_detected` because the exact operator
audit could not prove the required fused native-GQA kernel path; one fixed-L
eager run finalized `aborted` after a terminal process-query ambiguity. This
taxonomy is fail-closed: all 79 operator audits recorded no query-head-sized
KV temporary, so Phase 3 does not make a positive physical-materialization
claim. Eleven eager allocation audits also recorded allocator events. All
eight graph replay audits recorded zero allocation events and passed
eager/graph numerical agreement, but those facts cannot override G1.

Reporting-only descendant SHA
`ade0e86d2243ff193f684e008f99f35403dca293` produced immutable report
`phase3-g1-20260722t115413439499z-457123b1-e225cd`, status FAIL, SHA-256
`060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`.
Independent rederivation and repository governance validation pass. At that
immutable report publication, B-009 through B-013 remained open.

At Phase 0 start, the only top-level inputs were AGENTS.md,
CODEX_WORKFLOW.md, and Archive.zip. No implementation, model config, CUDA
extension, Dockerfile, build system, tests, artifact directory, or prior result
was present.

## Inputs

- Archive: /home/rockrock/cmu_paper/Archive.zip
- SHA-256: 20e5b6be5c3060012c48446d1b51067996cd4f13df1d6a73ee8eeb8f855e3ab1
- Contents: 23 PDFs plus 23 AppleDouble metadata files
- Extracted bytes: 123,745,542
- Extraction destination: literature/raw/
- Raw-tree writable entries: zero
- Source archive writable: no
- Checksum records: 47, covering the archive and all 46 extracted files
- Manifest records: 47 data rows
- Archive code executed: none

Static PDF checks found no encryption, declared JavaScript, or embedded files.
qpdf is unavailable; this residual defense-in-depth gap is non-gating and is
recorded in B-008/R-014.

## Source pins and commit plans for later validation

| Source | Exact revision | Phase 0 role |
|---|---|---|
| vLLM v0.25.1 | 752a3a504485790a2e8491cacbb35c137339ad34 | TurboQuant source/reference candidate |
| KIVI | 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6 + Decision 0018 patch | patched official-source authority; B-019 resolved; Phase 7 reference PASS; no paper-era equivalence claim |
| KVQuant | 57a238357f0ffe50084670fcd5781c9848f80ea2 | official-paper calibration/reference candidate |
| lm-evaluation-harness | c9bbec6e7de418b9082379da82797522eb173054 | direct KIVI Reference Lane dependency |

These are exact source pins, not admission decisions. KVQuant's three embedded
Transformers-derived trees are fixed by outer-commit tree hashes, while exact
upstream lineage remains unresolved; LOCK.json also assigns commit-resolution
plans to the GPTQ, GPTQ-for-LLaMA, and SqueezeLLM attributions. During the
Phase 0 audit, no upstream setup, binary, macro, kernel, or benchmark was
executed. Its temporary source snapshots were used only for read-only
inspection and are outside the repository. Phase 5's later bounded vLLM
reference execution is recorded separately above.

## Phase and gate ledger

| Phase/gate | Status | Evidence |
|---|---|---|
| Phase 0 repository/input audit | PASS | literature manifests; method notes; source lock; decision, risks, blockers, tasks |
| G0 native-host hardware certification | PASS | `docs/evidence/e00/e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32/`; prior immutable FAIL retained |
| Phase 2 repository/contracts/tooling | PASS | strict schemas and examples; fail-closed CLI; append-only local writer; 54 Phase 2 tests; repository checks |
| G1 BF16 baseline | PASS — native_host_admission only | Native-host report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` independently replays the unchanged 20 runs and 80 operations and passes all 20 criteria. Phase 6A eager/graph artifacts establish container parity only; no new unified or claim-bearing G1 result was created. |
| Phase 4 common method adapter | PASS | BF16 delegates through `KVCacheMethod`; fixed-L/growing, allocation, graph, and path checks pass; `docs/evidence/phase4/method-admission.json`; no quantized method implemented |
| Phase 5 TurboQuant reference lane | PASS | Exact vLLM v0.25.1 source/environment lock; 3 mandatory and 1 held-out deterministic fixtures; official store/append/decode paths; no measurement adapter or timing |
| Phase 6 TurboQuant measurement adapter | PASS | Execution HEAD `0df5bb4d445d48e6cba17e30723733f8de35cb14`; current-HEAD sanitizer 3/3 PASS; frozen bounded grid 9/9 PASS; 167-object admission root published COMPLETE-last and cleanly retrieved. |
| Phase 6A Measurement Container and R2 prerequisites | PASS | Exact image built and scanned; container G0 and both BF16 parity smokes PASS; private R2 state and indefinite lock verified; synthetic and 222-object G0 roots cleanly retrieved; Decision 0016 accepted. B-009/B-010 RESOLVED. |
| G2-TQ | PASS | All three mandatory configurations pass the frozen admission criteria; final root `f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d` is durably published and cleanly retrieved. |
| Phase 7 KIVI reference lane | PASS | Exact patched source, locked image, official extension, SM120/PTX/JIT, sanitizer, four fixtures, rollover/bytes/GQA/trace, and 30-object R2 publication plus clean retrieval pass. |
| Phase 8 KIVI measurement adapter | PASS | Execution SHA `462325e9df809d3bcf24a06361bf004bc7383d73`; exact fixtures, rollover, byte accounting, path/allocation audits, Graph, sanitizer, and bounded grid 10/10 PASS; 331-object inner and 341-object report-bearing outer roots are COMPLETE-last and cleanly retrieved. |
| G2-KIVI | PASS | All three mandatory configurations are admitted and held-out k4v2 conforms; MethodAdmissionReport derives 17/17 PASS checks. |
| Phase 9 KVQuant calibration | PASS | Exact Decision 0021 patched source; isolated image; frozen 16 x 2048 train tokens; 32 K plus 32 V Fisher artifacts; three complete quantizer families; reproducibility; 68-object COMPLETE-last R2 root and clean retrieval. |
| Phase 10 KVQuant reference lane | PASS | Decisions 0021/0023; exact calibration binding; nine source-faithful dense/sparse/sink/store/append/decode fixtures; SM120/PTX/JIT/sanitizer PASS; 113-object root `32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab` COMPLETE-last and cleanly retrieved. |
| Phase 11 KVQuant measurement adapter | PASS | Decision 0027 q4 deterministic decode binding; all nine corrected fixtures; exact byte/path/allocation/GQA controls; fixed-L Graph; zero-error sanitizer; bounded grid 9/9 PASS; 165-object inner root `0834410509ea7324a41715e0e84e09617bf9b188b10394a234f9a57e804dd1f2` COMPLETE-last and cleanly retrieved. |
| Phase 11R-Q23 KVQuant current-source re-admission | PASS | Decision 0029 source and extension binding; unchanged Adapter/cache/session; all nine corrected fixtures; fresh bounded grid 9/9; path/allocation/GQA/Graph/sanitizer PASS; successor MethodAdmissionReport SHA-256 `9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2`; 212-object inner root `8ea533b9544e99140aec04b4cb9b1ad26f271273206d170e7abefa195c0581aa` COMPLETE-last and cleanly retrieved. |
| G2-KVQ | PASS | All three bit widths satisfy the current Decision 0029 method-specific admission criteria. The successor MethodAdmissionReport SHA-256 is `9cfed618cee9514a1071392d0a2dca327dcf6acd33d81ac72cc477c7880c09e2`; historical Decision 0027 report SHA-256 `59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a` remains unchanged. |
| G1-G5 unified admission | PASS | Phase 12R campaign `phase12-20260731t062914664948z-6165f78d-c78b9a`; 30/30 completed, all 10 configurations stable, root `42ab15b6617d072f9b0825b701d1df4519caa110166b8edd48b8359fe8e588e5` COMPLETE-last and cleanly retrieved |
| Phase 13B compressed batch geometry | PASS | All nine compressed configurations pass B=1/4/8 static-cache admission; 27/27 matrix, Graph, allocation, stream, and sanitizer checks pass; 52-object R2 root is COMPLETE-last and cleanly retrieved. |
| Phase 13R Pilot | BLOCKED | Fresh 810-record campaign stopped fail-closed after 14 completed runs when precomputed-feasible `tq_3bit_nc` B=8/L=98304 exhausted memory during prefix construction; root `94104865452017fbfd3c87fffd34e82b12248e379ef2680b22153dd5b71d90b8` is COMPLETE-last and cleanly retrieved. |
| Phase 13F peak feasibility | PASS | Decision 0031 includes model, cache/workspace, prefix attention/MLP and control tensors, Graph reserve, and the frozen 0.88 limit; all 810 records deterministically classify 684 feasible and 126 capacity-infeasible. Root `5090b193c046637cb3836f7d5a3ee5ebbad95d9a459672ab4aa3ff0ddb756589` is COMPLETE-last and cleanly retrieved. |
| Phase 13R2 Pilot | BLOCKED | Fresh Decision 0031 campaign preserved 23 completed, one 7,200-second supervisor-timeout failure, 660 aborted, and 126 capacity-infeasible records; root 581b02a6ca1a09c976a899b2b5d7eeb7897c0ad8f7ed8ad9fb11be5f6475f327 is COMPLETE-last and cleanly retrieved. |
| Phase 13T supervisor remediation | PASS | Decision 0032; the long-context diagnostic completed with normal forward progress; ordered finite stage deadlines pass focused and full regression tests; 16-object root `5e98c8103c6e15ca0877e39f7b9363a208ba8722bb6993ef67a04a06c9f795cb` is COMPLETE-last and cleanly retrieved. |
| Phase 13R q4 workspace remediation | PASS | Decision 0036 capacity-derived q4 workspace; all nine fixture regressions, B=4/8 L=16384 targeted admission, Graph/allocation/stream/sanitizer controls, successor q4 admission, and three-process q4 G5 refresh pass; 14-object root `9f027d64424844d0d62311daad5740e2b76960d1d5e7b8be26aa1ccba100a8db` is COMPLETE-last and cleanly retrieved. |
| Phase 13 successor Pilot | PASS | Fresh campaign `phase13-20260822t150835736582z-4ddd7b17-3a8fb3`; 810/810 records, 684 completed, 126 capacity-infeasible, 228 stable points, maximum CV 0.221851%, no failures or selective reruns; 17,384-object root `feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531` is COMPLETE-last and cleanly retrieved. |
| Phase 13D knee densification | PASS | Campaign family `phase13d-20260825t030556684636z-a06837a3-83761a` preserves Segment A's 51 valid runs and the original failed-finalization run unchanged. Segment B completes one replacement plus the frozen remaining 200 records without rerunning Segment A or regenerating 84 prefix states. All 252 logical records pass QC; root `a8559a5e01edaad949df1e128c4bddff37638801cfc89f2eb8d4894c31ef82d2` is COMPLETE-last and cleanly retrieved. |
| Phase 14 CUDA Graph A/B / Phase 14C closure | PASS | The original campaign/report/root remain immutable and retain five eager `unstable` groups. Phase 14C separates 105 stable A/B conditions into 14 fully identifiable, four unstable-eager, and two no-positive-eager-slope comparisons. Complete launch-floor-only support is 0 of 14; the negative mechanism result is closure, not a timing failure. Seven-object closure root `4cd29ea1b94201f493db8cef9ebd01933b4c4f573185eff317ec5af81e9fb000` is COMPLETE-last and cleanly retrieved. |
| Phase 15 profiler subset | PASS | 32/32 Nsys and 22/22 NCU selected profiles; all-ten common-point physical traffic; 893-object root `641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8` COMPLETE-last and cleanly retrieved. Phase 16 READY; Full Scan CLOSED. |
| Post-performance quality validation | LOCKED | Decision 0005; `PERFORMANCE_DATA_FROZEN` absent |

## Phase 0 acceptance

- Unknown archive code was not executed: pass.
- Every supplied archive/extracted input has a SHA-256 record: pass.
- Every directly fetched source/dependency has an exact commit; every currently
  identified embedded or attributed repository has an explicit commit-resolution
  plan: pass.
- Risk coverage includes CUDA compatibility, Graph support, GQA replication,
  full-prefix dequantization, legacy dependencies, and OOM: pass.
- Required status, risk, blocker, decision, method-note, provenance, and E00-E18
  planning records exist: pass.
- Graph A/B ownership and ignored-artifact audit policy are explicit: pass.

## Next action

Decision 0016 continues to authorize Measurement Lane CUDA only in the exact
recorded image digest. All earlier failed and passing reports, campaigns,
runs, fixtures, and publication roots remain unchanged.

Phase 13B, Phase 13F, Phase 13T, and all prefix-remediation evidence remain
preserved. The blocked campaign
`phase13-20260804t111810342595z-a127b0d1-8649c3` remains checksum-identical,
and none of its 69 completed runs entered the successful successor campaign.
The Phase 13D campaign family
`phase13d-20260825t030556684636z-a06837a3-83761a` remains append-only. Its 51
original valid runs, failed-finalization record, and 84 prefix states remain
unchanged; the authorized continuation is a separate segment with its own ID
and execution heads. The original Phase 14 BLOCKED report and R2 root remain
unchanged; Phase 14C retains the five unstable eager groups, closes the 105
stable conditions with explicit denominators, and records a negative mechanism
result. Phase 15 directly records reduced CPU submission-call count and GPU
idle under Graph together with heterogeneous device behavior, preserves the
Phase 14 negative pure-launch-floor result, and measures same-work physical
traffic without using profiler durations as benchmark timing. Phase 16 is
READY for a separately proposed task but has not started. Full Scan,
performance claims, and quality execution remain closed.
