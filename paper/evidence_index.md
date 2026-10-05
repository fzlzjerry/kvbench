# Evidence index for `manuscript.md`

This is a writing aid that maps key claims and numbers to the repository material they come from. All paths are repository-relative. "Direct read" means a value or a min/max/count read from rows of an existing table; no new statistic, refit, or rescoring was performed. Table, figure, and section numbers follow the current manuscript; Table A1 of its Appendix A maps the v1.0 numbering used by the v1.0 manuscript and release notes (Tables 5–9 → A5, A6, 5, 6, 7; Figures 2, 3 → A1, A2; Sections 4.1–4.3 → Section 4 and Appendix E; Sections 6.2, 6.3 → Appendix A, Section 6.2). `src/<pkg>/` stands for the harness's Python package directory under `src/`, and no remote URL is recorded here.

**Path abbreviations**

| Short | Path |
|---|---|
| FS | `artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac` (full scan) |
| P14 | `artifacts/phase14/phase14-20260826t115110887808z-47ba4220-42fc95` (eager/Graph experiment) |
| P15 | `artifacts/phase15/phase15-20260828t144810363697z-446b334e-90460f` (profiler subset) |
| P17 | `artifacts/phase17/phase17-20260916t164055992658z-9e20208a-c944c5` (modeling) |
| P18 | `artifacts/phase18/phase18-20260917t024901884906z-a866fa68-6cfb61` (reporting audit and reproduction package) |
| Q1A | `artifacts/q1a/q1a-20260919t141635000000z-6e0c7803-4c6a18f2` |
| JR | `artifacts/joint_results/q3-q4-20261004t034803222550z-f406314c-cb983f` |

**Name mapping (manuscript → data).** TQ-4bit/TQ-k3v4/TQ-3bit = `tq_4bit_nc`/`tq_k3v4_nc`/`tq_3bit_nc`; KIVI-k4v4/k2v4/k2v2 = `k4v4`/`k2v4`/`k2v2`; KVQuant-4/3/2 = `kvq4`/`kvq3`/`kvq2`. Model candidate "Surface" = `RQ2` in the modeling outputs. The manuscript's r_DRAM (= D_BF16 / D_method, measured DRAM bytes; this GPU uses GDDR7) is the field `r_hbm` in the profiler outputs and earlier reports; ρ_hbm likewise. Stages: correctness checks = Q0; Fast/Full perplexity = Q1A/Q1B; LongBench-E = Q2A; LongBench v2 no-CoT = Q2B; CoT stress = Q2C; joint admission/join = Q3/Q4. Full scan = Phase 16R; eager/Graph experiment = Phase 14/14C; profiler subset = Phase 15; modeling = Phase 17; reporting audit/reproduction = Phase 18; unified admission = Phase 12; pilot/densification = Phase 13/13D.

## Scope, implementations, methodology (Sections 1–2)

| Claim / number | Source | Location |
|---|---|---|
| Model Llama-3.1-8B-Instruct, fixed revision, 32 layers, 32 Q / 8 KV heads, head dim 128; BF16 backend = forced Flash SDPA with native GQA, no fallback | `docs/decisions/0007-phase3-primary-model-and-bf16-backend.md` | Decision items 1, 7–12 |
| 17.2 GB BF16 cache for one 128K sequence | FS/wall-closure/point_summary.parquet | row bf16, B=1, context_label 131072: `logical_bf16_bytes` (direct read) |
| GPU, container digest, PyTorch 2.12.1 / CUDA 13.0 | `docs/decisions/0007-…`, `docs/decisions/0016-measurement-container-execution-authority.md`, `configs/hardware/rtx_pro_6000.yaml` | — |
| TurboQuant = vLLM v0.25.1 path, Hadamard + Lloyd–Max keys, FP16 norm, uniform values, omits QJL | `docs/method_notes/turboquant.md` | "Pinned reference authority", "Algorithm and current source semantics" |
| TurboQuant keeps first/last two layers in BF16 | `src/<pkg>/third_party/vllm_turboquant/config.py` (`get_boundary_skip_layers`, n=2); FS/wall-closure/point_summary.parquet | `residual_bytes` = 4/32 of BF16 bytes for tq_* rows |
| KIVI: post-paper official snapshot; group 32 / residual 32; native-GQA residual patch exact in BF16; FP16-only CUDA ABI staged | `docs/method_notes/kivi.md`; `docs/decisions/0017-…`, `0018-…` | "Algorithm", "B-019 remediation result", "Phase 8 measurement adapter result" |
| KVQuant: patched upstream (Llama-3.1 RoPE, native GQA, cap 12 = six per tail of 1,024-wide row, deterministic ties, 5 FP16 sinks); pre-RoPE key boundary | `docs/decisions/0020-kvquant-upstream-gqa-patch.md`, `0021-…`, `0026-…`; `configs/methods/kvquant.yaml` | Decision items 4–9 |
| KVQuant graph-safe / deterministic kernel changes (caller-owned buffers; ordered tile reduction replacing atomics; shared-memory initialization) | `docs/decisions/0024-…`, `0025-…`, `0027-…`, `0029-…` | Decision sections |
| KVQuant calibration: sixteen 2,048-token WikiText-2 train windows, Fisher-weighted | `docs/method_notes/kvquant.md` | "Frozen calibration state" |
| Timing endpoint, warmup 64 / measured 256, five processes, blocked randomization | `docs/measurement_protocol.md` §§2–7; `configs/plans/full_scan.yaml` (`measurement`) | — |
| Point latency = median of five process medians; CV across processes | FS/wall-closure/point_summary.parquet | `median_ms`, `process_medians_ms`, `cv` |
| Gates G1–G5 all PASS for ten configurations at B=1, L=4096, Graph | `docs/phase_reports/phase12-unified-admission.md` | configuration table |
| Pilot: knee density insufficient for 25/30; 84 densification contexts | `docs/phase_reports/phase13-successor-pilot-scan.md`; `docs/phase_reports/phase13d-knee-densification.md` | "Provisional fits"; FS/outer/adaptive_grid.parquet (84 rows) |
| 534 points (450 base + 84 adaptive); 441 feasible / 93 infeasible; 2,670 slots = 2,205 + 465; 38 replacements; max CV 1.305%; zero mismatches | `docs/phase_reports/phase16r-full-scan.md` | "Execution and QC" |
| Feasibility limit 88% of device memory (89.73 GB) | `configs/hardware/rtx_pro_6000.yaml` (`max_memory_fraction`); FS/outer/capacity_amplification.parquet | `memory_limit_bytes` |
| Ratio definitions r_nom, r_alloc, r_hbm, A | `docs/experiment_contract.md` §5; `docs/decisions/0019-…`; `docs/phase_reports/phase15-profiler-subset.md` | "Same-work physical traffic" |

## Bytes, traffic, latency, capacity, GQA, Graph (Section 3)

| Claim / number | Source | Location |
|---|---|---|
| Table 2 (r_nom, r_alloc at 4K and 128K; boundary-layer, workspace, metadata, outlier bytes) | FS/wall-closure/point_summary.parquet | rows B=1, context_label 4096 and 131072: `r_nominal`, `r_alloc`, `residual_bytes`, `workspace_bytes`, `scale_zero_bytes`, `metadata_bytes`, `outlier_value_bytes`+`outlier_index_bytes` (direct read) |
| Table 3 DRAM GB, r_hbm, A, L2 hit, SM activity, occupancy | `docs/phase_reports/phase15-profiler-subset.md`; P15/decode_traffic.parquet; P15/hbm_ratios.parquet | "Same-work physical traffic" table; `cache_path_dram_bytes`, `total_decode_dram_bytes`, `l2_hit_rate`, `sm_activity`, `achieved_occupancy` |
| Table 3 frozen columns regenerate from the v1.0 joint-results package (checked 2026-10-05: all ten rows match digit for digit) | JR/inputs/performance_profiler_feature_join.parquet; JR/inputs/wall_same_work_ratios.parquet | `decode_traffic_records` (B = 1, context 131072: `cache_path_dram_bytes`, `total_decode_dram_bytes`), `traffic_amplification_records` (`r_alloc`, `r_hbm`, `A_traffic`); `performance_only_ratio` |
| Table 3 r_alloc column and S column | FS/wall-closure/point_summary.parquet; FS/wall-closure/same_work_ratios.parquet | B=1, context_label 131072 rows: `r_alloc`; `performance_only_ratio` |
| KIVI residual-copy traffic not isolatable; fully classified DRAM bytes; no expanded-K/V signature | `docs/phase_reports/phase15-profiler-subset.md` | last paragraph of "Same-work physical traffic" |
| KVQuant decode work is issued once per sequence (Section 3.3 explanation of near-linear batch scaling) | `src/<pkg>/adapters/kvquant.py`; `src/<pkg>/runtime/cuda_graph.py` | static code reading only: `append_decode` loops `for batch_idx in range(cache.batch_size)` over `_pack_nonsink_token` (lines 879–889); `_decode_compressed` calls the key decode kernel per sequence on batch-1 slices (955–976) and the deterministic value-decode API per sequence (992–1018, 1046–1084); graph capture uses one side stream (cuda_graph.py 115–126) |
| KVQuant B = 1 cost still reflects other adapter changes (Section 6.1) | `src/<pkg>/adapters/kvquant.py`; Section 2.2 sources above | static code reading only: one decode-kernel launch per sequence at B = 1 (no batch loop iterations beyond one); other adapter code such as per-query-head sink `torch.bmm` loops (lines 928–942, 1023–1037) and the deterministic value-decode APIs (1046–1084) remain |
| r_alloc = C_logical / C_alloc (Section 2.5); BF16 r_alloc slightly below 1 | `scripts/phase13_pilot.py` lines 5036–5037 (`r_alloc = logical / allocated`); `scripts/phase17_modeling.py` line 318 (modeling reads the stored per-point `r_alloc`); FS/wall-closure/point_summary.parquet | bf16 B=1, 4K: `logical_bf16_bytes` / `allocated_bytes` = 0.9997 |
| A = (D_method / D_BF16) / (C_alloc,method / C_alloc,BF16), base = BF16 actual allocation (Section 2.5) | `scripts/phase15_profiler_subset.py` lines 2799–2831 (`rho_alloc = allocated / bf16_allocated`, both actual allocated bytes) and 838–862 (`rho_hbm` = method / BF16 cache-path DRAM bytes; `A_traffic = rho_hbm / rho_alloc`); P15/traffic_amplification.parquet | Table 3's r_alloc column uses the logical-payload base (point summary); at B = 1, 128K the two bases agree to four significant digits (e.g., TQ-4bit 2.23058 vs 2.2306), so no table value changes |
| r_nom = [f + (1 − f)(q_K + q_V)/32]^(−1), f = 4/32 for TurboQuant | FS/wall-closure/point_summary.parquet | `r_nominal` matches the formula for all nine configurations (e.g., TQ-4bit 2.909, TQ-k3v4 3.160, TQ-3bit 3.459, KIVI/KVQuant 4.00 / 5.33 / 8.00) |
| Code and data availability (Appendix A) | `git remote -v` (origin: the project's GitHub repository, renamed before release; URL withheld for anonymity); GitHub API (repository public, commit 0641de4b present, no license file detected, checked 2026-10-04); author decisions: Apache-2.0, packages as Release assets, raw measurements on request; `git ls-files artifacts` (only `artifacts/README.md` tracked) | commands from P18/README.md and JR/QUALITY_VALIDATION_REPORT.md; commands rewritten with python3 and <output_dir>; joint-results command taken from its report, as the package has no README |
| Profiled DRAM bytes include writes ("transfers", Section 3.2) | `scripts/phase15_profiler_subset.py` lines 800–821; P15/metric_map.json | totals sum `dram__bytes_op_read` and `dram__bytes_op_write` |
| Of the 108 BF16-infeasible pairs, 27 = capacity points, 81 both infeasible (Section 3.3) | FS/wall-closure/same_work_ratios.parquet joined with FS/wall-closure/point_summary.parquet and FS/outer/capacity_amplification.parquet | direct read: 108 rows with `null_reason = bf16_capacity_infeasible`; method `disposition` stable for 27 (all in the capacity table), capacity_infeasible for 81 |
| Table 4 latencies | FS/wall-closure/point_summary.parquet | `median_ms` at the listed (config, B, context_label) rows (direct read) |
| 357 measured ratios, all < 1; family ranges KIVI 0.471–0.842, TQ 0.048–0.702, KVQuant 0.008–0.092 | FS/wall-closure/same_work_ratios.parquet | rows with `calculated = true` (count and min/max are direct reads) |
| 123 null ratios = 108 BF16-infeasible + 15 no exact BF16 observation | FS/wall-closure/same_work_ratios.parquet | `null_reason` counts (direct read) |
| Figure 1a (v1.0 Figure 1, redrawn; panel a of `paper/scripts/fig1_samework_ceiling.py`, one panel per family) | Source data: FS/wall-closure/same_work_ratios.parquet (`historical_context`, `performance_only_ratio`, one series per configuration × batch). Script of the earlier single-panel figure: `paper/scripts/fig1_same_work_ratios.py` | Original plot FS/wall-closure/plots/host-wall-ratios.svg came from `scripts/phase16_wall_closure.py` → `scripts/phase13_pilot.py::_svg_line_plot`; the paper script carries a verbatim copy of that function and `--verify-original` reproduces the original SVG byte-for-byte. The redraw keeps the same 45 series, points, and x/y coordinate mapping; it changes only the title wording ("Host-wall" → "Wall-clock", matching the manuscript's terminology), the legend (complete, 9 colors × 5 markers instead of a truncated 45-entry list with 12 cycling colors), tick placement and labels, and axis titles |
| 27 capacity points at B·L = 393,216; BF16 114.3–114.4 GB vs limit 89.73 GB; compressed 72.1–86.0 GB | FS/outer/capacity_amplification.parquet | all 27 rows: `bf16_predicted_bytes`, `method_predicted_bytes`, `memory_limit_bytes`, `bf16_infeasible_reason` |
| Transformers eager uses `repeat_kv`; math-SDPA control dispatched `expand`/`clone` | `docs/decisions/0007-…` | "Context", item 9, "Rejected alternatives" |
| KIVI snapshot residual path: 32-head copy, 65,536 → 262,144 bytes | `docs/decisions/0017-kivi-source-authority-and-gqa-materialization.md`; `docs/method_notes/kivi.md` | "Phase 7 source-audit result" |
| Eager/Graph experiment design: B ∈ {1,4}, six contexts, 120 conditions, 10 infeasible, 105 stable, 5 unstable eager | `docs/phase_reports/phase14-analysis-closure.md`; P14/graph_ab_pairs.parquet | `pair_status` counts |
| Pure launch-floor criterion and 0/14 result; 4 inconclusive; 2 no positive slope | `docs/phase_reports/phase14-analysis-closure.md`; P14/graph_effects.parquet | `floor_material_threshold` 1.05, `slope_similarity_interval` [0.8,1.2], `launch_floor_interpretation_supported` |
| Method patterns: KIVI floor ratio 9.84–10.31, slope 0.29–0.31 (B=1), knee −61,440; TQ floor 1.05–1.59; KVQuant ≈1.00 | P14/graph_effects.parquet | `floor_ratio`, `slope_ratio`, `knee_shift_tokens` (direct read) |
| CUDA API calls 10,832 / 12,400 / 54,608–63,312 / 53,328 vs eight graph launches; idle lower 16/16; kernel count/order unchanged | `docs/phase_reports/phase15-profiler-subset.md` | "CUDA Graph mechanism" |
| Figure A1 (v1.0 Figure 2, unchanged) | P14/plots/graph-ratio-vs-context.svg | copied byte-identical to `figures/fig2_graph_ab_ratio.svg`; `graph_ratio` = eager/Graph wall median (P14/graph_ab_pairs.parquet) |

## Modeling (Section 4, Appendix E)

| Claim / number | Source | Location |
|---|---|---|
| Model D form, basis, bounds, 12 coefficients; candidates E/Surface(RQ2)/F_shape/F_diagnostic; holdouts; selection rule; targets | P17/candidate_spec.json; `docs/reproduction/phase18/predictor.md` | `candidates`, `holdouts`, `model_selection`, `targets` |
| Selection used outer results; score not unbiased | P17/model_target_status.json | `selection.outer_selection_score_unbiased = false` |
| Table A5 (v1.0 Table 5) macro scores and complexities | P17/model_target_status.json | `selection.selection_scores` |
| Table A6 (v1.0 Table 6) per-cell median / P95 | `docs/phase_reports/phase17-modeling.md` | holdout table (all 16 rows) |
| All 456 sign-accuracy records have observed label "slower" (Section 4, Appendix E) | `scripts/phase17_modeling.py` lines 936–958 (label = observed BF16/method ratio ≥ 1); P18/data/out_of_fold_predictions.csv collapsed by the package's own helper | direct count for model D: 357 (batch holdout) + 99 (context-band holdout) records, all observed < 1; D predicts "faster" for 31, reproducing 425/456 |
| Targets missed: 0.0792, 1.6453, 425/456, 346/513; knee not evaluable | P17/model_target_status.json | `targets` |
| Macro values are equal-weight means of 11 cell statistics, not pooled quantiles | P18/reporting_audit.md | first paragraph |
| BF16 edge failure: B=1, L=4096 observed 11.7736 ms, predicted 668.623 ms, error 5,579%; edge n=15 median 21.5% / P95 4,017%; interior n=27 median 2.46% / P95 6.02% | P18/reporting_audit.md; P18/bf16_batch_holdout_tail.csv | edge/interior lines; top row |
| Figure A2 (v1.0 Figure 3, redrawn) | Source data: P18/data/out_of_fold_predictions.csv, collapsed to logical points by the package's own `audit()` and filtered to the three geometry protocols as in `reproduce()` (968 points). Script: `paper/scripts/fig3_heldout_predicted_vs_measured.py` | Original plot `docs/evidence/phase18/figures/predicted_vs_measured.svg` came from P18/reproduce.py::`_svg_scatter`; `--verify-original` reproduces it byte-for-byte. The redraw keeps the same points, log10 transform, shared axis range, square plot area, and identity line; it adds axes, 1-2-5 ticks in ms, axis titles, and a small legend |
| Knees: 23 identified, 14 weak, 12 linear/constant, 1 insufficient span; 1,000 bootstrap draws | P18/data/knee_estimates.csv | `fit_status` counts (direct read); P17/candidate_spec.json `bootstrap` |
| Fold parameter vectors not saved | P18/reporting_audit.md | last paragraph |

## Quality and joint results (Section 5)

| Claim / number | Source | Location |
|---|---|---|
| Cache-sensitive protocol: 16 conditioning tokens; PPL incremental teacher forcing; identical tokens for all configs | `docs/phase_reports/qp1-quality-contract.md`; `configs/quality/quality_contract.yaml` | "Tokenizer, datasets, and selected inputs"; `ppl.mode` |
| Stage order; native prefill for finalists only | `CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md` | §14 stage order; §10.4; execution block (`secondary_protocol`) |
| Margins (PPL 1% / 2% / 5%; LongBench-E 2 / 3 / 5 / 1 pp; v2 2 / 5 / 5 / 1 pp / 95%) | `configs/quality/gates/quality_margins.yaml` | all fields |
| Datasets: C4 2,048 docs from first 4,096 rows; LongBench-E 13 tasks / 3,668; v2 503 IDs / 321 eligible / 182 excluded | `docs/phase_reports/qp1-quality-contract.md` | "Tokenizer, datasets, and selected inputs" |
| Protocol history: quality protocol preregistered before performance (2026-07-22); bindings after performance, before quality outputs; B=1 amendment after Q0 failure, before Fast PPL | Git history (commit "preregister post-performance quality validation protocol", 2026-07-22); `docs/phase_reports/qp1-quality-contract.md` "Gates and resolved choices"; Q1A/scope_amendment.json `chronology`; `docs/evidence/q1a/scope-amendment-approval.json` | — |
| Q0: 70/70 core probes, suffix 10/10, eager/Graph 100/100 tokens zero error, cache dependence 9/9; batch invariance fails 10/10, max errors 0.34–3.03 | `docs/phase_reports/q0-cache-sensitive-correctness.md` | "Per-configuration result", "Other Q0 controls" |
| Batch failure = shape-dependent BF16 projection arithmetic; first divergence layer-0 q_proj/v_proj | `docs/phase_reports/q0-batch-diagnosis.md` | "Result" |
| Table 5 (v1.0 Table 7) Fast PPL (ΔNLL, CIs, relative PPL, statuses); KVQuant rows in Table A7 | `docs/phase_reports/q1a-fast-ppl.md`; Q1A/fast_ppl_summary.json | results table; `decisions[*].aggregate_statistics` |
| TQ-4bit relative-PPL CI [0.782%, 1.288%] | Q1A/fast_ppl_summary.json | `tq_4bit_nc.aggregate_statistics.ci_lower/upper_relative_ppl_change` |
| KVQuant degradation present across datasets and lengths; retained as observed | `docs/phase_reports/q1a-fast-ppl.md`; Q1A/fast_ppl_summary.json | KVQuant paragraph; `length_statistics` |
| KVQuant cause unresolved, not attributed to upstream universally | JR/QUALITY_VALIDATION_REPORT.md | "Interpretation and reproducibility" |
| Full PPL k4v4: WikiText-2 +0.173736% [0.001228, 0.002242]; C4 +0.181803% [0.001145, 0.002571]; max cell +0.352% | `docs/phase_reports/q1b-full-ppl.md` | dataset and length tables |
| LongBench-E: 54.3810 vs 54.2490, drop 0.1321, CI [−0.2908, 0.5449]; categories −0.732…0.775; lengths 0.015–0.223; invalid 0/0 | `docs/phase_reports/q2a-b1-longbench-e.md` | macro paragraph |
| Table 6 (v1.0 Table 8) v2 overall, categories, length guard, invalid, retention, contingency 86/8/3/224 | `docs/phase_reports/q2b-b1-longbench-v2.md`; JR/joint_result_summary.json | tables; `q2b.overall`, `q2b_guardrails` |
| Code category N=15, 7/15 vs 6/15, 6.6667 pp, CI [0, 20] pp | JR/QUALITY_VALIDATION_REPORT.md | v2 category table, row "Code Repository Understanding" |
| Budget stops 253/256; invalid 24/25; long-dialogue 20/39 each | `docs/phase_reports/q2b-b1-longbench-v2.md` | invalid and stop-reason paragraphs |
| Table 7 (v1.0 Table 9); 8 fail / 1 inconclusive / 0 qualified; CoT stress `not_run_no_eligible_finalist`; native prefill `not_run_conditional_finalists_only` | JR/QUALITY_VALIDATION_REPORT.md; JR/joint_result_summary.json; JR/quality_admission_table.csv | stage table; `compressed_status_counts`, `q2c_status`, `native_prefill_status` |
| Join: 2,670 slots, 0 unmatched, 0 duplicates, 600 B=1 rows, 2,070 outside scope; qualified table empty; best speedup null | JR/joint_result_summary.json; JR/qualified_compressed_candidates.parquet | `performance_join.*`, `best_qualified_compressed_speedup` |

## Appendices

| Claim / number | Source | Location |
|---|---|---|
| Container digests, freeze tag, roots | `docs/phase_reports/qp0-tag-correction-closure.md`; `docs/phase_reports/phase16r-full-scan.md`; `docs/phase_reports/phase15-profiler-subset.md`; `docs/phase_reports/phase17-modeling.md`; `docs/phase_reports/q3-q4-joint-results.md` | identity and publication lines |
| CPU-only reproduction packages | P18/README.md, P18/reproduction_manifest.json; `docs/phase_reports/q3-q4-joint-results.md` "Reproduction and custody" | — |
| Timeline dates | campaign identifiers and report dates in `docs/phase_reports/`; `docs/status.md` | — |
| First full-scan summary used CUDA-event medians; host-wall closure re-derived from raw files | `docs/phase_reports/phase16r-full-scan.md` | "Immutable analysis correction" |
| Earlier blocked pilot and stopped scan attempt contributed no timing | `docs/phase_reports/phase13-successor-pilot-scan.md`; `docs/phase_reports/phase16r-full-scan.md` | preservation paragraphs |

## Figure regeneration

Run from the repository root; both scripts write only under `paper/`:

```bash
.phase17-venv/bin/python paper/scripts/fig1_same_work_ratios.py --verify-original --output paper/figures/fig1_same_work_ratios.svg
python3 -B paper/scripts/fig3_heldout_predicted_vs_measured.py --verify-original --output paper/figures/fig3_heldout_predicted_vs_measured.svg
```

## Reference verification (checked 2026-10-04)

Metadata for every arXiv entry was checked against the arXiv API (`export.arxiv.org/api/query`, latest version metadata). Versions in the list are the local PDF versions for [1]–[23] and the latest arXiv version for [24]–[35]. Venues are recorded here only; the reference format carries no venue.

| Ref | arXiv ID | First author per arXiv | Year (arXiv v1) | Venue and how it was checked |
|---|---|---|---|---|
| [24] Llama 3 Herd | 2407.21783v3 | Aaron Grattafiori (561 authors) | 2024 | none listed on arXiv |
| [25] GQA | 2305.13245v3 | Joshua Ainslie | 2023 | EMNLP 2023 (arXiv comment; ACL Anthology 2023.emnlp-main.298) |
| [26] PyTorch | 1912.01703v1 | Adam Paszke | 2019 | NeurIPS 2019 (arXiv comment) |
| [27] FlashAttention-2 | 2307.08691v1 | Tri Dao (sole author) | 2023 | ICLR 2024 (https://iclr.cc/virtual/2024/poster/17889, page title "ICLR Poster FlashAttention-2: …") |
| [28] vLLM / PagedAttention | 2309.06180v1 | Woosuk Kwon | 2023 | SOSP 2023 (arXiv comment); software release v0.25.1 tag resolves to commit 752a3a5 (GitHub API), matching `docs/method_notes/turboquant.md` |
| [29] Pointer Sentinel (WikiText-2) | 1609.07843v1 | Stephen Merity | 2016 | ICLR 2017 (OpenReview PDF https://openreview.net/pdf?id=Byj72udxe, header "Published as a conference paper at ICLR 2017"; direct download returned 403, header confirmed via search index) |
| [33] T5 (C4) | 1910.10683v4 | Colin Raffel | 2019 | JMLR 21, 2020 (jmlr.org/papers/v21/20-074.html) |
| [34] LongBench | 2308.14508v2 | Yushi Bai | 2023 | ACL 2024 (arXiv comment; ACL Anthology 2024.acl-long.172) |
| [35] LongBench v2 | 2412.15204v2 | Yushi Bai | 2024 | ACL 2025 (ACL Anthology 2025.acl-long.183) |
| [30] Nsight Compute | — | NVIDIA | — | https://docs.nvidia.com/nsight-compute/ (HTTP 200, title "Nsight Compute Documentation") |
| [31] CUDA Graphs | — | NVIDIA | — | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html (HTTP 200, "4.2. CUDA Graphs"); the former CUDA C++ Programming Guide URL redirects to this guide |
| [32] Nsight Systems | — | NVIDIA | — | https://docs.nvidia.com/nsight-systems/ (HTTP 200, title "Nsight Systems") |

Existing entries [1]–[23]: first authors match arXiv for 22 of 23 (the second author of [20], B. Van Durme, also matches). One correction: [21] InnerQ is listed on arXiv under "Sayed Mohammadreza Tayaranian Hosseini" (the PDF title page prints "Mohammadreza Tayaranian"), so the entry now reads "S. M. Tayaranian Hosseini et al." Titles follow the rule "published title if formally published, otherwise current arXiv title". [14] Palu was published at ICLR 2025 (https://iclr.cc/virtual/2025/poster/29993) as "Palu: KV-Cache Compression with Low-Rank Projection", which is the title already in the list, so it is unchanged; the current arXiv title ("Palu: Compressing KV-Cache with Low-Rank Projection") differs. [20] Compactor has no formal publication record found (arXiv, Hugging Face Papers, and ResearchGate list only the preprint), so its entry now uses the current arXiv title "Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores" in place of the local-PDF title "Compactor: Calibrated KV Cache Compression with Approximate Leverage Scores".

## Post-hoc review round (2026-10-04/05)

Everything below was defined after the frozen results were known (post hoc) and reads only frozen roots; no timing scan or quality run was repeated. Code and sealed outputs are in `paper/posthoc/` (copy list in its README).

**Additional path abbreviations**

| Short | Path | Root |
|---|---|---|
| PA | `paper/posthoc/artifacts/posthoc-a-20261004t143713987904z-0641de4b-ab87cf` (traffic model, ceilings, bandwidth, elasticity, capacity, decomposition) | `3dc84c64…` |
| PBO | `paper/posthoc/artifacts/posthoc-b-offline-20261004t145657101765z-0641de4b-63e796` (kernel breakdowns from P15 traces) | `c347e20a…` |
| PBG | `artifacts/posthoc/posthoc-b-gpu-20261004t202246814451z-0641de4b-e22fff` (Nsight diagnostics, raw) | `4cdbb3ae…` |
| PBA | `paper/posthoc/artifacts/posthoc-b-analysis-20261004t202531220352z-0641de4b-74462d` (Nsight diagnostics, analysis) | `39efeb1c…` |
| PC | `paper/posthoc/artifacts/posthoc-c-exclusions-20261004t164203027878z-0641de4b-e16598` (KVQuant quality exclusion record) | `47296f7c…` |
| PD | `paper/posthoc/artifacts/posthoc-d-20261005t064839061781z-b0ed1058-d6fb96` (final revision: cache-only ceiling S_cache, KIVI-k4v4 kernel breakdown) | `b0969338…` |

| Claim / number | Source | Location |
|---|---|---|
| KIVI reads 2.8–4.3× fewer cache bytes (abstract, §3.2, §6.2) | PA/a2_bandwidth_b1.csv | `r_dram_v2` for k4v4/k2v4/k2v2 (2.81/3.41/4.31) |
| Cache-path BW at B = 1: BF16 89%, KIVI 8–13%, KVQuant 1–2% with the frozen classification (Table 3) and 2–4% reclassified (Table A2), TurboQuant ≈1% (abstract, §3.2) | PA/a2_bandwidth_b1.csv | `bw_cache_marginal_v1_pct_peak` (frozen classification) and `bw_cache_marginal_v2_pct_peak` (reclassified; equal for all but KVQuant); numerator D_cache(128K) alone: `bw_cache_v1_pct_peak` (BF16 92.2%, KIVI 8.0–12.9%) |
| Total BW column of Table 3 | PA/a2_bandwidth_b1.csv | `bw_total_pct_peak` |
| Peak 1,792 GB/s and its cross-check | PA/summary.json | `peak` (`ncu_device_attributes.derived_peak_bytes_per_s`) |
| KVQuant reclassified (post hoc, Table A2; Table 3 keeps the frozen values): cache DRAM 8.08/6.78/5.62 GB, r_DRAM 2.15/2.56/3.09, A 1.46/1.58/1.79, cache-path BW 3.6/3.0/2.4%; moved bytes 3.53/3.00/2.46 GB; 2.2–4.3× fewer cache bytes over all configurations with it | PA/traffic_profiles.csv, PA/a2_bandwidth_b1.csv, PA/traffic_reclassified_kernels.csv | `cache_v2`; `r_dram_v2`, `amplification_A_v2`; per-kernel rows |
| Phase 15 rule that put the KVQuant key kernels in other_model | `scripts/phase15_profiler_subset.py` | `classify_kernel` (generic rule matching "rope") |
| α_m, non-cache fit (n0, n1), residuals 0.8% vs 3.0% | PA/summary.json | `traffic_model.alpha`, `traffic_model.noncache_fit` |
| Traffic model at other B = 1 profiles: cache within 2.8%, total within 0.5% | PA/traffic_model_validation.csv | `cache_v2_relative_error`, `total_v2_relative_error` |
| Traffic model at B = 8 and 16 | PBA/traffic_model_check.csv | `total_v2_relative_error`, `cache_v2_relative_error` |
| S_eq 1.00–1.96, BF16 72–85% of peak, no point with S above S_eq (§3.2) | PA/summary.json, PA/a1_roofline_ratios.csv | `a1.by_family.*.s_eq_port_v2_range`, `a1.bf16_bw_fraction_of_peak_range`, `count_s_measured_gt_s_eq_port_v2` |
| S_cache 1.02–2.17 at the 357 same-work points, none with S above it; S/S_cache 28–65% (KIVI), 3–58% (TurboQuant†), 0.6–8% (KVQuant) (§3.2, Figure 1b, §6.1) | PD/summary.json, PD/d1_cache_ceiling_points.csv | `d1_cache_ceiling.s_cache_range`, `count_s_measured_gt_s_cache`, `by_family.*.s_over_s_cache_range`; BW_eff,BF16 from PA/a5_decomposition_fits.csv (checked equal) |
| Below 16K tokens per batch S_eq and S_cache stay below 1.05 (39 points, maxima 1.049); at B = 1, 4K the cache path is 2.9% of BF16's step, S_cache 1.02, S_roof 1.37–1.39 (§3.2, App. C) | PD/summary.json, PD/d1_cache_ceiling_points.csv | `min_tokens_with_s_cache_gt_1.05` = `min_tokens_with_s_eq_gt_1.05` = 16,385; `b1_4k` rows; `bf16_cache_time_share` |
| S_roof 1.37–2.78 as the comparison ceiling (App. C); BF16 non-cache time at least 11.3 ms | PD/summary.json | `s_roof_alg_range` (equal to PA), `bf16_noncache_time_ms_min` |
| Stored-bytes variant: TurboQuant S_cache 1.02–1.70 → 1.02–1.88, others within 0.01 (App. C) | PD/summary.json | `by_family.*.s_cache_range`, `s_cache_stored_range` |
| Elasticity slopes 0.142 / −0.057 / 1.243 vs 0.304 / 0.242 / 0.553; 40 included, 92 below threshold, 30 incomplete | PA/a3_elasticity_pooled.json, PA/exclusions.json | pooled rows; `counts` |
| Steady-state capacity: 505 vs 441 feasible; BF16 74.1 GB at 393,216 tokens; 40.3 GB MLP transient; 61 steady-only points, none measured; TQ-4bit B4 128K 55.5 vs 93.4 GB | PA/a4_capacity_summary.json, PA/a4_capacity_points.csv | summary fields; rows bf16/tq_4bit_nc |
| Decomposition: BF16 c0 ≈ 12 ms, BW_eff 1.60–1.66 TB/s; TQ 14→208 GB/s; KIVI 0.14–0.23 → 0.38–0.48 TB/s; KVQuant c0 per sequence; R² ranges (§4, Table A3) | PA/a5_decomposition_fits.csv | `c0_ms`, `bw_eff_cache_bytes_per_s`, `r_squared` |
| TurboQuant split count 4 in the port; vLLM v0.25.1 default 32; fixtures use 4 | `src/<pkg>/runtime/turboquant_cache.py:22`; `.reference/vllm-source-v0.25.1/.../vllm/config/attention.py:33`; `reference/turboquant/generate_fixtures.py:50` | constants |
| Kernel source identical to upstream except imports; launch parameters | `src/<pkg>/third_party/vllm_turboquant/triton_turboquant_decode.py` vs upstream `vllm/v1/attention/ops/triton_turboquant_decode.py`; `src/<pkg>/adapters/turboquant.py:345–455` | diff; launch call |
| Stage-1 grid (B, 32, 4), 32 threads; share 55.7/90.6/97.2% at 4K/32K/128K; 1.4% of 9,024 resident warps | PBO/turboquant_geometry.json | `nsys` rows; `ncu_128k` |
| Split-32 diagnostic: stage-1 7.3–7.9× (B = 1) and 1.8–2.7× (B = 8) faster; 1.6–3.2× BF16 traced kernel time; as-ported traces within 2.5% of wall time (§3.3, Table A4) | PBA/split_diagnostic.csv | ratio columns; `frozen_host_wall_ms_split4` |
| In all 33 traced runs (12 with 32 splits): historical cache and pointers unchanged, adapter fingerprint validated, outputs finite (App. D) | PBA/worker_checks.json | per-job `historical_cache_unchanged`, `cache_pointers_stable`, `adapter_fingerprint_validation_error` (null), `output_finite`; `tq_split_override` for the 12 split-32 jobs. The host-side mechanism test (`test_partB_split_override.py`, synthetic data, not run in the measurement container, not sealed) is not cited |
| KVQuant selection kernel: grid/block (1,1,1), 64 launches per sequence and step, 1.9–2.7 ms per launch, 80.2/72.6/67.5/63.3% at 24K/48K/64K/128K, 136/130/123/175 ms (§3.3, App. D) | PBO/kvquant_select_kernel.json | rows |
| Selection kernel source and `<<<rows, 1>>>` launch | `third_party/patches/kvquant/0002-graphsafe-kvq3-deterministic.patch` | `SelectFixedOutliers1024Cap12Kernel` |
| Per-sequence calls on batch-1 slices (keys and values) | `src/<pkg>/adapters/kvquant.py:630–700` | `select_fixed_outliers_1024_cap12_out` calls |
| KVQuant quality exclusion (reason code, ΔNLL ≈ 3.4 at all bit widths) | PC/exclusion_record.json; `docs/evidence/q1a/fast-ppl.json` | `exclusions[0]`; `decisions` |
| BF16 GEMMs are not batch-invariant | He and Thinking Machines Lab (2025), reference [36] | — |
| FP8 KV cache supported in vLLM and TensorRT-LLM | references [37], [38] | — |
| R2 publication receipts | `docs/evidence/posthoc-review/*-r2-publish.json`, `*-r2-verify.json` | status PASS |
| TurboQuant† = as ported, split = 4, defined once in §2.2 and used for every TurboQuant ratio or latency; the split-32 range 1.6–3.2× BF16 (Nsight Systems GPU kernel time, not wall-clock) appears only in §2.2, §3.3, the Table 3/4 notes, App. D, §6.1, and §6.2; ceilings carry no marker | PBA/split_diagnostic.csv; `paper/latex/paper.tex` macros `\ported`, `\TQdiag` | min/max of the 32-split/BF16 step-time ratio over the nine points with a BF16 trace (1.60–3.16) |
| Post-hoc analyses regenerate unchanged (Appendix A) | rerun of `partA_analysis.py`, `partB_offline.py`, `partB_analysis.py`, `partC_exclusion_record.py`, and `partD_final_analyses.py` on the retained roots, 2026-10-05 | every sealed data file of PA, PBO, PBA, PC, and PD identical; the only difference is how the excluded staging directory's path was written in PBA/exclusions.json and the report (absolute in the sealed run, relative in the rerun) |
| Item numbers used by the v1.0 release notes (Appendix A, Table A1) | `paper/release/RELEASE_ASSETS_README.md` (as published with v1.0; unchanged) | "Paper item" columns |
| KIVI-k4v4 at B = 1, 4K (Graph trace, GPU kernel time per step): quantized-cache kernels 0.84 ms; 3,072 FP16 bmm and 2,048 FP16 add kernels 5.9 ms; BF16 attention kernels 0.66 ms; 6,826 vs 1,354 kernels per step (§4) | PD/d2_kivi_kernel_breakdown.csv, PD/summary.json (from P15 `nsys_events_classified.parquet`, runs `nsys-k4v4-b1-l4096-cuda_graph-attempt0`, `nsys-bf16-b1-l4096-cuda_graph-attempt0`) | categories `kivi_quantized_cache_gemv`, `fp16_bmm`, `fp16_add`, `bf16_flash_attention`; per-step = total / 8 Graph replays |
| The FP16 bmm and add kernels are issued per query head for the residual window and the current token (§4) | `src/<pkg>/adapters/kivi.py` lines 764–776 and 860–871 | static code reading: `for query_head in range(32)` loops with `torch.bmm(...)` and `add_` |
| Findings paragraph (§1) | rows of §3.1–§3.6, §4 and Appendix E, and §5 above | summary only; no new number |
| Conclusions: the released CPU packages regenerate the frozen tables; post-hoc analyses and raw data on request (§6.2, App. A) | v1.0 release assets; Table 3 check above | — |
