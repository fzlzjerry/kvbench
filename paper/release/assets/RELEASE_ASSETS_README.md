# kvbench v1.0 release assets

This release has two CPU-only reproduction packages for the paper *Bytes Are Not Latency* (`paper/manuscript.md` in the repository). Neither package needs a GPU, model weights, or network access. Each archive unpacks to one top-level directory, and all commands below run from that directory. `<output_dir>` must not exist yet.

`SHA256SUMS` lists the checksum of each archive.

## kvbench-modeling-repro.tar.gz: modeling results

Requires Python 3 (standard library only).

```bash
python3 reproduce.py reproduce --package . --output <output_dir>
```

The command regenerates the reporting audit, the BF16 batch-holdout error table, six SVG figures, and four predictor examples from the stored out-of-fold predictions and model-comparison tables. It does not refit any model.

| Paper item | Source in the package |
|---|---|
| Table 5 (candidate scores) and the four predictive targets (Section 4.2) | `data/model_target_status.json`, `data/model_comparison.csv` |
| Table 6 (model D held-out errors) | `data/model_comparison.csv` |
| BF16 batch-edge extrapolation numbers (Section 4.2) | regenerated `reporting_audit.json` and `bf16_batch_holdout_tail.csv` |
| Knee counts (Section 4.3) | `data/knee_estimates.csv` |
| Figure 3 | `data/out_of_fold_predictions.csv`; redraw with `paper/scripts/fig3_heldout_predicted_vs_measured.py --package <package root>` |

The offline predictor also runs from the package:

```bash
python3 reproduce.py predict --package . --method-config kvq4 --batch 1 --context 4096
```

## kvbench-joint-results-repro.tar.gz: joint quality–performance results

Requires Python 3 with pyarrow.

```bash
python3 reproduce.py --reproduce . --output <output_dir>
```

The command checks the stored inputs against the package's checksum ledger, then regenerates seven products into `<output_dir>`:

- the per-stage outcome table;
- the joined quality–performance table;
- the (empty) table of qualified configurations;
- the result summary;
- the eleven discordant LongBench v2 pairs;
- the identity mapping;
- the validation report.

Each regenerated file can be compared with the copy stored at the package root, for example `cmp <output_dir>/quality_admission_table.csv quality_admission_table.csv`. The command runs no inference and does not redo any bootstrap.

| Paper item | Source in the package |
|---|---|
| Table 9 (joint outcome) | regenerated `quality_admission_table.csv` |
| Joint outcome and join counts (Section 5.7) | regenerated `joint_result_summary.json`, `performance_quality_join.parquet`, `qualified_compressed_candidates.parquet` |
| Table 7 and Section 5.4 (perplexity) | `inputs/q1a_summary.json`, `inputs/q1b_summary.json` |
| Section 5.5 (LongBench-E) | `inputs/q2a_summary.json` |
| Table 8 and Section 5.6 (LongBench v2) | `inputs/q2b_summary.json`; discordant pairs in regenerated `q2b_discordant_pairs.csv` |
| Tables 2 and 4 (allocated bytes, latency) | `inputs/wall_point_summary.parquet` |
| Figure 1 and Section 3.3 (same-work ratios) | `inputs/wall_same_work_ratios.parquet`; redraw with `paper/scripts/fig1_same_work_ratios.py --table <package root>/inputs/wall_same_work_ratios.parquet` |
| Section 3.4 (capacity points) | `inputs/performance_capacity_amplification.parquet` |
| Table 3 (DRAM traffic, r_DRAM, A, profiler counters) | `inputs/performance_profiler_feature_join.parquet` |

## Package contents

Both packages are published exactly as they were sealed at the end of the study. Their manifests, checksum ledgers, completion markers, and provenance records are included unchanged. File, directory, and field names inside the packages follow the internal stage names used during the study.

The raw timing samples and profiler traces are not included; they are available from the author on request.
