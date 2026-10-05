# Post-hoc review round (2026-10-04)

POST-HOC: everything here was defined after the frozen performance and quality
results were known. Nothing re-runs a timing scan or a quality run, changes a
measured configuration or adapter, or edits frozen evidence. Profiler
durations are never reported as benchmark timing. New outputs are sealed into
fresh content-addressed directories (manifest.json, artifact_inventory.json,
checksums.sha256, COMPLETE) that `scripts/r2_artifact.py` publishes unchanged.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `posthoc_common.py` | host | input-root verification, sealing, SVG helpers |
| `partA_analysis.py` | host, `.phase17-venv` (numpy, pyarrow) | Part A: ceilings, BW_eff, elasticity, steady-state capacity, decomposition |
| `partB_offline.py` | host, `.phase17-venv` (+ ncu CLI, optional) | Part B from existing Phase 15 traces |
| `partB_worker.py` | measurement container | one profiled decode session; optional TurboQuant split override |
| `partB_gpu_diagnostics.py` | host, root, system `python3` | runs every GPU job in its own container, seals the raw output |
| `partB_analysis.py` | host, `.phase17-venv` | Part B analysis of the sealed GPU raw artifact (split diagnostic, traffic-model check) |
| `partC_exclusion_record.py` | host, `.phase17-venv` | KVQuant quality exclusion record (AGENTS.md #12) |
| `partD_final_analyses.py` | host, `.phase17-venv` | final revision: cache-only ceiling S_cache and the KIVI-k4v4 fixed-cost kernel breakdown (imports `partA_analysis.py` unchanged) |
| `partE_kivi_adjusted.py` | host, `.phase17-venv` | final revision: KIVI S_adj with the fixed-cost excess removed (sealed Part A fits only, no refit) |
| `test_partB_split_override.py` | host GPU, small | mechanism test of the split override (no model) |
| `README.md` | | this file |

The files were written in `paper/posthoc/` during the review round, because the
repository was root-owned, and copied byte for byte to `analysis/posthoc_review/`,
the committed location. Each sealed artifact also carries the scripts that produced
it in its `code/` directory, identical to these files. The scripts locate the
repository from their own path, so they work from either directory. The copy was
made file by file (never `cp -a dir/. target`):

```bash
mkdir -p analysis/posthoc_review
for f in README.md posthoc_common.py partA_analysis.py partB_offline.py partB_worker.py partB_gpu_diagnostics.py partB_analysis.py partC_exclusion_record.py partD_final_analyses.py partE_kivi_adjusted.py test_partB_split_override.py; do cp paper/posthoc/$f analysis/posthoc_review/$f; done
```

Rerunning the six CPU analyses on the retained roots (2026-10-05) reproduced every
sealed data file of 3dc84c64, c347e20a, 39efeb1c, 47296f7c, b0969338, and 1998be3e; the only difference
was how the excluded staging directory's path was written in the Part B analysis
exclusions (absolute in the sealed run, relative in the rerun):

```bash
P=.phase17-venv/bin/python
$P analysis/posthoc_review/partA_analysis.py --parent <new_dir>
$P analysis/posthoc_review/partB_offline.py --parent <new_dir>
$P analysis/posthoc_review/partC_exclusion_record.py --parent <new_dir>
$P analysis/posthoc_review/partD_final_analyses.py --parent <new_dir>
$P analysis/posthoc_review/partE_kivi_adjusted.py --parent <new_dir>
$P analysis/posthoc_review/partB_analysis.py artifacts/posthoc/posthoc-b-gpu-20261004t202246814451z-0641de4b-e22fff \
    --part-a paper/posthoc/artifacts/posthoc-a-20261004t143713987904z-0641de4b-ab87cf --parent <new_dir> \
    --excluded-staging artifacts/posthoc/.staging-partb-20261004t150133z=interrupted_duplicate_smoke_launch_not_used
```

## Sealed artifacts

| Artifact | Root SHA-256 | Inputs |
|---|---|---|
| `paper/posthoc/artifacts/posthoc-a-20261004t143713987904z-0641de4b-ab87cf` | `3dc84c64946d748bc4b73b49fa4fad90118dd04b3baa7fe5ad86b5326036d224` | 5605558b (host wall), d7458767 (outer), 641fc02d (Phase 15) |
| `paper/posthoc/artifacts/posthoc-b-offline-20261004t145657101765z-0641de4b-63e796` | `c347e20ac7406e6720a57ad6fc4c62620f55458c856da11c5dd22ac9ebe65cdc` | 641fc02d (Phase 15) |
| `artifacts/posthoc/posthoc-b-gpu-20261004t202246814451z-0641de4b-e22fff` (raw, 22 GB, root-owned) | `4cdbb3aed07f52d885dc8464edf9af20b950603b1979fe9804f14de0fedea119` | measurement container, execution repository ec534d99 |
| `paper/posthoc/artifacts/posthoc-b-analysis-20261004t202531220352z-0641de4b-74462d` | `39efeb1c273b2637a8e8b3ca60edd1f41e476530decbe85a750bf134071092fb` | 4cdbb3ae, 3dc84c64, 5605558b, 641fc02d |
| `paper/posthoc/artifacts/posthoc-c-exclusions-20261004t164203027878z-0641de4b-e16598` | `47296f7c98a0d0da54bb0cc3c60692ecf179ddd97743bcec89285f5126c139f7` | q1a 23d11321, joint 2d609efb |
| `paper/posthoc/artifacts/posthoc-d-20261005t064839061781z-b0ed1058-d6fb96` | `b0969338edac1270050001419faf2ae8b4add0d4c0976cb8fb1e4bae9a7c8d4d` | 5605558b, d7458767, 641fc02d, 3dc84c64 (checked equal) |
| `paper/posthoc/artifacts/posthoc-e-20261005t074653228884z-bbefea1c-dec6cf` | `1998be3eef24582e2e05c8c8c0015c4aac9d6b4bc3b2efd5a028ade6bd5e3db9` | 3dc84c64, 5605558b |

`artifacts/posthoc/.staging-partb-20261004t150133z` is an interrupted duplicate smoke launch (its runner ended after the first job). It is retained in place, not analyzed, and recorded in the Part B analysis exclusions with reason `interrupted_duplicate_smoke_launch_not_used`.

The sealed artifacts stay where they were written (`paper/posthoc/artifacts/` and `artifacts/posthoc/`, not tracked by Git, like all raw outputs) and are published on R2 by root hash. Re-running a script creates a new artifact with a new ID; it never replaces one. R2 receipts are in `docs/evidence/posthoc-review/`.

## Publish to R2 (root, repository root)

```bash
set -a; . ./.env; set +a
mkdir -p docs/evidence/posthoc-review
python3 scripts/r2_artifact.py publish paper/posthoc/artifacts/<artifact> > docs/evidence/posthoc-review/<artifact>-r2-publish.json
python3 scripts/r2_artifact.py verify <root_sha256> > docs/evidence/posthoc-review/<artifact>-r2-verify.json
```

## GPU diagnostics (root, measurement host)

```bash
python3 analysis/posthoc_review/partB_gpu_diagnostics.py plan        # job list, estimates, no GPU
python3 analysis/posthoc_review/partB_gpu_diagnostics.py smoke       # 2 jobs, ~3 min, unsealed
python3 analysis/posthoc_review/partB_gpu_diagnostics.py run --resume <staging dir printed by smoke>
```

- Jobs 1 (TurboQuant split 4 vs 32, nsys, Graph): TQ-4bit/k3v4/3bit x {B1 32K, B1 128K,
  B8 32K, B8 48K} x {4, 32}, plus BF16 at {B1 32K, B1 128K, B8 32K}. B8 128K is
  capacity-infeasible for TurboQuant, so B8 48K (largest Full Scan B = 8 context) replaces it.
- Jobs 2 (traffic-model check, ncu with Phase 15 metrics and defaults): BF16, TQ-4bit,
  KIVI-k4v4 at B8 32K and B16 16K.
- Optional `--with-kvquant-ncu`: KVQuant-4 at the same points (~3.2 h + ~6 h, reports
  ~43 GB + ~79 GB; each job is skipped with a recorded reason if free disk is too low).
- Estimated: ~3.8 h by default, ~13 h with KVQuant. Output while running:
  `artifacts/posthoc/.staging-partb-<stamp>/`; at the end a sealed
  `artifacts/posthoc/posthoc-b-gpu-...` and the R2 commands are printed.
- Sessions are built exactly as in the Full Scan (execution repository ec534d99; BF16
  restores its snapshot, compressed methods reconstruct from the logical prefix). The
  split override swaps the decode split count and scratch only inside each
  `_decode_compressed` call; `test_partB_split_override.py` checks that cache state,
  fingerprints and pointers are unchanged and outputs agree with 4 splits.
