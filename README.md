# kvbench

kvbench is the measurement harness behind the paper *Bytes Are Not Latency: Allocated Size, Measured Traffic, Decode Speed, and Quality of KV-Cache Quantization on a Single Blackwell GPU* (`paper/manuscript.md`).

It integrates adapted implementations of TurboQuant, KIVI, and KVQuant into one full-model decode harness for Llama-3.1-8B-Instruct on a single NVIDIA RTX PRO 6000 Blackwell GPU. Ten configurations (BF16 and three settings per method) are compared on allocated cache bytes, measured DRAM traffic, same-work decode latency, and cache-sensitive quality. In the paper, allocated compression is below nominal compression for every configuration, and cache-path DRAM traffic at a common profiled point is 2.8–5.5× lower than BF16. Even so, no compressed configuration is faster than BF16 at any of the 357 same-work points. Under a staged quality protocol at physical batch size 1, no compressed configuration passes every stage.

## Installation

The package needs Python 3.11 or newer:

```bash
git clone https://github.com/fzlzjerry/kvbench.git
cd kvbench
make install
```

`make install` creates `.venv` with Python 3.12 and installs the hash-pinned environment used for the paper. It installs PyTorch 2.12.1 for CUDA 13.0 and Triton 3.7.1 from `preflight/requirements-e00.txt`, and Transformers 4.57.6, NumPy, and safetensors from `preflight/requirements-phase3.txt` into `.deps`. The same locks define the measurement container in `docker/measurement.Dockerfile`.

KIVI and KVQuant run from their upstream repositories at the commits pinned in `third_party/LOCK.json`, with the patches in `third_party/patches/` applied. The TurboQuant kernels are vendored from vLLM v0.25.1 in `src/kvbench/third_party/vllm_turboquant/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/kvbench/` | Harness: method adapters (BF16, TurboQuant, KIVI, KVQuant), static caches and byte accounting, fixed-length and growing-context runners, CUDA Graph capture, configuration schemas, vendored TurboQuant kernels |
| `third_party/` | Pinned upstream sources (`third_party/LOCK.json`) and the KIVI and KVQuant patches |
| `configs/` | Model, hardware, method, scan-plan, and quality-evaluation configurations |
| `scripts/` | Method validation drivers, KVQuant calibration and validation workers, prefix-state construction, offline modeling reproduction |
| `reference/` | Numerical reference fixtures used by the golden tests |
| `tests/` | CPU unit and schema tests, CUDA tests, and CUDA Graph tests |
| `docker/` | Container definitions for the measurement, quality, calibration, and reference environments |
| `preflight/` | Hash-pinned Python dependency locks and shared file-writing helpers used by the scripts |
| `docs/` | Experiment contract and measurement protocol; their hashes are part of every scan-plan fingerprint |
| `paper/` | Manuscript, figures, and figure scripts |

## Tests

```bash
make test         # CPU unit and schema tests (GPUs hidden)
make test-cuda    # CUDA kernel, adapter, and allocation tests (needs the GPU)
make test-graph   # CUDA Graph capture and replay tests (needs the GPU)
```

Some tests read local inputs that are not in this repository. These are the KVQuant calibration bundle under `calibration/kvquant/`, and the upstream KIVI and KVQuant checkouts at the pinned commits.

## Reproducing the paper

### Modeling and joint quality–performance results (CPU only)

Two CPU reproduction packages are attached to the [v1.0 release](https://github.com/fzlzjerry/kvbench/releases/tag/v1.0); `RELEASE_ASSETS_README.md` in the release lists which tables and figures each one covers. Unpack an archive and run from its top-level directory:

```bash
# Modeling package (Python standard library only)
python3 reproduce.py reproduce --package . --output <output_dir>

# Joint quality–performance package (needs pyarrow)
python3 reproduce.py --reproduce . --output <output_dir>
```

`<output_dir>` must not exist yet. Figures 1 and 3 can be redrawn from the same packages:

```bash
python3 paper/scripts/fig1_same_work_ratios.py \
    --table <joint-results package>/inputs/wall_same_work_ratios.parquet \
    --output paper/figures/fig1_same_work_ratios.svg
python3 paper/scripts/fig3_heldout_predicted_vs_measured.py \
    --package <modeling package> \
    --output paper/figures/fig3_heldout_predicted_vs_measured.svg
```

Figure 2 and the raw timing samples come from the raw measurements, which are available from the author on request.

### GPU timing

The timing, profiling, and quality runs used experiment drivers that check recorded validation and provenance files before they start. The drivers, their execution orders, and those records are on the `main` branch at commit `0641de4b`, not on this release branch. To re-run the measurements:

1. Use one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB) and Docker with GPU support.
2. Build the measurement image from `docker/measurement.Dockerfile`. The paper used image `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
3. Obtain `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659` (gated under the Llama 3.1 Community License).
4. Check out KIVI and KVQuant at the commits in `third_party/LOCK.json` and apply the patches in `third_party/patches/`. Build the KVQuant calibration bundle with the calibration worker in `scripts/`.
5. Check out `0641de4b` and run the scan drivers there. The scan plan is `configs/plans/full_scan.yaml`.

## Paper

The manuscript is `paper/manuscript.md`, with figures in `paper/figures/` and figure scripts in `paper/scripts/`.

## License

The code is released under the Apache License 2.0 (`LICENSE`). Vendored and patched third-party code keeps its upstream terms, listed in `NOTICE`.

## Research history

This release branch contains the code, configurations, tests, and paper. The complete research history is on the `main` branch at commit `0641de4b`. It includes the measurement protocol and decision records, the execution plans, the run receipts, and the scan drivers.
