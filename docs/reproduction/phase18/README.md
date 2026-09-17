# Phase 18 CPU reproduction package

This compact package reproduces the frozen Phase 17 summaries and figures. It
does not launch CUDA, download models/data, rerun experiments, refit models, or
evaluate quality.

Default reproduction from the package root:

```bash
python3 reproduce.py reproduce --package . --output /tmp/kvbench-phase18-reproduced
```

Offline prediction example:

```bash
python3 reproduce.py predict --package . --method-config kvq4 --batch 1 --context 4096
```

The default path uses only the Python standard library. It reads frozen CSV
copies of the compact Phase 17 tables, regenerates the reporting audit, six SVG
figures, and four predictor examples. It performs no network or subprocess
operation.

Optional, operator-invoked refit from the repository (not run by Phase 18):

```bash
PYTHONPATH=src .phase17-venv/bin/python -m scripts.phase17_modeling
```

That optional command creates a new append-only Phase 17 bundle and is not
needed to reproduce this package. GPU experiment reproduction remains governed
by the historical phase plans and is intentionally not invoked here.

Bundle ID: `phase18-20260917t024901884906z-a866fa68-6cfb61`. Quality status: `unvalidated`.
