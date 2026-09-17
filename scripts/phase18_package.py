#!/usr/bin/env python3
"""Build and validate the compact CPU-only Phase 18 reproduction bundle."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from kvbench.runtime.artifacts import sha256_file
from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from scripts.r2_artifact import validate_local_artifact
from scripts.phase18_offline import audit, reproduce


ROOT = Path(__file__).resolve().parents[1]
PHASE17_ID = "phase17-20260916t164055992658z-9e20208a-c944c5"
PHASE17_ROOT = ROOT / "artifacts/phase17" / PHASE17_ID
PHASE17_SHA256 = "05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb"
PHASE16_HOST_ROOT = "5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e"
PHASE14_ROOT = "4cd29ea1b94201f493db8cef9ebd01933b4c4f573185eff317ec5af81e9fb000"
PHASE15_ROOT = "641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8"
ARTIFACT_ROOT = ROOT / "artifacts/phase18"
ID_RE = re.compile(r"phase18-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z")


class Phase18PackageError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase18PackageError(f"JSON object required: {path}")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parquet_to_csv(source: Path, target: Path) -> None:
    write_csv(target, [dict(row) for row in pq.read_table(source).to_pylist()])


def source_manifest(execution_head: str) -> dict[str, Any]:
    files = {
        name: sha256_file(PHASE17_ROOT / name)
        for name in (
            "analysis_frame.parquet", "out_of_fold_predictions.parquet",
            "model_comparison.parquet", "knee_estimates.parquet",
            "split_manifest.json", "candidate_spec.json", "input_manifest.json",
            "model_target_status.json", "prediction_metrics.json",
        )
    }
    return {
        "schema_version": "kvbench-phase18-source-manifest-1.0.0",
        "phase18_execution_head": execution_head,
        "phase17_bundle_id": PHASE17_ID,
        "phase17_root_sha256": PHASE17_SHA256,
        "phase17_r2_uri": f"r2://kvbench-artifacts/kvbench/sha256/{PHASE17_SHA256}/",
        "phase16_host_wall_root_sha256": PHASE16_HOST_ROOT,
        "phase14_closure_root_sha256": PHASE14_ROOT,
        "phase15_root_sha256": PHASE15_ROOT,
        "source_files": files,
        "accepted_process_observations": 2205,
        "feasible_logical_points": 441,
        "capacity_infeasible_process_records": 465,
        "infrastructure_replacements": 38,
        "historical_bundles_copied": False,
        "historical_download_performed": False,
    }


def render_audit_note(report: Mapping[str, Any], tail: Sequence[Mapping[str, Any]]) -> str:
    macro = report["macro_metric_definition"]
    bf16 = report["bf16_leave_one_batch_out"]
    domain_lines = "\n".join(
        f"- {row['domain_status']}: n={row['logical_points']}, median={100*row['median_absolute_relative_error']:.3f}%, P95={100*row['p95_relative_error']:.3f}%, max={100*row['maximum_absolute_relative_error']:.3f}%"
        for row in bf16["domain_summary"]
    )
    first = tail[0]
    return f"""# Phase 18 reporting audit

The frozen selected-model labels are explicitly:

- `macro_mean_of_cell_median_relative_errors` = `{macro['median_value']:.15f}`;
- `macro_mean_of_cell_p95_relative_errors` = `{macro['p95_value']:.15f}`.

They are equal-weight means of 11 applicable geometry-holdout cell statistics,
not pooled quantiles. BF16 leave-one-config-out is not applicable and session
holdout remains separate. No target status changed.

The BF16 leave-one-batch logical-point median error is
`{100*bf16['median_absolute_relative_error']:.3f}%` and its P95 is
`{100*bf16['p95_relative_error']:.3f}%`. The largest existing error is held-out
B={first['held_out_batch']}, historical L={first['historical_context']}:
observed `{first['observed_host_wall_ms']:.9f}` ms and predicted
`{first['predicted_host_wall_ms']:.9f}` ms, absolute relative error
`{100*first['absolute_relative_error']:.3f}%`.

{domain_lines}

The stored rows use milliseconds per full-batch decode step, without division
by B. Fold/config joins, historical-L scaling, point-dependent BF16 `r_alloc`,
and finite positive outputs pass. D emits positive latency directly, so no
missing inverse transform exists. The failure is concentrated in short-context
B=1 edge extrapolation and is retained without clipping, exclusion, or refit.
Phase 17 did not persist each fold's fitted parameter vector separately; this
audit therefore binds OOF values to the immutable prediction table, candidate
specification, and execution code rather than misidentifying the final
full-data BF16-D parameters as fold parameters.
"""


def render_research_report(audit_report: Mapping[str, Any]) -> str:
    macro = audit_report["macro_metric_definition"]
    return f"""# Method-conditioned KV-cache decode performance study

## Research question and scope

The study tested whether full-model fixed-L decode latency on one NVIDIA RTX
PRO 6000 Blackwell can be predicted from method-conditioned batch B,
historical context L, and allocated compression ratio `r_alloc`. The scalar
byte law and knee structure were tested rather than assumed. The endpoint is
host-wall milliseconds per full-batch CUDA Graph decode step for the frozen
Llama-3.1-8B-Instruct revision and ten admitted BF16, TurboQuant-vLLM,
project-patched KIVI, and project-patched KVQuant configurations.

## Dataset and custody

Phase 16R supplies 2,205 accepted process observations at 441 feasible logical
points, five independent replicate slots per point, plus 465 explicit
capacity-infeasible records. Thirty-eight infrastructure replacements retain
one accepted observation per slot. No Pilot, admission, profiler, or stopped
campaign timing enters the Phase 17 response table.

## Models and validation

The frozen candidates were E (scalar byte law), RQ2 (B/L/r surface), D
(positive knee response surface), F_shape (D plus static byte-shape features),
and diagnostic-only F_diagnostic (observed kernel count). Four distinct tests
were retained: leave-one-batch, leave-one-compressed-config, leave-context-band,
and repeat-session holdout. D is the scientific selection and deployment
default, but its outer selection score is not an unbiased evaluation of a
further model-selection procedure.

The primary labels are equal-cell macro means over 11 applicable geometry
cells: median-relative-error `{macro['median_value']:.15f}` and P95-relative-
error `{macro['p95_value']:.15f}`. They are not pooled quantiles. The frozen
5% and 10% targets are missed. Same-work sign accuracy remains 425/456
(93.2018%, missed), pairwise ranking remains 346/513 (67.4464%, missed), and
knee error is not evaluable without independent truth. TurboQuant
leave-one-config transfer remains 29.850% median and 44.554% P95; KIVI is
2.915%/14.117% and KVQuant 4.651%/16.210%. The approximately 67.45% ranking
accuracy is not a reliable automatic method selector.

E to RQ2 to D is a relative improvement among tested candidates under the
declared rule, not universal proof that B/L/r is sufficient. F_shape's lack of
robust improvement does not show metadata or workspace costs are irrelevant.
F_diagnostic is not deployable because held-out-run kernel count is required.

## Error tail and knees

The BF16 batch-holdout median is small but its P95 is 1625.690%. The audit
finds correct full-batch units, joins, scaling, point-dependent BF16 ratio, and
finite positive outputs. The tail is a real D-model predictive failure at
short-context B=1 edge extrapolation, retained in every primary metric and
figure. Twenty-three of 50 local curves have identifiable in-range knees;
27 are weak/non-identifiable, insufficient-span, or prefer linear/constant
descriptions. The 1,000 shared-session bootstrap is conditional on five sampled
sessions, and no independent knee truth exists.

## Mechanism and claim boundaries

Phase 14 found 0/14 fully identifiable comparisons supporting the complete
pure launch-floor-only criterion. Phase 15 directly found fewer CPU submissions
and lower GPU idle in 16/16 anchor pairs, while retaining the
`method_specific_mixed` interpretation. Its HBM/L2 data remain scoped only to
B=1, historical L=131071, Graph mode. They are not filled into Full Scan rows
and do not establish critical-path causality.

Quality has not been evaluated. All predicted ratios are performance-only and
quality-unvalidated; capacity feasibility is separate. The next quality work
must follow the existing performance-freeze protocol and human approval gate.
"""


def render_readme(bundle_id: str) -> str:
    return f"""# Phase 18 CPU reproduction package

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

Bundle ID: `{bundle_id}`. Quality status: `unvalidated`.
"""


def render_predictor_doc() -> str:
    return """# Offline predictor

The default model is frozen model D, evaluated separately by method family:

`T = tau(B,r) + s(B,r) * max(L/131071 - lambda(B,r), 0)`.

For each of tau, s, and lambda, standardized `log(B)` and `log(r_alloc)` form
the basis `[1, z_B, z_r, z_B*z_r]`. Tau and slope use exponentiated bounded
linear predictors; lambda uses `0.02 + 0.96*sigmoid(.)`. Exact scaler means,
scales, coefficients, bounds, residual intervals, training domains, and model
hashes are in `models/*-D.json`. Output units are milliseconds per full-batch
decode step.

For a supported configuration, `r_alloc` is calculated at the requested B/L
from the frozen allocated-byte formula, including metadata, residual/sink,
sparse, and workspace storage. It is point-dependent. An explicit numeric r
requires a method family and does not establish validity for an unseen
quantizer. Only the three measured compressed configurations per family are
within configuration support.

The interval is the stored empirical new-process OOF log-residual interval,
not fitted-mean uncertainty. The BF16 latency and ratio are fully predicted;
no measured baseline is substituted. Domain status does not assert memory
feasibility. Extrapolations are never clipped. All outputs retain
`quality_status=unvalidated` and `performance_claim_eligible=false`.
"""


def payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in excluded)


def seal(stage: Path, bundle_id: str, execution_head: str) -> Path:
    write_exclusive(stage / "manifest.json", json_bytes({
        "schema_version": "kvbench-phase18-artifact-manifest-1.0.0",
        "run_id": bundle_id, "status": "PASS", "created_at_utc": utc_now(),
        "execution_git_sha": execution_head, "source_phase17_root_sha256": PHASE17_SHA256,
        "append_only": True, "complete_written_last": True,
        "cpu_only": True, "model_refit": False, "gpu_launched": False,
        "quality_status": "unvalidated", "performance_claim_eligible": False,
    }))
    items = [{"path": path.relative_to(stage).as_posix(), "role": "phase18_cpu_reproduction", "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in payload_paths(stage, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"})]
    write_exclusive(stage / "artifact_inventory.json", json_bytes({"schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": bundle_id, "files": items, "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"]}))
    ledger = "".join(f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n" for path in payload_paths(stage, {"checksums.sha256", "COMPLETE"})).encode()
    write_exclusive(stage / "checksums.sha256", ledger)
    write_exclusive(stage / "COMPLETE", json_bytes({"schema_version": "kvbench-completion-1.0.0", "run_id": bundle_id, "status": "PASS", "manifest_sha256": sha256_file(stage / "manifest.json"), "artifact_inventory_sha256": sha256_file(stage / "artifact_inventory.json"), "checksum_ledger_path": "checksums.sha256", "checksum_ledger_sha256": sha256_file(stage / "checksums.sha256"), "written_last": True}))
    final = ARTIFACT_ROOT / bundle_id
    rename_noreplace(stage, final)
    for path in sorted(final.rglob("*"), reverse=True): path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate_bundle(final)
    return final


def validate_bundle(root: Path) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    required = {"README.md", "predictor.md", "research_report.md", "reporting_audit.json", "reporting_audit.md", "bf16_batch_holdout_tail.csv", "reproduce.py", "reproduction_manifest.json", "source_manifest.json", "quality_release_manifest_draft.json", "models/index.json", "data/out_of_fold_predictions.csv", "data/model_comparison.csv", "data/knee_estimates.csv", "data/analysis_frame.csv", "data/split_manifest.json", "manifest.json", "artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    missing = sorted(value for value in required if not (root / value).is_file())
    if missing: raise Phase18PackageError(f"bundle files missing: {missing}")
    report, tail, _ = audit(root)
    if abs(report["macro_metric_definition"]["median_value"] - 0.0791985632276268) > 1e-15 or abs(report["macro_metric_definition"]["p95_value"] - 1.645283992575687) > 1e-15:
        raise Phase18PackageError("macro metrics differ")
    if len(tail) != 20 or report["bf16_leave_one_batch_out"]["diagnosis"] != "real_predictive_failure_concentrated_in_B1_short_context_edge_extrapolation":
        raise Phase18PackageError("BF16 tail audit differs")
    manifest = strict_json(root / "manifest.json")
    if manifest.get("gpu_launched") is not False or manifest.get("model_refit") is not False or manifest.get("quality_status") != "unvalidated":
        raise Phase18PackageError("bundle scope differs")
    return {"status": "PASS", "root_sha256": artifact.root_sha256, "object_count": len(artifact.files), "tail_rows": len(tail)}


def build(bundle_id: str | None = None) -> Path:
    if validate_local_artifact(PHASE17_ROOT, environ={}).root_sha256 != PHASE17_SHA256:
        raise Phase18PackageError("Phase 17 root differs")
    execution_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identifier = bundle_id or f"phase18-{datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%f')[:22]}z-{execution_head[:8]}-{secrets.token_hex(3)}"
    if ID_RE.fullmatch(identifier) is None: raise Phase18PackageError("invalid bundle ID")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = ARTIFACT_ROOT / f".{identifier}.{secrets.token_hex(8)}.staging"
    if stage.exists() or (ARTIFACT_ROOT / identifier).exists(): raise Phase18PackageError("bundle ID exists")
    stage.mkdir(mode=0o700)
    (stage / "data").mkdir(); (stage / "models").mkdir()
    sources = source_manifest(execution_head)
    write_exclusive(stage / "source_manifest.json", json_bytes(sources))
    for stem in ("analysis_frame", "out_of_fold_predictions", "model_comparison", "knee_estimates"):
        parquet_to_csv(PHASE17_ROOT / f"{stem}.parquet", stage / "data" / f"{stem}.csv")
    for name in ("split_manifest.json", "candidate_spec.json", "input_manifest.json", "model_target_status.json", "prediction_metrics.json"):
        shutil.copy2(PHASE17_ROOT / name, stage / "data" / name)
    for model in (PHASE17_ROOT / "models").glob("*.json"): shutil.copy2(model, stage / "models" / model.name)
    shutil.copy2(ROOT / "scripts/phase18_offline.py", stage / "reproduce.py")
    mechanism = {"schema_version": "kvbench-phase18-mechanism-summary-1.0.0", "phase14_closure_root_sha256": PHASE14_ROOT, "phase14_identifiable_comparisons": 14, "phase14_launch_floor_support": 0, "phase14_interpretation": "heterogeneous_not_pure_launch_floor", "phase15_root_sha256": PHASE15_ROOT, "phase15_cpu_submission_reduced_pairs": 16, "phase15_gpu_idle_reduced_pairs": 16, "phase15_classification": "method_specific_mixed", "phase15_common_point_label": "B1/L131071 Graph", "profiler_features_scope": "exact_common_point_only"}
    write_exclusive(stage / "data/mechanism_summary.json", json_bytes(mechanism))
    report, tail, _ = audit(stage)
    write_exclusive(stage / "reporting_audit.json", json_bytes(report))
    write_csv(stage / "bf16_batch_holdout_tail.csv", tail)
    write_exclusive(stage / "reporting_audit.md", render_audit_note(report, tail).encode())
    write_exclusive(stage / "README.md", render_readme(identifier).encode())
    write_exclusive(stage / "predictor.md", render_predictor_doc().encode())
    write_exclusive(stage / "research_report.md", render_research_report(report).encode())
    write_exclusive(stage / "requirements.txt", b"# Default reproduction uses Python 3.11+ standard library only.\n# Optional refit environment: numpy==2.5.1, pyarrow==25.0.0\n")
    quality = {"schema_version": "kvbench-phase18-quality-release-manifest-draft-1.0.0", "status": "DRAFT_NOT_AUTHORITY", "quality_state": "LOCKED", "performance_data_frozen": "absent", "quality_evaluation_executed": False, "next_authorized_task": "separate_QP-0_performance_freeze", "qp0_prerequisites": ["verify_no_active_performance_process", "all_planned_runs_terminal", "raw_samples_present", "artifact_checksums_present", "record_git_working_tree", "record_container_digest", "record_gpu_hardware_manifest", "create_performance_inventory", "create_freeze_git_tag", "create_locked_hot_path_list", "populate_quality_contract_from_manifest", "stop_at_human_approval_gate"], "performance_freeze_created": False, "performance_freeze_tag_created": False, "quality_contract_approved": False}
    write_exclusive(stage / "quality_release_manifest_draft.json", json_bytes(quality))
    with tempfile.TemporaryDirectory(prefix="phase18-reproduction-") as tmp:
        result = reproduce(stage, Path(tmp))
        generated = Path(tmp)
        shutil.copytree(generated / "figures", stage / "figures")
        shutil.copy2(generated / "predictor_examples.json", stage / "prediction_examples.json")
        shutil.copy2(generated / "reproduction_result.json", stage / "default_reproduction_result.json")
    write_exclusive(stage / "reproduction_manifest.json", json_bytes({"schema_version": "kvbench-phase18-reproduction-manifest-1.0.0", "bundle_id": identifier, "execution_git_sha": execution_head, "default_command": "python3 reproduce.py reproduce --package . --output /tmp/kvbench-phase18-reproduced", "predictor_command": "python3 reproduce.py predict --package . --method-config kvq4 --batch 1 --context 4096", "optional_refit_command": "PYTHONPATH=src .phase17-venv/bin/python -m scripts.phase17_modeling", "optional_refit_executed": False, "default_result": result, "forbidden_runtime_imports": ["torch", "cuda", "kvbench.adapters"], "gpu_launched": False, "network_accessed": False, "quality_evaluation_executed": False}))
    final = seal(stage, identifier, execution_head)
    print(json.dumps({"status": "PASS", "bundle_id": identifier, "path": str(final), **validate_bundle(final)}, sort_keys=True))
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_bundle(args.validate), sort_keys=True)); return 0
    build(args.bundle_id); return 0


if __name__ == "__main__":
    raise SystemExit(main())
