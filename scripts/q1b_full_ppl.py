#!/usr/bin/env python3
"""Execute approved physical-B=1 Q1B Full PPL for BF16 and KIVI k4v4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

from scripts import q0_cache_sensitive_correctness as q0
from scripts import q1a_fast_ppl as q1a


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("bf16", "k4v4")
ALL_CONFIGS = q0.CONFIGS
DATASETS = ("wikitext2_test", "c4_validation")
PREFIX_LENGTHS = (4096, 16384, 24576, 28672, 32768, 65536, 98304, 130560)
ANCHORS_PER_LENGTH = 64
SCORED_HORIZON = 256
BURN_IN_TOKENS = 1
AUTHORIZATION = ROOT / "docs/evidence/q1b/execution-authorization.json"
Q1A_EVIDENCE = ROOT / "docs/evidence/q1a/fast-ppl.json"
Q1A_EVIDENCE_SHA256 = "91c101df6144b1c9f0f2b1e2d2878279c6c1434cce804e033728a1eb87103a9d"
Q1A_CAMPAIGN = ROOT / "artifacts/q1a/q1a-20260919t141635000000z-6e0c7803-4c6a18f2"
Q1A_ROOT = "23d11321522bc9d997f2d26f9f6c4110e3332d0aedc47e3f53b8e1f7f547a5d0"
CONTRACT_BUNDLE = q0.CONTRACT_BUNDLE
ANCHOR_MANIFEST = CONTRACT_BUNDLE / "selected_inputs/ppl/anchor_manifest.json"
INPUT_MANIFEST = CONTRACT_BUNDLE / "input_manifest.json"
QUALITY_MARGINS = CONTRACT_BUNDLE / "quality_margins.json"
FAST_STATUS = {
    "bf16": "baseline_valid",
    "tq_4bit_nc": "inconclusive",
    "tq_k3v4_nc": "fail",
    "tq_3bit_nc": "fail",
    "k4v4": "pass",
    "k2v4": "fail",
    "k2v2": "fail",
    "kvq4": "fail",
    "kvq3": "fail",
    "kvq2": "fail",
}


class Q1BError(RuntimeError):
    """Q1B failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return q0.load_json(path)


def sha256_file(path: Path) -> str:
    return q1a.sha256_file(path)


def _manifest_file_hash(relative: str) -> str:
    manifest = load_json(INPUT_MANIFEST)
    matches = [item for item in manifest["files"] if item["path"] == relative]
    if len(matches) != 1:
        raise Q1BError(f"input manifest path missing: {relative}")
    path = CONTRACT_BUNDLE / relative
    if sha256_file(path) != matches[0]["sha256"]:
        raise Q1BError(f"frozen input changed: {relative}")
    return matches[0]["sha256"]


def verify_authority() -> dict[str, Any]:
    q1a.verify_amendment()
    if sha256_file(Q1A_EVIDENCE) != Q1A_EVIDENCE_SHA256:
        raise Q1BError("Q1A eligibility evidence changed")
    q1a_evidence = load_json(Q1A_EVIDENCE)
    observed = {row["configuration"]: row["fast_stage_status"] for row in q1a_evidence["decisions"]}
    if observed != FAST_STATUS or q1a_evidence.get("q1b_eligible_configurations") != ["k4v4"]:
        raise Q1BError("Q1A eligibility differs")
    from scripts.r2_artifact import validate_local_artifact
    artifact = validate_local_artifact(Q1A_CAMPAIGN)
    if artifact.root_sha256 != Q1A_ROOT:
        raise Q1BError("Q1A local root differs")
    authorization = load_json(AUTHORIZATION)
    expected = {
        "original_contract_id": q1a.BASE_CONTRACT_ID,
        "original_contract_sha256": q1a.BASE_CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID,
        "amendment_sha256": q1a.AMENDMENT_SHA256,
        "q1a_root_sha256": Q1A_ROOT,
        "authorized_configurations": list(CONFIGS),
        "authorization_scope": ["Q1B_FULL_PPL_B1"],
        "physical_quality_batch_size": 1,
        "original_cross_batch_gate": "failed",
        "quality_transfer_to_other_batches": "not_established",
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise Q1BError(f"Q1B authorization differs: {key}")
    return {
        "status": "PASS",
        "q1a_root_sha256": artifact.root_sha256,
        "authorization_sha256": sha256_file(AUTHORIZATION),
    }


def eligibility() -> dict[str, Any]:
    rows = []
    for configuration in ALL_CONFIGS:
        status = FAST_STATUS[configuration]
        if configuration == "bf16":
            q1b_status = "paired_baseline_required"
            selected = True
        elif configuration == "k4v4":
            q1b_status = "selected_fast_pass"
            selected = True
        else:
            q1b_status = "not_run_due_to_fast_stage_status"
            selected = False
        rows.append({
            "configuration": configuration,
            "fast_stage_status": status,
            "selected_for_q1b": selected,
            "q1b_status": q1b_status,
        })
    return {
        "schema_version": "kvbench-q1b-eligibility-1.0.0",
        "source_q1a_root_sha256": Q1A_ROOT,
        "selected_configurations": list(CONFIGS),
        "rows": rows,
    }


def full_plan() -> dict[str, Any]:
    _manifest_file_hash("selected_inputs/ppl/anchor_manifest.json")
    manifest = load_json(ANCHOR_MANIFEST)
    anchors: list[dict[str, Any]] = []
    streams: dict[str, dict[str, Any]] = {}
    overlap = {"exact_full_observations": 0, "same_prefix_partial_horizon": 0}
    for dataset in DATASETS:
        relative = f"selected_inputs/ppl/{dataset}/token_stream.i32.gz"
        file_sha = _manifest_file_hash(relative)
        source = manifest["datasets"][dataset]
        streams[dataset] = {
            "path": relative,
            "file_sha256": file_sha,
            "logical_stream_sha256": source["stream_sha256"],
            "stream_token_count": source["stream_token_count"],
        }
        fast_by_id = {
            row["anchor_id"]: row
            for rows in source["stages"]["fast"].values()
            for row in rows
        }
        for length in PREFIX_LENGTHS:
            rows = source["stages"]["full"][str(length)]
            if len(rows) != ANCHORS_PER_LENGTH:
                raise Q1BError("Full-PPL anchor cardinality differs")
            for row in rows:
                if row["scored_token_count"] != SCORED_HORIZON:
                    raise Q1BError("Full-PPL scored horizon differs")
                if row["prefix_end"] - row["prefix_start"] != length:
                    raise Q1BError("Full-PPL prefix length differs")
                if row["burn_in_index"] != row["prefix_end"]:
                    raise Q1BError("Full-PPL burn-in split differs")
                if row["target_start"] != row["burn_in_index"] + 1:
                    raise Q1BError("Full-PPL target split differs")
                if row["target_end_exclusive"] - row["target_start"] != SCORED_HORIZON:
                    raise Q1BError("Full-PPL target horizon differs")
                if row["scoring_mask"] != {"value": 1, "length": SCORED_HORIZON}:
                    raise Q1BError("Full-PPL scoring mask differs")
                if length + BURN_IN_TOKENS + SCORED_HORIZON > 131072:
                    raise Q1BError("Full-PPL model-length bound exceeded")
                fast = fast_by_id.get(row["anchor_id"])
                if fast is not None:
                    same_prefix = all(fast[key] == row[key] for key in ("prefix_start", "prefix_end", "target_start"))
                    if same_prefix and fast["target_end_exclusive"] == row["target_end_exclusive"]:
                        overlap["exact_full_observations"] += 1
                    elif same_prefix:
                        overlap["same_prefix_partial_horizon"] += 1
                anchors.append({"dataset": dataset, "prefix_length": length, **row})
    per_configuration = len(DATASETS) * len(PREFIX_LENGTHS) * ANCHORS_PER_LENGTH
    if len(anchors) != per_configuration:
        raise Q1BError("Full-PPL plan cardinality differs")
    return {
        "schema_version": "kvbench-q1b-full-ppl-plan-1.0.0",
        "original_contract_id": q1a.BASE_CONTRACT_ID,
        "effective_amendment_id": q1a.AMENDMENT_ID,
        "physical_model_batch_size": 1,
        "execution_mode": "cache_sensitive_incremental_teacher_forcing",
        "burn_in_decode_tokens": BURN_IN_TOKENS,
        "scored_horizon": SCORED_HORIZON,
        "datasets": list(DATASETS),
        "prefix_lengths": list(PREFIX_LENGTHS),
        "anchors_per_length": ANCHORS_PER_LENGTH,
        "anchors_per_configuration": per_configuration,
        "scored_tokens_per_configuration": per_configuration * SCORED_HORIZON,
        "configuration_count": len(CONFIGS),
        "total_anchor_units": per_configuration * len(CONFIGS),
        "total_scored_tokens": per_configuration * SCORED_HORIZON * len(CONFIGS),
        "fast_full_overlap": overlap,
        "streams": streams,
        "anchors": anchors,
    }


def _unit_dir(campaign: Path, configuration: str, anchor: Mapping[str, Any]) -> Path:
    return campaign / "units" / configuration / anchor["dataset"] / f"l{anchor['prefix_length']}" / anchor["anchor_id"]


def _unit_complete(campaign: Path, configuration: str, anchor: Mapping[str, Any]) -> bool:
    root = _unit_dir(campaign, configuration, anchor)
    return (root / "result.json").is_file() and (root / "COMPLETE").is_file()


def _finalize_unit(campaign: Path, configuration: str, anchor: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    root = _unit_dir(campaign, configuration, anchor)
    if root.exists():
        raise Q1BError("Full-PPL unit already exists")
    root.mkdir(parents=True)
    q0.write_json_new(root / "result.json", result)
    q0.write_json_new(root / "COMPLETE", {
        "schema_version": "kvbench-q1b-full-ppl-unit-complete-1.0.0",
        "configuration": configuration,
        "dataset": anchor["dataset"],
        "prefix_length": anchor["prefix_length"],
        "anchor_id": anchor["anchor_id"],
        "status": result["status"],
        "result_sha256": sha256_file(root / "result.json"),
        "written_last": True,
    })


def _accept_anchor_result(campaign: Path, configuration: str, anchor: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    if result.get("status") != "PASS" or result.get("scored_token_count") != SCORED_HORIZON:
        raise Q1BError("successful Full-PPL path produced an invalid result")
    _finalize_unit(campaign, configuration, anchor, result)


def run_worker(campaign: Path, configuration: str, execution_head: str) -> dict[str, Any]:
    if configuration not in CONFIGS:
        raise Q1BError("configuration is outside Q1B authorization")
    if os.environ.get("KVBENCH_QUALITY_IMAGE_DIGEST") != q0.QUALITY_IMAGE:
        raise Q1BError("Quality image identity differs")
    observed_head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if observed_head != execution_head:
        raise Q1BError("Q1B execution HEAD differs")
    if subprocess.run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=ROOT, check=True, capture_output=True, text=True).stdout:
        raise Q1BError("Q1B execution source is not clean")
    verify_authority()
    q0.verify_locked_paths()
    plan = full_plan()
    contract = load_json(q0.CONTRACT)["quality_contract"]
    fingerprints = {item["method_config_id"]: item["method_config_fingerprint"] for item in contract["configurations"]["items"]}

    import gc
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    streams = {
        dataset: q1a.read_i32_gzip(CONTRACT_BUNDLE / plan["streams"][dataset]["path"])
        for dataset in DATASETS
    }
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    completed = 0
    failed = 0
    with torch.inference_mode(), forced_flash_execution():
        for anchor in plan["anchors"]:
            if _unit_complete(campaign, configuration, anchor):
                completed += 1
                continue
            try:
                result = q1a.score_anchor(
                    loaded,
                    configuration,
                    streams[anchor["dataset"]],
                    anchor,
                    fingerprints[configuration],
                    expected_horizon=SCORED_HORIZON,
                    result_schema_version="kvbench-q1b-full-ppl-anchor-1.0.0",
                )
                result["quality_stage"] = "full_ppl"
                result["source_q1a_root_sha256"] = Q1A_ROOT
                _accept_anchor_result(campaign, configuration, anchor, result)
            except BaseException as error:
                failed += 1
                if _unit_complete(campaign, configuration, anchor):
                    raise
                _finalize_unit(campaign, configuration, anchor, {
                    "schema_version": "kvbench-q1b-full-ppl-anchor-failure-1.0.0",
                    "status": "FAIL",
                    "configuration": configuration,
                    "dataset": anchor["dataset"],
                    "prefix_length": anchor["prefix_length"],
                    "anchor_id": anchor["anchor_id"],
                    "failure_class": "runtime_or_correctness_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "retry_permitted": False,
                })
                raise
            completed += 1
            print(json.dumps({"configuration": configuration, "completed": completed, "total": len(plan["anchors"])}, sort_keys=True), flush=True)
            del result
            gc.collect()
            torch.cuda.empty_cache()
    return {"configuration": configuration, "completed": completed, "failed": failed}


def _absolute_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = sum(int(row["scored_token_count"]) for row in rows)
    nll_sum = sum(float(row["nll_sum"]) for row in rows)
    mean_nll = nll_sum / count
    return {
        "anchor_count": len(rows),
        "scored_token_count": count,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "cluster_count": len({str(row["cluster_id"]) for row in rows}),
    }


def _paired_summary(rows: Sequence[Mapping[str, Any]], margins: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = margins["bootstrap"]
    stats = q1a.paired_cluster_bootstrap(rows, seed=int(bootstrap["seed"]), draws=int(bootstrap["draws"]))
    stats.update({
        "relative_ppl_change": math.expm1(stats["point_delta_nll"]),
        "ci_lower_relative_ppl_change": math.expm1(stats["ci_lower_delta_nll"]),
        "ci_upper_relative_ppl_change": math.expm1(stats["ci_upper_delta_nll"]),
        "anchor_count": len(rows),
        "scored_token_count": sum(int(row["scored_token_count"]) for row in rows),
        "cluster_count": len({str(row["cluster_id"]) for row in rows}),
    })
    return stats


def classify_full(dataset_stats: Sequence[Mapping[str, Any]], length_stats: Sequence[Mapping[str, Any]], margins: Mapping[str, Any]) -> tuple[str, list[str]]:
    margin = math.log1p(float(margins["ppl"]["global_relative_ppl_increase_max"]))
    hard = float(margins["ppl"]["length_hard_fail_relative_increase"])
    reasons: list[str] = []
    hard_rows = [row for row in length_stats if float(row["relative_ppl_change"]) > hard]
    if hard_rows:
        reasons.extend(f"length_hard_fail:{row['dataset']}:L{row['prefix_length']}" for row in hard_rows)
        return "fail", reasons
    states = []
    for row in dataset_stats:
        if float(row["ci_lower_delta_nll"]) > margin:
            state = "fail"
        elif float(row["ci_upper_delta_nll"]) <= margin:
            state = "pass"
        else:
            state = "inconclusive"
        states.append(state)
        reasons.append(f"dataset_{state}:{row['dataset']}")
    if "fail" in states:
        return "fail", reasons
    if states and all(state == "pass" for state in states):
        return "pass", reasons
    return "inconclusive", reasons


def analyze(campaign: Path) -> dict[str, Any]:
    plan = load_json(campaign / "full_ppl_plan.json")
    margins = q1a.quality_margins()
    results: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for configuration in CONFIGS:
        for anchor in plan["anchors"]:
            path = _unit_dir(campaign, configuration, anchor) / "result.json"
            if not path.is_file():
                failures.append({"configuration": configuration, "anchor_id": anchor["anchor_id"], "reason": "missing"})
                continue
            result = load_json(path)
            if result.get("status") != "PASS":
                failures.append({"configuration": configuration, "anchor_id": anchor["anchor_id"], "reason": result.get("failure_class", "failed")})
                continue
            key = (configuration, anchor["dataset"], int(anchor["prefix_length"]), anchor["anchor_id"])
            results[key] = result
    baseline_valid = (
        sum(key[0] == "bf16" for key in results) == len(plan["anchors"])
        and not any(item["configuration"] == "bf16" for item in failures)
    )
    absolute: dict[str, Any] = {}
    for configuration in CONFIGS:
        own = [row for key, row in results.items() if key[0] == configuration]
        absolute[configuration] = {
            "aggregate": _absolute_summary(own) if own else {},
            "datasets": [
                {"dataset": dataset, **_absolute_summary([row for row in own if row["dataset"] == dataset])}
                for dataset in DATASETS if any(row["dataset"] == dataset for row in own)
            ],
            "lengths": [
                {"dataset": dataset, "prefix_length": length, **_absolute_summary([
                    row for row in own if row["dataset"] == dataset and int(row["prefix_length"]) == length
                ])}
                for dataset in DATASETS for length in PREFIX_LENGTHS
                if any(row["dataset"] == dataset and int(row["prefix_length"]) == length for row in own)
            ],
        }
    paired_rows: list[dict[str, Any]] = []
    for anchor in plan["anchors"]:
        base_key = ("bf16", anchor["dataset"], int(anchor["prefix_length"]), anchor["anchor_id"])
        method_key = ("k4v4", anchor["dataset"], int(anchor["prefix_length"]), anchor["anchor_id"])
        if base_key not in results or method_key not in results:
            continue
        base, method = results[base_key], results[method_key]
        delta = float(method["mean_nll"]) - float(base["mean_nll"])
        paired_rows.append({
            "dataset": anchor["dataset"],
            "prefix_length": int(anchor["prefix_length"]),
            "anchor_id": anchor["anchor_id"],
            "cluster_id": anchor["cluster_id"],
            "scored_token_count": int(method["scored_token_count"]),
            "bf16_mean_nll": float(base["mean_nll"]),
            "k4v4_mean_nll": float(method["mean_nll"]),
            "delta_nll": delta,
            "relative_ppl_change": math.expm1(delta),
        })
    complete_pairing = baseline_valid and len(paired_rows) == len(plan["anchors"]) and not failures
    dataset_stats: list[dict[str, Any]] = []
    length_stats: list[dict[str, Any]] = []
    aggregate_stats: dict[str, Any] = {}
    if complete_pairing:
        for dataset in DATASETS:
            selected = [row for row in paired_rows if row["dataset"] == dataset]
            dataset_stats.append({"dataset": dataset, **_paired_summary(selected, margins)})
        for dataset in DATASETS:
            for length in PREFIX_LENGTHS:
                selected = [row for row in paired_rows if row["dataset"] == dataset and row["prefix_length"] == length]
                stats = _paired_summary(selected, margins)
                stats.update({
                    "dataset": dataset,
                    "prefix_length": length,
                    "review_threshold_exceeded": stats["relative_ppl_change"] > float(margins["ppl"]["length_review_relative_increase"]),
                    "hard_fail_threshold_exceeded": stats["relative_ppl_change"] > float(margins["ppl"]["length_hard_fail_relative_increase"]),
                })
                length_stats.append(stats)
        aggregate_stats = _paired_summary(paired_rows, margins)
        aggregate_stats["aggregation"] = "token_weighted_over_frozen_equal_cardinality_dataset_length_cells"
        full_status, gate_reasons = classify_full(dataset_stats, length_stats, margins)
    else:
        full_status, gate_reasons = "inconclusive", ["incomplete_or_invalid_pairing"]
    return {
        "summary": {
            "schema_version": "kvbench-q1b-full-ppl-summary-1.0.0",
            "status": "COMPLETE" if not failures else "PARTIAL",
            "original_contract_id": q1a.BASE_CONTRACT_ID,
            "effective_amendment_id": q1a.AMENDMENT_ID,
            "original_cross_batch_gate": "failed",
            "physical_quality_batch_size": 1,
            "quality_transfer_to_other_batches": "not_established",
            "planned_anchor_units": int(plan["total_anchor_units"]),
            "completed_anchor_units": len(results),
            "failed_or_missing_anchor_units": len(failures),
            "total_scored_tokens": sum(int(row["scored_token_count"]) for row in results.values()),
            "baseline_valid": baseline_valid,
            "k4v4_full_ppl_status": full_status,
            "gate_reasons": gate_reasons,
            "q2a_eligible_configurations": ["k4v4"] if full_status == "pass" else [],
            "absolute_statistics": absolute,
            "aggregate_paired_statistics": aggregate_stats,
            "dataset_paired_statistics": dataset_stats,
            "length_paired_statistics": length_stats,
            "excluded_configurations": eligibility()["rows"],
            "fast_full_overlap": plan["fast_full_overlap"],
            "longbench_started": False,
            "performance_execution_started": False,
        },
        "paired_rows": paired_rows,
        "failures": failures,
    }


def _worker_command(campaign: Path, configuration: str, execution_head: str) -> list[str]:
    campaign = campaign.resolve()
    command = q0._docker_worker_command(campaign, configuration, execution_head)
    q0_marker = f"dst=/home/rockrock/cmu_paper/artifacts/q0/{campaign.name}"
    container_campaign = f"/home/rockrock/cmu_paper/artifacts/q1b/{campaign.name}"
    for index, value in enumerate(command):
        if value.startswith("type=bind,") and q0_marker in value:
            command[index] = f"type=bind,src={campaign},dst={container_campaign}"
            break
    else:
        raise Q1BError("Q1B campaign mount is absent")
    module_index = command.index("-m")
    return command[:module_index] + [
        "-m", "scripts.q1b_full_ppl", "--run-worker",
        "--campaign", container_campaign,
        "--configuration", configuration,
        "--execution-head", execution_head,
    ]


def run_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    campaign = campaign.resolve()
    if campaign.exists():
        raise Q1BError("Q1B campaign already exists")
    observed_head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if observed_head != execution_head:
        raise Q1BError("Q1B execution HEAD differs")
    if subprocess.run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=ROOT, check=True, capture_output=True, text=True).stdout:
        raise Q1BError("Q1B execution source is not clean")
    authority = verify_authority()
    protected = q0.verify_locked_paths()
    plan = full_plan()
    campaign.mkdir(parents=True)
    shutil.copy2(AUTHORIZATION, campaign / "execution_authorization.json")
    q0.write_json_new(campaign / "eligibility.json", eligibility())
    q0.write_json_new(campaign / "full_ppl_plan.json", plan)
    q0.write_json_new(campaign / "entry.json", {
        "schema_version": "kvbench-q1b-entry-1.0.0",
        "status": "PASS",
        "execution_head": execution_head,
        "authority": authority,
        "protected_paths": protected,
        "quality_image_digest": q0.QUALITY_IMAGE,
    })
    attempts: list[dict[str, Any]] = []
    with q0.gpu_lock():
        for configuration in CONFIGS:
            q0._gpu_idle()
            attempt_root = campaign / "worker_attempts" / configuration / "attempt-00"
            attempt_root.mkdir(parents=True)
            command = _worker_command(campaign, configuration, execution_head)
            with (attempt_root / "stdout.txt").open("wb") as stdout, (attempt_root / "stderr.txt").open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, check=False, stdout=stdout, stderr=stderr)
            q0.write_json_new(attempt_root / "attempt.json", {
                "schema_version": "kvbench-q1b-worker-attempt-1.0.0",
                "configuration": configuration,
                "returncode": process.returncode,
                "execution_head": execution_head,
                "replacement_of": None,
            })
            attempts.append({"configuration": configuration, "returncode": process.returncode})
            q0._gpu_idle()
            if process.returncode != 0:
                break
    q0.write_json_new(campaign / "worker_index.json", {
        "schema_version": "kvbench-q1b-worker-index-1.0.0",
        "attempts": attempts,
    })
    analysis = analyze(campaign)
    q0.write_json_new(campaign / "paired_full_ppl.json", {"rows": analysis["paired_rows"]})
    q0.write_json_new(campaign / "failures.json", {"rows": analysis["failures"]})
    q0.write_json_new(campaign / "full_ppl_summary.json", analysis["summary"])
    return analysis["summary"]


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Q1B B=1 Full PPL",
        "",
        f"Execution status: **{summary['status']}**",
        f"BF16 baseline: **{'VALID' if summary['baseline_valid'] else 'INVALID'}**",
        f"k4v4 Full PPL: **{summary['k4v4_full_ppl_status'].upper()}**",
        "",
        f"Completed anchors: {summary['completed_anchor_units']}/{summary['planned_anchor_units']}.",
        "Original cross-batch gate: FAILED and unchanged. Physical quality batch: B=1 only.",
        "",
        "| Dataset | BF16 NLL / PPL | k4v4 NLL / PPL | Delta NLL 95% CI | Relative PPL |",
        "|---|---:|---:|---:|---:|",
    ]
    absolute = summary["absolute_statistics"]
    paired = {row["dataset"]: row for row in summary["dataset_paired_statistics"]}
    for dataset in DATASETS:
        base = next(row for row in absolute["bf16"]["datasets"] if row["dataset"] == dataset)
        method = next(row for row in absolute["k4v4"]["datasets"] if row["dataset"] == dataset)
        delta = paired[dataset]
        lines.append(
            f"| {dataset} | {base['mean_nll']:.8g} / {base['ppl']:.8g} | "
            f"{method['mean_nll']:.8g} / {method['ppl']:.8g} | "
            f"{delta['point_delta_nll']:.8g} [{delta['ci_lower_delta_nll']:.8g}, {delta['ci_upper_delta_nll']:.8g}] | "
            f"{100.0 * delta['relative_ppl_change']:.6g}% |"
        )
    lines.extend([
        "",
        "The immutable summary contains all dataset/length absolute and paired statistics.",
        f"Gate reasons: {', '.join(summary['gate_reasons'])}.",
        "Q2A LongBench-E is not started.",
        "",
    ])
    return "\n".join(lines)


def finalize_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    if (campaign / "COMPLETE").exists():
        raise Q1BError("Q1B campaign already finalized")
    summary = load_json(campaign / "full_ppl_summary.json")
    q0.write_new(campaign / "q1b_report.md", _report(summary).encode())
    q0.write_json_new(campaign / "manifest.json", {
        "schema_version": "kvbench-q1b-full-ppl-campaign-1.0.0",
        "run_id": campaign.name,
        "status": summary["status"],
        "execution_head": execution_head,
        "original_contract_id": q1a.BASE_CONTRACT_ID,
        "effective_amendment_id": q1a.AMENDMENT_ID,
        "quality_image_digest": q0.QUALITY_IMAGE,
        "physical_quality_batch_size": 1,
        "original_cross_batch_gate": "failed",
    })
    payloads = sorted(path for path in campaign.rglob("*") if path.is_file())
    q0.write_json_new(campaign / "artifact_inventory.json", {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": campaign.name,
        "files": [
            {"path": path.relative_to(campaign).as_posix(), "role": q0._role(path.relative_to(campaign).as_posix()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in payloads
        ],
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    })
    ledger_paths = q0._checksum_ledger_payloads(campaign)
    q0.write_new(campaign / "checksums.sha256", "".join(f"{sha256_file(path)}  {path.relative_to(campaign).as_posix()}\n" for path in ledger_paths).encode())
    q0.write_json_new(campaign / "COMPLETE", {
        "schema_version": "kvbench-q1b-full-ppl-complete-1.0.0",
        "run_id": campaign.name,
        "status": summary["status"],
        "manifest_sha256": sha256_file(campaign / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(campaign / "artifact_inventory.json"),
        "checksum_ledger_sha256": sha256_file(campaign / "checksums.sha256"),
        "checksum_ledger_path": "checksums.sha256",
        "written_last": True,
    })
    for path in sorted(campaign.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    campaign.chmod(0o555)
    from scripts.r2_artifact import validate_local_artifact
    artifact = validate_local_artifact(campaign)
    return {"status": summary["status"], "campaign_id": campaign.name, "root_sha256": artifact.root_sha256, "object_count": len(artifact.files)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true")
    action.add_argument("--print-plan", action="store_true")
    action.add_argument("--run-worker", action="store_true")
    action.add_argument("--run-campaign", action="store_true")
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--configuration", choices=CONFIGS)
    parser.add_argument("--execution-head")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        print(json.dumps({"authority": verify_authority(), "eligibility": eligibility()}, sort_keys=True))
        return 0
    if args.print_plan:
        print(json.dumps(full_plan(), sort_keys=True))
        return 0
    if args.run_worker:
        if args.campaign is None or args.configuration is None or not args.execution_head:
            raise Q1BError("worker arguments are required")
        print(json.dumps(run_worker(args.campaign, args.configuration, args.execution_head), sort_keys=True))
        return 0
    if args.run_campaign:
        if args.campaign is None or not args.execution_head:
            raise Q1BError("campaign arguments are required")
        print(json.dumps(run_campaign(args.campaign, args.execution_head), sort_keys=True))
        return 0
    if args.finalize:
        if args.campaign is None or not args.execution_head:
            raise Q1BError("finalization arguments are required")
        print(json.dumps(finalize_campaign(args.campaign, args.execution_head), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
