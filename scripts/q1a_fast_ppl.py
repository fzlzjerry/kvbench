#!/usr/bin/env python3
"""Execute B=1-scoped Q1A cache-sensitive Fast PPL."""

from __future__ import annotations

import argparse
import array
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

from scripts import q0_cache_sensitive_correctness as q0


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT_ID = q0.CONTRACT_ID
BASE_CONTRACT_SHA256 = q0.CONTRACT_SHA256
AMENDMENT_ID = "quality-q1a-b1-20260919t135650246825z-a89ddc18"
AMENDMENT_SHA256 = "f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441"
AMENDMENT = ROOT / "configs/quality/amendments" / f"{AMENDMENT_ID}.json"
APPROVAL = ROOT / "docs/evidence/q1a/scope-amendment-approval.json"
CONTRACT_BUNDLE = q0.CONTRACT_BUNDLE
ANCHOR_MANIFEST = CONTRACT_BUNDLE / "selected_inputs/ppl/anchor_manifest.json"
INPUT_MANIFEST = CONTRACT_BUNDLE / "input_manifest.json"
QUALITY_MARGINS = CONTRACT_BUNDLE / "quality_margins.json"
SOURCE_Q0 = ROOT / "artifacts/q0/q0-20260919t054207209666z-b85111f1-27c21357-closure-5b143f1"
SOURCE_Q0_ROOT = "2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0"
BATCH_DIAGNOSIS_ROOT = "eedee1596bb36692da8557fc001c45593f50f930e62c27227a7151ab6aa88ba6"
DATASETS = ("wikitext2_test", "c4_validation")
PREFIX_LENGTHS = (4096, 24576, 32768, 65536)
ANCHORS_PER_LENGTH = 16
SCORED_HORIZON = 128
BURN_IN_TOKENS = 1
CONFIGS = q0.CONFIGS
REQUIRED_B1_STAGES = tuple(f"core-l{length}" for length in q0.CORE_LENGTHS) + (
    "longbench-suffix",
    "graph-invariance",
    "cache-dependence",
)


class Q1AError(RuntimeError):
    """Q1A failed closed."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return q0.load_json(path)


def verify_amendment() -> dict[str, Any]:
    if sha256_file(q0.CONTRACT) != BASE_CONTRACT_SHA256:
        raise Q1AError("original quality contract changed")
    if sha256_file(CONTRACT_BUNDLE / "contract_snapshot.yaml") != BASE_CONTRACT_SHA256:
        raise Q1AError("original contract snapshot changed")
    if sha256_file(AMENDMENT) != AMENDMENT_SHA256:
        raise Q1AError("scope amendment bytes differ")
    amendment = load_json(AMENDMENT)
    approval = load_json(APPROVAL)
    if amendment.get("amendment_id") != AMENDMENT_ID:
        raise Q1AError("scope amendment identity differs")
    if amendment.get("base_contract") != {
        "contract_id": BASE_CONTRACT_ID,
        "contract_sha256": BASE_CONTRACT_SHA256,
        "bundle_root": q0.QP1_ROOT,
    }:
        raise Q1AError("scope amendment base authority differs")
    scope = amendment.get("scope", {})
    expected_scope = {
        "original_cross_batch_gate": "failed",
        "quality_evaluated_batch_sizes": [1],
        "quality_transfer_to_other_batches": "not_established",
        "amendment_reason": "scope_restriction_after_Q0_batch_failure",
        "quality_status": "unvalidated",
    }
    if scope != expected_scope:
        raise Q1AError("scope amendment semantics differ")
    if (
        approval.get("status") != "APPROVED"
        or approval.get("amendment_id") != AMENDMENT_ID
        or approval.get("amendment_sha256") != AMENDMENT_SHA256
        or approval.get("authorization_scope")
        != ["MATERIALIZE_EXACT_B1_SCOPE_AMENDMENT", "Q1A_FAST_PPL_B1"]
    ):
        raise Q1AError("scope amendment approval differs")
    return {"status": "PASS", "amendment": amendment, "approval": approval}


def _q0_result(configuration: str, stage: str) -> tuple[Path, dict[str, Any]]:
    continuation = SOURCE_Q0 / "continuations/kvquant-q0-wrapper-fix/units" / configuration / stage / "result.json"
    primary = SOURCE_Q0 / "units" / configuration / stage / "result.json"
    path = continuation if continuation.is_file() else primary
    return path, load_json(path)


def b1_q0_eligibility() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for configuration in CONFIGS:
        stages: list[dict[str, Any]] = []
        for stage in REQUIRED_B1_STAGES:
            path, result = _q0_result(configuration, stage)
            stages.append({
                "stage": stage,
                "status": result.get("status"),
                "result_path": path.relative_to(ROOT).as_posix(),
                "result_sha256": sha256_file(path),
            })
        batch_path, batch_result = _q0_result(configuration, "batch-invariance")
        eligible = all(item["status"] == "PASS" for item in stages)
        rows.append({
            "configuration": configuration,
            "physical_quality_batch_size": 1,
            "eligible": eligible,
            "required_b1_stages": stages,
            "original_cross_batch_gate": batch_result.get("status"),
            "original_cross_batch_result_path": batch_path.relative_to(ROOT).as_posix(),
            "original_cross_batch_result_sha256": sha256_file(batch_path),
            "quality_transfer_to_other_batches": "not_established",
        })
    return {
        "schema_version": "kvbench-q1a-b1-q0-eligibility-1.0.0",
        "effective_amendment_id": AMENDMENT_ID,
        "source_q0_root": SOURCE_Q0_ROOT,
        "rows": rows,
        "eligible_count": sum(row["eligible"] for row in rows),
    }


def _manifest_file_hash(relative: str) -> str:
    manifest = load_json(INPUT_MANIFEST)
    matches = [item for item in manifest["files"] if item["path"] == relative]
    if len(matches) != 1:
        raise Q1AError(f"input manifest path missing: {relative}")
    path = CONTRACT_BUNDLE / relative
    if sha256_file(path) != matches[0]["sha256"]:
        raise Q1AError(f"frozen input changed: {relative}")
    return matches[0]["sha256"]


def fast_plan() -> dict[str, Any]:
    _manifest_file_hash("selected_inputs/ppl/anchor_manifest.json")
    manifest = load_json(ANCHOR_MANIFEST)
    rows: list[dict[str, Any]] = []
    streams: dict[str, dict[str, Any]] = {}
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
        for length in PREFIX_LENGTHS:
            anchors = source["stages"]["fast"][str(length)]
            if len(anchors) != ANCHORS_PER_LENGTH:
                raise Q1AError("Fast-PPL anchor cardinality differs")
            for anchor in anchors:
                if anchor["scored_token_count"] != SCORED_HORIZON:
                    raise Q1AError("Fast-PPL scored horizon differs")
                if anchor["prefix_end"] - anchor["prefix_start"] != length:
                    raise Q1AError("Fast-PPL prefix length differs")
                if anchor["burn_in_index"] != anchor["prefix_end"]:
                    raise Q1AError("Fast-PPL burn-in split differs")
                if anchor["target_start"] != anchor["burn_in_index"] + 1:
                    raise Q1AError("Fast-PPL target split differs")
                if anchor["target_end_exclusive"] - anchor["target_start"] != SCORED_HORIZON:
                    raise Q1AError("Fast-PPL target horizon differs")
                if anchor["scoring_mask"] != {"value": 1, "length": SCORED_HORIZON}:
                    raise Q1AError("Fast-PPL scoring mask differs")
                rows.append({"dataset": dataset, "prefix_length": length, **anchor})
    expected = len(DATASETS) * len(PREFIX_LENGTHS) * ANCHORS_PER_LENGTH
    if len(rows) != expected:
        raise Q1AError("Fast-PPL plan cardinality differs")
    return {
        "schema_version": "kvbench-q1a-fast-ppl-plan-1.0.0",
        "effective_amendment_id": AMENDMENT_ID,
        "physical_model_batch_size": 1,
        "execution_mode": "cache_sensitive_incremental_teacher_forcing",
        "burn_in_decode_tokens": BURN_IN_TOKENS,
        "scored_horizon": SCORED_HORIZON,
        "datasets": list(DATASETS),
        "prefix_lengths": list(PREFIX_LENGTHS),
        "anchors_per_length": ANCHORS_PER_LENGTH,
        "anchors_per_configuration": expected,
        "scored_tokens_per_configuration": expected * SCORED_HORIZON,
        "configuration_count": len(CONFIGS),
        "total_anchor_units": expected * len(CONFIGS),
        "total_scored_tokens": expected * SCORED_HORIZON * len(CONFIGS),
        "streams": streams,
        "anchors": rows,
    }


def quality_margins() -> dict[str, Any]:
    margins = load_json(QUALITY_MARGINS)
    expected = {
        "confidence_level": 0.95,
        "draws": 10000,
        "seed": 20260722,
        "ppl_unit": "paired_anchor_clustered_by_source_document",
    }
    if any(margins.get("bootstrap", {}).get(key) != value for key, value in expected.items()):
        raise Q1AError("approved Fast-PPL bootstrap contract differs")
    ppl = margins.get("ppl", {})
    if ppl.get("global_relative_ppl_increase_max") != 0.01:
        raise Q1AError("approved global PPL margin differs")
    if ppl.get("length_review_relative_increase") != 0.02:
        raise Q1AError("approved PPL length-review margin differs")
    if ppl.get("length_hard_fail_relative_increase") != 0.05:
        raise Q1AError("approved PPL length hard-fail margin differs")
    if ppl.get("primary_comparison") != "paired_delta_nll":
        raise Q1AError("approved PPL primary comparison differs")
    return margins


def read_i32_gzip(path: Path) -> list[int]:
    values = array.array("i")
    with gzip.open(path, "rb") as source:
        values.frombytes(source.read())
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


def teacher_forced_loss(logits: Any, target: Any, mask_value: int) -> tuple[float, int]:
    """Consume current logits before the corresponding true target is decoded."""

    import torch
    import torch.nn.functional as functional

    if mask_value not in (0, 1):
        raise Q1AError("scoring mask must be binary")
    if not bool(torch.isfinite(logits).all()):
        raise Q1AError("Fast-PPL logits contain NaN/Inf")
    if mask_value == 0:
        return 0.0, 0
    loss = functional.cross_entropy(logits.float().reshape(1, -1), target.reshape(1), reduction="sum")
    if not bool(torch.isfinite(loss)):
        raise Q1AError("Fast-PPL NLL is NaN/Inf")
    return float(loss.item()), 1


def score_anchor(
    loaded: Any,
    configuration: str,
    stream: Sequence[int],
    anchor: Mapping[str, Any],
    contract_fingerprint: str,
) -> dict[str, Any]:
    import torch

    length = int(anchor["prefix_length"])
    prefix_ids = list(stream[int(anchor["prefix_start"]):int(anchor["prefix_end"])])
    burn_in = int(stream[int(anchor["burn_in_index"])])
    targets = list(stream[int(anchor["target_start"]):int(anchor["target_end_exclusive"])])
    if len(prefix_ids) != length or len(targets) != SCORED_HORIZON:
        raise Q1AError("Fast-PPL input slice is truncated")
    output_steps = 1 + len(targets)
    method, cache, endpoint, positions, rope = q0._allocate_state(
        loaded,
        configuration,
        batch=1,
        capacity=length + output_steps,
        mode="growing",
        prefix_length=length,
        output_steps=output_steps,
    )
    selected = q0.family(configuration)
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=cache.device).reshape(1, length)
    burn_tensor = torch.tensor([[burn_in]], dtype=torch.long, device=cache.device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=cache.device)
    nll_sum = 0.0
    scored_count = 0
    finite = True
    last_output = None
    with torch.inference_mode():
        q0._prefill(endpoint, prefix, selected)
        cache.prepare_growing(length, output_steps)
        cache.select_growing_step(0)
        logits = endpoint.decode(burn_tensor, positions[0], rope[0]).squeeze(1)
        q0._finish_step(cache, selected)
        for index, token in enumerate(targets):
            contribution, count = teacher_forced_loss(logits[0], target_tensor[index], 1)
            nll_sum += contribution
            scored_count += count
            cache.select_growing_step(index + 1)
            logits = endpoint.decode(target_tensor[index].reshape(1, 1), positions[index + 1], rope[index + 1]).squeeze(1)
            q0._finish_step(cache, selected)
        last_output = logits.detach().to(device="cpu", dtype=torch.float32, copy=True)
        torch.cuda.synchronize(device=cache.device)
        finite = bool(torch.isfinite(last_output).all()) and math.isfinite(nll_sum)
    if scored_count != SCORED_HORIZON or not finite:
        raise Q1AError("Fast-PPL scored output is incomplete")
    accounting = cache.accounting()
    result = {
        "schema_version": "kvbench-q1a-fast-ppl-anchor-1.0.0",
        "status": "PASS",
        "configuration": configuration,
        "method_family": selected,
        "contract_method_config_fingerprint": contract_fingerprint,
        "live_adapter_config_fingerprint": method.config_fingerprint(cache.layout_fingerprint()),
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "effective_amendment_id": AMENDMENT_ID,
        "effective_amendment_sha256": AMENDMENT_SHA256,
        "physical_model_batch_size": 1,
        "quality_transfer_to_other_batches": "not_established",
        "dataset": anchor["dataset"],
        "anchor_id": anchor["anchor_id"],
        "cluster_id": anchor["cluster_id"],
        "source_document_id": anchor["source_document_id"],
        "prefix_length": length,
        "burn_in_decode_tokens": 1,
        "scored_token_count": scored_count,
        "scoring_mask": anchor["scoring_mask"],
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / scored_count,
        "ppl": math.exp(nll_sum / scored_count),
        "active_context": int(cache.active_context),
        "expected_active_context": length + output_steps,
        "trailing_decode_executed": True,
        "finite": finite,
        "output_checksum": q0.output_digest(last_output),
        "allocated_bytes": int(accounting.allocated_bytes),
        "compressed_cache_read_authority": "existing_Q0_cache_dependence_unit",
        "run_kind": "quality",
        "normal_performance_timing": False,
    }
    if result["active_context"] != result["expected_active_context"]:
        raise Q1AError("Fast-PPL cache state did not advance exactly")
    return result


def _unit_dir(campaign: Path, configuration: str, anchor: Mapping[str, Any]) -> Path:
    return campaign / "units" / configuration / anchor["dataset"] / f"l{anchor['prefix_length']}" / anchor["anchor_id"]


def _unit_complete(campaign: Path, configuration: str, anchor: Mapping[str, Any]) -> bool:
    root = _unit_dir(campaign, configuration, anchor)
    return (root / "result.json").is_file() and (root / "COMPLETE").is_file()


def _finalize_unit(campaign: Path, configuration: str, anchor: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    root = _unit_dir(campaign, configuration, anchor)
    if root.exists():
        raise Q1AError("Fast-PPL unit already exists")
    root.mkdir(parents=True)
    q0.write_json_new(root / "result.json", result)
    q0.write_json_new(root / "COMPLETE", {
        "schema_version": "kvbench-q1a-fast-ppl-unit-complete-1.0.0",
        "configuration": configuration,
        "dataset": anchor["dataset"],
        "prefix_length": anchor["prefix_length"],
        "anchor_id": anchor["anchor_id"],
        "status": result["status"],
        "result_sha256": sha256_file(root / "result.json"),
        "written_last": True,
    })


def run_worker(campaign: Path, configuration: str, execution_head: str) -> dict[str, Any]:
    if os.environ.get("KVBENCH_QUALITY_IMAGE_DIGEST") != q0.QUALITY_IMAGE:
        raise Q1AError("Quality image identity differs")
    observed_head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if observed_head != execution_head:
        raise Q1AError("Q1A execution HEAD differs")
    if subprocess.run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=ROOT, check=True, capture_output=True, text=True).stdout:
        raise Q1AError("Q1A execution source is not clean")
    verify_amendment()
    q0.verify_locked_paths()
    eligibility = b1_q0_eligibility()
    row = next(item for item in eligibility["rows"] if item["configuration"] == configuration)
    if not row["eligible"]:
        raise Q1AError("configuration is not B=1 Q0 eligible")
    plan = fast_plan()
    contract = load_json(q0.CONTRACT)["quality_contract"]
    fingerprints = {item["method_config_id"]: item["method_config_fingerprint"] for item in contract["configurations"]["items"]}

    import gc
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    streams = {
        dataset: read_i32_gzip(CONTRACT_BUNDLE / plan["streams"][dataset]["path"])
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
                result = score_anchor(loaded, configuration, streams[anchor["dataset"]], anchor, fingerprints[configuration])
            except BaseException as error:
                failed += 1
                _finalize_unit(campaign, configuration, anchor, {
                    "schema_version": "kvbench-q1a-fast-ppl-anchor-failure-1.0.0",
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


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise Q1AError("quantile requires observations")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(rows: Sequence[Mapping[str, Any]], *, seed: int, draws: int) -> dict[str, float]:
    clusters: dict[str, list[tuple[float, int]]] = {}
    for row in rows:
        clusters.setdefault(str(row["cluster_id"]), []).append((float(row["delta_nll"]), int(row["scored_token_count"])))
    keys = sorted(clusters)
    if not keys:
        raise Q1AError("paired bootstrap has no clusters")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        numerator = 0.0
        denominator = 0
        for _cluster in keys:
            selected = clusters[keys[rng.randrange(len(keys))]]
            for delta, count in selected:
                numerator += delta * count
                denominator += count
        estimates.append(numerator / denominator)
    point_num = sum(float(row["delta_nll"]) * int(row["scored_token_count"]) for row in rows)
    point_den = sum(int(row["scored_token_count"]) for row in rows)
    return {
        "point_delta_nll": point_num / point_den,
        "ci_lower_delta_nll": quantile(estimates, 0.025),
        "ci_upper_delta_nll": quantile(estimates, 0.975),
    }


def classify_fast(
    aggregate_stats: Mapping[str, Any],
    length_stats: Sequence[Mapping[str, Any]],
    margins: Mapping[str, Any] | None = None,
) -> str:
    frozen = quality_margins() if margins is None else margins
    ppl = frozen["ppl"]
    margin = math.log1p(float(ppl["global_relative_ppl_increase_max"]))
    hard_fail = float(ppl["length_hard_fail_relative_increase"])
    if any(float(row["relative_ppl_change"]) > hard_fail for row in length_stats):
        return "fail"
    if float(aggregate_stats["ci_lower_delta_nll"]) > margin:
        return "fail"
    if float(aggregate_stats["ci_upper_delta_nll"]) <= margin:
        return "pass"
    return "inconclusive"


def analyze(campaign: Path) -> dict[str, Any]:
    plan = load_json(campaign / "fast_ppl_plan.json")
    margins = quality_margins()
    results: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for configuration in CONFIGS:
        for anchor in plan["anchors"]:
            path = _unit_dir(campaign, configuration, anchor) / "result.json"
            if not path.is_file():
                failures.append({"configuration": configuration, "anchor": anchor, "reason": "missing"})
                continue
            result = load_json(path)
            if result.get("status") != "PASS":
                failures.append({"configuration": configuration, "anchor": anchor, "reason": result.get("failure_class", "failed")})
                continue
            key = (configuration, anchor["dataset"], int(anchor["prefix_length"]), anchor["anchor_id"])
            results[key] = result
    baseline_rows = [key for key in results if key[0] == "bf16"]
    baseline_valid = (
        len(baseline_rows) == len(plan["anchors"])
        and not any(item["configuration"] == "bf16" for item in failures)
    )
    paired_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for configuration in CONFIGS:
        if configuration == "bf16":
            decisions.append({
                "configuration": configuration,
                "fast_stage_status": "baseline_valid" if baseline_valid else "baseline_invalid",
                "q1b_eligible": False,
                "scientific_self_comparison": False,
            })
            continue
        config_failed = any(item["configuration"] == configuration for item in failures)
        for anchor in plan["anchors"]:
            key = (configuration, anchor["dataset"], int(anchor["prefix_length"]), anchor["anchor_id"])
            base_key = ("bf16", anchor["dataset"], int(anchor["prefix_length"]), anchor["anchor_id"])
            if key not in results or base_key not in results:
                continue
            method = results[key]
            base = results[base_key]
            delta = float(method["mean_nll"]) - float(base["mean_nll"])
            paired_rows.append({
                "configuration": configuration,
                "dataset": anchor["dataset"],
                "prefix_length": int(anchor["prefix_length"]),
                "anchor_id": anchor["anchor_id"],
                "cluster_id": anchor["cluster_id"],
                "scored_token_count": int(method["scored_token_count"]),
                "bf16_mean_nll": float(base["mean_nll"]),
                "method_mean_nll": float(method["mean_nll"]),
                "delta_nll": delta,
                "relative_ppl_change": math.expm1(delta),
            })
        own_rows = [row for row in paired_rows if row["configuration"] == configuration]
        dataset_stats: list[dict[str, Any]] = []
        length_stats: list[dict[str, Any]] = []
        aggregate_stats: dict[str, Any] = {}
        if baseline_valid and not config_failed and len(own_rows) == len(plan["anchors"]):
            bootstrap = margins["bootstrap"]
            for dataset in DATASETS:
                selected = [row for row in own_rows if row["dataset"] == dataset]
                stats = paired_cluster_bootstrap(
                    selected,
                    seed=int(bootstrap["seed"]),
                    draws=int(bootstrap["draws"]),
                )
                dataset_stats.append({
                    "dataset": dataset,
                    **stats,
                    "relative_ppl_change": math.expm1(stats["point_delta_nll"]),
                    "ci_lower_relative_ppl_change": math.expm1(stats["ci_lower_delta_nll"]),
                    "ci_upper_relative_ppl_change": math.expm1(stats["ci_upper_delta_nll"]),
                    "anchor_count": len(selected),
                    "scored_token_count": sum(row["scored_token_count"] for row in selected),
                })
            for dataset in DATASETS:
                for length in PREFIX_LENGTHS:
                    selected = [row for row in own_rows if row["dataset"] == dataset and row["prefix_length"] == length]
                    delta = sum(row["delta_nll"] * row["scored_token_count"] for row in selected) / sum(row["scored_token_count"] for row in selected)
                    length_stats.append({
                        "dataset": dataset,
                        "prefix_length": length,
                        "delta_nll": delta,
                        "relative_ppl_change": math.expm1(delta),
                        "review_threshold_exceeded": math.expm1(delta) > float(margins["ppl"]["length_review_relative_increase"]),
                        "hard_fail_threshold_exceeded": math.expm1(delta) > float(margins["ppl"]["length_hard_fail_relative_increase"]),
                        "anchor_count": len(selected),
                    })
            aggregate_stats = paired_cluster_bootstrap(
                own_rows,
                seed=int(bootstrap["seed"]),
                draws=int(bootstrap["draws"]),
            )
            aggregate_stats.update({
                "aggregation": "token_weighted_over_frozen_equal_cardinality_dataset_length_cells",
                "relative_ppl_change": math.expm1(aggregate_stats["point_delta_nll"]),
                "ci_lower_relative_ppl_change": math.expm1(aggregate_stats["ci_lower_delta_nll"]),
                "ci_upper_relative_ppl_change": math.expm1(aggregate_stats["ci_upper_delta_nll"]),
                "anchor_count": len(own_rows),
                "scored_token_count": sum(row["scored_token_count"] for row in own_rows),
            })
            status = classify_fast(aggregate_stats, length_stats, margins)
        else:
            status = "inconclusive"
        decisions.append({
            "configuration": configuration,
            "fast_stage_status": status,
            "q1b_eligible": status == "pass",
            "paired_anchor_count": len(own_rows),
            "aggregate_statistics": aggregate_stats,
            "dataset_statistics": dataset_stats,
            "length_statistics": length_stats,
            "failure_or_missing_count": sum(item["configuration"] == configuration for item in failures),
        })
    summary = {
        "schema_version": "kvbench-q1a-fast-ppl-summary-1.0.0",
        "status": "COMPLETE" if not failures else "PARTIAL",
        "effective_amendment_id": AMENDMENT_ID,
        "effective_amendment_sha256": AMENDMENT_SHA256,
        "original_cross_batch_gate": "failed",
        "physical_quality_batch_size": 1,
        "quality_transfer_to_other_batches": "not_established",
        "baseline_valid": baseline_valid,
        "planned_anchor_units": int(plan["total_anchor_units"]),
        "completed_anchor_units": len(results),
        "failed_or_missing_anchor_units": len(failures),
        "paired_rows": paired_rows,
        "decisions": decisions,
        "q1b_eligible_configurations": [row["configuration"] for row in decisions if row.get("q1b_eligible")],
        "full_ppl_started": False,
        "longbench_started": False,
        "performance_execution_started": False,
    }
    return summary


def _worker_command(campaign: Path, configuration: str, execution_head: str) -> list[str]:
    command = q0._docker_worker_command(campaign, configuration, execution_head)
    container_campaign = f"/home/rockrock/cmu_paper/artifacts/q1a/{campaign.name}"
    source_marker = f"src={campaign},"
    for index, value in enumerate(command):
        if value.startswith("type=bind,") and source_marker in value:
            command[index] = f"type=bind,src={campaign},dst={container_campaign}"
            break
    else:
        raise Q1AError("Q1A campaign mount is absent")
    module_index = command.index("-m")
    return command[:module_index] + [
        "-m", "scripts.q1a_fast_ppl", "--run-worker",
        "--campaign", container_campaign,
        "--configuration", configuration,
        "--execution-head", execution_head,
    ]


def run_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    if campaign.exists():
        raise Q1AError("Q1A campaign already exists")
    if subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip() != execution_head:
        raise Q1AError("Q1A execution HEAD differs")
    if subprocess.run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=ROOT, check=True, capture_output=True, text=True).stdout:
        raise Q1AError("Q1A execution source is not clean")
    amendment = verify_amendment()
    protected = q0.verify_locked_paths()
    eligibility = b1_q0_eligibility()
    if eligibility["eligible_count"] != len(CONFIGS):
        raise Q1AError("not all configurations are B=1 Q0 eligible")
    plan = fast_plan()
    campaign.mkdir(parents=True)
    shutil.copy2(AMENDMENT, campaign / "scope_amendment.json")
    shutil.copy2(APPROVAL, campaign / "scope_amendment_approval.json")
    q0.write_json_new(campaign / "b1_q0_eligibility.json", eligibility)
    q0.write_json_new(campaign / "fast_ppl_plan.json", plan)
    q0.write_json_new(campaign / "entry.json", {
        "schema_version": "kvbench-q1a-entry-1.0.0",
        "status": "PASS",
        "execution_head": execution_head,
        "amendment": amendment,
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
            stdout_path = attempt_root / "stdout.txt"
            stderr_path = attempt_root / "stderr.txt"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, check=False, stdout=stdout, stderr=stderr)
            q0.write_json_new(attempt_root / "attempt.json", {
                "schema_version": "kvbench-q1a-worker-attempt-1.0.0",
                "configuration": configuration,
                "returncode": process.returncode,
                "execution_head": execution_head,
                "replacement_of": None,
            })
            attempts.append({"configuration": configuration, "returncode": process.returncode})
            q0._gpu_idle()
    q0.write_json_new(campaign / "worker_index.json", {
        "schema_version": "kvbench-q1a-worker-index-1.0.0",
        "attempts": attempts,
    })
    summary = analyze(campaign)
    q0.write_json_new(campaign / "paired_fast_ppl.json", {"rows": summary.pop("paired_rows")})
    q0.write_json_new(campaign / "fast_ppl_summary.json", summary)
    return summary


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Q1A B=1 Fast PPL",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"Physical quality batch: B=1. Original cross-batch gate: FAILED and unchanged.",
        f"Completed anchors: {summary['completed_anchor_units']}/{summary['planned_anchor_units']}.",
        "",
        "| Configuration | Fast status | Q1B eligible | WikiText delta NLL / CI | C4 delta NLL / CI |",
        "|---|---|---|---|---|",
    ]
    for row in summary["decisions"]:
        if row["configuration"] == "bf16":
            lines.append(f"| bf16 | {row['fast_stage_status']} | no | baseline | baseline |")
            continue
        stats = {item["dataset"]: item for item in row["dataset_statistics"]}
        def cell(dataset: str) -> str:
            if dataset not in stats:
                return "unavailable"
            item = stats[dataset]
            return f"{item['point_delta_nll']:.8g} / [{item['ci_lower_delta_nll']:.8g}, {item['ci_upper_delta_nll']:.8g}]"
        lines.append(
            f"| {row['configuration']} | {row['fast_stage_status']} | {'yes' if row['q1b_eligible'] else 'no'} | "
            f"{cell('wikitext2_test')} | {cell('c4_validation')} |"
        )
    lines.extend([
        "",
        "All comparisons use paired anchors and 10,000 document-cluster bootstrap draws.",
        "B=1 evidence is not transferable to B=2/4/8/16 performance. Full PPL and",
        "LongBench were not started.",
        "",
    ])
    return "\n".join(lines)


def finalize_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    if (campaign / "COMPLETE").exists():
        raise Q1AError("Q1A campaign already finalized")
    summary = load_json(campaign / "fast_ppl_summary.json")
    q0.write_new(campaign / "q1a_report.md", _report(summary).encode())
    q0.write_json_new(campaign / "manifest.json", {
        "schema_version": "kvbench-q1a-fast-ppl-campaign-1.0.0",
        "run_id": campaign.name,
        "status": summary["status"],
        "execution_head": execution_head,
        "effective_amendment_id": AMENDMENT_ID,
        "effective_amendment_sha256": AMENDMENT_SHA256,
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
        "schema_version": "kvbench-q1a-fast-ppl-complete-1.0.0",
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
        print(json.dumps({"amendment": verify_amendment(), "eligibility": b1_q0_eligibility()}, sort_keys=True))
        return 0
    if args.print_plan:
        print(json.dumps(fast_plan(), sort_keys=True))
        return 0
    if args.run_worker:
        if args.campaign is None or args.configuration is None or not args.execution_head:
            raise Q1AError("worker arguments are required")
        print(json.dumps(run_worker(args.campaign, args.configuration, args.execution_head), sort_keys=True))
        return 0
    if args.run_campaign:
        if args.campaign is None or not args.execution_head:
            raise Q1AError("campaign arguments are required")
        print(json.dumps(run_campaign(args.campaign, args.execution_head), sort_keys=True))
        return 0
    if args.finalize:
        if args.campaign is None or not args.execution_head:
            raise Q1AError("finalization arguments are required")
        print(json.dumps(finalize_campaign(args.campaign, args.execution_head), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
