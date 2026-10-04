#!/usr/bin/env python3
"""Evidence-only Q3/Q4 aggregation; standalone CPU reproduction with pyarrow.

No benchmark/scorer/runtime import, model loading, network, bootstrap, or fit.
The stored scientific outcomes are inputs, not values to optimize here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
STARTING_HEAD = "21251cd56a94cbed46925ef84d3ecdc44896d797"
CONTRACT_SHA = "b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1"
AMENDMENT_SHA = "f794f4a59899c7dfe853c39b7011b2d0633d68acf5a3e2ddeb5851f4d564b441"
CONFIGS = ("bf16", "tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc", "k4v4", "k2v4", "k2v2", "kvq4", "kvq3", "kvq2")
FREEZE = Path("artifacts/performance_freeze/qp0-freeze-20260917t115200000000z-83536c37-bf28e4")
FAMILY = Path("artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac")
EMPTY_REASON = "no_compressed_configuration_satisfied_all_required_evaluated_gates"
PRODUCTS = ("quality_admission_table.csv", "performance_quality_join.parquet",
            "qualified_compressed_candidates.parquet", "q2b_discordant_pairs.csv",
            "joint_result_summary.json", "QUALITY_VALIDATION_REPORT.md", "identity_mapping.json")


class JointError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise JointError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    write_new(path, encoded(value))


def unique(rows, key, label):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, f"duplicate {label}: {k}")
        result[k] = row
    return result


def point_key(row):
    return (row["method_config_fingerprint"], row["batch_size"],
            row.get("context_label", row.get("context_length")),
            row.get("historical_context", row.get("actual_historical_context")))


def source_paths(repo):
    """A fixed compact handoff, not a historical campaign scan."""
    paths = {
        "contract": "configs/quality/quality_contract.yaml",
        "amendment": "configs/quality/amendments/quality-q1a-b1-20260919t135650246825z-a89ddc18.json",
        "approval": "docs/evidence/q0/quality-contract-approval.json",
        "amendment_approval": "docs/evidence/q1a/scope-amendment-approval.json",
        "freeze_correction": "docs/evidence/qp0/freeze-tag-correction.json",
        "q2b_evidence": "docs/evidence/q2b/longbench-v2.json",
        "phase18_report": "docs/phase_reports/phase18-reproduction.md",
        "phase17_report": "docs/phase_reports/phase17-modeling.md",
    }
    for name in ("manifest.json", "performance_inventory.parquet", "locked_hot_path_hashes.json",
                 "external_kernel_identities.json", "hardware_identity.json", "quality_image.json", "release_sources.json"):
        paths["freeze_" + Path(name).stem] = FREEZE / name
    for name in ("raw_run_index.parquet", "point_summary.parquet", "same_work_ratios.parquet"):
        paths["wall_" + Path(name).stem] = FAMILY / "wall-closure" / name
    for name in ("feasibility.parquet", "capacity_amplification.parquet", "profiler_feature_join.parquet"):
        paths["performance_" + Path(name).stem] = FAMILY / "outer" / name
    summaries = {"q1a": "fast_ppl_summary.json", "q1b": "full_ppl_summary.json",
                 "q2a": "longbench_e_summary.json", "q2b": "summary.json"}
    reports = {"q1a": "q1a-fast-ppl.md", "q1b": "q1b-full-ppl.md",
               "q2a": "q2a-b1-longbench-e.md", "q2b": "q2b-b1-longbench-v2.md"}
    for phase, name in summaries.items():
        receipt = Path(f"docs/evidence/{phase}/r2-publication.json")
        campaign = Path("artifacts") / phase / load(repo / receipt)["campaign_id"]
        paths[phase + "_receipt"] = receipt
        paths[phase + "_summary"] = campaign / name
        paths[phase + "_entry"] = campaign / "entry.json"
        paths[phase + "_manifest"] = campaign / "manifest.json"
        paths[phase + "_report"] = Path("docs/phase_reports") / reports[phase]
        if phase == "q1a":
            paths["b1_eligibility"] = campaign / "b1_q0_eligibility.json"
        if phase == "q2b":
            paths["q2b_pairs"] = campaign / "paired_scores.json"
    # These ten small existing records document the LIMITED, shape-specific
    # eager/Graph relationship; they do not certify every performance shape.
    for row in load(repo / paths["b1_eligibility"])["rows"]:
        graph = next(s for s in row["required_b1_stages"] if s["stage"] == "graph-invariance")
        paths["q0_graph_" + row["configuration"]] = graph["result_path"]
    return {key: Path(value) for key, value in paths.items()}


def selected_ledger_binding(repo, path):
    """Check only the selected compact file against its existing local ledger."""
    for parent in path.parents:
        if parent == repo or repo not in parent.parents:
            break
        ledger = parent / "checksums.sha256"
        if ledger.is_file():
            relative = path.relative_to(parent).as_posix()
            entries = dict((line.split("  ", 1)[1], line.split("  ", 1)[0])
                           for line in ledger.read_text().splitlines())
            require(entries.get(relative) == sha(path), f"source ledger mismatch: {path}")
            return {"path": ledger.relative_to(repo).as_posix(), "entry": relative,
                    "expected_sha256": entries[relative]}
    return None


def read_inputs(directory):
    provenance = load(directory / "source_manifest.json")
    data = {}
    for item in provenance["files"]:
        path = directory / item["bundle_path"]
        require(sha(path) == item["sha256"], f"compact source changed: {item['key']}")
        data[item["key"]] = (pq.read_table(path).to_pylist() if path.suffix == ".parquet"
                             else path.read_text() if path.suffix == ".md" else load(path))
    data["provenance"] = provenance
    return data


def validate_authority(data):
    files = {r["key"]: r for r in data["provenance"]["files"]}
    require(files["contract"]["sha256"] == CONTRACT_SHA, "contract SHA differs")
    require(files["amendment"]["sha256"] == AMENDMENT_SHA, "amendment SHA differs")
    c, a, f = data["contract"]["quality_contract"], data["amendment"], data["freeze_manifest"]
    require(data["approval"]["status"] == "APPROVED" and data["approval"]["contract_sha256"] == CONTRACT_SHA, "approval differs")
    require(data["amendment_approval"]["status"] == "APPROVED" and data["amendment_approval"]["amendment_sha256"] == AMENDMENT_SHA, "amendment approval differs")
    require(a["scope"]["quality_evaluated_batch_sizes"] == [1] and a["scope"]["original_cross_batch_gate"] == "failed", "B1 scope differs")
    require(a["base_contract"]["contract_id"] == c["id"], "base contract differs")
    require(tuple(i["method_config_id"] for i in c["configurations"]["items"]) == CONFIGS, "configuration set differs")
    require(data["freeze_correction"]["freeze_bundle"]["manifest_sha256"] == files["freeze_manifest"]["sha256"], "corrected freeze binding differs")
    require(data["freeze_correction"]["active_freeze_binding"]["tag_name"] == c["provenance"]["performance_freeze_tag"], "freeze tag differs")
    require({i["method_config_id"]: i["method_config_fingerprint"] for i in c["configurations"]["items"]} == f["method_config_fingerprints"], "contract/freeze fingerprint discrepancy")
    # Q0's comparison digest covers path/SHA pairs, whereas the freeze's own
    # aggregate also includes Git blob/size metadata. Respect both encodings.
    pairs = [{"path": r["path"], "sha256": r["sha256"]} for r in data["freeze_locked_hot_path_hashes"]["files"]]
    comparison_sha = hashlib.sha256((json.dumps(pairs, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    for phase in ("q1a", "q1b", "q2a", "q2b"):
        entry = data[phase + "_entry"]
        require(entry["status"] == "PASS" and entry["protected_paths"]["status"] == "PASS", f"{phase} source lineage absent")
        require(entry["protected_paths"]["authority_sha256"] == files["freeze_locked_hot_path_hashes"]["sha256"] and
                entry["protected_paths"]["files_aggregate_sha256"] == comparison_sha, f"{phase} protected lineage differs")
        require(entry["quality_image_digest"] == f["quality_image_digest"], f"{phase} Quality image differs")
        receipt = data[phase + "_receipt"]
        require(receipt["status"] == "PASS" and receipt["complete_last"] and receipt["clean_retrieval"] in (True, "PASS"), f"{phase} durable receipt absent")
    require(data["q2b_summary"]["margins"] == c["gates"]["longbench_v2"], "v2 margins differ")
    return c


def registry(data):
    c = validate_authority(data)
    q0 = unique(data["b1_eligibility"]["rows"], lambda r: r["configuration"], "Q0 configuration")
    fast = unique(data["q1a_summary"]["decisions"], lambda r: r["configuration"], "Fast configuration")
    require(set(q0) == set(fast) == set(CONFIGS), "incomplete configuration coverage")
    full, e, v = (data[k] for k in ("q1b_summary", "q2a_summary", "q2b_summary"))
    require(full["failed_or_missing_anchor_units"] == 0 and e["missing_or_failed_outputs"] == 0 and v["missing_outputs"] == 0, "stage execution incomplete")
    require(full["baseline_valid"] and data["q1a_summary"]["baseline_valid"], "baseline invalid")
    output = []
    for item in c["configurations"]["items"]:
        name = item["method_config_id"]
        eligible = q0[name]["eligible"]
        require(eligible and all(r["status"] == "PASS" for r in q0[name]["required_b1_stages"]), "accepted B1 Q0 mapping differs")
        fs = fast[name]["fast_stage_status"].upper()
        full_status = "NOT_RUN_DUE_TO_FAST_STAGE_STATUS"
        e_status = "NOT_RUN_DUE_TO_ELIGIBILITY"
        v_status = "NOT_RUN_DUE_TO_ELIGIBILITY"
        primary = "NOT_RUN"
        if name == "bf16":
            full_status = e_status = v_status = "BASELINE_VALID"
            final, failed_stage, reason = "reference_baseline", None, "uncompressed_reference_not_a_compressed_candidate"
        else:
            if fs == "PASS":
                require(name == "k4v4", "new downstream configuration lacks stored evidence")
                full_status = full["k4v4_full_ppl_status"].upper()
                e_status = e["gate_status"].upper()
                v_status = v["scientific_verdict"].upper()
                primary = data["q2b_evidence"]["guardrails"]["primary_noninferiority"]["status"]
            stages = [("Q1A", fs), ("Q1B", full_status), ("Q2A", e_status), ("Q2B", v_status)]
            failed = next(((s, value) for s, value in stages if value in ("FAIL", "INCONCLUSIVE")), None)
            if failed:
                failed_stage, value = failed
                final = "quality_fail" if value == "FAIL" else "quality_inconclusive"
                reason = ";".join(v["gate_reasons"]) if failed_stage == "Q2B" else f"{failed_stage}_{value.lower()}_under_frozen_margins"
            else:
                require(all(value == "PASS" for _, value in stages), "unresolved stage chain")
                final, failed_stage, reason = "quality_pass_finalist", None, "all_required_evaluated_gates_pass"
        graph = data["q0_graph_" + name]
        expected_graph = next(s for s in q0[name]["required_b1_stages"] if s["stage"] == "graph-invariance")
        source_files = {r["key"]: r for r in data["provenance"]["files"]}
        require(source_files["q0_graph_" + name]["sha256"] == expected_graph["result_sha256"], "Q0 graph evidence binding differs")
        require(graph["status"] == "PASS" and graph["configuration"] == name, "stored graph relationship differs")
        output.append({
            "method_config_id": name, "method_config_fingerprint": item["method_config_fingerprint"],
            "method_family": item["method_family"], "parameters_json": json.dumps(item["parameters"], sort_keys=True),
            "is_compressed": name != "bf16", "quality_contract_id": c["id"], "quality_contract_sha256": CONTRACT_SHA,
            "effective_amendment_id": data["amendment"]["amendment_id"], "amendment_sha256": AMENDMENT_SHA,
            "physical_quality_batch_size": 1, "original_cross_batch_gate": q0[name]["original_cross_batch_gate"],
            "b1_q0_eligibility": "eligible", "fast_ppl_status": fs, "full_ppl_status": full_status,
            "longbench_e_status": e_status, "longbench_v2_status": v_status,
            "v2_primary_noninferiority": primary, "joint_quality_status": final, "failure_stage": failed_stage,
            "reason": reason, "fully_qualified_compressed": name != "bf16" and final == "quality_pass_finalist",
            "q2c_status": "not_run_no_eligible_finalist" if v_status != "PASS" else "not_run_requires_separate_authorization",
            "native_prefill_status": "not_run_conditional_finalists_only",
            "native_prefill_contract_rule": c["execution"]["secondary_protocol"],
            "entered_v2_finalist_validation": name != "bf16" and v_status in ("PASS", "FAIL", "INCONCLUSIVE"),
            "quality_transfer_to_other_batches": "not_established",
            "quality_run_ids_json": json.dumps({p: data[p + "_receipt"]["campaign_id"] for p in ("q1a", "q1b", "q2a", "q2b") if p == "q1a" or name in ("bf16", "k4v4")}, sort_keys=True),
            "source_roots_json": json.dumps({p: data[p + "_receipt"]["root_sha256"] for p in ("q1a", "q1b", "q2a", "q2b")}, sort_keys=True),
        })
    return output


def arithmetic_check(summary, paired):
    """Independent arithmetic on saved classifications; no parser or bootstrap."""
    unique(paired, lambda r: r["sample_id"], "v2 sample")
    require(len(paired) == summary["eligible"] == summary["paired_count"], "v2 paired coverage differs")
    require(summary["selected"] == summary["eligible"] + summary["excluded"], "v2 exclusions differ")
    groups = [(summary["overall"], paired)]
    for field, table in (("category", "category_results"), ("length_bucket", "length_results")):
        groups.extend((row, [p for p in paired if p[field] == row["name"]]) for row in summary[table])
    for recorded, rows in groups:
        require(bool(rows), "empty stored v2 group")
        counts = Counter((bool(r["bf16_correct"]), bool(r["k4v4_correct"])) for r in rows)
        n11, n10, n01, n00 = (counts[x] for x in ((True, True), (True, False), (False, True), (False, False)))
        n = len(rows)
        values = {"pairs": n, "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                  "bf16_accuracy": (n11+n10)/n, "k4v4_accuracy": (n11+n01)/n,
                  "accuracy_drop_pp": 100*(n10-n01)/n, "retention_denominator": n11+n10,
                  "retention": n11/(n11+n10) if n11+n10 else None}
        for key, val in values.items():
            require(recorded[key] is None if val is None else math.isclose(recorded[key], val, abs_tol=1e-12), f"saved v2 arithmetic differs: {key}")
        for config in ("bf16", "k4v4"):
            require(sum(r[config + "_invalid"] for r in rows) == recorded["invalid_counts"][config], "invalid count differs")
        increase = 100 * (recorded["invalid_counts"]["k4v4"] - recorded["invalid_counts"]["bf16"]) / n
        require(math.isclose(increase, recorded["invalid_increase_pp"], abs_tol=1e-12), "invalid increase differs")
    for config in ("bf16", "k4v4"):
        require(dict(Counter(r[config + "_stop_reason"] for r in paired)) == summary["generation"][config]["stop_reasons"], "saved stop counts differ")
    return [dict(r, transition="correct_to_wrong" if r["bf16_correct"] else "wrong_to_correct")
            for r in paired if r["bf16_correct"] != r["k4v4_correct"]]


def identity_mappings(data, reg):
    c = data["contract"]["quality_contract"]
    locked = {r["path"]: r["sha256"] for r in data["freeze_locked_hot_path_hashes"]["files"]}
    result = {}
    for row in reg:
        name, family = row["method_config_id"], row["method_family"]
        ext = data["freeze_external_kernel_identities"]["families"][family]
        path = f"src/kvbench/adapters/{family}.py"
        require(locked[path] == ext["performance_repo_source_hashes"][path], "frozen adapter authorities conflict")
        graph = data["q0_graph_" + name]
        result[row["method_config_fingerprint"]] = {
            "method_config_id": name, "contract_method_config_fingerprint": row["method_config_fingerprint"],
            "adapter_source_hash": locked[path], "kernel_source_hash": ext["kernel_source_aggregate_sha256"],
            "kernel_binary_hash": ext.get("kernel_binary_sha256") or ext.get("extension_sha256") or ext["kernel_source_aggregate_sha256"],
            "kernel_binary_identity_kind": "admission_extension_or_frozen_binary" if ext.get("kernel_binary_sha256") or ext.get("extension_sha256") else "jit_source_authority_no_single_static_binary",
            "model_checkpoint": c["model"]["checkpoint"], "model_revision": c["model"]["revision"],
            "tokenizer_revision": c["model"]["tokenizer_revision"],
            "container_digest": data["freeze_manifest"]["measurement_container_digest"],
            "quality_container_digest": data["freeze_manifest"]["quality_image_digest"],
            "lineage": "exact contract/freeze configuration plus inherited protected-path PASS and frozen external kernel authority",
            "graph_eager_relationship": {k: graph[k] for k in ("status", "eager", "graph", "fixed_l_graph_relationship_only")},
            "mode_scope_limit": "Q0 fixed-L eager/Graph at stored shape only; not arbitrary growing-to-fixed or cross-length equivalence",
        }
    return result


def join_performance(data, reg):
    maps = identity_mappings(data, reg)
    quality = unique(reg, lambda r: r["method_config_fingerprint"], "quality fingerprint")
    inventory = data["freeze_performance_inventory"]
    unique(inventory, lambda r: r["logical_record_id"], "inventory replicate slot")
    unique(inventory, lambda r: r["run_id"], "accepted inventory run")
    wall = unique(data["wall_raw_run_index"], lambda r: r["logical_record_id"], "host-wall replicate slot")
    require(set(wall) == {r["logical_record_id"] for r in inventory}, "host-wall slot coverage differs")
    points = unique(data["wall_point_summary"], point_key, "point summary")
    feasible = unique(data["performance_feasibility"], point_key, "feasibility")
    ratios = unique(data["wall_same_work_ratios"], point_key, "same-work ratio")
    output = []
    for source in inventory:
        row = dict(source)
        obs = wall[row["logical_record_id"]]
        # The accepted observation and replacement identity must agree exactly.
        for ik, wk in (("run_id", "run_id"), ("method_config_fingerprint", "method_config_fingerprint"),
                       ("batch_size", "batch_size"), ("context_length", "context_label"),
                       ("actual_historical_context", "historical_context"), ("graph_mode", "graph_mode"),
                       ("replacement_of", "replacement_of"), ("run_status", "status"),
                       ("executed_kernel_path_fingerprint", "kernel_path_fingerprint")):
            require(row[ik] == obs[wk], f"frozen run identity mismatch: {ik}")
        require(obs["run_kind"] == "timing" and obs["latency_basis"] == "host_wall" and obs["r_hbm"] is None, "timing scope differs")
        require(row["quality_status"] == "unvalidated" and row["performance_claim_eligible"] is False, "frozen claim fields differ")
        mapping = maps.get(row["method_config_fingerprint"])
        q = quality.get(row["method_config_fingerprint"])
        fields = ("method_config_id", "adapter_source_hash", "kernel_source_hash", "kernel_binary_hash",
                  "kernel_binary_identity_kind", "model_checkpoint", "model_revision", "tokenizer_revision", "container_digest")
        matched = mapping is not None and all(row[k] == mapping[k] for k in fields)
        pk = point_key(row)
        point, feas = points.get(pk), feasible.get(pk)
        require(point is not None and feas is not None, "missing exact point/feasibility key")
        completed = row["run_status"] == "completed"
        require(completed == row["accepted_observation"], "accepted observation status mismatch")
        require((feas["status"] == "feasible") == completed, "feasibility/status mismatch")
        require(obs["result_sha256"] == row["raw_artifact_sha256"] if completed else obs["manifest_sha256"] == row["raw_artifact_sha256"], "raw source checksum mismatch")
        ratio = ratios.get(pk)
        if ratio:
            require(ratio["latency_basis"] == "host_wall" and ratio["graph_mode"] == obs["graph_mode"] and
                    ratio["runner_kind"] == obs["runner_kind"] and ratio["warmup_steps"] == obs["warmup_steps"] and
                    ratio["measured_steps"] == obs["measured_steps"] and ratio["process_replicates"] == point["replicate_count"], "ratio work identity differs")
            require(not ratio["calculated"] or (completed and point["disposition"] == "stable"), "ratio uses invalid point")
        row.update({
            "total_attended_context": obs["total_attended_context"], "runner_kind": obs["runner_kind"],
            "warmup_steps": obs["warmup_steps"], "measured_steps": obs["measured_steps"], "process_replicates": point["replicate_count"],
            "run_kind": "timing", "r_hbm": None, "latency_basis": "host_wall",
            "host_wall_process_median_ms": obs.get("host_wall_process_median_ms") if completed else None,
            "logical_point_host_wall_median_ms": point["median_ms"] if completed else None,
            "logical_point_cv": point["cv"], "logical_point_status": point["disposition"],
            "allocated_bytes": point["allocated_bytes"], "rho_alloc": point["rho_alloc"], "r_alloc": point["r_alloc"],
            "feasibility_status": feas["status"], "predicted_required_bytes": feas["predicted_required_bytes"],
            "memory_limit_bytes": feas["limit_bytes"], "feasibility_reason": feas["reason"],
            "performance_only_ratio": ratio["performance_only_ratio"] if ratio else None,
            "ratio_null_reason": ratio["null_reason"] if ratio else "uncompressed_baseline_no_compressed_ratio",
            "ratio_grain": "logical_point_repeated_as_metadata_not_extra_observation",
            "quality_metadata_mapping_status": "exact_source_identity_match" if matched else "unresolved_source_identity",
            "b1_config_quality_outcome": q["joint_quality_status"] if matched else None,
            "b1_config_failure_stage": q["failure_stage"] if matched else None,
            "b1_config_full_ppl_status": q["full_ppl_status"] if matched else None,
            "b1_config_longbench_e_status": q["longbench_e_status"] if matched else None,
            "b1_config_v2_status": q["longbench_v2_status"] if matched else None,
            "quality_scope_status": "outside_evaluated_quality_scope" if row["batch_size"] != 1 else
                "b1_configuration_metadata_only_not_row_level_qualification",
            "row_quality_claim_eligible": False,
            "row_quality_claim_reason": "physical_batch_not_evaluated" if row["batch_size"] != 1 else
                "no_all_gate_compressed_passer_and_no_universal_length_domain_growing_to_fixed_mapping",
            "original_cross_batch_gate": "FAILED", "source_quality_roots_json": q["source_roots_json"] if matched else None,
        })
        output.append(row)
    counts = {"slots": len(output), "accepted": sum(r["accepted_observation"] for r in output),
              "capacity_infeasible": sum(r["run_status"] == "capacity_infeasible" for r in output),
              "replacements": sum(bool(r["replacement_of"]) for r in output)}
    require(counts == data["freeze_manifest"]["inventory"], "join changed frozen inventory counts")
    counts.update({"unmatched_identity_rows": sum(r["quality_metadata_mapping_status"] != "exact_source_identity_match" for r in output),
                   "duplicate_slot_count": 0, "outside_b1_scope_rows": sum(r["batch_size"] != 1 for r in output),
                   "b1_metadata_rows": sum(r["batch_size"] == 1 for r in output),
                   "logical_points": len(points), "stored_measured_ratios": sum(r["calculated"] for r in ratios.values()),
                   "stored_ratio_rows": len(ratios), "capacity_amplification_points": len(data["performance_capacity_amplification"]),
                   "row_quality_claim_eligible_count": 0})
    return output, counts, maps


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_new(path, stream.getvalue().encode())


def write_parquet(path, rows, schema=None):
    table = pa.Table.from_pylist(rows, schema=schema)
    with Path(path).open("xb") as stream:
        pq.write_table(table, stream, compression="zstd")


def render_report(summary, reg, data):
    v, g = summary["q2b"], data["q2b_evidence"]["guardrails"]
    s, j = v["overall"], summary["performance_join"]
    c = data["contract"]["quality_contract"]
    e = data["q2a_summary"]["macro"]
    code = next(r for r in v["category_results"] if r["name"] == "Code Repository Understanding")
    ppl = "; ".join(f"{r['dataset']}: relative PPL +{100*r['relative_ppl_change']:.6f}%, paired delta NLL {r['point_delta_nll']:.6f} [{r['ci_lower_delta_nll']:.6f}, {r['ci_upper_delta_nll']:.6f}]"
                    for r in data["q1b_summary"]["dataset_paired_statistics"])
    lines = ["# Q3/Q4 Quality Validation and Scoped Performance Results", "",
             "Execution COMPLETE (CPU aggregation). Scientific result: no fully qualified compressed candidate.", "",
             f"Contract `{c['id']}` / `{CONTRACT_SHA}`; physical-B=1 amendment `{data['amendment']['amendment_id']}` / `{AMENDMENT_SHA}`.",
             f"Performance freeze `{c['provenance']['performance_freeze_tag']}`. Original protocol preregistered before performance; exact manifest binding after performance results were known. The B=1 restriction followed Q0 batch failures, before Fast PPL scores. Subsequent stages selected only prior eligible configurations.",
             "Original cross-batch Q0 remains FAILED. All ten have accepted B=1 Q0 eligibility. No result transfers to B=2/4/8/16.", "",
             "| Configuration | Fast PPL | Full PPL | LongBench-E | v2 no-CoT | Final B=1 configuration outcome |",
             "|---|---|---|---|---|---|"]
    for r in reg:
        cells = [r[k] for k in ("method_config_id", "fast_ppl_status", "full_ppl_status", "longbench_e_status", "longbench_v2_status", "joint_quality_status")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Compressed candidates: {summary['compressed_status_counts']}. Fully qualified: **{summary['fully_qualified_compressed_candidate_count']}**; best configuration and speedup are **null**, not 0x or 1x. BF16 is an uncompressed reference, not a compression winner or deployment recommendation.",
              "Unexecuted stages are eligibility exclusions, not failed measurements. k4v4 entered and failed Q2B; it is not relabeled quality_pass_primary.",
              "Q2C: **not_run_no_eligible_finalist**. Native-prefill: **not_run_conditional_finalists_only**, the original accepted conditional rule; no unperformed secondary test is passed and no new eligibility waiver is inferred.", "",
              "## Positive stages and v2 limitation", "",
              "k4v4 retains Full-PPL PASS on WikiText-2 and C4 and LongBench-E PASS. Full-PPL dataset-specific paired intervals and every length result remain in inputs/q1b_summary.json; all Fast results remain in inputs/q1a_summary.json.",
              ppl + ". Both upper bounds meet the unchanged 1% dataset margin.",
              f"LongBench-E equal-task macro: BF16 {e['bf16_score_points']:.4f}, k4v4 {e['k4v4_score_points']:.4f}, drop {e['drop_score_points']:.4f} score points; saved paired 95% CI {e['paired_ci95_score_points']}. Official task/category/length results remain in inputs/q2a_summary.json; these scores are not interchangeable with PPL or v2 accuracy.",
              f"v2: BF16 {s['n11']+s['n10']}/{s['pairs']} = {100*s['bf16_accuracy']:.4f}%; k4v4 {s['n11']+s['n01']}/{s['pairs']} = {100*s['k4v4_accuracy']:.4f}%. Drop {s['accuracy_drop_pp']:.6f} pp; saved paired 95% CI {s['paired_ci95_pp']} pp. Overall non-inferiority **{g['primary_noninferiority']['status']}** under the unchanged 2 pp margin, scientific combined gate **{v['scientific_verdict']}**.",
              f"Contingency n11/n10/n01/n00 = {s['n11']}/{s['n10']}/{s['n01']}/{s['n00']}. Retention {s['n11']}/{s['retention_denominator']} = {100*s['retention']:.4f}%, CI {s['retention_ci95']}; point estimate below 95% fails. This is not a CI-lower-bound criterion.",
              f"Invalid counts {s['invalid_counts']}, increase {s['invalid_increase_pp']:.6f} pp: PASS at 1 pp. Invalid answers stay in the denominator. Length guardrails PASS. The 11 discordant saved pairs appear in q2b_discordant_pairs.csv; no answers were re-parsed or regenerated.", "",
              "| v2 category / length | N | BF16 accuracy % | k4v4 accuracy % | Drop pp | Saved 95% CI pp |",
              "|---|---:|---:|---:|---:|---|"]
    for r in v["category_results"] + v["length_results"]:
        lines.append(f"| {r['name']} | {r['pairs']} | {100*r['bf16_accuracy']:.4f} | {100*r['k4v4_accuracy']:.4f} | {r['accuracy_drop_pp']:.4f} | {r['paired_ci95_pp']} |")
    lines += ["", f"Hard failures: {v['gate_reasons']}. Code-category N={code['pairs']}: {code['n11']+code['n10']}/{code['pairs']} versus {code['n11']+code['n01']}/{code['pairs']}, drop {code['accuracy_drop_pp']:.4f} pp, CI {code['paired_ci95_pp']} pp, exceeding the 5 pp limit. Small subgroup denominators and null retention intervals remain explicit in the saved summary. The overall CI alone does not establish mean degradation greater than 2 pp; crossing zero is not equivalence.",
              f"Scope: {v['selected']} frozen IDs; {v['eligible']} eligible, {v['excluded']} predeclared length exclusions. B=1, no-CoT, 16 original prompt conditioning tokens, eight generated tokens maximum and the pinned parser. Budget stops {v['generation']['bf16']['stop_reasons']['budget_exhausted']}/{s['pairs']} BF16 and {v['generation']['k4v4']['stop_reasons']['budget_exhausted']}/{s['pairs']} k4v4; no evidence here establishes that budget stops caused failure.", "",
              "## Frozen performance sidecar", "",
              f"{j['slots']} replicate slots retained: {j['accepted']} accepted observations, {j['capacity_infeasible']} infeasible slots, {j['replacements']} replacement links (not extra observations). {j['unmatched_identity_rows']} unmatched identities and {j['duplicate_slot_count']} duplicate slots.",
              f"Join uses contract fingerprint plus exact frozen model/tokenizer, adapter, kernel-source/binary and container identity, and the original accepted run/shape mapping. {j['outside_b1_scope_rows']} B>1 rows are outside evaluated quality scope, NOT failed quality measurements. {j['b1_metadata_rows']} B=1 rows receive configuration metadata only, not arbitrary length/domain qualification.",
              "Stored Q0 eager/Graph relationships are exact-shape, fixed-L controls only (see identity_mapping.json); quality growing-context runtime fingerprints are not equated to a universal performance fingerprint. Protected-path comparison receipts are inherited, not re-executed.",
              f"{j['stored_measured_ratios']} frozen host-wall same-work ratios are reused at logical-point grain; repeated sidecar values do not add statistical weight. {j['capacity_amplification_points']} capacity-amplification points remain separate in the compact input table. No predictor fills null ratios. Infeasible rows have no invented latency.",
              "All historical quality_status=unvalidated and performance-only fields remain unchanged. Every derived timing row keeps r_hbm=null. The retained profiler-feature table is scoped only to Phase15 B=1/L=131071/Graph; no traffic is extrapolated. No quality-preserving headline or qualified frontier is supported.", "",
              "## Interpretation and reproducibility", "",
              "Quality failures alone do not establish a timing/runtime defect. KVQuant severe Fast-PPL degradation is observed for these exact implementations; its cause remains unresolved, not attributed to the upstream algorithm universally.",
              "Phase17/18 limitations are unchanged: all four predictive targets missed; selected outer scores have a model-selection limitation; only 23/50 knees identified and knee error not evaluable. BF16 held-out B=1/L4096 predicted 668.6230 ms versus observed 11.7736 ms (relative error 55.7899): a real edge extrapolation failure, not repaired or refitted. Macro metric labels and absence of separate saved fold parameter vectors remain explicit in the source reports.",
              "Future changes to budgets, thresholds, implementations or studies require separate protocols. None started. No GPU, inference, admission, performance rerun, bootstrap regeneration, refit, image build, or historical bulk verification occurred.",
              "Reproduce from this compact bundle (Python with pyarrow; no model, GPU or network):", "",
              "```sh", "python reproduce.py --reproduce . --output /new/empty/q3-q4-results", "```", "",
              "The source manifest binds all compact inputs and ledger entries. Artifact COMPLETE/inventory/checksums and the external publication receipt bind this new bundle. Prior roots are referenced, never bulk-copied. Stop for human review.", ""]
    return "\n".join(lines)


def derive(data, output):
    output.mkdir(parents=True, exist_ok=False)
    reg = registry(data)
    discordant = arithmetic_check(data["q2b_summary"], data["q2b_pairs"]["rows"])
    joined, counts, maps = join_performance(data, reg)
    compressed = [r for r in reg if r["is_compressed"]]
    qualified = [r for r in compressed if r["fully_qualified_compressed"]]
    require(not qualified, "this closure requires empty qualified set; inspect changed scientific inputs")
    summary = {
        "schema_version": "kvbench-q3-q4-joint-results-1.0.0", "execution_status": "COMPLETE", "cpu_only": True,
        "starting_head": data["provenance"]["starting_head"], "execution_head": data["provenance"]["execution_head"],
        "contract_id": data["contract"]["quality_contract"]["id"], "contract_sha256": CONTRACT_SHA,
        "amendment_id": data["amendment"]["amendment_id"], "amendment_sha256": AMENDMENT_SHA,
        "performance_freeze": data["freeze_correction"]["active_freeze_binding"],
        "source_roots": {p: data[p + "_receipt"]["root_sha256"] for p in ("q1a", "q1b", "q2a", "q2b")},
        "inherited_performance_source_roots": data["freeze_release_sources"]["roots"],
        "qp1_root": data["amendment"]["base_contract"]["bundle_root"],
        "q0_roots": data["amendment"]["motivation_evidence"],
        "performance_host_wall_root": data["freeze_manifest"]["phase16_primary_host_wall_root_sha256"],
        "compressed_configuration_count": len(compressed), "compressed_status_counts": dict(Counter(r["joint_quality_status"] for r in compressed)),
        "fully_qualified_compressed_candidate_count": len(qualified), "best_qualified_compressed_config": None,
        "best_qualified_compressed_speedup": None, "reason": EMPTY_REASON,
        "original_cross_batch_gate": "FAILED", "physical_quality_batch_sizes": [1],
        "q2c_status": "not_run_no_eligible_finalist", "native_prefill_status": "not_run_conditional_finalists_only",
        "performance_join": counts, "q2b": data["q2b_summary"], "q2b_guardrails": data["q2b_evidence"]["guardrails"],
        "discordant_pairs": len(discordant), "bootstrap_regenerated": False, "inference_executed": False,
        "performance_refit": False, "contract_modified": False,
        "modeling_limitations": data["freeze_release_sources"]["modeling_limitations"],
        "stage_execution_counts": {"q1a": data["q1a_summary"]["completed_anchor_units"], "q1b": data["q1b_summary"]["completed_anchor_units"],
                                   "q2a": data["q2a_summary"]["completed_outputs"], "q2b": data["q2b_summary"]["completed_outputs"]},
    }
    write_csv(output / "quality_admission_table.csv", reg)
    write_parquet(output / "performance_quality_join.parquet", joined)
    schema = pa.schema([("method_config_id", pa.string()), ("method_config_fingerprint", pa.string()),
                        ("physical_quality_batch_size", pa.int64()), ("joint_quality_status", pa.string()),
                        ("quality_preserving_speedup", pa.float64()), ("reason", pa.string())],
                       metadata={b"empty_reason": EMPTY_REASON.encode()})
    write_parquet(output / "qualified_compressed_candidates.parquet", [], schema)
    write_csv(output / "q2b_discordant_pairs.csv", discordant)
    write_json(output / "joint_result_summary.json", summary)
    write_json(output / "identity_mapping.json", {"mapping_kind": "exact_configuration_source_lineage_not_universal_runtime_fingerprint", "rows": list(maps.values())})
    write_new(output / "QUALITY_VALIDATION_REPORT.md", render_report(summary, reg, data).encode())
    return summary


def build(repo, focused_tests):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fz")
    bundle_id = f"q3-q4-{stamp}-{head[:8]}-{secrets.token_hex(3)}"
    root = repo / "artifacts/joint_results" / bundle_id
    root.mkdir(parents=True, exist_ok=False)
    files = []
    for key, relative in source_paths(repo).items():
        source = repo / relative
        binding = selected_ledger_binding(repo, source)
        destination = "inputs/" + key + (".parquet" if source.suffix == ".parquet" else ".md" if source.suffix == ".md" else ".json")
        write_new(root / destination, source.read_bytes())
        files.append({"key": key, "source_path": relative.as_posix(), "bundle_path": destination,
                      "sha256": sha(source), "size_bytes": source.stat().st_size, "existing_ledger_binding": binding})
    provenance = {"schema_version": "kvbench-q3-q4-compact-sources-1.0.0", "starting_head": STARTING_HEAD,
                  "execution_head": head, "files": files, "no_historical_bulk_revalidation": True}
    write_json(root / "source_manifest.json", provenance)
    prompt = repo / "Q3_Q4_CPU_JOINT_ADMISSION_PROMPT.md"
    write_json(root / "execution_authorization.json", {
        "status": "APPROVED", "approved_by": "operator", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_scope": ["Q3_CPU_JOINT_ADMISSION", "Q4_CPU_PERFORMANCE_QUALITY_JOIN", "NEW_COMPACT_RESULT_PUBLICATION"],
        "approval_text": "I authorize CPU-only Q3 joint admission followed by Q4 quality/performance result joining, using existing evidence. Preserve k4v4 Full-PPL and LongBench-E PASS, Q2B FAIL and INCONCLUSIVE overall v2 non-inferiority separately; retain all nine compressed configurations, zero qualifiers and B=1 scope; publish only the new compact result bundle, then stop.",
        "operator_prompt_sha256": sha(prompt), "frozen_contract_modified": False})
    write_new(root / "operator_prompt.md", prompt.read_bytes())
    write_new(root / "reproduce.py", Path(__file__).read_bytes())
    write_new(root / "focused_tests.json", focused_tests.read_bytes())
    summary = derive(read_inputs(root), root / "derived")
    # Atomic moves within this new, unfinished bundle; no source evidence touched.
    for name in PRODUCTS:
        (root / "derived" / name).rename(root / name)
    (root / "derived").rmdir()
    write_json(root / "manifest.json", {"schema_version": "kvbench-q3-q4-manifest-1.0.0", "run_id": bundle_id,
        "status": "COMPLETE", "execution_head": head, "starting_head": STARTING_HEAD, "cpu_only": True,
        "source_roots": summary["source_roots"], "fully_qualified_compressed_candidate_count": 0,
        "append_only": True, "complete_written_last": True})
    return root


def seal(root):
    """Use the existing repository inventory/ledger/COMPLETE publication format."""
    require(not (root / "COMPLETE").exists(), "bundle already finalized")
    payloads = sorted(p for p in root.rglob("*") if p.is_file())
    write_json(root / "artifact_inventory.json", {"schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": root.name,
        "files": [{"path": p.relative_to(root).as_posix(), "role": "q3_q4_cpu_evidence", "size_bytes": p.stat().st_size,
                   "sha256": sha(p)} for p in payloads], "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"]})
    ledger = "".join(f"{sha(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(root.rglob("*")) if p.is_file())
    write_new(root / "checksums.sha256", ledger.encode())
    write_json(root / "COMPLETE", {"schema_version": "kvbench-completion-1.0.0", "run_id": root.name, "status": "COMPLETE",
        "manifest_sha256": sha(root / "manifest.json"), "artifact_inventory_sha256": sha(root / "artifact_inventory.json"),
        "checksum_ledger_path": "checksums.sha256", "checksum_ledger_sha256": sha(root / "checksums.sha256"), "written_last": True})
    for p in sorted(root.rglob("*"), reverse=True):
        p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)
    from scripts.r2_artifact import validate_local_artifact
    artifact = validate_local_artifact(root, environ={})
    return {"status": "PASS", "root_sha256": artifact.root_sha256, "object_count": len(artifact.files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--build", action="store_true")
    command.add_argument("--reproduce", type=Path)
    command.add_argument("--seal", type=Path)
    parser.add_argument("--focused-tests", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.build:
        require(args.focused_tests is not None, "focused test receipt required")
        print(build(ROOT, args.focused_tests))
    elif args.seal:
        print(json.dumps(seal(args.seal)))
    else:
        require(args.output is not None, "new reproduction output required")
        summary = derive(read_inputs(args.reproduce), args.output)
        print(json.dumps({"status": "COMPLETE", "slots": summary["performance_join"]["slots"],
                          "qualified": summary["fully_qualified_compressed_candidate_count"], "gpu_or_network": False}))


if __name__ == "__main__":
    main()
