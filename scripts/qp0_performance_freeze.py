#!/usr/bin/env python3
"""Assemble and validate the compact QP-0 performance freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.r2_artifact import validate_local_artifact


ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE_SOURCE = "83536c37433875cda98c36e2848e05692e9407d0"
FAMILY_ID = "phase16-20260831t123029614620z-ec534d99-de80ac"
FAMILY = ROOT / "artifacts/phase16" / FAMILY_ID
RUN_INDEX = FAMILY / "wall-closure/raw_run_index.parquet"
FAMILY_MANIFEST = FAMILY / "outer/family_manifest.json"
CONTAINER_DIGEST = "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
MODEL_CONFIG = ROOT / "configs/models/primary_gqa_model.yaml"
QUALITY_IMAGE = ROOT / "docker/quality.image.json"
ARTIFACT_ROOT = ROOT / "artifacts/performance_freeze"
ID_RE = re.compile(r"qp0-freeze-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}")

SEGMENT_ROOTS = [
    "7f683da8c427ffa78c3617a102dd47bf6cc533da6cc10c35f8dee50aa7b68f4d",
    "a9e99b51a6d35c14733266bf34718fdb936ec144d4e4598f9c830b4d8213a355",
    "c859c77ebe820e743f23d53400fb36768bf8cc3953244285ecd244459d3511fd",
    "56152a5a3151170c1e3919bb24af35adafcd39aec5cf91b610f3c5445364b168",
    "78695bf14b4262debbc145a379a5d7f3c5936d1c6e8a295c55b125d8effaa09d",
]
ROOTS = {
    "phase13_pilot_design": "feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531",
    "phase13d_densification_design": "a8559a5e01edaad949df1e128c4bddff37638801cfc89f2eb8d4894c31ef82d2",
    "phase14_graph_ab_timing": "22a613b07c1ee6d3e9a0a7fc81df6065ccc8a10bf783b1a69aded3c2eb8068f0",
    "phase14_analysis_closure": "4cd29ea1b94201f493db8cef9ebd01933b4c4f573185eff317ec5af81e9fb000",
    "phase15_profiler_mechanism": "641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8",
    "phase16_outer_cuda_event_secondary": "d74587675dd59b464d81c6e82885d3c9706c681a9da216ad1a7fe4c6ccd88daa",
    "phase16_host_wall_primary": "5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e",
    "phase17_modeling": "05d5c4e82259ca8bf2d43e62a689c1180f091da1fd166c79106230ab455f98bb",
    "phase18_reproduction": "cd6ee2324d0163088102e29e45b4ad124b5d524fb1900717faddf997da6805e2",
}

LOCKED_PATHS = [
    "src/kvbench/adapters/",
    "src/kvbench/runtime/",
    "src/kvbench/third_party/vllm_turboquant/",
    "configs/methods/",
    "configs/models/",
    "scripts/phase12_unified_admission.py",
    "scripts/phase13_pilot.py",
    "scripts/phase16_full_scan.py",
    "scripts/phase16_logical_prefix.py",
]


class FreezeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def git(*args: str, binary: bool = False) -> bytes | str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=not binary)


def method_family(config_id: str) -> str:
    if config_id == "bf16":
        return "bf16"
    if config_id.startswith("tq_"):
        return "turboquant"
    if config_id.startswith("k") and not config_id.startswith("kvq"):
        return "kivi"
    if config_id.startswith("kvq"):
        return "kvquant"
    raise FreezeError(f"unknown method config: {config_id}")


def aggregate_hash(source: str, paths: list[str]) -> str:
    payload = bytearray()
    for path in sorted(paths):
        data = git("show", f"{source}:{path}", binary=True)
        assert isinstance(data, bytes)
        payload.extend(path.encode() + b"\0" + hashlib.sha256(data).digest())
    return sha256_bytes(bytes(payload))


def locked_hashes(source: str) -> dict[str, Any]:
    names_raw = git("ls-tree", "-r", "--name-only", source)
    assert isinstance(names_raw, str)
    names = names_raw.splitlines()
    selected = sorted(
        name for name in names
        if any(name == item.rstrip("/") or name.startswith(item) for item in LOCKED_PATHS)
    )
    if not selected:
        raise FreezeError("locked path expansion is empty")
    files = []
    for name in selected:
        blob_raw = git("rev-parse", f"{source}:{name}")
        assert isinstance(blob_raw, str)
        data = git("show", f"{source}:{name}", binary=True)
        assert isinstance(data, bytes)
        files.append({"path": name, "git_blob": blob_raw.strip(), "sha256": sha256_bytes(data), "size_bytes": len(data)})
    return {
        "schema_version": "kvbench-qp0-locked-hot-path-hashes-1.0.0",
        "performance_source_commit": source,
        "locked_paths": LOCKED_PATHS,
        "tracked_file_count": len(files),
        "files": files,
        "files_sha256": sha256_bytes(json_bytes(files)),
    }


def external_identities(family_manifest: dict[str, Any]) -> dict[str, Any]:
    refs = {
        "bf16": {
            "authority": "configs/methods/bf16.yaml",
            "kernel_binary_sha256": "b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984",
        },
        "turboquant": {
            "authority": "docs/evidence/phase6/turboquant-method-admission.json",
            "source_commit": "752a3a504485790a2e8491cacbb35c137339ad34",
            "source_tree": "3ec7a4eb00f9bc8fec399bea6cf7de27a7936372",
            "binary_identity": "jit_source_authority_no_single_static_binary",
        },
        "kivi": {
            "authority": "docs/evidence/phase8/kivi-method-admission.json",
            "official_base_commit": "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6",
            "patched_tree": "b617493dea5aff1a754cd27ad6be12ac512b2aee",
            "extension_sha256": "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9",
        },
        "kvquant": {
            "authority": ["docs/evidence/phase11rq23/kvquant-method-admission.json", "docs/evidence/phase13rq4/kvquant-q4-method-admission.json"],
            "source_commit": "34b0bdfa83082e1f30387d9ac5cca369006e089c",
            "source_tree": "1f85af65fe03061583ffe8bd91e47d7ecffdd312",
            "aggregate_patch_sha256": "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a",
            "extension_sha256": "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d",
            "calibration_root_sha256": "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf",
        },
    }
    for value in refs.values():
        authorities = value["authority"] if isinstance(value["authority"], list) else [value["authority"]]
        value["authority_sha256"] = {path: sha256_file(ROOT / path) for path in authorities}
    files = family_manifest["timing_critical_hashes"]["files"]
    family_sources = {
        "bf16": ["src/kvbench/adapters/bf16.py", "src/kvbench/runtime/bf16_endpoint.py", "src/kvbench/runtime/fixed_l_runner.py"],
        "turboquant": ["src/kvbench/adapters/turboquant.py", "src/kvbench/runtime/turboquant_cache.py"],
        "kivi": ["src/kvbench/adapters/kivi.py", "src/kvbench/runtime/kivi_cache.py"],
        "kvquant": ["src/kvbench/adapters/kvquant.py", "src/kvbench/runtime/kvquant_cache.py"],
    }
    for family, paths in family_sources.items():
        refs[family]["performance_repo_source_hashes"] = {path: files.get(path) for path in paths}
        refs[family]["kernel_source_aggregate_sha256"] = aggregate_hash(PERFORMANCE_SOURCE, [path for path in paths if path in files])
    return {"schema_version": "kvbench-qp0-external-identities-1.0.0", "families": refs}


def process_registry_check() -> dict[str, Any]:
    needles = ("phase13_pilot.py", "phase14_graph_ab.py", "phase16_full_scan.py", "phase16_full_scan_continuation.py")
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(value in command for value in needles):
            matches.append({"pid": int(entry.name), "command": command})
    return {
        "schema_version": "kvbench-qp0-process-registry-check-1.0.0",
        "captured_at_utc": utc_now(),
        "method": "single_proc_cmdline_scan_for_known_performance_coordinators",
        "active_performance_processes": matches,
        "status": "PASS" if not matches else "FAIL",
    }


def image_identity() -> dict[str, Any]:
    payload = load_json(QUALITY_IMAGE)
    required = {"base_image_digest", "quality_image_digest", "dependency_delta", "runtime_identity", "status"}
    if not required.issubset(payload) or payload["base_image_digest"] != CONTAINER_DIGEST or payload["status"] != "PASS":
        raise FreezeError("quality image identity is missing or invalid")
    return payload


def build_inventory(family: dict[str, Any], model: dict[str, Any], external: dict[str, Any]) -> tuple[pa.Table, dict[str, int]]:
    rows = pq.read_table(RUN_INDEX).to_pylist()
    if len(rows) != 2670:
        raise FreezeError(f"expected 2670 inventory slots, found {len(rows)}")
    segment_roots = {index: root for index, root in enumerate(SEGMENT_ROOTS)}
    output = []
    for row in rows:
        config_id = row["method_config_id"]
        family_name = method_family(config_id)
        authority = external["families"][family_name]
        completed = row["status"] == "completed"
        relative = row["result_path"] if completed else row["manifest_path"]
        checksum = row["result_sha256"] if completed else row["manifest_sha256"]
        if not relative or not checksum:
            raise FreezeError(f"missing raw locator/checksum for {row['run_id']}")
        binary = authority.get("kernel_binary_sha256") or authority.get("extension_sha256") or authority["kernel_source_aggregate_sha256"]
        output.append({
            "run_id": row["run_id"],
            "logical_record_id": row["logical_record_id"],
            "method": family_name,
            "method_config_id": config_id,
            "method_config_fingerprint": row["method_config_fingerprint"],
            "model_checkpoint": model["model_id"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
            "adapter_source_hash": family["timing_critical_hashes"]["files"][f"src/kvbench/adapters/{family_name if family_name != 'bf16' else 'bf16'}.py"],
            "kernel_source_hash": authority["kernel_source_aggregate_sha256"],
            "kernel_binary_hash": binary,
            "kernel_binary_identity_kind": "admission_extension_or_frozen_binary" if authority.get("extension_sha256") or authority.get("kernel_binary_sha256") else "jit_source_authority_no_single_static_binary",
            "executed_kernel_path_fingerprint": row["kernel_path_fingerprint"],
            "container_digest": CONTAINER_DIGEST,
            "git_sha": family["execution_git_sha"],
            "batch_size": row["batch_size"],
            "context_length": row["context_label"],
            "actual_historical_context": row["historical_context"],
            "graph_mode": row["graph_mode"],
            "run_status": row["status"],
            "raw_artifact_path": f"r2://kvbench-artifacts/kvbench/sha256/{segment_roots[row['replicate_index']]}/{relative}",
            "raw_artifact_sha256": checksum,
            "segment_root_sha256": segment_roots[row["replicate_index"]],
            "replicate_index": row["replicate_index"],
            "replacement_of": row["replacement_of"],
            "accepted_observation": completed,
            "quality_status": "unvalidated",
            "claim_eligibility": "performance_only",
            "performance_claim_eligible": False,
        })
    counts = {
        "slots": len(output),
        "accepted": sum(item["run_status"] == "completed" for item in output),
        "capacity_infeasible": sum(item["run_status"] == "capacity_infeasible" for item in output),
        "replacements": sum(bool(item["replacement_of"]) for item in output),
    }
    if counts != {"slots": 2670, "accepted": 2205, "capacity_infeasible": 465, "replacements": 38}:
        raise FreezeError(f"inventory accounting differs: {counts}")
    return pa.Table.from_pylist(output), counts


def quality_contract(freeze_id: str, tag: str, fingerprints: dict[str, str]) -> dict[str, Any]:
    model = load_json(MODEL_CONFIG)
    method_params: dict[str, Any] = {}
    for path in ("bf16.yaml", "turboquant.yaml", "kivi.yaml", "kvquant.yaml"):
        payload = load_json(ROOT / "configs/methods" / path)
        for variant in payload["variants"]:
            raw = variant["variant_id"]
            config_id = {"turboquant_4bit_nc": "tq_4bit_nc", "turboquant_k3v4_nc": "tq_k3v4_nc", "turboquant_3bit_nc": "tq_3bit_nc"}.get(raw, raw)
            if config_id in fingerprints:
                method_params[config_id] = variant["parameters"]
    configurations = [
        {"method_config_id": config_id, "method_family": method_family(config_id), "method_config_fingerprint": fingerprints[config_id], "parameters": method_params[config_id]}
        for config_id in ("bf16", "tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc", "k4v4", "k2v4", "k2v2", "kvq4", "kvq3", "kvq2")
    ]
    return {
        "quality_contract": {
            "id": f"quality-{freeze_id}",
            "status": "requires_human_approval",
            "approval": {"approved": False, "approved_by": None, "approved_at_utc": None},
            "provenance": {
                "performance_freeze_tag": tag,
                "performance_freeze_manifest": f"artifacts/performance_freeze/{freeze_id}/manifest.json",
                "protocol_commit_sha": "a7b8285dd8ed2fb598efbb3312e9f55064a0ee64",
                "protocol_preregistered_before_performance": True,
                "manifest_binding_prepared_after_performance_results_known": True,
            },
            "model": {
                "checkpoint": model["model_id"], "revision": model["revision"],
                "tokenizer_id": model["tokenizer_id"], "tokenizer_revision": model["tokenizer_revision"],
                "tokenizer_config_sha256": model["tokenizer_config_sha256"],
                "chat_template_hash": {"status": "pending_qp1", "reason": "not_present_in_frozen_performance_manifest"},
                "max_model_length": model["target_context_length"], "weight_dtype": model["weight_dtype"],
                "rope": model["rope"], "require_same_checkpoint_as_performance": True,
            },
            "execution": {
                "quality_batch_size": 1, "temperature": 0.0, "top_p": 1.0, "do_sample": False,
                "seed": 20260722, "primary_protocol": "cache_sensitive",
                "secondary_protocol": "native_prefill_finalists_only", "decode_conditioning_tokens": 16,
                "require_compressed_cache_read_assertion": True,
                "implementation_equivalence": "exact frozen adapter/kernel/config identity; quality code outside locked paths",
            },
            "configurations": {"source": "performance_inventory", "require_exact_method_config_fingerprint": True, "run_q0_for_all": True, "run_fast_ppl_for_all": True, "items": configurations},
            "ppl": {
                "mode": "incremental_teacher_forcing", "burn_in_decode_tokens": 1,
                "datasets": [
                    {"name": "wikitext-2-raw-v1", "split": "test", "revision": {"status": "pending_qp1"}},
                    {"name": "c4", "split": "validation", "revision": {"status": "pending_qp1"}, "sample_ids_file": {"status": "pending_qp1"}},
                ],
                "document_boundary": {"separator": "eos", "ignored_tokens_after_boundary": 32},
                "fast": {"prefix_lengths": [4096, 24576, 32768, 65536], "anchors_per_length": 16, "scored_tokens_per_anchor": 128},
                "full": {"prefix_lengths": [4096, 16384, 24576, 28672, 32768, 65536, 98304, 130560], "anchors_per_length": 64, "scored_tokens_per_anchor": 256},
                "primary_metric": "delta_nll",
            },
            "longbench_e": {"revision": {"status": "pending_qp1"}, "sample_ids": {"status": "pending_qp1"}, "cache_sensitive": True, "decode_conditioning_tokens": 16, "official_prompts": True, "official_metrics": True, "truncate_overlength_primary": False, "tasks": ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]},
            "longbench_v2": {"revision": {"status": "pending_qp1"}, "sample_ids": {"status": "pending_qp1"}, "cache_sensitive": True, "decode_conditioning_tokens": 16, "cot_primary": False, "cot_finalist_stress": True, "max_new_tokens_no_cot": 8, "valid_answers": ["A", "B", "C", "D"], "finalist_rule": ["best_quality_among_primary_passers", "highest_r_alloc_among_primary_passers", "do_not_use_speed_for_selection"]},
            "invariance": {"graph_eager_sample_count": 100, "batch_sample_count": 100, "batch_sizes": [1, 4, 8]},
            "gates": {
                "ppl": {"global_relative_increase_max": 0.01, "bucket_review_threshold": 0.02, "bucket_hard_fail": 0.05},
                "longbench_e": {"macro_score_drop_max_points": 2.0, "category_review_points": 3.0, "category_hard_fail_points": 5.0, "invalid_output_increase_max_pp": 1.0},
                "longbench_v2": {"accuracy_drop_max_pp": 2.0, "any_length_bucket_drop_max_pp": 5.0, "any_category_drop_max_pp": 5.0, "invalid_output_increase_max_pp": 1.0, "baseline_correct_retention_min": 0.95},
                "decision_states": ["pass", "fail", "inconclusive"],
            },
            "pending_qp1_items": ["quality_evaluation_dependency_lock", "chat_template_hash", "dataset_revisions", "sample_id_sets", "prompt_template_hashes", "dataset_and_prompt_materialization", "formal_human_approval"],
            "quality_execution": "LOCKED",
        }
    }


def payload_files(root: Path, excluded: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in excluded)


def seal(stage: Path, freeze_id: str) -> Path:
    inventory_rows = pq.read_table(stage / "performance_inventory.parquet").num_rows
    items = [{"path": path.relative_to(stage).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in payload_files(stage, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"})]
    write_new(stage / "artifact_inventory.json", json_bytes({"schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": freeze_id, "files": items, "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"]}))
    ledger = "".join(f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n" for path in payload_files(stage, {"checksums.sha256", "COMPLETE"})).encode()
    write_new(stage / "checksums.sha256", ledger)
    write_new(stage / "COMPLETE", json_bytes({"schema_version": "kvbench-completion-1.0.0", "run_id": freeze_id, "status": "PASS", "manifest_sha256": sha256_file(stage / "manifest.json"), "artifact_inventory_sha256": sha256_file(stage / "artifact_inventory.json"), "checksum_ledger_path": "checksums.sha256", "checksum_ledger_sha256": sha256_file(stage / "checksums.sha256"), "written_last": True}))
    final = ARTIFACT_ROOT / freeze_id
    os.rename(stage, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    result = validate(final)
    if result["inventory_count"] != inventory_rows:
        raise FreezeError("sealed inventory row count drifted")
    return final


def build(freeze_id: str, metadata_commit: str) -> dict[str, Any]:
    if not ID_RE.fullmatch(freeze_id):
        raise FreezeError("invalid freeze ID")
    if (ARTIFACT_ROOT / freeze_id).exists():
        raise FreezeError("freeze ID already exists")
    family = load_json(FAMILY_MANIFEST)
    if family["execution_git_sha"] != "ec534d9958d5616eef981c628b87a92c7c809872" or family["authorized_container_digest"] != CONTAINER_DIGEST:
        raise FreezeError("Full Scan execution identity differs")
    process_check = process_registry_check()
    if process_check["status"] != "PASS":
        raise FreezeError("an active performance process remains")
    model = load_json(MODEL_CONFIG)
    image = image_identity()
    hot = locked_hashes(PERFORMANCE_SOURCE)
    external = external_identities(family)
    table, counts = build_inventory(family, model, external)
    tag = f"perf-freeze-20260917-{PERFORMANCE_SOURCE[:8]}"
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = ARTIFACT_ROOT / f".{freeze_id}.{secrets.token_hex(8)}.staging"
    stage.mkdir(mode=0o700)
    pq.write_table(table, stage / "performance_inventory.parquet", compression="zstd")
    write_new(stage / "locked_paths.txt", ("\n".join(LOCKED_PATHS) + "\n").encode())
    write_new(stage / "locked_hot_path_hashes.json", json_bytes(hot))
    write_new(stage / "external_kernel_identities.json", json_bytes(external))
    write_new(stage / "process_registry_check.json", json_bytes(process_check))
    hardware_path = ROOT / "docs/evidence/e00/e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32/manifest.json"
    hardware = load_json(hardware_path)
    write_new(stage / "hardware_identity.json", json_bytes({"schema_version": "kvbench-qp0-hardware-reference-1.0.0", "source_path": hardware_path.relative_to(ROOT).as_posix(), "source_sha256": sha256_file(hardware_path), "gpu": hardware["gpu"], "note": "historical measurement hardware identity; not replaced with current host identity"}))
    write_new(stage / "quality_image.json", json_bytes(image))
    contract = quality_contract(freeze_id, tag, family["method_fingerprints"])
    write_new(stage / "quality_contract.yaml", json_bytes(contract))
    write_new(stage / "quality_contract_handoff.json", json_bytes({"schema_version": "kvbench-qp0-quality-handoff-1.0.0", "status": "PENDING_QP1_HUMAN_APPROVAL", "quality_execution": "LOCKED", "resolved": ["performance_manifest_binding", "model_and_tokenizer_identity", "ten_exact_method_fingerprints", "cache_sensitive_protocol", "quality_margins_and_decision_states"], "pending_qp1_items": contract["quality_contract"]["pending_qp1_items"], "quality_evaluation_executed": False}))
    role_roots = [{"role": role, "root_sha256": digest, "uri": f"r2://kvbench-artifacts/kvbench/sha256/{digest}/", "verification": "inherited_publication_receipt_and_checksum_ledger"} for role, digest in ROOTS.items()]
    role_roots.extend({"role": f"phase16_replicate_{index}_raw_timing", "root_sha256": digest, "uri": f"r2://kvbench-artifacts/kvbench/sha256/{digest}/", "verification": "inherited_publication_receipt_and_checksum_ledger"} for index, digest in enumerate(SEGMENT_ROOTS))
    release_sources = {
        "schema_version": "kvbench-qp0-release-sources-1.0.0",
        "roots": role_roots,
        "source_hash_verification_scope": "existing compact manifests, checksum ledgers, and publication receipts; no historical bulk reread",
        "stopped_campaigns_role": "historical_failure_evidence_only_not_inventory_observations",
        "host_wall_supplement_role": "authority_correction_join_not_additional_observation",
        "modeling_limitations": ["all four predeclared predictive targets missed", "knee error not evaluable", "23_of_50_knees_identified", "outer_selected_scores_have_selection_limitation", "individual_fold_parameter_vectors_not_saved"],
    }
    write_new(stage / "release_sources.json", json_bytes(release_sources))
    manifest = {
        "schema_version": "kvbench-qp0-performance-freeze-manifest-1.0.0",
        "run_id": freeze_id, "freeze_id": freeze_id, "status": "PASS", "created_at_utc": utc_now(),
        "performance_source_commit": PERFORMANCE_SOURCE, "qp0_metadata_commit": metadata_commit,
        "intended_local_tag": tag, "tag_target": PERFORMANCE_SOURCE,
        "inventory": counts, "full_scan_family_id": FAMILY_ID,
        "phase16_primary_host_wall_root_sha256": ROOTS["phase16_host_wall_primary"],
        "method_config_fingerprints": family["method_fingerprints"],
        "measurement_container_digest": CONTAINER_DIGEST,
        "quality_image_digest": image["quality_image_digest"],
        "quality_status": "unvalidated", "claim_eligibility": "performance_only",
        "performance_data_frozen": True, "quality_execution": "LOCKED",
        "quality_contract_status": "requires_human_approval", "quality_evaluation_executed": False,
        "append_only": True, "complete_written_last": True,
    }
    write_new(stage / "manifest.json", json_bytes(manifest))
    final = seal(stage, freeze_id)
    return {"path": str(final), **validate(final)}


def validate(root: Path) -> dict[str, Any]:
    local = validate_local_artifact(root, environ={})
    required = {"performance_inventory.parquet", "manifest.json", "locked_paths.txt", "locked_hot_path_hashes.json", "external_kernel_identities.json", "hardware_identity.json", "quality_image.json", "quality_contract.yaml", "quality_contract_handoff.json", "process_registry_check.json", "release_sources.json", "artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise FreezeError(f"missing bundle files: {missing}")
    manifest = load_json(root / "manifest.json")
    table = pq.read_table(root / "performance_inventory.parquet")
    rows = table.to_pylist()
    counts = {"accepted": sum(r["run_status"] == "completed" for r in rows), "capacity_infeasible": sum(r["run_status"] == "capacity_infeasible" for r in rows), "replacements": sum(bool(r["replacement_of"]) for r in rows)}
    if len(rows) != 2670 or counts != {"accepted": 2205, "capacity_infeasible": 465, "replacements": 38}:
        raise FreezeError(f"invalid inventory accounting: {len(rows)} {counts}")
    if any(r["quality_status"] != "unvalidated" or r["claim_eligibility"] != "performance_only" or r["performance_claim_eligible"] is not False for r in rows):
        raise FreezeError("inventory quality/claim labels differ")
    if manifest.get("performance_source_commit") != PERFORMANCE_SOURCE or manifest.get("quality_execution") != "LOCKED" or manifest.get("quality_evaluation_executed") is not False:
        raise FreezeError("freeze scope differs")
    contract = load_json(root / "quality_contract.yaml")["quality_contract"]
    if contract["status"] != "requires_human_approval" or contract["approval"]["approved"] is not False or len(contract["configurations"]["items"]) != 10:
        raise FreezeError("quality contract approval/configuration state differs")
    hot = load_json(root / "locked_hot_path_hashes.json")
    if hot["performance_source_commit"] != PERFORMANCE_SOURCE or not hot["files"]:
        raise FreezeError("locked hashes differ")
    if load_json(root / "process_registry_check.json")["status"] != "PASS":
        raise FreezeError("process registry was not clean")
    return {"status": "PASS", "root_sha256": local.root_sha256, "object_count": len(local.files), "inventory_count": len(rows), **counts, "locked_file_count": hot["tracked_file_count"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--freeze-id", required=True)
    build_parser.add_argument("--metadata-commit", required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = build(args.freeze_id, args.metadata_commit) if args.command == "build" else validate(args.artifact)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
