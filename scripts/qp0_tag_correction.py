#!/usr/bin/env python3
"""Build and validate the compact QP-0 tag-correction closure bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
from typing import Any

from scripts.r2_artifact import validate_local_artifact


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "83536c37433875cda98c36e2848e05692e9407d0"
FREEZE_ID = "qp0-freeze-20260917t115200000000z-83536c37-bf28e4"
FREEZE_ROOT = "9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f"
MANIFEST_SHA = "91db28a33940e9bbdda6c723a2678ae9459e73f813b20bfecb6c97222ba38fc5"
ORIGINAL_TAG = "perf-freeze-20260917-83536c37"
ORIGINAL_TAG_OBJECT = "c5049cf73c78a96729c4e8a791ae77acf7a1963c"
CORRECTED_TAG = "perf-freeze-20260917-83536c37-r1"
CORRECTED_TAG_OBJECT = "773b12c32ae312bfae9b782d772364f6bbbc008e"
SOURCE_BUNDLE = ROOT / "artifacts/performance_freeze" / FREEZE_ID
OUTPUT_ROOT = ROOT / "artifacts/performance_freeze_corrections"
ID_RE = re.compile(r"qp0-tag-correction-[0-9]{8}t[0-9]{12}z-[0-9a-f]{7}-[0-9a-f]{6}")


class CorrectionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def git_ref(name: str, *, peel: bool = False) -> str:
    expression = f"{name}^{{}}" if peel else name
    return subprocess.check_output(
        ["git", "rev-parse", expression], cwd=ROOT, text=True
    ).strip()


def tag_text(name: str) -> str:
    return subprocess.check_output(
        ["git", "cat-file", "-p", name], cwd=ROOT, text=True
    )


def verify_authority() -> None:
    manifest = SOURCE_BUNDLE / "manifest.json"
    actual = sha256_file(manifest)
    ledger = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (SOURCE_BUNDLE / "checksums.sha256").read_text().splitlines()
    }
    inventory = load_json(SOURCE_BUNDLE / "artifact_inventory.json")
    inventory_by_path = {item["path"]: item["sha256"] for item in inventory["files"]}
    if actual != MANIFEST_SHA or ledger.get("manifest.json") != actual or inventory_by_path.get("manifest.json") != actual:
        raise CorrectionError("finalized manifest identity differs")
    if git_ref(ORIGINAL_TAG) != ORIGINAL_TAG_OBJECT or git_ref(ORIGINAL_TAG, peel=True) != SOURCE:
        raise CorrectionError("original tag identity changed")
    if git_ref(CORRECTED_TAG) != CORRECTED_TAG_OBJECT or git_ref(CORRECTED_TAG, peel=True) != SOURCE:
        raise CorrectionError("corrected tag identity differs")
    annotation = tag_text(CORRECTED_TAG)
    for required in (MANIFEST_SHA, FREEZE_ROOT, ORIGINAL_TAG, ORIGINAL_TAG_OBJECT):
        if required not in annotation:
            raise CorrectionError("corrected tag annotation is incomplete")
    source = validate_local_artifact(SOURCE_BUNDLE, environ={})
    if source.root_sha256 != FREEZE_ROOT:
        raise CorrectionError("original freeze bundle root changed")


def payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )


def build(bundle_id: str, execution_head: str) -> dict[str, Any]:
    if not ID_RE.fullmatch(bundle_id):
        raise CorrectionError("invalid correction bundle ID")
    if git_ref("HEAD") != execution_head:
        raise CorrectionError("execution HEAD mismatch")
    verify_authority()
    final = OUTPUT_ROOT / bundle_id
    if final.exists():
        raise CorrectionError("correction bundle ID already exists")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = OUTPUT_ROOT / f".{bundle_id}.{secrets.token_hex(8)}.staging"
    stage.mkdir(mode=0o700)
    source_files = {
        "freeze-tag-correction.json": ROOT / "docs/evidence/qp0/freeze-tag-correction.json",
        "qp0-tag-correction-closure.md": ROOT / "docs/phase_reports/qp0-tag-correction-closure.md",
        "performance-data-frozen-marker.json": ROOT / "PERFORMANCE_DATA_FROZEN",
    }
    for destination, source in source_files.items():
        write_new(stage / destination, source.read_bytes())
    contract = load_json(ROOT / "configs/quality/quality_contract.yaml")["quality_contract"]
    handoff = {
        "schema_version": "kvbench-qp0-corrected-quality-handoff-1.0.0",
        "status": "QP1_PREPARATION_READY",
        "active_freeze_tag": CORRECTED_TAG,
        "freeze_bundle_root_sha256": FREEZE_ROOT,
        "quality_contract_status": contract["status"],
        "quality_contract_approved": contract["approval"]["approved"],
        "quality_execution": contract["quality_execution"],
        "quality_evaluation_executed": False,
        "pending_qp1_items": contract["pending_qp1_items"],
    }
    write_new(stage / "quality-contract-handoff.json", json_bytes(handoff))
    reference = {
        "schema_version": "kvbench-qp0-original-freeze-reference-1.0.0",
        "freeze_id": FREEZE_ID,
        "root_sha256": FREEZE_ROOT,
        "uri": f"r2://kvbench-artifacts/kvbench/sha256/{FREEZE_ROOT}/",
        "manifest_sha256": MANIFEST_SHA,
        "publication_receipt": "docs/evidence/qp0/r2-publication.json",
        "verification": "existing_receipt_accepted_no_repeat_retrieval",
    }
    write_new(stage / "original-freeze-reference.json", json_bytes(reference))
    manifest = {
        "schema_version": "kvbench-qp0-tag-correction-bundle-1.0.0",
        "run_id": bundle_id,
        "status": "PASS",
        "execution_git_sha": execution_head,
        "scope": "metadata_only_tag_annotation_correction",
        "performance_source_commit": SOURCE,
        "freeze_id": FREEZE_ID,
        "freeze_bundle_root_sha256": FREEZE_ROOT,
        "freeze_manifest_sha256": MANIFEST_SHA,
        "original_tag": ORIGINAL_TAG,
        "original_tag_object_id": ORIGINAL_TAG_OBJECT,
        "corrected_tag": CORRECTED_TAG,
        "corrected_tag_object_id": CORRECTED_TAG_OBJECT,
        "quality_contract_approved": False,
        "quality_execution": "LOCKED",
        "quality_evaluation_executed": False,
        "complete_written_last": True,
    }
    write_new(stage / "manifest.json", json_bytes(manifest))
    excluded = {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    items = [
        {
            "path": path.relative_to(stage).as_posix(),
            "role": "qp0_tag_correction",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in payload_paths(stage, excluded)
    ]
    write_new(stage / "artifact_inventory.json", json_bytes({
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": bundle_id,
        "files": items,
        "excluded_control_files": [
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ],
    }))
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
        for path in payload_paths(stage, {"checksums.sha256", "COMPLETE"})
    ).encode()
    write_new(stage / "checksums.sha256", ledger)
    write_new(stage / "COMPLETE", json_bytes({
        "schema_version": "kvbench-completion-1.0.0",
        "run_id": bundle_id,
        "status": "PASS",
        "manifest_sha256": sha256_file(stage / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(stage / "artifact_inventory.json"),
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": sha256_file(stage / "checksums.sha256"),
        "written_last": True,
    }))
    os.rename(stage, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    return {"path": str(final), **validate(final)}


def validate(root: Path) -> dict[str, Any]:
    verify_authority()
    local = validate_local_artifact(root, environ={})
    manifest = load_json(root / "manifest.json")
    correction = load_json(root / "freeze-tag-correction.json")
    marker = load_json(root / "performance-data-frozen-marker.json")
    handoff = load_json(root / "quality-contract-handoff.json")
    if manifest.get("corrected_tag_object_id") != CORRECTED_TAG_OBJECT:
        raise CorrectionError("bundle corrected tag identity differs")
    if correction.get("active_freeze_binding", {}).get("tag_object_id") != CORRECTED_TAG_OBJECT:
        raise CorrectionError("correction receipt tag identity differs")
    if marker.get("manifest_sha256") != MANIFEST_SHA or not marker.get("performance_freeze_complete"):
        raise CorrectionError("freeze marker identity differs")
    if marker.get("quality_contract_approved") is not False or marker.get("quality_execution") != "LOCKED":
        raise CorrectionError("freeze marker incorrectly authorizes quality")
    if handoff.get("quality_contract_approved") is not False or handoff.get("quality_execution") != "LOCKED":
        raise CorrectionError("quality handoff state differs")
    return {
        "status": "PASS",
        "bundle_id": manifest["run_id"],
        "root_sha256": local.root_sha256,
        "object_count": len(local.files),
        "corrected_tag_object_id": CORRECTED_TAG_OBJECT,
        "quality_execution": "LOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--bundle-id", required=True)
    build_parser.add_argument("--execution-head", required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = (
        build(args.bundle_id, args.execution_head)
        if args.command == "build"
        else validate(args.artifact)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
