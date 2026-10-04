"""Narrow successor geometry authority for Phase 16G."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


PHASE16G_CONFIGURATIONS = (
    "bf16",
    "tq_4bit_nc",
    "tq_k3v4_nc",
    "tq_3bit_nc",
    "k4v4",
    "k2v4",
    "k2v2",
    "kvq4",
    "kvq3",
    "kvq2",
)
PHASE16G_NEW_BATCH_SIZES = (2, 16)
PHASE16G_ADMITTED_BATCH_SIZES = (1, 2, 4, 8, 16)
PHASE16G_CONTEXT_LENGTH = 4096
PHASE16G_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
PHASE16G_INDEX_SCHEMA = "kvbench-phase16g-geometry-admission-index-1.0.0"
PHASE16G_PREFIX_SCHEMA = "kvbench-phase16g-prefix-state-3.0.0"
PHASE16G_EXECUTION_GIT_SHA = "6c829edad5cd0401051a20ee42fc61eeb612a9ba"
PHASE16G_DECISION_PATH = "docs/decisions/0039-full-scan-batch-geometry-admission.md"
PHASE16G_DECISION_SHA256 = (
    "f059336e0911439b5446385098991206f1cef7a7e30d29283626206604300c3c"
)
PHASE16G_GEOMETRY_REPORT_PATH = (
    "docs/evidence/phase16g/batch-geometry-admission.json"
)
PHASE16G_GEOMETRY_REPORT_SHA256 = (
    "d245796fbe7c9e13a5e9105872a1dffce0335c1f830bdfee8ef71676917334ca"
)
PHASE16G_SOURCE_AUTHORITY_COMMIT = (
    "b862af64346a0dba2650b2c213ebd1d3b5b99ef2"
)
PHASE16G_Q4_SOURCE_AUTHORITY_COMMIT = (
    "d7ef70b4fc4e9f0392b7e32dc749d0e57e9d7c84"
)
PHASE16G_SOURCE_TRANSITIONS = {
    "src/kvbench/runtime/turboquant_cache.py": {
        "predecessors": {
            PHASE16G_SOURCE_AUTHORITY_COMMIT: (
                "92551d9daf9c0af2b830655c414c137a52ef5e6dcaf26506efef31200e037b15"
            ),
        },
        "execution_blob": "aa03e0b412eab52112d2e57fefbf724755c3154d",
        "execution_sha256": (
            "8879f863c8db252c141bf9bbd68fba59f1ecb8178ece2712327b5684e7a3a76c"
        ),
        "transition_commits": (
            "fed1d184c2a26d0f818f3395d0ec7d4f6fe7159f",
        ),
    },
    "src/kvbench/runtime/kivi_cache.py": {
        "predecessors": {
            PHASE16G_SOURCE_AUTHORITY_COMMIT: (
                "5a466e0b80c50e891a18b40b058cbf46eeb8221508ef7be0ab47f164f9c08400"
            ),
        },
        "execution_blob": "6e713b659a26b2c82071836f0fdda5d6c412d626",
        "execution_sha256": (
            "27e9f32a27e86c3276d72184399859ec57048a80513a66a67102f04cc43d0cac"
        ),
        "transition_commits": (
            "a2e8fc012f7801a7d007a8be93ce5efcad03ccd5",
            "fed1d184c2a26d0f818f3395d0ec7d4f6fe7159f",
        ),
    },
    "src/kvbench/runtime/kvquant_cache.py": {
        "predecessors": {
            PHASE16G_SOURCE_AUTHORITY_COMMIT: (
                "24e0152347edbfca46ae8d911b424b2d0a1da9237d886c8075b0c071e605ee96"
            ),
            PHASE16G_Q4_SOURCE_AUTHORITY_COMMIT: (
                "b9729bc80a187f48edcc97137e7bb8fec8000185895bab4beb5f70d0315cbd6d"
            ),
        },
        "execution_blob": "450c75f02cc705d49a3929f3febe2810c1cab908",
        "execution_sha256": (
            "7d3e20f002ee172e5fccbed70a56074eb267a4d34ee676cf5ee8696f6664ef7c"
        ),
        "transition_commits": (
            "fed1d184c2a26d0f818f3395d0ec7d4f6fe7159f",
            PHASE16G_Q4_SOURCE_AUTHORITY_COMMIT,
        ),
    },
    "src/kvbench/runtime/kvquant_session.py": {
        "predecessors": {
            PHASE16G_SOURCE_AUTHORITY_COMMIT: (
                "5a92fdb3d71ef3241774f6e66883d5aa994a2f341b64e272e37368e67895d780"
            ),
            PHASE16G_Q4_SOURCE_AUTHORITY_COMMIT: (
                "e785066338e8249512808ea3a0a0ee3cca888f9edffae380679d7c798858ebba"
            ),
        },
        "execution_blob": "25cba756e832b4912b9a4ad6af138fb849644158",
        "execution_sha256": (
            "73f4d3d36fa038c156830d85c42ee54fe9ebcb776422a474b279d9e8b0bb2437"
        ),
        "transition_commits": (
            "a2e8fc012f7801a7d007a8be93ce5efcad03ccd5",
            PHASE16G_Q4_SOURCE_AUTHORITY_COMMIT,
        ),
    },
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Phase16GGeometryError(RuntimeError):
    """A geometry key, record, or successor authority is invalid."""


def geometry_key(configuration: str, batch_size: int) -> str:
    if configuration not in PHASE16G_CONFIGURATIONS:
        raise Phase16GGeometryError("geometry configuration is unknown")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise Phase16GGeometryError("geometry batch size is invalid")
    return f"{configuration}/B{batch_size}"


def validate_geometry_index(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact B=2/B=16 evidence without rewriting B=1/4/8 records."""

    records = payload.get("new_geometry_records")
    expected_keys = {
        geometry_key(configuration, batch)
        for configuration in PHASE16G_CONFIGURATIONS
        for batch in PHASE16G_NEW_BATCH_SIZES
    }
    if (
        payload.get("schema_version") != PHASE16G_INDEX_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("decision_id") != "0039"
        or payload.get("authorized_container_digest")
        != PHASE16G_CONTAINER_DIGEST
        or payload.get("prefix_format_version") != PHASE16G_PREFIX_SCHEMA
        or payload.get("admitted_full_scan_batches")
        != list(PHASE16G_ADMITTED_BATCH_SIZES)
        or payload.get("existing_geometry_batches_unchanged") != [1, 4, 8]
        or not isinstance(records, Mapping)
        or set(records) != expected_keys
    ):
        raise Phase16GGeometryError("geometry admission index contract differs")
    for key, record in records.items():
        if not isinstance(record, Mapping):
            raise Phase16GGeometryError("geometry admission record differs")
        expected = geometry_key(
            str(record.get("configuration")),
            int(record.get("batch_size", 0)),
        )
        if (
            key != expected
            or record.get("batch_size") not in PHASE16G_NEW_BATCH_SIZES
            or record.get("status") != "PASS"
            or record.get("short_eager") != "PASS"
            or record.get("short_cuda_graph") != "PASS"
            or record.get("prefix_restore") != "PASS"
            or record.get("allocation") != "PASS"
            or record.get("execution_path") != "PASS"
        ):
            raise Phase16GGeometryError("geometry admission result differs")
        for field in (
            "method_config_fingerprint",
            "adapter_config_fingerprint",
            "cache_layout_fingerprint",
        ):
            value = record.get(field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise Phase16GGeometryError(f"geometry {field} differs")
    return dict(payload)


def require_admitted_geometry(
    payload: Mapping[str, Any],
    *,
    configuration: str,
    batch_size: int,
) -> str:
    """Keep admission policy outside the shape-faithful prefix loader."""

    validated = validate_geometry_index(payload)
    key = geometry_key(configuration, batch_size)
    if batch_size in (1, 4, 8):
        authorities = validated.get("existing_geometry_authorities")
        if not isinstance(authorities, Mapping) or configuration not in authorities:
            raise Phase16GGeometryError("existing geometry authority is absent")
        return key
    record = validated["new_geometry_records"].get(key)
    if not isinstance(record, Mapping) or record.get("status") != "PASS":
        raise Phase16GGeometryError("requested geometry is not admitted")
    return key


def _authority_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise Phase16GGeometryError("Phase 16G authority path is unsafe")
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise Phase16GGeometryError("Phase 16G authority path is absent") from error
    if not path.is_file():
        raise Phase16GGeometryError("Phase 16G authority path is unsafe")
    return path


def _authority_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("/usr/bin/git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or result.stderr:
        raise Phase16GGeometryError("Phase 16G Git authority is unavailable")
    return result.stdout


def validate_source_transition(repository_root: Path) -> dict[str, Any]:
    """Validate only the checksum-bound Decision 0039 source transition."""

    root = repository_root.resolve(strict=True)
    decision = _authority_file(root, PHASE16G_DECISION_PATH)
    report_path = _authority_file(root, PHASE16G_GEOMETRY_REPORT_PATH)
    if (
        _authority_sha256(decision) != PHASE16G_DECISION_SHA256
        or _authority_sha256(report_path) != PHASE16G_GEOMETRY_REPORT_SHA256
    ):
        raise Phase16GGeometryError("Phase 16G decision or report checksum differs")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Phase16GGeometryError("Phase 16G report JSON differs") from error
    validate_geometry_index(report)
    if report.get("execution_git_sha") != PHASE16G_EXECUTION_GIT_SHA:
        raise Phase16GGeometryError("Phase 16G execution Git SHA differs")
    head = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    ancestry = subprocess.run(
        (
            "/usr/bin/git",
            "merge-base",
            "--is-ancestor",
            PHASE16G_EXECUTION_GIT_SHA,
            head,
        ),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0 or ancestry.stdout or ancestry.stderr:
        raise Phase16GGeometryError("Phase 16G execution SHA is not an ancestor")
    sources: dict[str, dict[str, Any]] = {}
    for relative, expected in PHASE16G_SOURCE_TRANSITIONS.items():
        execution_bytes = _git_output(
            root, "show", f"{PHASE16G_EXECUTION_GIT_SHA}:{relative}"
        )
        execution_blob = (
            _git_output(
                root, "rev-parse", f"{PHASE16G_EXECUTION_GIT_SHA}:{relative}"
            )
            .decode("ascii")
            .strip()
        )
        execution_sha256 = hashlib.sha256(execution_bytes).hexdigest()
        current_bytes = _git_output(root, "show", f"{head}:{relative}")
        current_file_sha256 = _authority_sha256(_authority_file(root, relative))
        history = tuple(
            line
            for line in _git_output(
                root,
                "log",
                "--format=%H",
                f"{PHASE16G_SOURCE_AUTHORITY_COMMIT}..{PHASE16G_EXECUTION_GIT_SHA}",
                "--",
                relative,
            )
            .decode("ascii")
            .splitlines()
            if line
        )
        post_history = _git_output(
            root,
            "log",
            "--format=%H",
            f"{PHASE16G_EXECUTION_GIT_SHA}..{head}",
            "--",
            relative,
        ).strip()
        predecessors = expected["predecessors"]
        if not isinstance(predecessors, Mapping):
            raise Phase16GGeometryError("Phase 16G predecessor map differs")
        predecessor_records: dict[str, str] = {}
        for commit, expected_sha256 in predecessors.items():
            predecessor_sha256 = hashlib.sha256(
                _git_output(root, "show", f"{commit}:{relative}")
            ).hexdigest()
            if predecessor_sha256 != expected_sha256:
                raise Phase16GGeometryError(
                    f"Phase 16G predecessor source differs: {relative}"
                )
            predecessor_records[str(commit)] = predecessor_sha256
        if (
            execution_blob != expected["execution_blob"]
            or execution_sha256 != expected["execution_sha256"]
            or hashlib.sha256(current_bytes).hexdigest() != execution_sha256
            or current_file_sha256 != execution_sha256
            or history != expected["transition_commits"]
            or post_history
        ):
            raise Phase16GGeometryError(
                f"Phase 16G source transition is unrecognized: {relative}"
            )
        sources[relative] = {
            "predecessor_sha256s": predecessor_records,
            "execution_blob": execution_blob,
            "execution_sha256": execution_sha256,
            "transition_commits": list(history),
            "post_execution_transition_count": 0,
        }
    return {
        "decision": "0039",
        "decision_sha256": PHASE16G_DECISION_SHA256,
        "geometry_report_sha256": PHASE16G_GEOMETRY_REPORT_SHA256,
        "execution_git_sha": PHASE16G_EXECUTION_GIT_SHA,
        "current_git_sha": head,
        "sources": sources,
    }


def source_transition_recognized(
    authority: Mapping[str, Any],
    *,
    relative_path: str,
    predecessor_sha256: str,
    current_sha256: str,
) -> bool:
    """Match one old report source to its exact Phase 16G successor blob."""

    sources = authority.get("sources")
    if not isinstance(sources, Mapping):
        return False
    record = sources.get(relative_path)
    if not isinstance(record, Mapping):
        return False
    predecessors = record.get("predecessor_sha256s")
    return (
        isinstance(predecessors, Mapping)
        and predecessor_sha256 in predecessors.values()
        and current_sha256 == record.get("execution_sha256")
    )
