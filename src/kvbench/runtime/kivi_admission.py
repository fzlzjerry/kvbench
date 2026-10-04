"""Fail-closed, evidence-only admission helpers for Phase 8 KIVI.

This module does not execute CUDA, collect profiles, publish artifacts, or
introduce another runner.  It interprets already-produced common-runner
evidence and joins it into the strict Phase 8 report schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from kvbench.adapters.kivi import (
    KIVI_BGEMV2_HOST_STUB_OFFSET,
    KIVI_BGEMV4_HOST_STUB_OFFSET,
    KIVI_DECISION_0018_PATCH_SHA256,
    KIVI_EXTENSION_SHA256,
    KIVI_FIXTURE_ROOT_SHA256,
    KIVI_NEW_PACK_SHA256,
    KIVI_OFFICIAL_BASE_TREE,
    KIVI_OFFICIAL_COMMIT,
    KIVI_PATCHED_TREE,
    KIVIMethodAdapter,
)
from kvbench.adapters import kivi as kivi_adapter_module
from kvbench.runtime.turboquant_admission import (
    TurboQuantAdmissionError,
    require_authorized_cuda_environment,
)
from kvbench.runtime.artifacts import sha256_file, validate_run_directory
from kvbench.runtime.backend import BACKEND_IDENTITY
from kvbench.runtime.kivi_allocation import (
    KIVIAllocationBinding,
    KIVIAllocationError,
    replay_preserved_kivi_allocation_attribution,
)
from kvbench.runtime.kivi_session import (
    build_kivi_operation_keys,
)
from kvbench.runtime.kivi_cache import KIVIStaticCache
from kvbench.schema import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
    canonical_json_bytes,
    sha256_hex,
)
from kvbench.schema.base import (
    require_git_sha,
    require_relative_path,
    require_sha256,
)
from kvbench.schema.method_admission import MethodAdmissionEvidenceReference
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase8 import (
    PHASE8_ADMISSION_CHECK_IDS,
    PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    PHASE8_BASE_TREE,
    PHASE8_DECISION_0018_PATCH_SHA256,
    PHASE8_EXTENSION_SHA256,
    PHASE8_FIXTURE_ROOT_DIGEST,
    PHASE8_HELD_OUT_CONFIG,
    PHASE8_MANDATORY_CONFIGS,
    PHASE8_OFFICIAL_COMMIT,
    PHASE8_PATCHED_TREE,
    RECIPROCAL_ABS_TOLERANCE,
    Phase8AdmissionCheck,
    Phase8AdmissionGates,
    Phase8MethodAdmissionReport,
    Phase8RunManifest,
)
from kvbench.schema.phase16g import (
    Phase16GGeometryError,
    source_transition_recognized,
    validate_source_transition,
)
from kvbench.schema.phase13b import Phase13BMethodAdmissionReport


PHASE8_IMAGE_ENVIRONMENT_VARIABLE = "KVBENCH_AUTHORIZED_IMAGE_DIGEST"
PHASE8_CONTAINER_ENVIRONMENT_VARIABLE = "KVBENCH_EXECUTION_ENVIRONMENT"
PHASE8_CONTAINER_ENVIRONMENT_VALUE = "measurement_container"
OFFICIAL_KIVI_KERNEL_FAMILIES = (
    "bgemv2_kernel_outer_dim",
    "bgemv4_kernel_outer_dim",
)
OFFICIAL_KIVI_HOST_STUB_OFFSETS = {
    2: KIVI_BGEMV2_HOST_STUB_OFFSET,
    4: KIVI_BGEMV4_HOST_STUB_OFFSET,
}
PHASE8_FIXTURE_TEST_PATH = "tests/cuda/test_phase8_kivi_cuda.py"
PHASE8_GRAPH_TEST_PATH = "tests/graph/test_phase8_kivi_graph.py"
PHASE8_SANITIZER_PROBE_PATH = (
    "tests/cuda/phase8_kivi_sanitizer_probe.py"
)
PHASE8_ADAPTER_PATH = "src/kvbench/adapters/kivi.py"
PHASE8_CACHE_PATH = "src/kvbench/runtime/kivi_cache.py"
PHASE8_ENDPOINT_PATH = "src/kvbench/runtime/bf16_endpoint.py"
PHASE8_EXECUTION_GIT_SHA = (
    "462325e9df809d3bcf24a06361bf004bc7383d73"
)
PHASE8_HISTORICAL_ADAPTER_SHA256 = (
    "d47efdb9a9b6e34aaf3f8465a33b6f2bc550680ad369cfb1a3e4d6f0222bccc8"
)
PHASE8_HISTORICAL_ADAPTER_VERSION = (
    "kvbench-kivi-method-adapter-1.0.0"
)
PHASE8_HISTORICAL_CACHE_SHA256 = (
    "0c99bb6b6bf9e84074f5e087d545988912285d4c5621c10ee8e7920cac0844a5"
)
PHASE8_HISTORICAL_ENDPOINT_SHA256 = (
    "8aa48ec285fb9c7853bc19ae10bd8afc07a04d1d6f522f53e67e705a424a27b9"
)
PHASE8_DECISION_0026_ENDPOINT_COMMIT = (
    "781b416748e2bddca8ea5c23cd0f51a63a066276"
)
PHASE8_DECISION_0026_ENDPOINT_SHA256 = (
    "9095e9a2a9c01e1ea6afb2f1cefcee46a964a82caae7b819a125757b59244a9b"
)
PHASE13B_DECISION_0030_PATH = (
    "docs/decisions/0030-compressed-static-cache-batch-geometry.md"
)
PHASE13B_DECISION_0030_SHA256 = (
    "84c2eb943b35afba312eaf599f8ec8f1d4a82169daa2d5c5fc5d127f0a965e62"
)
PHASE13B_DECISION_0030_COMMIT = (
    "2af47459e109207ac21167ddd88e8ca79d815490"
)
PHASE13B_SOURCE_AUTHORITY_COMMIT = (
    "b862af64346a0dba2650b2c213ebd1d3b5b99ef2"
)
PHASE13B_KIVI_REPORT_PATH = (
    "docs/evidence/phase13b/kivi-method-admission.json"
)
PHASE13B_KIVI_REPORT_SHA256 = (
    "1e91730ac56af37e03d80edce7979a509d52049428faad89f61e61dc6bd48c51"
)
PHASE13B_KIVI_HISTORICAL_REPORT_SHA256 = (
    "3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a"
)
PHASE13B_KIVI_SOURCE_AUTHORITY = {
    PHASE8_ADAPTER_PATH: {
        "historical_sha256": PHASE8_HISTORICAL_ADAPTER_SHA256,
        "authority_sha256": (
            "d4fa3cf8e576bc6a080b2132e1976f195c9756fd82f8b9ffcc0d4251b08c16f0"
        ),
    },
    PHASE8_CACHE_PATH: {
        "historical_sha256": PHASE8_HISTORICAL_CACHE_SHA256,
        "authority_sha256": (
            "5a466e0b80c50e891a18b40b058cbf46eeb8221508ef7be0ab47f164f9c08400"
        ),
    },
}
PHASE8_METHOD_CONFIG_PATH = "configs/methods/kivi.yaml"
PHASE8_BOUNDED_GRID_SCHEMA = (
    "kvbench-phase8-kivi-bounded-grid-1.0.0"
)
PHASE8_CANDIDATE_SCHEMA = (
    "kvbench-phase8-kivi-admission-candidate-1.0.0"
)
PHASE8_INNER_RECEIPT_SCHEMA = (
    "kvbench-phase8-kivi-admission-r2-publication-1.0.0"
)
_PROHIBITED_RECEIPT_KEYS = frozenset(
    {
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "cloudflare_api_token",
        "hf_token",
        "hugging_face_hub_token",
        "r2_account_id",
        "authorization",
    }
)
_FORBIDDEN_INNER_REPORT_VALIDATION_KEYS = frozenset(
    {
        "retrieved_report_valid",
        "method_admission_report_valid",
    }
)


class KIVIAdmissionError(RuntimeError):
    """Phase 8 evidence is absent, inconsistent, or outside authority."""


@dataclass(frozen=True, slots=True)
class Phase8HistoricalSourceAuthority:
    """Execution-time source blobs plus the accepted endpoint transition."""

    execution_git_sha: str
    current_git_sha: str
    adapter_source_sha256: str
    cache_source_sha256: str
    endpoint_source_sha256: str
    endpoint_transition_commit: str


def _phase8_git(
    repository_root: Path,
    *arguments: str,
    binary: bool = False,
) -> bytes | str:
    """Run one non-interactive Git object query in a scrubbed environment."""

    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
        "LANG": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        result = subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise KIVIAdmissionError(
            "historical KIVI Git object query failed"
        ) from error
    if result.returncode != 0:
        raise KIVIAdmissionError(
            "historical KIVI Git object query failed"
        )
    if binary:
        return result.stdout
    try:
        return result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise KIVIAdmissionError(
            "historical KIVI Git identity is not ASCII"
        ) from error


def _phase8_git_blob_sha256(
    repository_root: Path,
    *,
    revision: str,
    relative_path: str,
) -> str:
    payload = _phase8_git(
        repository_root,
        "cat-file",
        "blob",
        f"{revision}:{relative_path}",
        binary=True,
    )
    if not isinstance(payload, bytes):
        raise KIVIAdmissionError("historical KIVI Git blob is invalid")
    return hashlib.sha256(payload).hexdigest()


def _phase8_git_path_history(
    repository_root: Path,
    *,
    start_commit: str,
    end_commit: str,
    relative_paths: tuple[str, ...],
) -> str:
    """Return full path history, including changes discarded by merges."""

    history = _phase8_git(
        repository_root,
        "rev-list",
        "--full-history",
        "--reverse",
        f"{start_commit}..{end_commit}",
        "--",
        *relative_paths,
    )
    if not isinstance(history, str):
        raise KIVIAdmissionError(
            "historical KIVI Git path history is invalid"
        )
    return history


def _phase8_regular_source_sha256(
    repository_root: Path,
    relative_path: str,
) -> str:
    path = repository_root / relative_path
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise KIVIAdmissionError(
            f"current KIVI source is absent: {relative_path}"
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or resolved != path
    ):
        raise KIVIAdmissionError(
            f"current KIVI source is unsafe: {relative_path}"
        )
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise KIVIAdmissionError(
            f"current KIVI source is unreadable: {relative_path}"
        ) from error


def _validate_phase13b_kivi_successor_transition(
    repository_root: Path,
    *,
    execution_git_sha: str,
    current_git_sha: str,
) -> None:
    """Recognize only Decision 0030's checksum-bound KIVI successor."""

    decision = _resolve_within(
        repository_root.joinpath(*PHASE13B_DECISION_0030_PATH.split("/")),
        root=repository_root,
        label="Decision 0030",
        require_file=True,
    )
    report_path = _resolve_within(
        repository_root.joinpath(*PHASE13B_KIVI_REPORT_PATH.split("/")),
        root=repository_root,
        label="Phase 13B KIVI successor report",
        require_file=True,
    )
    if sha256_file(decision) != PHASE13B_DECISION_0030_SHA256:
        raise KIVIAdmissionError("Decision 0030 checksum differs")
    if sha256_file(report_path) != PHASE13B_KIVI_REPORT_SHA256:
        raise KIVIAdmissionError(
            "Phase 13B KIVI successor report checksum differs"
        )
    try:
        report = Phase13BMethodAdmissionReport.from_dict(
            _strict_json(
                report_path,
                label="Phase 13B KIVI successor report",
            )
        )
    except (TypeError, ValueError) as error:
        raise KIVIAdmissionError(
            "Phase 13B KIVI successor report schema differs"
        ) from error
    expected_sources = {
        path: authority["authority_sha256"]
        for path, authority in PHASE13B_KIVI_SOURCE_AUTHORITY.items()
    }
    if (
        report.method_family != "kivi"
        or report.creation_git_sha != PHASE13B_SOURCE_AUTHORITY_COMMIT
        or report.historical_report_path
        != "docs/evidence/phase8/kivi-method-admission.json"
        or report.historical_report_sha256
        != PHASE13B_KIVI_HISTORICAL_REPORT_SHA256
        or report.source_hashes != expected_sources
        or not report.b1_numerical_preserved
        or report.cuda_source_changed
    ):
        raise KIVIAdmissionError(
            "Phase 13B KIVI successor authority differs"
        )

    for ancestor, descendant in (
        (execution_git_sha, PHASE13B_DECISION_0030_COMMIT),
        (PHASE13B_DECISION_0030_COMMIT, PHASE13B_SOURCE_AUTHORITY_COMMIT),
        (PHASE13B_SOURCE_AUTHORITY_COMMIT, current_git_sha),
    ):
        ancestry = _phase8_git(
            repository_root,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        )
        if ancestry != "":
            raise KIVIAdmissionError(
                "Decision 0030 KIVI successor ancestry differs"
            )

    phase16g_authority: dict[str, Any] | None = None
    for relative_path, expected in PHASE13B_KIVI_SOURCE_AUTHORITY.items():
        historical = _phase8_git_blob_sha256(
            repository_root,
            revision=execution_git_sha,
            relative_path=relative_path,
        )
        authority = _phase8_git_blob_sha256(
            repository_root,
            revision=PHASE13B_SOURCE_AUTHORITY_COMMIT,
            relative_path=relative_path,
        )
        current_head = _phase8_git_blob_sha256(
            repository_root,
            revision=current_git_sha,
            relative_path=relative_path,
        )
        current_file = _phase8_regular_source_sha256(
            repository_root,
            relative_path,
        )
        transition = _phase8_git_path_history(
            repository_root,
            start_commit=execution_git_sha,
            end_commit=PHASE13B_SOURCE_AUTHORITY_COMMIT,
            relative_paths=(relative_path,),
        )
        post_authority = _phase8_git_path_history(
            repository_root,
            start_commit=PHASE13B_SOURCE_AUTHORITY_COMMIT,
            end_commit=current_git_sha,
            relative_paths=(relative_path,),
        )
        if (
            historical != expected["historical_sha256"]
            or authority != expected["authority_sha256"]
        ):
            source_label = (
                "adapter"
                if relative_path == PHASE8_ADAPTER_PATH
                else "cache"
            )
            raise KIVIAdmissionError(
                f"KIVI {source_label} authority changed outside the exact "
                "Decision 0030 successor"
            )
        if transition != PHASE13B_DECISION_0030_COMMIT:
            raise KIVIAdmissionError(
                "KIVI adapter or cache changed outside the exact "
                "Decision 0030 successor"
            )
        if (
            current_head != expected["authority_sha256"]
            or current_file != expected["authority_sha256"]
            or post_authority != ""
        ):
            if phase16g_authority is None:
                try:
                    phase16g_authority = validate_source_transition(
                        repository_root
                    )
                except Phase16GGeometryError as error:
                    raise KIVIAdmissionError(
                        "Decision 0039 KIVI source authority differs"
                    ) from error
            if not source_transition_recognized(
                phase16g_authority,
                relative_path=relative_path,
                predecessor_sha256=authority,
                current_sha256=current_head,
            ) or current_file != current_head:
                raise KIVIAdmissionError(
                    "KIVI adapter or cache changed outside the exact "
                    "Decision 0039 successor"
                )


def resolve_phase8_historical_source_authority(
    *,
    repository_root: Path,
    execution_git_sha: str,
    manifest_adapter_sha256: str,
) -> Phase8HistoricalSourceAuthority:
    """Bind Phase 8 replay to its commit and Decision 0026's sole transition."""

    try:
        require_git_sha(execution_git_sha)
        require_sha256(manifest_adapter_sha256)
    except ValueError as error:
        raise KIVIAdmissionError(
            "historical KIVI source identity is invalid"
        ) from error
    if execution_git_sha != PHASE8_EXECUTION_GIT_SHA:
        raise KIVIAdmissionError(
            "historical KIVI execution commit differs from Phase 8 authority"
        )
    resolved_execution = _phase8_git(
        repository_root,
        "rev-parse",
        "--verify",
        f"{execution_git_sha}^{{commit}}",
    )
    current_git_sha = _phase8_git(
        repository_root,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    )
    try:
        if not isinstance(current_git_sha, str):
            raise ValueError("current Git SHA is not text")
        require_git_sha(current_git_sha)
    except ValueError as error:
        raise KIVIAdmissionError(
            "current KIVI Git identity is invalid"
        ) from error
    if (
        resolved_execution != execution_git_sha
    ):
        raise KIVIAdmissionError(
            "historical KIVI execution commit did not resolve exactly"
        )
    ancestry = _phase8_git(
        repository_root,
        "merge-base",
        "--is-ancestor",
        execution_git_sha,
        current_git_sha,
    )
    if ancestry != "":
        raise KIVIAdmissionError(
            "historical KIVI execution commit is not a current ancestor"
        )
    endpoint_commits = _phase8_git_path_history(
        repository_root,
        start_commit=execution_git_sha,
        end_commit=current_git_sha,
        relative_paths=(PHASE8_ENDPOINT_PATH,),
    )
    if endpoint_commits != PHASE8_DECISION_0026_ENDPOINT_COMMIT:
        raise KIVIAdmissionError(
            "KIVI endpoint transition is not the exact Decision 0026 commit"
        )
    _validate_phase13b_kivi_successor_transition(
        repository_root,
        execution_git_sha=execution_git_sha,
        current_git_sha=current_git_sha,
    )

    historical_adapter = _phase8_git_blob_sha256(
        repository_root,
        revision=execution_git_sha,
        relative_path=PHASE8_ADAPTER_PATH,
    )
    historical_cache = _phase8_git_blob_sha256(
        repository_root,
        revision=execution_git_sha,
        relative_path=PHASE8_CACHE_PATH,
    )
    historical_endpoint = _phase8_git_blob_sha256(
        repository_root,
        revision=execution_git_sha,
        relative_path=PHASE8_ENDPOINT_PATH,
    )
    transition_parent_endpoint = _phase8_git_blob_sha256(
        repository_root,
        revision=f"{PHASE8_DECISION_0026_ENDPOINT_COMMIT}^",
        relative_path=PHASE8_ENDPOINT_PATH,
    )
    transition_endpoint = _phase8_git_blob_sha256(
        repository_root,
        revision=PHASE8_DECISION_0026_ENDPOINT_COMMIT,
        relative_path=PHASE8_ENDPOINT_PATH,
    )
    current_head_adapter = _phase8_git_blob_sha256(
        repository_root,
        revision=current_git_sha,
        relative_path=PHASE8_ADAPTER_PATH,
    )
    current_head_cache = _phase8_git_blob_sha256(
        repository_root,
        revision=current_git_sha,
        relative_path=PHASE8_CACHE_PATH,
    )
    current_head_endpoint = _phase8_git_blob_sha256(
        repository_root,
        revision=current_git_sha,
        relative_path=PHASE8_ENDPOINT_PATH,
    )
    current_adapter = _phase8_regular_source_sha256(
        repository_root,
        PHASE8_ADAPTER_PATH,
    )
    current_cache = _phase8_regular_source_sha256(
        repository_root,
        PHASE8_CACHE_PATH,
    )
    current_endpoint = _phase8_regular_source_sha256(
        repository_root,
        PHASE8_ENDPOINT_PATH,
    )
    if (
        historical_adapter != PHASE8_HISTORICAL_ADAPTER_SHA256
        or manifest_adapter_sha256 != PHASE8_HISTORICAL_ADAPTER_SHA256
        or current_head_adapter
        != PHASE13B_KIVI_SOURCE_AUTHORITY[PHASE8_ADAPTER_PATH][
            "authority_sha256"
        ]
        or current_adapter
        != PHASE13B_KIVI_SOURCE_AUTHORITY[PHASE8_ADAPTER_PATH][
            "authority_sha256"
        ]
    ):
        raise KIVIAdmissionError(
            "KIVI adapter authority changed after Phase 8"
        )
    if (
        historical_cache != PHASE8_HISTORICAL_CACHE_SHA256
        or current_head_cache
        != PHASE13B_KIVI_SOURCE_AUTHORITY[PHASE8_CACHE_PATH][
            "authority_sha256"
        ]
        or current_cache
        != PHASE13B_KIVI_SOURCE_AUTHORITY[PHASE8_CACHE_PATH][
            "authority_sha256"
        ]
    ):
        raise KIVIAdmissionError(
            "KIVI cache authority changed after Phase 8"
        )
    if (
        historical_endpoint != PHASE8_HISTORICAL_ENDPOINT_SHA256
        or transition_parent_endpoint
        != PHASE8_HISTORICAL_ENDPOINT_SHA256
        or transition_endpoint != PHASE8_DECISION_0026_ENDPOINT_SHA256
        or current_head_endpoint != PHASE8_DECISION_0026_ENDPOINT_SHA256
        or current_endpoint != PHASE8_DECISION_0026_ENDPOINT_SHA256
    ):
        raise KIVIAdmissionError(
            "KIVI endpoint blobs do not match the exact historical transition"
        )
    return Phase8HistoricalSourceAuthority(
        execution_git_sha=execution_git_sha,
        current_git_sha=current_git_sha,
        adapter_source_sha256=historical_adapter,
        cache_source_sha256=historical_cache,
        endpoint_source_sha256=historical_endpoint,
        endpoint_transition_commit=PHASE8_DECISION_0026_ENDPOINT_COMMIT,
    )


def _phase8_historical_backend_fingerprint(
    source_authority: Phase8HistoricalSourceAuthority,
) -> str:
    payload = {
        "schema_version": "kvbench-phase8-kivi-backend-fingerprint-1.0.0",
        "prefill_backend": BACKEND_IDENTITY,
        "decode_backend": {
            "implementation": (
                "patched_official_kivi_direct_compressed_decode"
            ),
            "official_commit": KIVI_OFFICIAL_COMMIT,
            "official_base_tree": KIVI_OFFICIAL_BASE_TREE,
            "patched_tree": KIVI_PATCHED_TREE,
            "decision_0018_patch_sha256": (
                KIVI_DECISION_0018_PATCH_SHA256
            ),
            "extension_sha256": KIVI_EXTENSION_SHA256,
            "new_pack_sha256": KIVI_NEW_PACK_SHA256,
            "fixture_root_sha256": KIVI_FIXTURE_ROOT_SHA256,
            "cuda_abi": "float16",
            "model_boundary": "bfloat16_to_float16_to_bfloat16",
            "kernel_families": [
                "bgemv2_kernel_outer_dim",
                "bgemv4_kernel_outer_dim",
            ],
        },
        "local_sources": {
            "adapters/kivi.py": source_authority.adapter_source_sha256,
            "runtime/bf16_endpoint.py": (
                source_authority.endpoint_source_sha256
            ),
            "runtime/kivi_cache.py": source_authority.cache_source_sha256,
        },
    }
    return sha256_hex(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class Phase8GridPoint:
    """One and only one preregistered Phase 8 bounded-admission point."""

    configuration: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    context_length: int
    output_steps: int


PHASE8_ADMISSION_GRID = (
    Phase8GridPoint("k4v4", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    Phase8GridPoint(
        "k4v4", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1
    ),
    Phase8GridPoint("k2v4", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    Phase8GridPoint(
        "k2v4", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1
    ),
    Phase8GridPoint("k2v2", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    Phase8GridPoint(
        "k2v2", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1
    ),
    Phase8GridPoint("k4v4", RunnerKind.FIXED_L, GraphMode.EAGER, 4096, 1),
    Phase8GridPoint(
        "k4v4", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 4096, 1
    ),
    Phase8GridPoint(
        "k4v4", RunnerKind.GROWING_CONTEXT, GraphMode.EAGER, 31, 4
    ),
    Phase8GridPoint("k4v2", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
)


def require_authorized_kivi_environment(
    declared_digest: str,
) -> dict[str, Any]:
    """Reject native CUDA and every image except Decision 0016 authority."""

    if declared_digest != PHASE8_AUTHORIZED_CONTAINER_DIGEST:
        raise KIVIAdmissionError(
            "Phase 8 CUDA requires the exact authorized Measurement Container"
        )
    try:
        identity = require_authorized_cuda_environment(declared_digest)
    except TurboQuantAdmissionError as error:
        raise KIVIAdmissionError(str(error)) from error
    if (
        identity.get("container_digest")
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or identity.get("execution_environment")
        != PHASE8_CONTAINER_ENVIRONMENT_VALUE
        or os.environ.get(PHASE8_IMAGE_ENVIRONMENT_VARIABLE)
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or os.environ.get(PHASE8_CONTAINER_ENVIRONMENT_VARIABLE)
        != PHASE8_CONTAINER_ENVIRONMENT_VALUE
    ):
        raise KIVIAdmissionError(
            "authorized Measurement Container identity did not round-trip"
        )
    return {
        **identity,
        "phase": "phase8_kivi_admission",
        "native_host_cuda_rejected": True,
    }


def _manifest_grid_point(manifest: Phase8RunManifest) -> Phase8GridPoint:
    return Phase8GridPoint(
        manifest.method_configuration,
        manifest.runner_kind,
        manifest.graph_mode,
        manifest.context_length,
        manifest.output_steps,
    )


def require_exact_phase8_grid(
    manifests: Sequence[Phase8RunManifest],
) -> tuple[Phase8RunManifest, ...]:
    """Require the ten completed points, once each, in preregistered order."""

    records = tuple(manifests)
    if (
        len(records) != len(PHASE8_ADMISSION_GRID)
        or any(type(record) is not Phase8RunManifest for record in records)
        or tuple(_manifest_grid_point(record) for record in records)
        != PHASE8_ADMISSION_GRID
    ):
        raise KIVIAdmissionError(
            "bounded admission manifests differ from the exact ten-point grid"
        )
    if len({record.run_id for record in records}) != len(records):
        raise KIVIAdmissionError("bounded admission run IDs are not unique")
    for record in records:
        expected_capacity = record.context_length + record.output_steps
        expected_active = (
            record.context_length
            if record.runner_kind is RunnerKind.FIXED_L
            else expected_capacity
        )
        if (
            record.status is not RunStatus.COMPLETED
            or record.capacity != expected_capacity
            or record.accounting.capacity != expected_capacity
            or record.accounting.active_context != expected_active
            or record.inventory_path is None
            or record.failure_reason is not None
        ):
            raise KIVIAdmissionError(
                f"bounded admission run is incomplete: {record.run_id}"
            )
    return records


@dataclass(frozen=True, slots=True)
class KIVIExecutionPathAudit:
    """Static interpretation of already-collected KIVI path evidence."""

    passed: bool
    kernel_families: tuple[str, ...]
    two_bit_kernel_verified: bool
    four_bit_kernel_verified: bool
    extension_identity_verified: bool
    source_identity_verified: bool
    patched_authority_verified: bool
    host_stub_offsets_verified: bool
    native_gqa_indexing_verified: bool
    full_prefix_dequantization_detected: bool
    full_prefix_temporary_detected: bool
    gqa_materialization_detected: bool
    query_head_sized_kv_temporary_detected: bool
    measured_torch_cat_detected: bool
    host_synchronization_detected: bool
    cache_growth_detected: bool
    backend_fallback_detected: bool
    stable_post_warmup_path: bool
    reasons: tuple[str, ...]
    kernel_sequence_sha256: str

    def __post_init__(self) -> None:
        require_sha256(
            self.kernel_sequence_sha256,
            field_name="kernel_sequence_sha256",
        )
        if (
            self.kernel_families != OFFICIAL_KIVI_KERNEL_FAMILIES
            or self.passed != (not self.reasons)
            or len(set(self.reasons)) != len(self.reasons)
        ):
            raise ValueError("KIVI execution-path audit is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "kernel_families": list(self.kernel_families),
            "two_bit_kernel_verified": self.two_bit_kernel_verified,
            "four_bit_kernel_verified": self.four_bit_kernel_verified,
            "extension_identity_verified": self.extension_identity_verified,
            "source_identity_verified": self.source_identity_verified,
            "patched_authority_verified": self.patched_authority_verified,
            "host_stub_offsets_verified": self.host_stub_offsets_verified,
            "native_gqa_indexing_verified": self.native_gqa_indexing_verified,
            "full_prefix_dequantization_detected": (
                self.full_prefix_dequantization_detected
            ),
            "full_prefix_temporary_detected": (
                self.full_prefix_temporary_detected
            ),
            "gqa_materialization_detected": (
                self.gqa_materialization_detected
            ),
            "query_head_sized_kv_temporary_detected": (
                self.query_head_sized_kv_temporary_detected
            ),
            "measured_torch_cat_detected": self.measured_torch_cat_detected,
            "host_synchronization_detected": (
                self.host_synchronization_detected
            ),
            "cache_growth_detected": self.cache_growth_detected,
            "backend_fallback_detected": self.backend_fallback_detected,
            "stable_post_warmup_path": self.stable_post_warmup_path,
            "reasons": list(self.reasons),
            "kernel_sequence_sha256": self.kernel_sequence_sha256,
            "timings_retained": False,
            "performance_claim_eligible": False,
            "r_hbm": None,
        }


def _kernel_names(value: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of kernel names")
    names = tuple(value)
    if not names or any(type(name) is not str or not name for name in names):
        raise ValueError(f"{label} must contain non-empty kernel names")
    return names


def _query_head_sized_kv_temporary(
    role: str,
    shape_value: Sequence[int],
) -> bool:
    shape = tuple(shape_value)
    if (
        not role
        or any(type(item) is not int or item <= 0 for item in shape)
        or len(shape) < 2
    ):
        raise ValueError("temporary shape evidence is invalid")
    lowered = role.casefold()
    is_kv = (
        "key" in lowered
        or "value" in lowered
        or lowered.startswith("kv")
    )
    if not is_kv:
        return False
    if any(
        token in lowered
        for token in (
            "logit",
            "output",
            "workspace",
            "weight",
            "query",
            "merge",
        )
    ):
        return False
    if len(shape) == 3:
        return shape[0] == 32 or shape[1] == 32
    return shape[-3] == 32 or shape[-2] == 32


def kivi_adapter_hot_path_source() -> str:
    """Return the explicit Phase 8 adapter/cache call closure under audit."""

    return "\n".join(
        inspect.getsource(function)
        for function in (
            KIVIMethodAdapter._runtime,
            KIVIMethodAdapter._require_cache,
            KIVIMethodAdapter._handle,
            KIVIMethodAdapter._validate_update,
            KIVIMethodAdapter._quantization_scratch,
            KIVIMethodAdapter._quantize_into,
            KIVIMethodAdapter._store_historical_k,
            KIVIMethodAdapter._store_historical_v,
            KIVIMethodAdapter._commit_token,
            KIVIMethodAdapter._layer_context,
            KIVIMethodAdapter.store_prefill,
            KIVIMethodAdapter.append_decode,
            KIVIMethodAdapter._decode_compressed,
            KIVIMethodAdapter.decode_attention,
            kivi_adapter_module._KIVIDirectGEMVLauncher.launch_into,
            KIVIStaticCache._check_layer,
            KIVIStaticCache.select_growing_step,
            KIVIStaticCache.update,
            KIVIStaticCache.fixed_scratch_overwrite,
        )
    )


def derive_kivi_static_execution_precheck(
    *,
    adapter_source_sha256: str,
    hot_path_source: str | None = None,
    observer_source: str | None = None,
) -> dict[str, Any]:
    """Independently derive the static half of the execution-path audit."""

    require_sha256(
        adapter_source_sha256,
        field_name="adapter_source_sha256",
    )
    source = (
        kivi_adapter_hot_path_source()
        if hot_path_source is None
        else hot_path_source
    )
    if type(source) is not str or not source:
        raise ValueError("KIVI hot-path source must be nonempty")
    lowered = source.casefold()
    forbidden = (
        "torch.cat(",
        "repeat_kv",
        "repeat_interleave",
        "dequantize_full",
        "dequantize_prefix",
        ".cpu(",
        ".item(",
        ".tolist(",
        "synchronize(",
    )
    forbidden_present = tuple(
        token for token in forbidden if token in lowered
    )
    observed_launcher_source = (
        inspect.getsource(
            kivi_adapter_module._KIVIDirectGEMVLauncher
        )
        if observer_source is None
        else observer_source
    )
    if (
        type(observed_launcher_source) is not str
        or not observed_launcher_source
    ):
        raise ValueError("KIVI launcher observer source must be nonempty")
    observer_available = bool(
        "def begin_observation(" in observed_launcher_source
        and "def end_observation(" in observed_launcher_source
        and "self._observation.append(" in observed_launcher_source
    )
    authority_passed = (
        KIVI_OFFICIAL_COMMIT,
        KIVI_OFFICIAL_BASE_TREE,
        KIVI_PATCHED_TREE,
        KIVI_DECISION_0018_PATCH_SHA256,
        KIVI_EXTENSION_SHA256,
        KIVI_NEW_PACK_SHA256,
        KIVI_FIXTURE_ROOT_SHA256,
        KIVI_BGEMV2_HOST_STUB_OFFSET,
        KIVI_BGEMV4_HOST_STUB_OFFSET,
    ) == (
        PHASE8_OFFICIAL_COMMIT,
        PHASE8_BASE_TREE,
        PHASE8_PATCHED_TREE,
        PHASE8_DECISION_0018_PATCH_SHA256,
        PHASE8_EXTENSION_SHA256,
        KIVI_NEW_PACK_SHA256,
        PHASE8_FIXTURE_ROOT_DIGEST,
        OFFICIAL_KIVI_HOST_STUB_OFFSETS[2],
        OFFICIAL_KIVI_HOST_STUB_OFFSETS[4],
    )
    passed = not forbidden_present and observer_available and authority_passed
    return {
        "schema_version": (
            "kvbench-phase8-kivi-static-execution-precheck-1.0.0"
        ),
        "passed": passed,
        "forbidden_tokens_present": list(forbidden_present),
        "launcher_observer_available": observer_available,
        "authority_passed": authority_passed,
        "adapter_source_sha256": adapter_source_sha256,
        "instrumented_runtime_observation_required": True,
        "runtime_path_claimed_by_static_precheck": False,
    }


def audit_kivi_execution_path(
    *,
    kernel_names: Sequence[str],
    repeated_kernel_names: Sequence[str],
    runtime_event_names: Sequence[str],
    temporary_shapes: Mapping[str, Sequence[int]],
    adapter_hot_path_source: str,
    observed_extension_sha256: str,
    observed_new_pack_sha256: str,
    official_commit: str,
    official_base_tree: str,
    patched_tree: str,
    decision_0018_patch_sha256: str,
    fixture_root_digest: str,
    host_stub_offsets: Mapping[int, int],
    backend_fallback_observed: bool,
    cache_growth_observed: bool,
) -> KIVIExecutionPathAudit:
    """Apply the frozen KIVI path criteria without collecting a new trace."""

    first = _kernel_names(kernel_names, "kernel_names")
    repeated = _kernel_names(
        repeated_kernel_names, "repeated_kernel_names"
    )
    runtime = tuple(runtime_event_names)
    if any(type(name) is not str or not name for name in runtime):
        raise ValueError("runtime event names must be non-empty strings")
    if type(adapter_hot_path_source) is not str or not adapter_hot_path_source:
        raise ValueError("adapter hot-path source must be non-empty")
    if (
        type(backend_fallback_observed) is not bool
        or type(cache_growth_observed) is not bool
        or not temporary_shapes
    ):
        raise ValueError("execution-path observations are incomplete")
    require_sha256(
        observed_extension_sha256,
        field_name="observed_extension_sha256",
    )
    require_sha256(
        observed_new_pack_sha256,
        field_name="observed_new_pack_sha256",
    )

    two_bit = any(OFFICIAL_KIVI_KERNEL_FAMILIES[0] in name for name in first)
    four_bit = any(OFFICIAL_KIVI_KERNEL_FAMILIES[1] in name for name in first)
    extension_identity = observed_extension_sha256 == KIVI_EXTENSION_SHA256
    source_identity = observed_new_pack_sha256 == KIVI_NEW_PACK_SHA256
    patched_authority = (
        official_commit,
        official_base_tree,
        patched_tree,
        decision_0018_patch_sha256,
        fixture_root_digest,
    ) == (
        KIVI_OFFICIAL_COMMIT,
        KIVI_OFFICIAL_BASE_TREE,
        KIVI_PATCHED_TREE,
        KIVI_DECISION_0018_PATCH_SHA256,
        KIVI_FIXTURE_ROOT_SHA256,
    )
    offsets = dict(host_stub_offsets)
    offsets_verified = offsets == OFFICIAL_KIVI_HOST_STUB_OFFSETS
    source = adapter_hot_path_source.casefold()
    all_kernel_text = "\n".join((*first, *repeated)).casefold()
    full_prefix_dequant = any(
        token in all_kernel_text or token in source
        for token in (
            "full_prefix_dequant",
            "dequantize_full",
            "dequantize_prefix",
            "complete_prefix_dequant",
        )
    )
    full_prefix_temporary = any(
        token in role.casefold()
        for role in temporary_shapes
        for token in ("full_prefix", "complete_prefix", "prefix_kv")
    )
    gqa_materialization = any(
        token in all_kernel_text or token in source
        for token in (
            "repeat_kv",
            "repeat_interleave",
            ".expand(",
        )
    )
    hq_temporary = any(
        _query_head_sized_kv_temporary(role, shape)
        for role, shape in temporary_shapes.items()
    )
    measured_cat = "torch.cat(" in source or ".cat(" in source
    host_sync = any(
        token in name
        for name in runtime
        for token in (
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
        )
    ) or any(
        token in source
        for token in (".cpu(", ".item(", ".tolist(", "synchronize(")
    )
    native_gqa = (
        "kv_head = query_head // kivi_gqa_group_size" in source
        and not gqa_materialization
    )
    stable = first == repeated
    fallback = (
        backend_fallback_observed
        or not two_bit
        or not four_bit
        or any(
            token in all_kernel_text
            for token in (
                "flash_fwd",
                "scaled_dot_product",
                "efficient_attention",
                "xformers",
                "cudnn_attention",
            )
        )
    )
    cache_growth = cache_growth_observed or measured_cat

    checks = {
        "official_bgemv2_kernel_absent": two_bit,
        "official_bgemv4_kernel_absent": four_bit,
        "extension_identity_mismatch": extension_identity,
        "new_pack_source_identity_mismatch": source_identity,
        "patched_authority_mismatch": patched_authority,
        "official_host_stub_offsets_mismatch": offsets_verified,
        "native_gqa_indexing_unverified": native_gqa,
        "full_prefix_dequantization_detected": not full_prefix_dequant,
        "full_prefix_temporary_detected": not full_prefix_temporary,
        "gqa_materialization_detected": not gqa_materialization,
        "query_head_sized_kv_temporary_detected": not hq_temporary,
        "measured_torch_cat_detected": not measured_cat,
        "host_synchronization_detected": not host_sync,
        "cache_growth_detected": not cache_growth,
        "backend_fallback_detected": not fallback,
        "post_warmup_path_unstable": stable,
    }
    reasons = tuple(reason for reason, passed in checks.items() if not passed)
    sequence = "\n".join((*first, *repeated)).encode("utf-8")
    return KIVIExecutionPathAudit(
        passed=not reasons,
        kernel_families=OFFICIAL_KIVI_KERNEL_FAMILIES,
        two_bit_kernel_verified=two_bit,
        four_bit_kernel_verified=four_bit,
        extension_identity_verified=extension_identity,
        source_identity_verified=source_identity,
        patched_authority_verified=patched_authority,
        host_stub_offsets_verified=offsets_verified,
        native_gqa_indexing_verified=native_gqa,
        full_prefix_dequantization_detected=full_prefix_dequant,
        full_prefix_temporary_detected=full_prefix_temporary,
        gqa_materialization_detected=gqa_materialization,
        query_head_sized_kv_temporary_detected=hq_temporary,
        measured_torch_cat_detected=measured_cat,
        host_synchronization_detected=host_sync,
        cache_growth_detected=cache_growth,
        backend_fallback_detected=fallback,
        stable_post_warmup_path=stable,
        reasons=reasons,
        kernel_sequence_sha256=hashlib.sha256(sequence).hexdigest(),
    )


def summarize_phase8_accounting(
    manifests: Sequence[Phase8RunManifest],
) -> dict[str, Any]:
    """Return canonical physical-capacity ratios for the exact grid."""

    records = require_exact_phase8_grid(manifests)
    points: list[dict[str, Any]] = []
    for record in records:
        accounting = record.accounting
        product_error = abs(accounting.r_alloc * accounting.rho_alloc - 1.0)
        if (
            product_error > RECIPROCAL_ABS_TOLERANCE
            or accounting.breakdown.total != accounting.allocated_bytes
            or accounting.predicted_relative_error >= 0.01
            or accounting.r_hbm is not None
        ):
            raise KIVIAdmissionError(
                f"canonical accounting failed: {record.run_id}"
            )
        points.append(
            {
                "run_id": record.run_id,
                "configuration": record.method_configuration,
                "runner_kind": record.runner_kind.value,
                "graph_mode": record.graph_mode.value,
                "context_length": record.context_length,
                "output_steps": record.output_steps,
                "allocated_bytes": accounting.allocated_bytes,
                "active_storage_bytes": accounting.active_storage_bytes,
                "logical_bf16_allocated_bytes": (
                    accounting.logical_bf16_allocated_bytes
                ),
                "logical_bf16_active_bytes": (
                    accounting.logical_bf16_active_bytes
                ),
                "rho_alloc": accounting.rho_alloc,
                "r_alloc": accounting.r_alloc,
                "reciprocal_product_error": product_error,
                "predicted_relative_error": (
                    accounting.predicted_relative_error
                ),
                "temporary_peak_bytes": accounting.temporary_peak_bytes,
                "r_hbm": None,
            }
        )
    body = {
        "schema_version": "kvbench-phase8-accounting-summary-1.0.0",
        "points": points,
        "canonical_ratio_semantics": {
            "rho_alloc": "allocated_bytes / logical_bf16_allocated_bytes",
            "r_alloc": "logical_bf16_allocated_bytes / allocated_bytes",
        },
        "r_hbm": None,
        "performance_claim_eligible": False,
    }
    return {
        **body,
        "summary_sha256": sha256_hex(canonical_json_bytes(body)),
    }


@dataclass(frozen=True, slots=True)
class Phase8DurablePublication:
    """Strict host-side outcome parsed from the inner publication receipt."""

    local_root_digest: str
    r2_uri: str
    bucket_lock_identity: str
    object_count: int
    complete_uploaded_last: bool
    inventory_valid: bool
    checksum_valid: bool
    clean_retrieval: bool
    retrieved_bundle_valid: bool

    def __post_init__(self) -> None:
        require_sha256(
            self.local_root_digest, field_name="local_root_digest"
        )
        if (
            type(self.r2_uri) is not str
            or type(self.bucket_lock_identity) is not str
            or type(self.object_count) is not int
            or self.object_count <= 0
            or type(self.complete_uploaded_last) is not bool
            or type(self.inventory_valid) is not bool
            or type(self.checksum_valid) is not bool
            or type(self.clean_retrieval) is not bool
            or type(self.retrieved_bundle_valid) is not bool
        ):
            raise ValueError("durable publication outcome is invalid")

    @property
    def publication_passed(self) -> bool:
        expected_uri = (
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{self.local_root_digest}/"
        )
        return bool(
            self.complete_uploaded_last
            and self.inventory_valid
            and self.checksum_valid
            and self.r2_uri == expected_uri
            and self.bucket_lock_identity.strip()
        )

    @property
    def retrieval_passed(self) -> bool:
        return bool(
            self.publication_passed
            and self.clean_retrieval
            and self.retrieved_bundle_valid
        )


@dataclass(frozen=True, slots=True)
class Phase8DerivedAdmissionEvidence:
    """All G2-KIVI inputs independently derived from one finalized bundle."""

    manifests: tuple[Phase8RunManifest, ...]
    checks: tuple[Phase8AdmissionCheck, ...]
    evidence_references: tuple[MethodAdmissionEvidenceReference, ...]
    execution_path_audit: KIVIExecutionPathAudit
    durable_publication: Phase8DurablePublication


@dataclass(frozen=True, slots=True)
class _ValidatedInnerArtifact:
    root_sha256: str
    files: tuple[Path, ...]


def _strict_json_value(path: Path, *, label: str) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise KIVIAdmissionError(f"{label} is absent or invalid") from error
    return payload


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _strict_json_value(path, label=label)
    if not isinstance(payload, dict):
        raise KIVIAdmissionError(f"{label} must be a JSON object")
    return payload


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KIVIAdmissionError(f"{label} must be an object")
    return value


def _require_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise KIVIAdmissionError(f"{label} must be an array")
    return value


def _reject_receipt_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                isinstance(key, str)
                and key.casefold() in _PROHIBITED_RECEIPT_KEYS
            ):
                raise KIVIAdmissionError(
                    "publication receipt contains a credential field"
                )
            if key in _FORBIDDEN_INNER_REPORT_VALIDATION_KEYS:
                raise KIVIAdmissionError(
                    "inner receipt cannot validate the MethodAdmissionReport"
                )
            _reject_receipt_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_receipt_keys(child)


def _reject_artifact_credential_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                isinstance(key, str)
                and key.casefold() in _PROHIBITED_RECEIPT_KEYS
            ):
                raise KIVIAdmissionError(
                    "inner artifact contains a credential field"
                )
            _reject_artifact_credential_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_artifact_credential_keys(child)


def _resolve_within(
    path: Path,
    *,
    root: Path,
    label: str,
    require_file: bool,
) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise KIVIAdmissionError(f"{label} is absent or escaped") from error
    if path.is_symlink() or (
        require_file and not resolved.is_file()
    ) or (not require_file and not resolved.is_dir()):
        raise KIVIAdmissionError(f"{label} is unsafe")
    return resolved


def _validate_finalized_inner_artifact(
    directory: Path,
) -> _ValidatedInnerArtifact:
    validation = validate_run_directory(directory, expect_final_name=True)
    if (
        not validation.valid
        or not validation.complete
        or validation.status != RunStatus.COMPLETED.value
    ):
        raise KIVIAdmissionError(
            "Phase 8 inner bundle failed immutable validation"
        )
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    try:
        root_metadata = directory.lstat()
    except OSError as error:
        raise KIVIAdmissionError("Phase 8 inner bundle is unsafe") from error
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_mode & write_bits
    ):
        raise KIVIAdmissionError("Phase 8 inner bundle is unsafe")
    files: list[Path] = []
    for candidate in sorted(directory.rglob("*")):
        relative = candidate.relative_to(directory)
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise KIVIAdmissionError(
                "Phase 8 inner bundle contains a symlink"
            )
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_mode & write_bits:
                raise KIVIAdmissionError(
                    "Phase 8 inner bundle remains writable"
                )
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & write_bits
        ):
            raise KIVIAdmissionError(
                "Phase 8 inner bundle contains an unsafe file"
            )
        if any(
            part == ".env" or part.startswith(".env.")
            for part in relative.parts
        ):
            raise KIVIAdmissionError(
                "Phase 8 inner bundle contains a prohibited env file"
            )
        if candidate.suffix == ".json":
            _reject_artifact_credential_keys(
                _strict_json_value(
                    candidate,
                    label=f"inner JSON {relative.as_posix()}",
                )
            )
        files.append(candidate)
    canonical = "".join(
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}\n"
        for path in files
    ).encode("utf-8")
    return _ValidatedInnerArtifact(
        root_sha256=hashlib.sha256(canonical).hexdigest(),
        files=tuple(files),
    )


def _inner_r2_uri(root_digest: str) -> str:
    return (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_digest}/"
    )


def _parse_publication_receipt(
    *,
    receipt_path: Path,
    evidence_root: Path,
    inner_root_digest: str,
    inner_object_count: int,
    source_run_id: str,
    source_git_sha: str,
) -> Phase8DurablePublication:
    path = _resolve_within(
        receipt_path,
        root=evidence_root,
        label="Phase 8 inner publication receipt",
        require_file=True,
    )
    receipt = _strict_json(
        path,
        label="Phase 8 inner publication receipt",
    )
    _reject_receipt_keys(receipt)
    local = _require_mapping(
        receipt.get("local_validation"),
        label="receipt local_validation",
    )
    publication = _require_mapping(
        receipt.get("publication"),
        label="receipt publication",
    )
    retrieval = _require_mapping(
        receipt.get("clean_retrieval"),
        label="receipt clean_retrieval",
    )
    lock = _require_mapping(
        receipt.get("bucket_lock"),
        label="receipt bucket_lock",
    )
    expected_uri = _inner_r2_uri(inner_root_digest)
    lock_rule_name = lock.get("lock_rule_name")
    verified_at = lock.get("verified_at_utc")
    if (
        receipt.get("schema_version") != PHASE8_INNER_RECEIPT_SCHEMA
        or receipt.get("admission_status") != "PASS"
        or receipt.get("artifact_status") != "completed"
        or receipt.get("source_git_sha") != source_git_sha
        or receipt.get("source_run_id") != source_run_id
        or receipt.get("credential_values_recorded") is not False
        or receipt.get("env_file_read") is not False
        or local.get("valid") is not True
        or local.get("complete") is not True
        or local.get("status") != "completed"
        or local.get("root_sha256") != inner_root_digest
        or local.get("object_count") != inner_object_count
        or local.get("complete_marker_valid") is not True
        or local.get("inventory_valid") is not True
        or local.get("checksum_ledger_valid") is not True
        or local.get("root_digest_valid") is not True
        or local.get("bundle_validation_valid") is not True
        or publication.get("result") != "PASS"
        or publication.get("root_sha256") != inner_root_digest
        or publication.get("uri") != expected_uri
        or publication.get("object_count") != inner_object_count
        or publication.get("content_addressed") is not True
        or publication.get("conditional_writes") is not True
        or publication.get("complete_last") is not True
        or retrieval.get("result") != "PASS"
        or retrieval.get("root_sha256") != inner_root_digest
        or retrieval.get("object_count") != inner_object_count
        or retrieval.get("destination_initially_empty") is not True
        or retrieval.get("complete_marker_valid") is not True
        or retrieval.get("inventory_valid") is not True
        or retrieval.get("checksum_ledger_valid") is not True
        or retrieval.get("root_digest_valid") is not True
        or retrieval.get("bundle_validation_valid") is not True
        or retrieval.get("unexpected_objects") is not False
        or lock.get("provider") != "cloudflare_r2"
        or lock.get("bucket") != "kvbench-artifacts"
        or lock.get("endpoint")
        != "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
        or lock.get("endpoint_class") != "cloudflare_r2_s3"
        or lock.get("bucket_exists") is not True
        or lock.get("verification_result") != "PASS"
        or lock.get("enabled") is not True
        or lock.get("public_state_result") != "PASS"
        or lock.get("managed_r2_dev_enabled") is not False
        or lock.get("public_r2_dev") is not False
        or type(lock.get("custom_domain_count")) is not int
        or lock.get("custom_domain_count") < 0
        or type(lock.get("enabled_custom_domain_count")) is not int
        or lock.get("enabled_custom_domain_count") != 0
        or lock.get("public_custom_domain") is not False
        or not isinstance(lock.get("lock_rule_id"), str)
        or not lock["lock_rule_id"].strip()
        or lock["lock_rule_id"] != lock["lock_rule_id"].strip()
        or "lock_rule_name" not in lock
        or (
            lock_rule_name is not None
            and (
                not isinstance(lock_rule_name, str)
                or not lock_rule_name.strip()
            )
        )
        or lock.get("lock_scope") != "exact"
        or lock.get("covered_prefix") != "kvbench/sha256/"
        or lock.get("lock_prefix") != "kvbench/sha256/"
        or lock.get("retention_type") != "Indefinite"
        or lock.get("retention_condition") != "Indefinite"
        or lock.get("bucket_public") is not False
        or not isinstance(verified_at, str)
        or not verified_at.endswith("Z")
        or "T" not in verified_at
    ):
        raise KIVIAdmissionError(
            "Phase 8 inner publication receipt does not bind the bundle"
        )
    return Phase8DurablePublication(
        local_root_digest=inner_root_digest,
        r2_uri=expected_uri,
        bucket_lock_identity=lock["lock_rule_id"],
        object_count=inner_object_count,
        complete_uploaded_last=True,
        inventory_valid=True,
        checksum_valid=True,
        clean_retrieval=True,
        retrieved_bundle_valid=True,
    )


def _require_source_digest(
    *,
    repository_root: Path,
    relative_path: str,
    observed_sha256: object,
    label: str,
) -> None:
    require_sha256(str(observed_sha256), field_name=f"{label} SHA-256")
    source = _resolve_within(
        repository_root.joinpath(*relative_path.split("/")),
        root=repository_root,
        label=label,
        require_file=True,
    )
    if sha256_file(source) != observed_sha256:
        raise KIVIAdmissionError(f"{label} identity differs")


def _require_historical_source_digest(
    *,
    repository_root: Path,
    execution_git_sha: str,
    relative_path: str,
    observed_sha256: object,
    label: str,
) -> None:
    require_sha256(str(observed_sha256), field_name=f"{label} SHA-256")
    if (
        _phase8_git_blob_sha256(
            repository_root,
            revision=execution_git_sha,
            relative_path=relative_path,
        )
        != observed_sha256
    ):
        raise KIVIAdmissionError(f"{label} identity differs")


def _supervision_passed(
    value: object,
    *,
    expected_argv: Sequence[str] | None = None,
    expected_stdout_sha256: str | None = None,
    expected_stderr_sha256: str | None = None,
) -> bool:
    """Validate the exact generic direct-child supervision evidence schema."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "identity",
        "command",
        "timeout",
        "returncode",
        "pidfd",
        "direct_child",
        "final_reap",
        "stdout",
        "stderr",
    }:
        return False
    identity = value.get("identity")
    command = value.get("command")
    timeout = value.get("timeout")
    pidfd = value.get("pidfd")
    direct_child = value.get("direct_child")
    final_reap = value.get("final_reap")
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    if (
        value.get("schema_version")
        != "kvbench-generic-supervised-command-result-1.0.0"
        or not isinstance(identity, Mapping)
        or set(identity) != {"pid", "start_time_ticks", "parent_pid"}
        or not isinstance(command, Mapping)
        or set(command)
        != {
            "argv",
            "working_directory",
            "environment_sha256",
            "command_fingerprint",
            "shell",
        }
        or not isinstance(timeout, Mapping)
        or set(timeout)
        != {
            "timeout_seconds",
            "timed_out",
            "terminate_requested",
            "kill_requested",
        }
        or not isinstance(pidfd, Mapping)
        or set(pidfd) != {"supported", "opened", "descriptor", "closed"}
        or not isinstance(direct_child, Mapping)
        or set(direct_child)
        != {
            "verified",
            "expected_parent_pid",
            "parent_pid_verified",
            "start_time_ticks_verified",
            "process_handle_retained",
        }
        or not isinstance(final_reap, Mapping)
        or set(final_reap) != {"completed", "count"}
        or not isinstance(stdout, Mapping)
        or set(stdout) != {"bytes", "sha256"}
        or not isinstance(stderr, Mapping)
        or set(stderr) != {"bytes", "sha256"}
    ):
        return False

    argv = command.get("argv")
    working_directory = command.get("working_directory")
    timeout_seconds = timeout.get("timeout_seconds")
    pidfd_opened = pidfd.get("opened")
    descriptor = pidfd.get("descriptor")
    if (
        type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or type(identity.get("start_time_ticks")) is not int
        or identity["start_time_ticks"] < 0
        or type(identity.get("parent_pid")) is not int
        or identity["parent_pid"] <= 0
        or not isinstance(argv, list)
        or not argv
        or any(type(item) is not str or not item for item in argv)
        or (
            expected_argv is not None
            and argv != list(expected_argv)
        )
        or type(working_directory) is not str
        or not Path(working_directory).is_absolute()
        or type(command.get("environment_sha256")) is not str
        or type(command.get("command_fingerprint")) is not str
        or command.get("shell") is not False
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
        or timeout.get("timed_out") is not False
        or timeout.get("terminate_requested") is not False
        or timeout.get("kill_requested") is not False
        or type(value.get("returncode")) is not int
        or value["returncode"] != 0
        or type(pidfd.get("supported")) is not bool
        or type(pidfd_opened) is not bool
        or type(pidfd.get("closed")) is not bool
        or pidfd_opened is not pidfd.get("closed")
        or (pidfd_opened and pidfd.get("supported") is not True)
        or (
            pidfd_opened
            and (type(descriptor) is not int or descriptor < 0)
        )
        or (not pidfd_opened and descriptor is not None)
        or direct_child.get("verified") is not True
        or direct_child.get("expected_parent_pid")
        != identity["parent_pid"]
        or direct_child.get("parent_pid_verified") is not True
        or direct_child.get("start_time_ticks_verified") is not True
        or direct_child.get("process_handle_retained") is not True
        or final_reap.get("completed") is not True
        or final_reap.get("count") != 1
        or type(stdout.get("bytes")) is not int
        or stdout["bytes"] < 0
        or type(stderr.get("bytes")) is not int
        or stderr["bytes"] < 0
    ):
        return False
    try:
        require_sha256(
            command["environment_sha256"],
            field_name="supervised environment SHA-256",
        )
        require_sha256(
            command["command_fingerprint"],
            field_name="supervised command SHA-256",
        )
        require_sha256(stdout.get("sha256"), field_name="supervised stdout SHA-256")
        require_sha256(stderr.get("sha256"), field_name="supervised stderr SHA-256")
    except (TypeError, ValueError):
        return False
    return bool(
        (
            expected_stdout_sha256 is None
            or stdout["sha256"] == expected_stdout_sha256
        )
        and (
            expected_stderr_sha256 is None
            or stderr["sha256"] == expected_stderr_sha256
        )
    )


def _validate_exact_test_result(
    *,
    bundle_root: Path,
    repository_root: Path,
    evidence_name: str,
    source_path: str,
    required_test_names: Sequence[str],
    expected_count: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    prefix = bundle_root / "validation" / evidence_name
    result_path = prefix / "result.json"
    stdout_path = prefix / "stdout.txt"
    stderr_path = prefix / "stderr.txt"
    result = _strict_json(
        result_path,
        label=f"{evidence_name} result",
    )
    try:
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
    except OSError as error:
        raise KIVIAdmissionError(
            f"{evidence_name} output is absent"
        ) from error
    combined = (stdout + b"\n" + stderr).decode(
        "utf-8", errors="replace"
    )
    command = result.get("command")
    if (
        result.get("schema_version")
        != "kvbench-phase8-exact-container-test-1.0.0"
        or result.get("evidence_name") != evidence_name
        or result.get("source_path") != source_path
        or not isinstance(command, list)
        or len(command) < 3
        or command[-1] != "-v"
        or not isinstance(command[-2], str)
        or not command[-2].endswith(source_path)
        or result.get("exit_code") != 0
        or result.get("timed_out") is not False
        or result.get("container_digest")
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or result.get("performance_timing") is not False
        or result.get("passed") is not True
        or result.get("stdout_sha256")
        != hashlib.sha256(stdout).hexdigest()
        or result.get("stderr_sha256")
        != hashlib.sha256(stderr).hexdigest()
        or not _supervision_passed(
            result.get("process_supervision"),
            expected_argv=command,
            expected_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            expected_stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        )
        or f"Ran {expected_count} test" not in combined
        or "OK" not in combined
        or "FAILED" in combined
        or "skipped=" in combined
        or any(name not in combined for name in required_test_names)
    ):
        raise KIVIAdmissionError(
            f"{evidence_name} structured result did not pass"
        )
    _require_source_digest(
        repository_root=repository_root,
        relative_path=source_path,
        observed_sha256=result.get("source_sha256"),
        label=evidence_name,
    )
    return result, (
        result_path.relative_to(bundle_root).as_posix(),
        stdout_path.relative_to(bundle_root).as_posix(),
        stderr_path.relative_to(bundle_root).as_posix(),
    )


def _last_json_object(data: bytes) -> dict[str, Any] | None:
    for line in reversed(
        data.decode("utf-8", errors="replace").splitlines()
    ):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _validate_sanitizer_result(
    *,
    bundle_root: Path,
    repository_root: Path,
    execution_git_sha: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    prefix = bundle_root / "validation" / "sanitizer"
    result_path = prefix / "result.json"
    stdout_path = prefix / "stdout.txt"
    stderr_path = prefix / "stderr.txt"
    result = _strict_json(result_path, label="Compute Sanitizer result")
    try:
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
    except OSError as error:
        raise KIVIAdmissionError(
            "Compute Sanitizer output is absent"
        ) from error
    combined = (stdout + b"\n" + stderr).decode(
        "utf-8", errors="replace"
    )
    probe = _last_json_object(stdout)
    if probe is None:
        raise KIVIAdmissionError("Compute Sanitizer probe is absent")
    authority = _require_mapping(
        probe.get("authority"),
        label="sanitizer authority",
    )
    configurations = _require_list(
        probe.get("configurations"),
        label="sanitizer configurations",
    )
    by_configuration = {
        item.get("configuration"): item
        for item in configurations
        if isinstance(item, Mapping)
        and isinstance(item.get("configuration"), str)
    }
    expected_authority = {
        "official_commit": PHASE8_OFFICIAL_COMMIT,
        "official_base_tree": PHASE8_BASE_TREE,
        "patched_tree": PHASE8_PATCHED_TREE,
        "decision_0018_patch_sha256": (
            PHASE8_DECISION_0018_PATCH_SHA256
        ),
        "extension_sha256": PHASE8_EXTENSION_SHA256,
        "new_pack_sha256": KIVI_NEW_PACK_SHA256,
        "fixture_root_sha256": PHASE8_FIXTURE_ROOT_DIGEST,
    }
    k4v4 = by_configuration.get("k4v4")
    k2v2 = by_configuration.get("k2v2")
    tool = result.get("tool_identity")
    if (
        result.get("schema_version")
        != "kvbench-phase8-kivi-sanitizer-result-1.0.0"
        or result.get("probe_source_path")
        != PHASE8_SANITIZER_PROBE_PATH
        or result.get("extension_sha256") != PHASE8_EXTENSION_SHA256
        or result.get("container_digest")
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or result.get("exit_code") != 0
        or result.get("timed_out") is not False
        or result.get("stdout_sha256")
        != hashlib.sha256(stdout).hexdigest()
        or result.get("stderr_sha256")
        != hashlib.sha256(stderr).hexdigest()
        or result.get("memcheck_summaries_passed") is not True
        or result.get("probe_passed") is not True
        or result.get("rollover_covered") is not True
        or result.get("kernel_families")
        != list(OFFICIAL_KIVI_KERNEL_FAMILIES)
        or result.get("performance_timing") is not False
        or result.get("passed") is not True
        or not _supervision_passed(
            result.get("process_supervision"),
            expected_argv=result.get("command"),
            expected_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            expected_stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        )
        or not isinstance(tool, Mapping)
        or tool.get("exit_code") != 0
        or not isinstance(tool.get("sha256"), str)
        or len(tool["sha256"]) != 64
        or "LEAK SUMMARY: 0 bytes leaked in 0 allocations"
        not in combined
        or "ERROR SUMMARY: 0 errors" not in combined
        or probe.get("status") != "pass"
        or probe.get("container_digest")
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or probe.get("kernel_families")
        != list(OFFICIAL_KIVI_KERNEL_FAMILIES)
        or dict(authority) != expected_authority
        or set(by_configuration) != {"k4v4", "k2v2"}
        or not isinstance(k4v4, Mapping)
        or k4v4.get("gemv_bits") != [4]
        or k4v4.get("rollover") != "L31_to_L33"
        or k4v4.get("active_context") != 33
        or k4v4.get("key_history_tokens") != 32
        or k4v4.get("key_residual_tokens") != 1
        or k4v4.get("value_history_tokens") != 1
        or k4v4.get("value_residual_tokens") != 32
        or k4v4.get("finite") is not True
        or k4v4.get("token_movement") != "exact"
        or not isinstance(k2v2, Mapping)
        or k2v2.get("gemv_bits") != [2]
        or k2v2.get("active_context") != 33
        or k2v2.get("finite") is not True
        or k2v2.get("token_movement") != "exact"
    ):
        raise KIVIAdmissionError(
            "Compute Sanitizer structured result did not pass"
        )
    _require_source_digest(
        repository_root=repository_root,
        relative_path=PHASE8_SANITIZER_PROBE_PATH,
        observed_sha256=result.get("probe_source_sha256"),
        label="Compute Sanitizer probe",
    )
    _require_historical_source_digest(
        repository_root=repository_root,
        execution_git_sha=execution_git_sha,
        relative_path=PHASE8_ADAPTER_PATH,
        observed_sha256=result.get("adapter_source_sha256"),
        label="KIVI adapter",
    )
    return result, (
        result_path.relative_to(bundle_root).as_posix(),
        stdout_path.relative_to(bundle_root).as_posix(),
        stderr_path.relative_to(bundle_root).as_posix(),
    )


@dataclass(frozen=True, slots=True)
class _ValidatedPointEvidence:
    manifest: Phase8RunManifest
    point: Mapping[str, Any]
    first_kernel_names: tuple[str, ...]
    repeated_kernel_names: tuple[str, ...]
    relative_paths: tuple[str, ...]
    allocation_relative_paths: tuple[str, ...]


def _execution_audit_from_bundle(
    *,
    bundle_root: Path,
    repository_root: Path,
    points: Sequence[_ValidatedPointEvidence],
    execution_git_sha: str,
) -> tuple[KIVIExecutionPathAudit, tuple[str, ...]]:
    static_path = (
        bundle_root
        / "validation"
        / "execution-path-static-precheck.json"
    )
    execution_path = bundle_root / "validation" / "execution-path.json"
    static = _strict_json(
        static_path,
        label="static execution-path precheck",
    )
    execution = _strict_json(
        execution_path,
        label="execution-path audit",
    )
    first_names = tuple(
        name for point in points for name in point.first_kernel_names
    )
    repeated_names = tuple(
        name for point in points for name in point.repeated_kernel_names
    )
    sequence_digest = hashlib.sha256(
        "\n".join((*first_names, *repeated_names)).encode("utf-8")
    ).hexdigest()
    adapter_sha256 = execution.get("adapter_source_sha256")
    _require_historical_source_digest(
        repository_root=repository_root,
        execution_git_sha=execution_git_sha,
        relative_path=PHASE8_ADAPTER_PATH,
        observed_sha256=adapter_sha256,
        label="KIVI adapter",
    )
    try:
        adapter_payload = _phase8_git(
            repository_root,
            "cat-file",
            "blob",
            f"{execution_git_sha}:{PHASE8_ADAPTER_PATH}",
            binary=True,
        )
        cache_payload = _phase8_git(
            repository_root,
            "cat-file",
            "blob",
            f"{execution_git_sha}:{PHASE8_CACHE_PATH}",
            binary=True,
        )
        if not isinstance(adapter_payload, bytes) or not isinstance(
            cache_payload, bytes
        ):
            raise UnicodeError("historical source payload is not bytes")
        adapter_source_text = adapter_payload.decode("utf-8")
        cache_source_text = cache_payload.decode("utf-8")
    except (KIVIAdmissionError, UnicodeError) as error:
        raise KIVIAdmissionError(
            "KIVI execution-path source is unreadable"
        ) from error
    audited_source = f"{adapter_source_text}\n{cache_source_text}"
    observed_offsets = execution.get("host_stub_offsets")
    derived_static = derive_kivi_static_execution_precheck(
        adapter_source_sha256=adapter_sha256,
        hot_path_source=audited_source,
        observer_source=adapter_source_text,
    )
    if (
        static != derived_static
        or derived_static.get("passed") is not True
        or observed_offsets
        != {
            str(bits): offset
            for bits, offset in OFFICIAL_KIVI_HOST_STUB_OFFSETS.items()
        }
        or execution.get("runtime_launcher_probe_count") != len(points)
        or execution.get("runtime_observation_instrumented_separately")
        is not True
        or execution.get("normal_timing_instrumented") is not False
        or execution.get("kernel_sequence_sha256") != sequence_digest
    ):
        raise KIVIAdmissionError(
            "execution-path evidence is inconsistent"
        )
    first_manifest = points[0].manifest
    derived_audit = audit_kivi_execution_path(
        kernel_names=first_names,
        repeated_kernel_names=repeated_names,
        runtime_event_names=("cudaLaunchKernel",),
        temporary_shapes={
            "query_staging": (1, 32, 128),
            "key_staging": (1, 8, 1, 128),
            "value_staging": (1, 8, 1, 128),
            "logits_workspace": (
                1,
                32,
                first_manifest.capacity,
            ),
            "output_buffer": (1, 32, 128),
        },
        adapter_hot_path_source=audited_source,
        observed_extension_sha256=KIVI_EXTENSION_SHA256,
        observed_new_pack_sha256=KIVI_NEW_PACK_SHA256,
        official_commit=KIVI_OFFICIAL_COMMIT,
        official_base_tree=KIVI_OFFICIAL_BASE_TREE,
        patched_tree=KIVI_PATCHED_TREE,
        decision_0018_patch_sha256=(
            KIVI_DECISION_0018_PATCH_SHA256
        ),
        fixture_root_digest=KIVI_FIXTURE_ROOT_SHA256,
        host_stub_offsets=OFFICIAL_KIVI_HOST_STUB_OFFSETS,
        backend_fallback_observed=False,
        cache_growth_observed=False,
    )
    expected_execution = {
        **derived_audit.to_dict(),
        "adapter_source_path": PHASE8_ADAPTER_PATH,
        "adapter_source_sha256": adapter_sha256,
        "host_stub_offsets": {
            str(bits): offset
            for bits, offset in OFFICIAL_KIVI_HOST_STUB_OFFSETS.items()
        },
        "runtime_launcher_probe_count": len(points),
        "runtime_observation_instrumented_separately": True,
        "normal_timing_instrumented": False,
    }
    if execution != expected_execution or not derived_audit.passed:
        raise KIVIAdmissionError("execution-path audit did not pass")
    return derived_audit, (
        static_path.relative_to(bundle_root).as_posix(),
        execution_path.relative_to(bundle_root).as_posix(),
    )


def _validate_allocation_record(
    value: object,
    *,
    manifest: Phase8RunManifest,
    run_root: Path,
    bundle_root: Path,
    repository_root: Path,
) -> tuple[str, ...]:
    allocation = _require_mapping(value, label="point allocation")
    operations = _require_list(
        allocation.get("operation_allocations"),
        label="point allocation operations",
    )
    graph_required = manifest.graph_mode is GraphMode.CUDA_GRAPH
    try:
        operation_keys = build_kivi_operation_keys(
            configuration=manifest.method_configuration,
            runner_kind=manifest.runner_kind,
            graph_mode=manifest.graph_mode,
            starting_context=manifest.context_length,
            output_steps=manifest.output_steps,
        )
    except ValueError as error:
        raise KIVIAdmissionError(
            "point allocation operation set is outside the frozen grid"
        ) from error
    expected_operation_count = len(operation_keys)
    expected_outputs: list[dict[str, Any]] = []
    evidence_paths: list[str] = []
    if (
        set(allocation)
        != {
            "passed",
            "operation_allocations",
            "unknown_allocation_count",
            "all_ephemeral_allocations_attributed",
            "graph_required",
            "graph_passed",
            "pointers_stable",
            "outputs",
        }
        or allocation.get("passed") is not True
        or len(operations) != expected_operation_count
        or allocation.get("unknown_allocation_count") != 0
        or allocation.get("all_ephemeral_allocations_attributed")
        is not (None if graph_required else True)
        or allocation.get("graph_required") is not graph_required
        or allocation.get("graph_passed") is not True
        or allocation.get("pointers_stable") is not True
    ):
        raise KIVIAdmissionError("point allocation summary did not pass")
    source_authority = resolve_phase8_historical_source_authority(
        repository_root=repository_root,
        execution_git_sha=manifest.git_sha,
        manifest_adapter_sha256=manifest.adapter_source_sha256,
    )
    backend_identity = _phase8_historical_backend_fingerprint(
        source_authority
    )
    for step, (operation, operation_key) in enumerate(
        zip(operations, operation_keys, strict=True)
    ):
        item = _require_mapping(operation, label="allocation operation")
        if set(item) != {
            "raw",
            "criterion",
            "raw_files",
            "raw_evidence_root",
            "raw_evidence_sha256",
        }:
            raise KIVIAdmissionError(
                "allocation operation envelope is not exact"
            )
        expected_binding = KIVIAllocationBinding(
            configuration=manifest.method_configuration,
            runner_kind=manifest.runner_kind.value,
            graph_mode=manifest.graph_mode.value,
            historical_context=operation_key.historical_context,
            attended_context=operation_key.attended_context,
            operation_fingerprint_sha256=(
                operation_key.operation_fingerprint_sha256
            ),
            cache_layout_fingerprint=(
                manifest.cache_layout_fingerprint
            ),
            method_fingerprint=manifest.method_fingerprint,
            backend_identity=backend_identity,
            adapter_source_sha256=(
                source_authority.adapter_source_sha256
            ),
            cache_source_sha256=source_authority.cache_source_sha256,
            endpoint_source_sha256=(
                source_authority.endpoint_source_sha256
            ),
            authorized_container_digest=(
                PHASE8_AUTHORIZED_CONTAINER_DIGEST
            ),
            official_commit=PHASE8_OFFICIAL_COMMIT,
            patched_tree=PHASE8_PATCHED_TREE,
            decision_0018_patch_sha256=(
                PHASE8_DECISION_0018_PATCH_SHA256
            ),
            extension_sha256=PHASE8_EXTENSION_SHA256,
        )
        expected_root = f"allocation/operations/step-{step:04d}"
        if item.get("raw_evidence_root") != expected_root:
            raise KIVIAdmissionError(
                "allocation raw evidence root differs"
            )
        raw_files = _require_mapping(
            item.get("raw_files"),
            label="allocation raw-file index",
        )
        evidence_directory = run_root / expected_root
        try:
            replay = replay_preserved_kivi_allocation_attribution(
                evidence_directory,
                raw_files=raw_files,
                expected_binding=expected_binding,
            )
        except KIVIAllocationError as error:
            raise KIVIAdmissionError(
                "allocation raw evidence did not replay semantically"
            ) from error
        observed_hashes = _require_mapping(
            item.get("raw_evidence_sha256"),
            label="allocation raw evidence hashes",
        )
        if (
            item.get("raw") != replay.summary["raw"]
            or item.get("criterion") != replay.summary["criterion"]
            or raw_files != replay.summary["raw_files"]
            or dict(observed_hashes)
            != dict(replay.file_sha256_by_basename)
        ):
            raise KIVIAdmissionError(
                "allocation summary differs from raw semantic replay"
            )
        witness_output = _require_mapping(
            replay.operation_witness.get("measured_output"),
            label="allocation witnessed output",
        )
        expected_outputs.append(dict(witness_output))
        evidence_paths.extend(
            (
                evidence_directory / basename
            ).relative_to(bundle_root).as_posix()
            for basename in replay.file_sha256_by_basename
        )
    if allocation.get("outputs") != expected_outputs:
        raise KIVIAdmissionError(
            "allocation witnessed outputs differ from point outputs"
        )
    return tuple(evidence_paths)


def _validate_launcher_probe(
    value: object,
    *,
    manifest: Phase8RunManifest,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    probe = _require_mapping(value, label="launcher observation")
    first = _require_list(
        probe.get("first_sequence"),
        label="first launcher sequence",
    )
    repeated = _require_list(
        probe.get("second_sequence"),
        label="second launcher sequence",
    )
    if not first or first != repeated:
        raise KIVIAdmissionError("launcher sequences are absent or unstable")
    expected_bits = {manifest.k_bits, manifest.v_bits}
    observed_bits: set[int] = set()
    names: list[str] = []
    for record in first:
        item = _require_mapping(record, label="launcher record")
        bits = item.get("bits")
        family = item.get("kernel_family")
        if (
            bits not in {2, 4}
            or family != f"bgemv{bits}_kernel_outer_dim"
            or item.get("group_size") != 32
            or item.get("num_query_heads") != 32
            or item.get("num_kv_heads") != 8
        ):
            raise KIVIAdmissionError("launcher record differs from KIVI")
        observed_bits.add(bits)
        names.append(family)
    first_digest = probe.get("first_output_sha256")
    second_digest = probe.get("second_output_sha256")
    if (
        probe.get("schema_version")
        != "kvbench-phase8-kivi-runtime-launch-observation-1.0.0"
        or probe.get("passed") is not True
        or probe.get("configuration") != manifest.method_configuration
        or probe.get("runner_kind") != manifest.runner_kind.value
        or probe.get("graph_mode") != manifest.graph_mode.value
        or probe.get("first_sequence") != probe.get("second_sequence")
        or observed_bits != expected_bits
        or probe.get("expected_bits") != sorted(expected_bits)
        or probe.get("observed_bits") != sorted(expected_bits)
        or not isinstance(first_digest, str)
        or not isinstance(second_digest, str)
        or first_digest != second_digest
        or len(first_digest) != 64
        or probe.get("first_output_finite") is not True
        or probe.get("second_output_finite") is not True
        or probe.get("stable_post_warmup_sequence") is not True
        or probe.get("instrumented_audit_separate") is not True
        or probe.get("allocation_audit_instrumented") is not False
        or probe.get("normal_timing_instrumented") is not False
        or probe.get("host_synchronization_outside_hot_path") is not True
    ):
        raise KIVIAdmissionError("launcher observation did not pass")
    return tuple(names), tuple(names)


def _validate_point_run(
    *,
    run_root: Path,
    manifest: Phase8RunManifest,
    bundle_root: Path,
    repository_root: Path,
) -> _ValidatedPointEvidence:
    point_path = run_root / "validation" / "point.json"
    allocation_path = run_root / "allocation" / "full-model.json"
    accounting_path = run_root / "accounting" / "bytes.json"
    launcher_path = (
        run_root / "execution-path" / "launcher-observation.json"
    )
    gqa_path = run_root / "gqa" / "full-model.json"
    numerical_path = run_root / "numerical" / "output.json"
    runner_path = run_root / "raw" / "runner.json"
    method_path = run_root / "config" / "method.json"
    environment_path = (
        run_root / "environment" / "container_identity.json"
    )
    point = _strict_json(point_path, label="point validation")
    allocation = _strict_json(
        allocation_path,
        label="point allocation",
    )
    accounting = _strict_json(
        accounting_path,
        label="point byte accounting",
    )
    launcher = _strict_json(
        launcher_path,
        label="point launcher observation",
    )
    gqa = _strict_json(gqa_path, label="point GQA evidence")
    numerical = _strict_json(
        numerical_path,
        label="point numerical evidence",
    )
    runner = _strict_json(runner_path, label="point runner result")
    method = _strict_json(method_path, label="point method record")
    environment = _strict_json(
        environment_path,
        label="point container identity",
    )
    expected_active = (
        manifest.context_length
        if manifest.runner_kind is RunnerKind.FIXED_L
        else manifest.capacity
    )
    graph_required = manifest.graph_mode is GraphMode.CUDA_GRAPH
    if (
        point.get("schema_version")
        != "kvbench-phase8-kivi-point-validation-1.0.0"
        or point.get("run_id") != manifest.run_id
        or point.get("git_sha") != manifest.git_sha
        or point.get("container_digest")
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or point.get("configuration") != manifest.method_configuration
        or point.get("runner_kind") != manifest.runner_kind.value
        or point.get("graph_mode") != manifest.graph_mode.value
        or point.get("batch_size") != 1
        or point.get("context_length") != manifest.context_length
        or point.get("output_steps") != manifest.output_steps
        or point.get("capacity") != manifest.capacity
        or point.get("accounting") != manifest.accounting.to_dict()
        or accounting != manifest.accounting.to_dict()
        or point.get("runtime_committed_context") != expected_active
        or point.get("byte_breakdown_sum")
        != manifest.accounting.allocated_bytes
        or point.get("allocated_bytes")
        != manifest.accounting.allocated_bytes
        or point.get("rho_alloc") != manifest.accounting.rho_alloc
        or point.get("r_alloc") != manifest.accounting.r_alloc
        or point.get("reciprocal_product_error")
        > RECIPROCAL_ABS_TOLERANCE
        or point.get("output_finite") is not True
        or point.get("cache_pointers_stable") is not True
        or point.get("historical_cache_unchanged") is not True
        or point.get("native_gqa") is not True
        or point.get("speedup_calculated") is not False
        or point.get("r_hbm") is not None
        or point.get("quality_status") != "unvalidated"
        or point.get("performance_claim_eligible") is not False
        or point.get("measurement_scope")
        != "measurement_container_admission"
        or point.get("passed") is not True
    ):
        raise KIVIAdmissionError(
            f"point evidence differs: {manifest.run_id}"
        )
    expected_rollover = (
        [31, 32, 33, 34]
        if manifest.runner_kind is RunnerKind.GROWING_CONTEXT
        else None
    )
    if point.get("rollover_active_lengths") != expected_rollover:
        raise KIVIAdmissionError(
            f"point rollover evidence differs: {manifest.run_id}"
        )
    if allocation != point.get("allocation"):
        raise KIVIAdmissionError("point allocation copies differ")
    allocation_evidence_paths = _validate_allocation_record(
        allocation,
        manifest=manifest,
        run_root=run_root,
        bundle_root=bundle_root,
        repository_root=repository_root,
    )
    first_names, repeated_names = _validate_launcher_probe(
        launcher,
        manifest=manifest,
    )
    if launcher != point.get("launcher_probe"):
        raise KIVIAdmissionError("point launcher copies differ")
    geometry = _require_mapping(
        gqa.get("geometry"),
        label="GQA geometry",
    )
    runner_geometry = _require_mapping(
        runner.get("gqa_cache_geometry"),
        label="runner GQA geometry",
    )
    if (
        gqa.get("native_gqa") is not True
        or gqa.get("mapping") != "query_head // 4"
        or geometry != runner_geometry
        or geometry.get("native_kv_head_storage") is not True
        or geometry.get("gqa_materialized") is not False
        or geometry.get("num_kv_heads") != 8
        or geometry.get("num_query_heads") != 32
    ):
        raise KIVIAdmissionError("native eight-head GQA evidence differs")
    runner_accounting = _require_mapping(
        runner.get("cache_accounting"),
        label="runner cache accounting",
    )
    runner_breakdown = _require_mapping(
        runner.get("cache_byte_breakdown"),
        label="runner cache byte breakdown",
    )
    if (
        runner_accounting.get("allocated_bytes")
        != manifest.accounting.allocated_bytes
        or runner_accounting.get("predicted_tensor_bytes")
        != manifest.accounting.predicted_allocated_bytes
        or runner_accounting.get("active_context") != expected_active
        or sum(
            value
            for value in runner_breakdown.values()
            if type(value) is int
        )
        != manifest.accounting.allocated_bytes
        or runner.get("cache_layout_fingerprint")
        != manifest.cache_layout_fingerprint
        or runner.get("output_finite") is not True
        or runner.get("cache_pointers_stable") is not True
        or runner.get("historical_cache_unchanged") is not True
        or runner.get("measurement_scope")
        != "measurement_container_admission"
        or runner.get("speedup_calculated") is not False
    ):
        raise KIVIAdmissionError("common-runner evidence differs")
    if (
        numerical.get("output_finite") is not True
        or numerical.get("decode_atol") != 0.02
        or numerical.get("decode_rtol") != 0.02
        or not isinstance(numerical.get("output_checksum"), str)
        or len(numerical["output_checksum"]) != 64
    ):
        raise KIVIAdmissionError("numerical evidence differs")
    if graph_required:
        graph = _require_mapping(
            numerical.get("graph"),
            label="point graph evidence",
        )
        comparison = _require_mapping(
            numerical.get("eager_graph_comparison"),
            label="eager/graph comparison",
        )
        if (
            graph.get("fallback") is not False
            or graph.get("consecutive_replay_outputs_exact") is not True
            or comparison.get("passed") is not True
        ):
            raise KIVIAdmissionError("point graph evidence did not pass")
    elif (
        numerical.get("graph") is not None
        or numerical.get("eager_graph_comparison") is not None
    ):
        raise KIVIAdmissionError("eager point contains graph evidence")
    expected_role = (
        "held_out_validation"
        if manifest.method_configuration == PHASE8_HELD_OUT_CONFIG
        else "mandatory"
    )
    if (
        method.get("schema_version")
        != "kvbench-phase8-kivi-method-record-1.0.0"
        or method.get("method") != "kivi"
        or method.get("method_config_id") != "kivi"
        or method.get("method_config_path") != PHASE8_METHOD_CONFIG_PATH
        or method.get("configuration") != manifest.method_configuration
        or method.get("admission_role") != expected_role
        or method.get("key_bits") != manifest.k_bits
        or method.get("value_bits") != manifest.v_bits
        or method.get("group_size") != 32
        or method.get("residual_length") != 32
        or method.get("method_config_fingerprint")
        != manifest.method_config_fingerprint
        or method.get("method_fingerprint")
        != manifest.method_fingerprint
        or method.get("adapter_version") != manifest.adapter_version
        or method.get("adapter_source_sha256")
        != manifest.adapter_source_sha256
        or method.get("cache_layout_fingerprint")
        != manifest.cache_layout_fingerprint
        or method.get("official_base_commit")
        != PHASE8_OFFICIAL_COMMIT
        or method.get("official_base_tree") != PHASE8_BASE_TREE
        or method.get("patched_tree") != PHASE8_PATCHED_TREE
        or method.get("decision_0018_patch_sha256")
        != PHASE8_DECISION_0018_PATCH_SHA256
        or method.get("extension_sha256") != PHASE8_EXTENSION_SHA256
        or method.get("fixture_root_digest")
        != PHASE8_FIXTURE_ROOT_DIGEST
        or method.get("dtype_boundary")
        != "bf16_to_fp16_official_kivi_to_bf16"
        or method.get("gqa_mapping") != "query_head // 4"
        or method.get("native_kv_head_storage") is not True
        or method.get("r_hbm") is not None
    ):
        raise KIVIAdmissionError("point method authority differs")
    _require_source_digest(
        repository_root=repository_root,
        relative_path=PHASE8_METHOD_CONFIG_PATH,
        observed_sha256=method.get("method_config_file_sha256"),
        label="KIVI method configuration",
    )
    if (
        environment.get("container_digest")
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or environment.get("execution_environment")
        != PHASE8_CONTAINER_ENVIRONMENT_VALUE
        or environment.get("git_sha") != manifest.git_sha
        or environment.get("image_mutated") is not False
        or environment.get("packages_installed") is not False
        or environment.get("network_enabled") is not False
        or environment.get("credentials_passed") is not False
    ):
        raise KIVIAdmissionError("point container authority differs")
    relative_paths = tuple(
        path.relative_to(bundle_root).as_posix()
        for path in (
            point_path,
            allocation_path,
            accounting_path,
            launcher_path,
            gqa_path,
            numerical_path,
            runner_path,
            method_path,
            environment_path,
            run_root / "manifest.json",
            run_root / "artifact_inventory.json",
            run_root / "checksums.sha256",
            run_root / "COMPLETE",
        )
    )
    return _ValidatedPointEvidence(
        manifest=manifest,
        point=point,
        first_kernel_names=first_names,
        repeated_kernel_names=repeated_names,
        relative_paths=relative_paths,
        allocation_relative_paths=allocation_evidence_paths,
    )


def _validate_bounded_grid_record(
    *,
    bundle_root: Path,
    manifests: Sequence[Phase8RunManifest],
    points: Sequence[_ValidatedPointEvidence],
) -> tuple[dict[str, Any], str]:
    path = bundle_root / "validation" / "bounded-grid.json"
    bounded = _strict_json(path, label="bounded admission grid")
    run_ids = [manifest.run_id for manifest in manifests]
    expected_plan = [
        {
            "configuration": item.configuration,
            "runner_kind": item.runner_kind.value,
            "graph_mode": item.graph_mode.value,
            "batch_size": 1,
            "context_length": item.context_length,
            "output_steps": item.output_steps,
            "engineering_samples": 1,
        }
        for item in PHASE8_ADMISSION_GRID
    ]
    observed_points = _require_list(
        bounded.get("points"),
        label="bounded grid point records",
    )
    if (
        bounded.get("schema_version") != PHASE8_BOUNDED_GRID_SCHEMA
        or bounded.get("plan") != expected_plan
        or bounded.get("run_ids") != run_ids
        or bounded.get("embedded_run_ids") != run_ids[1:]
        or bounded.get("bundle_root_point_run_id") != run_ids[0]
        or observed_points != [dict(point.point) for point in points]
        or bounded.get("attempted") != len(PHASE8_ADMISSION_GRID)
        or bounded.get("passed") != len(PHASE8_ADMISSION_GRID)
        or bounded.get("failed") != 0
        or bounded.get("speedup_calculated") is not False
        or bounded.get("performance_claim_eligible") is not False
        or bounded.get("measurement_scope")
        != "measurement_container_admission"
    ):
        raise KIVIAdmissionError(
            "bounded admission grid differs from the frozen plan"
        )
    return bounded, path.relative_to(bundle_root).as_posix()


def _validate_candidate_record(
    bundle_root: Path,
    *,
    creation_git_sha: str,
) -> str:
    path = bundle_root / "validation" / "admission-candidate.json"
    candidate = _strict_json(path, label="local admission candidate")
    independently_checked = (
        "fixture_conformance",
        "byte_accounting",
        "residual_rollover",
        "token_integrity",
        "static_cache",
        "no_measured_torch_cat",
        "direct_compressed_decode",
        "native_gqa",
        "no_unknown_allocation",
        "graph_capture_replay",
        "graph_zero_replay_allocation",
        "no_backend_fallback",
        "compute_sanitizer",
        "bounded_admission_grid",
    )
    if (
        candidate.get("schema_version") != PHASE8_CANDIDATE_SCHEMA
        or candidate.get("status")
        != "LOCAL_CHECKS_PASS_PUBLICATION_PENDING"
        or candidate.get("git_sha") != creation_git_sha
        or candidate.get("container_digest")
        != PHASE8_AUTHORIZED_CONTAINER_DIGEST
        or any(candidate.get(key) is not True for key in independently_checked)
        or candidate.get("immutable_checksums")
        != "pending_finalization"
        or candidate.get("durable_publication") != "pending_host_side"
        or candidate.get("clean_retrieval") != "pending_host_side"
        or candidate.get("g2_kivi") != "NOT_EVALUATED"
        or candidate.get("global_g2") != "NOT_EVALUATED"
        or candidate.get("quality_execution") != "LOCKED"
        or candidate.get("full_scan") != "CLOSED"
        or candidate.get("performance_data_frozen") is not False
        or candidate.get("quality_benchmark_executed") is not False
        or candidate.get("performance_claim_eligible") is not False
        or candidate.get("speedup_calculated") is not False
        or candidate.get("r_hbm") is not None
    ):
        raise KIVIAdmissionError("local admission candidate differs")
    derivation = _require_mapping(
        candidate.get("derivation"),
        label="candidate derivation",
    )
    if derivation.get("literal_gate_overrides") is not False:
        raise KIVIAdmissionError(
            "local admission candidate used literal gate overrides"
        )
    return path.relative_to(bundle_root).as_posix()


def _evidence_reference(
    *,
    evidence_id: str,
    path: Path,
    evidence_root: Path,
) -> MethodAdmissionEvidenceReference:
    resolved = _resolve_within(
        path,
        root=evidence_root,
        label=f"evidence {evidence_id}",
        require_file=True,
    )
    return MethodAdmissionEvidenceReference(
        evidence_id=evidence_id,
        path=resolved.relative_to(evidence_root).as_posix(),
        sha256=sha256_file(resolved),
    )


def _verify_evidence_references(
    *,
    evidence_root: Path,
    checks: Sequence[Phase8AdmissionCheck],
    references: Sequence[MethodAdmissionEvidenceReference],
) -> tuple[MethodAdmissionEvidenceReference, ...]:
    try:
        root = evidence_root.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise KIVIAdmissionError("evidence root is unavailable") from error
    if evidence_root.is_symlink() or not root.is_dir():
        raise KIVIAdmissionError("evidence root must be a real directory")
    refs = tuple(references)
    by_id = {reference.evidence_id: reference for reference in refs}
    if len(by_id) != len(refs):
        raise KIVIAdmissionError("evidence reference IDs are not unique")
    referenced = {
        evidence_id
        for check in checks
        for evidence_id in check.evidence_ids
    }
    if referenced != set(by_id):
        raise KIVIAdmissionError("checks and evidence references do not join")
    observed_paths: dict[str, str] = {}
    for reference in refs:
        require_relative_path(reference.path, field_name="evidence path")
        require_sha256(reference.sha256, field_name="evidence SHA-256")
        candidate = root.joinpath(*reference.path.split("/"))
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise KIVIAdmissionError(
                f"evidence path escaped or is absent: {reference.path}"
            ) from error
        if candidate.is_symlink() or not resolved.is_file():
            raise KIVIAdmissionError(
                f"evidence is not a regular file: {reference.path}"
            )
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != reference.sha256:
            raise KIVIAdmissionError(
                f"evidence SHA-256 mismatch: {reference.path}"
            )
        previous = observed_paths.setdefault(reference.path, digest)
        if previous != digest:
            raise KIVIAdmissionError(
                f"evidence path has conflicting identities: {reference.path}"
            )
    return refs


def derive_phase8_admission_evidence(
    *,
    evidence_root: Path,
    inner_bundle_root: Path,
    publication_receipt_path: Path,
    creation_git_sha: str,
) -> Phase8DerivedAdmissionEvidence:
    """Derive all 17 checks from immutable bytes, never caller verdicts."""

    try:
        require_git_sha(creation_git_sha)
    except ValueError as error:
        raise KIVIAdmissionError("creation Git SHA is invalid") from error
    root = _resolve_within(
        evidence_root,
        root=evidence_root,
        label="evidence root",
        require_file=False,
    )
    bundle = _resolve_within(
        inner_bundle_root,
        root=root,
        label="Phase 8 inner bundle",
        require_file=False,
    )
    artifact = _validate_finalized_inner_artifact(bundle)
    repository_validation = validate_run_directory(
        bundle,
        expect_final_name=True,
    )
    if (
        not repository_validation.valid
        or not repository_validation.complete
        or repository_validation.status != RunStatus.COMPLETED.value
    ):
        raise KIVIAdmissionError(
            "Phase 8 inner bundle lifecycle is invalid"
        )

    bounded_preview = _strict_json(
        bundle / "validation" / "bounded-grid.json",
        label="bounded admission grid",
    )
    run_ids = bounded_preview.get("run_ids")
    if (
        not isinstance(run_ids, list)
        or len(run_ids) != len(PHASE8_ADMISSION_GRID)
        or len(set(run_ids)) != len(PHASE8_ADMISSION_GRID)
        or any(type(run_id) is not str for run_id in run_ids)
        or bounded_preview.get("embedded_run_ids") != run_ids[1:]
    ):
        raise KIVIAdmissionError("bounded run index is invalid")

    manifests: list[Phase8RunManifest] = []
    run_roots: list[Path] = []
    for index, run_id in enumerate(run_ids):
        run_root = (
            bundle
            if index == 0
            else bundle / "grid-runs" / run_id
        )
        validation = validate_run_directory(
            run_root,
            expect_final_name=True,
        )
        if (
            not validation.valid
            or not validation.complete
            or validation.status != RunStatus.COMPLETED.value
        ):
            raise KIVIAdmissionError(
                f"bounded run immutable controls failed: {run_id}"
            )
        manifest = Phase8RunManifest.from_dict(
            _strict_json(
                run_root / "manifest.json",
                label=f"bounded run manifest {run_id}",
            )
        )
        if manifest.run_id != run_id:
            raise KIVIAdmissionError("bounded run identity differs")
        manifests.append(manifest)
        run_roots.append(run_root)
    records = require_exact_phase8_grid(manifests)
    if (
        bundle.name != records[0].run_id
        or {record.git_sha for record in records} != {creation_git_sha}
    ):
        raise KIVIAdmissionError(
            "inner bundle Git or root-run identity differs"
        )
    summarize_phase8_accounting(records)
    if (
        _phase8_git_blob_sha256(
            root,
            revision=creation_git_sha,
            relative_path=PHASE8_ADAPTER_PATH,
        )
        != records[0].adapter_source_sha256
    ):
        raise KIVIAdmissionError("KIVI adapter identity differs")
    if any(
        record.adapter_source_sha256
        != records[0].adapter_source_sha256
        for record in records
    ):
        raise KIVIAdmissionError(
            "bounded grid adapter source identities differ"
        )

    points = tuple(
        _validate_point_run(
            run_root=run_root,
            manifest=manifest,
            bundle_root=bundle,
            repository_root=root,
        )
        for run_root, manifest in zip(run_roots, records, strict=True)
    )
    _, bounded_relative = _validate_bounded_grid_record(
        bundle_root=bundle,
        manifests=records,
        points=points,
    )
    candidate_relative = _validate_candidate_record(
        bundle,
        creation_git_sha=creation_git_sha,
    )
    _, fixture_relatives = _validate_exact_test_result(
        bundle_root=bundle,
        repository_root=root,
        evidence_name="fixture-conformance",
        source_path=PHASE8_FIXTURE_TEST_PATH,
        required_test_names=(
            "test_all_four_frozen_configurations_store_append_and_rollover",
            "test_cache_position_requires_cuda_int64",
        ),
        expected_count=2,
    )
    _, graph_relatives = _validate_exact_test_result(
        bundle_root=bundle,
        repository_root=root,
        evidence_name="graph-harness",
        source_path=PHASE8_GRAPH_TEST_PATH,
        required_test_names=(
            "test_mandatory_configs_capture_direct_decode_without_replay_allocation",
        ),
        expected_count=1,
    )
    _, sanitizer_relatives = _validate_sanitizer_result(
        bundle_root=bundle,
        repository_root=root,
        execution_git_sha=creation_git_sha,
    )
    execution_audit, execution_relatives = _execution_audit_from_bundle(
        bundle_root=bundle,
        repository_root=root,
        points=points,
        execution_git_sha=creation_git_sha,
    )
    durable = _parse_publication_receipt(
        receipt_path=publication_receipt_path,
        evidence_root=root,
        inner_root_digest=artifact.root_sha256,
        inner_object_count=len(artifact.files),
        source_run_id=records[0].run_id,
        source_git_sha=creation_git_sha,
    )

    references: dict[str, MethodAdmissionEvidenceReference] = {}

    def add_reference(
        evidence_id: str,
        path: Path,
    ) -> str:
        if evidence_id in references:
            raise KIVIAdmissionError("duplicate derived evidence ID")
        references[evidence_id] = _evidence_reference(
            evidence_id=evidence_id,
            path=path,
            evidence_root=root,
        )
        return evidence_id

    bundle_relative = bundle.relative_to(root)

    def inner_path(relative: str) -> Path:
        return root / bundle_relative.joinpath(*relative.split("/"))

    fixture_ids = tuple(
        add_reference(
            f"fixture_{index}",
            inner_path(relative),
        )
        for index, relative in enumerate(fixture_relatives)
    )
    graph_ids = tuple(
        add_reference(
            f"graph_harness_{index}",
            inner_path(relative),
        )
        for index, relative in enumerate(graph_relatives)
    )
    sanitizer_ids = tuple(
        add_reference(
            f"sanitizer_{index}",
            inner_path(relative),
        )
        for index, relative in enumerate(sanitizer_relatives)
    )
    execution_ids = tuple(
        add_reference(
            f"execution_path_{index}",
            inner_path(relative),
        )
        for index, relative in enumerate(execution_relatives)
    )
    bounded_id = add_reference(
        "bounded_grid",
        inner_path(bounded_relative),
    )
    candidate_id = add_reference(
        "local_candidate",
        inner_path(candidate_relative),
    )
    point_ids: list[dict[str, str]] = []
    point_names = (
        "validation",
        "allocation",
        "accounting",
        "launcher",
        "gqa",
        "numerical",
        "runner",
        "method",
        "environment",
        "manifest",
        "inventory",
        "ledger",
        "complete",
    )
    for index, point in enumerate(points):
        ids: dict[str, str] = {}
        for name, relative in zip(
            point_names,
            point.relative_paths,
            strict=True,
        ):
            evidence_id = f"point_{index:02d}_{name}"
            ids[name] = add_reference(
                evidence_id,
                inner_path(relative),
            )
        point_ids.append(ids)
    point_allocation_raw_ids: list[tuple[str, ...]] = []
    for index, point in enumerate(points):
        raw_ids = tuple(
            add_reference(
                f"point_{index:02d}_allocation_raw_{raw_index:03d}",
                inner_path(relative),
            )
            for raw_index, relative in enumerate(
                point.allocation_relative_paths
            )
        )
        if not raw_ids:
            raise KIVIAdmissionError(
                "point raw allocation evidence references are absent"
            )
        point_allocation_raw_ids.append(raw_ids)
    publication_id = add_reference(
        "inner_publication_receipt",
        publication_receipt_path,
    )

    def ids_for(name: str) -> tuple[str, ...]:
        return tuple(item[name] for item in point_ids)

    growing_index = next(
        index
        for index, record in enumerate(records)
        if record.runner_kind is RunnerKind.GROWING_CONTEXT
    )
    graph_indices = tuple(
        index
        for index, record in enumerate(records)
        if record.graph_mode is GraphMode.CUDA_GRAPH
    )
    graph_point_ids = tuple(
        evidence_id
        for index in graph_indices
        for evidence_id in (
            point_ids[index]["allocation"],
            point_ids[index]["numerical"],
            *point_allocation_raw_ids[index],
        )
    )
    all_allocation_raw_ids = tuple(
        evidence_id
        for point_raw_ids in point_allocation_raw_ids
        for evidence_id in point_raw_ids
    )
    check_evidence = {
        "fixture_conformance": fixture_ids,
        "byte_accounting": (
            bounded_id,
            *ids_for("accounting"),
            *ids_for("runner"),
            *ids_for("manifest"),
        ),
        "residual_rollover": (
            *fixture_ids,
            *sanitizer_ids,
            point_ids[growing_index]["validation"],
        ),
        "token_integrity": (
            *fixture_ids,
            *sanitizer_ids,
            point_ids[growing_index]["validation"],
            *ids_for("numerical"),
        ),
        "static_cache": (
            *execution_ids,
            *ids_for("allocation"),
            *all_allocation_raw_ids,
        ),
        "no_measured_torch_cat": execution_ids,
        "direct_compressed_decode": (
            *execution_ids,
            *ids_for("launcher"),
        ),
        "native_gqa": (
            *execution_ids,
            *ids_for("gqa"),
        ),
        "no_unknown_allocation": (
            *ids_for("allocation"),
            *all_allocation_raw_ids,
        ),
        "graph_capture_replay": (*graph_ids, *graph_point_ids),
        "graph_zero_replay_allocation": (
            *graph_ids,
            *(
                point_ids[index]["allocation"]
                for index in graph_indices
            ),
            *(
                evidence_id
                for index in graph_indices
                for evidence_id in point_allocation_raw_ids[index]
            ),
        ),
        "no_backend_fallback": (
            *execution_ids,
            *(
                point_ids[index]["numerical"]
                for index in graph_indices
            ),
        ),
        "compute_sanitizer": sanitizer_ids,
        "bounded_admission_grid": (
            bounded_id,
            candidate_id,
            *ids_for("validation"),
            *ids_for("method"),
            *ids_for("environment"),
            *ids_for("manifest"),
        ),
        "immutable_checksums": (
            *ids_for("inventory"),
            *ids_for("ledger"),
            *ids_for("complete"),
        ),
        "durable_publication": (publication_id,),
        "clean_retrieval": (publication_id,),
    }
    checks = tuple(
        Phase8AdmissionCheck(
            check_id=check_id,
            status=GateDisposition.PASS,
            summary=f"{check_id} independently derived from finalized evidence",
            evidence_ids=check_evidence[check_id],
        )
        for check_id in PHASE8_ADMISSION_CHECK_IDS
    )
    verified_references = _verify_evidence_references(
        evidence_root=root,
        checks=checks,
        references=tuple(references.values()),
    )
    return Phase8DerivedAdmissionEvidence(
        manifests=records,
        checks=checks,
        evidence_references=verified_references,
        execution_path_audit=execution_audit,
        durable_publication=durable,
    )


def _aggregate_fingerprints(
    manifests: Sequence[Phase8RunManifest],
    field_name: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for configuration in (*PHASE8_MANDATORY_CONFIGS, PHASE8_HELD_OUT_CONFIG):
        values = [
            {
                "runner_kind": record.runner_kind.value,
                "graph_mode": record.graph_mode.value,
                "context_length": record.context_length,
                "output_steps": record.output_steps,
                "fingerprint": getattr(record, field_name),
            }
            for record in manifests
            if record.method_configuration == configuration
        ]
        payload = {
            "schema_version": (
                "kvbench-phase8-per-configuration-fingerprint-aggregate-1.0.0"
            ),
            "configuration": configuration,
            "fingerprint_field": field_name,
            "grid_values": values,
        }
        result[configuration] = sha256_hex(canonical_json_bytes(payload))
    return result


def _report_disposition(
    checks: Sequence[Phase8AdmissionCheck],
) -> GateDisposition:
    statuses = tuple(check.status for check in checks)
    if any(
        status not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }
        for status in statuses
    ):
        raise KIVIAdmissionError(
            "admission report requires terminal check dispositions"
        )
    if GateDisposition.FAIL in statuses:
        return GateDisposition.FAIL
    if GateDisposition.BLOCKED in statuses:
        return GateDisposition.BLOCKED
    if GateDisposition.PARTIAL in statuses:
        return GateDisposition.PARTIAL
    return GateDisposition.PASS


def build_phase8_method_admission_report(
    *,
    created_at_utc: str,
    creation_git_sha: str,
    evidence_root: Path,
    inner_bundle_root: Path,
    publication_receipt_path: Path,
) -> Phase8MethodAdmissionReport:
    """Build G2-KIVI only from a finalized bundle and strict R2 receipt."""

    derived = derive_phase8_admission_evidence(
        evidence_root=evidence_root,
        inner_bundle_root=inner_bundle_root,
        publication_receipt_path=publication_receipt_path,
        creation_git_sha=creation_git_sha,
    )
    records = derived.manifests
    ordered_checks = derived.checks
    references = derived.evidence_references
    execution_path_audit = derived.execution_path_audit
    durable_publication = derived.durable_publication
    by_id = {check.check_id: check for check in ordered_checks}
    execution_ids = (
        "static_cache",
        "no_measured_torch_cat",
        "direct_compressed_decode",
        "native_gqa",
        "no_backend_fallback",
    )
    execution_checks_pass = all(
        by_id[check_id].status is GateDisposition.PASS
        for check_id in execution_ids
    )
    if execution_checks_pass != execution_path_audit.passed:
        raise KIVIAdmissionError(
            "execution-path checks disagree with the derived path audit"
        )
    if by_id["byte_accounting"].status is not GateDisposition.PASS:
        raise KIVIAdmissionError(
            "valid canonical accounting cannot be reported as non-PASS"
        )
    if by_id["bounded_admission_grid"].status is not GateDisposition.PASS:
        raise KIVIAdmissionError(
            "the exact completed grid cannot be reported as non-PASS"
        )
    if (
        by_id["durable_publication"].status is GateDisposition.PASS
    ) != durable_publication.publication_passed:
        raise KIVIAdmissionError(
            "durable-publication check disagrees with its receipt"
        )
    if (
        by_id["clean_retrieval"].status is GateDisposition.PASS
    ) != durable_publication.retrieval_passed:
        raise KIVIAdmissionError(
            "clean-retrieval check disagrees with its receipt"
        )

    adapter_versions = {record.adapter_version for record in records}
    adapter_sources = {record.adapter_source_sha256 for record in records}
    creation_shas = {record.git_sha for record in records}
    if (
        adapter_versions != {PHASE8_HISTORICAL_ADAPTER_VERSION}
        or len(adapter_sources) != 1
        or creation_shas != {creation_git_sha}
    ):
        raise KIVIAdmissionError(
            "bounded grid does not share one exact Git/adapter identity"
        )
    disposition = _report_disposition(ordered_checks)
    if disposition is GateDisposition.PASS and (
        not execution_path_audit.passed
        or not durable_publication.retrieval_passed
    ):
        raise KIVIAdmissionError(
            "G2-KIVI cannot pass before local and durable evidence pass"
        )
    blockers = (
        ()
        if disposition is GateDisposition.PASS
        else tuple(
            f"{check.check_id}: {check.summary}"
            for check in ordered_checks
            if check.status is not GateDisposition.PASS
        )
    )
    return Phase8MethodAdmissionReport(
        schema_version=Phase8MethodAdmissionReport.SCHEMA_VERSION,
        created_at_utc=created_at_utc,
        status=disposition,
        mandatory_configurations=PHASE8_MANDATORY_CONFIGS,
        held_out_configuration=PHASE8_HELD_OUT_CONFIG,
        admitted_configurations=(
            PHASE8_MANDATORY_CONFIGS
            if disposition is GateDisposition.PASS
            else ()
        ),
        method_fingerprints=_aggregate_fingerprints(
            records, "method_fingerprint"
        ),
        cache_layout_fingerprints=_aggregate_fingerprints(
            records, "cache_layout_fingerprint"
        ),
        adapter_version=PHASE8_HISTORICAL_ADAPTER_VERSION,
        adapter_source_sha256=next(iter(adapter_sources)),
        official_base_commit=PHASE8_OFFICIAL_COMMIT,
        official_base_tree=PHASE8_BASE_TREE,
        patched_tree=PHASE8_PATCHED_TREE,
        decision_0018_patch_sha256=PHASE8_DECISION_0018_PATCH_SHA256,
        extension_sha256=PHASE8_EXTENSION_SHA256,
        fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
        authorized_container_digest=PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        checks=ordered_checks,
        evidence_references=references,
        gates=Phase8AdmissionGates(
            g0=GateDisposition.PASS,
            g1=GateDisposition.PASS,
            g2_tq=GateDisposition.PASS,
            g2_kivi=disposition,
            global_g2=GateDisposition.NOT_EVALUATED,
            g3=GateDisposition.NOT_EVALUATED,
            g4=GateDisposition.NOT_EVALUATED,
            g5=GateDisposition.NOT_EVALUATED,
            full_scan_state="CLOSED",
        ),
        blockers=blockers,
        local_root_digest=durable_publication.local_root_digest,
        r2_uri=durable_publication.r2_uri,
        bucket_lock_identity=durable_publication.bucket_lock_identity,
        clean_retrieval=durable_publication.retrieval_passed,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        quality_status=QualityValidationState.UNVALIDATED,
        quality_execution=QualityExecutionState.LOCKED,
        performance_claim_eligible=False,
        performance_data_frozen=False,
        quality_benchmark_executed=False,
        speedup_calculated=False,
        r_hbm=None,
        measurement_scope=(
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ),
        creation_git_sha=creation_git_sha,
    )
