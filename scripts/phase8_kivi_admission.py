#!/usr/bin/env python3
"""Build or validate the bounded local Phase 8 KIVI admission bundle.

CUDA execution is fail-closed to the exact Decision 0016 Measurement
Container.  This driver creates only local, append-only engineering evidence;
R2 publication and the G2-KIVI MethodAdmissionReport remain host-side steps.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from kvbench.adapters import build_method_adapter
from kvbench.adapters import kivi as kivi_adapter_module
from kvbench.adapters.kivi import (
    KIVI_ADAPTER_VERSION,
    KIVI_BGEMV2_HOST_STUB_OFFSET,
    KIVI_BGEMV4_HOST_STUB_OFFSET,
    KIVI_DECISION_0018_PATCH_SHA256,
    KIVI_EXTENSION_SHA256,
    KIVI_FIXTURE_ROOT_SHA256,
    KIVIMethodAdapter,
    KIVI_NEW_PACK_SHA256,
    KIVI_OFFICIAL_BASE_TREE,
    KIVI_OFFICIAL_COMMIT,
    KIVI_PATCHED_TREE,
)
from kvbench.config import load_config
from kvbench.runtime.artifacts import (
    ArtifactRun,
    phase8_artifact_store,
    sha256_file,
    validate_run_directory,
)
from kvbench.runtime.backend import forced_flash_execution
from kvbench.runtime.fixed_l_runner import run_fixed_l
from kvbench.runtime.growing_context_runner import run_growing_context
from kvbench.runtime.kivi_admission import (
    OFFICIAL_KIVI_HOST_STUB_OFFSETS,
    OFFICIAL_KIVI_KERNEL_FAMILIES,
    PHASE8_ADMISSION_GRID,
    KIVIExecutionPathAudit,
    audit_kivi_execution_path,
    derive_kivi_static_execution_precheck,
    kivi_adapter_hot_path_source,
    require_authorized_kivi_environment,
    require_exact_phase8_grid,
    summarize_phase8_accounting,
)
from kvbench.runtime.kivi_cache import (
    KIVI_CONFIG_BITS,
    KIVI_GROUP_SIZE,
    KIVI_RESIDUAL_LENGTH,
    KIVIStaticCache,
)
from kvbench.runtime.kivi_allocation import (
    KIVIAllocationBinding,
    collect_kivi_allocation_attribution,
    raw_file_sha256,
)
from kvbench.runtime.kivi_session import (
    build_kivi_endpoint_session,
    build_kivi_operation_keys,
    kivi_runtime_context,
)
from kvbench.runtime.model_loader import (
    MODEL_ID,
    MODEL_REVISION,
    load_frozen_model,
    validate_loaded_frozen_model_receipt,
)
from kvbench.runtime.numerical import tensor_sha256_untimed
from kvbench.runtime.process_supervision import run_supervised_command
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
)
from kvbench.schema import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    MethodConfig,
    MethodConfigFingerprint,
    MethodName,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
    VariantRole,
    canonical_json_bytes,
    sha256_hex,
)
from kvbench.schema.phase8 import (
    PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    PHASE8_BASE_TREE,
    PHASE8_DECISION_0018_PATCH_SHA256,
    PHASE8_EXTENSION_SHA256,
    PHASE8_FIXTURE_ROOT_DIGEST,
    PHASE8_OFFICIAL_COMMIT,
    PHASE8_PATCHED_TREE,
    RECIPROCAL_ABS_TOLERANCE,
    Phase8ByteAccounting,
    Phase8ByteBreakdown,
    Phase8RunManifest,
)
from preflight.run_preflight import sanitizer_error_count


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = Path(__file__).resolve()
ADAPTER_PATH = REPOSITORY_ROOT / "src" / "kvbench" / "adapters" / "kivi.py"
CACHE_PATH = (
    REPOSITORY_ROOT / "src" / "kvbench" / "runtime" / "kivi_cache.py"
)
ENDPOINT_PATH = (
    REPOSITORY_ROOT / "src" / "kvbench" / "runtime" / "bf16_endpoint.py"
)
METHOD_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "methods" / "kivi.yaml"
FIXTURE_TEST = REPOSITORY_ROOT / "tests" / "cuda" / "test_phase8_kivi_cuda.py"
GRAPH_TEST = REPOSITORY_ROOT / "tests" / "graph" / "test_phase8_kivi_graph.py"
SANITIZER_PROBE = (
    REPOSITORY_ROOT / "tests" / "cuda" / "phase8_kivi_sanitizer_probe.py"
)
CONTAINER_PYTHON = Path("/opt/kvbench/.venv/bin/python")
CONTAINER_PHASE3_SITE = Path("/opt/kvbench/.phase3/site-packages")
SANITIZER = Path("/usr/local/cuda-13.0/bin/compute-sanitizer")
_BITS = dict(KIVI_CONFIG_BITS)
_EXPECTED_VARIANT_ROLES = {
    "k4v4": VariantRole.MAIN,
    "k2v4": VariantRole.MAIN,
    "k2v2": VariantRole.MAIN,
    "k4v2": VariantRole.HELD_OUT,
}
_HOST_ONLY_R2_ENVIRONMENT = (
    "R2_ACCOUNT_ID",
    "R2_BUCKET",
    "R2_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "KVBENCH_R2_PREFIX",
)
_CHILD_ENVIRONMENT_PASSTHROUGH = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    "KVBENCH_KIVI_SOURCE_ROOT",
)


class Phase8KIVIDriverError(RuntimeError):
    """The bounded KIVI driver could not preserve its frozen contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Phase8KIVIDriverError("required Git identity query failed")
    return result.stdout.strip()


def _require_clean_git() -> str:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise Phase8KIVIDriverError(
            "Phase 8 admission requires a clean worktree"
        )
    head = _git("rev-parse", "HEAD")
    if len(head) != 40:
        raise Phase8KIVIDriverError("Phase 8 Git SHA is invalid")
    return head


def _load_kivi_method_config() -> MethodConfig:
    document = load_config(METHOD_CONFIG_PATH)
    if (
        type(document) is not MethodConfig
        or document.method is not MethodName.KIVI
        or document.method_config_id != "kivi"
        or document.source_lock_id != "kivi"
        or document.source_revision != KIVI_OFFICIAL_COMMIT
        or tuple(variant.variant_id for variant in document.variants)
        != tuple(_BITS)
    ):
        raise Phase8KIVIDriverError(
            "canonical KIVI method configuration authority differs"
        )
    for variant in document.variants:
        expected_bits = _BITS[variant.variant_id]
        parameters = variant.parameters
        if (
            variant.role is not _EXPECTED_VARIANT_ROLES[variant.variant_id]
            or (
                parameters.k_bits,
                parameters.v_bits,
                parameters.group_size,
                parameters.residual_length,
            )
            != (*expected_bits, KIVI_GROUP_SIZE, KIVI_RESIDUAL_LENGTH)
        ):
            raise Phase8KIVIDriverError(
                "canonical KIVI variant semantics differ"
            )
    return document


def _method_variant(configuration: str) -> tuple[MethodConfig, Any]:
    if configuration not in _BITS:
        raise ValueError("KIVI configuration is not frozen")
    config = _load_kivi_method_config()
    variant = next(
        item
        for item in config.variants
        if item.variant_id == configuration
    )
    return config, variant


def _method_config_fingerprint(configuration: str) -> str:
    config, _ = _method_variant(configuration)
    return MethodConfigFingerprint.from_config(
        config,
        configuration,
    ).sha256


def _canonical_factory_method(configuration: str) -> KIVIMethodAdapter:
    config, variant = _method_variant(configuration)
    method = build_method_adapter(
        config,
        kivi_runtime_context(),
        variant_id=variant.variant_id,
    )
    if (
        type(method) is not KIVIMethodAdapter
        or method.config_name != configuration
        or (method.k_bits, method.v_bits) != _BITS[configuration]
    ):
        raise Phase8KIVIDriverError(
            "canonical config did not resolve to the exact KIVI adapter"
        )
    return method


def _run_id(
    *,
    git_sha: str,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    context_length: int,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:-3] + "z"
    runner = "fixed" if runner_kind is RunnerKind.FIXED_L else "growing"
    graph = "graph" if graph_mode is GraphMode.CUDA_GRAPH else "eager"
    return (
        f"phase8-{stamp}-{git_sha[:8]}-{secrets.token_hex(3)}-"
        f"{configuration}-{runner}-l{context_length}-{graph}"
    )


def _phase8_byte_accounting(
    cache: Any,
    *,
    active_context: int,
) -> Phase8ByteAccounting:
    if (
        type(active_context) is not int
        or active_context <= 0
        or active_context > cache.capacity
    ):
        raise Phase8KIVIDriverError(
            "admission active context is outside physical capacity"
        )
    observed = cache.byte_breakdown()
    required = {
        "quantized_k_payload",
        "quantized_v_payload",
        "key_scales",
        "key_zero_points",
        "value_scales",
        "value_zero_points",
        "other_metadata",
        "residual_k",
        "residual_v",
        "fp16_staging",
        "quantization_staging",
        "padding_alignment",
        "persistent_workspace",
        "value_rollover_shift_scratch",
        "block_group_rounding_bytes",
    }
    if set(observed) != required:
        raise Phase8KIVIDriverError("KIVI byte categories differ")
    breakdown = Phase8ByteBreakdown(
        quantized_historical_k_payload=observed["quantized_k_payload"],
        quantized_historical_v_payload=observed["quantized_v_payload"],
        k_scales=observed["key_scales"],
        k_zeros=observed["key_zero_points"],
        v_scales=observed["value_scales"],
        v_zeros=observed["value_zero_points"],
        other_metadata=observed["other_metadata"],
        residual_k=observed["residual_k"],
        residual_v=observed["residual_v"],
        fp16_staging=observed["fp16_staging"],
        quantization_staging=observed["quantization_staging"],
        padding_alignment=observed["padding_alignment"],
        persistent_workspace=observed["persistent_workspace"],
        value_rollover_shift_scratch=observed[
            "value_rollover_shift_scratch"
        ],
        block_group_rounding=observed["block_group_rounding_bytes"],
    )
    raw = cache.accounting()
    allocated = int(raw.allocated_bytes)
    predicted = int(raw.predicted_tensor_bytes)
    logical_allocated = int(cache.logical_bf16_storage_bytes)
    rho_alloc = allocated / logical_allocated
    r_alloc = logical_allocated / allocated
    return Phase8ByteAccounting(
        capacity=cache.capacity,
        active_context=active_context,
        allocated_bytes=allocated,
        predicted_allocated_bytes=predicted,
        active_storage_bytes=int(
            cache.active_storage_bytes(active_context)
        ),
        logical_bf16_allocated_bytes=logical_allocated,
        logical_bf16_active_bytes=int(
            cache.active_logical_bf16_bytes(active_context)
        ),
        rho_alloc=rho_alloc,
        r_alloc=r_alloc,
        predicted_relative_error=abs(predicted - allocated) / allocated,
        temporary_peak_bytes=int(raw.temporary_peak_bytes),
        breakdown=breakdown,
        r_hbm=None,
    )


def _manifest(
    *,
    run_id: str,
    git_sha: str,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    context_length: int,
    output_steps: int,
    cache_layout_fingerprint: str,
    method_fingerprint: str,
    accounting: Phase8ByteAccounting,
    created_at_utc: str,
) -> Phase8RunManifest:
    capacity = context_length + output_steps
    expected_active = (
        context_length
        if runner_kind is RunnerKind.FIXED_L
        else context_length + output_steps
    )
    if (
        accounting.capacity != capacity
        or accounting.active_context != expected_active
    ):
        raise Phase8KIVIDriverError("manifest accounting context differs")
    k_bits, v_bits = _BITS[configuration]
    return Phase8RunManifest(
        schema_version=Phase8RunManifest.SCHEMA_VERSION,
        artifact_schema_version=Phase8RunManifest.ARTIFACT_SCHEMA_VERSION,
        run_id=run_id,
        status=RunStatus.CREATED,
        git_sha=git_sha,
        git_dirty=False,
        created_at_utc=created_at_utc,
        started_at_utc=None,
        finished_at_utc=None,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        method_configuration=configuration,
        k_bits=k_bits,
        v_bits=v_bits,
        method_config_fingerprint=_method_config_fingerprint(configuration),
        method_fingerprint=method_fingerprint,
        adapter_version=KIVI_ADAPTER_VERSION,
        adapter_source_sha256=sha256_file(ADAPTER_PATH),
        official_base_commit=PHASE8_OFFICIAL_COMMIT,
        official_base_tree=PHASE8_BASE_TREE,
        patched_tree=PHASE8_PATCHED_TREE,
        decision_0018_patch_sha256=PHASE8_DECISION_0018_PATCH_SHA256,
        extension_sha256=PHASE8_EXTENSION_SHA256,
        fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
        group_size=KIVI_GROUP_SIZE,
        residual_length=KIVI_RESIDUAL_LENGTH,
        dtype_boundary="bf16_to_fp16_official_kivi_to_bf16",
        cache_layout_fingerprint=cache_layout_fingerprint,
        authorized_container_digest=PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        batch_size=1,
        context_length=context_length,
        output_steps=output_steps,
        capacity=capacity,
        accounting=accounting,
        quality_status=QualityValidationState.UNVALIDATED,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        performance_claim_eligible=False,
        measurement_scope=(
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ),
        quality_execution=QualityExecutionState.LOCKED,
        quality_benchmark_executed=False,
        performance_data_frozen=False,
        speedup_calculated=False,
        full_scan_state="CLOSED",
        inventory_path=None,
        failure_reason=None,
    )


def _terminal_manifest(
    initial: Phase8RunManifest,
    *,
    started_at_utc: str,
    status: RunStatus,
    failure_reason: str | None,
) -> Phase8RunManifest:
    return dataclasses.replace(
        initial,
        status=status,
        started_at_utc=started_at_utc,
        finished_at_utc=_utc_now(),
        inventory_path="artifact_inventory.json",
        failure_reason=failure_reason,
    )


def _method_record(manifest: Phase8RunManifest) -> dict[str, Any]:
    config, variant = _method_variant(manifest.method_configuration)
    canonical_config_sha256 = sha256_hex(canonical_json_bytes(config))
    return {
        "schema_version": "kvbench-phase8-kivi-method-record-1.0.0",
        "method": "kivi",
        "method_config_id": config.method_config_id,
        "method_config_path": METHOD_CONFIG_PATH.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "method_config_file_sha256": sha256_file(METHOD_CONFIG_PATH),
        "method_config_canonical_sha256": canonical_config_sha256,
        "configuration": manifest.method_configuration,
        "variant_role": variant.role.value,
        "admission_role": (
            "held_out_validation"
            if variant.role is VariantRole.HELD_OUT
            else "mandatory"
        ),
        "key_bits": manifest.k_bits,
        "value_bits": manifest.v_bits,
        "group_size": manifest.group_size,
        "residual_length": manifest.residual_length,
        "method_config_fingerprint": manifest.method_config_fingerprint,
        "method_fingerprint": manifest.method_fingerprint,
        "adapter_version": manifest.adapter_version,
        "adapter_source_sha256": manifest.adapter_source_sha256,
        "cache_layout_fingerprint": manifest.cache_layout_fingerprint,
        "official_base_commit": manifest.official_base_commit,
        "official_base_tree": manifest.official_base_tree,
        "patched_tree": manifest.patched_tree,
        "decision_0018_patch_sha256": (
            manifest.decision_0018_patch_sha256
        ),
        "extension_sha256": manifest.extension_sha256,
        "fixture_root_digest": manifest.fixture_root_digest,
        "dtype_boundary": manifest.dtype_boundary,
        "gqa_mapping": "query_head // 4",
        "native_kv_head_storage": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "r_hbm": None,
    }


def _write_common_records(
    run: ArtifactRun,
    initial: Phase8RunManifest,
    environment: Mapping[str, Any],
) -> None:
    run.write_json("config/method.json", _method_record(initial))
    run.write_json(
        "config/point.json",
        {
            "schema_version": "kvbench-phase8-kivi-point-config-1.0.0",
            "runner_kind": initial.runner_kind.value,
            "graph_mode": initial.graph_mode.value,
            "batch_size": 1,
            "context_length": initial.context_length,
            "output_steps": initial.output_steps,
            "capacity": initial.capacity,
            "engineering_samples": 1,
            "speedup_calculated": False,
            "driver_source_path": DRIVER_PATH.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "driver_source_sha256": sha256_file(DRIVER_PATH),
        },
    )
    run.write_json(
        "environment/container_identity.json",
        {
            "schema_version": "kvbench-phase8-container-runtime-1.0.0",
            **dict(environment),
            "git_sha": initial.git_sha,
            "image_mutated": False,
            "packages_installed": False,
            "network_enabled": False,
            "credentials_passed": False,
        },
    )
    run.write_json(
        "environment/hardware_manifest.json",
        {
            "schema_version": "kvbench-phase8-live-hardware-1.0.0",
            "gpu_name": environment["gpu_name"],
            "gpu_uuid": environment["gpu_uuid"],
            "compute_capability": environment["compute_capability"],
            "cuda_runtime": environment["cuda_runtime"],
            "container_digest": environment["container_digest"],
        },
    )


def _safe_reason(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message or 'unspecified failure'}"[:500]


def _write_failure_payloads(
    run: ArtifactRun,
    *,
    stage: str,
    reason: str,
) -> None:
    if not (run.stage / "raw" / "runner.json").is_file():
        run.write_json(
            "raw/runner.json",
            {
                "status": "not_completed",
                "stage": stage,
                "reason": reason,
                "performance_claim_eligible": False,
                "speedup_calculated": False,
            },
        )
    if not (run.stage / "validation" / "point.json").is_file():
        run.write_json(
            "validation/point.json",
            {
                "passed": False,
                "stage": stage,
                "reason": reason,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "measurement_scope": "measurement_container_admission",
            },
        )


def _load_receipt_record(receipt: Any) -> dict[str, Any]:
    fields = (
        "schema_version",
        "parameter_binding_kind",
        "frozen_identity_sha256",
        "snapshot_file_ledger_sha256",
        "loader_source_sha256",
        "model_object_id",
        "tokenizer_object_id",
        "model_class_module",
        "model_class_name",
        "tokenizer_class_module",
        "tokenizer_class_name",
        "tokenizer_runtime_sha256",
        "parameter_runtime_sha256",
        "parameter_tensor_count",
        "parameter_element_count",
        "receipt_sha256",
    )
    return {name: getattr(receipt, name) for name in fields}


def _child_environment() -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in _CHILD_ENVIRONMENT_PASSTHROUGH
        if name in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(CONTAINER_PHASE3_SITE),
                    str(REPOSITORY_ROOT / "src"),
                    str(REPOSITORY_ROOT),
                )
            ),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
    )
    if any(name in environment for name in _HOST_ONLY_R2_ENVIRONMENT):
        raise Phase8KIVIDriverError(
            "host-only R2 name entered the supervised child environment"
        )
    return environment


def _supervision_evidence_passed(record: Mapping[str, Any]) -> bool:
    evidence = record.get("process_supervision")
    if not isinstance(evidence, Mapping):
        return False
    timeout = evidence.get("timeout")
    direct_child = evidence.get("direct_child")
    final_reap = evidence.get("final_reap")
    return bool(
        evidence.get("schema_version")
        == "kvbench-generic-supervised-command-result-1.0.0"
        and evidence.get("returncode") == 0
        and isinstance(timeout, Mapping)
        and timeout.get("timed_out") is False
        and isinstance(direct_child, Mapping)
        and direct_child.get("verified") is True
        and direct_child.get("parent_pid_verified") is True
        and direct_child.get("start_time_ticks_verified") is True
        and direct_child.get("process_handle_retained") is True
        and isinstance(final_reap, Mapping)
        and final_reap.get("completed") is True
        and final_reap.get("count") == 1
    )


def _run_exact_python_check(
    run: ArtifactRun,
    *,
    evidence_name: str,
    source: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    if not CONTAINER_PYTHON.is_file() or not source.is_file():
        raise Phase8KIVIDriverError(
            f"required exact-container check is unavailable: {evidence_name}"
        )
    command = (str(CONTAINER_PYTHON), str(source), "-v")
    started = _utc_now()
    monotonic_started = time.monotonic()
    result = run_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=_child_environment(),
        timeout_seconds=float(timeout_seconds),
    )
    stdout = result.stdout
    stderr = result.stderr
    exit_code = result.returncode
    timed_out = result.timed_out
    process_supervision = result.to_dict()
    combined = stdout + b"\n" + stderr
    record = {
        "process_supervision": process_supervision,
    }
    supervision_passed = _supervision_evidence_passed(record)
    passed = bool(
        not timed_out
        and exit_code == 0
        and b"OK" in combined
        and b"skipped=" not in combined
        and supervision_passed
    )
    prefix = f"validation/{evidence_name}"
    run.write_bytes(f"{prefix}/stdout.txt", stdout)
    run.write_bytes(f"{prefix}/stderr.txt", stderr)
    record = {
        "schema_version": "kvbench-phase8-exact-container-test-1.0.0",
        "evidence_name": evidence_name,
        "command": list(command),
        "source_path": source.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_sha256": sha256_file(source),
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "diagnostic_duration_seconds": float(
            time.monotonic() - monotonic_started
        ),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        "performance_timing": False,
        "process_supervision": process_supervision,
        "process_supervision_passed": supervision_passed,
        "passed": passed,
    }
    run.write_json(f"{prefix}/result.json", record)
    if not passed:
        raise Phase8KIVIDriverError(
            f"exact-container {evidence_name} check failed"
        )
    return record


def _sanitizer_tool_identity() -> dict[str, Any]:
    if not SANITIZER.is_file() or not CONTAINER_PYTHON.is_file():
        raise Phase8KIVIDriverError("locked Compute Sanitizer is unavailable")
    result = run_supervised_command(
        (str(SANITIZER), "--version"),
        working_directory=str(REPOSITORY_ROOT),
        environment=_child_environment(),
        timeout_seconds=30.0,
    )
    if result.returncode != 0:
        raise Phase8KIVIDriverError("Compute Sanitizer identity query failed")
    supervision = result.to_dict()
    if not _supervision_evidence_passed(
        {"process_supervision": supervision}
    ):
        raise Phase8KIVIDriverError(
            "Compute Sanitizer identity query was not supervised"
        )
    return {
        "path": str(SANITIZER),
        "resolved_path": str(SANITIZER.resolve(strict=True)),
        "sha256": sha256_file(SANITIZER.resolve(strict=True)),
        "version_stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "version_stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
        "version_stdout": result.stdout.decode("utf-8", errors="replace"),
        "version_stderr": result.stderr.decode("utf-8", errors="replace"),
        "exit_code": result.returncode,
        "process_supervision": supervision,
    }


def _memcheck_summaries_pass(stdout: bytes, stderr: bytes) -> bool:
    combined = stdout + b"\n" + stderr
    text = combined.decode("utf-8", errors="replace")
    return bool(
        sanitizer_error_count("memcheck", text) == 0
        and "LEAK SUMMARY: 0 bytes leaked in 0 allocations" in text
        and "ERROR SUMMARY: 0 errors" in text
    )


def _last_json_object(stdout: bytes) -> dict[str, Any] | None:
    for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _run_sanitizer(run: ArtifactRun) -> dict[str, Any]:
    tool = _sanitizer_tool_identity()
    sanitizer_environment = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    environment = _child_environment()
    environment.update(sanitizer_environment)
    command = (
        str(SANITIZER),
        "--tool",
        "memcheck",
        "--error-exitcode",
        "99",
        "--target-processes",
        "application-only",
        "--leak-check",
        "full",
        str(CONTAINER_PYTHON),
        str(SANITIZER_PROBE),
        "--image-config-digest",
        PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    )
    started = _utc_now()
    monotonic_started = time.monotonic()
    result = run_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=environment,
        timeout_seconds=900.0,
    )
    stdout = result.stdout
    stderr = result.stderr
    exit_code = result.returncode
    timed_out = result.timed_out
    process_supervision = result.to_dict()
    supervision_passed = _supervision_evidence_passed(
        {"process_supervision": process_supervision}
    )
    probe = _last_json_object(stdout)
    summaries_passed = _memcheck_summaries_pass(stdout, stderr)
    probe_passed = bool(
        probe is not None
        and probe.get("status") == "pass"
        and probe.get("kernel_families")
        == list(OFFICIAL_KIVI_KERNEL_FAMILIES)
        and {
            item.get("configuration")
            for item in probe.get("configurations", [])
            if isinstance(item, Mapping)
        }
        == {"k4v4", "k2v2"}
    )
    passed = (
        not timed_out
        and exit_code == 0
        and probe_passed
        and summaries_passed
        and supervision_passed
    )
    prefix = "validation/sanitizer"
    run.write_bytes(f"{prefix}/stdout.txt", stdout)
    run.write_bytes(f"{prefix}/stderr.txt", stderr)
    record = {
        "schema_version": "kvbench-phase8-kivi-sanitizer-result-1.0.0",
        "command": list(command),
        "probe_source_path": SANITIZER_PROBE.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "probe_source_sha256": sha256_file(SANITIZER_PROBE),
        "adapter_source_sha256": sha256_file(ADAPTER_PATH),
        "extension_sha256": PHASE8_EXTENSION_SHA256,
        "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        "tool_identity": tool,
        "environment": sanitizer_environment,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "diagnostic_duration_seconds": float(
            time.monotonic() - monotonic_started
        ),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "memcheck_summaries_passed": summaries_passed,
        "probe_passed": probe_passed,
        "rollover_covered": True,
        "kernel_families": list(OFFICIAL_KIVI_KERNEL_FAMILIES),
        "performance_timing": False,
        "process_supervision": process_supervision,
        "process_supervision_passed": supervision_passed,
        "sanitizer_descendant_coverage": {
            "direct_child": str(SANITIZER),
            "target_processes": "application-only",
            "target_executable": str(CONTAINER_PYTHON),
            "target_probe": SANITIZER_PROBE.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "bound_by_command_argv": True,
        },
        "passed": passed,
    }
    run.write_json(f"{prefix}/result.json", record)
    if not passed:
        raise Phase8KIVIDriverError("minimal KIVI sanitizer matrix failed")
    return record


def _adapter_hot_path_source() -> str:
    return kivi_adapter_hot_path_source()


def _hot_path_temporary_shapes(cache: Any) -> dict[str, tuple[int, ...]]:
    return {
        "query_staging": tuple(cache.query_fp16_staging.shape),
        "key_staging": tuple(cache.key_fp16_staging.shape),
        "value_staging": tuple(cache.value_fp16_staging.shape),
        "logits_workspace": tuple(cache.decode_logits.shape),
        "output_buffer": tuple(cache.output_buffer.shape),
    }


def _static_execution_path_precheck() -> dict[str, Any]:
    return derive_kivi_static_execution_precheck(
        adapter_source_sha256=sha256_file(ADAPTER_PATH)
    )


def _capture_launcher_probe(session: Any) -> dict[str, Any]:
    """Observe two eager post-warmup calls outside allocation and timing."""

    import torch

    first_key = session.operation_keys[0]
    step = (
        0
        if first_key.runner_kind is RunnerKind.FIXED_L
        else len(session.operation_keys) - 1
    )
    operation = (
        session._fixed_operation
        if first_key.runner_kind is RunnerKind.FIXED_L
        else session._growing_operations[step]
    )
    if operation is None:
        raise Phase8KIVIDriverError(
            "untimed eager launcher probe operation is unavailable"
        )
    launcher = session.method._runtime()[2]

    def capture_once() -> tuple[tuple[Any, ...], str, bool]:
        session.prepare_audit_step(step)
        launcher.begin_observation()
        try:
            output = operation()
            torch.cuda.synchronize(device=session.cache_device)
            cpu_output = output.detach().cpu()
        finally:
            records = launcher.end_observation()
        return (
            records,
            tensor_sha256_untimed(cpu_output),
            bool(torch.isfinite(cpu_output).all().item()),
        )

    first_records, first_digest, first_finite = capture_once()
    second_records, second_digest, second_finite = capture_once()
    first_payload = tuple(record.to_dict() for record in first_records)
    second_payload = tuple(record.to_dict() for record in second_records)
    expected_bits = set(_BITS[first_key.configuration])
    observed_bits = {
        int(record["bits"])
        for record in first_payload
    }
    records_valid = bool(
        first_payload
        and all(
            record["kernel_family"]
            == f"bgemv{record['bits']}_kernel_outer_dim"
            and record["group_size"] == KIVI_GROUP_SIZE
            and record["num_query_heads"] == 32
            and record["num_kv_heads"] == 8
            for record in first_payload
        )
    )
    passed = bool(
        first_payload == second_payload
        and first_digest == second_digest
        and first_finite
        and second_finite
        and observed_bits == expected_bits
        and records_valid
    )
    return {
        "schema_version": (
            "kvbench-phase8-kivi-runtime-launch-observation-1.0.0"
        ),
        "passed": passed,
        "configuration": first_key.configuration,
        "runner_kind": first_key.runner_kind.value,
        "graph_mode": first_key.graph_mode.value,
        "observed_step": step,
        "first_sequence": list(first_payload),
        "second_sequence": list(second_payload),
        "first_output_sha256": first_digest,
        "second_output_sha256": second_digest,
        "first_output_finite": first_finite,
        "second_output_finite": second_finite,
        "stable_post_warmup_sequence": first_payload == second_payload,
        "expected_bits": sorted(expected_bits),
        "observed_bits": sorted(observed_bits),
        "instrumented_audit_separate": True,
        "allocation_audit_instrumented": False,
        "normal_timing_instrumented": False,
        "host_synchronization_outside_hot_path": True,
    }


def _execution_path_audit(
    *,
    temporary_shapes: Mapping[str, Sequence[int]],
    launcher_probes: Sequence[Mapping[str, Any]],
    backend_fallback_observed: bool,
    cache_growth_observed: bool,
) -> KIVIExecutionPathAudit:
    source = _adapter_hot_path_source()
    first_kernels = tuple(
        str(record["kernel_family"])
        for probe in launcher_probes
        for record in probe["first_sequence"]
    )
    second_kernels = tuple(
        str(record["kernel_family"])
        for probe in launcher_probes
        for record in probe["second_sequence"]
    )
    if (
        not first_kernels
        or first_kernels != second_kernels
        or not all(probe.get("passed") is True for probe in launcher_probes)
    ):
        raise Phase8KIVIDriverError(
            "runtime launcher observations are absent or unstable"
        )
    return audit_kivi_execution_path(
        kernel_names=first_kernels,
        repeated_kernel_names=second_kernels,
        runtime_event_names=("cudaLaunchKernel",),
        temporary_shapes=temporary_shapes,
        adapter_hot_path_source=source,
        observed_extension_sha256=KIVI_EXTENSION_SHA256,
        observed_new_pack_sha256=KIVI_NEW_PACK_SHA256,
        official_commit=KIVI_OFFICIAL_COMMIT,
        official_base_tree=KIVI_OFFICIAL_BASE_TREE,
        patched_tree=KIVI_PATCHED_TREE,
        decision_0018_patch_sha256=KIVI_DECISION_0018_PATCH_SHA256,
        fixture_root_digest=KIVI_FIXTURE_ROOT_SHA256,
        host_stub_offsets={
            2: KIVI_BGEMV2_HOST_STUB_OFFSET,
            4: KIVI_BGEMV4_HOST_STUB_OFFSET,
        },
        backend_fallback_observed=backend_fallback_observed,
        cache_growth_observed=cache_growth_observed,
    )


def _deterministic_ids(
    torch: Any,
    *,
    length: int,
    offset: int,
    device: Any,
) -> Any:
    values = torch.arange(length, dtype=torch.long, device=device).reshape(
        1, length
    )
    return (values + offset) % 120_000 + 1_000


def _audit_session(
    session: Any,
) -> tuple[
    list[tuple[str, bool]],
    dict[str, Any],
    dict[str, bytes],
]:
    import torch

    outputs: list[tuple[str, bool]] = []
    allocations: list[dict[str, Any]] = []
    evidence_files: dict[str, bytes] = {}
    pointers_before = session.current_cache_pointers()
    for step in range(len(session.operation_keys)):
        session.prepare_audit_step(step)
        output = session.execute_audit_step(step)
        torch.cuda.synchronize(device=session.cache_device)
        cpu_output = output.detach().cpu()
        outputs.append(
            (
                tensor_sha256_untimed(cpu_output),
                bool(torch.isfinite(cpu_output).all().item()),
            )
        )
        operation_key = session.operation_keys[step]
        binding = KIVIAllocationBinding(
            configuration=operation_key.configuration,
            runner_kind=operation_key.runner_kind.value,
            graph_mode=operation_key.graph_mode.value,
            historical_context=operation_key.historical_context,
            attended_context=operation_key.attended_context,
            operation_fingerprint_sha256=(
                operation_key.operation_fingerprint_sha256
            ),
            cache_layout_fingerprint=(
                session.cache_layout_fingerprint()
            ),
            method_fingerprint=session.adapter_config_fingerprint,
            backend_identity=(
                session.method.runtime_context.backend_fingerprint
            ),
            adapter_source_sha256=sha256_file(ADAPTER_PATH),
            cache_source_sha256=sha256_file(CACHE_PATH),
            endpoint_source_sha256=sha256_file(ENDPOINT_PATH),
            authorized_container_digest=(
                PHASE8_AUTHORIZED_CONTAINER_DIGEST
            ),
            official_commit=KIVI_OFFICIAL_COMMIT,
            patched_tree=KIVI_PATCHED_TREE,
            decision_0018_patch_sha256=(
                KIVI_DECISION_0018_PATCH_SHA256
            ),
            extension_sha256=KIVI_EXTENSION_SHA256,
        )

        def capture_state() -> dict[str, Any]:
            pointer_digest = sha256_hex(
                canonical_json_bytes(
                    session.current_cache_pointers()
                )
            )
            return {
                "cache_pointers_sha256": pointer_digest,
                "active_context": session.active_context,
            }

        def capture_output(value: Any) -> dict[str, Any]:
            cpu_value = value.detach().cpu()
            return {
                "sha256": tensor_sha256_untimed(cpu_value),
                "finite": bool(torch.isfinite(cpu_value).all().item()),
            }

        attributed = collect_kivi_allocation_attribution(
            lambda step=step: session.execute_audit_step(step),
            prepare_operation=(
                lambda step=step: session.prepare_audit_step(step)
            ),
            capture_state=capture_state,
            capture_output=capture_output,
            binding=binding,
            device=session.cache_device,
        )
        allocation = copy.deepcopy(dict(attributed.summary))
        evidence_root = f"allocation/operations/step-{step:04d}"
        allocation["raw_evidence_root"] = evidence_root
        allocation["raw_evidence_sha256"] = {
            name: raw_file_sha256(payload)
            for name, payload in sorted(attributed.files.items())
        }
        allocations.append(allocation)
        for name, payload in attributed.files.items():
            relative = f"{evidence_root}/{name}"
            if relative in evidence_files:
                raise Phase8KIVIDriverError(
                    "allocator raw evidence path is duplicated"
                )
            evidence_files[relative] = payload
    graph_required = (
        session.operation_keys[0].graph_mode is GraphMode.CUDA_GRAPH
    )
    graph_passed = (
        not graph_required
        or (
            session.graph is not None
            and session.graph_evidence is not None
            and session.graph_evidence.get("fallback") is False
            and session.graph_evidence.get(
                "consecutive_replay_outputs_exact"
            )
            is True
            and session.eager_graph_comparison is not None
            and session.eager_graph_comparison.passed
        )
    )
    allocation_passed = all(
        item["criterion"]["passed"] for item in allocations
    )
    pointers_stable = pointers_before == session.current_cache_pointers()
    return outputs, {
        "passed": (
            allocation_passed
            and graph_passed
            and pointers_stable
            and all(finite for _, finite in outputs)
        ),
        "operation_allocations": allocations,
        "unknown_allocation_count": 0 if allocation_passed else None,
        "all_ephemeral_allocations_attributed": (
            allocation_passed if not graph_required else None
        ),
        "graph_required": graph_required,
        "graph_passed": graph_passed,
        "pointers_stable": pointers_stable,
        "outputs": [
            {"sha256": digest, "finite": finite}
            for digest, finite in outputs
        ],
    }, evidence_files


def _normalize_runner_scope(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only the known legacy nested timing scope, fail closed."""

    normalized = copy.deepcopy(dict(result))
    timing = normalized.get("timing")
    if (
        normalized.get("measurement_scope")
        != MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION.value
        or normalized.get("performance_claim_eligible") is not False
        or not isinstance(timing, dict)
        or timing.get("sample_count") != 1
        or not isinstance(timing.get("samples"), list)
        or len(timing["samples"]) != 1
        or timing.get("measurement_scope")
        != MeasurementScope.NATIVE_HOST_ADMISSION.value
        or timing.get("paper_claim_eligible") is not False
    ):
        raise Phase8KIVIDriverError(
            "common runner measurement-scope evidence differs"
        )
    timing["measurement_scope"] = (
        MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION.value
    )
    timing["scope_normalization"] = {
        "legacy_value_observed": (
            MeasurementScope.NATIVE_HOST_ADMISSION.value
        ),
        "canonical_value": (
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION.value
        ),
        "timing_semantics_changed": False,
    }
    normalized["speedup_calculated"] = False
    normalized["quality_status"] = "unvalidated"
    return normalized


def _run_common_runner(
    session: Any,
    runner_kind: RunnerKind,
) -> dict[str, Any]:
    if runner_kind is RunnerKind.FIXED_L:
        raw = run_fixed_l(
            session,
            measured_steps=1,
            measured_batches=1,
        ).to_dict()
    elif runner_kind is RunnerKind.GROWING_CONTEXT:
        raw = run_growing_context(session, expected_steps=4).to_dict()
    else:
        raise Phase8KIVIDriverError("runner kind is outside Phase 8")
    return _normalize_runner_scope(raw)


def _execute_point(
    *,
    loaded: Any,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    context_length: int,
    output_steps: int,
    global_audits_passed: bool,
) -> tuple[
    Any,
    dict[str, Any],
    dict[str, Any],
    Phase8ByteAccounting,
    dict[str, bytes],
]:
    import torch

    canonical_method = _canonical_factory_method(configuration)
    del canonical_method
    keys = build_kivi_operation_keys(
        configuration=configuration,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        starting_context=context_length,
        output_steps=output_steps,
    )
    offset = (
        10_000
        + list(_BITS).index(configuration) * 20_000
        + context_length
    )
    prefix = _deterministic_ids(
        torch,
        length=context_length,
        offset=offset,
        device=torch.device("cuda:0"),
    )
    decode = _deterministic_ids(
        torch,
        length=output_steps,
        offset=offset + context_length + 257,
        device=torch.device("cuda:0"),
    )
    with forced_flash_execution():
        session = build_kivi_endpoint_session(
            loaded=loaded,
            operation_keys=keys,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
        launcher_probe = _capture_launcher_probe(session)
        observed, audit, allocation_evidence_files = _audit_session(
            session
        )
        session.admit(
            observed_outputs=observed,
            execution_path_passed=(
                global_audits_passed and launcher_probe["passed"]
            ),
            allocation_passed=audit["passed"],
            graph_passed=audit["graph_passed"],
        )
    runner_result = _run_common_runner(session, runner_kind)
    accounting = _phase8_byte_accounting(
        session.cache,
        active_context=(
            context_length
            if runner_kind is RunnerKind.FIXED_L
            else context_length + output_steps
        ),
    )
    runtime_accounting = runner_result["cache_accounting"]
    breakdown_sum = sum(
        int(value)
        for value in runner_result["cache_byte_breakdown"].values()
    )
    ratios = session.method_allocation_ratios()
    point = {
        "schema_version": "kvbench-phase8-kivi-point-validation-1.0.0",
        "configuration": configuration,
        "runner_kind": runner_kind.value,
        "graph_mode": graph_mode.value,
        "batch_size": 1,
        "context_length": context_length,
        "output_steps": output_steps,
        "capacity": context_length + output_steps,
        "allocation": audit,
        "launcher_probe": launcher_probe,
        "accounting": accounting.to_dict(),
        "runtime_committed_context": int(runtime_accounting["active_context"]),
        "byte_breakdown_sum": breakdown_sum,
        "allocated_bytes": accounting.allocated_bytes,
        "rho_alloc": accounting.rho_alloc,
        "r_alloc": accounting.r_alloc,
        "reciprocal_product_error": abs(
            accounting.rho_alloc * accounting.r_alloc - 1.0
        ),
        "output_finite": runner_result["output_finite"],
        "cache_pointers_stable": runner_result["cache_pointers_stable"],
        "historical_cache_unchanged": runner_result[
            "historical_cache_unchanged"
        ],
        "native_gqa": (
            runner_result["gqa_cache_geometry"]["native_kv_head_storage"]
            and not runner_result["gqa_cache_geometry"]["gqa_materialized"]
        ),
        "rollover_active_lengths": runner_result.get("active_lengths"),
        "speedup_calculated": False,
        "r_hbm": None,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
    }
    point["passed"] = all(
        (
            global_audits_passed,
            launcher_probe["passed"],
            audit["passed"],
            accounting.predicted_relative_error < 0.01,
            breakdown_sum == accounting.allocated_bytes,
            int(runtime_accounting["allocated_bytes"])
            == accounting.allocated_bytes,
            int(runtime_accounting["predicted_tensor_bytes"])
            == accounting.predicted_allocated_bytes,
            int(runtime_accounting["active_context"])
            == accounting.active_context,
            runner_result["cache_layout_fingerprint"]
            == session.cache_layout_fingerprint(),
            point["output_finite"],
            point["cache_pointers_stable"],
            point["historical_cache_unchanged"],
            point["native_gqa"],
            point["reciprocal_product_error"]
            <= RECIPROCAL_ABS_TOLERANCE,
            ratios["rho_alloc"] == accounting.rho_alloc,
            ratios["r_alloc"] == accounting.r_alloc,
            runner_result["measurement_scope"]
            == "measurement_container_admission",
            runner_result["timing"]["measurement_scope"]
            == "measurement_container_admission",
            (
                runner_kind is not RunnerKind.GROWING_CONTEXT
                or runner_result.get("active_lengths") == [31, 32, 33, 34]
            ),
        )
    )
    if point["passed"] is not True:
        raise Phase8KIVIDriverError("bounded KIVI admission point failed")
    return (
        session,
        runner_result,
        point,
        accounting,
        allocation_evidence_files,
    )


def _create_started_run(
    *,
    store: Any,
    git_sha: str,
    environment: Mapping[str, Any],
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    context_length: int,
    output_steps: int,
    cache_layout_fingerprint: str,
    method_fingerprint: str,
    accounting: Phase8ByteAccounting,
) -> tuple[ArtifactRun, Phase8RunManifest, str]:
    created = _utc_now()
    identifier = _run_id(
        git_sha=git_sha,
        configuration=configuration,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        context_length=context_length,
    )
    initial = _manifest(
        run_id=identifier,
        git_sha=git_sha,
        configuration=configuration,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        context_length=context_length,
        output_steps=output_steps,
        cache_layout_fingerprint=cache_layout_fingerprint,
        method_fingerprint=method_fingerprint,
        accounting=accounting,
        created_at_utc=created,
    )
    run = store.create(identifier, initial)
    run.start()
    started = _utc_now()
    _write_common_records(run, initial, environment)
    return run, initial, started


def _write_point_records(
    run: ArtifactRun,
    *,
    runner_result: Mapping[str, Any],
    point: Mapping[str, Any],
    allocation_evidence_files: Mapping[str, bytes],
) -> None:
    if not allocation_evidence_files:
        raise Phase8KIVIDriverError(
            "point lacks raw allocation attribution evidence"
        )
    for relative, payload in sorted(allocation_evidence_files.items()):
        run.write_bytes(relative, payload)
    run.write_json("raw/runner.json", runner_result)
    run.write_json("validation/point.json", point)
    run.write_json("allocation/full-model.json", dict(point["allocation"]))
    run.write_json(
        "execution-path/launcher-observation.json",
        dict(point["launcher_probe"]),
    )
    run.write_json("accounting/bytes.json", dict(point["accounting"]))
    run.write_json(
        "gqa/full-model.json",
        {
            "native_gqa": point["native_gqa"],
            "mapping": "query_head // 4",
            "geometry": runner_result["gqa_cache_geometry"],
        },
    )
    run.write_json(
        "numerical/output.json",
        {
            "output_checksum": runner_result["output_checksum"],
            "output_finite": runner_result["output_finite"],
            "graph": runner_result.get("graph"),
            "eager_graph_comparison": runner_result.get(
                "eager_graph_comparison"
            ),
            "decode_atol": 0.02,
            "decode_rtol": 0.02,
        },
    )


def _release_cuda_objects() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _artifact_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise Phase8KIVIDriverError(
                "Phase 8 artifact contains a symlink"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise Phase8KIVIDriverError(
                "Phase 8 artifact contains an unsafe entry"
            )
        files[candidate.relative_to(root).as_posix()] = candidate
    return files


def _embed_completed_run(bundle: ArtifactRun, completed: Path) -> None:
    validation = validate_run_directory(completed)
    if not validation.valid or not validation.complete:
        raise Phase8KIVIDriverError(
            "point run is not complete before embedding"
        )
    if completed.name == bundle.run_id:
        raise Phase8KIVIDriverError("bundle cannot recursively embed itself")
    for relative, source in _artifact_files(completed).items():
        bundle.write_bytes(
            f"grid-runs/{completed.name}/{relative}",
            source.read_bytes(),
        )


def _embedded_run_matches(
    bundle_root: Path,
    run_id: str,
    completed: Path,
) -> bool:
    embedded = bundle_root / "grid-runs" / run_id
    if not embedded.is_dir() or embedded.is_symlink():
        return False
    try:
        embedded_files = _artifact_files(embedded)
        completed_files = _artifact_files(completed)
    except Phase8KIVIDriverError:
        return False
    if set(embedded_files) != set(completed_files):
        return False
    return all(
        source.stat().st_size == embedded_files[relative].stat().st_size
        and sha256_file(source) == sha256_file(embedded_files[relative])
        for relative, source in completed_files.items()
    )


def validate_local_admission(bundle_path: Path) -> dict[str, Any]:
    """Validate one explicitly selected bundle without discarding history."""

    if bundle_path.is_symlink():
        raise Phase8KIVIDriverError(
            "selected Phase 8 admission bundle is unsafe"
        )
    try:
        bundle_root = bundle_path.resolve(strict=True)
    except OSError as error:
        raise Phase8KIVIDriverError(
            "selected Phase 8 admission bundle is absent"
        ) from error
    artifact_root = bundle_root.parent
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise Phase8KIVIDriverError("Phase 8 artifact root is absent")
    records: dict[str, Phase8RunManifest] = {}
    completed_bundle_ids: list[str] = []
    historical_failed_ids: list[str] = []
    for path in sorted(artifact_root.iterdir()):
        if path.name.startswith("."):
            continue
        if not path.is_dir() or path.is_symlink():
            raise Phase8KIVIDriverError("Phase 8 artifact root is unsafe")
        validation = validate_run_directory(path)
        if not validation.valid or not validation.complete:
            raise Phase8KIVIDriverError(
                f"Phase 8 run validation failed: {path.name}"
            )
        manifest = Phase8RunManifest.from_dict(
            json.loads(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
        )
        if manifest.run_id != path.name:
            raise Phase8KIVIDriverError("Phase 8 run identity differs")
        records[manifest.run_id] = manifest
        if manifest.status is not RunStatus.COMPLETED:
            historical_failed_ids.append(manifest.run_id)
        elif (path / "validation" / "bounded-grid.json").is_file():
            completed_bundle_ids.append(manifest.run_id)
    if (
        bundle_root.parent != artifact_root
        or bundle_root.name not in records
        or records[bundle_root.name].status is not RunStatus.COMPLETED
        or bundle_root.name not in completed_bundle_ids
    ):
        raise Phase8KIVIDriverError(
            "selected Phase 8 admission bundle is not completed"
        )
    bounded = json.loads(
        (bundle_root / "validation" / "bounded-grid.json").read_text(
            encoding="utf-8"
        )
    )
    run_ids = bounded.get("run_ids")
    embedded_ids = bounded.get("embedded_run_ids")
    if (
        not isinstance(run_ids, list)
        or len(run_ids) != len(PHASE8_ADMISSION_GRID)
        or len(set(run_ids)) != len(PHASE8_ADMISSION_GRID)
        or embedded_ids != run_ids[1:]
    ):
        raise Phase8KIVIDriverError("Phase 8 bundle run index is invalid")
    manifests: list[Phase8RunManifest] = []
    for run_id in run_ids:
        manifest = records.get(run_id)
        if manifest is None or manifest.status is not RunStatus.COMPLETED:
            raise Phase8KIVIDriverError(
                "Phase 8 bundle references an incomplete run"
            )
        manifests.append(manifest)
    require_exact_phase8_grid(manifests)
    if any(
        not _embedded_run_matches(
            bundle_root, run_id, artifact_root / run_id
        )
        for run_id in embedded_ids
    ):
        raise Phase8KIVIDriverError(
            "embedded Phase 8 point differs from its source"
        )
    accounting = summarize_phase8_accounting(manifests)
    candidate = json.loads(
        (bundle_root / "validation" / "admission-candidate.json").read_text(
            encoding="utf-8"
        )
    )
    if candidate.get("status") != "LOCAL_CHECKS_PASS_PUBLICATION_PENDING":
        raise Phase8KIVIDriverError("local admission candidate differs")
    selected_ids = set(run_ids)
    historical_completed_ids = sorted(
        run_id
        for run_id, manifest in records.items()
        if manifest.status is RunStatus.COMPLETED
        and run_id not in selected_ids
        and run_id not in completed_bundle_ids
    )
    return {
        "status": "PASS",
        "scope": "phase8_kivi_local_inner_bundle",
        "artifact_root": str(artifact_root),
        "bundle_run_id": bundle_root.name,
        "bundle_path": str(bundle_root),
        "run_count": len(manifests),
        "run_ids": run_ids,
        "historical_completed_bundle_run_ids": sorted(
            run_id
            for run_id in completed_bundle_ids
            if run_id != bundle_root.name
        ),
        "historical_completed_run_ids": historical_completed_ids,
        "historical_failed_run_ids": sorted(historical_failed_ids),
        "selection_kind": "explicit_bundle_path",
        "bounded_grid_exact": True,
        "complete_marker_valid": True,
        "inventory_valid": True,
        "checksum_valid": True,
        "accounting_summary_sha256": accounting["summary_sha256"],
        "durable_publication": "PENDING_HOST_SIDE",
        "g2_kivi": "NOT_EVALUATED_PUBLICATION_PENDING",
        "performance_claim_eligible": False,
        "speedup_calculated": False,
    }


def _preallocate_first_identity() -> tuple[
    Any, Any, str, str, Phase8ByteAccounting
]:
    first = PHASE8_ADMISSION_GRID[0]
    method = _canonical_factory_method(first.configuration)
    method.prepare_runtime()
    cache = method.allocate(
        batch_size=1,
        capacity=first.context_length + first.output_steps,
        device="cuda:0",
        workspace_bytes=0,
    )
    cache.initialize_deterministic()
    layout = cache.layout_fingerprint()
    fingerprint = method.config_fingerprint(layout)
    accounting = _phase8_byte_accounting(
        cache,
        active_context=first.context_length,
    )
    return method, cache, layout, fingerprint, accounting


def _derive_local_candidate(
    *,
    git_sha: str,
    fixture: Mapping[str, Any],
    graph: Mapping[str, Any],
    sanitizer: Mapping[str, Any],
    execution_path: KIVIExecutionPathAudit,
    point_records: Sequence[Mapping[str, Any]],
    run_ids: Sequence[str],
) -> dict[str, Any]:
    points = tuple(point_records)
    allocation_operations = tuple(
        operation
        for point in points
        for operation in point["allocation"]["operation_allocations"]
    )
    graph_points = tuple(
        point for point in points if point["graph_mode"] == "cuda_graph"
    )
    growing_points = tuple(
        point for point in points if point["runner_kind"] == "growing_context"
    )
    observed_grid = tuple(
        (
            point["configuration"],
            point["runner_kind"],
            point["graph_mode"],
            point["context_length"],
            point["output_steps"],
        )
        for point in points
    )
    expected_grid = tuple(
        (
            item.configuration,
            item.runner_kind.value,
            item.graph_mode.value,
            item.context_length,
            item.output_steps,
        )
        for item in PHASE8_ADMISSION_GRID
    )
    fixture_conformance = fixture.get("passed") is True
    sanitizer_passed = sanitizer.get("passed") is True
    byte_accounting = bool(
        points
        and all(
            point["accounting"]["predicted_relative_error"] < 0.01
            and point["byte_breakdown_sum"] == point["allocated_bytes"]
            and point["reciprocal_product_error"]
            <= RECIPROCAL_ABS_TOLERANCE
            and point["r_hbm"] is None
            for point in points
        )
    )
    residual_rollover = bool(
        len(growing_points) == 1
        and growing_points[0]["rollover_active_lengths"]
        == [31, 32, 33, 34]
        and sanitizer.get("rollover_covered") is True
    )
    token_integrity = bool(
        fixture_conformance
        and sanitizer.get("probe_passed") is True
        and residual_rollover
    )
    static_cache = bool(
        execution_path.passed
        and not execution_path.cache_growth_detected
        and all(point["cache_pointers_stable"] for point in points)
    )
    no_unknown_allocation = bool(
        allocation_operations
        and all(
            operation["criterion"]["passed"]
            and operation["criterion"]["unknown_allocation_count"] == 0
            and operation["criterion"]["persistent_allocated_delta"] == 0
            and operation["criterion"]["persistent_reserved_delta"] == 0
            for operation in allocation_operations
        )
    )
    graph_capture_replay = bool(
        graph.get("passed") is True
        and graph_points
        and all(
            point["allocation"]["graph_passed"] is True
            for point in graph_points
        )
    )
    graph_zero_replay_allocation = bool(
        graph_capture_replay
        and all(
            operation["criterion"]["strict_graph_zero_events"] is True
            for point in graph_points
            for operation in point["allocation"]["operation_allocations"]
        )
    )
    bounded_grid = bool(
        observed_grid == expected_grid
        and len(run_ids) == len(PHASE8_ADMISSION_GRID)
        and len(set(run_ids)) == len(PHASE8_ADMISSION_GRID)
        and all(point["passed"] is True for point in points)
    )
    child_process_supervision = all(
        _supervision_evidence_passed(record)
        for record in (fixture, graph, sanitizer)
    )
    local_checks = {
        "fixture_conformance": fixture_conformance,
        "byte_accounting": byte_accounting,
        "residual_rollover": residual_rollover,
        "token_integrity": token_integrity,
        "static_cache": static_cache,
        "no_measured_torch_cat": bool(
            execution_path.passed
            and not execution_path.measured_torch_cat_detected
        ),
        "direct_compressed_decode": bool(
            execution_path.passed
            and execution_path.two_bit_kernel_verified
            and execution_path.four_bit_kernel_verified
            and not execution_path.full_prefix_dequantization_detected
            and not execution_path.full_prefix_temporary_detected
        ),
        "native_gqa": bool(
            execution_path.native_gqa_indexing_verified
            and not execution_path.gqa_materialization_detected
            and not execution_path.query_head_sized_kv_temporary_detected
            and all(point["native_gqa"] for point in points)
        ),
        "no_unknown_allocation": no_unknown_allocation,
        "graph_capture_replay": graph_capture_replay,
        "graph_zero_replay_allocation": graph_zero_replay_allocation,
        "no_backend_fallback": bool(
            execution_path.passed
            and not execution_path.backend_fallback_detected
        ),
        "compute_sanitizer": sanitizer_passed,
        "child_process_supervision": child_process_supervision,
        "bounded_admission_grid": bounded_grid,
    }
    local_passed = all(local_checks.values())
    return {
        "schema_version": (
            "kvbench-phase8-kivi-admission-candidate-1.0.0"
        ),
        "status": (
            "LOCAL_CHECKS_PASS_PUBLICATION_PENDING"
            if local_passed
            else "LOCAL_CHECKS_FAILED"
        ),
        "git_sha": git_sha,
        "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        **local_checks,
        "immutable_checksums": "pending_finalization",
        "durable_publication": "pending_host_side",
        "clean_retrieval": "pending_host_side",
        "g2_kivi": "NOT_EVALUATED",
        "global_g2": "NOT_EVALUATED",
        "quality_execution": "LOCKED",
        "full_scan": "CLOSED",
        "performance_data_frozen": False,
        "quality_benchmark_executed": False,
        "performance_claim_eligible": False,
        "speedup_calculated": False,
        "r_hbm": None,
        "derivation": {
            "fixture_record_passed": fixture_conformance,
            "graph_record_passed": graph.get("passed") is True,
            "sanitizer_record_passed": sanitizer_passed,
            "child_process_supervision_passed": child_process_supervision,
            "execution_path_record_passed": execution_path.passed,
            "point_record_count": len(points),
            "allocation_operation_count": len(allocation_operations),
            "run_id_count": len(run_ids),
            "literal_gate_overrides": False,
        },
    }


def run_admission() -> dict[str, Any]:
    git_sha = _require_clean_git()
    environment = require_authorized_kivi_environment(
        PHASE8_AUTHORIZED_CONTAINER_DIGEST
    )
    store = phase8_artifact_store(REPOSITORY_ROOT)
    first = PHASE8_ADMISSION_GRID[0]
    pre_method, pre_cache, layout, fingerprint, accounting = (
        _preallocate_first_identity()
    )
    temporary_shapes = _hot_path_temporary_shapes(pre_cache)
    bundle_run, bundle_initial, bundle_started = _create_started_run(
        store=store,
        git_sha=git_sha,
        environment=environment,
        configuration=first.configuration,
        runner_kind=first.runner_kind,
        graph_mode=first.graph_mode,
        context_length=first.context_length,
        output_steps=first.output_steps,
        cache_layout_fingerprint=layout,
        method_fingerprint=fingerprint,
        accounting=accounting,
    )
    stage = "fixture_conformance"
    active_run: ArtifactRun | None = None
    active_initial: Phase8RunManifest | None = None
    active_started: str | None = None
    run_ids: list[str] = []
    point_records: list[dict[str, Any]] = []
    embedded_ids: list[str] = []
    try:
        static_path = _static_execution_path_precheck()
        bundle_run.write_json(
            "validation/execution-path-static-precheck.json",
            static_path,
        )
        if static_path["passed"] is not True:
            raise Phase8KIVIDriverError(
                "KIVI static execution-path precheck failed"
            )
        fixture = _run_exact_python_check(
            bundle_run,
            evidence_name="fixture-conformance",
            source=FIXTURE_TEST,
        )
        stage = "graph_harness"
        graph = _run_exact_python_check(
            bundle_run,
            evidence_name="graph-harness",
            source=GRAPH_TEST,
        )
        stage = "compute_sanitizer"
        sanitizer = _run_sanitizer(bundle_run)

        pre_cache = None
        pre_method = None
        _release_cuda_objects()
        stage = "model_load"
        loaded = load_frozen_model()
        validate_loaded_frozen_model_receipt(loaded)
        model_fingerprint = sha256_hex(
            canonical_json_bytes(loaded.identity.to_dict())
        )
        bundle_run.write_json(
            "validation/model_identity.json",
            {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "fingerprint": model_fingerprint,
                "identity": loaded.identity.to_dict(),
                "load_receipt": _load_receipt_record(loaded.receipt),
            },
        )
        global_audits_passed = bool(
            fixture["passed"]
            and graph["passed"]
            and sanitizer["passed"]
            and static_path["passed"]
        )
        for index, point_spec in enumerate(PHASE8_ADMISSION_GRID):
            stage = f"bounded_grid_{index}"
            (
                session,
                runner_result,
                point,
                point_accounting,
                allocation_evidence_files,
            ) = _execute_point(
                loaded=loaded,
                configuration=point_spec.configuration,
                runner_kind=point_spec.runner_kind,
                graph_mode=point_spec.graph_mode,
                context_length=point_spec.context_length,
                output_steps=point_spec.output_steps,
                global_audits_passed=global_audits_passed,
            )
            if index == 0:
                run = bundle_run
                initial = bundle_initial
                started = bundle_started
                if (
                    session.cache_layout_fingerprint()
                    != initial.cache_layout_fingerprint
                    or session.adapter_config_fingerprint
                    != initial.method_fingerprint
                    or point_accounting != initial.accounting
                ):
                    raise Phase8KIVIDriverError(
                        "bundle identity differs from the warmed first point"
                    )
            else:
                run, initial, started = _create_started_run(
                    store=store,
                    git_sha=git_sha,
                    environment=environment,
                    configuration=point_spec.configuration,
                    runner_kind=point_spec.runner_kind,
                    graph_mode=point_spec.graph_mode,
                    context_length=point_spec.context_length,
                    output_steps=point_spec.output_steps,
                    cache_layout_fingerprint=(
                        session.cache_layout_fingerprint()
                    ),
                    method_fingerprint=(
                        session.adapter_config_fingerprint
                    ),
                    accounting=point_accounting,
                )
                active_run = run
                active_initial = initial
                active_started = started
            point = {
                **point,
                "run_id": initial.run_id,
                "git_sha": git_sha,
                "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            }
            _write_point_records(
                run,
                runner_result=runner_result,
                point=point,
                allocation_evidence_files=allocation_evidence_files,
            )
            run_ids.append(initial.run_id)
            point_records.append(point)
            if index != 0:
                completed = run.finalize(
                    _terminal_manifest(
                        initial,
                        started_at_utc=started,
                        status=RunStatus.COMPLETED,
                        failure_reason=None,
                    )
                )
                active_run = None
                active_initial = None
                active_started = None
                _embed_completed_run(bundle_run, completed)
                embedded_ids.append(initial.run_id)
            session = None
            _release_cuda_objects()

        stage = "execution_path"
        launcher_probes = tuple(
            point["launcher_probe"] for point in point_records
        )
        execution_path = _execution_path_audit(
            temporary_shapes=temporary_shapes,
            launcher_probes=launcher_probes,
            backend_fallback_observed=bool(
                graph.get("passed") is not True
                or any(
                    probe.get("passed") is not True
                    for probe in launcher_probes
                )
            ),
            cache_growth_observed=any(
                point["cache_pointers_stable"] is not True
                for point in point_records
            ),
        )
        bundle_run.write_json(
            "validation/execution-path.json",
            {
                **execution_path.to_dict(),
                "adapter_source_path": ADAPTER_PATH.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "adapter_source_sha256": sha256_file(ADAPTER_PATH),
                "host_stub_offsets": {
                    str(bits): offset
                    for bits, offset in OFFICIAL_KIVI_HOST_STUB_OFFSETS.items()
                },
                "runtime_launcher_probe_count": len(launcher_probes),
                "runtime_observation_instrumented_separately": True,
                "normal_timing_instrumented": False,
            },
        )
        if not execution_path.passed:
            raise Phase8KIVIDriverError("KIVI execution-path audit failed")

        stage = "bundle_finalization"
        bounded = {
            "schema_version": "kvbench-phase8-kivi-bounded-grid-1.0.0",
            "plan": [
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
            ],
            "run_ids": run_ids,
            "embedded_run_ids": embedded_ids,
            "bundle_root_point_run_id": bundle_initial.run_id,
            "points": point_records,
            "attempted": len(PHASE8_ADMISSION_GRID),
            "passed": len(point_records),
            "failed": 0,
            "speedup_calculated": False,
            "performance_claim_eligible": False,
            "measurement_scope": "measurement_container_admission",
        }
        bundle_run.write_json("validation/bounded-grid.json", bounded)
        candidate = _derive_local_candidate(
            git_sha=git_sha,
            fixture=fixture,
            graph=graph,
            sanitizer=sanitizer,
            execution_path=execution_path,
            point_records=point_records,
            run_ids=run_ids,
        )
        bundle_run.write_json(
            "validation/admission-candidate.json",
            candidate,
        )
        if candidate["status"] != "LOCAL_CHECKS_PASS_PUBLICATION_PENDING":
            raise Phase8KIVIDriverError(
                "derived local admission checks did not all pass"
            )
        bundle_path = bundle_run.finalize(
            _terminal_manifest(
                bundle_initial,
                started_at_utc=bundle_started,
                status=RunStatus.COMPLETED,
                failure_reason=None,
            )
        )
        validation = validate_local_admission(bundle_path)
        return {
            **validation,
            "git_sha": git_sha,
            "container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            "bundle_path": str(bundle_path),
            "fixture_conformance": candidate["fixture_conformance"],
            "execution_path": execution_path.passed,
            "graph_harness": candidate["graph_capture_replay"],
            "sanitizer": candidate["compute_sanitizer"],
        }
    except Exception as error:
        reason = _safe_reason(error)
        if (
            active_run is not None
            and active_initial is not None
            and active_started is not None
        ):
            try:
                _write_failure_payloads(
                    active_run,
                    stage=stage,
                    reason=reason,
                )
                active_run.write_json(
                    "validation/admission-failure.json",
                    {
                        "stage": stage,
                        "reason": reason,
                        "g2_kivi": "BLOCKED",
                        "performance_claim_eligible": False,
                    },
                )
                active_run.finalize(
                    _terminal_manifest(
                        active_initial,
                        started_at_utc=active_started,
                        status=RunStatus.RUNTIME_FAILED,
                        failure_reason=reason,
                    )
                )
            except Exception:
                pass
        try:
            _write_failure_payloads(
                bundle_run,
                stage=stage,
                reason=reason,
            )
            bundle_run.write_json(
                "validation/admission-failure.json",
                {
                    "stage": stage,
                    "reason": reason,
                    "g2_kivi": "BLOCKED",
                    "performance_claim_eligible": False,
                },
            )
            bundle_run.finalize(
                _terminal_manifest(
                    bundle_initial,
                    started_at_utc=bundle_started,
                    status=RunStatus.RUNTIME_FAILED,
                    failure_reason=reason,
                )
            )
        except Exception:
            pass
        raise


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the finalized local Phase 8 inner admission bundle",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="explicit finalized Phase 8 inner admission bundle path",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if arguments.validate_only != (arguments.artifact is not None):
            raise Phase8KIVIDriverError(
                "--validate-only requires exactly one explicit --artifact"
            )
        result = (
            validate_local_admission(arguments.artifact)
            if arguments.validate_only
            else run_admission()
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": _safe_reason(error),
                    "g2_kivi": "BLOCKED",
                    "performance_claim_eligible": False,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
