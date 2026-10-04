#!/usr/bin/env python3
"""Run or validate the bounded non-claim Phase 6 TurboQuant admission."""

from __future__ import annotations

import argparse
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

from kvbench.adapters.turboquant import (
    TURBOQUANT_ADAPTER_VERSION,
    TurboQuantMethodAdapter,
)
from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.artifacts import (
    ArtifactRun,
    phase6_artifact_store,
    sha256_file,
    validate_run_directory,
)
from kvbench.runtime.backend import forced_flash_execution
from kvbench.runtime.fixed_l_runner import run_fixed_l
from kvbench.runtime.growing_context_runner import run_growing_context
from kvbench.runtime.model_loader import (
    MODEL_ID,
    MODEL_REVISION,
    load_frozen_model,
    validate_loaded_frozen_model_receipt,
)
from kvbench.runtime.numerical import tensor_sha256_untimed
from kvbench.runtime.turboquant_admission import (
    TurboQuantAdmissionError,
    evaluate_fixture_configuration,
    mandatory_configuration_summary,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_cache import (
    TURBOQUANT_BF16_LAYERS,
    TURBOQUANT_COMPRESSED_LAYERS,
    TURBOQUANT_MANDATORY_CONFIGS,
    TURBOQUANT_SLOT_SIZES,
)
from kvbench.runtime.turboquant_session import (
    build_turboquant_endpoint_session,
    build_turboquant_operation_keys,
    phase6_backend_fingerprint,
    turboquant_runtime_context,
)
from kvbench.schema import (
    ClaimClass,
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    MethodName,
    Phase6BackendIdentity,
    Phase6RunManifest,
    QualityExecutionState,
    QualityValidationState,
    RunKind,
    RunnerKind,
    RunStatus,
    canonical_json_bytes,
    sha256_hex,
)
from preflight.run_preflight import sanitizer_error_count

from kvbench.schema.phase6 import (
    AUTHORIZED_CONTAINER_DIGEST,
    DECODE_SOURCE_SHA256,
    FIXTURE_ROOT_LEDGER_SHA256,
    FIXTURE_SET_SHA256,
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_TREE,
    STAGE2_SOURCE_SHA256,
    STORE_SOURCE_SHA256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPOSITORY_ROOT / "src" / "kvbench" / "adapters" / "turboquant.py"
)
SANITIZER = Path("/usr/local/cuda-13.0/bin/compute-sanitizer")
CONTAINER_PYTHON = Path("/opt/kvbench/.venv/bin/python")
SANITIZER_PROBE = (
    REPOSITORY_ROOT / "tests" / "cuda" / "phase6_turboquant_sanitizer_probe.py"
)
GRID = (
    ("turboquant_4bit_nc", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    (
        "turboquant_4bit_nc",
        RunnerKind.FIXED_L,
        GraphMode.CUDA_GRAPH,
        128,
        1,
    ),
    ("turboquant_k3v4_nc", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    (
        "turboquant_k3v4_nc",
        RunnerKind.FIXED_L,
        GraphMode.CUDA_GRAPH,
        128,
        1,
    ),
    ("turboquant_3bit_nc", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    (
        "turboquant_3bit_nc",
        RunnerKind.FIXED_L,
        GraphMode.CUDA_GRAPH,
        128,
        1,
    ),
    ("turboquant_4bit_nc", RunnerKind.FIXED_L, GraphMode.EAGER, 4096, 1),
    (
        "turboquant_4bit_nc",
        RunnerKind.FIXED_L,
        GraphMode.CUDA_GRAPH,
        4096,
        1,
    ),
    (
        "turboquant_4bit_nc",
        RunnerKind.GROWING_CONTEXT,
        GraphMode.EAGER,
        128,
        4,
    ),
)
_PARAMETERS: Mapping[str, Mapping[str, int]] = {
    "turboquant_4bit_nc": {"key_bits": 4, "value_bits": 4},
    "turboquant_k3v4_nc": {"key_bits": 3, "value_bits": 4},
    "turboquant_3bit_nc": {"key_bits": 3, "value_bits": 3},
}


class Phase6DriverError(RuntimeError):
    """The bounded admission driver could not preserve its contract."""


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
        raise Phase6DriverError("required Git identity query failed")
    return result.stdout.strip()


def _require_clean_git() -> str:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise Phase6DriverError("Phase 6 admission requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    if len(head) != 40:
        raise Phase6DriverError("Phase 6 Git SHA is invalid")
    return head


def _method_config_fingerprint(configuration: str) -> str:
    if configuration not in TURBOQUANT_MANDATORY_CONFIGS:
        raise ValueError("method configuration is not mandatory")
    payload = {
        "schema_version": "kvbench-phase6-method-config-1.0.0",
        "method": "turboquant",
        "configuration": configuration,
        "source_commit": PINNED_SOURCE_COMMIT,
        "source_tree": PINNED_SOURCE_TREE,
        "key_bits": _PARAMETERS[configuration]["key_bits"],
        "value_bits": _PARAMETERS[configuration]["value_bits"],
        "key_path": "mse",
        "norm_correction": True,
        "block_size": 16,
        "decode_split_count": 4,
        "slot_size_bytes": TURBOQUANT_SLOT_SIZES[configuration],
        "compressed_layers": list(TURBOQUANT_COMPRESSED_LAYERS),
        "bf16_layers": list(TURBOQUANT_BF16_LAYERS),
    }
    return sha256_hex(canonical_json_bytes(payload))


def _backend_identity() -> Phase6BackendIdentity:
    return Phase6BackendIdentity(
        schema_version=Phase6BackendIdentity.SCHEMA_VERSION,
        backend_id="pytorch_flash_turboquant",
        backend_fingerprint=phase6_backend_fingerprint(),
        store_kernel_family="_tq_fused_store_mse",
        decode_kernel_families=(
            "_tq_decode_stage1",
            "_fwd_kernel_stage2",
        ),
        store_source_sha256=STORE_SOURCE_SHA256,
        decode_source_sha256=DECODE_SOURCE_SHA256,
        stage2_source_sha256=STAGE2_SOURCE_SHA256,
    )


def _run_id(
    *,
    git_sha: str,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    context_length: int,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:-3] + "z"
    config = configuration.removeprefix("turboquant_")
    runner = "fixed" if runner_kind is RunnerKind.FIXED_L else "growing"
    graph = "graph" if graph_mode is GraphMode.CUDA_GRAPH else "eager"
    return (
        f"phase6-{stamp}-{git_sha[:8]}-{secrets.token_hex(3)}-"
        f"{config}-{runner}-l{context_length}-{graph}"
    )


def _b018_sanitizer_run_id(*, git_sha: str, configuration: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:-3] + "z"
    config = configuration.removeprefix("turboquant_")
    return (
        f"phase6-b018-{stamp}-{git_sha[:8]}-{secrets.token_hex(3)}-"
        f"{config}-sanitizer"
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
    adapter_config_fingerprint: str,
    created_at_utc: str,
) -> Phase6RunManifest:
    return Phase6RunManifest(
        schema_version=Phase6RunManifest.SCHEMA_VERSION,
        artifact_schema_version=(
            Phase6RunManifest.ARTIFACT_SCHEMA_VERSION
        ),
        run_id=run_id,
        status=RunStatus.CREATED,
        created_at_utc=created_at_utc,
        started_at_utc=None,
        finished_at_utc=None,
        run_kind=RunKind.PHASE6_ADMISSION,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        claim_class=ClaimClass.NONE,
        measurement_scope=(
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ),
        performance_claim_eligible=False,
        git_sha=git_sha,
        git_dirty=False,
        container_digest=AUTHORIZED_CONTAINER_DIGEST,
        method=MethodName.TURBOQUANT,
        method_config_id=configuration,
        method_config_fingerprint=_method_config_fingerprint(configuration),
        adapter_version=TURBOQUANT_ADAPTER_VERSION,
        adapter_source_sha256=sha256_file(ADAPTER_PATH),
        adapter_config_fingerprint=adapter_config_fingerprint,
        pinned_source_commit=PINNED_SOURCE_COMMIT,
        pinned_source_tree=PINNED_SOURCE_TREE,
        fixture_set_sha256=FIXTURE_SET_SHA256,
        fixture_root_ledger_sha256=FIXTURE_ROOT_LEDGER_SHA256,
        cache_layout_fingerprint=cache_layout_fingerprint,
        slot_size_bytes=TURBOQUANT_SLOT_SIZES[configuration],
        compressed_layers=TURBOQUANT_COMPRESSED_LAYERS,
        bf16_layers=TURBOQUANT_BF16_LAYERS,
        backend_identity=_backend_identity(),
        batch_size=1,
        context_length=context_length,
        output_steps=output_steps,
        quality_status=QualityValidationState.UNVALIDATED,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        quality_execution=QualityExecutionState.LOCKED,
        performance_data_frozen=False,
        quality_benchmark_executed=False,
        speedup_calculated=False,
        r_hbm=None,
        full_scan_state="CLOSED",
        inventory_path=None,
        failure_reason=None,
    )


def _terminal_manifest(
    initial: Phase6RunManifest,
    *,
    started_at_utc: str,
    status: RunStatus,
    failure_reason: str | None,
) -> Phase6RunManifest:
    return dataclasses.replace(
        initial,
        status=status,
        started_at_utc=started_at_utc,
        finished_at_utc=_utc_now(),
        inventory_path="artifact_inventory.json",
        failure_reason=failure_reason,
    )


def _method_record(
    configuration: str,
    *,
    adapter_config_fingerprint: str,
    cache_layout_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase6-method-record-1.0.0",
        "method": "turboquant",
        "method_config_id": configuration,
        "method_config_fingerprint": _method_config_fingerprint(configuration),
        "adapter_version": TURBOQUANT_ADAPTER_VERSION,
        "adapter_source_sha256": sha256_file(ADAPTER_PATH),
        "adapter_config_fingerprint": adapter_config_fingerprint,
        "cache_layout_fingerprint": cache_layout_fingerprint,
        "pinned_source_commit": PINNED_SOURCE_COMMIT,
        "pinned_source_tree": PINNED_SOURCE_TREE,
        "fixture_set_sha256": FIXTURE_SET_SHA256,
        "fixture_root_ledger_sha256": FIXTURE_ROOT_LEDGER_SHA256,
        "slot_size_bytes": TURBOQUANT_SLOT_SIZES[configuration],
        "block_size": 16,
        "compressed_layers": list(TURBOQUANT_COMPRESSED_LAYERS),
        "bf16_layers": list(TURBOQUANT_BF16_LAYERS),
        "backend_identity": _backend_identity().to_dict(),
        "r_hbm": None,
    }


def _write_common_records(
    run: ArtifactRun,
    initial: Phase6RunManifest,
    environment: Mapping[str, Any],
) -> None:
    run.write_json(
        "config/method.json",
        _method_record(
            initial.method_config_id,
            adapter_config_fingerprint=initial.adapter_config_fingerprint,
            cache_layout_fingerprint=initial.cache_layout_fingerprint,
        ),
    )
    run.write_json(
        "environment/container_identity.json",
        {
            "schema_version": "kvbench-phase6-container-runtime-1.0.0",
            **dict(environment),
            "git_sha": initial.git_sha,
            "image_mutated": False,
            "packages_installed": False,
            "network_enabled": False,
            "credentials_passed": False,
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


def _sanitizer_tool_identity() -> dict[str, Any]:
    if not SANITIZER.is_file() or not CONTAINER_PYTHON.is_file():
        raise Phase6DriverError("locked Compute Sanitizer tool is unavailable")
    result = subprocess.run(
        (str(SANITIZER), "--version"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Phase6DriverError("Compute Sanitizer identity query failed")
    return {
        "path": str(SANITIZER),
        "resolved_path": str(SANITIZER.resolve(strict=True)),
        "sha256": sha256_file(SANITIZER.resolve(strict=True)),
        "version_stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "version_stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
        "version_stdout": result.stdout.decode("utf-8", errors="replace"),
        "version_stderr": result.stderr.decode("utf-8", errors="replace"),
        "exit_code": result.returncode,
    }


def _memcheck_summaries_pass(stdout: bytes, stderr: bytes) -> bool:
    combined = stdout + b"\n" + stderr
    text = combined.decode("utf-8", errors="replace")
    return sanitizer_error_count("memcheck", text) == 0


def _run_sanitizer_configuration(
    run: ArtifactRun,
    configuration: str,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if configuration not in TURBOQUANT_MANDATORY_CONFIGS:
        raise ValueError("sanitizer configuration is not mandatory")
    tool_identity = (
        _sanitizer_tool_identity() if identity is None else dict(identity)
    )
    sanitizer_environment = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    child_environment = os.environ.copy()
    child_environment.update(sanitizer_environment)
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
        "--configuration",
        configuration,
        "--image-config-digest",
        AUTHORIZED_CONTAINER_DIGEST,
    )
    started = _utc_now()
    monotonic_started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            timeout=900,
            env=child_environment,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code: int | None = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        exit_code = None
        timed_out = True
    summaries_passed = _memcheck_summaries_pass(stdout, stderr)
    probe_passed = b'"status":"pass"' in stdout
    passed = (
        not timed_out
        and exit_code == 0
        and probe_passed
        and summaries_passed
    )
    prefix = f"validation/sanitizer/{configuration}"
    run.write_bytes(f"{prefix}/stdout.txt", stdout)
    run.write_bytes(f"{prefix}/stderr.txt", stderr)
    record = {
        "schema_version": "kvbench-phase6-sanitizer-result-1.0.0",
        "configuration": configuration,
        "command": list(command),
        "tool_identity": tool_identity,
        "environment": sanitizer_environment,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "duration_seconds": float(time.monotonic() - monotonic_started),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "memcheck_summaries_passed": summaries_passed,
        "probe_passed": probe_passed,
        "passed": passed,
    }
    run.write_json(f"{prefix}/result.json", record)
    return record


def _run_sanitizers(run: ArtifactRun) -> dict[str, Any]:
    identity = _sanitizer_tool_identity()
    records: dict[str, Any] = {}
    for configuration in TURBOQUANT_MANDATORY_CONFIGS:
        record = _run_sanitizer_configuration(
            run,
            configuration,
            identity=identity,
        )
        records[configuration] = record
        if record["passed"] is not True:
            raise TurboQuantAdmissionError(
                f"Compute Sanitizer failed for {configuration}"
            )
    return {
        "schema_version": "kvbench-phase6-sanitizer-summary-1.0.0",
        "tool_identity": identity,
        "configurations": records,
        "passed": all(item["passed"] for item in records.values()),
    }


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


def _full_model_allocation_criterion(
    evidence: Mapping[str, Any],
    *,
    graph_required: bool,
    turboquant_hot_path_zero_allocation: bool,
    attended_context: int,
) -> dict[str, Any]:
    """Apply the frozen eager/Graph criteria by narrow composition."""

    required = (
        "audit_available",
        "allocation_event_count",
        "allocation_event_bytes",
        "allocated_delta",
        "reserved_delta",
        "event_counts",
    )
    if any(name not in evidence for name in required):
        raise Phase6DriverError("full-model allocation evidence is incomplete")
    if (
        isinstance(attended_context, bool)
        or not isinstance(attended_context, int)
        or attended_context <= 0
    ):
        raise Phase6DriverError("allocation context is invalid")
    audit_available = evidence["audit_available"] is True
    allocated_zero = evidence["allocated_delta"] == 0
    reserved_zero = evidence["reserved_delta"] == 0
    expected_event_count: int
    expected_event_bytes: int
    if graph_required:
        expected_event_count = 0
        expected_event_bytes = 0
        zero_events = (
            evidence["allocation_event_count"] == 0
            and sum(int(value) for value in evidence["event_counts"].values())
            == 0
        )
        passed = (
            audit_available
            and zero_events
            and allocated_zero
            and reserved_zero
        )
        criterion_id = "phase3_graph_zero_allocation_v1"
        categories: dict[str, int] = {}
        fully_attributed = False
    else:
        zero_events = evidence["allocation_event_count"] == 0
        split_count = (attended_context + 127) // 128
        expected_event_count = 898
        expected_event_bytes = (
            9_637_132
            + 4 * (8_344 + 16_512 * split_count)
        )
        exact_frozen_ephemeral_set = (
            evidence["allocation_event_count"] == expected_event_count
            and evidence["allocation_event_bytes"] == expected_event_bytes
        )
        passed = (
            audit_available
            and turboquant_hot_path_zero_allocation
            and exact_frozen_ephemeral_set
            and allocated_zero
            and reserved_zero
        )
        criterion_id = "phase6_composed_frozen_eager_attribution_v1"
        categories = {
            "fixed_shared_activation": 881,
            "framework_bookkeeping": 8,
            "context_scaled_workspace": 8,
            "fixed_output": 1,
            "turboquant_hot_path": 0,
        }
        fully_attributed = passed
    return {
        "criterion_id": criterion_id,
        "passed": passed,
        "audit_available": audit_available,
        "allocation_event_count": int(evidence["allocation_event_count"]),
        "allocation_event_bytes": int(evidence["allocation_event_bytes"]),
        "expected_allocation_event_count": expected_event_count,
        "expected_allocation_event_bytes": expected_event_bytes,
        "attended_context": attended_context,
        "persistent_allocated_delta": int(evidence["allocated_delta"]),
        "persistent_reserved_delta": int(evidence["reserved_delta"]),
        "strict_graph_zero_events": (
            zero_events if graph_required else None
        ),
        "fully_attributed_bounded_ephemeral": fully_attributed,
        "unknown_allocation_count": 0 if passed else None,
        "categories": categories,
        "composition_authority": {
            "surrounding_model": (
                "G1_and_Phase6A_frozen_attributed_ephemeral_path"
            ),
            "turboquant_hot_path": (
                "mandatory_fixture_store_append_decode_zero_event_audits"
            ),
            "criteria_redefined": False,
        },
    }


def _audit_session(
    session: Any,
    *,
    turboquant_hot_path_zero_allocation: bool,
) -> tuple[list[tuple[str, bool]], dict[str, Any]]:
    import torch

    outputs: list[tuple[str, bool]] = []
    allocations: list[dict[str, Any]] = []
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
        session.prepare_audit_step(step)
        audit = audit_cuda_allocations(
            lambda step=step: session.execute_audit_step(step),
            device=session.cache_device,
        )
        raw_allocation = audit.to_dict()
        allocations.append(
            {
                "raw": raw_allocation,
                "criterion": _full_model_allocation_criterion(
                    raw_allocation,
                    graph_required=(
                        session.operation_keys[0].graph_mode
                        is GraphMode.CUDA_GRAPH
                    ),
                    turboquant_hot_path_zero_allocation=(
                        turboquant_hot_path_zero_allocation
                    ),
                    attended_context=(
                        session.operation_keys[step].attended_context
                    ),
                ),
            }
        )
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
    }


def _execute_point(
    *,
    loaded: Any,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    context_length: int,
    output_steps: int,
    global_audits_passed: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    keys = build_turboquant_operation_keys(
        configuration=configuration,
        runner_kind=runner_kind,
        graph_mode=graph_mode,
        starting_context=context_length,
        output_steps=output_steps,
    )
    offset = (
        10_000
        + list(TURBOQUANT_MANDATORY_CONFIGS).index(configuration) * 20_000
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
        session = build_turboquant_endpoint_session(
            loaded=loaded,
            operation_keys=keys,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
        observed, audit = _audit_session(
            session,
            turboquant_hot_path_zero_allocation=global_audits_passed,
        )
        session.admit(
            observed_outputs=observed,
            execution_path_passed=global_audits_passed,
            allocation_passed=audit["passed"],
            graph_passed=audit["graph_passed"],
        )
    if runner_kind is RunnerKind.FIXED_L:
        result = run_fixed_l(
            session,
            measured_steps=1,
            measured_batches=1,
        ).to_dict()
    else:
        result = run_growing_context(
            session,
            expected_steps=4,
        ).to_dict()
    accounting = result["cache_accounting"]
    allocated = int(accounting["allocated_bytes"])
    predicted = int(accounting["predicted_tensor_bytes"])
    relative_error = abs(predicted - allocated) / allocated
    breakdown_sum = sum(int(value) for value in result["cache_byte_breakdown"].values())
    point = {
        "schema_version": "kvbench-phase6-point-validation-1.0.0",
        "configuration": configuration,
        "runner_kind": runner_kind.value,
        "graph_mode": graph_mode.value,
        "batch_size": 1,
        "context_length": context_length,
        "output_steps": output_steps,
        "allocation": audit,
        "predicted_allocated_relative_error": relative_error,
        "byte_breakdown_sum": breakdown_sum,
        "allocated_bytes": allocated,
        "output_finite": result["output_finite"],
        "cache_pointers_stable": result["cache_pointers_stable"],
        "historical_cache_unchanged": result[
            "historical_cache_unchanged"
        ],
        "native_gqa": (
            result["gqa_cache_geometry"]["native_kv_head_storage"]
            and not result["gqa_cache_geometry"]["gqa_materialized"]
        ),
        "speedup_calculated": False,
        "r_hbm": None,
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
    }
    point["passed"] = all(
        (
            audit["passed"],
            relative_error < 0.01,
            breakdown_sum == allocated,
            point["output_finite"],
            point["cache_pointers_stable"],
            point["historical_cache_unchanged"],
            point["native_gqa"],
            result["measurement_scope"]
            == "measurement_container_admission",
        )
    )
    if point["passed"] is not True:
        raise Phase6DriverError("bounded admission point failed")
    return session, result, point


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
    adapter_config_fingerprint: str,
    run_id_override: str | None = None,
) -> tuple[ArtifactRun, Phase6RunManifest, str]:
    created = _utc_now()
    identifier = run_id_override or _run_id(
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
        adapter_config_fingerprint=adapter_config_fingerprint,
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
) -> None:
    run.write_json("raw/runner.json", runner_result)
    run.write_json("validation/point.json", point)
    run.write_json(
        "allocation/full-model.json",
        dict(point["allocation"]),
    )
    run.write_json(
        "gqa/full-model.json",
        {
            "native_gqa": point["native_gqa"],
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
            raise Phase6DriverError("Phase 6 artifact contains a symlink")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise Phase6DriverError("Phase 6 artifact contains an unsafe entry")
        relative = candidate.relative_to(root).as_posix()
        files[relative] = candidate
    return files


def _embed_completed_run(bundle: ArtifactRun, completed: Path) -> None:
    validation = validate_run_directory(completed)
    if not validation.valid or not validation.complete:
        raise Phase6DriverError("point run is not complete before embedding")
    if completed.name == bundle.run_id:
        raise Phase6DriverError("bundle cannot recursively embed itself")
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
    except Phase6DriverError:
        return False
    if set(embedded_files) != set(completed_files):
        return False
    return all(
        source.stat().st_size == embedded_files[relative].stat().st_size
        and sha256_file(source) == sha256_file(embedded_files[relative])
        for relative, source in completed_files.items()
    )


def _validate_completed_grid(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise Phase6DriverError("Phase 6 artifact root is absent")
    records: dict[str, dict[str, Any]] = {}
    bundle_ids: list[str] = []
    failed_ids: list[str] = []
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue
        if not path.is_dir() or path.is_symlink():
            raise Phase6DriverError("Phase 6 artifact root is unsafe")
        validation = validate_run_directory(path)
        if not validation.valid or not validation.complete:
            raise Phase6DriverError(
                f"Phase 6 run validation failed: {path.name}"
            )
        manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or run_id != path.name:
            raise Phase6DriverError("Phase 6 artifact identity differs")
        records[run_id] = manifest
        if manifest.get("status") != "completed":
            failed_ids.append(run_id)
        elif (path / "validation" / "bounded-grid.json").is_file():
            bundle_ids.append(run_id)
    if len(bundle_ids) != 1:
        raise Phase6DriverError(
            "Phase 6 requires exactly one completed admission bundle"
        )
    bundle_path = root / bundle_ids[0] / "validation" / "bounded-grid.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    run_ids = bundle.get("run_ids")
    if (
        not isinstance(run_ids, list)
        or len(run_ids) != len(GRID)
        or len(set(run_ids)) != len(GRID)
        or any(not isinstance(item, str) for item in run_ids)
    ):
        raise Phase6DriverError("Phase 6 bundle run index is invalid")
    embedded_ids = bundle.get("embedded_run_ids")
    if (
        not isinstance(embedded_ids, list)
        or embedded_ids != run_ids[1:]
    ):
        raise Phase6DriverError("Phase 6 embedded run index is invalid")
    selected: list[dict[str, Any]] = []
    for run_id in run_ids:
        manifest = records.get(run_id)
        if manifest is None or manifest.get("status") != "completed":
            raise Phase6DriverError(
                "Phase 6 bundle references an incomplete run"
            )
        selected.append(manifest)
    bundle_root = root / bundle_ids[0]
    if any(
        not _embedded_run_matches(bundle_root, run_id, root / run_id)
        for run_id in embedded_ids
    ):
        raise Phase6DriverError("embedded Phase 6 point differs from source")
    observed = {
        (
            item["method_config_id"],
            item["runner_kind"],
            item["graph_mode"],
            item["context_length"],
            item["output_steps"],
        )
        for item in selected
    }
    expected = {
        (config, runner.value, graph.value, context, output)
        for config, runner, graph, context, output in GRID
    }
    if observed != expected or len(selected) != len(GRID):
        raise Phase6DriverError("completed Phase 6 grid is not exact")
    return {
        "status": "PASS",
        "artifact_root": str(root),
        "run_count": len(selected),
        "run_ids": run_ids,
        "bundle_run_id": bundle_ids[0],
        "historical_failed_run_ids": failed_ids,
        "bounded_grid_exact": True,
        "performance_claim_eligible": False,
        "speedup_calculated": False,
    }


def run_admission() -> dict[str, Any]:
    git_sha = _require_clean_git()
    environment = require_authorized_cuda_environment(
        AUTHORIZED_CONTAINER_DIGEST
    )
    store = phase6_artifact_store(REPOSITORY_ROOT)
    first = GRID[0]
    pre_method = TurboQuantMethodAdapter(
        turboquant_runtime_context(),
        first[0],
    )
    pre_cache = pre_method.allocate(
        batch_size=1,
        capacity=first[3] + first[4],
        device="cuda:0",
        workspace_bytes=0,
    )
    bundle_run, bundle_initial, bundle_started = _create_started_run(
        store=store,
        git_sha=git_sha,
        environment=environment,
        configuration=first[0],
        runner_kind=first[1],
        graph_mode=first[2],
        context_length=first[3],
        output_steps=first[4],
        cache_layout_fingerprint=pre_cache.layout_fingerprint(),
        adapter_config_fingerprint=pre_method.config_fingerprint(
            pre_cache.layout_fingerprint()
        ),
    )
    stage = "fixture_audits"
    run_ids: list[str] = []
    point_records: list[dict[str, Any]] = []
    embedded_run_ids: list[str] = []
    active_run: ArtifactRun | None = None
    active_initial: Phase6RunManifest | None = None
    active_started: str | None = None
    try:
        fixture_records = {
            configuration: evaluate_fixture_configuration(
                configuration,
                evidence_directory=(
                    bundle_run.stage / "validation" / configuration
                ),
            )
            for configuration in TURBOQUANT_MANDATORY_CONFIGS
        }
        fixture_summary = mandatory_configuration_summary(fixture_records)
        bundle_run.write_json(
            "validation/fixture-summary.json",
            fixture_summary,
        )
        if fixture_summary["passed"] is not True:
            raise Phase6DriverError("mandatory fixture audits failed")

        stage = "compute_sanitizer"
        sanitizer_summary = _run_sanitizers(bundle_run)
        bundle_run.write_json(
            "validation/sanitizer-summary.json",
            sanitizer_summary,
        )
        if sanitizer_summary["passed"] is not True:
            raise Phase6DriverError("mandatory sanitizer matrix failed")

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

        global_audits_passed = fixture_summary["passed"] is True
        for index, (
            configuration,
            runner_kind,
            graph_mode,
            context_length,
            output_steps,
        ) in enumerate(GRID):
            stage = f"bounded_grid_{index}"
            session, runner_result, point = _execute_point(
                loaded=loaded,
                configuration=configuration,
                runner_kind=runner_kind,
                graph_mode=graph_mode,
                context_length=context_length,
                output_steps=output_steps,
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
                    != initial.adapter_config_fingerprint
                ):
                    raise Phase6DriverError(
                        "bundle cache identity differs from warmed session"
                    )
            else:
                run, initial, started = _create_started_run(
                    store=store,
                    git_sha=git_sha,
                    environment=environment,
                    configuration=configuration,
                    runner_kind=runner_kind,
                    graph_mode=graph_mode,
                    context_length=context_length,
                    output_steps=output_steps,
                    cache_layout_fingerprint=(
                        session.cache_layout_fingerprint()
                    ),
                    adapter_config_fingerprint=(
                        session.adapter_config_fingerprint
                    ),
                )
                active_run = run
                active_initial = initial
                active_started = started
            point = {
                **point,
                "run_id": initial.run_id,
                "git_sha": git_sha,
                "container_digest": AUTHORIZED_CONTAINER_DIGEST,
            }
            _write_point_records(
                run,
                runner_result=runner_result,
                point=point,
            )
            run_ids.append(initial.run_id)
            point_records.append(point)
            if index != 0:
                completed_path = run.finalize(
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
                _embed_completed_run(bundle_run, completed_path)
                embedded_run_ids.append(initial.run_id)
            session = None
            _release_cuda_objects()

        stage = "bundle_finalization"
        bounded_summary = {
            "schema_version": "kvbench-phase6-bounded-grid-1.0.0",
            "plan": [
                {
                    "configuration": config,
                    "runner_kind": runner.value,
                    "graph_mode": graph.value,
                    "batch_size": 1,
                    "context_length": context,
                    "output_steps": output,
                }
                for config, runner, graph, context, output in GRID
            ],
            "run_ids": run_ids,
            "embedded_run_ids": embedded_run_ids,
            "bundle_root_point_run_id": bundle_initial.run_id,
            "points": point_records,
            "attempted": len(GRID),
            "passed": len(point_records),
            "failed": 0,
            "capacity_infeasible": 0,
            "speedup_calculated": False,
            "performance_claim_eligible": False,
        }
        bundle_run.write_json(
            "validation/bounded-grid.json",
            bounded_summary,
        )
        bundle_run.write_json(
            "validation/admission-candidate.json",
            {
                "schema_version": (
                    "kvbench-phase6-admission-candidate-1.0.0"
                ),
                "status": "LOCAL_CHECKS_PASS_PUBLICATION_PENDING",
                "git_sha": git_sha,
                "container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "fixture_conformance": True,
                "byte_accounting": True,
                "static_cache_skip_policy": True,
                "store_append_correctness": True,
                "decode_tolerance": True,
                "finite_output": True,
                "execution_path": True,
                "allocation": True,
                "cuda_graph": True,
                "compute_sanitizer": True,
                "bounded_admission_grid": True,
                "immutable_checksums": "pending_finalization",
                "durable_publication": "pending_host_side",
                "g2_tq": "NOT_EVALUATED",
                "global_g2": "NOT_EVALUATED",
                "quality_execution": "LOCKED",
                "full_scan": "CLOSED",
                "performance_data_frozen": False,
                "performance_claim_eligible": False,
                "speedup_calculated": False,
                "r_hbm": None,
            },
        )
        bundle_path = bundle_run.finalize(
            _terminal_manifest(
                bundle_initial,
                started_at_utc=bundle_started,
                status=RunStatus.COMPLETED,
                failure_reason=None,
            )
        )
        validation = _validate_completed_grid(
            REPOSITORY_ROOT / "artifacts" / "phase6"
        )
        return {
            **validation,
            "git_sha": git_sha,
            "container_digest": AUTHORIZED_CONTAINER_DIGEST,
            "bundle_run_id": bundle_initial.run_id,
            "bundle_path": str(bundle_path),
            "fixture_summary": fixture_summary["passed"],
            "sanitizer_summary": sanitizer_summary["passed"],
            "g2_tq": "NOT_EVALUATED_PUBLICATION_PENDING",
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
                        "g2_tq": "BLOCKED",
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
                    "g2_tq": "BLOCKED",
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


def _b018_cache_identity(configuration: str) -> tuple[str, str]:
    method = TurboQuantMethodAdapter(
        turboquant_runtime_context(),
        configuration,
    )
    cache: Any | None = None
    try:
        cache = method.allocate(
            batch_size=1,
            capacity=18,
            device="cuda:0",
            workspace_bytes=0,
        )
        layout_fingerprint = cache.layout_fingerprint()
        adapter_fingerprint = method.config_fingerprint(layout_fingerprint)
        return layout_fingerprint, adapter_fingerprint
    finally:
        if cache is not None:
            import torch

            torch.cuda.synchronize(device=cache.device)
            cache.release_owned_cuda_resources_for_sanitizer()
        cache = None
        method = None
        from kvbench.third_party.vllm_turboquant.compat import (
            _build_hadamard_cached,
        )

        _build_hadamard_cached.cache_clear()
        _release_cuda_objects()


def _b018_artifact_checksums(
    completed: Path,
    configuration: str,
) -> dict[str, str]:
    prefix = Path("validation") / "sanitizer" / configuration
    relatives = (
        Path("manifest.json"),
        Path("artifact_inventory.json"),
        Path("checksums.sha256"),
        Path("COMPLETE"),
        prefix / "stdout.txt",
        prefix / "stderr.txt",
        prefix / "result.json",
    )
    return {
        relative.as_posix(): sha256_file(completed / relative)
        for relative in relatives
    }


def run_b018_sanitizer_only() -> dict[str, Any]:
    """Run only the three sequential B-018 memcheck probes."""

    git_sha = _require_clean_git()
    environment = require_authorized_cuda_environment(
        AUTHORIZED_CONTAINER_DIGEST
    )
    store = phase6_artifact_store(REPOSITORY_ROOT)
    identity = _sanitizer_tool_identity()
    records: list[dict[str, Any]] = []
    for configuration in TURBOQUANT_MANDATORY_CONFIGS:
        layout_fingerprint, adapter_fingerprint = _b018_cache_identity(
            configuration
        )
        run, initial, started = _create_started_run(
            store=store,
            git_sha=git_sha,
            environment=environment,
            configuration=configuration,
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.EAGER,
            context_length=128,
            output_steps=1,
            cache_layout_fingerprint=layout_fingerprint,
            adapter_config_fingerprint=adapter_fingerprint,
            run_id_override=_b018_sanitizer_run_id(
                git_sha=git_sha,
                configuration=configuration,
            ),
        )
        finalized = False
        try:
            sanitizer = _run_sanitizer_configuration(
                run,
                configuration,
                identity=identity,
            )
            summary = {
                "schema_version": (
                    "kvbench-phase6-b018-sanitizer-summary-1.0.0"
                ),
                "configuration": configuration,
                "sanitizer_only": True,
                "cuda_graph_created": False,
                "bounded_grid_attempted": False,
                "result": sanitizer,
                "passed": sanitizer["passed"],
            }
            run.write_json("validation/sanitizer-summary.json", summary)
            run.write_json(
                "raw/runner.json",
                {
                    "schema_version": (
                        "kvbench-phase6-b018-sanitizer-runner-1.0.0"
                    ),
                    "configuration": configuration,
                    "operation": "store_append_decode",
                    "sanitizer_only": True,
                    "cuda_graph_created": False,
                    "bounded_grid_attempted": False,
                    "probe_passed": sanitizer["probe_passed"],
                    "exit_code": sanitizer["exit_code"],
                    "memcheck_summaries_passed": sanitizer[
                        "memcheck_summaries_passed"
                    ],
                    "performance_claim_eligible": False,
                },
            )
            run.write_json(
                "validation/point.json",
                {
                    "schema_version": (
                        "kvbench-phase6-b018-sanitizer-validation-1.0.0"
                    ),
                    "configuration": configuration,
                    "passed": sanitizer["passed"],
                    "sanitizer_only": True,
                    "bounded_grid": "NOT_EVALUATED",
                    "g2_tq": "BLOCKED",
                    "quality_execution": "LOCKED",
                    "full_scan": "CLOSED",
                    "performance_claim_eligible": False,
                    "speedup_calculated": False,
                    "r_hbm": None,
                },
            )
            if sanitizer["passed"] is not True:
                raise TurboQuantAdmissionError(
                    f"Compute Sanitizer failed for {configuration}; "
                    f"run_id={initial.run_id}"
                )
            completed = run.finalize(
                _terminal_manifest(
                    initial,
                    started_at_utc=started,
                    status=RunStatus.ABORTED,
                    failure_reason=(
                        "b018_sanitizer_only_scope_complete_"
                        "bounded_grid_not_authorized"
                    ),
                )
            )
            finalized = True
            validation = validate_run_directory(completed)
            if not validation.valid or not validation.complete:
                raise Phase6DriverError(
                    f"B-018 artifact validation failed: {initial.run_id}"
                )
            records.append(
                {
                    "configuration": configuration,
                    "run_id": initial.run_id,
                    "artifact_path": str(completed),
                    "artifact_status": RunStatus.ABORTED.value,
                    "probe_passed": True,
                    "exit_code": 0,
                    "memcheck_summaries_passed": True,
                    "checksums": _b018_artifact_checksums(
                        completed,
                        configuration,
                    ),
                }
            )
        except Exception as error:
            if not finalized:
                reason = _safe_reason(error)
                try:
                    _write_failure_payloads(
                        run,
                        stage="b018_compute_sanitizer",
                        reason=reason,
                    )
                    run.write_json(
                        "validation/admission-failure.json",
                        {
                            "stage": "b018_compute_sanitizer",
                            "reason": reason,
                            "g2_tq": "BLOCKED",
                            "bounded_grid_attempted": False,
                            "performance_claim_eligible": False,
                        },
                    )
                    run.finalize(
                        _terminal_manifest(
                            initial,
                            started_at_utc=started,
                            status=RunStatus.RUNTIME_FAILED,
                            failure_reason=reason,
                        )
                    )
                except Exception:
                    pass
            raise
    return {
        "status": "PASS",
        "scope": "phase6_b018_sanitizer_only",
        "git_sha": git_sha,
        "container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "configurations": records,
        "bounded_grid_attempted": False,
        "g2_tq": "BLOCKED",
        "quality_execution": "LOCKED",
        "full_scan": "CLOSED",
        "performance_claim_eligible": False,
        "speedup_calculated": False,
    }

def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate finalized local Phase 6 admission artifacts",
    )
    mode.add_argument(
        "--b018-sanitizer-only",
        action="store_true",
        help="run only the three sequential B-018 sanitizer probes",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if arguments.validate_only:
            result = _validate_completed_grid(
                REPOSITORY_ROOT / "artifacts" / "phase6"
            )
        elif arguments.b018_sanitizer_only:
            result = run_b018_sanitizer_only()
        else:
            result = run_admission()
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": _safe_reason(error),
                    "g2_tq": "BLOCKED",
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
