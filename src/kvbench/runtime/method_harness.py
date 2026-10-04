"""Thin reusable facades over the validated Phase 3 admission helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from kvbench.adapters import KVCacheMethod
from kvbench.runtime.backend import forced_flash_execution
from kvbench.runtime.cuda_graph import (
    FullModelGraphValidation,
    GraphCaptureError,
    validate_full_model_fixed_graph,
)
from kvbench.runtime.numerical import (
    FullModelReferenceResult,
    NumericalComparison,
    compare_tensors_untimed,
    small_attention_reference,
    validate_full_model_reference,
)


@dataclass(frozen=True, slots=True)
class MethodCorrectnessHarnessResult:
    passed: bool
    small_tensor: NumericalComparison
    full_model: FullModelReferenceResult
    graph: FullModelGraphValidation

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "small_tensor": self.small_tensor.to_dict(),
            "full_model": self.full_model.to_dict(),
            "graph": self.graph.to_dict(),
            "static_cache_checked": True,
            "fixed_l_checked": True,
            "growing_context_checked": True,
            "eager_checked": True,
            "graph_checked": True,
            "nan_inf_checked": True,
            "timing_collected": False,
        }


def run_graph_harness(
    method: KVCacheMethod,
    model: Any,
    prefix_input_ids: Any,
    current_input_ids: Any,
) -> FullModelGraphValidation:
    """Reuse the existing capture/replay, pointer, output, and allocation test."""

    if not method.supports_cuda_graph():
        raise GraphCaptureError("method does not declare CUDA Graph support")
    result = validate_full_model_fixed_graph(
        model,
        prefix_input_ids,
        current_input_ids,
        method=method,
    )
    if result.graph.get("fallback") is not False:
        raise GraphCaptureError("graph harness detected eager fallback")
    return result


def run_correctness_harness(
    method: KVCacheMethod,
    model: Any,
    prefix_input_ids: Any,
    decode_input_ids: Any,
    *,
    attention_module: Any,
    small_query: Any,
    small_key: Any,
    small_value: Any,
    scale: float,
) -> MethodCorrectnessHarnessResult:
    """Run existing small, fixed-L, growing, eager, and graph validations."""

    reference = small_attention_reference(
        small_query,
        small_key,
        small_value,
        is_causal=int(small_query.shape[-2]) > 1,
        scale=scale,
    )
    with forced_flash_execution():
        observed = method.decode_attention(
            attention_module,
            small_query,
            small_key,
            small_value,
            scaling=scale,
        )
    small = compare_tensors_untimed(
        observed,
        reference,
        atol=0.02,
        rtol=0.02,
    )
    full = validate_full_model_reference(
        model,
        prefix_input_ids,
        decode_input_ids,
        method=method,
    )
    graph = run_graph_harness(
        method,
        model,
        prefix_input_ids,
        decode_input_ids[:, :1],
    )
    return MethodCorrectnessHarnessResult(
        passed=small.passed and full.passed and graph.passed,
        small_tensor=small,
        full_model=full,
        graph=graph,
    )


@dataclass(frozen=True, slots=True)
class MethodAllocationHarnessResult:
    passed: bool
    predicted_bytes: int
    allocated_cache_bytes: int
    byte_breakdown: dict[str, int]
    workspace_bytes: int
    persistent_allocated_delta: int
    persistent_reserved_delta: int
    categorized_eager_allocations: dict[str, int]
    graph_replay_allocations: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "predicted_bytes": self.predicted_bytes,
            "allocated_cache_bytes": self.allocated_cache_bytes,
            "byte_breakdown": dict(self.byte_breakdown),
            "workspace_bytes": self.workspace_bytes,
            "persistent_allocated_delta": self.persistent_allocated_delta,
            "persistent_reserved_delta": self.persistent_reserved_delta,
            "categorized_eager_allocations": dict(
                self.categorized_eager_allocations
            ),
            "graph_replay_allocations": dict(self.graph_replay_allocations),
            "criterion_source": "frozen_phase3_allocation_verdicts",
            "timing_collected": False,
        }


def _integer_field(evidence: Mapping[str, object], name: str) -> int:
    value = evidence.get(name)
    if type(value) is not int:
        raise ValueError(f"allocation evidence field is not an integer: {name}")
    return value


def _event_counts(evidence: Mapping[str, object]) -> dict[str, int]:
    raw = evidence.get("event_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("allocation evidence lacks categorized event counts")
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if type(key) is not str or type(value) is not int or value < 0:
            raise ValueError("allocation event counts are malformed")
        counts[key] = value
    return dict(sorted(counts.items()))


def summarize_allocation_harness(
    method: KVCacheMethod,
    cache_state: Any,
    *,
    eager_evidence: Mapping[str, object],
    graph_replay_evidence: Mapping[str, object],
) -> MethodAllocationHarnessResult:
    """Combine existing frozen eager/graph verdicts with method byte ownership."""

    eager_passed = eager_evidence.get("passed")
    graph_passed = graph_replay_evidence.get("passed")
    if type(eager_passed) is not bool or type(graph_passed) is not bool:
        raise ValueError("allocation evidence lacks derived pass booleans")
    breakdown = dict(sorted(method.byte_breakdown(cache_state).items()))
    allocated = method.allocated_bytes(cache_state)
    if (
        not breakdown
        or any(type(value) is not int or value < 0 for value in breakdown.values())
        or sum(breakdown.values()) != allocated
    ):
        raise ValueError("method byte breakdown does not equal owned storage")
    graph_counts = _event_counts(graph_replay_evidence)
    graph_zero = sum(graph_counts.values()) == 0
    return MethodAllocationHarnessResult(
        passed=eager_passed and graph_passed and graph_zero,
        predicted_bytes=method.logical_bf16_bytes(cache_state),
        allocated_cache_bytes=allocated,
        byte_breakdown=breakdown,
        workspace_bytes=breakdown.get("workspace_bytes", 0),
        persistent_allocated_delta=_integer_field(
            eager_evidence,
            "allocated_delta",
        ),
        persistent_reserved_delta=_integer_field(
            eager_evidence,
            "reserved_delta",
        ),
        categorized_eager_allocations=_event_counts(eager_evidence),
        graph_replay_allocations=graph_counts,
    )


@dataclass(frozen=True, slots=True)
class ExecutionPathAuditFacade:
    passed: bool
    backend_identity_verified: bool
    device_kernel_family_verified: bool
    allocation_categories_verified: bool
    temporary_tensor_shapes_verified: bool
    gqa_replication_detected: bool
    full_prefix_temporary_detected: bool
    host_synchronization_detected: bool
    backend_fallback_detected: bool
    full_prefix_dequantization: Literal["not_applicable", "verified_false"]

    def to_dict(self) -> dict[str, bool | str]:
        return {
            "passed": self.passed,
            "backend_identity_verified": self.backend_identity_verified,
            "device_kernel_family_verified": self.device_kernel_family_verified,
            "allocation_categories_verified": self.allocation_categories_verified,
            "temporary_tensor_shapes_verified": (
                self.temporary_tensor_shapes_verified
            ),
            "gqa_replication_detected": self.gqa_replication_detected,
            "full_prefix_temporary_detected": (
                self.full_prefix_temporary_detected
            ),
            "host_synchronization_detected": (
                self.host_synchronization_detected
            ),
            "backend_fallback_detected": self.backend_fallback_detected,
            "full_prefix_dequantization": self.full_prefix_dequantization,
            "evidence_source": "existing_phase3_audits",
        }


def execution_path_audit_facade(
    *,
    backend_identity_verified: bool,
    device_kernel_family_verified: bool,
    allocation_categories_verified: bool,
    temporary_tensor_shapes_verified: bool,
    gqa_replication_detected: bool,
    full_prefix_temporary_detected: bool,
    host_synchronization_detected: bool,
    backend_fallback_detected: bool,
    full_prefix_dequantization: Literal[
        "not_applicable",
        "verified_false",
    ] = "not_applicable",
) -> ExecutionPathAuditFacade:
    """Expose already derived Phase 3 audit facts without collecting a new trace."""

    booleans = (
        backend_identity_verified,
        device_kernel_family_verified,
        allocation_categories_verified,
        temporary_tensor_shapes_verified,
        gqa_replication_detected,
        full_prefix_temporary_detected,
        host_synchronization_detected,
        backend_fallback_detected,
    )
    if any(type(value) is not bool for value in booleans):
        raise ValueError("execution-path audit inputs must be booleans")
    passed = (
        backend_identity_verified
        and device_kernel_family_verified
        and allocation_categories_verified
        and temporary_tensor_shapes_verified
        and not gqa_replication_detected
        and not full_prefix_temporary_detected
        and not host_synchronization_detected
        and not backend_fallback_detected
    )
    return ExecutionPathAuditFacade(
        passed=passed,
        backend_identity_verified=backend_identity_verified,
        device_kernel_family_verified=device_kernel_family_verified,
        allocation_categories_verified=allocation_categories_verified,
        temporary_tensor_shapes_verified=temporary_tensor_shapes_verified,
        gqa_replication_detected=gqa_replication_detected,
        full_prefix_temporary_detected=full_prefix_temporary_detected,
        host_synchronization_detected=host_synchronization_detected,
        backend_fallback_detected=backend_fallback_detected,
        full_prefix_dequantization=full_prefix_dequantization,
    )
