"""Narrow raw allocator attribution for Phase 8 KIVI admission."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from kvbench.runtime.allocation import collect_cuda_allocator_raw
from kvbench.runtime.allocation_attribution import (
    DECISION_0009_POLICY_CATALOG_ID,
    DECISION_0009_POLICY_CATALOG_SHA256,
    PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
    PHASE3_OUTPUT_DTYPE_BYTES,
    PHASE3_OUTPUT_WIDTH,
    AllocationClass,
    AllocationAttributionError,
    AllocationCriterionResult,
    AllocationGeometry,
    AllocatorTraceAttribution,
    MemoryDeltaEvidence,
    RawAllocatorEvidenceFiles,
    allocator_counters_from_memory_stats,
    attribute_allocator_trace,
    build_history_integrity_evidence,
    evaluate_strict_graph_criterion,
    instantiate_decision_0013_phase8_kivi_rules,
    memory_delta_from_raw_samples,
    preserve_allocator_evidence,
    raw_memory_accounting_sample_from_mapping,
    read_verified_allocator_evidence,
)
from kvbench.schema import canonical_json_bytes, sha256_hex


_SHA256_CHARS = frozenset("0123456789abcdef")
_CONFIGURATIONS = frozenset({"k4v4", "k2v4", "k2v2", "k4v2"})
_RUNNERS = frozenset({"fixed_l", "growing_context"})
_GRAPH_MODES = frozenset({"eager", "cuda_graph"})
_FORBIDDEN_CLASSES = frozenset(
    {
        AllocationClass.CACHE_GROWTH,
        AllocationClass.GQA_EXPANSION,
        AllocationClass.CONTEXT_SCALED_WORKSPACE,
        AllocationClass.AUDIT_INSTRUMENTATION,
        AllocationClass.UNKNOWN,
    }
)


class KIVIAllocationError(RuntimeError):
    """Phase 8 KIVI allocator evidence is incomplete or inconsistent."""


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise KIVIAllocationError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class KIVIAllocationBinding:
    """Exact Phase 8 operation/source/layout/container composition authority."""

    configuration: str
    runner_kind: str
    graph_mode: str
    historical_context: int
    attended_context: int
    operation_fingerprint_sha256: str
    cache_layout_fingerprint: str
    method_fingerprint: str
    backend_identity: str
    adapter_source_sha256: str
    cache_source_sha256: str
    endpoint_source_sha256: str
    authorized_container_digest: str
    official_commit: str
    patched_tree: str
    decision_0018_patch_sha256: str
    extension_sha256: str
    group_size: int = 32
    residual_length: int = 32
    schema_version: str = (
        "kvbench-phase8-kivi-allocation-binding-1.0.0"
    )

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "kvbench-phase8-kivi-allocation-binding-1.0.0"
            or self.configuration not in _CONFIGURATIONS
            or self.runner_kind not in _RUNNERS
            or self.graph_mode not in _GRAPH_MODES
            or type(self.historical_context) is not int
            or self.historical_context <= 0
            or self.attended_context != self.historical_context + 1
            or self.group_size != 32
            or self.residual_length != 32
        ):
            raise KIVIAllocationError("KIVI allocation binding geometry differs")
        if self.runner_kind == "growing_context" and self.graph_mode != "eager":
            raise KIVIAllocationError(
                "growing-context allocation binding must be eager"
            )
        for name in (
            "operation_fingerprint_sha256",
            "cache_layout_fingerprint",
            "method_fingerprint",
            "adapter_source_sha256",
            "cache_source_sha256",
            "endpoint_source_sha256",
            "decision_0018_patch_sha256",
            "extension_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not self.backend_identity:
            raise KIVIAllocationError("KIVI backend identity is absent")
        if (
            not self.authorized_container_digest.startswith("sha256:")
            or len(self.authorized_container_digest) != 71
        ):
            raise KIVIAllocationError(
                "authorized container config digest is invalid"
            )
        _require_sha256(
            self.authorized_container_digest.removeprefix("sha256:"),
            "authorized container config digest",
        )
        if (
            type(self.official_commit) is not str
            or len(self.official_commit) != 40
            or type(self.patched_tree) is not str
            or len(self.patched_tree) != 40
        ):
            raise KIVIAllocationError("KIVI source authority is invalid")

    @property
    def execution_mode(self) -> str:
        return "cuda_graph" if self.graph_mode == "cuda_graph" else "eager"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "configuration": self.configuration,
            "runner_kind": self.runner_kind,
            "graph_mode": self.graph_mode,
            "execution_mode": self.execution_mode,
            "historical_context": self.historical_context,
            "attended_context": self.attended_context,
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
            "cache_layout_fingerprint": self.cache_layout_fingerprint,
            "method_fingerprint": self.method_fingerprint,
            "backend_identity": self.backend_identity,
            "adapter_source_sha256": self.adapter_source_sha256,
            "cache_source_sha256": self.cache_source_sha256,
            "endpoint_source_sha256": self.endpoint_source_sha256,
            "authorized_container_digest": self.authorized_container_digest,
            "official_commit": self.official_commit,
            "patched_tree": self.patched_tree,
            "decision_0018_patch_sha256": (
                self.decision_0018_patch_sha256
            ),
            "extension_sha256": self.extension_sha256,
            "group_size": self.group_size,
            "residual_length": self.residual_length,
            "allocation_policy_catalog_id": (
                DECISION_0009_POLICY_CATALOG_ID
            ),
            "allocation_policy_catalog_sha256": (
                DECISION_0009_POLICY_CATALOG_SHA256
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> KIVIAllocationBinding:
        """Parse one exact binding; reject aliases and unknown fields."""

        binding = cls(
            schema_version=value.get("schema_version"),
            configuration=value.get("configuration"),
            runner_kind=value.get("runner_kind"),
            graph_mode=value.get("graph_mode"),
            historical_context=value.get("historical_context"),
            attended_context=value.get("attended_context"),
            operation_fingerprint_sha256=value.get(
                "operation_fingerprint_sha256"
            ),
            cache_layout_fingerprint=value.get(
                "cache_layout_fingerprint"
            ),
            method_fingerprint=value.get("method_fingerprint"),
            backend_identity=value.get("backend_identity"),
            adapter_source_sha256=value.get("adapter_source_sha256"),
            cache_source_sha256=value.get("cache_source_sha256"),
            endpoint_source_sha256=value.get("endpoint_source_sha256"),
            authorized_container_digest=value.get(
                "authorized_container_digest"
            ),
            official_commit=value.get("official_commit"),
            patched_tree=value.get("patched_tree"),
            decision_0018_patch_sha256=value.get(
                "decision_0018_patch_sha256"
            ),
            extension_sha256=value.get("extension_sha256"),
            group_size=value.get("group_size"),
            residual_length=value.get("residual_length"),
        )
        if binding.to_dict() != dict(value):
            raise KIVIAllocationError(
                "serialized KIVI allocation binding is not exact"
            )
        return binding

    @property
    def identity_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class KIVIAttributedAllocation:
    """Derived verdict plus the complete append-only raw evidence bytes."""

    summary: Mapping[str, Any]
    files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class KIVIPreservedAllocationReplay:
    """Independent semantic replay of one preserved raw operation."""

    summary: Mapping[str, Any]
    file_sha256_by_basename: Mapping[str, str]
    operation_witness: Mapping[str, Any]
    expected_allocation_event_count: int
    expected_allocation_event_bytes: int


def _geometry(binding: KIVIAllocationBinding) -> AllocationGeometry:
    return AllocationGeometry(
        batch=1,
        query_heads=32,
        kv_heads=8,
        context=binding.attended_context,
        head_dim=128,
        dtype_bytes=2,
        query_length=1,
        operation_output_width=PHASE3_OUTPUT_WIDTH,
        operation_output_dtype_bytes=PHASE3_OUTPUT_DTYPE_BYTES,
    )


def _evaluate_eager(
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
) -> AllocationCriterionResult:
    reasons: list[str] = []
    history = attribution.history_integrity
    if attribution.integrity_errors:
        reasons.append("allocator_trace_integrity_failure")
    if history is None:
        reasons.append("allocator_history_integrity_missing")
    else:
        reasons.extend(history.failure_reasons())
        if history.raw_trace_sha256 != attribution.trace_sha256:
            reasons.append("attributed_trace_sha256_mismatch")
    if any(
        not allocation.python_stack or not allocation.cpp_stack
        for allocation in attribution.allocations
    ):
        reasons.append("allocator_allocation_stack_incomplete")
    if not attribution.counters.complete:
        reasons.append("allocator_counter_evidence_incomplete")
    if not attribution.all_block_sizes_proven:
        reasons.append("allocated_block_size_evidence_incomplete")
    if not attribution.all_lifetimes_fully_freed:
        reasons.append("allocation_lifetime_not_fully_freed")
    if not attribution.all_allocations_cache_reused:
        reasons.append("allocation_not_reused_from_cache")
    if attribution.segment_alloc_count or attribution.segment_free_count:
        reasons.append("allocator_segment_event_detected")
    counters = attribution.counters
    for name in (
        "device_allocation_count",
        "device_free_count",
        "allocation_retry_count",
        "oom_count",
    ):
        if getattr(counters, name) != 0:
            reasons.append(f"allocator_{name}_nonzero_or_unavailable")
    if memory.allocated_delta != 0:
        reasons.append("persistent_allocated_delta_nonzero")
    if memory.reserved_delta != 0:
        reasons.append("persistent_reserved_delta_nonzero")
    if memory.device_used_delta != 0:
        reasons.append("persistent_device_used_delta_nonzero_or_unavailable")
    if memory.non_pytorch_delta != 0:
        reasons.append("persistent_non_pytorch_delta_nonzero_or_unavailable")

    policies = attribution.rules.permitted_allocation_policies
    allowed_ids = {policy.policy_id for policy in policies}
    for policy in policies:
        observed = [
            allocation
            for allocation in attribution.allocations
            if allocation.policy_id == policy.policy_id
        ]
        if len(observed) != policy.exact_count:
            reasons.append(
                f"allocation_policy_count_bound_failed:{policy.policy_id}"
            )
        if (
            sum(allocation.requested_bytes for allocation in observed)
            != policy.exact_total_requested_bytes
        ):
            reasons.append(
                f"allocation_policy_byte_bound_failed:{policy.policy_id}"
            )
    for allocation in attribution.allocations:
        if (
            allocation.event_class in _FORBIDDEN_CLASSES
            or allocation.policy_id not in allowed_ids
        ):
            reasons.append(
                f"forbidden_or_unattributed_allocation:"
                f"{allocation.allocation_id}"
            )
    no_context_dependent = all(
        allocation.dependencies.context is False
        for allocation in attribution.allocations
    )
    if not no_context_dependent:
        reasons.append("context_dependent_allocation_detected")
    unique = tuple(dict.fromkeys(reasons))
    passed = not unique
    return AllocationCriterionResult(
        criterion_id=(
            "phase8_kivi_decision_0013_composed_eager_attribution_v1"
        ),
        passed=passed,
        failure_reasons=unique,
        allocation_event_count=len(attribution.allocations),
        class_counts=attribution.class_counts(),
        no_context_dependent_allocation=no_context_dependent,
        fully_attributed_bounded_ephemeral=passed,
        strict_graph_zero_events=None,
    )


def _validate_witness(
    value: Mapping[str, Any],
    *,
    binding: KIVIAllocationBinding,
) -> None:
    if (
        value.get("schema_version")
        != "kvbench-phase8-kivi-allocation-operation-witness-1.0.0"
        or value.get("binding_sha256") != binding.identity_sha256
        or value.get("operation_fingerprint_sha256")
        != binding.operation_fingerprint_sha256
    ):
        raise KIVIAllocationError("allocation operation witness identity differs")
    before = value.get("state_before")
    after = value.get("state_after")
    output = value.get("measured_output")
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or not isinstance(output, Mapping)
        or before.get("cache_pointers_sha256")
        != after.get("cache_pointers_sha256")
        or before.get("active_context") != binding.historical_context
        or (
            binding.runner_kind == "fixed_l"
            and after.get("active_context") != binding.historical_context
        )
        or (
            binding.runner_kind == "growing_context"
            and after.get("active_context") != binding.attended_context
        )
        or output.get("finite") is not True
    ):
        raise KIVIAllocationError("allocation operation witness did not pass")
    _require_sha256(
        before.get("cache_pointers_sha256"),
        "pre-operation cache pointer fingerprint",
    )
    _require_sha256(
        output.get("sha256"),
        "allocation-audit output fingerprint",
    )


def derive_kivi_allocation_attribution(
    *,
    binding: KIVIAllocationBinding,
    snapshot: Mapping[str, Any],
    trace: tuple[Mapping[str, Any], ...],
    memory_stats_before: Mapping[str, Any],
    memory_stats_after: Mapping[str, Any],
    memory_accounting_before: Mapping[str, Any],
    memory_accounting_after: Mapping[str, Any],
    operation_witness: Mapping[str, Any],
    expected_snapshot_sha256: str | None = None,
    expected_trace_sha256: str | None = None,
) -> tuple[
    AllocatorTraceAttribution,
    MemoryDeltaEvidence,
    AllocationCriterionResult,
]:
    """Independently replay raw Phase 8 bytes; no caller verdict is consumed."""

    _validate_witness(operation_witness, binding=binding)
    geometry = _geometry(binding)
    rules = instantiate_decision_0013_phase8_kivi_rules(
        geometry=geometry,
        backend_identity=binding.backend_identity,
        composition_binding_sha256=binding.identity_sha256,
    )
    history = build_history_integrity_evidence(
        snapshot,
        trace,
        max_entries=PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
        stack_mode="all",
        expected_snapshot_sha256=expected_snapshot_sha256,
        expected_trace_sha256=expected_trace_sha256,
    )
    counters = allocator_counters_from_memory_stats(
        memory_stats_before,
        memory_stats_after,
    )
    attribution = attribute_allocator_trace(
        trace,
        geometry=geometry,
        counters=counters,
        rules=rules,
        backend_identity=binding.backend_identity,
        expected_trace_sha256=history.expected_raw_trace_sha256,
        history_integrity=history,
    )
    before = raw_memory_accounting_sample_from_mapping(
        memory_accounting_before
    )
    after = raw_memory_accounting_sample_from_mapping(
        memory_accounting_after
    )
    if (
        before.operation_fingerprint_sha256
        != binding.operation_fingerprint_sha256
        or after.operation_fingerprint_sha256
        != binding.operation_fingerprint_sha256
    ):
        raise KIVIAllocationError(
            "raw memory samples differ from the KIVI operation binding"
        )
    memory = memory_delta_from_raw_samples(before, after)
    criterion = (
        evaluate_strict_graph_criterion(attribution, memory)
        if binding.execution_mode == "cuda_graph"
        else _evaluate_eager(attribution, memory)
    )
    return attribution, memory, criterion


def _derived_operation_summary(
    *,
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
    criterion: AllocationCriterionResult,
    binding: KIVIAllocationBinding,
) -> tuple[dict[str, Any], int, int, int]:
    expected_count = sum(
        policy.exact_count
        for policy in attribution.rules.permitted_allocation_policies
    )
    expected_bytes = sum(
        policy.exact_total_requested_bytes
        for policy in attribution.rules.permitted_allocation_policies
    )
    observed_bytes = sum(
        allocation.requested_bytes
        for allocation in attribution.allocations
    )
    unknown_count = sum(
        allocation.event_class in _FORBIDDEN_CLASSES
        or allocation.policy_id is None
        for allocation in attribution.allocations
    )
    summary = {
        "raw": {
            "audit_available": True,
            "allocation_event_count": len(attribution.allocations),
            "allocation_event_bytes": observed_bytes,
            "allocated_delta": memory.allocated_delta,
            "reserved_delta": memory.reserved_delta,
            "event_counts": dict(
                sorted(
                    Counter(
                        event.action for event in attribution.events
                    ).items()
                )
            ),
        },
        "criterion": {
            **criterion.to_dict(),
            "expected_allocation_event_count": expected_count,
            "expected_allocation_event_bytes": expected_bytes,
            "allocation_event_bytes": observed_bytes,
            "attended_context": binding.attended_context,
            "persistent_allocated_delta": memory.allocated_delta,
            "persistent_reserved_delta": memory.reserved_delta,
            "unknown_allocation_count": unknown_count,
            "categories": attribution.class_counts(),
            "attribution_rules_sha256": attribution.rules.identity_sha256,
            "composition_binding_sha256": binding.identity_sha256,
        },
    }
    return summary, expected_count, expected_bytes, unknown_count


def _raw_files_from_mapping(
    value: Mapping[str, Any],
) -> RawAllocatorEvidenceFiles:
    expected_keys = {
        "snapshot_file",
        "snapshot_sha256",
        "trace_file",
        "trace_sha256",
        "memory_stats_before_file",
        "memory_stats_before_sha256",
        "memory_stats_after_file",
        "memory_stats_after_sha256",
        "memory_accounting_before_file",
        "memory_accounting_before_sha256",
        "memory_accounting_after_file",
        "memory_accounting_after_sha256",
        "operation_witness_file",
        "operation_witness_sha256",
        "audit_file",
        "audit_sha256_file",
        "audit_sha256",
    }
    if set(value) != expected_keys:
        raise KIVIAllocationError(
            "KIVI allocator raw-file index is not exact"
        )
    try:
        return RawAllocatorEvidenceFiles(
            snapshot_file=value.get("snapshot_file"),
            snapshot_sha256=value.get("snapshot_sha256"),
            trace_file=value.get("trace_file"),
            trace_sha256=value.get("trace_sha256"),
            memory_stats_before_file=value.get(
                "memory_stats_before_file"
            ),
            memory_stats_before_sha256=value.get(
                "memory_stats_before_sha256"
            ),
            memory_stats_after_file=value.get("memory_stats_after_file"),
            memory_stats_after_sha256=value.get(
                "memory_stats_after_sha256"
            ),
            memory_accounting_before_file=value.get(
                "memory_accounting_before_file"
            ),
            memory_accounting_before_sha256=value.get(
                "memory_accounting_before_sha256"
            ),
            memory_accounting_after_file=value.get(
                "memory_accounting_after_file"
            ),
            memory_accounting_after_sha256=value.get(
                "memory_accounting_after_sha256"
            ),
            operation_witness_file=value.get("operation_witness_file"),
            operation_witness_sha256=value.get(
                "operation_witness_sha256"
            ),
            audit_file=value.get("audit_file"),
            audit_sha256_file=value.get("audit_sha256_file"),
            audit_sha256=value.get("audit_sha256"),
        )
    except AllocationAttributionError as error:
        raise KIVIAllocationError(str(error)) from error


def _parse_canonical_mapping(
    payload: bytes,
    *,
    label: str,
) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KIVIAllocationError(
            f"{label} is not valid JSON"
        ) from error
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise KIVIAllocationError(
            f"{label} is not canonical JSON"
        ) from error
    if not isinstance(value, Mapping) or canonical != payload:
        raise KIVIAllocationError(
            f"{label} is not a canonical JSON object"
        )
    return value


def _parse_canonical_trace(
    payload: bytes,
) -> tuple[Mapping[str, Any], ...]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KIVIAllocationError(
            "allocator trace is not valid JSON"
        ) from error
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise KIVIAllocationError(
            "allocator trace is not canonical JSON"
        ) from error
    if (
        not isinstance(value, list)
        or not all(isinstance(item, Mapping) for item in value)
        or canonical != payload
    ):
        raise KIVIAllocationError(
            "allocator trace is not a canonical event list"
        )
    return tuple(value)


def replay_preserved_kivi_allocation_attribution(
    directory: Path,
    *,
    raw_files: Mapping[str, Any],
    expected_binding: KIVIAllocationBinding,
) -> KIVIPreservedAllocationReplay:
    """Rehash and re-derive one operation without consuming stored verdicts."""

    files = _raw_files_from_mapping(raw_files)
    try:
        payloads = read_verified_allocator_evidence(directory, files)
    except (OSError, AllocationAttributionError) as error:
        raise KIVIAllocationError(
            "KIVI raw allocator evidence failed checksum verification"
        ) from error
    snapshot = _parse_canonical_mapping(
        payloads["snapshot"],
        label="allocator snapshot",
    )
    trace = _parse_canonical_trace(payloads["trace"])
    stats_before = _parse_canonical_mapping(
        payloads["stats_before"],
        label="allocator memory stats before",
    )
    stats_after = _parse_canonical_mapping(
        payloads["stats_after"],
        label="allocator memory stats after",
    )
    accounting_before = _parse_canonical_mapping(
        payloads["accounting_before"],
        label="allocator memory accounting before",
    )
    accounting_after = _parse_canonical_mapping(
        payloads["accounting_after"],
        label="allocator memory accounting after",
    )
    witness = _parse_canonical_mapping(
        payloads["operation_witness"],
        label="allocation operation witness",
    )
    audit = _parse_canonical_mapping(
        payloads["audit"],
        label="allocation audit",
    )
    audit_binding_value = audit.get("binding")
    if not isinstance(audit_binding_value, Mapping):
        raise KIVIAllocationError(
            "allocation audit binding is absent"
        )
    observed_binding = KIVIAllocationBinding.from_mapping(
        audit_binding_value
    )
    if observed_binding != expected_binding:
        raise KIVIAllocationError(
            "allocation audit binding differs from the point authority"
        )
    try:
        attribution, memory, criterion = (
            derive_kivi_allocation_attribution(
                binding=expected_binding,
                snapshot=snapshot,
                trace=trace,
                memory_stats_before=stats_before,
                memory_stats_after=stats_after,
                memory_accounting_before=accounting_before,
                memory_accounting_after=accounting_after,
                operation_witness=witness,
                expected_snapshot_sha256=files.snapshot_sha256,
                expected_trace_sha256=files.trace_sha256,
            )
        )
    except (AllocationAttributionError, KIVIAllocationError) as error:
        raise KIVIAllocationError(
            "KIVI raw allocator semantic replay failed"
        ) from error
    (
        derived_summary,
        expected_count,
        expected_bytes,
        unknown_count,
    ) = _derived_operation_summary(
        attribution=attribution,
        memory=memory,
        criterion=criterion,
        binding=expected_binding,
    )
    expected_audit = {
        "schema_version": (
            "kvbench-phase8-kivi-allocation-attribution-1.0.0"
        ),
        "run_kind": "allocation_audit",
        "evidence_status": "complete",
        "execution_mode": expected_binding.execution_mode,
        "binding": expected_binding.to_dict(),
        "binding_sha256": expected_binding.identity_sha256,
        "memory": memory.to_dict(),
        "attribution": attribution.to_dict(),
        "criterion": criterion.to_dict(),
        "expected_allocation_event_count": expected_count,
        "expected_allocation_event_bytes": expected_bytes,
        "observed_allocation_event_bytes": derived_summary["raw"][
            "allocation_event_bytes"
        ],
        "unknown_allocation_count": unknown_count,
        "operation_witness": dict(witness),
        "profiler_timing_reported": False,
        "instrumented_duration_reported_as_timing": False,
        "normal_benchmark_timing_eligible": False,
        "raw_files": files.to_dict(include_audit_sha256=False),
    }
    if dict(audit) != expected_audit:
        raise KIVIAllocationError(
            "stored allocation audit differs from raw semantic replay"
        )
    if (
        not criterion.passed
        or unknown_count != 0
        or derived_summary["criterion"][
            "expected_allocation_event_count"
        ]
        != expected_count
        or derived_summary["criterion"][
            "expected_allocation_event_bytes"
        ]
        != expected_bytes
    ):
        raise KIVIAllocationError(
            "raw-derived KIVI allocation criterion did not pass"
        )
    logical_payload_names = {
        "snapshot": files.snapshot_file,
        "trace": files.trace_file,
        "stats_before": files.memory_stats_before_file,
        "stats_after": files.memory_stats_after_file,
        "accounting_before": files.memory_accounting_before_file,
        "accounting_after": files.memory_accounting_after_file,
        "operation_witness": files.operation_witness_file,
        "audit": files.audit_file,
        "audit_digest": files.audit_sha256_file,
    }
    file_sha256 = {
        basename: hashlib.sha256(payloads[key]).hexdigest()
        for key, basename in logical_payload_names.items()
    }
    return KIVIPreservedAllocationReplay(
        summary={
            **derived_summary,
            "raw_files": files.to_dict(),
        },
        file_sha256_by_basename=file_sha256,
        operation_witness=dict(witness),
        expected_allocation_event_count=expected_count,
        expected_allocation_event_bytes=expected_bytes,
    )


def _read_preserved_files(
    directory: Path,
    files: RawAllocatorEvidenceFiles,
) -> dict[str, bytes]:
    names = (
        files.snapshot_file,
        files.trace_file,
        files.memory_stats_before_file,
        files.memory_stats_after_file,
        files.memory_accounting_before_file,
        files.memory_accounting_after_file,
        files.operation_witness_file,
        files.audit_file,
        files.audit_sha256_file,
    )
    return {name: (directory / name).read_bytes() for name in names}


def collect_kivi_allocation_attribution(
    operation: Callable[[], Any],
    *,
    prepare_operation: Callable[[], Any],
    capture_state: Callable[[], Mapping[str, Any]],
    capture_output: Callable[[Any], Mapping[str, Any]],
    binding: KIVIAllocationBinding,
    device: Any,
) -> KIVIAttributedAllocation:
    """Collect, semantically derive, and preserve one exact KIVI operation."""

    raw = collect_cuda_allocator_raw(
        operation,
        operation_fingerprint_sha256=(
            binding.operation_fingerprint_sha256
        ),
        prepare_operation=prepare_operation,
        capture_output=capture_output,
        capture_state=capture_state,
        device=device,
        max_entries=PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
    )
    if (
        raw.state_before is None
        or raw.state_after is None
        or raw.output_witness is None
    ):
        raise KIVIAllocationError(
            "raw allocator collection lacks operation witnesses"
        )
    witness = {
        "schema_version": (
            "kvbench-phase8-kivi-allocation-operation-witness-1.0.0"
        ),
        "binding_sha256": binding.identity_sha256,
        "operation_fingerprint_sha256": (
            binding.operation_fingerprint_sha256
        ),
        "state_before": dict(raw.state_before),
        "state_after": dict(raw.state_after),
        "measured_output": dict(raw.output_witness),
    }
    attribution, memory, criterion = derive_kivi_allocation_attribution(
        binding=binding,
        snapshot=raw.snapshot,
        trace=raw.trace,
        memory_stats_before=raw.memory_stats_before,
        memory_stats_after=raw.memory_stats_after,
        memory_accounting_before=raw.memory_accounting_before,
        memory_accounting_after=raw.memory_accounting_after,
        operation_witness=witness,
    )
    (
        derived_summary,
        expected_count,
        expected_bytes,
        unknown_count,
    ) = _derived_operation_summary(
        attribution=attribution,
        memory=memory,
        criterion=criterion,
        binding=binding,
    )
    observed_bytes = derived_summary["raw"]["allocation_event_bytes"]
    audit_payload = {
        "schema_version": (
            "kvbench-phase8-kivi-allocation-attribution-1.0.0"
        ),
        "run_kind": "allocation_audit",
        "evidence_status": "complete",
        "execution_mode": binding.execution_mode,
        "binding": binding.to_dict(),
        "binding_sha256": binding.identity_sha256,
        "memory": memory.to_dict(),
        "attribution": attribution.to_dict(),
        "criterion": criterion.to_dict(),
        "expected_allocation_event_count": expected_count,
        "expected_allocation_event_bytes": expected_bytes,
        "observed_allocation_event_bytes": observed_bytes,
        "unknown_allocation_count": unknown_count,
        "operation_witness": witness,
        "profiler_timing_reported": False,
        "instrumented_duration_reported_as_timing": False,
        "normal_benchmark_timing_eligible": False,
    }
    with tempfile.TemporaryDirectory(
        prefix="kvbench-phase8-kivi-allocation-",
        dir="/tmp",
    ) as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        history = attribution.history_integrity
        if history is None:
            raise KIVIAllocationError(
                "allocator history integrity evidence is absent"
            )
        preserved = preserve_allocator_evidence(
            directory,
            snapshot=raw.snapshot,
            trace=raw.trace,
            memory_stats_before=raw.memory_stats_before,
            memory_stats_after=raw.memory_stats_after,
            memory_accounting_before=raw.memory_accounting_before,
            memory_accounting_after=raw.memory_accounting_after,
            operation_witness=witness,
            expected_snapshot_sha256=history.raw_snapshot_sha256,
            expected_trace_sha256=history.raw_trace_sha256,
            audit_payload=audit_payload,
        )
        file_bytes = _read_preserved_files(directory, preserved)
    summary = {**derived_summary, "raw_files": preserved.to_dict()}
    return KIVIAttributedAllocation(summary=summary, files=file_bytes)


def raw_file_sha256(value: bytes) -> str:
    """Expose exact file identities for the point-level evidence index."""

    return hashlib.sha256(value).hexdigest()
