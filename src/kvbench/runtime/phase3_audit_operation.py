"""Strict operation identities shared by Phase 3 audit evidence.

This module deliberately contains no collection or GPU integration.  It turns
one exact preregistered process point into the operation keys that dispatch and
allocation evidence can join without sharing their subsystem-specific mode
vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import re
from typing import ClassVar, Literal, TypeAlias

from kvbench.schema.base import (
    GraphMode,
    RunnerKind,
    StrictModel,
    canonical_json_bytes,
    require_git_sha,
    require_identifier,
    require_run_id,
    require_schema,
    require_sha256,
    sha256_hex,
)
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
    derive_phase3_point_fingerprint,
)


DispatchExecutionMode: TypeAlias = Literal["eager", "cuda_graph_replay"]
AllocationExecutionMode: TypeAlias = Literal["eager", "cuda_graph"]

PHASE3_AUDIT_OPERATION_SCHEMA_VERSION = (
    "kvbench-phase3-audit-operation-key-1.0.0"
)

_PHASE3_POINT_RE = re.compile(
    r"\A(?P<runner>fixed_l|growing_context)-"
    r"b(?P<batch>[1-9][0-9]*)-"
    r"l(?P<context>[1-9][0-9]*)-"
    r"(?P<graph>eager|cuda_graph)-"
    r"r(?P<replicate>[1-9][0-9]*)\Z"
)


def _mapped_execution_modes(
    graph_mode: GraphMode,
) -> tuple[DispatchExecutionMode, AllocationExecutionMode]:
    if graph_mode is GraphMode.EAGER:
        return "eager", "eager"
    if graph_mode is GraphMode.CUDA_GRAPH:
        return "cuda_graph_replay", "cuda_graph"
    raise ValueError("graph_mode must use the manifest graph-mode vocabulary")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3AuditOperationKey(StrictModel):
    """Versioned identity for exactly one measured Phase 3 decode operation."""

    schema_version: str
    run_id: str
    point_id: str
    point_fingerprint: str
    process_replicate: int
    runner_kind: RunnerKind
    graph_mode: GraphMode
    dispatch_execution_mode: DispatchExecutionMode
    allocation_execution_mode: AllocationExecutionMode
    decode_step: int
    batch_size: int
    historical_context: int
    attended_context: int
    capacity: int
    cache_layout_fingerprint: str
    execution_git_sha: str
    plan_fingerprint: str
    hardware_identity_sha256: str
    software_identity_sha256: str
    model_identity_sha256: str
    backend_identity_sha256: str
    source_identity_sha256: str
    operation_fingerprint_sha256: str

    SCHEMA_VERSION: ClassVar[str] = PHASE3_AUDIT_OPERATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        string_fields = (
            "schema_version",
            "run_id",
            "point_id",
            "point_fingerprint",
            "dispatch_execution_mode",
            "allocation_execution_mode",
            "cache_layout_fingerprint",
            "execution_git_sha",
            "plan_fingerprint",
            "hardware_identity_sha256",
            "software_identity_sha256",
            "model_identity_sha256",
            "backend_identity_sha256",
            "source_identity_sha256",
            "operation_fingerprint_sha256",
        )
        if any(type(getattr(self, name)) is not str for name in string_fields):
            raise ValueError("operation-key string fields must be strings")
        integer_fields = (
            "process_replicate",
            "decode_step",
            "batch_size",
            "historical_context",
            "attended_context",
            "capacity",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ValueError("operation-key integer fields must be integers")
        if type(self.runner_kind) is not RunnerKind:
            raise ValueError("runner_kind must use the manifest runner vocabulary")
        if type(self.graph_mode) is not GraphMode:
            raise ValueError("graph_mode must use the manifest graph-mode vocabulary")

        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        require_git_sha(self.execution_git_sha)
        require_identifier(self.point_id, field_name="point_id")
        for name in (
            "point_fingerprint",
            "cache_layout_fingerprint",
            "plan_fingerprint",
            "hardware_identity_sha256",
            "software_identity_sha256",
            "model_identity_sha256",
            "backend_identity_sha256",
            "source_identity_sha256",
            "operation_fingerprint_sha256",
        ):
            require_sha256(getattr(self, name), field_name=name)

        match = _PHASE3_POINT_RE.fullmatch(self.point_id)
        if match is None:
            raise ValueError("point_id is not an exact Phase 3 process-point ID")
        expected_point_fingerprint = derive_phase3_point_fingerprint(self.point_id)
        if self.point_fingerprint != expected_point_fingerprint:
            raise ValueError("point_fingerprint does not match the frozen point")
        encoded_point = (
            match.group("runner"),
            match.group("graph"),
            int(match.group("replicate")),
            int(match.group("batch")),
        )
        declared_point = (
            self.runner_kind.value,
            self.graph_mode.value,
            self.process_replicate,
            self.batch_size,
        )
        if declared_point != encoded_point:
            raise ValueError("operation fields do not match point_id")

        plan_path = (
            PHASE3_FIXED_PLAN_PATH
            if self.runner_kind is RunnerKind.FIXED_L
            else PHASE3_GROWING_PLAN_PATH
        )
        if self.plan_fingerprint != PHASE3_PLAN_FINGERPRINTS[plan_path]:
            raise ValueError("plan_fingerprint does not match the frozen point")

        starting_context = int(match.group("context"))
        if self.runner_kind is RunnerKind.FIXED_L:
            expected_geometry = (
                0,
                starting_context,
                starting_context + 1,
                starting_context + 1,
            )
        else:
            if self.graph_mode is not GraphMode.EAGER:
                raise ValueError("growing-context operations must be eager")
            if self.decode_step not in range(16):
                raise ValueError("growing-context decode_step must be in [0, 15]")
            historical_context = starting_context + self.decode_step
            expected_geometry = (
                self.decode_step,
                historical_context,
                historical_context + 1,
                starting_context + 16,
            )
        observed_geometry = (
            self.decode_step,
            self.historical_context,
            self.attended_context,
            self.capacity,
        )
        if observed_geometry != expected_geometry:
            raise ValueError("operation geometry does not match runner semantics")

        dispatch_mode, allocation_mode = _mapped_execution_modes(self.graph_mode)
        if self.dispatch_execution_mode != dispatch_mode:
            raise ValueError("dispatch_execution_mode does not match graph_mode")
        if self.allocation_execution_mode != allocation_mode:
            raise ValueError("allocation_execution_mode does not match graph_mode")

        if self.operation_fingerprint_sha256 != self._derive_fingerprint():
            raise ValueError("operation_fingerprint_sha256 does not match the key")

    @classmethod
    def from_point(
        cls,
        *,
        run_id: str,
        point: Phase3ProcessPoint,
        decode_step: int,
        cache_layout_fingerprint: str,
        execution_git_sha: str,
        plan_fingerprint: str,
        hardware_identity_sha256: str,
        software_identity_sha256: str,
        model_identity_sha256: str,
        backend_identity_sha256: str,
        source_identity_sha256: str,
    ) -> Phase3AuditOperationKey:
        """Build one key from an already validated frozen process point."""

        if type(point) is not Phase3ProcessPoint:
            raise TypeError("point must be a Phase3ProcessPoint")
        if type(decode_step) is not int:
            raise ValueError("decode_step must be an integer")
        if point.runner_kind is RunnerKind.FIXED_L:
            if decode_step != 0:
                raise ValueError("fixed-L points contain exactly decode step 0")
        elif decode_step not in range(16):
            raise ValueError("growing-context decode_step must be in [0, 15]")

        historical_context = point.context_length + decode_step
        dispatch_mode, allocation_mode = _mapped_execution_modes(point.graph_mode)
        unsigned_payload = {
            "schema_version": cls.SCHEMA_VERSION,
            "run_id": run_id,
            "point_id": point.point_id,
            "point_fingerprint": derive_phase3_point_fingerprint(point.point_id),
            "process_replicate": point.process_replicate,
            "runner_kind": point.runner_kind.value,
            "graph_mode": point.graph_mode.value,
            "dispatch_execution_mode": dispatch_mode,
            "allocation_execution_mode": allocation_mode,
            "decode_step": decode_step,
            "batch_size": point.batch_size,
            "historical_context": historical_context,
            "attended_context": historical_context + 1,
            "capacity": point.context_length + point.output_steps,
            "cache_layout_fingerprint": cache_layout_fingerprint,
            "execution_git_sha": execution_git_sha,
            "plan_fingerprint": plan_fingerprint,
            "hardware_identity_sha256": hardware_identity_sha256,
            "software_identity_sha256": software_identity_sha256,
            "model_identity_sha256": model_identity_sha256,
            "backend_identity_sha256": backend_identity_sha256,
            "source_identity_sha256": source_identity_sha256,
        }
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            run_id=run_id,
            point_id=point.point_id,
            point_fingerprint=unsigned_payload["point_fingerprint"],
            process_replicate=point.process_replicate,
            runner_kind=point.runner_kind,
            graph_mode=point.graph_mode,
            dispatch_execution_mode=dispatch_mode,
            allocation_execution_mode=allocation_mode,
            decode_step=decode_step,
            batch_size=point.batch_size,
            historical_context=historical_context,
            attended_context=historical_context + 1,
            capacity=point.context_length + point.output_steps,
            cache_layout_fingerprint=cache_layout_fingerprint,
            execution_git_sha=execution_git_sha,
            plan_fingerprint=plan_fingerprint,
            hardware_identity_sha256=hardware_identity_sha256,
            software_identity_sha256=software_identity_sha256,
            model_identity_sha256=model_identity_sha256,
            backend_identity_sha256=backend_identity_sha256,
            source_identity_sha256=source_identity_sha256,
            operation_fingerprint_sha256=sha256_hex(
                canonical_json_bytes(unsigned_payload)
            ),
        )

    def _unsigned_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        del payload["operation_fingerprint_sha256"]
        return payload

    def _derive_fingerprint(self) -> str:
        return sha256_hex(canonical_json_bytes(self._unsigned_payload()))


def validate_phase3_audit_operation_set(
    operations: Sequence[Phase3AuditOperationKey],
) -> tuple[Phase3AuditOperationKey, ...]:
    """Validate complete, ordered operation coverage for exactly one point."""

    if isinstance(operations, (str, bytes)):
        raise TypeError("operations must be a sequence of operation keys")
    frozen = tuple(operations)
    if not frozen:
        raise ValueError("operation set must be nonempty")
    if any(type(operation) is not Phase3AuditOperationKey for operation in frozen):
        raise TypeError("operation set entries must be Phase3AuditOperationKey values")

    first = frozen[0]
    shared_fields = (
        "schema_version",
        "run_id",
        "point_id",
        "point_fingerprint",
        "process_replicate",
        "runner_kind",
        "graph_mode",
        "dispatch_execution_mode",
        "allocation_execution_mode",
        "batch_size",
        "capacity",
        "cache_layout_fingerprint",
        "execution_git_sha",
        "plan_fingerprint",
        "hardware_identity_sha256",
        "software_identity_sha256",
        "model_identity_sha256",
        "backend_identity_sha256",
        "source_identity_sha256",
    )
    expected_identity = tuple(getattr(first, name) for name in shared_fields)
    if any(
        tuple(getattr(operation, name) for name in shared_fields)
        != expected_identity
        for operation in frozen[1:]
    ):
        raise ValueError("operation set mixes point or identity bindings")

    expected_steps = (
        (0,)
        if first.runner_kind is RunnerKind.FIXED_L
        else tuple(range(16))
    )
    observed_steps = tuple(operation.decode_step for operation in frozen)
    if observed_steps != expected_steps:
        raise ValueError("operation set is incomplete, duplicated, or out of order")
    fingerprints = tuple(
        operation.operation_fingerprint_sha256 for operation in frozen
    )
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("operation fingerprints must be unique")
    return frozen
