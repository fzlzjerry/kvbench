"""Raw paired allocator controls for the Phase 3 GQA proof.

The collector in this module is deliberately untimed and returns bytes only.
It records one isolated public-SDPA GQA control and one held-constant MHA
control.  The parser and verifier independently reconstruct allocator facts;
no serialized verdict or caller-supplied allocation classification is trusted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import re
import time
from typing import Any, Literal, TypeAlias

from kvbench.errors import SchemaValidationError
from kvbench.runtime.allocation_attribution import (
    AllocationClass,
    AllocationGeometry,
    AllocatorCounterEvidence,
    AllocatorTraceAttribution,
    AttributionRules,
    MemoryDeltaEvidence,
    PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
    PHASE3_ALLOCATION_WARMUP_ITERATIONS,
    PHASE3_BACKEND_IDENTITY_SHA256,
    PHASE3_CUDA_RUNTIME_VERSION,
    PHASE3_DEVICE,
    PHASE3_DEVICE_INDEX,
    PHASE3_OUTPUT_DTYPE,
    PHASE3_RECORDER_CONFIGURATION,
    PHASE3_TORCH_VERSION,
    RawMemoryAccountingSample,
    allocator_counters_from_memory_stats,
    allocator_snapshot_sha256,
    allocator_trace_from_snapshot,
    allocator_trace_sha256,
    attribute_allocator_trace,
    build_history_integrity_evidence,
    cuda_allocator_rounded_minimum,
    evaluate_strict_graph_criterion,
    memory_delta_from_raw_samples,
)
from kvbench.runtime.backend import backend_identity, forced_flash_execution
from kvbench.runtime.cuda_graph import CapturedFixedGraph, capture_fixed_graph
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey


AllocatorControlRole: TypeAlias = Literal["gqa", "mha_control"]

GQA_ALLOCATOR_CONTROL_SCHEMA = (
    "kvbench-phase3-gqa-allocator-control-observation-2.0.0"
)
MHA_ALLOCATOR_CONTROL_SCHEMA = (
    "kvbench-phase3-mha-allocator-control-observation-2.0.0"
)
PHASE3_ALLOCATOR_CONTROL_SCHEMAS = {
    "gqa": GQA_ALLOCATOR_CONTROL_SCHEMA,
    "mha_control": MHA_ALLOCATOR_CONTROL_SCHEMA,
}
PHASE3_ALLOCATOR_CONTROL_HEADS = 32
PHASE3_ALLOCATOR_CONTROL_GQA_KV_HEADS = 8
PHASE3_ALLOCATOR_CONTROL_MHA_KV_HEADS = 32
PHASE3_ALLOCATOR_CONTROL_LAYERS = 32
PHASE3_ALLOCATOR_CONTROL_HEAD_DIM = 128
PHASE3_ALLOCATOR_CONTROL_QUERY_LENGTH = 1
PHASE3_ALLOCATOR_CONTROL_DTYPE_BYTES = 2
PHASE3_ALLOCATOR_CONTROL_SCALE = PHASE3_ALLOCATOR_CONTROL_HEAD_DIM**-0.5
PHASE3_ALLOCATOR_CONTROL_CAUSAL = False
PHASE3_ALLOCATOR_CONTROL_DROPOUT = 0.0
PHASE3_ALLOCATOR_CONTROL_ENABLE_GQA = True
PHASE3_ALLOCATOR_CONTROL_MAX_BYTES = 256 * 1024 * 1024

# The public entry point is present in the C++ stack of the frozen public SDPA
# call.  The two internal names are accepted only as a pair.  A matching stack
# is still insufficient by itself: exact byte formula and paired multiplicity
# checks are mandatory below.
PUBLIC_FLASH_CPP_FRAME = "at::native::_scaled_dot_product_flash_attention_cuda"
INTERNAL_SPLIT_K_CPP_FRAMES = (
    "pytorch_flash::set_params_splitkv",
    "pytorch_flash::mha_fwd",
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_TORCH: Any | None = None


class Phase3AllocatorControlError(ValueError):
    """Raw allocator-control evidence is absent, malformed, or inconsistent."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Phase3AllocatorControlError(
            "allocator-control evidence is not JSON serializable"
        ) from error


def canonical_phase3_allocator_control_bytes(
    value: Mapping[str, object],
) -> bytes:
    """Serialize an allocator-control observation canonically."""

    if not isinstance(value, Mapping):
        raise Phase3AllocatorControlError(
            "allocator-control payload must be a mapping"
        )
    return _canonical_json_bytes(dict(value))


def _strict_canonical_object(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw:
        raise Phase3AllocatorControlError(
            "allocator-control bytes are absent"
        )
    if len(raw) > PHASE3_ALLOCATOR_CONTROL_MAX_BYTES:
        raise Phase3AllocatorControlError(
            "allocator-control bytes exceed the frozen size bound"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Phase3AllocatorControlError(
            "allocator-control bytes are not UTF-8"
        ) from error

    def reject_constant(value: str) -> None:
        raise Phase3AllocatorControlError(
            f"allocator-control JSON contains non-finite constant {value}"
        )

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Phase3AllocatorControlError(
                    f"allocator-control JSON contains duplicate key {key}"
                )
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except Phase3AllocatorControlError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise Phase3AllocatorControlError(
            "allocator-control bytes are malformed JSON"
        ) from error
    if type(parsed) is not dict:
        raise Phase3AllocatorControlError(
            "allocator-control root must be an object"
        )
    if _canonical_json_bytes(parsed) != raw:
        raise Phase3AllocatorControlError(
            "allocator-control bytes are not canonical"
        )
    return parsed


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise Phase3AllocatorControlError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise Phase3AllocatorControlError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise Phase3AllocatorControlError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise Phase3AllocatorControlError(
            f"{label} must be a nonempty string"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Phase3AllocatorControlError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise Phase3AllocatorControlError(f"{label} must be boolean")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        raise Phase3AllocatorControlError(f"{label} is not a SHA-256 digest")
    return text


def _canonical_contiguous_stride(
    shape: tuple[int, ...]
) -> tuple[int, ...]:
    if not shape or any(type(item) is not int or item <= 0 for item in shape):
        raise Phase3AllocatorControlError(
            "canonical stride requires a positive shape"
        )
    running = 1
    reversed_strides: list[int] = []
    for dimension in reversed(shape):
        reversed_strides.append(running)
        running *= dimension
    return tuple(reversed(reversed_strides))


def _strides_are_contiguous(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> bool:
    if len(shape) != len(stride):
        return False
    expected = 1
    for dimension, observed_stride in zip(
        reversed(shape), reversed(stride), strict=True
    ):
        if dimension != 1 and observed_stride != expected:
            return False
        expected *= dimension
    return True


def _storage_range(
    tensor: AllocatorControlTensorObservation,
) -> tuple[int, int]:
    return (
        tensor.storage_data_ptr,
        tensor.storage_data_ptr + tensor.storage_bytes,
    )


def _require_nonaliasing_control_tensors(
    tensors: Mapping[str, AllocatorControlTensorObservation],
) -> None:
    names = tuple(tensors)
    for index, left_name in enumerate(names):
        left_start, left_end = _storage_range(tensors[left_name])
        for right_name in names[index + 1 :]:
            right_start, right_end = _storage_range(tensors[right_name])
            if left_start < right_end and right_start < left_end:
                raise Phase3AllocatorControlError(
                    "allocator-control tensor storage aliases: "
                    f"{left_name}/{right_name}"
                )


@dataclass(frozen=True, slots=True)
class AllocatorControlGeometry:
    """Frozen held constants for one control."""

    batch: int
    query_heads: int
    kv_heads: int
    context: int
    head_dim: int
    query_length: int
    dtype: str
    dtype_bytes: int
    is_causal: bool
    scale: float
    dropout_p: float
    enable_gqa: bool
    execution_mode: str

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_heads",
            "kv_heads",
            "context",
            "head_dim",
            "query_length",
            "dtype_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise Phase3AllocatorControlError(
                    f"allocator-control {name} must be positive"
                )
        if type(self.scale) is not float or not math.isfinite(self.scale):
            raise Phase3AllocatorControlError(
                "allocator-control scale must be a finite JSON float"
            )
        if type(self.dropout_p) is not float or not math.isfinite(
            self.dropout_p
        ):
            raise Phase3AllocatorControlError(
                "allocator-control dropout must be a finite JSON float"
            )
        if type(self.is_causal) is not bool or type(self.enable_gqa) is not bool:
            raise Phase3AllocatorControlError(
                "allocator-control causal/GQA flags must be boolean"
            )
        if self.execution_mode not in {"eager", "cuda_graph_replay"}:
            raise Phase3AllocatorControlError(
                "allocator-control execution mode is invalid"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> AllocatorControlGeometry:
        _require_exact_keys(
            value,
            {
                "batch",
                "query_heads",
                "kv_heads",
                "context",
                "head_dim",
                "query_length",
                "dtype",
                "dtype_bytes",
                "is_causal",
                "scale",
                "dropout_p",
                "enable_gqa",
                "execution_mode",
            },
            "allocator-control geometry",
        )
        scale = value["scale"]
        dropout = value["dropout_p"]
        if type(scale) is not float or type(dropout) is not float:
            raise Phase3AllocatorControlError(
                "allocator-control scale/dropout must be JSON floats"
            )
        return cls(
            batch=_integer(value["batch"], "geometry.batch", minimum=1),
            query_heads=_integer(
                value["query_heads"], "geometry.query_heads", minimum=1
            ),
            kv_heads=_integer(
                value["kv_heads"], "geometry.kv_heads", minimum=1
            ),
            context=_integer(
                value["context"], "geometry.context", minimum=1
            ),
            head_dim=_integer(
                value["head_dim"], "geometry.head_dim", minimum=1
            ),
            query_length=_integer(
                value["query_length"], "geometry.query_length", minimum=1
            ),
            dtype=_string(value["dtype"], "geometry.dtype"),
            dtype_bytes=_integer(
                value["dtype_bytes"], "geometry.dtype_bytes", minimum=1
            ),
            is_causal=_boolean(
                value["is_causal"], "geometry.is_causal"
            ),
            scale=scale,
            dropout_p=dropout,
            enable_gqa=_boolean(
                value["enable_gqa"], "geometry.enable_gqa"
            ),
            execution_mode=_string(
                value["execution_mode"], "geometry.execution_mode"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch": self.batch,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "context": self.context,
            "head_dim": self.head_dim,
            "query_length": self.query_length,
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "is_causal": self.is_causal,
            "scale": self.scale,
            "dropout_p": self.dropout_p,
            "enable_gqa": self.enable_gqa,
            "execution_mode": self.execution_mode,
        }

    def allocation_geometry(self) -> AllocationGeometry:
        return AllocationGeometry(
            batch=self.batch,
            query_heads=self.query_heads,
            kv_heads=self.kv_heads,
            context=self.context,
            head_dim=self.head_dim,
            dtype_bytes=self.dtype_bytes,
            query_length=self.query_length,
        )


@dataclass(frozen=True, slots=True)
class AllocatorControlTensorObservation:
    """Pointer, storage, shape, and optional content identity."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    element_size: int
    logical_bytes: int
    storage_bytes: int
    storage_offset: int
    data_ptr: int
    storage_data_ptr: int
    is_contiguous: bool
    content_sha256: str | None

    def __post_init__(self) -> None:
        if not self.shape or len(self.shape) != len(self.stride):
            raise Phase3AllocatorControlError(
                "tensor shape/stride ranks differ"
            )
        if any(type(item) is not int or item <= 0 for item in self.shape):
            raise Phase3AllocatorControlError(
                "tensor shape dimensions must be positive"
            )
        if any(type(item) is not int or item < 0 for item in self.stride):
            raise Phase3AllocatorControlError(
                "tensor strides must be nonnegative"
            )
        for name in (
            "element_size",
            "logical_bytes",
            "storage_bytes",
            "data_ptr",
            "storage_data_ptr",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise Phase3AllocatorControlError(
                    f"tensor {name} must be positive"
                )
        if type(self.storage_offset) is not int or self.storage_offset < 0:
            raise Phase3AllocatorControlError(
                "tensor storage offset must be nonnegative"
            )
        if type(self.is_contiguous) is not bool:
            raise Phase3AllocatorControlError(
                "tensor contiguity must be boolean"
            )
        expected_elements = math.prod(self.shape)
        if self.logical_bytes != expected_elements * self.element_size:
            raise Phase3AllocatorControlError(
                "tensor logical byte count differs from shape"
            )
        if self.storage_bytes < self.logical_bytes:
            raise Phase3AllocatorControlError(
                "tensor storage is smaller than logical tensor bytes"
            )
        expected_data_ptr = self.storage_data_ptr + (
            self.storage_offset * self.element_size
        )
        if self.data_ptr != expected_data_ptr:
            raise Phase3AllocatorControlError(
                "tensor data pointer differs from storage offset"
            )

        maximum_element = self.storage_offset + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(
                self.shape, self.stride, strict=True
            )
        )
        if (maximum_element + 1) * self.element_size > self.storage_bytes:
            raise Phase3AllocatorControlError(
                "tensor view extends beyond its recorded storage"
            )
        if self.is_contiguous != _strides_are_contiguous(
            self.shape, self.stride
        ):
            raise Phase3AllocatorControlError(
                "tensor contiguity differs from recorded strides"
            )
        if not self.dtype or not self.device:
            raise Phase3AllocatorControlError(
                "tensor dtype/device must be present"
            )
        if self.content_sha256 is not None and (
            _SHA256_RE.fullmatch(self.content_sha256) is None
        ):
            raise Phase3AllocatorControlError(
                "tensor content identity is not SHA-256"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, label: str
    ) -> AllocatorControlTensorObservation:
        _require_exact_keys(
            value,
            {
                "shape",
                "stride",
                "dtype",
                "device",
                "element_size",
                "logical_bytes",
                "storage_bytes",
                "storage_offset",
                "data_ptr",
                "storage_data_ptr",
                "is_contiguous",
                "content_sha256",
            },
            label,
        )
        shape = tuple(
            _integer(item, f"{label}.shape", minimum=1)
            for item in _array(value["shape"], f"{label}.shape")
        )
        stride = tuple(
            _integer(item, f"{label}.stride")
            for item in _array(value["stride"], f"{label}.stride")
        )
        raw_content = value["content_sha256"]
        content = (
            None
            if raw_content is None
            else _sha256(raw_content, f"{label}.content_sha256")
        )
        return cls(
            shape=shape,
            stride=stride,
            dtype=_string(value["dtype"], f"{label}.dtype"),
            device=_string(value["device"], f"{label}.device"),
            element_size=_integer(
                value["element_size"], f"{label}.element_size", minimum=1
            ),
            logical_bytes=_integer(
                value["logical_bytes"], f"{label}.logical_bytes", minimum=1
            ),
            storage_bytes=_integer(
                value["storage_bytes"], f"{label}.storage_bytes", minimum=1
            ),
            storage_offset=_integer(
                value["storage_offset"], f"{label}.storage_offset"
            ),
            data_ptr=_integer(
                value["data_ptr"], f"{label}.data_ptr", minimum=1
            ),
            storage_data_ptr=_integer(
                value["storage_data_ptr"],
                f"{label}.storage_data_ptr",
                minimum=1,
            ),
            is_contiguous=_boolean(
                value["is_contiguous"], f"{label}.is_contiguous"
            ),
            content_sha256=content,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "element_size": self.element_size,
            "logical_bytes": self.logical_bytes,
            "storage_bytes": self.storage_bytes,
            "storage_offset": self.storage_offset,
            "data_ptr": self.data_ptr,
            "storage_data_ptr": self.storage_data_ptr,
            "is_contiguous": self.is_contiguous,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class AllocatorControlGraphBinding:
    captured: bool
    output_data_ptr: int | None
    capture_stream_id: int | None

    def __post_init__(self) -> None:
        if type(self.captured) is not bool:
            raise Phase3AllocatorControlError(
                "graph capture flag must be boolean"
            )
        if self.captured:
            if (
                type(self.output_data_ptr) is not int
                or self.output_data_ptr <= 0
                or type(self.capture_stream_id) is not int
                or self.capture_stream_id <= 0
            ):
                raise Phase3AllocatorControlError(
                    "captured graph lacks pointer/stream identity"
                )
        elif self.output_data_ptr is not None or self.capture_stream_id is not None:
            raise Phase3AllocatorControlError(
                "eager control cannot carry graph pointer identity"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> AllocatorControlGraphBinding:
        _require_exact_keys(
            value,
            {"captured", "output_data_ptr", "capture_stream_id"},
            "graph replay binding",
        )
        captured = _boolean(value["captured"], "graph_binding.captured")
        if captured:
            output_ptr = _integer(
                value["output_data_ptr"],
                "graph_binding.output_data_ptr",
                minimum=1,
            )
            stream_id = _integer(
                value["capture_stream_id"],
                "graph_binding.capture_stream_id",
                minimum=1,
            )
        else:
            if value["output_data_ptr"] is not None or value[
                "capture_stream_id"
            ] is not None:
                raise Phase3AllocatorControlError(
                    "eager graph binding contains non-null identity"
                )
            output_ptr = None
            stream_id = None
        return cls(captured, output_ptr, stream_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "captured": self.captured,
            "output_data_ptr": self.output_data_ptr,
            "capture_stream_id": self.capture_stream_id,
        }


@dataclass(frozen=True, slots=True)
class Phase3AllocatorControlObservation:
    """One canonical raw control, with no serialized scientific verdict."""

    schema_version: str
    role: AllocatorControlRole
    operation_key: Phase3AuditOperationKey
    operation_fingerprint_sha256: str
    geometry: AllocatorControlGeometry
    backend_identity_sha256: str
    dispatch_trace_sha256: str
    dispatch_trace_size_bytes: int
    query: AllocatorControlTensorObservation
    key: AllocatorControlTensorObservation
    value: AllocatorControlTensorObservation
    query_after: AllocatorControlTensorObservation
    key_after: AllocatorControlTensorObservation
    value_after: AllocatorControlTensorObservation
    output: AllocatorControlTensorObservation
    graph_binding: AllocatorControlGraphBinding
    warmup_iterations: int
    recorder_configuration: dict[str, object]
    allocator_snapshot: dict[str, object]
    allocator_history: tuple[Mapping[str, Any], ...]
    allocator_snapshot_sha256: str
    allocator_history_sha256: str
    memory_stats_before: dict[str, object]
    memory_stats_after: dict[str, object]
    accounting_before: RawMemoryAccountingSample
    accounting_after: RawMemoryAccountingSample
    runtime: dict[str, object]
    collection_started_ns: int
    collection_finished_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "operation_key": self.operation_key.to_dict(),
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
            "geometry": self.geometry.to_dict(),
            "backend_identity_sha256": self.backend_identity_sha256,
            "dispatch_trace_binding": {
                "sha256": self.dispatch_trace_sha256,
                "size_bytes": self.dispatch_trace_size_bytes,
            },
            "tensor_observations": {
                "query": self.query.to_dict(),
                "key": self.key.to_dict(),
                "value": self.value.to_dict(),
            },
            "post_operation_tensor_observations": {
                "query": self.query_after.to_dict(),
                "key": self.key_after.to_dict(),
                "value": self.value_after.to_dict(),
            },
            "output_metadata": self.output.to_dict(),
            "graph_replay_binding": self.graph_binding.to_dict(),
            "warmup_iterations": self.warmup_iterations,
            "recorder_configuration": dict(self.recorder_configuration),
            "allocator_snapshot": self.allocator_snapshot,
            "allocator_history": [dict(item) for item in self.allocator_history],
            "allocator_snapshot_sha256": self.allocator_snapshot_sha256,
            "allocator_history_sha256": self.allocator_history_sha256,
            "memory_stats_before": self.memory_stats_before,
            "memory_stats_after": self.memory_stats_after,
            "device_accounting_before": self.accounting_before.to_dict(),
            "device_accounting_after": self.accounting_after.to_dict(),
            "runtime": dict(self.runtime),
            "collection_started_ns": self.collection_started_ns,
            "collection_finished_ns": self.collection_finished_ns,
            "timing_governance": {
                "allocator_instrumented": True,
                "performance_timing_reported": False,
                "normal_benchmark_timing_eligible": False,
            },
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def _parse_accounting(
    value: Mapping[str, object], *, role: str
) -> RawMemoryAccountingSample:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "operation_fingerprint_sha256",
            "sample_role",
            "timestamp_ns",
            "device",
            "device_index",
            "gpu_uuid",
            "allocated_bytes",
            "reserved_bytes",
            "device_free_bytes",
            "device_total_bytes",
            "device_used_bytes",
        },
        f"device accounting {role}",
    )
    sample = RawMemoryAccountingSample(
        schema_version=_string(
            value["schema_version"], f"accounting.{role}.schema_version"
        ),
        operation_fingerprint_sha256=_sha256(
            value["operation_fingerprint_sha256"],
            f"accounting.{role}.operation_fingerprint_sha256",
        ),
        sample_role=_string(
            value["sample_role"], f"accounting.{role}.sample_role"
        ),
        timestamp_ns=_integer(
            value["timestamp_ns"],
            f"accounting.{role}.timestamp_ns",
            minimum=1,
        ),
        device=_string(value["device"], f"accounting.{role}.device"),
        device_index=_integer(
            value["device_index"], f"accounting.{role}.device_index"
        ),
        gpu_uuid=_string(
            value["gpu_uuid"], f"accounting.{role}.gpu_uuid"
        ),
        allocated_bytes=_integer(
            value["allocated_bytes"], f"accounting.{role}.allocated_bytes"
        ),
        reserved_bytes=_integer(
            value["reserved_bytes"], f"accounting.{role}.reserved_bytes"
        ),
        device_free_bytes=_integer(
            value["device_free_bytes"],
            f"accounting.{role}.device_free_bytes",
        ),
        device_total_bytes=_integer(
            value["device_total_bytes"],
            f"accounting.{role}.device_total_bytes",
            minimum=1,
        ),
    )
    if value["device_used_bytes"] != sample.device_used_bytes:
        raise Phase3AllocatorControlError(
            f"device accounting {role} used-byte derivation differs"
        )
    return sample


def _validate_frozen_observation(
    observation: Phase3AllocatorControlObservation,
) -> None:
    role = observation.role
    expected_schema = PHASE3_ALLOCATOR_CONTROL_SCHEMAS[role]
    if observation.schema_version != expected_schema:
        raise Phase3AllocatorControlError(
            "allocator-control schema differs from its role"
        )
    operation = observation.operation_key
    geometry = observation.geometry
    expected_kv_heads = (
        PHASE3_ALLOCATOR_CONTROL_GQA_KV_HEADS
        if role == "gqa"
        else PHASE3_ALLOCATOR_CONTROL_MHA_KV_HEADS
    )
    expected_geometry = (
        operation.batch_size,
        PHASE3_ALLOCATOR_CONTROL_HEADS,
        expected_kv_heads,
        operation.attended_context,
        PHASE3_ALLOCATOR_CONTROL_HEAD_DIM,
        PHASE3_ALLOCATOR_CONTROL_QUERY_LENGTH,
        PHASE3_OUTPUT_DTYPE,
        PHASE3_ALLOCATOR_CONTROL_DTYPE_BYTES,
        PHASE3_ALLOCATOR_CONTROL_CAUSAL,
        PHASE3_ALLOCATOR_CONTROL_SCALE,
        PHASE3_ALLOCATOR_CONTROL_DROPOUT,
        PHASE3_ALLOCATOR_CONTROL_ENABLE_GQA,
        operation.dispatch_execution_mode,
    )
    observed_geometry = (
        geometry.batch,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.context,
        geometry.head_dim,
        geometry.query_length,
        geometry.dtype,
        geometry.dtype_bytes,
        geometry.is_causal,
        geometry.scale,
        geometry.dropout_p,
        geometry.enable_gqa,
        geometry.execution_mode,
    )
    if observed_geometry != expected_geometry:
        raise Phase3AllocatorControlError(
            "allocator-control geometry differs from the frozen operation"
        )
    if (
        observation.operation_fingerprint_sha256
        != operation.operation_fingerprint_sha256
    ):
        raise Phase3AllocatorControlError(
            "allocator-control operation fingerprint is inconsistent"
        )
    if (
        observation.backend_identity_sha256
        != PHASE3_BACKEND_IDENTITY_SHA256
        or observation.backend_identity_sha256
        != operation.backend_identity_sha256
    ):
        raise Phase3AllocatorControlError(
            "allocator-control backend identity is not the frozen backend"
        )
    if observation.dispatch_trace_size_bytes <= 0:
        raise Phase3AllocatorControlError(
            "allocator-control dispatch trace binding is empty"
        )
    expected_shapes = {
        "query": (
            geometry.batch,
            geometry.query_heads,
            geometry.query_length,
            geometry.head_dim,
        ),
        "key": (
            geometry.batch,
            geometry.kv_heads,
            geometry.context,
            geometry.head_dim,
        ),
        "value": (
            geometry.batch,
            geometry.kv_heads,
            geometry.context,
            geometry.head_dim,
        ),
        "output": (
            geometry.batch,
            geometry.query_heads,
            geometry.query_length,
            geometry.head_dim,
        ),
    }
    tensors = {
        "query": observation.query,
        "key": observation.key,
        "value": observation.value,
        "output": observation.output,
    }
    for name, tensor in tensors.items():
        if (
            tensor.shape != expected_shapes[name]
            or tensor.dtype != PHASE3_OUTPUT_DTYPE
            or tensor.device != PHASE3_DEVICE
            or tensor.element_size != PHASE3_ALLOCATOR_CONTROL_DTYPE_BYTES
        ):
            raise Phase3AllocatorControlError(
                f"allocator-control {name} tensor differs from geometry"
            )
    canonical_query_stride = _canonical_contiguous_stride(
        expected_shapes["query"]
    )
    if (
        observation.query.stride != canonical_query_stride
        or observation.query.storage_offset != 0
        or observation.query.storage_bytes
        != observation.query.logical_bytes
        or not observation.query.is_contiguous
    ):
        raise Phase3AllocatorControlError(
            "allocator-control query layout is not canonical"
        )
    if (
        not _strides_are_contiguous(
            observation.output.shape,
            observation.output.stride,
        )
        or observation.output.storage_offset != 0
        or observation.output.storage_bytes
        != observation.output.logical_bytes
        or not observation.output.is_contiguous
    ):
        raise Phase3AllocatorControlError(
            "allocator-control output layout is not canonical"
        )
    if role == "mha_control":
        expected_kv_stride = _canonical_contiguous_stride(
            expected_shapes["key"]
        )
        for name, tensor in (
            ("key", observation.key),
            ("value", observation.value),
        ):
            if (
                tensor.stride != expected_kv_stride
                or tensor.storage_offset != 0
                or tensor.storage_bytes != tensor.logical_bytes
                or not tensor.is_contiguous
            ):
                raise Phase3AllocatorControlError(
                    f"MHA {name} storage is not exact native-head storage"
                )
    else:
        capacity = operation.capacity
        per_layer_elements = (
            geometry.batch
            * PHASE3_ALLOCATOR_CONTROL_GQA_KV_HEADS
            * capacity
            * PHASE3_ALLOCATOR_CONTROL_HEAD_DIM
        )
        expected_storage_bytes = (
            PHASE3_ALLOCATOR_CONTROL_LAYERS
            * per_layer_elements
            * PHASE3_ALLOCATOR_CONTROL_DTYPE_BYTES
        )
        expected_kv_stride = (
            PHASE3_ALLOCATOR_CONTROL_GQA_KV_HEADS
            * capacity
            * PHASE3_ALLOCATOR_CONTROL_HEAD_DIM,
            capacity * PHASE3_ALLOCATOR_CONTROL_HEAD_DIM,
            PHASE3_ALLOCATOR_CONTROL_HEAD_DIM,
            1,
        )
        for name, tensor in (
            ("key", observation.key),
            ("value", observation.value),
        ):
            layer_index, remainder = divmod(
                tensor.storage_offset, per_layer_elements
            )
            if (
                tensor.storage_bytes != expected_storage_bytes
                or tensor.stride != expected_kv_stride
                or remainder != 0
                or not 0 <= layer_index < PHASE3_ALLOCATOR_CONTROL_LAYERS
                or tensor.is_contiguous
                != (geometry.context == operation.capacity)
            ):
                raise Phase3AllocatorControlError(
                    f"GQA {name} backing is not bounded native-KV cache storage"
                )
        if observation.key.storage_offset != observation.value.storage_offset:
            raise Phase3AllocatorControlError(
                "GQA key/value cache layers differ"
            )
    post_tensors = {
        "query": observation.query_after,
        "key": observation.key_after,
        "value": observation.value_after,
    }
    for name, tensor in post_tensors.items():
        if (
            tensor.shape != expected_shapes[name]
            or tensor.dtype != PHASE3_OUTPUT_DTYPE
            or tensor.device != PHASE3_DEVICE
            or tensor.element_size != PHASE3_ALLOCATOR_CONTROL_DTYPE_BYTES
        ):
            raise Phase3AllocatorControlError(
                f"post-operation allocator-control {name} tensor differs from geometry"
            )
    if any(
        tensor.content_sha256 is None
        for tensor in (
            observation.query,
            observation.key,
            observation.value,
            observation.query_after,
            observation.key_after,
            observation.value_after,
        )
    ):
        raise Phase3AllocatorControlError(
            "allocator-control input tensor lacks content identity"
        )
    if observation.output.content_sha256 is not None:
        raise Phase3AllocatorControlError(
            "allocator-control output cannot carry an input content digest"
        )
    for name, before, after in (
        ("query", observation.query, observation.query_after),
        ("key", observation.key, observation.key_after),
        ("value", observation.value, observation.value_after),
    ):
        if before != after:
            raise Phase3AllocatorControlError(
                "allocator-control operation mutated "
                f"{name} storage, metadata, or content"
            )
    _require_nonaliasing_control_tensors(tensors)
    _require_nonaliasing_control_tensors(post_tensors)
    graph_expected = geometry.execution_mode == "cuda_graph_replay"
    if observation.graph_binding.captured != graph_expected:
        raise Phase3AllocatorControlError(
            "allocator-control graph binding differs from execution mode"
        )
    if graph_expected and (
        observation.graph_binding.output_data_ptr
        != observation.output.data_ptr
    ):
        raise Phase3AllocatorControlError(
            "captured graph output pointer differs from output metadata"
        )
    if observation.recorder_configuration != PHASE3_RECORDER_CONFIGURATION:
        raise Phase3AllocatorControlError(
            "allocator-control recorder configuration is not frozen"
        )
    if observation.warmup_iterations != PHASE3_ALLOCATION_WARMUP_ITERATIONS:
        raise Phase3AllocatorControlError(
            "allocator-control warmup count is not frozen"
        )
    if observation.allocator_snapshot_sha256 != allocator_snapshot_sha256(
        observation.allocator_snapshot
    ):
        raise Phase3AllocatorControlError(
            "allocator-control snapshot digest mismatch"
        )
    if observation.allocator_history_sha256 != allocator_trace_sha256(
        observation.allocator_history
    ):
        raise Phase3AllocatorControlError(
            "allocator-control history digest mismatch"
        )
    if (
        observation.accounting_before.operation_fingerprint_sha256
        != operation.operation_fingerprint_sha256
        or observation.accounting_after.operation_fingerprint_sha256
        != operation.operation_fingerprint_sha256
        or observation.accounting_before.sample_role != "before"
        or observation.accounting_after.sample_role != "after"
    ):
        raise Phase3AllocatorControlError(
            "allocator-control accounting is not operation-bound"
        )
    if not (
        observation.collection_started_ns
        <= observation.accounting_before.timestamp_ns
        <= observation.accounting_after.timestamp_ns
        <= observation.collection_finished_ns
    ):
        raise Phase3AllocatorControlError(
            "allocator-control timestamps are not ordered"
        )
    runtime = observation.runtime
    if runtime != {
        "torch_version": PHASE3_TORCH_VERSION,
        "cuda_runtime_version": PHASE3_CUDA_RUNTIME_VERSION,
        "device": PHASE3_DEVICE,
        "device_index": PHASE3_DEVICE_INDEX,
        "gpu_uuid": observation.accounting_before.gpu_uuid,
    }:
        raise Phase3AllocatorControlError(
            "allocator-control runtime identity is not frozen"
        )
    if observation.accounting_after.gpu_uuid != runtime["gpu_uuid"]:
        raise Phase3AllocatorControlError(
            "allocator-control accounting GPU identities differ"
        )


def parse_phase3_allocator_control_bytes(
    raw: bytes,
) -> Phase3AllocatorControlObservation:
    """Parse one bounded, canonical, duplicate-free control observation."""

    payload = _strict_canonical_object(raw)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "role",
            "operation_key",
            "operation_fingerprint_sha256",
            "geometry",
            "backend_identity_sha256",
            "dispatch_trace_binding",
            "tensor_observations",
            "post_operation_tensor_observations",
            "output_metadata",
            "graph_replay_binding",
            "warmup_iterations",
            "recorder_configuration",
            "allocator_snapshot",
            "allocator_history",
            "allocator_snapshot_sha256",
            "allocator_history_sha256",
            "memory_stats_before",
            "memory_stats_after",
            "device_accounting_before",
            "device_accounting_after",
            "runtime",
            "collection_started_ns",
            "collection_finished_ns",
            "timing_governance",
        },
        "allocator-control observation",
    )
    role = _string(payload["role"], "allocator-control role")
    if role not in PHASE3_ALLOCATOR_CONTROL_SCHEMAS:
        raise Phase3AllocatorControlError(
            "allocator-control role is not gqa or mha_control"
        )
    dispatch = _object(payload["dispatch_trace_binding"], "dispatch binding")
    _require_exact_keys(
        dispatch, {"sha256", "size_bytes"}, "dispatch binding"
    )
    tensors = _object(payload["tensor_observations"], "tensor observations")
    _require_exact_keys(tensors, {"query", "key", "value"}, "tensor observations")
    post_tensors = _object(
        payload["post_operation_tensor_observations"],
        "post-operation tensor observations",
    )
    _require_exact_keys(
        post_tensors,
        {"query", "key", "value"},
        "post-operation tensor observations",
    )
    timing = _object(payload["timing_governance"], "timing governance")
    _require_exact_keys(
        timing,
        {
            "allocator_instrumented",
            "performance_timing_reported",
            "normal_benchmark_timing_eligible",
        },
        "timing governance",
    )
    if timing != {
        "allocator_instrumented": True,
        "performance_timing_reported": False,
        "normal_benchmark_timing_eligible": False,
    }:
        raise Phase3AllocatorControlError(
            "allocator-control timing governance is invalid"
        )
    try:
        operation = Phase3AuditOperationKey.from_dict(
            _object(payload["operation_key"], "operation key")
        )
    except (SchemaValidationError, TypeError, ValueError) as error:
        raise Phase3AllocatorControlError(
            "allocator-control operation key is invalid"
        ) from error
    snapshot = _object(payload["allocator_snapshot"], "allocator snapshot")
    history_values = _array(payload["allocator_history"], "allocator history")
    history: list[Mapping[str, Any]] = []
    for item in history_values:
        history.append(_object(item, "allocator history event"))
    stats_before = _object(payload["memory_stats_before"], "memory stats before")
    stats_after = _object(payload["memory_stats_after"], "memory stats after")
    runtime = _object(payload["runtime"], "runtime identity")
    _require_exact_keys(
        runtime,
        {
            "torch_version",
            "cuda_runtime_version",
            "device",
            "device_index",
            "gpu_uuid",
        },
        "runtime identity",
    )
    observation = Phase3AllocatorControlObservation(
        schema_version=_string(payload["schema_version"], "schema version"),
        role=role,  # type: ignore[arg-type]
        operation_key=operation,
        operation_fingerprint_sha256=_sha256(
            payload["operation_fingerprint_sha256"],
            "operation fingerprint",
        ),
        geometry=AllocatorControlGeometry.from_mapping(
            _object(payload["geometry"], "geometry")
        ),
        backend_identity_sha256=_sha256(
            payload["backend_identity_sha256"], "backend identity"
        ),
        dispatch_trace_sha256=_sha256(
            dispatch["sha256"], "dispatch trace digest"
        ),
        dispatch_trace_size_bytes=_integer(
            dispatch["size_bytes"], "dispatch trace size", minimum=1
        ),
        query=AllocatorControlTensorObservation.from_mapping(
            _object(tensors["query"], "query tensor"), label="query tensor"
        ),
        key=AllocatorControlTensorObservation.from_mapping(
            _object(tensors["key"], "key tensor"), label="key tensor"
        ),
        value=AllocatorControlTensorObservation.from_mapping(
            _object(tensors["value"], "value tensor"), label="value tensor"
        ),
        query_after=AllocatorControlTensorObservation.from_mapping(
            _object(post_tensors["query"], "post-operation query tensor"),
            label="post-operation query tensor",
        ),
        key_after=AllocatorControlTensorObservation.from_mapping(
            _object(post_tensors["key"], "post-operation key tensor"),
            label="post-operation key tensor",
        ),
        value_after=AllocatorControlTensorObservation.from_mapping(
            _object(post_tensors["value"], "post-operation value tensor"),
            label="post-operation value tensor",
        ),
        output=AllocatorControlTensorObservation.from_mapping(
            _object(payload["output_metadata"], "output metadata"),
            label="output metadata",
        ),
        graph_binding=AllocatorControlGraphBinding.from_mapping(
            _object(payload["graph_replay_binding"], "graph replay binding")
        ),
        warmup_iterations=_integer(
            payload["warmup_iterations"], "warmup iterations", minimum=1
        ),
        recorder_configuration=_object(
            payload["recorder_configuration"], "recorder configuration"
        ),
        allocator_snapshot=snapshot,
        allocator_history=tuple(history),
        allocator_snapshot_sha256=_sha256(
            payload["allocator_snapshot_sha256"], "snapshot digest"
        ),
        allocator_history_sha256=_sha256(
            payload["allocator_history_sha256"], "history digest"
        ),
        memory_stats_before=stats_before,
        memory_stats_after=stats_after,
        accounting_before=_parse_accounting(
            _object(
                payload["device_accounting_before"],
                "device accounting before",
            ),
            role="before",
        ),
        accounting_after=_parse_accounting(
            _object(
                payload["device_accounting_after"],
                "device accounting after",
            ),
            role="after",
        ),
        runtime=runtime,
        collection_started_ns=_integer(
            payload["collection_started_ns"],
            "collection start timestamp",
            minimum=1,
        ),
        collection_finished_ns=_integer(
            payload["collection_finished_ns"],
            "collection finish timestamp",
            minimum=1,
        ),
    )
    _validate_frozen_observation(observation)
    return observation


@dataclass(frozen=True, slots=True)
class AllocatorControlAllocationFact:
    allocation_id: int
    requested_bytes: int
    formula_id: str
    num_splits: int | None


@dataclass(frozen=True, slots=True)
class Phase3AllocatorControlReplay:
    observation: Phase3AllocatorControlObservation
    geometry: AllocationGeometry
    counters: AllocatorCounterEvidence
    memory: MemoryDeltaEvidence
    attribution: AllocatorTraceAttribution
    allocation_facts: tuple[AllocatorControlAllocationFact, ...]
    failure_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failure_reasons

    @property
    def split_k_facts(self) -> tuple[AllocatorControlAllocationFact, ...]:
        return tuple(
            fact
            for fact in self.allocation_facts
            if fact.formula_id.startswith("flash_split_k_")
        )


def _frame_matches(name: str, marker: str) -> bool:
    return name == marker or name.startswith(marker + "(") or name.startswith(
        marker + "<"
    )


def _has_exact_cpp_frame(allocation: Any, marker: str) -> bool:
    return any(
        _frame_matches(frame.name, marker)
        for frame in allocation.cpp_stack
    )


def _has_flash_stack(allocation: Any) -> bool:
    return _has_exact_cpp_frame(
        allocation, PUBLIC_FLASH_CPP_FRAME
    ) or _has_exact_cpp_frame(
        allocation, INTERNAL_SPLIT_K_CPP_FRAMES[1]
    )


def _has_split_k_stack(allocation: Any) -> bool:
    return all(
        _has_exact_cpp_frame(allocation, marker)
        for marker in INTERNAL_SPLIT_K_CPP_FRAMES
    )


def _allocation_formula(
    allocation: Any,
    geometry: AllocationGeometry,
    *,
    role: AllocatorControlRole,
) -> tuple[str, int | None]:
    size = allocation.requested_bytes
    flash_stack = _has_flash_stack(allocation)
    if role == "gqa" and size in {
        geometry.expanded_kv_single_bytes,
        geometry.expanded_kv_combined_bytes,
    }:
        return "expanded_kv", None
    if size in {
        geometry.native_kv_single_bytes,
        geometry.native_kv_combined_bytes,
    }:
        return "native_or_context_kv", None
    if size == 8 and flash_stack:
        return "flash_fixed_scalar_8", None
    if size == 16 and flash_stack:
        return "flash_fixed_scalar_16", None
    if size == geometry.output_bytes and flash_stack:
        return "fixed_attention_output", None
    if size == geometry.flash_lse_bytes and flash_stack:
        return "fixed_flash_lse", None
    output_base = geometry.flash_split_k_output_accumulator_bytes(1)
    lse_base = geometry.flash_split_k_lse_bytes(1)
    if _has_split_k_stack(allocation):
        if size % output_base == 0 and size // output_base > 1:
            return "flash_split_k_output_accumulator", size // output_base
        if size % lse_base == 0 and size // lse_base > 1:
            return "flash_split_k_lse", size // lse_base
    context_plane = (
        geometry.batch
        * geometry.context
        * geometry.head_dim
        * geometry.dtype_bytes
    )
    if size % context_plane == 0:
        return "context_scaled_unknown", None
    return AllocationClass.UNKNOWN.value, None


_EAGER_REQUIRED_FORMULAS = (
    "flash_fixed_scalar_8",
    "flash_fixed_scalar_16",
    "fixed_attention_output",
    "fixed_flash_lse",
)
_EAGER_SPLIT_FORMULAS = (
    "flash_split_k_output_accumulator",
    "flash_split_k_lse",
)


def _eager_formula_failure_reasons(
    facts: Sequence[AllocatorControlAllocationFact],
) -> tuple[str, ...]:
    counts = Counter(fact.formula_id for fact in facts)
    reasons: list[str] = []
    for formula in _EAGER_REQUIRED_FORMULAS:
        observed = counts[formula]
        if observed == 0:
            reasons.append(f"eager_expected_allocation_missing:{formula}")
        elif observed != 1:
            reasons.append(f"eager_expected_allocation_duplicate:{formula}")
    split_output = tuple(
        fact
        for fact in facts
        if fact.formula_id == "flash_split_k_output_accumulator"
    )
    split_lse = tuple(
        fact
        for fact in facts
        if fact.formula_id == "flash_split_k_lse"
    )
    if not split_output and not split_lse:
        reasons.append("eager_split_k_pair_missing")
    elif bool(split_output) != bool(split_lse):
        reasons.append("eager_split_k_pair_incomplete")
    else:
        if len(split_output) != 1 or len(split_lse) != 1:
            reasons.append("eager_split_k_pair_duplicate")
        elif (
            split_output[0].num_splits is None
            or split_output[0].num_splits != split_lse[0].num_splits
        ):
            reasons.append("eager_split_k_pair_formula_mismatch")
    allowed = frozenset((*_EAGER_REQUIRED_FORMULAS, *_EAGER_SPLIT_FORMULAS))
    if any(fact.formula_id not in allowed for fact in facts):
        reasons.append("eager_allocation_set_contains_unexpected_formula")
    expected_total = len(_EAGER_REQUIRED_FORMULAS) + len(
        _EAGER_SPLIT_FORMULAS
    )
    if len(facts) != expected_total:
        reasons.append("eager_allocation_set_cardinality_mismatch")
    return tuple(dict.fromkeys(reasons))


def replay_phase3_allocator_control(
    raw: bytes,
    *,
    expected_operation_key: Phase3AuditOperationKey,
    dispatch_trace_raw: bytes,
) -> Phase3AllocatorControlReplay:
    """Independently reconstruct one control from its raw observations."""

    if type(expected_operation_key) is not Phase3AuditOperationKey:
        raise Phase3AllocatorControlError(
            "expected operation key has the wrong type"
        )
    if type(dispatch_trace_raw) is not bytes or not dispatch_trace_raw:
        raise Phase3AllocatorControlError("dispatch trace bytes are absent")
    observation = parse_phase3_allocator_control_bytes(raw)
    if observation.operation_key != expected_operation_key:
        raise Phase3AllocatorControlError(
            "allocator control differs from the expected operation key"
        )
    if (
        len(dispatch_trace_raw) != observation.dispatch_trace_size_bytes
        or hashlib.sha256(dispatch_trace_raw).hexdigest()
        != observation.dispatch_trace_sha256
    ):
        raise Phase3AllocatorControlError(
            "allocator control is not bound to the supplied dispatch trace"
        )
    try:
        selected_trace = allocator_trace_from_snapshot(
            observation.allocator_snapshot, PHASE3_DEVICE_INDEX
        )
    except (TypeError, ValueError) as error:
        raise Phase3AllocatorControlError(
            "allocator-control snapshot trace cannot be reconstructed"
        ) from error
    if selected_trace != list(observation.allocator_history):
        raise Phase3AllocatorControlError(
            "allocator history differs from the selected snapshot trace"
        )
    geometry = observation.geometry.allocation_geometry()
    counters = allocator_counters_from_memory_stats(
        observation.memory_stats_before,
        observation.memory_stats_after,
    )
    memory = memory_delta_from_raw_samples(
        observation.accounting_before, observation.accounting_after
    )
    history = build_history_integrity_evidence(
        observation.allocator_snapshot,
        observation.allocator_history,
        max_entries=PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
        stack_mode="all",
        expected_snapshot_sha256=observation.allocator_snapshot_sha256,
        expected_trace_sha256=observation.allocator_history_sha256,
    )
    rules = AttributionRules(
        frozen_backend_identity=observation.backend_identity_sha256,
        flash_split_k_cpp_markers=(PUBLIC_FLASH_CPP_FRAME,),
    )
    attribution = attribute_allocator_trace(
        observation.allocator_history,
        geometry=geometry,
        counters=counters,
        rules=rules,
        backend_identity=observation.backend_identity_sha256,
        expected_trace_sha256=observation.allocator_history_sha256,
        history_integrity=history,
    )
    facts: list[AllocatorControlAllocationFact] = []
    reasons: list[str] = []
    if attribution.integrity_errors:
        reasons.append("allocator_trace_integrity_failure")
    reasons.extend(history.failure_reasons())
    if not counters.complete:
        reasons.append("allocator_counter_evidence_incomplete")
    if not attribution.all_block_sizes_proven:
        reasons.append("allocated_block_size_evidence_incomplete")
    if not attribution.all_lifetimes_fully_freed:
        reasons.append("allocation_lifetime_not_fully_freed")
    if any(
        not item.python_stack or not item.cpp_stack
        for item in attribution.allocations
    ):
        reasons.append("allocator_allocation_stack_incomplete")
    if attribution.segment_alloc_count or attribution.segment_free_count:
        reasons.append("segment_alloc_or_free_detected")
    if counters.segment_allocation_count != 0 or counters.segment_free_count != 0:
        reasons.append("segment_counter_nonzero_or_unavailable")
    if counters.device_allocation_count != 0 or counters.device_free_count != 0:
        reasons.append("device_allocation_or_free_detected_or_unavailable")
    if counters.allocation_retry_count != 0 or counters.oom_count != 0:
        reasons.append("allocator_retry_or_oom_detected_or_unavailable")
    if (
        memory.allocated_delta != 0
        or memory.reserved_delta != 0
        or memory.device_used_delta != 0
        or memory.non_pytorch_delta != 0
    ):
        reasons.append("persistent_or_non_pytorch_memory_delta_nonzero")

    for allocation in attribution.allocations:
        formula, splits = _allocation_formula(
            allocation, geometry, role=observation.role
        )
        facts.append(
            AllocatorControlAllocationFact(
                allocation_id=allocation.allocation_id,
                requested_bytes=allocation.requested_bytes,
                formula_id=formula,
                num_splits=splits,
            )
        )
        if formula == "expanded_kv":
            reasons.append("expanded_kv_allocation_detected")
        elif formula == "native_or_context_kv":
            reasons.append("native_or_context_kv_allocation_detected")
        elif formula == "context_scaled_unknown":
            reasons.append("context_scaled_unknown_allocation_detected")
        elif formula == AllocationClass.UNKNOWN.value:
            reasons.append("unknown_allocation_detected")
    if observation.geometry.execution_mode == "cuda_graph_replay":
        graph = evaluate_strict_graph_criterion(attribution, memory)
        reasons.extend(graph.failure_reasons)
    else:
        reasons.extend(_eager_formula_failure_reasons(facts))
        output_facts = tuple(
            fact
            for fact in facts
            if fact.formula_id == "fixed_attention_output"
        )
        if len(output_facts) == 1:
            output_allocation = next(
                item
                for item in attribution.allocations
                if item.allocation_id == output_facts[0].allocation_id
            )
            if output_allocation.address != observation.output.data_ptr:
                reasons.append("fixed_attention_output_pointer_mismatch")
        if not attribution.all_allocations_cache_reused:
            reasons.append("eager_allocation_not_reused_from_cache")
    return Phase3AllocatorControlReplay(
        observation=observation,
        geometry=geometry,
        counters=counters,
        memory=memory,
        attribution=attribution,
        allocation_facts=tuple(facts),
        failure_reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class Phase3PairedAllocatorControlVerification:
    """Independent paired result; this is not a production gate decision."""

    gqa: Phase3AllocatorControlReplay
    mha_control: Phase3AllocatorControlReplay
    split_k_pair_multiplicity: tuple[tuple[int, int], ...]
    failure_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failure_reasons


def _held_constants(geometry: AllocatorControlGeometry) -> tuple[object, ...]:
    return (
        geometry.batch,
        geometry.query_heads,
        geometry.context,
        geometry.head_dim,
        geometry.query_length,
        geometry.dtype,
        geometry.dtype_bytes,
        geometry.is_causal,
        geometry.scale,
        geometry.dropout_p,
        geometry.enable_gqa,
        geometry.execution_mode,
    )


def _non_split_formula_counter(
    replay: Phase3AllocatorControlReplay,
) -> Counter[tuple[str, int]]:
    return Counter(
        (fact.formula_id, fact.requested_bytes)
        for fact in replay.allocation_facts
        if (
            fact.formula_id != AllocationClass.UNKNOWN.value
            and fact.formula_id not in _EAGER_SPLIT_FORMULAS
        )
    )


def verify_phase3_paired_allocator_controls(
    *,
    gqa_raw: bytes,
    mha_control_raw: bytes,
    operation_key: Phase3AuditOperationKey,
    gqa_dispatch_trace_raw: bytes,
    mha_dispatch_trace_raw: bytes,
) -> Phase3PairedAllocatorControlVerification:
    """Verify held constants and each geometry's exact split-K formula pair."""

    gqa = replay_phase3_allocator_control(
        gqa_raw,
        expected_operation_key=operation_key,
        dispatch_trace_raw=gqa_dispatch_trace_raw,
    )
    mha = replay_phase3_allocator_control(
        mha_control_raw,
        expected_operation_key=operation_key,
        dispatch_trace_raw=mha_dispatch_trace_raw,
    )
    reasons: list[str] = []
    if gqa.observation.role != "gqa" or mha.observation.role != "mha_control":
        reasons.append("allocator_control_roles_mismatch")
    if _held_constants(gqa.observation.geometry) != _held_constants(
        mha.observation.geometry
    ):
        reasons.append("allocator_control_held_constants_mismatch")
    if (
        gqa.observation.geometry.query_heads
        != PHASE3_ALLOCATOR_CONTROL_HEADS
        or gqa.observation.geometry.kv_heads
        != PHASE3_ALLOCATOR_CONTROL_GQA_KV_HEADS
        or mha.observation.geometry.query_heads
        != PHASE3_ALLOCATOR_CONTROL_HEADS
        or mha.observation.geometry.kv_heads
        != PHASE3_ALLOCATOR_CONTROL_MHA_KV_HEADS
    ):
        reasons.append("allocator_control_geometry_mismatch")
    if gqa.observation.backend_identity_sha256 != (
        mha.observation.backend_identity_sha256
    ):
        reasons.append("allocator_control_backend_identity_mismatch")
    if gqa.observation.runtime != mha.observation.runtime:
        reasons.append("allocator_control_runtime_identity_mismatch")
    if gqa.observation.recorder_configuration != (
        mha.observation.recorder_configuration
    ):
        reasons.append("allocator_control_recorder_configuration_mismatch")
    if gqa.observation.query != mha.observation.query:
        reasons.append("allocator_control_query_identity_or_content_mismatch")
    for prefix, replay in (("gqa", gqa), ("mha_control", mha)):
        reasons.extend(
            f"{prefix}:{reason}" for reason in replay.failure_reasons
        )

    def split_pair_multiplicity(
        prefix: str,
        replay: Phase3AllocatorControlReplay,
    ) -> tuple[tuple[int, int], ...]:
        multiplicities: list[tuple[int, int]] = []
        split_values = sorted(
            {
                fact.num_splits
                for fact in replay.split_k_facts
                if fact.num_splits is not None
            }
        )
        for splits in split_values:
            output_count = sum(
                1
                for fact in replay.split_k_facts
                if fact.formula_id
                == "flash_split_k_output_accumulator"
                and fact.num_splits == splits
            )
            lse_count = sum(
                1
                for fact in replay.split_k_facts
                if fact.formula_id == "flash_split_k_lse"
                and fact.num_splits == splits
            )
            if output_count <= 0 or output_count != lse_count:
                reasons.append(
                    f"{prefix}_split_k_output_lse_pair_mismatch"
                )
            multiplicities.append((splits, output_count))
        return tuple(multiplicities)

    gqa_multiplicities = split_pair_multiplicity("gqa", gqa)
    split_pair_multiplicity("mha_control", mha)
    if _non_split_formula_counter(gqa) != _non_split_formula_counter(mha):
        reasons.append("allocator_control_non_split_formula_mismatch")
    return Phase3PairedAllocatorControlVerification(
        gqa=gqa,
        mha_control=mha,
        # The production endpoint is GQA, so its independently verified split
        # count is the one bound into production allocation replay.
        split_k_pair_multiplicity=gqa_multiplicities,
        failure_reasons=tuple(dict.fromkeys(reasons)),
    )


def _torch_module() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise Phase3AllocatorControlError(
                "PyTorch is unavailable for allocator controls"
            ) from error
    return _TORCH


def _tensor_content_sha256(torch: Any, tensor: Any) -> str:
    header = _canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": [int(item) for item in tensor.shape]}
    )
    contiguous = tensor.detach().contiguous()
    host_bytes = contiguous.view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(header + b"\0" + host_bytes).hexdigest()


def _tensor_observation(
    torch: Any, tensor: Any, *, include_content: bool
) -> AllocatorControlTensorObservation:
    if not isinstance(tensor, torch.Tensor):
        raise Phase3AllocatorControlError(
            "allocator-control tensor input is not a tensor"
        )
    storage = tensor.untyped_storage()
    return AllocatorControlTensorObservation(
        shape=tuple(int(item) for item in tensor.shape),
        stride=tuple(int(item) for item in tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        element_size=int(tensor.element_size()),
        logical_bytes=int(tensor.numel()) * int(tensor.element_size()),
        storage_bytes=int(storage.nbytes()),
        storage_offset=int(tensor.storage_offset()),
        data_ptr=int(tensor.data_ptr()),
        storage_data_ptr=int(storage.data_ptr()),
        is_contiguous=bool(tensor.is_contiguous()),
        content_sha256=(
            _tensor_content_sha256(torch, tensor) if include_content else None
        ),
    )


def _memory_accounting_sample(
    torch: Any,
    device: Any,
    *,
    operation_fingerprint_sha256: str,
    sample_role: str,
    gpu_uuid: str,
) -> RawMemoryAccountingSample:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    except Exception as error:
        raise Phase3AllocatorControlError(
            "device memory accounting is unavailable"
        ) from error
    return RawMemoryAccountingSample(
        schema_version="kvbench-phase3-memory-accounting-2.0.0",
        operation_fingerprint_sha256=operation_fingerprint_sha256,
        sample_role=sample_role,
        timestamp_ns=time.time_ns(),
        device=str(device),
        device_index=PHASE3_DEVICE_INDEX,
        gpu_uuid=gpu_uuid,
        allocated_bytes=int(torch.cuda.memory_allocated(device=device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device=device)),
        device_free_bytes=int(free_bytes),
        device_total_bytes=int(total_bytes),
    )


def _validate_live_control_tensors(
    torch: Any,
    *,
    role: AllocatorControlRole,
    operation_key: Phase3AuditOperationKey,
    query: Any,
    key: Any,
    value: Any,
) -> AllocatorControlGeometry:
    if any(not isinstance(item, torch.Tensor) for item in (query, key, value)):
        raise Phase3AllocatorControlError("Q/K/V controls must be tensors")
    kv_heads = (
        PHASE3_ALLOCATOR_CONTROL_GQA_KV_HEADS
        if role == "gqa"
        else PHASE3_ALLOCATOR_CONTROL_MHA_KV_HEADS
    )
    expected = (
        (
            operation_key.batch_size,
            PHASE3_ALLOCATOR_CONTROL_HEADS,
            PHASE3_ALLOCATOR_CONTROL_QUERY_LENGTH,
            PHASE3_ALLOCATOR_CONTROL_HEAD_DIM,
        ),
        (
            operation_key.batch_size,
            kv_heads,
            operation_key.attended_context,
            PHASE3_ALLOCATOR_CONTROL_HEAD_DIM,
        ),
    )
    if tuple(query.shape) != expected[0] or tuple(key.shape) != expected[1] or (
        tuple(value.shape) != expected[1]
    ):
        raise Phase3AllocatorControlError(
            "live allocator-control tensors differ from frozen geometry"
        )
    if any(
        item.dtype != torch.bfloat16 or str(item.device) != PHASE3_DEVICE
        for item in (query, key, value)
    ):
        raise Phase3AllocatorControlError(
            "live allocator-control tensors must be BF16 on cuda:0"
        )
    if key.data_ptr() == value.data_ptr():
        raise Phase3AllocatorControlError(
            "live allocator-control K/V pointers alias"
        )
    return AllocatorControlGeometry(
        batch=operation_key.batch_size,
        query_heads=PHASE3_ALLOCATOR_CONTROL_HEADS,
        kv_heads=kv_heads,
        context=operation_key.attended_context,
        head_dim=PHASE3_ALLOCATOR_CONTROL_HEAD_DIM,
        query_length=PHASE3_ALLOCATOR_CONTROL_QUERY_LENGTH,
        dtype=PHASE3_OUTPUT_DTYPE,
        dtype_bytes=PHASE3_ALLOCATOR_CONTROL_DTYPE_BYTES,
        is_causal=PHASE3_ALLOCATOR_CONTROL_CAUSAL,
        scale=float(PHASE3_ALLOCATOR_CONTROL_SCALE),
        dropout_p=float(PHASE3_ALLOCATOR_CONTROL_DROPOUT),
        enable_gqa=PHASE3_ALLOCATOR_CONTROL_ENABLE_GQA,
        execution_mode=operation_key.dispatch_execution_mode,
    )


def _collect_one_control(
    *,
    torch: Any,
    role: AllocatorControlRole,
    operation_key: Phase3AuditOperationKey,
    query: Any,
    key: Any,
    value: Any,
    dispatch_trace_raw: bytes,
    runtime_identity: Mapping[str, object],
) -> bytes:
    geometry = _validate_live_control_tensors(
        torch,
        role=role,
        operation_key=operation_key,
        query=query,
        key=key,
        value=value,
    )
    if type(dispatch_trace_raw) is not bytes or not dispatch_trace_raw:
        raise Phase3AllocatorControlError("dispatch trace bytes are absent")
    query_observation = _tensor_observation(torch, query, include_content=True)
    key_observation = _tensor_observation(torch, key, include_content=True)
    value_observation = _tensor_observation(torch, value, include_content=True)
    selected = torch.device(PHASE3_DEVICE)
    properties = torch.cuda.get_device_properties(PHASE3_DEVICE_INDEX)
    gpu_uuid = str(getattr(properties, "uuid", ""))
    if not gpu_uuid:
        raise Phase3AllocatorControlError(
            "allocator control requires a GPU UUID"
        )

    def direct_operation() -> Any:
        with forced_flash_execution():
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=PHASE3_ALLOCATOR_CONTROL_DROPOUT,
                is_causal=PHASE3_ALLOCATOR_CONTROL_CAUSAL,
                scale=float(PHASE3_ALLOCATOR_CONTROL_SCALE),
                enable_gqa=PHASE3_ALLOCATOR_CONTROL_ENABLE_GQA,
            )

    graph: CapturedFixedGraph | None = None
    if geometry.execution_mode == "cuda_graph_replay":
        graph = capture_fixed_graph(
            direct_operation,
            warmup_steps=PHASE3_ALLOCATION_WARMUP_ITERATIONS,
            device=selected,
        )
        operation = graph.replay
    else:
        operation = direct_operation
    for _ in range(PHASE3_ALLOCATION_WARMUP_ITERATIONS):
        warmup_result = operation()
        del warmup_result
    torch.cuda.synchronize(device=selected)

    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
    snapshot_function = getattr(torch.cuda.memory, "_snapshot", None)
    if not callable(recorder) or not callable(snapshot_function):
        raise Phase3AllocatorControlError(
            "PyTorch allocator history APIs are unavailable"
        )
    started = time.time_ns()
    accounting_before = _memory_accounting_sample(
        torch,
        selected,
        operation_fingerprint_sha256=(
            operation_key.operation_fingerprint_sha256
        ),
        sample_role="before",
        gpu_uuid=gpu_uuid,
    )
    stats_before = dict(torch.cuda.memory_stats(device=selected))
    recorder(
        enabled=PHASE3_RECORDER_CONFIGURATION["enabled"],
        context=PHASE3_RECORDER_CONFIGURATION["context"],
        stacks=PHASE3_RECORDER_CONFIGURATION["stacks"],
        max_entries=PHASE3_RECORDER_CONFIGURATION["max_entries"],
        device=selected,
        clear_history=PHASE3_RECORDER_CONFIGURATION["clear_history"],
    )
    recorder_enabled = True
    try:
        output = operation()
        torch.cuda.synchronize(device=selected)
        output_observation = _tensor_observation(
            torch, output, include_content=False
        )
        if graph is None:
            del output
            torch.cuda.synchronize(device=selected)
        stats_after = dict(torch.cuda.memory_stats(device=selected))
        accounting_after = _memory_accounting_sample(
            torch,
            selected,
            operation_fingerprint_sha256=(
                operation_key.operation_fingerprint_sha256
            ),
            sample_role="after",
            gpu_uuid=gpu_uuid,
        )
        snapshot = snapshot_function(device=selected)
        if not isinstance(snapshot, Mapping):
            raise Phase3AllocatorControlError(
                "allocator-control snapshot is malformed"
            )
        snapshot_dict = dict(snapshot)
    finally:
        if recorder_enabled:
            recorder(enabled=None, device=selected)
    query_after = _tensor_observation(torch, query, include_content=True)
    key_after = _tensor_observation(torch, key, include_content=True)
    value_after = _tensor_observation(torch, value, include_content=True)
    finished = time.time_ns()
    history = allocator_trace_from_snapshot(
        snapshot_dict, PHASE3_DEVICE_INDEX
    )
    graph_binding = AllocatorControlGraphBinding(
        captured=graph is not None,
        output_data_ptr=(None if graph is None else graph.output_data_ptr),
        capture_stream_id=(None if graph is None else graph.capture_stream_id),
    )
    observation = Phase3AllocatorControlObservation(
        schema_version=PHASE3_ALLOCATOR_CONTROL_SCHEMAS[role],
        role=role,
        operation_key=operation_key,
        operation_fingerprint_sha256=(
            operation_key.operation_fingerprint_sha256
        ),
        geometry=geometry,
        backend_identity_sha256=PHASE3_BACKEND_IDENTITY_SHA256,
        dispatch_trace_sha256=hashlib.sha256(dispatch_trace_raw).hexdigest(),
        dispatch_trace_size_bytes=len(dispatch_trace_raw),
        query=query_observation,
        key=key_observation,
        value=value_observation,
        query_after=query_after,
        key_after=key_after,
        value_after=value_after,
        output=output_observation,
        graph_binding=graph_binding,
        warmup_iterations=PHASE3_ALLOCATION_WARMUP_ITERATIONS,
        recorder_configuration=dict(PHASE3_RECORDER_CONFIGURATION),
        allocator_snapshot=snapshot_dict,
        allocator_history=tuple(history),
        allocator_snapshot_sha256=allocator_snapshot_sha256(snapshot_dict),
        allocator_history_sha256=allocator_trace_sha256(history),
        memory_stats_before=stats_before,
        memory_stats_after=stats_after,
        accounting_before=accounting_before,
        accounting_after=accounting_after,
        runtime=dict(runtime_identity),
        collection_started_ns=started,
        collection_finished_ns=finished,
    )
    _validate_frozen_observation(observation)
    return observation.canonical_bytes()


def collect_phase3_paired_allocator_controls(
    *,
    operation_key: Phase3AuditOperationKey,
    query: Any,
    gqa_key: Any,
    gqa_value: Any,
    mha_key: Any,
    mha_value: Any,
    gqa_dispatch_trace_raw: bytes,
    mha_dispatch_trace_raw: bytes,
) -> tuple[bytes, bytes]:
    """Collect untimed, held-constant raw GQA/MHA allocator observations."""

    if type(operation_key) is not Phase3AuditOperationKey:
        raise Phase3AllocatorControlError(
            "operation_key must be Phase3AuditOperationKey"
        )
    if operation_key.backend_identity_sha256 != PHASE3_BACKEND_IDENTITY_SHA256:
        raise Phase3AllocatorControlError(
            "operation key does not bind the frozen backend"
        )
    torch = _torch_module()
    identity_raw = _canonical_json_bytes(backend_identity())
    if hashlib.sha256(identity_raw).hexdigest() != PHASE3_BACKEND_IDENTITY_SHA256:
        raise Phase3AllocatorControlError(
            "live backend identity differs from the frozen digest"
        )
    if str(torch.__version__) != PHASE3_TORCH_VERSION or str(
        torch.version.cuda
    ) != PHASE3_CUDA_RUNTIME_VERSION:
        raise Phase3AllocatorControlError(
            "live PyTorch/CUDA runtime differs from the frozen build"
        )
    if query is None:
        raise Phase3AllocatorControlError("shared query tensor is absent")
    properties = torch.cuda.get_device_properties(PHASE3_DEVICE_INDEX)
    runtime = {
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "device": PHASE3_DEVICE,
        "device_index": PHASE3_DEVICE_INDEX,
        "gpu_uuid": str(getattr(properties, "uuid", "")),
    }
    gqa_raw = _collect_one_control(
        torch=torch,
        role="gqa",
        operation_key=operation_key,
        query=query,
        key=gqa_key,
        value=gqa_value,
        dispatch_trace_raw=gqa_dispatch_trace_raw,
        runtime_identity=runtime,
    )
    mha_raw = _collect_one_control(
        torch=torch,
        role="mha_control",
        operation_key=operation_key,
        query=query,
        key=mha_key,
        value=mha_value,
        dispatch_trace_raw=mha_dispatch_trace_raw,
        runtime_identity=runtime,
    )
    return gqa_raw, mha_raw
