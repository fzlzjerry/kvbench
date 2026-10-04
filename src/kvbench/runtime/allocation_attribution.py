"""Fail-closed CUDA allocator attribution for Phase 3 remediation.

Collection remains separate from this module: these routines validate and
attribute an already captured chronological allocator trace.  Profiler or
audit execution therefore cannot be mistaken for normal benchmark timing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

from kvbench.errors import SchemaValidationError
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.schema.phase3 import derive_cache_layout_fingerprint


CUDA_ALLOCATOR_MIN_BLOCK_BYTES = 512
PHASE3_ALLOCATION_AUDIT_SCHEMA_VERSION = 3
PHASE3_ALLOCATION_WARMUP_ITERATIONS = 3
PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES = 100_000
PHASE3_TORCH_VERSION = "2.12.1+cu130"
PHASE3_CUDA_RUNTIME_VERSION = "13.0"
PHASE3_DEVICE = "cuda:0"
PHASE3_DEVICE_INDEX = 0
PHASE3_OUTPUT_WIDTH = 128_256
PHASE3_OUTPUT_DTYPE = "torch.bfloat16"
PHASE3_OUTPUT_DTYPE_BYTES = 2
PHASE3_HIDDEN_WIDTH = 4_096
PHASE3_INTERMEDIATE_WIDTH = 14_336
PHASE3_LAYER_COUNT = 32
PHASE3_NORM_COUNT = 2 * PHASE3_LAYER_COUNT + 1
PHASE3_EXTERNAL_PROVENANCE_STATUS = "external_run_join_unverified"
PHASE3_RECORDER_CONFIGURATION = {
    "enabled": "all",
    "context": "all",
    "stacks": "all",
    "max_entries": PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES,
    "clear_history": True,
}
FLASH_SPLIT_K_CPP_MARKERS = (
    "pytorch_flash::set_params_splitkv",
    "pytorch_flash::mha_fwd",
    "at::native::_scaled_dot_product_flash_attention_cuda",
)
_TORCH: Any | None = None


class AllocationAttributionError(ValueError):
    """Attribution evidence or configuration is structurally invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AllocationAttributionError(
            "allocator evidence is not canonical JSON"
        ) from error


def _require_exact_mapping_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise AllocationAttributionError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )


class AllocationClass(StrEnum):
    """Preregistered Phase 3 allocation classes."""

    CACHE_GROWTH = "cache_growth"
    GQA_EXPANSION = "gqa_expansion"
    CONTEXT_SCALED_WORKSPACE = "context_scaled_workspace"
    FIXED_OUTPUT = "fixed_output"
    FIXED_SHARED_ACTIVATION = "fixed_shared_activation"
    FRAMEWORK_BOOKKEEPING = "framework_bookkeeping"
    AUDIT_INSTRUMENTATION = "audit_instrumentation"
    UNKNOWN = "unknown"


class CacheReuseStatus(StrEnum):
    VERIFIED_REUSE = "verified_reuse"
    NEW_TRACE_SEGMENT = "new_trace_segment"
    UNVERIFIED = "unverified"


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AllocationAttributionError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class AllocationGeometry:
    """Frozen geometry and exact byte formulas for one operation."""

    batch: int
    query_heads: int
    kv_heads: int
    context: int
    head_dim: int
    dtype_bytes: int
    query_length: int = 1
    operation_output_width: int | None = None
    operation_output_dtype_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "query_heads",
            "kv_heads",
            "context",
            "head_dim",
            "dtype_bytes",
            "query_length",
        ):
            _positive_int(name, getattr(self, name))
        if self.kv_heads > self.query_heads:
            raise AllocationAttributionError(
                "kv_heads cannot exceed query_heads"
            )
        if (self.operation_output_width is None) != (
            self.operation_output_dtype_bytes is None
        ):
            raise AllocationAttributionError(
                "operation output width and dtype bytes must both be set or unset"
            )
        if self.operation_output_width is not None:
            _positive_int(
                "operation_output_width", self.operation_output_width
            )
            assert self.operation_output_dtype_bytes is not None
            _positive_int(
                "operation_output_dtype_bytes",
                self.operation_output_dtype_bytes,
            )

    @property
    def output_bytes(self) -> int:
        """Isolated attention output bytes, not endpoint/logit bytes."""

        return (
            self.batch
            * self.query_heads
            * self.query_length
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def hidden_bf16_bytes(self) -> int:
        return self.output_bytes

    @property
    def hidden_fp32_bytes(self) -> int:
        return (
            self.batch
            * self.query_length
            * self.query_heads
            * self.head_dim
            * 4
        )

    @property
    def intermediate_bf16_bytes(self) -> int:
        return (
            self.batch
            * self.query_length
            * PHASE3_INTERMEDIATE_WIDTH
            * self.dtype_bytes
        )

    @property
    def batch_fp32_scalar_bytes(self) -> int:
        return self.batch * self.query_length * 4

    @property
    def operation_output_bytes(self) -> int | None:
        if (
            self.operation_output_width is None
            or self.operation_output_dtype_bytes is None
        ):
            return None
        return (
            self.batch
            * self.query_length
            * self.operation_output_width
            * self.operation_output_dtype_bytes
        )

    @property
    def kv_projection_bytes(self) -> int:
        """Combined current-token K/V projection output."""

        return (
            2
            * self.batch
            * self.kv_heads
            * self.query_length
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def kv_projection_single_bytes(self) -> int:
        return (
            self.batch
            * self.kv_heads
            * self.query_length
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def flash_lse_bytes(self) -> int:
        """One FP32 LSE value per batch/query-head/query token."""

        return self.batch * self.query_heads * self.query_length * 4

    @property
    def expanded_kv_single_bytes(self) -> int:
        return (
            self.batch
            * self.query_heads
            * self.context
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def expanded_kv_combined_bytes(self) -> int:
        return 2 * self.expanded_kv_single_bytes

    @property
    def native_kv_single_bytes(self) -> int:
        return (
            self.batch
            * self.kv_heads
            * self.context
            * self.head_dim
            * self.dtype_bytes
        )

    @property
    def native_kv_combined_bytes(self) -> int:
        return 2 * self.native_kv_single_bytes

    def flash_split_k_output_accumulator_bytes(self, splits: int) -> int:
        """FP32 partial-output storage for frozen Flash split-K."""

        _positive_int("splits", splits)
        return (
            splits
            * self.batch
            * self.query_heads
            * self.query_length
            * self.head_dim
            * 4
        )

    def flash_split_k_lse_bytes(self, splits: int) -> int:
        """FP32 log-sum-exp storage for frozen Flash split-K."""

        _positive_int("splits", splits)
        return (
            splits
            * self.batch
            * self.query_heads
            * self.query_length
            * 4
        )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "batch": self.batch,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "context": self.context,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
            "query_length": self.query_length,
            "output_bytes": self.output_bytes,
            "hidden_bf16_bytes": self.hidden_bf16_bytes,
            "hidden_fp32_bytes": self.hidden_fp32_bytes,
            "intermediate_bf16_bytes": self.intermediate_bf16_bytes,
            "batch_fp32_scalar_bytes": self.batch_fp32_scalar_bytes,
            "operation_output_width": self.operation_output_width,
            "operation_output_dtype_bytes": (
                self.operation_output_dtype_bytes
            ),
            "operation_output_bytes": self.operation_output_bytes,
            "kv_projection_bytes": self.kv_projection_bytes,
            "kv_projection_single_bytes": (
                self.kv_projection_single_bytes
            ),
            "flash_lse_bytes": self.flash_lse_bytes,
            "expanded_kv_single_bytes": self.expanded_kv_single_bytes,
            "expanded_kv_combined_bytes": self.expanded_kv_combined_bytes,
            "native_kv_single_bytes": self.native_kv_single_bytes,
            "native_kv_combined_bytes": self.native_kv_combined_bytes,
        }


@dataclass(frozen=True, slots=True)
class DependencyFlags:
    """Whether requested bytes depend on each frozen dimension.

    None means the dependency is unproven and is fail-closed by the gate.
    """

    batch: bool | None
    context: bool | None
    query_heads: bool | None
    kv_heads: bool | None

    def to_dict(self) -> dict[str, bool | None]:
        return {
            "batch": self.batch,
            "context": self.context,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
        }


UNKNOWN_DEPENDENCIES = DependencyFlags(None, None, None, None)


@dataclass(frozen=True, slots=True)
class StackFrame:
    """One normalized Python or C++ stack frame."""

    name: str
    filename: str | None = None
    line: int | None = None

    def render(self) -> str:
        location = self.filename or ""
        if self.line is not None:
            location = f"{location}:{self.line}"
        return f"{self.name} {location}".strip()

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "name": self.name,
            "filename": self.filename,
            "line": self.line,
        }


@dataclass(frozen=True, slots=True)
class CanonicalFrameSelector:
    """Exact function plus canonical source-suffix selector.

    Source suffixes are path-component matches, not arbitrary substrings.  A
    production selector must be present in the checked-in policy catalog.
    """

    function_name: str
    source_suffix: str

    def __post_init__(self) -> None:
        if not self.function_name or not self.source_suffix:
            raise AllocationAttributionError(
                "canonical frame selector fields must be nonempty"
            )
        normalized = self.source_suffix.replace("\\", "/").strip("/")
        if normalized != self.source_suffix or ".." in normalized.split("/"):
            raise AllocationAttributionError(
                "canonical frame source_suffix must be normalized and relative"
            )

    def matches(self, frame: StackFrame) -> bool:
        if frame.name != self.function_name or frame.filename is None:
            return False
        filename = frame.filename.replace("\\", "/").rstrip("/")
        return filename == self.source_suffix or filename.endswith(
            f"/{self.source_suffix}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "function_name": self.function_name,
            "source_suffix": self.source_suffix,
        }


@dataclass(frozen=True, slots=True)
class TraceEventEvidence:
    """One allocator trace event retained in original chronological order."""

    index: int
    action: str
    address: int | None
    size_bytes: int | None
    stream: int | None
    allocated_block_bytes: int | None
    python_stack: tuple[StackFrame, ...]
    cpp_stack: tuple[StackFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "address": self.address,
            "size_bytes": self.size_bytes,
            "stream": self.stream,
            "allocated_block_bytes": self.allocated_block_bytes,
            "python_stack": [frame.to_dict() for frame in self.python_stack],
            "cpp_stack": [frame.to_dict() for frame in self.cpp_stack],
        }


@dataclass(frozen=True, slots=True)
class AllocationLifetime:
    """One address generation from alloc through free completion."""

    allocation_id: int
    address: int
    requested_bytes: int
    rounded_minimum_bytes: int
    allocated_block_bytes: int | None
    allocated_block_size_proven: bool
    stream: int | None
    alloc_event_index: int
    free_requested_event_index: int | None
    free_completed_event_index: int | None
    fully_freed: bool
    cache_reuse_status: CacheReuseStatus
    triggered_segment_alloc: bool
    python_stack: tuple[StackFrame, ...]
    cpp_stack: tuple[StackFrame, ...]
    event_class: AllocationClass
    size_formula: str | None
    policy_id: str | None
    formula_parameters: tuple[tuple[str, int], ...]
    dependencies: DependencyFlags

    @property
    def reused_from_cache(self) -> bool | None:
        if self.cache_reuse_status is CacheReuseStatus.UNVERIFIED:
            return None
        return self.cache_reuse_status is CacheReuseStatus.VERIFIED_REUSE

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "address": self.address,
            "requested_bytes": self.requested_bytes,
            "rounded_minimum_bytes": self.rounded_minimum_bytes,
            "allocated_block_bytes": self.allocated_block_bytes,
            "allocated_block_size_proven": self.allocated_block_size_proven,
            "stream": self.stream,
            "alloc_event_index": self.alloc_event_index,
            "free_requested_event_index": self.free_requested_event_index,
            "free_completed_event_index": self.free_completed_event_index,
            "fully_freed": self.fully_freed,
            "reused_from_cache": self.reused_from_cache,
            "cache_reuse_status": self.cache_reuse_status.value,
            "triggered_segment_alloc": self.triggered_segment_alloc,
            "python_stack": [frame.to_dict() for frame in self.python_stack],
            "cpp_stack": [frame.to_dict() for frame in self.cpp_stack],
            "event_class": self.event_class.value,
            "size_formula": self.size_formula,
            "policy_id": self.policy_id,
            "formula_parameters": dict(self.formula_parameters),
            "dependencies": self.dependencies.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AllocatorCounterEvidence:
    """Cumulative counters spanning exactly the traced operation.

    Nullable fields preserve unavailable evidence.  Gate evaluation treats
    every unavailable counter as a failure.
    """

    allocation_count: int | None
    requested_bytes: int | None
    allocated_block_bytes: int | None
    device_allocation_count: int | None
    device_free_count: int | None
    free_count: int | None = None
    freed_requested_bytes: int | None = None
    freed_block_bytes: int | None = None
    segment_allocation_count: int | None = None
    segment_free_count: int | None = None
    allocation_retry_count: int | None = None
    oom_count: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "allocation_count",
            "requested_bytes",
            "allocated_block_bytes",
            "device_allocation_count",
            "device_free_count",
            "free_count",
            "freed_requested_bytes",
            "freed_block_bytes",
            "segment_allocation_count",
            "segment_free_count",
            "allocation_retry_count",
            "oom_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise AllocationAttributionError(
                    f"{name} must be a nonnegative integer or None"
                )

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.allocation_count,
                self.requested_bytes,
                self.allocated_block_bytes,
                self.device_allocation_count,
                self.device_free_count,
                self.free_count,
                self.freed_requested_bytes,
                self.freed_block_bytes,
                self.segment_allocation_count,
                self.segment_free_count,
                self.allocation_retry_count,
                self.oom_count,
            )
        )

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "allocation_count": self.allocation_count,
            "requested_bytes": self.requested_bytes,
            "allocated_block_bytes": self.allocated_block_bytes,
            "device_allocation_count": self.device_allocation_count,
            "device_free_count": self.device_free_count,
            "free_count": self.free_count,
            "freed_requested_bytes": self.freed_requested_bytes,
            "freed_block_bytes": self.freed_block_bytes,
            "segment_allocation_count": self.segment_allocation_count,
            "segment_free_count": self.segment_free_count,
            "allocation_retry_count": self.allocation_retry_count,
            "oom_count": self.oom_count,
            "complete": self.complete,
        }


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class AllocatorHistoryIntegrityEvidence:
    """Completeness and immutable-raw-evidence proof for allocator history."""

    stack_mode: str
    ring_capacity: int
    observed_trace_entries: int
    raw_snapshot_sha256: str
    expected_raw_snapshot_sha256: str
    raw_trace_sha256: str
    expected_raw_trace_sha256: str

    def __post_init__(self) -> None:
        _positive_int("ring_capacity", self.ring_capacity)
        if (
            isinstance(self.observed_trace_entries, bool)
            or not isinstance(self.observed_trace_entries, int)
            or self.observed_trace_entries < 0
        ):
            raise AllocationAttributionError(
                "observed_trace_entries must be a nonnegative integer"
            )
        for name in (
            "raw_snapshot_sha256",
            "expected_raw_snapshot_sha256",
            "raw_trace_sha256",
            "expected_raw_trace_sha256",
        ):
            value = getattr(self, name)
            if not _valid_sha256(value):
                raise AllocationAttributionError(
                    f"{name} must be a lowercase SHA-256 digest"
                )

    @property
    def ring_saturated(self) -> bool:
        return self.observed_trace_entries >= self.ring_capacity

    @property
    def passed(self) -> bool:
        return (
            self.stack_mode == "all"
            and not self.ring_saturated
            and self.raw_snapshot_sha256
            == self.expected_raw_snapshot_sha256
            and self.raw_trace_sha256 == self.expected_raw_trace_sha256
        )

    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.stack_mode != "all":
            reasons.append("allocator_history_python_cpp_stacks_unverified")
        if self.ring_saturated:
            reasons.append("allocator_history_ring_saturated")
        if self.raw_snapshot_sha256 != self.expected_raw_snapshot_sha256:
            reasons.append("raw_allocator_snapshot_sha256_mismatch")
        if self.raw_trace_sha256 != self.expected_raw_trace_sha256:
            reasons.append("raw_allocator_trace_sha256_mismatch")
        return tuple(reasons)

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "stack_mode": self.stack_mode,
            "ring_capacity": self.ring_capacity,
            "observed_trace_entries": self.observed_trace_entries,
            "ring_saturated": self.ring_saturated,
            "raw_snapshot_sha256": self.raw_snapshot_sha256,
            "expected_raw_snapshot_sha256": (
                self.expected_raw_snapshot_sha256
            ),
            "raw_trace_sha256": self.raw_trace_sha256,
            "expected_raw_trace_sha256": self.expected_raw_trace_sha256,
            "passed": self.passed,
        }


_POLICY_FORMULA_CONTRACTS = {
    "attention_output_geometry_bytes_v1": (
        AllocationClass.FIXED_OUTPUT,
        DependencyFlags(True, False, True, False),
    ),
    "operation_output_geometry_bytes_v1": (
        AllocationClass.FIXED_OUTPUT,
        DependencyFlags(True, False, False, False),
    ),
    "kv_projection_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, False, True),
    ),
    "kv_projection_single_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, False, True),
    ),
    "hidden_bf16_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, True, False),
    ),
    "hidden_fp32_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, True, False),
    ),
    "intermediate_bf16_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, False, False),
    ),
    "batch_fp32_scalar_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, False, False),
    ),
    "attention_output_shared_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, True, False),
    ),
    "flash_lse_geometry_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(True, False, True, False),
    ),
    "fixed_shared_scalar_exact_bytes_v1": (
        AllocationClass.FIXED_SHARED_ACTIVATION,
        DependencyFlags(False, False, False, False),
    ),
    "framework_scalar_exact_bytes_v1": (
        AllocationClass.FRAMEWORK_BOOKKEEPING,
        DependencyFlags(False, False, False, False),
    ),
}
_EAGER_PERMITTED_CLASSES = (
    AllocationClass.FIXED_OUTPUT,
    AllocationClass.FIXED_SHARED_ACTIVATION,
    AllocationClass.FRAMEWORK_BOOKKEEPING,
)
DECISION_0009_ID = "0009"
DECISION_0009_SHA256 = (
    "7421b8877b3305ccccea6006a4d34bf459ba774e7326ef470ad0d57cdddc0b03"
)
DECISION_0009_POLICY_CATALOG_ID = (
    "phase3-decision-0009-eager-allocation-policy-catalog-v1"
)
DECISION_0009_POLICY_CATALOG_SHA256 = (
    "9e14a4cb11e5818b795ddd60a1b6b2000999ff24a24340f667c6644050cb589d"
)
DECISION_0013_SHA256 = (
    "458a6c4c1add90fab5a6aea370c968f2f2806dda80677fb77e128c4c9cf83190"
)
PHASE3_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PHASE3_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
PHASE3_MODEL_CONFIG_SHA256 = (
    "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e"
)
PHASE3_BACKEND_CONFIG_ID = (
    "pytorch_2.12.1_flash_sdpa_fa2_2.5.7_forced_gqa"
)
PHASE3_CACHE_LAYOUT_ID = "layers_batch_kv_heads_context_head_dim"
PHASE3_CACHE_IMPLEMENTATION_SHA256 = (
    "6cd86da351e302e20afefc5f001019be21b5f23b171177ac847592b8cf33c340"
)
PHASE3_ADAPTER_SOURCE_SHA256 = (
    "0d185b132987627abe0272a4e66f14202ac5a06c832fe0cc59050c9d7a403313"
)
PHASE3_KERNEL_BINARY_SHA256 = (
    "b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984"
)
PHASE3_FIXED_PLAN_PATH = "configs/plans/phase3_bf16_fixed_l.yaml"
PHASE3_GROWING_PLAN_PATH = "configs/plans/phase3_bf16_growing.yaml"
PHASE3_FIXED_PLAN_SHA256 = (
    "d8f2b4e61f6569d5b8cb75b84bbf36a3b60927b0575d514c2d9bd0aac7da6a2d"
)
PHASE3_GROWING_PLAN_SHA256 = (
    "4598647f5ba04deff187d11346c0695b857464d06729b96b46b838080d80cd63"
)
PHASE3_RANDOM_SEED = 20260722
PHASE3_BACKEND_IDENTITY_SHA256 = (
    "0841ae768cf05df38adbf803b5019460491572b9bf205d87f703428d2cfbc354"
)


def _phase3_backend_identity_payload() -> dict[str, Any]:
    return {
        "schema_version": "kvbench.phase3-bf16-backend.v1",
        "backend_id": "torch_sdpa_flash_gqa",
        "torch_version": "2.12.1+cu130",
        "torch_git_sha": "7269437d655783a26cba32aa88195b741ff496aa",
        "cuda_runtime_version": "13.0",
        "cudnn_version": "9.20.0",
        "triton_version": "3.7.1",
        "flash_generation": "FA2",
        "flash_version": "2.5.7",
        "dispatch_api": (
            "torch.nn.functional.scaled_dot_product_attention"
        ),
        "selected_backend": "flash_attention",
        "enable_gqa": True,
        "compile_mode": "disabled",
        "source_artifacts": [
            {
                "path": (
                    "include/ATen/native/transformers/cuda/flash_attn/"
                    "flash_api.h"
                ),
                "sha256": (
                    "1474aa79d8aa6ce39984dbc3c0aad9dba283ab819f034370e5cfb70980524ee7"
                ),
            },
            {
                "path": "lib/libtorch_cuda.so",
                "sha256": PHASE3_KERNEL_BINARY_SHA256,
            },
            {
                "path": "nn/attention/__init__.py",
                "sha256": (
                    "56e10b6f965cc050db782dd4dc472097c9b02ec5b5fe3ab2c8b04055c0b0bbe0"
                ),
            },
            {
                "path": "nn/attention/varlen.py",
                "sha256": (
                    "2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea"
                ),
            },
            {
                "path": "nn/functional.py",
                "sha256": (
                    "27493186ee22f811b553e31d9c804d4d46716d1be62d034d731537f66f27ef19"
                ),
            },
        ],
    }


PHASE3_BACKEND_IDENTITY = _canonical_json_bytes(
    _phase3_backend_identity_payload()
).decode("utf-8")


def _phase3_backend_identity_is_intact() -> bool:
    return (
        hashlib.sha256(PHASE3_BACKEND_IDENTITY.encode("utf-8")).hexdigest()
        == PHASE3_BACKEND_IDENTITY_SHA256
    )


def _phase3_cache_layout_fingerprint(
    *, runner_kind: str, batch: int, starting_context: int
) -> tuple[int, str]:
    output_steps = 1 if runner_kind == "fixed_l" else 16
    capacity = starting_context + output_steps
    workspace_bytes = 32 * batch * (32 + 8) * 1 * 64 * 2
    return capacity, derive_cache_layout_fingerprint(
        num_layers=32,
        batch_size=batch,
        num_kv_heads=8,
        capacity=capacity,
        head_dim=128,
        device=PHASE3_DEVICE,
        workspace_bytes=workspace_bytes,
        implementation_sha256=PHASE3_CACHE_IMPLEMENTATION_SHA256,
    )


# Decision 0013 source-backed template rows:
# policy, class, formula, count, exact bytes, Python function/source,
# C++ function/source. Geometry formulas resolve bytes per operation key.
_DECISION_0013_POLICY_ROWS = (
    (
        "embedding_hidden",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_bf16_geometry_bytes_v1",
        1,
        None,
        "embedding",
        "torch/nn/functional.py",
        "torch::autograd::THPVariable_embedding(_object*, _object*, _object*)",
        "python_torch_functions_0.cpp",
    ),
    (
        "norm_to_fp32",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_fp32_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_to(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "norm_pow",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_fp32_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_pow(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "norm_mean",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "batch_fp32_scalar_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_mean(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "norm_scalar_add",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "batch_fp32_scalar_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_add(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "norm_rsqrt",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "batch_fp32_scalar_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_rsqrt(_object*, _object*, _object*)",
        "python_torch_functions_1.cpp",
    ),
    (
        "norm_mul_fp32",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_fp32_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_mul(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "norm_to_bf16",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_bf16_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_to(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "norm_weight_mul",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_bf16_geometry_bytes_v1",
        PHASE3_NORM_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_mul(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "kv_linear",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "kv_projection_single_geometry_bytes_v1",
        2 * PHASE3_LAYER_COUNT,
        None,
        "forward",
        "torch/nn/modules/linear.py",
        "torch::autograd::THPVariable_linear(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "hidden_linear",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_bf16_geometry_bytes_v1",
        3 * PHASE3_LAYER_COUNT,
        None,
        "forward",
        "torch/nn/modules/linear.py",
        "torch::autograd::THPVariable_linear(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "residual_add",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "hidden_bf16_geometry_bytes_v1",
        2 * PHASE3_LAYER_COUNT,
        None,
        "_base_forward",
        "src/kvbench/runtime/bf16_endpoint.py",
        "torch::autograd::THPVariable_add(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "intermediate_linear",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "intermediate_bf16_geometry_bytes_v1",
        2 * PHASE3_LAYER_COUNT,
        None,
        "forward",
        "torch/nn/modules/linear.py",
        "torch::autograd::THPVariable_linear(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "intermediate_silu",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "intermediate_bf16_geometry_bytes_v1",
        PHASE3_LAYER_COUNT,
        None,
        "silu",
        "torch/nn/functional.py",
        "torch::autograd::THPVariable_silu(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "intermediate_mul",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "intermediate_bf16_geometry_bytes_v1",
        PHASE3_LAYER_COUNT,
        None,
        "forward",
        "transformers/models/llama/modeling_llama.py",
        "torch::autograd::THPVariable_mul(_object*, _object*, _object*)",
        "python_variable_methods.cpp",
    ),
    (
        "flash_scalar_8",
        AllocationClass.FRAMEWORK_BOOKKEEPING.value,
        "framework_scalar_exact_bytes_v1",
        PHASE3_LAYER_COUNT,
        8,
        "flash_attention_forward",
        "src/kvbench/runtime/backend.py",
        "torch::autograd::THPVariable_scaled_dot_product_attention(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "flash_scalar_16",
        AllocationClass.FRAMEWORK_BOOKKEEPING.value,
        "framework_scalar_exact_bytes_v1",
        PHASE3_LAYER_COUNT,
        16,
        "flash_attention_forward",
        "src/kvbench/runtime/backend.py",
        "torch::autograd::THPVariable_scaled_dot_product_attention(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "flash_lse",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "flash_lse_geometry_bytes_v1",
        PHASE3_LAYER_COUNT,
        None,
        "flash_attention_forward",
        "src/kvbench/runtime/backend.py",
        "torch::autograd::THPVariable_scaled_dot_product_attention(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "flash_output",
        AllocationClass.FIXED_SHARED_ACTIVATION.value,
        "attention_output_shared_geometry_bytes_v1",
        PHASE3_LAYER_COUNT,
        None,
        "flash_attention_forward",
        "src/kvbench/runtime/backend.py",
        "torch::autograd::THPVariable_scaled_dot_product_attention(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
    (
        "operation_output",
        AllocationClass.FIXED_OUTPUT.value,
        "operation_output_geometry_bytes_v1",
        1,
        None,
        "forward",
        "torch/nn/modules/linear.py",
        "torch::autograd::THPVariable_linear(_object*, _object*, _object*)",
        "python_nn_functions.cpp",
    ),
)


def _decision_0009_catalog_payload() -> dict[str, Any]:
    templates = [
        {
            "policy_id": f"phase3_decision_0013_{row[0]}_v1",
            "event_class": row[1],
            "formula_id": row[2],
            "exact_count": row[3],
            "exact_requested_bytes": row[4],
            "required_python_frame": {
                "function_name": row[5],
                "source_suffix": row[6],
            },
            "required_cpp_frame": {
                "function_name": row[7],
                "source_suffix": row[8],
            },
        }
        for row in _DECISION_0013_POLICY_ROWS
    ]
    templates.append(
        {
            "policy_id": "phase3_decision_0013_flash_split_k_v1",
            "event_class": AllocationClass.CONTEXT_SCALED_WORKSPACE.value,
            "formula_ids": [
                "flash_split_k_output_accumulator",
                "flash_split_k_lse",
            ],
            "full_endpoint_pair_count": PHASE3_LAYER_COUNT,
            "required_cpp_markers": list(FLASH_SPLIT_K_CPP_MARKERS),
            "paired_control_required": True,
        }
    )
    return {
        "catalog_id": DECISION_0009_POLICY_CATALOG_ID,
        "criterion_decision_id": DECISION_0009_ID,
        "criterion_decision_sha256": DECISION_0009_SHA256,
        "catalog_amendment_decision_id": "0013",
        "catalog_amendment_decision_sha256": DECISION_0013_SHA256,
        "production_templates": templates,
    }


def _decision_0009_catalog_is_intact() -> bool:
    return (
        hashlib.sha256(
            _canonical_json_bytes(_decision_0009_catalog_payload())
        ).hexdigest()
        == DECISION_0009_POLICY_CATALOG_SHA256
    )


def _decision_0009_catalog_requires_split_k_raw() -> bool:
    templates = _decision_0009_catalog_payload()["production_templates"]
    return any(
        isinstance(item, Mapping)
        and item.get("event_class")
        == AllocationClass.CONTEXT_SCALED_WORKSPACE.value
        for item in templates
    )


def _phase3_point_fingerprint(
    *,
    point_id: str,
    runner_kind: str,
    execution_mode: str,
    batch: int,
    starting_context: int,
    process_replicate: int,
) -> str:
    fixed = runner_kind == "fixed_l"
    payload = {
        "schema": "kvbench-phase3-process-point-1.0.0",
        "point_id": point_id,
        "plan_path": (
            PHASE3_FIXED_PLAN_PATH if fixed else PHASE3_GROWING_PLAN_PATH
        ),
        "plan_fingerprint": (
            PHASE3_FIXED_PLAN_SHA256
            if fixed
            else PHASE3_GROWING_PLAN_SHA256
        ),
        "runner_kind": runner_kind,
        "graph_mode": execution_mode,
        "batch_size": batch,
        "context_length": starting_context,
        "output_steps": 1 if fixed else 16,
        "warmup_count": 16 if fixed else 1,
        "measured_count": 32 if fixed else 1,
        "measured_batches": 5 if fixed else 1,
        "count_unit": "decode_operations" if fixed else "trajectories",
        "random_seed": PHASE3_RANDOM_SEED,
        "process_replicate": process_replicate,
        "capacity": starting_context + (1 if fixed else 16),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class AllocationClassPolicy:
    """Frozen provenance, formula, multiplicity, and byte contract.

    A size match alone is never enough to permit an eager allocation.  Every
    permitted class requires nonempty Python and C++ provenance markers and an
    exact preregistered count/byte total.  The policy is serialized into, and
    hashed with, the allocator attribution evidence.
    """

    policy_id: str
    event_class: AllocationClass
    formula_id: str
    allowed_requested_bytes: frozenset[int]
    required_python_frames: tuple[CanonicalFrameSelector, ...]
    required_cpp_frames: tuple[CanonicalFrameSelector, ...]
    dependencies: DependencyFlags
    exact_count: int
    exact_total_requested_bytes: int

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise AllocationAttributionError("policy_id must be nonempty")
        contract = _POLICY_FORMULA_CONTRACTS.get(self.formula_id)
        if contract is None:
            raise AllocationAttributionError(
                "allocation policy formula_id is not recognized"
            )
        expected_class, expected_dependencies = contract
        if self.event_class is not expected_class:
            raise AllocationAttributionError(
                "allocation policy formula_id does not match its class"
            )
        if self.dependencies != expected_dependencies:
            raise AllocationAttributionError(
                "allocation policy dependencies do not match its formula"
            )
        if not self.allowed_requested_bytes:
            raise AllocationAttributionError(
                "allowed_requested_bytes must be nonempty"
            )
        for size in self.allowed_requested_bytes:
            _positive_int("allowed_requested_bytes entry", size)
        for name in (
            "required_python_frames",
            "required_cpp_frames",
        ):
            selectors = getattr(self, name)
            if not selectors:
                raise AllocationAttributionError(
                    f"{name} must contain exact canonical selectors"
                )
        _positive_int("exact_count", self.exact_count)
        _positive_int(
            "exact_total_requested_bytes",
            self.exact_total_requested_bytes,
        )
        if len(self.allowed_requested_bytes) != 1:
            raise AllocationAttributionError(
                "each allocation policy must freeze exactly one requested size"
            )
        frozen_size = next(iter(self.allowed_requested_bytes))
        if self.exact_total_requested_bytes != frozen_size * self.exact_count:
            raise AllocationAttributionError(
                "exact total bytes must equal frozen size times exact count"
            )
        if self.formula_id in {
            "fixed_shared_scalar_exact_bytes_v1",
            "framework_scalar_exact_bytes_v1",
        } and (frozen_size > 4096 or self.exact_count > 64):
            raise AllocationAttributionError(
                "scalar allocation policy exceeds frozen size/count bounds"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "event_class": self.event_class.value,
            "formula_id": self.formula_id,
            "allowed_requested_bytes": sorted(self.allowed_requested_bytes),
            "required_python_frames": [
                selector.to_dict() for selector in self.required_python_frames
            ],
            "required_cpp_frames": [
                selector.to_dict() for selector in self.required_cpp_frames
            ],
            "dependencies": self.dependencies.to_dict(),
            "exact_count": self.exact_count,
            "exact_total_requested_bytes": (
                self.exact_total_requested_bytes
            ),
        }


@dataclass(frozen=True, slots=True)
class SplitKCompositeRawInputs:
    """Checksum identities and verifier-derived paired-control multiplicity."""

    gqa_dispatch_trace_sha256: str
    mha_dispatch_trace_sha256: str
    gqa_allocator_control_sha256: str
    mha_allocator_control_sha256: str
    split_k_pair_multiplicity: tuple[tuple[int, int], ...] = ()
    raw_bytes_verified: bool = False
    raw_composite_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "gqa_dispatch_trace_sha256",
            "mha_dispatch_trace_sha256",
            "gqa_allocator_control_sha256",
            "mha_allocator_control_sha256",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise AllocationAttributionError(
                    f"{name} must be a lowercase SHA-256 digest"
                )
        for pair in self.split_k_pair_multiplicity:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or any(
                    type(value) is not int or value <= 0
                    for value in pair
                )
            ):
                raise AllocationAttributionError(
                    "split-K multiplicity entries must be positive pairs"
                )
        if self.split_k_pair_multiplicity != tuple(
            sorted(set(self.split_k_pair_multiplicity))
        ):
            raise AllocationAttributionError(
                "split-K multiplicity must be unique and ordered"
            )
        if self.raw_bytes_verified:
            if not _valid_sha256(self.raw_composite_sha256):
                raise AllocationAttributionError(
                    "raw-byte-verified split-K inputs require a composite digest"
                )
        elif (
            self.raw_composite_sha256 is not None
            or self.split_k_pair_multiplicity
        ):
            raise AllocationAttributionError(
                "unverified split-K inputs cannot claim derived facts"
            )

    @classmethod
    def from_raw_bytes(
        cls,
        *,
        gqa_dispatch_trace: bytes,
        mha_dispatch_trace: bytes,
        gqa_allocator_control: bytes,
        mha_allocator_control: bytes,
        split_k_pair_multiplicity: tuple[tuple[int, int], ...] = (),
    ) -> SplitKCompositeRawInputs:
        raw_values = {
            "gqa_dispatch_trace": gqa_dispatch_trace,
            "mha_dispatch_trace": mha_dispatch_trace,
            "gqa_allocator_control": gqa_allocator_control,
            "mha_allocator_control": mha_allocator_control,
        }
        if not all(
            isinstance(value, bytes) and value for value in raw_values.values()
        ):
            raise AllocationAttributionError(
                "split-K composite inputs must be nonempty raw bytes"
            )
        digests = {
            name: hashlib.sha256(value).hexdigest()
            for name, value in raw_values.items()
        }
        multiplicity_payload = [
            {"num_splits": splits, "pair_count": count}
            for splits, count in split_k_pair_multiplicity
        ]
        composite = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "raw_sha256": digests,
                    "split_k_pair_multiplicity": multiplicity_payload,
                }
            )
        ).hexdigest()
        return cls(
            gqa_dispatch_trace_sha256=digests["gqa_dispatch_trace"],
            mha_dispatch_trace_sha256=digests["mha_dispatch_trace"],
            gqa_allocator_control_sha256=digests[
                "gqa_allocator_control"
            ],
            mha_allocator_control_sha256=digests[
                "mha_allocator_control"
            ],
            split_k_pair_multiplicity=split_k_pair_multiplicity,
            raw_bytes_verified=True,
            raw_composite_sha256=composite,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gqa_dispatch_trace_sha256": self.gqa_dispatch_trace_sha256,
            "mha_dispatch_trace_sha256": self.mha_dispatch_trace_sha256,
            "gqa_allocator_control_sha256": (
                self.gqa_allocator_control_sha256
            ),
            "mha_allocator_control_sha256": (
                self.mha_allocator_control_sha256
            ),
            "split_k_pair_multiplicity": [
                {"num_splits": splits, "pair_count": count}
                for splits, count in self.split_k_pair_multiplicity
            ],
            "raw_bytes_verified": self.raw_bytes_verified,
            "raw_composite_sha256": self.raw_composite_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProductionAllocationBinding:
    """Checked Phase 3 run/point/source/backend/operation binding."""

    operation_key: Phase3AuditOperationKey
    run_id: str
    point_id: str
    point_fingerprint: str
    runner_kind: str
    execution_mode: str
    batch: int
    starting_context: int
    decode_step: int
    process_replicate: int
    historical_context: int
    attended_context: int
    geometry: AllocationGeometry
    cache_layout_id: str
    cache_capacity: int
    cache_implementation_sha256: str
    cache_layout_fingerprint: str
    model_id: str
    model_revision: str
    model_config_sha256: str
    adapter_source_sha256: str
    kernel_binary_sha256: str
    backend_config_id: str
    backend_identity: str
    backend_identity_sha256: str
    operation_fingerprint_sha256: str
    split_k_raw_inputs: SplitKCompositeRawInputs | None
    external_provenance_status: str = PHASE3_EXTERNAL_PROVENANCE_STATUS
    decision_id: str = DECISION_0009_ID
    decision_sha256: str = DECISION_0009_SHA256
    policy_catalog_id: str = DECISION_0009_POLICY_CATALOG_ID
    policy_catalog_sha256: str = DECISION_0009_POLICY_CATALOG_SHA256

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if type(self.operation_key) is not Phase3AuditOperationKey:
            errors.append("production_binding_operation_key_invalid")
            return tuple(errors)
        if not self.run_id:
            errors.append("production_binding_run_id_missing")
        for name in (
            "cache_layout_fingerprint",
            "point_fingerprint",
            "model_config_sha256",
            "adapter_source_sha256",
            "kernel_binary_sha256",
            "backend_identity_sha256",
            "operation_fingerprint_sha256",
            "decision_sha256",
            "policy_catalog_sha256",
        ):
            value = getattr(self, name)
            if not _valid_sha256(value):
                errors.append(f"production_binding_digest_invalid:{name}")
        expected_point = _production_point_id(
            self.runner_kind,
            self.execution_mode,
            self.batch,
            self.starting_context,
            self.decode_step,
            self.process_replicate,
        )
        if expected_point is None or self.point_id != expected_point:
            errors.append("production_binding_point_not_preregistered")
        expected_fingerprint = (
            None
            if expected_point is None
            else _phase3_point_fingerprint(
                point_id=expected_point,
                runner_kind=self.runner_kind,
                execution_mode=self.execution_mode,
                batch=self.batch,
                starting_context=self.starting_context,
                process_replicate=self.process_replicate,
            )
        )
        if self.point_fingerprint != expected_fingerprint:
            errors.append("production_binding_point_fingerprint_mismatch")
        key = self.operation_key
        key_join = {
            "run_id": key.run_id,
            "point_id": key.point_id,
            "point_fingerprint": key.point_fingerprint,
            "runner_kind": key.runner_kind.value,
            "execution_mode": key.allocation_execution_mode,
            "batch": key.batch_size,
            "decode_step": key.decode_step,
            "process_replicate": key.process_replicate,
            "historical_context": key.historical_context,
            "attended_context": key.attended_context,
            "cache_capacity": key.capacity,
            "cache_layout_fingerprint": key.cache_layout_fingerprint,
            "backend_identity_sha256": key.backend_identity_sha256,
            "operation_fingerprint_sha256": (
                key.operation_fingerprint_sha256
            ),
        }
        for name, expected in key_join.items():
            if getattr(self, name) != expected:
                errors.append(f"production_binding_operation_key_mismatch:{name}")
        if self.starting_context != key.historical_context - key.decode_step:
            errors.append(
                "production_binding_operation_key_mismatch:starting_context"
            )
        if self.external_provenance_status != PHASE3_EXTERNAL_PROVENANCE_STATUS:
            errors.append("production_binding_external_provenance_status_invalid")
        expected_historical = self.starting_context + self.decode_step
        expected_attended = expected_historical + 1
        if self.historical_context != expected_historical:
            errors.append("production_binding_historical_context_mismatch")
        if self.attended_context != expected_attended:
            errors.append("production_binding_attended_context_mismatch")
        expected_geometry = AllocationGeometry(
            batch=self.batch,
            query_heads=32,
            kv_heads=8,
            context=expected_attended,
            head_dim=128,
            dtype_bytes=2,
            query_length=1,
            operation_output_width=PHASE3_OUTPUT_WIDTH,
            operation_output_dtype_bytes=PHASE3_OUTPUT_DTYPE_BYTES,
        )
        if self.geometry != expected_geometry:
            errors.append("production_binding_geometry_mismatch")
        frozen_values = {
            "cache_layout_id": PHASE3_CACHE_LAYOUT_ID,
            "model_id": PHASE3_MODEL_ID,
            "model_revision": PHASE3_MODEL_REVISION,
            "model_config_sha256": PHASE3_MODEL_CONFIG_SHA256,
            "adapter_source_sha256": PHASE3_ADAPTER_SOURCE_SHA256,
            "kernel_binary_sha256": PHASE3_KERNEL_BINARY_SHA256,
            "backend_config_id": PHASE3_BACKEND_CONFIG_ID,
            "cache_implementation_sha256": (
                PHASE3_CACHE_IMPLEMENTATION_SHA256
            ),
            "decision_id": DECISION_0009_ID,
            "decision_sha256": DECISION_0009_SHA256,
            "policy_catalog_id": DECISION_0009_POLICY_CATALOG_ID,
            "policy_catalog_sha256": (
                DECISION_0009_POLICY_CATALOG_SHA256
            ),
        }
        for name, expected in frozen_values.items():
            if getattr(self, name) != expected:
                errors.append(f"production_binding_frozen_mismatch:{name}")
        expected_capacity, expected_cache_fingerprint = (
            _phase3_cache_layout_fingerprint(
                runner_kind=self.runner_kind,
                batch=self.batch,
                starting_context=self.starting_context,
            )
        )
        if self.cache_capacity != expected_capacity:
            errors.append("production_binding_cache_capacity_mismatch")
        if self.cache_layout_fingerprint != expected_cache_fingerprint:
            errors.append("production_binding_cache_fingerprint_mismatch")
        expected_backend_sha = hashlib.sha256(
            self.backend_identity.encode("utf-8")
        ).hexdigest()
        if (
            not _phase3_backend_identity_is_intact()
            or self.backend_identity != PHASE3_BACKEND_IDENTITY
            or self.backend_identity_sha256 != expected_backend_sha
            or self.backend_identity_sha256
            != PHASE3_BACKEND_IDENTITY_SHA256
        ):
            errors.append("production_binding_backend_identity_mismatch")
        if (
            self.operation_fingerprint_sha256
            != self.operation_key.operation_fingerprint_sha256
        ):
            errors.append("production_binding_operation_fingerprint_mismatch")
        if not _decision_0009_catalog_is_intact():
            errors.append("decision_0009_policy_catalog_hash_mismatch")
        if (
            self.execution_mode == "eager"
            and _decision_0009_catalog_requires_split_k_raw()
            and (
                self.split_k_raw_inputs is None
                or not self.split_k_raw_inputs.raw_bytes_verified
                or not self.split_k_raw_inputs.split_k_pair_multiplicity
            )
        ):
            errors.append("production_binding_split_k_raw_unverified")
        return tuple(dict.fromkeys(errors))

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_key": self.operation_key.to_dict(),
            "run_id": self.run_id,
            "point_id": self.point_id,
            "point_fingerprint": self.point_fingerprint,
            "runner_kind": self.runner_kind,
            "execution_mode": self.execution_mode,
            "batch": self.batch,
            "starting_context": self.starting_context,
            "decode_step": self.decode_step,
            "process_replicate": self.process_replicate,
            "historical_context": self.historical_context,
            "attended_context": self.attended_context,
            "geometry": self.geometry.to_dict(),
            "cache_layout_id": self.cache_layout_id,
            "cache_capacity": self.cache_capacity,
            "cache_implementation_sha256": (
                self.cache_implementation_sha256
            ),
            "cache_layout_fingerprint": self.cache_layout_fingerprint,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_config_sha256": self.model_config_sha256,
            "adapter_source_sha256": self.adapter_source_sha256,
            "kernel_binary_sha256": self.kernel_binary_sha256,
            "backend_config_id": self.backend_config_id,
            "backend_identity": self.backend_identity,
            "backend_identity_sha256": self.backend_identity_sha256,
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
            "split_k_raw_inputs": (
                None
                if self.split_k_raw_inputs is None
                else self.split_k_raw_inputs.to_dict()
            ),
            "external_provenance_status": self.external_provenance_status,
            "decision_id": self.decision_id,
            "decision_sha256": self.decision_sha256,
            "policy_catalog_id": self.policy_catalog_id,
            "policy_catalog_sha256": self.policy_catalog_sha256,
        }


def _production_point_id(
    runner_kind: str,
    execution_mode: str,
    batch: int,
    starting_context: int,
    decode_step: int,
    process_replicate: int,
) -> str | None:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (
            batch,
            starting_context,
            decode_step,
            process_replicate,
        )
    ):
        return None
    if batch not in {1, 4}:
        return None
    if runner_kind == "fixed_l":
        if (
            execution_mode not in {"eager", "cuda_graph"}
            or starting_context not in {128, 4096, 16384}
            or decode_step != 0
        ):
            return None
        maximum_replicate = (
            3
            if batch == 1 and starting_context == 4096
            else 1
        )
        if process_replicate not in set(range(1, maximum_replicate + 1)):
            return None
    elif runner_kind == "growing_context":
        if (
            execution_mode != "eager"
            or starting_context not in {128, 4096}
            or decode_step not in set(range(16))
            or process_replicate != 1
        ):
            return None
    else:
        return None
    return (
        f"{runner_kind}-b{batch}-l{starting_context}-"
        f"{execution_mode}-r{process_replicate}"
    )


def build_phase3_production_allocation_binding(
    *,
    operation_key: Phase3AuditOperationKey,
    backend_identity: str,
    split_k_raw_inputs: SplitKCompositeRawInputs | None = None,
) -> ProductionAllocationBinding:
    """Instantiate only a preregistered Phase 3 point deterministically."""

    if type(operation_key) is not Phase3AuditOperationKey:
        raise AllocationAttributionError(
            "operation_key must be an exact Phase3AuditOperationKey"
        )
    run_id = operation_key.run_id
    runner_kind = operation_key.runner_kind.value
    execution_mode = operation_key.allocation_execution_mode
    batch = operation_key.batch_size
    starting_context = (
        operation_key.historical_context - operation_key.decode_step
    )
    decode_step = operation_key.decode_step
    process_replicate = operation_key.process_replicate
    point_id = _production_point_id(
        runner_kind,
        execution_mode,
        batch,
        starting_context,
        decode_step,
        process_replicate,
    )
    if point_id is None:
        raise AllocationAttributionError(
            "allocation binding point is outside the preregistered grid"
        )
    if backend_identity != PHASE3_BACKEND_IDENTITY:
        raise AllocationAttributionError(
            "backend_identity must equal the frozen canonical Phase 3 identity"
        )
    historical_context = starting_context + decode_step
    attended_context = historical_context + 1
    point_fingerprint = _phase3_point_fingerprint(
        point_id=point_id,
        runner_kind=runner_kind,
        execution_mode=execution_mode,
        batch=batch,
        starting_context=starting_context,
        process_replicate=process_replicate,
    )
    geometry = AllocationGeometry(
        batch=batch,
        query_heads=32,
        kv_heads=8,
        context=attended_context,
        head_dim=128,
        dtype_bytes=2,
        query_length=1,
        operation_output_width=PHASE3_OUTPUT_WIDTH,
        operation_output_dtype_bytes=PHASE3_OUTPUT_DTYPE_BYTES,
    )
    backend_sha = hashlib.sha256(backend_identity.encode("utf-8")).hexdigest()
    cache_capacity, cache_layout_fingerprint = (
        _phase3_cache_layout_fingerprint(
            runner_kind=runner_kind,
            batch=batch,
            starting_context=starting_context,
        )
    )
    if operation_key.point_id != point_id:
        raise AllocationAttributionError(
            "operation_key point differs from the preregistered allocation point"
        )
    if operation_key.point_fingerprint != point_fingerprint:
        raise AllocationAttributionError(
            "operation_key point fingerprint differs from the frozen point"
        )
    if operation_key.cache_layout_fingerprint != cache_layout_fingerprint:
        raise AllocationAttributionError(
            "operation_key cache fingerprint differs from the frozen layout"
        )
    if operation_key.capacity != cache_capacity:
        raise AllocationAttributionError(
            "operation_key capacity differs from the frozen layout"
        )
    if operation_key.backend_identity_sha256 != backend_sha:
        raise AllocationAttributionError(
            "operation_key backend identity differs from the frozen backend"
        )
    binding = ProductionAllocationBinding(
        operation_key=operation_key,
        run_id=run_id,
        point_id=point_id,
        point_fingerprint=point_fingerprint,
        runner_kind=runner_kind,
        execution_mode=execution_mode,
        batch=batch,
        starting_context=starting_context,
        decode_step=decode_step,
        process_replicate=process_replicate,
        historical_context=historical_context,
        attended_context=attended_context,
        geometry=geometry,
        cache_layout_id=PHASE3_CACHE_LAYOUT_ID,
        cache_capacity=cache_capacity,
        cache_implementation_sha256=PHASE3_CACHE_IMPLEMENTATION_SHA256,
        cache_layout_fingerprint=cache_layout_fingerprint,
        model_id=PHASE3_MODEL_ID,
        model_revision=PHASE3_MODEL_REVISION,
        model_config_sha256=PHASE3_MODEL_CONFIG_SHA256,
        adapter_source_sha256=PHASE3_ADAPTER_SOURCE_SHA256,
        kernel_binary_sha256=PHASE3_KERNEL_BINARY_SHA256,
        backend_config_id=PHASE3_BACKEND_CONFIG_ID,
        backend_identity=backend_identity,
        backend_identity_sha256=backend_sha,
        operation_fingerprint_sha256=(
            operation_key.operation_fingerprint_sha256
        ),
        split_k_raw_inputs=split_k_raw_inputs,
    )
    errors = binding.validation_errors()
    if errors:
        raise AllocationAttributionError(
            "invalid production allocation binding: " + ", ".join(errors)
        )
    return binding


@dataclass(frozen=True, slots=True)
class OperationCacheStateWitness:
    """Raw cache geometry, pointers, and region digests around one decode."""

    active_length: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_strides: tuple[int, ...]
    value_strides: tuple[int, ...]
    key_dtype: str
    value_dtype: str
    key_device: str
    value_device: str
    key_data_ptr: int
    value_data_ptr: int
    historical_prefix_sha256: str
    destination_slot_sha256: str
    destination_slot_is_sentinel: bool
    layout_fingerprint: str

    def __post_init__(self) -> None:
        _positive_int("operation witness active_length", self.active_length)
        for name in ("key_shape", "value_shape", "key_strides", "value_strides"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(values) != 5
                or any(type(item) is not int or item <= 0 for item in values)
            ):
                raise AllocationAttributionError(
                    f"operation witness {name} must contain five positive integers"
                )
        for name in ("key_dtype", "value_dtype", "key_device", "value_device"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise AllocationAttributionError(
                    f"operation witness {name} must be nonempty"
                )
        for name in ("key_data_ptr", "value_data_ptr"):
            _positive_int(f"operation witness {name}", getattr(self, name))
        if self.key_data_ptr == self.value_data_ptr:
            raise AllocationAttributionError(
                "operation witness key/value pointers must be distinct"
            )
        for name in (
            "historical_prefix_sha256",
            "destination_slot_sha256",
            "layout_fingerprint",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise AllocationAttributionError(
                    f"operation witness {name} must be SHA-256"
                )
        if type(self.destination_slot_is_sentinel) is not bool:
            raise AllocationAttributionError(
                "operation witness destination sentinel flag must be boolean"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> OperationCacheStateWitness:
        _require_exact_mapping_keys(
            value,
            {
                "active_length",
                "key_shape",
                "value_shape",
                "key_strides",
                "value_strides",
                "key_dtype",
                "value_dtype",
                "key_device",
                "value_device",
                "key_data_ptr",
                "value_data_ptr",
                "historical_prefix_sha256",
                "destination_slot_sha256",
                "destination_slot_is_sentinel",
                "layout_fingerprint",
            },
            "operation cache witness",
        )
        sequence_fields = (
            "key_shape",
            "value_shape",
            "key_strides",
            "value_strides",
        )
        if any(not isinstance(value.get(name), list) for name in sequence_fields):
            raise AllocationAttributionError(
                "operation cache witness shapes/strides must be arrays"
            )
        return cls(
            active_length=value.get("active_length"),
            key_shape=tuple(value["key_shape"]),
            value_shape=tuple(value["value_shape"]),
            key_strides=tuple(value["key_strides"]),
            value_strides=tuple(value["value_strides"]),
            key_dtype=value.get("key_dtype"),
            value_dtype=value.get("value_dtype"),
            key_device=value.get("key_device"),
            value_device=value.get("value_device"),
            key_data_ptr=value.get("key_data_ptr"),
            value_data_ptr=value.get("value_data_ptr"),
            historical_prefix_sha256=value.get("historical_prefix_sha256"),
            destination_slot_sha256=value.get("destination_slot_sha256"),
            destination_slot_is_sentinel=value.get(
                "destination_slot_is_sentinel"
            ),
            layout_fingerprint=value.get("layout_fingerprint"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_length": self.active_length,
            "key_shape": list(self.key_shape),
            "value_shape": list(self.value_shape),
            "key_strides": list(self.key_strides),
            "value_strides": list(self.value_strides),
            "key_dtype": self.key_dtype,
            "value_dtype": self.value_dtype,
            "key_device": self.key_device,
            "value_device": self.value_device,
            "key_data_ptr": self.key_data_ptr,
            "value_data_ptr": self.value_data_ptr,
            "historical_prefix_sha256": self.historical_prefix_sha256,
            "destination_slot_sha256": self.destination_slot_sha256,
            "destination_slot_is_sentinel": (
                self.destination_slot_is_sentinel
            ),
            "layout_fingerprint": self.layout_fingerprint,
        }


def _cache_state_validation_errors(
    state: OperationCacheStateWitness,
    label: str,
    binding: ProductionAllocationBinding,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_shape = (
        32,
        binding.batch,
        8,
        binding.cache_capacity,
        128,
    )
    expected_strides = (
        binding.batch * 8 * binding.cache_capacity * 128,
        8 * binding.cache_capacity * 128,
        binding.cache_capacity * 128,
        128,
        1,
    )
    if state.key_shape != expected_shape or state.value_shape != expected_shape:
        errors.append(f"operation_witness_{label}_cache_shape_mismatch")
    if (
        state.key_strides != expected_strides
        or state.value_strides != expected_strides
    ):
        errors.append(f"operation_witness_{label}_cache_strides_mismatch")
    if (
        state.key_dtype != PHASE3_OUTPUT_DTYPE
        or state.value_dtype != PHASE3_OUTPUT_DTYPE
    ):
        errors.append(f"operation_witness_{label}_cache_dtype_mismatch")
    if (
        state.key_device != PHASE3_DEVICE
        or state.value_device != PHASE3_DEVICE
    ):
        errors.append(f"operation_witness_{label}_cache_device_mismatch")
    workspace_bytes = 32 * binding.batch * (32 + 8) * 1 * 64 * 2
    try:
        derived_layout = derive_cache_layout_fingerprint(
            num_layers=state.key_shape[0],
            batch_size=state.key_shape[1],
            num_kv_heads=state.key_shape[2],
            capacity=state.key_shape[3],
            head_dim=state.key_shape[4],
            device=state.key_device,
            workspace_bytes=workspace_bytes,
            implementation_sha256=PHASE3_CACHE_IMPLEMENTATION_SHA256,
        )
    except ValueError:
        derived_layout = ""
    if (
        state.layout_fingerprint != derived_layout
        or state.layout_fingerprint != binding.cache_layout_fingerprint
    ):
        errors.append(f"operation_witness_{label}_cache_layout_mismatch")
    return tuple(errors)


def _phase3_zero_destination_sentinel_sha256(
    binding: ProductionAllocationBinding,
) -> str:
    shape = (32, binding.batch, 8, 1, 128)
    header = _canonical_json_bytes(
        {"shape": list(shape), "dtype": PHASE3_OUTPUT_DTYPE}
    )
    tensor_digest = hashlib.sha256()
    tensor_digest.update(header)
    tensor_digest.update(b"\0")
    tensor_digest.update(
        b"\0"
        * (
            shape[0]
            * shape[1]
            * shape[2]
            * shape[3]
            * shape[4]
            * PHASE3_OUTPUT_DTYPE_BYTES
        )
    )
    component = tensor_digest.hexdigest()
    return hashlib.sha256(f"{component}:{component}".encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class OperationOutputWitness:
    """Untimed checksum and metadata for the witnessed decode output."""

    sha256: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    finite: bool

    def __post_init__(self) -> None:
        if not _valid_sha256(self.sha256):
            raise AllocationAttributionError(
                "operation output witness must contain a SHA-256"
            )
        if not self.shape:
            raise AllocationAttributionError(
                "operation output witness shape must be nonempty"
            )
        for dimension in self.shape:
            _positive_int("operation output dimension", dimension)
        if not self.dtype or not self.device:
            raise AllocationAttributionError(
                "operation output witness dtype/device must be nonempty"
            )
        if not isinstance(self.finite, bool):
            raise AllocationAttributionError(
                "operation output witness finite flag must be boolean"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> OperationOutputWitness:
        _require_exact_mapping_keys(
            value,
            {"sha256", "shape", "dtype", "device", "finite"},
            "operation output witness",
        )
        raw_shape = value.get("shape")
        if not isinstance(raw_shape, list):
            raise AllocationAttributionError(
                "operation output witness shape must be a list"
            )
        return cls(
            sha256=value.get("sha256"),
            shape=tuple(raw_shape),
            dtype=value.get("dtype"),
            device=value.get("device"),
            finite=value.get("finite"),
        )

    def metadata(self) -> tuple[tuple[int, ...], str, str]:
        return self.shape, self.dtype, self.device

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "finite": self.finite,
        }


@dataclass(frozen=True, slots=True)
class OperationWitnessCallbacks:
    """Caller callbacks used by the untimed allocation audit.

    ``capture_output`` is invoked while allocator history is active so any
    accidental CUDA allocation remains visible and fails the gate.  Production
    CUDA callbacks should delegate to :func:`capture_output_witness_d2h`, which
    copies an already-contiguous result directly to CPU before hashing and
    checking finiteness.
    """

    capture_cache_state: Callable[[], OperationCacheStateWitness]
    capture_output: Callable[[Any], OperationOutputWitness]

    def __post_init__(self) -> None:
        if not callable(self.capture_cache_state) or not callable(
            self.capture_output
        ):
            raise AllocationAttributionError(
                "operation witness callbacks must be callable"
            )


@dataclass(frozen=True, slots=True)
class OperationWitnessEvidence:
    """Raw-derived witness tying the audited callable to a cache transition."""

    operation_key: Phase3AuditOperationKey
    operation_fingerprint_sha256: str
    reference_before: OperationCacheStateWitness
    reference_after: OperationCacheStateWitness
    reference_output: OperationOutputWitness
    measured_before: OperationCacheStateWitness
    measured_after: OperationCacheStateWitness
    measured_output: OperationOutputWitness | None
    recorder_configuration: dict[str, Any]

    def __post_init__(self) -> None:
        if not _valid_sha256(self.operation_fingerprint_sha256):
            raise AllocationAttributionError(
                "operation witness fingerprint must be SHA-256"
            )
        if type(self.operation_key) is not Phase3AuditOperationKey:
            raise AllocationAttributionError(
                "operation witness key must be a Phase3AuditOperationKey"
            )
        if self.recorder_configuration != PHASE3_RECORDER_CONFIGURATION:
            raise AllocationAttributionError(
                "operation witness recorder configuration is not frozen"
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> OperationWitnessEvidence:
        _require_exact_mapping_keys(
            value,
            {
                "operation_key",
                "operation_fingerprint_sha256",
                "reference_before",
                "reference_after",
                "reference_output",
                "measured_before",
                "measured_after",
                "measured_output",
                "recorder_configuration",
            },
            "operation witness evidence",
        )
        key = value.get("operation_key")
        state_fields = (
            "reference_before",
            "reference_after",
            "measured_before",
            "measured_after",
        )
        output = value.get("reference_output")
        measured_output = value.get("measured_output")
        recorder_configuration = value.get("recorder_configuration")
        if (
            not isinstance(key, dict)
            or any(not isinstance(value.get(name), Mapping) for name in state_fields)
            or not isinstance(output, Mapping)
            or (
                measured_output is not None
                and not isinstance(measured_output, Mapping)
            )
            or not isinstance(recorder_configuration, dict)
        ):
            raise AllocationAttributionError(
                "operation witness state/output payload is malformed"
            )
        return cls(
            operation_key=Phase3AuditOperationKey.from_dict(key),
            operation_fingerprint_sha256=value.get(
                "operation_fingerprint_sha256"
            ),
            reference_before=OperationCacheStateWitness.from_mapping(
                value["reference_before"]
            ),
            reference_after=OperationCacheStateWitness.from_mapping(
                value["reference_after"]
            ),
            reference_output=OperationOutputWitness.from_mapping(output),
            measured_before=OperationCacheStateWitness.from_mapping(
                value["measured_before"]
            ),
            measured_after=OperationCacheStateWitness.from_mapping(
                value["measured_after"]
            ),
            measured_output=(
                None
                if measured_output is None
                else OperationOutputWitness.from_mapping(measured_output)
            ),
            recorder_configuration=dict(recorder_configuration),
        )

    def validation_errors(
        self, binding: ProductionAllocationBinding
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if self.operation_key != binding.operation_key:
            errors.append("operation_witness_operation_key_mismatch")
        if (
            self.operation_fingerprint_sha256
            != binding.operation_fingerprint_sha256
        ):
            errors.append("operation_witness_fingerprint_mismatch")
        states = {
            "reference_before": self.reference_before,
            "reference_after": self.reference_after,
            "measured_before": self.measured_before,
            "measured_after": self.measured_after,
        }
        for label, state in states.items():
            errors.extend(_cache_state_validation_errors(state, label, binding))
        expected_after_length = (
            binding.historical_context
            if binding.runner_kind == "fixed_l"
            else binding.attended_context
        )
        if self.reference_before.active_length != binding.historical_context:
            errors.append("operation_witness_before_length_mismatch")
        if self.reference_after.active_length != expected_after_length:
            errors.append("operation_witness_after_length_mismatch")
        if self.measured_before != self.reference_before:
            errors.append("operation_witness_prepare_not_reproducible")
        if self.measured_after != self.reference_after:
            errors.append("operation_witness_post_state_mismatch")
        pointer_transcript = {
            (state.key_data_ptr, state.value_data_ptr)
            for state in states.values()
        }
        if len(pointer_transcript) != 1:
            errors.append("operation_witness_cache_pointers_changed")
        if len(
            {state.historical_prefix_sha256 for state in states.values()}
        ) != 1:
            errors.append("operation_witness_historical_prefix_changed")
        for prefix in ("reference", "measured"):
            before = states[f"{prefix}_before"]
            after = states[f"{prefix}_after"]
            expected_sentinel = _phase3_zero_destination_sentinel_sha256(
                binding
            )
            if not before.destination_slot_is_sentinel:
                errors.append(
                    f"operation_witness_{prefix}_destination_not_prepared"
                )
            if before.destination_slot_sha256 != expected_sentinel:
                errors.append(
                    f"operation_witness_{prefix}_destination_sentinel_mismatch"
                )
            if after.destination_slot_is_sentinel:
                errors.append(
                    f"operation_witness_{prefix}_destination_not_written"
                )
            if before.destination_slot_sha256 == after.destination_slot_sha256:
                errors.append(
                    f"operation_witness_{prefix}_destination_unchanged"
                )
        expected_output = (
            binding.batch,
            binding.geometry.query_length,
            PHASE3_OUTPUT_WIDTH,
        )
        if self.reference_output.shape != expected_output:
            errors.append("operation_witness_output_geometry_mismatch")
        if self.reference_output.dtype != PHASE3_OUTPUT_DTYPE:
            errors.append("operation_witness_output_dtype_mismatch")
        if self.reference_output.device != PHASE3_DEVICE:
            errors.append("operation_witness_output_device_mismatch")
        if not self.reference_output.finite:
            errors.append("operation_witness_output_nonfinite")
        if self.measured_output is None:
            errors.append("operation_witness_measured_output_missing")
        else:
            if self.measured_output.shape != expected_output:
                errors.append(
                    "operation_witness_measured_output_geometry_mismatch"
                )
            if self.measured_output.dtype != PHASE3_OUTPUT_DTYPE:
                errors.append(
                    "operation_witness_measured_output_dtype_mismatch"
                )
            if self.measured_output.device != PHASE3_DEVICE:
                errors.append(
                    "operation_witness_measured_output_device_mismatch"
                )
            if not self.measured_output.finite:
                errors.append(
                    "operation_witness_measured_output_nonfinite"
                )
            if self.measured_output != self.reference_output:
                errors.append("operation_witness_measured_output_mismatch")
        if self.recorder_configuration != PHASE3_RECORDER_CONFIGURATION:
            errors.append("operation_witness_recorder_configuration_mismatch")
        return tuple(dict.fromkeys(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_key": self.operation_key.to_dict(),
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
            "reference_before": self.reference_before.to_dict(),
            "reference_after": self.reference_after.to_dict(),
            "reference_output": self.reference_output.to_dict(),
            "measured_before": self.measured_before.to_dict(),
            "measured_after": self.measured_after.to_dict(),
            "measured_output": (
                None
                if self.measured_output is None
                else self.measured_output.to_dict()
            ),
            "recorder_configuration": dict(self.recorder_configuration),
        }


def _operation_output_metadata(
    value: Any,
) -> tuple[tuple[int, ...], str, str]:
    raw_shape = getattr(value, "shape", None)
    raw_dtype = getattr(value, "dtype", None)
    raw_device = getattr(value, "device", None)
    if raw_shape is None or raw_dtype is None or raw_device is None:
        raise AllocationAttributionError(
            "audited operation must return a tensor-like output"
        )
    try:
        shape = tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError) as error:
        raise AllocationAttributionError(
            "audited operation output shape is invalid"
        ) from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise AllocationAttributionError(
            "audited operation output shape is invalid"
        )
    return shape, str(raw_dtype), str(raw_device)


def capture_output_witness_d2h(value: Any) -> OperationOutputWitness:
    """Hash a contiguous CUDA output after a blocking direct-to-CPU copy.

    No contiguity conversion or numerical operation is performed on CUDA.  If
    PyTorch nevertheless emits a CUDA allocator event for this callback, the
    callback executes inside the recorder window and the event remains part of
    the raw audit evidence.
    """

    shape, dtype, device = _operation_output_metadata(value)
    if device != PHASE3_DEVICE:
        raise AllocationAttributionError(
            "operation output D2H witness requires frozen cuda:0"
        )
    is_contiguous = getattr(value, "is_contiguous", None)
    if not callable(is_contiguous) or is_contiguous() is not True:
        raise AllocationAttributionError(
            "operation output D2H witness requires a contiguous CUDA tensor"
        )
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise AllocationAttributionError(
            "operation output D2H witness requires a detachable tensor"
        )
    detached = detach()
    to = getattr(detached, "to", None)
    if not callable(to):
        raise AllocationAttributionError(
            "operation output D2H witness cannot copy the tensor to CPU"
        )
    try:
        cpu_value = to(
            device="cpu",
            non_blocking=False,
            copy=True,
        )
    except (TypeError, RuntimeError) as error:
        raise AllocationAttributionError(
            "operation output D2H witness copy failed"
        ) from error
    cpu_device = getattr(cpu_value, "device", None)
    cpu_is_contiguous = getattr(cpu_value, "is_contiguous", None)
    if (
        str(cpu_device) != "cpu"
        or not callable(cpu_is_contiguous)
        or cpu_is_contiguous() is not True
    ):
        raise AllocationAttributionError(
            "operation output D2H witness did not produce contiguous CPU data"
        )
    torch = _torch_module()
    try:
        byte_view = cpu_value.view(torch.uint8)
        storage_offset = int(byte_view.storage_offset())
        storage = byte_view.untyped_storage()
        expected_storage_bytes = int(byte_view.numel())
        if storage_offset != 0 or storage.nbytes() != expected_storage_bytes:
            raise AllocationAttributionError(
                "operation output D2H witness CPU storage is not exact"
            )
        raw_bytes = bytes(storage)
        finite = bool(torch.isfinite(cpu_value).all().item())
    except AllocationAttributionError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise AllocationAttributionError(
            "operation output D2H witness CPU inspection failed"
        ) from error
    canonical_header = _canonical_json_bytes(
        {
            "shape": list(shape),
            "dtype": dtype,
        }
    )
    digest = hashlib.sha256()
    digest.update(canonical_header)
    digest.update(b"\0")
    digest.update(raw_bytes)
    return OperationOutputWitness(
        sha256=digest.hexdigest(),
        shape=shape,
        dtype=dtype,
        device=device,
        finite=finite,
    )


def _allocation_audit_capture_output(
    callback: Callable[[Any], OperationOutputWitness], value: Any
) -> OperationOutputWitness:
    """Keep output-witness CUDA allocations identifiable in trace stacks."""

    return callback(value)


@dataclass(frozen=True, slots=True)
class AttributionRules:
    """Preregistered facts not inferable from one allocator trace."""

    frozen_backend_identity: str | None = None
    permitted_allocation_policies: tuple[AllocationClassPolicy, ...] = ()
    split_k_expected_pair_count: int | None = None
    policy_authority: str = "structural_test_only"
    policy_catalog_id: str | None = None
    policy_catalog_sha256: str | None = None
    production_binding_sha256: str | None = None
    cache_stack_markers: tuple[str, ...] = (
        "static_cache",
        "cache.update",
        "cache_growth",
    )
    audit_instrumentation_python_frames: tuple[CanonicalFrameSelector, ...] = (
        CanonicalFrameSelector(
            function_name="_allocation_audit_capture_output",
            source_suffix="src/kvbench/runtime/allocation_attribution.py",
        ),
    )
    flash_split_k_cpp_markers: tuple[str, ...] = FLASH_SPLIT_K_CPP_MARKERS

    def __post_init__(self) -> None:
        if self.frozen_backend_identity == "":
            raise AllocationAttributionError(
                "frozen_backend_identity must be nonempty when supplied"
            )
        if self.policy_authority not in {
            "structural_test_only",
            "decision_0009_production",
            "decision_0013_direct_compressed_composition",
            "decision_0013_phase8_kivi_composition",
        }:
            raise AllocationAttributionError("unknown policy authority")
        if self.policy_authority in {
            "decision_0009_production",
            "decision_0013_direct_compressed_composition",
            "decision_0013_phase8_kivi_composition",
        }:
            if (
                self.policy_catalog_id != DECISION_0009_POLICY_CATALOG_ID
                or self.policy_catalog_sha256
                != DECISION_0009_POLICY_CATALOG_SHA256
                or self.production_binding_sha256 is None
                or not _valid_sha256(self.production_binding_sha256)
                or self.frozen_backend_identity is None
            ):
                raise AllocationAttributionError(
                    "production rules lack Decision-0009 trust binding"
                )
        policy_ids = [
            policy.policy_id for policy in self.permitted_allocation_policies
        ]
        if len(policy_ids) != len(set(policy_ids)):
            raise AllocationAttributionError(
                "allocation policy IDs must be unique"
            )
        if (
            not self.audit_instrumentation_python_frames
            or any(
                type(selector) is not CanonicalFrameSelector
                for selector in self.audit_instrumentation_python_frames
            )
        ):
            raise AllocationAttributionError(
                "audit instrumentation requires exact canonical frames"
            )
        if self.split_k_expected_pair_count is not None:
            _positive_int(
                "split_k_expected_pair_count",
                self.split_k_expected_pair_count,
            )
        for name in (
            "cache_stack_markers",
            "flash_split_k_cpp_markers",
        ):
            markers = getattr(self, name)
            if not markers or any(not marker for marker in markers):
                raise AllocationAttributionError(
                    f"{name} must contain nonempty markers"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen_backend_identity": self.frozen_backend_identity,
            "permitted_allocation_policies": [
                policy.to_dict()
                for policy in self.permitted_allocation_policies
            ],
            "split_k_expected_pair_count": self.split_k_expected_pair_count,
            "policy_authority": self.policy_authority,
            "policy_catalog_id": self.policy_catalog_id,
            "policy_catalog_sha256": self.policy_catalog_sha256,
            "production_binding_sha256": self.production_binding_sha256,
            "cache_stack_markers": list(self.cache_stack_markers),
            "audit_instrumentation_python_frames": [
                selector.to_dict()
                for selector in self.audit_instrumentation_python_frames
            ],
            "flash_split_k_cpp_markers": list(
                self.flash_split_k_cpp_markers
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


def instantiate_decision_0009_production_rules(
    binding: ProductionAllocationBinding,
) -> AttributionRules:
    """Resolve the Decision 0013 catalog for one frozen operation key."""

    errors = binding.validation_errors()
    if errors:
        raise AllocationAttributionError(
            "invalid production allocation binding: " + ", ".join(errors)
        )
    policies: list[AllocationClassPolicy] = []
    for template in _decision_0009_catalog_payload()[
        "production_templates"
    ]:
        if (
            template.get("event_class")
            == AllocationClass.CONTEXT_SCALED_WORKSPACE.value
        ):
            continue
        formula_id = template["formula_id"]
        event_class = AllocationClass(template["event_class"])
        contract = _POLICY_FORMULA_CONTRACTS[formula_id]
        if contract[0] is not event_class:
            raise AllocationAttributionError(
                "catalog formula and event class differ"
            )
        exact_bytes = template["exact_requested_bytes"]
        if exact_bytes is None:
            exact_bytes = _policy_geometry_bytes(
                formula_id, binding.geometry
            )
        if type(exact_bytes) is not int or exact_bytes <= 0:
            raise AllocationAttributionError(
                "catalog formula did not resolve to positive bytes"
            )
        count = template["exact_count"]
        if type(count) is not int or count <= 0:
            raise AllocationAttributionError(
                "catalog multiplicity is invalid"
            )
        python_frame = template["required_python_frame"]
        cpp_frame = template["required_cpp_frame"]
        policies.append(
            AllocationClassPolicy(
                policy_id=template["policy_id"],
                event_class=event_class,
                formula_id=formula_id,
                allowed_requested_bytes=frozenset({exact_bytes}),
                required_python_frames=(
                    CanonicalFrameSelector(
                        python_frame["function_name"],
                        python_frame["source_suffix"],
                    ),
                ),
                required_cpp_frames=(
                    CanonicalFrameSelector(
                        cpp_frame["function_name"],
                        cpp_frame["source_suffix"],
                    ),
                ),
                dependencies=contract[1],
                exact_count=count,
                exact_total_requested_bytes=exact_bytes * count,
            )
        )
    split_pair_count: int | None = None
    if binding.execution_mode == "eager":
        assert binding.split_k_raw_inputs is not None
        multiplicity = (
            binding.split_k_raw_inputs.split_k_pair_multiplicity
        )
        if len(multiplicity) != 1 or multiplicity[0][1] != 1:
            raise AllocationAttributionError(
                "paired split-K control multiplicity is not one exact pair"
            )
        split_pair_count = PHASE3_LAYER_COUNT
    return AttributionRules(
        frozen_backend_identity=binding.backend_identity,
        permitted_allocation_policies=tuple(policies),
        split_k_expected_pair_count=split_pair_count,
        policy_authority="decision_0009_production",
        policy_catalog_id=DECISION_0009_POLICY_CATALOG_ID,
        policy_catalog_sha256=DECISION_0009_POLICY_CATALOG_SHA256,
        production_binding_sha256=binding.identity_sha256,
    )


def _instantiate_decision_0013_direct_compressed_rules(
    *,
    geometry: AllocationGeometry,
    backend_identity: str,
    composition_binding_sha256: str,
    policy_authority: str,
) -> AttributionRules:
    """Bind direct compressed decode to ordinary Decision 0013 allocations.

    A direct compressed-cache method replaces every Flash-attention call in
    the 32-layer endpoint. The retained model operations therefore use the
    exact ordinary Decision 0013 policies, while the four Flash-specific
    policies and split-K workspaces are forbidden.
    """

    if type(geometry) is not AllocationGeometry:
        raise AllocationAttributionError(
            "direct-compressed rules require AllocationGeometry"
        )
    if not isinstance(backend_identity, str) or not backend_identity:
        raise AllocationAttributionError(
            "direct-compressed backend identity must be nonempty"
        )
    if not _valid_sha256(composition_binding_sha256):
        raise AllocationAttributionError(
            "direct-compressed composition binding must be a SHA-256"
        )
    if policy_authority not in {
        "decision_0013_direct_compressed_composition",
        "decision_0013_phase8_kivi_composition",
    }:
        raise AllocationAttributionError(
            "direct-compressed policy authority is invalid"
        )
    if not _decision_0009_catalog_is_intact():
        raise AllocationAttributionError(
            "Decision 0013 allocation catalog identity differs"
        )

    policies: list[AllocationClassPolicy] = []
    for template in _decision_0009_catalog_payload()[
        "production_templates"
    ]:
        if (
            template.get("event_class")
            == AllocationClass.CONTEXT_SCALED_WORKSPACE.value
        ):
            continue
        python_frame = template["required_python_frame"]
        if (
            python_frame.get("source_suffix")
            == "src/kvbench/runtime/backend.py"
        ):
            continue
        formula_id = template["formula_id"]
        event_class = AllocationClass(template["event_class"])
        contract = _POLICY_FORMULA_CONTRACTS[formula_id]
        if contract[0] is not event_class:
            raise AllocationAttributionError(
                "catalog formula and event class differ"
            )
        exact_bytes = template["exact_requested_bytes"]
        if exact_bytes is None:
            exact_bytes = _policy_geometry_bytes(formula_id, geometry)
        if type(exact_bytes) is not int or exact_bytes <= 0:
            raise AllocationAttributionError(
                "catalog formula did not resolve to positive bytes"
            )
        count = template["exact_count"]
        if type(count) is not int or count <= 0:
            raise AllocationAttributionError(
                "catalog multiplicity is invalid"
            )
        cpp_frame = template["required_cpp_frame"]
        policies.append(
            AllocationClassPolicy(
                policy_id=template["policy_id"],
                event_class=event_class,
                formula_id=formula_id,
                allowed_requested_bytes=frozenset({exact_bytes}),
                required_python_frames=(
                    CanonicalFrameSelector(
                        python_frame["function_name"],
                        python_frame["source_suffix"],
                    ),
                ),
                required_cpp_frames=(
                    CanonicalFrameSelector(
                        cpp_frame["function_name"],
                        cpp_frame["source_suffix"],
                    ),
                ),
                dependencies=contract[1],
                exact_count=count,
                exact_total_requested_bytes=exact_bytes * count,
            )
        )
    return AttributionRules(
        frozen_backend_identity=backend_identity,
        permitted_allocation_policies=tuple(policies),
        split_k_expected_pair_count=None,
        policy_authority=policy_authority,
        policy_catalog_id=DECISION_0009_POLICY_CATALOG_ID,
        policy_catalog_sha256=DECISION_0009_POLICY_CATALOG_SHA256,
        production_binding_sha256=composition_binding_sha256,
    )


def instantiate_decision_0013_direct_compressed_rules(
    *,
    geometry: AllocationGeometry,
    backend_identity: str,
    composition_binding_sha256: str,
) -> AttributionRules:
    """Create method-neutral Decision 0013 direct-compressed rules."""

    return _instantiate_decision_0013_direct_compressed_rules(
        geometry=geometry,
        backend_identity=backend_identity,
        composition_binding_sha256=composition_binding_sha256,
        policy_authority="decision_0013_direct_compressed_composition",
    )


def instantiate_decision_0013_phase8_kivi_rules(
    *,
    geometry: AllocationGeometry,
    backend_identity: str,
    composition_binding_sha256: str,
) -> AttributionRules:
    """Preserve the exact Phase 8 KIVI composition identity."""

    return _instantiate_decision_0013_direct_compressed_rules(
        geometry=geometry,
        backend_identity=backend_identity,
        composition_binding_sha256=composition_binding_sha256,
        policy_authority="decision_0013_phase8_kivi_composition",
    )


@dataclass(frozen=True, slots=True)
class AllocatorTraceAttribution:
    """Validated trace, address lifetimes, and formula classifications."""

    trace_sha256: str
    expected_trace_sha256: str | None
    backend_identity: str | None
    geometry: AllocationGeometry
    rules: AttributionRules
    counters: AllocatorCounterEvidence
    events: tuple[TraceEventEvidence, ...]
    allocations: tuple[AllocationLifetime, ...]
    action_counts: dict[str, int]
    segment_alloc_count: int
    segment_free_count: int
    integrity_errors: tuple[str, ...]
    history_integrity: AllocatorHistoryIntegrityEvidence | None = None

    @property
    def all_block_sizes_proven(self) -> bool:
        return all(
            item.allocated_block_size_proven for item in self.allocations
        )

    @property
    def all_lifetimes_fully_freed(self) -> bool:
        return all(item.fully_freed for item in self.allocations)

    @property
    def all_allocations_cache_reused(self) -> bool:
        return all(
            item.cache_reuse_status is CacheReuseStatus.VERIFIED_REUSE
            for item in self.allocations
        )

    def class_counts(self) -> dict[str, int]:
        counts = Counter(item.event_class.value for item in self.allocations)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_sha256": self.trace_sha256,
            "expected_trace_sha256": self.expected_trace_sha256,
            "backend_identity": self.backend_identity,
            "geometry": self.geometry.to_dict(),
            "attribution_rules": self.rules.to_dict(),
            "attribution_rules_sha256": self.rules.identity_sha256,
            "counters": self.counters.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "allocations": [item.to_dict() for item in self.allocations],
            "action_counts": dict(sorted(self.action_counts.items())),
            "class_counts": self.class_counts(),
            "segment_alloc_count": self.segment_alloc_count,
            "segment_free_count": self.segment_free_count,
            "all_block_sizes_proven": self.all_block_sizes_proven,
            "all_lifetimes_fully_freed": self.all_lifetimes_fully_freed,
            "all_allocations_cache_reused": (
                self.all_allocations_cache_reused
            ),
            "integrity_errors": list(self.integrity_errors),
            "history_integrity": (
                None
                if self.history_integrity is None
                else self.history_integrity.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class MemoryDeltaEvidence:
    """Persistent PyTorch and device memory around the isolated operation."""

    allocated_before: int
    allocated_after: int
    reserved_before: int
    reserved_after: int
    device_used_before: int | None
    device_used_after: int | None

    def __post_init__(self) -> None:
        for name in (
            "allocated_before",
            "allocated_after",
            "reserved_before",
            "reserved_after",
            "device_used_before",
            "device_used_after",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise AllocationAttributionError(
                    f"{name} must be a nonnegative integer or None"
                )
        if (self.device_used_before is None) != (
            self.device_used_after is None
        ):
            raise AllocationAttributionError(
                "device-used samples must both be present or both absent"
            )

    @property
    def allocated_delta(self) -> int:
        return self.allocated_after - self.allocated_before

    @property
    def reserved_delta(self) -> int:
        return self.reserved_after - self.reserved_before

    @property
    def device_used_delta(self) -> int | None:
        if (
            self.device_used_before is None
            or self.device_used_after is None
        ):
            return None
        return self.device_used_after - self.device_used_before

    @property
    def non_pytorch_delta(self) -> int | None:
        """Device growth unexplained by PyTorch reserved-memory growth."""

        device_delta = self.device_used_delta
        if device_delta is None:
            return None
        return device_delta - self.reserved_delta

    def to_dict(self) -> dict[str, int | None]:
        return {
            "allocated_before": self.allocated_before,
            "allocated_after": self.allocated_after,
            "allocated_delta": self.allocated_delta,
            "reserved_before": self.reserved_before,
            "reserved_after": self.reserved_after,
            "reserved_delta": self.reserved_delta,
            "device_used_before": self.device_used_before,
            "device_used_after": self.device_used_after,
            "device_used_delta": self.device_used_delta,
            "non_pytorch_delta": self.non_pytorch_delta,
        }


@dataclass(frozen=True, slots=True)
class RawMemoryAccountingSample:
    """Raw PyTorch allocator and device mem_get_info sample."""

    schema_version: str
    operation_fingerprint_sha256: str
    sample_role: str
    timestamp_ns: int
    device: str
    device_index: int
    gpu_uuid: str
    allocated_bytes: int
    reserved_bytes: int
    device_free_bytes: int
    device_total_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != "kvbench-phase3-memory-accounting-2.0.0":
            raise AllocationAttributionError(
                "raw memory accounting schema is not frozen"
            )
        if not _valid_sha256(self.operation_fingerprint_sha256):
            raise AllocationAttributionError(
                "raw memory accounting operation fingerprint is invalid"
            )
        if self.sample_role not in {"before", "after"}:
            raise AllocationAttributionError(
                "raw memory accounting role must be before or after"
            )
        _positive_int("raw memory accounting timestamp", self.timestamp_ns)
        if self.device != PHASE3_DEVICE or self.device_index != PHASE3_DEVICE_INDEX:
            raise AllocationAttributionError(
                "raw memory accounting device must be frozen cuda:0"
            )
        if not isinstance(self.gpu_uuid, str) or not self.gpu_uuid:
            raise AllocationAttributionError(
                "raw memory accounting GPU UUID must be nonempty"
            )
        for name in (
            "allocated_bytes",
            "reserved_bytes",
            "device_free_bytes",
            "device_total_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AllocationAttributionError(
                    f"{name} must be a nonnegative integer"
                )
        if self.allocated_bytes > self.reserved_bytes:
            raise AllocationAttributionError(
                "allocated bytes cannot exceed reserved bytes"
            )
        if self.reserved_bytes > self.device_total_bytes:
            raise AllocationAttributionError(
                "reserved bytes cannot exceed device total bytes"
            )
        if self.device_free_bytes > self.device_total_bytes:
            raise AllocationAttributionError(
                "device free bytes cannot exceed total bytes"
            )

    @property
    def device_used_bytes(self) -> int:
        return self.device_total_bytes - self.device_free_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
            "sample_role": self.sample_role,
            "timestamp_ns": self.timestamp_ns,
            "device": self.device,
            "device_index": self.device_index,
            "gpu_uuid": self.gpu_uuid,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "device_free_bytes": self.device_free_bytes,
            "device_total_bytes": self.device_total_bytes,
            "device_used_bytes": self.device_used_bytes,
        }


def memory_delta_from_raw_samples(
    before: RawMemoryAccountingSample,
    after: RawMemoryAccountingSample,
) -> MemoryDeltaEvidence:
    if before.sample_role != "before" or after.sample_role != "after":
        raise AllocationAttributionError(
            "raw memory accounting samples have wrong roles"
        )
    comparable = (
        before.operation_fingerprint_sha256
        == after.operation_fingerprint_sha256
        and before.device == after.device
        and before.device_index == after.device_index
        and before.gpu_uuid == after.gpu_uuid
        and before.device_total_bytes == after.device_total_bytes
    )
    if not comparable:
        raise AllocationAttributionError(
            "raw memory accounting provenance differs between samples"
        )
    if before.timestamp_ns > after.timestamp_ns:
        raise AllocationAttributionError(
            "raw memory accounting timestamps are out of order"
        )
    return MemoryDeltaEvidence(
        allocated_before=before.allocated_bytes,
        allocated_after=after.allocated_bytes,
        reserved_before=before.reserved_bytes,
        reserved_after=after.reserved_bytes,
        device_used_before=before.device_used_bytes,
        device_used_after=after.device_used_bytes,
    )


@dataclass(frozen=True, slots=True)
class AllocationCriterionResult:
    """Machine-readable gate result.

    An eager pass is described only as attributed bounded ephemeral allocation,
    never as zero-allocation.
    """

    criterion_id: str
    passed: bool
    failure_reasons: tuple[str, ...]
    allocation_event_count: int
    class_counts: dict[str, int]
    no_context_dependent_allocation: bool
    fully_attributed_bounded_ephemeral: bool
    strict_graph_zero_events: bool | None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "criterion_id": self.criterion_id,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "allocation_event_count": self.allocation_event_count,
            "class_counts": dict(sorted(self.class_counts.items())),
            "no_context_dependent_allocation": (
                self.no_context_dependent_allocation
            ),
            "fully_attributed_bounded_ephemeral": (
                self.fully_attributed_bounded_ephemeral
            ),
        }
        if self.strict_graph_zero_events is not None:
            value["strict_graph_zero_events"] = self.strict_graph_zero_events
        return value


@dataclass(slots=True)
class _OpenAllocation:
    allocation_id: int
    event: TraceEventEvidence
    rounded_minimum_bytes: int
    explicit_block_bytes: int | None
    cache_reuse_status: CacheReuseStatus
    triggered_segment_alloc: bool
    free_requested_event_index: int | None = None


def cuda_allocator_rounded_minimum(requested_bytes: int) -> int:
    """Frozen 512-byte lower-bound rounding used for aggregate proofs."""

    _positive_int("requested_bytes", requested_bytes)
    return max(
        CUDA_ALLOCATOR_MIN_BLOCK_BYTES,
        (
            (requested_bytes + CUDA_ALLOCATOR_MIN_BLOCK_BYTES - 1)
            // CUDA_ALLOCATOR_MIN_BLOCK_BYTES
        )
        * CUDA_ALLOCATOR_MIN_BLOCK_BYTES,
    )


def allocator_trace_sha256(
    trace: Sequence[Mapping[str, Any]],
) -> str:
    """Hash a raw trace canonically so later mutation can be rejected."""

    return hashlib.sha256(_canonical_json_bytes(list(trace))).hexdigest()


def allocator_snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    """Hash the complete raw allocator snapshot canonically."""

    return hashlib.sha256(_canonical_json_bytes(snapshot)).hexdigest()


def verify_allocator_trace_sha256(
    trace: Sequence[Mapping[str, Any]],
    expected_sha256: str,
) -> bool:
    """Return whether a raw trace still matches its recorded digest."""

    return allocator_trace_sha256(trace) == expected_sha256


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _ranges_overlap(
    left_start: int,
    left_size: int,
    right_start: int,
    right_size: int,
) -> bool:
    return left_start < right_start + right_size and right_start < (
        left_start + left_size
    )


def _frame_from_value(value: object) -> StackFrame | None:
    if isinstance(value, str):
        return StackFrame(name=value)
    if not isinstance(value, Mapping):
        return None
    raw_name = value.get("name", value.get("function", ""))
    raw_filename = value.get("filename", value.get("file"))
    raw_line = value.get("line", value.get("line_number"))
    name = raw_name if isinstance(raw_name, str) else ""
    filename = raw_filename if isinstance(raw_filename, str) else None
    line = _optional_int(raw_line)
    if not name and filename is None:
        return None
    return StackFrame(name=name, filename=filename, line=line)


def _frames(value: object) -> tuple[StackFrame, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    result: list[StackFrame] = []
    for item in value:
        frame = _frame_from_value(item)
        if frame is not None:
            result.append(frame)
    return tuple(result)


def _looks_python(frame: StackFrame) -> bool:
    filename = (frame.filename or "").lower()
    return (
        filename.endswith((".py", ".pyw"))
        or filename in {"<stdin>", "<string>"}
    )


def _event_stacks(
    raw: Mapping[str, Any],
) -> tuple[tuple[StackFrame, ...], tuple[StackFrame, ...]]:
    python_stack = _frames(raw.get("python_stack"))
    cpp_stack = _frames(raw.get("cpp_stack"))
    if python_stack or cpp_stack:
        return python_stack, cpp_stack
    generic = _frames(raw.get("frames"))
    return (
        tuple(frame for frame in generic if _looks_python(frame)),
        tuple(frame for frame in generic if not _looks_python(frame)),
    )


def _python_stack_text(lifetime: AllocationLifetime) -> str:
    return "\n".join(frame.render() for frame in lifetime.python_stack)


def _cpp_stack_text(lifetime: AllocationLifetime) -> str:
    return "\n".join(frame.render() for frame in lifetime.cpp_stack)


def _stack_text(lifetime: AllocationLifetime) -> str:
    return "\n".join(
        frame.render()
        for frame in (*lifetime.python_stack, *lifetime.cpp_stack)
    )


def _has_any_marker(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _has_all_markers(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return bool(markers) and all(
        marker.lower() in lowered for marker in markers
    )


def _has_all_frame_selectors(
    frames: Sequence[StackFrame],
    selectors: Sequence[CanonicalFrameSelector],
) -> bool:
    return bool(selectors) and all(
        any(selector.matches(frame) for frame in frames)
        for selector in selectors
    )


def _policy_geometry_bytes(
    formula_id: str,
    geometry: AllocationGeometry,
) -> int | None:
    if formula_id == "attention_output_geometry_bytes_v1":
        return geometry.output_bytes
    if formula_id == "attention_output_shared_geometry_bytes_v1":
        return geometry.output_bytes
    if formula_id == "operation_output_geometry_bytes_v1":
        return geometry.operation_output_bytes
    if formula_id == "kv_projection_geometry_bytes_v1":
        return geometry.kv_projection_bytes
    if formula_id == "kv_projection_single_geometry_bytes_v1":
        return geometry.kv_projection_single_bytes
    if formula_id == "hidden_bf16_geometry_bytes_v1":
        return geometry.hidden_bf16_bytes
    if formula_id == "hidden_fp32_geometry_bytes_v1":
        return geometry.hidden_fp32_bytes
    if formula_id == "intermediate_bf16_geometry_bytes_v1":
        return geometry.intermediate_bf16_bytes
    if formula_id == "batch_fp32_scalar_geometry_bytes_v1":
        return geometry.batch_fp32_scalar_bytes
    if formula_id == "flash_lse_geometry_bytes_v1":
        return geometry.flash_lse_bytes
    return None


def _classify(
    lifetime: AllocationLifetime,
    geometry: AllocationGeometry,
    rules: AttributionRules,
) -> AllocationLifetime:
    size = lifetime.requested_bytes
    stack = _stack_text(lifetime)
    cpp_stack = _cpp_stack_text(lifetime)

    policy_matches: list[AllocationClassPolicy] = []
    for policy in rules.permitted_allocation_policies:
        if size not in policy.allowed_requested_bytes:
            continue
        geometry_bytes = _policy_geometry_bytes(policy.formula_id, geometry)
        if policy.formula_id == "operation_output_geometry_bytes_v1" and (
            geometry_bytes is None
        ):
            continue
        if geometry_bytes is not None and size != geometry_bytes:
            continue
        if not _has_all_frame_selectors(
            lifetime.python_stack, policy.required_python_frames
        ) or not _has_all_frame_selectors(
            lifetime.cpp_stack, policy.required_cpp_frames
        ):
            continue
        policy_matches.append(policy)
    independently_proven_non_attention_collision = (
        len(policy_matches) == 1
        and policy_matches[0].formula_id
        in {
            "operation_output_geometry_bytes_v1",
            "fixed_shared_scalar_exact_bytes_v1",
            "framework_scalar_exact_bytes_v1",
        }
        and not _has_any_marker(
            stack,
            (
                "attention",
                "scaled_dot_product",
                "sdpa",
                "flash",
                "repeat",
                "expand",
                "copy",
            ),
        )
    )

    if (
        geometry.query_heads > geometry.kv_heads
        and not independently_proven_non_attention_collision
    ):
        if size == geometry.expanded_kv_combined_bytes:
            return replace(
                lifetime,
                event_class=AllocationClass.GQA_EXPANSION,
                size_formula="expanded_kv_combined",
                dependencies=DependencyFlags(True, True, True, False),
            )
        if size == geometry.expanded_kv_single_bytes:
            return replace(
                lifetime,
                event_class=AllocationClass.GQA_EXPANSION,
                size_formula="expanded_kv_single",
                dependencies=DependencyFlags(True, True, True, False),
            )

    if (
        size == geometry.native_kv_combined_bytes
        and not independently_proven_non_attention_collision
    ):
        return replace(
            lifetime,
            event_class=AllocationClass.CACHE_GROWTH,
            size_formula="native_kv_combined",
            dependencies=DependencyFlags(True, True, False, True),
        )
    if (
        size == geometry.native_kv_single_bytes
        and not independently_proven_non_attention_collision
    ):
        return replace(
            lifetime,
            event_class=AllocationClass.CACHE_GROWTH,
            size_formula="native_kv_single",
            dependencies=DependencyFlags(True, True, False, True),
        )
    if _has_any_marker(stack, rules.cache_stack_markers):
        return replace(
            lifetime,
            event_class=AllocationClass.CACHE_GROWTH,
            dependencies=DependencyFlags(None, True, None, None),
        )
    if _has_all_frame_selectors(
        lifetime.python_stack,
        rules.audit_instrumentation_python_frames,
    ):
        return replace(
            lifetime,
            event_class=AllocationClass.AUDIT_INSTRUMENTATION,
        )

    split_formula: str | None = None
    derived_splits: int | None = None
    output_base = geometry.flash_split_k_output_accumulator_bytes(1)
    lse_base = geometry.flash_split_k_lse_bytes(1)
    if _has_any_marker(cpp_stack, rules.flash_split_k_cpp_markers):
        if size % output_base == 0 and size // output_base > 1:
            split_formula = "flash_split_k_output_accumulator"
            derived_splits = size // output_base
        elif size % lse_base == 0 and size // lse_base > 1:
            split_formula = "flash_split_k_lse"
            derived_splits = size // lse_base
    if (
        split_formula is not None
        and derived_splits is not None
        and _has_all_markers(cpp_stack, rules.flash_split_k_cpp_markers)
    ):
        return replace(
            lifetime,
            event_class=AllocationClass.CONTEXT_SCALED_WORKSPACE,
            size_formula=split_formula,
            formula_parameters=(("num_splits", derived_splits),),
            dependencies=DependencyFlags(True, True, True, False),
        )
    if len(policy_matches) == 1:
        policy = policy_matches[0]
        return replace(
            lifetime,
            event_class=policy.event_class,
            size_formula=policy.formula_id,
            policy_id=policy.policy_id,
            dependencies=policy.dependencies,
        )
    if len(policy_matches) > 1:
        return replace(lifetime, size_formula="ambiguous_permitted_policy")
    if split_formula is not None and derived_splits is not None:
        return replace(
            lifetime,
            size_formula=f"{split_formula}_stack_unverified",
            formula_parameters=(("num_splits", derived_splits),),
        )
    return lifetime


def _make_lifetime(
    opened: _OpenAllocation,
    free_completed_index: int | None,
) -> AllocationLifetime:
    event = opened.event
    if event.address is None or event.size_bytes is None:
        raise AssertionError("invalid allocation cannot form a lifetime")
    return AllocationLifetime(
        allocation_id=opened.allocation_id,
        address=event.address,
        requested_bytes=event.size_bytes,
        rounded_minimum_bytes=opened.rounded_minimum_bytes,
        allocated_block_bytes=opened.explicit_block_bytes,
        allocated_block_size_proven=opened.explicit_block_bytes is not None,
        stream=event.stream,
        alloc_event_index=event.index,
        free_requested_event_index=opened.free_requested_event_index,
        free_completed_event_index=free_completed_index,
        fully_freed=free_completed_index is not None,
        cache_reuse_status=opened.cache_reuse_status,
        triggered_segment_alloc=opened.triggered_segment_alloc,
        python_stack=event.python_stack,
        cpp_stack=event.cpp_stack,
        event_class=AllocationClass.UNKNOWN,
        size_formula=None,
        policy_id=None,
        formula_parameters=(),
        dependencies=UNKNOWN_DEPENDENCIES,
    )


def _reconcile_block_sizes(
    allocations: list[AllocationLifetime],
    counters: AllocatorCounterEvidence,
    errors: list[str],
) -> list[AllocationLifetime]:
    observed_count = len(allocations)
    observed_requested = sum(item.requested_bytes for item in allocations)
    if counters.allocation_count is None:
        errors.append("allocator_counter_allocation_count_missing")
    elif counters.allocation_count != observed_count:
        errors.append("allocator_counter_allocation_count_mismatch")
    if counters.requested_bytes is None:
        errors.append("allocator_counter_requested_bytes_missing")
    elif counters.requested_bytes != observed_requested:
        errors.append("allocator_counter_requested_bytes_mismatch")

    explicit_total = 0
    unresolved_minimum = 0
    for item in allocations:
        block = item.allocated_block_bytes
        if block is None:
            unresolved_minimum += item.rounded_minimum_bytes
        else:
            if (
                block < item.rounded_minimum_bytes
                or block % CUDA_ALLOCATOR_MIN_BLOCK_BYTES != 0
            ):
                errors.append(
                    "invalid_allocated_block_size:"
                    f"allocation_id={item.allocation_id}"
                )
            explicit_total += block

    aggregate = counters.allocated_block_bytes
    if aggregate is None:
        errors.append("allocator_counter_allocated_block_bytes_missing")
        if unresolved_minimum:
            errors.append("allocated_block_sizes_unresolved")
        return allocations
    minimum_total = explicit_total + unresolved_minimum
    if aggregate < minimum_total:
        errors.append(
            "allocator_counter_allocated_block_bytes_below_minimum"
        )
    elif unresolved_minimum and aggregate == minimum_total:
        allocations = [
            replace(
                item,
                allocated_block_bytes=item.rounded_minimum_bytes,
                allocated_block_size_proven=True,
            )
            if item.allocated_block_bytes is None
            else item
            for item in allocations
        ]
    elif unresolved_minimum:
        errors.append("allocated_block_sizes_unresolved")
    elif aggregate != explicit_total:
        errors.append("allocator_counter_allocated_block_bytes_mismatch")
    return allocations


def _validate_completion_counters(
    allocations: Sequence[AllocationLifetime],
    counters: AllocatorCounterEvidence,
    *,
    segment_alloc_count: int,
    segment_free_count: int,
    errors: list[str],
) -> None:
    freed = [item for item in allocations if item.fully_freed]
    if counters.free_count is None:
        errors.append("allocator_counter_free_count_missing")
    elif counters.free_count != len(freed):
        errors.append("allocator_counter_free_count_mismatch")
    freed_requested = sum(item.requested_bytes for item in freed)
    if counters.freed_requested_bytes is None:
        errors.append("allocator_counter_freed_requested_bytes_missing")
    elif counters.freed_requested_bytes != freed_requested:
        errors.append("allocator_counter_freed_requested_bytes_mismatch")
    freed_blocks = [item.allocated_block_bytes for item in freed]
    if counters.freed_block_bytes is None:
        errors.append("allocator_counter_freed_block_bytes_missing")
    elif all(value is not None for value in freed_blocks) and (
        counters.freed_block_bytes
        != sum(value for value in freed_blocks if value is not None)
    ):
        errors.append("allocator_counter_freed_block_bytes_mismatch")
    if counters.segment_allocation_count is None:
        errors.append("allocator_counter_segment_allocation_count_missing")
    elif counters.segment_allocation_count != segment_alloc_count:
        errors.append("allocator_counter_segment_allocation_count_mismatch")
    if counters.segment_free_count is None:
        errors.append("allocator_counter_segment_free_count_missing")
    elif counters.segment_free_count != segment_free_count:
        errors.append("allocator_counter_segment_free_count_mismatch")


def attribute_allocator_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    geometry: AllocationGeometry,
    counters: AllocatorCounterEvidence,
    rules: AttributionRules | None = None,
    backend_identity: str | None = None,
    expected_trace_sha256: str | None = None,
    history_integrity: AllocatorHistoryIntegrityEvidence | None = None,
) -> AllocatorTraceAttribution:
    """Parse a trace with a reuse-safe chronological address state machine."""

    selected_rules = rules or AttributionRules()
    digest = allocator_trace_sha256(trace)
    errors: list[str] = []
    if expected_trace_sha256 is not None and digest != expected_trace_sha256:
        errors.append("allocator_trace_sha256_mismatch")
    if backend_identity == "":
        errors.append("backend_identity_empty")
    if (
        selected_rules.frozen_backend_identity is not None
        and backend_identity != selected_rules.frozen_backend_identity
    ):
        errors.append("backend_identity_mismatch")

    events: list[TraceEventEvidence] = []
    counts: Counter[str] = Counter()
    active: dict[int, _OpenAllocation] = {}
    completed: list[AllocationLifetime] = []
    active_trace_segments: dict[int, tuple[int, int]] = {}
    unclaimed_segment_triggers: set[int] = set()
    next_allocation_id = 0

    for index, raw in enumerate(trace):
        if not isinstance(raw, Mapping):
            errors.append(f"malformed_trace_event:index={index}")
            event = TraceEventEvidence(
                index, "malformed", None, None, None, None, (), ()
            )
            events.append(event)
            counts[event.action] += 1
            continue
        raw_action = raw.get("action")
        action = raw_action if isinstance(raw_action, str) else "malformed"
        address = _optional_int(raw.get("addr", raw.get("address")))
        size = _optional_int(raw.get("size"))
        stream = _optional_int(raw.get("stream"))
        block = _optional_int(raw.get("allocated_block_size"))
        python_stack, cpp_stack = _event_stacks(raw)
        event = TraceEventEvidence(
            index,
            action,
            address,
            size,
            stream,
            block,
            python_stack,
            cpp_stack,
        )
        events.append(event)
        counts[action] += 1

        if action in {
            "alloc",
            "free_requested",
            "free_completed",
            "segment_alloc",
            "segment_free",
        } and (stream is None or stream < 0):
            errors.append(
                f"allocator_event_stream_missing_or_invalid:index={index}"
            )

        if action == "alloc":
            if (
                address is None
                or address < 0
                or size is None
                or size <= 0
            ):
                errors.append(f"invalid_alloc_event:index={index}")
                continue
            if address in active:
                errors.append(
                    f"alloc_before_prior_free_completed:index={index}"
                )
                continue
            if block is not None and block <= 0:
                errors.append(
                    f"invalid_allocated_block_size:index={index}"
                )
                block = None
            allocation_extent = block or cuda_allocator_rounded_minimum(size)
            if any(
                _ranges_overlap(
                    address,
                    allocation_extent,
                    active_address,
                    (
                        opened.explicit_block_bytes
                        or opened.rounded_minimum_bytes
                    ),
                )
                for active_address, opened in active.items()
            ):
                errors.append(
                    f"overlapping_live_allocation_range:index={index}"
                )
            segment_match = next(
                (
                    (start, segment_size, segment_index)
                    for start, (
                        segment_size,
                        segment_index,
                    ) in active_trace_segments.items()
                    if start <= address < start + segment_size
                ),
                None,
            )
            if segment_match is not None:
                segment_start, segment_size, segment_index = segment_match
                if address + allocation_extent > segment_start + segment_size:
                    errors.append(
                        "allocation_exceeds_trace_segment:"
                        f"index={index}"
                    )
                triggered_segment_alloc = (
                    segment_index in unclaimed_segment_triggers
                )
                unclaimed_segment_triggers.discard(segment_index)
            else:
                triggered_segment_alloc = False
            active[address] = _OpenAllocation(
                allocation_id=next_allocation_id,
                event=event,
                rounded_minimum_bytes=(
                    cuda_allocator_rounded_minimum(size)
                ),
                explicit_block_bytes=block,
                cache_reuse_status=(
                    CacheReuseStatus.UNVERIFIED
                    if segment_match is None
                    else CacheReuseStatus.NEW_TRACE_SEGMENT
                ),
                triggered_segment_alloc=triggered_segment_alloc,
            )
            next_allocation_id += 1
            continue

        if action in ("free_requested", "free_completed"):
            if address is None or address < 0:
                errors.append(f"invalid_free_event:index={index}")
                continue
            opened = active.get(address)
            if opened is None:
                errors.append(
                    f"free_without_matching_alloc:index={index}"
                )
                continue
            if size is None:
                errors.append(f"free_event_size_missing:index={index}")
            elif size != opened.event.size_bytes:
                errors.append(f"free_event_size_mismatch:index={index}")
            if action == "free_requested":
                if opened.free_requested_event_index is not None:
                    errors.append(
                        f"duplicate_free_requested:index={index}"
                    )
                else:
                    opened.free_requested_event_index = index
                continue
            if opened.free_requested_event_index is None:
                errors.append(
                    "free_completed_before_free_requested:"
                    f"index={index}"
                )
                continue
            completed.append(_make_lifetime(opened, index))
            del active[address]
            continue

        if action == "segment_alloc":
            if (
                address is None
                or address < 0
                or size is None
                or size <= 0
            ):
                errors.append(f"invalid_segment_alloc:index={index}")
            elif any(
                _ranges_overlap(address, size, start, segment_size)
                for start, (segment_size, _) in active_trace_segments.items()
            ):
                errors.append(f"overlapping_segment_alloc:index={index}")
            elif any(
                _ranges_overlap(
                    address,
                    size,
                    allocation_address,
                    (
                        opened.explicit_block_bytes
                        or opened.rounded_minimum_bytes
                    ),
                )
                for allocation_address, opened in active.items()
            ):
                errors.append(
                    f"segment_alloc_overlaps_live_allocation:index={index}"
                )
            else:
                active_trace_segments[address] = (size, index)
                unclaimed_segment_triggers.add(index)
            continue
        if action == "segment_free":
            if address is None or address < 0:
                errors.append(f"invalid_segment_free:index={index}")
            elif address not in active_trace_segments:
                errors.append(
                    f"segment_free_without_matching_alloc:index={index}"
                )
            else:
                segment_size, _ = active_trace_segments[address]
                if size is None:
                    errors.append(
                        f"segment_free_size_missing:index={index}"
                    )
                elif size != segment_size:
                    errors.append(
                        f"segment_free_size_mismatch:index={index}"
                    )
                elif any(
                    _ranges_overlap(
                        address,
                        segment_size,
                        allocation_address,
                        (
                            opened.explicit_block_bytes
                            or opened.rounded_minimum_bytes
                        ),
                    )
                    for allocation_address, opened in active.items()
                ):
                    errors.append(
                        "segment_free_with_live_allocation:"
                        f"index={index}"
                    )
                else:
                    _, segment_index = active_trace_segments[address]
                    del active_trace_segments[address]
                    unclaimed_segment_triggers.discard(segment_index)
            continue
        if action in ("oom", "alloc_retry"):
            errors.append(
                f"allocator_failure_event:{action}:index={index}"
            )
        elif action not in ("snapshot", "malformed"):
            errors.append(
                f"unknown_allocator_action:{action}:index={index}"
            )
        elif action == "malformed":
            errors.append(f"missing_allocator_action:index={index}")

    for opened in active.values():
        errors.append(
            "allocation_not_fully_freed:"
            f"allocation_id={opened.allocation_id}"
        )
        completed.append(_make_lifetime(opened, None))
    for start, (segment_size, segment_index) in active_trace_segments.items():
        errors.append(
            "trace_segment_not_freed:"
            f"start={start}:size={segment_size}:index={segment_index}"
        )
    completed.sort(key=lambda item: item.allocation_id)
    completed = _reconcile_block_sizes(completed, counters, errors)
    _validate_completion_counters(
        completed,
        counters,
        segment_alloc_count=int(counts.get("segment_alloc", 0)),
        segment_free_count=int(counts.get("segment_free", 0)),
        errors=errors,
    )
    segment_integrity_failed = any(
        "segment" in error for error in errors
    )
    cache_reuse_proven = (
        not segment_integrity_failed
        and int(counts.get("segment_alloc", 0)) == 0
        and int(counts.get("segment_free", 0)) == 0
        and counters.segment_allocation_count == 0
        and counters.segment_free_count == 0
        and counters.device_allocation_count == 0
        and counters.device_free_count == 0
    )
    if cache_reuse_proven:
        completed = [
            replace(
                item,
                cache_reuse_status=CacheReuseStatus.VERIFIED_REUSE,
            )
            if item.cache_reuse_status is CacheReuseStatus.UNVERIFIED
            else item
            for item in completed
        ]
    completed = [
        _classify(item, geometry, selected_rules) for item in completed
    ]
    return AllocatorTraceAttribution(
        trace_sha256=digest,
        expected_trace_sha256=expected_trace_sha256,
        backend_identity=backend_identity,
        geometry=geometry,
        rules=selected_rules,
        counters=counters,
        events=tuple(events),
        allocations=tuple(completed),
        action_counts=dict(counts),
        segment_alloc_count=int(counts.get("segment_alloc", 0)),
        segment_free_count=int(counts.get("segment_free", 0)),
        integrity_errors=tuple(dict.fromkeys(errors)),
        history_integrity=history_integrity,
    )


def _append_eager_memory_failures(
    reasons: list[str],
    memory: MemoryDeltaEvidence,
) -> None:
    if memory.allocated_delta > 0:
        reasons.append("persistent_allocated_growth")
    if memory.reserved_delta > 0:
        reasons.append("persistent_reserved_growth")
    if memory.device_used_delta is None:
        reasons.append("device_used_delta_unavailable")
    elif memory.device_used_delta > 0:
        reasons.append("persistent_device_used_growth")
    if memory.non_pytorch_delta is None:
        reasons.append("non_pytorch_delta_unavailable")
    elif memory.non_pytorch_delta != 0:
        reasons.append("persistent_non_pytorch_delta_nonzero")


def _append_graph_memory_failures(
    reasons: list[str],
    memory: MemoryDeltaEvidence,
) -> None:
    if memory.allocated_delta != 0:
        reasons.append("graph_allocated_delta_nonzero")
    if memory.reserved_delta != 0:
        reasons.append("graph_reserved_delta_nonzero")
    if memory.device_used_delta is None:
        reasons.append("device_used_delta_unavailable")
    elif memory.device_used_delta != 0:
        reasons.append("graph_device_used_delta_nonzero")
    if memory.non_pytorch_delta is None:
        reasons.append("non_pytorch_delta_unavailable")
    elif memory.non_pytorch_delta != 0:
        reasons.append("graph_non_pytorch_delta_nonzero")


def _append_common_failures(
    reasons: list[str],
    attribution: AllocatorTraceAttribution,
) -> None:
    if attribution.integrity_errors:
        reasons.append("allocator_trace_integrity_failure")
    history = attribution.history_integrity
    if history is None:
        reasons.append("allocator_history_integrity_missing")
    else:
        reasons.extend(history.failure_reasons())
        if history.raw_trace_sha256 != attribution.trace_sha256:
            reasons.append("attributed_trace_sha256_mismatch")
    if any(
        not item.python_stack or not item.cpp_stack
        for item in attribution.allocations
    ):
        reasons.append("allocator_allocation_stack_incomplete")
    if not attribution.counters.complete:
        reasons.append("allocator_counter_evidence_incomplete")
    if not attribution.all_block_sizes_proven:
        reasons.append("allocated_block_size_evidence_incomplete")
    if not attribution.all_lifetimes_fully_freed:
        reasons.append("allocation_lifetime_not_fully_freed")
    if attribution.segment_alloc_count:
        reasons.append("segment_alloc_detected")
    if attribution.segment_free_count:
        reasons.append("segment_free_detected")
    if attribution.counters.device_allocation_count != 0:
        reasons.append("device_allocation_detected_or_unavailable")
    if attribution.counters.device_free_count != 0:
        reasons.append("device_free_detected_or_unavailable")
    if attribution.counters.allocation_retry_count != 0:
        reasons.append("allocator_retry_detected_or_unavailable")
    if attribution.counters.oom_count != 0:
        reasons.append("allocator_oom_detected_or_unavailable")


def _append_policy_bound_failures(
    reasons: list[str],
    attribution: AllocatorTraceAttribution,
) -> None:
    policies = attribution.rules.permitted_allocation_policies
    allocations = attribution.allocations
    for policy in policies:
        observed = [
            item for item in allocations if item.policy_id == policy.policy_id
        ]
        if len(observed) != policy.exact_count:
            reasons.append(
                "allocation_policy_count_bound_failed:"
                f"{policy.policy_id}"
            )
        if (
            sum(item.requested_bytes for item in observed)
            != policy.exact_total_requested_bytes
        ):
            reasons.append(
                "allocation_policy_byte_bound_failed:"
                f"{policy.policy_id}"
            )
        geometry_bytes = _policy_geometry_bytes(
            policy.formula_id, attribution.geometry
        )
        geometry_formula = policy.formula_id.endswith(
            "_geometry_bytes_v1"
        )
        if geometry_formula and (
            geometry_bytes is None
            or policy.allowed_requested_bytes
            != frozenset({geometry_bytes})
            or policy.exact_total_requested_bytes
            != geometry_bytes * policy.exact_count
        ):
            reasons.append(
                "allocation_policy_formula_geometry_mismatch:"
                f"{policy.policy_id}"
            )

    for event_class in _EAGER_PERMITTED_CLASSES:
        class_policies = [
            policy for policy in policies if policy.event_class is event_class
        ]
        observed = [
            item for item in allocations if item.event_class is event_class
        ]
        if len(observed) != sum(
            policy.exact_count for policy in class_policies
        ):
            reasons.append(
                "allocation_class_count_bound_failed:"
                f"{event_class.value}"
            )
        if sum(item.requested_bytes for item in observed) != sum(
            policy.exact_total_requested_bytes for policy in class_policies
        ):
            reasons.append(
                "allocation_class_byte_bound_failed:"
                f"{event_class.value}"
            )


def _evaluate_eager_criterion(
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
    *,
    production_binding: ProductionAllocationBinding | None,
    require_production_authority: bool,
) -> AllocationCriterionResult:
    """Evaluate the attributed-bounded-ephemeral eager criterion.

    Split-K workspaces fail closed until a verifier rederives the controls from
    checksum-bound raw dispatch and allocator artifacts.  No caller-supplied
    booleans, kernel names, counts, or digests can make this criterion pass.
    """

    reasons: list[str] = []
    if require_production_authority:
        if production_binding is None:
            reasons.append("production_allocation_binding_missing")
        else:
            reasons.extend(production_binding.validation_errors())
            if production_binding.execution_mode != "eager":
                reasons.append("production_binding_execution_mode_mismatch")
            if production_binding.geometry != attribution.geometry:
                reasons.append("production_binding_geometry_mismatch")
            if production_binding.backend_identity != attribution.backend_identity:
                reasons.append("production_binding_backend_identity_mismatch")
            try:
                expected_rules = instantiate_decision_0009_production_rules(
                    production_binding
                )
            except AllocationAttributionError:
                reasons.append("production_policy_instantiation_failed")
            else:
                if attribution.rules != expected_rules:
                    reasons.append("production_policy_rules_mismatch")
        if attribution.rules.frozen_backend_identity is None:
            reasons.append("frozen_backend_identity_missing")
    _append_common_failures(reasons, attribution)
    _append_eager_memory_failures(reasons, memory)
    if not attribution.all_allocations_cache_reused:
        reasons.append("allocation_not_reused_from_cache")
    _append_policy_bound_failures(reasons, attribution)

    forbidden = {
        AllocationClass.CACHE_GROWTH,
        AllocationClass.GQA_EXPANSION,
        AllocationClass.AUDIT_INSTRUMENTATION,
        AllocationClass.UNKNOWN,
    }
    for event_class in forbidden:
        if any(
            item.event_class is event_class
            for item in attribution.allocations
        ):
            reasons.append(
                f"forbidden_allocation_class:{event_class.value}"
            )

    workspaces = [
        item
        for item in attribution.allocations
        if item.event_class is AllocationClass.CONTEXT_SCALED_WORKSPACE
    ]
    if workspaces:
        workspace_splits = tuple(
            dict(item.formula_parameters).get("num_splits")
            for item in workspaces
        )
        observed_splits = frozenset(
            split
            for split in workspace_splits
            if isinstance(split, int) and not isinstance(split, bool)
        )
        formula_split_counts = Counter(
            (
                item.size_formula,
                dict(item.formula_parameters).get("num_splits"),
            )
            for item in workspaces
        )
        expected_pair_count = attribution.rules.split_k_expected_pair_count
        control_splits: frozenset[int] = frozenset()
        if require_production_authority and production_binding is not None:
            raw_inputs = production_binding.split_k_raw_inputs
            if raw_inputs is None or not raw_inputs.raw_bytes_verified:
                reasons.append("flash_split_k_paired_control_unverified")
            else:
                control_splits = frozenset(
                    splits
                    for splits, pair_count in (
                        raw_inputs.split_k_pair_multiplicity
                    )
                    if pair_count == 1
                )
        else:
            reasons.append("flash_split_k_independent_verifier_unavailable")
        if expected_pair_count is None:
            reasons.append("flash_split_k_pair_multiplicity_unregistered")
        if (
            len(observed_splits) != 1
            or any(split is None for split in workspace_splits)
            or expected_pair_count is None
            or (
                require_production_authority
                and observed_splits != control_splits
            )
            or any(
                formula_split_counts[
                    ("flash_split_k_output_accumulator", split)
                ]
                != expected_pair_count
                or formula_split_counts[("flash_split_k_lse", split)]
                != expected_pair_count
                for split in observed_splits
            )
        ):
            reasons.append("flash_split_k_workspace_pair_mismatch")
        if any(
            not _has_all_markers(
                _cpp_stack_text(item),
                attribution.rules.flash_split_k_cpp_markers,
            )
            for item in workspaces
        ):
            reasons.append("flash_split_k_stack_attribution_failed")
    elif require_production_authority:
        reasons.append("flash_split_k_workspace_pair_missing")

    no_context_dependent = all(
        item.dependencies.context is False
        for item in attribution.allocations
    )
    fully_attributed = not any(
        item.event_class in forbidden for item in attribution.allocations
    )
    unique = tuple(dict.fromkeys(reasons))
    return AllocationCriterionResult(
        criterion_id=(
            "phase3_eager_attributed_ephemeral_v1"
            if require_production_authority
            else "phase3_eager_structural_test_only_v1"
        ),
        passed=not unique,
        failure_reasons=unique,
        allocation_event_count=len(attribution.allocations),
        class_counts=attribution.class_counts(),
        no_context_dependent_allocation=no_context_dependent,
        fully_attributed_bounded_ephemeral=(
            not unique and fully_attributed
        ),
        strict_graph_zero_events=None,
    )


def evaluate_refined_eager_criterion(
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
    *,
    production_binding: ProductionAllocationBinding | None = None,
) -> AllocationCriterionResult:
    """Evaluate production eager evidence against Decisions 0009 and 0013.

    Arbitrary caller policies are never production authority. The checked-in
    catalog, paired raw controls, exact stacks, formulas, and multiplicities
    are all required.
    """

    return _evaluate_eager_criterion(
        attribution,
        memory,
        production_binding=production_binding,
        require_production_authority=True,
    )


def evaluate_structural_eager_criterion_for_test(
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
) -> AllocationCriterionResult:
    """Exercise parser/policy mechanics without granting production status."""

    return _evaluate_eager_criterion(
        attribution,
        memory,
        production_binding=None,
        require_production_authority=False,
    )


def evaluate_strict_graph_criterion(
    attribution: AllocatorTraceAttribution,
    memory: MemoryDeltaEvidence,
) -> AllocationCriterionResult:
    """Require exactly zero graph-replay allocation and device events."""

    reasons: list[str] = []
    _append_common_failures(reasons, attribution)
    _append_graph_memory_failures(reasons, memory)
    if attribution.action_counts.get("alloc", 0) != 0 or (
        attribution.allocations
    ):
        reasons.append("graph_allocation_event_detected")
    counters = attribution.counters
    if any(
        value != 0
        for value in (
            counters.allocation_count,
            counters.requested_bytes,
            counters.allocated_block_bytes,
            counters.free_count,
            counters.freed_requested_bytes,
            counters.freed_block_bytes,
            counters.segment_allocation_count,
            counters.segment_free_count,
            counters.device_allocation_count,
            counters.device_free_count,
            counters.allocation_retry_count,
            counters.oom_count,
        )
    ):
        reasons.append(
            "graph_allocator_counters_nonzero_or_unavailable"
        )
    unique = tuple(dict.fromkeys(reasons))
    strict_zero = not unique
    return AllocationCriterionResult(
        criterion_id="phase3_graph_zero_allocation_v1",
        passed=strict_zero,
        failure_reasons=unique,
        allocation_event_count=len(attribution.allocations),
        class_counts=attribution.class_counts(),
        no_context_dependent_allocation=strict_zero,
        fully_attributed_bounded_ephemeral=False,
        strict_graph_zero_events=strict_zero,
    )


class AllocationExecutionMode(StrEnum):
    """Allocation criteria have deliberately distinct execution modes."""

    EAGER = "eager"
    CUDA_GRAPH = "cuda_graph"


@dataclass(frozen=True, slots=True)
class RawAllocatorEvidenceFiles:
    """Fixed append-only filenames and hashes inside caller-owned staging."""

    snapshot_file: str
    snapshot_sha256: str
    trace_file: str
    trace_sha256: str
    memory_stats_before_file: str
    memory_stats_before_sha256: str
    memory_stats_after_file: str
    memory_stats_after_sha256: str
    memory_accounting_before_file: str
    memory_accounting_before_sha256: str
    memory_accounting_after_file: str
    memory_accounting_after_sha256: str
    operation_witness_file: str
    operation_witness_sha256: str
    audit_file: str
    audit_sha256_file: str
    audit_sha256: str | None

    def __post_init__(self) -> None:
        filenames = (
            self.snapshot_file,
            self.trace_file,
            self.memory_stats_before_file,
            self.memory_stats_after_file,
            self.memory_accounting_before_file,
            self.memory_accounting_after_file,
            self.operation_witness_file,
            self.audit_file,
            self.audit_sha256_file,
        )
        for name in filenames:
            if not name or Path(name).name != name:
                raise AllocationAttributionError(
                    "allocator evidence filename must be a basename"
                )
        if len(set(filenames)) != len(filenames):
            raise AllocationAttributionError(
                "allocator evidence filenames must be unique"
            )
        for name in (
            "snapshot_sha256",
            "trace_sha256",
            "memory_stats_before_sha256",
            "memory_stats_after_sha256",
            "memory_accounting_before_sha256",
            "memory_accounting_after_sha256",
            "operation_witness_sha256",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise AllocationAttributionError(
                    f"{name} must be a lowercase SHA-256 digest"
                )
        if self.audit_sha256 is not None and not _valid_sha256(
            self.audit_sha256
        ):
            raise AllocationAttributionError(
                "audit_sha256 must be a lowercase SHA-256 digest or None"
            )

    def to_dict(self, *, include_audit_sha256: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "snapshot_file": self.snapshot_file,
            "snapshot_sha256": self.snapshot_sha256,
            "trace_file": self.trace_file,
            "trace_sha256": self.trace_sha256,
            "memory_stats_before_file": self.memory_stats_before_file,
            "memory_stats_before_sha256": self.memory_stats_before_sha256,
            "memory_stats_after_file": self.memory_stats_after_file,
            "memory_stats_after_sha256": self.memory_stats_after_sha256,
            "memory_accounting_before_file": (
                self.memory_accounting_before_file
            ),
            "memory_accounting_before_sha256": (
                self.memory_accounting_before_sha256
            ),
            "memory_accounting_after_file": (
                self.memory_accounting_after_file
            ),
            "memory_accounting_after_sha256": (
                self.memory_accounting_after_sha256
            ),
            "operation_witness_file": self.operation_witness_file,
            "operation_witness_sha256": (
                self.operation_witness_sha256
            ),
            "audit_file": self.audit_file,
            "audit_sha256_file": self.audit_sha256_file,
        }
        if include_audit_sha256:
            value["audit_sha256"] = self.audit_sha256
        return value


@dataclass(frozen=True, slots=True)
class PartialAllocatorEvidenceFile:
    evidence_key: str
    file: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.evidence_key
            or not self.evidence_key.replace("_", "").isalnum()
            or Path(self.file).name != self.file
            or not _valid_sha256(self.sha256)
        ):
            raise AllocationAttributionError(
                "partial allocator evidence metadata is invalid"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "evidence_key": self.evidence_key,
            "file": self.file,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FailedAllocatorAuditFiles:
    """Checksum-bound evidence retained when collection cannot finish."""

    audit_file: str
    audit_sha256_file: str
    audit_sha256: str
    partial_files: tuple[PartialAllocatorEvidenceFile, ...] = ()

    def __post_init__(self) -> None:
        for name in (self.audit_file, self.audit_sha256_file):
            if not name or Path(name).name != name:
                raise AllocationAttributionError(
                    "allocator evidence filename must be a basename"
                )
        if not _valid_sha256(self.audit_sha256):
            raise AllocationAttributionError(
                "audit_sha256 must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "audit_file": self.audit_file,
            "audit_sha256_file": self.audit_sha256_file,
            "audit_sha256": self.audit_sha256,
            "partial_files": [
                item.to_dict() for item in self.partial_files
            ],
        }


@dataclass(frozen=True, slots=True)
class CollectedAllocationAudit:
    """Production evidence for one warmed, isolated, instrumented operation."""

    execution_mode: AllocationExecutionMode
    device: str
    device_index: int
    gpu_uuid: str
    torch_version: str
    cuda_runtime_version: str | None
    backend_identity: str
    warmup_iterations: int
    prepare_present: bool
    prepare_attempted: bool
    prepare_completed: bool
    prepare_attempt_count: int
    prepare_completion_count: int
    max_history_entries: int
    recorder_configuration: dict[str, Any]
    collection_started_ns: int
    collection_finished_ns: int
    operation_error: str | None
    production_binding: ProductionAllocationBinding
    operation_witness: OperationWitnessEvidence
    memory: MemoryDeltaEvidence
    attribution: AllocatorTraceAttribution
    criterion: AllocationCriterionResult
    raw_files: RawAllocatorEvidenceFiles

    def to_dict(
        self, *, include_audit_sha256: bool = True
    ) -> dict[str, Any]:
        return {
            "schema_version": PHASE3_ALLOCATION_AUDIT_SCHEMA_VERSION,
            "run_kind": "allocation_audit",
            "evidence_status": "complete",
            "execution_mode": self.execution_mode.value,
            "device": self.device,
            "device_index": self.device_index,
            "gpu_uuid": self.gpu_uuid,
            "torch_version": self.torch_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "backend_identity": self.backend_identity,
            "warmup_iterations": self.warmup_iterations,
            "prepare_present": self.prepare_present,
            "prepare_attempted": self.prepare_attempted,
            "prepare_completed": self.prepare_completed,
            "prepare_attempt_count": self.prepare_attempt_count,
            "prepare_completion_count": self.prepare_completion_count,
            "max_history_entries": self.max_history_entries,
            "recorder_configuration": dict(self.recorder_configuration),
            "collection_started_ns": self.collection_started_ns,
            "collection_finished_ns": self.collection_finished_ns,
            "operation_error": self.operation_error,
            "production_binding": self.production_binding.to_dict(),
            "production_binding_sha256": (
                self.production_binding.identity_sha256
            ),
            "operation_key": self.production_binding.operation_key.to_dict(),
            "external_provenance_status": (
                self.production_binding.external_provenance_status
            ),
            "operation_witness": self.operation_witness.to_dict(),
            "memory": self.memory.to_dict(),
            "attribution": self.attribution.to_dict(),
            "criterion": self.criterion.to_dict(),
            "raw_files": self.raw_files.to_dict(
                include_audit_sha256=include_audit_sha256
            ),
            "profiler_timing_reported": False,
            "instrumented_duration_reported_as_timing": False,
            "normal_benchmark_timing_eligible": False,
        }


_STAT_KEYS = {
    "allocation_count": "allocation.all.allocated",
    "requested_bytes": "requested_bytes.all.allocated",
    "allocated_block_bytes": "allocated_bytes.all.allocated",
    "free_count": "allocation.all.freed",
    "freed_requested_bytes": "requested_bytes.all.freed",
    "freed_block_bytes": "allocated_bytes.all.freed",
    "segment_allocation_count": "segment.all.allocated",
    "segment_free_count": "segment.all.freed",
    "device_allocation_count": "num_device_alloc",
    "device_free_count": "num_device_free",
    "allocation_retry_count": "num_alloc_retries",
    "oom_count": "num_ooms",
}


def _counter_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    key: str,
) -> int | None:
    left = before.get(key)
    right = after.get(key)
    if (
        isinstance(left, bool)
        or not isinstance(left, int)
        or isinstance(right, bool)
        or not isinstance(right, int)
        or right < left
    ):
        return None
    return right - left


def allocator_counters_from_memory_stats(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> AllocatorCounterEvidence:
    """Derive complete cumulative deltas; missing/decreasing keys stay null."""

    values = {
        name: _counter_delta(before, after, key)
        for name, key in _STAT_KEYS.items()
    }
    return AllocatorCounterEvidence(
        allocation_count=values["allocation_count"],
        requested_bytes=values["requested_bytes"],
        allocated_block_bytes=values["allocated_block_bytes"],
        device_allocation_count=values["device_allocation_count"],
        device_free_count=values["device_free_count"],
        free_count=values["free_count"],
        freed_requested_bytes=values["freed_requested_bytes"],
        freed_block_bytes=values["freed_block_bytes"],
        segment_allocation_count=values["segment_allocation_count"],
        segment_free_count=values["segment_free_count"],
        allocation_retry_count=values["allocation_retry_count"],
        oom_count=values["oom_count"],
    )


def allocator_trace_from_snapshot(
    snapshot: Mapping[str, Any],
    device_index: int,
) -> list[Mapping[str, Any]]:
    """Select and validate one device's raw chronological trace."""

    if isinstance(device_index, bool) or device_index < 0:
        raise AllocationAttributionError(
            "device_index must be a nonnegative integer"
        )
    traces = snapshot.get("device_traces")
    if not isinstance(traces, list) or device_index >= len(traces):
        raise AllocationAttributionError(
            "allocator snapshot lacks selected device trace"
        )
    selected = traces[device_index]
    if not isinstance(selected, list):
        raise AllocationAttributionError(
            "allocator snapshot device trace is malformed"
        )
    if not all(isinstance(event, Mapping) for event in selected):
        raise AllocationAttributionError(
            "allocator snapshot contains a malformed trace event"
        )
    return list(selected)


def build_history_integrity_evidence(
    snapshot: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    *,
    max_entries: int,
    stack_mode: str = "all",
    expected_snapshot_sha256: str | None = None,
    expected_trace_sha256: str | None = None,
) -> AllocatorHistoryIntegrityEvidence:
    """Build ring/stack/hash evidence without trusting serialized verdicts."""

    _positive_int("max_entries", max_entries)
    snapshot_digest = allocator_snapshot_sha256(snapshot)
    trace_digest = allocator_trace_sha256(trace)
    return AllocatorHistoryIntegrityEvidence(
        stack_mode=stack_mode,
        ring_capacity=max_entries,
        observed_trace_entries=len(trace),
        raw_snapshot_sha256=snapshot_digest,
        expected_raw_snapshot_sha256=(
            snapshot_digest
            if expected_snapshot_sha256 is None
            else expected_snapshot_sha256
        ),
        raw_trace_sha256=trace_digest,
        expected_raw_trace_sha256=(
            trace_digest
            if expected_trace_sha256 is None
            else expected_trace_sha256
        ),
    )


def _staging_directory_fd(staging_directory: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open(os.fspath(staging_directory), flags)
    except OSError as error:
        raise AllocationAttributionError(
            "allocator evidence staging path must be an existing real directory"
        ) from error


def _preflight_names(directory_fd: int, names: Sequence[str]) -> None:
    for name in names:
        if not name or Path(name).name != name:
            raise AllocationAttributionError(
                "allocator evidence filename must be a basename"
            )
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise FileExistsError(
            f"allocator evidence destination already exists: {name}"
        )


def _exclusive_write(directory_fd: int, name: str, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
    )
    descriptor = os.open(name, flags, 0o640, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short allocator evidence write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def preserve_allocator_evidence(
    staging_directory: Path,
    *,
    snapshot: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    memory_stats_before: Mapping[str, Any],
    memory_stats_after: Mapping[str, Any],
    memory_accounting_before: Mapping[str, Any],
    memory_accounting_after: Mapping[str, Any],
    operation_witness: Mapping[str, Any],
    expected_snapshot_sha256: str,
    expected_trace_sha256: str,
    audit_payload: Mapping[str, Any],
) -> RawAllocatorEvidenceFiles:
    """Write raw evidence without replacement through a pinned directory fd.

    Files already written are deliberately retained if a later write fails.
    The caller's artifact finalizer owns promotion and COMPLETE publication.
    """

    snapshot_payload = _canonical_json_bytes(snapshot)
    trace_payload = _canonical_json_bytes(list(trace))
    stats_before_payload = _canonical_json_bytes(memory_stats_before)
    stats_after_payload = _canonical_json_bytes(memory_stats_after)
    accounting_before_payload = _canonical_json_bytes(
        memory_accounting_before
    )
    accounting_after_payload = _canonical_json_bytes(memory_accounting_after)
    operation_witness_payload = _canonical_json_bytes(operation_witness)
    snapshot_digest = hashlib.sha256(snapshot_payload).hexdigest()
    trace_digest = hashlib.sha256(trace_payload).hexdigest()
    if snapshot_digest != expected_snapshot_sha256:
        raise AllocationAttributionError(
            "raw allocator snapshot changed before preservation"
        )
    if trace_digest != expected_trace_sha256:
        raise AllocationAttributionError(
            "raw allocator trace changed before preservation"
        )
    files = RawAllocatorEvidenceFiles(
        snapshot_file="allocator_snapshot.json",
        snapshot_sha256=snapshot_digest,
        trace_file="allocator_trace.json",
        trace_sha256=trace_digest,
        memory_stats_before_file="allocator_memory_stats_before.json",
        memory_stats_before_sha256=hashlib.sha256(
            stats_before_payload
        ).hexdigest(),
        memory_stats_after_file="allocator_memory_stats_after.json",
        memory_stats_after_sha256=hashlib.sha256(
            stats_after_payload
        ).hexdigest(),
        memory_accounting_before_file=(
            "allocator_memory_accounting_before.json"
        ),
        memory_accounting_before_sha256=hashlib.sha256(
            accounting_before_payload
        ).hexdigest(),
        memory_accounting_after_file=(
            "allocator_memory_accounting_after.json"
        ),
        memory_accounting_after_sha256=hashlib.sha256(
            accounting_after_payload
        ).hexdigest(),
        operation_witness_file="allocation_operation_witness.json",
        operation_witness_sha256=hashlib.sha256(
            operation_witness_payload
        ).hexdigest(),
        audit_file="allocation_audit.json",
        audit_sha256_file="allocation_audit.sha256",
        audit_sha256=None,
    )
    authoritative_raw_files = files.to_dict(include_audit_sha256=False)
    existing_raw_files = audit_payload.get("raw_files")
    if (
        existing_raw_files is not None
        and existing_raw_files != authoritative_raw_files
    ):
        raise AllocationAttributionError(
            "allocator audit raw-file references do not match raw evidence"
        )
    existing_operation_witness = audit_payload.get("operation_witness")
    if (
        existing_operation_witness is not None
        and existing_operation_witness != operation_witness
    ):
        raise AllocationAttributionError(
            "allocator audit operation witness differs from raw evidence"
        )
    bound_audit_payload = dict(audit_payload)
    bound_audit_payload["raw_files"] = authoritative_raw_files
    bound_audit_payload["operation_witness"] = dict(operation_witness)
    audit_bytes = _canonical_json_bytes(bound_audit_payload)
    audit_digest = hashlib.sha256(audit_bytes).hexdigest()
    files = replace(files, audit_sha256=audit_digest)
    audit_digest_bytes = (
        f"{audit_digest}  {files.audit_file}\n".encode("ascii")
    )
    directory_fd = _staging_directory_fd(staging_directory)
    try:
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
        _preflight_names(directory_fd, names)
        _exclusive_write(directory_fd, files.snapshot_file, snapshot_payload)
        _exclusive_write(directory_fd, files.trace_file, trace_payload)
        _exclusive_write(
            directory_fd,
            files.memory_stats_before_file,
            stats_before_payload,
        )
        _exclusive_write(
            directory_fd,
            files.memory_stats_after_file,
            stats_after_payload,
        )
        _exclusive_write(
            directory_fd,
            files.memory_accounting_before_file,
            accounting_before_payload,
        )
        _exclusive_write(
            directory_fd,
            files.memory_accounting_after_file,
            accounting_after_payload,
        )
        _exclusive_write(
            directory_fd,
            files.operation_witness_file,
            operation_witness_payload,
        )
        _exclusive_write(directory_fd, files.audit_file, audit_bytes)
        _exclusive_write(
            directory_fd,
            files.audit_sha256_file,
            audit_digest_bytes,
        )
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return files


def _read_no_follow(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AllocationAttributionError(
                "allocator evidence must be a regular single-link file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _read_verified_allocator_evidence(
    staging_directory: Path,
    files: RawAllocatorEvidenceFiles,
) -> dict[str, bytes]:
    """Read each required file exactly once and verify those exact bytes."""

    if files.audit_sha256 is None:
        raise AllocationAttributionError(
            "allocator audit digest is unavailable"
        )
    directory_fd = _staging_directory_fd(staging_directory)
    try:
        payloads = {
            "snapshot": _read_no_follow(directory_fd, files.snapshot_file),
            "trace": _read_no_follow(directory_fd, files.trace_file),
            "stats_before": _read_no_follow(
                directory_fd, files.memory_stats_before_file
            ),
            "stats_after": _read_no_follow(
                directory_fd, files.memory_stats_after_file
            ),
            "accounting_before": _read_no_follow(
                directory_fd, files.memory_accounting_before_file
            ),
            "accounting_after": _read_no_follow(
                directory_fd, files.memory_accounting_after_file
            ),
            "operation_witness": _read_no_follow(
                directory_fd, files.operation_witness_file
            ),
            "audit": _read_no_follow(directory_fd, files.audit_file),
            "audit_digest": _read_no_follow(
                directory_fd, files.audit_sha256_file
            ),
        }
    finally:
        os.close(directory_fd)
    expected = {
        "snapshot": files.snapshot_sha256,
        "trace": files.trace_sha256,
        "stats_before": files.memory_stats_before_sha256,
        "stats_after": files.memory_stats_after_sha256,
        "accounting_before": files.memory_accounting_before_sha256,
        "accounting_after": files.memory_accounting_after_sha256,
        "operation_witness": files.operation_witness_sha256,
        "audit": files.audit_sha256,
    }
    for key, digest in expected.items():
        if hashlib.sha256(payloads[key]).hexdigest() != digest:
            raise AllocationAttributionError(
                f"allocator evidence digest mismatch: {key}"
            )
    expected_ledger = (
        f"{files.audit_sha256}  {files.audit_file}\n".encode("ascii")
    )
    if payloads["audit_digest"] != expected_ledger:
        raise AllocationAttributionError(
            "allocator audit digest ledger mismatch"
        )
    try:
        audit_payload = json.loads(payloads["audit"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AllocationAttributionError(
            "allocator audit is not valid JSON"
        ) from error
    if not isinstance(audit_payload, Mapping) or audit_payload.get(
        "raw_files"
    ) != files.to_dict(include_audit_sha256=False):
        raise AllocationAttributionError(
            "allocator audit raw-file index mismatch"
        )
    return payloads


def verify_preserved_allocator_evidence(
    staging_directory: Path,
    files: RawAllocatorEvidenceFiles,
) -> bool:
    """Rehash all preserved evidence through no-follow opens."""

    try:
        _read_verified_allocator_evidence(staging_directory, files)
    except (OSError, AllocationAttributionError):
        return False
    return True


def read_verified_allocator_evidence(
    staging_directory: Path,
    files: RawAllocatorEvidenceFiles,
) -> dict[str, bytes]:
    """Return the exact no-follow, checksum-verified raw evidence bytes."""

    return _read_verified_allocator_evidence(staging_directory, files)


@dataclass(frozen=True, slots=True)
class SemanticAllocatorEvidenceValidation:
    passed: bool
    failure_reasons: tuple[str, ...]
    memory: MemoryDeltaEvidence | None
    attribution: AllocatorTraceAttribution | None
    criterion: AllocationCriterionResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "memory": None if self.memory is None else self.memory.to_dict(),
            "attribution": (
                None
                if self.attribution is None
                else self.attribution.to_dict()
            ),
            "criterion": (
                None
                if self.criterion is None
                else self.criterion.to_dict()
            ),
        }


def _parse_canonical_json_bytes(payload: bytes) -> object:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AllocationAttributionError(
            "allocator raw evidence is not valid JSON"
        ) from error
    if _canonical_json_bytes(parsed) != payload:
        raise AllocationAttributionError(
            "allocator raw evidence is not canonical JSON"
        )
    return parsed


def _raw_memory_sample_from_mapping(
    value: Mapping[str, Any],
) -> RawMemoryAccountingSample:
    _require_exact_mapping_keys(
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
        "raw memory accounting sample",
    )
    sample = RawMemoryAccountingSample(
        schema_version=value.get("schema_version"),
        operation_fingerprint_sha256=value.get(
            "operation_fingerprint_sha256"
        ),
        sample_role=value.get("sample_role"),
        timestamp_ns=value.get("timestamp_ns"),
        device=value.get("device"),
        device_index=value.get("device_index"),
        gpu_uuid=value.get("gpu_uuid"),
        allocated_bytes=value.get("allocated_bytes"),
        reserved_bytes=value.get("reserved_bytes"),
        device_free_bytes=value.get("device_free_bytes"),
        device_total_bytes=value.get("device_total_bytes"),
    )
    if value.get("device_used_bytes") != sample.device_used_bytes:
        raise AllocationAttributionError(
            "serialized device-used bytes do not match free/total samples"
        )
    return sample


def raw_memory_accounting_sample_from_mapping(
    value: Mapping[str, Any],
) -> RawMemoryAccountingSample:
    """Parse one canonical raw memory sample for an external semantic replay."""

    return _raw_memory_sample_from_mapping(value)


def validate_preserved_allocator_evidence_semantically(
    staging_directory: Path,
    files: RawAllocatorEvidenceFiles,
    *,
    production_binding: ProductionAllocationBinding,
) -> SemanticAllocatorEvidenceValidation:
    """Replay the complete gate from canonical raw evidence.

    The serialized criterion ``passed`` field is deliberately ignored.
    """

    reasons: list[str] = list(production_binding.validation_errors())
    try:
        verified_payloads = _read_verified_allocator_evidence(
            staging_directory, files
        )
    except (OSError, AllocationAttributionError) as error:
        reasons.append("preserved_allocator_hash_verification_failed")
        reasons.append(
            f"preserved_allocator_read_failed:{type(error).__name__}:{error}"
        )
        return SemanticAllocatorEvidenceValidation(
            False, tuple(dict.fromkeys(reasons)), None, None, None
        )
    raw_payloads = {
        key: verified_payloads[key]
        for key in (
            "snapshot",
            "trace",
            "stats_before",
            "stats_after",
            "accounting_before",
            "accounting_after",
            "operation_witness",
            "audit",
        )
    }
    try:
        parsed = {
            key: _parse_canonical_json_bytes(value)
            for key, value in raw_payloads.items()
        }
        snapshot = parsed["snapshot"]
        trace = parsed["trace"]
        stats_before = parsed["stats_before"]
        stats_after = parsed["stats_after"]
        accounting_before = parsed["accounting_before"]
        accounting_after = parsed["accounting_after"]
        raw_operation_witness = parsed["operation_witness"]
        audit = parsed["audit"]
        if not isinstance(snapshot, Mapping):
            raise AllocationAttributionError("snapshot must be an object")
        if not isinstance(trace, list) or not all(
            isinstance(item, Mapping) for item in trace
        ):
            raise AllocationAttributionError("trace must be an event list")
        if not isinstance(stats_before, Mapping) or not isinstance(
            stats_after, Mapping
        ):
            raise AllocationAttributionError("memory stats must be objects")
        if not isinstance(accounting_before, Mapping) or not isinstance(
            accounting_after, Mapping
        ):
            raise AllocationAttributionError(
                "memory accounting samples must be objects"
            )
        if not isinstance(raw_operation_witness, Mapping):
            raise AllocationAttributionError(
                "operation witness must be an object"
            )
        if not isinstance(audit, Mapping):
            raise AllocationAttributionError("audit must be an object")
        _require_exact_mapping_keys(
            audit,
            {
                "schema_version",
                "run_kind",
                "evidence_status",
                "execution_mode",
                "device",
                "device_index",
                "gpu_uuid",
                "torch_version",
                "cuda_runtime_version",
                "backend_identity",
                "warmup_iterations",
                "prepare_present",
                "prepare_attempted",
                "prepare_completed",
                "prepare_attempt_count",
                "prepare_completion_count",
                "max_history_entries",
                "recorder_configuration",
                "collection_started_ns",
                "collection_finished_ns",
                "operation_error",
                "production_binding",
                "production_binding_sha256",
                "operation_key",
                "external_provenance_status",
                "operation_witness",
                "memory",
                "attribution",
                "criterion",
                "raw_files",
                "profiler_timing_reported",
                "instrumented_duration_reported_as_timing",
                "normal_benchmark_timing_eligible",
            },
            "allocation audit envelope",
        )
        if (
            audit.get("schema_version")
            != PHASE3_ALLOCATION_AUDIT_SCHEMA_VERSION
            or audit.get("run_kind") != "allocation_audit"
            or audit.get("evidence_status") != "complete"
        ):
            raise AllocationAttributionError(
                "allocation audit schema/kind/status is invalid"
            )
        if audit.get("execution_mode") != production_binding.execution_mode:
            raise AllocationAttributionError(
                "allocation audit execution mode differs from operation key"
            )
        if (
            audit.get("device") != PHASE3_DEVICE
            or audit.get("device_index") != PHASE3_DEVICE_INDEX
        ):
            raise AllocationAttributionError(
                "allocation audit device is not frozen cuda:0"
            )
        if (
            audit.get("torch_version") != PHASE3_TORCH_VERSION
            or audit.get("cuda_runtime_version")
            != PHASE3_CUDA_RUNTIME_VERSION
        ):
            raise AllocationAttributionError(
                "allocation audit runtime identity is not frozen"
            )
        if audit.get("backend_identity") != production_binding.backend_identity:
            raise AllocationAttributionError(
                "allocation audit backend identity mismatch"
            )
        if (
            audit.get("warmup_iterations")
            != PHASE3_ALLOCATION_WARMUP_ITERATIONS
            or audit.get("max_history_entries")
            != PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES
            or audit.get("recorder_configuration")
            != PHASE3_RECORDER_CONFIGURATION
        ):
            raise AllocationAttributionError(
                "allocation audit collection configuration is not frozen"
            )
        if (
            audit.get("prepare_present") is not True
            or audit.get("prepare_attempted") is not True
            or audit.get("prepare_completed") is not True
            or audit.get("prepare_attempt_count") != 2
            or audit.get("prepare_completion_count") != 2
        ):
            raise AllocationAttributionError(
                "allocation audit prepare protocol is incomplete"
            )
        started = audit.get("collection_started_ns")
        finished = audit.get("collection_finished_ns")
        if (
            isinstance(started, bool)
            or not isinstance(started, int)
            or isinstance(finished, bool)
            or not isinstance(finished, int)
            or started <= 0
            or finished < started
        ):
            raise AllocationAttributionError(
                "allocation audit timestamps are invalid"
            )
        if (
            audit.get("profiler_timing_reported") is not False
            or audit.get("instrumented_duration_reported_as_timing") is not False
            or audit.get("normal_benchmark_timing_eligible") is not False
        ):
            raise AllocationAttributionError(
                "allocation audit timing-governance flags are invalid"
            )
        if (
            audit.get("operation_key")
            != production_binding.operation_key.to_dict()
            or audit.get("external_provenance_status")
            != PHASE3_EXTERNAL_PROVENANCE_STATUS
        ):
            raise AllocationAttributionError(
                "allocation audit operation-key join is invalid"
            )
        operation_witness = OperationWitnessEvidence.from_mapping(
            raw_operation_witness
        )
        witness_errors = operation_witness.validation_errors(
            production_binding
        )
        device_index = PHASE3_DEVICE_INDEX
        selected_trace = allocator_trace_from_snapshot(snapshot, device_index)
        if list(trace) != selected_trace:
            raise AllocationAttributionError(
                "raw trace differs from selected snapshot device trace"
            )
        before_sample = _raw_memory_sample_from_mapping(accounting_before)
        after_sample = _raw_memory_sample_from_mapping(accounting_after)
        gpu_uuid = audit.get("gpu_uuid")
        if (
            not isinstance(gpu_uuid, str)
            or not gpu_uuid
            or before_sample.gpu_uuid != gpu_uuid
            or after_sample.gpu_uuid != gpu_uuid
            or before_sample.operation_fingerprint_sha256
            != production_binding.operation_fingerprint_sha256
            or after_sample.operation_fingerprint_sha256
            != production_binding.operation_fingerprint_sha256
            or not (
                started
                <= before_sample.timestamp_ns
                <= after_sample.timestamp_ns
                <= finished
            )
        ):
            raise AllocationAttributionError(
                "allocation accounting provenance/timestamps mismatch envelope"
            )
        memory = memory_delta_from_raw_samples(before_sample, after_sample)
        counters = allocator_counters_from_memory_stats(
            stats_before, stats_after
        )
        rules = instantiate_decision_0009_production_rules(
            production_binding
        )
        max_entries = PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES
        history = build_history_integrity_evidence(
            snapshot,
            trace,
            max_entries=max_entries,
            stack_mode="all",
            expected_snapshot_sha256=files.snapshot_sha256,
            expected_trace_sha256=files.trace_sha256,
        )
        attributed = attribute_allocator_trace(
            trace,
            geometry=production_binding.geometry,
            counters=counters,
            rules=rules,
            backend_identity=production_binding.backend_identity,
            expected_trace_sha256=files.trace_sha256,
            history_integrity=history,
        )
        if witness_errors:
            attributed = replace(
                attributed,
                integrity_errors=tuple(
                    dict.fromkeys(
                        (*attributed.integrity_errors, *witness_errors)
                    )
                ),
            )
        if audit.get("operation_error") is not None:
            attributed = replace(
                attributed,
                integrity_errors=tuple(
                    dict.fromkeys(
                        (
                            *attributed.integrity_errors,
                            "instrumented_operation_failed",
                        )
                    )
                ),
            )
        criterion = (
            evaluate_refined_eager_criterion(
                attributed,
                memory,
                production_binding=production_binding,
            )
            if production_binding.execution_mode == "eager"
            else evaluate_strict_graph_criterion(attributed, memory)
        )
    except (
        AllocationAttributionError,
        SchemaValidationError,
        TypeError,
        ValueError,
    ) as error:
        reasons.append(
            "semantic_allocator_replay_failed:"
            f"{type(error).__name__}:{error}"
        )
        return SemanticAllocatorEvidenceValidation(
            False, tuple(dict.fromkeys(reasons)), None, None, None
        )

    if audit.get("production_binding") != production_binding.to_dict() or (
        audit.get("production_binding_sha256")
        != production_binding.identity_sha256
    ):
        reasons.append("serialized_production_binding_mismatch")
    if audit.get("operation_witness") != operation_witness.to_dict():
        reasons.append("serialized_operation_witness_mismatch")
    if audit.get("memory") != memory.to_dict():
        reasons.append("serialized_memory_derivation_mismatch")
    if audit.get("attribution") != attributed.to_dict():
        reasons.append("serialized_attribution_derivation_mismatch")
    serialized_criterion = audit.get("criterion")
    if not isinstance(serialized_criterion, Mapping):
        reasons.append("serialized_criterion_missing")
    else:
        serialized_without_passed = dict(serialized_criterion)
        serialized_without_passed.pop("passed", None)
        derived_without_passed = criterion.to_dict()
        derived_without_passed.pop("passed", None)
        if serialized_without_passed != derived_without_passed:
            reasons.append("serialized_criterion_derivation_mismatch")
    unique = tuple(dict.fromkeys(reasons))
    return SemanticAllocatorEvidenceValidation(
        passed=not unique and criterion.passed,
        failure_reasons=unique,
        memory=memory,
        attribution=attributed,
        criterion=criterion,
    )


def preserve_failed_allocator_audit(
    staging_directory: Path,
    *,
    audit_payload: Mapping[str, Any],
    partial_payloads: Mapping[str, object] | None = None,
) -> FailedAllocatorAuditFiles:
    """Preserve a no-replace, checksum-bound failed collection record."""

    partial_bytes: dict[str, bytes] = {}
    partial_files: list[PartialAllocatorEvidenceFile] = []
    for key, value in sorted((partial_payloads or {}).items()):
        if not key or not key.replace("_", "").isalnum():
            raise AllocationAttributionError(
                "partial allocator evidence key is invalid"
            )
        payload = _canonical_json_bytes(value)
        filename = f"allocation_audit_failed_{key}.json"
        partial_bytes[key] = payload
        partial_files.append(
            PartialAllocatorEvidenceFile(
                evidence_key=key,
                file=filename,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    bound_payload = dict(audit_payload)
    bound_payload["partial_files"] = [
        item.to_dict() for item in partial_files
    ]
    raw_snapshot_available = any(
        item.evidence_key == "snapshot" for item in partial_files
    )
    bound_payload["raw_snapshot_available"] = raw_snapshot_available
    bound_payload["raw_snapshot_preserved"] = raw_snapshot_available
    if not _valid_failed_allocator_audit_payload(bound_payload):
        raise AllocationAttributionError(
            "failed allocator audit payload is incomplete or malformed"
        )
    audit_bytes = _canonical_json_bytes(bound_payload)
    audit_digest = hashlib.sha256(audit_bytes).hexdigest()
    files = FailedAllocatorAuditFiles(
        audit_file="allocation_audit_failed.json",
        audit_sha256_file="allocation_audit_failed.sha256",
        audit_sha256=audit_digest,
        partial_files=tuple(partial_files),
    )
    digest_bytes = (
        f"{audit_digest}  {files.audit_file}\n".encode("ascii")
    )
    directory_fd = _staging_directory_fd(staging_directory)
    try:
        _preflight_names(
            directory_fd,
            (
                *(item.file for item in files.partial_files),
                files.audit_file,
                files.audit_sha256_file,
            ),
        )
        for item in files.partial_files:
            _exclusive_write(
                directory_fd,
                item.file,
                partial_bytes[item.evidence_key],
            )
        _exclusive_write(directory_fd, files.audit_file, audit_bytes)
        _exclusive_write(
            directory_fd, files.audit_sha256_file, digest_bytes
        )
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return files


def verify_preserved_failed_allocator_audit(
    staging_directory: Path,
    files: FailedAllocatorAuditFiles,
) -> bool:
    """Verify a failed-audit JSON and its detached digest ledger."""

    directory_fd = _staging_directory_fd(staging_directory)
    try:
        audit = _read_no_follow(directory_fd, files.audit_file)
        digest_file = _read_no_follow(
            directory_fd, files.audit_sha256_file
        )
        partial_contents = {
            item.evidence_key: _read_no_follow(directory_fd, item.file)
            for item in files.partial_files
        }
    except OSError:
        return False
    finally:
        os.close(directory_fd)
    try:
        audit_payload = json.loads(audit)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(
        audit_payload, Mapping
    ) or not _valid_failed_allocator_audit_payload(audit_payload):
        return False
    if audit_payload.get("partial_files") != [
        item.to_dict() for item in files.partial_files
    ]:
        return False
    for item in files.partial_files:
        content = partial_contents[item.evidence_key]
        if hashlib.sha256(content).hexdigest() != item.sha256:
            return False
        try:
            parsed = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if _canonical_json_bytes(parsed) != content:
            return False
    return (
        hashlib.sha256(audit).hexdigest() == files.audit_sha256
        and digest_file
        == f"{files.audit_sha256}  {files.audit_file}\n".encode("ascii")
    )


def _valid_failed_allocator_audit_payload(
    payload: Mapping[str, Any],
) -> bool:
    started = payload.get("collection_started_ns")
    finished = payload.get("collection_finished_ns")
    rules = payload.get("attribution_rules")
    rules_sha = payload.get("attribution_rules_sha256")
    binding = payload.get("production_binding")
    binding_sha = payload.get("production_binding_sha256")
    partial_files = payload.get("partial_files")
    if not isinstance(partial_files, list) or not all(
        isinstance(item, Mapping)
        and isinstance(item.get("evidence_key"), str)
        and isinstance(item.get("file"), str)
        and isinstance(item.get("sha256"), str)
        and _valid_sha256(item["sha256"])
        for item in partial_files
    ):
        return False
    snapshot_preserved = any(
        item.get("evidence_key") == "snapshot" for item in partial_files
    )
    failure_stage = payload.get("failure_stage")
    attempt_count = payload.get("prepare_attempt_count")
    completion_count = payload.get("prepare_completion_count")
    early_stages = {
        "input_validation",
        "runtime_initialization",
        "device_validation",
        "allocator_api_validation",
        "warmup",
        "warmup_sync",
    }
    one_completed_stages = {
        "operation_witness_prepare_sync",
        "operation_witness_before",
        "operation_witness_operation",
        "operation_witness_output",
        "operation_witness_after",
        "operation_witness_cleanup_sync",
    }
    if failure_stage in early_stages:
        expected_prepare_counts = (0, 0)
    elif failure_stage == "operation_witness_prepare":
        expected_prepare_counts = (1, 0)
    elif failure_stage in one_completed_stages:
        expected_prepare_counts = (1, 1)
    elif failure_stage == "post_witness_prepare":
        expected_prepare_counts = (2, 1)
    else:
        expected_prepare_counts = (2, 2)
    binding_key = (
        binding.get("operation_key") if isinstance(binding, Mapping) else None
    )
    warmup_iterations = payload.get("warmup_iterations")
    max_history_entries = payload.get("max_history_entries")
    collection_configuration_valid = (
        isinstance(warmup_iterations, int)
        and not isinstance(warmup_iterations, bool)
        and warmup_iterations > 0
        and isinstance(max_history_entries, int)
        and not isinstance(max_history_entries, bool)
        and max_history_entries > 0
        and (
            failure_stage == "input_validation"
            or (
                warmup_iterations
                == PHASE3_ALLOCATION_WARMUP_ITERATIONS
                and max_history_entries
                == PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES
            )
        )
    )
    return (
        payload.get("schema_version")
        == PHASE3_ALLOCATION_AUDIT_SCHEMA_VERSION
        and payload.get("run_kind") == "allocation_audit"
        and payload.get("execution_mode") in {"eager", "cuda_graph"}
        and payload.get("evidence_status") == "failed"
        and isinstance(failure_stage, str)
        and bool(failure_stage)
        and isinstance(payload.get("failure_type"), str)
        and bool(payload.get("failure_type"))
        and payload.get("raw_snapshot_available") is snapshot_preserved
        and payload.get("raw_snapshot_preserved") is snapshot_preserved
        and isinstance(payload.get("backend_identity"), str)
        and bool(payload.get("backend_identity"))
        and isinstance(payload.get("geometry"), Mapping)
        and isinstance(rules, Mapping)
        and isinstance(rules_sha, str)
        and _valid_sha256(rules_sha)
        and hashlib.sha256(_canonical_json_bytes(rules)).hexdigest()
        == rules_sha
        and isinstance(binding, Mapping)
        and isinstance(payload.get("prepare_present"), bool)
        and isinstance(payload.get("prepare_attempted"), bool)
        and isinstance(payload.get("prepare_completed"), bool)
        and isinstance(attempt_count, int)
        and not isinstance(attempt_count, bool)
        and isinstance(completion_count, int)
        and not isinstance(completion_count, bool)
        and (attempt_count, completion_count) == expected_prepare_counts
        and payload.get("prepare_attempted") is (attempt_count > 0)
        and payload.get("prepare_completed") is (completion_count == 2)
        and (
            payload.get("prepare_present") is True
            or (attempt_count, completion_count) == (0, 0)
        )
        and payload.get("operation_key") == binding_key
        and payload.get("external_provenance_status")
        == PHASE3_EXTERNAL_PROVENANCE_STATUS
        and payload.get("recorder_configuration")
        == PHASE3_RECORDER_CONFIGURATION
        and collection_configuration_valid
        and isinstance(binding_sha, str)
        and _valid_sha256(binding_sha)
        and hashlib.sha256(_canonical_json_bytes(binding)).hexdigest()
        == binding_sha
        and isinstance(started, int)
        and not isinstance(started, bool)
        and isinstance(finished, int)
        and not isinstance(finished, bool)
        and finished >= started
        and payload.get("profiler_timing_reported") is False
        and payload.get("instrumented_duration_reported_as_timing") is False
        and payload.get("normal_benchmark_timing_eligible") is False
    )


def _torch_module() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise AllocationAttributionError(
                "PyTorch is unavailable for allocator collection"
            ) from error
    return _TORCH


def _raw_device_memory_info(
    torch: Any, device: Any
) -> tuple[int, int]:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    except Exception as error:
        raise AllocationAttributionError(
            "device-level memory accounting is unavailable"
        ) from error
    if (
        isinstance(free_bytes, bool)
        or not isinstance(free_bytes, int)
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or free_bytes < 0
        or total_bytes < free_bytes
    ):
        raise AllocationAttributionError(
            "device-level memory accounting is malformed"
        )
    return free_bytes, total_bytes


def _raw_memory_accounting_sample(
    torch: Any,
    device: Any,
    *,
    sample_role: str,
    operation_fingerprint_sha256: str,
    device_index: int,
    gpu_uuid: str,
) -> RawMemoryAccountingSample:
    free_bytes, total_bytes = _raw_device_memory_info(torch, device)
    return RawMemoryAccountingSample(
        schema_version="kvbench-phase3-memory-accounting-2.0.0",
        operation_fingerprint_sha256=operation_fingerprint_sha256,
        sample_role=sample_role,
        timestamp_ns=time.time_ns(),
        device=str(device),
        device_index=device_index,
        gpu_uuid=gpu_uuid,
        allocated_bytes=int(torch.cuda.memory_allocated(device=device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device=device)),
        device_free_bytes=free_bytes,
        device_total_bytes=total_bytes,
    )


def _validate_evidence_destination(staging_directory: Path) -> None:
    directory_fd = _staging_directory_fd(staging_directory)
    try:
        _preflight_names(
            directory_fd,
            (
                "allocator_snapshot.json",
                "allocator_trace.json",
                "allocator_memory_stats_before.json",
                "allocator_memory_stats_after.json",
                "allocator_memory_accounting_before.json",
                "allocator_memory_accounting_after.json",
                "allocation_operation_witness.json",
                "allocation_audit.json",
                "allocation_audit.sha256",
                "allocation_audit_failed.json",
                "allocation_audit_failed.sha256",
            ),
        )
    finally:
        os.close(directory_fd)


def collect_cuda_allocation_attribution(
    operation: Callable[[], Any],
    *,
    production_binding: ProductionAllocationBinding,
    staging_directory: Path,
    operation_witness: OperationWitnessCallbacks,
    warmup_operation: Callable[[], Any] | None = None,
    prepare_operation: Callable[[], Any] | None = None,
    warmup_iterations: int = 3,
    max_entries: int = 100_000,
    device: Any | None = None,
) -> CollectedAllocationAudit:
    """Collect one untimed allocation audit after explicit warmup.

    The operation and optional warmup closure must preserve the caller's
    frozen decode semantics.  No duration from this function is reportable as
    benchmark timing.  The three evidence files are written exclusively into
    an existing caller-owned staging directory; this function does not write
    COMPLETE or promote the directory.
    """

    if type(production_binding) is not ProductionAllocationBinding:
        raise AllocationAttributionError(
            "production_binding must be a ProductionAllocationBinding"
        )
    binding_errors = production_binding.validation_errors()
    if binding_errors:
        raise AllocationAttributionError(
            "invalid production allocation binding: "
            + ", ".join(binding_errors)
        )
    mode = AllocationExecutionMode(production_binding.execution_mode)
    geometry = production_binding.geometry
    backend_identity = production_binding.backend_identity
    rules = instantiate_decision_0009_production_rules(production_binding)
    _validate_evidence_destination(staging_directory)

    collection_started_ns = time.time_ns()
    warmup = operation if warmup_operation is None else warmup_operation
    partial_payloads: dict[str, object] = {}
    operation_error: str | None = None
    snapshot: Mapping[str, Any] | None = None
    trace: list[Mapping[str, Any]] | None = None
    stats_before: Mapping[str, Any] | None = None
    stats_after: Mapping[str, Any] | None = None
    accounting_before: RawMemoryAccountingSample | None = None
    accounting_after: RawMemoryAccountingSample | None = None
    witness_before: OperationCacheStateWitness | None = None
    witness_after: OperationCacheStateWitness | None = None
    witness_output: OperationOutputWitness | None = None
    measured_before: OperationCacheStateWitness | None = None
    measured_after: OperationCacheStateWitness | None = None
    measured_output: OperationOutputWitness | None = None
    operation_witness_evidence: OperationWitnessEvidence | None = None
    torch: Any | None = None
    selected: Any | None = None
    device_index: int | None = None
    gpu_uuid: str | None = None
    recorder: Any | None = None
    snapshot_function: Any | None = None
    torch_version: str | None = None
    cuda_runtime_version: str | None = None
    prepare_attempt_count = 0
    prepare_completion_count = 0
    failure_stage = "input_validation"
    recorder_enabled = False
    recorder_disable_error: Exception | None = None
    collection_error_before_replay: Exception | None = None
    try:
        if not callable(operation):
            raise AllocationAttributionError("operation must be callable")
        if warmup_operation is not None and not callable(warmup_operation):
            raise AllocationAttributionError(
                "warmup_operation must be callable"
            )
        if prepare_operation is not None and not callable(prepare_operation):
            raise AllocationAttributionError(
                "prepare_operation must be callable"
            )
        if not isinstance(operation_witness, OperationWitnessCallbacks):
            raise AllocationAttributionError(
                "operation_witness must be OperationWitnessCallbacks"
            )
        if warmup_iterations != PHASE3_ALLOCATION_WARMUP_ITERATIONS:
            raise AllocationAttributionError(
                "warmup_iterations must equal the frozen allocation-audit value"
            )
        if max_entries != PHASE3_ALLOCATION_MAX_HISTORY_ENTRIES:
            raise AllocationAttributionError(
                "max_entries must equal the frozen allocation-audit value"
            )
        if prepare_operation is None:
            raise AllocationAttributionError(
                "production allocation audit requires prepare_operation"
            )
        failure_stage = "runtime_initialization"
        torch = _torch_module()
        torch_version = str(torch.__version__)
        cuda_runtime_version = (
            None
            if getattr(torch.version, "cuda", None) is None
            else str(torch.version.cuda)
        )
        if (
            torch_version != PHASE3_TORCH_VERSION
            or cuda_runtime_version != PHASE3_CUDA_RUNTIME_VERSION
        ):
            raise AllocationAttributionError(
                "allocator audit runtime differs from the frozen build"
            )
        failure_stage = "device_validation"
        selected = torch.device(
            f"cuda:{torch.cuda.current_device()}" if device is None else device
        )
        if selected.type != "cuda":
            raise AllocationAttributionError(
                "allocator history collection requires a CUDA device"
            )
        device_index = (
            int(torch.cuda.current_device())
            if selected.index is None
            else int(selected.index)
        )
        if str(selected) != PHASE3_DEVICE or device_index != PHASE3_DEVICE_INDEX:
            raise AllocationAttributionError(
                "allocator history collection requires frozen cuda:0"
            )
        properties = torch.cuda.get_device_properties(device_index)
        gpu_uuid = str(getattr(properties, "uuid", ""))
        if not gpu_uuid:
            raise AllocationAttributionError(
                "allocator history collection requires a GPU UUID"
            )
        failure_stage = "allocator_api_validation"
        recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
        snapshot_function = getattr(torch.cuda.memory, "_snapshot", None)
        if not callable(recorder) or not callable(snapshot_function):
            raise AllocationAttributionError(
                "PyTorch allocator history APIs are unavailable"
            )
        failure_stage = "warmup"
        for _ in range(warmup_iterations):
            warmup_result = warmup()
            del warmup_result
        failure_stage = "warmup_sync"
        torch.cuda.synchronize(device=selected)
        failure_stage = "operation_witness_prepare"
        prepare_attempt_count += 1
        prepare_result = prepare_operation()
        prepare_completion_count += 1
        del prepare_result
        failure_stage = "operation_witness_prepare_sync"
        torch.cuda.synchronize(device=selected)
        failure_stage = "operation_witness_before"
        witness_before = operation_witness.capture_cache_state()
        if not isinstance(witness_before, OperationCacheStateWitness):
            raise AllocationAttributionError(
                "operation witness before callback returned wrong type"
            )
        failure_stage = "operation_witness_operation"
        witness_result = operation()
        torch.cuda.synchronize(device=selected)
        witness_result_metadata = _operation_output_metadata(witness_result)
        failure_stage = "operation_witness_output"
        witness_output = operation_witness.capture_output(witness_result)
        if not isinstance(witness_output, OperationOutputWitness):
            raise AllocationAttributionError(
                "operation witness output callback returned wrong type"
            )
        if witness_output.metadata() != witness_result_metadata:
            raise AllocationAttributionError(
                "operation witness output metadata differs from returned output"
            )
        failure_stage = "operation_witness_after"
        witness_after = operation_witness.capture_cache_state()
        if not isinstance(witness_after, OperationCacheStateWitness):
            raise AllocationAttributionError(
                "operation witness after callback returned wrong type"
            )
        del witness_result
        failure_stage = "operation_witness_cleanup_sync"
        torch.cuda.synchronize(device=selected)
        partial_payloads["operation_witness_untimed"] = {
            "operation_key": production_binding.operation_key.to_dict(),
            "operation_fingerprint_sha256": (
                production_binding.operation_fingerprint_sha256
            ),
            "reference_before": witness_before.to_dict(),
            "reference_after": witness_after.to_dict(),
            "reference_output": witness_output.to_dict(),
        }
        failure_stage = "post_witness_prepare"
        prepare_attempt_count += 1
        prepare_result = prepare_operation()
        prepare_completion_count += 1
        del prepare_result
        failure_stage = "post_witness_prepare_sync"
        torch.cuda.synchronize(device=selected)
        failure_stage = "post_witness_prepare_validation"
        measured_before = operation_witness.capture_cache_state()
        if not isinstance(measured_before, OperationCacheStateWitness):
            raise AllocationAttributionError(
                "operation witness prepared-state callback returned wrong type"
            )
        failure_stage = "pre_operation_memory_accounting"
        accounting_before = _raw_memory_accounting_sample(
            torch,
            selected,
            sample_role="before",
            operation_fingerprint_sha256=(
                production_binding.operation_fingerprint_sha256
            ),
            device_index=device_index,
            gpu_uuid=gpu_uuid,
        )
        partial_payloads["memory_accounting_before"] = (
            accounting_before.to_dict()
        )
        failure_stage = "pre_operation_memory_stats"
        raw_stats_before = torch.cuda.memory_stats(device=selected)
        if not isinstance(raw_stats_before, Mapping):
            raise AllocationAttributionError(
                "pre-operation CUDA memory stats are malformed"
            )
        stats_before = dict(raw_stats_before)
        partial_payloads["memory_stats_before"] = stats_before
        failure_stage = "allocator_history_enable"
        recorder(
            enabled=PHASE3_RECORDER_CONFIGURATION["enabled"],
            context=PHASE3_RECORDER_CONFIGURATION["context"],
            stacks=PHASE3_RECORDER_CONFIGURATION["stacks"],
            max_entries=PHASE3_RECORDER_CONFIGURATION["max_entries"],
            device=selected,
            clear_history=PHASE3_RECORDER_CONFIGURATION["clear_history"],
        )
        recorder_enabled = True
        result: Any | None = None
        try:
            result = operation()
        except Exception as error:
            operation_error = (
                f"{type(error).__module__}.{type(error).__qualname__}: {error}"
            )
        failure_stage = "post_operation_sync"
        torch.cuda.synchronize(device=selected)
        if result is not None:
            failure_stage = "measured_output_witness"
            result_metadata = _operation_output_metadata(result)
            measured_output = _allocation_audit_capture_output(
                operation_witness.capture_output,
                result,
            )
            if not isinstance(measured_output, OperationOutputWitness):
                raise AllocationAttributionError(
                    "measured output callback returned wrong type"
                )
            if measured_output.metadata() != result_metadata:
                raise AllocationAttributionError(
                    "measured output witness metadata differs from returned output"
                )
        del result
        failure_stage = "measured_output_cleanup_sync"
        torch.cuda.synchronize(device=selected)
        failure_stage = "post_operation_memory_stats"
        stats_after = torch.cuda.memory_stats(device=selected)
        if not isinstance(stats_after, Mapping):
            raise AllocationAttributionError(
                "post-operation CUDA memory stats are malformed"
            )
        stats_after = dict(stats_after)
        partial_payloads["memory_stats_after"] = stats_after
        failure_stage = "post_operation_memory_accounting"
        accounting_after = _raw_memory_accounting_sample(
            torch,
            selected,
            sample_role="after",
            operation_fingerprint_sha256=(
                production_binding.operation_fingerprint_sha256
            ),
            device_index=device_index,
            gpu_uuid=gpu_uuid,
        )
        partial_payloads["memory_accounting_after"] = (
            accounting_after.to_dict()
        )
        failure_stage = "allocator_snapshot"
        raw_snapshot = snapshot_function(device=selected)
        if not isinstance(raw_snapshot, Mapping):
            raise AllocationAttributionError(
                "CUDA allocator snapshot is malformed"
            )
        snapshot = raw_snapshot
        partial_payloads["snapshot"] = snapshot
    except Exception as error:
        collection_error_before_replay = error
    finally:
        if recorder_enabled:
            try:
                assert callable(recorder)
                recorder(enabled=None, device=selected)
            except Exception as error:
                recorder_disable_error = error
                if collection_error_before_replay is None:
                    failure_stage = "allocator_history_disable"

    try:
        if collection_error_before_replay is not None:
            raise collection_error_before_replay
        if recorder_disable_error is not None:
            raise recorder_disable_error
        failure_stage = "operation_witness_measured_post"
        measured_after = operation_witness.capture_cache_state()
        if not isinstance(measured_after, OperationCacheStateWitness):
            raise AllocationAttributionError(
                "operation witness measured-post callback returned wrong type"
            )
        if (
            witness_before is None
            or witness_after is None
            or witness_output is None
            or measured_before is None
            or measured_after is None
        ):
            raise AllocationAttributionError(
                "successful collection path lacks untimed operation witness"
            )
        operation_witness_evidence = OperationWitnessEvidence(
            operation_key=production_binding.operation_key,
            operation_fingerprint_sha256=(
                production_binding.operation_fingerprint_sha256
            ),
            reference_before=witness_before,
            reference_after=witness_after,
            reference_output=witness_output,
            measured_before=measured_before,
            measured_after=measured_after,
            measured_output=measured_output,
            recorder_configuration=dict(PHASE3_RECORDER_CONFIGURATION),
        )
        partial_payloads["operation_witness"] = (
            operation_witness_evidence.to_dict()
        )
        if (
            snapshot is None
            or stats_before is None
            or stats_after is None
            or accounting_before is None
            or accounting_after is None
        ):
            raise AllocationAttributionError(
                "successful collection path lacks required raw evidence"
            )
        failure_stage = "allocator_trace_parse"
        trace = allocator_trace_from_snapshot(snapshot, device_index)
        partial_payloads["trace"] = trace
        failure_stage = "memory_derivation"
        memory = memory_delta_from_raw_samples(
            accounting_before, accounting_after
        )
        counters = allocator_counters_from_memory_stats(
            stats_before, stats_after
        )
        failure_stage = "history_integrity_derivation"
        history = build_history_integrity_evidence(
            snapshot,
            trace,
            max_entries=max_entries,
            stack_mode="all",
        )
        failure_stage = "allocation_attribution"
        attributed = attribute_allocator_trace(
            trace,
            geometry=geometry,
            counters=counters,
            rules=rules,
            backend_identity=backend_identity,
            expected_trace_sha256=history.expected_raw_trace_sha256,
            history_integrity=history,
        )
        witness_errors = operation_witness_evidence.validation_errors(
            production_binding
        )
        if witness_errors:
            attributed = replace(
                attributed,
                integrity_errors=tuple(
                    dict.fromkeys(
                        (*attributed.integrity_errors, *witness_errors)
                    )
                ),
            )
        if operation_error is not None:
            attributed = replace(
                attributed,
                integrity_errors=tuple(
                    dict.fromkeys(
                        (
                            *attributed.integrity_errors,
                            "instrumented_operation_failed",
                        )
                    )
                ),
            )
        failure_stage = "criterion_evaluation"
        criterion = (
            evaluate_refined_eager_criterion(
                attributed,
                memory,
                production_binding=production_binding,
            )
            if mode is AllocationExecutionMode.EAGER
            else evaluate_strict_graph_criterion(attributed, memory)
        )
        stats_before_bytes = _canonical_json_bytes(stats_before)
        stats_after_bytes = _canonical_json_bytes(stats_after)
        accounting_before_dict = accounting_before.to_dict()
        accounting_after_dict = accounting_after.to_dict()
        accounting_before_bytes = _canonical_json_bytes(
            accounting_before_dict
        )
        accounting_after_bytes = _canonical_json_bytes(
            accounting_after_dict
        )
        files = RawAllocatorEvidenceFiles(
            snapshot_file="allocator_snapshot.json",
            snapshot_sha256=history.raw_snapshot_sha256,
            trace_file="allocator_trace.json",
            trace_sha256=history.raw_trace_sha256,
            memory_stats_before_file="allocator_memory_stats_before.json",
            memory_stats_before_sha256=hashlib.sha256(
                stats_before_bytes
            ).hexdigest(),
            memory_stats_after_file="allocator_memory_stats_after.json",
            memory_stats_after_sha256=hashlib.sha256(
                stats_after_bytes
            ).hexdigest(),
            memory_accounting_before_file=(
                "allocator_memory_accounting_before.json"
            ),
            memory_accounting_before_sha256=hashlib.sha256(
                accounting_before_bytes
            ).hexdigest(),
            memory_accounting_after_file=(
                "allocator_memory_accounting_after.json"
            ),
            memory_accounting_after_sha256=hashlib.sha256(
                accounting_after_bytes
            ).hexdigest(),
            operation_witness_file="allocation_operation_witness.json",
            operation_witness_sha256=hashlib.sha256(
                _canonical_json_bytes(operation_witness_evidence.to_dict())
            ).hexdigest(),
            audit_file="allocation_audit.json",
            audit_sha256_file="allocation_audit.sha256",
            audit_sha256=None,
        )
        if (
            torch_version is None
            or cuda_runtime_version is None
            or device_index is None
            or gpu_uuid is None
        ):
            raise AllocationAttributionError(
                "successful collection lacks frozen runtime/device identity"
            )
        collected = CollectedAllocationAudit(
            execution_mode=mode,
            device=str(selected),
            device_index=device_index,
            gpu_uuid=gpu_uuid,
            torch_version=torch_version,
            cuda_runtime_version=cuda_runtime_version,
            backend_identity=backend_identity,
            warmup_iterations=warmup_iterations,
            prepare_present=prepare_operation is not None,
            prepare_attempted=prepare_attempt_count > 0,
            prepare_completed=prepare_completion_count == 2,
            prepare_attempt_count=prepare_attempt_count,
            prepare_completion_count=prepare_completion_count,
            max_history_entries=max_entries,
            recorder_configuration=dict(PHASE3_RECORDER_CONFIGURATION),
            collection_started_ns=collection_started_ns,
            collection_finished_ns=time.time_ns(),
            operation_error=operation_error,
            production_binding=production_binding,
            operation_witness=operation_witness_evidence,
            memory=memory,
            attribution=attributed,
            criterion=criterion,
            raw_files=files,
        )
        failure_stage = "evidence_preservation"
        preserved = preserve_allocator_evidence(
            staging_directory,
            snapshot=snapshot,
            trace=trace,
            memory_stats_before=stats_before,
            memory_stats_after=stats_after,
            memory_accounting_before=accounting_before_dict,
            memory_accounting_after=accounting_after_dict,
            operation_witness=operation_witness_evidence.to_dict(),
            expected_snapshot_sha256=history.raw_snapshot_sha256,
            expected_trace_sha256=history.raw_trace_sha256,
            audit_payload=collected.to_dict(include_audit_sha256=False),
        )
        collected = replace(collected, raw_files=preserved)
        failure_stage = "semantic_replay_validation"
        semantic = validate_preserved_allocator_evidence_semantically(
            staging_directory,
            preserved,
            production_binding=production_binding,
        )
        if semantic.failure_reasons:
            raise AllocationAttributionError(
                "semantic replay mismatch: "
                + ", ".join(semantic.failure_reasons)
            )
        return collected
    except Exception as collection_error:
        collection_finished_ns = time.time_ns()
        error_type = (
            f"{type(collection_error).__module__}."
            f"{type(collection_error).__qualname__}"
        )
        failed_payload: dict[str, Any] = {
            "schema_version": PHASE3_ALLOCATION_AUDIT_SCHEMA_VERSION,
            "run_kind": "allocation_audit",
            "execution_mode": mode.value,
            "evidence_status": "failed",
            "failure_stage": failure_stage,
            "failure_type": error_type,
            "failure_message": str(collection_error),
            "recorder_disable_error": (
                None
                if recorder_disable_error is None
                else (
                    f"{type(recorder_disable_error).__module__}."
                    f"{type(recorder_disable_error).__qualname__}: "
                    f"{recorder_disable_error}"
                )
            ),
            "raw_snapshot_available": False,
            "raw_snapshot_preserved": False,
            "device": None if selected is None else str(selected),
            "device_index": device_index,
            "gpu_uuid": gpu_uuid,
            "torch_version": torch_version,
            "cuda_runtime_version": cuda_runtime_version,
            "backend_identity": backend_identity,
            "geometry": geometry.to_dict(),
            "production_binding": production_binding.to_dict(),
            "production_binding_sha256": (
                production_binding.identity_sha256
            ),
            "operation_key": production_binding.operation_key.to_dict(),
            "external_provenance_status": (
                production_binding.external_provenance_status
            ),
            "attribution_rules": rules.to_dict(),
            "attribution_rules_sha256": rules.identity_sha256,
            "warmup_iterations": warmup_iterations,
            "prepare_present": prepare_operation is not None,
            "prepare_attempted": prepare_attempt_count > 0,
            "prepare_completed": prepare_completion_count == 2,
            "prepare_attempt_count": prepare_attempt_count,
            "prepare_completion_count": prepare_completion_count,
            "max_history_entries": max_entries,
            "recorder_configuration": dict(PHASE3_RECORDER_CONFIGURATION),
            "collection_started_ns": collection_started_ns,
            "collection_finished_ns": collection_finished_ns,
            "operation_error": operation_error,
            "profiler_timing_reported": False,
            "instrumented_duration_reported_as_timing": False,
            "normal_benchmark_timing_eligible": False,
        }
        try:
            failed_files = preserve_failed_allocator_audit(
                staging_directory,
                audit_payload=failed_payload,
                partial_payloads=partial_payloads,
            )
        except Exception as preservation_error:
            raise AllocationAttributionError(
                "allocator collection failed at "
                f"{failure_stage}; failed-evidence preservation also failed"
            ) from preservation_error
        raise AllocationAttributionError(
            "allocator collection failed at "
            f"{failure_stage}; checksum-bound failed evidence preserved as "
            f"{failed_files.audit_file} ({failed_files.audit_sha256})"
        ) from collection_error
