"""Untimed CUDA-device dispatch evidence for the Phase 3 GQA audit.

Chrome parsing and proof evaluation intentionally have no PyTorch dependency.
The collection helper imports PyTorch lazily and returns only raw-artifact
identity: profiler timestamps and durations never become benchmark timing.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any
import warnings

from kvbench.runtime.gqa_taxonomy import classify_gqa_evidence
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.schema import (
    GQAVerdict,
    derive_cache_layout_fingerprint,
    derive_phase3_point_fingerprint,
)
from kvbench.schema.base import require_identifier, require_run_id


FLASH_FORWARD_FAMILY = "pytorch_flash::flash_fwd_kernel"
FLASH_SPLIT_KV_FAMILY = "pytorch_flash::flash_fwd_splitkv"
DEVICE_EVENT_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
MATERIALIZATION_CLASSIFICATIONS = frozenset(
    {
        "repeat_materialization",
        "expand_materialization",
    }
)
COPY_CANDIDATE_CLASSIFICATIONS = frozenset(
    {"device_copy_candidate", "copy_candidate", "transpose_copy_candidate"}
)
FROZEN_CONTROL_BATCH_SIZE = 1
FROZEN_CONTROL_CONTEXT_LENGTH = 128
FROZEN_CONTROL_QUERY_LENGTH = 1
REQUIRED_SUT_SOURCES = (
    "src/kvbench/runtime/backend.py",
    "src/kvbench/runtime/bf16_endpoint.py",
    "src/kvbench/runtime/static_cache.py",
)
PHASE3_CACHE_LAYOUT_NAME = "layers_batch_kv_heads_context_head_dim"
PHASE3_CACHE_LAYOUT_SCHEMA = "kvbench-bf16-static-cache-layout-1.0.0"
PHASE3_NUM_LAYERS = 32
PHASE3_NUM_QUERY_HEADS = 32
PHASE3_NUM_KV_HEADS = 8
PHASE3_HEAD_DIM = 128
PHASE3_DTYPE = "torch.bfloat16"
PHASE3_DTYPE_BYTES = 2
EAGER_EXECUTION_MODE = "eager"
CUDA_GRAPH_REPLAY_EXECUTION_MODE = "cuda_graph_replay"
DISPATCH_EXECUTION_MODES = frozenset(
    {EAGER_EXECUTION_MODE, CUDA_GRAPH_REPLAY_EXECUTION_MODE}
)
PHASE3_FLASH_RELATED_FAMILIES = (
    (FLASH_FORWARD_FAMILY, FLASH_SPLIT_KV_FAMILY),
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_PHASE3_POINT_RE = re.compile(
    r"\A(?P<runner>fixed_l|growing_context)-"
    r"b(?P<batch>[1-9][0-9]*)-l(?P<context>[1-9][0-9]*)-"
    r"(?P<graph>eager|cuda_graph)-r(?P<replicate>[1-9][0-9]*)\Z"
)
_FLASH_FORWARD_KERNEL_RE = re.compile(
    r"pytorch_flash::flash_fwd_kernel(?:<|\()"
)
_FLASH_SPLIT_FORWARD_KERNEL_RE = re.compile(
    r"pytorch_flash::flash_fwd_splitkv_kernel(?:<|\()"
)
_FLASH_SPLIT_COMBINE_KERNEL_RE = re.compile(
    r"pytorch_flash::flash_fwd_splitkv_combine_kernel(?:<|\()"
)
_EVENT_CLASSIFICATIONS = frozenset(
    {
        "flash_attention",
        "device_copy_candidate",
        "device_memset",
        "repeat_materialization",
        "expand_materialization",
        "transpose_copy_candidate",
        "copy_candidate",
        "unknown_kernel",
    }
)
_DEVICE_LIKE_CATEGORY_RE = re.compile(
    r"(?:cuda|gpu|kernel|memcpy|memset)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_SOURCE_PATTERNS = (
    ("repeat_kv", re.compile(r"\brepeat_kv\b")),
    ("repeat_interleave", re.compile(r"\brepeat_interleave\b")),
    ("tensor_repeat", re.compile(r"\.repeat\s*\(")),
    ("tensor_expand", re.compile(r"\.expand\s*\(")),
    (
        "replication_copy",
        re.compile(
            r"\b(?:expanded|query_head|replicated)_"
            r"(?:kv|key|value)[A-Za-z0-9_]*"
            r"\.copy_?\s*\("
        ),
    ),
    ("torch_cat", re.compile(r"\btorch\.cat\s*\(")),
    ("dynamic_cache", re.compile(r"\bDynamicCache\b")),
)
_SELECTED_SOURCE_FUNCTION_PATHS = {
    REQUIRED_SUT_SOURCES[0]: ("flash_attention_forward",),
    REQUIRED_SUT_SOURCES[1]: (
        "BF16DecodeEndpoint._attention",
        "BF16DecodeEndpoint._base_forward",
        "BF16DecodeEndpoint.decode",
    ),
    REQUIRED_SUT_SOURCES[2]: ("BF16StaticCache.update",),
}


def phase3_source_identity_sha256(
    source_sha256_by_path: Mapping[str, str],
) -> str:
    """Derive the exact ordered identity of the three selected SUT sources."""

    if set(source_sha256_by_path) != set(REQUIRED_SUT_SOURCES):
        raise ValueError("Phase 3 source identity requires the exact SUT source set")
    ordered: list[dict[str, str]] = []
    for path in REQUIRED_SUT_SOURCES:
        digest = source_sha256_by_path[path]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("Phase 3 source identity contains an invalid digest")
        ordered.append({"path": path, "sha256": digest})
    raw = json.dumps(
        {
            "schema_version": "kvbench-phase3-sut-source-identity-1.0.0",
            "sources": ordered,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class GQADeviceDispatchError(RuntimeError):
    """Device-dispatch evidence could not be collected or validated."""


class ChromeTraceValidationError(GQADeviceDispatchError):
    """A Chrome trace cannot support a fail-closed CUDA audit."""


def _strict_json_loads(raw: bytes) -> object:
    if not isinstance(raw, bytes) or not raw:
        raise ChromeTraceValidationError("Chrome trace bytes are absent")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ChromeTraceValidationError("Chrome trace is not UTF-8") from error

    def reject_constant(value: str) -> None:
        raise ChromeTraceValidationError(
            f"Chrome trace contains a non-finite constant: {value}"
        )

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ChromeTraceValidationError(
                    f"Chrome trace contains duplicate key: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except ChromeTraceValidationError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise ChromeTraceValidationError("Chrome trace is malformed JSON") from error


def _required_trace_integer(
    args: Mapping[str, object],
    key: str,
    *,
    positive: bool,
) -> int:
    value = args.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChromeTraceValidationError(
            "CUDA device event lacks integer " + repr(key)
        )
    minimum = 1 if positive else 0
    if value < minimum:
        raise ChromeTraceValidationError(
            "CUDA device event has invalid " + repr(key)
        )
    return value


def _optional_trace_integer(args: Mapping[str, object], key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChromeTraceValidationError(
            "CUDA device event has invalid optional " + repr(key)
        )
    return value


def _optional_positive_trace_integer(
    args: Mapping[str, object],
    key: str,
) -> int | None:
    if key not in args:
        return None
    value = args[key]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ChromeTraceValidationError(
            "CUDA device event has invalid optional " + repr(key)
        )
    return value


def _optional_graph_identity(
    args: Mapping[str, object],
) -> tuple[int | None, int | None]:
    graph_key = "graph id"
    node_key = "graph node id"
    graph_present = graph_key in args
    node_present = node_key in args
    if not graph_present and not node_present:
        return None, None
    if graph_present != node_present:
        raise ChromeTraceValidationError(
            "CUDA device graph identity is only partially present"
        )
    graph_id = args[graph_key]
    graph_node_id = args[node_key]
    if (
        not isinstance(graph_id, int)
        or isinstance(graph_id, bool)
        or graph_id < 0
        or not isinstance(graph_node_id, int)
        or isinstance(graph_node_id, bool)
        or graph_node_id < 0
    ):
        raise ChromeTraceValidationError(
            "CUDA device graph identity is invalid"
        )
    if graph_id == 0 and graph_node_id == 0:
        return None, None
    if graph_id > 0 and graph_node_id > 0:
        return graph_id, graph_node_id
    raise ChromeTraceValidationError(
        "CUDA device graph identity mixes sentinel and positive values"
    )


def normalize_kernel_family(name: str) -> str | None:
    """Return a frozen Flash forward family without matching templates."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("kernel name must be a non-empty string")
    if "pytorch_flash::flash_fwd_splitkv" in name:
        return FLASH_SPLIT_KV_FAMILY
    if FLASH_FORWARD_FAMILY in name:
        return FLASH_FORWARD_FAMILY
    return None


def classify_device_event(category: str, name: str) -> tuple[str, str | None]:
    """Classify a CUDA activity without high-level ATen child names."""

    if category not in DEVICE_EVENT_CATEGORIES:
        raise ValueError("unsupported CUDA device event category")
    family = normalize_kernel_family(name)
    if family is not None:
        return "flash_attention", family
    if category == "gpu_memcpy":
        return "device_copy_candidate", None
    if category == "gpu_memset":
        return "device_memset", None
    lowered = name.casefold().replace("-", "_")
    if "repeat_interleave" in lowered or "repeat_kv" in lowered:
        return "repeat_materialization", None
    if "repeat" in lowered:
        return "repeat_materialization", None
    if "expand" in lowered and (
        "copy" in lowered or "kernel" in lowered or "material" in lowered
    ):
        return "expand_materialization", None
    if "transpose" in lowered and ("copy" in lowered or "kernel" in lowered):
        return "transpose_copy_candidate", None
    if "copy" in lowered or "memcpy" in lowered:
        return "copy_candidate", None
    return "unknown_kernel", None


@dataclass(frozen=True, slots=True)
class CUDADeviceEvent:
    """Parsed device event with ordering but no trace timing fields."""

    order: int
    category: str
    name: str
    stream: int
    correlation_id: int
    external_id: int | None
    device: int | None
    context: int | None
    classification: str
    kernel_family: str | None
    copy_bytes: int | None = None
    copy_direction: str | None = None
    memory_bytes: int | None = None
    memory_role: str | None = None
    graph_id: int | None = None
    graph_node_id: int | None = None

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("device event order must be nonnegative")
        if self.category not in DEVICE_EVENT_CATEGORIES:
            raise ValueError("device event category is unsupported")
        if not self.name.strip():
            raise ValueError("device event name must be non-empty")
        if (
            not isinstance(self.stream, int)
            or isinstance(self.stream, bool)
            or self.stream < 0
            or not isinstance(self.correlation_id, int)
            or isinstance(self.correlation_id, bool)
            or self.correlation_id <= 0
        ):
            raise ValueError("device event identifiers are invalid")
        if self.external_id is not None and (
            not isinstance(self.external_id, int)
            or isinstance(self.external_id, bool)
            or self.external_id <= 0
        ):
            raise ValueError("device event external ID is invalid")
        if self.device is not None and self.device < 0:
            raise ValueError("device event device is invalid")
        if self.context is not None and self.context < 0:
            raise ValueError("device event context is invalid")
        if self.classification not in _EVENT_CLASSIFICATIONS:
            raise ValueError("device event classification is unsupported")
        classified_as_flash = self.classification == "flash_attention"
        if classified_as_flash != (self.kernel_family is not None):
            raise ValueError("Flash classification and kernel family differ")
        is_copy = self.classification in COPY_CANDIDATE_CLASSIFICATIONS
        if self.copy_bytes is not None and (
            not is_copy
            or not isinstance(self.copy_bytes, int)
            or isinstance(self.copy_bytes, bool)
            or self.copy_bytes <= 0
        ):
            raise ValueError("device copy byte evidence is invalid")
        if self.copy_direction is not None and (
            not is_copy or not self.copy_direction.strip()
        ):
            raise ValueError("device copy direction evidence is invalid")
        is_memset = self.classification == "device_memset"
        if self.memory_bytes is not None and (
            not is_memset
            or not isinstance(self.memory_bytes, int)
            or isinstance(self.memory_bytes, bool)
            or self.memory_bytes <= 0
        ):
            raise ValueError("device memset byte evidence is invalid")
        if self.memory_role is not None and (
            not is_memset or not self.memory_role.strip()
        ):
            raise ValueError("device memset role evidence is invalid")
        if (self.graph_id is None) != (self.graph_node_id is None):
            raise ValueError("device graph and graph-node IDs must coexist")
        for value in (self.graph_id, self.graph_node_id):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError("device graph identifier is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "category": self.category,
            "name": self.name,
            "stream": self.stream,
            "correlation_id": self.correlation_id,
            "external_id": self.external_id,
            "device": self.device,
            "context": self.context,
            "classification": self.classification,
            "kernel_family": self.kernel_family,
            "copy_bytes": self.copy_bytes,
            "copy_direction": self.copy_direction,
            "memory_bytes": self.memory_bytes,
            "memory_role": self.memory_role,
            "graph_id": self.graph_id,
            "graph_node_id": self.graph_node_id,
        }


@dataclass(frozen=True, slots=True)
class _TraceEventCandidate:
    timestamp: float
    duration: float
    original_index: int
    category: str
    name: str
    stream: int
    correlation_id: int
    external_id: int | None
    device: int | None
    context: int | None
    classification: str
    kernel_family: str | None
    copy_bytes: int | None
    copy_direction: str | None
    memory_bytes: int | None
    memory_role: str | None
    graph_id: int | None
    graph_node_id: int | None


def _trace_event_list(raw: bytes) -> list[dict[str, object]]:
    payload = _strict_json_loads(raw)
    if not isinstance(payload, dict):
        raise ChromeTraceValidationError("Chrome trace root must be an object")
    trace_events = payload.get("traceEvents")
    if not isinstance(trace_events, list):
        raise ChromeTraceValidationError("Chrome trace lacks traceEvents")
    if any(not isinstance(item, dict) for item in trace_events):
        raise ChromeTraceValidationError("Chrome trace event must be an object")
    return trace_events


def _finite_trace_number(event: Mapping[str, object], key: str) -> float:
    value = event.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ChromeTraceValidationError(
            f"Chrome trace event lacks finite {key!r}"
        )
    return float(value)


def _complete_interval(event: Mapping[str, object]) -> tuple[float, float]:
    if event.get("ph") != "X":
        raise ChromeTraceValidationError("scoped trace event must be complete")
    start = _finite_trace_number(event, "ts")
    duration = _finite_trace_number(event, "dur")
    if duration < 0.0:
        raise ChromeTraceValidationError("trace event duration must be nonnegative")
    return start, start + duration


def _event_args(event: Mapping[str, object]) -> Mapping[str, object]:
    args = event.get("args")
    if not isinstance(args, dict):
        raise ChromeTraceValidationError("Chrome trace event lacks arguments")
    return args


def _trace_integer(event: Mapping[str, object], key: str) -> int:
    value = event.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChromeTraceValidationError(
            f"Chrome trace event lacks integer {key!r}"
        )
    return value


def _interval_contains(
    outer: tuple[float, float],
    inner: tuple[float, float],
) -> bool:
    tolerance = 0.01
    return bool(
        inner[0] + tolerance >= outer[0]
        and inner[1] <= outer[1] + tolerance
    )


def _intervals_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    tolerance = 0.01
    return bool(
        left[0] < right[1] - tolerance
        and right[0] < left[1] - tolerance
    )


def _reject_unrecognized_device_like_in_marker_events(
    trace_events: list[dict[str, object]],
    *,
    gpu_marker: Mapping[str, object],
    isolated_host_interval: tuple[float, float],
) -> None:
    """Fail closed on device-looking activity hidden under a new category.

    PyTorch profiler category spelling is not a proof boundary.  Known host
    linkage categories remain allowed, but a complete event anywhere in the
    isolated host launch or its asynchronous GPU annotation whose category
    looks device-like (or carries an explicit device/stream/context tuple)
    cannot be ignored merely because its category is unfamiliar.  Scoping only
    to either interval would miss preceding host-scoped activity or GPU work
    which completes after the host launch returns.
    """

    known_non_device_categories = frozenset(
        {
            "cpu_op",
            "cuda_runtime",
            "user_annotation",
            "gpu_user_annotation",
        }
    )
    explicit_device_argument_keys = frozenset(
        {"device", "stream", "context", "graph id", "graph node id"}
    )
    gpu_interval = _complete_interval(gpu_marker)
    for event in trace_events:
        if event is gpu_marker:
            continue
        category = event.get("cat")
        if not isinstance(category, str):
            continue
        if category in DEVICE_EVENT_CATEGORIES or category in (
            known_non_device_categories
        ):
            continue
        if event.get("ph") != "X":
            continue
        try:
            interval = _complete_interval(event)
        except ChromeTraceValidationError:
            continue
        if not (
            _intervals_overlap(isolated_host_interval, interval)
            or _intervals_overlap(gpu_interval, interval)
        ):
            continue
        args = event.get("args")
        argument_keys = frozenset(args) if isinstance(args, dict) else frozenset()
        looks_device_like = bool(
            _DEVICE_LIKE_CATEGORY_RE.search(category)
            or (
                {"stream", "context"}.issubset(argument_keys)
                and bool(argument_keys & explicit_device_argument_keys)
            )
        )
        if looks_device_like:
            raise ChromeTraceValidationError(
                "isolated device-like event uses an unrecognized category"
            )


def _copy_metadata(
    *,
    classification: str,
    name: str,
    args: Mapping[str, object],
) -> tuple[int | None, str | None]:
    if classification not in COPY_CANDIDATE_CLASSIFICATIONS:
        return None, None
    size_values: list[int] = []
    for key in ("bytes", "Bytes", "size", "Size"):
        value = args.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ChromeTraceValidationError("device copy byte count is invalid")
        size_values.append(value)
    if len(set(size_values)) > 1:
        raise ChromeTraceValidationError("device copy byte counts disagree")
    direction_values: list[str] = []
    for key in (
        "copy direction",
        "Copy direction",
        "copy kind",
        "Copy kind",
        "Memcpy type",
    ):
        value = args.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ChromeTraceValidationError("device copy direction is invalid")
        direction_values.append(value.strip())
    if len(set(direction_values)) > 1:
        raise ChromeTraceValidationError("device copy directions disagree")
    if direction_values:
        direction = direction_values[0]
    else:
        match = re.search(r"\b([DH])to([DH])\b", name, flags=re.IGNORECASE)
        direction = None if match is None else match.group(0)
    size = None if not size_values else size_values[0]
    return size, direction


def _memset_metadata(
    *,
    classification: str,
    args: Mapping[str, object],
) -> tuple[int | None, str | None]:
    if classification != "device_memset":
        return None, None
    sizes: list[int] = []
    for key in ("bytes", "Bytes", "size", "Size"):
        value = args.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ChromeTraceValidationError("device memset byte count is invalid")
        sizes.append(value)
    if len(set(sizes)) > 1:
        raise ChromeTraceValidationError("device memset byte counts disagree")
    roles: list[str] = []
    for key in ("memory role", "Memory role", "role"):
        value = args.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ChromeTraceValidationError("device memset role is invalid")
        roles.append(value.strip())
    if len(set(roles)) > 1:
        raise ChromeTraceValidationError("device memset roles disagree")
    return (None if not sizes else sizes[0], None if not roles else roles[0])


def _parse_device_candidates(
    trace_events: list[dict[str, object]],
    *,
    require_external_id: bool = True,
) -> tuple[_TraceEventCandidate, ...]:
    if not isinstance(require_external_id, bool):
        raise TypeError("external-ID requirement must be boolean")
    parsed: list[_TraceEventCandidate] = []
    for original_index, raw_event in enumerate(trace_events):
        category = raw_event.get("cat")
        if not isinstance(category, str) or category not in DEVICE_EVENT_CATEGORIES:
            continue
        interval = _complete_interval(raw_event)
        name = raw_event.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ChromeTraceValidationError("CUDA device event lacks a name")
        args = _event_args(raw_event)
        classification, family = classify_device_event(category, name)
        copy_bytes, copy_direction = _copy_metadata(
            classification=classification,
            name=name,
            args=args,
        )
        memory_bytes, memory_role = _memset_metadata(
            classification=classification,
            args=args,
        )
        graph_id, graph_node_id = _optional_graph_identity(args)
        parsed.append(
            _TraceEventCandidate(
                timestamp=interval[0],
                duration=interval[1] - interval[0],
                original_index=original_index,
                category=category,
                name=name,
                stream=_required_trace_integer(args, "stream", positive=False),
                correlation_id=_required_trace_integer(
                    args, "correlation", positive=True
                ),
                external_id=(
                    _required_trace_integer(args, "External id", positive=True)
                    if require_external_id
                    else _optional_positive_trace_integer(args, "External id")
                ),
                device=_optional_trace_integer(args, "device"),
                context=_optional_trace_integer(args, "context"),
                classification=classification,
                kernel_family=family,
                copy_bytes=copy_bytes,
                copy_direction=copy_direction,
                memory_bytes=memory_bytes,
                memory_role=memory_role,
                graph_id=graph_id,
                graph_node_id=graph_node_id,
            )
        )
    return tuple(parsed)


def _canonical_device_events(
    parsed: tuple[_TraceEventCandidate, ...],
) -> tuple[CUDADeviceEvent, ...]:
    ordered = sorted(parsed, key=lambda item: (item.timestamp, item.original_index))
    return tuple(
        CUDADeviceEvent(
            order=order,
            category=item.category,
            name=item.name,
            stream=item.stream,
            correlation_id=item.correlation_id,
            external_id=item.external_id,
            device=item.device,
            context=item.context,
            classification=item.classification,
            kernel_family=item.kernel_family,
            copy_bytes=item.copy_bytes,
            copy_direction=item.copy_direction,
            memory_bytes=item.memory_bytes,
            memory_role=item.memory_role,
            graph_id=item.graph_id,
            graph_node_id=item.graph_node_id,
        )
        for order, item in enumerate(ordered)
    )


def parse_chrome_cuda_events(raw: bytes) -> tuple[CUDADeviceEvent, ...]:
    """Strictly parse CUDA events from untouched Chrome JSON bytes.

    Device timestamps establish event order only. Neither timestamps nor
    durations are retained in the returned evidence.
    """

    return _canonical_device_events(_parse_device_candidates(_trace_event_list(raw)))


@dataclass(frozen=True, slots=True)
class TraceScopeEvidence:
    """Marker-to-CPU-to-runtime-to-device linkage with timing removed."""

    marker: str
    marker_external_id: int
    cpu_process_id: int
    cpu_thread_id: int
    sdpa_external_id: int
    nested_cpu_external_ids: tuple[int, ...]
    runtime_correlations: tuple[int, ...]
    gpu_stream: int

    def __post_init__(self) -> None:
        if not self.marker.strip():
            raise ValueError("trace scope marker is absent")
        for name in (
            "marker_external_id",
            "cpu_process_id",
            "cpu_thread_id",
            "sdpa_external_id",
        ):
            _positive_integer(getattr(self, name), name)
        if self.gpu_stream < 0:
            raise ValueError("trace scope GPU stream is invalid")
        if (
            not self.nested_cpu_external_ids
            or self.sdpa_external_id not in self.nested_cpu_external_ids
            or self.nested_cpu_external_ids
            != tuple(sorted(set(self.nested_cpu_external_ids)))
        ):
            raise ValueError("trace scope CPU external IDs are not canonical")
        if (
            not self.runtime_correlations
            or self.runtime_correlations
            != tuple(sorted(set(self.runtime_correlations)))
            or any(value <= 0 for value in self.runtime_correlations)
        ):
            raise ValueError("trace scope correlations are not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "marker_external_id": self.marker_external_id,
            "cpu_process_id": self.cpu_process_id,
            "cpu_thread_id": self.cpu_thread_id,
            "sdpa_external_id": self.sdpa_external_id,
            "nested_cpu_external_ids": list(self.nested_cpu_external_ids),
            "runtime_correlations": list(self.runtime_correlations),
            "gpu_stream": self.gpu_stream,
            "timestamps_retained": False,
            "durations_retained": False,
        }


@dataclass(frozen=True, slots=True)
class CUDAGraphTraceScopeEvidence:
    """Marker-to-cudaGraphLaunch-to-device linkage with timing removed."""

    marker: str
    marker_external_id: int
    cpu_process_id: int
    cpu_thread_id: int
    graph_launch_external_id: int | None
    runtime_correlations: tuple[int, ...]
    gpu_stream: int
    graph_id: int
    graph_node_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.marker.strip():
            raise ValueError("graph trace scope marker is absent")
        for name in (
            "marker_external_id",
            "cpu_process_id",
            "cpu_thread_id",
        ):
            _positive_integer(getattr(self, name), name)
        if self.graph_launch_external_id is not None:
            _positive_integer(
                self.graph_launch_external_id,
                "graph_launch_external_id",
            )
        if self.gpu_stream < 0:
            raise ValueError("graph trace scope GPU stream is invalid")
        _positive_integer(self.graph_id, "graph_id")
        if (
            not self.graph_node_ids
            or len(self.graph_node_ids) != len(set(self.graph_node_ids))
        ):
            raise ValueError("graph trace node IDs are absent or duplicated")
        for node_id in self.graph_node_ids:
            _positive_integer(node_id, "graph_node_id")
        if len(self.runtime_correlations) != 1:
            raise ValueError("graph trace requires one launch correlation")
        correlation = self.runtime_correlations[0]
        if (
            not isinstance(correlation, int)
            or isinstance(correlation, bool)
            or correlation <= 0
        ):
            raise ValueError("graph trace launch correlation is invalid")

    @property
    def nested_cpu_external_ids(self) -> tuple[int, ...]:
        if self.graph_launch_external_id is None:
            return ()
        return (self.graph_launch_external_id,)

    @property
    def external_id_linkage(self) -> str:
        return (
            "absent_in_raw"
            if self.graph_launch_external_id is None
            else "present_and_matched"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "marker_external_id": self.marker_external_id,
            "cpu_process_id": self.cpu_process_id,
            "cpu_thread_id": self.cpu_thread_id,
            "dispatch_root": "cudaGraphLaunch",
            "graph_launch_external_id": self.graph_launch_external_id,
            "external_id_linkage": self.external_id_linkage,
            "nested_cpu_external_ids": list(self.nested_cpu_external_ids),
            "runtime_correlations": list(self.runtime_correlations),
            "gpu_stream": self.gpu_stream,
            "graph_id": self.graph_id,
            "graph_node_ids": list(self.graph_node_ids),
            "timestamps_retained": False,
            "durations_retained": False,
        }


@dataclass(frozen=True, slots=True)
class ScopedCUDAActivities:
    scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence
    device_events: tuple[CUDADeviceEvent, ...]

    def __post_init__(self) -> None:
        if not self.device_events:
            raise ValueError("scoped CUDA activity is absent")
        if any(event.stream != self.scope.gpu_stream for event in self.device_events):
            raise ValueError("scoped CUDA event stream differs from marker")


def _is_device_producing_runtime(name: str) -> bool:
    lowered = name.casefold()
    return "launch" in lowered or "memcpy" in lowered or "memset" in lowered


def parse_scoped_chrome_cuda_events(
    raw: bytes,
    *,
    marker: str,
    require_kernel_launch_runtime: bool = False,
) -> ScopedCUDAActivities:
    """Require an unambiguous marker/CPU/runtime/device correlation chain."""

    if not isinstance(marker, str) or not marker.strip():
        raise ValueError("dispatch trace marker must be non-empty")
    if not isinstance(require_kernel_launch_runtime, bool):
        raise TypeError("kernel-launch correlation requirement must be boolean")
    trace_events = _trace_event_list(raw)
    host_markers = [
        event
        for event in trace_events
        if event.get("cat") == "user_annotation" and event.get("name") == marker
    ]
    gpu_markers = [
        event
        for event in trace_events
        if event.get("cat") == "gpu_user_annotation"
        and event.get("name") == marker
    ]
    if len(host_markers) != 1 or len(gpu_markers) != 1:
        raise ChromeTraceValidationError(
            "dispatch marker does not have one host/GPU annotation pair"
        )
    host_marker = host_markers[0]
    gpu_marker = gpu_markers[0]
    host_interval = _complete_interval(host_marker)
    gpu_interval = _complete_interval(gpu_marker)
    host_args = _event_args(host_marker)
    gpu_args = _event_args(gpu_marker)
    _reject_unrecognized_device_like_in_marker_events(
        trace_events,
        gpu_marker=gpu_marker,
        isolated_host_interval=host_interval,
    )
    marker_external_id = _required_trace_integer(
        host_args, "External id", positive=True
    )
    if (
        _required_trace_integer(gpu_args, "External id", positive=True)
        != marker_external_id
    ):
        raise ChromeTraceValidationError("host/GPU marker external IDs differ")
    cpu_pid = _trace_integer(host_marker, "pid")
    cpu_tid = _trace_integer(host_marker, "tid")
    gpu_stream = _trace_integer(gpu_marker, "tid")
    root_sdpa = [
        event
        for event in trace_events
        if event.get("cat") == "cpu_op"
        and event.get("name") == "aten::scaled_dot_product_attention"
        and event.get("pid") == cpu_pid
        and event.get("tid") == cpu_tid
        and _interval_contains(host_interval, _complete_interval(event))
    ]
    if len(root_sdpa) != 1:
        raise ChromeTraceValidationError(
            "dispatch marker does not contain exactly one root SDPA CPU op"
        )
    sdpa_interval = _complete_interval(root_sdpa[0])
    sdpa_external_id = _required_trace_integer(
        _event_args(root_sdpa[0]), "External id", positive=True
    )
    nested_cpu_ids: list[int] = []
    for event in trace_events:
        if (
            event.get("cat") == "cpu_op"
            and event.get("pid") == cpu_pid
            and event.get("tid") == cpu_tid
            and _interval_contains(sdpa_interval, _complete_interval(event))
        ):
            nested_cpu_ids.append(
                _required_trace_integer(
                    _event_args(event), "External id", positive=True
                )
            )
    if len(nested_cpu_ids) != len(set(nested_cpu_ids)):
        raise ChromeTraceValidationError("nested CPU external IDs are ambiguous")
    nested_ids = frozenset(nested_cpu_ids)
    runtime_by_link: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    producing_runtime_links: set[tuple[int, int]] = set()
    for event in trace_events:
        if (
            event.get("cat") != "cuda_runtime"
            or event.get("pid") != cpu_pid
            or event.get("tid") != cpu_tid
            or not _interval_contains(sdpa_interval, _complete_interval(event))
        ):
            continue
        args = _event_args(event)
        external_id = args.get("External id")
        if external_id not in nested_ids:
            continue
        if not isinstance(external_id, int) or isinstance(external_id, bool):
            raise ChromeTraceValidationError("runtime external ID is invalid")
        correlation = _required_trace_integer(args, "correlation", positive=True)
        link = (external_id, correlation)
        runtime_by_link.setdefault(link, []).append(event)
        name = event.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ChromeTraceValidationError("CUDA runtime event lacks a name")
        if _is_device_producing_runtime(name):
            producing_runtime_links.add(link)
    candidates = _parse_device_candidates(trace_events)
    selected: list[_TraceEventCandidate] = []
    for candidate in candidates:
        interval = (
            candidate.timestamp,
            candidate.timestamp + candidate.duration,
        )
        linked_cpu = candidate.external_id in nested_ids
        within_gpu_marker = _interval_contains(gpu_interval, interval)
        overlaps_host_marker = _intervals_overlap(host_interval, interval)
        on_marker_stream = candidate.stream == gpu_stream
        link = (candidate.external_id, candidate.correlation_id)
        runtime_matches = runtime_by_link.get(link, [])
        related = linked_cpu or within_gpu_marker or overlaps_host_marker
        fully_linked = bool(
            linked_cpu
            and within_gpu_marker
            and on_marker_stream
            and len(runtime_matches) == 1
            and link in producing_runtime_links
        )
        if related and not fully_linked:
            raise ChromeTraceValidationError(
                "CUDA device event is ambiguously or incompletely marker-linked"
            )
        if fully_linked:
            runtime_name = runtime_matches[0].get("name")
            if (
                require_kernel_launch_runtime
                and candidate.category == "kernel"
                and (
                    not isinstance(runtime_name, str)
                    or "launch" not in runtime_name.casefold()
                )
            ):
                raise ChromeTraceValidationError(
                    "CUDA kernel is not linked to a launch runtime event"
                )
            selected.append(candidate)
    if not selected:
        raise ChromeTraceValidationError("marked SDPA has no linked CUDA activity")
    selected_links = {
        (candidate.external_id, candidate.correlation_id)
        for candidate in selected
    }
    if producing_runtime_links != selected_links:
        raise ChromeTraceValidationError(
            "marked SDPA runtime/device correlations are unmatched"
        )
    correlations = tuple(sorted(candidate.correlation_id for candidate in selected))
    scope = TraceScopeEvidence(
        marker=marker,
        marker_external_id=marker_external_id,
        cpu_process_id=cpu_pid,
        cpu_thread_id=cpu_tid,
        sdpa_external_id=sdpa_external_id,
        nested_cpu_external_ids=tuple(sorted(nested_ids)),
        runtime_correlations=correlations,
        gpu_stream=gpu_stream,
    )
    return ScopedCUDAActivities(
        scope=scope,
        device_events=_canonical_device_events(tuple(selected)),
    )


def parse_scoped_chrome_cuda_graph_events(
    raw: bytes,
    *,
    marker: str,
) -> ScopedCUDAActivities:
    """Require one marker/cudaGraphLaunch/device correlation chain."""

    if not isinstance(marker, str) or not marker.strip():
        raise ValueError("dispatch trace marker must be non-empty")
    trace_events = _trace_event_list(raw)
    host_markers = [
        event
        for event in trace_events
        if event.get("cat") == "user_annotation" and event.get("name") == marker
    ]
    gpu_markers = [
        event
        for event in trace_events
        if event.get("cat") == "gpu_user_annotation"
        and event.get("name") == marker
    ]
    if len(host_markers) != 1 or len(gpu_markers) != 1:
        raise ChromeTraceValidationError(
            "graph marker does not have one host/GPU annotation pair"
        )
    host_marker = host_markers[0]
    gpu_marker = gpu_markers[0]
    host_interval = _complete_interval(host_marker)
    gpu_interval = _complete_interval(gpu_marker)
    host_args = _event_args(host_marker)
    gpu_args = _event_args(gpu_marker)
    _reject_unrecognized_device_like_in_marker_events(
        trace_events,
        gpu_marker=gpu_marker,
        isolated_host_interval=host_interval,
    )
    marker_external_id = _required_trace_integer(
        host_args, "External id", positive=True
    )
    if (
        _required_trace_integer(gpu_args, "External id", positive=True)
        != marker_external_id
    ):
        raise ChromeTraceValidationError("graph host/GPU marker IDs differ")
    cpu_pid = _trace_integer(host_marker, "pid")
    cpu_tid = _trace_integer(host_marker, "tid")
    gpu_stream = _trace_integer(gpu_marker, "tid")
    launches = [
        event
        for event in trace_events
        if event.get("cat") == "cuda_runtime"
        and event.get("pid") == cpu_pid
        and event.get("tid") == cpu_tid
        and isinstance(event.get("name"), str)
        and "cudagraphlaunch" in str(event.get("name")).casefold()
        and _interval_contains(host_interval, _complete_interval(event))
    ]
    if len(launches) != 1:
        raise ChromeTraceValidationError(
            "graph marker does not contain exactly one cudaGraphLaunch"
        )
    launch_args = _event_args(launches[0])
    launch_external_id = _optional_positive_trace_integer(
        launch_args,
        "External id",
    )
    launch_correlation = _required_trace_integer(
        launch_args, "correlation", positive=True
    )
    launch_interval = _complete_interval(launches[0])
    if launch_interval[1] > gpu_interval[0] + 0.01:
        raise ChromeTraceValidationError(
            "cudaGraphLaunch does not precede its GPU marker"
        )
    candidates = _parse_device_candidates(
        trace_events,
        require_external_id=False,
    )
    selected: list[_TraceEventCandidate] = []
    for candidate in candidates:
        interval = (candidate.timestamp, candidate.timestamp + candidate.duration)
        within_gpu_marker = _interval_contains(gpu_interval, interval)
        overlaps_host_marker = _intervals_overlap(host_interval, interval)
        if not within_gpu_marker:
            if candidate.correlation_id == launch_correlation:
                raise ChromeTraceValidationError(
                    "graph-launch-correlated device event is outside the GPU marker"
                )
            if overlaps_host_marker:
                raise ChromeTraceValidationError(
                    "host-overlapping graph device event is outside the GPU marker"
                )
            continue
        if candidate.correlation_id != launch_correlation:
            raise ChromeTraceValidationError(
                "in-marker graph device event has the wrong correlation"
            )
        if candidate.stream != gpu_stream:
            raise ChromeTraceValidationError(
                "in-marker graph device event has the wrong stream"
            )
        if candidate.external_id != launch_external_id:
            raise ChromeTraceValidationError(
                "graph launch/device External-ID presence or value differs"
            )
        if candidate.graph_id is None or candidate.graph_node_id is None:
            raise ChromeTraceValidationError(
                "graph device event lacks graph or graph-node identity"
            )
        selected.append(candidate)
    if not selected:
        raise ChromeTraceValidationError("cudaGraphLaunch has no linked CUDA activity")
    canonical_events = _canonical_device_events(tuple(selected))
    graph_ids = {event.graph_id for event in canonical_events}
    if len(graph_ids) != 1:
        raise ChromeTraceValidationError(
            "in-marker graph device events have mixed graph IDs"
        )
    graph_node_ids = tuple(
        event.graph_node_id
        for event in canonical_events
        if event.graph_node_id is not None
    )
    if len(graph_node_ids) != len(set(graph_node_ids)):
        raise ChromeTraceValidationError(
            "in-marker graph device events have duplicate graph-node IDs"
        )
    graph_id = next(iter(graph_ids))
    if graph_id is None:  # pragma: no cover - guarded above
        raise AssertionError("validated graph identity disappeared")
    scope = CUDAGraphTraceScopeEvidence(
        marker=marker,
        marker_external_id=marker_external_id,
        cpu_process_id=cpu_pid,
        cpu_thread_id=cpu_tid,
        graph_launch_external_id=launch_external_id,
        runtime_correlations=(launch_correlation,),
        gpu_stream=gpu_stream,
        graph_id=graph_id,
        graph_node_ids=graph_node_ids,
    )
    return ScopedCUDAActivities(
        scope=scope,
        device_events=canonical_events,
    )


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class KVByteEvidence:
    """Exact native and hypothetical query-head-expanded K/V byte counts."""

    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    context_length: int
    head_dim: int
    dtype_bytes: int
    native_kv_bytes: int
    expanded_kv_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "context_length": self.context_length,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
            "native_kv_bytes": self.native_kv_bytes,
            "expanded_kv_bytes": self.expanded_kv_bytes,
        }


def calculate_kv_bytes(
    *,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    context_length: int,
    head_dim: int,
    dtype_bytes: int,
) -> KVByteEvidence:
    """Calculate both preregistered K+V formulas exactly."""

    values = {
        "batch_size": _positive_integer(batch_size, "batch_size"),
        "num_query_heads": _positive_integer(
            num_query_heads, "num_query_heads"
        ),
        "num_kv_heads": _positive_integer(num_kv_heads, "num_kv_heads"),
        "context_length": _positive_integer(context_length, "context_length"),
        "head_dim": _positive_integer(head_dim, "head_dim"),
        "dtype_bytes": _positive_integer(dtype_bytes, "dtype_bytes"),
    }
    if values["num_query_heads"] % values["num_kv_heads"] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    common = (
        2
        * values["batch_size"]
        * values["context_length"]
        * values["head_dim"]
        * values["dtype_bytes"]
    )
    return KVByteEvidence(
        **values,
        native_kv_bytes=common * values["num_kv_heads"],
        expanded_kv_bytes=common * values["num_query_heads"],
    )


@dataclass(frozen=True, slots=True)
class RawTraceArtifact:
    """Identity of an untouched dispatch trace, with no timing metric."""

    relative_path: str
    sha256: str
    size_bytes: int
    execution_mode: str = EAGER_EXECUTION_MODE

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError("raw trace artifact path must be safe and relative")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("raw trace SHA-256 is invalid")
        if not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("raw trace size must be positive")
        if self.execution_mode not in DISPATCH_EXECUTION_MODES:
            raise ValueError("raw trace execution mode is unsupported")

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        relative_path: str,
        execution_mode: str = EAGER_EXECUTION_MODE,
    ) -> RawTraceArtifact:
        raw = path.read_bytes()
        return cls(
            relative_path=relative_path,
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
            execution_mode=execution_mode,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_kind": "dispatch_audit",
            "trace_mechanism": "torch.profiler_cpu_cuda",
            "activities": ["CPU", "CUDA"],
            "profiler_instrumented": True,
            "benchmark_timing_eligible": False,
            "raw_trace_untouched": True,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "execution_mode": self.execution_mode,
            "cuda_graph_capture_contract": (
                {
                    "helper": "kvbench.runtime.cuda_graph.capture_fixed_graph",
                    "dedicated_side_stream": True,
                    "capture_error_mode": "global",
                }
                if self.execution_mode == CUDA_GRAPH_REPLAY_EXECUTION_MODE
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class BackendControlEvidence:
    """Fail-closed Flash forcing, eligibility, and build identity evidence."""

    enabled_backends: tuple[str, ...]
    flash_eligible: bool
    fused_backend_name: str | None
    rejected_control_failed: bool
    rejected_control_error: str | None
    rejected_control_warnings: tuple[str, ...]
    rejected_control_synchronized: bool
    source_build_fingerprint: str
    source_build_verified: bool
    eligibility_diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.source_build_fingerprint):
            raise ValueError("backend source/build fingerprint is invalid")
        if any(not item.strip() for item in self.eligibility_diagnostics):
            raise ValueError("backend diagnostics must be non-empty strings")
        if self.rejected_control_error is not None and not (
            isinstance(self.rejected_control_error, str)
            and self.rejected_control_error.strip()
        ):
            raise ValueError("rejected-control error must be non-empty")
        if any(not item.strip() for item in self.rejected_control_warnings):
            raise ValueError("rejected-control warnings must be non-empty")

    @property
    def passed(self) -> bool:
        return bool(
            self.enabled_backends == ("FLASH_ATTENTION",)
            and self.flash_eligible
            and self.fused_backend_name == "FLASH_ATTENTION"
            and self.rejected_control_failed
            and self.rejected_control_synchronized
            and self.source_build_verified
        )

    @property
    def control_transcript_verified(self) -> bool:
        expected_rejection = bool(
            self.rejected_control_error
            == "No available kernel. Aborting execution."
            and any(
                "Flash attention kernel not used because:" in item
                for item in self.rejected_control_warnings
            )
            and any(
                "Expected query, key and value to all be of dtype: "
                "{Half, BFloat16}." in item
                for item in self.rejected_control_warnings
            )
            and self.rejected_control_synchronized
        )
        if self.rejected_control_failed != expected_rejection:
            return False
        if len(self.eligibility_diagnostics) < 3:
            return False
        enabled_line, eligible_line, choice_line = (
            self.eligibility_diagnostics[-3:]
        )
        if enabled_line != "enabled_backends=" + ",".join(
            self.enabled_backends
        ):
            return False
        if eligible_line != (
            f"can_use_flash_attention={self.flash_eligible}"
        ):
            return False
        prefix = "fused_sdp_choice="
        if not choice_line.startswith(prefix):
            return False
        try:
            choice = int(choice_line.removeprefix(prefix))
        except ValueError:
            return False
        return bool(
            (choice == 1)
            == (self.fused_backend_name == "FLASH_ATTENTION")
        )

    @property
    def control_transcript_sha256(self) -> str:
        raw = json.dumps(
            {
                "eligibility_diagnostics": self.eligibility_diagnostics,
                "enabled_backends": self.enabled_backends,
                "flash_eligible": self.flash_eligible,
                "fused_backend_name": self.fused_backend_name,
                "rejected_control_error": self.rejected_control_error,
                "rejected_control_failed": self.rejected_control_failed,
                "rejected_control_synchronized": (
                    self.rejected_control_synchronized
                ),
                "rejected_control_warnings": self.rejected_control_warnings,
                "source_build_fingerprint": self.source_build_fingerprint,
                "source_build_verified": self.source_build_verified,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled_backends": list(self.enabled_backends),
            "flash_eligible": self.flash_eligible,
            "fused_backend_name": self.fused_backend_name,
            "rejected_control_failed": self.rejected_control_failed,
            "rejected_control_error": self.rejected_control_error,
            "rejected_control_warnings": list(self.rejected_control_warnings),
            "rejected_control_synchronized": self.rejected_control_synchronized,
            "source_build_fingerprint": self.source_build_fingerprint,
            "source_build_verified": self.source_build_verified,
            "eligibility_diagnostics": list(self.eligibility_diagnostics),
            "control_transcript_sha256": self.control_transcript_sha256,
            "control_transcript_verified": self.control_transcript_verified,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class TensorShapeEvidence:
    """Tensor metadata read without copying tensor contents to the host."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    element_size: int
    storage_bytes: int
    storage_offset: int
    is_contiguous: bool

    def __post_init__(self) -> None:
        if not self.shape or len(self.shape) != len(self.stride):
            raise ValueError("tensor shape and stride ranks must match")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.shape
        ):
            raise ValueError("tensor dimensions must be positive integers")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.stride
        ):
            raise ValueError("tensor strides must be nonnegative integers")
        _positive_integer(self.element_size, "element_size")
        _positive_integer(self.storage_bytes, "storage_bytes")
        if (
            not isinstance(self.storage_offset, int)
            or isinstance(self.storage_offset, bool)
            or self.storage_offset < 0
        ):
            raise ValueError("tensor storage offset must be nonnegative")
        if not isinstance(self.is_contiguous, bool):
            raise ValueError("tensor contiguity must be boolean")
        if not self.dtype.strip() or not self.device.strip():
            raise ValueError("tensor dtype and device must be recorded")

    @property
    def logical_bytes(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * self.element_size

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
            "is_contiguous": self.is_contiguous,
        }


@dataclass(frozen=True, slots=True)
class Phase3DispatchPointBinding:
    """Exact shared operation identity for one production dispatch audit."""

    operation_key: Phase3AuditOperationKey

    def __post_init__(self) -> None:
        if type(self.operation_key) is not Phase3AuditOperationKey:
            raise ValueError(
                "dispatch point requires an exact Phase3AuditOperationKey"
            )

    @classmethod
    def create(
        cls,
        *,
        operation_key: Phase3AuditOperationKey,
    ) -> Phase3DispatchPointBinding:
        return cls(operation_key=operation_key)

    @property
    def run_id(self) -> str:
        return self.operation_key.run_id

    @property
    def point_id(self) -> str:
        return self.operation_key.point_id

    @property
    def point_fingerprint(self) -> str:
        return self.operation_key.point_fingerprint

    @property
    def runner_kind(self) -> str:
        return self.operation_key.runner_kind.value

    @property
    def graph_mode(self) -> str:
        return self.operation_key.graph_mode.value

    @property
    def process_replicate(self) -> int:
        return self.operation_key.process_replicate

    @property
    def batch_size(self) -> int:
        return self.operation_key.batch_size

    @property
    def starting_context(self) -> int:
        return self.operation_key.historical_context - self.operation_key.decode_step

    @property
    def active_context(self) -> int:
        return self.operation_key.attended_context

    @property
    def capacity(self) -> int:
        return self.operation_key.capacity

    @property
    def decode_step(self) -> int:
        return self.operation_key.decode_step

    @property
    def operation_fingerprint_sha256(self) -> str:
        return self.operation_key.operation_fingerprint_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_key": self.operation_key.to_dict(),
            "run_id": self.run_id,
            "point_id": self.point_id,
            "point_fingerprint": self.point_fingerprint,
            "runner_kind": self.runner_kind,
            "graph_mode": self.graph_mode,
            "process_replicate": self.process_replicate,
            "batch_size": self.batch_size,
            "starting_context": self.starting_context,
            "active_context": self.active_context,
            "capacity": self.capacity,
            "decode_step": self.decode_step,
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class StaticCacheLayoutEvidence:
    """Declared and checksum-bound native-KV static-cache layout."""

    layout_name: str
    tensor_shape: tuple[int, ...]
    tensor_stride: tuple[int, ...]
    dtype: str
    element_size: int
    device: str
    workspace_bytes: int
    implementation_source_path: str
    implementation_sha256: str
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if self.layout_name != PHASE3_CACHE_LAYOUT_NAME:
            raise ValueError("cache layout name differs from frozen Phase 3 layout")
        if len(self.tensor_shape) != 5 or len(self.tensor_stride) != 5:
            raise ValueError("cache declaration must be rank five")
        if any(value <= 0 for value in self.tensor_shape):
            raise ValueError("cache declaration dimensions must be positive")
        if self.tensor_shape[0] != PHASE3_NUM_LAYERS:
            raise ValueError("cache declaration has the wrong layer count")
        if self.tensor_shape[2] != PHASE3_NUM_KV_HEADS:
            raise ValueError("cache declaration has the wrong backing head count")
        if self.tensor_shape[4] != PHASE3_HEAD_DIM:
            raise ValueError("cache declaration has the wrong head dimension")
        expected_stride = (
            self.tensor_shape[1]
            * self.tensor_shape[2]
            * self.tensor_shape[3]
            * self.tensor_shape[4],
            self.tensor_shape[2] * self.tensor_shape[3] * self.tensor_shape[4],
            self.tensor_shape[3] * self.tensor_shape[4],
            self.tensor_shape[4],
            1,
        )
        if self.tensor_stride != expected_stride:
            raise ValueError("cache declaration strides differ from frozen layout")
        if self.dtype != PHASE3_DTYPE or self.element_size != PHASE3_DTYPE_BYTES:
            raise ValueError("cache declaration must use BF16")
        if not self.device.startswith("cuda:"):
            raise ValueError("cache declaration must identify one CUDA device")
        if (
            not isinstance(self.workspace_bytes, int)
            or isinstance(self.workspace_bytes, bool)
            or self.workspace_bytes < 0
        ):
            raise ValueError("cache workspace bytes must be nonnegative")
        if self.implementation_source_path != REQUIRED_SUT_SOURCES[2]:
            raise ValueError("cache implementation source path differs")
        if not _SHA256_RE.fullmatch(self.implementation_sha256):
            raise ValueError("cache implementation SHA-256 is invalid")
        expected_fingerprint = derive_cache_layout_fingerprint(
            num_layers=self.tensor_shape[0],
            batch_size=self.tensor_shape[1],
            num_kv_heads=self.tensor_shape[2],
            capacity=self.tensor_shape[3],
            head_dim=self.tensor_shape[4],
            device=self.device,
            workspace_bytes=self.workspace_bytes,
            implementation_sha256=self.implementation_sha256,
        )
        if self.layout_fingerprint != expected_fingerprint:
            raise ValueError("cache layout fingerprint differs from declaration")

    @property
    def batch_size(self) -> int:
        return self.tensor_shape[1]

    @property
    def capacity(self) -> int:
        return self.tensor_shape[3]

    @property
    def single_tensor_storage_bytes(self) -> int:
        elements = 1
        for dimension in self.tensor_shape:
            elements *= dimension
        return elements * self.element_size

    @classmethod
    def create(
        cls,
        *,
        batch_size: int,
        capacity: int,
        device: str,
        workspace_bytes: int,
        implementation_sha256: str,
        layout_fingerprint: str,
    ) -> StaticCacheLayoutEvidence:
        shape = (
            PHASE3_NUM_LAYERS,
            batch_size,
            PHASE3_NUM_KV_HEADS,
            capacity,
            PHASE3_HEAD_DIM,
        )
        stride = (
            batch_size * PHASE3_NUM_KV_HEADS * capacity * PHASE3_HEAD_DIM,
            PHASE3_NUM_KV_HEADS * capacity * PHASE3_HEAD_DIM,
            capacity * PHASE3_HEAD_DIM,
            PHASE3_HEAD_DIM,
            1,
        )
        return cls(
            layout_name=PHASE3_CACHE_LAYOUT_NAME,
            tensor_shape=shape,
            tensor_stride=stride,
            dtype=PHASE3_DTYPE,
            element_size=PHASE3_DTYPE_BYTES,
            device=device,
            workspace_bytes=workspace_bytes,
            implementation_source_path=REQUIRED_SUT_SOURCES[2],
            implementation_sha256=implementation_sha256,
            layout_fingerprint=layout_fingerprint,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PHASE3_CACHE_LAYOUT_SCHEMA,
            "layout_name": self.layout_name,
            "tensor_shape": list(self.tensor_shape),
            "tensor_stride": list(self.tensor_stride),
            "dtype": self.dtype,
            "element_size": self.element_size,
            "device": self.device,
            "workspace_bytes": self.workspace_bytes,
            "implementation_source_path": self.implementation_source_path,
            "implementation_sha256": self.implementation_sha256,
            "layout_fingerprint": self.layout_fingerprint,
            "single_tensor_storage_bytes": self.single_tensor_storage_bytes,
            "native_kv_head_backing": True,
        }


@dataclass(frozen=True, slots=True)
class StaticCacheViewBindingEvidence:
    """Bind active per-layer K/V views to full native-KV cache tensors."""

    layout: StaticCacheLayoutEvidence
    layer_index: int
    active_context: int
    key_backing: TensorShapeEvidence
    value_backing: TensorShapeEvidence
    key_view: TensorShapeEvidence
    value_view: TensorShapeEvidence
    key_backing_storage_ptr: int
    value_backing_storage_ptr: int
    key_view_storage_ptr: int
    value_view_storage_ptr: int
    key_view_shares_backing_storage: bool
    value_view_shares_backing_storage: bool
    key_value_backing_storages_distinct: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.layer_index, int)
            or isinstance(self.layer_index, bool)
            or not 0 <= self.layer_index < PHASE3_NUM_LAYERS
        ):
            raise ValueError("cache layer index is invalid")
        _positive_integer(self.active_context, "active_context")
        if self.active_context > self.layout.capacity:
            raise ValueError("active cache view exceeds declared capacity")
        for field in (
            "key_backing_storage_ptr",
            "value_backing_storage_ptr",
            "key_view_storage_ptr",
            "value_view_storage_ptr",
        ):
            _positive_integer(getattr(self, field), field)
        for field in (
            "key_view_shares_backing_storage",
            "value_view_shares_backing_storage",
            "key_value_backing_storages_distinct",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError("cache storage-binding flags must be boolean")
        expected_flags = (
            self.key_view_storage_ptr == self.key_backing_storage_ptr,
            self.value_view_storage_ptr == self.value_backing_storage_ptr,
            self.key_backing_storage_ptr != self.value_backing_storage_ptr,
        )
        observed_flags = (
            self.key_view_shares_backing_storage,
            self.value_view_shares_backing_storage,
            self.key_value_backing_storages_distinct,
        )
        if observed_flags != expected_flags:
            raise ValueError("cache storage-binding flags differ from pointers")

    @property
    def storage_pointer_transcript_sha256(self) -> str:
        transcript = json.dumps(
            {
                "key_backing_storage_ptr": self.key_backing_storage_ptr,
                "key_view_storage_ptr": self.key_view_storage_ptr,
                "value_backing_storage_ptr": self.value_backing_storage_ptr,
                "value_view_storage_ptr": self.value_view_storage_ptr,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(transcript).hexdigest()

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        expected_backing_shape = self.layout.tensor_shape
        expected_backing_stride = self.layout.tensor_stride
        expected_storage_bytes = self.layout.single_tensor_storage_bytes
        expected_view_shape = (
            self.layout.batch_size,
            PHASE3_NUM_KV_HEADS,
            self.active_context,
            PHASE3_HEAD_DIM,
        )
        expected_view_stride = expected_backing_stride[1:]
        expected_offset = self.layer_index * expected_backing_stride[0]
        for label, tensor in (
            ("key_backing", self.key_backing),
            ("value_backing", self.value_backing),
        ):
            if tensor.shape != expected_backing_shape:
                reasons.append(f"{label}_shape_mismatch")
            if tensor.stride != expected_backing_stride:
                reasons.append(f"{label}_stride_mismatch")
            if tensor.storage_bytes != expected_storage_bytes:
                reasons.append(f"{label}_storage_bytes_mismatch")
            if tensor.storage_offset != 0:
                reasons.append(f"{label}_storage_offset_mismatch")
            if not tensor.is_contiguous:
                reasons.append(f"{label}_not_contiguous")
        for label, tensor in (
            ("key_view", self.key_view),
            ("value_view", self.value_view),
        ):
            if tensor.shape != expected_view_shape:
                reasons.append(f"{label}_shape_mismatch")
            if tensor.stride != expected_view_stride:
                reasons.append(f"{label}_stride_mismatch")
            if tensor.storage_bytes != expected_storage_bytes:
                reasons.append(f"{label}_storage_bytes_mismatch")
            if tensor.storage_offset != expected_offset:
                reasons.append(f"{label}_storage_offset_mismatch")
        for label, tensor in (
            ("key_backing", self.key_backing),
            ("value_backing", self.value_backing),
            ("key_view", self.key_view),
            ("value_view", self.value_view),
        ):
            if tensor.dtype != self.layout.dtype:
                reasons.append(f"{label}_dtype_mismatch")
            if tensor.element_size != self.layout.element_size:
                reasons.append(f"{label}_element_size_mismatch")
            if tensor.device != self.layout.device:
                reasons.append(f"{label}_device_mismatch")
        if not self.key_view_shares_backing_storage:
            reasons.append("key_view_not_bound_to_backing")
        if not self.value_view_shares_backing_storage:
            reasons.append("value_view_not_bound_to_backing")
        if not self.key_value_backing_storages_distinct:
            reasons.append("key_value_backing_storage_alias")
        return tuple(reasons)

    @property
    def verified(self) -> bool:
        return not self.failure_reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "layout": self.layout.to_dict(),
            "layer_index": self.layer_index,
            "active_context": self.active_context,
            "key_backing": self.key_backing.to_dict(),
            "value_backing": self.value_backing.to_dict(),
            "key_view": self.key_view.to_dict(),
            "value_view": self.value_view.to_dict(),
            "storage_pointers": {
                "key_backing": self.key_backing_storage_ptr,
                "key_view": self.key_view_storage_ptr,
                "value_backing": self.value_backing_storage_ptr,
                "value_view": self.value_view_storage_ptr,
                "sha256": self.storage_pointer_transcript_sha256,
            },
            "key_view_shares_backing_storage": (
                self.key_view_shares_backing_storage
            ),
            "value_view_shares_backing_storage": (
                self.value_view_shares_backing_storage
            ),
            "key_value_backing_storages_distinct": (
                self.key_value_backing_storages_distinct
            ),
            "active_view_storage_may_exceed_logical_bytes": True,
            "verified": self.verified,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class SourceFindingEvidence:
    """Typed source-audit finding with truthful taxonomy semantics."""

    code: str
    evidence_type: str
    positive_materialization_evidence: bool

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("source finding code is absent")
        if self.evidence_type not in {
            "direct_gqa_replication_path",
            "source_contract_violation",
        }:
            raise ValueError("source finding evidence type is unsupported")
        expected_type = (
            "direct_gqa_replication_path"
            if self.positive_materialization_evidence
            else "source_contract_violation"
        )
        if self.evidence_type != expected_type:
            raise ValueError("source finding taxonomy is inconsistent")

    @classmethod
    def from_code(
        cls,
        code: str,
        *,
        positive_materialization_evidence: bool,
    ) -> SourceFindingEvidence:
        return cls(
            code=code,
            evidence_type=(
                "direct_gqa_replication_path"
                if positive_materialization_evidence
                else "source_contract_violation"
            ),
            positive_materialization_evidence=(
                positive_materialization_evidence
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "evidence_type": self.evidence_type,
            "positive_materialization_evidence": (
                self.positive_materialization_evidence
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceFileEvidence:
    """Identity and forbidden-path findings for one selected SUT source."""

    relative_path: str
    sha256: str
    findings: tuple[str, ...]
    direct_replication_findings: tuple[str, ...] = ()
    selected_function_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError("source audit path must be safe and relative")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("source audit SHA-256 is invalid")
        if any(not finding.strip() for finding in self.findings):
            raise ValueError("source audit findings must be non-empty strings")
        if any(not item.strip() for item in self.selected_function_paths):
            raise ValueError("selected source function path is absent")
        if (
            self.findings != tuple(dict.fromkeys(self.findings))
            or self.direct_replication_findings
            != tuple(dict.fromkeys(self.direct_replication_findings))
            or not set(self.direct_replication_findings).issubset(self.findings)
            or self.direct_replication_findings
            != tuple(
                finding
                for finding in self.findings
                if finding in frozenset(self.direct_replication_findings)
            )
            or self.selected_function_paths
            != tuple(dict.fromkeys(self.selected_function_paths))
        ):
            raise ValueError("source audit findings are not canonical")

    @property
    def passed(self) -> bool:
        expected_paths = _SELECTED_SOURCE_FUNCTION_PATHS.get(self.relative_path)
        return bool(
            not self.findings
            and (
                expected_paths is None
                or self.selected_function_paths == expected_paths
            )
        )

    @property
    def selected_execution_path_verified(self) -> bool:
        expected_paths = _SELECTED_SOURCE_FUNCTION_PATHS.get(self.relative_path)
        return bool(
            expected_paths is None
            or self.selected_function_paths == expected_paths
        )

    @property
    def typed_findings(self) -> tuple[SourceFindingEvidence, ...]:
        direct = frozenset(self.direct_replication_findings)
        return tuple(
            SourceFindingEvidence.from_code(
                item,
                positive_materialization_evidence=item in direct,
            )
            for item in self.findings
        )

    @property
    def positive_materialization_findings(self) -> tuple[str, ...]:
        return tuple(
            finding.code
            for finding in self.typed_findings
            if finding.positive_materialization_evidence
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "findings": list(self.findings),
            "direct_replication_findings": list(
                self.direct_replication_findings
            ),
            "selected_function_paths": list(self.selected_function_paths),
            "selected_execution_path_verified": (
                self.selected_execution_path_verified
            ),
            "typed_findings": [item.to_dict() for item in self.typed_findings],
            "passed": self.passed,
        }


def source_materialization_evidence(
    sources: tuple[SourceFileEvidence, ...],
) -> tuple[str, ...]:
    """Return canonical positive source evidence with file identity."""

    return tuple(
        sorted(
            f"source:{source.relative_path}:{finding}"
            for source in sources
            for finding in source.positive_materialization_findings
        )
    )


@dataclass(frozen=True, slots=True)
class SourceShapeEvidence:
    """Source identity and exact Q/K/V/output metadata for one control."""

    sources: tuple[SourceFileEvidence, ...]
    query: TensorShapeEvidence
    key: TensorShapeEvidence
    value: TensorShapeEvidence
    output: TensorShapeEvidence
    native_kv_storage_verified: bool

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("source audit must retain at least one source")
        paths = tuple(source.relative_path for source in self.sources)
        if len(set(paths)) != len(paths) or paths != tuple(sorted(paths)):
            raise ValueError("source audit paths must be unique and sorted")

    @property
    def source_verified(self) -> bool:
        return all(source.passed for source in self.sources)

    def shape_verified_for(self, control: DispatchControlEvidence) -> bool:
        expected_query = (
            control.batch_size,
            control.num_query_heads,
            control.query_length,
            control.head_dim,
        )
        expected_kv = (
            control.batch_size,
            control.num_kv_heads,
            control.context_length,
            control.head_dim,
        )
        tensors = (self.query, self.key, self.value, self.output)
        return bool(
            self.native_kv_storage_verified
            and self.query.shape == expected_query
            and self.key.shape == expected_kv
            and self.value.shape == expected_kv
            and self.output.shape == expected_query
            and all(tensor.dtype == "torch.bfloat16" for tensor in tensors)
            and len({tensor.device for tensor in tensors}) == 1
            and all(tensor.element_size == 2 for tensor in tensors)
            and all(all(stride > 0 for stride in tensor.stride) for tensor in tensors)
            and self.key.storage_bytes == self.key.logical_bytes
            and self.value.storage_bytes == self.value.logical_bytes
            and self.key.storage_offset == 0
            and self.value.storage_offset == 0
            and self.key.is_contiguous
            and self.value.is_contiguous
        )

    def to_dict(self, *, control: DispatchControlEvidence) -> dict[str, object]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "query": self.query.to_dict(),
            "key": self.key.to_dict(),
            "value": self.value.to_dict(),
            "output": self.output.to_dict(),
            "native_kv_storage_verified": self.native_kv_storage_verified,
            "source_verified": self.source_verified,
            "shape_verified": self.shape_verified_for(control),
        }


@dataclass(frozen=True, slots=True)
class DispatchControlEvidence:
    """One held-constant GQA or MHA direct device control."""

    role: str
    batch_size: int
    context_length: int
    query_length: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    dtype_bytes: int
    is_causal: bool
    warmup_count: int
    backend: BackendControlEvidence
    raw_trace: RawTraceArtifact | None
    trace_scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence | None
    device_events: tuple[CUDADeviceEvent, ...]
    execution_mode: str = EAGER_EXECUTION_MODE

    def __post_init__(self) -> None:
        if self.role not in {"gqa", "mha_control"}:
            raise ValueError("dispatch control role is unsupported")
        for name in (
            "batch_size",
            "context_length",
            "query_length",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "dtype_bytes",
            "warmup_count",
        ):
            _positive_integer(getattr(self, name), name)
        if self.num_query_heads != 32 or self.head_dim != 128:
            raise ValueError("dispatch control differs from frozen HQ/D geometry")
        expected_kv_heads = 8 if self.role == "gqa" else 32
        if self.num_kv_heads != expected_kv_heads:
            raise ValueError("dispatch control differs from frozen KV geometry")
        if self.dtype != "torch.bfloat16" or self.dtype_bytes != 2:
            raise ValueError("dispatch control must use frozen BF16")
        if self.execution_mode not in DISPATCH_EXECUTION_MODES:
            raise ValueError("dispatch control execution mode is unsupported")
        if (
            self.raw_trace is not None
            and self.raw_trace.execution_mode != self.execution_mode
        ):
            raise ValueError("dispatch control and raw trace modes differ")
        if self.trace_scope is not None:
            expected_scope = (
                TraceScopeEvidence
                if self.execution_mode == EAGER_EXECUTION_MODE
                else CUDAGraphTraceScopeEvidence
            )
            if not isinstance(self.trace_scope, expected_scope):
                raise ValueError("dispatch trace scope differs from execution mode")
        if tuple(event.order for event in self.device_events) != tuple(
            range(len(self.device_events))
        ):
            raise ValueError("dispatch device event ordering is not canonical")
        if self.trace_scope is not None:
            correlations = tuple(
                sorted({event.correlation_id for event in self.device_events})
            )
            if correlations != self.trace_scope.runtime_correlations:
                raise ValueError("trace scope correlations differ from device events")
            if any(
                event.stream != self.trace_scope.gpu_stream
                for event in self.device_events
            ):
                raise ValueError("device events differ from retained trace scope")
            if isinstance(self.trace_scope, TraceScopeEvidence):
                if any(
                    event.external_id
                    not in self.trace_scope.nested_cpu_external_ids
                    for event in self.device_events
                ):
                    raise ValueError(
                        "eager device External IDs differ from trace scope"
                    )
            else:
                graph_ids = {event.graph_id for event in self.device_events}
                graph_node_ids = tuple(
                    event.graph_node_id for event in self.device_events
                )
                if (
                    any(
                        event.external_id
                        != self.trace_scope.graph_launch_external_id
                        for event in self.device_events
                    )
                    or graph_ids != {self.trace_scope.graph_id}
                    or graph_node_ids != self.trace_scope.graph_node_ids
                ):
                    raise ValueError(
                        "graph device identities differ from trace scope"
                    )

    @property
    def byte_evidence(self) -> KVByteEvidence:
        return calculate_kv_bytes(
            batch_size=self.batch_size,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            context_length=self.context_length,
            head_dim=self.head_dim,
            dtype_bytes=self.dtype_bytes,
        )

    @property
    def attention_families(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    event.kernel_family
                    for event in self.device_events
                    if event.kernel_family is not None
                }
            )
        )

    @property
    def events_before_attention(self) -> tuple[CUDADeviceEvent, ...]:
        attention_orders = [
            event.order
            for event in self.device_events
            if event.classification == "flash_attention"
        ]
        if not attention_orders:
            return self.device_events
        first_attention = min(attention_orders)
        return tuple(
            event for event in self.device_events if event.order < first_attention
        )

    def held_constants(self) -> tuple[object, ...]:
        return (
            self.batch_size,
            self.context_length,
            self.query_length,
            self.num_query_heads,
            self.head_dim,
            self.dtype,
            self.dtype_bytes,
            self.is_causal,
            self.warmup_count,
            self.execution_mode,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "geometry": {
                "batch_size": self.batch_size,
                "context_length": self.context_length,
                "query_length": self.query_length,
                "num_query_heads": self.num_query_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "dtype": self.dtype,
                "dtype_bytes": self.dtype_bytes,
                "is_causal": self.is_causal,
                "warmup_count": self.warmup_count,
                "execution_mode": self.execution_mode,
            },
            "backend": self.backend.to_dict(),
            "raw_trace": None if self.raw_trace is None else self.raw_trace.to_dict(),
            "trace_scope": (
                None if self.trace_scope is None else self.trace_scope.to_dict()
            ),
            "byte_evidence": self.byte_evidence.to_dict(),
            "device_events": [event.to_dict() for event in self.device_events],
            "attention_families": list(self.attention_families),
            "events_before_attention": [
                event.to_dict() for event in self.events_before_attention
            ],
        }


@dataclass(frozen=True, slots=True)
class FlashKernelSequenceEvidence:
    """Strict standard-forward or split-K-forward-plus-combine sequence."""

    family: str | None
    variant: str
    kernel_names: tuple[str, ...]
    kernel_orders: tuple[int, ...]
    forward_orders: tuple[int, ...]
    combine_orders: tuple[int, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.family not in {
            None,
            FLASH_FORWARD_FAMILY,
            FLASH_SPLIT_KV_FAMILY,
        }:
            raise ValueError("Flash sequence family is unsupported")
        if self.variant not in {
            "standard_forward",
            "split_k_forward_combine",
            "unverified",
        }:
            raise ValueError("Flash sequence variant is unsupported")
        if len(self.kernel_names) != len(self.kernel_orders):
            raise ValueError("Flash sequence kernel names and orders differ")
        if self.kernel_orders != tuple(sorted(set(self.kernel_orders))):
            raise ValueError("Flash sequence kernel orders are not canonical")
        if self.forward_orders != tuple(sorted(set(self.forward_orders))):
            raise ValueError("Flash forward orders are not canonical")
        if self.combine_orders != tuple(sorted(set(self.combine_orders))):
            raise ValueError("Flash combine orders are not canonical")
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("Flash sequence reasons must be non-empty")
        if self.passed != (self.family is not None):
            raise ValueError("Flash sequence family and verdict differ")

    @property
    def passed(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "variant": self.variant,
            "kernel_names": list(self.kernel_names),
            "kernel_orders": list(self.kernel_orders),
            "forward_orders": list(self.forward_orders),
            "combine_orders": list(self.combine_orders),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def analyze_flash_kernel_sequence(
    events: tuple[CUDADeviceEvent, ...],
) -> FlashKernelSequenceEvidence:
    """Identify one unambiguous Flash forward sequence from scoped events."""

    flash_events = tuple(
        event for event in events if event.classification == "flash_attention"
    )
    reasons: list[str] = []
    if not flash_events:
        reasons.append("flash_attention_kernel_absent")
    unrelated_kernels = tuple(
        event
        for event in events
        if event.category == "kernel" and event.classification == "unknown_kernel"
    )
    if unrelated_kernels:
        reasons.append("unrelated_scoped_kernel_present")
    families = {
        event.kernel_family
        for event in flash_events
        if event.kernel_family is not None
    }
    if len(families) > 1:
        reasons.append("ambiguous_flash_kernel_families")

    strict_standard = tuple(
        event
        for event in flash_events
        if _FLASH_FORWARD_KERNEL_RE.search(event.name) is not None
    )
    strict_split_forward = tuple(
        event
        for event in flash_events
        if _FLASH_SPLIT_FORWARD_KERNEL_RE.search(event.name) is not None
    )
    strict_split_combine = tuple(
        event
        for event in flash_events
        if _FLASH_SPLIT_COMBINE_KERNEL_RE.search(event.name) is not None
    )
    recognized_orders = {
        event.order
        for event in (
            *strict_standard,
            *strict_split_forward,
            *strict_split_combine,
        )
    }
    if any(event.order not in recognized_orders for event in flash_events):
        reasons.append("unrecognized_flash_kernel_component")

    family: str | None = None
    variant = "unverified"
    forward_orders: tuple[int, ...] = ()
    combine_orders: tuple[int, ...] = ()
    if not reasons and families == {FLASH_FORWARD_FAMILY}:
        if len(flash_events) != 1 or len(strict_standard) != 1:
            reasons.append("ambiguous_standard_flash_sequence")
        else:
            family = FLASH_FORWARD_FAMILY
            variant = "standard_forward"
            forward_orders = (strict_standard[0].order,)
    elif not reasons and families == {FLASH_SPLIT_KV_FAMILY}:
        if (
            len(flash_events) != 2
            or len(strict_split_forward) != 1
            or len(strict_split_combine) != 1
        ):
            reasons.append("split_k_requires_one_forward_and_one_combine")
        elif strict_split_forward[0].order >= strict_split_combine[0].order:
            reasons.append("split_k_combine_does_not_follow_forward")
        else:
            family = FLASH_SPLIT_KV_FAMILY
            variant = "split_k_forward_combine"
            forward_orders = (strict_split_forward[0].order,)
            combine_orders = (strict_split_combine[0].order,)
    elif not reasons:
        reasons.append("flash_kernel_family_unverified")

    if reasons:
        family = None
        variant = "unverified"
        forward_orders = ()
        combine_orders = ()
    return FlashKernelSequenceEvidence(
        family=family,
        variant=variant,
        kernel_names=tuple(event.name for event in flash_events),
        kernel_orders=tuple(event.order for event in flash_events),
        forward_orders=forward_orders,
        combine_orders=combine_orders,
        reasons=tuple(reasons),
    )


def compare_geometry_bound_kernel_sequences(
    gqa: FlashKernelSequenceEvidence,
    mha: FlashKernelSequenceEvidence,
    *,
    explicitly_related: Set[tuple[str, str]] = frozenset(),
) -> KernelFamilyComparison:
    """Compare exact families with no implicit standard/split-K approval."""

    if not gqa.passed or not mha.passed:
        relation = "unverified"
    elif gqa.family == mha.family:
        relation = "same"
    elif (gqa.family, mha.family) in explicitly_related or (
        mha.family,
        gqa.family,
    ) in explicitly_related:
        relation = "related"
    else:
        relation = "unrelated"
    return KernelFamilyComparison(
        relation=relation,
        gqa_families=() if gqa.family is None else (gqa.family,),
        mha_families=() if mha.family is None else (mha.family,),
    )


@dataclass(frozen=True, slots=True)
class GeometryBoundSourceShapeEvidence:
    """Source and tensor evidence for a production-cache GQA/MHA pair."""

    sources: tuple[SourceFileEvidence, ...]
    gqa_query: TensorShapeEvidence
    gqa_output: TensorShapeEvidence
    cache: StaticCacheViewBindingEvidence
    mha_query: TensorShapeEvidence
    mha_key: TensorShapeEvidence
    mha_value: TensorShapeEvidence
    mha_output: TensorShapeEvidence

    def __post_init__(self) -> None:
        paths = tuple(source.relative_path for source in self.sources)
        if paths != REQUIRED_SUT_SOURCES:
            raise ValueError("geometry-bound source evidence differs from the SUT")
        if len(set(paths)) != len(paths):
            raise ValueError("geometry-bound source paths contain duplicates")

    @property
    def source_verified(self) -> bool:
        source_by_path = {source.relative_path: source for source in self.sources}
        return bool(
            all(source.passed for source in self.sources)
            and all(
                source.selected_function_paths
                == _SELECTED_SOURCE_FUNCTION_PATHS[source.relative_path]
                for source in self.sources
            )
            and source_by_path[REQUIRED_SUT_SOURCES[2]].sha256
            == self.cache.layout.implementation_sha256
        )

    def shape_verified_for(
        self,
        gqa: DispatchControlEvidence,
        mha: DispatchControlEvidence,
    ) -> bool:
        expected_query = (
            gqa.batch_size,
            PHASE3_NUM_QUERY_HEADS,
            1,
            PHASE3_HEAD_DIM,
        )
        expected_gqa_kv = (
            gqa.batch_size,
            PHASE3_NUM_KV_HEADS,
            gqa.context_length,
            PHASE3_HEAD_DIM,
        )
        expected_mha_kv = (
            mha.batch_size,
            PHASE3_NUM_QUERY_HEADS,
            mha.context_length,
            PHASE3_HEAD_DIM,
        )
        expected_query_stride = (
            PHASE3_NUM_QUERY_HEADS * PHASE3_HEAD_DIM,
            PHASE3_HEAD_DIM,
            PHASE3_HEAD_DIM,
            1,
        )
        expected_mha_kv_stride = (
            PHASE3_NUM_QUERY_HEADS * mha.context_length * PHASE3_HEAD_DIM,
            mha.context_length * PHASE3_HEAD_DIM,
            PHASE3_HEAD_DIM,
            1,
        )
        query_tensors = (
            self.gqa_query,
            self.mha_query,
        )
        output_tensors = (
            self.gqa_output,
            self.mha_output,
        )
        tensors = (
            self.gqa_query,
            self.gqa_output,
            self.cache.key_view,
            self.cache.value_view,
            self.mha_query,
            self.mha_key,
            self.mha_value,
            self.mha_output,
        )
        return bool(
            self.cache.verified
            and self.cache.active_context == gqa.context_length
            and self.cache.layout.batch_size == gqa.batch_size
            and self.gqa_query.shape == expected_query
            and self.gqa_output.shape == expected_query
            and self.cache.key_view.shape == expected_gqa_kv
            and self.cache.value_view.shape == expected_gqa_kv
            and self.mha_query.shape == expected_query
            and self.mha_output.shape == expected_query
            and self.mha_key.shape == expected_mha_kv
            and self.mha_value.shape == expected_mha_kv
            and all(
                tensor.stride == expected_query_stride
                for tensor in query_tensors
            )
            and all(
                all(
                    dimension == 1 or observed == expected
                    for dimension, observed, expected in zip(
                        tensor.shape,
                        tensor.stride,
                        expected_query_stride,
                        strict=True,
                    )
                )
                for tensor in output_tensors
            )
            and self.mha_key.stride == expected_mha_kv_stride
            and self.mha_value.stride == expected_mha_kv_stride
            and all(
                tensor.storage_bytes == tensor.logical_bytes
                and tensor.storage_offset == 0
                and tensor.is_contiguous
                for tensor in (*query_tensors, *output_tensors)
            )
            and self.gqa_query == self.mha_query
            and all(tensor.dtype == PHASE3_DTYPE for tensor in tensors)
            and all(tensor.element_size == PHASE3_DTYPE_BYTES for tensor in tensors)
            and len({tensor.device for tensor in tensors}) == 1
            and self.mha_key.storage_bytes == self.mha_key.logical_bytes
            and self.mha_value.storage_bytes == self.mha_value.logical_bytes
            and self.mha_key.storage_offset == 0
            and self.mha_value.storage_offset == 0
            and self.mha_key.is_contiguous
            and self.mha_value.is_contiguous
        )

    def to_dict(
        self,
        *,
        gqa: DispatchControlEvidence,
        mha: DispatchControlEvidence,
    ) -> dict[str, object]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "source_paths": [source.relative_path for source in self.sources],
            "source_verified": self.source_verified,
            "gqa_query": self.gqa_query.to_dict(),
            "gqa_output": self.gqa_output.to_dict(),
            "cache": self.cache.to_dict(),
            "mha_query": self.mha_query.to_dict(),
            "mha_key": self.mha_key.to_dict(),
            "mha_value": self.mha_value.to_dict(),
            "mha_output": self.mha_output.to_dict(),
            "shape_verified": self.shape_verified_for(gqa, mha),
        }


@dataclass(frozen=True, slots=True)
class KernelFamilyComparison:
    """Relationship between the held-constant GQA and MHA controls."""

    relation: str
    gqa_families: tuple[str, ...]
    mha_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.relation not in {"same", "related", "unrelated", "unverified"}:
            raise ValueError("kernel family relation is unsupported")

    @property
    def passed(self) -> bool:
        return self.relation in {"same", "related"}

    def to_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "gqa_families": list(self.gqa_families),
            "mha_families": list(self.mha_families),
            "passed": self.passed,
        }


def compare_kernel_families(
    gqa: DispatchControlEvidence | None,
    mha: DispatchControlEvidence | None,
    *,
    explicitly_related: Set[tuple[str, str]] = frozenset(),
) -> KernelFamilyComparison:
    """Compare normalized families with no implicit related-family waiver."""

    gqa_families = () if gqa is None else gqa.attention_families
    mha_families = () if mha is None else mha.attention_families
    if len(gqa_families) != 1 or len(mha_families) != 1:
        relation = "unverified"
    elif gqa_families == mha_families:
        relation = "same"
    elif (
        (gqa_families[0], mha_families[0]) in explicitly_related
        or (mha_families[0], gqa_families[0]) in explicitly_related
    ):
        relation = "related"
    else:
        relation = "unrelated"
    return KernelFamilyComparison(
        relation=relation,
        gqa_families=gqa_families,
        mha_families=mha_families,
    )


@dataclass(frozen=True, slots=True)
class GQAProofEvaluation:
    """Combined result for every preregistered GQA proof layer."""

    verdict: GQAVerdict
    dispatch_verified: bool
    no_replication_kernel_verified: bool
    allocation_verified: bool
    source_verified: bool
    shape_verified: bool
    family_comparison: KernelFamilyComparison
    positive_materialization_evidence: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "dispatch_verified": self.dispatch_verified,
            "no_replication_kernel_verified": (
                self.no_replication_kernel_verified
            ),
            "allocation_verified": self.allocation_verified,
            "source_verified": self.source_verified,
            "shape_verified": self.shape_verified,
            "family_comparison": self.family_comparison.to_dict(),
            "positive_materialization_evidence": list(
                self.positive_materialization_evidence
            ),
            "reasons": list(self.reasons),
            "performance_timing_reported": False,
        }


def _preceding_activity_is_nonmaterializing(
    control: DispatchControlEvidence,
) -> bool:
    return not control.events_before_attention


def evaluate_gqa_device_dispatch(
    *,
    gqa: DispatchControlEvidence | None,
    mha: DispatchControlEvidence | None,
    allocation_verified: bool,
    source_verified: bool,
    shape_verified: bool,
    expanded_kv_allocation_detected: bool = False,
    expanded_kv_tensor_detected: bool = False,
    explicitly_related_families: Set[tuple[str, str]] = frozenset(),
    source_positive_materialization_evidence: tuple[str, ...] = (),
) -> GQAProofEvaluation:
    """Evaluate device proof through the corrected four-state taxonomy."""

    family = compare_kernel_families(
        gqa,
        mha,
        explicitly_related=explicitly_related_families,
    )
    held_constants_match = bool(
        gqa is not None
        and mha is not None
        and gqa.held_constants() == mha.held_constants()
    )
    source_build_match = bool(
        gqa is not None
        and mha is not None
        and gqa.backend.source_build_fingerprint
        == mha.backend.source_build_fingerprint
    )
    dispatch_verified = bool(
        held_constants_match
        and source_build_match
        and gqa is not None
        and mha is not None
        and gqa.raw_trace is not None
        and mha.raw_trace is not None
        and gqa.trace_scope is not None
        and mha.trace_scope is not None
        and gqa.backend.passed
        and mha.backend.passed
        and family.passed
    )

    positive: list[str] = list(source_positive_materialization_evidence)
    if expanded_kv_allocation_detected:
        positive.append("expanded_kv_allocation")
    if expanded_kv_tensor_detected:
        positive.append("expanded_kv_tensor")
    if gqa is not None:
        expanded_copy_sizes = {
            gqa.byte_evidence.expanded_kv_bytes,
            gqa.byte_evidence.expanded_kv_bytes // 2,
        }
        positive.extend(
            f"device_event:{event.classification}:{event.name}"
            for event in gqa.events_before_attention
            if event.classification in MATERIALIZATION_CLASSIFICATIONS
        )
        positive.extend(
            f"expanded_kv_copy:{event.copy_bytes}:{event.name}"
            for event in gqa.events_before_attention
            if event.classification in COPY_CANDIDATE_CLASSIFICATIONS
            and event.copy_bytes in expanded_copy_sizes
        )

    no_replication_kernel_verified = bool(
        dispatch_verified
        and gqa is not None
        and mha is not None
        and _preceding_activity_is_nonmaterializing(gqa)
        and _preceding_activity_is_nonmaterializing(mha)
        and not any(
            event.classification
            in (MATERIALIZATION_CLASSIFICATIONS | COPY_CANDIDATE_CLASSIFICATIONS)
            for event in (*gqa.device_events, *mha.device_events)
        )
    )
    verdict = classify_gqa_evidence(
        materialization_evidence=bool(positive),
        dispatch_verified=dispatch_verified,
        no_replication_kernel_verified=no_replication_kernel_verified,
        allocation_verified=allocation_verified,
        source_verified=source_verified,
        shape_verified=shape_verified,
    )
    reasons: list[str] = []
    if positive:
        reasons.append("positive materialization evidence exists")
    if not held_constants_match:
        reasons.append("GQA and MHA controls do not hold constants fixed")
    if not source_build_match:
        reasons.append("GQA and MHA backend source/build identities differ")
    if not dispatch_verified:
        reasons.append("device dispatch is not fully verified")
    if dispatch_verified and not no_replication_kernel_verified:
        reasons.append("no-preceding-materialization proof is incomplete")
    if not allocation_verified:
        reasons.append("allocation-size proof is incomplete")
    if not source_verified:
        reasons.append("source proof is incomplete")
    if not shape_verified:
        reasons.append("shape and storage proof is incomplete")
    return GQAProofEvaluation(
        verdict=verdict,
        dispatch_verified=dispatch_verified,
        no_replication_kernel_verified=no_replication_kernel_verified,
        allocation_verified=allocation_verified,
        source_verified=source_verified,
        shape_verified=shape_verified,
        family_comparison=family,
        positive_materialization_evidence=tuple(positive),
        reasons=tuple(reasons),
    )


def evaluate_geometry_bound_gqa_device_dispatch(
    *,
    point: Phase3DispatchPointBinding,
    gqa: DispatchControlEvidence,
    mha: DispatchControlEvidence,
    gqa_sequence: FlashKernelSequenceEvidence,
    mha_sequence: FlashKernelSequenceEvidence,
    source_shape: GeometryBoundSourceShapeEvidence,
    explicitly_related_families: Set[tuple[str, str]] = frozenset(),
) -> GQAProofEvaluation:
    """Evaluate dispatch-only evidence bound to one production cache geometry."""

    family = compare_geometry_bound_kernel_sequences(
        gqa_sequence,
        mha_sequence,
        explicitly_related=explicitly_related_families,
    )
    held_constants_match = gqa.held_constants() == mha.held_constants()
    source_build_match = (
        gqa.backend.source_build_fingerprint
        == mha.backend.source_build_fingerprint
    )
    expected_execution_mode = (
        CUDA_GRAPH_REPLAY_EXECUTION_MODE
        if point.graph_mode == "cuda_graph"
        else EAGER_EXECUTION_MODE
    )
    point_geometry_matches = bool(
        gqa.batch_size == point.batch_size
        and mha.batch_size == point.batch_size
        and gqa.context_length == point.active_context
        and mha.context_length == point.active_context
        and source_shape.cache.active_context == point.active_context
        and source_shape.cache.layout.capacity == point.capacity
        and source_shape.cache.layout.batch_size == point.batch_size
        and gqa.execution_mode == expected_execution_mode
        and mha.execution_mode == expected_execution_mode
    )
    dispatch_verified = bool(
        held_constants_match
        and source_build_match
        and point_geometry_matches
        and gqa.raw_trace is not None
        and mha.raw_trace is not None
        and gqa.trace_scope is not None
        and mha.trace_scope is not None
        and gqa.backend.passed
        and mha.backend.passed
        and gqa.backend.control_transcript_verified
        and mha.backend.control_transcript_verified
        and gqa_sequence.passed
        and mha_sequence.passed
        and family.passed
    )

    positive: list[str] = list(
        source_materialization_evidence(source_shape.sources)
    )
    expanded_copy_sizes = {
        gqa.byte_evidence.expanded_kv_bytes,
        gqa.byte_evidence.expanded_kv_bytes // 2,
    }
    positive.extend(
        f"device_event:{event.classification}:{event.name}"
        for event in gqa.events_before_attention
        if event.classification in MATERIALIZATION_CLASSIFICATIONS
    )
    positive.extend(
        f"expanded_kv_copy:{event.copy_bytes}:{event.name}"
        for event in gqa.events_before_attention
        if event.classification in COPY_CANDIDATE_CLASSIFICATIONS
        and event.copy_bytes in expanded_copy_sizes
    )
    forbidden_device_activity = any(
        event.classification
        in (MATERIALIZATION_CLASSIFICATIONS | COPY_CANDIDATE_CLASSIFICATIONS)
        for event in (*gqa.device_events, *mha.device_events)
    )
    no_replication_kernel_verified = bool(
        dispatch_verified
        and _preceding_activity_is_nonmaterializing(gqa)
        and _preceding_activity_is_nonmaterializing(mha)
        and not forbidden_device_activity
    )
    source_verified = source_shape.source_verified
    shape_verified = source_shape.shape_verified_for(gqa, mha)
    allocation_verified = False
    verdict = classify_gqa_evidence(
        materialization_evidence=bool(positive),
        dispatch_verified=dispatch_verified,
        no_replication_kernel_verified=no_replication_kernel_verified,
        allocation_verified=allocation_verified,
        source_verified=source_verified,
        shape_verified=shape_verified,
    )
    reasons: list[str] = []
    if positive:
        reasons.append("positive materialization evidence exists")
    if not held_constants_match:
        reasons.append("GQA and MHA production controls differ")
    if not source_build_match:
        reasons.append("GQA and MHA backend identities differ")
    if not point_geometry_matches:
        reasons.append("dispatch geometry differs from the bound Phase 3 point")
    if not gqa_sequence.passed:
        reasons.extend(f"gqa_sequence:{item}" for item in gqa_sequence.reasons)
    if not mha_sequence.passed:
        reasons.extend(f"mha_sequence:{item}" for item in mha_sequence.reasons)
    if not family.passed:
        reasons.append("GQA and MHA Flash kernel families are not related")
    if dispatch_verified and not no_replication_kernel_verified:
        reasons.append("no-preceding-materialization proof is incomplete")
    if not source_verified:
        reasons.append("source proof is incomplete")
    if not shape_verified:
        reasons.append("production cache shape/storage binding is incomplete")
    reasons.append("allocation proof is pending checksum-bound Task C evidence")
    return GQAProofEvaluation(
        verdict=verdict,
        dispatch_verified=dispatch_verified,
        no_replication_kernel_verified=no_replication_kernel_verified,
        allocation_verified=False,
        source_verified=source_verified,
        shape_verified=shape_verified,
        family_comparison=family,
        positive_materialization_evidence=tuple(positive),
        reasons=tuple(reasons),
    )


def collect_torch_profiler_trace(
    operation: Callable[[], Any],
    output_path: Path,
    *,
    artifact_relative_path: str,
    marker: str,
    warmup_count: int,
    device: Any,
    execution_mode: str = EAGER_EXECUTION_MODE,
) -> RawTraceArtifact:
    """Export one untouched CPU+CUDA Chrome trace outside benchmark timing."""

    if not callable(operation):
        raise TypeError("dispatch operation must be callable")
    _positive_integer(warmup_count, "warmup_count")
    if execution_mode not in DISPATCH_EXECUTION_MODES:
        raise ValueError("dispatch trace execution mode is unsupported")
    if not marker.strip():
        raise ValueError("dispatch trace marker must be non-empty")
    path = Path(output_path)
    if os.path.lexists(path):
        raise GQADeviceDispatchError("raw trace output already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise GQADeviceDispatchError("raw trace parent must be a real directory")
    staging_name = tempfile.mkdtemp(
        prefix=f".{path.name}.",
        suffix=".staging-dir",
        dir=path.parent,
    )
    staging_directory = Path(staging_name)
    staging_path = staging_directory / "trace.chrome.json"
    reserved_stat = os.lstat(staging_directory)
    if (
        not stat.S_ISDIR(reserved_stat.st_mode)
        or stat.S_IMODE(reserved_stat.st_mode) & 0o077
    ):
        raise GQADeviceDispatchError("raw trace staging directory is not private")

    torch = importlib.import_module("torch")
    profiler = importlib.import_module("torch.profiler")
    retained_output: Any = None
    captured_graph: Any = None
    if execution_mode == EAGER_EXECUTION_MODE:
        for _ in range(warmup_count):
            operation()
        torch.cuda.synchronize(device=device)
    else:
        graph_module = importlib.import_module("kvbench.runtime.cuda_graph")
        captured_graph = graph_module.capture_fixed_graph(
            operation,
            warmup_steps=warmup_count,
            device=device,
        )
        retained_output = captured_graph.output
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as trace:
        with torch.autograd.profiler.record_function(marker):
            if execution_mode == EAGER_EXECUTION_MODE:
                retained_output = operation()
            else:
                retained_output = captured_graph.replay()
    torch.cuda.synchronize(device=device)
    trace.export_chrome_trace(str(staging_path))
    staged_directory_stat = os.lstat(staging_directory)
    staged_stat = os.lstat(staging_path)
    if (
        not stat.S_ISDIR(staged_directory_stat.st_mode)
        or staged_directory_stat.st_dev != reserved_stat.st_dev
        or staged_directory_stat.st_ino != reserved_stat.st_ino
        or not stat.S_ISREG(staged_stat.st_mode)
        or staged_stat.st_size <= 0
    ):
        raise GQADeviceDispatchError(
            "profiler replaced or invalidated the private trace staging area"
        )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    trace_fd = os.open(staging_path, os.O_RDONLY | nofollow)
    try:
        opened_stat = os.fstat(trace_fd)
        if (
            opened_stat.st_dev != staged_stat.st_dev
            or opened_stat.st_ino != staged_stat.st_ino
        ):
            raise GQADeviceDispatchError("raw trace file changed during validation")
        os.fsync(trace_fd)
        try:
            os.link(staging_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise GQADeviceDispatchError(
                "raw trace destination appeared before no-replace promotion"
            ) from error
        published_stat = os.lstat(path)
        if (
            not stat.S_ISREG(published_stat.st_mode)
            or published_stat.st_dev != opened_stat.st_dev
            or published_stat.st_ino != opened_stat.st_ino
        ):
            raise GQADeviceDispatchError("published raw trace inode differs")
        artifact = RawTraceArtifact.from_path(
            path,
            relative_path=artifact_relative_path,
            execution_mode=execution_mode,
        )
    finally:
        os.close(trace_fd)
    os.unlink(staging_path)
    os.rmdir(staging_directory)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    del captured_graph, retained_output
    return artifact


@dataclass(frozen=True, slots=True)
class BackendIdentityEvidence:
    """Canonical, checksum-bound copy of the frozen backend manifest."""

    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.canonical_json)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("backend identity JSON is invalid") from error
        if not isinstance(payload, dict):
            raise ValueError("backend identity must be a JSON object")
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical != self.canonical_json:
            raise ValueError("backend identity JSON is not canonical")
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("backend identity SHA-256 differs")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> BackendIdentityEvidence:
        canonical = json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            canonical_json=canonical,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def to_dict(self) -> dict[str, object]:
        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):  # pragma: no cover - guarded at creation
            raise AssertionError("validated backend identity changed type")
        return {"sha256": self.sha256, "manifest": payload}


def audit_gqa_source_files(
    source_root: Path,
    source_paths: tuple[Path, ...],
) -> tuple[SourceFileEvidence, ...]:
    """Hash and scan the selected frozen SUT sources, failing on ambiguity."""

    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        raise GQADeviceDispatchError("source root must be a real directory")
    root = root.resolve(strict=True)
    if not source_paths:
        raise GQADeviceDispatchError("source audit paths are absent")
    evidence: list[SourceFileEvidence] = []
    for raw_path in source_paths:
        supplied = Path(raw_path)
        candidate = supplied if supplied.is_absolute() else root / supplied
        if not candidate.is_file() or candidate.is_symlink():
            raise GQADeviceDispatchError("source audit target is not a real file")
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise GQADeviceDispatchError(
                "source audit target escapes the source root"
            ) from error
        evidence.append(source_file_evidence_from_bytes(relative, resolved.read_bytes()))
    evidence.sort(key=lambda item: item.relative_path)
    paths = tuple(item.relative_path for item in evidence)
    if len(set(paths)) != len(paths):
        raise GQADeviceDispatchError("source audit paths contain duplicates")
    return tuple(evidence)


def _source_expression_identifiers(node: ast.AST) -> tuple[str, ...]:
    identifiers: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            identifiers.append(item.id)
        elif isinstance(item, ast.Attribute):
            identifiers.append(item.attr)
    return tuple(identifiers)


def _identifier_names_kv_tensor(identifier: str) -> bool:
    lowered = identifier.casefold()
    if lowered in {"k", "v", "kv", "key", "value"}:
        return True
    tokens = frozenset(filter(None, re.split(r"[^a-z0-9]+", lowered)))
    return bool(
        tokens & {"key", "value", "kv"}
        or ({"k", "cache"} <= tokens)
        or ({"v", "cache"} <= tokens)
        or ({"k", "states"} <= tokens)
        or ({"v", "states"} <= tokens)
    )


def _expression_names_kv_tensor(node: ast.AST) -> bool:
    return any(
        _identifier_names_kv_tensor(identifier)
        for identifier in _source_expression_identifiers(node)
    )


def _identifier_names_replication_target(identifier: str) -> bool:
    lowered = identifier.casefold()
    return bool(
        any(
            token in lowered
            for token in (
                "expanded_kv",
                "replicated_kv",
                "query_head_kv",
                "expanded_key",
                "expanded_value",
                "replicated_key",
                "replicated_value",
            )
        )
    )


def _expression_names_replication_target(node: ast.AST) -> bool:
    return any(
        _identifier_names_replication_target(identifier)
        for identifier in _source_expression_identifiers(node)
    )


def _call_leaf_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _qualified_source_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[f"{node.name}.{child.name}"] = child
    return functions


def _selected_executable_nodes(
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    selected_paths: tuple[str, ...],
) -> tuple[ast.AST, ...]:
    retained: list[ast.AST] = []
    for path in selected_paths:
        function = functions.get(path)
        if function is None:
            continue
        pending = list(reversed(function.body))
        while pending:
            node = pending.pop()
            retained.append(node)
            children = tuple(ast.iter_child_nodes(node))
            for child in reversed(children):
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
                ):
                    continue
                pending.append(child)
    return tuple(retained)


def _call_expands_kv(call: ast.Call) -> bool:
    if _call_leaf_name(call) != "expand":
        return False
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    first_argument = call.args[0] if call.args else None
    return bool(
        (receiver is not None and _expression_names_kv_tensor(receiver))
        or (
            first_argument is not None
            and _expression_names_kv_tensor(first_argument)
        )
    )


def _direct_source_replication_findings(
    selected_nodes: tuple[ast.AST, ...],
) -> tuple[str, ...]:
    """Classify direct replication only on exact selected execution paths."""

    direct: set[str] = set()
    for node in selected_nodes:
        if not isinstance(node, ast.Call):
            continue
        leaf = _call_leaf_name(node)
        if leaf == "repeat_kv":
            direct.add("repeat_kv")
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        first_argument = node.args[0] if node.args else None
        operates_on_kv = bool(
            (receiver is not None and _expression_names_kv_tensor(receiver))
            or (
                first_argument is not None
                and _expression_names_kv_tensor(first_argument)
            )
        )
        if leaf == "repeat_interleave" and operates_on_kv:
            direct.add("repeat_interleave")
        if leaf == "repeat" and operates_on_kv:
            direct.add("tensor_repeat")
        if leaf in {"clone", "contiguous"} and any(
            _call_expands_kv(item)
            for item in ast.walk(node)
            if isinstance(item, ast.Call) and item is not node
        ):
            direct.add("tensor_expand")
        if (
            leaf in {"copy", "copy_"}
            and receiver is not None
            and _expression_names_replication_target(receiver)
        ):
            direct.add("replication_copy")

    assignment_nodes = (
        node
        for node in selected_nodes
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
    )
    for assignment in assignment_nodes:
        if isinstance(assignment, ast.Assign):
            targets = tuple(assignment.targets)
            value = assignment.value
        else:
            targets = (assignment.target,)
            value = assignment.value
        if value is None:
            continue
        if not any(_expression_names_replication_target(item) for item in targets):
            continue
        for call in (
            item for item in ast.walk(value) if isinstance(item, ast.Call)
        ):
            leaf = _call_leaf_name(call)
            if leaf == "repeat_interleave":
                direct.add("repeat_interleave")
            elif leaf == "repeat":
                direct.add("tensor_repeat")
            elif leaf == "expand":
                direct.add("tensor_expand")
    return tuple(
        label
        for label, _pattern in _FORBIDDEN_SOURCE_PATTERNS
        if label in direct
    )


def source_file_evidence_from_bytes(
    relative_path: str,
    raw: bytes,
) -> SourceFileEvidence:
    """Rebuild one source finding record from retained bytes only."""

    if not isinstance(raw, bytes) or not raw:
        raise GQADeviceDispatchError("source audit bytes are absent")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GQADeviceDispatchError(
            f"source audit target is not UTF-8: {relative_path}"
        ) from error
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise GQADeviceDispatchError("source audit target is not valid Python") from error
    expected_paths = _SELECTED_SOURCE_FUNCTION_PATHS.get(relative_path, ())
    functions = _qualified_source_functions(tree)
    selected_paths = tuple(path for path in expected_paths if path in functions)
    selected_nodes = _selected_executable_nodes(functions, selected_paths)
    findings = tuple(
        label
        for label, pattern in _FORBIDDEN_SOURCE_PATTERNS
        if pattern.search(text) is not None
    )
    return SourceFileEvidence(
        relative_path=relative_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        findings=findings,
        direct_replication_findings=(
            _direct_source_replication_findings(selected_nodes)
        ),
        selected_function_paths=selected_paths,
    )


def tensor_shape_evidence(tensor: Any) -> TensorShapeEvidence:
    """Read tensor metadata without copying tensor contents to the host."""

    required = (
        "shape",
        "stride",
        "dtype",
        "device",
        "element_size",
        "untyped_storage",
        "storage_offset",
        "is_contiguous",
    )
    if any(not hasattr(tensor, attribute) for attribute in required):
        raise GQADeviceDispatchError("shape evidence requires a tensor object")
    storage = tensor.untyped_storage()
    return TensorShapeEvidence(
        shape=tuple(int(item) for item in tensor.shape),
        stride=tuple(int(item) for item in tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        element_size=int(tensor.element_size()),
        storage_bytes=int(storage.nbytes()),
        storage_offset=int(tensor.storage_offset()),
        is_contiguous=bool(tensor.is_contiguous()),
    )


def static_cache_view_binding_evidence(
    *,
    layout: StaticCacheLayoutEvidence,
    layer_index: int,
    active_context: int,
    key_backing: Any,
    value_backing: Any,
    key_view: Any,
    value_view: Any,
) -> StaticCacheViewBindingEvidence:
    """Capture metadata and storage identity without tensor-to-host conversion."""

    key_backing_storage = key_backing.untyped_storage()
    value_backing_storage = value_backing.untyped_storage()
    key_view_storage = key_view.untyped_storage()
    value_view_storage = value_view.untyped_storage()
    key_backing_pointer = int(key_backing_storage.data_ptr())
    value_backing_pointer = int(value_backing_storage.data_ptr())
    key_view_pointer = int(key_view_storage.data_ptr())
    value_view_pointer = int(value_view_storage.data_ptr())
    return StaticCacheViewBindingEvidence(
        layout=layout,
        layer_index=layer_index,
        active_context=active_context,
        key_backing=tensor_shape_evidence(key_backing),
        value_backing=tensor_shape_evidence(value_backing),
        key_view=tensor_shape_evidence(key_view),
        value_view=tensor_shape_evidence(value_view),
        key_backing_storage_ptr=key_backing_pointer,
        value_backing_storage_ptr=value_backing_pointer,
        key_view_storage_ptr=key_view_pointer,
        value_view_storage_ptr=value_view_pointer,
        key_view_shares_backing_storage=bool(
            key_view_pointer == key_backing_pointer
        ),
        value_view_shares_backing_storage=bool(
            value_view_pointer == value_backing_pointer
        ),
        key_value_backing_storages_distinct=bool(
            key_backing_pointer != value_backing_pointer
        ),
    )


def _backend_enum_name(backend: object) -> str:
    name = getattr(backend, "name", None)
    if isinstance(name, str) and name:
        return name
    rendered = str(backend)
    return rendered.rsplit(".", maxsplit=1)[-1]


@dataclass(frozen=True, slots=True)
class _RejectedBackendControl:
    passed: bool
    error: str | None
    warning_messages: tuple[str, ...]
    synchronized: bool


def _collect_rejected_backend_control(
    torch: Any,
    *,
    device: Any,
    num_kv_heads: int,
    is_causal: bool,
    scale: float,
    forced_flash_execution: Callable[[], Any],
) -> _RejectedBackendControl:
    query = torch.empty((1, 32, 1, 128), dtype=torch.float64, device=device)
    key = torch.empty((1, num_kv_heads, 1, 128), dtype=torch.float64, device=device)
    value = torch.empty_like(key)
    torch.cuda.synchronize(device=device)
    error_message: str | None = None
    output: Any = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            with forced_flash_execution():
                output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=float(scale),
                    enable_gqa=True,
                )
        except RuntimeError as error:
            error_message = str(error).strip()
    warning_messages = tuple(str(item.message) for item in captured)
    try:
        torch.cuda.synchronize(device=device)
    except RuntimeError as error:
        raise GQADeviceDispatchError(
            "rejected-backend control left an asynchronous CUDA error"
        ) from error
    del output, query, key, value
    expected_error = "No available kernel. Aborting execution."
    has_flash_diagnostic = any(
        "Flash attention kernel not used because:" in item
        for item in warning_messages
    )
    has_dtype_diagnostic = any(
        "Expected query, key and value to all be of dtype: {Half, BFloat16}."
        in item
        for item in warning_messages
    )
    return _RejectedBackendControl(
        passed=bool(
            error_message == expected_error
            and has_flash_diagnostic
            and has_dtype_diagnostic
        ),
        error=error_message,
        warning_messages=warning_messages,
        synchronized=True,
    )


def _collect_backend_control(
    torch: Any,
    *,
    query: Any,
    key: Any,
    value: Any,
    is_causal: bool,
    scale: float,
    identity: BackendIdentityEvidence,
    forced_flash_execution: Callable[[], Any],
) -> BackendControlEvidence:
    current_backends = getattr(
        torch.nn.attention,
        "_cur_sdpa_kernel_backends",
        None,
    )
    if not callable(current_backends):
        raise GQADeviceDispatchError(
            "enabled-SDPA-backend diagnostics are unavailable"
        )
    diagnostics: list[str] = []
    with forced_flash_execution():
        enabled = tuple(_backend_enum_name(item) for item in current_backends())
        params = torch.backends.cuda.SDPAParams(
            query,
            key,
            value,
            None,
            0.0,
            is_causal,
            True,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            flash_eligible = bool(
                torch.backends.cuda.can_use_flash_attention(params, debug=True)
            )
        diagnostics.extend(str(item.message) for item in captured)
        try:
            choice = int(
                torch._fused_sdp_choice(
                    query,
                    key,
                    value,
                    None,
                    0.0,
                    is_causal,
                    scale=float(scale),
                    enable_gqa=True,
                )
            )
        except RuntimeError:
            choice = -1
    expected = int(torch.nn.attention.SDPBackend.FLASH_ATTENTION.value)
    fused_backend_name = "FLASH_ATTENTION" if choice == expected else None
    diagnostics.extend(
        (
            "enabled_backends=" + ",".join(enabled),
            f"can_use_flash_attention={flash_eligible}",
            f"fused_sdp_choice={choice}",
        )
    )
    rejected = _collect_rejected_backend_control(
        torch,
        device=query.device,
        num_kv_heads=int(key.shape[1]),
        is_causal=is_causal,
        scale=scale,
        forced_flash_execution=forced_flash_execution,
    )
    return BackendControlEvidence(
        enabled_backends=enabled,
        flash_eligible=flash_eligible,
        fused_backend_name=fused_backend_name,
        rejected_control_failed=rejected.passed,
        rejected_control_error=rejected.error,
        rejected_control_warnings=rejected.warning_messages,
        rejected_control_synchronized=rejected.synchronized,
        source_build_fingerprint=identity.sha256,
        source_build_verified=True,
        eligibility_diagnostics=tuple(diagnostics),
    )


def _validate_control_tensors(
    torch: Any,
    *,
    role: str,
    query: Any,
    key: Any,
    value: Any,
) -> None:
    if role not in {"gqa", "mha_control"}:
        raise ValueError("dispatch control role is unsupported")
    if not all(isinstance(item, torch.Tensor) for item in (query, key, value)):
        raise GQADeviceDispatchError("dispatch controls require tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise GQADeviceDispatchError("dispatch control tensors must be rank four")
    expected_kv_heads = 8 if role == "gqa" else 32
    if int(query.shape[1]) != 32 or int(key.shape[1]) != expected_kv_heads:
        raise GQADeviceDispatchError("dispatch control head geometry differs")
    if int(query.shape[-1]) != 128 or int(key.shape[-1]) != 128:
        raise GQADeviceDispatchError("dispatch control head dimension differs")
    if int(query.shape[-2]) != FROZEN_CONTROL_QUERY_LENGTH:
        raise GQADeviceDispatchError("dispatch control must isolate one decode query")
    if tuple(key.shape) != tuple(value.shape):
        raise GQADeviceDispatchError("dispatch control K/V shapes differ")
    if int(query.shape[0]) != int(key.shape[0]):
        raise GQADeviceDispatchError("dispatch control batch dimensions differ")
    if int(query.shape[0]) != FROZEN_CONTROL_BATCH_SIZE:
        raise GQADeviceDispatchError("dispatch control batch differs from frozen one")
    if int(key.shape[-2]) != FROZEN_CONTROL_CONTEXT_LENGTH:
        raise GQADeviceDispatchError("dispatch control context differs from frozen one")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype:
        raise GQADeviceDispatchError("dispatch control must use BF16 Q/K/V")
    if value.dtype != query.dtype:
        raise GQADeviceDispatchError("dispatch control V dtype differs")
    if query.device.type != "cuda":
        raise GQADeviceDispatchError("dispatch control must use CUDA tensors")
    if key.device != query.device or value.device != query.device:
        raise GQADeviceDispatchError("dispatch control devices differ")


def _validate_geometry_bound_control_tensors(
    torch: Any,
    *,
    role: str,
    query: Any,
    key: Any,
    value: Any,
    active_context: int,
) -> None:
    """Validate arbitrary Phase 3 B/context while freezing model geometry."""

    if role not in {"gqa", "mha_control"}:
        raise ValueError("geometry-bound dispatch role is unsupported")
    if not all(isinstance(item, torch.Tensor) for item in (query, key, value)):
        raise GQADeviceDispatchError("geometry-bound controls require tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise GQADeviceDispatchError("geometry-bound controls must be rank four")
    expected_kv_heads = (
        PHASE3_NUM_KV_HEADS if role == "gqa" else PHASE3_NUM_QUERY_HEADS
    )
    if (
        int(query.shape[1]) != PHASE3_NUM_QUERY_HEADS
        or int(key.shape[1]) != expected_kv_heads
    ):
        raise GQADeviceDispatchError("geometry-bound control heads differ")
    if (
        int(query.shape[-1]) != PHASE3_HEAD_DIM
        or int(key.shape[-1]) != PHASE3_HEAD_DIM
    ):
        raise GQADeviceDispatchError("geometry-bound head dimension differs")
    if int(query.shape[-2]) != 1:
        raise GQADeviceDispatchError("geometry-bound control must use one query")
    if tuple(key.shape) != tuple(value.shape):
        raise GQADeviceDispatchError("geometry-bound K/V shapes differ")
    if int(query.shape[0]) != int(key.shape[0]):
        raise GQADeviceDispatchError("geometry-bound batch dimensions differ")
    if int(key.shape[-2]) != active_context:
        raise GQADeviceDispatchError("K/V view differs from active context")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype:
        raise GQADeviceDispatchError("geometry-bound controls must use BF16")
    if value.dtype != query.dtype:
        raise GQADeviceDispatchError("geometry-bound V dtype differs")
    if query.device.type != "cuda":
        raise GQADeviceDispatchError("geometry-bound controls require CUDA")
    if key.device != query.device or value.device != query.device:
        raise GQADeviceDispatchError("geometry-bound control devices differ")


def _read_verified_trace(
    path: Path,
    artifact: RawTraceArtifact,
    *,
    marker: str,
    require_kernel_launch_runtime: bool = False,
) -> ScopedCUDAActivities:
    if not path.is_file() or path.is_symlink():
        raise GQADeviceDispatchError("raw trace is not a real file")
    raw = path.read_bytes()
    if len(raw) != artifact.size_bytes:
        raise GQADeviceDispatchError("raw trace size changed after collection")
    if hashlib.sha256(raw).hexdigest() != artifact.sha256:
        raise GQADeviceDispatchError("raw trace digest changed after collection")
    if artifact.execution_mode == CUDA_GRAPH_REPLAY_EXECUTION_MODE:
        return parse_scoped_chrome_cuda_graph_events(raw, marker=marker)
    return parse_scoped_chrome_cuda_events(
        raw,
        marker=marker,
        require_kernel_launch_runtime=require_kernel_launch_runtime,
    )


@dataclass(frozen=True, slots=True)
class GQADeviceDispatchAudit:
    """Small frozen operator controls, not campaign/cache integration proof."""

    backend_identity: BackendIdentityEvidence
    gqa: DispatchControlEvidence
    mha: DispatchControlEvidence
    gqa_source_shape: SourceShapeEvidence
    mha_source_shape: SourceShapeEvidence
    raw_trace_bytes_verified: bool
    allocation_proof_sha256: str | None
    expanded_kv_allocation_detected: bool
    expanded_kv_tensor_detected: bool
    explicitly_related_families: tuple[tuple[str, str], ...]
    evaluation: GQAProofEvaluation

    def __post_init__(self) -> None:
        if self.gqa.role != "gqa" or self.mha.role != "mha_control":
            raise ValueError("dispatch audit controls have incorrect roles")
        if (
            self.gqa.batch_size != FROZEN_CONTROL_BATCH_SIZE
            or self.mha.batch_size != FROZEN_CONTROL_BATCH_SIZE
            or self.gqa.context_length != FROZEN_CONTROL_CONTEXT_LENGTH
            or self.mha.context_length != FROZEN_CONTROL_CONTEXT_LENGTH
            or self.gqa.query_length != FROZEN_CONTROL_QUERY_LENGTH
            or self.mha.query_length != FROZEN_CONTROL_QUERY_LENGTH
            or self.gqa.is_causal
            or self.mha.is_causal
        ):
            raise ValueError("dispatch audit differs from frozen small controls")
        source_paths = tuple(
            item.relative_path for item in self.gqa_source_shape.sources
        )
        if source_paths != REQUIRED_SUT_SOURCES:
            raise ValueError("dispatch audit source set differs from the SUT")
        if self.mha_source_shape.sources != self.gqa_source_shape.sources:
            raise ValueError("GQA/MHA source audit evidence differs")
        if not self.raw_trace_bytes_verified:
            raise ValueError("dispatch audit requires reverified raw traces")
        if self.gqa.raw_trace is None or self.mha.raw_trace is None:
            raise ValueError("dispatch audit raw traces are absent")
        fingerprints = {
            self.gqa.backend.source_build_fingerprint,
            self.mha.backend.source_build_fingerprint,
        }
        if fingerprints != {self.backend_identity.sha256}:
            raise ValueError("backend controls differ from retained identity")
        if self.allocation_proof_sha256 is not None and not _SHA256_RE.fullmatch(
            self.allocation_proof_sha256
        ):
            raise ValueError("allocation proof SHA-256 is invalid")
        if (
            self.evaluation.allocation_verified
            or self.expanded_kv_allocation_detected
        ) and self.allocation_proof_sha256 is None:
            raise ValueError("allocation conclusions require a checksum-bound proof")
        if self.explicitly_related_families != tuple(
            sorted(set(self.explicitly_related_families))
        ):
            raise ValueError("related kernel families are not canonical")
        expected = evaluate_gqa_device_dispatch(
            gqa=self.gqa,
            mha=self.mha,
            allocation_verified=self.evaluation.allocation_verified,
            source_verified=(
                self.gqa_source_shape.source_verified
                and self.mha_source_shape.source_verified
            ),
            shape_verified=(
                self.gqa_source_shape.shape_verified_for(self.gqa)
                and self.mha_source_shape.shape_verified_for(self.mha)
            ),
            expanded_kv_allocation_detected=(
                self.expanded_kv_allocation_detected
            ),
            expanded_kv_tensor_detected=self.expanded_kv_tensor_detected,
            explicitly_related_families=set(self.explicitly_related_families),
            source_positive_materialization_evidence=(
                source_materialization_evidence(self.gqa_source_shape.sources)
            ),
        )
        if expected != self.evaluation:
            raise ValueError("dispatch audit evaluation is not reproducible")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "kvbench-phase3-gqa-device-dispatch-audit-1.0.0",
            "run_kind": "dispatch_audit",
            "measurement_scope": "small_frozen_isolated_operator_controls",
            "control_geometry_scope": {
                "batch_size": FROZEN_CONTROL_BATCH_SIZE,
                "context_length": FROZEN_CONTROL_CONTEXT_LENGTH,
                "query_length": FROZEN_CONTROL_QUERY_LENGTH,
                "campaign_geometries_covered": False,
                "production_endpoint_binding": "integration_required",
                "actual_cache_geometry_binding": "integration_required",
            },
            "profiler_instrumented": True,
            "performance_timing_reported": False,
            "benchmark_timing_eligible": False,
            "backend_identity": self.backend_identity.to_dict(),
            "raw_trace_bytes_verified": self.raw_trace_bytes_verified,
            "gqa": self.gqa.to_dict(),
            "mha_control": self.mha.to_dict(),
            "source_shape": {
                "gqa": self.gqa_source_shape.to_dict(control=self.gqa),
                "mha_control": self.mha_source_shape.to_dict(control=self.mha),
            },
            "allocation_size_proof": {
                "verified": self.evaluation.allocation_verified,
                "sha256": self.allocation_proof_sha256,
                "expanded_kv_allocation_detected": (
                    self.expanded_kv_allocation_detected
                ),
                "expanded_kv_tensor_detected": self.expanded_kv_tensor_detected,
            },
            "explicitly_related_families": [
                list(pair) for pair in self.explicitly_related_families
            ],
            "evaluation": self.evaluation.to_dict(),
        }


def _geometry_bound_trace_marker(
    point: Phase3DispatchPointBinding,
    role: str,
) -> str:
    if role not in {"gqa", "mha_control"}:
        raise ValueError("geometry-bound dispatch role is unsupported")
    return (
        f"kvbench.phase3.geometry_dispatch.{point.run_id}."
        f"{point.point_id}.step{point.decode_step}."
        f"{point.operation_fingerprint_sha256}.{role}"
    )


@dataclass(frozen=True, slots=True)
class GeometryBoundRawTraceValidationEvidence:
    """Raw-derived trace scopes, device events, sequences, and digest."""

    execution_mode: str
    gqa_raw_sha256: str
    gqa_raw_size_bytes: int
    mha_raw_sha256: str
    mha_raw_size_bytes: int
    gqa_scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence
    mha_scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence
    gqa_device_events: tuple[CUDADeviceEvent, ...]
    mha_device_events: tuple[CUDADeviceEvent, ...]
    gqa_kernel_sequence: FlashKernelSequenceEvidence
    mha_kernel_sequence: FlashKernelSequenceEvidence
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.execution_mode not in DISPATCH_EXECUTION_MODES:
            raise ValueError("raw validation execution mode is unsupported")
        for value in (self.gqa_raw_sha256, self.mha_raw_sha256):
            if not _SHA256_RE.fullmatch(value):
                raise ValueError("raw validation SHA-256 is invalid")
        for value in (self.gqa_raw_size_bytes, self.mha_raw_size_bytes):
            _positive_integer(value, "raw_trace_size_bytes")
        expected_scope = (
            TraceScopeEvidence
            if self.execution_mode == EAGER_EXECUTION_MODE
            else CUDAGraphTraceScopeEvidence
        )
        if not isinstance(self.gqa_scope, expected_scope) or not isinstance(
            self.mha_scope, expected_scope
        ):
            raise ValueError("raw validation scope differs from execution mode")
        all_events = (*self.gqa_device_events, *self.mha_device_events)
        if not all_events:
            raise ValueError("raw validation device events are absent")
        if any(type(event.device) is not int or event.device != 0 for event in all_events):
            raise ValueError("Phase 3 raw CUDA events must bind to device zero")
        contexts = {event.context for event in all_events}
        if (
            len(contexts) != 1
            or any(type(context) is not int or context <= 0 for context in contexts)
        ):
            raise ValueError(
                "Phase 3 raw CUDA events require one common positive context"
            )
        if analyze_flash_kernel_sequence(self.gqa_device_events) != (
            self.gqa_kernel_sequence
        ):
            raise ValueError("raw GQA kernel sequence is not reproducible")
        if analyze_flash_kernel_sequence(self.mha_device_events) != (
            self.mha_kernel_sequence
        ):
            raise ValueError("raw MHA kernel sequence is not reproducible")
        if self.evidence_sha256 != self._derive_sha256():
            raise ValueError("raw trace validation digest differs")

    def _payload(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "gqa_raw_sha256": self.gqa_raw_sha256,
            "gqa_raw_size_bytes": self.gqa_raw_size_bytes,
            "mha_raw_sha256": self.mha_raw_sha256,
            "mha_raw_size_bytes": self.mha_raw_size_bytes,
            "gqa_scope": self.gqa_scope.to_dict(),
            "mha_scope": self.mha_scope.to_dict(),
            "gqa_device_events": [
                event.to_dict() for event in self.gqa_device_events
            ],
            "mha_device_events": [
                event.to_dict() for event in self.mha_device_events
            ],
            "gqa_kernel_sequence": self.gqa_kernel_sequence.to_dict(),
            "mha_kernel_sequence": self.mha_kernel_sequence.to_dict(),
        }

    def _derive_sha256(self) -> str:
        raw = json.dumps(
            self._payload(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        execution_mode: str,
        gqa_raw_sha256: str,
        gqa_raw_size_bytes: int,
        mha_raw_sha256: str,
        mha_raw_size_bytes: int,
        gqa_scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence,
        mha_scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence,
        gqa_device_events: tuple[CUDADeviceEvent, ...],
        mha_device_events: tuple[CUDADeviceEvent, ...],
    ) -> GeometryBoundRawTraceValidationEvidence:
        gqa_sequence = analyze_flash_kernel_sequence(gqa_device_events)
        mha_sequence = analyze_flash_kernel_sequence(mha_device_events)
        payload = {
            "execution_mode": execution_mode,
            "gqa_raw_sha256": gqa_raw_sha256,
            "gqa_raw_size_bytes": gqa_raw_size_bytes,
            "mha_raw_sha256": mha_raw_sha256,
            "mha_raw_size_bytes": mha_raw_size_bytes,
            "gqa_scope": gqa_scope.to_dict(),
            "mha_scope": mha_scope.to_dict(),
            "gqa_device_events": [event.to_dict() for event in gqa_device_events],
            "mha_device_events": [event.to_dict() for event in mha_device_events],
            "gqa_kernel_sequence": gqa_sequence.to_dict(),
            "mha_kernel_sequence": mha_sequence.to_dict(),
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            execution_mode=execution_mode,
            gqa_raw_sha256=gqa_raw_sha256,
            gqa_raw_size_bytes=gqa_raw_size_bytes,
            mha_raw_sha256=mha_raw_sha256,
            mha_raw_size_bytes=mha_raw_size_bytes,
            gqa_scope=gqa_scope,
            mha_scope=mha_scope,
            gqa_device_events=gqa_device_events,
            mha_device_events=mha_device_events,
            gqa_kernel_sequence=gqa_sequence,
            mha_kernel_sequence=mha_sequence,
            evidence_sha256=digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "evidence_sha256": self.evidence_sha256}


def revalidate_geometry_bound_raw_traces(
    *,
    point: Phase3DispatchPointBinding,
    gqa_artifact: RawTraceArtifact,
    mha_artifact: RawTraceArtifact,
    gqa_raw: bytes,
    mha_raw: bytes,
) -> GeometryBoundRawTraceValidationEvidence:
    """Purely reparse retained raw traces and rebuild all dispatch evidence."""

    execution_mode = (
        CUDA_GRAPH_REPLAY_EXECUTION_MODE
        if point.graph_mode == "cuda_graph"
        else EAGER_EXECUTION_MODE
    )
    for label, artifact, raw in (
        ("gqa", gqa_artifact, gqa_raw),
        ("mha_control", mha_artifact, mha_raw),
    ):
        if artifact.execution_mode != execution_mode:
            raise GQADeviceDispatchError(
                f"{label} raw trace execution mode differs from point"
            )
        if len(raw) != artifact.size_bytes:
            raise GQADeviceDispatchError(f"{label} raw trace size differs")
        if hashlib.sha256(raw).hexdigest() != artifact.sha256:
            raise GQADeviceDispatchError(f"{label} raw trace digest differs")
    parser = (
        parse_scoped_chrome_cuda_graph_events
        if execution_mode == CUDA_GRAPH_REPLAY_EXECUTION_MODE
        else parse_scoped_chrome_cuda_events
    )
    if execution_mode == EAGER_EXECUTION_MODE:
        gqa_scoped = parser(
            gqa_raw,
            marker=_geometry_bound_trace_marker(point, "gqa"),
            require_kernel_launch_runtime=True,
        )
        mha_scoped = parser(
            mha_raw,
            marker=_geometry_bound_trace_marker(point, "mha_control"),
            require_kernel_launch_runtime=True,
        )
    else:
        gqa_scoped = parser(
            gqa_raw,
            marker=_geometry_bound_trace_marker(point, "gqa"),
        )
        mha_scoped = parser(
            mha_raw,
            marker=_geometry_bound_trace_marker(point, "mha_control"),
        )
    return GeometryBoundRawTraceValidationEvidence.create(
        execution_mode=execution_mode,
        gqa_raw_sha256=gqa_artifact.sha256,
        gqa_raw_size_bytes=gqa_artifact.size_bytes,
        mha_raw_sha256=mha_artifact.sha256,
        mha_raw_size_bytes=mha_artifact.size_bytes,
        gqa_scope=gqa_scoped.scope,
        mha_scope=mha_scoped.scope,
        gqa_device_events=gqa_scoped.device_events,
        mha_device_events=mha_scoped.device_events,
    )


@dataclass(frozen=True, slots=True)
class Phase3GeometryBoundGQADeviceDispatchAudit:
    """Dispatch-only proof bound to one production Phase 3 cache geometry."""

    point: Phase3DispatchPointBinding
    backend_identity: BackendIdentityEvidence
    gqa: DispatchControlEvidence
    mha: DispatchControlEvidence
    source_shape: GeometryBoundSourceShapeEvidence
    gqa_kernel_sequence: FlashKernelSequenceEvidence
    mha_kernel_sequence: FlashKernelSequenceEvidence
    trace_validation: GeometryBoundRawTraceValidationEvidence
    explicitly_related_families: tuple[tuple[str, str], ...]
    related_family_policy_sha256: str | None
    evaluation: GQAProofEvaluation

    def __post_init__(self) -> None:
        expected_execution_mode = (
            CUDA_GRAPH_REPLAY_EXECUTION_MODE
            if self.point.graph_mode == "cuda_graph"
            else EAGER_EXECUTION_MODE
        )
        if self.gqa.role != "gqa" or self.mha.role != "mha_control":
            raise ValueError("geometry-bound controls have incorrect roles")
        if (
            self.gqa.batch_size != self.point.batch_size
            or self.mha.batch_size != self.point.batch_size
            or self.gqa.context_length != self.point.active_context
            or self.mha.context_length != self.point.active_context
            or self.gqa.query_length != 1
            or self.mha.query_length != 1
            or self.gqa.is_causal
            or self.mha.is_causal
            or self.gqa.execution_mode != expected_execution_mode
            or self.mha.execution_mode != expected_execution_mode
        ):
            raise ValueError("geometry-bound controls differ from the Phase 3 point")
        cache = self.source_shape.cache
        if (
            cache.active_context != self.point.active_context
            or cache.layout.batch_size != self.point.batch_size
            or cache.layout.capacity != self.point.capacity
        ):
            raise ValueError("static-cache evidence differs from the point binding")
        if self.gqa.raw_trace is None or self.mha.raw_trace is None:
            raise ValueError("geometry-bound raw traces are absent")
        if self.gqa.trace_scope is None or self.mha.trace_scope is None:
            raise ValueError("geometry-bound trace scopes are absent")
        if self.gqa.trace_scope.marker != _geometry_bound_trace_marker(
            self.point, "gqa"
        ):
            raise ValueError("GQA trace marker differs from run/point binding")
        if self.mha.trace_scope.marker != _geometry_bound_trace_marker(
            self.point, "mha_control"
        ):
            raise ValueError("MHA trace marker differs from run/point binding")
        source_paths = tuple(
            item.relative_path for item in self.source_shape.sources
        )
        if source_paths != REQUIRED_SUT_SOURCES:
            raise ValueError("geometry-bound source paths differ from the SUT")
        fingerprints = {
            self.gqa.backend.source_build_fingerprint,
            self.mha.backend.source_build_fingerprint,
        }
        if fingerprints != {self.backend_identity.sha256}:
            raise ValueError("geometry-bound backend controls differ from identity")
        operation_key = self.point.operation_key
        if operation_key.backend_identity_sha256 != self.backend_identity.sha256:
            raise ValueError("dispatch backend differs from operation key")
        if (
            operation_key.cache_layout_fingerprint
            != cache.layout.layout_fingerprint
        ):
            raise ValueError("dispatch cache layout differs from operation key")
        source_identity = phase3_source_identity_sha256(
            {
                item.relative_path: item.sha256
                for item in self.source_shape.sources
            }
        )
        if operation_key.source_identity_sha256 != source_identity:
            raise ValueError("dispatch source bundle differs from operation key")
        if self.explicitly_related_families != tuple(
            sorted(set(self.explicitly_related_families))
        ):
            raise ValueError("related-family policy pairs are not canonical")
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(item, str) or not item for item in pair)
            for pair in self.explicitly_related_families
        ):
            raise ValueError("related-family policy pair is malformed")
        allowed_pairs = set(PHASE3_FLASH_RELATED_FAMILIES)
        if any(
            pair not in allowed_pairs and (pair[1], pair[0]) not in allowed_pairs
            for pair in self.explicitly_related_families
        ):
            raise ValueError("related-family policy contains an unsupported pair")
        if self.related_family_policy_sha256 is not None and not (
            _SHA256_RE.fullmatch(self.related_family_policy_sha256)
        ):
            raise ValueError("related-family policy SHA-256 is invalid")
        if bool(self.explicitly_related_families) != bool(
            self.related_family_policy_sha256
        ):
            raise ValueError(
                "related families and decision-record digest must coexist"
            )
        if self.explicitly_related_families:
            raise ValueError(
                "no checksum-pinned related-family decision is approved"
            )
        if (
            self.trace_validation.execution_mode != expected_execution_mode
            or self.trace_validation.gqa_raw_sha256 != self.gqa.raw_trace.sha256
            or self.trace_validation.gqa_raw_size_bytes
            != self.gqa.raw_trace.size_bytes
            or self.trace_validation.mha_raw_sha256 != self.mha.raw_trace.sha256
            or self.trace_validation.mha_raw_size_bytes
            != self.mha.raw_trace.size_bytes
            or self.trace_validation.gqa_scope != self.gqa.trace_scope
            or self.trace_validation.mha_scope != self.mha.trace_scope
            or self.trace_validation.gqa_device_events != self.gqa.device_events
            or self.trace_validation.mha_device_events != self.mha.device_events
        ):
            raise ValueError("raw-derived trace validation differs from controls")
        if analyze_flash_kernel_sequence(self.gqa.device_events) != (
            self.gqa_kernel_sequence
        ):
            raise ValueError("GQA kernel sequence is not reproducible")
        if analyze_flash_kernel_sequence(self.mha.device_events) != (
            self.mha_kernel_sequence
        ):
            raise ValueError("MHA kernel sequence is not reproducible")
        expected = evaluate_geometry_bound_gqa_device_dispatch(
            point=self.point,
            gqa=self.gqa,
            mha=self.mha,
            gqa_sequence=self.gqa_kernel_sequence,
            mha_sequence=self.mha_kernel_sequence,
            source_shape=self.source_shape,
            explicitly_related_families=set(
                self.explicitly_related_families
            ),
        )
        if expected != self.evaluation:
            raise ValueError("geometry-bound dispatch evaluation is not reproducible")
        if self.evaluation.allocation_verified or self.evaluation.verdict is (
            GQAVerdict.NONMATERIALIZATION_VERIFIED
        ):
            raise ValueError("dispatch-only evidence cannot verify allocation")

    def to_dict(self) -> dict[str, object]:
        assert self.gqa.raw_trace is not None
        assert self.mha.raw_trace is not None
        return {
            "schema_version": (
                "kvbench-phase3-geometry-bound-gqa-device-dispatch-audit-1.0.0"
            ),
            "run_kind": "dispatch_audit",
            "measurement_scope": "phase3_point_bound_static_cache_operator_controls",
            "profiler_instrumented": True,
            "performance_timing_reported": False,
            "benchmark_timing_eligible": False,
            "point_binding": self.point.to_dict(),
            "backend_identity": self.backend_identity.to_dict(),
            "source_shape": self.source_shape.to_dict(
                gqa=self.gqa,
                mha=self.mha,
            ),
            "raw_trace_validation": self.trace_validation.to_dict(),
            "raw_trace_bytes_verified": True,
            "strict_marker_dispatch_launch_correlation_verified": True,
            "dispatch_trace_sha256": {
                "gqa": self.gqa.raw_trace.sha256,
                "mha_control": self.mha.raw_trace.sha256,
            },
            "gqa": self.gqa.to_dict(),
            "mha_control": self.mha.to_dict(),
            "kernel_sequences": {
                "gqa": self.gqa_kernel_sequence.to_dict(),
                "mha_control": self.mha_kernel_sequence.to_dict(),
            },
            "kernel_family_policy": {
                "standard_family": FLASH_FORWARD_FAMILY,
                "split_k_family": FLASH_SPLIT_KV_FAMILY,
                "split_k_contract": (
                    "exactly_one_forward_followed_by_exactly_one_combine"
                ),
                "explicitly_related_pairs": [
                    list(pair) for pair in self.explicitly_related_families
                ],
                "candidate_related_pairs": [
                    list(pair) for pair in PHASE3_FLASH_RELATED_FAMILIES
                ],
                "decision_record_sha256": self.related_family_policy_sha256,
                "decision_approved": bool(self.explicitly_related_families),
                "unrelated_or_ambiguous_rejected": True,
            },
            "allocation_size_proof": {
                "verified": False,
                "dispatch_trace_hash_binding_required": True,
                "binding_owner": "phase3_task_c_allocation_attribution",
                "raw_derived_binding_inputs": {
                    "trace_validation_sha256": (
                        self.trace_validation.evidence_sha256
                    ),
                    "gqa_trace": {
                        "relative_path": self.gqa.raw_trace.relative_path,
                        "sha256": self.gqa.raw_trace.sha256,
                        "size_bytes": self.gqa.raw_trace.size_bytes,
                    },
                    "mha_control_trace": {
                        "relative_path": self.mha.raw_trace.relative_path,
                        "sha256": self.mha.raw_trace.sha256,
                        "size_bytes": self.mha.raw_trace.size_bytes,
                    },
                    "gqa_kv_bytes": self.gqa.byte_evidence.to_dict(),
                    "mha_control_kv_bytes": self.mha.byte_evidence.to_dict(),
                },
            },
            "evaluation": self.evaluation.to_dict(),
        }


PHASE3_GEOMETRY_BOUND_DISPATCH_OBSERVATION_SCHEMA = (
    "kvbench-phase3-geometry-bound-dispatch-observation-1.0.0"
)
_MAX_CANONICAL_EVIDENCE_JSON_BYTES = 256 * 1024 * 1024
_APPROVED_RELATED_FAMILY_DECISION_SHA256 = frozenset()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_canonical_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    """Parse one bounded, duplicate-free canonical JSON object."""

    if type(raw) is not bytes or not raw:
        raise GQADeviceDispatchError(f"{label} bytes are absent")
    if len(raw) > _MAX_CANONICAL_EVIDENCE_JSON_BYTES:
        raise GQADeviceDispatchError(f"{label} exceeds the size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GQADeviceDispatchError(f"{label} is not UTF-8") from error

    def reject_constant(value: str) -> None:
        raise GQADeviceDispatchError(
            f"{label} contains a non-finite constant: {value}"
        )

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        observed: dict[str, object] = {}
        for key, value in pairs:
            if key in observed:
                raise GQADeviceDispatchError(
                    f"{label} contains duplicate key: {key}"
                )
            observed[key] = value
        return observed

    try:
        payload = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except GQADeviceDispatchError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise GQADeviceDispatchError(f"{label} is malformed JSON") from error
    if type(payload) is not dict:
        raise GQADeviceDispatchError(f"{label} must be a JSON object")
    if _canonical_json_bytes(payload) != raw:
        raise GQADeviceDispatchError(f"{label} bytes are not canonical")
    return payload


def _require_exact_json_keys(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if frozenset(payload) != expected:
        raise GQADeviceDispatchError(f"{label} has an unexpected field set")


def _json_object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise GQADeviceDispatchError(f"{label} must be an object")
    return value


def _json_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise GQADeviceDispatchError(f"{label} must be a non-empty string")
    return value


def _json_optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _json_string(value, label=label)


def _json_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise GQADeviceDispatchError(f"{label} must be boolean")
    return value


def _json_integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GQADeviceDispatchError(f"{label} must be an integer >= {minimum}")
    return value


def _json_string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise GQADeviceDispatchError(f"{label} must be an array")
    result = tuple(
        _json_string(item, label=f"{label} entry")
        for item in value
    )
    return result


def _json_integer_tuple(
    value: object,
    *,
    label: str,
    minimum: int,
) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise GQADeviceDispatchError(f"{label} must be a non-empty array")
    return tuple(
        _json_integer(item, label=f"{label} entry", minimum=minimum)
        for item in value
    )


@dataclass(frozen=True, slots=True)
class BackendControlObservation:
    """Nonderived outputs of one forced-backend control."""

    enabled_backends: tuple[str, ...]
    flash_eligible: bool
    fused_backend_name: str | None
    rejected_control_failed: bool
    rejected_control_error: str | None
    rejected_control_warnings: tuple[str, ...]
    rejected_control_synchronized: bool
    source_build_verified: bool
    eligibility_diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "flash_eligible",
            "rejected_control_failed",
            "rejected_control_synchronized",
            "source_build_verified",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError("backend-control observation flags must be boolean")
        for values in (
            self.enabled_backends,
            self.rejected_control_warnings,
            self.eligibility_diagnostics,
        ):
            if any(type(item) is not str or not item for item in values):
                raise ValueError("backend-control strings must be non-empty")
        for value in (self.fused_backend_name, self.rejected_control_error):
            if value is not None and (type(value) is not str or not value):
                raise ValueError("optional backend-control strings are invalid")

    @classmethod
    def from_evidence(
        cls,
        evidence: BackendControlEvidence,
    ) -> BackendControlObservation:
        return cls(
            enabled_backends=evidence.enabled_backends,
            flash_eligible=evidence.flash_eligible,
            fused_backend_name=evidence.fused_backend_name,
            rejected_control_failed=evidence.rejected_control_failed,
            rejected_control_error=evidence.rejected_control_error,
            rejected_control_warnings=evidence.rejected_control_warnings,
            rejected_control_synchronized=(
                evidence.rejected_control_synchronized
            ),
            source_build_verified=evidence.source_build_verified,
            eligibility_diagnostics=evidence.eligibility_diagnostics,
        )

    def to_evidence(self, *, backend_identity_sha256: str) -> BackendControlEvidence:
        return BackendControlEvidence(
            enabled_backends=self.enabled_backends,
            flash_eligible=self.flash_eligible,
            fused_backend_name=self.fused_backend_name,
            rejected_control_failed=self.rejected_control_failed,
            rejected_control_error=self.rejected_control_error,
            rejected_control_warnings=self.rejected_control_warnings,
            rejected_control_synchronized=self.rejected_control_synchronized,
            source_build_fingerprint=backend_identity_sha256,
            source_build_verified=self.source_build_verified,
            eligibility_diagnostics=self.eligibility_diagnostics,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled_backends": list(self.enabled_backends),
            "flash_eligible": self.flash_eligible,
            "fused_backend_name": self.fused_backend_name,
            "rejected_control_failed": self.rejected_control_failed,
            "rejected_control_error": self.rejected_control_error,
            "rejected_control_warnings": list(self.rejected_control_warnings),
            "rejected_control_synchronized": self.rejected_control_synchronized,
            "source_build_verified": self.source_build_verified,
            "eligibility_diagnostics": list(self.eligibility_diagnostics),
        }


def _parse_backend_control_observation(
    value: object,
    *,
    label: str,
) -> BackendControlObservation:
    payload = _json_object(value, label=label)
    _require_exact_json_keys(
        payload,
        frozenset(
            {
                "enabled_backends",
                "flash_eligible",
                "fused_backend_name",
                "rejected_control_failed",
                "rejected_control_error",
                "rejected_control_warnings",
                "rejected_control_synchronized",
                "source_build_verified",
                "eligibility_diagnostics",
            }
        ),
        label=label,
    )
    return BackendControlObservation(
        enabled_backends=_json_string_tuple(
            payload["enabled_backends"], label=f"{label}.enabled_backends"
        ),
        flash_eligible=_json_bool(
            payload["flash_eligible"], label=f"{label}.flash_eligible"
        ),
        fused_backend_name=_json_optional_string(
            payload["fused_backend_name"],
            label=f"{label}.fused_backend_name",
        ),
        rejected_control_failed=_json_bool(
            payload["rejected_control_failed"],
            label=f"{label}.rejected_control_failed",
        ),
        rejected_control_error=_json_optional_string(
            payload["rejected_control_error"],
            label=f"{label}.rejected_control_error",
        ),
        rejected_control_warnings=_json_string_tuple(
            payload["rejected_control_warnings"],
            label=f"{label}.rejected_control_warnings",
        ),
        rejected_control_synchronized=_json_bool(
            payload["rejected_control_synchronized"],
            label=f"{label}.rejected_control_synchronized",
        ),
        source_build_verified=_json_bool(
            payload["source_build_verified"],
            label=f"{label}.source_build_verified",
        ),
        eligibility_diagnostics=_json_string_tuple(
            payload["eligibility_diagnostics"],
            label=f"{label}.eligibility_diagnostics",
        ),
    )


def _tensor_observation_dict(tensor: TensorShapeEvidence) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride),
        "dtype": tensor.dtype,
        "device": tensor.device,
        "element_size": tensor.element_size,
        "storage_bytes": tensor.storage_bytes,
        "storage_offset": tensor.storage_offset,
        "is_contiguous": tensor.is_contiguous,
    }


def _parse_tensor_observation(value: object, *, label: str) -> TensorShapeEvidence:
    payload = _json_object(value, label=label)
    _require_exact_json_keys(
        payload,
        frozenset(
            {
                "shape",
                "stride",
                "dtype",
                "device",
                "element_size",
                "storage_bytes",
                "storage_offset",
                "is_contiguous",
            }
        ),
        label=label,
    )
    return TensorShapeEvidence(
        shape=_json_integer_tuple(
            payload["shape"], label=f"{label}.shape", minimum=1
        ),
        stride=_json_integer_tuple(
            payload["stride"], label=f"{label}.stride", minimum=0
        ),
        dtype=_json_string(payload["dtype"], label=f"{label}.dtype"),
        device=_json_string(payload["device"], label=f"{label}.device"),
        element_size=_json_integer(
            payload["element_size"], label=f"{label}.element_size", minimum=1
        ),
        storage_bytes=_json_integer(
            payload["storage_bytes"], label=f"{label}.storage_bytes", minimum=1
        ),
        storage_offset=_json_integer(
            payload["storage_offset"],
            label=f"{label}.storage_offset",
            minimum=0,
        ),
        is_contiguous=_json_bool(
            payload["is_contiguous"], label=f"{label}.is_contiguous"
        ),
    )


def canonical_phase3_geometry_bound_dispatch_audit_bytes(
    audit: Phase3GeometryBoundGQADeviceDispatchAudit,
) -> bytes:
    """Serialize a fully validated derived audit without presentation whitespace."""

    if type(audit) is not Phase3GeometryBoundGQADeviceDispatchAudit:
        raise TypeError("audit must be Phase3GeometryBoundGQADeviceDispatchAudit")
    return _canonical_json_bytes(audit.to_dict())


@dataclass(frozen=True, slots=True)
class Phase3GeometryBoundDispatchObservation:
    """Minimal raw observations that cannot be recovered from retained bytes."""

    schema_version: str
    operation_fingerprint_sha256: str
    derived_audit_sha256: str
    gqa_warmup_count: int
    mha_warmup_count: int
    gqa_trace_relative_path: str
    mha_trace_relative_path: str
    gqa_backend: BackendControlObservation
    mha_backend: BackendControlObservation
    gqa_query: TensorShapeEvidence
    gqa_output: TensorShapeEvidence
    cache_key_backing: TensorShapeEvidence
    cache_value_backing: TensorShapeEvidence
    cache_key_view: TensorShapeEvidence
    cache_value_view: TensorShapeEvidence
    mha_query: TensorShapeEvidence
    mha_key: TensorShapeEvidence
    mha_value: TensorShapeEvidence
    mha_output: TensorShapeEvidence
    cache_workspace_bytes: int
    cache_layer_index: int
    cache_key_backing_storage_ptr: int
    cache_value_backing_storage_ptr: int
    cache_key_view_storage_ptr: int
    cache_value_view_storage_ptr: int
    explicitly_related_families: tuple[tuple[str, str], ...]
    related_family_policy_sha256: str | None

    def __post_init__(self) -> None:
        if self.schema_version != PHASE3_GEOMETRY_BOUND_DISPATCH_OBSERVATION_SCHEMA:
            raise ValueError("dispatch observation schema differs")
        for digest in (
            self.operation_fingerprint_sha256,
            self.derived_audit_sha256,
        ):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("dispatch observation SHA-256 is invalid")
        for count in (self.gqa_warmup_count, self.mha_warmup_count):
            _positive_integer(count, "warmup_count")
        expected_names = (
            (self.gqa_trace_relative_path, "gqa.geometry.chrome.json"),
            (self.mha_trace_relative_path, "mha.geometry.chrome.json"),
        )
        for relative_path, expected_name in expected_names:
            path = PurePosixPath(relative_path)
            if (
                not relative_path
                or path.is_absolute()
                or "." in path.parts
                or ".." in path.parts
                or path.name != expected_name
            ):
                raise ValueError("dispatch observation trace path is invalid")
        if self.gqa_trace_relative_path == self.mha_trace_relative_path:
            raise ValueError("dispatch observation trace paths alias")
        if type(self.gqa_backend) is not BackendControlObservation or type(
            self.mha_backend
        ) is not BackendControlObservation:
            raise ValueError("dispatch backend observations have the wrong type")
        tensors = (
            self.gqa_query,
            self.gqa_output,
            self.cache_key_backing,
            self.cache_value_backing,
            self.cache_key_view,
            self.cache_value_view,
            self.mha_query,
            self.mha_key,
            self.mha_value,
            self.mha_output,
        )
        if any(type(tensor) is not TensorShapeEvidence for tensor in tensors):
            raise ValueError("dispatch tensor observations have the wrong type")
        if any(tensor.device != "cuda:0" for tensor in tensors):
            raise ValueError("dispatch tensor observations must bind to cuda:0")
        if (
            type(self.cache_workspace_bytes) is not int
            or self.cache_workspace_bytes < 0
            or type(self.cache_layer_index) is not int
            or not 0 <= self.cache_layer_index < PHASE3_NUM_LAYERS
        ):
            raise ValueError("dispatch cache scalar observation is invalid")
        pointers = (
            self.cache_key_backing_storage_ptr,
            self.cache_value_backing_storage_ptr,
            self.cache_key_view_storage_ptr,
            self.cache_value_view_storage_ptr,
        )
        for pointer in pointers:
            _positive_integer(pointer, "cache_storage_pointer")
        if (
            self.cache_key_backing_storage_ptr
            != self.cache_key_view_storage_ptr
            or self.cache_value_backing_storage_ptr
            != self.cache_value_view_storage_ptr
            or self.cache_key_backing_storage_ptr
            == self.cache_value_backing_storage_ptr
        ):
            raise ValueError("dispatch cache storage pointers are not bound")
        if self.explicitly_related_families != tuple(
            sorted(set(self.explicitly_related_families))
        ):
            raise ValueError("dispatch related-family pairs are not canonical")
        if any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(item) is not str or not item for item in pair)
            for pair in self.explicitly_related_families
        ):
            raise ValueError("dispatch related-family pair is malformed")
        allowed_pairs = set(PHASE3_FLASH_RELATED_FAMILIES)
        if any(
            pair not in allowed_pairs and (pair[1], pair[0]) not in allowed_pairs
            for pair in self.explicitly_related_families
        ):
            raise ValueError("dispatch related-family pair is unsupported")
        if self.related_family_policy_sha256 is not None and (
            _SHA256_RE.fullmatch(self.related_family_policy_sha256) is None
        ):
            raise ValueError("dispatch related-family decision SHA-256 is invalid")
        if bool(self.explicitly_related_families) != bool(
            self.related_family_policy_sha256
        ):
            raise ValueError(
                "dispatch related families and decision digest must coexist"
            )

    @classmethod
    def from_audit(
        cls,
        audit: Phase3GeometryBoundGQADeviceDispatchAudit,
    ) -> Phase3GeometryBoundDispatchObservation:
        if type(audit) is not Phase3GeometryBoundGQADeviceDispatchAudit:
            raise TypeError("audit must be Phase3GeometryBoundGQADeviceDispatchAudit")
        if audit.gqa.raw_trace is None or audit.mha.raw_trace is None:
            raise GQADeviceDispatchError("dispatch audit raw trace metadata is absent")
        cache = audit.source_shape.cache
        audit_raw = canonical_phase3_geometry_bound_dispatch_audit_bytes(audit)
        return cls(
            schema_version=PHASE3_GEOMETRY_BOUND_DISPATCH_OBSERVATION_SCHEMA,
            operation_fingerprint_sha256=(
                audit.point.operation_fingerprint_sha256
            ),
            derived_audit_sha256=hashlib.sha256(audit_raw).hexdigest(),
            gqa_warmup_count=audit.gqa.warmup_count,
            mha_warmup_count=audit.mha.warmup_count,
            gqa_trace_relative_path=audit.gqa.raw_trace.relative_path,
            mha_trace_relative_path=audit.mha.raw_trace.relative_path,
            gqa_backend=BackendControlObservation.from_evidence(
                audit.gqa.backend
            ),
            mha_backend=BackendControlObservation.from_evidence(
                audit.mha.backend
            ),
            gqa_query=audit.source_shape.gqa_query,
            gqa_output=audit.source_shape.gqa_output,
            cache_key_backing=cache.key_backing,
            cache_value_backing=cache.value_backing,
            cache_key_view=cache.key_view,
            cache_value_view=cache.value_view,
            mha_query=audit.source_shape.mha_query,
            mha_key=audit.source_shape.mha_key,
            mha_value=audit.source_shape.mha_value,
            mha_output=audit.source_shape.mha_output,
            cache_workspace_bytes=cache.layout.workspace_bytes,
            cache_layer_index=cache.layer_index,
            cache_key_backing_storage_ptr=cache.key_backing_storage_ptr,
            cache_value_backing_storage_ptr=cache.value_backing_storage_ptr,
            cache_key_view_storage_ptr=cache.key_view_storage_ptr,
            cache_value_view_storage_ptr=cache.value_view_storage_ptr,
            explicitly_related_families=audit.explicitly_related_families,
            related_family_policy_sha256=audit.related_family_policy_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation_fingerprint_sha256": (
                self.operation_fingerprint_sha256
            ),
            "derived_audit_sha256": self.derived_audit_sha256,
            "warmup_counts": {
                "gqa": self.gqa_warmup_count,
                "mha_control": self.mha_warmup_count,
            },
            "trace_relative_paths": {
                "gqa": self.gqa_trace_relative_path,
                "mha_control": self.mha_trace_relative_path,
            },
            "backend_controls": {
                "gqa": self.gqa_backend.to_dict(),
                "mha_control": self.mha_backend.to_dict(),
            },
            "tensor_observations": {
                "gqa_query": _tensor_observation_dict(self.gqa_query),
                "gqa_output": _tensor_observation_dict(self.gqa_output),
                "cache_key_backing": _tensor_observation_dict(
                    self.cache_key_backing
                ),
                "cache_value_backing": _tensor_observation_dict(
                    self.cache_value_backing
                ),
                "cache_key_view": _tensor_observation_dict(self.cache_key_view),
                "cache_value_view": _tensor_observation_dict(
                    self.cache_value_view
                ),
                "mha_query": _tensor_observation_dict(self.mha_query),
                "mha_key": _tensor_observation_dict(self.mha_key),
                "mha_value": _tensor_observation_dict(self.mha_value),
                "mha_output": _tensor_observation_dict(self.mha_output),
            },
            "cache_observation": {
                "workspace_bytes": self.cache_workspace_bytes,
                "layer_index": self.cache_layer_index,
                "storage_pointers": {
                    "key_backing": self.cache_key_backing_storage_ptr,
                    "value_backing": self.cache_value_backing_storage_ptr,
                    "key_view": self.cache_key_view_storage_ptr,
                    "value_view": self.cache_value_view_storage_ptr,
                },
            },
            "kernel_family_policy": {
                "explicitly_related_pairs": [
                    list(pair) for pair in self.explicitly_related_families
                ],
                "decision_record_sha256": self.related_family_policy_sha256,
            },
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def phase3_geometry_bound_dispatch_observation_bytes(
    audit: Phase3GeometryBoundGQADeviceDispatchAudit,
) -> bytes:
    """Create the canonical nonderived-observation artifact for one audit."""

    return Phase3GeometryBoundDispatchObservation.from_audit(
        audit
    ).canonical_bytes()


def phase3_geometry_bound_dispatch_evidence_bytes(
    audit: Phase3GeometryBoundGQADeviceDispatchAudit,
) -> bytes:
    """Return the single formal ``b011_audit`` raw evidence artifact.

    This artifact is the canonical nonderived observation envelope.  The
    derived audit is deliberately not a second required raw file: validation
    regenerates it and compares its SHA-256 with the digest in this envelope.
    """

    return phase3_geometry_bound_dispatch_observation_bytes(audit)


def _parse_related_family_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not list:
        raise GQADeviceDispatchError(
            "dispatch observation related-family pairs must be an array"
        )
    pairs: list[tuple[str, str]] = []
    for raw_pair in value:
        if type(raw_pair) is not list or len(raw_pair) != 2:
            raise GQADeviceDispatchError(
                "dispatch observation related-family pair is malformed"
            )
        pairs.append(
            (
                _json_string(raw_pair[0], label="related family"),
                _json_string(raw_pair[1], label="related family"),
            )
        )
    return tuple(pairs)


def parse_phase3_geometry_bound_dispatch_observation_bytes(
    raw: bytes,
) -> Phase3GeometryBoundDispatchObservation:
    """Strictly parse canonical, duplicate-free raw observation bytes."""

    payload = _strict_canonical_json_object(
        raw,
        label="Phase 3 dispatch observation",
    )
    _require_exact_json_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "operation_fingerprint_sha256",
                "derived_audit_sha256",
                "warmup_counts",
                "trace_relative_paths",
                "backend_controls",
                "tensor_observations",
                "cache_observation",
                "kernel_family_policy",
            }
        ),
        label="Phase 3 dispatch observation",
    )
    warmups = _json_object(payload["warmup_counts"], label="warmup_counts")
    paths = _json_object(
        payload["trace_relative_paths"], label="trace_relative_paths"
    )
    backends = _json_object(
        payload["backend_controls"], label="backend_controls"
    )
    tensors = _json_object(
        payload["tensor_observations"], label="tensor_observations"
    )
    cache = _json_object(
        payload["cache_observation"], label="cache_observation"
    )
    policy = _json_object(
        payload["kernel_family_policy"], label="kernel_family_policy"
    )
    _require_exact_json_keys(
        warmups, frozenset({"gqa", "mha_control"}), label="warmup_counts"
    )
    _require_exact_json_keys(
        paths,
        frozenset({"gqa", "mha_control"}),
        label="trace_relative_paths",
    )
    _require_exact_json_keys(
        backends,
        frozenset({"gqa", "mha_control"}),
        label="backend_controls",
    )
    tensor_names = frozenset(
        {
            "gqa_query",
            "gqa_output",
            "cache_key_backing",
            "cache_value_backing",
            "cache_key_view",
            "cache_value_view",
            "mha_query",
            "mha_key",
            "mha_value",
            "mha_output",
        }
    )
    _require_exact_json_keys(tensors, tensor_names, label="tensor_observations")
    _require_exact_json_keys(
        cache,
        frozenset({"workspace_bytes", "layer_index", "storage_pointers"}),
        label="cache_observation",
    )
    pointers = _json_object(
        cache["storage_pointers"], label="cache_observation.storage_pointers"
    )
    _require_exact_json_keys(
        pointers,
        frozenset({"key_backing", "value_backing", "key_view", "value_view"}),
        label="cache_observation.storage_pointers",
    )
    _require_exact_json_keys(
        policy,
        frozenset({"explicitly_related_pairs", "decision_record_sha256"}),
        label="kernel_family_policy",
    )
    try:
        return Phase3GeometryBoundDispatchObservation(
            schema_version=_json_string(
                payload["schema_version"], label="schema_version"
            ),
            operation_fingerprint_sha256=_json_string(
                payload["operation_fingerprint_sha256"],
                label="operation_fingerprint_sha256",
            ),
            derived_audit_sha256=_json_string(
                payload["derived_audit_sha256"],
                label="derived_audit_sha256",
            ),
            gqa_warmup_count=_json_integer(
                warmups["gqa"], label="warmup_counts.gqa", minimum=1
            ),
            mha_warmup_count=_json_integer(
                warmups["mha_control"],
                label="warmup_counts.mha_control",
                minimum=1,
            ),
            gqa_trace_relative_path=_json_string(
                paths["gqa"], label="trace_relative_paths.gqa"
            ),
            mha_trace_relative_path=_json_string(
                paths["mha_control"],
                label="trace_relative_paths.mha_control",
            ),
            gqa_backend=_parse_backend_control_observation(
                backends["gqa"], label="backend_controls.gqa"
            ),
            mha_backend=_parse_backend_control_observation(
                backends["mha_control"],
                label="backend_controls.mha_control",
            ),
            gqa_query=_parse_tensor_observation(
                tensors["gqa_query"], label="tensor_observations.gqa_query"
            ),
            gqa_output=_parse_tensor_observation(
                tensors["gqa_output"], label="tensor_observations.gqa_output"
            ),
            cache_key_backing=_parse_tensor_observation(
                tensors["cache_key_backing"],
                label="tensor_observations.cache_key_backing",
            ),
            cache_value_backing=_parse_tensor_observation(
                tensors["cache_value_backing"],
                label="tensor_observations.cache_value_backing",
            ),
            cache_key_view=_parse_tensor_observation(
                tensors["cache_key_view"],
                label="tensor_observations.cache_key_view",
            ),
            cache_value_view=_parse_tensor_observation(
                tensors["cache_value_view"],
                label="tensor_observations.cache_value_view",
            ),
            mha_query=_parse_tensor_observation(
                tensors["mha_query"], label="tensor_observations.mha_query"
            ),
            mha_key=_parse_tensor_observation(
                tensors["mha_key"], label="tensor_observations.mha_key"
            ),
            mha_value=_parse_tensor_observation(
                tensors["mha_value"], label="tensor_observations.mha_value"
            ),
            mha_output=_parse_tensor_observation(
                tensors["mha_output"], label="tensor_observations.mha_output"
            ),
            cache_workspace_bytes=_json_integer(
                cache["workspace_bytes"],
                label="cache_observation.workspace_bytes",
                minimum=0,
            ),
            cache_layer_index=_json_integer(
                cache["layer_index"],
                label="cache_observation.layer_index",
                minimum=0,
            ),
            cache_key_backing_storage_ptr=_json_integer(
                pointers["key_backing"],
                label="cache_observation.storage_pointers.key_backing",
                minimum=1,
            ),
            cache_value_backing_storage_ptr=_json_integer(
                pointers["value_backing"],
                label="cache_observation.storage_pointers.value_backing",
                minimum=1,
            ),
            cache_key_view_storage_ptr=_json_integer(
                pointers["key_view"],
                label="cache_observation.storage_pointers.key_view",
                minimum=1,
            ),
            cache_value_view_storage_ptr=_json_integer(
                pointers["value_view"],
                label="cache_observation.storage_pointers.value_view",
                minimum=1,
            ),
            explicitly_related_families=_parse_related_family_pairs(
                policy["explicitly_related_pairs"]
            ),
            related_family_policy_sha256=_json_optional_string(
                policy["decision_record_sha256"],
                label="kernel_family_policy.decision_record_sha256",
            ),
        )
    except (TypeError, ValueError) as error:
        raise GQADeviceDispatchError(
            "Phase 3 dispatch observation is invalid"
        ) from error


def _validate_related_family_decision_raw(
    observation: Phase3GeometryBoundDispatchObservation,
    related_family_decision_raw: bytes | None,
) -> None:
    if not observation.explicitly_related_families:
        if related_family_decision_raw is not None:
            raise GQADeviceDispatchError(
                "related-family decision bytes exist without a requested policy"
            )
        return
    if related_family_decision_raw is None:
        raise GQADeviceDispatchError(
            "related-family policy lacks exact decision-record bytes"
        )
    decision = _strict_canonical_json_object(
        related_family_decision_raw,
        label="related-family decision",
    )
    _require_exact_json_keys(
        decision,
        frozenset(
            {
                "schema_version",
                "scope",
                "approved",
                "explicitly_related_pairs",
            }
        ),
        label="related-family decision",
    )
    decision_pairs = _parse_related_family_pairs(
        decision["explicitly_related_pairs"]
    )
    if (
        decision["schema_version"]
        != "kvbench-phase3-related-kernel-family-decision-1.0.0"
        or decision["scope"] != "phase3_geometry_bound_dispatch"
        or type(decision["approved"]) is not bool
        or decision["approved"] is not True
        or decision_pairs != observation.explicitly_related_families
    ):
        raise GQADeviceDispatchError(
            "related-family decision does not approve the exact policy"
        )
    digest = hashlib.sha256(related_family_decision_raw).hexdigest()
    if digest != observation.related_family_policy_sha256:
        raise GQADeviceDispatchError(
            "related-family decision digest differs from observation"
        )
    if digest not in _APPROVED_RELATED_FAMILY_DECISION_SHA256:
        raise GQADeviceDispatchError(
            "related-family decision is not checksum-approved by this build"
        )


def revalidate_phase3_geometry_bound_dispatch_audit_from_raw(
    *,
    observation_raw: bytes,
    operation_key: Phase3AuditOperationKey,
    gqa_raw: bytes,
    mha_raw: bytes,
    backend_identity_raw: bytes,
    source_bytes_by_path: Mapping[str, bytes],
    audit_raw: bytes | None = None,
    related_family_decision_raw: bytes | None = None,
) -> Phase3GeometryBoundGQADeviceDispatchAudit:
    """Rebuild an audit exclusively from canonical observations and raw bytes.

    The optional serialized audit supplies no trusted verdict, boolean, parsed
    event, source finding, or shape result.  The formal single-file contract
    stores ``observation_raw`` as ``b011_audit``; the derived audit is regenerated
    and its SHA-256 must match the digest recorded there.  ``audit_raw`` exists
    only for callers that also want exact equality with a presentation artifact.
    """

    if type(operation_key) is not Phase3AuditOperationKey:
        raise TypeError("operation_key must be Phase3AuditOperationKey")
    for label, raw in (
        ("GQA dispatch trace", gqa_raw),
        ("MHA dispatch trace", mha_raw),
        ("backend identity", backend_identity_raw),
    ):
        if type(raw) is not bytes or not raw:
            raise GQADeviceDispatchError(f"{label} bytes are absent")
    if audit_raw is not None:
        _strict_canonical_json_object(
            audit_raw,
            label="derived dispatch audit",
        )
    observation = parse_phase3_geometry_bound_dispatch_observation_bytes(
        observation_raw
    )
    if (
        observation.operation_fingerprint_sha256
        != operation_key.operation_fingerprint_sha256
    ):
        raise GQADeviceDispatchError(
            "dispatch observation differs from the supplied operation key"
        )
    if audit_raw is not None and hashlib.sha256(audit_raw).hexdigest() != (
        observation.derived_audit_sha256
    ):
        raise GQADeviceDispatchError(
            "derived dispatch audit differs from the recorded SHA-256"
        )
    _validate_related_family_decision_raw(
        observation,
        related_family_decision_raw,
    )

    backend_payload = _strict_canonical_json_object(
        backend_identity_raw,
        label="backend identity",
    )
    try:
        backend_identity = BackendIdentityEvidence.from_payload(backend_payload)
    except (TypeError, ValueError) as error:
        raise GQADeviceDispatchError("backend identity is invalid") from error
    if backend_identity.canonical_json.encode("utf-8") != backend_identity_raw:
        raise GQADeviceDispatchError("backend identity bytes changed on rebuild")
    if operation_key.backend_identity_sha256 != backend_identity.sha256:
        raise GQADeviceDispatchError(
            "backend identity differs from the operation key"
        )

    if not isinstance(source_bytes_by_path, Mapping) or set(
        source_bytes_by_path
    ) != set(REQUIRED_SUT_SOURCES):
        raise GQADeviceDispatchError("raw source bundle paths are incomplete")
    retained_source_bytes: dict[str, bytes] = {}
    for path in REQUIRED_SUT_SOURCES:
        raw = source_bytes_by_path[path]
        if type(raw) is not bytes or not raw:
            raise GQADeviceDispatchError(
                "raw source bundle contains absent or non-byte content"
            )
        retained_source_bytes[path] = raw
    try:
        sources = tuple(
            source_file_evidence_from_bytes(path, retained_source_bytes[path])
            for path in REQUIRED_SUT_SOURCES
        )
        source_identity = phase3_source_identity_sha256(
            {source.relative_path: source.sha256 for source in sources}
        )
    except (TypeError, ValueError) as error:
        raise GQADeviceDispatchError("raw source bundle is invalid") from error
    if operation_key.source_identity_sha256 != source_identity:
        raise GQADeviceDispatchError(
            "raw source bundle differs from the operation key"
        )

    point = Phase3DispatchPointBinding.create(operation_key=operation_key)
    static_cache_source = next(
        source
        for source in sources
        if source.relative_path == REQUIRED_SUT_SOURCES[2]
    )
    try:
        cache_layout = StaticCacheLayoutEvidence.create(
            batch_size=operation_key.batch_size,
            capacity=operation_key.capacity,
            device="cuda:0",
            workspace_bytes=observation.cache_workspace_bytes,
            implementation_sha256=static_cache_source.sha256,
            layout_fingerprint=operation_key.cache_layout_fingerprint,
        )
        cache = StaticCacheViewBindingEvidence(
            layout=cache_layout,
            layer_index=observation.cache_layer_index,
            active_context=operation_key.attended_context,
            key_backing=observation.cache_key_backing,
            value_backing=observation.cache_value_backing,
            key_view=observation.cache_key_view,
            value_view=observation.cache_value_view,
            key_backing_storage_ptr=(
                observation.cache_key_backing_storage_ptr
            ),
            value_backing_storage_ptr=(
                observation.cache_value_backing_storage_ptr
            ),
            key_view_storage_ptr=observation.cache_key_view_storage_ptr,
            value_view_storage_ptr=observation.cache_value_view_storage_ptr,
            key_view_shares_backing_storage=(
                observation.cache_key_view_storage_ptr
                == observation.cache_key_backing_storage_ptr
            ),
            value_view_shares_backing_storage=(
                observation.cache_value_view_storage_ptr
                == observation.cache_value_backing_storage_ptr
            ),
            key_value_backing_storages_distinct=(
                observation.cache_key_backing_storage_ptr
                != observation.cache_value_backing_storage_ptr
            ),
        )
    except (TypeError, ValueError) as error:
        raise GQADeviceDispatchError(
            "cache observation differs from operation/source identity"
        ) from error

    execution_mode = operation_key.dispatch_execution_mode
    gqa_artifact = RawTraceArtifact(
        relative_path=observation.gqa_trace_relative_path,
        sha256=hashlib.sha256(gqa_raw).hexdigest(),
        size_bytes=len(gqa_raw),
        execution_mode=execution_mode,
    )
    mha_artifact = RawTraceArtifact(
        relative_path=observation.mha_trace_relative_path,
        sha256=hashlib.sha256(mha_raw).hexdigest(),
        size_bytes=len(mha_raw),
        execution_mode=execution_mode,
    )
    try:
        trace_validation = revalidate_geometry_bound_raw_traces(
            point=point,
            gqa_artifact=gqa_artifact,
            mha_artifact=mha_artifact,
            gqa_raw=gqa_raw,
            mha_raw=mha_raw,
        )
    except (TypeError, ValueError) as error:
        raise GQADeviceDispatchError(
            "raw dispatch traces do not satisfy the Phase 3 binding"
        ) from error

    def rebuild_control(
        *,
        role: str,
        warmup_count: int,
        backend: BackendControlObservation,
        artifact: RawTraceArtifact,
        scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence,
        events: tuple[CUDADeviceEvent, ...],
    ) -> DispatchControlEvidence:
        return DispatchControlEvidence(
            role=role,
            batch_size=operation_key.batch_size,
            context_length=operation_key.attended_context,
            query_length=1,
            num_query_heads=PHASE3_NUM_QUERY_HEADS,
            num_kv_heads=(
                PHASE3_NUM_KV_HEADS
                if role == "gqa"
                else PHASE3_NUM_QUERY_HEADS
            ),
            head_dim=PHASE3_HEAD_DIM,
            dtype=PHASE3_DTYPE,
            dtype_bytes=PHASE3_DTYPE_BYTES,
            is_causal=False,
            warmup_count=warmup_count,
            backend=backend.to_evidence(
                backend_identity_sha256=backend_identity.sha256
            ),
            raw_trace=artifact,
            trace_scope=scope,
            device_events=events,
            execution_mode=execution_mode,
        )

    try:
        gqa = rebuild_control(
            role="gqa",
            warmup_count=observation.gqa_warmup_count,
            backend=observation.gqa_backend,
            artifact=gqa_artifact,
            scope=trace_validation.gqa_scope,
            events=trace_validation.gqa_device_events,
        )
        mha = rebuild_control(
            role="mha_control",
            warmup_count=observation.mha_warmup_count,
            backend=observation.mha_backend,
            artifact=mha_artifact,
            scope=trace_validation.mha_scope,
            events=trace_validation.mha_device_events,
        )
        source_shape = GeometryBoundSourceShapeEvidence(
            sources=sources,
            gqa_query=observation.gqa_query,
            gqa_output=observation.gqa_output,
            cache=cache,
            mha_query=observation.mha_query,
            mha_key=observation.mha_key,
            mha_value=observation.mha_value,
            mha_output=observation.mha_output,
        )
        evaluation = evaluate_geometry_bound_gqa_device_dispatch(
            point=point,
            gqa=gqa,
            mha=mha,
            gqa_sequence=trace_validation.gqa_kernel_sequence,
            mha_sequence=trace_validation.mha_kernel_sequence,
            source_shape=source_shape,
            explicitly_related_families=set(
                observation.explicitly_related_families
            ),
        )
        rebuilt = Phase3GeometryBoundGQADeviceDispatchAudit(
            point=point,
            backend_identity=backend_identity,
            gqa=gqa,
            mha=mha,
            source_shape=source_shape,
            gqa_kernel_sequence=trace_validation.gqa_kernel_sequence,
            mha_kernel_sequence=trace_validation.mha_kernel_sequence,
            trace_validation=trace_validation,
            explicitly_related_families=(
                observation.explicitly_related_families
            ),
            related_family_policy_sha256=(
                observation.related_family_policy_sha256
            ),
            evaluation=evaluation,
        )
    except (TypeError, ValueError) as error:
        raise GQADeviceDispatchError(
            "raw observations cannot reconstruct a valid dispatch audit"
        ) from error

    rebuilt_raw = canonical_phase3_geometry_bound_dispatch_audit_bytes(rebuilt)
    if hashlib.sha256(rebuilt_raw).hexdigest() != (
        observation.derived_audit_sha256
    ):
        raise GQADeviceDispatchError(
            "reconstructed dispatch audit differs from recorded SHA-256"
        )
    if audit_raw is not None and rebuilt_raw != audit_raw:
        raise GQADeviceDispatchError(
            "serialized derived fields differ from raw-derived dispatch audit"
        )
    return rebuilt


def revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
    *,
    b011_audit_raw: bytes,
    operation_key: Phase3AuditOperationKey,
    gqa_raw: bytes,
    mha_raw: bytes,
    backend_identity_raw: bytes,
    source_bytes_by_path: Mapping[str, bytes],
    related_family_decision_raw: bytes | None = None,
) -> Phase3GeometryBoundGQADeviceDispatchAudit:
    """Revalidate the formal one-file ``b011_audit`` observation artifact."""

    return revalidate_phase3_geometry_bound_dispatch_audit_from_raw(
        observation_raw=b011_audit_raw,
        operation_key=operation_key,
        gqa_raw=gqa_raw,
        mha_raw=mha_raw,
        backend_identity_raw=backend_identity_raw,
        source_bytes_by_path=source_bytes_by_path,
        related_family_decision_raw=related_family_decision_raw,
    )


_PHASE3_ALLOCATION_JOIN_CLASSES = frozenset(
    {
        "cache_growth",
        "gqa_expansion",
        "context_scaled_workspace",
        "fixed_output",
        "fixed_shared_activation",
        "framework_bookkeeping",
        "audit_instrumentation",
        "unknown",
    }
)


@dataclass(frozen=True, slots=True)
class Phase3AllocationJoinEventFact:
    """One raw-replayed allocation lifetime used by the GQA join."""

    event_index: int
    event_class: str
    requested_bytes: int
    allocated_block_bytes: int | None
    triggered_segment_alloc: bool

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("allocation join event index is invalid")
        if self.event_class not in _PHASE3_ALLOCATION_JOIN_CLASSES:
            raise ValueError("allocation join event class is unsupported")
        _positive_integer(self.requested_bytes, "allocation_requested_bytes")
        if self.allocated_block_bytes is not None:
            _positive_integer(
                self.allocated_block_bytes,
                "allocation_block_bytes",
            )
        if type(self.triggered_segment_alloc) is not bool:
            raise ValueError("allocation segment flag must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "event_class": self.event_class,
            "requested_bytes": self.requested_bytes,
            "allocated_block_bytes": self.allocated_block_bytes,
            "triggered_segment_alloc": self.triggered_segment_alloc,
        }


@dataclass(frozen=True, slots=True)
class Phase3AllocationTensorJoinFact:
    """One exact declared tensor reconstructed from the raw witness."""

    tensor_index: int
    role: str
    shape: tuple[int, ...]
    storage_bytes: int

    def __post_init__(self) -> None:
        if type(self.tensor_index) is not int or self.tensor_index < 0:
            raise ValueError("allocation tensor index is invalid")
        if type(self.role) is not str or not self.role:
            raise ValueError("allocation tensor role is invalid")
        require_identifier(self.role, field_name="allocation_tensor_role")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(item) is not int or item <= 0 for item in self.shape)
        ):
            raise ValueError("allocation tensor shape is invalid")
        _positive_integer(self.storage_bytes, "allocation_tensor_storage_bytes")

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_index": self.tensor_index,
            "role": self.role,
            "shape": list(self.shape),
            "storage_bytes": self.storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class Phase3AllocationRawEvidence:
    """The complete canonical B-012 evidence set for one operation.

    These are bytes, not a serialized validation result.  The B-011 join
    replays them through Task C's semantic validator before deriving any
    allocation fact.
    """

    snapshot_raw: bytes
    trace_raw: bytes
    memory_stats_before_raw: bytes
    memory_stats_after_raw: bytes
    memory_accounting_before_raw: bytes
    memory_accounting_after_raw: bytes
    operation_witness_raw: bytes
    audit_raw: bytes
    audit_sha256_ledger_raw: bytes

    def __post_init__(self) -> None:
        for name in (
            "snapshot_raw",
            "trace_raw",
            "memory_stats_before_raw",
            "memory_stats_after_raw",
            "memory_accounting_before_raw",
            "memory_accounting_after_raw",
            "operation_witness_raw",
            "audit_raw",
            "audit_sha256_ledger_raw",
        ):
            value = getattr(self, name)
            if type(value) is not bytes or not value:
                raise ValueError(
                    "allocation raw evidence contains absent/non-byte content"
                )


@dataclass(frozen=True, slots=True, init=False)
class Phase3AllocationJoinFacts:
    """Raw-bound, nonboolean facts emitted by independent Task C replay.

    Construction hashes the exact canonical allocation audit and allocator
    trace.  The class intentionally has no public generated initializer and no
    ``passed``/``allocation_verified`` input.
    """

    operation_key: Phase3AuditOperationKey
    production_binding_sha256: str
    allocation_audit_sha256: str
    allocator_trace_sha256: str
    expected_allocator_trace_sha256: str
    gqa_dispatch_trace_sha256: str
    mha_dispatch_trace_sha256: str
    dispatch_trace_validation_sha256: str
    semantic_failure_reasons: tuple[str, ...]
    trace_integrity_errors: tuple[str, ...]
    criterion_id: str
    criterion_failure_reasons: tuple[str, ...]
    criterion_allocation_event_count: int
    criterion_class_counts: tuple[tuple[str, int], ...]
    allocation_events: tuple[Phase3AllocationJoinEventFact, ...]
    method_tensors: tuple[Phase3AllocationTensorJoinFact, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.operation_key) is not Phase3AuditOperationKey:
            raise ValueError("allocation join operation key is invalid")
        for digest in (
            self.production_binding_sha256,
            self.allocation_audit_sha256,
            self.allocator_trace_sha256,
            self.expected_allocator_trace_sha256,
            self.gqa_dispatch_trace_sha256,
            self.mha_dispatch_trace_sha256,
            self.dispatch_trace_validation_sha256,
            self.evidence_sha256,
        ):
            if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("allocation join digest is invalid")
        for reasons in (
            self.semantic_failure_reasons,
            self.trace_integrity_errors,
            self.criterion_failure_reasons,
        ):
            if (
                type(reasons) is not tuple
                or reasons != tuple(dict.fromkeys(reasons))
                or any(type(item) is not str or not item for item in reasons)
            ):
                raise ValueError("allocation join reasons are not canonical")
        if type(self.criterion_id) is not str or not self.criterion_id:
            raise ValueError("allocation join criterion ID is invalid")
        if (
            type(self.criterion_allocation_event_count) is not int
            or self.criterion_allocation_event_count < 0
            or self.criterion_allocation_event_count
            != len(self.allocation_events)
        ):
            raise ValueError("allocation join event count is inconsistent")
        if any(
            type(event) is not Phase3AllocationJoinEventFact
            for event in self.allocation_events
        ) or tuple(event.event_index for event in self.allocation_events) != tuple(
            range(len(self.allocation_events))
        ):
            raise ValueError("allocation join events are not canonical")
        observed_counts: dict[str, int] = {}
        for event in self.allocation_events:
            observed_counts[event.event_class] = (
                observed_counts.get(event.event_class, 0) + 1
            )
        expected_counts = tuple(sorted(observed_counts.items()))
        if self.criterion_class_counts != expected_counts:
            raise ValueError("allocation join class counts are inconsistent")
        if (
            not self.method_tensors
            or any(
                type(tensor) is not Phase3AllocationTensorJoinFact
                for tensor in self.method_tensors
            )
            or tuple(tensor.tensor_index for tensor in self.method_tensors)
            != tuple(range(len(self.method_tensors)))
        ):
            raise ValueError("allocation tensor inventory is not canonical")
        if self.evidence_sha256 != self._derive_sha256():
            raise ValueError("allocation join evidence SHA-256 differs")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "kvbench-phase3-allocation-gqa-join-facts-1.0.0",
            "operation_key": self.operation_key.to_dict(),
            "production_binding_sha256": self.production_binding_sha256,
            "allocation_audit_sha256": self.allocation_audit_sha256,
            "allocator_trace_sha256": self.allocator_trace_sha256,
            "expected_allocator_trace_sha256": (
                self.expected_allocator_trace_sha256
            ),
            "gqa_dispatch_trace_sha256": self.gqa_dispatch_trace_sha256,
            "mha_dispatch_trace_sha256": self.mha_dispatch_trace_sha256,
            "dispatch_trace_validation_sha256": (
                self.dispatch_trace_validation_sha256
            ),
            "semantic_failure_reasons": list(self.semantic_failure_reasons),
            "trace_integrity_errors": list(self.trace_integrity_errors),
            "criterion_id": self.criterion_id,
            "criterion_failure_reasons": list(
                self.criterion_failure_reasons
            ),
            "criterion_allocation_event_count": (
                self.criterion_allocation_event_count
            ),
            "criterion_class_counts": {
                key: value for key, value in self.criterion_class_counts
            },
            "allocation_events": [
                event.to_dict() for event in self.allocation_events
            ],
            "method_tensors": [
                tensor.to_dict() for tensor in self.method_tensors
            ],
        }

    def _derive_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self._payload())).hexdigest()

    @classmethod
    def from_raw_evidence(
        cls,
        *,
        operation_key: Phase3AuditOperationKey,
        production_binding: object,
        raw_evidence: Phase3AllocationRawEvidence,
        gqa_dispatch_trace_sha256: str,
        mha_dispatch_trace_sha256: str,
        dispatch_trace_validation_sha256: str,
    ) -> Phase3AllocationJoinFacts:
        """Derive join facts by replaying every canonical B-012 raw file."""

        if type(operation_key) is not Phase3AuditOperationKey:
            raise TypeError("operation_key must be Phase3AuditOperationKey")
        if type(raw_evidence) is not Phase3AllocationRawEvidence:
            raise TypeError("raw_evidence must be Phase3AllocationRawEvidence")
        allocation_module = importlib.import_module(
            "kvbench.runtime.allocation_attribution"
        )
        production_binding_type = allocation_module.ProductionAllocationBinding
        if type(production_binding) is not production_binding_type:
            raise TypeError(
                "production_binding must be ProductionAllocationBinding"
            )
        if production_binding.operation_key != operation_key:
            raise GQADeviceDispatchError(
                "allocation binding differs from the joined operation key"
            )
        allocation_audit = _strict_canonical_json_object(
            raw_evidence.audit_raw,
            label="allocation audit",
        )
        if (
            allocation_audit.get("schema_version")
            != allocation_module.PHASE3_ALLOCATION_AUDIT_SCHEMA_VERSION
            or allocation_audit.get("run_kind") != "allocation_audit"
            or allocation_audit.get("evidence_status") != "complete"
            or allocation_audit.get("operation_key")
            != operation_key.to_dict()
        ):
            raise GQADeviceDispatchError(
                "allocation audit is not a complete formal operation envelope"
            )
        raw_files = allocation_audit.get("raw_files")
        if type(raw_files) is not dict:
            raise GQADeviceDispatchError(
                "allocation audit lacks its exact raw-file index"
            )
        expected_raw_file_keys = frozenset(
            {
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
            }
        )
        _require_exact_json_keys(
            raw_files,
            expected_raw_file_keys,
            label="allocation audit raw_files",
        )
        try:
            files = allocation_module.RawAllocatorEvidenceFiles(
                **raw_files,
                audit_sha256=hashlib.sha256(raw_evidence.audit_raw).hexdigest(),
            )
        except (
            TypeError,
            ValueError,
            allocation_module.AllocationAttributionError,
        ) as error:
            raise GQADeviceDispatchError(
                "allocation audit raw-file index is invalid"
            ) from error
        expected_ledger = (
            f"{files.audit_sha256}  {files.audit_file}\n".encode("ascii")
        )
        if raw_evidence.audit_sha256_ledger_raw != expected_ledger:
            raise GQADeviceDispatchError(
                "allocation audit SHA-256 ledger differs from exact audit bytes"
            )

        payload_by_name = {
            files.snapshot_file: raw_evidence.snapshot_raw,
            files.trace_file: raw_evidence.trace_raw,
            files.memory_stats_before_file: (
                raw_evidence.memory_stats_before_raw
            ),
            files.memory_stats_after_file: raw_evidence.memory_stats_after_raw,
            files.memory_accounting_before_file: (
                raw_evidence.memory_accounting_before_raw
            ),
            files.memory_accounting_after_file: (
                raw_evidence.memory_accounting_after_raw
            ),
            files.operation_witness_file: raw_evidence.operation_witness_raw,
            files.audit_file: raw_evidence.audit_raw,
            files.audit_sha256_file: raw_evidence.audit_sha256_ledger_raw,
        }
        if len(payload_by_name) != 9:
            raise GQADeviceDispatchError(
                "allocation raw-file index contains aliased filenames"
            )
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-allocation-replay-",
            dir="/tmp",
        ) as temporary:
            temporary_path = Path(temporary)
            os.chmod(temporary_path, 0o700)
            for filename, raw in payload_by_name.items():
                descriptor = os.open(
                    temporary_path / filename,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    0o600,
                )
                try:
                    view = memoryview(raw)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("short allocation replay write")
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            semantic = (
                allocation_module.validate_preserved_allocator_evidence_semantically(
                    temporary_path,
                    files,
                    production_binding=production_binding,
                )
            )
        if (
            semantic.attribution is None
            or semantic.criterion is None
            or semantic.memory is None
        ):
            raise GQADeviceDispatchError(
                "complete B-012 raw evidence could not be semantically replayed"
            )
        attribution = semantic.attribution
        criterion = semantic.criterion
        expected_trace_sha256 = attribution.expected_trace_sha256
        if (
            type(expected_trace_sha256) is not str
            or _SHA256_RE.fullmatch(expected_trace_sha256) is None
        ):
            raise GQADeviceDispatchError(
                "semantic allocation replay lacks an expected trace digest"
            )
        if attribution.trace_sha256 != hashlib.sha256(
            raw_evidence.trace_raw
        ).hexdigest():
            raise GQADeviceDispatchError(
                "semantic allocation replay differs from exact trace bytes"
            )
        try:
            witness_payload = _strict_canonical_json_object(
                raw_evidence.operation_witness_raw,
                label="allocation operation witness",
            )
            witness = allocation_module.OperationWitnessEvidence.from_mapping(
                witness_payload
            )
        except (
            TypeError,
            ValueError,
            allocation_module.AllocationAttributionError,
        ) as error:
            raise GQADeviceDispatchError(
                "allocation operation witness cannot be reconstructed"
            ) from error
        if witness.operation_key != operation_key:
            raise GQADeviceDispatchError(
                "allocation witness differs from the joined operation key"
            )

        allocation_events = tuple(
            Phase3AllocationJoinEventFact(
                event_index=index,
                event_class=allocation.event_class.value,
                requested_bytes=allocation.requested_bytes,
                allocated_block_bytes=allocation.allocated_block_bytes,
                triggered_segment_alloc=allocation.triggered_segment_alloc,
            )
            for index, allocation in enumerate(attribution.allocations)
        )
        cache_shape = witness.reference_before.key_shape
        cache_storage_bytes = math.prod(cache_shape) * PHASE3_DTYPE_BYTES
        output_shape = witness.reference_output.shape
        output_storage_bytes = math.prod(output_shape) * PHASE3_DTYPE_BYTES
        method_tensors = (
            Phase3AllocationTensorJoinFact(
                tensor_index=0,
                role="native_kv_cache_key",
                shape=cache_shape,
                storage_bytes=cache_storage_bytes,
            ),
            Phase3AllocationTensorJoinFact(
                tensor_index=1,
                role="native_kv_cache_value",
                shape=witness.reference_before.value_shape,
                storage_bytes=cache_storage_bytes,
            ),
            Phase3AllocationTensorJoinFact(
                tensor_index=2,
                role="decode_logits",
                shape=output_shape,
                storage_bytes=output_storage_bytes,
            ),
        )

        instance = object.__new__(cls)
        values: dict[str, object] = {
            "operation_key": operation_key,
            "production_binding_sha256": production_binding.identity_sha256,
            "allocation_audit_sha256": hashlib.sha256(
                raw_evidence.audit_raw
            ).hexdigest(),
            "allocator_trace_sha256": attribution.trace_sha256,
            "expected_allocator_trace_sha256": expected_trace_sha256,
            "gqa_dispatch_trace_sha256": gqa_dispatch_trace_sha256,
            "mha_dispatch_trace_sha256": mha_dispatch_trace_sha256,
            "dispatch_trace_validation_sha256": (
                dispatch_trace_validation_sha256
            ),
            "semantic_failure_reasons": tuple(semantic.failure_reasons),
            "trace_integrity_errors": tuple(attribution.integrity_errors),
            "criterion_id": criterion.criterion_id,
            "criterion_failure_reasons": tuple(criterion.failure_reasons),
            "criterion_allocation_event_count": criterion.allocation_event_count,
            "criterion_class_counts": tuple(
                sorted(criterion.class_counts.items())
            ),
            "allocation_events": allocation_events,
            "method_tensors": method_tensors,
            "evidence_sha256": "0" * 64,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "evidence_sha256", instance._derive_sha256())
        instance.__post_init__()
        return instance

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "evidence_sha256": self.evidence_sha256}


def combine_phase3_geometry_bound_gqa_allocation_verdict(
    *,
    dispatch_audit: Phase3GeometryBoundGQADeviceDispatchAudit,
    allocation_facts: Phase3AllocationJoinFacts,
) -> GQAProofEvaluation:
    """Join independently replayed dispatch/allocation evidence fail closed."""

    if type(dispatch_audit) is not Phase3GeometryBoundGQADeviceDispatchAudit:
        raise TypeError(
            "dispatch_audit must be Phase3GeometryBoundGQADeviceDispatchAudit"
        )
    if type(allocation_facts) is not Phase3AllocationJoinFacts:
        raise TypeError("allocation_facts must be Phase3AllocationJoinFacts")
    if allocation_facts.evidence_sha256 != allocation_facts._derive_sha256():
        raise GQADeviceDispatchError("allocation join facts were mutated")
    operation_key = dispatch_audit.point.operation_key
    if allocation_facts.operation_key != operation_key:
        raise GQADeviceDispatchError(
            "dispatch/allocation operation keys differ"
        )
    if dispatch_audit.gqa.raw_trace is None or dispatch_audit.mha.raw_trace is None:
        raise GQADeviceDispatchError("dispatch trace identities are absent")
    expected_trace_binding = (
        dispatch_audit.gqa.raw_trace.sha256,
        dispatch_audit.mha.raw_trace.sha256,
        dispatch_audit.trace_validation.evidence_sha256,
    )
    observed_trace_binding = (
        allocation_facts.gqa_dispatch_trace_sha256,
        allocation_facts.mha_dispatch_trace_sha256,
        allocation_facts.dispatch_trace_validation_sha256,
    )
    if observed_trace_binding != expected_trace_binding:
        raise GQADeviceDispatchError(
            "allocation facts are not bound to the exact dispatch traces"
        )

    dispatch = evaluate_geometry_bound_gqa_device_dispatch(
        point=dispatch_audit.point,
        gqa=dispatch_audit.gqa,
        mha=dispatch_audit.mha,
        gqa_sequence=dispatch_audit.gqa_kernel_sequence,
        mha_sequence=dispatch_audit.mha_kernel_sequence,
        source_shape=dispatch_audit.source_shape,
        explicitly_related_families=set(
            dispatch_audit.explicitly_related_families
        ),
    )
    expanded_single = dispatch_audit.gqa.byte_evidence.expanded_kv_bytes // 2
    expanded_combined = dispatch_audit.gqa.byte_evidence.expanded_kv_bytes
    expanded_sizes = {expanded_single, expanded_combined}
    expected_expanded_shape = (
        operation_key.batch_size,
        PHASE3_NUM_QUERY_HEADS,
        operation_key.attended_context,
        PHASE3_HEAD_DIM,
    )
    allocation_positive: list[str] = []
    for event in allocation_facts.allocation_events:
        if event.event_class == "gqa_expansion":
            allocation_positive.append(
                "allocation_event:gqa_expansion:"
                f"{event.event_index}:{event.requested_bytes}"
            )
        for size_kind, size in (
            ("requested", event.requested_bytes),
            ("allocated_block", event.allocated_block_bytes),
        ):
            if size in expanded_sizes:
                allocation_positive.append(
                    "allocation_expanded_kv_size:"
                    f"{event.event_index}:{size_kind}:{size}"
                )
    for tensor in allocation_facts.method_tensors:
        if tensor.storage_bytes in expanded_sizes:
            allocation_positive.append(
                "tensor_expanded_kv_size:"
                f"{tensor.role}:{tensor.storage_bytes}"
            )
        if tensor.shape == expected_expanded_shape:
            allocation_positive.append(
                "tensor_expanded_kv_shape:" + tensor.role
            )
    positive = tuple(
        dict.fromkeys(
            (*dispatch.positive_materialization_evidence, *allocation_positive)
        )
    )

    expected_criterion_id = (
        "phase3_graph_zero_allocation_v1"
        if operation_key.allocation_execution_mode == "cuda_graph"
        else "phase3_eager_attributed_ephemeral_v1"
    )
    allocation_failures: list[str] = []
    if (
        allocation_facts.allocator_trace_sha256
        != allocation_facts.expected_allocator_trace_sha256
    ):
        allocation_failures.append("allocator_trace_sha256_mismatch")
    allocation_failures.extend(
        "semantic:" + reason
        for reason in allocation_facts.semantic_failure_reasons
    )
    allocation_failures.extend(
        "trace_integrity:" + reason
        for reason in allocation_facts.trace_integrity_errors
    )
    allocation_failures.extend(
        "criterion:" + reason
        for reason in allocation_facts.criterion_failure_reasons
    )
    if any(
        event.triggered_segment_alloc
        for event in allocation_facts.allocation_events
    ):
        allocation_failures.append("segment_alloc_detected")
    if allocation_facts.criterion_id != expected_criterion_id:
        allocation_failures.append("allocation_criterion_id_mismatch")
    if (
        operation_key.allocation_execution_mode == "cuda_graph"
        and allocation_facts.allocation_events
    ):
        allocation_failures.append("graph_allocation_event_detected")
    if allocation_positive:
        allocation_failures.append("expanded_gqa_allocation_or_tensor_detected")
    allocation_verified = not allocation_failures

    verdict = classify_gqa_evidence(
        materialization_evidence=bool(positive),
        dispatch_verified=dispatch.dispatch_verified,
        no_replication_kernel_verified=(
            dispatch.no_replication_kernel_verified
        ),
        allocation_verified=allocation_verified,
        source_verified=dispatch.source_verified,
        shape_verified=dispatch.shape_verified,
    )
    reasons: list[str] = []
    if positive:
        reasons.append("positive materialization evidence exists")
    if not dispatch.dispatch_verified:
        reasons.append("device dispatch is not fully verified")
    if not dispatch.no_replication_kernel_verified:
        reasons.append("no-preceding-materialization proof is incomplete")
    if not dispatch.source_verified:
        reasons.append("source proof is incomplete")
    if not dispatch.shape_verified:
        reasons.append("shape and storage proof is incomplete")
    reasons.extend(allocation_failures)
    if not allocation_verified:
        reasons.append("allocation-size proof is incomplete")
    return GQAProofEvaluation(
        verdict=verdict,
        dispatch_verified=dispatch.dispatch_verified,
        no_replication_kernel_verified=(
            dispatch.no_replication_kernel_verified
        ),
        allocation_verified=allocation_verified,
        source_verified=dispatch.source_verified,
        shape_verified=dispatch.shape_verified,
        family_comparison=dispatch.family_comparison,
        positive_materialization_evidence=positive,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def revalidate_phase3_geometry_bound_dispatch_audit(
    audit: Phase3GeometryBoundGQADeviceDispatchAudit,
    *,
    gqa_raw: bytes,
    mha_raw: bytes,
    backend_identity_raw: bytes,
    source_bytes_by_path: Mapping[str, bytes],
) -> Phase3GeometryBoundGQADeviceDispatchAudit:
    """Purely rebuild a serialized audit's raw trace and source bindings."""

    if not isinstance(backend_identity_raw, bytes) or not backend_identity_raw:
        raise GQADeviceDispatchError("raw backend identity is absent")
    try:
        backend_payload = json.loads(backend_identity_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise GQADeviceDispatchError("raw backend identity is invalid") from error
    if not isinstance(backend_payload, dict):
        raise GQADeviceDispatchError("raw backend identity must be an object")
    rebuilt_backend_identity = BackendIdentityEvidence.from_payload(
        backend_payload
    )
    if (
        backend_identity_raw
        != rebuilt_backend_identity.canonical_json.encode("utf-8")
        or rebuilt_backend_identity != audit.backend_identity
    ):
        raise GQADeviceDispatchError(
            "retained backend identity bytes differ from audit identity"
        )
    if set(source_bytes_by_path) != set(REQUIRED_SUT_SOURCES):
        raise GQADeviceDispatchError("raw source bundle paths are incomplete")
    rebuilt_sources = tuple(
        source_file_evidence_from_bytes(path, source_bytes_by_path[path])
        for path in REQUIRED_SUT_SOURCES
    )
    if rebuilt_sources != audit.source_shape.sources:
        raise GQADeviceDispatchError("retained source bytes differ from audit identity")
    if audit.gqa.raw_trace is None or audit.mha.raw_trace is None:
        raise GQADeviceDispatchError("audit raw trace metadata is absent")
    trace_validation = revalidate_geometry_bound_raw_traces(
        point=audit.point,
        gqa_artifact=audit.gqa.raw_trace,
        mha_artifact=audit.mha.raw_trace,
        gqa_raw=gqa_raw,
        mha_raw=mha_raw,
    )
    gqa = replace(
        audit.gqa,
        trace_scope=trace_validation.gqa_scope,
        device_events=trace_validation.gqa_device_events,
    )
    mha = replace(
        audit.mha,
        trace_scope=trace_validation.mha_scope,
        device_events=trace_validation.mha_device_events,
    )
    source_shape = replace(audit.source_shape, sources=rebuilt_sources)
    evaluation = evaluate_geometry_bound_gqa_device_dispatch(
        point=audit.point,
        gqa=gqa,
        mha=mha,
        gqa_sequence=trace_validation.gqa_kernel_sequence,
        mha_sequence=trace_validation.mha_kernel_sequence,
        source_shape=source_shape,
        explicitly_related_families=set(
            audit.explicitly_related_families
        ),
    )
    rebuilt = Phase3GeometryBoundGQADeviceDispatchAudit(
        point=audit.point,
        backend_identity=rebuilt_backend_identity,
        gqa=gqa,
        mha=mha,
        source_shape=source_shape,
        gqa_kernel_sequence=trace_validation.gqa_kernel_sequence,
        mha_kernel_sequence=trace_validation.mha_kernel_sequence,
        trace_validation=trace_validation,
        explicitly_related_families=audit.explicitly_related_families,
        related_family_policy_sha256=(
            audit.related_family_policy_sha256
        ),
        evaluation=evaluation,
    )
    if rebuilt != audit:
        raise GQADeviceDispatchError(
            "raw-derived dispatch audit differs from retained audit"
        )
    return rebuilt


def collect_gqa_mha_device_dispatch(
    *,
    gqa_query: Any,
    gqa_key: Any,
    gqa_value: Any,
    mha_query: Any,
    mha_key: Any,
    mha_value: Any,
    output_directory: Path,
    artifact_relative_root: str,
    source_root: Path,
    source_paths: tuple[Path, ...],
    is_causal: bool,
    scale: float,
    warmup_count: int,
    allocation_verified: bool,
    allocation_proof_sha256: str | None = None,
    expanded_kv_allocation_detected: bool = False,
    expanded_kv_tensor_detected: bool = False,
    explicitly_related_families: Set[tuple[str, str]] = frozenset(),
) -> GQADeviceDispatchAudit:
    """Collect small frozen public-SDPA controls for B-011 only.

    This does not bind the proof to every campaign geometry, the production
    endpoint, or an actual model cache. Those remain explicit integration work.
    """

    _positive_integer(warmup_count, "warmup_count")
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be boolean")
    if is_causal:
        raise ValueError("frozen small dispatch controls use decode causality")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise TypeError("scale must be numeric")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be finite and positive")
    if float(scale) != 128**-0.5:
        raise ValueError("dispatch control scale differs from frozen head scale")
    relative_root = PurePosixPath(artifact_relative_root)
    if (
        not artifact_relative_root
        or relative_root.is_absolute()
        or "." in relative_root.parts
        or ".." in relative_root.parts
    ):
        raise ValueError("artifact trace root must be safe and relative")
    output_root = Path(output_directory)
    if not output_root.is_dir() or output_root.is_symlink():
        raise GQADeviceDispatchError("trace output directory must be real")
    torch = importlib.import_module("torch")
    backend_module = importlib.import_module("kvbench.runtime.backend")
    forced_flash_execution = backend_module.forced_flash_execution
    identity_payload = backend_module.backend_identity()
    identity = BackendIdentityEvidence.from_payload(identity_payload)
    controls = (
        ("gqa", gqa_query, gqa_key, gqa_value),
        ("mha_control", mha_query, mha_key, mha_value),
    )
    for role, query, key, value in controls:
        _validate_control_tensors(
            torch,
            role=role,
            query=query,
            key=key,
            value=value,
        )
    if (
        tuple(int(item) for item in gqa_query.shape[:1] + gqa_query.shape[2:])
        != tuple(int(item) for item in mha_query.shape[:1] + mha_query.shape[2:])
        or int(gqa_key.shape[-2]) != int(mha_key.shape[-2])
        or gqa_query.dtype != mha_query.dtype
        or gqa_query.device != mha_query.device
    ):
        raise GQADeviceDispatchError("GQA/MHA held-constant controls differ")
    sources = audit_gqa_source_files(source_root, source_paths)
    if tuple(item.relative_path for item in sources) != REQUIRED_SUT_SOURCES:
        raise GQADeviceDispatchError(
            "composite dispatch audit requires the exact frozen SUT source set"
        )
    backend_controls = {
        role: _collect_backend_control(
            torch,
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            scale=float(scale),
            identity=identity,
            forced_flash_execution=forced_flash_execution,
        )
        for role, query, key, value in controls
    }

    def operation(query: Any, key: Any, value: Any) -> Callable[[], Any]:
        def invoke() -> Any:
            with forced_flash_execution():
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=float(scale),
                    enable_gqa=True,
                )

        return invoke

    retained: dict[str, tuple[DispatchControlEvidence, SourceShapeEvidence]] = {}
    for role, query, key, value in controls:
        filename = "gqa.chrome.json" if role == "gqa" else "mha.chrome.json"
        trace_path = output_root / filename
        relative_path = (relative_root / filename).as_posix()
        marker = f"kvbench.phase3.{role}.dispatch_audit"
        invoke = operation(query, key, value)
        artifact = collect_torch_profiler_trace(
            invoke,
            trace_path,
            artifact_relative_path=relative_path,
            marker=marker,
            warmup_count=warmup_count,
            device=query.device,
        )
        scoped = _read_verified_trace(trace_path, artifact, marker=marker)
        output = invoke()
        output_shape = tensor_shape_evidence(output)
        del output
        torch.cuda.synchronize(device=query.device)
        control = DispatchControlEvidence(
            role=role,
            batch_size=int(query.shape[0]),
            context_length=int(key.shape[-2]),
            query_length=int(query.shape[-2]),
            num_query_heads=int(query.shape[1]),
            num_kv_heads=int(key.shape[1]),
            head_dim=int(query.shape[-1]),
            dtype=str(query.dtype),
            dtype_bytes=int(query.element_size()),
            is_causal=is_causal,
            warmup_count=warmup_count,
            backend=backend_controls[role],
            raw_trace=artifact,
            trace_scope=scoped.scope,
            device_events=scoped.device_events,
        )
        query_shape = tensor_shape_evidence(query)
        key_shape = tensor_shape_evidence(key)
        value_shape = tensor_shape_evidence(value)
        source_shape = SourceShapeEvidence(
            sources=sources,
            query=query_shape,
            key=key_shape,
            value=value_shape,
            output=output_shape,
            native_kv_storage_verified=bool(
                key_shape.storage_bytes == key_shape.logical_bytes
                and value_shape.storage_bytes == value_shape.logical_bytes
                and key_shape.storage_offset == 0
                and value_shape.storage_offset == 0
                and key_shape.is_contiguous
                and value_shape.is_contiguous
            ),
        )
        retained[role] = (control, source_shape)
    gqa, gqa_source_shape = retained["gqa"]
    mha, mha_source_shape = retained["mha_control"]
    related = tuple(sorted(set(explicitly_related_families)))
    source_verified = bool(
        gqa_source_shape.source_verified and mha_source_shape.source_verified
    )
    shape_verified = bool(
        gqa_source_shape.shape_verified_for(gqa)
        and mha_source_shape.shape_verified_for(mha)
    )
    evaluation = evaluate_gqa_device_dispatch(
        gqa=gqa,
        mha=mha,
        allocation_verified=allocation_verified,
        source_verified=source_verified,
        shape_verified=shape_verified,
        expanded_kv_allocation_detected=expanded_kv_allocation_detected,
        expanded_kv_tensor_detected=expanded_kv_tensor_detected,
        explicitly_related_families=set(related),
        source_positive_materialization_evidence=(
            source_materialization_evidence(gqa_source_shape.sources)
        ),
    )
    return GQADeviceDispatchAudit(
        backend_identity=identity,
        gqa=gqa,
        mha=mha,
        gqa_source_shape=gqa_source_shape,
        mha_source_shape=mha_source_shape,
        raw_trace_bytes_verified=True,
        allocation_proof_sha256=allocation_proof_sha256,
        expanded_kv_allocation_detected=expanded_kv_allocation_detected,
        expanded_kv_tensor_detected=expanded_kv_tensor_detected,
        explicitly_related_families=related,
        evaluation=evaluation,
    )


def collect_phase3_geometry_bound_gqa_mha_device_dispatch(
    *,
    operation_key: Phase3AuditOperationKey,
    cache_layout_fingerprint: str,
    cache_workspace_bytes: int,
    cache_layer_index: int,
    cache_key_backing: Any,
    cache_value_backing: Any,
    gqa_query: Any,
    gqa_key_view: Any,
    gqa_value_view: Any,
    mha_query: Any,
    mha_key: Any,
    mha_value: Any,
    output_directory: Path,
    artifact_relative_root: str,
    source_root: Path,
    source_paths: tuple[Path, ...],
    is_causal: bool,
    scale: float,
    warmup_count: int,
    explicitly_related_families: Set[tuple[str, str]] = frozenset(),
    related_family_policy_sha256: str | None = None,
) -> Phase3GeometryBoundGQADeviceDispatchAudit:
    """Collect dispatch-only evidence for one actual Phase 3 cache view.

    This path deliberately has no allocator-verdict argument. Task C must bind
    its raw allocator evidence to the retained GQA/MHA trace hashes before the
    combined non-materialization verdict can become verified.
    """

    if type(operation_key) is not Phase3AuditOperationKey:
        raise TypeError("operation_key must be Phase3AuditOperationKey")
    point = Phase3DispatchPointBinding.create(operation_key=operation_key)
    active_context = point.active_context
    cache_capacity = point.capacity
    _positive_integer(warmup_count, "warmup_count")
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be boolean")
    if is_causal:
        raise ValueError("geometry-bound decode controls must be non-causal")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise TypeError("scale must be numeric")
    if not math.isfinite(float(scale)) or float(scale) != PHASE3_HEAD_DIM**-0.5:
        raise ValueError("geometry-bound scale differs from frozen head scale")
    try:
        related_families = tuple(sorted(set(explicitly_related_families)))
    except TypeError as error:
        raise ValueError("related-family policy pairs are malformed") from error
    if any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(not isinstance(item, str) or not item for item in pair)
        for pair in related_families
    ):
        raise ValueError("related-family policy pair is malformed")
    allowed_pairs = set(PHASE3_FLASH_RELATED_FAMILIES)
    if any(
        pair not in allowed_pairs and (pair[1], pair[0]) not in allowed_pairs
        for pair in related_families
    ):
        raise ValueError("related-family policy contains an unsupported pair")
    if related_family_policy_sha256 is not None and not _SHA256_RE.fullmatch(
        related_family_policy_sha256
    ):
        raise ValueError("related-family policy SHA-256 is invalid")
    if bool(related_families) != bool(related_family_policy_sha256):
        raise ValueError(
            "related families and decision-record digest must coexist"
        )
    if related_families:
        raise ValueError(
            "no checksum-pinned related-family decision is approved"
        )
    relative_root = PurePosixPath(artifact_relative_root)
    if (
        not artifact_relative_root
        or relative_root.is_absolute()
        or "." in relative_root.parts
        or ".." in relative_root.parts
    ):
        raise ValueError("artifact trace root must be safe and relative")
    output_root = Path(output_directory)
    if not output_root.is_dir() or output_root.is_symlink():
        raise GQADeviceDispatchError("trace output directory must be real")

    torch = importlib.import_module("torch")
    controls = (
        ("gqa", gqa_query, gqa_key_view, gqa_value_view),
        ("mha_control", mha_query, mha_key, mha_value),
    )
    for role, query, key, value in controls:
        _validate_geometry_bound_control_tensors(
            torch,
            role=role,
            query=query,
            key=key,
            value=value,
            active_context=active_context,
        )
    if not isinstance(cache_key_backing, torch.Tensor) or not isinstance(
        cache_value_backing, torch.Tensor
    ):
        raise GQADeviceDispatchError("static-cache backings require tensors")
    if (
        tuple(int(item) for item in gqa_query.shape[:1] + gqa_query.shape[2:])
        != tuple(int(item) for item in mha_query.shape[:1] + mha_query.shape[2:])
        or gqa_query.dtype != mha_query.dtype
        or gqa_query.device != mha_query.device
    ):
        raise GQADeviceDispatchError("geometry-bound held constants differ")
    if mha_query is not gqa_query or int(mha_query.data_ptr()) != int(
        gqa_query.data_ptr()
    ):
        raise GQADeviceDispatchError(
            "GQA/MHA controls must share the exact query tensor"
        )
    if int(gqa_query.shape[0]) != point.batch_size:
        raise GQADeviceDispatchError(
            "geometry-bound query batch differs from operation key"
        )
    execution_mode = (
        CUDA_GRAPH_REPLAY_EXECUTION_MODE
        if point.graph_mode == "cuda_graph"
        else EAGER_EXECUTION_MODE
    )
    sources = audit_gqa_source_files(source_root, source_paths)
    if tuple(item.relative_path for item in sources) != REQUIRED_SUT_SOURCES:
        raise GQADeviceDispatchError(
            "geometry-bound dispatch requires the exact frozen SUT source set"
        )
    source_by_path = {source.relative_path: source for source in sources}
    cache_source = source_by_path[REQUIRED_SUT_SOURCES[2]]
    layout = StaticCacheLayoutEvidence.create(
        batch_size=point.batch_size,
        capacity=cache_capacity,
        device=str(gqa_query.device),
        workspace_bytes=cache_workspace_bytes,
        implementation_sha256=cache_source.sha256,
        layout_fingerprint=cache_layout_fingerprint,
    )
    cache_binding = static_cache_view_binding_evidence(
        layout=layout,
        layer_index=cache_layer_index,
        active_context=active_context,
        key_backing=cache_key_backing,
        value_backing=cache_value_backing,
        key_view=gqa_key_view,
        value_view=gqa_value_view,
    )
    if not cache_binding.verified:
        raise GQADeviceDispatchError(
            "static-cache view binding failed: "
            + ",".join(cache_binding.failure_reasons)
        )

    backend_module = importlib.import_module("kvbench.runtime.backend")
    forced_flash_execution = backend_module.forced_flash_execution
    identity = BackendIdentityEvidence.from_payload(
        backend_module.backend_identity()
    )
    backend_controls = {
        role: _collect_backend_control(
            torch,
            query=query,
            key=key,
            value=value,
            is_causal=False,
            scale=float(scale),
            identity=identity,
            forced_flash_execution=forced_flash_execution,
        )
        for role, query, key, value in controls
    }

    retained_operation_outputs: dict[str, Any] = {}

    def operation(
        role: str,
        query: Any,
        key: Any,
        value: Any,
    ) -> Callable[[], Any]:
        def invoke() -> Any:
            with forced_flash_execution():
                output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=float(scale),
                    enable_gqa=True,
                )
            retained_operation_outputs[role] = output
            return output

        return invoke

    retained_controls: dict[str, DispatchControlEvidence] = {}
    output_shapes: dict[str, TensorShapeEvidence] = {}
    for role, query, key, value in controls:
        filename = (
            "gqa.geometry.chrome.json"
            if role == "gqa"
            else "mha.geometry.chrome.json"
        )
        trace_path = output_root / filename
        relative_path = (relative_root / filename).as_posix()
        marker = _geometry_bound_trace_marker(point, role)
        invoke = operation(role, query, key, value)
        artifact = collect_torch_profiler_trace(
            invoke,
            trace_path,
            artifact_relative_path=relative_path,
            marker=marker,
            warmup_count=warmup_count,
            device=query.device,
            execution_mode=execution_mode,
        )
        scoped = _read_verified_trace(
            trace_path,
            artifact,
            marker=marker,
            require_kernel_launch_runtime=True,
        )
        output_shapes[role] = tensor_shape_evidence(
            retained_operation_outputs.pop(role)
        )
        retained_controls[role] = DispatchControlEvidence(
            role=role,
            batch_size=int(query.shape[0]),
            context_length=int(key.shape[-2]),
            query_length=int(query.shape[-2]),
            num_query_heads=int(query.shape[1]),
            num_kv_heads=int(key.shape[1]),
            head_dim=int(query.shape[-1]),
            dtype=str(query.dtype),
            dtype_bytes=int(query.element_size()),
            is_causal=False,
            warmup_count=warmup_count,
            backend=backend_controls[role],
            raw_trace=artifact,
            trace_scope=scoped.scope,
            device_events=scoped.device_events,
            execution_mode=execution_mode,
        )

    gqa = retained_controls["gqa"]
    mha = retained_controls["mha_control"]
    assert gqa.raw_trace is not None
    assert mha.raw_trace is not None
    trace_validation = revalidate_geometry_bound_raw_traces(
        point=point,
        gqa_artifact=gqa.raw_trace,
        mha_artifact=mha.raw_trace,
        gqa_raw=(output_root / "gqa.geometry.chrome.json").read_bytes(),
        mha_raw=(output_root / "mha.geometry.chrome.json").read_bytes(),
    )
    source_shape = GeometryBoundSourceShapeEvidence(
        sources=sources,
        gqa_query=tensor_shape_evidence(gqa_query),
        gqa_output=output_shapes["gqa"],
        cache=cache_binding,
        mha_query=tensor_shape_evidence(mha_query),
        mha_key=tensor_shape_evidence(mha_key),
        mha_value=tensor_shape_evidence(mha_value),
        mha_output=output_shapes["mha_control"],
    )
    gqa_sequence = analyze_flash_kernel_sequence(gqa.device_events)
    mha_sequence = analyze_flash_kernel_sequence(mha.device_events)
    evaluation = evaluate_geometry_bound_gqa_device_dispatch(
        point=point,
        gqa=gqa,
        mha=mha,
        gqa_sequence=gqa_sequence,
        mha_sequence=mha_sequence,
        source_shape=source_shape,
        explicitly_related_families=set(related_families),
    )
    return Phase3GeometryBoundGQADeviceDispatchAudit(
        point=point,
        backend_identity=identity,
        gqa=gqa,
        mha=mha,
        source_shape=source_shape,
        gqa_kernel_sequence=gqa_sequence,
        mha_kernel_sequence=mha_sequence,
        trace_validation=trace_validation,
        explicitly_related_families=related_families,
        related_family_policy_sha256=related_family_policy_sha256,
        evaluation=evaluation,
    )
