"""Dependency-free strict model primitives and canonical serialization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import types
from abc import ABC
from datetime import datetime, timezone
from enum import Enum, StrEnum
from pathlib import PurePosixPath
from typing import (
    Any,
    ClassVar,
    Literal,
    Mapping,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from kvbench.errors import PathPart, SchemaValidationError


SCHEMA_PREFIX = "kvbench"
CANONICALIZATION_VERSION = "kvbench-json-v1"
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")
OCI_DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
RUN_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,127}\Z")
IDENTIFIER_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    BUILD_FAILED = "build_failed"
    RUNTIME_FAILED = "runtime_failed"
    NUMERICAL_FAILED = "numerical_failed"
    MODEL_IDENTITY_UNRESOLVED = "model_identity_unresolved"
    MODEL_ACCESS_BLOCKED = "model_access_blocked"
    BACKEND_UNSUPPORTED = "backend_unsupported"
    ALLOCATION_FAILED = "allocation_failed"
    STATE_DRIFT_DETECTED = "state_drift_detected"
    GQA_MATERIALIZATION_DETECTED = "gqa_materialization_detected"
    GQA_DISPATCH_UNVERIFIED = "gqa_dispatch_unverified"
    GQA_NONMATERIALIZATION_UNPROVEN = "gqa_nonmaterialization_unproven"
    GRAPH_CAPTURE_FAILED = "graph_capture_failed"
    GRAPH_REPLAY_FAILED = "graph_replay_failed"
    PROFILER_FAILED = "profiler_failed"
    CAPACITY_INFEASIBLE = "capacity_infeasible"
    UNSTABLE = "unstable"
    BACKEND_FALLBACK = "backend_fallback"
    UNSUPPORTED_GEOMETRY = "unsupported_geometry"
    ABORTED = "aborted"

    @property
    def is_terminal(self) -> bool:
        return self not in {
            RunStatus.CREATED,
            RunStatus.RUNNING,
            RunStatus.FINALIZING,
        }

    @property
    def is_failure(self) -> bool:
        return self.is_terminal and self is not RunStatus.COMPLETED


class GQAVerdict(StrEnum):
    """Evidence verdict for the frozen Phase 3 GQA proof contract."""

    MATERIALIZATION_DETECTED = "gqa_materialization_detected"
    DISPATCH_UNVERIFIED = "gqa_dispatch_unverified"
    NONMATERIALIZATION_UNPROVEN = "gqa_nonmaterialization_unproven"
    NONMATERIALIZATION_VERIFIED = "gqa_nonmaterialization_verified"


class RunKind(StrEnum):
    HARDWARE_PREFLIGHT = "hardware_preflight"
    CORRECTNESS = "correctness"
    TIMING = "timing"
    NSYS = "nsys"
    NCU = "ncu"
    SYNTHETIC = "synthetic"
    PHASE3_ADMISSION = "phase3_admission"
    PHASE6_ADMISSION = "phase6_admission"


class MeasurementScope(StrEnum):
    NATIVE_HOST_ADMISSION = "native_host_admission"
    MEASUREMENT_CONTAINER_ADMISSION = "measurement_container_admission"


class ClaimClass(StrEnum):
    NONE = "none"
    SAME_WORK_LATENCY = "same_work_latency"
    CAPACITY_AMPLIFICATION = "capacity_amplification"
    MECHANISM_ONLY = "mechanism_only"


class GraphMode(StrEnum):
    EAGER = "eager"
    CUDA_GRAPH = "cuda_graph"


class RunnerKind(StrEnum):
    FIXED_L = "fixed_l"
    GROWING_CONTEXT = "growing_context"


class ResolutionState(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"


class QualityValidationState(StrEnum):
    UNVALIDATED = "unvalidated"
    NOT_APPLICABLE = "not_applicable"


class ClaimEligibility(StrEnum):
    PERFORMANCE_ONLY = "performance_only"
    NONE = "none"


class QualityExecutionState(StrEnum):
    LOCKED = "locked"


T = TypeVar("T", bound="StrictModel")


def _schema_error(path: tuple[PathPart, ...], message: str) -> SchemaValidationError:
    return SchemaValidationError(message, path=path)


def _parse_value(annotation: Any, value: Any, path: tuple[PathPart, ...]) -> Any:
    if annotation is Any:
        raise _schema_error(path, "untyped values are not allowed in strict schemas")

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        if not any(type(value) is type(candidate) and value == candidate for candidate in args):
            raise _schema_error(path, "value is not one of the allowed literals")
        return value

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        failures: list[SchemaValidationError] = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _parse_value(option, value, path)
            except SchemaValidationError as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        raise _schema_error(path, "value does not match any allowed schema type")

    if origin is tuple:
        if not isinstance(value, list):
            raise _schema_error(path, "expected an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _parse_value(args[0], item, (*path, index))
                for index, item in enumerate(value)
            )
        if len(value) != len(args):
            raise _schema_error(path, "array has the wrong number of items")
        return tuple(
            _parse_value(item_type, item, (*path, index))
            for index, (item_type, item) in enumerate(zip(args, value, strict=True))
        )

    if origin is list:
        if not isinstance(value, list):
            raise _schema_error(path, "expected an array")
        return [
            _parse_value(args[0], item, (*path, index))
            for index, item in enumerate(value)
        ]

    if origin is dict:
        if not isinstance(value, dict):
            raise _schema_error(path, "expected an object")
        key_type, value_type = args
        parsed: dict[Any, Any] = {}
        for key, item in value.items():
            parsed_key = _parse_value(key_type, key, (*path, "<key>"))
            parsed[parsed_key] = _parse_value(value_type, item, (*path, str(key)))
        return parsed

    if annotation is str:
        if type(value) is not str:
            raise _schema_error(path, "expected a string")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise _schema_error(path, "expected a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise _schema_error(path, "expected an integer")
        return value
    if annotation is float:
        if type(value) is not float or not math.isfinite(value):
            raise _schema_error(path, "expected a finite JSON number with a decimal point")
        return value
    if annotation is type(None):
        if value is not None:
            raise _schema_error(path, "expected null")
        return None

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if type(value) is not str:
            raise _schema_error(path, "expected an enum string")
        try:
            return annotation(value)
        except ValueError as error:
            raise _schema_error(path, "invalid enum value") from error

    if isinstance(annotation, type) and issubclass(annotation, StrictModel):
        return annotation.from_dict(value, path=path)

    raise _schema_error(path, "unsupported schema annotation")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, StrictModel):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values cannot be canonically serialized")
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact, sorted UTF-8 JSON bytes without a trailing newline."""

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class StrictModel(ABC):
    """Mixin for frozen dataclasses with strict, unknown-field-free parsing."""

    schema_version: ClassVar[str]

    @classmethod
    def from_dict(
        cls: type[T],
        value: Any,
        *,
        path: tuple[PathPart, ...] = (),
    ) -> T:
        if not isinstance(value, dict) or any(type(key) is not str for key in value):
            raise _schema_error(path, "expected an object with string keys")
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        fields = {field.name: field for field in dataclasses.fields(cls)}
        unknown = sorted(set(value) - set(fields))
        if unknown:
            raise _schema_error((*path, unknown[0]), "unknown field")

        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for name, field in fields.items():
            if name not in value:
                if field.default is not dataclasses.MISSING:
                    continue
                if field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
                    continue
                raise _schema_error((*path, name), "missing required field")
            kwargs[name] = _parse_value(hints[name], value[name], (*path, name))
        try:
            return cls(**kwargs)
        except SchemaValidationError:
            raise
        except ValueError as error:
            raise _schema_error(path, str(error)) from error

    def to_dict(self) -> dict[str, Any]:
        if not dataclasses.is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass")
        return {
            field.name: _json_value(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    def fingerprint(self) -> str:
        return sha256_hex(self.canonical_bytes())


def require_schema(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"schema_version must be {expected}")


def require_identifier(value: str, *, field_name: str = "identifier") -> None:
    if IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} is malformed")


def require_run_id(value: str) -> None:
    if RUN_ID_RE.fullmatch(value) is None or value in {".", ".."} or value.startswith("."):
        raise ValueError("run_id is unsafe or malformed")


def require_sha256(value: str, *, field_name: str = "SHA-256") -> None:
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")


def require_git_sha(value: str) -> None:
    if GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError("git_sha must be a full 40-character lowercase hexadecimal SHA")


def require_oci_digest(value: str) -> None:
    if OCI_DIGEST_RE.fullmatch(value) is None:
        raise ValueError("container_digest must be an sha256 OCI digest")


def require_relative_path(value: str, *, field_name: str = "path") -> None:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field_name} is not a safe POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} is not a safe POSIX relative path")


def require_utc_timestamp(value: str, *, field_name: str = "timestamp") -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


@dataclasses.dataclass(frozen=True, slots=True)
class Resolution(StrictModel):
    schema_version: str
    status: ResolutionState
    blockers: tuple[str, ...]
    reason: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench.resolution.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.status is ResolutionState.RESOLVED:
            if self.blockers or self.reason is not None:
                raise ValueError("resolved state cannot carry blockers or a reason")
        else:
            if not self.blockers or not self.reason:
                raise ValueError("unresolved or blocked state requires blockers and a reason")
        if any(not blocker.strip() for blocker in self.blockers):
            raise ValueError("blocker identifiers must be non-empty")
        if len(set(self.blockers)) != len(self.blockers):
            raise ValueError("blocker identifiers must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class QualityStatus(StrictModel):
    schema_version: str
    quality_status: QualityValidationState
    claim_eligibility: ClaimEligibility
    quality_execution: QualityExecutionState
    performance_data_frozen: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench.quality-status.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.quality_execution is not QualityExecutionState.LOCKED:
            raise ValueError("quality execution must remain locked in Phase 2")
        if self.performance_data_frozen:
            raise ValueError("PERFORMANCE_DATA_FROZEN must remain absent in Phase 2")
        if self.quality_status is QualityValidationState.UNVALIDATED:
            if self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY:
                raise ValueError("unvalidated performance data must be performance_only")
        elif self.claim_eligibility is not ClaimEligibility.NONE:
            raise ValueError("not-applicable quality metadata cannot be claim eligible")
