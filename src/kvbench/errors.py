"""Structured, non-secret-bearing errors used by Phase 2 tooling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    CONFIG_LOAD_ERROR = "config_load_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"
    ADMISSION_BLOCKED = "admission_blocked"
    PHASE_NOT_IMPLEMENTED = "phase_not_implemented"
    ARTIFACT_SAFETY_ERROR = "artifact_safety_error"
    ARTIFACT_CONFLICT = "artifact_conflict"
    ARTIFACT_STATE_ERROR = "artifact_state_error"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    PROVENANCE_ERROR = "provenance_error"


PathPart = str | int


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """A safe machine-readable error without the rejected input value."""

    code: ErrorCode
    message: str
    path: tuple[PathPart, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "path": list(self.path),
        }


class KVBenchError(Exception):
    """Base exception carrying a stable error code and a field path."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        path: tuple[PathPart, ...] = (),
    ) -> None:
        self.detail = ErrorDetail(code=code, message=message, path=path)
        super().__init__(self._render())

    @property
    def code(self) -> ErrorCode:
        return self.detail.code

    @property
    def path(self) -> tuple[PathPart, ...]:
        return self.detail.path

    def to_dict(self) -> dict[str, Any]:
        return self.detail.to_dict()

    def _render(self) -> str:
        location = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in self.detail.path
        ).lstrip(".")
        prefix = f"{location}: " if location else ""
        return f"{self.detail.code.value}: {prefix}{self.detail.message}"


class ConfigLoadError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.CONFIG_LOAD_ERROR, message, path=path)


class SchemaValidationError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.SCHEMA_VALIDATION_ERROR, message, path=path)


class ProvenanceError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.PROVENANCE_ERROR, message, path=path)


class AdmissionError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.ADMISSION_BLOCKED, message, path=path)


class PhaseNotImplementedError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.PHASE_NOT_IMPLEMENTED, message, path=path)


class ArtifactSafetyError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.ARTIFACT_SAFETY_ERROR, message, path=path)


class ArtifactConflictError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.ARTIFACT_CONFLICT, message, path=path)


class ArtifactStateError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.ARTIFACT_STATE_ERROR, message, path=path)


class ChecksumMismatchError(KVBenchError):
    def __init__(self, message: str, *, path: tuple[PathPart, ...] = ()) -> None:
        super().__init__(ErrorCode.CHECKSUM_MISMATCH, message, path=path)
