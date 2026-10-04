"""Run-owned process identity and supervision primitives.

The Phase 3 registry deliberately contains no polling loop and never calls
``subprocess.Popen.poll`` or ``wait``.  Its coordinator supplies observations
from pidfd readiness or ``waitid(..., WNOWAIT)``.  The small generic command
runner is separate: it supervises one direct child through bounded
``communicate`` calls and makes no handshake or device-exclusivity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
from typing import Callable, ClassVar, Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,255}\Z")
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]+\Z")


class ProcessSupervisionError(RuntimeError):
    """A process identity, handshake, or ownership assertion failed closed."""


class ProcessIdentityUnavailable(ProcessSupervisionError):
    """A process identity could not be recovered from procfs."""


class HandshakeStage(str, Enum):
    """The exact ordered supervisor/worker handshake."""

    WORKER_STARTED = "worker_started"
    CUDA_CONTEXT_CREATED = "cuda_context_created"
    MEASUREMENT_STARTED = "measurement_started"
    MEASUREMENT_FINISHED = "measurement_finished"
    EVIDENCE_FLUSHED = "evidence_flushed"
    WORKER_EXITING = "worker_exiting"
    SUPERVISOR_REAPED = "supervisor_reaped"

    @property
    def sequence(self) -> int:
        return _HANDSHAKE_STAGES.index(self) + 1


_HANDSHAKE_STAGES = tuple(HandshakeStage)
_WORKER_STAGES = _HANDSHAKE_STAGES[:-1]


class ProcObservationDisposition(str, Enum):
    """Result of comparing a procfs observation with the registered worker."""

    EXACT = "exact"
    DISAPPEARED_RETAINED = "disappeared_retained"
    PID_REUSE_DETECTED = "pid_reuse_detected"


class SnapshotDisposition(str, Enum):
    """Ownership disposition for one device-process snapshot."""

    CLEAN = "clean"
    OWNED_ONLY = "owned_only"
    FOREIGN_PROCESS_DETECTED = "foreign_process_detected"
    PID_REUSE_DETECTED = "pid_reuse_detected"
    UNVERIFIED_REGISTERED_PID = "unverified_registered_pid"


class OwnershipDisposition(str, Enum):
    """Terminal run-owned process verdict."""

    OWNED_COMPLETED = "owned_completed"
    OWNED_WORKER_FAILURE = "owned_worker_failure"
    FOREIGN_PROCESS_DETECTED = "foreign_process_detected"
    PID_REUSE_DETECTED = "pid_reuse_detected"
    UNVERIFIED_PROCESS_DETECTED = "unverified_process_detected"


def _require_positive_integer(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProcessSupervisionError(f"{label} must be a positive integer")


def _require_nonnegative_integer(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProcessSupervisionError(f"{label} must be a nonnegative integer")


def _require_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise ProcessSupervisionError(f"{label} is not a safe identifier")


def _require_gpu_uuid(value: str) -> None:
    if not isinstance(value, str) or not _GPU_UUID.fullmatch(value):
        raise ProcessSupervisionError("GPU UUID is invalid")


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProcessSupervisionError(f"{label} is not a SHA-256 digest")


def command_fingerprint(
    argv: Sequence[str],
    *,
    working_directory: str,
    environment_sha256: str,
) -> str:
    """Bind the exact worker command without serializing its environment."""

    if (
        isinstance(argv, (str, bytes))
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ProcessSupervisionError("worker argv must be a nonempty string list")
    if not isinstance(working_directory, str) or not Path(
        working_directory
    ).is_absolute():
        raise ProcessSupervisionError("worker working directory must be absolute")
    _require_sha256(environment_sha256, "environment fingerprint")
    payload = {
        "argv": list(argv),
        "environment_sha256": environment_sha256,
        "working_directory": working_directory,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_fingerprint(environment: Mapping[str, str]) -> str:
    """Bind an explicit child environment without returning any of its values."""

    if not isinstance(environment, Mapping):
        raise ProcessSupervisionError("supervised environment must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "\x00" in key
            or "=" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ProcessSupervisionError(
                "supervised environment contains an invalid entry"
            )
        if key in normalized:
            raise ProcessSupervisionError(
                "supervised environment contains a duplicate key"
            )
        normalized[key] = value
    encoded = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProcessIdentity:
    """One PID protected against reuse by its procfs start time."""

    pid: int
    start_time_ticks: int
    parent_pid: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.pid, "PID")
        _require_nonnegative_integer(self.start_time_ticks, "process start time")
        _require_positive_integer(self.parent_pid, "parent PID")

    def to_dict(self) -> dict[str, int]:
        return {
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "parent_pid": self.parent_pid,
        }


@dataclass(frozen=True)
class SupervisedCommandResult:
    """Immutable result for a direct child that has no worker handshake."""

    SCHEMA_VERSION: ClassVar[str] = (
        "kvbench-generic-supervised-command-result-1.0.0"
    )

    argv: tuple[str, ...]
    working_directory: str
    environment_sha256: str
    command_sha256: str
    process_identity: ProcessIdentity
    supervisor_pid: int
    timeout_seconds: float
    timed_out: bool
    terminate_requested: bool
    kill_requested: bool
    returncode: int
    stdout: bytes
    stderr: bytes
    pidfd_supported: bool
    pidfd_opened: bool
    pidfd: int | None
    pidfd_closed: bool
    direct_child_verified: bool
    process_handle_retained: bool
    final_reap_completed: bool
    final_reap_count: int

    def __post_init__(self) -> None:
        if (
            not self.argv
            or any(not isinstance(item, str) or not item for item in self.argv)
        ):
            raise ProcessSupervisionError(
                "supervised result argv must be a nonempty string tuple"
            )
        if (
            not isinstance(self.working_directory, str)
            or not Path(self.working_directory).is_absolute()
        ):
            raise ProcessSupervisionError(
                "supervised result working directory must be absolute"
            )
        _require_sha256(
            self.environment_sha256,
            "supervised environment fingerprint",
        )
        _require_sha256(self.command_sha256, "supervised command fingerprint")
        _require_positive_integer(self.supervisor_pid, "supervisor PID")
        if self.process_identity.parent_pid != self.supervisor_pid:
            raise ProcessSupervisionError(
                "supervised result parent PID differs from supervisor"
            )
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ProcessSupervisionError(
                "supervised result timeout must be finite and positive"
            )
        for value, label in (
            (self.timed_out, "timeout"),
            (self.terminate_requested, "terminate request"),
            (self.kill_requested, "kill request"),
            (self.pidfd_supported, "pidfd support"),
            (self.pidfd_opened, "pidfd opened"),
            (self.pidfd_closed, "pidfd closed"),
            (self.direct_child_verified, "direct-child verification"),
            (self.process_handle_retained, "process-handle retention"),
            (self.final_reap_completed, "final reap"),
        ):
            if not isinstance(value, bool):
                raise ProcessSupervisionError(
                    f"supervised result {label} flag is invalid"
                )
        if not isinstance(self.returncode, int) or isinstance(
            self.returncode, bool
        ):
            raise ProcessSupervisionError(
                "supervised result return code must be an integer"
            )
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise ProcessSupervisionError(
                "supervised command output must be bytes"
            )
        if self.pidfd_opened != (self.pidfd is not None):
            raise ProcessSupervisionError(
                "supervised result pidfd metadata is inconsistent"
            )
        if self.pidfd_opened and not self.pidfd_supported:
            raise ProcessSupervisionError(
                "supervised result opened an unsupported pidfd"
            )
        if self.pidfd is not None:
            _require_nonnegative_integer(self.pidfd, "pidfd")
        if self.pidfd_opened != self.pidfd_closed:
            raise ProcessSupervisionError(
                "supervised result must close every opened pidfd"
            )
        if (
            not self.direct_child_verified
            or not self.process_handle_retained
            or not self.final_reap_completed
            or self.final_reap_count != 1
        ):
            raise ProcessSupervisionError(
                "supervised result lacks direct-child final-reap proof"
            )
        if not self.timed_out and (
            self.terminate_requested or self.kill_requested
        ):
            raise ProcessSupervisionError(
                "supervised result signals require a timeout"
            )
        if self.kill_requested and not self.terminate_requested:
            raise ProcessSupervisionError(
                "supervised result kill requires an earlier terminate request"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize evidence without exposing environment or output contents."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "identity": self.process_identity.to_dict(),
            "command": {
                "argv": list(self.argv),
                "working_directory": self.working_directory,
                "environment_sha256": self.environment_sha256,
                "command_fingerprint": self.command_sha256,
                "shell": False,
            },
            "timeout": {
                "timeout_seconds": float(self.timeout_seconds),
                "timed_out": self.timed_out,
                "terminate_requested": self.terminate_requested,
                "kill_requested": self.kill_requested,
            },
            "returncode": self.returncode,
            "pidfd": {
                "supported": self.pidfd_supported,
                "opened": self.pidfd_opened,
                "descriptor": self.pidfd,
                "closed": self.pidfd_closed,
            },
            "direct_child": {
                "verified": self.direct_child_verified,
                "expected_parent_pid": self.supervisor_pid,
                "parent_pid_verified": True,
                "start_time_ticks_verified": True,
                "process_handle_retained": self.process_handle_retained,
            },
            "final_reap": {
                "completed": self.final_reap_completed,
                "count": self.final_reap_count,
            },
            "stdout": {
                "bytes": len(self.stdout),
                "sha256": hashlib.sha256(self.stdout).hexdigest(),
            },
            "stderr": {
                "bytes": len(self.stderr),
                "sha256": hashlib.sha256(self.stderr).hexdigest(),
            },
        }


@dataclass(frozen=True)
class StageSupervisedCommandResult:
    """Immutable direct-child result with bounded stage supervision."""

    SCHEMA_VERSION: ClassVar[str] = (
        "kvbench-stage-supervised-command-result-1.0.0"
    )

    command: SupervisedCommandResult
    stage_timeouts: tuple[tuple[str, float], ...]
    stage_observations: tuple[tuple[int, str, str, int, str], ...]
    timeout_stage: str | None
    timeout_stage_elapsed_seconds: float | None
    observer_poll_seconds: float

    def __post_init__(self) -> None:
        if not self.stage_timeouts:
            raise ProcessSupervisionError("stage timeout contract is empty")
        names: set[str] = set()
        for name, seconds in self.stage_timeouts:
            _require_identifier(name, "supervised stage")
            if name in names:
                raise ProcessSupervisionError("supervised stage is duplicated")
            names.add(name)
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not math.isfinite(float(seconds))
                or seconds <= 0
            ):
                raise ProcessSupervisionError(
                    "supervised stage timeout must be finite and positive"
                )
        previous_sequence = 0
        started: set[str] = set()
        completed: set[str] = set()
        for sequence, stage, state, monotonic_ns, event_sha256 in self.stage_observations:
            if sequence != previous_sequence + 1:
                raise ProcessSupervisionError(
                    "supervised stage observation sequence differs"
                )
            previous_sequence = sequence
            _require_identifier(stage, "supervised observed stage")
            if stage not in names or state not in {"started", "completed"}:
                raise ProcessSupervisionError(
                    "supervised stage observation is outside the contract"
                )
            _require_nonnegative_integer(monotonic_ns, "stage monotonic time")
            _require_sha256(event_sha256, "stage event")
            if state == "started":
                if stage in started or stage in completed:
                    raise ProcessSupervisionError(
                        "supervised stage starts more than once"
                    )
                started.add(stage)
            else:
                if stage not in started or stage in completed:
                    raise ProcessSupervisionError(
                        "supervised stage completion is out of order"
                    )
                completed.add(stage)
        if self.timeout_stage is None:
            if self.timeout_stage_elapsed_seconds is not None:
                raise ProcessSupervisionError(
                    "stage timeout duration exists without a stage"
                )
        else:
            if self.timeout_stage not in names or not self.command.timed_out:
                raise ProcessSupervisionError(
                    "stage timeout identity is inconsistent"
                )
            if (
                not isinstance(self.timeout_stage_elapsed_seconds, (int, float))
                or isinstance(self.timeout_stage_elapsed_seconds, bool)
                or not math.isfinite(float(self.timeout_stage_elapsed_seconds))
                or self.timeout_stage_elapsed_seconds <= 0
            ):
                raise ProcessSupervisionError(
                    "stage timeout duration must be finite and positive"
                )
        if self.command.timed_out != (self.timeout_stage is not None):
            raise ProcessSupervisionError(
                "command and stage timeout verdicts differ"
            )
        if (
            not isinstance(self.observer_poll_seconds, (int, float))
            or isinstance(self.observer_poll_seconds, bool)
            or not math.isfinite(float(self.observer_poll_seconds))
            or self.observer_poll_seconds <= 0
        ):
            raise ProcessSupervisionError(
                "stage observer poll interval must be finite and positive"
            )

    @property
    def returncode(self) -> int:
        return self.command.returncode

    @property
    def timed_out(self) -> bool:
        return self.command.timed_out

    @property
    def stdout(self) -> bytes:
        return self.command.stdout

    @property
    def stderr(self) -> bytes:
        return self.command.stderr

    def to_dict(self) -> dict[str, object]:
        payload = self.command.to_dict()
        payload["schema_version"] = self.SCHEMA_VERSION
        payload["stage_supervision"] = {
            "timeouts_seconds": {
                name: float(seconds) for name, seconds in self.stage_timeouts
            },
            "observer_poll_seconds": float(self.observer_poll_seconds),
            "observations": [
                {
                    "sequence": sequence,
                    "stage": stage,
                    "state": state,
                    "monotonic_ns": monotonic_ns,
                    "event_sha256": event_sha256,
                }
                for sequence, stage, state, monotonic_ns, event_sha256
                in self.stage_observations
            ],
            "timeout_stage": self.timeout_stage,
            "timeout_stage_elapsed_seconds": (
                float(self.timeout_stage_elapsed_seconds)
                if self.timeout_stage_elapsed_seconds is not None
                else None
            ),
            "no_heartbeat_extension": True,
        }
        return payload


def _normalize_stage_observations(
    raw: Sequence[Mapping[str, object]],
    *,
    allowed_stages: frozenset[str],
) -> tuple[tuple[int, str, str, int, str], ...]:
    if isinstance(raw, (str, bytes)):
        raise ProcessSupervisionError("stage observations must be a sequence")
    normalized: list[tuple[int, str, str, int, str]] = []
    started: set[str] = set()
    completed: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "sequence",
            "stage",
            "state",
            "monotonic_ns",
            "event_sha256",
        }:
            raise ProcessSupervisionError("stage observation fields differ")
        sequence = item.get("sequence")
        stage = item.get("stage")
        state = item.get("state")
        monotonic_ns = item.get("monotonic_ns")
        event_sha256 = item.get("event_sha256")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence != len(normalized) + 1
            or not isinstance(stage, str)
            or stage not in allowed_stages
            or state not in {"started", "completed"}
            or not isinstance(monotonic_ns, int)
            or isinstance(monotonic_ns, bool)
            or monotonic_ns < 0
            or not isinstance(event_sha256, str)
            or _SHA256.fullmatch(event_sha256) is None
        ):
            raise ProcessSupervisionError("stage observation value differs")
        if state == "started":
            if stage in started or stage in completed:
                raise ProcessSupervisionError(
                    "supervised stage starts more than once"
                )
            started.add(stage)
        else:
            if stage not in started or stage in completed:
                raise ProcessSupervisionError(
                    "supervised stage completion is out of order"
                )
            completed.add(stage)
        normalized.append(
            (sequence, stage, str(state), monotonic_ns, event_sha256)
        )
    return tuple(normalized)


def read_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessIdentity:
    """Read a Linux process identity without relying on its command name."""

    _require_positive_integer(pid, "PID")
    stat_path = proc_root / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ProcessIdentityUnavailable(
            f"cannot read process identity for PID {pid}"
        ) from error
    closing = raw.rfind(")")
    first_space = raw.find(" ")
    if closing < 0 or first_space < 0 or closing + 2 > len(raw):
        raise ProcessIdentityUnavailable("process stat record is malformed")
    tail = raw[closing + 2 :].split()
    if len(tail) < 20:
        raise ProcessIdentityUnavailable("process stat record is incomplete")
    try:
        stat_pid = int(raw[:first_space])
        parent_pid = int(tail[1])
        start_time_ticks = int(tail[19])
    except (IndexError, ValueError) as error:
        raise ProcessIdentityUnavailable(
            "process stat identity fields are malformed"
        ) from error
    if stat_pid != pid:
        raise ProcessIdentityUnavailable("process stat PID differs")
    try:
        return ProcessIdentity(pid, start_time_ticks, parent_pid)
    except ProcessSupervisionError as error:
        raise ProcessIdentityUnavailable("process stat identity is invalid") from error


def _open_supervised_pidfd(pid: int) -> tuple[bool, int | None]:
    opener = getattr(os, "pidfd_open", None)
    if opener is None:
        return False, None
    try:
        return True, opener(pid, 0)
    except OSError:
        return True, None


def _signal_supervised_child(
    process: subprocess.Popen[bytes],
    *,
    expected_identity: ProcessIdentity,
    pidfd: int | None,
    requested_signal: int,
) -> bool:
    """Signal only the retained direct child, protected against PID reuse."""

    pidfd_sender = getattr(signal, "pidfd_send_signal", None)
    if pidfd is not None and callable(pidfd_sender):
        try:
            pidfd_sender(pidfd, requested_signal)
        except ProcessLookupError:
            return False
        except OSError as error:
            raise ProcessSupervisionError(
                "cannot signal supervised child through pidfd"
            ) from error
        return True
    if process.pid != expected_identity.pid:
        raise ProcessSupervisionError(
            "supervised process handle PID changed before signal"
        )
    try:
        current_identity = read_process_identity(process.pid)
    except ProcessIdentityUnavailable:
        return False
    if current_identity != expected_identity:
        raise ProcessSupervisionError(
            "supervised child identity changed before signal"
        )
    try:
        process.send_signal(requested_signal)
    except ProcessLookupError:
        return False
    except OSError as error:
        raise ProcessSupervisionError(
            "cannot signal verified supervised child"
        ) from error
    return True


def run_supervised_command(
    argv: Sequence[str],
    *,
    working_directory: str,
    environment: Mapping[str, str],
    timeout_seconds: float,
    termination_grace_seconds: float = 5.0,
) -> SupervisedCommandResult:
    """Run and reap one explicit direct child without handshake assertions."""

    if (
        isinstance(argv, (str, bytes))
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ProcessSupervisionError(
            "supervised argv must be a nonempty string list"
        )
    if (
        not isinstance(working_directory, str)
        or not Path(working_directory).is_absolute()
        or not Path(working_directory).is_dir()
    ):
        raise ProcessSupervisionError(
            "supervised working directory must be an existing absolute directory"
        )
    for value, label in (
        (timeout_seconds, "execution timeout"),
        (termination_grace_seconds, "termination grace"),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ProcessSupervisionError(
                f"supervised {label} must be finite and positive"
            )
    environment_sha256 = environment_fingerprint(environment)
    argv_tuple = tuple(argv)
    command_sha256 = command_fingerprint(
        argv_tuple,
        working_directory=working_directory,
        environment_sha256=environment_sha256,
    )
    supervisor_pid = os.getpid()
    process = subprocess.Popen(
        argv_tuple,
        cwd=working_directory,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    pidfd_supported, pidfd = _open_supervised_pidfd(process.pid)
    pidfd_closed = False
    identity: ProcessIdentity | None = None
    final_reap_count = 0
    timed_out = False
    terminate_requested = False
    kill_requested = False
    try:
        try:
            identity = read_process_identity(process.pid)
            if (
                identity.pid != process.pid
                or identity.parent_pid != supervisor_pid
            ):
                raise ProcessSupervisionError(
                    "spawned command is not the supervisor's direct child"
                )
        except ProcessSupervisionError:
            process.kill()
            process.communicate()
            final_reap_count = 1
            raise

        try:
            stdout, stderr = process.communicate(
                timeout=float(timeout_seconds)
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_requested = True
            _signal_supervised_child(
                process,
                expected_identity=identity,
                pidfd=pidfd,
                requested_signal=signal.SIGTERM,
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=float(termination_grace_seconds)
                )
            except subprocess.TimeoutExpired:
                kill_requested = True
                _signal_supervised_child(
                    process,
                    expected_identity=identity,
                    pidfd=pidfd,
                    requested_signal=signal.SIGKILL,
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=float(termination_grace_seconds)
                    )
                except subprocess.TimeoutExpired as error:
                    raise ProcessSupervisionError(
                        "supervised child did not exit after verified SIGKILL"
                    ) from error
        final_reap_count = 1
        if not isinstance(process.returncode, int) or isinstance(
            process.returncode, bool
        ):
            raise ProcessSupervisionError(
                "supervised child lacks a final return code"
            )
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise ProcessSupervisionError(
                "supervised child output was not captured as bytes"
            )
    finally:
        if pidfd is not None:
            os.close(pidfd)
            pidfd_closed = True

    if identity is None:
        raise ProcessSupervisionError(
            "supervised child identity was not retained"
        )
    return SupervisedCommandResult(
        argv=argv_tuple,
        working_directory=working_directory,
        environment_sha256=environment_sha256,
        command_sha256=command_sha256,
        process_identity=identity,
        supervisor_pid=supervisor_pid,
        timeout_seconds=float(timeout_seconds),
        timed_out=timed_out,
        terminate_requested=terminate_requested,
        kill_requested=kill_requested,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        pidfd_supported=pidfd_supported,
        pidfd_opened=pidfd is not None,
        pidfd=pidfd,
        pidfd_closed=pidfd_closed,
        direct_child_verified=True,
        process_handle_retained=True,
        final_reap_completed=final_reap_count == 1,
        final_reap_count=final_reap_count,
    )


def run_stage_supervised_command(
    argv: Sequence[str],
    *,
    working_directory: str,
    environment: Mapping[str, str],
    stage_timeouts: Mapping[str, float],
    stage_observer: Callable[[], Sequence[Mapping[str, object]]],
    startup_stage: str,
    transition_stage: str,
    observer_poll_seconds: float = 5.0,
    termination_grace_seconds: float = 5.0,
) -> StageSupervisedCommandResult:
    """Run one child with immutable, non-heartbeat stage deadlines."""

    if (
        isinstance(argv, (str, bytes))
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ProcessSupervisionError(
            "stage-supervised argv must be a nonempty string list"
        )
    if (
        not isinstance(working_directory, str)
        or not Path(working_directory).is_absolute()
        or not Path(working_directory).is_dir()
    ):
        raise ProcessSupervisionError(
            "stage-supervised working directory must be an existing absolute directory"
        )
    if not isinstance(stage_timeouts, Mapping) or not stage_timeouts:
        raise ProcessSupervisionError("stage timeout contract is empty")
    normalized_timeouts: list[tuple[str, float]] = []
    seen_stages: set[str] = set()
    for stage, timeout in stage_timeouts.items():
        _require_identifier(stage, "supervised stage")
        if stage in seen_stages:
            raise ProcessSupervisionError("supervised stage is duplicated")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ProcessSupervisionError(
                "stage timeout must be finite and positive"
            )
        seen_stages.add(stage)
        normalized_timeouts.append((stage, float(timeout)))
    for stage, label in (
        (startup_stage, "startup stage"),
        (transition_stage, "transition stage"),
    ):
        _require_identifier(stage, label)
        if stage not in seen_stages:
            raise ProcessSupervisionError(
                f"{label} is absent from the timeout contract"
            )
    for value, label in (
        (observer_poll_seconds, "observer poll interval"),
        (termination_grace_seconds, "termination grace"),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ProcessSupervisionError(
                f"stage-supervised {label} must be finite and positive"
            )
    if not callable(stage_observer):
        raise ProcessSupervisionError("stage observer is not callable")

    timeout_by_stage = dict(normalized_timeouts)
    allowed_stages = frozenset(timeout_by_stage)
    environment_sha256 = environment_fingerprint(environment)
    argv_tuple = tuple(argv)
    command_sha256 = command_fingerprint(
        argv_tuple,
        working_directory=working_directory,
        environment_sha256=environment_sha256,
    )
    supervisor_pid = os.getpid()
    process_started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        argv_tuple,
        cwd=working_directory,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    pidfd_supported, pidfd = _open_supervised_pidfd(process.pid)
    pidfd_closed = False
    identity: ProcessIdentity | None = None
    observations: tuple[tuple[int, str, str, int, str], ...] = ()
    active_stage = startup_stage
    active_stage_started_ns = process_started_ns
    timeout_stage: str | None = None
    timeout_stage_elapsed_seconds: float | None = None
    final_reap_count = 0
    terminate_requested = False
    kill_requested = False
    stdout = b""
    stderr = b""

    def observe() -> None:
        nonlocal observations, active_stage, active_stage_started_ns
        current = _normalize_stage_observations(
            stage_observer(),
            allowed_stages=allowed_stages,
        )
        if (
            len(current) < len(observations)
            or current[: len(observations)] != observations
        ):
            raise ProcessSupervisionError(
                "stage evidence changed after observation"
            )
        now_ns = time.monotonic_ns()
        for event in current[len(observations) :]:
            _, stage, state, event_ns, _ = event
            if event_ns < process_started_ns or event_ns > now_ns:
                raise ProcessSupervisionError(
                    "stage event monotonic time is outside process lifetime"
                )
            if state == "started":
                active_stage = stage
                active_stage_started_ns = event_ns
            else:
                active_stage = transition_stage
                active_stage_started_ns = event_ns
        observations = current

    try:
        try:
            identity = read_process_identity(process.pid)
            if (
                identity.pid != process.pid
                or identity.parent_pid != supervisor_pid
            ):
                raise ProcessSupervisionError(
                    "spawned command is not the supervisor's direct child"
                )
        except ProcessSupervisionError:
            process.kill()
            process.communicate()
            final_reap_count = 1
            raise

        try:
            while True:
                observe()
                elapsed_seconds = (
                    time.monotonic_ns() - active_stage_started_ns
                ) / 1_000_000_000.0
                remaining_seconds = (
                    timeout_by_stage[active_stage] - elapsed_seconds
                )
                if remaining_seconds <= 0:
                    timeout_stage = active_stage
                    timeout_stage_elapsed_seconds = elapsed_seconds
                    break
                try:
                    stdout, stderr = process.communicate(
                        timeout=min(
                            float(observer_poll_seconds),
                            remaining_seconds,
                        )
                    )
                    final_reap_count = 1
                    observe()
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            if process.returncode is None and identity is not None:
                _signal_supervised_child(
                    process,
                    expected_identity=identity,
                    pidfd=pidfd,
                    requested_signal=signal.SIGKILL,
                )
            try:
                process.communicate(timeout=float(termination_grace_seconds))
                final_reap_count = 1
            except subprocess.TimeoutExpired as error:
                raise ProcessSupervisionError(
                    "stage-supervised child did not exit after verified SIGKILL"
                ) from error
            raise

        if timeout_stage is not None:
            terminate_requested = True
            _signal_supervised_child(
                process,
                expected_identity=identity,
                pidfd=pidfd,
                requested_signal=signal.SIGTERM,
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=float(termination_grace_seconds)
                )
            except subprocess.TimeoutExpired:
                kill_requested = True
                _signal_supervised_child(
                    process,
                    expected_identity=identity,
                    pidfd=pidfd,
                    requested_signal=signal.SIGKILL,
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=float(termination_grace_seconds)
                    )
                except subprocess.TimeoutExpired as error:
                    raise ProcessSupervisionError(
                        "stage-supervised child did not exit after verified SIGKILL"
                    ) from error
            final_reap_count = 1
            observe()
        if not isinstance(process.returncode, int) or isinstance(
            process.returncode, bool
        ):
            raise ProcessSupervisionError(
                "stage-supervised child lacks a final return code"
            )
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise ProcessSupervisionError(
                "stage-supervised child output was not captured as bytes"
            )
    finally:
        if pidfd is not None:
            os.close(pidfd)
            pidfd_closed = True

    if identity is None:
        raise ProcessSupervisionError(
            "stage-supervised child identity was not retained"
        )
    nontransition_stage_total = sum(
        timeout for stage, timeout in normalized_timeouts
        if stage != transition_stage
    )
    evidence_total_timeout = (
        nontransition_stage_total
        + timeout_by_stage[transition_stage] * len(normalized_timeouts)
    )
    command = SupervisedCommandResult(
        argv=argv_tuple,
        working_directory=working_directory,
        environment_sha256=environment_sha256,
        command_sha256=command_sha256,
        process_identity=identity,
        supervisor_pid=supervisor_pid,
        timeout_seconds=evidence_total_timeout,
        timed_out=timeout_stage is not None,
        terminate_requested=terminate_requested,
        kill_requested=kill_requested,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        pidfd_supported=pidfd_supported,
        pidfd_opened=pidfd is not None,
        pidfd=pidfd,
        pidfd_closed=pidfd_closed,
        direct_child_verified=True,
        process_handle_retained=True,
        final_reap_completed=final_reap_count == 1,
        final_reap_count=final_reap_count,
    )
    return StageSupervisedCommandResult(
        command=command,
        stage_timeouts=tuple(normalized_timeouts),
        stage_observations=observations,
        timeout_stage=timeout_stage,
        timeout_stage_elapsed_seconds=timeout_stage_elapsed_seconds,
        observer_poll_seconds=float(observer_poll_seconds),
    )


@dataclass(frozen=True)
class WorkerIdentity:
    """Durable scientific identity of the directly spawned worker."""

    process: ProcessIdentity
    run_id: str
    gpu_uuid: str
    spawned_at_utc: str
    expected_command_fingerprint: str

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "run ID")
        _require_gpu_uuid(self.gpu_uuid)
        if not isinstance(self.spawned_at_utc, str) or not self.spawned_at_utc:
            raise ProcessSupervisionError("spawn timestamp is absent")
        _require_sha256(
            self.expected_command_fingerprint,
            "expected command fingerprint",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.process.to_dict(),
            "run_id": self.run_id,
            "gpu_uuid": self.gpu_uuid,
            "spawned_at_utc": self.spawned_at_utc,
            "expected_command_fingerprint": self.expected_command_fingerprint,
        }


@dataclass(frozen=True)
class ProcessHandleMetadata:
    """Serializable proof that a direct process handle was retained."""

    process_handle_kind: str
    process_handle_retained: bool
    pidfd_supported: bool
    pidfd_opened: bool
    pidfd: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.process_handle_kind, str) or not self.process_handle_kind:
            raise ProcessSupervisionError("process handle kind is absent")
        if not isinstance(self.process_handle_retained, bool):
            raise ProcessSupervisionError("process handle retention flag is invalid")
        if not isinstance(self.pidfd_supported, bool) or not isinstance(
            self.pidfd_opened, bool
        ):
            raise ProcessSupervisionError("pidfd support flags are invalid")
        if not self.process_handle_retained:
            raise ProcessSupervisionError("process handle must be retained")
        if self.pidfd_opened and not self.pidfd_supported:
            raise ProcessSupervisionError("pidfd cannot be opened when unsupported")
        if self.pidfd_opened != (self.pidfd is not None):
            raise ProcessSupervisionError("pidfd metadata is inconsistent")
        if self.pidfd is not None:
            _require_nonnegative_integer(self.pidfd, "pidfd")

    def to_dict(self) -> dict[str, object]:
        return {
            "process_handle_kind": self.process_handle_kind,
            "process_handle_retained": self.process_handle_retained,
            "pidfd_supported": self.pidfd_supported,
            "pidfd_opened": self.pidfd_opened,
            "pidfd": self.pidfd,
        }


@dataclass(frozen=True)
class HandshakeEvent:
    """One strict worker or supervisor lifecycle event."""

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase3-worker-handshake-event-1.0.0"

    sequence: int
    stage: HandshakeStage
    recorded_at_utc: str
    run_id: str
    gpu_uuid: str
    pid: int
    process_start_time_ticks: int
    parent_pid: int
    command_fingerprint: str
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.sequence != self.stage.sequence:
            raise ProcessSupervisionError("handshake sequence and stage differ")
        if not isinstance(self.recorded_at_utc, str) or not self.recorded_at_utc:
            raise ProcessSupervisionError("handshake timestamp is absent")
        _require_identifier(self.run_id, "handshake run ID")
        _require_gpu_uuid(self.gpu_uuid)
        _require_positive_integer(self.pid, "handshake PID")
        _require_nonnegative_integer(
            self.process_start_time_ticks,
            "handshake process start time",
        )
        _require_positive_integer(self.parent_pid, "handshake parent PID")
        _require_sha256(self.command_fingerprint, "handshake command fingerprint")
        if self.stage is HandshakeStage.EVIDENCE_FLUSHED:
            if self.evidence_sha256 is None:
                raise ProcessSupervisionError(
                    "evidence_flushed requires the durable evidence digest"
                )
            _require_sha256(self.evidence_sha256, "worker evidence digest")
        elif self.evidence_sha256 is not None:
            raise ProcessSupervisionError(
                "only evidence_flushed may bind an evidence digest"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "sequence": self.sequence,
            "stage": self.stage.value,
            "recorded_at_utc": self.recorded_at_utc,
            "run_id": self.run_id,
            "gpu_uuid": self.gpu_uuid,
            "pid": self.pid,
            "process_start_time_ticks": self.process_start_time_ticks,
            "parent_pid": self.parent_pid,
            "command_fingerprint": self.command_fingerprint,
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HandshakeEvent:
        """Parse one event without accepting unknown or coerced fields."""

        expected = {
            "schema_version",
            "sequence",
            "stage",
            "recorded_at_utc",
            "run_id",
            "gpu_uuid",
            "pid",
            "process_start_time_ticks",
            "parent_pid",
            "command_fingerprint",
            "evidence_sha256",
        }
        if set(payload) != expected or payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ProcessSupervisionError("handshake event fields differ")
        sequence = payload.get("sequence")
        stage_value = payload.get("stage")
        recorded_at_utc = payload.get("recorded_at_utc")
        run_id = payload.get("run_id")
        gpu_uuid = payload.get("gpu_uuid")
        pid = payload.get("pid")
        start_ticks = payload.get("process_start_time_ticks")
        parent_pid = payload.get("parent_pid")
        fingerprint = payload.get("command_fingerprint")
        evidence = payload.get("evidence_sha256")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not isinstance(stage_value, str)
            or not isinstance(recorded_at_utc, str)
            or not isinstance(run_id, str)
            or not isinstance(gpu_uuid, str)
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or not isinstance(start_ticks, int)
            or isinstance(start_ticks, bool)
            or not isinstance(parent_pid, int)
            or isinstance(parent_pid, bool)
            or not isinstance(fingerprint, str)
            or evidence is not None
            and not isinstance(evidence, str)
        ):
            raise ProcessSupervisionError("handshake event value types differ")
        try:
            stage = HandshakeStage(stage_value)
        except ValueError as error:
            raise ProcessSupervisionError("handshake event stage is invalid") from error
        return cls(
            sequence=sequence,
            stage=stage,
            recorded_at_utc=recorded_at_utc,
            run_id=run_id,
            gpu_uuid=gpu_uuid,
            pid=pid,
            process_start_time_ticks=start_ticks,
            parent_pid=parent_pid,
            command_fingerprint=fingerprint,
            evidence_sha256=evidence,
        )


def handshake_event_path(directory: Path, stage: HandshakeStage) -> Path:
    """Return the fixed lexical path for one ordered handshake event."""

    return directory / f"{stage.sequence:04d}-{stage.value}.json"


def _canonical_event_bytes(event: HandshakeEvent) -> bytes:
    return (
        json.dumps(
            event.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _published_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return fields that cannot change during the legitimate link transition."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _validate_private_publication_parent(path: Path) -> Path:
    if not path.is_absolute() or not path.name:
        raise ProcessSupervisionError(
            "published evidence path must be absolute and named"
        )
    parent = path.parent
    try:
        metadata = parent.lstat()
    except OSError as error:
        raise ProcessSupervisionError(
            "published evidence parent is absent"
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ProcessSupervisionError(
            "published evidence parent must be private and process-owned"
        )
    return parent


def publish_bytes_no_replace(path: Path, payload: bytes) -> None:
    """Fsync complete bytes before exclusive same-directory publication."""

    if type(payload) is not bytes or not payload:
        raise ProcessSupervisionError(
            "published evidence payload must be nonempty bytes"
        )
    parent = _validate_private_publication_parent(path)
    temporary = parent / (
        f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except OSError as error:
        raise ProcessSupervisionError(
            "published evidence temporary path is unavailable"
        ) from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProcessSupervisionError(
                    "published evidence write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except OSError as error:
        raise ProcessSupervisionError(
            "published evidence target already exists or is unsafe"
        ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(parent)


def read_published_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    """Read complete published bytes through one no-follow descriptor."""

    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ProcessSupervisionError(
            "published evidence size limit is invalid"
        )
    _validate_private_publication_parent(path)
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ProcessSupervisionError(
            "published evidence cannot be inspected"
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.geteuid()
        or before.st_nlink not in {1, 2}
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise ProcessSupervisionError("published evidence file is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise ProcessSupervisionError(
            "platform lacks no-follow publication reads"
        )
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProcessSupervisionError(
            "published evidence cannot be opened safely"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            _published_file_identity(opened)
            != _published_file_identity(before)
            or opened.st_nlink not in {1, 2}
        ):
            raise ProcessSupervisionError(
                "published evidence changed while opening"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(16 * 1024, remaining))
            if not chunk:
                raise ProcessSupervisionError(
                    "published evidence was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProcessSupervisionError("published evidence grew while reading")
        after = os.fstat(descriptor)
        if (
            _published_file_identity(after)
            != _published_file_identity(opened)
            or after.st_nlink not in {1, 2}
        ):
            raise ProcessSupervisionError(
                "published evidence changed while reading"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_handshake_event(directory: Path, event: HandshakeEvent) -> Path:
    """Atomically publish one fsynced, no-replace event in a real directory."""

    if not directory.is_absolute():
        raise ProcessSupervisionError("handshake directory must be absolute")
    target = handshake_event_path(directory, event.stage)
    publish_bytes_no_replace(target, _canonical_event_bytes(event))
    return target


def read_handshake_event(path: Path) -> HandshakeEvent:
    """Read one atomically published canonical handshake event."""

    raw = read_published_bytes(path, maximum_bytes=16 * 1024)
    if not raw.endswith(b"\n") or b"\n" in raw[:-1] or b"\r" in raw:
        raise ProcessSupervisionError("handshake event framing is invalid")
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError("duplicate key")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            raw[:-1].decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("non-finite JSON")
            ),
        )
    except (UnicodeError, ValueError) as error:
        raise ProcessSupervisionError("handshake event JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ProcessSupervisionError("handshake event is not an object")
    event = HandshakeEvent.from_dict(payload)
    if raw != _canonical_event_bytes(event):
        raise ProcessSupervisionError("handshake event is not canonical")
    if path.name != handshake_event_path(path.parent, event.stage).name:
        raise ProcessSupervisionError("handshake event filename differs")
    return event


@dataclass(frozen=True)
class DeviceProcessObservation:
    """One raw compute-process identity from device telemetry."""

    gpu_uuid: str
    pid: int
    process_start_time_ticks: int | None

    def __post_init__(self) -> None:
        _require_gpu_uuid(self.gpu_uuid)
        _require_positive_integer(self.pid, "observed PID")
        if self.process_start_time_ticks is not None:
            _require_nonnegative_integer(
                self.process_start_time_ticks,
                "observed process start time",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu_uuid": self.gpu_uuid,
            "pid": self.pid,
            "process_start_time_ticks": self.process_start_time_ticks,
        }


@dataclass(frozen=True)
class SnapshotVerdict:
    """Exact classification of one device-process snapshot."""

    disposition: SnapshotDisposition
    owned: tuple[DeviceProcessObservation, ...]
    foreign: tuple[DeviceProcessObservation, ...]
    pid_reuse: tuple[DeviceProcessObservation, ...]
    unverified: tuple[DeviceProcessObservation, ...]

    @property
    def hard_failure(self) -> bool:
        return self.disposition not in {
            SnapshotDisposition.CLEAN,
            SnapshotDisposition.OWNED_ONLY,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "hard_failure": self.hard_failure,
            "owned": [item.to_dict() for item in self.owned],
            "foreign": [item.to_dict() for item in self.foreign],
            "pid_reuse": [item.to_dict() for item in self.pid_reuse],
            "unverified": [item.to_dict() for item in self.unverified],
        }


@dataclass(frozen=True)
class OwnershipOutcome:
    """Terminal ownership result after the supervisor has reaped the worker."""

    disposition: OwnershipDisposition
    reason: str
    returncode: int
    observed_stages: tuple[HandshakeStage, ...]
    missing_worker_stages: tuple[HandshakeStage, ...]
    evidence_flushed: bool
    worker_exiting_observed: bool
    full_handshake_observed: bool
    exclusivity_passed: bool
    owned_completion_basis: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "reason": self.reason,
            "returncode": self.returncode,
            "observed_stages": [stage.value for stage in self.observed_stages],
            "missing_worker_stages": [
                stage.value for stage in self.missing_worker_stages
            ],
            "evidence_flushed": self.evidence_flushed,
            "worker_exiting_observed": self.worker_exiting_observed,
            "full_handshake_observed": self.full_handshake_observed,
            "exclusivity_passed": self.exclusivity_passed,
            "owned_completion_basis": self.owned_completion_basis,
        }


class RunOwnedProcessRegistry:
    """Retain worker ownership independently of procfs lifetime and polling."""

    SCHEMA_VERSION = "kvbench-phase3-process-registry-3.0.0"
    OWNED_COMPLETION_POLICY = (
        "zero_exit_after_durable_evidence_flush_worker_exiting_optional"
    )

    def __init__(
        self,
        *,
        identity: WorkerIdentity,
        process_handle: object,
        handle_metadata: ProcessHandleMetadata,
    ) -> None:
        self._identity = identity
        self._process_handle = process_handle
        self._handle_metadata = handle_metadata
        self._worker_events: list[HandshakeEvent] = []
        self._reaped_event: HandshakeEvent | None = None
        self._exit_observed = False
        self._proc_disappeared = False
        self._returncode: int | None = None
        self._snapshot_count = 0
        self._registered_compute_observed = False
        self._hard_disposition: OwnershipDisposition | None = None
        self._hard_reason: str | None = None

    @classmethod
    def register_spawn(
        cls,
        *,
        process_identity: ProcessIdentity,
        expected_supervisor_pid: int,
        process_handle: object,
        pidfd_supported: bool,
        pidfd: int | None,
        run_id: str,
        gpu_uuid: str,
        spawned_at_utc: str,
        expected_command_fingerprint: str,
    ) -> RunOwnedProcessRegistry:
        """Register the direct child before any telemetry poll or reap."""

        _require_positive_integer(expected_supervisor_pid, "supervisor PID")
        if process_identity.parent_pid != expected_supervisor_pid:
            raise ProcessSupervisionError(
                "spawned worker parent PID differs from the supervisor"
            )
        if process_handle is None:
            raise ProcessSupervisionError("spawned worker process handle is absent")
        handle_kind = (
            f"{type(process_handle).__module__}."
            f"{type(process_handle).__qualname__}"
        )
        metadata = ProcessHandleMetadata(
            process_handle_kind=handle_kind,
            process_handle_retained=True,
            pidfd_supported=pidfd_supported,
            pidfd_opened=pidfd is not None,
            pidfd=pidfd,
        )
        identity = WorkerIdentity(
            process=process_identity,
            run_id=run_id,
            gpu_uuid=gpu_uuid,
            spawned_at_utc=spawned_at_utc,
            expected_command_fingerprint=expected_command_fingerprint,
        )
        return cls(
            identity=identity,
            process_handle=process_handle,
            handle_metadata=metadata,
        )

    @property
    def identity(self) -> WorkerIdentity:
        return self._identity

    @property
    def process_handle(self) -> object:
        return self._process_handle

    @property
    def pidfd(self) -> int | None:
        return self._handle_metadata.pidfd

    @property
    def exit_observed(self) -> bool:
        return self._exit_observed

    @property
    def reaped(self) -> bool:
        return self._reaped_event is not None

    @property
    def observed_worker_stages(self) -> tuple[HandshakeStage, ...]:
        return tuple(event.stage for event in self._worker_events)

    def _event(
        self,
        stage: HandshakeStage,
        *,
        recorded_at_utc: str,
        command_fingerprint_value: str,
        evidence_sha256: str | None,
    ) -> HandshakeEvent:
        process = self._identity.process
        return HandshakeEvent(
            sequence=stage.sequence,
            stage=stage,
            recorded_at_utc=recorded_at_utc,
            run_id=self._identity.run_id,
            gpu_uuid=self._identity.gpu_uuid,
            pid=process.pid,
            process_start_time_ticks=process.start_time_ticks,
            parent_pid=process.parent_pid,
            command_fingerprint=command_fingerprint_value,
            evidence_sha256=evidence_sha256,
        )

    def record_worker_stage(
        self,
        stage: HandshakeStage,
        *,
        recorded_at_utc: str,
        observed_command_fingerprint: str | None = None,
        evidence_sha256: str | None = None,
    ) -> HandshakeEvent:
        """Create and ingest the next worker-owned handshake event."""

        fingerprint = (
            self._identity.expected_command_fingerprint
            if observed_command_fingerprint is None
            else observed_command_fingerprint
        )
        event = self._event(
            stage,
            recorded_at_utc=recorded_at_utc,
            command_fingerprint_value=fingerprint,
            evidence_sha256=evidence_sha256,
        )
        self.ingest_worker_event(event)
        return event

    def ingest_worker_event(self, event: HandshakeEvent) -> None:
        """Validate identity and exact prefix ordering before retaining an event."""

        if self.reaped:
            raise ProcessSupervisionError("worker event arrived after supervisor reap")
        if event.stage is HandshakeStage.SUPERVISOR_REAPED:
            raise ProcessSupervisionError("worker cannot publish supervisor_reaped")
        if len(self._worker_events) >= len(_WORKER_STAGES):
            raise ProcessSupervisionError("worker handshake already reached its end")
        expected_stage = _WORKER_STAGES[len(self._worker_events)]
        if event.stage is not expected_stage:
            raise ProcessSupervisionError(
                f"out-of-order worker stage: expected {expected_stage.value}"
            )
        process = self._identity.process
        if (
            event.run_id != self._identity.run_id
            or event.gpu_uuid != self._identity.gpu_uuid
            or event.pid != process.pid
            or event.process_start_time_ticks != process.start_time_ticks
            or event.parent_pid != process.parent_pid
        ):
            raise ProcessSupervisionError("worker handshake identity differs")
        if event.command_fingerprint != self._identity.expected_command_fingerprint:
            raise ProcessSupervisionError("worker command fingerprint differs")
        self._worker_events.append(event)

    def refresh_handshake_directory(self, directory: Path) -> int:
        """Ingest every newly visible contiguous worker event exactly once."""

        ingested = 0
        while len(self._worker_events) < len(_WORKER_STAGES):
            stage = _WORKER_STAGES[len(self._worker_events)]
            path = handshake_event_path(directory, stage)
            if not path.exists() and not path.is_symlink():
                break
            self.ingest_worker_event(read_handshake_event(path))
            ingested += 1
        return ingested

    def note_exit_observed(self) -> None:
        """Retain a pidfd/waitid exit observation without reaping the child."""

        self._exit_observed = True

    def record_supervisor_reaped(
        self,
        returncode: int,
        *,
        recorded_at_utc: str,
    ) -> HandshakeEvent:
        """Record the sole final reap after non-reaping exit observation."""

        if not self._exit_observed:
            raise ProcessSupervisionError(
                "supervisor cannot reap before non-reaping exit observation"
            )
        if self.reaped:
            raise ProcessSupervisionError("worker has already been reaped")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise ProcessSupervisionError("worker return code must be an integer")
        event = self._event(
            HandshakeStage.SUPERVISOR_REAPED,
            recorded_at_utc=recorded_at_utc,
            command_fingerprint_value=self._identity.expected_command_fingerprint,
            evidence_sha256=None,
        )
        self._returncode = returncode
        self._reaped_event = event
        return event

    def _record_hard_failure(
        self,
        disposition: OwnershipDisposition,
        reason: str,
    ) -> None:
        priorities = {
            OwnershipDisposition.UNVERIFIED_PROCESS_DETECTED: 1,
            OwnershipDisposition.FOREIGN_PROCESS_DETECTED: 2,
            OwnershipDisposition.PID_REUSE_DETECTED: 3,
        }
        existing_priority = (
            0
            if self._hard_disposition is None
            else priorities.get(self._hard_disposition, 0)
        )
        if priorities.get(disposition, 0) > existing_priority:
            self._hard_disposition = disposition
            self._hard_reason = reason

    def note_unverified_device_evidence(self, reason: str) -> None:
        """Make a malformed or failed device query a sticky exclusivity failure."""

        if not isinstance(reason, str) or not reason:
            raise ProcessSupervisionError(
                "unverified device evidence reason must be nonempty"
            )
        self._record_hard_failure(
            OwnershipDisposition.UNVERIFIED_PROCESS_DETECTED,
            reason,
        )

    def observe_proc_start_time(
        self,
        observed_start_time_ticks: int | None,
    ) -> ProcObservationDisposition:
        """Retain ownership if procfs vanishes; reject a changed start time."""

        if observed_start_time_ticks is None:
            self._proc_disappeared = True
            return ProcObservationDisposition.DISAPPEARED_RETAINED
        _require_nonnegative_integer(
            observed_start_time_ticks,
            "observed process start time",
        )
        if observed_start_time_ticks == self._identity.process.start_time_ticks:
            return ProcObservationDisposition.EXACT
        self._record_hard_failure(
            OwnershipDisposition.PID_REUSE_DETECTED,
            "registered PID has a different process start time",
        )
        return ProcObservationDisposition.PID_REUSE_DETECTED

    def classify_device_snapshot(
        self,
        observations: Sequence[DeviceProcessObservation],
    ) -> SnapshotVerdict:
        """Fail on every process not joined to the exact registered worker."""

        if isinstance(observations, (str, bytes)):
            raise ProcessSupervisionError("device observations must be a sequence")
        seen: set[tuple[str, int, int | None]] = set()
        owned: list[DeviceProcessObservation] = []
        foreign: list[DeviceProcessObservation] = []
        pid_reuse: list[DeviceProcessObservation] = []
        unverified: list[DeviceProcessObservation] = []
        process = self._identity.process
        for observation in observations:
            if not isinstance(observation, DeviceProcessObservation):
                raise ProcessSupervisionError("device observation type is invalid")
            key = (
                observation.gpu_uuid,
                observation.pid,
                observation.process_start_time_ticks,
            )
            if key in seen:
                raise ProcessSupervisionError("duplicate device process observation")
            seen.add(key)
            if observation.pid == process.pid and (
                observation.process_start_time_ticks is not None
                and observation.process_start_time_ticks != process.start_time_ticks
            ):
                pid_reuse.append(observation)
            elif (
                observation.pid == process.pid
                and observation.gpu_uuid == self._identity.gpu_uuid
                and observation.process_start_time_ticks == process.start_time_ticks
            ):
                owned.append(observation)
            elif (
                observation.pid == process.pid
                and observation.gpu_uuid == self._identity.gpu_uuid
                and observation.process_start_time_ticks is None
                and (self._proc_disappeared or self._exit_observed)
            ):
                owned.append(observation)
            elif (
                observation.pid == process.pid
                and observation.gpu_uuid == self._identity.gpu_uuid
                and observation.process_start_time_ticks is None
            ):
                unverified.append(observation)
            else:
                foreign.append(observation)
        if pid_reuse:
            disposition = SnapshotDisposition.PID_REUSE_DETECTED
            self._record_hard_failure(
                OwnershipDisposition.PID_REUSE_DETECTED,
                "device snapshot observed registered PID with a new start time",
            )
        elif foreign:
            disposition = SnapshotDisposition.FOREIGN_PROCESS_DETECTED
            self._record_hard_failure(
                OwnershipDisposition.FOREIGN_PROCESS_DETECTED,
                "device snapshot contains an unregistered process",
            )
        elif unverified:
            disposition = SnapshotDisposition.UNVERIFIED_REGISTERED_PID
            self._record_hard_failure(
                OwnershipDisposition.UNVERIFIED_PROCESS_DETECTED,
                "device snapshot PID lacks a retained identity basis",
            )
        elif owned:
            disposition = SnapshotDisposition.OWNED_ONLY
            self._registered_compute_observed = True
        else:
            disposition = SnapshotDisposition.CLEAN
        self._snapshot_count += 1
        return SnapshotVerdict(
            disposition=disposition,
            owned=tuple(owned),
            foreign=tuple(foreign),
            pid_reuse=tuple(pid_reuse),
            unverified=tuple(unverified),
        )

    def terminal_outcome(self) -> OwnershipOutcome:
        """Derive the terminal verdict without re-reading procfs."""

        if self._reaped_event is None or self._returncode is None:
            raise ProcessSupervisionError("worker has not been supervisor-reaped")
        worker_stages = self.observed_worker_stages
        observed = (*worker_stages, HandshakeStage.SUPERVISOR_REAPED)
        missing = tuple(stage for stage in _WORKER_STAGES if stage not in worker_stages)
        evidence_flushed = HandshakeStage.EVIDENCE_FLUSHED in worker_stages
        worker_exiting = HandshakeStage.WORKER_EXITING in worker_stages
        full = worker_stages == _WORKER_STAGES
        if self._hard_disposition is not None:
            disposition = self._hard_disposition
            reason = self._hard_reason or "process exclusivity failed"
            exclusivity_passed = False
            owned_completion_basis = None
        elif self._returncode == 0 and evidence_flushed:
            disposition = OwnershipDisposition.OWNED_COMPLETED
            reason = (
                "registered worker exited after durable evidence flush"
                if not worker_exiting
                else "registered worker completed the ordered handshake"
            )
            exclusivity_passed = True
            owned_completion_basis = (
                "zero_exit_after_evidence_flushed"
                if not worker_exiting
                else "full_ordered_handshake_zero_exit"
            )
        else:
            disposition = OwnershipDisposition.OWNED_WORKER_FAILURE
            reason = (
                "registered worker exited before evidence_flushed"
                if not evidence_flushed
                else f"registered worker exited with return code {self._returncode}"
            )
            exclusivity_passed = True
            owned_completion_basis = None
        return OwnershipOutcome(
            disposition=disposition,
            reason=reason,
            returncode=self._returncode,
            observed_stages=observed,
            missing_worker_stages=missing,
            evidence_flushed=evidence_flushed,
            worker_exiting_observed=worker_exiting,
            full_handshake_observed=full,
            exclusivity_passed=exclusivity_passed,
            owned_completion_basis=owned_completion_basis,
        )

    def to_evidence(self) -> dict[str, object]:
        """Serialize registry state without dereferencing a vanished PID."""

        events = [event.to_dict() for event in self._worker_events]
        if self._reaped_event is not None:
            events.append(self._reaped_event.to_dict())
        outcome = None if not self.reaped else self.terminal_outcome().to_dict()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "owned_completion_policy": self.OWNED_COMPLETION_POLICY,
            "evidence_flushed_required_for_owned_completion": True,
            "zero_returncode_required_for_owned_completion": True,
            "worker_exiting_required_for_owned_completion": False,
            "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed": True,
            "identity": self._identity.to_dict(),
            "handle": self._handle_metadata.to_dict(),
            "handshake_events": events,
            "exit_observed_without_reaping": self._exit_observed,
            "supervisor_reaped": self.reaped,
            "proc_disappeared_after_registration": self._proc_disappeared,
            "device_snapshot_count": self._snapshot_count,
            "registered_compute_observed": self._registered_compute_observed,
            "outcome": outcome,
        }
