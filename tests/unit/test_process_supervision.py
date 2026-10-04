"""Deterministic controls for Phase 3 run-owned process supervision."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from kvbench.runtime import process_supervision
from kvbench.runtime.phase3_coordinator import (
    PYTHON_EXECUTABLE,
    Phase3CoordinatorError,
    _nonreaping_exit_observed,
    _process_snapshot,
    _reap_registered_worker,
    _registry_snapshot_verdict,
    _terminate_registered_worker,
    _terminate_unregistered_worker,
    _worker_argv as coordinator_worker_argv,
)
from kvbench.runtime.phase3_worker import _worker_argv as worker_worker_argv
from kvbench.runtime.process_supervision import (
    DeviceProcessObservation,
    HandshakeEvent,
    HandshakeStage,
    OwnershipDisposition,
    ProcObservationDisposition,
    ProcessIdentity,
    ProcessIdentityUnavailable,
    ProcessSupervisionError,
    RunOwnedProcessRegistry,
    SnapshotDisposition,
    command_fingerprint,
    publish_bytes_no_replace,
    read_published_bytes,
    read_handshake_event,
    read_process_identity,
    write_handshake_event,
)


ENVIRONMENT_SHA256 = "1" * 64
EVIDENCE_SHA256 = "2" * 64
OTHER_SHA256 = "3" * 64
RECORDED_AT = "2026-07-22T12:00:00.000000Z"


class FakeProcessHandle:
    """Non-reaping process handle used by deterministic unit tests."""


def expected_fingerprint() -> str:
    return command_fingerprint(
        ("/usr/bin/python3", "-m", "kvbench", "phase3-worker"),
        working_directory="/home/rockrock/cmu_paper",
        environment_sha256=ENVIRONMENT_SHA256,
    )


def make_registry(
    *,
    pidfd_supported: bool = True,
    pidfd: int | None = 9,
) -> RunOwnedProcessRegistry:
    return RunOwnedProcessRegistry.register_spawn(
        process_identity=ProcessIdentity(
            pid=432362,
            start_time_ticks=10973359,
            parent_pid=431000,
        ),
        expected_supervisor_pid=431000,
        process_handle=FakeProcessHandle(),
        pidfd_supported=pidfd_supported,
        pidfd=pidfd,
        run_id="phase3-unit-run",
        gpu_uuid="GPU-unit-0001",
        spawned_at_utc=RECORDED_AT,
        expected_command_fingerprint=expected_fingerprint(),
    )


def record_through(
    registry: RunOwnedProcessRegistry,
    terminal_stage: HandshakeStage,
) -> None:
    for stage in tuple(HandshakeStage)[:-1]:
        registry.record_worker_stage(
            stage,
            recorded_at_utc=RECORDED_AT,
            evidence_sha256=(
                EVIDENCE_SHA256
                if stage is HandshakeStage.EVIDENCE_FLUSHED
                else None
            ),
        )
        if stage is terminal_stage:
            return
    raise AssertionError("terminal stage is not worker-owned")


def reap(registry: RunOwnedProcessRegistry, returncode: int) -> None:
    registry.note_exit_observed()
    registry.record_supervisor_reaped(
        returncode,
        recorded_at_utc=RECORDED_AT,
    )


def process_record(
    *,
    pid: int = 432362,
    start_time_ticks: int = 10973359,
    gpu_uuid: str = "GPU-unit-0001",
) -> dict[str, object]:
    return {
        "gpu_uuid": gpu_uuid,
        "pid": pid,
        "process_start_time_ticks": start_time_ticks,
    }


def process_snapshot(
    *,
    allowed: tuple[dict[str, object], ...] = (),
    foreign: tuple[dict[str, object], ...] = (),
    unknown: tuple[dict[str, object], ...] = (),
    errors: tuple[str, ...] = (),
    query_exit_code: int = 0,
) -> dict[str, object]:
    return {
        "allowed_compute_processes": list(allowed),
        "foreign_compute_processes": list(foreign),
        "unknown_processes": list(unknown),
        "errors": list(errors),
        "query_exit_code": query_exit_code,
    }


class ProcessSnapshotInterpreterTests(unittest.TestCase):
    def test_process_snapshot_uses_frozen_project_python(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                b'{"allowed_compute_processes":[],"errors":[],'
                b'"foreign_compute_processes":[],"query_exit_code":0,'
                b'"unknown_processes":[]}\n'
            ),
            stderr=b"",
        )
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.subprocess.run",
            return_value=completed,
        ) as run:
            result = _process_snapshot()

        self.assertEqual(result["query_exit_code"], 0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], str(PYTHON_EXECUTABLE))


class ProcessIdentityTests(unittest.TestCase):
    def test_proc_stat_parser_handles_parentheses_and_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            process_root = root / "77"
            process_root.mkdir()
            tail = ["S", "55", *(["0"] * 17), "123456"]
            (process_root / "stat").write_text(
                f"77 (worker ) name) {' '.join(tail)}\n",
                encoding="utf-8",
            )
            identity = read_process_identity(77, proc_root=root)

        self.assertEqual(identity.pid, 77)
        self.assertEqual(identity.parent_pid, 55)
        self.assertEqual(identity.start_time_ticks, 123456)

    def test_missing_proc_identity_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            with self.assertRaises(ProcessIdentityUnavailable):
                read_process_identity(77, proc_root=Path(raw_root))

    def test_command_fingerprint_binds_environment_and_argv(self) -> None:
        first = expected_fingerprint()
        second = command_fingerprint(
            ("/usr/bin/python3", "-m", "kvbench", "different-worker"),
            working_directory="/home/rockrock/cmu_paper",
            environment_sha256=ENVIRONMENT_SHA256,
        )
        self.assertNotEqual(first, second)


class HandshakeTests(unittest.TestCase):
    def test_fast_normal_exit_before_telemetry_poll_is_owned_completed(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.WORKER_EXITING)
        reap(registry, 0)

        outcome = registry.terminal_outcome()
        self.assertIs(outcome.disposition, OwnershipDisposition.OWNED_COMPLETED)
        self.assertTrue(outcome.full_handshake_observed)
        self.assertEqual(
            outcome.owned_completion_basis,
            "full_ordered_handshake_zero_exit",
        )
        self.assertEqual(registry.to_evidence()["device_snapshot_count"], 0)

    def test_exit_before_first_poll_and_required_handshake_is_owned_failure(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.WORKER_STARTED)
        reap(registry, 1)

        outcome = registry.terminal_outcome()
        self.assertIs(
            outcome.disposition,
            OwnershipDisposition.OWNED_WORKER_FAILURE,
        )
        self.assertTrue(outcome.exclusivity_passed)
        self.assertFalse(outcome.evidence_flushed)

    def test_exit_after_evidence_flush_without_exiting_event_is_completed(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        reap(registry, 0)

        outcome = registry.terminal_outcome()
        self.assertIs(outcome.disposition, OwnershipDisposition.OWNED_COMPLETED)
        self.assertTrue(outcome.evidence_flushed)
        self.assertFalse(outcome.worker_exiting_observed)
        self.assertFalse(outcome.full_handshake_observed)
        self.assertEqual(
            outcome.owned_completion_basis,
            "zero_exit_after_evidence_flushed",
        )
        evidence = registry.to_evidence()
        self.assertEqual(
            evidence["schema_version"],
            "kvbench-phase3-process-registry-3.0.0",
        )
        self.assertFalse(evidence["worker_exiting_required_for_owned_completion"])
        self.assertTrue(
            evidence[
                "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed"
            ]
        )

    def test_abnormal_exit_after_evidence_flush_is_owned_failure(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        reap(registry, -9)

        outcome = registry.terminal_outcome()
        self.assertIs(
            outcome.disposition,
            OwnershipDisposition.OWNED_WORKER_FAILURE,
        )
        self.assertIn("return code -9", outcome.reason)

    def test_out_of_order_and_duplicate_stages_fail_closed(self) -> None:
        registry = make_registry()
        with self.assertRaises(ProcessSupervisionError):
            registry.record_worker_stage(
                HandshakeStage.CUDA_CONTEXT_CREATED,
                recorded_at_utc=RECORDED_AT,
            )
        registry.record_worker_stage(
            HandshakeStage.WORKER_STARTED,
            recorded_at_utc=RECORDED_AT,
        )
        with self.assertRaises(ProcessSupervisionError):
            registry.record_worker_stage(
                HandshakeStage.WORKER_STARTED,
                recorded_at_utc=RECORDED_AT,
            )

    def test_command_fingerprint_and_identity_mismatch_fail_closed(self) -> None:
        registry = make_registry()
        with self.assertRaises(ProcessSupervisionError):
            registry.record_worker_stage(
                HandshakeStage.WORKER_STARTED,
                recorded_at_utc=RECORDED_AT,
                observed_command_fingerprint=OTHER_SHA256,
            )
        wrong_identity = HandshakeEvent(
            sequence=1,
            stage=HandshakeStage.WORKER_STARTED,
            recorded_at_utc=RECORDED_AT,
            run_id="phase3-different-run",
            gpu_uuid="GPU-unit-0001",
            pid=432362,
            process_start_time_ticks=10973359,
            parent_pid=431000,
            command_fingerprint=expected_fingerprint(),
        )
        with self.assertRaises(ProcessSupervisionError):
            registry.ingest_worker_event(wrong_identity)

    def test_malformed_evidence_stage_and_worker_reap_event_are_rejected(self) -> None:
        with self.assertRaises(ProcessSupervisionError):
            HandshakeEvent(
                sequence=5,
                stage=HandshakeStage.EVIDENCE_FLUSHED,
                recorded_at_utc=RECORDED_AT,
                run_id="phase3-unit-run",
                gpu_uuid="GPU-unit-0001",
                pid=432362,
                process_start_time_ticks=10973359,
                parent_pid=431000,
                command_fingerprint=expected_fingerprint(),
            )
        registry = make_registry()
        supervisor_event = HandshakeEvent(
            sequence=7,
            stage=HandshakeStage.SUPERVISOR_REAPED,
            recorded_at_utc=RECORDED_AT,
            run_id="phase3-unit-run",
            gpu_uuid="GPU-unit-0001",
            pid=432362,
            process_start_time_ticks=10973359,
            parent_pid=431000,
            command_fingerprint=expected_fingerprint(),
        )
        with self.assertRaises(ProcessSupervisionError):
            registry.ingest_worker_event(supervisor_event)

    def test_reap_requires_prior_non_reaping_exit_observation(self) -> None:
        registry = make_registry()
        with self.assertRaises(ProcessSupervisionError):
            registry.record_supervisor_reaped(0, recorded_at_utc=RECORDED_AT)


class HandshakePersistenceTests(unittest.TestCase):
    def test_atomic_event_round_trip_refresh_and_no_replace(self) -> None:
        source = make_registry()
        event = source.record_worker_stage(
            HandshakeStage.WORKER_STARTED,
            recorded_at_utc=RECORDED_AT,
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory).resolve()
            path = write_handshake_event(directory, event)

            self.assertEqual(read_handshake_event(path), event)
            destination = make_registry()
            self.assertEqual(destination.refresh_handshake_directory(directory), 1)
            self.assertEqual(
                destination.observed_worker_stages,
                (HandshakeStage.WORKER_STARTED,),
            )
            with self.assertRaises(ProcessSupervisionError):
                write_handshake_event(directory, event)

    def test_visible_hard_link_window_is_complete_and_readable(self) -> None:
        source = make_registry()
        event = source.record_worker_stage(
            HandshakeStage.WORKER_STARTED,
            recorded_at_utc=RECORDED_AT,
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory).resolve()
            with mock.patch.object(
                Path,
                "unlink",
                autospec=True,
                return_value=None,
            ):
                path = write_handshake_event(directory, event)
            self.assertEqual(path.stat().st_nlink, 2)
            self.assertEqual(read_handshake_event(path), event)

    def test_final_name_is_absent_during_short_complete_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory).resolve()
            target = directory / "ready.json"
            payload = b'{"ready":true}\n'
            original_write = process_supervision.os.write
            observations: list[bool] = []

            def short_write(descriptor: int, value: object) -> int:
                observations.append(target.exists())
                return original_write(descriptor, bytes(value)[:1])

            with mock.patch.object(
                process_supervision.os,
                "write",
                side_effect=short_write,
            ):
                publish_bytes_no_replace(target, payload)
            self.assertTrue(observations)
            self.assertFalse(any(observations))
            self.assertEqual(
                read_published_bytes(target, maximum_bytes=1024),
                payload,
            )

    def test_zero_progress_write_never_publishes_final_name(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory).resolve()
            target = directory / "release.token"
            with mock.patch.object(
                process_supervision.os,
                "write",
                return_value=0,
            ):
                with self.assertRaisesRegex(
                    ProcessSupervisionError,
                    "made no progress",
                ):
                    publish_bytes_no_replace(target, b"release\n")
            self.assertFalse(target.exists())

    def test_published_reader_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory).resolve()
            source = directory / "source"
            source.write_bytes(b"payload")
            source.chmod(0o600)
            target = directory / "target"
            target.symlink_to(source)
            with self.assertRaises(ProcessSupervisionError):
                read_published_bytes(target, maximum_bytes=1024)

    def test_coordinator_and_worker_reconstruct_the_same_command(self) -> None:
        point = SimpleNamespace(
            point_id="fixed-l-b1-l64-eager-r1",
            process_replicate=2,
        )
        coordinator = coordinator_worker_argv(
            "configs/phase3-fixed-l.yaml",
            point,
            "phase3-remediation-unit",
        )
        worker = worker_worker_argv(
            "configs/phase3-fixed-l.yaml",
            point.point_id,
            point.process_replicate,
            "phase3-remediation-unit",
        )
        self.assertEqual(coordinator, worker)


class DeviceProcessOwnershipTests(unittest.TestCase):
    def test_pid_disappearance_retains_registered_child_ownership(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        disposition = registry.observe_proc_start_time(None)
        self.assertIs(
            disposition,
            ProcObservationDisposition.DISAPPEARED_RETAINED,
        )
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=None,
                ),
            )
        )
        self.assertIs(verdict.disposition, SnapshotDisposition.OWNED_ONLY)
        reap(registry, 0)
        self.assertIs(
            registry.terminal_outcome().disposition,
            OwnershipDisposition.OWNED_COMPLETED,
        )

    def test_simulated_pid_reuse_is_a_sticky_hard_failure(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        self.assertIs(
            registry.observe_proc_start_time(10973360),
            ProcObservationDisposition.PID_REUSE_DETECTED,
        )
        reap(registry, 0)
        outcome = registry.terminal_outcome()
        self.assertIs(
            outcome.disposition,
            OwnershipDisposition.PID_REUSE_DETECTED,
        )
        self.assertFalse(outcome.exclusivity_passed)

    def test_same_pid_with_new_start_time_in_snapshot_is_pid_reuse(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=10973360,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.PID_REUSE_DETECTED,
        )
        self.assertTrue(verdict.hard_failure)

    def test_unregistered_foreign_process_is_a_hard_failure(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=777777,
                    process_start_time_ticks=900,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.FOREIGN_PROCESS_DETECTED,
        )
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        reap(registry, 0)
        self.assertIs(
            registry.terminal_outcome().disposition,
            OwnershipDisposition.FOREIGN_PROCESS_DETECTED,
        )

    def test_worker_and_foreign_process_coexistence_remains_a_hard_failure(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=10973359,
                ),
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=777777,
                    process_start_time_ticks=900,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.FOREIGN_PROCESS_DETECTED,
        )
        self.assertEqual(len(verdict.owned), 1)
        self.assertEqual(len(verdict.foreign), 1)

    def test_missing_start_time_without_disappearance_basis_fails_closed(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=None,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.UNVERIFIED_REGISTERED_PID,
        )
        self.assertTrue(verdict.hard_failure)

    def test_process_handle_and_pidfd_metadata_are_preserved(self) -> None:
        registry = make_registry(pidfd=9)
        evidence = registry.to_evidence()
        self.assertEqual(
            evidence["schema_version"],
            "kvbench-phase3-process-registry-3.0.0",
        )
        handle = evidence["handle"]
        self.assertIsInstance(handle, dict)
        assert isinstance(handle, dict)
        self.assertTrue(handle["process_handle_retained"])
        self.assertTrue(handle["pidfd_supported"])
        self.assertTrue(handle["pidfd_opened"])
        self.assertEqual(handle["pidfd"], 9)
        self.assertIsInstance(registry.process_handle, FakeProcessHandle)

    def test_process_handle_fallback_is_retained_without_pidfd_support(self) -> None:
        registry = make_registry(pidfd_supported=False, pidfd=None)
        evidence = registry.to_evidence()
        handle = evidence["handle"]
        self.assertIsInstance(handle, dict)
        assert isinstance(handle, dict)
        self.assertTrue(handle["process_handle_retained"])
        self.assertFalse(handle["pidfd_supported"])
        self.assertFalse(handle["pidfd_opened"])
        self.assertIsNone(handle["pidfd"])


class CoordinatorOwnershipJoinTests(unittest.TestCase):
    def test_exact_registered_row_is_owned_despite_raw_foreign_bucket(self) -> None:
        registry = make_registry()
        verdict = _registry_snapshot_verdict(
            process_snapshot(foreign=(process_record(),)),
            registry,
            terminal_resolution_allowed=False,
        )

        self.assertTrue(verdict["passed"])
        self.assertEqual(
            verdict["registry_verdict"]["disposition"],
            SnapshotDisposition.OWNED_ONLY.value,
        )

    def test_live_registered_compute_apps_pmon_race_is_owned(self) -> None:
        registry = make_registry()
        verdict = _registry_snapshot_verdict(
            process_snapshot(
                unknown=(process_record(),),
                errors=(
                    "compute_apps GPU GPU-unit-0001 PID 432362 "
                    "has no pmon process type",
                ),
                query_exit_code=2,
            ),
            registry,
            terminal_resolution_allowed=False,
        )

        self.assertTrue(verdict["passed"])
        self.assertFalse(verdict["query_evidence_hard_failure"])
        self.assertFalse(verdict["terminal_registered_process_resolution"])
        self.assertEqual(
            verdict["registry_verdict"]["disposition"],
            SnapshotDisposition.OWNED_ONLY.value,
        )

    def test_terminal_stale_registered_row_is_owned(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        registry.note_exit_observed()
        verdict = _registry_snapshot_verdict(
            process_snapshot(
                unknown=(process_record(start_time_ticks=0),),
                errors=(
                    "compute_apps GPU GPU-unit-0001 PID 432362 "
                    "has no pmon process type",
                    "cannot read /proc/432362/stat: FileNotFoundError",
                ),
                query_exit_code=2,
            ),
            registry,
            terminal_resolution_allowed=True,
        )

        self.assertTrue(verdict["passed"])
        self.assertTrue(verdict["terminal_registered_process_resolution"])
        self.assertEqual(
            verdict["registry_verdict"]["disposition"],
            SnapshotDisposition.OWNED_ONLY.value,
        )

    def test_unrelated_snapshot_error_remains_a_hard_failure(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        verdict = _registry_snapshot_verdict(
            process_snapshot(
                foreign=(process_record(),),
                errors=("pmon exited with status 1",),
                query_exit_code=2,
            ),
            registry,
            terminal_resolution_allowed=True,
        )

        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["terminal_registered_process_resolution"])
        self.assertTrue(verdict["query_evidence_hard_failure"])
        reap(registry, 0)
        outcome = registry.terminal_outcome()
        self.assertIs(
            outcome.disposition,
            OwnershipDisposition.UNVERIFIED_PROCESS_DETECTED,
        )
        self.assertFalse(outcome.exclusivity_passed)

    def test_pid_reuse_and_foreign_coexistence_remain_hard_failures(self) -> None:
        pmon_race_error = (
            "compute_apps GPU GPU-unit-0001 PID 432362 "
            "has no pmon process type",
        )
        reused_registry = make_registry()
        reused = _registry_snapshot_verdict(
            process_snapshot(
                unknown=(process_record(start_time_ticks=10973360),),
                errors=pmon_race_error,
                query_exit_code=2,
            ),
            reused_registry,
            terminal_resolution_allowed=False,
        )
        self.assertFalse(reused["passed"])
        self.assertTrue(reused["query_evidence_hard_failure"])
        self.assertEqual(
            reused["registry_verdict"]["disposition"],
            SnapshotDisposition.PID_REUSE_DETECTED.value,
        )

        foreign_registry = make_registry()
        coexistence = _registry_snapshot_verdict(
            process_snapshot(
                unknown=(process_record(),),
                foreign=(
                    process_record(pid=777777, start_time_ticks=900),
                ),
                errors=pmon_race_error,
                query_exit_code=2,
            ),
            foreign_registry,
            terminal_resolution_allowed=False,
        )
        self.assertFalse(coexistence["passed"])
        self.assertTrue(coexistence["query_evidence_hard_failure"])
        self.assertEqual(
            coexistence["registry_verdict"]["disposition"],
            SnapshotDisposition.FOREIGN_PROCESS_DETECTED.value,
        )

    def test_nonreaping_waitid_observation_does_not_use_process_poll(self) -> None:
        registry = make_registry(pidfd_supported=False, pidfd=None)
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.os.waitid",
            return_value=object(),
        ) as waitid:
            self.assertTrue(_nonreaping_exit_observed(registry))

        waitid.assert_called_once()
        flags = waitid.call_args.args[2]
        self.assertTrue(flags & os.WEXITED)
        self.assertTrue(flags & os.WNOHANG)
        self.assertTrue(flags & os.WNOWAIT)
        self.assertTrue(registry.exit_observed)
        self.assertFalse(registry.reaped)

    def test_nonreaping_waitid_reap_race_fails_closed(self) -> None:
        registry = make_registry(pidfd_supported=False, pidfd=None)
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.os.waitid",
            side_effect=ChildProcessError,
        ):
            with self.assertRaises(Phase3CoordinatorError):
                _nonreaping_exit_observed(registry)


class CoordinatorRegisteredReapOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._identity_patcher = mock.patch(
            "kvbench.runtime.phase3_coordinator.read_process_identity",
            return_value=ProcessIdentity(
                pid=432362,
                start_time_ticks=10973359,
                parent_pid=431000,
            ),
        )
        self._identity_patcher.start()
        self.addCleanup(self._identity_patcher.stop)

    class OrderedProcess:
        def __init__(
            self,
            order: list[str],
            registry: RunOwnedProcessRegistry,
            returncode: int,
        ) -> None:
            self.pid = registry.identity.process.pid
            self.returncode: int | None = None
            self.wait_calls = 0
            self._order = order
            self._registry = registry
            self._wait_returncode = returncode

        def wait(self, *, timeout: float) -> int:
            self._order.append("wait")
            if not self._registry.exit_observed:
                raise AssertionError("wait occurred before exit observation")
            self.wait_calls += 1
            if self.wait_calls != 1:
                raise AssertionError("registered worker was waited more than once")
            self.returncode = self._wait_returncode
            return self._wait_returncode

    @staticmethod
    def _record_wrapper(
        registry: RunOwnedProcessRegistry,
        order: list[str],
    ) -> object:
        original = registry.record_supervisor_reaped

        def record(returncode: int, *, recorded_at_utc: str) -> object:
            order.append("record")
            return original(returncode, recorded_at_utc=recorded_at_utc)

        return mock.patch.object(
            registry,
            "record_supervisor_reaped",
            side_effect=record,
        )

    def test_rapid_normal_exit_after_flush_is_owned_and_ordered(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        process = self.OrderedProcess(order, registry, 0)

        def signal_request(pid: int, requested_signal: int) -> None:
            order.append(f"signal:{requested_signal}")
            raise ProcessLookupError

        def observe(
            observed_registry: RunOwnedProcessRegistry,
            *,
            timeout_seconds: float,
        ) -> bool:
            order.append("observe")
            observed_registry.note_exit_observed()
            return True

        with (
            mock.patch(
                "kvbench.runtime.phase3_coordinator.os.killpg",
                side_effect=signal_request,
            ),
            mock.patch(
                "kvbench.runtime.phase3_coordinator."
                "_wait_for_registered_exit_observation",
                side_effect=observe,
            ),
            self._record_wrapper(registry, order),
        ):
            returncode, _ = _terminate_registered_worker(process, registry)

        self.assertEqual(returncode, 0)
        self.assertEqual(order, ["signal:15", "observe", "wait", "record"])
        self.assertEqual(process.wait_calls, 1)
        self.assertIs(
            registry.terminal_outcome().disposition,
            OwnershipDisposition.OWNED_COMPLETED,
        )

    def test_abnormal_exit_is_owned_failure_and_ordered(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        process = self.OrderedProcess(order, registry, 17)

        def observe(
            observed_registry: RunOwnedProcessRegistry,
            *,
            timeout_seconds: float,
        ) -> bool:
            order.append("observe")
            observed_registry.note_exit_observed()
            return True

        with (
            mock.patch(
                "kvbench.runtime.phase3_coordinator.os.killpg",
                side_effect=lambda pid, sig: order.append(f"signal:{sig}"),
            ),
            mock.patch(
                "kvbench.runtime.phase3_coordinator."
                "_wait_for_registered_exit_observation",
                side_effect=observe,
            ),
            self._record_wrapper(registry, order),
        ):
            returncode, _ = _terminate_registered_worker(process, registry)

        self.assertEqual(returncode, 17)
        self.assertEqual(order, ["signal:15", "observe", "wait", "record"])
        self.assertIs(
            registry.terminal_outcome().disposition,
            OwnershipDisposition.OWNED_WORKER_FAILURE,
        )

    def test_term_timeout_escalates_to_kill_before_single_reap(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        process = self.OrderedProcess(order, registry, -9)
        observations = iter((False, True))

        def observe(
            observed_registry: RunOwnedProcessRegistry,
            *,
            timeout_seconds: float,
        ) -> bool:
            order.append("observe")
            result = next(observations)
            if result:
                observed_registry.note_exit_observed()
            return result

        with (
            mock.patch(
                "kvbench.runtime.phase3_coordinator.os.killpg",
                side_effect=lambda pid, sig: order.append(f"signal:{sig}"),
            ),
            mock.patch(
                "kvbench.runtime.phase3_coordinator."
                "_wait_for_registered_exit_observation",
                side_effect=observe,
            ),
            self._record_wrapper(registry, order),
        ):
            returncode, _ = _terminate_registered_worker(process, registry)

        self.assertEqual(returncode, -9)
        self.assertEqual(
            order,
            ["signal:15", "observe", "signal:9", "observe", "wait", "record"],
        )
        self.assertEqual(process.wait_calls, 1)

    def test_kill_observation_timeout_fails_before_wait_or_record(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        process = self.OrderedProcess(order, registry, -9)
        with (
            mock.patch(
                "kvbench.runtime.phase3_coordinator.os.killpg",
                side_effect=lambda pid, sig: order.append(f"signal:{sig}"),
            ),
            mock.patch(
                "kvbench.runtime.phase3_coordinator."
                "_wait_for_registered_exit_observation",
                side_effect=lambda registry, timeout_seconds: (
                    order.append("observe") or False
                ),
            ),
            self._record_wrapper(registry, order),
        ):
            with self.assertRaisesRegex(
                Phase3CoordinatorError,
                "not observed after SIGKILL",
            ):
                _terminate_registered_worker(process, registry)

        self.assertEqual(
            order,
            ["signal:15", "observe", "signal:9", "observe"],
        )
        self.assertEqual(process.wait_calls, 0)
        self.assertFalse(registry.reaped)

    def test_direct_reap_before_observation_is_rejected_without_wait(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        process = self.OrderedProcess(order, registry, 0)
        with self.assertRaisesRegex(
            Phase3CoordinatorError,
            "cannot be waited before",
        ):
            _reap_registered_worker(
                process,
                registry,
                timeout_seconds=10.0,
            )
        self.assertEqual(process.wait_calls, 0)

    def test_direct_normal_reap_waits_once_then_records_once(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        record_through(registry, HandshakeStage.WORKER_EXITING)
        registry.note_exit_observed()
        process = self.OrderedProcess(order, registry, 0)

        with self._record_wrapper(registry, order):
            returncode, event = _reap_registered_worker(
                process,
                registry,
                timeout_seconds=10.0,
            )

        self.assertEqual(returncode, 0)
        self.assertEqual(event.stage, HandshakeStage.SUPERVISOR_REAPED)
        self.assertEqual(order, ["wait", "record"])
        self.assertEqual(process.wait_calls, 1)
        self.assertTrue(registry.reaped)
        self.assertIs(
            registry.terminal_outcome().disposition,
            OwnershipDisposition.OWNED_COMPLETED,
        )

    def test_pid_reuse_before_term_fails_without_signaling(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        process = self.OrderedProcess(order, registry, 0)
        reused = ProcessIdentity(
            pid=registry.identity.process.pid,
            start_time_ticks=registry.identity.process.start_time_ticks + 1,
            parent_pid=registry.identity.process.parent_pid,
        )
        with (
            mock.patch(
                "kvbench.runtime.phase3_coordinator.read_process_identity",
                return_value=reused,
            ),
            mock.patch(
                "kvbench.runtime.phase3_coordinator.os.killpg"
            ) as killpg,
        ):
            with self.assertRaisesRegex(
                Phase3CoordinatorError,
                "identity changed before SIGTERM",
            ):
                _terminate_registered_worker(process, registry)

        killpg.assert_not_called()
        self.assertEqual(process.wait_calls, 0)
        self.assertFalse(registry.exit_observed)
        self.assertFalse(registry.reaped)

    def test_pid_reuse_between_term_and_kill_blocks_kill(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=False, pidfd=None)
        process = self.OrderedProcess(order, registry, -9)
        expected = registry.identity.process
        reused = ProcessIdentity(
            pid=expected.pid,
            start_time_ticks=expected.start_time_ticks + 1,
            parent_pid=expected.parent_pid,
        )
        identity_reads = mock.Mock(side_effect=(expected, reused))
        with (
            mock.patch(
                "kvbench.runtime.phase3_coordinator.read_process_identity",
                identity_reads,
            ),
            mock.patch(
                "kvbench.runtime.phase3_coordinator.os.killpg",
                side_effect=lambda pid, sig: order.append(f"signal:{sig}"),
            ) as killpg,
            mock.patch(
                "kvbench.runtime.phase3_coordinator."
                "_wait_for_registered_exit_observation",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(
                Phase3CoordinatorError,
                "identity changed before SIGKILL",
            ):
                _terminate_registered_worker(process, registry)

        self.assertEqual(order, ["signal:15"])
        self.assertEqual(killpg.call_count, 1)
        self.assertEqual(identity_reads.call_count, 2)
        self.assertEqual(process.wait_calls, 0)
        self.assertFalse(registry.reaped)

    def test_pidfd_is_preferred_over_reusable_pid_signaling(self) -> None:
        order: list[str] = []
        registry = make_registry(pidfd_supported=True, pidfd=19)
        process = self.OrderedProcess(order, registry, -15)

        def observe(
            observed_registry: RunOwnedProcessRegistry,
            *,
            timeout_seconds: float,
        ) -> bool:
            observed_registry.note_exit_observed()
            return True

        with (
            mock.patch(
                "kvbench.runtime.phase3_coordinator.signal.pidfd_send_signal"
            ) as pidfd_send_signal,
            mock.patch(
                "kvbench.runtime.phase3_coordinator.os.killpg"
            ) as killpg,
            mock.patch(
                "kvbench.runtime.phase3_coordinator.read_process_identity"
            ) as identity_read,
            mock.patch(
                "kvbench.runtime.phase3_coordinator."
                "_wait_for_registered_exit_observation",
                side_effect=observe,
            ),
            self._record_wrapper(registry, order),
        ):
            returncode, _ = _terminate_registered_worker(process, registry)

        self.assertEqual(returncode, -15)
        pidfd_send_signal.assert_called_once_with(19, 15)
        identity_read.assert_not_called()
        killpg.assert_not_called()
        self.assertEqual(order, ["wait", "record"])


class CoordinatorUnregisteredCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = ProcessIdentity(
            pid=517171,
            start_time_ticks=701,
            parent_pid=431000,
        )
        self._identity_patcher = mock.patch(
            "kvbench.runtime.phase3_coordinator.read_process_identity",
            return_value=self.identity,
        )
        self._identity_patcher.start()
        self.addCleanup(self._identity_patcher.stop)

    @staticmethod
    def _process(
        *,
        returncode: int | None = None,
        wait: mock.Mock | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            pid=517171,
            returncode=returncode,
            wait=wait if wait is not None else mock.Mock(),
        )

    def test_already_exited_unregistered_worker_needs_no_cleanup(self) -> None:
        process = self._process(returncode=7)
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.os.killpg"
        ) as killpg:
            self.assertEqual(_terminate_unregistered_worker(process), 7)

        killpg.assert_not_called()
        process.wait.assert_not_called()

    def test_term_reaps_unregistered_worker(self) -> None:
        process = self._process(wait=mock.Mock(return_value=-15))
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.os.killpg"
        ) as killpg:
            self.assertEqual(
                _terminate_unregistered_worker(
                    process,
                    expected_identity=self.identity,
                ),
                -15,
            )

        self.assertEqual([call.args[1] for call in killpg.call_args_list], [15])
        process.wait.assert_called_once_with(timeout=10)

    def test_term_timeout_escalates_and_reaps_unregistered_worker(self) -> None:
        timeout = subprocess.TimeoutExpired(cmd="phase3-worker", timeout=10)
        process = self._process(wait=mock.Mock(side_effect=(timeout, -9)))
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.os.killpg"
        ) as killpg:
            self.assertEqual(
                _terminate_unregistered_worker(
                    process,
                    expected_identity=self.identity,
                ),
                -9,
            )

        self.assertEqual(
            [call.args[1] for call in killpg.call_args_list],
            [15, 9],
        )
        self.assertEqual(process.wait.call_count, 2)
        process.wait.assert_has_calls(
            [mock.call(timeout=10), mock.call(timeout=10)]
        )

    def test_kill_wait_timeout_is_not_hidden(self) -> None:
        timeouts = (
            subprocess.TimeoutExpired(cmd="phase3-worker", timeout=10),
            subprocess.TimeoutExpired(cmd="phase3-worker", timeout=10),
        )
        process = self._process(wait=mock.Mock(side_effect=timeouts))
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.os.killpg"
        ) as killpg:
            with self.assertRaises(subprocess.TimeoutExpired):
                _terminate_unregistered_worker(
                    process,
                    expected_identity=self.identity,
                )

        self.assertEqual(
            [call.args[1] for call in killpg.call_args_list],
            [15, 9],
        )
        self.assertEqual(process.wait.call_count, 2)

    def test_live_unregistered_worker_without_stable_identity_is_not_signaled(
        self,
    ) -> None:
        process = self._process(wait=mock.Mock(return_value=-15))
        with mock.patch(
            "kvbench.runtime.phase3_coordinator.os.killpg"
        ) as killpg:
            with self.assertRaisesRegex(
                Phase3CoordinatorError,
                "lacks a stable identity before SIGTERM",
            ):
                _terminate_unregistered_worker(process)

        killpg.assert_not_called()
        process.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
