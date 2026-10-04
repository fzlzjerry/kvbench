"""Focused tests for Phase 8 generic direct-child supervision."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest import mock

from kvbench.runtime.process_supervision import (
    ProcessIdentity,
    ProcessSupervisionError,
    environment_fingerprint,
    run_supervised_command,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class GenericSupervisedCommandTests(unittest.TestCase):
    @staticmethod
    def environment() -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "KVBENCH_TEST_SECRET": "must-not-appear-in-evidence",
        }

    def test_direct_child_output_and_evidence_are_bound_without_environment(self) -> None:
        result = run_supervised_command(
            (
                sys.executable,
                "-c",
                (
                    "import sys,time;"
                    "sys.stdout.buffer.write(b'fixture-ok\\n');"
                    "sys.stderr.buffer.write(b'fixture-note\\n');"
                    "time.sleep(0.05)"
                ),
            ),
            working_directory=str(REPOSITORY_ROOT),
            environment=self.environment(),
            timeout_seconds=5.0,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"fixture-ok\n")
        self.assertEqual(result.stderr, b"fixture-note\n")
        self.assertFalse(result.timed_out)
        self.assertTrue(result.direct_child_verified)
        self.assertEqual(result.process_identity.parent_pid, os.getpid())
        self.assertEqual(result.final_reap_count, 1)
        evidence = result.to_dict()
        self.assertEqual(evidence["command"]["shell"], False)
        self.assertEqual(
            evidence["command"]["working_directory"],
            str(REPOSITORY_ROOT),
        )
        self.assertEqual(evidence["final_reap"], {"completed": True, "count": 1})
        self.assertNotIn("handshake", evidence)
        self.assertNotIn("exclusivity", evidence)
        self.assertNotIn(
            self.environment()["KVBENCH_TEST_SECRET"],
            repr(evidence),
        )

    def test_environment_fingerprint_is_deterministic_and_value_sensitive(self) -> None:
        first = environment_fingerprint({"B": "two", "A": "one"})
        reordered = environment_fingerprint({"A": "one", "B": "two"})
        changed = environment_fingerprint({"A": "one", "B": "three"})

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)
        self.assertNotIn("one", first)

    def test_spawn_uses_explicit_argv_cwd_environment_and_no_shell(self) -> None:
        class FakeProcess:
            pid = 712345
            returncode: int | None = None

            def communicate(self, *, timeout: float | None = None) -> tuple[bytes, bytes]:
                self.returncode = 0
                return b"out", b"err"

            def kill(self) -> None:
                raise AssertionError("normal child must not be killed")

            def send_signal(self, requested_signal: int) -> None:
                raise AssertionError(
                    f"normal child must not receive {requested_signal}"
                )

        process = FakeProcess()
        environment = self.environment()
        identity = ProcessIdentity(
            pid=process.pid,
            start_time_ticks=998877,
            parent_pid=os.getpid(),
        )
        with (
            mock.patch(
                "kvbench.runtime.process_supervision.subprocess.Popen",
                return_value=process,
            ) as popen,
            mock.patch(
                "kvbench.runtime.process_supervision.read_process_identity",
                return_value=identity,
            ),
            mock.patch(
                "kvbench.runtime.process_supervision._open_supervised_pidfd",
                return_value=(False, None),
            ),
        ):
            result = run_supervised_command(
                ("/usr/bin/example", "--fixture"),
                working_directory=str(REPOSITORY_ROOT),
                environment=environment,
                timeout_seconds=3.0,
            )

        popen.assert_called_once_with(
            ("/usr/bin/example", "--fixture"),
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        self.assertEqual(result.process_identity, identity)
        self.assertEqual(result.final_reap_count, 1)

    def test_timeout_terminates_then_kills_only_the_verified_identity(self) -> None:
        class TimedProcess:
            pid = 723456
            returncode: int | None = None

            def __init__(self) -> None:
                self.communicate_calls = 0
                self.signals: list[int] = []

            def communicate(self, *, timeout: float | None = None) -> tuple[bytes, bytes]:
                self.communicate_calls += 1
                if self.communicate_calls < 3:
                    raise subprocess.TimeoutExpired(("worker",), timeout)
                self.returncode = -signal.SIGKILL
                return b"partial-out", b"partial-err"

            def kill(self) -> None:
                raise AssertionError("registered cleanup path must not be used")

            def send_signal(self, requested_signal: int) -> None:
                self.signals.append(requested_signal)

        process = TimedProcess()
        identity = ProcessIdentity(
            pid=process.pid,
            start_time_ticks=887766,
            parent_pid=os.getpid(),
        )
        with (
            mock.patch(
                "kvbench.runtime.process_supervision.subprocess.Popen",
                return_value=process,
            ),
            mock.patch(
                "kvbench.runtime.process_supervision.read_process_identity",
                return_value=identity,
            ) as identity_reader,
            mock.patch(
                "kvbench.runtime.process_supervision._open_supervised_pidfd",
                return_value=(False, None),
            ),
        ):
            result = run_supervised_command(
                ("/usr/bin/worker",),
                working_directory=str(REPOSITORY_ROOT),
                environment=self.environment(),
                timeout_seconds=0.25,
                termination_grace_seconds=0.25,
            )

        self.assertTrue(result.timed_out)
        self.assertTrue(result.terminate_requested)
        self.assertTrue(result.kill_requested)
        self.assertEqual(process.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(identity_reader.call_count, 3)
        self.assertEqual(result.returncode, -signal.SIGKILL)
        self.assertEqual(result.final_reap_count, 1)

    def test_timeout_refuses_to_signal_a_changed_process_identity(self) -> None:
        class TimedProcess:
            pid = 734567
            returncode: int | None = None

            def __init__(self) -> None:
                self.signals: list[int] = []

            def communicate(self, *, timeout: float | None = None) -> tuple[bytes, bytes]:
                raise subprocess.TimeoutExpired(("worker",), timeout)

            def kill(self) -> None:
                raise AssertionError("registered cleanup path must not be used")

            def send_signal(self, requested_signal: int) -> None:
                self.signals.append(requested_signal)

        process = TimedProcess()
        original = ProcessIdentity(
            pid=process.pid,
            start_time_ticks=776655,
            parent_pid=os.getpid(),
        )
        changed = ProcessIdentity(
            pid=process.pid,
            start_time_ticks=776656,
            parent_pid=os.getpid(),
        )
        with (
            mock.patch(
                "kvbench.runtime.process_supervision.subprocess.Popen",
                return_value=process,
            ),
            mock.patch(
                "kvbench.runtime.process_supervision.read_process_identity",
                side_effect=(original, changed),
            ),
            mock.patch(
                "kvbench.runtime.process_supervision._open_supervised_pidfd",
                return_value=(False, None),
            ),
        ):
            with self.assertRaisesRegex(
                ProcessSupervisionError,
                "identity changed",
            ):
                run_supervised_command(
                    ("/usr/bin/worker",),
                    working_directory=str(REPOSITORY_ROOT),
                    environment=self.environment(),
                    timeout_seconds=0.25,
                )

        self.assertEqual(process.signals, [])

    def test_rejects_implicit_command_and_invalid_execution_bounds(self) -> None:
        with self.assertRaises(ProcessSupervisionError):
            run_supervised_command(
                "echo unsafe",
                working_directory=str(REPOSITORY_ROOT),
                environment=self.environment(),
                timeout_seconds=1.0,
            )
        with self.assertRaises(ProcessSupervisionError):
            run_supervised_command(
                ("/usr/bin/echo",),
                working_directory="relative",
                environment=self.environment(),
                timeout_seconds=1.0,
            )
        with self.assertRaises(ProcessSupervisionError):
            run_supervised_command(
                ("/usr/bin/echo",),
                working_directory=str(REPOSITORY_ROOT),
                environment=self.environment(),
                timeout_seconds=float("inf"),
            )
        with self.assertRaises(ProcessSupervisionError):
            environment_fingerprint({"INVALID=NAME": "value"})


if __name__ == "__main__":
    unittest.main()
