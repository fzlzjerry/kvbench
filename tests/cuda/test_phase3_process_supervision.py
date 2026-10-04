"""Real CUDA foreign-process control for Phase 3 process ownership."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import unittest

import torch

from kvbench.runtime.phase3_coordinator import (
    _process_snapshot,
    _registry_snapshot_verdict,
)
from kvbench.runtime.process_supervision import (
    RunOwnedProcessRegistry,
    SnapshotDisposition,
    read_process_identity,
)


NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
RESULT_SCHEMA_VERSION = "kvbench-phase3-real-foreign-cuda-control-1.0.0"
RESULT_PREFIX = "PHASE3_PROCESS_SUPERVISION_RESULT="
CHILD_READY_TIMEOUT_SECONDS = 30.0
SNAPSHOT_DISCOVERY_TIMEOUT_SECONDS = 30.0
CHILD_CODE = """
import os
import time
import torch

torch.cuda.init()
held_tensor = torch.empty((1,), dtype=torch.uint8, device="cuda:0")
torch.cuda.synchronize(device="cuda:0")
print(f"READY {os.getpid()}", flush=True)
time.sleep(120)
"""


class Phase3ForeignCudaProcessTests(unittest.TestCase):
    @staticmethod
    def _unsupported_reason() -> str | None:
        if not torch.cuda.is_available():
            return (
                "explicit unsupported CUDA runtime: "
                "torch.cuda.is_available() is false"
            )
        if not NVIDIA_SMI.is_file():
            return (
                "explicit unsupported CUDA runtime: "
                "/usr/bin/nvidia-smi absent"
            )
        if torch.cuda.device_count() < 1:
            return "explicit unsupported CUDA runtime: no visible CUDA device"
        return None

    @staticmethod
    def _emit_result(result: dict[str, object]) -> None:
        print(
            RESULT_PREFIX
            + json.dumps(result, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    @staticmethod
    def _terminate_child(child: subprocess.Popen[str]) -> None:
        if child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5.0)

    def test_actual_cuda_context_is_unregistered_foreign_by_pid_and_start(
        self,
    ) -> None:
        unsupported_reason = self._unsupported_reason()
        if unsupported_reason is not None:
            self._emit_result(
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "status": "skipped_unsupported",
                    "admission_eligible": False,
                    "reason": unsupported_reason,
                    "child_pid": None,
                    "child_start_time_ticks": None,
                    "foreign_process_detected": False,
                    "cleanup_completed": True,
                }
            )
            self.skipTest(unsupported_reason)

        result: dict[str, object] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "executed_failed",
            "admission_eligible": False,
            "reason": "control did not complete",
            "child_pid": None,
            "child_start_time_ticks": None,
            "foreign_process_detected": False,
            "cleanup_completed": False,
        }
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        child: subprocess.Popen[str] | None = None
        try:
            child = subprocess.Popen(
                (sys.executable, "-c", CHILD_CODE),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                start_new_session=True,
            )
            result["child_pid"] = child.pid
            if child.stdout is None or child.stderr is None:
                self.fail("CUDA child pipes were not created")
            readable, _, _ = select.select(
                (child.stdout,),
                (),
                (),
                CHILD_READY_TIMEOUT_SECONDS,
            )
            if not readable:
                self.fail("CUDA child did not report readiness before timeout")
            ready_line = child.stdout.readline().strip()
            if ready_line != f"READY {child.pid}":
                stderr = child.stderr.read() if child.poll() is not None else ""
                self.fail(
                    "CUDA child failed before readiness: "
                    f"stdout={ready_line!r}, stderr={stderr!r}"
                )
            child_identity = read_process_identity(child.pid)
            result["child_start_time_ticks"] = child_identity.start_time_ticks
            deadline = time.monotonic() + SNAPSHOT_DISCOVERY_TIMEOUT_SECONDS
            matching_record: dict[str, object] | None = None
            last_snapshot: dict[str, object] | None = None
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    stderr = child.stderr.read()
                    self.fail(
                        "CUDA child exited before device-process discovery: "
                        f"returncode={child.returncode}, stderr={stderr!r}"
                    )
                last_snapshot = _process_snapshot()
                if (
                    last_snapshot.get("query_exit_code") != 0
                    or last_snapshot.get("errors") != []
                ):
                    self.fail(
                        "actual device-process snapshot failed closed: "
                        f"{last_snapshot!r}"
                    )
                foreign = last_snapshot.get("foreign_compute_processes")
                if not isinstance(foreign, list):
                    self.fail("device-process snapshot foreign bucket is malformed")
                candidates = [
                    item
                    for item in foreign
                    if isinstance(item, dict) and item.get("pid") == child.pid
                ]
                if candidates:
                    if len(candidates) != 1:
                        self.fail("CUDA child has duplicate foreign records")
                    matching_record = candidates[0]
                    break
                time.sleep(0.5)
            if matching_record is None:
                self.fail(
                    "actual device-process snapshot never classified CUDA child "
                    f"as foreign: last_snapshot={last_snapshot!r}"
                )
            self.assertEqual(
                matching_record.get("process_start_time_ticks"),
                child_identity.start_time_ticks,
            )
            self.assertEqual(matching_record.get("pid"), child_identity.pid)

            supervisor_identity = read_process_identity(os.getpid())
            registry = RunOwnedProcessRegistry.register_spawn(
                process_identity=supervisor_identity,
                expected_supervisor_pid=supervisor_identity.parent_pid,
                process_handle=object(),
                pidfd_supported=False,
                pidfd=None,
                run_id="phase3-real-foreign-cuda-control",
                gpu_uuid=str(matching_record["gpu_uuid"]),
                spawned_at_utc="2026-07-23T00:00:00.000000Z",
                expected_command_fingerprint="1" * 64,
            )
            verdict = _registry_snapshot_verdict(
                last_snapshot,
                registry,
                terminal_resolution_allowed=False,
            )
            self.assertFalse(verdict["passed"])
            registry_verdict = verdict["registry_verdict"]
            self.assertEqual(
                registry_verdict["disposition"],
                SnapshotDisposition.FOREIGN_PROCESS_DETECTED.value,
            )
            self.assertTrue(
                any(
                    item["pid"] == child.pid
                    and item["process_start_time_ticks"]
                    == child_identity.start_time_ticks
                    for item in registry_verdict["foreign"]
                )
            )
            result.update(
                {
                    "status": "executed_passed",
                    "admission_eligible": True,
                    "reason": None,
                    "foreign_process_detected": True,
                }
            )
        except BaseException as error:
            result.update(
                {
                    "status": "executed_failed",
                    "admission_eligible": False,
                    "reason": (
                        f"{type(error).__name__}: "
                        f"{' '.join(str(error).split())}"
                    )[:1000],
                }
            )
            raise
        finally:
            try:
                if child is not None:
                    self._terminate_child(child)
                    if child.stdout is not None:
                        child.stdout.close()
                    if child.stderr is not None:
                        child.stderr.close()
                result["cleanup_completed"] = True
            except BaseException as cleanup_error:
                result.update(
                    {
                        "status": "executed_failed",
                        "admission_eligible": False,
                        "reason": (
                            f"cleanup {type(cleanup_error).__name__}: "
                            f"{' '.join(str(cleanup_error).split())}"
                        )[:1000],
                        "cleanup_completed": False,
                    }
                )
                raise
            finally:
                self._emit_result(result)
