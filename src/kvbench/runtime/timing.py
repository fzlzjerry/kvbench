"""CUDA-event and host-wall timing with Phase 3 batch boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import time
from typing import Any


_TORCH: Any | None = None


class TimingFailure(RuntimeError):
    """A measured batch did not complete all declared operations."""

    def __init__(self, message: str, *, submitted_operations: int) -> None:
        self.submitted_operations = submitted_operations
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RawTimingSample:
    """One preserved synchronized batch total and exact divisor."""

    batch_index: int
    host_total_ns: int
    cuda_total_ms: float
    completed_operations: int
    failed_operations: int

    @property
    def host_ns_per_operation(self) -> float:
        return self.host_total_ns / self.completed_operations

    @property
    def cuda_ms_per_operation(self) -> float:
        return self.cuda_total_ms / self.completed_operations

    def to_dict(self) -> dict[str, int | float]:
        return {
            "batch_index": self.batch_index,
            "host_total_ns": self.host_total_ns,
            "cuda_total_ms": self.cuda_total_ms,
            "completed_operations": self.completed_operations,
            "failed_operations": self.failed_operations,
            "host_ns_per_operation": self.host_ns_per_operation,
            "cuda_ms_per_operation": self.cuda_ms_per_operation,
        }


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Raw timing samples plus the final device output reference."""

    samples: tuple[RawTimingSample, ...]
    last_output: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "sample_count": len(self.samples),
            "paper_claim_eligible": False,
            "measurement_scope": "native_host_admission",
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise TimingFailure(
                "PyTorch is unavailable",
                submitted_operations=0,
            ) from error
    return _TORCH


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def warmup_operations(
    operation: Callable[[], Any],
    *,
    count: int,
    device: Any | None = None,
) -> Any:
    """Submit all untimed warmups and synchronize only once at completion."""

    torch = _torch()
    warmups = _positive_int(count, "count")
    output: Any = None
    for _ in range(warmups):
        output = operation()
    torch.cuda.synchronize(device=device)
    return output


def measure_fixed_batches(
    operation: Callable[[], Any],
    *,
    operations_per_batch: int,
    batches: int,
    device: Any | None = None,
) -> TimingResult:
    """Measure fixed-shape operations without per-operation synchronization."""

    torch = _torch()
    count = _positive_int(operations_per_batch, "operations_per_batch")
    batch_count = _positive_int(batches, "batches")
    event_pairs = tuple(
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(batch_count)
    )
    samples: list[RawTimingSample] = []
    last_output: Any = None
    for batch_index, (start_event, end_event) in enumerate(event_pairs):
        torch.cuda.synchronize(device=device)
        submitted = 0
        host_start = time.perf_counter_ns()
        start_event.record()
        try:
            for _ in range(count):
                last_output = operation()
                submitted += 1
            end_event.record()
            torch.cuda.synchronize(device=device)
        except BaseException as error:
            raise TimingFailure(
                "measured fixed batch did not complete",
                submitted_operations=batch_index * count + submitted,
            ) from error
        host_end = time.perf_counter_ns()
        samples.append(
            RawTimingSample(
                batch_index=batch_index,
                host_total_ns=host_end - host_start,
                cuda_total_ms=float(start_event.elapsed_time(end_event)),
                completed_operations=count,
                failed_operations=0,
            )
        )
    return TimingResult(samples=tuple(samples), last_output=last_output)


def measure_growing_trajectory(
    operation: Callable[[int], Any],
    *,
    output_steps: int,
    device: Any | None = None,
) -> TimingResult:
    """Measure one complete growing trajectory with one terminal synchronization."""

    torch = _torch()
    steps = _positive_int(output_steps, "output_steps")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device=device)
    submitted = 0
    last_output: Any = None
    host_start = time.perf_counter_ns()
    start_event.record()
    try:
        for step in range(steps):
            last_output = operation(step)
            submitted += 1
        end_event.record()
        torch.cuda.synchronize(device=device)
    except BaseException as error:
        raise TimingFailure(
            "measured growing trajectory did not complete",
            submitted_operations=submitted,
        ) from error
    host_end = time.perf_counter_ns()
    sample = RawTimingSample(
        batch_index=0,
        host_total_ns=host_end - host_start,
        cuda_total_ms=float(start_event.elapsed_time(end_event)),
        completed_operations=steps,
        failed_operations=0,
    )
    return TimingResult(samples=(sample,), last_output=last_output)
