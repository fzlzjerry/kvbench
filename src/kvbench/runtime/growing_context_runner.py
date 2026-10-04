"""Growing-context Phase 3 timing over an admitted endpoint session."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

from kvbench.runtime.allocation import (
    NormalTimingMemoryEvidence,
    capture_cuda_memory_snapshot,
)
from kvbench.runtime.backend import forced_flash_execution
from kvbench.runtime.numerical import tensor_sha256_untimed
from kvbench.runtime.turboquant_session import (
    EndpointSessionError,
    MeasurementEndpointSession,
    require_endpoint_session,
    session_measurement_scope,
)
from kvbench.runtime.telemetry import (
    TelemetryError,
    TelemetrySnapshot,
    collect_telemetry,
    telemetry_sampling_interval_seconds,
)
from kvbench.runtime.timing import TimingResult, measure_growing_trajectory
from kvbench.schema import RunnerKind


_TORCH: Any | None = None


class GrowingContextRunnerError(RuntimeError):
    """The growing-context runner could not preserve its frozen semantics."""


@dataclass(frozen=True, slots=True)
class GrowingStepEvidence:
    """Untimed output witness from the admitted ordered audit."""

    step: int
    historical_active_length: int
    output_checksum: str
    output_finite: bool

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "step": self.step,
            "historical_active_length": self.historical_active_length,
            "output_checksum": self.output_checksum,
            "output_finite": self.output_finite,
        }


@dataclass(frozen=True, slots=True)
class GrowingContextRunResult:
    """Machine-readable eager trajectory evidence with raw step sequence."""

    starting_context: int
    output_steps: int
    batch_size: int
    output_checksum: str
    output_finite: bool
    active_lengths: tuple[int, ...]
    step_evidence: tuple[GrowingStepEvidence, ...]
    historical_checksum_before: str
    historical_checksum_after: str
    historical_cache_unchanged: bool
    cache_accounting: dict[str, int]
    cache_layout_fingerprint: str
    adapter_config_fingerprint: str
    cache_byte_breakdown: dict[str, int]
    cache_pointers_stable: bool
    timing: TimingResult
    memory_evidence: NormalTimingMemoryEvidence
    gqa_cache_geometry: dict[str, Any]
    telemetry_before: dict[str, Any]
    telemetry_after: dict[str, Any]
    telemetry_sampling_interval_seconds: float | None
    measurement_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": "growing_context",
            "context_convention": "historical_active_length_before_append",
            "starting_context": self.starting_context,
            "output_steps": self.output_steps,
            "batch_size": self.batch_size,
            "output_checksum": self.output_checksum,
            "output_finite": self.output_finite,
            "active_lengths": list(self.active_lengths),
            "step_evidence": [item.to_dict() for item in self.step_evidence],
            "historical_checksum_before": self.historical_checksum_before,
            "historical_checksum_after": self.historical_checksum_after,
            "historical_cache_unchanged": self.historical_cache_unchanged,
            "cache_accounting": self.cache_accounting,
            "cache_layout_fingerprint": self.cache_layout_fingerprint,
            "adapter_config_fingerprint": self.adapter_config_fingerprint,
            "cache_byte_breakdown": self.cache_byte_breakdown,
            "cache_pointers_stable": self.cache_pointers_stable,
            "timing": self.timing.to_dict(),
            "timing_skipped_reason": None,
            "step_checksum_source": "untimed_allocation_audit_before_timing",
            "allocation": None,
            "memory_evidence": self.memory_evidence.to_dict(),
            "gqa_cache_geometry": self.gqa_cache_geometry,
            "audit_evidence_source": "checksum_bound_raw_audit_index",
            "audit_operation_count": self.output_steps,
            "session_state": "measured",
            "telemetry_before": self.telemetry_before,
            "telemetry_after": self.telemetry_after,
            "telemetry_sampling_interval_seconds": (
                self.telemetry_sampling_interval_seconds
            ),
            "telemetry_stability_inference": False,
            "quality_status": "unvalidated",
            "claim_eligibility": "performance_only",
            "performance_claim_eligible": False,
            "measurement_scope": self.measurement_scope,
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise GrowingContextRunnerError("PyTorch is unavailable") from error
    return _TORCH


def _telemetry_or_error() -> tuple[dict[str, Any], TelemetrySnapshot | None]:
    try:
        snapshot = collect_telemetry()
        return snapshot.to_dict(), snapshot
    except TelemetryError as error:
        return (
            {
                "error": type(error).__name__,
                "available": False,
                "raw_snapshot": False,
                "stability_inference": False,
            },
            None,
        )


def run_growing_context(
    session: MeasurementEndpointSession,
    *,
    expected_steps: int = 16,
) -> GrowingContextRunResult:
    """Time one ordered trajectory from one admitted growing session."""

    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps <= 0
    ):
        raise GrowingContextRunnerError(
            "expected_steps must be a positive integer"
        )
    try:
        session = require_endpoint_session(session)
        measurement_scope = session_measurement_scope(session).value
    except EndpointSessionError as error:
        raise GrowingContextRunnerError(
            "growing runner requires an endpoint session"
        ) from error
    first = session.operation_keys[0]
    output_steps = len(session.operation_keys)
    if (
        first.runner_kind is not RunnerKind.GROWING_CONTEXT
        or output_steps != expected_steps
        or session.state != "ready"
    ):
        raise GrowingContextRunnerError("growing session is not admitted")
    torch = _torch()
    device = session.cache_device
    pointers_before = session.current_cache_pointers()
    history_before = session.historical_prefix_sha256
    setup_memory = capture_cuda_memory_snapshot(
        "post_setup",
        device=device,
    )

    with torch.inference_mode(), forced_flash_execution():
        operations = session.growing_measurement_callables()

        def measured_step(step: int) -> Any:
            return operations[step]()

        telemetry_before, telemetry_before_snapshot = _telemetry_or_error()
        torch.cuda.synchronize(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
        timing_memory_before = capture_cuda_memory_snapshot(
            "normal_timing_before",
            device=device,
        )
        timing = measure_growing_trajectory(
            measured_step,
            output_steps=output_steps,
            device=device,
        )
        timing_memory_after = capture_cuda_memory_snapshot(
            "normal_timing_after",
            device=device,
        )
        telemetry_after, telemetry_after_snapshot = _telemetry_or_error()
        output_checksum = tensor_sha256_untimed(timing.last_output)
        measured_output_finite = bool(
            torch.isfinite(timing.last_output).all().item()
        )

    step_evidence = tuple(
        GrowingStepEvidence(
            step=step,
            historical_active_length=(
                session.operation_keys[step].historical_context
            ),
            output_checksum=session.audit_output(step)[0],
            output_finite=session.audit_output(step)[1],
        )
        for step in range(output_steps)
    )
    if output_checksum != step_evidence[-1].output_checksum:
        raise GrowingContextRunnerError(
            "growing measured output differs from its admitted audit output"
        )
    if session.active_context != first.historical_context + output_steps:
        raise GrowingContextRunnerError(
            "growing measured trajectory ended at the wrong active length"
        )
    history_after = session.current_historical_prefix_sha256()
    pointers_after = session.current_cache_pointers()
    session.mark_measured()
    telemetry_interval = (
        None
        if telemetry_before_snapshot is None
        or telemetry_after_snapshot is None
        else telemetry_sampling_interval_seconds(
            telemetry_before_snapshot,
            telemetry_after_snapshot,
        )
    )
    accounting = session.method_cache_accounting()
    accounting["model_baseline_allocated_bytes"] = (
        session.model_memory.allocated_bytes
    )
    memory_evidence = NormalTimingMemoryEvidence(
        model_baseline=session.model_memory,
        post_cache_allocation=session.cache_memory,
        post_setup=setup_memory,
        timing_before=timing_memory_before,
        timing_after=timing_memory_after,
        timing_executed=True,
    )
    return GrowingContextRunResult(
        starting_context=first.historical_context,
        output_steps=output_steps,
        batch_size=first.batch_size,
        output_checksum=output_checksum,
        output_finite=(
            measured_output_finite
            and all(item.output_finite for item in step_evidence)
        ),
        active_lengths=tuple(
            item.historical_active_length for item in step_evidence
        ),
        step_evidence=step_evidence,
        historical_checksum_before=history_before,
        historical_checksum_after=history_after,
        historical_cache_unchanged=history_before == history_after,
        cache_accounting=accounting,
        cache_layout_fingerprint=session.cache_layout_fingerprint(),
        adapter_config_fingerprint=session.adapter_config_fingerprint,
        cache_byte_breakdown=session.method_byte_breakdown(),
        cache_pointers_stable=pointers_before == pointers_after,
        timing=timing,
        memory_evidence=memory_evidence,
        gqa_cache_geometry=session.gqa_cache_geometry(),
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
        telemetry_sampling_interval_seconds=telemetry_interval,
        measurement_scope=measurement_scope,
    )
