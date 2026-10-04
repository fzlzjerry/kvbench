"""Fixed-historical-context Phase 3 timing over an admitted endpoint session."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

from kvbench.runtime.allocation import (
    NormalTimingMemoryEvidence,
    capture_cuda_memory_snapshot,
)
from kvbench.runtime.backend import forced_flash_execution
from kvbench.runtime.numerical import (
    NumericalComparison,
    tensor_sha256_untimed,
)
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
from kvbench.runtime.timing import TimingResult, measure_fixed_batches
from kvbench.schema import RunnerKind


_TORCH: Any | None = None


class FixedLRunnerError(RuntimeError):
    """The fixed-L runner could not preserve its frozen semantics."""


@dataclass(frozen=True, slots=True)
class FixedLRunResult:
    """Machine-readable fixed-L runtime evidence with no scientific claim."""

    context_length: int
    total_attended_length: int
    batch_size: int
    graph_mode: str
    output_checksum: str
    output_finite: bool
    historical_checksum_before: str
    historical_checksum_after: str
    historical_cache_unchanged: bool
    cache_pointers_stable: bool
    cache_accounting: dict[str, int]
    cache_layout_fingerprint: str
    adapter_config_fingerprint: str
    cache_byte_breakdown: dict[str, int]
    timing: TimingResult
    memory_evidence: NormalTimingMemoryEvidence
    gqa_cache_geometry: dict[str, Any]
    graph: dict[str, Any] | None
    eager_graph_comparison: NumericalComparison | None
    telemetry_before: dict[str, Any]
    telemetry_after: dict[str, Any]
    telemetry_sampling_interval_seconds: float | None
    measurement_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": "fixed_l",
            "context_convention": "historical_prefix_length",
            "context_length": self.context_length,
            "total_attended_length": self.total_attended_length,
            "batch_size": self.batch_size,
            "graph_mode": self.graph_mode,
            "output_checksum": self.output_checksum,
            "output_finite": self.output_finite,
            "historical_checksum_before": self.historical_checksum_before,
            "historical_checksum_after": self.historical_checksum_after,
            "historical_cache_unchanged": self.historical_cache_unchanged,
            "cache_pointers_stable": self.cache_pointers_stable,
            "cache_accounting": self.cache_accounting,
            "cache_layout_fingerprint": self.cache_layout_fingerprint,
            "adapter_config_fingerprint": self.adapter_config_fingerprint,
            "cache_byte_breakdown": self.cache_byte_breakdown,
            "timing": self.timing.to_dict(),
            "timing_skipped_reason": None,
            "allocation": None,
            "memory_evidence": self.memory_evidence.to_dict(),
            "gqa_cache_geometry": self.gqa_cache_geometry,
            "graph": self.graph,
            "eager_graph_comparison": (
                None
                if self.eager_graph_comparison is None
                else self.eager_graph_comparison.to_dict()
            ),
            "audit_evidence_source": "checksum_bound_raw_audit_index",
            "audit_operation_count": 1,
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
            raise FixedLRunnerError("PyTorch is unavailable") from error
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


def run_fixed_l(
    session: MeasurementEndpointSession,
    *,
    measured_steps: int = 32,
    measured_batches: int = 5,
) -> FixedLRunResult:
    """Time only the exact callable retained by one admitted fixed-L session."""

    try:
        session = require_endpoint_session(session)
        measurement_scope = session_measurement_scope(session).value
    except EndpointSessionError as error:
        raise FixedLRunnerError(
            "fixed-L runner requires an endpoint session"
        ) from error
    operation_key = session.operation_keys[0]
    if (
        len(session.operation_keys) != 1
        or operation_key.runner_kind is not RunnerKind.FIXED_L
        or session.state != "ready"
    ):
        raise FixedLRunnerError("fixed-L session is not admitted")
    torch = _torch()
    device = session.cache_device
    pointers_before = session.current_cache_pointers()
    history_before = session.historical_prefix_sha256
    setup_memory = capture_cuda_memory_snapshot(
        "post_setup",
        device=device,
    )

    with torch.inference_mode(), forced_flash_execution():
        operation = session.fixed_measurement_callable()
        telemetry_before, telemetry_before_snapshot = _telemetry_or_error()
        torch.cuda.synchronize(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
        timing_memory_before = capture_cuda_memory_snapshot(
            "normal_timing_before",
            device=device,
        )
        timing = measure_fixed_batches(
            operation,
            operations_per_batch=measured_steps,
            batches=measured_batches,
            device=device,
        )
        timing_memory_after = capture_cuda_memory_snapshot(
            "normal_timing_after",
            device=device,
        )
        telemetry_after, telemetry_after_snapshot = _telemetry_or_error()
        output_checksum = tensor_sha256_untimed(timing.last_output)
        output_finite = bool(torch.isfinite(timing.last_output).all().item())

    audit_checksum, audit_finite = session.audit_output(0)
    if output_checksum != audit_checksum or not audit_finite:
        raise FixedLRunnerError(
            "fixed-L measured output differs from its admitted audit output"
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
    return FixedLRunResult(
        context_length=operation_key.historical_context,
        total_attended_length=operation_key.attended_context,
        batch_size=operation_key.batch_size,
        graph_mode=operation_key.graph_mode.value,
        output_checksum=output_checksum,
        output_finite=output_finite,
        historical_checksum_before=history_before,
        historical_checksum_after=history_after,
        historical_cache_unchanged=history_before == history_after,
        cache_pointers_stable=pointers_before == pointers_after,
        cache_accounting=accounting,
        cache_layout_fingerprint=session.cache_layout_fingerprint(),
        adapter_config_fingerprint=session.adapter_config_fingerprint,
        cache_byte_breakdown=session.method_byte_breakdown(),
        timing=timing,
        memory_evidence=memory_evidence,
        gqa_cache_geometry=session.gqa_cache_geometry(),
        graph=(
            None
            if session.graph_evidence is None
            else dict(session.graph_evidence)
        ),
        eager_graph_comparison=session.eager_graph_comparison,
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
        telemetry_sampling_interval_seconds=telemetry_interval,
        measurement_scope=measurement_scope,
    )
