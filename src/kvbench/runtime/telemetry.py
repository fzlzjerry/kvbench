"""Non-invasive nvidia-smi telemetry sampled outside decode boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import subprocess
import time
from typing import Any


NVIDIA_SMI = "/usr/bin/nvidia-smi"
_QUERY_FIELDS = (
    "timestamp",
    "name",
    "uuid",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "memory.used",
)


class TelemetryError(RuntimeError):
    """Telemetry could not be collected without ambiguity."""


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """One raw outside-boundary GPU telemetry snapshot."""

    timestamp: str
    collected_at_utc: str
    host_query_started_ns: int
    host_query_finished_ns: int
    host_monotonic_ns: int
    gpu_name: str
    gpu_uuid: str
    power_watts: float
    temperature_celsius: float
    sm_clock_mhz: float
    memory_clock_mhz: float
    vram_used_mib: float
    ecc_mode: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "collected_at_utc": self.collected_at_utc,
            "host_query_started_ns": self.host_query_started_ns,
            "host_query_finished_ns": self.host_query_finished_ns,
            "host_monotonic_ns": self.host_monotonic_ns,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "power_watts": self.power_watts,
            "temperature_celsius": self.temperature_celsius,
            "sm_clock_mhz": self.sm_clock_mhz,
            "memory_clock_mhz": self.memory_clock_mhz,
            "vram_used_mib": self.vram_used_mib,
            "ecc_mode": self.ecc_mode,
            "raw_snapshot": True,
            "stability_inference": False,
        }


def _run_query(fields: tuple[str, ...]) -> list[str]:
    result = subprocess.run(
        [
            NVIDIA_SMI,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    if result.returncode != 0:
        raise TelemetryError("nvidia-smi telemetry query failed")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise TelemetryError("telemetry requires exactly one visible GPU")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != len(fields):
        raise TelemetryError("nvidia-smi telemetry field count differs")
    return values


def collect_telemetry() -> TelemetrySnapshot:
    """Collect one snapshot; callers must invoke it outside measured work."""

    host_started = time.monotonic_ns()
    values = _run_query(_QUERY_FIELDS)
    ecc: str | None = None
    try:
        ecc = _run_query(("ecc.mode.current",))[0]
    except TelemetryError:
        ecc = None
    host_finished = time.monotonic_ns()
    try:
        numeric = [float(item) for item in values[3:]]
    except ValueError as error:
        raise TelemetryError("nvidia-smi telemetry contains nonnumeric data") from error
    return TelemetrySnapshot(
        timestamp=values[0],
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        host_query_started_ns=host_started,
        host_query_finished_ns=host_finished,
        host_monotonic_ns=(host_started + host_finished) // 2,
        gpu_name=values[1],
        gpu_uuid=values[2],
        power_watts=numeric[0],
        temperature_celsius=numeric[1],
        sm_clock_mhz=numeric[2],
        memory_clock_mhz=numeric[3],
        vram_used_mib=numeric[4],
        ecc_mode=ecc,
    )


def telemetry_sampling_interval_seconds(
    before: TelemetrySnapshot,
    after: TelemetrySnapshot,
) -> float:
    """Return the raw host-midpoint interval without a stability inference."""

    delta_ns = after.host_monotonic_ns - before.host_monotonic_ns
    if delta_ns < 0:
        raise TelemetryError("telemetry host timestamps are not monotonic")
    return delta_ns / 1_000_000_000.0
