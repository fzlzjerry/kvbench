"""Phase 2 admission reporting without experiment execution."""

from __future__ import annotations

import dataclasses
from typing import Any

from kvbench.config import ExperimentBundle


@dataclasses.dataclass(frozen=True, slots=True)
class AdmissionReport:
    schema_version: str
    admitted: bool
    blockers: tuple[str, ...]
    full_scan_state: str
    quality_execution: str
    performance_data_frozen: bool
    execution_attempted: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_admission(bundle: ExperimentBundle) -> AdmissionReport:
    """Report explicit blockers; Phase 2 never opens an admission gate."""

    blockers = tuple(sorted(set(bundle.blockers)))
    return AdmissionReport(
        schema_version="kvbench-admission-report-1.0.0",
        admitted=not blockers,
        blockers=blockers,
        full_scan_state=bundle.plan.admission.full_scan_state,
        quality_execution=bundle.plan.quality.quality_execution.value,
        performance_data_frozen=bundle.plan.quality.performance_data_frozen,
        execution_attempted=False,
    )
