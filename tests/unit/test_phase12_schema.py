"""Focused fail-closed tests for the Phase 12 unified-admission schema."""

from __future__ import annotations

import dataclasses
import math
import statistics
import unittest

from kvbench.schema.base import QualityExecutionState
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase12 import (
    PHASE12_AUTHORIZED_CONTAINER_DIGEST,
    PHASE12_CONFIG_FINGERPRINTS,
    PHASE12_HELD_OUT_CONFIGURATIONS,
    PHASE12_MAIN_CONFIGURATIONS,
    PHASE12_MODEL_ID,
    PHASE12_MODEL_REVISION,
    PHASE12_RANDOMIZATION_SEEDS,
    PHASE12_RANDOMIZED_ORDERS,
    derive_phase12_randomized_order,
    Phase12ByteAccounting,
    Phase12ConfigurationAdmission,
    Phase12EvidenceReference,
    Phase12ExcludedConfiguration,
    Phase12G5Disposition,
    Phase12G5Run,
    Phase12G5Statistics,
    Phase12GlobalGates,
    Phase12PriorGateEvidence,
    Phase12PublicationState,
    Phase12RandomizedOrder,
    Phase12UnifiedAdmissionReport,
)


def _evidence(config_id: str, gate: str) -> Phase12EvidenceReference:
    return Phase12EvidenceReference(
        evidence_id=f"{config_id}_{gate.lower()}",
        path=f"evidence/{config_id}/{gate.lower()}.json",
        sha256=f"{sum(map(ord, config_id + gate)):064x}",
    )


def _accounting() -> Phase12ByteAccounting:
    predicted = 100
    allocated = 100
    logical = 200
    return Phase12ByteAccounting(
        data_payload_bytes=50,
        metadata_bytes=10,
        sparse_bytes=10,
        sink_residual_bytes=10,
        padding_bytes=10,
        workspace_bytes=10,
        predicted_allocated_bytes=predicted,
        allocated_bytes=allocated,
        logical_bf16_bytes=logical,
        rho_alloc=allocated / logical,
        r_alloc=logical / allocated,
        predicted_relative_error=0.0,
        r_hbm=None,
    )


def _configuration(
    config_id: str,
    *,
    statuses: dict[str, GateDisposition] | None = None,
) -> Phase12ConfigurationAdmission:
    observed = statuses or {}
    gates = tuple(
        Phase12PriorGateEvidence(
            gate=gate,  # type: ignore[arg-type]
            status=observed.get(gate, GateDisposition.PASS),
            criteria_satisfied=(
                observed.get(gate, GateDisposition.PASS)
                is GateDisposition.PASS
            ),
            evidence=(_evidence(config_id, gate),),
        )
        for gate in ("G1", "G2", "G3", "G4")
    )
    return Phase12ConfigurationAdmission(
        method_config_id=config_id,
        method_config_fingerprint=PHASE12_CONFIG_FINGERPRINTS[config_id],
        prior_gates=gates,
        byte_accounting=_accounting(),
        no_fallback=True,
        speedup_calculated=False,
    )


def _orders() -> tuple[Phase12RandomizedOrder, ...]:
    return tuple(
        Phase12RandomizedOrder(
            replicate_index=index,
            seed=seed,
            configurations=PHASE12_RANDOMIZED_ORDERS[index],
        )
        for index, seed in enumerate(PHASE12_RANDOMIZATION_SEEDS)
    )


def _runs(
    *,
    medians: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[Phase12G5Run, ...]:
    by_config = medians or {}
    runs: list[Phase12G5Run] = []
    for replicate_index, (seed, order) in enumerate(
        zip(
            PHASE12_RANDOMIZATION_SEEDS,
            PHASE12_RANDOMIZED_ORDERS,
            strict=True,
        )
    ):
        for order_index, config_id in enumerate(order):
            process_median = by_config.get(config_id, (1.0, 1.0, 1.0))[
                replicate_index
            ]
            ordinal = replicate_index * len(PHASE12_MAIN_CONFIGURATIONS) + order_index
            runs.append(
                Phase12G5Run(
                    run_id=f"phase12-r{replicate_index}-{order_index:02d}-{config_id}",
                    method_config_id=config_id,
                    replicate_index=replicate_index,
                    seed=seed,
                    order_index=order_index,
                    manifest_path=(
                        f"runs/phase12-r{replicate_index}-{order_index:02d}-"
                        f"{config_id}/manifest.json"
                    ),
                    manifest_sha256=f"{ordinal + 1:064x}",
                    process_median_ms=process_median,
                    output_checksum=f"{100 + PHASE12_MAIN_CONFIGURATIONS.index(config_id):064x}",
                    kernel_path_fingerprint=f"{200 + PHASE12_MAIN_CONFIGURATIONS.index(config_id):064x}",
                    allocation_fingerprint=f"{300 + PHASE12_MAIN_CONFIGURATIONS.index(config_id):064x}",
                    temperature_min_c=40.0,
                    temperature_max_c=50.0,
                    sm_clock_min_mhz=1000,
                    sm_clock_max_mhz=1200,
                    memory_clock_min_mhz=1500,
                    memory_clock_max_mhz=1800,
                    power_min_w=100.0,
                    power_max_w=200.0,
                    finite_output=True,
                    no_backend_fallback=True,
                    allocation_stable=True,
                    kernel_path_stable=True,
                    gpu_exclusive=True,
                    speedup_calculated=False,
                )
            )
    return tuple(runs)


def _summary(
    config_id: str,
    matching_runs: tuple[Phase12G5Run, ...],
    *,
    disposition: Phase12G5Disposition | None = None,
) -> Phase12G5Statistics:
    medians = tuple(item.process_median_ms for item in matching_runs)
    cv = statistics.stdev(medians) / statistics.mean(medians)
    output_agreement = len({item.output_checksum for item in matching_runs}) == 1
    kernel_agreement = (
        len({item.kernel_path_fingerprint for item in matching_runs}) == 1
    )
    allocation_agreement = (
        len({item.allocation_fingerprint for item in matching_runs}) == 1
    )
    inferred = (
        Phase12G5Disposition.PASS
        if cv <= 0.03
        and output_agreement
        and kernel_agreement
        and allocation_agreement
        else Phase12G5Disposition.UNSTABLE
    )
    return Phase12G5Statistics(
        method_config_id=config_id,
        run_ids=tuple(item.run_id for item in matching_runs),
        process_medians_ms=medians,
        median_ms=float(statistics.median(medians)),
        minimum_ms=min(medians),
        maximum_ms=max(medians),
        mean_ms=statistics.mean(medians),
        standard_deviation_ms=statistics.stdev(medians),
        coefficient_of_variation=cv,
        temperature_min_c=min(item.temperature_min_c for item in matching_runs),
        temperature_max_c=max(item.temperature_max_c for item in matching_runs),
        sm_clock_min_mhz=min(item.sm_clock_min_mhz for item in matching_runs),
        sm_clock_max_mhz=max(item.sm_clock_max_mhz for item in matching_runs),
        memory_clock_min_mhz=min(
            item.memory_clock_min_mhz for item in matching_runs
        ),
        memory_clock_max_mhz=max(
            item.memory_clock_max_mhz for item in matching_runs
        ),
        power_min_w=min(item.power_min_w for item in matching_runs),
        power_max_w=max(item.power_max_w for item in matching_runs),
        output_checksum_agreement=output_agreement,
        kernel_path_agreement=kernel_agreement,
        allocation_agreement=allocation_agreement,
        disposition=disposition or inferred,
    )


def _statistics(runs: tuple[Phase12G5Run, ...]) -> tuple[Phase12G5Statistics, ...]:
    return tuple(
        _summary(
            config_id,
            tuple(item for item in runs if item.method_config_id == config_id),
        )
        for config_id in PHASE12_MAIN_CONFIGURATIONS
    )


def _gates(
    *,
    g1: GateDisposition = GateDisposition.PASS,
    g5: GateDisposition = GateDisposition.PASS,
) -> Phase12GlobalGates:
    all_pass = g1 is GateDisposition.PASS and g5 is GateDisposition.PASS
    return Phase12GlobalGates(
        g0=GateDisposition.PASS,
        g1=g1,
        g2=GateDisposition.PASS,
        g3=GateDisposition.PASS,
        g4=GateDisposition.PASS,
        g5=g5,
        pilot_state="READY" if all_pass else "NOT_READY",
        full_scan_state="CLOSED",
        quality_execution=QualityExecutionState.LOCKED,
        performance_data_frozen=False,
    )


def _report(
    *,
    configurations: tuple[Phase12ConfigurationAdmission, ...] | None = None,
    runs: tuple[Phase12G5Run, ...] | None = None,
    statistics_records: tuple[Phase12G5Statistics, ...] | None = None,
    gates: Phase12GlobalGates | None = None,
    publication_state: Phase12PublicationState = Phase12PublicationState.PASS,
) -> Phase12UnifiedAdmissionReport:
    observed_runs = runs or _runs()
    published = publication_state is Phase12PublicationState.PASS
    return Phase12UnifiedAdmissionReport(
        schema_version=Phase12UnifiedAdmissionReport.SCHEMA_VERSION,
        created_at_utc="2026-07-30T00:00:00Z",
        campaign_id="phase12-unified-001",
        execution_git_sha="2bc6aaa1d05b08d50f4c01bbc0b2863dd8689fe1",
        authorized_container_digest=PHASE12_AUTHORIZED_CONTAINER_DIGEST,
        model_id=PHASE12_MODEL_ID,
        model_revision=PHASE12_MODEL_REVISION,
        tokenizer_id=PHASE12_MODEL_ID,
        tokenizer_revision=PHASE12_MODEL_REVISION,
        runner_kind="fixed_l",
        graph_mode="cuda_graph",
        batch_size=1,
        context_length=4096,
        warmup_steps=64,
        measured_steps=128,
        measured_batches=5,
        independent_process_replicates=3,
        cv_threshold=0.03,
        configurations=configurations
        or tuple(_configuration(item) for item in PHASE12_MAIN_CONFIGURATIONS),
        excluded_configurations=tuple(
            Phase12ExcludedConfiguration(
                method_config_id=item,
                reason="validation_only_control",
            )
            for item in PHASE12_HELD_OUT_CONFIGURATIONS
        ),
        randomized_orders=_orders(),
        runs=observed_runs,
        g5_statistics=statistics_records or _statistics(observed_runs),
        publication_state=publication_state,
        publication_receipt=(
            Phase12EvidenceReference(
                evidence_id="phase12_r2_publication",
                path="docs/evidence/phase12/r2-publication.json",
                sha256="9" * 64,
            )
            if published
            else None
        ),
        published_root_sha256="8" * 64 if published else None,
        r2_uri=(
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{'8' * 64}/"
            if published
            else None
        ),
        object_count=64 if published else None,
        complete_last=published,
        clean_retrieval=published,
        gates=gates or _gates(),
        speedup_calculated=False,
    )


class Phase12SchemaTests(unittest.TestCase):
    def test_randomized_orders_derive_from_each_frozen_seed(self) -> None:
        self.assertEqual(
            tuple(
                derive_phase12_randomized_order(seed)
                for seed in PHASE12_RANDOMIZATION_SEEDS
            ),
            PHASE12_RANDOMIZED_ORDERS,
        )
        with self.assertRaisesRegex(ValueError, "seed is not frozen"):
            derive_phase12_randomized_order(0)

    def test_exact_contract_passes_and_round_trips(self) -> None:
        report = _report()
        parsed = Phase12UnifiedAdmissionReport.from_dict(report.to_dict())
        self.assertEqual(parsed, report)
        self.assertEqual(
            tuple(item.method_config_id for item in report.configurations),
            PHASE12_MAIN_CONFIGURATIONS,
        )
        self.assertEqual(len(report.runs), 30)
        self.assertEqual(report.gates.pilot_state, "READY")

    def test_publication_is_required_before_global_g5_and_pilot(self) -> None:
        report = _report(
            publication_state=Phase12PublicationState.PENDING,
            gates=Phase12GlobalGates(
                g0=GateDisposition.PASS,
                g1=GateDisposition.PASS,
                g2=GateDisposition.PASS,
                g3=GateDisposition.PASS,
                g4=GateDisposition.PASS,
                g5=GateDisposition.NOT_EVALUATED,
                pilot_state="NOT_READY",
                full_scan_state="CLOSED",
                quality_execution=QualityExecutionState.LOCKED,
                performance_data_frozen=False,
            ),
        )
        self.assertIs(report.publication_state, Phase12PublicationState.PENDING)
        self.assertIs(report.gates.g5, GateDisposition.NOT_EVALUATED)
        with self.assertRaises(ValueError):
            dataclasses.replace(report, gates=_gates())
        with self.assertRaises(ValueError):
            dataclasses.replace(
                report,
                publication_state=Phase12PublicationState.PASS,
            )
        with self.assertRaises(ValueError):
            _report(
                gates=report.gates,
                publication_state=Phase12PublicationState.PASS,
            )

    def test_rejects_main_set_fingerprint_and_held_out_drift(self) -> None:
        report = _report()
        with self.assertRaises(ValueError):
            dataclasses.replace(
                report,
                configurations=report.configurations[:-1],
            )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                report.configurations[0],
                method_config_fingerprint="f" * 64,
            )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                report,
                excluded_configurations=report.excluded_configurations[::-1],
            )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                report.configurations[0],
                method_config_id="k8v4",
            )

    def test_rejects_seed_order_missing_and_duplicate_replicate(self) -> None:
        report = _report()
        with self.assertRaises(ValueError):
            dataclasses.replace(
                report.randomized_orders[0],
                seed=20260731,
            )
        with self.assertRaises(ValueError):
            dataclasses.replace(report, runs=report.runs[:-1])
        duplicated = (*report.runs[:-1], report.runs[0])
        with self.assertRaises(ValueError):
            dataclasses.replace(report, runs=duplicated)

    def test_cv_threshold_boundary_and_unstable_state(self) -> None:
        boundary_runs = _runs(medians={"bf16": (97.0, 100.0, 103.0)})
        boundary_stats = _statistics(boundary_runs)
        bf16 = boundary_stats[0]
        self.assertTrue(math.isclose(bf16.coefficient_of_variation, 0.03))
        self.assertIs(bf16.disposition, Phase12G5Disposition.PASS)
        _report(runs=boundary_runs, statistics_records=boundary_stats)

        unstable_runs = _runs(medians={"bf16": (90.0, 100.0, 110.0)})
        unstable_stats = _statistics(unstable_runs)
        self.assertIs(
            unstable_stats[0].disposition,
            Phase12G5Disposition.UNSTABLE,
        )
        _report(
            runs=unstable_runs,
            statistics_records=unstable_stats,
            gates=_gates(g5=GateDisposition.FAIL),
        )
        with self.assertRaises(ValueError):
            _report(
                runs=unstable_runs,
                statistics_records=unstable_stats,
                gates=_gates(),
            )

    def test_rejects_checksum_path_allocation_and_kernel_drift_for_pass(self) -> None:
        report = _report()
        with self.assertRaises(ValueError):
            dataclasses.replace(report.runs[0], manifest_path="../manifest.json")
        with self.assertRaises(ValueError):
            dataclasses.replace(report.runs[0], allocation_stable=False)
        with self.assertRaises(ValueError):
            dataclasses.replace(report.runs[0], kernel_path_stable=False)

        changed_run = dataclasses.replace(
            report.runs[0],
            output_checksum="f" * 64,
        )
        changed_runs = (changed_run, *report.runs[1:])
        with self.assertRaises(ValueError):
            dataclasses.replace(report, runs=changed_runs)

    def test_rejects_fallback_exclusivity_nan_and_speedup(self) -> None:
        run = _report().runs[0]
        for changes in (
            {"no_backend_fallback": False},
            {"gpu_exclusive": False},
            {"finite_output": False},
            {"process_median_ms": float("nan")},
            {"speedup_calculated": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(run, **changes)
        with self.assertRaises(ValueError):
            dataclasses.replace(_report(), speedup_calculated=True)

    def test_rejects_bad_accounting_r_hbm_and_ratio(self) -> None:
        accounting = _accounting()
        with self.assertRaises(ValueError):
            dataclasses.replace(accounting, predicted_allocated_bytes=101)
        with self.assertRaises(ValueError):
            dataclasses.replace(accounting, rho_alloc=1.0)
        payload = accounting.to_dict()
        payload["r_hbm"] = 1.0
        with self.assertRaises(Exception):
            Phase12ByteAccounting.from_dict(payload)

    def test_no_majority_voting_and_pilot_requires_all_gates(self) -> None:
        configurations = list(
            _configuration(item) for item in PHASE12_MAIN_CONFIGURATIONS
        )
        configurations[0] = _configuration(
            PHASE12_MAIN_CONFIGURATIONS[0],
            statuses={"G1": GateDisposition.FAIL},
        )
        with self.assertRaises(ValueError):
            _report(configurations=tuple(configurations), gates=_gates())
        _report(
            configurations=tuple(configurations),
            gates=_gates(g1=GateDisposition.FAIL),
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                _gates(g5=GateDisposition.FAIL),
                pilot_state="READY",
            )

    def test_full_scan_and_quality_remain_locked(self) -> None:
        gates = _gates()
        payload = gates.to_dict()
        payload["full_scan_state"] = "OPEN"
        with self.assertRaises(Exception):
            Phase12GlobalGates.from_dict(payload)
        with self.assertRaises(ValueError):
            dataclasses.replace(
                gates,
                quality_execution=QualityExecutionState.LOCKED,
                performance_data_frozen=True,
            )


if __name__ == "__main__":
    unittest.main()
