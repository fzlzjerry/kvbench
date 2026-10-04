"""Focused strict Phase 8 byte and bounded-grid schema tests."""

from __future__ import annotations

import dataclasses
import unittest

from kvbench.schema import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
)
from kvbench.schema.phase8 import (
    PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    PHASE8_BASE_TREE,
    PHASE8_DECISION_0018_PATCH_SHA256,
    PHASE8_EXTENSION_SHA256,
    PHASE8_FIXTURE_ROOT_DIGEST,
    PHASE8_OFFICIAL_COMMIT,
    PHASE8_PATCHED_TREE,
    Phase8ByteAccounting,
    Phase8ByteBreakdown,
    Phase8RunManifest,
)


def _breakdown() -> Phase8ByteBreakdown:
    return Phase8ByteBreakdown(
        quantized_historical_k_payload=10,
        quantized_historical_v_payload=10,
        k_scales=2,
        k_zeros=2,
        v_scales=2,
        v_zeros=2,
        other_metadata=0,
        residual_k=10,
        residual_v=10,
        fp16_staging=10,
        quantization_staging=10,
        padding_alignment=0,
        persistent_workspace=10,
        value_rollover_shift_scratch=10,
        block_group_rounding=2,
    )


def _accounting() -> Phase8ByteAccounting:
    breakdown = _breakdown()
    return Phase8ByteAccounting(
        capacity=129,
        active_context=128,
        allocated_bytes=breakdown.total,
        predicted_allocated_bytes=breakdown.total,
        active_storage_bytes=64,
        logical_bf16_allocated_bytes=516,
        logical_bf16_active_bytes=512,
        rho_alloc=breakdown.total / 516,
        r_alloc=516 / breakdown.total,
        predicted_relative_error=0.0,
        temporary_peak_bytes=0,
        breakdown=breakdown,
        r_hbm=None,
    )


def _manifest() -> Phase8RunManifest:
    return Phase8RunManifest(
        schema_version=Phase8RunManifest.SCHEMA_VERSION,
        artifact_schema_version=Phase8RunManifest.ARTIFACT_SCHEMA_VERSION,
        run_id="phase8-k4v4-fixed-128-eager-001",
        status=RunStatus.COMPLETED,
        git_sha="5" * 40,
        git_dirty=False,
        created_at_utc="2026-07-27T00:00:00Z",
        started_at_utc="2026-07-27T00:00:01Z",
        finished_at_utc="2026-07-27T00:01:00Z",
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=GraphMode.EAGER,
        method_configuration="k4v4",
        k_bits=4,
        v_bits=4,
        method_config_fingerprint="1" * 64,
        method_fingerprint="2" * 64,
        adapter_version="phase8_kivi_1",
        adapter_source_sha256="3" * 64,
        official_base_commit=PHASE8_OFFICIAL_COMMIT,
        official_base_tree=PHASE8_BASE_TREE,
        patched_tree=PHASE8_PATCHED_TREE,
        decision_0018_patch_sha256=PHASE8_DECISION_0018_PATCH_SHA256,
        extension_sha256=PHASE8_EXTENSION_SHA256,
        fixture_root_digest=PHASE8_FIXTURE_ROOT_DIGEST,
        group_size=32,
        residual_length=32,
        dtype_boundary="bf16_to_fp16_official_kivi_to_bf16",
        cache_layout_fingerprint="4" * 64,
        authorized_container_digest=PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        batch_size=1,
        context_length=128,
        output_steps=1,
        capacity=129,
        accounting=_accounting(),
        quality_status=QualityValidationState.UNVALIDATED,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        performance_claim_eligible=False,
        measurement_scope=MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
        quality_execution=QualityExecutionState.LOCKED,
        quality_benchmark_executed=False,
        performance_data_frozen=False,
        speedup_calculated=False,
        full_scan_state="CLOSED",
        inventory_path="artifact_inventory.json",
        failure_reason=None,
    )


class Phase8KIVISchemaTests(unittest.TestCase):
    def test_canonical_accounting_and_no_hbm(self) -> None:
        accounting = _accounting()
        self.assertAlmostEqual(
            accounting.r_alloc * accounting.rho_alloc, 1.0, places=12
        )
        self.assertIsNone(accounting.r_hbm)
        self.assertEqual(accounting.breakdown.total, accounting.allocated_bytes)

    def test_bounded_grid_manifest(self) -> None:
        manifest = _manifest()
        self.assertFalse(manifest.speedup_calculated)
        self.assertFalse(manifest.performance_claim_eligible)

    def test_rejects_non_grid_and_inverted_ratio(self) -> None:
        with self.assertRaises(ValueError):
            dataclasses.replace(_manifest(), context_length=256)
        accounting = _accounting()
        with self.assertRaises(ValueError):
            dataclasses.replace(
                accounting,
                rho_alloc=accounting.r_alloc,
                r_alloc=accounting.rho_alloc,
            )

    def test_rejects_r_hbm_field(self) -> None:
        payload = _accounting().to_dict()
        payload["r_hbm"] = 1.0
        with self.assertRaises(Exception):
            Phase8ByteAccounting.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
