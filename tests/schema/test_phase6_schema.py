"""Strict Phase 6 manifest and G2-TQ report schema tests."""

from __future__ import annotations

from copy import deepcopy
import unittest

from kvbench.schema import (
    AUTHORIZED_CONTAINER_DIGEST,
    FIXTURE_ROOT_LEDGER_SHA256,
    FIXTURE_SET_SHA256,
    MANDATORY_CONFIG_SLOT_SIZES,
    PHASE6_METHOD_ADMISSION_CHECK_IDS,
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_TREE,
    MethodAdmissionReportV2,
    Phase6RunManifest,
)
from kvbench.schema.base import SchemaValidationError
from kvbench.schema.phase6 import (
    DECODE_SOURCE_SHA256,
    STAGE2_SOURCE_SHA256,
    STORE_SOURCE_SHA256,
)


def _run_payload(
    config_name: str = "turboquant_4bit_nc",
    *,
    runner: str = "fixed_l",
    graph: str = "eager",
    context: int = 128,
    output_steps: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": Phase6RunManifest.SCHEMA_VERSION,
        "artifact_schema_version": Phase6RunManifest.ARTIFACT_SCHEMA_VERSION,
        "run_id": "phase6-schema-fixture",
        "status": "created",
        "created_at_utc": "2026-07-25T01:02:03Z",
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_kind": "phase6_admission",
        "runner_kind": runner,
        "graph_mode": graph,
        "claim_class": "none",
        "measurement_scope": "measurement_container_admission",
        "performance_claim_eligible": False,
        "git_sha": "0" * 40,
        "git_dirty": False,
        "container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "method": "turboquant",
        "method_config_id": config_name,
        "method_config_fingerprint": "1" * 64,
        "adapter_version": "kvbench-turboquant-method-adapter-1.0.0",
        "adapter_source_sha256": "2" * 64,
        "adapter_config_fingerprint": "3" * 64,
        "pinned_source_commit": PINNED_SOURCE_COMMIT,
        "pinned_source_tree": PINNED_SOURCE_TREE,
        "fixture_set_sha256": FIXTURE_SET_SHA256,
        "fixture_root_ledger_sha256": FIXTURE_ROOT_LEDGER_SHA256,
        "cache_layout_fingerprint": "4" * 64,
        "slot_size_bytes": MANDATORY_CONFIG_SLOT_SIZES[config_name],
        "compressed_layers": list(range(2, 30)),
        "bf16_layers": [0, 1, 30, 31],
        "backend_identity": {
            "schema_version": "kvbench-phase6-backend-identity-1.0.0",
            "backend_id": "pytorch_flash_turboquant",
            "backend_fingerprint": "5" * 64,
            "store_kernel_family": "_tq_fused_store_mse",
            "decode_kernel_families": [
                "_tq_decode_stage1",
                "_fwd_kernel_stage2",
            ],
            "store_source_sha256": STORE_SOURCE_SHA256,
            "decode_source_sha256": DECODE_SOURCE_SHA256,
            "stage2_source_sha256": STAGE2_SOURCE_SHA256,
        },
        "batch_size": 1,
        "context_length": context,
        "output_steps": output_steps,
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "quality_execution": "locked",
        "performance_data_frozen": False,
        "quality_benchmark_executed": False,
        "speedup_calculated": False,
        "r_hbm": None,
        "full_scan_state": "CLOSED",
        "inventory_path": None,
        "failure_reason": None,
    }


def _report_payload() -> dict[str, object]:
    mandatory = list(MANDATORY_CONFIG_SLOT_SIZES)
    return {
        "schema_version": MethodAdmissionReportV2.SCHEMA_VERSION,
        "created_at_utc": "2026-07-25T02:03:04Z",
        "status": "PASS",
        "method_name": "turboquant",
        "mandatory_config_ids": mandatory,
        "admitted_config_ids": mandatory,
        "method_config_fingerprints": {
            item: "1" * 64 for item in mandatory
        },
        "adapter_version": "kvbench-turboquant-method-adapter-1.0.0",
        "adapter_source_sha256": "2" * 64,
        "adapter_config_fingerprints": {
            item: "3" * 64 for item in mandatory
        },
        "source_commit": PINNED_SOURCE_COMMIT,
        "source_tree": PINNED_SOURCE_TREE,
        "fixture_set_sha256": FIXTURE_SET_SHA256,
        "container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "model_identity": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "0" * 40,
            "fingerprint": "4" * 64,
        },
        "backend_identity": {
            "backend_id": "pytorch_flash_turboquant",
            "fingerprint": "5" * 64,
        },
        "cache_layout_fingerprints": {
            item: "6" * 64 for item in mandatory
        },
        "checks": [
            {
                "check_id": check_id,
                "status": "PASS",
                "summary": "Verified by immutable Phase 6 evidence.",
                "evidence_ids": ["phase6_bundle"],
            }
            for check_id in PHASE6_METHOD_ADMISSION_CHECK_IDS
        ],
        "reproducibility_status": "PASS",
        "evidence_references": [
            {
                "evidence_id": "phase6_bundle",
                "path": "artifacts/phase6/root_manifest.json",
                "sha256": "7" * 64,
            }
        ],
        "gates": {
            "g0": "PASS",
            "g1": "PASS",
            "g2_tq": "PASS",
            "global_g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
            "full_scan_state": "CLOSED",
        },
        "blockers": [],
        "claim_eligibility": "performance_only",
        "quality_status": "unvalidated",
        "quality_execution": "locked",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "quality_benchmark_executed": False,
        "speedup_calculated": False,
        "r_hbm": None,
        "measurement_scope": "measurement_container_admission",
        "creation_git_sha": "8" * 40,
    }


class Phase6SchemaTests(unittest.TestCase):
    def test_every_bounded_grid_shape_round_trips(self) -> None:
        payloads = [
            _run_payload(config, graph=graph)
            for config in MANDATORY_CONFIG_SLOT_SIZES
            for graph in ("eager", "cuda_graph")
        ]
        payloads.extend(
            [
                _run_payload(context=4096, graph="eager"),
                _run_payload(context=4096, graph="cuda_graph"),
                _run_payload(
                    runner="growing_context",
                    graph="eager",
                    output_steps=4,
                ),
            ]
        )
        self.assertEqual(len(payloads), 9)
        for payload in payloads:
            with self.subTest(payload=payload):
                parsed = Phase6RunManifest.from_dict(payload)
                self.assertEqual(parsed.to_dict(), payload)

    def test_manifest_rejects_authority_grid_and_governance_drift(self) -> None:
        mutations = (
            ("container_digest", "sha256:" + "0" * 64),
            ("method_config_id", "turboquant_k8v4"),
            ("slot_size_bytes", 133),
            ("compressed_layers", list(range(3, 30))),
            ("measurement_scope", "native_host_admission"),
            ("performance_claim_eligible", True),
            ("speedup_calculated", True),
            ("r_hbm", 1),
            ("full_scan_state", "OPEN"),
        )
        for field, value in mutations:
            payload = _run_payload()
            payload[field] = value
            with self.subTest(field=field):
                with self.assertRaises(SchemaValidationError):
                    Phase6RunManifest.from_dict(payload)
        for payload in (
            _run_payload("turboquant_k3v4_nc", context=4096),
            _run_payload(
                runner="growing_context",
                graph="cuda_graph",
                output_steps=4,
            ),
            _run_payload(
                "turboquant_3bit_nc",
                runner="growing_context",
                output_steps=4,
            ),
        ):
            with self.assertRaises(SchemaValidationError):
                Phase6RunManifest.from_dict(payload)

    def test_method_admission_v2_pass_and_strict_rejections(self) -> None:
        payload = _report_payload()
        parsed = MethodAdmissionReportV2.from_dict(payload)
        self.assertEqual(parsed.to_dict(), payload)
        mutations = []
        missing_check = deepcopy(payload)
        missing_check["checks"] = missing_check["checks"][:-1]
        mutations.append(missing_check)
        failed_check = deepcopy(payload)
        failed_check["checks"][0]["status"] = "FAIL"
        mutations.append(failed_check)
        global_g2 = deepcopy(payload)
        global_g2["gates"]["global_g2"] = "PASS"
        mutations.append(global_g2)
        missing_publication = deepcopy(payload)
        missing_publication["checks"][-1]["evidence_ids"] = []
        mutations.append(missing_publication)
        speedup = deepcopy(payload)
        speedup["speedup_calculated"] = True
        mutations.append(speedup)
        blocker = deepcopy(payload)
        blocker["blockers"] = ["unexpected"]
        mutations.append(blocker)
        for candidate in mutations:
            with self.subTest(candidate=candidate):
                with self.assertRaises(SchemaValidationError):
                    MethodAdmissionReportV2.from_dict(candidate)


if __name__ == "__main__":
    unittest.main()
