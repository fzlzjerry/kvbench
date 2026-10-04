"""Strict schema tests for the compact Phase 4 method admission report."""

from __future__ import annotations

import copy
import json
import unittest

from kvbench.errors import SchemaValidationError
from kvbench.schema import MethodAdmissionReport


def _report_payload() -> dict[str, object]:
    evidence_id = "phase3-g1-report"
    result = {
        "status": "PASS",
        "summary": "existing Phase 3 evidence revalidated through the adapter",
        "evidence_ids": [evidence_id],
    }
    return {
        "schema_version": MethodAdmissionReport.SCHEMA_VERSION,
        "created_at_utc": "2026-07-24T00:00:00Z",
        "status": "PASS",
        "method_name": "bf16",
        "method_config_id": "bf16",
        "method_config_fingerprint": "1" * 64,
        "adapter_version": "kvbench-bf16-method-adapter-1.0.0",
        "adapter_config_fingerprint": "2" * 64,
        "model_identity": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "3" * 40,
            "fingerprint": "4" * 64,
        },
        "backend_identity": {
            "backend_id": "torch_sdpa_flash_gqa",
            "fingerprint": "5" * 64,
        },
        "cache_layout_fingerprint": "6" * 64,
        "correctness": copy.deepcopy(result),
        "byte_accounting": copy.deepcopy(result),
        "execution_path": copy.deepcopy(result),
        "graph": copy.deepcopy(result),
        "reproducibility_status": "PASS",
        "evidence_references": [
            {
                "evidence_id": evidence_id,
                "path": "artifacts/phase3_reports/report.json",
                "sha256": "7" * 64,
            }
        ],
        "gates": {
            "g0": "PASS",
            "g1": "PASS",
            "g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
            "full_scan_state": "closed",
        },
        "blockers": ["B-009", "B-010"],
        "claim_eligibility": "performance_only",
        "quality_status": "unvalidated",
        "quality_execution": "locked",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "quality_benchmark_executed": False,
        "measurement_scope": "native_host_admission",
        "creation_git_sha": "8" * 40,
    }


class MethodAdmissionReportTests(unittest.TestCase):
    def test_exact_bf16_report_round_trips_canonically(self) -> None:
        report = MethodAdmissionReport.from_dict(_report_payload())
        raw = report.canonical_bytes()
        reparsed = MethodAdmissionReport.from_dict(json.loads(raw))
        self.assertEqual(reparsed.canonical_bytes(), raw)
        self.assertEqual(reparsed.gates.g1.value, "PASS")
        self.assertFalse(reparsed.performance_claim_eligible)

    def test_unknown_field_and_unjoined_evidence_fail_closed(self) -> None:
        unknown = _report_payload()
        unknown["raw_trace"] = {}
        with self.assertRaises(SchemaValidationError):
            MethodAdmissionReport.from_dict(unknown)

        unjoined = _report_payload()
        unjoined["graph"]["evidence_ids"] = ["missing"]  # type: ignore[index]
        with self.assertRaises(SchemaValidationError):
            MethodAdmissionReport.from_dict(unjoined)

    def test_pass_gate_and_quality_loopholes_are_rejected(self) -> None:
        for path, value in (
            (("correctness", "status"), "FAIL"),
            (("gates", "g2"), "PASS"),
            (("gates", "full_scan_state"), "open"),
            (("quality_execution",), "unlocked"),
            (("performance_claim_eligible",), True),
            (("performance_data_frozen",), True),
        ):
            with self.subTest(path=path):
                payload = _report_payload()
                target: dict[str, object] = payload
                for part in path[:-1]:
                    target = target[part]  # type: ignore[assignment]
                target[path[-1]] = value
                with self.assertRaises(SchemaValidationError):
                    MethodAdmissionReport.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
