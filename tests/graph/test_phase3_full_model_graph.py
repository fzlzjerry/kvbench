"""Untimed exact-checkpoint full-model CUDA Graph validation."""

from __future__ import annotations

import unittest

import torch
from kvbench.adapters import (
    build_method_adapter,
    declared_bf16_runtime_context,
)

from kvbench.runtime.cuda_graph import validate_full_model_fixed_graph
from kvbench.runtime.model_loader import load_frozen_model
from kvbench.schema import GraphMode
from tests.cuda.test_phase3_full_model import (
    collect_exact_endpoint_audit,
    retained_fixed_output_checksums_untimed,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3FullModelGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_frozen_model(device="cuda:0")
        cls.adapter = build_method_adapter(
            "bf16",
            declared_bf16_runtime_context(cls.loaded.model),
        )

    def test_short_fixed_shape_capture_replay_is_exact_and_allocation_free(
        self,
    ) -> None:
        prefix = torch.arange(
            3_000,
            3_008,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        current = torch.tensor([[4_000]], dtype=torch.long, device="cuda:0")
        result = validate_full_model_fixed_graph(
            self.loaded.model,
            prefix,
            current,
            method=self.adapter,
        )
        self.assertTrue(result.passed, result.to_dict())
        self.assertTrue(result.eager_replay_comparison["passed"])
        self.assertTrue(result.replay_outputs_exact)
        self.assertTrue(result.replay_copies_independent)
        self.assertEqual(
            result.first_replay_checksum,
            result.second_replay_checksum,
        )
        self.assertTrue(result.cache_pointers_stable)
        self.assertTrue(result.historical_cache_unchanged)
        self.assertTrue(result.replay_allocation["passed"])
        self.assertEqual(result.replay_allocation["allocation_event_count"], 0)
        serialized = result.to_dict()
        self.assertFalse(serialized["timing_collected"])
        self.assertFalse(serialized["performance_claim_eligible"])

    def test_exact_endpoint_retained_graph_audit_admits_without_timing(
        self,
    ) -> None:
        session, record, evidence_root = collect_exact_endpoint_audit(
            self.loaded,
            graph_mode=GraphMode.CUDA_GRAPH,
        )
        print(f"preserved_endpoint_graph_audit={evidence_root}")
        self.assertEqual(session.method.name, "bf16")
        self.assertEqual(len(session.adapter_config_fingerprint), 64)
        self.assertEqual(record.status, "completed")
        self.assertEqual(session.state, "ready")
        self.assertTrue(session.provenance_payload()["graph_retained"])
        self.assertIsNotNone(session.graph_evidence)
        self.assertTrue(
            session.graph_evidence["consecutive_replay_outputs_exact"]
        )
        audit_checksum, audit_finite = session.audit_output(0)
        observed = retained_fixed_output_checksums_untimed(session)
        self.assertTrue(audit_finite)
        self.assertEqual(set(observed), {audit_checksum})



if __name__ == "__main__":
    unittest.main()
