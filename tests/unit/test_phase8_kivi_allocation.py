"""Focused CPU tests for raw Phase 8 KIVI allocator attribution."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from kvbench.runtime.allocation_attribution import (
    AllocationClass,
    cuda_allocator_rounded_minimum,
    instantiate_decision_0013_phase8_kivi_rules,
    preserve_allocator_evidence,
)
from kvbench.runtime.kivi_allocation import (
    KIVIAllocationBinding,
    KIVIAllocationError,
    _derived_operation_summary,
    derive_kivi_allocation_attribution,
    replay_preserved_kivi_allocation_attribution,
)


def _binding(*, graph_mode: str = "eager") -> KIVIAllocationBinding:
    return KIVIAllocationBinding(
        configuration="k4v4",
        runner_kind="fixed_l",
        graph_mode=graph_mode,
        historical_context=128,
        attended_context=129,
        operation_fingerprint_sha256="1" * 64,
        cache_layout_fingerprint="2" * 64,
        method_fingerprint="3" * 64,
        backend_identity="4" * 64,
        adapter_source_sha256="5" * 64,
        cache_source_sha256="6" * 64,
        endpoint_source_sha256="7" * 64,
        authorized_container_digest=f"sha256:{'8' * 64}",
        official_commit="9" * 40,
        patched_tree="a" * 40,
        decision_0018_patch_sha256="b" * 64,
        extension_sha256="c" * 64,
    )


def _memory_stats(
    *,
    count: int = 0,
    requested_bytes: int = 0,
    block_bytes: int = 0,
) -> dict[str, int]:
    return {
        "allocation.all.allocated": count,
        "requested_bytes.all.allocated": requested_bytes,
        "allocated_bytes.all.allocated": block_bytes,
        "allocation.all.freed": count,
        "requested_bytes.all.freed": requested_bytes,
        "allocated_bytes.all.freed": block_bytes,
        "segment.all.allocated": 0,
        "segment.all.freed": 0,
        "num_device_alloc": 0,
        "num_device_free": 0,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }


def _accounting(
    binding: KIVIAllocationBinding,
    *,
    role: str,
    timestamp: int,
) -> dict[str, object]:
    return {
        "schema_version": "kvbench-phase3-memory-accounting-2.0.0",
        "operation_fingerprint_sha256": (
            binding.operation_fingerprint_sha256
        ),
        "sample_role": role,
        "timestamp_ns": timestamp,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_uuid": "GPU-phase8-unit-test",
        "allocated_bytes": 1_024,
        "reserved_bytes": 2_048,
        "device_free_bytes": 8_192,
        "device_total_bytes": 10_240,
        "device_used_bytes": 2_048,
    }


def _witness(binding: KIVIAllocationBinding) -> dict[str, object]:
    state = {
        "cache_pointers_sha256": "d" * 64,
        "active_context": binding.historical_context,
    }
    return {
        "schema_version": (
            "kvbench-phase8-kivi-allocation-operation-witness-1.0.0"
        ),
        "binding_sha256": binding.identity_sha256,
        "operation_fingerprint_sha256": (
            binding.operation_fingerprint_sha256
        ),
        "state_before": state,
        "state_after": dict(state),
        "measured_output": {"sha256": "e" * 64, "finite": True},
    }


def _catalog_trace(
    binding: KIVIAllocationBinding,
) -> tuple[list[dict[str, object]], int, int]:
    from kvbench.runtime.kivi_allocation import _geometry

    rules = instantiate_decision_0013_phase8_kivi_rules(
        geometry=_geometry(binding),
        backend_identity=binding.backend_identity,
        composition_binding_sha256=binding.identity_sha256,
    )
    trace: list[dict[str, object]] = []
    address = 0x100000
    requested_total = 0
    block_total = 0
    for policy in rules.permitted_allocation_policies:
        size = next(iter(policy.allowed_requested_bytes))
        block = cuda_allocator_rounded_minimum(size)
        python_frame = policy.required_python_frames[0]
        cpp_frame = policy.required_cpp_frames[0]
        for _ in range(policy.exact_count):
            trace.extend(
                (
                    {
                        "action": "alloc",
                        "addr": address,
                        "size": size,
                        "stream": 7,
                        "allocated_block_size": block,
                        "python_stack": [
                            {
                                "name": python_frame.function_name,
                                "filename": python_frame.source_suffix,
                                "line": 1,
                            }
                        ],
                        "cpp_stack": [
                            {
                                "name": cpp_frame.function_name,
                                "filename": cpp_frame.source_suffix,
                                "line": 1,
                            }
                        ],
                    },
                    {
                        "action": "free_requested",
                        "addr": address,
                        "size": size,
                        "stream": 7,
                    },
                    {
                        "action": "free_completed",
                        "addr": address,
                        "size": size,
                        "stream": 7,
                    },
                )
            )
            address += block + 0x1000
            requested_total += size
            block_total += block
    return trace, requested_total, block_total


def _derive(
    binding: KIVIAllocationBinding,
    trace: list[dict[str, object]],
    requested_total: int,
    block_total: int,
):
    snapshot = {"device_traces": [trace]}
    return derive_kivi_allocation_attribution(
        binding=binding,
        snapshot=snapshot,
        trace=tuple(trace),
        memory_stats_before=_memory_stats(),
        memory_stats_after=_memory_stats(
            count=sum(item.get("action") == "alloc" for item in trace),
            requested_bytes=requested_total,
            block_bytes=block_total,
        ),
        memory_accounting_before=_accounting(
            binding, role="before", timestamp=10
        ),
        memory_accounting_after=_accounting(
            binding, role="after", timestamp=20
        ),
        operation_witness=_witness(binding),
    )


class Phase8KIVIAllocationTests(unittest.TestCase):
    def test_decision_0013_composition_excludes_every_flash_policy(self) -> None:
        binding = _binding()
        trace, requested, blocks = _catalog_trace(binding)
        attribution, _, criterion = _derive(
            binding, trace, requested, blocks
        )
        self.assertTrue(criterion.passed)
        expected_count = sum(
            policy.exact_count
            for policy in attribution.rules.permitted_allocation_policies
        )
        expected_bytes = sum(
            policy.exact_total_requested_bytes
            for policy in attribution.rules.permitted_allocation_policies
        )
        self.assertEqual(
            criterion.allocation_event_count,
            expected_count,
        )
        self.assertEqual(requested, expected_bytes)
        self.assertEqual(
            attribution.class_counts(),
            {
                "fixed_output": 1,
                "fixed_shared_activation": 873,
            },
        )
        self.assertNotIn(
            AllocationClass.FRAMEWORK_BOOKKEEPING.value,
            attribution.class_counts(),
        )
        self.assertNotIn(
            AllocationClass.CONTEXT_SCALED_WORKSPACE.value,
            attribution.class_counts(),
        )

    def test_same_count_and_bytes_with_tampered_frame_fails_closed(self) -> None:
        binding = _binding()
        trace, requested, blocks = _catalog_trace(binding)
        attribution, _, criterion = _derive(
            binding, trace, requested, blocks
        )
        self.assertTrue(criterion.passed)
        tampered = copy.deepcopy(trace)
        first_alloc = next(
            item for item in tampered if item.get("action") == "alloc"
        )
        first_alloc["python_stack"][0]["name"] = "tampered_frame"
        changed, _, changed_criterion = _derive(
            binding, tampered, requested, blocks
        )
        self.assertEqual(
            len(changed.allocations), len(attribution.allocations)
        )
        self.assertEqual(
            sum(item.requested_bytes for item in changed.allocations),
            sum(item.requested_bytes for item in attribution.allocations),
        )
        self.assertFalse(changed_criterion.passed)
        self.assertEqual(
            changed.class_counts().get(AllocationClass.UNKNOWN.value),
            1,
        )
        self.assertIn(
            "forbidden_or_unattributed_allocation:0",
            changed_criterion.failure_reasons,
        )

    def test_preserved_replay_rejects_same_aggregate_tampered_frame(
        self,
    ) -> None:
        binding = _binding()
        trace, requested, blocks = _catalog_trace(binding)
        attribution, memory, criterion = _derive(
            binding,
            trace,
            requested,
            blocks,
        )
        summary, expected_count, expected_bytes, unknown_count = (
            _derived_operation_summary(
                attribution=attribution,
                memory=memory,
                criterion=criterion,
                binding=binding,
            )
        )
        witness = _witness(binding)
        audit_payload = {
            "schema_version": (
                "kvbench-phase8-kivi-allocation-attribution-1.0.0"
            ),
            "run_kind": "allocation_audit",
            "evidence_status": "complete",
            "execution_mode": binding.execution_mode,
            "binding": binding.to_dict(),
            "binding_sha256": binding.identity_sha256,
            "memory": memory.to_dict(),
            "attribution": attribution.to_dict(),
            "criterion": criterion.to_dict(),
            "expected_allocation_event_count": expected_count,
            "expected_allocation_event_bytes": expected_bytes,
            "observed_allocation_event_bytes": summary["raw"][
                "allocation_event_bytes"
            ],
            "unknown_allocation_count": unknown_count,
            "operation_witness": witness,
            "profiler_timing_reported": False,
            "instrumented_duration_reported_as_timing": False,
            "normal_benchmark_timing_eligible": False,
        }
        tampered = copy.deepcopy(trace)
        first_alloc = next(
            item for item in tampered if item.get("action") == "alloc"
        )
        first_alloc["python_stack"][0]["name"] = "tampered_frame"
        changed, _, changed_criterion = _derive(
            binding,
            tampered,
            requested,
            blocks,
        )
        self.assertEqual(
            len(changed.allocations),
            len(attribution.allocations),
        )
        self.assertEqual(
            sum(item.requested_bytes for item in changed.allocations),
            requested,
        )
        self.assertFalse(changed_criterion.passed)
        changed_history = changed.history_integrity
        self.assertIsNotNone(changed_history)
        assert changed_history is not None
        snapshot = {"device_traces": [tampered]}
        before_stats = _memory_stats()
        after_stats = _memory_stats(
            count=len(changed.allocations),
            requested_bytes=requested,
            block_bytes=blocks,
        )
        with tempfile.TemporaryDirectory(
            prefix="phase8-kivi-replay-negative-"
        ) as temporary:
            files = preserve_allocator_evidence(
                Path(temporary),
                snapshot=snapshot,
                trace=tampered,
                memory_stats_before=before_stats,
                memory_stats_after=after_stats,
                memory_accounting_before=_accounting(
                    binding,
                    role="before",
                    timestamp=10,
                ),
                memory_accounting_after=_accounting(
                    binding,
                    role="after",
                    timestamp=20,
                ),
                operation_witness=witness,
                expected_snapshot_sha256=(
                    changed_history.raw_snapshot_sha256
                ),
                expected_trace_sha256=changed_history.raw_trace_sha256,
                audit_payload=audit_payload,
            )
            with self.assertRaises(KIVIAllocationError):
                replay_preserved_kivi_allocation_attribution(
                    Path(temporary),
                    raw_files=files.to_dict(),
                    expected_binding=binding,
                )

    def test_graph_replay_requires_strict_raw_zero(self) -> None:
        binding = _binding(graph_mode="cuda_graph")
        attribution, _, criterion = _derive(binding, [], 0, 0)
        self.assertEqual(attribution.allocations, ())
        self.assertTrue(criterion.passed)
        self.assertTrue(criterion.strict_graph_zero_events)


if __name__ == "__main__":
    unittest.main()
