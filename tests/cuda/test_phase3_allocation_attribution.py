from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

import kvbench.runtime.allocation_attribution as attribution_module
from kvbench.runtime.allocation_attribution import (
    AllocationClass,
    OperationCacheStateWitness,
    OperationOutputWitness,
    OperationWitnessCallbacks,
    PHASE3_BACKEND_IDENTITY,
    ProductionAllocationBinding,
    SplitKCompositeRawInputs,
    build_phase3_production_allocation_binding,
    capture_output_witness_d2h,
    collect_cuda_allocation_attribution,
    validate_preserved_allocator_evidence_semantically,
    verify_preserved_allocator_evidence,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.backend import backend_identity, forced_flash_execution
from kvbench.runtime.numerical import (
    cache_history_sha256_untimed,
    tensor_sha256_untimed,
)
from kvbench.schema import GraphMode, RunnerKind
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)


def _binding(*, execution_mode: str) -> ProductionAllocationBinding:
    graph_mode = GraphMode(execution_mode)
    point = Phase3ProcessPoint(
        point_id=f"fixed_l-b1-l128-{graph_mode.value}-r1",
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=graph_mode,
        batch_size=1,
        context_length=128,
        output_steps=1,
        process_replicate=1,
        stability_member=False,
    )
    _, cache_fingerprint = attribution_module._phase3_cache_layout_fingerprint(
        runner_kind="fixed_l",
        batch=1,
        starting_context=128,
    )
    operation_key = Phase3AuditOperationKey.from_point(
        run_id=f"phase3-remediation-cuda-allocation-{execution_mode}",
        point=point,
        decode_step=0,
        cache_layout_fingerprint=cache_fingerprint,
        execution_git_sha="1" * 40,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
            PHASE3_FIXED_PLAN_PATH
        ],
        hardware_identity_sha256="2" * 64,
        software_identity_sha256="3" * 64,
        model_identity_sha256="4" * 64,
        backend_identity_sha256=hashlib.sha256(
            PHASE3_BACKEND_IDENTITY.encode("utf-8")
        ).hexdigest(),
        source_identity_sha256="5" * 64,
    )
    split_inputs = (
        SplitKCompositeRawInputs.from_raw_bytes(
            gqa_dispatch_trace=b"gqa-dispatch-control",
            mha_dispatch_trace=b"mha-dispatch-control",
            gqa_allocator_control=b"gqa-allocator-control",
            mha_allocator_control=b"mha-allocator-control",
            split_k_pair_multiplicity=((2, 1),),
        )
        if execution_mode == "eager"
        else None
    )
    return build_phase3_production_allocation_binding(
        operation_key=operation_key,
        backend_identity=PHASE3_BACKEND_IDENTITY,
        split_k_raw_inputs=split_inputs,
    )


class _CudaAttentionHarness:
    def __init__(
        self,
        *,
        binding: ProductionAllocationBinding,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.binding = binding
        self.query = query
        self.key = key
        self.value = value
        self.current_key = torch.randn_like(key[:, :, :1, :])
        self.current_value = torch.randn_like(value[:, :, :1, :])
        self.active_length = binding.historical_context
        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_output: torch.Tensor | None = None

    def prepare(self) -> None:
        self.key[:, :, -1:, :].zero_()
        self.value[:, :, -1:, :].zero_()
        self.active_length = self.binding.historical_context

    def _attention(self) -> torch.Tensor:
        self.key[:, :, -1:, :].copy_(self.current_key)
        self.value[:, :, -1:, :].copy_(self.current_value)
        output = torch.nn.functional.scaled_dot_product_attention(
            self.query,
            self.key,
            self.value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=128**-0.5,
            enable_gqa=True,
        )
        self.active_length = self.binding.historical_context
        return output

    def eager_operation(self) -> torch.Tensor:
        with forced_flash_execution():
            return self._attention()

    def graph_operation(self) -> torch.Tensor:
        if self.graph is None or self.graph_output is None:
            raise RuntimeError("graph harness is not captured")
        self.graph.replay()
        self.active_length = self.binding.historical_context
        return self.graph_output

    def capture_graph(self) -> None:
        graph = torch.cuda.CUDAGraph()
        with forced_flash_execution(), torch.cuda.graph(graph):
            output = self._attention()
        self.graph = graph
        self.graph_output = output
        self.prepare()

    def capture_cache_state(self) -> OperationCacheStateWitness:
        key = self.key.unsqueeze(0)
        value = self.value.unsqueeze(0)
        prefix = cache_history_sha256_untimed(
            key,
            value,
            historical_length=self.binding.historical_context,
        )
        destination_key = self.key[
            :, :, self.binding.historical_context :, :
        ]
        destination_value = self.value[
            :, :, self.binding.historical_context :, :
        ]
        destination = hashlib.sha256(
            (
                tensor_sha256_untimed(destination_key)
                + ":"
                + tensor_sha256_untimed(destination_value)
            ).encode("ascii")
        ).hexdigest()
        destination_is_sentinel = bool(
            torch.count_nonzero(destination_key).item() == 0
            and torch.count_nonzero(destination_value).item() == 0
        )
        return OperationCacheStateWitness(
            active_length=self.active_length,
            key_shape=tuple(int(item) for item in key.shape),
            value_shape=tuple(int(item) for item in value.shape),
            key_strides=tuple(int(item) for item in key.stride()),
            value_strides=tuple(int(item) for item in value.stride()),
            key_dtype=str(key.dtype),
            value_dtype=str(value.dtype),
            key_device=str(key.device),
            value_device=str(value.device),
            key_data_ptr=int(key.data_ptr()),
            value_data_ptr=int(value.data_ptr()),
            historical_prefix_sha256=prefix,
            destination_slot_sha256=destination,
            destination_slot_is_sentinel=destination_is_sentinel,
            layout_fingerprint=self.binding.cache_layout_fingerprint,
        )

    def capture_output(
        self, output: torch.Tensor
    ) -> OperationOutputWitness:
        return capture_output_witness_d2h(output)

    @property
    def witness(self) -> OperationWitnessCallbacks:
        return OperationWitnessCallbacks(
            capture_cache_state=self.capture_cache_state,
            capture_output=self.capture_output,
        )


class Phase3AllocationAttributionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.device = torch.device("cuda:0")
        observed_identity = json.dumps(
            backend_identity(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if observed_identity != PHASE3_BACKEND_IDENTITY:
            raise AssertionError("runtime backend is not the frozen identity")

    def _inputs(
        self, binding: ProductionAllocationBinding
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.randn(
            (1, 32, 1, 128),
            device=self.device,
            dtype=torch.bfloat16,
        )
        key = torch.randn(
            (1, 8, binding.attended_context, 128),
            device=self.device,
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key)
        return query, key, value

    def test_output_witness_uses_canonical_tensor_checksum(self) -> None:
        output = torch.arange(
            2 * 3 * 16,
            dtype=torch.bfloat16,
            device=self.device,
        ).reshape(2, 3, 16)
        witness = capture_output_witness_d2h(output)

        self.assertEqual(
            witness.sha256,
            tensor_sha256_untimed(output),
        )

    def test_isolated_attention_is_rejected_as_not_the_full_endpoint(
        self,
    ) -> None:
        binding = _binding(execution_mode="eager")
        query, key, value = self._inputs(binding)
        harness = _CudaAttentionHarness(
            binding=binding,
            query=query,
            key=key,
            value=value,
        )
        with tempfile.TemporaryDirectory(
            prefix="phase3-allocation-attribution-eager-"
        ) as temporary:
            staging = Path(temporary)
            result = collect_cuda_allocation_attribution(
                harness.eager_operation,
                production_binding=binding,
                staging_directory=staging,
                operation_witness=harness.witness,
                prepare_operation=harness.prepare,
                warmup_iterations=3,
                max_entries=100_000,
                device=self.device,
            )
            self.assertFalse(result.criterion.passed)
            self.assertIn(
                "operation_witness_output_geometry_mismatch",
                result.attribution.integrity_errors,
            )
            self.assertIn(
                "operation_witness_reference_before_cache_shape_mismatch",
                result.attribution.integrity_errors,
            )
            self.assertGreater(len(result.attribution.allocations), 0)
            self.assertFalse(
                any(
                    item.event_class is AllocationClass.GQA_EXPANSION
                    for item in result.attribution.allocations
                )
            )
            self.assertTrue(
                verify_preserved_allocator_evidence(
                    staging, result.raw_files
                )
            )
            semantic = validate_preserved_allocator_evidence_semantically(
                staging,
                result.raw_files,
                production_binding=binding,
            )
            self.assertFalse(semantic.passed)
            self.assertEqual(semantic.failure_reasons, ())
            self.assertIsNotNone(semantic.criterion)
            assert semantic.criterion is not None
            self.assertFalse(semantic.criterion.passed)
            self.assertFalse(
                result.to_dict()[
                    "instrumented_duration_reported_as_timing"
                ]
            )

    def test_toy_graph_replay_cannot_claim_production_allocation_pass(
        self,
    ) -> None:
        binding = _binding(execution_mode="cuda_graph")
        query, key, value = self._inputs(binding)
        harness = _CudaAttentionHarness(
            binding=binding,
            query=query,
            key=key,
            value=value,
        )
        harness.capture_graph()
        assert harness.graph_output is not None
        self.assertEqual(harness.graph_output.shape, query.shape)
        with tempfile.TemporaryDirectory(
            prefix="phase3-allocation-attribution-graph-"
        ) as temporary:
            staging = Path(temporary)
            result = collect_cuda_allocation_attribution(
                harness.graph_operation,
                production_binding=binding,
                staging_directory=staging,
                operation_witness=harness.witness,
                prepare_operation=harness.prepare,
                warmup_iterations=3,
                max_entries=100_000,
                device=self.device,
            )
            self.assertFalse(result.criterion.passed)
            self.assertIn(
                "operation_witness_output_geometry_mismatch",
                result.attribution.integrity_errors,
            )
            self.assertIn(
                "operation_witness_reference_before_cache_shape_mismatch",
                result.attribution.integrity_errors,
            )
            semantic = validate_preserved_allocator_evidence_semantically(
                staging,
                result.raw_files,
                production_binding=binding,
            )
            self.assertFalse(semantic.passed)
            self.assertEqual(semantic.failure_reasons, ())
            self.assertIsNotNone(semantic.criterion)
            assert semantic.criterion is not None
            self.assertFalse(semantic.criterion.passed)


if __name__ == "__main__":
    unittest.main()
