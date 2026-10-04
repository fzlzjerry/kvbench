"""Actual CUDA operator controls for the Phase 3 B-011 dispatch gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from kvbench.runtime.phase3_allocator_controls import (
    collect_phase3_paired_allocator_controls,
    verify_phase3_paired_allocator_controls,
)
from kvbench.runtime.backend import backend_identity, forced_flash_execution
from kvbench.runtime.gqa_device_dispatch import (
    CUDA_GRAPH_REPLAY_EXECUTION_MODE,
    FLASH_FORWARD_FAMILY,
    MATERIALIZATION_CLASSIFICATIONS,
    collect_gqa_mha_device_dispatch,
    collect_phase3_geometry_bound_gqa_mha_device_dispatch,
    collect_torch_profiler_trace,
    parse_scoped_chrome_cuda_graph_events,
    phase3_source_identity_sha256,
    revalidate_phase3_geometry_bound_dispatch_audit,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.static_cache import BF16StaticCache
from kvbench.schema import GQAVerdict, GraphMode, RunnerKind
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_HARDWARE_FINGERPRINT,
    PHASE3_MODEL_FINGERPRINT,
    PHASE3_PLAN_FINGERPRINTS,
    PHASE3_SOFTWARE_FINGERPRINT,
    Phase3ProcessPoint,
)


def _operation_key(
    *,
    graph_mode: GraphMode,
    run_id: str,
    cache_layout_fingerprint: str,
    repository_root: Path,
) -> Phase3AuditOperationKey:
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
    backend_payload = backend_identity()
    backend_raw = json.dumps(
        backend_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    source_hashes = {
        path: hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
        for path in (
            "src/kvbench/runtime/backend.py",
            "src/kvbench/runtime/bf16_endpoint.py",
            "src/kvbench/runtime/static_cache.py",
        )
    }
    return Phase3AuditOperationKey.from_point(
        run_id=run_id,
        point=point,
        decode_step=0,
        cache_layout_fingerprint=cache_layout_fingerprint,
        execution_git_sha="5" * 40,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[PHASE3_FIXED_PLAN_PATH],
        hardware_identity_sha256=PHASE3_HARDWARE_FINGERPRINT,
        software_identity_sha256=PHASE3_SOFTWARE_FINGERPRINT,
        model_identity_sha256=PHASE3_MODEL_FINGERPRINT,
        backend_identity_sha256=hashlib.sha256(backend_raw).hexdigest(),
        source_identity_sha256=phase3_source_identity_sha256(source_hashes),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3GQADeviceDispatchCudaTests(unittest.TestCase):
    def _collect_paired_allocator_control(
        self,
        *,
        graph_mode: GraphMode,
    ) -> tuple[object, object, Path]:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(20260722)
        batch_size = 1
        starting_context = 128
        active_context = starting_context + 1
        capacity = active_context
        workspace_bytes = 32 * batch_size * (32 + 8) * 1 * 64 * 2
        cache = BF16StaticCache(
            num_layers=32,
            batch_size=batch_size,
            num_kv_heads=8,
            capacity=capacity,
            head_dim=128,
            device=device,
            workspace_bytes=workspace_bytes,
        )
        query = torch.randn(
            (batch_size, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        gqa_key = cache.keys[0, :, :, :active_context, :]
        gqa_value = cache.values[0, :, :, :active_context, :]
        mha_key = torch.randn(
            (batch_size, 32, active_context, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        mha_value = torch.randn_like(mha_key, generator=generator)
        repository_root = Path(__file__).resolve().parents[2]
        output_directory = Path(
            tempfile.mkdtemp(
                prefix=(
                    "kvbench-phase3-paired-allocator-"
                    f"{graph_mode.value}-"
                ),
                dir="/tmp",
            )
        )
        operation_key = _operation_key(
            graph_mode=graph_mode,
            run_id=f"phase3-b011-b012-{graph_mode.value}-control",
            cache_layout_fingerprint=cache.layout_fingerprint(),
            repository_root=repository_root,
        )
        dispatch = collect_phase3_geometry_bound_gqa_mha_device_dispatch(
            operation_key=operation_key,
            cache_layout_fingerprint=cache.layout_fingerprint(),
            cache_workspace_bytes=workspace_bytes,
            cache_layer_index=0,
            cache_key_backing=cache.keys,
            cache_value_backing=cache.values,
            gqa_query=query,
            gqa_key_view=gqa_key,
            gqa_value_view=gqa_value,
            mha_query=query,
            mha_key=mha_key,
            mha_value=mha_value,
            output_directory=output_directory,
            artifact_relative_root=(
                f"dispatch/{graph_mode.value}-allocator-traces"
            ),
            source_root=repository_root,
            source_paths=tuple(
                Path(item)
                for item in (
                    "src/kvbench/runtime/backend.py",
                    "src/kvbench/runtime/bf16_endpoint.py",
                    "src/kvbench/runtime/static_cache.py",
                )
            ),
            is_causal=False,
            scale=128**-0.5,
            warmup_count=3,
        )
        (output_directory / "dispatch-audit.json").write_text(
            json.dumps(dispatch.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        gqa_trace_raw = (
            output_directory / "gqa.geometry.chrome.json"
        ).read_bytes()
        mha_trace_raw = (
            output_directory / "mha.geometry.chrome.json"
        ).read_bytes()
        gqa_raw, mha_raw = collect_phase3_paired_allocator_controls(
            operation_key=operation_key,
            query=query,
            gqa_key=gqa_key,
            gqa_value=gqa_value,
            mha_key=mha_key,
            mha_value=mha_value,
            gqa_dispatch_trace_raw=gqa_trace_raw,
            mha_dispatch_trace_raw=mha_trace_raw,
        )
        (output_directory / "gqa.allocator-control.json").write_bytes(gqa_raw)
        (output_directory / "mha.allocator-control.json").write_bytes(mha_raw)
        verification = verify_phase3_paired_allocator_controls(
            gqa_raw=gqa_raw,
            mha_control_raw=mha_raw,
            operation_key=operation_key,
            gqa_dispatch_trace_raw=gqa_trace_raw,
            mha_dispatch_trace_raw=mha_trace_raw,
        )
        return dispatch, verification, output_directory

    def test_paired_allocator_controls_eager_and_graph(self) -> None:
        forbidden_formulas = {
            "expanded_kv",
            "native_or_context_kv",
            "context_scaled_unknown",
            "unknown",
        }
        for graph_mode in (GraphMode.EAGER, GraphMode.CUDA_GRAPH):
            with self.subTest(graph_mode=graph_mode.value):
                dispatch, verification, output_directory = (
                    self._collect_paired_allocator_control(
                        graph_mode=graph_mode
                    )
                )
                print(f"preserved_paired_allocator_control={output_directory}")
                self.assertTrue(dispatch.evaluation.dispatch_verified)
                expected_mode = (
                    "eager"
                    if graph_mode is GraphMode.EAGER
                    else CUDA_GRAPH_REPLAY_EXECUTION_MODE
                )
                self.assertEqual(dispatch.gqa.execution_mode, expected_mode)
                self.assertEqual(dispatch.mha.execution_mode, expected_mode)
                repository_root = Path(__file__).resolve().parents[2]
                source_bytes = {
                    path: (repository_root / path).read_bytes()
                    for path in (
                        "src/kvbench/runtime/backend.py",
                        "src/kvbench/runtime/bf16_endpoint.py",
                        "src/kvbench/runtime/static_cache.py",
                    )
                }
                rebuilt = revalidate_phase3_geometry_bound_dispatch_audit(
                    dispatch,
                    gqa_raw=(
                        output_directory / "gqa.geometry.chrome.json"
                    ).read_bytes(),
                    mha_raw=(
                        output_directory / "mha.geometry.chrome.json"
                    ).read_bytes(),
                    backend_identity_raw=(
                        dispatch.backend_identity.canonical_json.encode("utf-8")
                    ),
                    source_bytes_by_path=source_bytes,
                )
                self.assertEqual(rebuilt, dispatch)
                self.assertTrue(
                    verification.passed,
                    verification.failure_reasons,
                )
                for replay in (verification.gqa, verification.mha_control):
                    self.assertEqual(replay.failure_reasons, ())
                    self.assertFalse(
                        forbidden_formulas.intersection(
                            fact.formula_id
                            for fact in replay.allocation_facts
                        )
                    )
                if graph_mode is GraphMode.CUDA_GRAPH:
                    self.assertFalse(verification.gqa.allocation_facts)
                    self.assertFalse(verification.mha_control.allocation_facts)

    def test_public_flash_gqa_and_mha_controls_expose_device_kernels(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(20260722)
        query = torch.randn(
            (1, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        gqa_key = torch.randn(
            (1, 8, 128, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        gqa_value = torch.randn_like(gqa_key, generator=generator)
        mha_key = torch.randn(
            (1, 32, 128, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        mha_value = torch.randn_like(mha_key, generator=generator)
        repository_root = Path(__file__).resolve().parents[2]
        output_directory = Path(
            tempfile.mkdtemp(prefix="kvbench-phase3-dispatch-audit-", dir="/tmp")
        )
        audit = collect_gqa_mha_device_dispatch(
            gqa_query=query,
            gqa_key=gqa_key,
            gqa_value=gqa_value,
            mha_query=query,
            mha_key=mha_key,
            mha_value=mha_value,
            output_directory=output_directory,
            artifact_relative_root="dispatch/traces",
            source_root=repository_root,
            source_paths=(
                Path("src/kvbench/runtime/backend.py"),
                Path("src/kvbench/runtime/bf16_endpoint.py"),
                Path("src/kvbench/runtime/static_cache.py"),
            ),
            is_causal=False,
            scale=128**-0.5,
            warmup_count=3,
            # Task C supplies the independent allocator-history proof. This
            # operator gate must remain unproven, rather than inventing it.
            allocation_verified=False,
        )
        payload = audit.to_dict()
        (output_directory / "dispatch-audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"preserved_dispatch_audit={output_directory}")

        self.assertTrue(audit.raw_trace_bytes_verified)
        self.assertTrue(audit.gqa.backend.passed)
        self.assertTrue(audit.mha.backend.passed)
        self.assertEqual(audit.gqa.backend.enabled_backends, ("FLASH_ATTENTION",))
        self.assertEqual(audit.mha.backend.enabled_backends, ("FLASH_ATTENTION",))
        self.assertTrue(audit.evaluation.dispatch_verified)
        self.assertTrue(audit.evaluation.no_replication_kernel_verified)
        self.assertEqual(
            audit.evaluation.verdict,
            GQAVerdict.NONMATERIALIZATION_UNPROVEN,
        )
        self.assertEqual(audit.gqa.attention_families, (FLASH_FORWARD_FAMILY,))
        self.assertEqual(audit.mha.attention_families, (FLASH_FORWARD_FAMILY,))
        self.assertFalse(audit.gqa.events_before_attention)
        self.assertFalse(audit.mha.events_before_attention)
        self.assertFalse(
            any(
                event.classification in MATERIALIZATION_CLASSIFICATIONS
                for event in (*audit.gqa.device_events, *audit.mha.device_events)
            )
        )
        self.assertTrue(audit.gqa_source_shape.source_verified)
        self.assertTrue(audit.mha_source_shape.source_verified)
        self.assertTrue(audit.gqa_source_shape.shape_verified_for(audit.gqa))
        self.assertTrue(audit.mha_source_shape.shape_verified_for(audit.mha))
        self.assertEqual(audit.gqa.byte_evidence.native_kv_bytes, 524_288)
        self.assertEqual(audit.gqa.byte_evidence.expanded_kv_bytes, 2_097_152)
        for role, artifact in (
            ("gqa", audit.gqa.raw_trace),
            ("mha", audit.mha.raw_trace),
        ):
            self.assertIsNotNone(artifact)
            assert artifact is not None
            path = output_directory / f"{role}.chrome.json"
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), artifact.sha256)
            self.assertEqual(len(raw), artifact.size_bytes)
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in ('"latency"', '"duration"', '"wall_time_ms"'):
            self.assertNotIn(forbidden, rendered)

    def test_long_graph_control_allows_async_gpu_completion(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(20260723)
        query = torch.randn(
            (1, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        key = torch.randn(
            (1, 32, 16_385, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        value = torch.randn_like(key, generator=generator)
        marker = "kvbench.phase3.b016.long-mha-graph-control"

        def operation() -> torch.Tensor:
            with forced_flash_execution():
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=128**-0.5,
                    enable_gqa=True,
                )

        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-b016-long-graph-",
            dir="/tmp",
        ) as directory:
            trace_path = Path(directory) / "mha.geometry.chrome.json"
            artifact = collect_torch_profiler_trace(
                operation,
                trace_path,
                artifact_relative_path=trace_path.name,
                marker=marker,
                warmup_count=3,
                device=device,
                execution_mode=CUDA_GRAPH_REPLAY_EXECUTION_MODE,
            )
            raw = trace_path.read_bytes()
            parsed = parse_scoped_chrome_cuda_graph_events(
                raw,
                marker=marker,
            )
            self.assertEqual(hashlib.sha256(raw).hexdigest(), artifact.sha256)
            self.assertEqual(len(raw), artifact.size_bytes)
            self.assertTrue(parsed.device_events)
            self.assertTrue(
                all(
                    event.classification == "flash_attention"
                    for event in parsed.device_events
                )
            )
            self.assertFalse(
                any(
                    event.classification in MATERIALIZATION_CLASSIFICATIONS
                    for event in parsed.device_events
                )
            )


if __name__ == "__main__":
    unittest.main()
