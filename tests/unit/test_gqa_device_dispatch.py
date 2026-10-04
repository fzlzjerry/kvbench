"""Pure operator fixtures for the Phase 3 CUDA device-dispatch audit."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

from kvbench.runtime.gqa_device_dispatch import (
    BackendControlEvidence,
    BackendIdentityEvidence,
    COPY_CANDIDATE_CLASSIFICATIONS,
    CUDADeviceEvent,
    ChromeTraceValidationError,
    DispatchControlEvidence,
    GQADeviceDispatchError,
    FLASH_FORWARD_FAMILY,
    FLASH_SPLIT_KV_FAMILY,
    RawTraceArtifact,
    SourceFileEvidence,
    SourceShapeEvidence,
    TensorShapeEvidence,
    TraceScopeEvidence,
    _collect_rejected_backend_control,
    audit_gqa_source_files,
    calculate_kv_bytes,
    collect_torch_profiler_trace,
    compare_kernel_families,
    evaluate_gqa_device_dispatch,
    parse_chrome_cuda_events,
    parse_scoped_chrome_cuda_events,
)
from kvbench.schema import GQAVerdict


FLASH_NAME = (
    "void pytorch_flash::flash_fwd_kernel<traits>("
    "pytorch_flash::Flash_fwd_params)"
)
SPLIT_NAME = "void pytorch_flash::flash_fwd_splitkv_kernel<traits>(params)"
SPLIT_COMBINE_NAME = (
    "void pytorch_flash::flash_fwd_splitkv_combine_kernel<traits>(params)"
)


def cuda_event(
    name: str,
    *,
    category: str = "kernel",
    timestamp: float = 20.0,
    correlation: int = 43,
    external_id: int = 9,
    copy_bytes: int | None = None,
    omit: frozenset[str] = frozenset(),
) -> dict[str, object]:
    args: dict[str, object] = {
        "stream": 7,
        "correlation": correlation,
        "External id": external_id,
        "device": 0,
        "context": 1,
    }
    if copy_bytes is not None:
        args["bytes"] = copy_bytes
    for key in omit:
        args.pop(key, None)
    return {
        "ph": "X",
        "cat": category,
        "name": name,
        "ts": timestamp,
        "dur": 1.0,
        "args": args,
    }


def trace_bytes(*events: dict[str, object]) -> bytes:
    host_event = {
        "ph": "X",
        "cat": "cpu_op",
        "name": "aten::scaled_dot_product_attention",
        "ts": 1.0,
        "dur": 1000.0,
        "args": {"External id": 8},
    }
    return json.dumps(
        {"schemaVersion": 1, "traceEvents": [host_event, *events]},
        sort_keys=True,
    ).encode("utf-8")


def scoped_trace_bytes(
    marker: str,
    *events: dict[str, object],
) -> bytes:
    marker_external_id = 1
    scoped_events: list[dict[str, object]] = []
    runtime_events: list[dict[str, object]] = []
    for event in events:
        copied = dict(event)
        args = dict(copied["args"])  # type: ignore[arg-type]
        args["External id"] = 10
        copied["args"] = args
        scoped_events.append(copied)
        category = copied["cat"]
        runtime_name = {
            "gpu_memcpy": "cudaMemcpyAsync",
            "gpu_memset": "cudaMemsetAsync",
        }.get(category, "cudaLaunchKernel")
        runtime_events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": runtime_name,
                "pid": 101,
                "tid": 101,
                "ts": float(copied["ts"]) - 0.5,
                "dur": 0.25,
                "args": {
                    "External id": 10,
                    "correlation": args["correlation"],
                },
            }
        )
    gpu_start = min(float(event["ts"]) for event in scoped_events) - 0.1
    gpu_end = max(
        float(event["ts"]) + float(event["dur"]) for event in scoped_events
    ) + 0.1
    trace_events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": marker,
            "pid": 101,
            "tid": 101,
            "ts": 0.0,
            "dur": 200.0,
            "args": {"External id": marker_external_id},
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::scaled_dot_product_attention",
            "pid": 101,
            "tid": 101,
            "ts": 5.0,
            "dur": 180.0,
            "args": {"External id": 2},
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "arbitrary_nested_fused_dispatch",
            "pid": 101,
            "tid": 101,
            "ts": 10.0,
            "dur": 160.0,
            "args": {"External id": 10},
        },
        *runtime_events,
        *scoped_events,
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": marker,
            "pid": 0,
            "tid": 7,
            "ts": gpu_start,
            "dur": gpu_end - gpu_start,
            "args": {"External id": marker_external_id},
        },
    ]
    return json.dumps(
        {"schemaVersion": 1, "traceEvents": trace_events},
        sort_keys=True,
    ).encode("utf-8")


def raw_artifact(role: str) -> RawTraceArtifact:
    return RawTraceArtifact(
        relative_path=f"gqa/traces/{role}.chrome.json",
        sha256="a" * 64,
        size_bytes=1024,
    )


def passing_backend() -> BackendControlEvidence:
    return BackendControlEvidence(
        enabled_backends=("FLASH_ATTENTION",),
        flash_eligible=True,
        fused_backend_name="FLASH_ATTENTION",
        rejected_control_failed=True,
        rejected_control_error="No available kernel. Aborting execution.",
        rejected_control_warnings=(
            "Flash attention kernel not used because:",
            "Expected query, key and value to all be of dtype: {Half, BFloat16}.",
        ),
        rejected_control_synchronized=True,
        source_build_fingerprint="b" * 64,
        source_build_verified=True,
    )


def control(
    role: str,
    events: tuple[CUDADeviceEvent, ...],
    *,
    context_length: int = 128,
    backend: BackendControlEvidence | None = None,
    with_raw_trace: bool = True,
    with_trace_scope: bool = True,
) -> DispatchControlEvidence:
    correlations = tuple(sorted({event.correlation_id for event in events}))
    external_ids = tuple(sorted({2, *(event.external_id for event in events)}))
    trace_scope = None
    if with_trace_scope and events:
        trace_scope = TraceScopeEvidence(
            marker=f"kvbench.phase3.{role}.dispatch_audit",
            marker_external_id=1,
            cpu_process_id=101,
            cpu_thread_id=101,
            sdpa_external_id=2,
            nested_cpu_external_ids=external_ids,
            runtime_correlations=correlations,
            gpu_stream=7,
        )
    return DispatchControlEvidence(
        role=role,
        batch_size=1,
        context_length=context_length,
        query_length=1,
        num_query_heads=32,
        num_kv_heads=8 if role == "gqa" else 32,
        head_dim=128,
        dtype="torch.bfloat16",
        dtype_bytes=2,
        is_causal=False,
        warmup_count=3,
        backend=passing_backend() if backend is None else backend,
        raw_trace=raw_artifact(role) if with_raw_trace else None,
        trace_scope=trace_scope,
        device_events=events,
    )


def complete_evaluation(
    gqa: DispatchControlEvidence | None,
    mha: DispatchControlEvidence | None,
    **overrides: object,
):
    arguments: dict[str, object] = {
        "gqa": gqa,
        "mha": mha,
        "allocation_verified": True,
        "source_verified": True,
        "shape_verified": True,
    }
    arguments.update(overrides)
    return evaluate_gqa_device_dispatch(**arguments)  # type: ignore[arg-type]


class ChromeTraceParserTests(unittest.TestCase):
    def test_device_kernel_passes_without_fused_aten_child_name(self) -> None:
        events = parse_chrome_cuda_events(trace_bytes(cuda_event(FLASH_NAME)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kernel_family, FLASH_FORWARD_FAMILY)
        self.assertEqual(events[0].classification, "flash_attention")
        self.assertEqual(events[0].stream, 7)
        self.assertEqual(events[0].correlation_id, 43)
        self.assertEqual(events[0].external_id, 9)
        self.assertNotIn("timestamp", events[0].to_dict())
        self.assertNotIn("duration", events[0].to_dict())

    def test_device_events_are_sorted_without_retaining_timestamps(self) -> None:
        events = parse_chrome_cuda_events(
            trace_bytes(
                cuda_event(FLASH_NAME, timestamp=30.0),
                cuda_event("repeat_interleave_cuda_kernel", timestamp=10.0),
            )
        )
        self.assertEqual([event.order for event in events], [0, 1])
        self.assertEqual(events[0].classification, "repeat_materialization")
        self.assertEqual(events[1].classification, "flash_attention")

    def test_missing_cuda_activity_is_empty_not_positive_evidence(self) -> None:
        self.assertEqual(parse_chrome_cuda_events(trace_bytes()), ())

    def test_required_device_identifiers_are_strict(self) -> None:
        for key in ("stream", "correlation", "External id"):
            with self.subTest(key=key):
                raw = trace_bytes(cuda_event(FLASH_NAME, omit=frozenset({key})))
                with self.assertRaises(ChromeTraceValidationError):
                    parse_chrome_cuda_events(raw)

    def test_malformed_trace_inputs_fail_closed(self) -> None:
        malformed = (
            b"",
            b"not-json",
            b"[]",
            b'{"other": []}',
            b'{"traceEvents": [1]}',
            b'{"traceEvents": [], "bad": NaN}',
        )
        for raw in malformed:
            with self.subTest(raw=raw):
                with self.assertRaises(ChromeTraceValidationError):
                    parse_chrome_cuda_events(raw)

    def test_splitkv_main_and_combine_normalize_to_one_family(self) -> None:
        events = parse_chrome_cuda_events(
            trace_bytes(
                cuda_event(SPLIT_NAME, timestamp=20.0),
                cuda_event(SPLIT_COMBINE_NAME, timestamp=30.0),
            )
        )
        self.assertEqual(
            {event.kernel_family for event in events},
            {FLASH_SPLIT_KV_FAMILY},
        )

    def test_scoped_parser_requires_full_marker_correlation_chain(self) -> None:
        marker = "kvbench.phase3.gqa.dispatch_audit"
        scoped = parse_scoped_chrome_cuda_events(
            scoped_trace_bytes(marker, cuda_event(FLASH_NAME)),
            marker=marker,
        )
        self.assertEqual(scoped.scope.marker_external_id, 1)
        self.assertEqual(scoped.scope.sdpa_external_id, 2)
        self.assertEqual(scoped.scope.nested_cpu_external_ids, (2, 10))
        self.assertEqual(scoped.scope.runtime_correlations, (43,))
        self.assertEqual(scoped.scope.gpu_stream, 7)
        self.assertEqual(len(scoped.device_events), 1)
        rendered = scoped.scope.to_dict()
        self.assertNotIn("timestamp", rendered)
        self.assertNotIn("duration", rendered)
        self.assertIs(rendered["timestamps_retained"], False)

    def test_scoped_parser_rejects_duplicate_marker(self) -> None:
        marker = "kvbench.phase3.gqa.dispatch_audit"
        payload = json.loads(
            scoped_trace_bytes(marker, cuda_event(FLASH_NAME)).decode("utf-8")
        )
        payload["traceEvents"].append(dict(payload["traceEvents"][0]))
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_events(
                json.dumps(payload).encode("utf-8"),
                marker=marker,
            )

    def test_scoped_parser_rejects_unmatched_device_correlation(self) -> None:
        marker = "kvbench.phase3.gqa.dispatch_audit"
        payload = json.loads(
            scoped_trace_bytes(marker, cuda_event(FLASH_NAME)).decode("utf-8")
        )
        kernel = next(
            event for event in payload["traceEvents"] if event.get("cat") == "kernel"
        )
        kernel["args"]["correlation"] = 999
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_events(
                json.dumps(payload).encode("utf-8"),
                marker=marker,
            )

    def test_scoped_parser_rejects_overlapping_foreign_device_event(self) -> None:
        marker = "kvbench.phase3.gqa.dispatch_audit"
        payload = json.loads(
            scoped_trace_bytes(marker, cuda_event(FLASH_NAME)).decode("utf-8")
        )
        payload["traceEvents"].append(
            cuda_event(
                "foreign_kernel",
                correlation=999,
                external_id=999,
            )
        )
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_events(
                json.dumps(payload).encode("utf-8"),
                marker=marker,
            )

    def test_memcpy_retains_bytes_and_direction_as_candidate(self) -> None:
        events = parse_chrome_cuda_events(
            trace_bytes(
                cuda_event(
                    "Memcpy DtoD",
                    category="gpu_memcpy",
                    copy_bytes=524_288,
                )
            )
        )
        self.assertEqual(events[0].classification, "device_copy_candidate")
        self.assertIn(events[0].classification, COPY_CANDIDATE_CLASSIFICATIONS)
        self.assertEqual(events[0].copy_bytes, 524_288)
        self.assertEqual(events[0].copy_direction, "DtoD")


class ByteAndShapeProofTests(unittest.TestCase):
    def test_exact_gqa_and_mha_byte_formulas(self) -> None:
        gqa = calculate_kv_bytes(
            batch_size=1,
            num_query_heads=32,
            num_kv_heads=8,
            context_length=128,
            head_dim=128,
            dtype_bytes=2,
        )
        self.assertEqual(gqa.native_kv_bytes, 524_288)
        self.assertEqual(gqa.expanded_kv_bytes, 2_097_152)
        mha = calculate_kv_bytes(
            batch_size=1,
            num_query_heads=32,
            num_kv_heads=32,
            context_length=128,
            head_dim=128,
            dtype_bytes=2,
        )
        self.assertEqual(mha.native_kv_bytes, 2_097_152)
        self.assertEqual(mha.expanded_kv_bytes, 2_097_152)
        long_gqa = calculate_kv_bytes(
            batch_size=1,
            num_query_heads=32,
            num_kv_heads=8,
            context_length=4096,
            head_dim=128,
            dtype_bytes=2,
        )
        self.assertEqual(long_gqa.expanded_kv_bytes, 67_108_864)

    def test_source_and_shape_evidence_is_exact(self) -> None:
        gqa = control(
            "gqa",
            parse_chrome_cuda_events(trace_bytes(cuda_event(FLASH_NAME))),
        )
        query = TensorShapeEvidence(
            shape=(1, 32, 1, 128),
            stride=(4096, 128, 128, 1),
            dtype="torch.bfloat16",
            device="cuda:0",
            element_size=2,
            storage_bytes=8_192,
            storage_offset=0,
            is_contiguous=True,
        )
        key = TensorShapeEvidence(
            shape=(1, 8, 128, 128),
            stride=(131072, 16384, 128, 1),
            dtype="torch.bfloat16",
            device="cuda:0",
            element_size=2,
            storage_bytes=262_144,
            storage_offset=0,
            is_contiguous=True,
        )
        evidence = SourceShapeEvidence(
            sources=(
                SourceFileEvidence("src/backend.py", "c" * 64, ()),
                SourceFileEvidence("src/static_cache.py", "d" * 64, ()),
            ),
            query=query,
            key=key,
            value=key,
            output=query,
            native_kv_storage_verified=True,
        )
        self.assertTrue(evidence.source_verified)
        self.assertTrue(evidence.shape_verified_for(gqa))
        self.assertEqual(key.logical_bytes, 262_144)

    def test_backend_identity_is_canonical_and_checksum_bound(self) -> None:
        identity = BackendIdentityEvidence.from_payload(
            {"torch": "2.12.1+cu130", "sources": [{"sha256": "a" * 64}]}
        )
        self.assertEqual(
            identity.sha256,
            hashlib.sha256(identity.canonical_json.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            identity.to_dict()["manifest"],
            {"sources": [{"sha256": "a" * 64}], "torch": "2.12.1+cu130"},
        )
        with self.assertRaises(ValueError):
            BackendIdentityEvidence(identity.canonical_json, "0" * 64)

    def test_source_audit_hashes_selected_files_and_fails_on_replication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kvbench-source-unit-", dir="/tmp") as root:
            root_path = Path(root)
            clean = root_path / "clean.py"
            forbidden = root_path / "forbidden.py"
            clean.write_text("def attention():\n    return 1\n", encoding="utf-8")
            forbidden.write_text(
                "def attention(x):\n    return x.repeat(4, 1)\n",
                encoding="utf-8",
            )
            evidence = audit_gqa_source_files(
                root_path,
                (forbidden, clean),
            )
        self.assertEqual(
            tuple(item.relative_path for item in evidence),
            ("clean.py", "forbidden.py"),
        )
        self.assertTrue(evidence[0].passed)
        self.assertEqual(evidence[1].findings, ("tensor_repeat",))


class DeviceDispatchVerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        flash = parse_chrome_cuda_events(trace_bytes(cuda_event(FLASH_NAME)))
        self.gqa = control("gqa", flash)
        self.mha = control("mha_control", flash)

    def test_complete_device_proof_is_verified(self) -> None:
        result = complete_evaluation(self.gqa, self.mha)
        self.assertEqual(result.verdict, GQAVerdict.NONMATERIALIZATION_VERIFIED)
        self.assertTrue(result.dispatch_verified)
        self.assertTrue(result.no_replication_kernel_verified)
        self.assertEqual(result.family_comparison.relation, "same")

    def test_missing_cuda_kernel_is_dispatch_unverified(self) -> None:
        missing = control("gqa", parse_chrome_cuda_events(trace_bytes()))
        result = complete_evaluation(missing, self.mha)
        self.assertEqual(result.verdict, GQAVerdict.DISPATCH_UNVERIFIED)
        self.assertFalse(result.positive_materialization_evidence)

    def test_missing_raw_trace_is_dispatch_unverified(self) -> None:
        missing = control("gqa", self.gqa.device_events, with_raw_trace=False)
        result = complete_evaluation(missing, self.mha)
        self.assertEqual(result.verdict, GQAVerdict.DISPATCH_UNVERIFIED)

    def test_preceding_repeat_and_expand_are_positive_materialization(self) -> None:
        names = (
            ("repeat_interleave_cuda_kernel", "kernel"),
            ("expand_copy_cuda_kernel", "kernel"),
        )
        for name, category in names:
            with self.subTest(name=name):
                events = parse_chrome_cuda_events(
                    trace_bytes(
                        cuda_event(name, category=category, timestamp=10.0),
                        cuda_event(FLASH_NAME, timestamp=20.0),
                    )
                )
                result = complete_evaluation(control("gqa", events), self.mha)
                self.assertEqual(
                    result.verdict,
                    GQAVerdict.MATERIALIZATION_DETECTED,
                )
                self.assertTrue(result.positive_materialization_evidence)

    def test_generic_preceding_copy_is_unproven_not_materialization(self) -> None:
        for name, category in (
            ("Memcpy DtoD", "gpu_memcpy"),
            ("transpose_copy_cuda_kernel", "kernel"),
        ):
            with self.subTest(name=name):
                events = parse_chrome_cuda_events(
                    trace_bytes(
                        cuda_event(
                            name,
                            category=category,
                            timestamp=10.0,
                            copy_bytes=524_288 if category == "gpu_memcpy" else None,
                        ),
                        cuda_event(FLASH_NAME, timestamp=20.0),
                    )
                )
                result = complete_evaluation(control("gqa", events), self.mha)
                self.assertEqual(
                    result.verdict,
                    GQAVerdict.NONMATERIALIZATION_UNPROVEN,
                )
                self.assertFalse(result.positive_materialization_evidence)
                self.assertFalse(result.no_replication_kernel_verified)

    def test_preceding_expanded_size_copy_is_positive_evidence(self) -> None:
        events = parse_chrome_cuda_events(
            trace_bytes(
                cuda_event(
                    "Memcpy DtoD",
                    category="gpu_memcpy",
                    timestamp=10.0,
                    copy_bytes=1_048_576,
                ),
                cuda_event(FLASH_NAME, timestamp=20.0),
            )
        )
        result = complete_evaluation(control("gqa", events), self.mha)
        self.assertEqual(result.verdict, GQAVerdict.MATERIALIZATION_DETECTED)
        self.assertTrue(result.positive_materialization_evidence)

    def test_post_attention_copy_is_not_positive_replication_evidence(self) -> None:
        events = parse_chrome_cuda_events(
            trace_bytes(
                cuda_event(FLASH_NAME, timestamp=10.0),
                cuda_event(
                    "Memcpy DtoD",
                    category="gpu_memcpy",
                    timestamp=20.0,
                    copy_bytes=1_048_576,
                ),
            )
        )
        result = complete_evaluation(control("gqa", events), self.mha)
        self.assertEqual(
            result.verdict,
            GQAVerdict.NONMATERIALIZATION_UNPROVEN,
        )
        self.assertFalse(result.positive_materialization_evidence)

    def test_expanded_tensor_or_allocation_is_positive_evidence(self) -> None:
        for field in (
            "expanded_kv_allocation_detected",
            "expanded_kv_tensor_detected",
        ):
            with self.subTest(field=field):
                result = complete_evaluation(
                    self.gqa,
                    self.mha,
                    **{field: True},
                )
                self.assertEqual(
                    result.verdict,
                    GQAVerdict.MATERIALIZATION_DETECTED,
                )

    def test_incomplete_non_dispatch_component_is_unproven(self) -> None:
        for field in ("allocation_verified", "source_verified", "shape_verified"):
            with self.subTest(field=field):
                result = complete_evaluation(
                    self.gqa,
                    self.mha,
                    **{field: False},
                )
                self.assertEqual(
                    result.verdict,
                    GQAVerdict.NONMATERIALIZATION_UNPROVEN,
                )

    def test_unknown_preceding_kernel_is_unproven_not_materialization(self) -> None:
        events = parse_chrome_cuda_events(
            trace_bytes(
                cuda_event("opaque_workspace_kernel", timestamp=10.0),
                cuda_event(FLASH_NAME, timestamp=20.0),
            )
        )
        result = complete_evaluation(control("gqa", events), self.mha)
        self.assertEqual(
            result.verdict,
            GQAVerdict.NONMATERIALIZATION_UNPROVEN,
        )
        self.assertFalse(result.positive_materialization_evidence)

    def test_unrelated_family_fails_dispatch_proof(self) -> None:
        split = parse_chrome_cuda_events(trace_bytes(cuda_event(SPLIT_NAME)))
        split_mha = control("mha_control", split)
        comparison = compare_kernel_families(self.gqa, split_mha)
        self.assertEqual(comparison.relation, "unrelated")
        result = complete_evaluation(self.gqa, split_mha)
        self.assertEqual(result.verdict, GQAVerdict.DISPATCH_UNVERIFIED)

    def test_related_family_requires_an_explicit_mapping(self) -> None:
        split = parse_chrome_cuda_events(trace_bytes(cuda_event(SPLIT_NAME)))
        split_mha = control("mha_control", split)
        comparison = compare_kernel_families(
            self.gqa,
            split_mha,
            explicitly_related={(FLASH_FORWARD_FAMILY, FLASH_SPLIT_KV_FAMILY)},
        )
        self.assertEqual(comparison.relation, "related")
        self.assertTrue(comparison.passed)

    def test_backend_forcing_and_held_constants_fail_closed(self) -> None:
        rejected_backend = BackendControlEvidence(
            enabled_backends=("FLASH_ATTENTION", "MATH"),
            flash_eligible=True,
            fused_backend_name="FLASH_ATTENTION",
            rejected_control_failed=True,
            rejected_control_error="No available kernel. Aborting execution.",
            rejected_control_warnings=(
                "Flash attention kernel not used because:",
                "Expected query, key and value to all be of dtype: {Half, BFloat16}.",
            ),
            rejected_control_synchronized=True,
            source_build_fingerprint="b" * 64,
            source_build_verified=True,
        )
        bad_backend = control(
            "gqa",
            self.gqa.device_events,
            backend=rejected_backend,
        )
        self.assertEqual(
            complete_evaluation(bad_backend, self.mha).verdict,
            GQAVerdict.DISPATCH_UNVERIFIED,
        )
        mismatched_mha = control(
            "mha_control",
            self.mha.device_events,
            context_length=129,
        )
        self.assertEqual(
            complete_evaluation(self.gqa, mismatched_mha).verdict,
            GQAVerdict.DISPATCH_UNVERIFIED,
        )
        mismatched_build = BackendControlEvidence(
            enabled_backends=("FLASH_ATTENTION",),
            flash_eligible=True,
            fused_backend_name="FLASH_ATTENTION",
            rejected_control_failed=True,
            rejected_control_error="No available kernel. Aborting execution.",
            rejected_control_warnings=(
                "Flash attention kernel not used because:",
                "Expected query, key and value to all be of dtype: {Half, BFloat16}.",
            ),
            rejected_control_synchronized=True,
            source_build_fingerprint="e" * 64,
            source_build_verified=True,
        )
        mismatched_build_mha = control(
            "mha_control",
            self.mha.device_events,
            backend=mismatched_build,
        )
        self.assertEqual(
            complete_evaluation(self.gqa, mismatched_build_mha).verdict,
            GQAVerdict.DISPATCH_UNVERIFIED,
        )

    def test_evidence_contains_no_profiler_timing_metrics(self) -> None:
        payload = complete_evaluation(self.gqa, self.mha).to_dict()

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value).union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        observed = keys(payload)
        self.assertFalse(
            observed.intersection(
                {"ts", "dur", "duration", "latency", "wall_time_ms"}
            )
        )
        self.assertIs(payload["performance_timing_reported"], False)


class RejectedBackendControlTests(unittest.TestCase):
    @staticmethod
    def fake_torch(error_message: str, synchronize: mock.Mock):
        def rejected_attention(*args: object, **kwargs: object) -> None:
            del args, kwargs
            warnings.warn("Flash attention kernel not used because:")
            warnings.warn(
                "Expected query, key and value to all be of dtype: "
                "{Half, BFloat16}. Got double."
            )
            raise RuntimeError(error_message)

        return SimpleNamespace(
            float64="float64",
            empty=mock.Mock(return_value=object()),
            empty_like=mock.Mock(return_value=object()),
            cuda=SimpleNamespace(synchronize=synchronize),
            nn=SimpleNamespace(
                functional=SimpleNamespace(
                    scaled_dot_product_attention=rejected_attention
                )
            ),
        )

    def test_expected_dtype_rejection_is_synchronized_and_verified(self) -> None:
        synchronize = mock.Mock()
        result = _collect_rejected_backend_control(
            self.fake_torch(
                "No available kernel. Aborting execution.",
                synchronize,
            ),
            device="cuda:0",
            num_kv_heads=8,
            is_causal=False,
            scale=128**-0.5,
            forced_flash_execution=nullcontext,
        )
        self.assertTrue(result.passed)
        self.assertTrue(result.synchronized)
        self.assertEqual(synchronize.call_count, 2)

    def test_arbitrary_runtime_error_cannot_pass_rejected_control(self) -> None:
        synchronize = mock.Mock()
        result = _collect_rejected_backend_control(
            self.fake_torch("CUDA out of memory", synchronize),
            device="cuda:0",
            num_kv_heads=8,
            is_causal=False,
            scale=128**-0.5,
            forced_flash_execution=nullcontext,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error, "CUDA out of memory")
        self.assertEqual(synchronize.call_count, 2)

    def test_post_control_async_error_fails_closed(self) -> None:
        synchronize = mock.Mock(
            side_effect=[None, RuntimeError("stale asynchronous failure")]
        )
        with self.assertRaises(GQADeviceDispatchError):
            _collect_rejected_backend_control(
                self.fake_torch(
                    "No available kernel. Aborting execution.",
                    synchronize,
                ),
                device="cuda:0",
                num_kv_heads=8,
                is_causal=False,
                scale=128**-0.5,
                forced_flash_execution=nullcontext,
            )


class TraceCollectionHelperTests(unittest.TestCase):
    def test_collection_exports_untouched_trace_without_duration_result(self) -> None:
        raw = trace_bytes(cuda_event(FLASH_NAME))
        calls: list[str] = []

        class FakeTrace:
            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def export_chrome_trace(self, path: str) -> None:
                Path(path).write_bytes(raw)

        fake_profile = mock.Mock(return_value=FakeTrace())
        fake_profiler = SimpleNamespace(
            profile=fake_profile,
            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
        )
        synchronize = mock.Mock()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=synchronize),
            autograd=SimpleNamespace(
                profiler=SimpleNamespace(
                    record_function=lambda marker: nullcontext(marker)
                )
            ),
        )

        def import_module(name: str):
            return fake_torch if name == "torch" else fake_profiler

        def operation() -> object:
            calls.append("called")
            return object()

        with tempfile.TemporaryDirectory(prefix="kvbench-gqa-unit-", dir="/tmp") as root:
            path = Path(root) / "trace.json"
            with mock.patch(
                "kvbench.runtime.gqa_device_dispatch.importlib.import_module",
                side_effect=import_module,
            ):
                artifact = collect_torch_profiler_trace(
                    operation,
                    path,
                    artifact_relative_path="gqa/traces/gqa.chrome.json",
                    marker="kvbench.gqa.dispatch",
                    warmup_count=3,
                    device="cuda:0",
                )
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(len(calls), 4)
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(artifact.sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(artifact.size_bytes, len(raw))
        metadata = artifact.to_dict()
        self.assertEqual(metadata["run_kind"], "dispatch_audit")
        self.assertIs(metadata["benchmark_timing_eligible"], False)
        self.assertNotIn("duration", metadata)
        fake_profile.assert_called_once()

    def test_dangling_symlink_destination_is_rejected_before_collection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kvbench-gqa-unit-", dir="/tmp") as root:
            path = Path(root) / "trace.json"
            path.symlink_to(Path(root) / "absent.json")
            with self.assertRaises(GQADeviceDispatchError):
                collect_torch_profiler_trace(
                    lambda: object(),
                    path,
                    artifact_relative_path="gqa/traces/gqa.chrome.json",
                    marker="kvbench.gqa.dispatch",
                    warmup_count=1,
                    device="cuda:0",
                )
            self.assertTrue(os.path.lexists(path))
            self.assertTrue(path.is_symlink())

    def test_destination_race_cannot_replace_competing_file(self) -> None:
        raw = trace_bytes(cuda_event(FLASH_NAME))
        competitor = b"competitor"
        with tempfile.TemporaryDirectory(prefix="kvbench-gqa-unit-", dir="/tmp") as root:
            path = Path(root) / "trace.json"

            class FakeTrace:
                def __enter__(self):
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

                def export_chrome_trace(self, staging: str) -> None:
                    Path(staging).write_bytes(raw)
                    path.write_bytes(competitor)

            fake_profiler = SimpleNamespace(
                profile=mock.Mock(return_value=FakeTrace()),
                ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
            )
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(synchronize=mock.Mock()),
                autograd=SimpleNamespace(
                    profiler=SimpleNamespace(
                        record_function=lambda marker: nullcontext(marker)
                    )
                ),
            )

            def import_module(name: str):
                return fake_torch if name == "torch" else fake_profiler

            with mock.patch(
                "kvbench.runtime.gqa_device_dispatch.importlib.import_module",
                side_effect=import_module,
            ):
                with self.assertRaises(GQADeviceDispatchError):
                    collect_torch_profiler_trace(
                        lambda: object(),
                        path,
                        artifact_relative_path="gqa/traces/gqa.chrome.json",
                        marker="kvbench.gqa.dispatch",
                        warmup_count=1,
                        device="cuda:0",
                    )
            self.assertEqual(path.read_bytes(), competitor)

    def test_profiler_cannot_replace_reserved_staging_directory(self) -> None:
        raw = trace_bytes(cuda_event(FLASH_NAME))
        with tempfile.TemporaryDirectory(prefix="kvbench-gqa-unit-", dir="/tmp") as root:
            path = Path(root) / "trace.json"

            class ReplacingTrace:
                def __enter__(self):
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

                def export_chrome_trace(self, staging: str) -> None:
                    staging_path = Path(staging)
                    staging_directory = staging_path.parent
                    reserved_backup = staging_directory.with_name(
                        staging_directory.name + ".reserved"
                    )
                    staging_directory.replace(reserved_backup)
                    staging_directory.mkdir(mode=0o700)
                    staging_path.write_bytes(raw)

            fake_profiler = SimpleNamespace(
                profile=mock.Mock(return_value=ReplacingTrace()),
                ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
            )
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(synchronize=mock.Mock()),
                autograd=SimpleNamespace(
                    profiler=SimpleNamespace(
                        record_function=lambda marker: nullcontext(marker)
                    )
                ),
            )

            def import_module(name: str):
                return fake_torch if name == "torch" else fake_profiler

            with mock.patch(
                "kvbench.runtime.gqa_device_dispatch.importlib.import_module",
                side_effect=import_module,
            ):
                with self.assertRaises(GQADeviceDispatchError):
                    collect_torch_profiler_trace(
                        lambda: object(),
                        path,
                        artifact_relative_path="gqa/traces/gqa.chrome.json",
                        marker="kvbench.gqa.dispatch",
                        warmup_count=1,
                        device="cuda:0",
                    )
            self.assertFalse(os.path.lexists(path))


if __name__ == "__main__":
    unittest.main()
