"""CPU-only adversarial tests for production-geometry B-011 evidence."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kvbench.errors import SchemaValidationError
from kvbench.runtime.gqa_device_dispatch import (
    CUDA_GRAPH_REPLAY_EXECUTION_MODE,
    EAGER_EXECUTION_MODE,
    FLASH_FORWARD_FAMILY,
    FLASH_SPLIT_KV_FAMILY,
    PHASE3_CACHE_LAYOUT_NAME,
    REQUIRED_SUT_SOURCES,
    BackendControlEvidence,
    BackendIdentityEvidence,
    CUDADeviceEvent,
    ChromeTraceValidationError,
    CUDAGraphTraceScopeEvidence,
    DispatchControlEvidence,
    GQADeviceDispatchError,
    GeometryBoundSourceShapeEvidence,
    Phase3AllocationJoinFacts,
    Phase3AllocationRawEvidence,
    Phase3GeometryBoundGQADeviceDispatchAudit,
    Phase3DispatchPointBinding,
    RawTraceArtifact,
    SourceFileEvidence,
    StaticCacheLayoutEvidence,
    StaticCacheViewBindingEvidence,
    TensorShapeEvidence,
    TraceScopeEvidence,
    analyze_flash_kernel_sequence,
    canonical_phase3_geometry_bound_dispatch_audit_bytes,
    combine_phase3_geometry_bound_gqa_allocation_verdict,
    compare_geometry_bound_kernel_sequences,
    collect_phase3_geometry_bound_gqa_mha_device_dispatch,
    collect_torch_profiler_trace,
    evaluate_geometry_bound_gqa_device_dispatch,
    parse_scoped_chrome_cuda_graph_events,
    parse_scoped_chrome_cuda_events,
    parse_chrome_cuda_events,
    parse_phase3_geometry_bound_dispatch_observation_bytes,
    phase3_source_identity_sha256,
    phase3_geometry_bound_dispatch_evidence_bytes,
    phase3_geometry_bound_dispatch_observation_bytes,
    revalidate_geometry_bound_raw_traces,
    revalidate_phase3_geometry_bound_dispatch_audit,
    revalidate_phase3_geometry_bound_dispatch_audit_from_raw,
    revalidate_phase3_geometry_bound_dispatch_evidence_from_raw,
    source_file_evidence_from_bytes,
)
from kvbench.runtime import allocation_attribution as allocation_module
from kvbench.runtime.allocation_attribution import (
    PHASE3_BACKEND_IDENTITY,
    ProductionAllocationBinding,
    SplitKCompositeRawInputs,
    build_phase3_production_allocation_binding,
    collect_cuda_allocation_attribution,
    cuda_allocator_rounded_minimum,
    instantiate_decision_0009_production_rules,
)
from kvbench.runtime.phase3_allocator_controls import (
    verify_phase3_paired_allocator_controls,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.schema import (
    GQAVerdict,
    GraphMode,
    RunnerKind,
    derive_cache_layout_fingerprint,
)
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)
from tests.unit.test_allocation_attribution import (
    _FakeOperationHarness,
    _FakeTorch,
    alloc_lifetime,
)


FLASH_NAME = (
    "void pytorch_flash::flash_fwd_kernel<traits>("
    "pytorch_flash::Flash_fwd_params)"
)
SPLIT_NAME = "void pytorch_flash::flash_fwd_splitkv_kernel<traits>(params)"
SPLIT_COMBINE_NAME = (
    "void pytorch_flash::flash_fwd_splitkv_combine_kernel<traits>(params)"
)
STATIC_CACHE_SHA256 = "c" * 64
SELECTED_SOURCE_FUNCTIONS = {
    REQUIRED_SUT_SOURCES[0]: ("flash_attention_forward",),
    REQUIRED_SUT_SOURCES[1]: (
        "BF16DecodeEndpoint._attention",
        "BF16DecodeEndpoint._base_forward",
        "BF16DecodeEndpoint.decode",
    ),
    REQUIRED_SUT_SOURCES[2]: ("BF16StaticCache.update",),
}
SOURCE_IDENTITY_SHA256 = phase3_source_identity_sha256(
    {
        REQUIRED_SUT_SOURCES[0]: "d" * 64,
        REQUIRED_SUT_SOURCES[1]: "d" * 64,
        REQUIRED_SUT_SOURCES[2]: STATIC_CACHE_SHA256,
    }
)


def operation_key(
    *,
    graph_mode: GraphMode = GraphMode.EAGER,
    run_id: str = "phase3-remediation-dispatch-fixture",
    cache_layout_fingerprint: str | None = None,
    backend_identity_sha256: str = "b" * 64,
    source_identity_sha256: str = SOURCE_IDENTITY_SHA256,
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
    return Phase3AuditOperationKey.from_point(
        run_id=run_id,
        point=point,
        decode_step=0,
        cache_layout_fingerprint=(
            layout().layout_fingerprint
            if cache_layout_fingerprint is None
            else cache_layout_fingerprint
        ),
        execution_git_sha="5" * 40,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[PHASE3_FIXED_PLAN_PATH],
        hardware_identity_sha256="6" * 64,
        software_identity_sha256="7" * 64,
        model_identity_sha256="8" * 64,
        backend_identity_sha256=backend_identity_sha256,
        source_identity_sha256=source_identity_sha256,
    )


def growing_operation_key(*, decode_step: int) -> Phase3AuditOperationKey:
    point = Phase3ProcessPoint(
        point_id="growing_context-b1-l128-eager-r1",
        runner_kind=RunnerKind.GROWING_CONTEXT,
        graph_mode=GraphMode.EAGER,
        batch_size=1,
        context_length=128,
        output_steps=16,
        process_replicate=1,
        stability_member=False,
    )
    return Phase3AuditOperationKey.from_point(
        run_id="phase3-remediation-dispatch-growing-fixture",
        point=point,
        decode_step=decode_step,
        cache_layout_fingerprint=layout(capacity=144).layout_fingerprint,
        execution_git_sha="5" * 40,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[PHASE3_GROWING_PLAN_PATH],
        hardware_identity_sha256="6" * 64,
        software_identity_sha256="7" * 64,
        model_identity_sha256="8" * 64,
        backend_identity_sha256="b" * 64,
        source_identity_sha256=SOURCE_IDENTITY_SHA256,
    )


def dispatch_point(
    *, graph_mode: GraphMode = GraphMode.EAGER
) -> Phase3DispatchPointBinding:
    return Phase3DispatchPointBinding.create(
        operation_key=operation_key(graph_mode=graph_mode)
    )


def tensor(
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    storage_bytes: int,
    storage_offset: int = 0,
    contiguous: bool = True,
) -> TensorShapeEvidence:
    return TensorShapeEvidence(
        shape=shape,
        stride=stride,
        dtype="torch.bfloat16",
        device="cuda:0",
        element_size=2,
        storage_bytes=storage_bytes,
        storage_offset=storage_offset,
        is_contiguous=contiguous,
    )


def layout(
    *,
    batch: int = 1,
    capacity: int = 129,
    implementation_sha256: str = STATIC_CACHE_SHA256,
    workspace_bytes: int = 327_680,
) -> StaticCacheLayoutEvidence:
    fingerprint = derive_cache_layout_fingerprint(
        num_layers=32,
        batch_size=batch,
        num_kv_heads=8,
        capacity=capacity,
        head_dim=128,
        device="cuda:0",
        workspace_bytes=workspace_bytes,
        implementation_sha256=implementation_sha256,
    )
    return StaticCacheLayoutEvidence.create(
        batch_size=batch,
        capacity=capacity,
        device="cuda:0",
        workspace_bytes=workspace_bytes,
        implementation_sha256=implementation_sha256,
        layout_fingerprint=fingerprint,
    )


def cache_binding(
    *,
    active_context: int = 129,
    capacity: int = 129,
    layer_index: int = 0,
    implementation_sha256: str = STATIC_CACHE_SHA256,
    workspace_bytes: int = 327_680,
) -> StaticCacheViewBindingEvidence:
    declaration = layout(
        capacity=capacity,
        implementation_sha256=implementation_sha256,
        workspace_bytes=workspace_bytes,
    )
    storage_bytes = declaration.single_tensor_storage_bytes
    backing = tensor(
        shape=declaration.tensor_shape,
        stride=declaration.tensor_stride,
        storage_bytes=storage_bytes,
    )
    view = tensor(
        shape=(1, 8, active_context, 128),
        stride=declaration.tensor_stride[1:],
        storage_bytes=storage_bytes,
        storage_offset=layer_index * declaration.tensor_stride[0],
        contiguous=active_context == capacity,
    )
    return StaticCacheViewBindingEvidence(
        layout=declaration,
        layer_index=layer_index,
        active_context=active_context,
        key_backing=backing,
        value_backing=backing,
        key_view=view,
        value_view=view,
        key_backing_storage_ptr=0x1000,
        value_backing_storage_ptr=0x2000,
        key_view_storage_ptr=0x1000,
        value_view_storage_ptr=0x2000,
        key_view_shares_backing_storage=True,
        value_view_shares_backing_storage=True,
        key_value_backing_storages_distinct=True,
    )


def device_event(
    *,
    order: int,
    name: str,
    classification: str,
    family: str | None,
    category: str = "kernel",
    correlation: int | None = None,
    external_id: int | None = None,
    copy_bytes: int | None = None,
    memory_bytes: int | None = None,
    memory_role: str | None = None,
    graph_id: int | None = None,
    graph_node_id: int | None = None,
) -> CUDADeviceEvent:
    return CUDADeviceEvent(
        order=order,
        category=category,
        name=name,
        stream=7,
        correlation_id=40 + order if correlation is None else correlation,
        external_id=10 + order if external_id is None else external_id,
        device=0,
        context=1,
        classification=classification,
        kernel_family=family,
        copy_bytes=copy_bytes,
        copy_direction="DtoD" if copy_bytes is not None else None,
        memory_bytes=memory_bytes,
        memory_role=memory_role,
        graph_id=graph_id,
        graph_node_id=graph_node_id,
    )


def flash_event(order: int = 0) -> CUDADeviceEvent:
    return device_event(
        order=order,
        name=FLASH_NAME,
        classification="flash_attention",
        family=FLASH_FORWARD_FAMILY,
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
        eligibility_diagnostics=(
            "enabled_backends=FLASH_ATTENTION",
            "can_use_flash_attention=True",
            "fused_sdp_choice=1",
        ),
    )


def control(
    role: str,
    events: tuple[CUDADeviceEvent, ...],
    *,
    active_context: int = 129,
    execution_mode: str = EAGER_EXECUTION_MODE,
    marker: str | None = None,
) -> DispatchControlEvidence:
    actual_marker = f"fixture.{role}" if marker is None else marker
    if execution_mode == EAGER_EXECUTION_MODE:
        external_ids = tuple(
            sorted(
                {
                    2,
                    *(
                        event.external_id
                        for event in events
                        if event.external_id is not None
                    ),
                }
            )
        )
        correlations = tuple(sorted({event.correlation_id for event in events}))
        scope: TraceScopeEvidence | CUDAGraphTraceScopeEvidence = TraceScopeEvidence(
            marker=actual_marker,
            marker_external_id=1,
            cpu_process_id=101,
            cpu_thread_id=101,
            sdpa_external_id=2,
            nested_cpu_external_ids=external_ids,
            runtime_correlations=correlations,
            gpu_stream=7,
        )
    else:
        graph_correlation = events[0].correlation_id
        graph_external_id = events[0].external_id
        events = tuple(
            replace(
                event,
                correlation_id=graph_correlation,
                external_id=graph_external_id,
                graph_id=2,
                graph_node_id=0x200000000 + event.order,
            )
            for event in events
        )
        correlations = (graph_correlation,)
        scope = CUDAGraphTraceScopeEvidence(
            marker=actual_marker,
            marker_external_id=1,
            cpu_process_id=101,
            cpu_thread_id=101,
            graph_launch_external_id=graph_external_id,
            runtime_correlations=correlations,
            gpu_stream=7,
            graph_id=2,
            graph_node_ids=tuple(
                event.graph_node_id
                for event in events
                if event.graph_node_id is not None
            ),
        )
    return DispatchControlEvidence(
        role=role,
        batch_size=1,
        context_length=active_context,
        query_length=1,
        num_query_heads=32,
        num_kv_heads=8 if role == "gqa" else 32,
        head_dim=128,
        dtype="torch.bfloat16",
        dtype_bytes=2,
        is_causal=False,
        warmup_count=3,
        backend=passing_backend(),
        raw_trace=RawTraceArtifact(
            relative_path=f"dispatch/{role}.geometry.chrome.json",
            sha256="a" * 64,
            size_bytes=1024,
            execution_mode=execution_mode,
        ),
        trace_scope=scope,
        device_events=events,
        execution_mode=execution_mode,
    )


def source_shape(
    binding: StaticCacheViewBindingEvidence,
    *,
    source_findings: tuple[str, ...] = (),
    sources: tuple[SourceFileEvidence, ...] | None = None,
) -> GeometryBoundSourceShapeEvidence:
    query = tensor(
        shape=(1, 32, 1, 128),
        stride=(4096, 128, 128, 1),
        storage_bytes=8192,
    )
    mha_bytes = 1 * 32 * binding.active_context * 128 * 2
    mha_kv = tensor(
        shape=(1, 32, binding.active_context, 128),
        stride=(
            32 * binding.active_context * 128,
            binding.active_context * 128,
            128,
            1,
        ),
        storage_bytes=mha_bytes,
    )
    if sources is None:
        sources = tuple(
            SourceFileEvidence(
                relative_path=path,
                sha256=(
                    binding.layout.implementation_sha256
                    if path == REQUIRED_SUT_SOURCES[2]
                    else "d" * 64
                ),
                findings=source_findings if path == REQUIRED_SUT_SOURCES[0] else (),
                direct_replication_findings=(
                    source_findings if path == REQUIRED_SUT_SOURCES[0] else ()
                ),
                selected_function_paths=SELECTED_SOURCE_FUNCTIONS[path],
            )
            for path in REQUIRED_SUT_SOURCES
        )
    return GeometryBoundSourceShapeEvidence(
        sources=sources,
        gqa_query=query,
        gqa_output=query,
        cache=binding,
        mha_query=query,
        mha_key=mha_kv,
        mha_value=mha_kv,
        mha_output=query,
    )


def evaluation(
    gqa_events: tuple[CUDADeviceEvent, ...],
    mha_events: tuple[CUDADeviceEvent, ...] = (flash_event(),),
    *,
    source_findings: tuple[str, ...] = (),
):
    point = dispatch_point()
    gqa = control("gqa", gqa_events)
    mha = control("mha_control", mha_events)
    evidence = source_shape(
        cache_binding(),
        source_findings=source_findings,
    )
    return evaluate_geometry_bound_gqa_device_dispatch(
        point=point,
        gqa=gqa,
        mha=mha,
        gqa_sequence=analyze_flash_kernel_sequence(gqa_events),
        mha_sequence=analyze_flash_kernel_sequence(mha_events),
        source_shape=evidence,
    )


def mismatched_runtime_trace() -> bytes:
    marker = "kvbench.phase3.geometry.fixture"
    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": marker,
            "pid": 101,
            "tid": 101,
            "ts": 0.0,
            "dur": 100.0,
            "args": {"External id": 1},
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::scaled_dot_product_attention",
            "pid": 101,
            "tid": 101,
            "ts": 5.0,
            "dur": 80.0,
            "args": {"External id": 2},
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::_scaled_dot_product_flash_attention",
            "pid": 101,
            "tid": 101,
            "ts": 10.0,
            "dur": 60.0,
            "args": {"External id": 10},
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaMemcpyAsync",
            "pid": 101,
            "tid": 101,
            "ts": 15.0,
            "dur": 1.0,
            "args": {"External id": 10, "correlation": 43},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": FLASH_NAME,
            "ts": 20.0,
            "dur": 1.0,
            "args": {
                "stream": 7,
                "correlation": 43,
                "External id": 10,
                "device": 0,
                "context": 1,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": marker,
            "pid": 0,
            "tid": 7,
            "ts": 19.0,
            "dur": 3.0,
            "args": {"External id": 1},
        },
    ]
    return json.dumps({"schemaVersion": 1, "traceEvents": events}).encode()


def eager_launch_trace(marker: str) -> bytes:
    payload = json.loads(mismatched_runtime_trace().decode("utf-8"))
    for event in payload["traceEvents"]:
        if event.get("cat") in {"user_annotation", "gpu_user_annotation"}:
            event["name"] = marker
        if event.get("cat") == "cuda_runtime":
            event["name"] = "cudaLaunchKernel"
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def graph_launch_trace(marker: str, *, kernel_name: str = FLASH_NAME) -> bytes:
    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": marker,
            "pid": 101,
            "tid": 101,
            "ts": 0.0,
            "dur": 100.0,
            "args": {"External id": 1},
        },
        {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaGraphLaunch",
            "pid": 101,
            "tid": 101,
            "ts": 10.0,
            "dur": 1.0,
            "args": {"correlation": 43},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": kernel_name,
            "ts": 20.0,
            "dur": 1.0,
            "args": {
                "stream": 7,
                "correlation": 43,
                "device": 0,
                "context": 1,
                "graph id": 2,
                "graph node id": 8589934592,
            },
        },
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": marker,
            "pid": 0,
            "tid": 7,
            "ts": 19.0,
            "dur": 3.0,
            "args": {"External id": 1},
        },
    ]
    return json.dumps(
        {"schemaVersion": 1, "traceEvents": events},
        sort_keys=True,
    ).encode("utf-8")


def source_bundle() -> dict[str, bytes]:
    return {
        REQUIRED_SUT_SOURCES[0]: (
            b"def flash_attention_forward(query, key, value):\n"
            b"    return query\n"
        ),
        REQUIRED_SUT_SOURCES[1]: (
            b"class BF16DecodeEndpoint:\n"
            b"    def _attention(self):\n"
            b"        return 1\n"
            b"    def _base_forward(self):\n"
            b"        return self._attention()\n"
            b"    def decode(self):\n"
            b"        return self._base_forward()\n"
        ),
        REQUIRED_SUT_SOURCES[2]: (
            b"class BF16StaticCache:\n"
            b"    def update(self):\n"
            b"        return 1\n"
        ),
    }


def raw_replay_fixture(
    *,
    production_allocation_identity: bool = False,
) -> dict[str, object]:
    if production_allocation_identity:
        repository_root = Path(__file__).resolve().parents[2]
        raw_sources = {
            path: (repository_root / path).read_bytes()
            for path in REQUIRED_SUT_SOURCES
        }
        workspace_bytes = 32 * 1 * (32 + 8) * 1 * 64 * 2
        identity = BackendIdentityEvidence.from_payload(
            json.loads(PHASE3_BACKEND_IDENTITY)
        )
    else:
        raw_sources = source_bundle()
        workspace_bytes = 327_680
        identity = BackendIdentityEvidence.from_payload(
            {"backend_id": "torch_sdpa_flash_gqa", "build": "fixture"}
        )
    sources = tuple(
        source_file_evidence_from_bytes(path, raw_sources[path])
        for path in REQUIRED_SUT_SOURCES
    )
    static_sha256 = hashlib.sha256(
        raw_sources[REQUIRED_SUT_SOURCES[2]]
    ).hexdigest()
    evidence = source_shape(
        cache_binding(
            implementation_sha256=static_sha256,
            workspace_bytes=workspace_bytes,
        ),
        sources=sources,
    )
    key = operation_key(
        cache_layout_fingerprint=evidence.cache.layout.layout_fingerprint,
        backend_identity_sha256=identity.sha256,
        source_identity_sha256=phase3_source_identity_sha256(
            {item.relative_path: item.sha256 for item in sources}
        ),
    )
    point = Phase3DispatchPointBinding.create(operation_key=key)
    marker_prefix = (
        "kvbench.phase3.geometry_dispatch."
        "phase3-remediation-dispatch-fixture."
        "fixed_l-b1-l128-eager-r1.step0."
        f"{point.operation_fingerprint_sha256}."
    )
    gqa_marker = marker_prefix + "gqa"
    mha_marker = marker_prefix + "mha_control"
    gqa_raw = eager_launch_trace(gqa_marker)
    mha_raw = eager_launch_trace(mha_marker)
    gqa_artifact = RawTraceArtifact(
        relative_path="dispatch/gqa.geometry.chrome.json",
        sha256=hashlib.sha256(gqa_raw).hexdigest(),
        size_bytes=len(gqa_raw),
    )
    mha_artifact = RawTraceArtifact(
        relative_path="dispatch/mha.geometry.chrome.json",
        sha256=hashlib.sha256(mha_raw).hexdigest(),
        size_bytes=len(mha_raw),
    )
    trace_validation = revalidate_geometry_bound_raw_traces(
        point=point,
        gqa_artifact=gqa_artifact,
        mha_artifact=mha_artifact,
        gqa_raw=gqa_raw,
        mha_raw=mha_raw,
    )
    backend = replace(
        passing_backend(),
        source_build_fingerprint=identity.sha256,
    )
    gqa = replace(
        control(
            "gqa",
            trace_validation.gqa_device_events,
            marker=gqa_marker,
        ),
        backend=backend,
        raw_trace=gqa_artifact,
        trace_scope=trace_validation.gqa_scope,
    )
    mha = replace(
        control(
            "mha_control",
            trace_validation.mha_device_events,
            marker=mha_marker,
        ),
        backend=backend,
        raw_trace=mha_artifact,
        trace_scope=trace_validation.mha_scope,
    )
    gqa_sequence = analyze_flash_kernel_sequence(gqa.device_events)
    mha_sequence = analyze_flash_kernel_sequence(mha.device_events)
    result = evaluate_geometry_bound_gqa_device_dispatch(
        point=point,
        gqa=gqa,
        mha=mha,
        gqa_sequence=gqa_sequence,
        mha_sequence=mha_sequence,
        source_shape=evidence,
    )
    audit = Phase3GeometryBoundGQADeviceDispatchAudit(
        point=point,
        backend_identity=identity,
        gqa=gqa,
        mha=mha,
        source_shape=evidence,
        gqa_kernel_sequence=gqa_sequence,
        mha_kernel_sequence=mha_sequence,
        trace_validation=trace_validation,
        explicitly_related_families=(),
        related_family_policy_sha256=None,
        evaluation=result,
    )
    return {
        "audit": audit,
        "audit_raw": canonical_phase3_geometry_bound_dispatch_audit_bytes(
            audit
        ),
        "observation_raw": phase3_geometry_bound_dispatch_observation_bytes(
            audit
        ),
        "operation_key": key,
        "gqa_raw": gqa_raw,
        "mha_raw": mha_raw,
        "backend_identity_raw": identity.canonical_json.encode("utf-8"),
        "source_bytes_by_path": raw_sources,
    }


def paired_allocator_raw_evidence(
    fixture: dict[str, object],
) -> tuple[bytes, bytes, SplitKCompositeRawInputs]:
    from tests.unit.test_phase3_allocator_controls import (
        observation_bytes,
        successful_trace,
    )

    operation = fixture["operation_key"]
    gqa_dispatch = fixture["gqa_raw"]
    mha_dispatch = fixture["mha_raw"]
    if not isinstance(operation, Phase3AuditOperationKey):
        raise AssertionError("fixture operation key has the wrong type")
    if not isinstance(gqa_dispatch, bytes) or not isinstance(
        mha_dispatch, bytes
    ):
        raise AssertionError("fixture dispatch traces have the wrong type")
    control_trace = successful_trace()
    gqa_allocator = observation_bytes(
        role="gqa",
        key=operation,
        trace=control_trace,
        dispatch_raw=gqa_dispatch,
    )
    mha_allocator = observation_bytes(
        role="mha_control",
        key=operation,
        trace=control_trace,
        dispatch_raw=mha_dispatch,
    )
    paired = verify_phase3_paired_allocator_controls(
        gqa_raw=gqa_allocator,
        mha_control_raw=mha_allocator,
        operation_key=operation,
        gqa_dispatch_trace_raw=gqa_dispatch,
        mha_dispatch_trace_raw=mha_dispatch,
    )
    if not paired.passed:
        raise AssertionError(
            "synthetic paired allocator controls failed: "
            + ", ".join(paired.failure_reasons)
        )
    split_inputs = SplitKCompositeRawInputs.from_raw_bytes(
        gqa_dispatch_trace=gqa_dispatch,
        mha_dispatch_trace=mha_dispatch,
        gqa_allocator_control=gqa_allocator,
        mha_allocator_control=mha_allocator,
        split_k_pair_multiplicity=paired.split_k_pair_multiplicity,
    )
    return gqa_allocator, mha_allocator, split_inputs


def production_endpoint_trace(
    binding: ProductionAllocationBinding,
) -> list[dict[str, object]]:
    rules = instantiate_decision_0009_production_rules(binding)
    trace: list[dict[str, object]] = []
    next_address = 0x100000

    def append_lifetime(
        size: int,
        *,
        python_stack: list[dict[str, object]],
        cpp_stack: list[dict[str, object]],
    ) -> None:
        nonlocal next_address
        trace.extend(
            alloc_lifetime(
                size,
                address=next_address,
                stream=7,
                python_stack=python_stack,
                cpp_stack=cpp_stack,
                allocated_block_size=cuda_allocator_rounded_minimum(size),
            )
        )
        next_address += 0x100000

    for policy in rules.permitted_allocation_policies:
        size = next(iter(policy.allowed_requested_bytes))
        python_stack = [
            {
                "name": selector.function_name,
                "filename": f"/source/{selector.source_suffix}",
                "line": 10,
            }
            for selector in policy.required_python_frames
        ]
        cpp_stack = [
            {
                "name": selector.function_name,
                "filename": f"/source/{selector.source_suffix}",
                "line": 20,
            }
            for selector in policy.required_cpp_frames
        ]
        for _ in range(policy.exact_count):
            append_lifetime(
                size,
                python_stack=python_stack,
                cpp_stack=cpp_stack,
            )

    raw_inputs = binding.split_k_raw_inputs
    if raw_inputs is None:
        raise AssertionError("eager production binding lacks split-K inputs")
    split_cpp_stack = [
        {
            "name": name,
            "filename": "/source/flash_api.cpp",
            "line": 30 + index,
        }
        for index, name in enumerate(
            allocation_module.FLASH_SPLIT_K_CPP_MARKERS
        )
    ]
    split_python_stack = [
        {
            "name": "flash_attention_forward",
            "filename": "/source/src/kvbench/runtime/backend.py",
            "line": 40,
        }
    ]
    for splits, control_pair_count in raw_inputs.split_k_pair_multiplicity:
        if control_pair_count != 1:
            raise AssertionError("control fixture must contain one split-K pair")
        for _ in range(allocation_module.PHASE3_LAYER_COUNT):
            append_lifetime(
                binding.geometry.flash_split_k_output_accumulator_bytes(splits),
                python_stack=split_python_stack,
                cpp_stack=split_cpp_stack,
            )
            append_lifetime(
                binding.geometry.flash_split_k_lse_bytes(splits),
                python_stack=split_python_stack,
                cpp_stack=split_cpp_stack,
            )

    allocations = [event for event in trace if event["action"] == "alloc"]
    if len(allocations) != 1_066:
        raise AssertionError("source-backed fixture allocation count drifted")
    if sum(int(event["size"]) for event in allocations) != 10_960_908:
        raise AssertionError("source-backed fixture requested bytes drifted")
    return trace


def allocation_raw_evidence(
    fixture: dict[str, object],
    *,
    injected_allocation_bytes: int | None = None,
    mutate_audit: object | None = None,
) -> tuple[object, Phase3AllocationRawEvidence]:
    selected_operation = fixture["operation_key"]
    if not isinstance(selected_operation, Phase3AuditOperationKey):
        raise AssertionError("fixture operation key has the wrong type")
    _, _, split_inputs = paired_allocator_raw_evidence(fixture)
    binding = build_phase3_production_allocation_binding(
        operation_key=selected_operation,
        backend_identity=PHASE3_BACKEND_IDENTITY,
        split_k_raw_inputs=split_inputs,
    )
    endpoint_trace = production_endpoint_trace(binding)
    fake = _FakeTorch()
    harness = _FakeOperationHarness(binding)
    recorded = False

    def operation() -> object:
        nonlocal recorded
        if fake.cuda.memory.recording and not recorded:
            recorded = True
            recorded_trace = list(endpoint_trace)
            if injected_allocation_bytes is not None:
                recorded_trace.extend(
                    alloc_lifetime(
                        injected_allocation_bytes,
                        address=0x900000000,
                        stream=0,
                        allocated_block_size=cuda_allocator_rounded_minimum(
                            injected_allocation_bytes
                        ),
                    )
                )
            fake.cuda.memory.snapshot["device_traces"][0].extend(
                recorded_trace
            )
            allocations = [
                event
                for event in recorded_trace
                if event["action"] == "alloc"
            ]
            requested_bytes = sum(
                int(event["size"]) for event in allocations
            )
            block_bytes = sum(
                int(event["allocated_block_size"]) for event in allocations
            )
            fake.cuda.stats.update(
                {
                    "allocation.all.allocated": len(allocations),
                    "requested_bytes.all.allocated": requested_bytes,
                    "allocated_bytes.all.allocated": block_bytes,
                    "allocation.all.freed": len(allocations),
                    "requested_bytes.all.freed": requested_bytes,
                    "allocated_bytes.all.freed": block_bytes,
                }
            )
        return harness.operation()

    previous_torch = allocation_module._TORCH
    try:
        allocation_module._TORCH = fake
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            collected = collect_cuda_allocation_attribution(
                operation,
                production_binding=binding,
                staging_directory=staging,
                operation_witness=harness.callbacks,
                prepare_operation=harness.prepare,
                warmup_iterations=3,
                max_entries=100_000,
            )
            files = collected.raw_files
            audit_payload = json.loads(
                (staging / files.audit_file).read_text(encoding="utf-8")
            )
            if mutate_audit is not None:
                if not callable(mutate_audit):
                    raise AssertionError("audit mutation must be callable")
                mutate_audit(audit_payload)
            audit_raw = json.dumps(
                audit_payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            audit_digest = hashlib.sha256(audit_raw).hexdigest()
            raw = Phase3AllocationRawEvidence(
                snapshot_raw=(staging / files.snapshot_file).read_bytes(),
                trace_raw=(staging / files.trace_file).read_bytes(),
                memory_stats_before_raw=(
                    staging / files.memory_stats_before_file
                ).read_bytes(),
                memory_stats_after_raw=(
                    staging / files.memory_stats_after_file
                ).read_bytes(),
                memory_accounting_before_raw=(
                    staging / files.memory_accounting_before_file
                ).read_bytes(),
                memory_accounting_after_raw=(
                    staging / files.memory_accounting_after_file
                ).read_bytes(),
                operation_witness_raw=(
                    staging / files.operation_witness_file
                ).read_bytes(),
                audit_raw=audit_raw,
                audit_sha256_ledger_raw=(
                    f"{audit_digest}  {files.audit_file}\n".encode("ascii")
                ),
            )
    finally:
        allocation_module._TORCH = previous_torch
    return binding, raw


def allocation_join_facts(
    fixture: dict[str, object],
    *,
    injected_allocation_bytes: int | None = None,
    mutate_audit: object | None = None,
    operation: Phase3AuditOperationKey | None = None,
    gqa_dispatch_trace_sha256: str | None = None,
    mha_dispatch_trace_sha256: str | None = None,
    dispatch_trace_validation_sha256: str | None = None,
) -> Phase3AllocationJoinFacts:
    audit = fixture["audit"]
    if not isinstance(audit, Phase3GeometryBoundGQADeviceDispatchAudit):
        raise AssertionError("fixture audit has the wrong type")
    binding, raw = allocation_raw_evidence(
        fixture,
        injected_allocation_bytes=injected_allocation_bytes,
        mutate_audit=mutate_audit,
    )
    selected_operation = fixture["operation_key"] if operation is None else operation
    if not isinstance(selected_operation, Phase3AuditOperationKey):
        raise AssertionError("fixture operation key has the wrong type")
    assert audit.gqa.raw_trace is not None
    assert audit.mha.raw_trace is not None
    return Phase3AllocationJoinFacts.from_raw_evidence(
        operation_key=selected_operation,
        production_binding=binding,
        raw_evidence=raw,
        gqa_dispatch_trace_sha256=(
            audit.gqa.raw_trace.sha256
            if gqa_dispatch_trace_sha256 is None
            else gqa_dispatch_trace_sha256
        ),
        mha_dispatch_trace_sha256=(
            audit.mha.raw_trace.sha256
            if mha_dispatch_trace_sha256 is None
            else mha_dispatch_trace_sha256
        ),
        dispatch_trace_validation_sha256=(
            audit.trace_validation.evidence_sha256
            if dispatch_trace_validation_sha256 is None
            else dispatch_trace_validation_sha256
        ),
    )


class GraphCaptureCollectionTests(unittest.TestCase):
    def test_graph_trace_delegates_capture_to_production_helper(self) -> None:
        marker = "kvbench.phase3.geometry.graph-helper"
        raw = graph_launch_trace(marker)
        calls: list[str] = []
        static_output = object()

        class CapturedGraph:
            output = static_output

            def replay(self) -> object:
                calls.append("replay")
                return self.output

        capture_fixed_graph = mock.Mock(return_value=CapturedGraph())

        class FakeTrace:
            def __enter__(self) -> FakeTrace:
                return self

            def __exit__(self, *args: object) -> None:
                del args

            def export_chrome_trace(self, path: str) -> None:
                Path(path).write_bytes(raw)

        fake_profiler = type(
            "FakeProfiler",
            (),
            {
                "profile": mock.Mock(return_value=FakeTrace()),
                "ProfilerActivity": type(
                    "ProfilerActivity",
                    (),
                    {"CPU": "CPU", "CUDA": "CUDA"},
                ),
            },
        )
        synchronize = mock.Mock()
        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": type(
                    "FakeCuda",
                    (),
                    {"synchronize": synchronize},
                ),
                "autograd": type(
                    "FakeAutograd",
                    (),
                    {
                        "profiler": type(
                            "FakeAutogradProfiler",
                            (),
                            {
                                "record_function": staticmethod(
                                    lambda observed: nullcontext(observed)
                                )
                            },
                        )
                    },
                ),
            },
        )
        fake_graph_module = type(
            "FakeGraphModule",
            (),
            {"capture_fixed_graph": capture_fixed_graph},
        )

        def import_module(name: str) -> object:
            modules = {
                "torch": fake_torch,
                "torch.profiler": fake_profiler,
                "kvbench.runtime.cuda_graph": fake_graph_module,
            }
            return modules[name]

        def operation() -> object:
            calls.append("operation")
            return object()

        with tempfile.TemporaryDirectory(
            prefix="kvbench-graph-helper-unit-",
            dir="/tmp",
        ) as root:
            output = Path(root) / "trace.chrome.json"
            with mock.patch(
                "kvbench.runtime.gqa_device_dispatch.importlib.import_module",
                side_effect=import_module,
            ):
                artifact = collect_torch_profiler_trace(
                    operation,
                    output,
                    artifact_relative_path="dispatch/graph.chrome.json",
                    marker=marker,
                    warmup_count=3,
                    device="cuda:0",
                    execution_mode=CUDA_GRAPH_REPLAY_EXECUTION_MODE,
                )
            self.assertEqual(output.read_bytes(), raw)
        capture_fixed_graph.assert_called_once_with(
            operation,
            warmup_steps=3,
            device="cuda:0",
        )
        self.assertEqual(calls, ["replay"])
        self.assertEqual(synchronize.call_count, 1)
        contract = artifact.to_dict()["cuda_graph_capture_contract"]
        self.assertEqual(
            contract["helper"],
            "kvbench.runtime.cuda_graph.capture_fixed_graph",
        )
        self.assertIs(contract["dedicated_side_stream"], True)
        self.assertEqual(contract["capture_error_mode"], "global")


class Phase3PointAndCacheBindingTests(unittest.TestCase):
    def test_active_view_may_share_a_larger_native_kv_backing(self) -> None:
        binding = cache_binding(active_context=129, capacity=144, layer_index=7)
        self.assertTrue(binding.verified)
        self.assertGreater(
            binding.key_view.storage_bytes,
            binding.key_view.logical_bytes,
        )
        self.assertEqual(binding.layout.tensor_shape, (32, 1, 8, 144, 128))
        self.assertEqual(binding.layout.layout_name, PHASE3_CACHE_LAYOUT_NAME)

    def test_wrong_backing_heads_fail_binding(self) -> None:
        binding = cache_binding()
        wrong = replace(
            binding.key_backing,
            shape=(32, 1, 32, 129, 128),
        )
        observed = replace(binding, key_backing=wrong)
        self.assertFalse(observed.verified)
        self.assertIn("key_backing_shape_mismatch", observed.failure_reasons)

    def test_wrong_backing_capacity_fail_binding(self) -> None:
        binding = cache_binding()
        wrong = replace(
            binding.value_backing,
            shape=(32, 1, 8, 130, 128),
        )
        observed = replace(binding, value_backing=wrong)
        self.assertFalse(observed.verified)
        self.assertIn("value_backing_shape_mismatch", observed.failure_reasons)

    def test_storage_binding_flags_are_derived_from_pointer_evidence(self) -> None:
        binding = cache_binding()
        payload = binding.to_dict()
        self.assertEqual(
            payload["storage_pointers"]["key_backing"],
            payload["storage_pointers"]["key_view"],
        )
        self.assertRegex(
            payload["storage_pointers"]["sha256"],
            r"\A[0-9a-f]{64}\Z",
        )
        with self.assertRaises(ValueError):
            replace(binding, key_view_storage_ptr=0x3000)
        with self.assertRaises(ValueError):
            replace(binding, key_view_shares_backing_storage=False)

    def test_layout_fingerprint_is_geometry_and_source_bound(self) -> None:
        with self.assertRaises(ValueError):
            StaticCacheLayoutEvidence.create(
                batch_size=1,
                capacity=129,
                device="cuda:0",
                workspace_bytes=327_680,
                implementation_sha256=STATIC_CACHE_SHA256,
                layout_fingerprint="0" * 64,
            )

    def test_point_binding_rejects_wrong_capacity(self) -> None:
        payload = operation_key().to_dict()
        payload["capacity"] = 130
        with self.assertRaises(SchemaValidationError):
            Phase3DispatchPointBinding.create(
                operation_key=Phase3AuditOperationKey.from_dict(payload),
            )


class GeometryBoundKernelSequenceTests(unittest.TestCase):
    def test_related_family_policy_is_required_before_collection(self) -> None:
        common = {
            "operation_key": operation_key(),
            "cache_layout_fingerprint": "f" * 64,
            "cache_workspace_bytes": 1,
            "cache_layer_index": 0,
            "cache_key_backing": None,
            "cache_value_backing": None,
            "gqa_query": None,
            "gqa_key_view": None,
            "gqa_value_view": None,
            "mha_query": None,
            "mha_key": None,
            "mha_value": None,
            "output_directory": Path("."),
            "artifact_relative_root": "dispatch",
            "source_root": Path("."),
            "source_paths": (),
            "is_causal": False,
            "scale": 128**-0.5,
            "warmup_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "must coexist"):
            collect_phase3_geometry_bound_gqa_mha_device_dispatch(
                **common,
                explicitly_related_families={
                    (FLASH_FORWARD_FAMILY, FLASH_SPLIT_KV_FAMILY)
                },
            )
        with self.assertRaisesRegex(ValueError, "must coexist"):
            collect_phase3_geometry_bound_gqa_mha_device_dispatch(
                **common,
                related_family_policy_sha256="e" * 64,
            )

    def test_split_k_requires_forward_then_combine(self) -> None:
        split = (
            device_event(
                order=0,
                name=SPLIT_NAME,
                classification="flash_attention",
                family=FLASH_SPLIT_KV_FAMILY,
            ),
            device_event(
                order=1,
                name=SPLIT_COMBINE_NAME,
                classification="flash_attention",
                family=FLASH_SPLIT_KV_FAMILY,
            ),
        )
        parsed = analyze_flash_kernel_sequence(split)
        self.assertTrue(parsed.passed)
        self.assertEqual(parsed.variant, "split_k_forward_combine")
        self.assertEqual(parsed.forward_orders, (0,))
        self.assertEqual(parsed.combine_orders, (1,))
        comparison = compare_geometry_bound_kernel_sequences(
            analyze_flash_kernel_sequence((flash_event(),)),
            parsed,
        )
        self.assertEqual(comparison.relation, "unrelated")
        comparison = compare_geometry_bound_kernel_sequences(
            analyze_flash_kernel_sequence((flash_event(),)),
            parsed,
            explicitly_related={
                (FLASH_FORWARD_FAMILY, FLASH_SPLIT_KV_FAMILY)
            },
        )
        self.assertEqual(comparison.relation, "related")

    def test_split_k_missing_or_reversed_combine_fails(self) -> None:
        missing = analyze_flash_kernel_sequence(
            (
                device_event(
                    order=0,
                    name=SPLIT_NAME,
                    classification="flash_attention",
                    family=FLASH_SPLIT_KV_FAMILY,
                ),
            )
        )
        self.assertFalse(missing.passed)
        reversed_sequence = analyze_flash_kernel_sequence(
            (
                device_event(
                    order=0,
                    name=SPLIT_COMBINE_NAME,
                    classification="flash_attention",
                    family=FLASH_SPLIT_KV_FAMILY,
                ),
                device_event(
                    order=1,
                    name=SPLIT_NAME,
                    classification="flash_attention",
                    family=FLASH_SPLIT_KV_FAMILY,
                ),
            )
        )
        self.assertFalse(reversed_sequence.passed)
        self.assertIn(
            "split_k_combine_does_not_follow_forward",
            reversed_sequence.reasons,
        )

    def test_broad_split_prefix_cannot_hide_unrelated_kernel(self) -> None:
        parsed = analyze_flash_kernel_sequence(
            (
                device_event(
                    order=0,
                    name=(
                        "void pytorch_flash::"
                        "flash_fwd_splitkv_unrelated_kernel<traits>()"
                    ),
                    classification="flash_attention",
                    family=FLASH_SPLIT_KV_FAMILY,
                ),
            )
        )
        self.assertFalse(parsed.passed)
        self.assertIn("unrecognized_flash_kernel_component", parsed.reasons)

    def test_correlated_unknown_kernel_is_rejected_as_ambiguous(self) -> None:
        parsed = analyze_flash_kernel_sequence(
            (
                device_event(
                    order=0,
                    name="opaque_correlated_kernel",
                    classification="unknown_kernel",
                    family=None,
                ),
                flash_event(order=1),
            )
        )
        self.assertFalse(parsed.passed)
        self.assertIn("unrelated_scoped_kernel_present", parsed.reasons)

    def test_kernel_requires_launch_runtime_in_geometry_bound_mode(self) -> None:
        marker = "kvbench.phase3.geometry.fixture"
        raw = mismatched_runtime_trace()
        # Legacy frozen parsing remains unchanged unless the new path opts in.
        parsed = parse_scoped_chrome_cuda_events(raw, marker=marker)
        self.assertEqual(len(parsed.device_events), 1)
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_events(
                raw,
                marker=marker,
                require_kernel_launch_runtime=True,
            )

    def test_graph_parser_requires_marker_graph_launch_device_chain(self) -> None:
        marker = "kvbench.phase3.geometry.graph-fixture"
        raw = graph_launch_trace(marker)
        parsed = parse_scoped_chrome_cuda_graph_events(raw, marker=marker)
        self.assertIsInstance(parsed.scope, CUDAGraphTraceScopeEvidence)
        self.assertEqual(parsed.scope.runtime_correlations, (43,))
        self.assertIsNone(parsed.scope.graph_launch_external_id)
        self.assertEqual(parsed.scope.external_id_linkage, "absent_in_raw")
        self.assertEqual(parsed.scope.graph_id, 2)
        self.assertEqual(parsed.scope.graph_node_ids, (8589934592,))
        self.assertEqual(len(parsed.device_events), 1)
        self.assertIsNone(parsed.device_events[0].external_id)

        def payload_and_kernel() -> tuple[dict[str, object], dict[str, object]]:
            payload = json.loads(raw.decode("utf-8"))
            kernel = next(
                item
                for item in payload["traceEvents"]
                if item.get("cat") == "kernel"
            )
            return payload, kernel

        for label, mutation in (
            (
                "wrong_correlation",
                lambda kernel: kernel["args"].__setitem__("correlation", 999),
            ),
            (
                "wrong_stream",
                lambda kernel: kernel["args"].__setitem__("stream", 9),
            ),
            (
                "mismatched_external_id_presence",
                lambda kernel: kernel["args"].__setitem__("External id", 10),
            ),
            (
                "missing_graph_id",
                lambda kernel: kernel["args"].pop("graph id"),
            ),
            (
                "missing_graph_node_id",
                lambda kernel: kernel["args"].pop("graph node id"),
            ),
        ):
            with self.subTest(label=label):
                payload, kernel = payload_and_kernel()
                mutation(kernel)
                with self.assertRaises(ChromeTraceValidationError):
                    parse_scoped_chrome_cuda_graph_events(
                        json.dumps(payload).encode("utf-8"),
                        marker=marker,
                    )

    def test_graph_parser_accepts_gpu_completion_after_host_return(self) -> None:
        marker = "kvbench.phase3.geometry.graph-async-completion"
        payload = json.loads(graph_launch_trace(marker).decode("utf-8"))
        host = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "user_annotation"
        )
        host["dur"] = 20.0
        parsed = parse_scoped_chrome_cuda_graph_events(
            json.dumps(payload).encode("utf-8"),
            marker=marker,
        )
        self.assertEqual(parsed.scope.runtime_correlations, (43,))
        self.assertEqual(len(parsed.device_events), 1)

    def test_graph_parser_rejects_async_correlated_event_outside_gpu_marker(
        self,
    ) -> None:
        marker = "kvbench.phase3.geometry.graph-async-outside"
        payload = json.loads(graph_launch_trace(marker).decode("utf-8"))
        host = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "user_annotation"
        )
        host["dur"] = 20.0
        original = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "kernel"
        )
        outside = json.loads(json.dumps(original))
        outside["ts"] = 23.0
        outside["dur"] = 0.25
        outside["args"]["graph node id"] = 8589934593
        payload["traceEvents"].append(outside)
        with self.assertRaisesRegex(
            ChromeTraceValidationError,
            "graph-launch-correlated",
        ):
            parse_scoped_chrome_cuda_graph_events(
                json.dumps(payload).encode("utf-8"),
                marker=marker,
            )

    def test_graph_parser_rejects_unknown_gpu_event_after_host_return(
        self,
    ) -> None:
        marker = "kvbench.phase3.geometry.graph-async-unknown"
        payload = json.loads(graph_launch_trace(marker).decode("utf-8"))
        host = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "user_annotation"
        )
        gpu = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "gpu_user_annotation"
        )
        host["dur"] = 20.0
        gpu["dur"] = 5.0
        payload["traceEvents"].append(
            {
                "ph": "X",
                "cat": "cuda_kernel_v2",
                "name": "hidden_async_device_activity",
                "ts": 21.25,
                "dur": 0.25,
                "args": {
                    "stream": 7,
                    "correlation": 43,
                    "device": 0,
                    "context": 1,
                },
            }
        )
        with self.assertRaisesRegex(
            ChromeTraceValidationError,
            "unrecognized category",
        ):
            parse_scoped_chrome_cuda_graph_events(
                json.dumps(payload).encode("utf-8"),
                marker=marker,
            )

    def test_graph_parser_rejects_mixed_duplicate_and_out_of_scope_nodes(
        self,
    ) -> None:
        marker = "kvbench.phase3.geometry.graph-adversarial"
        raw = graph_launch_trace(marker)

        def with_second_kernel(
            *,
            timestamp: float,
            graph_id: int = 2,
            graph_node_id: int = 8589934593,
        ) -> dict[str, object]:
            payload = json.loads(raw.decode("utf-8"))
            original = next(
                item
                for item in payload["traceEvents"]
                if item.get("cat") == "kernel"
            )
            second = json.loads(json.dumps(original))
            second["ts"] = timestamp
            second["dur"] = 0.25
            second["args"]["graph id"] = graph_id
            second["args"]["graph node id"] = graph_node_id
            payload["traceEvents"].append(second)
            return payload

        candidates = (
            ("mixed_graph", with_second_kernel(timestamp=20.5, graph_id=3)),
            (
                "duplicate_node",
                with_second_kernel(
                    timestamp=20.5,
                    graph_node_id=8589934592,
                ),
            ),
            ("outside_gpu_marker", with_second_kernel(timestamp=30.0)),
        )
        for label, payload in candidates:
            with self.subTest(label=label):
                with self.assertRaises(ChromeTraceValidationError):
                    parse_scoped_chrome_cuda_graph_events(
                        json.dumps(payload).encode("utf-8"),
                        marker=marker,
                    )
        other_correlation = with_second_kernel(timestamp=30.0)
        other_correlation["traceEvents"][-1]["args"]["correlation"] = 99
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_graph_events(
                json.dumps(other_correlation).encode("utf-8"),
                marker=marker,
            )
        late_launch = json.loads(raw.decode("utf-8"))
        launch = next(
            item
            for item in late_launch["traceEvents"]
            if item.get("name") == "cudaGraphLaunch"
        )
        launch["ts"] = 19.5
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_graph_events(
                json.dumps(late_launch).encode("utf-8"),
                marker=marker,
            )

    def test_graph_parser_accepts_split_forward_and_combine_nodes(self) -> None:
        marker = "kvbench.phase3.geometry.graph-split"
        payload = json.loads(
            graph_launch_trace(marker, kernel_name=SPLIT_NAME).decode("utf-8")
        )
        original = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "kernel"
        )
        combine = json.loads(json.dumps(original))
        combine["name"] = SPLIT_COMBINE_NAME
        combine["ts"] = 20.5
        combine["dur"] = 0.25
        combine["args"]["graph node id"] = 8589934593
        payload["traceEvents"].append(combine)
        payload["traceEvents"].reverse()
        parsed = parse_scoped_chrome_cuda_graph_events(
            json.dumps(payload).encode("utf-8"),
            marker=marker,
        )
        self.assertEqual(
            parsed.scope.graph_node_ids,
            (8589934592, 8589934593),
        )
        sequence = analyze_flash_kernel_sequence(parsed.device_events)
        self.assertTrue(sequence.passed)
        self.assertEqual(sequence.variant, "split_k_forward_combine")

    def test_eager_parser_rejects_auxiliary_in_marker_device_activity(
        self,
    ) -> None:
        marker = "kvbench.phase3.geometry.eager-auxiliary"
        payload = json.loads(eager_launch_trace(marker).decode("utf-8"))
        payload["traceEvents"].append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "opaque_auxiliary_kernel",
                "ts": 30.0,
                "dur": 1.0,
                "args": {
                    "stream": 9,
                    "correlation": 99,
                    "External id": 99,
                    "device": 0,
                    "context": 1,
                },
            }
        )
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_events(
                json.dumps(payload).encode("utf-8"),
                marker=marker,
                require_kernel_launch_runtime=True,
            )

    def test_eager_zero_graph_identity_normalizes_to_absent(self) -> None:
        marker = "kvbench.phase3.geometry.eager-zero-graph"
        payload = json.loads(eager_launch_trace(marker).decode("utf-8"))
        kernel = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "kernel"
        )
        kernel["args"]["graph id"] = 0
        kernel["args"]["graph node id"] = 0
        parsed = parse_scoped_chrome_cuda_events(
            json.dumps(payload).encode("utf-8"),
            marker=marker,
            require_kernel_launch_runtime=True,
        )
        self.assertEqual(len(parsed.device_events), 1)
        self.assertIsNone(parsed.device_events[0].graph_id)
        self.assertIsNone(parsed.device_events[0].graph_node_id)

    def test_graph_identity_rejects_partial_mixed_and_invalid_pairs(self) -> None:
        marker = "kvbench.phase3.geometry.eager-invalid-graph"
        raw = eager_launch_trace(marker)

        def payload_and_args() -> tuple[dict[str, object], dict[str, object]]:
            payload = json.loads(raw.decode("utf-8"))
            kernel = next(
                item
                for item in payload["traceEvents"]
                if item.get("cat") == "kernel"
            )
            return payload, kernel["args"]

        candidates = (
            ("zero_positive", {"graph id": 0, "graph node id": 2}),
            ("positive_zero", {"graph id": 2, "graph node id": 0}),
            ("negative", {"graph id": -1, "graph node id": -1}),
            ("boolean", {"graph id": True, "graph node id": True}),
            ("partial", {"graph id": 2}),
        )
        for label, identity in candidates:
            with self.subTest(label=label):
                payload, args = payload_and_args()
                args.update(identity)
                with self.assertRaises(ChromeTraceValidationError):
                    parse_scoped_chrome_cuda_events(
                        json.dumps(payload).encode("utf-8"),
                        marker=marker,
                        require_kernel_launch_runtime=True,
                    )

    def test_graph_parser_rejects_zero_graph_identity(self) -> None:
        marker = "kvbench.phase3.geometry.graph-zero-identity"
        payload = json.loads(graph_launch_trace(marker).decode("utf-8"))
        kernel = next(
            item
            for item in payload["traceEvents"]
            if item.get("cat") == "kernel"
        )
        kernel["args"]["graph id"] = 0
        kernel["args"]["graph node id"] = 0
        with self.assertRaises(ChromeTraceValidationError):
            parse_scoped_chrome_cuda_graph_events(
                json.dumps(payload).encode("utf-8"),
                marker=marker,
            )

    def test_graph_raw_revalidation_derives_execution_mode_and_sequence(self) -> None:
        point = dispatch_point(graph_mode=GraphMode.CUDA_GRAPH)
        marker_prefix = (
            "kvbench.phase3.geometry_dispatch."
            "phase3-remediation-dispatch-fixture."
            "fixed_l-b1-l128-cuda_graph-r1.step0."
            f"{point.operation_fingerprint_sha256}."
        )
        gqa_marker = marker_prefix + "gqa"
        mha_marker = marker_prefix + "mha_control"
        gqa_raw = graph_launch_trace(gqa_marker)
        mha_raw = graph_launch_trace(mha_marker)
        gqa_artifact = RawTraceArtifact(
            relative_path="dispatch/gqa.graph.chrome.json",
            sha256=hashlib.sha256(gqa_raw).hexdigest(),
            size_bytes=len(gqa_raw),
            execution_mode=CUDA_GRAPH_REPLAY_EXECUTION_MODE,
        )
        mha_artifact = RawTraceArtifact(
            relative_path="dispatch/mha.graph.chrome.json",
            sha256=hashlib.sha256(mha_raw).hexdigest(),
            size_bytes=len(mha_raw),
            execution_mode=CUDA_GRAPH_REPLAY_EXECUTION_MODE,
        )
        validated = revalidate_geometry_bound_raw_traces(
            point=point,
            gqa_artifact=gqa_artifact,
            mha_artifact=mha_artifact,
            gqa_raw=gqa_raw,
            mha_raw=mha_raw,
        )
        self.assertEqual(
            validated.execution_mode,
            CUDA_GRAPH_REPLAY_EXECUTION_MODE,
        )
        self.assertIsInstance(validated.gqa_scope, CUDAGraphTraceScopeEvidence)
        self.assertTrue(validated.gqa_kernel_sequence.passed)
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_geometry_bound_raw_traces(
                point=point,
                gqa_artifact=replace(
                    gqa_artifact,
                    execution_mode=EAGER_EXECUTION_MODE,
                ),
                mha_artifact=mha_artifact,
                gqa_raw=gqa_raw,
                mha_raw=mha_raw,
            )

    def test_raw_memset_parser_retains_exact_bytes_and_role(self) -> None:
        raw = json.dumps(
            {
                "schemaVersion": 1,
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "gpu_memset",
                        "name": "Memset (Device)",
                        "ts": 1.0,
                        "dur": 1.0,
                        "args": {
                            "stream": 7,
                            "correlation": 43,
                            "External id": 10,
                            "device": 0,
                            "context": 1,
                            "bytes": 8192,
                            "memory role": "fixed_attention_workspace_zero",
                        },
                    }
                ],
            }
        ).encode("utf-8")
        event = parse_chrome_cuda_events(raw)[0]
        self.assertEqual(event.memory_bytes, 8192)
        self.assertEqual(event.memory_role, "fixed_attention_workspace_zero")
        payload = json.loads(raw.decode("utf-8"))
        payload["traceEvents"][0]["args"]["Size"] = 4096
        with self.assertRaises(ChromeTraceValidationError):
            parse_chrome_cuda_events(json.dumps(payload).encode("utf-8"))

    def test_graph_point_rejects_eager_control_mode(self) -> None:
        point = dispatch_point(graph_mode=GraphMode.CUDA_GRAPH)
        gqa = control("gqa", (flash_event(),))
        mha = control("mha_control", (flash_event(),))
        evidence = source_shape(cache_binding())
        result = evaluate_geometry_bound_gqa_device_dispatch(
            point=point,
            gqa=gqa,
            mha=mha,
            gqa_sequence=analyze_flash_kernel_sequence(gqa.device_events),
            mha_sequence=analyze_flash_kernel_sequence(mha.device_events),
            source_shape=evidence,
        )
        self.assertFalse(result.dispatch_verified)
        self.assertEqual(result.verdict, GQAVerdict.DISPATCH_UNVERIFIED)

    def test_query_layout_is_exact_and_output_layout_is_semantic(self) -> None:
        gqa = control("gqa", (flash_event(),))
        mha = control("mha_control", (flash_event(),))
        evidence = source_shape(cache_binding())
        self.assertTrue(evidence.shape_verified_for(gqa, mha))

        alternate_singleton_stride = replace(
            evidence.gqa_output,
            stride=(4096, 128, 4096, 1),
        )
        semantic_output = replace(
            evidence,
            gqa_output=alternate_singleton_stride,
        )
        self.assertNotEqual(
            semantic_output.gqa_output,
            semantic_output.mha_output,
        )
        self.assertTrue(semantic_output.shape_verified_for(gqa, mha))

        bad_query_stride = replace(
            evidence.gqa_query,
            stride=(4096, 129, 128, 1),
        )
        self.assertFalse(
            replace(
                evidence,
                gqa_query=bad_query_stride,
            ).shape_verified_for(gqa, mha)
        )

        invalid_outputs = (
            replace(
                alternate_singleton_stride,
                stride=(4096, 129, 4096, 1),
            ),
            replace(
                alternate_singleton_stride,
                stride=(4096, 128, 4096, 2),
            ),
            replace(
                alternate_singleton_stride,
                storage_bytes=alternate_singleton_stride.logical_bytes + 2,
            ),
            replace(alternate_singleton_stride, storage_offset=1),
            replace(alternate_singleton_stride, is_contiguous=False),
        )
        for output in invalid_outputs:
            with self.subTest(output=output):
                self.assertFalse(
                    replace(
                        semantic_output,
                        gqa_output=output,
                    ).shape_verified_for(gqa, mha)
                )


class GeometryBoundDispatchVerdictTests(unittest.TestCase):
    def test_backend_flags_cannot_override_a_mismatched_transcript(self) -> None:
        backend = replace(
            passing_backend(),
            eligibility_diagnostics=(
                "enabled_backends=FLASH_ATTENTION",
                "can_use_flash_attention=False",
                "fused_sdp_choice=1",
            ),
        )
        self.assertTrue(backend.flash_eligible)
        self.assertFalse(backend.control_transcript_verified)
        self.assertTrue(backend.passed)
        self.assertRegex(
            backend.control_transcript_sha256,
            r"\A[0-9a-f]{64}\Z",
        )
        point = dispatch_point()
        gqa = replace(control("gqa", (flash_event(),)), backend=backend)
        mha = control("mha_control", (flash_event(),))
        result = evaluate_geometry_bound_gqa_device_dispatch(
            point=point,
            gqa=gqa,
            mha=mha,
            gqa_sequence=analyze_flash_kernel_sequence(gqa.device_events),
            mha_sequence=analyze_flash_kernel_sequence(mha.device_events),
            source_shape=source_shape(cache_binding()),
        )
        self.assertFalse(result.dispatch_verified)
        self.assertEqual(result.verdict, GQAVerdict.DISPATCH_UNVERIFIED)

    def test_clean_dispatch_remains_allocation_unproven(self) -> None:
        result = evaluation((flash_event(),))
        self.assertTrue(result.dispatch_verified)
        self.assertTrue(result.no_replication_kernel_verified)
        self.assertFalse(result.allocation_verified)
        self.assertEqual(
            result.verdict,
            GQAVerdict.NONMATERIALIZATION_UNPROVEN,
        )

    def test_preceding_repeat_is_positive_materialization(self) -> None:
        events = (
            device_event(
                order=0,
                name="repeat_interleave_cuda_kernel",
                classification="repeat_materialization",
                family=None,
            ),
            flash_event(order=1),
        )
        result = evaluation(events)
        self.assertEqual(result.verdict, GQAVerdict.MATERIALIZATION_DETECTED)
        self.assertTrue(result.positive_materialization_evidence)

    def test_direct_source_replication_takes_taxonomy_precedence(self) -> None:
        for finding in (
            "repeat_kv",
            "repeat_interleave",
            "tensor_repeat",
            "tensor_expand",
            "replication_copy",
        ):
            with self.subTest(finding=finding):
                result = evaluation(
                    (flash_event(),),
                    source_findings=(finding,),
                )
                self.assertEqual(
                    result.verdict,
                    GQAVerdict.MATERIALIZATION_DETECTED,
                )
                self.assertTrue(
                    any(
                        item.endswith(":" + finding)
                        for item in result.positive_materialization_evidence
                    )
                )

    def test_raw_source_bytes_produce_typed_materialization_findings(self) -> None:
        direct = source_file_evidence_from_bytes(
            REQUIRED_SUT_SOURCES[0],
            (
                b"def flash_attention_forward(query, key, value):\n"
                b"    expanded_key.copy_(key)\n"
                b"    return query\n"
            ),
        )
        self.assertEqual(direct.findings, ("replication_copy",))
        self.assertEqual(
            direct.typed_findings[0].evidence_type,
            "direct_gqa_replication_path",
        )
        self.assertTrue(
            direct.typed_findings[0].positive_materialization_evidence
        )
        unrelated_contract = source_file_evidence_from_bytes(
            REQUIRED_SUT_SOURCES[0],
            (
                b"def flash_attention_forward(query, key, value):\n"
                b"    return torch.cat((query, value), dim=-1)\n"
            ),
        )
        self.assertEqual(unrelated_contract.findings, ("torch_cat",))
        self.assertFalse(
            unrelated_contract.typed_findings[0].positive_materialization_evidence
        )
        unselected = source_file_evidence_from_bytes(
            REQUIRED_SUT_SOURCES[0],
            (
                b"def flash_attention_forward(query, key, value):\n"
                b"    return query\n"
                b"def repeat_kv(value):\n"
                b"    return value\n"
                b"def unused(k_cache):\n"
                b"    return k_cache.repeat_interleave(4, dim=1)\n"
            ),
        )
        self.assertEqual(
            unselected.findings,
            ("repeat_kv", "repeat_interleave"),
        )
        self.assertFalse(unselected.positive_materialization_findings)
        selected_cases = (
            (
                b"    return k_cache.repeat_interleave(4, dim=1)\n",
                "repeat_interleave",
            ),
            (
                b"    return key_states.expand(1, 32, 1, 128).contiguous()\n",
                "tensor_expand",
            ),
            (
                b"    expanded_key.copy_(key_states)\n    return query\n",
                "replication_copy",
            ),
        )
        for body, expected in selected_cases:
            with self.subTest(expected=expected):
                observed = source_file_evidence_from_bytes(
                    REQUIRED_SUT_SOURCES[0],
                    (
                        b"def flash_attention_forward("
                        b"query, key_states, value):\n"
                        + body
                    ),
                )
                self.assertIn(expected, observed.findings)
                self.assertIn(
                    expected,
                    observed.positive_materialization_findings,
                )

    def test_missing_selected_source_path_cannot_verify_source(self) -> None:
        raw_sources = source_bundle()
        raw_sources[REQUIRED_SUT_SOURCES[0]] = (
            b"def similarly_named_forward(query, key, value):\n"
            b"    return query\n"
        )
        sources = tuple(
            source_file_evidence_from_bytes(path, raw_sources[path])
            for path in REQUIRED_SUT_SOURCES
        )
        static_sha256 = hashlib.sha256(
            raw_sources[REQUIRED_SUT_SOURCES[2]]
        ).hexdigest()
        evidence = source_shape(
            cache_binding(implementation_sha256=static_sha256),
            sources=sources,
        )
        self.assertFalse(evidence.source_verified)
        self.assertFalse(sources[0].passed)
        self.assertEqual(sources[0].selected_function_paths, ())

    def test_preceding_copy_blocks_clean_proof_without_inventing_materialization(self) -> None:
        events = (
            device_event(
                order=0,
                name="Memcpy DtoD",
                category="gpu_memcpy",
                classification="device_copy_candidate",
                family=None,
                copy_bytes=4096,
            ),
            flash_event(order=1),
        )
        result = evaluation(events)
        self.assertEqual(
            result.verdict,
            GQAVerdict.NONMATERIALIZATION_UNPROVEN,
        )
        self.assertFalse(result.no_replication_kernel_verified)
        self.assertFalse(result.positive_materialization_evidence)

    def test_preceding_memset_requires_destination_binding(self) -> None:
        unbound = device_event(
            order=0,
            name="Memset (Device)",
            category="gpu_memset",
            classification="device_memset",
            family=None,
            memory_bytes=8192,
            memory_role="fixed_attention_workspace_zero",
        )
        result = evaluation((unbound, flash_event(order=1)))
        self.assertFalse(result.no_replication_kernel_verified)
        self.assertEqual(
            result.verdict,
            GQAVerdict.NONMATERIALIZATION_UNPROVEN,
        )
        native_context_bytes = 2 * 1 * 8 * 129 * 128 * 2
        for event in (
            replace(unbound, memory_role=None),
            replace(unbound, memory_bytes=native_context_bytes),
        ):
            with self.subTest(event=event):
                failed = evaluation((event, flash_event(order=1)))
                self.assertFalse(failed.no_replication_kernel_verified)
                self.assertEqual(
                    failed.verdict,
                    GQAVerdict.NONMATERIALIZATION_UNPROVEN,
                )

    def test_other_preceding_device_activity_fails_closed(self) -> None:
        candidates = (
            device_event(
                order=0,
                name="opaque_device_kernel",
                classification="unknown_kernel",
                family=None,
            ),
            device_event(
                order=0,
                name="expand_copy_kernel",
                classification="expand_materialization",
                family=None,
            ),
            device_event(
                order=0,
                name="transpose_copy_kernel",
                classification="transpose_copy_candidate",
                family=None,
                copy_bytes=4096,
            ),
        )
        for event in candidates:
            with self.subTest(classification=event.classification):
                result = evaluation((event, flash_event(order=1)))
                self.assertFalse(result.no_replication_kernel_verified)

    def test_audit_binds_markers_identity_and_dispatch_hashes(self) -> None:
        raw_sources = source_bundle()
        sources = tuple(
            source_file_evidence_from_bytes(path, raw_sources[path])
            for path in REQUIRED_SUT_SOURCES
        )
        static_sha256 = hashlib.sha256(
            raw_sources[REQUIRED_SUT_SOURCES[2]]
        ).hexdigest()
        evidence = source_shape(
            cache_binding(implementation_sha256=static_sha256),
            sources=sources,
        )
        identity = BackendIdentityEvidence.from_payload(
            {"backend_id": "torch_sdpa_flash_gqa", "build": "fixture"}
        )
        point = Phase3DispatchPointBinding.create(
            operation_key=operation_key(
                cache_layout_fingerprint=evidence.cache.layout.layout_fingerprint,
                backend_identity_sha256=identity.sha256,
                source_identity_sha256=phase3_source_identity_sha256(
                    {item.relative_path: item.sha256 for item in sources}
                ),
            )
        )
        marker_prefix = (
            "kvbench.phase3.geometry_dispatch."
            "phase3-remediation-dispatch-fixture."
            "fixed_l-b1-l128-eager-r1.step0."
            f"{point.operation_fingerprint_sha256}."
        )
        gqa_marker = marker_prefix + "gqa"
        mha_marker = marker_prefix + "mha_control"
        gqa_raw = eager_launch_trace(gqa_marker)
        mha_raw = eager_launch_trace(mha_marker)
        gqa_artifact = RawTraceArtifact(
            relative_path="dispatch/gqa.geometry.chrome.json",
            sha256=hashlib.sha256(gqa_raw).hexdigest(),
            size_bytes=len(gqa_raw),
        )
        mha_artifact = RawTraceArtifact(
            relative_path="dispatch/mha.geometry.chrome.json",
            sha256=hashlib.sha256(mha_raw).hexdigest(),
            size_bytes=len(mha_raw),
        )
        trace_validation = revalidate_geometry_bound_raw_traces(
            point=point,
            gqa_artifact=gqa_artifact,
            mha_artifact=mha_artifact,
            gqa_raw=gqa_raw,
            mha_raw=mha_raw,
        )
        backend = replace(
            passing_backend(),
            source_build_fingerprint=identity.sha256,
        )
        gqa = replace(
            control(
                "gqa",
                trace_validation.gqa_device_events,
                marker=gqa_marker,
            ),
            backend=backend,
            raw_trace=gqa_artifact,
            trace_scope=trace_validation.gqa_scope,
        )
        mha = replace(
            control(
                "mha_control",
                trace_validation.mha_device_events,
                marker=mha_marker,
            ),
            backend=backend,
            raw_trace=mha_artifact,
            trace_scope=trace_validation.mha_scope,
        )
        gqa_sequence = analyze_flash_kernel_sequence(gqa.device_events)
        mha_sequence = analyze_flash_kernel_sequence(mha.device_events)
        result = evaluate_geometry_bound_gqa_device_dispatch(
            point=point,
            gqa=gqa,
            mha=mha,
            gqa_sequence=gqa_sequence,
            mha_sequence=mha_sequence,
            source_shape=evidence,
        )
        audit = Phase3GeometryBoundGQADeviceDispatchAudit(
            point=point,
            backend_identity=identity,
            gqa=gqa,
            mha=mha,
            source_shape=evidence,
            gqa_kernel_sequence=gqa_sequence,
            mha_kernel_sequence=mha_sequence,
            trace_validation=trace_validation,
            explicitly_related_families=(),
            related_family_policy_sha256=None,
            evaluation=result,
        )
        with self.assertRaises(ValueError):
            replace(audit, related_family_policy_sha256="e" * 64)
        payload = audit.to_dict()
        self.assertIs(payload["allocation_size_proof"]["verified"], False)
        task_c_inputs = payload["allocation_size_proof"][
            "raw_derived_binding_inputs"
        ]
        self.assertEqual(
            task_c_inputs["trace_validation_sha256"],
            trace_validation.evidence_sha256,
        )
        self.assertEqual(
            task_c_inputs["gqa_kv_bytes"]["expanded_kv_bytes"],
            2 * 1 * 32 * 129 * 128 * 2,
        )
        self.assertEqual(
            payload["dispatch_trace_sha256"]["gqa"],
            hashlib.sha256(gqa_raw).hexdigest(),
        )
        self.assertEqual(
            payload["point_binding"]["active_context"],
            129,
        )
        rebuilt = revalidate_phase3_geometry_bound_dispatch_audit(
            audit,
            gqa_raw=gqa_raw,
            mha_raw=mha_raw,
            backend_identity_raw=identity.canonical_json.encode("utf-8"),
            source_bytes_by_path=raw_sources,
        )
        self.assertEqual(rebuilt, audit)
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_audit(
                audit,
                gqa_raw=gqa_raw + b" ",
                mha_raw=mha_raw,
                backend_identity_raw=identity.canonical_json.encode("utf-8"),
                source_bytes_by_path=raw_sources,
            )
        tampered_sources = dict(raw_sources)
        tampered_sources[REQUIRED_SUT_SOURCES[0]] += b"# changed\n"
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_audit(
                audit,
                gqa_raw=gqa_raw,
                mha_raw=mha_raw,
                backend_identity_raw=identity.canonical_json.encode("utf-8"),
                source_bytes_by_path=tampered_sources,
            )
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_audit(
                audit,
                gqa_raw=gqa_raw,
                mha_raw=mha_raw,
                backend_identity_raw=(
                    identity.canonical_json.replace("fixture", "changed").encode(
                        "utf-8"
                    )
                ),
                source_bytes_by_path=raw_sources,
            )
        object.__setattr__(
            audit,
            "evaluation",
            replace(
                audit.evaluation,
                reasons=(*audit.evaluation.reasons, "tampered"),
            ),
        )
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_audit(
                audit,
                gqa_raw=gqa_raw,
                mha_raw=mha_raw,
                backend_identity_raw=identity.canonical_json.encode("utf-8"),
                source_bytes_by_path=raw_sources,
            )


class GeometryBoundRawByteReplayTests(unittest.TestCase):
    @staticmethod
    def canonical(payload: dict[str, object]) -> bytes:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def raw_arguments(
        self,
        fixture: dict[str, object],
        **updates: object,
    ) -> dict[str, object]:
        arguments = {
            "b011_audit_raw": fixture["observation_raw"],
            "operation_key": fixture["operation_key"],
            "gqa_raw": fixture["gqa_raw"],
            "mha_raw": fixture["mha_raw"],
            "backend_identity_raw": fixture["backend_identity_raw"],
            "source_bytes_by_path": fixture["source_bytes_by_path"],
        }
        arguments.update(updates)
        return arguments

    def test_single_b011_artifact_reconstructs_and_hashes_derived_audit(
        self,
    ) -> None:
        fixture = raw_replay_fixture()
        audit = fixture["audit"]
        self.assertIsInstance(
            audit,
            Phase3GeometryBoundGQADeviceDispatchAudit,
        )
        self.assertEqual(
            phase3_geometry_bound_dispatch_evidence_bytes(audit),
            fixture["observation_raw"],
        )
        parsed = parse_phase3_geometry_bound_dispatch_observation_bytes(
            fixture["observation_raw"]
        )
        self.assertEqual(parsed.canonical_bytes(), fixture["observation_raw"])
        self.assertEqual(
            parsed.derived_audit_sha256,
            hashlib.sha256(fixture["audit_raw"]).hexdigest(),
        )
        rebuilt = revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
            **self.raw_arguments(fixture)
        )
        self.assertEqual(rebuilt, audit)
        rebuilt_with_optional_presentation = (
            revalidate_phase3_geometry_bound_dispatch_audit_from_raw(
                observation_raw=fixture["observation_raw"],
                operation_key=fixture["operation_key"],
                gqa_raw=fixture["gqa_raw"],
                mha_raw=fixture["mha_raw"],
                backend_identity_raw=fixture["backend_identity_raw"],
                source_bytes_by_path=fixture["source_bytes_by_path"],
                audit_raw=fixture["audit_raw"],
            )
        )
        self.assertEqual(rebuilt_with_optional_presentation, audit)

    def test_observation_and_optional_audit_require_canonical_unique_json(
        self,
    ) -> None:
        fixture = raw_replay_fixture()
        observation_raw = fixture["observation_raw"]
        duplicate = b'{"schema_version":"duplicate",' + observation_raw[1:]
        for label, raw in (
            ("duplicate", duplicate),
            ("noncanonical", observation_raw + b"\n"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(GQADeviceDispatchError):
                    parse_phase3_geometry_bound_dispatch_observation_bytes(raw)
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_audit_from_raw(
                observation_raw=observation_raw,
                operation_key=fixture["operation_key"],
                gqa_raw=fixture["gqa_raw"],
                mha_raw=fixture["mha_raw"],
                backend_identity_raw=fixture["backend_identity_raw"],
                source_bytes_by_path=fixture["source_bytes_by_path"],
                audit_raw=fixture["audit_raw"] + b"\n",
            )

    def test_derived_verdict_flags_are_never_trusted(self) -> None:
        fixture = raw_replay_fixture()
        tampered_audit = json.loads(fixture["audit_raw"].decode("utf-8"))
        tampered_audit["evaluation"]["dispatch_verified"] = False
        tampered_audit["evaluation"]["verdict"] = (
            GQAVerdict.DISPATCH_UNVERIFIED.value
        )
        tampered_audit_raw = self.canonical(tampered_audit)
        tampered_observation = json.loads(
            fixture["observation_raw"].decode("utf-8")
        )
        tampered_observation["derived_audit_sha256"] = hashlib.sha256(
            tampered_audit_raw
        ).hexdigest()
        tampered_observation_raw = self.canonical(tampered_observation)
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                **self.raw_arguments(
                    fixture,
                    b011_audit_raw=tampered_observation_raw,
                )
            )
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_audit_from_raw(
                observation_raw=tampered_observation_raw,
                operation_key=fixture["operation_key"],
                gqa_raw=fixture["gqa_raw"],
                mha_raw=fixture["mha_raw"],
                backend_identity_raw=fixture["backend_identity_raw"],
                source_bytes_by_path=fixture["source_bytes_by_path"],
                audit_raw=tampered_audit_raw,
            )

    def test_trace_role_and_cross_step_swaps_are_rejected(self) -> None:
        fixture = raw_replay_fixture()
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                **self.raw_arguments(
                    fixture,
                    gqa_raw=fixture["mha_raw"],
                    mha_raw=fixture["gqa_raw"],
                )
            )
        swapped_paths = json.loads(fixture["observation_raw"].decode("utf-8"))
        paths = swapped_paths["trace_relative_paths"]
        paths["gqa"], paths["mha_control"] = (
            paths["mha_control"],
            paths["gqa"],
        )
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                **self.raw_arguments(
                    fixture,
                    b011_audit_raw=self.canonical(swapped_paths),
                )
            )

        wrong_step = growing_operation_key(decode_step=1)
        wrong_point = Phase3DispatchPointBinding.create(operation_key=wrong_step)
        gqa_raw = fixture["gqa_raw"]
        mha_raw = fixture["mha_raw"]
        with self.assertRaises(ChromeTraceValidationError):
            revalidate_geometry_bound_raw_traces(
                point=wrong_point,
                gqa_artifact=RawTraceArtifact(
                    relative_path="dispatch/gqa.geometry.chrome.json",
                    sha256=hashlib.sha256(gqa_raw).hexdigest(),
                    size_bytes=len(gqa_raw),
                ),
                mha_artifact=RawTraceArtifact(
                    relative_path="dispatch/mha.geometry.chrome.json",
                    sha256=hashlib.sha256(mha_raw).hexdigest(),
                    size_bytes=len(mha_raw),
                ),
                gqa_raw=gqa_raw,
                mha_raw=mha_raw,
            )

    def test_source_backend_cache_and_operation_mismatches_are_rejected(
        self,
    ) -> None:
        fixture = raw_replay_fixture()
        tampered_sources = dict(fixture["source_bytes_by_path"])
        tampered_sources[REQUIRED_SUT_SOURCES[0]] += b"# tampered\n"
        changed_backend = self.canonical(
            {"backend_id": "torch_sdpa_flash_gqa", "build": "changed"}
        )
        cache_mismatch = json.loads(
            fixture["observation_raw"].decode("utf-8")
        )
        cache_mismatch["cache_observation"]["workspace_bytes"] += 1
        candidates = (
            {"source_bytes_by_path": tampered_sources},
            {"backend_identity_raw": changed_backend},
            {"b011_audit_raw": self.canonical(cache_mismatch)},
            {"operation_key": growing_operation_key(decode_step=0)},
        )
        for updates in candidates:
            with self.subTest(updates=tuple(updates)):
                with self.assertRaises(GQADeviceDispatchError):
                    revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                        **self.raw_arguments(fixture, **updates)
                    )

    def test_unapproved_related_family_decision_is_rejected(self) -> None:
        fixture = raw_replay_fixture()
        pair = [FLASH_FORWARD_FAMILY, FLASH_SPLIT_KV_FAMILY]
        decision = self.canonical(
            {
                "schema_version": (
                    "kvbench-phase3-related-kernel-family-decision-1.0.0"
                ),
                "scope": "phase3_geometry_bound_dispatch",
                "approved": True,
                "explicitly_related_pairs": [pair],
            }
        )
        observation = json.loads(fixture["observation_raw"].decode("utf-8"))
        observation["kernel_family_policy"] = {
            "explicitly_related_pairs": [pair],
            "decision_record_sha256": hashlib.sha256(decision).hexdigest(),
        }
        with self.assertRaisesRegex(
            GQADeviceDispatchError,
            "not checksum-approved",
        ):
            revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                **self.raw_arguments(
                    fixture,
                    b011_audit_raw=self.canonical(observation),
                    related_family_decision_raw=decision,
                )
            )

    def test_raw_cuda_events_bind_device_zero_and_one_positive_context(
        self,
    ) -> None:
        fixture = raw_replay_fixture()

        def mutate_kernel(raw: bytes, key: str, value: int) -> bytes:
            payload = json.loads(raw.decode("utf-8"))
            kernel = next(
                event
                for event in payload["traceEvents"]
                if event.get("cat") == "kernel"
            )
            kernel["args"][key] = value
            return json.dumps(payload, sort_keys=True).encode("utf-8")

        candidates = (
            {
                "gqa_raw": mutate_kernel(fixture["gqa_raw"], "device", 1),
            },
            {
                "mha_raw": mutate_kernel(fixture["mha_raw"], "context", 2),
            },
            {
                "gqa_raw": mutate_kernel(fixture["gqa_raw"], "context", 0),
            },
        )
        for updates in candidates:
            with self.subTest(updates=tuple(updates)):
                with self.assertRaises(GQADeviceDispatchError):
                    revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                        **self.raw_arguments(fixture, **updates)
                    )

    def test_unrecognized_device_like_in_marker_category_is_rejected(
        self,
    ) -> None:
        fixture = raw_replay_fixture()
        payload = json.loads(fixture["gqa_raw"].decode("utf-8"))
        payload["traceEvents"].append(
            {
                "ph": "X",
                "cat": "cuda_kernel_v2",
                "name": "hidden_device_activity",
                "ts": 20.25,
                "dur": 0.1,
                "args": {
                    "stream": 7,
                    "correlation": 43,
                    "External id": 10,
                    "device": 0,
                    "context": 1,
                },
            }
        )
        with self.assertRaises(GQADeviceDispatchError):
            revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                **self.raw_arguments(
                    fixture,
                    gqa_raw=json.dumps(payload, sort_keys=True).encode("utf-8"),
                )
            )

    def test_unknown_device_category_before_gpu_marker_is_rejected(self) -> None:
        fixture = raw_replay_fixture()
        eager_payload = json.loads(fixture["gqa_raw"].decode("utf-8"))
        eager_marker = next(
            event["name"]
            for event in eager_payload["traceEvents"]
            if event.get("cat") == "user_annotation"
        )
        graph_payload = json.loads(graph_launch_trace("graph.marker").decode("utf-8"))
        hidden = {
            "ph": "X",
            "cat": "cuda_kernel_v2",
            "name": "hidden_pre_attention_materialization",
            "ts": 17.0,
            "dur": 0.25,
            "args": {
                "stream": 7,
                "correlation": 43,
                "External id": 10,
                "device": 0,
                "context": 1,
            },
        }
        eager_payload["traceEvents"].append(dict(hidden))
        graph_payload["traceEvents"].append(dict(hidden))
        with self.assertRaisesRegex(
            ChromeTraceValidationError,
            "unrecognized category",
        ):
            parse_scoped_chrome_cuda_events(
                json.dumps(eager_payload, sort_keys=True).encode("utf-8"),
                marker=eager_marker,
                require_kernel_launch_runtime=True,
            )
        with self.assertRaisesRegex(
            ChromeTraceValidationError,
            "unrecognized category",
        ):
            parse_scoped_chrome_cuda_graph_events(
                json.dumps(graph_payload, sort_keys=True).encode("utf-8"),
                marker="graph.marker",
            )


class GeometryBoundAllocationJoinTests(unittest.TestCase):
    def test_clean_raw_bound_allocation_facts_complete_the_proof(self) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        audit = fixture["audit"]
        self.assertIsInstance(
            audit,
            Phase3GeometryBoundGQADeviceDispatchAudit,
        )
        facts = allocation_join_facts(fixture)
        result = combine_phase3_geometry_bound_gqa_allocation_verdict(
            dispatch_audit=audit,
            allocation_facts=facts,
        )
        self.assertEqual(
            result.verdict,
            GQAVerdict.NONMATERIALIZATION_VERIFIED,
        )
        self.assertTrue(result.allocation_verified)
        self.assertEqual(
            result.dispatch_verified,
            audit.evaluation.dispatch_verified,
        )
        self.assertEqual(
            result.no_replication_kernel_verified,
            audit.evaluation.no_replication_kernel_verified,
        )
        self.assertEqual(
            result.source_verified,
            audit.evaluation.source_verified,
        )
        self.assertEqual(
            result.shape_verified,
            audit.evaluation.shape_verified,
        )
        self.assertFalse(result.positive_materialization_evidence)
        self.assertRegex(facts.evidence_sha256, r"\A[0-9a-f]{64}\Z")
        self.assertEqual(
            tuple(tensor.role for tensor in facts.method_tensors),
            ("native_kv_cache_key", "native_kv_cache_value", "decode_logits"),
        )

    def test_expanded_allocation_from_raw_has_taxonomy_precedence(self) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        audit = fixture["audit"]
        self.assertIsInstance(
            audit,
            Phase3GeometryBoundGQADeviceDispatchAudit,
        )
        expanded_single = audit.gqa.byte_evidence.expanded_kv_bytes // 2
        facts = allocation_join_facts(
            fixture,
            injected_allocation_bytes=expanded_single,
        )
        expanded_events = [
            event
            for event in facts.allocation_events
            if event.requested_bytes == expanded_single
        ]
        self.assertEqual(len(expanded_events), 1)
        result = combine_phase3_geometry_bound_gqa_allocation_verdict(
            dispatch_audit=audit,
            allocation_facts=facts,
        )
        self.assertEqual(result.verdict, GQAVerdict.MATERIALIZATION_DETECTED)
        self.assertFalse(result.allocation_verified)
        self.assertTrue(result.positive_materialization_evidence)

    def test_raw_unknown_allocation_criterion_failure_remains_unproven(self) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        audit = fixture["audit"]
        self.assertIsInstance(
            audit,
            Phase3GeometryBoundGQADeviceDispatchAudit,
        )
        facts = allocation_join_facts(
            fixture,
            injected_allocation_bytes=12_288,
        )
        result = combine_phase3_geometry_bound_gqa_allocation_verdict(
            dispatch_audit=audit,
            allocation_facts=facts,
        )
        self.assertEqual(result.verdict, GQAVerdict.NONMATERIALIZATION_UNPROVEN)
        self.assertFalse(result.allocation_verified)
        self.assertFalse(result.positive_materialization_evidence)
        self.assertTrue(facts.criterion_failure_reasons)

    def test_serialized_failing_criterion_cannot_yield_verified(self) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        audit = fixture["audit"]
        self.assertIsInstance(audit, Phase3GeometryBoundGQADeviceDispatchAudit)

        def mutate(payload: dict[str, object]) -> None:
            criterion = payload["criterion"]
            assert isinstance(criterion, dict)
            criterion["passed"] = False
            criterion["failure_reasons"] = ["invented_serialized_failure"]

        facts = allocation_join_facts(fixture, mutate_audit=mutate)
        result = combine_phase3_geometry_bound_gqa_allocation_verdict(
            dispatch_audit=audit,
            allocation_facts=facts,
        )
        self.assertEqual(result.verdict, GQAVerdict.NONMATERIALIZATION_UNPROVEN)
        self.assertFalse(result.allocation_verified)
        self.assertIn(
            "serialized_criterion_derivation_mismatch",
            facts.semantic_failure_reasons,
        )

    def test_nonformal_and_failed_allocation_envelopes_are_rejected(self) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        for status in ("failed", "partial"):
            with self.subTest(status=status):
                def mutate(payload: dict[str, object]) -> None:
                    payload["evidence_status"] = status

                with self.assertRaisesRegex(
                    GQADeviceDispatchError,
                    "complete formal",
                ):
                    allocation_join_facts(fixture, mutate_audit=mutate)

    def test_join_swaps_and_mutated_fact_digest_are_rejected(self) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        audit = fixture["audit"]
        self.assertIsInstance(
            audit,
            Phase3GeometryBoundGQADeviceDispatchAudit,
        )
        with self.assertRaisesRegex(GQADeviceDispatchError, "binding differs"):
            allocation_join_facts(
                fixture,
                operation=growing_operation_key(decode_step=0),
            )
        wrong_gqa_trace = allocation_join_facts(
            fixture,
            gqa_dispatch_trace_sha256="e" * 64,
        )
        with self.assertRaises(GQADeviceDispatchError):
            combine_phase3_geometry_bound_gqa_allocation_verdict(
                dispatch_audit=audit,
                allocation_facts=wrong_gqa_trace,
            )
        mutated = allocation_join_facts(fixture)
        object.__setattr__(mutated, "allocation_audit_sha256", "f" * 64)
        with self.assertRaisesRegex(GQADeviceDispatchError, "mutated"):
            combine_phase3_geometry_bound_gqa_allocation_verdict(
                dispatch_audit=audit,
                allocation_facts=mutated,
            )

    def test_join_contract_has_no_caller_verified_boolean(self) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        facts = allocation_join_facts(fixture)
        self.assertNotIn("passed", facts.to_dict())
        self.assertNotIn("allocation_verified", facts.to_dict())
        self.assertFalse(hasattr(Phase3AllocationJoinFacts, "from_raw_derived"))
        with self.assertRaises(TypeError):
            Phase3AllocationJoinFacts(allocation_verified=True)


if __name__ == "__main__":
    unittest.main()
