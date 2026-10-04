from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import kvbench.runtime.allocation_attribution as attribution_module
from kvbench.runtime.allocation_attribution import (
    AllocationAttributionError,
    AllocationClass,
    AllocationClassPolicy,
    AllocationGeometry,
    AllocatorCounterEvidence,
    AllocatorHistoryIntegrityEvidence,
    AttributionRules,
    CanonicalFrameSelector,
    DependencyFlags,
    FailedAllocatorAuditFiles,
    MemoryDeltaEvidence,
    OperationCacheStateWitness,
    OperationOutputWitness,
    OperationWitnessCallbacks,
    OperationWitnessEvidence,
    PHASE3_BACKEND_IDENTITY,
    PartialAllocatorEvidenceFile,
    ProductionAllocationBinding,
    SplitKCompositeRawInputs,
    allocator_counters_from_memory_stats,
    allocator_snapshot_sha256,
    allocator_trace_sha256,
    attribute_allocator_trace,
    build_history_integrity_evidence,
    build_phase3_production_allocation_binding,
    collect_cuda_allocation_attribution,
    cuda_allocator_rounded_minimum,
    evaluate_refined_eager_criterion as evaluate_production_eager_criterion,
    evaluate_structural_eager_criterion_for_test as evaluate_refined_eager_criterion,
    evaluate_strict_graph_criterion,
    instantiate_decision_0009_production_rules,
    preserve_allocator_evidence,
    preserve_failed_allocator_audit,
    verify_preserved_allocator_evidence,
    verify_preserved_failed_allocator_audit,
    validate_preserved_allocator_evidence_semantically,
    verify_allocator_trace_sha256,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.schema import GraphMode, RunnerKind
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)


Trace = list[dict[str, Any]]
BACKEND = "torch-2.12.1+cu130:flash-fa2:frozen-build"


def geometry(*, context: int = 4096) -> AllocationGeometry:
    return AllocationGeometry(
        batch=1,
        query_heads=32,
        kv_heads=8,
        context=context,
        head_dim=128,
        dtype_bytes=2,
    )


def zero_memory() -> MemoryDeltaEvidence:
    return MemoryDeltaEvidence(
        allocated_before=1024,
        allocated_after=1024,
        reserved_before=2048,
        reserved_after=2048,
        device_used_before=4096,
        device_used_after=4096,
    )


def zero_memory_stats() -> dict[str, int]:
    return {
        "allocation.all.allocated": 0,
        "requested_bytes.all.allocated": 0,
        "allocated_bytes.all.allocated": 0,
        "allocation.all.freed": 0,
        "requested_bytes.all.freed": 0,
        "allocated_bytes.all.freed": 0,
        "segment.all.allocated": 0,
        "segment.all.freed": 0,
        "num_device_alloc": 0,
        "num_device_free": 0,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }


def raw_accounting(
    *,
    binding: ProductionAllocationBinding | None = None,
    sample_role: str = "before",
    timestamp_ns: int = 10,
) -> dict[str, Any]:
    selected = production_binding() if binding is None else binding
    return {
        "schema_version": "kvbench-phase3-memory-accounting-2.0.0",
        "operation_fingerprint_sha256": (
            selected.operation_fingerprint_sha256
        ),
        "sample_role": sample_role,
        "timestamp_ns": timestamp_ns,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_uuid": "GPU-test-uuid",
        "allocated_bytes": 1024,
        "reserved_bytes": 2048,
        "device_free_bytes": 4096,
        "device_total_bytes": 8192,
        "device_used_bytes": 4096,
    }


def raw_sample_arguments() -> dict[str, Any]:
    binding = production_binding()
    return {
        "memory_stats_before": zero_memory_stats(),
        "memory_stats_after": zero_memory_stats(),
        "memory_accounting_before": raw_accounting(
            binding=binding, sample_role="before", timestamp_ns=10
        ),
        "memory_accounting_after": raw_accounting(
            binding=binding, sample_role="after", timestamp_ns=20
        ),
        "operation_witness": operation_witness_payload(binding),
    }


def output_policy(
    *,
    count: int = 1,
    python_markers: tuple[str, ...] = ("decode",),
    cpp_markers: tuple[str, ...] = ("at::empty",),
) -> AllocationClassPolicy:
    size = geometry().output_bytes
    return AllocationClassPolicy(
        policy_id="phase3_test_fixed_output_v1",
        event_class=AllocationClass.FIXED_OUTPUT,
        formula_id="attention_output_geometry_bytes_v1",
        allowed_requested_bytes=frozenset({size}),
        required_python_frames=tuple(
            CanonicalFrameSelector(marker, "model.py")
            for marker in python_markers
        ),
        required_cpp_frames=tuple(
            CanonicalFrameSelector(marker, "aten/Empty.cpp")
            for marker in cpp_markers
        ),
        dependencies=DependencyFlags(True, False, True, False),
        exact_count=count,
        exact_total_requested_bytes=count * size,
    )


def output_rules(*, count: int = 1) -> AttributionRules:
    return AttributionRules(
        permitted_allocation_policies=(output_policy(count=count),)
    )


def framework_policy(*, count: int = 1) -> AllocationClassPolicy:
    return AllocationClassPolicy(
        policy_id="phase3_test_framework_bookkeeping_v1",
        event_class=AllocationClass.FRAMEWORK_BOOKKEEPING,
        formula_id="framework_scalar_exact_bytes_v1",
        allowed_requested_bytes=frozenset({16}),
        required_python_frames=(
            CanonicalFrameSelector("decode", "model.py"),
        ),
        required_cpp_frames=(
            CanonicalFrameSelector("at::empty", "aten/Empty.cpp"),
        ),
        dependencies=DependencyFlags(False, False, False, False),
        exact_count=count,
        exact_total_requested_bytes=16 * count,
    )


def split_raw_inputs() -> SplitKCompositeRawInputs:
    return SplitKCompositeRawInputs.from_raw_bytes(
        gqa_dispatch_trace=b"gqa-dispatch",
        mha_dispatch_trace=b"mha-dispatch",
        gqa_allocator_control=b"gqa-allocator",
        mha_allocator_control=b"mha-allocator",
        split_k_pair_multiplicity=((2, 1),),
    )


def audit_operation_key(
    *,
    run_id: str = "phase3-remediation-test-run",
    runner_kind: str = "fixed_l",
    execution_mode: str = "cuda_graph",
    batch: int = 1,
    starting_context: int = 128,
    decode_step: int = 0,
    process_replicate: int = 1,
) -> Phase3AuditOperationKey:
    graph_mode = GraphMode(execution_mode)
    runner = RunnerKind(runner_kind)
    output_steps = 1 if runner is RunnerKind.FIXED_L else 16
    point = Phase3ProcessPoint(
        point_id=(
            f"{runner.value}-b{batch}-l{starting_context}-"
            f"{graph_mode.value}-r{process_replicate}"
        ),
        runner_kind=runner,
        graph_mode=graph_mode,
        batch_size=batch,
        context_length=starting_context,
        output_steps=output_steps,
        process_replicate=process_replicate,
        stability_member=(
            runner is RunnerKind.FIXED_L
            and batch == 1
            and starting_context == 4096
            and process_replicate > 1
        ),
    )
    _, cache_fingerprint = (
        attribution_module._phase3_cache_layout_fingerprint(
            runner_kind=runner_kind,
            batch=batch,
            starting_context=starting_context,
        )
    )
    plan_path = (
        PHASE3_FIXED_PLAN_PATH
        if runner is RunnerKind.FIXED_L
        else PHASE3_GROWING_PLAN_PATH
    )
    return Phase3AuditOperationKey.from_point(
        run_id=run_id,
        point=point,
        decode_step=decode_step,
        cache_layout_fingerprint=cache_fingerprint,
        execution_git_sha="1" * 40,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[plan_path],
        hardware_identity_sha256="2" * 64,
        software_identity_sha256="3" * 64,
        model_identity_sha256="4" * 64,
        backend_identity_sha256=hashlib.sha256(
            PHASE3_BACKEND_IDENTITY.encode("utf-8")
        ).hexdigest(),
        source_identity_sha256="5" * 64,
    )


def production_binding(
    *,
    run_id: str = "phase3-remediation-test-run",
    runner_kind: str = "fixed_l",
    execution_mode: str = "cuda_graph",
    batch: int = 1,
    starting_context: int = 128,
    decode_step: int = 0,
    process_replicate: int = 1,
    split_k_raw_inputs: SplitKCompositeRawInputs | None = None,
) -> ProductionAllocationBinding:
    if split_k_raw_inputs is None and execution_mode == "eager":
        split_k_raw_inputs = split_raw_inputs()
    return build_phase3_production_allocation_binding(
        operation_key=audit_operation_key(
            run_id=run_id,
            runner_kind=runner_kind,
            execution_mode=execution_mode,
            batch=batch,
            starting_context=starting_context,
            decode_step=decode_step,
            process_replicate=process_replicate,
        ),
        backend_identity=PHASE3_BACKEND_IDENTITY,
        split_k_raw_inputs=split_k_raw_inputs,
    )


def operation_witness_evidence(
    binding: ProductionAllocationBinding | None = None,
    *,
    measured_shape: tuple[int, ...] = (1, 1, 128_256),
    measured_sha256: str | None = None,
) -> OperationWitnessEvidence:
    selected = production_binding() if binding is None else binding
    shape = (32, selected.batch, 8, selected.cache_capacity, 128)
    strides = (
        selected.batch * 8 * selected.cache_capacity * 128,
        8 * selected.cache_capacity * 128,
        selected.cache_capacity * 128,
        128,
        1,
    )
    prefix_sha = hashlib.sha256(b"fake-cache-prefix").hexdigest()
    before = OperationCacheStateWitness(
        active_length=selected.historical_context,
        key_shape=shape,
        value_shape=shape,
        key_strides=strides,
        value_strides=strides,
        key_dtype="torch.bfloat16",
        value_dtype="torch.bfloat16",
        key_device="cuda:0",
        value_device="cuda:0",
        key_data_ptr=0x1000,
        value_data_ptr=0x2000,
        historical_prefix_sha256=prefix_sha,
        destination_slot_sha256=(
            attribution_module._phase3_zero_destination_sentinel_sha256(
                selected
            )
        ),
        destination_slot_is_sentinel=True,
        layout_fingerprint=selected.cache_layout_fingerprint,
    )
    after = OperationCacheStateWitness(
        active_length=(
            selected.historical_context
            if selected.runner_kind == "fixed_l"
            else selected.attended_context
        ),
        key_shape=shape,
        value_shape=shape,
        key_strides=strides,
        value_strides=strides,
        key_dtype="torch.bfloat16",
        value_dtype="torch.bfloat16",
        key_device="cuda:0",
        value_device="cuda:0",
        key_data_ptr=0x1000,
        value_data_ptr=0x2000,
        historical_prefix_sha256=prefix_sha,
        destination_slot_sha256=hashlib.sha256(
            b"fake-cache-written"
        ).hexdigest(),
        destination_slot_is_sentinel=False,
        layout_fingerprint=selected.cache_layout_fingerprint,
    )
    output = OperationOutputWitness(
        sha256=hashlib.sha256(b"fake-output").hexdigest(),
        shape=measured_shape,
        dtype="torch.bfloat16",
        device="cuda:0",
        finite=True,
    )
    return OperationWitnessEvidence(
        operation_key=selected.operation_key,
        operation_fingerprint_sha256=(
            selected.operation_fingerprint_sha256
        ),
        reference_before=before,
        reference_after=after,
        reference_output=output,
        measured_before=before,
        measured_after=after,
        measured_output=replace(
            output,
            sha256=(
                output.sha256
                if measured_sha256 is None
                else measured_sha256
            ),
        ),
        recorder_configuration=dict(
            attribution_module.PHASE3_RECORDER_CONFIGURATION
        ),
    )


def operation_witness_payload(
    binding: ProductionAllocationBinding | None = None,
) -> dict[str, Any]:
    return operation_witness_evidence(binding).to_dict()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def failed_audit_files(
    staging: Path,
) -> tuple[dict[str, Any], FailedAllocatorAuditFiles]:
    payload = json.loads(
        (staging / "allocation_audit_failed.json").read_text(
            encoding="utf-8"
        )
    )
    digest = (
        staging / "allocation_audit_failed.sha256"
    ).read_text(encoding="ascii").split()[0]
    files = FailedAllocatorAuditFiles(
        audit_file="allocation_audit_failed.json",
        audit_sha256_file="allocation_audit_failed.sha256",
        audit_sha256=digest,
        partial_files=tuple(
            PartialAllocatorEvidenceFile(
                evidence_key=item["evidence_key"],
                file=item["file"],
                sha256=item["sha256"],
            )
            for item in payload["partial_files"]
        ),
    )
    return payload, files


def rewrite_bound_raw_evidence(
    staging: Path,
    files: attribution_module.RawAllocatorEvidenceFiles,
    *,
    file_field: str,
    digest_field: str,
    payload: bytes,
) -> attribution_module.RawAllocatorEvidenceFiles:
    raw_name = getattr(files, file_field)
    (staging / raw_name).write_bytes(payload)
    rebound = replace(
        files,
        **{digest_field: hashlib.sha256(payload).hexdigest()},
    )
    audit_path = staging / rebound.audit_file
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["raw_files"] = rebound.to_dict(include_audit_sha256=False)
    audit_bytes = canonical_json_bytes(audit)
    audit_digest = hashlib.sha256(audit_bytes).hexdigest()
    audit_path.write_bytes(audit_bytes)
    (staging / rebound.audit_sha256_file).write_text(
        f"{audit_digest}  {rebound.audit_file}\n",
        encoding="ascii",
    )
    return replace(rebound, audit_sha256=audit_digest)


def rewrite_audit_payload(
    staging: Path,
    files: attribution_module.RawAllocatorEvidenceFiles,
    mutate: Any,
) -> attribution_module.RawAllocatorEvidenceFiles:
    audit_path = staging / files.audit_file
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    mutate(audit)
    audit_bytes = canonical_json_bytes(audit)
    audit_digest = hashlib.sha256(audit_bytes).hexdigest()
    audit_path.write_bytes(audit_bytes)
    (staging / files.audit_sha256_file).write_text(
        f"{audit_digest}  {files.audit_file}\n",
        encoding="ascii",
    )
    return replace(files, audit_sha256=audit_digest)


def alloc_lifetime(
    size: int,
    *,
    address: int = 0x2000,
    stream: int = 7,
    python_stack: list[dict[str, Any]] | None = None,
    cpp_stack: list[dict[str, Any]] | None = None,
    allocated_block_size: int | None = None,
) -> Trace:
    alloc: dict[str, Any] = {
        "action": "alloc",
        "addr": address,
        "size": size,
        "stream": stream,
        "python_stack": (
            [{"name": "decode", "filename": "model.py", "line": 10}]
            if python_stack is None
            else python_stack
        ),
        "cpp_stack": (
            [{"name": "at::empty", "filename": "aten/Empty.cpp", "line": 1}]
            if cpp_stack is None
            else cpp_stack
        ),
    }
    if allocated_block_size is not None:
        alloc["allocated_block_size"] = allocated_block_size
    return [
        alloc,
        {
            "action": "free_requested",
            "addr": address,
            "size": size,
            "stream": stream,
        },
        {
            "action": "free_completed",
            "addr": address,
            "size": size,
            "stream": stream,
        },
    ]


def counters_for(
    trace: Trace,
    *,
    allocation_count: int | None = None,
    requested_bytes: int | None = None,
    allocated_block_bytes: int | None = None,
    device_allocation_count: int | None = None,
    device_free_count: int | None = None,
) -> AllocatorCounterEvidence:
    sizes = [
        event["size"]
        for event in trace
        if event.get("action") == "alloc"
        and isinstance(event.get("size"), int)
    ]
    freed_sizes = [
        event["size"]
        for event in trace
        if event.get("action") == "free_completed"
        and isinstance(event.get("size"), int)
    ]
    segment_allocations = sum(
        event.get("action") == "segment_alloc" for event in trace
    )
    segment_frees = sum(
        event.get("action") == "segment_free" for event in trace
    )
    return AllocatorCounterEvidence(
        allocation_count=(
            len(sizes) if allocation_count is None else allocation_count
        ),
        requested_bytes=(
            sum(sizes) if requested_bytes is None else requested_bytes
        ),
        allocated_block_bytes=(
            sum(cuda_allocator_rounded_minimum(size) for size in sizes)
            if allocated_block_bytes is None
            else allocated_block_bytes
        ),
        device_allocation_count=(
            segment_allocations
            if device_allocation_count is None
            else device_allocation_count
        ),
        device_free_count=(
            segment_frees
            if device_free_count is None
            else device_free_count
        ),
        free_count=len(freed_sizes),
        freed_requested_bytes=sum(freed_sizes),
        freed_block_bytes=sum(
            cuda_allocator_rounded_minimum(size) for size in freed_sizes
        ),
        segment_allocation_count=segment_allocations,
        segment_free_count=segment_frees,
        allocation_retry_count=sum(
            event.get("action") == "alloc_retry" for event in trace
        ),
        oom_count=sum(event.get("action") == "oom" for event in trace),
    )


def attribution(
    trace: Trace,
    *,
    selected_geometry: AllocationGeometry | None = None,
    rules: AttributionRules | None = None,
    selected_counters: AllocatorCounterEvidence | None = None,
    expected_digest: str | None = None,
    backend_identity: str = BACKEND,
):
    trace_digest = allocator_trace_sha256(trace)
    snapshot_digest = "a" * 64
    return attribute_allocator_trace(
        trace,
        geometry=selected_geometry or geometry(),
        counters=selected_counters or counters_for(trace),
        rules=output_rules() if rules is None else rules,
        backend_identity=backend_identity,
        expected_trace_sha256=(
            trace_digest
            if expected_digest is None
            else expected_digest
        ),
        history_integrity=AllocatorHistoryIntegrityEvidence(
            stack_mode="all",
            ring_capacity=max(64, len(trace) + 1),
            observed_trace_entries=len(trace),
            raw_snapshot_sha256=snapshot_digest,
            expected_raw_snapshot_sha256=snapshot_digest,
            raw_trace_sha256=trace_digest,
            expected_raw_trace_sha256=trace_digest,
        ),
    )


def split_cpp_stack() -> list[dict[str, Any]]:
    return [
        {
            "name": "pytorch_flash::set_params_splitkv",
            "filename": "flash_api.cpp",
            "line": 1,
        },
        {
            "name": "pytorch_flash::mha_fwd",
            "filename": "flash_api.cpp",
            "line": 2,
        },
        {
            "name": "at::native::_scaled_dot_product_flash_attention_cuda",
            "filename": "attention.cu",
            "line": 3,
        },
    ]


class AllocationFormulaTests(unittest.TestCase):
    def test_expanded_and_native_kv_formulas_are_single_and_combined(self) -> None:
        value = geometry()
        self.assertEqual(value.output_bytes, 8192)
        self.assertEqual(value.expanded_kv_single_bytes, 33_554_432)
        self.assertEqual(value.expanded_kv_combined_bytes, 67_108_864)
        self.assertEqual(value.native_kv_single_bytes, 8_388_608)
        self.assertEqual(value.native_kv_combined_bytes, 16_777_216)
        self.assertEqual(
            value.flash_split_k_output_accumulator_bytes(32), 524_288
        )
        self.assertEqual(value.flash_split_k_lse_bytes(32), 4096)

    def test_attention_output_and_full_operation_output_are_distinct(self) -> None:
        value = AllocationGeometry(
            batch=1,
            query_heads=32,
            kv_heads=8,
            context=4096,
            head_dim=128,
            dtype_bytes=2,
            operation_output_width=32_000,
            operation_output_dtype_bytes=4,
        )
        self.assertEqual(value.output_bytes, 8192)
        self.assertEqual(value.operation_output_bytes, 128_000)
        assert value.operation_output_bytes is not None
        policy = AllocationClassPolicy(
            policy_id="phase3_test_operation_output_v1",
            event_class=AllocationClass.FIXED_OUTPUT,
            formula_id="operation_output_geometry_bytes_v1",
            allowed_requested_bytes=frozenset(
                {value.operation_output_bytes}
            ),
            required_python_frames=(
                CanonicalFrameSelector("decode", "model.py"),
            ),
            required_cpp_frames=(
                CanonicalFrameSelector("at::empty", "aten/Empty.cpp"),
            ),
            dependencies=DependencyFlags(True, False, False, False),
            exact_count=1,
            exact_total_requested_bytes=value.operation_output_bytes,
        )
        parsed = attribution(
            alloc_lifetime(value.operation_output_bytes),
            selected_geometry=value,
            rules=AttributionRules(
                permitted_allocation_policies=(policy,)
            ),
        )
        item = parsed.allocations[0]
        self.assertEqual(item.event_class, AllocationClass.FIXED_OUTPUT)
        self.assertEqual(
            item.size_formula, "operation_output_geometry_bytes_v1"
        )
        self.assertEqual(
            item.dependencies,
            DependencyFlags(True, False, False, False),
        )
        self.assertEqual(
            item.to_dict()["dependencies"],
            {
                "batch": True,
                "context": False,
                "query_heads": False,
                "kv_heads": False,
            },
        )
        self.assertTrue(
            evaluate_refined_eager_criterion(
                parsed, zero_memory()
            ).passed
        )

    def test_formula_specific_shared_dependencies_are_serialized(self) -> None:
        value = geometry()
        projection = AllocationClassPolicy(
            policy_id="phase3_test_kv_projection_v1",
            event_class=AllocationClass.FIXED_SHARED_ACTIVATION,
            formula_id="kv_projection_geometry_bytes_v1",
            allowed_requested_bytes=frozenset({value.kv_projection_bytes}),
            required_python_frames=(
                CanonicalFrameSelector("decode", "model.py"),
            ),
            required_cpp_frames=(
                CanonicalFrameSelector("at::empty", "aten/Empty.cpp"),
            ),
            dependencies=DependencyFlags(True, False, False, True),
            exact_count=1,
            exact_total_requested_bytes=value.kv_projection_bytes,
        )
        lse = AllocationClassPolicy(
            policy_id="phase3_test_flash_lse_v1",
            event_class=AllocationClass.FIXED_SHARED_ACTIVATION,
            formula_id="flash_lse_geometry_bytes_v1",
            allowed_requested_bytes=frozenset({value.flash_lse_bytes}),
            required_python_frames=(
                CanonicalFrameSelector("decode", "model.py"),
            ),
            required_cpp_frames=(
                CanonicalFrameSelector("at::empty", "aten/Empty.cpp"),
            ),
            dependencies=DependencyFlags(True, False, True, False),
            exact_count=1,
            exact_total_requested_bytes=value.flash_lse_bytes,
        )
        self.assertEqual(
            projection.to_dict()["dependencies"]["kv_heads"], True
        )
        self.assertEqual(
            lse.to_dict()["dependencies"]["query_heads"], True
        )
        with self.assertRaises(AllocationAttributionError):
            replace(
                projection,
                dependencies=DependencyFlags(True, False, True, False),
            )

    def test_output_sized_allocation_is_attributed_and_eager_permitted(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        parsed = attribution(trace)
        item = parsed.allocations[0]
        self.assertEqual(item.event_class, AllocationClass.FIXED_OUTPUT)
        self.assertEqual(item.allocated_block_bytes, 8192)
        self.assertTrue(item.allocated_block_size_proven)
        self.assertEqual(item.stream, 7)
        self.assertTrue(item.python_stack)
        self.assertTrue(item.cpp_stack)
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(
            result.criterion_id, "phase3_eager_structural_test_only_v1"
        )
        self.assertTrue(result.no_context_dependent_allocation)
        self.assertNotIn("zero_allocation", result.to_dict())

    def test_output_size_without_frozen_provenance_policy_is_unknown(self) -> None:
        parsed = attribution(
            alloc_lifetime(geometry().output_bytes),
            rules=AttributionRules(),
        )
        self.assertEqual(
            parsed.allocations[0].event_class, AllocationClass.UNKNOWN
        )
        self.assertFalse(
            evaluate_refined_eager_criterion(parsed, zero_memory()).passed
        )

    def test_permitted_policy_requires_formula_and_both_stack_provenances(self) -> None:
        with self.assertRaises(AllocationAttributionError):
            AllocationClassPolicy(
                policy_id="bad",
                event_class=AllocationClass.FIXED_OUTPUT,
                formula_id="self_asserted_formula",
                allowed_requested_bytes=frozenset({geometry().output_bytes}),
                required_python_frames=(
                    CanonicalFrameSelector("decode", "model.py"),
                ),
                required_cpp_frames=(
                    CanonicalFrameSelector(
                        "at::empty", "aten/Empty.cpp"
                    ),
                ),
                dependencies=DependencyFlags(True, False, True, False),
                exact_count=1,
                exact_total_requested_bytes=geometry().output_bytes,
            )
        with self.assertRaises(AllocationAttributionError):
            output_policy(cpp_markers=())

        trace = alloc_lifetime(
            geometry().output_bytes,
            cpp_stack=[{"name": "unrelated", "filename": "other.cpp"}],
        )
        parsed = attribution(trace, rules=output_rules())
        self.assertEqual(
            parsed.allocations[0].event_class, AllocationClass.UNKNOWN
        )

    def test_permitted_policy_exact_count_and_byte_bounds_are_enforced(self) -> None:
        one = alloc_lifetime(geometry().output_bytes)
        missing = attribution(one, rules=output_rules(count=2))
        missing_result = evaluate_refined_eager_criterion(
            missing, zero_memory()
        )
        self.assertIn(
            "allocation_policy_count_bound_failed:"
            "phase3_test_fixed_output_v1",
            missing_result.failure_reasons,
        )
        self.assertIn(
            "allocation_policy_byte_bound_failed:"
            "phase3_test_fixed_output_v1",
            missing_result.failure_reasons,
        )

        two = one + alloc_lifetime(
            geometry().output_bytes, address=0x3000
        )
        excess = attribution(two, rules=output_rules(count=1))
        excess_result = evaluate_refined_eager_criterion(
            excess, zero_memory()
        )
        self.assertIn(
            "allocation_class_count_bound_failed:fixed_output",
            excess_result.failure_reasons,
        )
        self.assertIn(
            "allocation_class_byte_bound_failed:fixed_output",
            excess_result.failure_reasons,
        )

    def test_single_and_combined_expanded_kv_are_positive_failures(self) -> None:
        value = geometry()
        trace = alloc_lifetime(value.expanded_kv_single_bytes, address=0x1000)
        trace += alloc_lifetime(
            value.expanded_kv_combined_bytes, address=0x2000
        )
        parsed = attribution(trace)
        self.assertEqual(
            [item.event_class for item in parsed.allocations],
            [AllocationClass.GQA_EXPANSION, AllocationClass.GQA_EXPANSION],
        )
        self.assertEqual(
            [item.size_formula for item in parsed.allocations],
            ["expanded_kv_single", "expanded_kv_combined"],
        )
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "forbidden_allocation_class:gqa_expansion",
            result.failure_reasons,
        )

    def test_proven_non_attention_output_retains_class_on_size_collision(self) -> None:
        base = geometry()
        value = replace(
            base,
            operation_output_width=(
                base.expanded_kv_single_bytes // 4
            ),
            operation_output_dtype_bytes=4,
        )
        assert value.operation_output_bytes is not None
        policy = AllocationClassPolicy(
            policy_id="phase3_test_collision_output_v1",
            event_class=AllocationClass.FIXED_OUTPUT,
            formula_id="operation_output_geometry_bytes_v1",
            allowed_requested_bytes=frozenset(
                {value.operation_output_bytes}
            ),
            required_python_frames=(
                CanonicalFrameSelector("decode", "model.py"),
            ),
            required_cpp_frames=(
                CanonicalFrameSelector("at::empty", "aten/Empty.cpp"),
            ),
            dependencies=DependencyFlags(True, False, False, False),
            exact_count=1,
            exact_total_requested_bytes=value.operation_output_bytes,
        )
        parsed = attribution(
            alloc_lifetime(value.operation_output_bytes),
            selected_geometry=value,
            rules=AttributionRules(
                permitted_allocation_policies=(policy,)
            ),
        )
        self.assertEqual(
            parsed.allocations[0].event_class,
            AllocationClass.FIXED_OUTPUT,
        )

        attention_trace = alloc_lifetime(
            value.operation_output_bytes,
            cpp_stack=[
                {
                    "name": "at::empty",
                    "filename": "aten/Empty.cpp",
                },
                {
                    "name": "flash_attention_copy",
                    "filename": "attention.cu",
                },
            ],
        )
        attention_parsed = attribution(
            attention_trace,
            selected_geometry=value,
            rules=AttributionRules(
                permitted_allocation_policies=(policy,)
            ),
        )
        self.assertEqual(
            attention_parsed.allocations[0].event_class,
            AllocationClass.GQA_EXPANSION,
        )

    def test_audit_stack_marker_cannot_mask_expanded_kv_size(self) -> None:
        trace = alloc_lifetime(
            geometry().expanded_kv_single_bytes,
            python_stack=[
                {
                    "name": "allocation_audit",
                    "filename": "audit.py",
                }
            ],
        )
        parsed = attribution(trace)
        self.assertEqual(
            parsed.allocations[0].event_class,
            AllocationClass.GQA_EXPANSION,
        )

    def test_outer_endpoint_audit_wrapper_is_unknown_not_instrumentation(
        self,
    ) -> None:
        trace = alloc_lifetime(
            12_345,
            python_stack=[
                {
                    "name": "collect_phase3_endpoint_allocation_audit",
                    "filename": (
                        "/workspace/src/kvbench/runtime/"
                        "phase3_endpoint_audit.py"
                    ),
                    "line": 972,
                }
            ],
        )
        parsed = attribution(trace, rules=AttributionRules())
        self.assertEqual(
            parsed.allocations[0].event_class,
            AllocationClass.UNKNOWN,
        )
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "forbidden_allocation_class:unknown",
            result.failure_reasons,
        )

    def test_exact_audit_instrumentation_frame_requires_name_and_source(
        self,
    ) -> None:
        canonical = {
            "name": "_allocation_audit_capture_output",
            "filename": (
                "/workspace/src/kvbench/runtime/"
                "allocation_attribution.py"
            ),
            "line": 2054,
        }
        parsed = attribution(
            alloc_lifetime(12_345, python_stack=[canonical]),
            rules=AttributionRules(),
        )
        self.assertEqual(
            parsed.allocations[0].event_class,
            AllocationClass.AUDIT_INSTRUMENTATION,
        )
        for spoof in (
            {**canonical, "name": "allocation_audit_capture_output"},
            {**canonical, "filename": "/workspace/spoof.py"},
        ):
            with self.subTest(spoof=spoof):
                spoofed = attribution(
                    alloc_lifetime(12_345, python_stack=[spoof]),
                    rules=AttributionRules(),
                )
                self.assertEqual(
                    spoofed.allocations[0].event_class,
                    AllocationClass.UNKNOWN,
                )

        with self.assertRaises(AllocationAttributionError):
            AttributionRules(audit_instrumentation_python_frames=())

    def test_outer_endpoint_audit_wrapper_does_not_preempt_output_policy(
        self,
    ) -> None:
        trace = alloc_lifetime(
            geometry().output_bytes,
            python_stack=[
                {"name": "decode", "filename": "model.py", "line": 10},
                {
                    "name": "collect_phase3_endpoint_allocation_audit",
                    "filename": (
                        "src/kvbench/runtime/phase3_endpoint_audit.py"
                    ),
                    "line": 972,
                },
            ],
        )
        parsed = attribution(trace, rules=output_rules())
        self.assertEqual(
            parsed.allocations[0].event_class,
            AllocationClass.FIXED_OUTPUT,
        )

    def test_outer_endpoint_audit_wrapper_does_not_preempt_split_k_formula(
        self,
    ) -> None:
        value = geometry()
        trace = alloc_lifetime(
            value.flash_split_k_output_accumulator_bytes(32),
            python_stack=[
                {
                    "name": "collect_phase3_endpoint_allocation_audit",
                    "filename": (
                        "src/kvbench/runtime/phase3_endpoint_audit.py"
                    ),
                }
            ],
            cpp_stack=split_cpp_stack(),
        )
        parsed = attribution(
            trace,
            rules=AttributionRules(
                frozen_backend_identity=BACKEND,
                split_k_expected_pair_count=1,
            ),
        )
        item = parsed.allocations[0]
        self.assertEqual(
            item.event_class,
            AllocationClass.CONTEXT_SCALED_WORKSPACE,
        )
        self.assertEqual(
            item.size_formula,
            "flash_split_k_output_accumulator",
        )
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "flash_split_k_independent_verifier_unavailable",
            result.failure_reasons,
        )

    def test_native_kv_sized_allocation_is_cache_growth(self) -> None:
        trace = alloc_lifetime(geometry().native_kv_combined_bytes)
        parsed = attribution(trace)
        self.assertEqual(
            parsed.allocations[0].event_class,
            AllocationClass.CACHE_GROWTH,
        )
        self.assertFalse(
            evaluate_refined_eager_criterion(parsed, zero_memory()).passed
        )

    def test_exact_flash_split_k_pair_fails_closed_without_raw_verifier(self) -> None:
        value = geometry()
        stack = split_cpp_stack()
        trace = alloc_lifetime(
            value.flash_split_k_output_accumulator_bytes(32),
            address=0x1000,
            cpp_stack=stack,
        )
        trace += alloc_lifetime(
            value.flash_split_k_lse_bytes(32),
            address=0x2000,
            cpp_stack=stack,
        )
        rules = AttributionRules(
            frozen_backend_identity=BACKEND,
            split_k_expected_pair_count=1,
        )
        parsed = attribution(trace, rules=rules)
        self.assertEqual(
            parsed.class_counts(), {"context_scaled_workspace": 2}
        )
        self.assertEqual(
            {
                dict(item.formula_parameters)["num_splits"]
                for item in parsed.allocations
            },
            {32},
        )
        self.assertTrue(
            all(
                item.dependencies
                == DependencyFlags(True, True, True, False)
                for item in parsed.allocations
            )
        )
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "flash_split_k_independent_verifier_unavailable",
            result.failure_reasons,
        )
        self.assertFalse(result.no_context_dependent_allocation)
        self.assertFalse(result.fully_attributed_bounded_ephemeral)

    def test_split_k_exact_pair_multiplicity_is_enforced(self) -> None:
        value = geometry()
        trace = alloc_lifetime(
            value.flash_split_k_output_accumulator_bytes(32),
            address=0x1000,
            cpp_stack=split_cpp_stack(),
        )
        trace += alloc_lifetime(
            value.flash_split_k_lse_bytes(32),
            address=0x2000,
            cpp_stack=split_cpp_stack(),
        )
        trace += alloc_lifetime(
            value.flash_split_k_output_accumulator_bytes(32),
            address=0x3000,
            cpp_stack=split_cpp_stack(),
        )
        trace += alloc_lifetime(
            value.flash_split_k_lse_bytes(32),
            address=0x4000,
            cpp_stack=split_cpp_stack(),
        )
        parsed = attribution(
            trace,
            rules=AttributionRules(
                frozen_backend_identity=BACKEND,
                split_k_expected_pair_count=1,
            ),
        )
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "flash_split_k_workspace_pair_mismatch", result.failure_reasons
        )

    def test_split_k_size_without_complete_stack_is_unknown(self) -> None:
        value = geometry()
        trace = alloc_lifetime(
            value.flash_split_k_output_accumulator_bytes(32),
            cpp_stack=[{"name": "pytorch_flash::mha_fwd"}],
        )
        parsed = attribution(
            trace,
            rules=AttributionRules(),
        )
        self.assertEqual(
            parsed.allocations[0].event_class, AllocationClass.UNKNOWN
        )
        self.assertEqual(
            parsed.allocations[0].size_formula,
            "flash_split_k_output_accumulator_stack_unverified",
        )
        self.assertFalse(
            evaluate_refined_eager_criterion(parsed, zero_memory()).passed
        )

    def test_python_stack_cannot_spoof_split_k_cpp_markers(self) -> None:
        value = geometry()
        python_stack = [
            {
                "name": marker,
                "filename": "spoof.py",
                "line": index,
            }
            for index, marker in enumerate(
                attribution_module.FLASH_SPLIT_K_CPP_MARKERS, start=1
            )
        ]
        trace = alloc_lifetime(
            value.flash_split_k_output_accumulator_bytes(32),
            python_stack=python_stack,
            cpp_stack=[{"name": "at::empty", "filename": "Empty.cpp"}],
        )
        parsed = attribution(trace, rules=AttributionRules())
        item = parsed.allocations[0]
        self.assertEqual(item.event_class, AllocationClass.UNKNOWN)
        self.assertIsNone(item.size_formula)

    def test_split_k_requires_independent_controls_and_formula_pair(self) -> None:
        value = geometry()
        trace = alloc_lifetime(
            value.flash_split_k_lse_bytes(32),
            cpp_stack=split_cpp_stack(),
        )
        parsed = attribution(
            trace,
            rules=AttributionRules(
                frozen_backend_identity=BACKEND,
                split_k_expected_pair_count=1,
            ),
        )
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "flash_split_k_independent_verifier_unavailable",
            result.failure_reasons,
        )
        self.assertIn(
            "flash_split_k_workspace_pair_mismatch", result.failure_reasons
        )


class AllocationLifecycleTests(unittest.TestCase):
    def test_unknown_event_fails_despite_zero_net_growth(self) -> None:
        trace = alloc_lifetime(12_345)
        parsed = attribution(trace)
        self.assertEqual(
            parsed.allocations[0].event_class, AllocationClass.UNKNOWN
        )
        self.assertEqual(parsed.allocations[0].allocated_block_bytes, 12_800)
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "forbidden_allocation_class:unknown", result.failure_reasons
        )

    def test_segment_alloc_is_device_allocation_and_not_cache_reuse(self) -> None:
        trace: Trace = [
            {
                "action": "segment_alloc",
                "addr": 0x1000,
                "size": 65536,
                "stream": 7,
            },
            *alloc_lifetime(geometry().output_bytes, address=0x2000),
            {
                "action": "segment_free",
                "addr": 0x1000,
                "size": 65536,
                "stream": 7,
            },
        ]
        parsed = attribution(trace)
        item = parsed.allocations[0]
        self.assertTrue(item.triggered_segment_alloc)
        self.assertFalse(item.reused_from_cache)
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn("segment_alloc_detected", result.failure_reasons)
        self.assertIn(
            "device_allocation_detected_or_unavailable",
            result.failure_reasons,
        )
        self.assertIn(
            "allocation_not_reused_from_cache", result.failure_reasons
        )

    def test_only_first_allocation_is_causally_bound_to_new_segment(self) -> None:
        size = geometry().output_bytes
        trace: Trace = [
            {
                "action": "segment_alloc",
                "addr": 0x1000,
                "size": 131_072,
                "stream": 7,
            },
            *alloc_lifetime(size, address=0x2000),
            *alloc_lifetime(size, address=0x6000),
            {
                "action": "segment_free",
                "addr": 0x1000,
                "size": 131_072,
                "stream": 7,
            },
        ]
        parsed = attribution(trace)
        self.assertEqual(
            [item.triggered_segment_alloc for item in parsed.allocations],
            [True, False],
        )
        self.assertEqual(parsed.segment_alloc_count, 1)

    def test_overlapping_trace_segments_fail_integrity(self) -> None:
        trace: Trace = [
            {
                "action": "segment_alloc",
                "addr": 0x1000,
                "size": 4096,
                "stream": 7,
            },
            {
                "action": "segment_alloc",
                "addr": 0x1800,
                "size": 4096,
                "stream": 7,
            },
        ]
        parsed = attribution(trace)
        self.assertTrue(
            any(
                error.startswith("overlapping_segment_alloc")
                for error in parsed.integrity_errors
            )
        )

    def test_segment_free_with_live_allocation_fails_integrity(self) -> None:
        size = geometry().output_bytes
        trace: Trace = [
            {
                "action": "segment_alloc",
                "addr": 0x1000,
                "size": 65536,
                "stream": 7,
            },
            alloc_lifetime(size, address=0x2000)[0],
            {
                "action": "segment_free",
                "addr": 0x1000,
                "size": 65536,
                "stream": 7,
            },
            *alloc_lifetime(size, address=0x2000)[1:],
        ]
        parsed = attribution(trace)
        self.assertTrue(
            any(
                error.startswith("segment_free_with_live_allocation")
                for error in parsed.integrity_errors
            )
        )

    def test_allocation_crossing_trace_segment_boundary_fails(self) -> None:
        trace: Trace = [
            {
                "action": "segment_alloc",
                "addr": 0x1000,
                "size": 4096,
                "stream": 7,
            },
            *alloc_lifetime(512, address=0x1F00),
            {
                "action": "segment_free",
                "addr": 0x1000,
                "size": 4096,
                "stream": 7,
            },
        ]
        parsed = attribution(trace)
        self.assertTrue(
            any(
                error.startswith("allocation_exceeds_trace_segment")
                for error in parsed.integrity_errors
            )
        )

    def test_missing_or_invalid_allocation_stream_fails_completeness(self) -> None:
        for event_index, invalid_stream in (
            (0, None),
            (0, -1),
            (1, True),
            (2, "stream-7"),
        ):
            with self.subTest(
                event_index=event_index, invalid_stream=invalid_stream
            ):
                trace = alloc_lifetime(geometry().output_bytes)
                if invalid_stream is None:
                    del trace[event_index]["stream"]
                else:
                    trace[event_index]["stream"] = invalid_stream
                parsed = attribution(trace)
                self.assertTrue(
                    any(
                        error.startswith(
                            "allocator_event_stream_missing_or_invalid"
                        )
                        for error in parsed.integrity_errors
                    )
                )
                self.assertFalse(
                    evaluate_refined_eager_criterion(
                        parsed, zero_memory()
                    ).passed
                )

        parsed_segment = attribution(
            [
                {
                    "action": "segment_alloc",
                    "addr": 0x1000,
                    "size": 4096,
                }
            ]
        )
        self.assertTrue(
            any(
                error.startswith(
                    "allocator_event_stream_missing_or_invalid"
                )
                for error in parsed_segment.integrity_errors
            )
        )

    def test_rapid_address_reuse_creates_distinct_lifetimes(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes, address=0x2000)
        trace += alloc_lifetime(16, address=0x2000)
        parsed = attribution(
            trace,
            rules=AttributionRules(
                permitted_allocation_policies=(
                    output_policy(),
                    framework_policy(),
                )
            ),
        )
        self.assertEqual(len(parsed.allocations), 2)
        self.assertEqual(
            [item.allocation_id for item in parsed.allocations], [0, 1]
        )
        self.assertEqual(
            [item.alloc_event_index for item in parsed.allocations], [0, 3]
        )
        self.assertTrue(parsed.all_lifetimes_fully_freed)

    def test_free_without_alloc_and_free_ordering_fail_closed(self) -> None:
        unmatched: Trace = [
            {"action": "free_requested", "addr": 0x1000, "size": 8}
        ]
        parsed = attribution(unmatched)
        self.assertTrue(
            any(
                error.startswith("free_without_matching_alloc")
                for error in parsed.integrity_errors
            )
        )

        out_of_order = alloc_lifetime(geometry().output_bytes)
        del out_of_order[1]
        parsed = attribution(out_of_order)
        self.assertTrue(
            any(
                error.startswith("free_completed_before_free_requested")
                for error in parsed.integrity_errors
            )
        )
        self.assertFalse(parsed.all_lifetimes_fully_freed)

    def test_alloc_before_prior_free_completion_fails(self) -> None:
        size = geometry().output_bytes
        trace: Trace = [
            {"action": "alloc", "addr": 0x1000, "size": size},
            {"action": "alloc", "addr": 0x1000, "size": size},
            {"action": "free_requested", "addr": 0x1000, "size": size},
            {"action": "free_completed", "addr": 0x1000, "size": size},
        ]
        parsed = attribution(trace)
        self.assertTrue(
            any(
                error.startswith("alloc_before_prior_free_completed")
                for error in parsed.integrity_errors
            )
        )
        self.assertIn(
            "allocator_counter_allocation_count_mismatch",
            parsed.integrity_errors,
        )

    def test_overlapping_live_allocation_ranges_fail(self) -> None:
        first = alloc_lifetime(1024, address=0x1000)
        second = alloc_lifetime(1024, address=0x1200)
        trace = [first[0], second[0], *first[1:], *second[1:]]
        parsed = attribution(trace)
        self.assertTrue(
            any(
                error.startswith("overlapping_live_allocation_range")
                for error in parsed.integrity_errors
            )
        )

    def test_cache_reuse_is_unverified_when_device_counter_is_missing(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        counters = replace(
            counters_for(trace), device_allocation_count=None
        )
        parsed = attribution(trace, selected_counters=counters)
        self.assertIsNone(parsed.allocations[0].reused_from_cache)
        self.assertEqual(
            parsed.allocations[0].to_dict()["cache_reuse_status"],
            "unverified",
        )
        self.assertFalse(parsed.all_allocations_cache_reused)

    def test_requested_and_block_counter_tamper_fail_closed(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        bad_requested = counters_for(trace, requested_bytes=8193)
        parsed = attribution(trace, selected_counters=bad_requested)
        self.assertIn(
            "allocator_counter_requested_bytes_mismatch",
            parsed.integrity_errors,
        )

        unresolved = counters_for(trace, allocated_block_bytes=8704)
        parsed = attribution(trace, selected_counters=unresolved)
        self.assertIn("allocated_block_sizes_unresolved", parsed.integrity_errors)
        self.assertIsNone(parsed.allocations[0].allocated_block_bytes)
        self.assertFalse(
            evaluate_refined_eager_criterion(parsed, zero_memory()).passed
        )

    def test_nullable_counter_and_invalid_explicit_block_fail_closed(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        missing = AllocatorCounterEvidence(1, 8192, None, 0, 0)
        parsed = attribution(trace, selected_counters=missing)
        self.assertFalse(parsed.all_block_sizes_proven)
        self.assertFalse(
            evaluate_refined_eager_criterion(parsed, zero_memory()).passed
        )

        trace = alloc_lifetime(
            geometry().output_bytes, allocated_block_size=512
        )
        invalid = counters_for(trace, allocated_block_bytes=512)
        parsed = attribution(trace, selected_counters=invalid)
        self.assertTrue(
            any(
                error.startswith("invalid_allocated_block_size")
                for error in parsed.integrity_errors
            )
        )

    def test_trace_digest_detects_mutation(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        digest = allocator_trace_sha256(trace)
        self.assertTrue(verify_allocator_trace_sha256(trace, digest))
        tampered = copy.deepcopy(trace)
        tampered[0]["stream"] = 99
        self.assertFalse(verify_allocator_trace_sha256(tampered, digest))
        parsed = attribution(tampered, expected_digest=digest)
        self.assertIn(
            "allocator_trace_sha256_mismatch", parsed.integrity_errors
        )


class ProductionPolicyTrustTests(unittest.TestCase):
    def test_caller_selected_policy_cannot_pass_production(self) -> None:
        binding = production_binding(execution_mode="eager")
        parsed = attribution(
            alloc_lifetime(binding.geometry.output_bytes),
            selected_geometry=binding.geometry,
            rules=output_rules(),
            backend_identity=binding.backend_identity,
        )
        structural = evaluate_refined_eager_criterion(
            parsed, zero_memory()
        )
        self.assertTrue(structural.passed, structural.failure_reasons)
        production = evaluate_production_eager_criterion(
            parsed,
            zero_memory(),
            production_binding=binding,
        )
        self.assertFalse(production.passed)
        self.assertIn(
            "production_policy_rules_mismatch",
            production.failure_reasons,
        )

    def test_exact_flash_output_policy_precedes_partial_split_heuristic(
        self,
    ) -> None:
        binding = production_binding(execution_mode="eager")
        rules = instantiate_decision_0009_production_rules(binding)
        trace = alloc_lifetime(
            binding.geometry.output_bytes,
            python_stack=[
                {
                    "name": "flash_attention_forward",
                    "filename": "/workspace/src/kvbench/runtime/backend.py",
                }
            ],
            cpp_stack=[
                {
                    "name": "pytorch_flash::mha_fwd",
                    "filename": "flash_api.cpp",
                },
                {
                    "name": (
                        "at::native::_scaled_dot_product_flash_attention_cuda"
                    ),
                    "filename": "attention.cu",
                },
                {
                    "name": (
                        "torch::autograd::"
                        "THPVariable_scaled_dot_product_attention"
                        "(_object*, _object*, _object*)"
                    ),
                    "filename": "python_nn_functions.cpp",
                },
            ],
            allocated_block_size=binding.geometry.output_bytes,
        )
        parsed = attribution(
            trace,
            selected_geometry=binding.geometry,
            rules=rules,
            backend_identity=binding.backend_identity,
        )
        item = parsed.allocations[0]
        self.assertEqual(
            item.event_class, AllocationClass.FIXED_SHARED_ACTIVATION
        )
        self.assertEqual(
            item.policy_id,
            "phase3_decision_0013_flash_output_v1",
        )

    def test_production_zero_event_point_fails_catalog_bounds(self) -> None:
        binding = production_binding(execution_mode="eager")
        self.assertIsNotNone(binding.split_k_raw_inputs)
        rules = instantiate_decision_0009_production_rules(binding)
        parsed = attribution(
            [],
            selected_geometry=binding.geometry,
            rules=rules,
            backend_identity=binding.backend_identity,
        )
        result = evaluate_production_eager_criterion(
            parsed,
            zero_memory(),
            production_binding=binding,
        )
        self.assertFalse(result.passed)
        self.assertIn(
            "allocation_policy_count_bound_failed:"
            "phase3_decision_0013_embedding_hidden_v1",
            result.failure_reasons,
        )
        self.assertIn(
            "flash_split_k_workspace_pair_missing",
            result.failure_reasons,
        )
        self.assertEqual(
            result.criterion_id,
            "phase3_eager_attributed_ephemeral_v1",
        )

    def test_split_k_raw_inputs_are_required_by_catalog(self) -> None:
        raw = SplitKCompositeRawInputs.from_raw_bytes(
            gqa_dispatch_trace=b"gqa-dispatch",
            mha_dispatch_trace=b"mha-dispatch",
            gqa_allocator_control=b"gqa-allocator",
            mha_allocator_control=b"mha-allocator",
            split_k_pair_multiplicity=((2, 1),),
        )
        self.assertTrue(raw.raw_bytes_verified)
        self.assertIsNotNone(raw.raw_composite_sha256)
        binding = production_binding(
            run_id="raw-split-input-test",
            runner_kind="fixed_l",
            execution_mode="eager",
            batch=1,
            starting_context=128,
            decode_step=0,
            process_replicate=1,
            split_k_raw_inputs=raw,
        )
        self.assertEqual(binding.split_k_raw_inputs, raw)
        self.assertEqual(binding.validation_errors(), ())
        self.assertEqual(
            binding.operation_fingerprint_sha256,
            binding.operation_key.operation_fingerprint_sha256,
        )
        self.assertEqual(
            binding.external_provenance_status,
            "external_run_join_unverified",
        )
        self.assertEqual(binding.geometry.operation_output_width, 128_256)
        self.assertEqual(binding.geometry.operation_output_dtype_bytes, 2)
        with self.assertRaisesRegex(
            AllocationAttributionError,
            "production_binding_split_k_raw_unverified",
        ):
            build_phase3_production_allocation_binding(
                operation_key=audit_operation_key(
                    run_id="missing-split-input-test",
                    execution_mode="eager",
                ),
                backend_identity=PHASE3_BACKEND_IDENTITY,
            )

    def test_production_binding_tamper_and_non_grid_point_are_rejected(self) -> None:
        binding = production_binding(execution_mode="eager")
        self.assertEqual(
            binding.point_id, "fixed_l-b1-l128-eager-r1"
        )
        self.assertEqual(
            binding.point_fingerprint,
            "1f441644857acc1b801654bfa702b835b2b17ef26f0de5e623c0da32309868c0",
        )
        tampered = replace(
            binding, operation_fingerprint_sha256="0" * 64
        )
        self.assertIn(
            "production_binding_operation_fingerprint_mismatch",
            tampered.validation_errors(),
        )
        self.assertIn(
            "production_binding_cache_fingerprint_mismatch",
            replace(
                binding, cache_layout_fingerprint="0" * 64
            ).validation_errors(),
        )
        with self.assertRaises(AllocationAttributionError):
            build_phase3_production_allocation_binding(
                operation_key=audit_operation_key(
                    run_id="caller-backend",
                    execution_mode="eager",
                ),
                backend_identity=BACKEND,
                split_k_raw_inputs=split_raw_inputs(),
            )
        with self.assertRaises(ValueError):
            audit_operation_key(
                run_id="bad-grid",
                execution_mode="eager",
                batch=2,
            )

    def test_production_binding_rejects_cross_run_operation_key(self) -> None:
        binding = production_binding(execution_mode="eager")
        other_key = audit_operation_key(
            run_id="phase3-remediation-other-run",
            execution_mode="eager",
        )
        errors = replace(binding, operation_key=other_key).validation_errors()
        self.assertIn(
            "production_binding_operation_key_mismatch:run_id", errors
        )
        self.assertIn(
            "production_binding_operation_key_mismatch:"
            "operation_fingerprint_sha256",
            errors,
        )
        self.assertIn(
            "production_binding_external_provenance_status_invalid",
            replace(
                binding, external_provenance_status="verified"
            ).validation_errors(),
        )


class AllocationCriterionTests(unittest.TestCase):
    def test_eager_positive_raw_growth_fails(self) -> None:
        parsed = attribution(alloc_lifetime(geometry().output_bytes))
        memory = MemoryDeltaEvidence(100, 101, 200, 200, 300, 300)
        result = evaluate_refined_eager_criterion(parsed, memory)
        self.assertFalse(result.passed)
        self.assertIn("persistent_allocated_growth", result.failure_reasons)

    def test_eager_benign_negative_raw_deltas_pass_when_residual_is_zero(self) -> None:
        parsed = attribution(alloc_lifetime(geometry().output_bytes))
        memory = MemoryDeltaEvidence(100, 90, 200, 150, 300, 250)
        self.assertEqual(memory.non_pytorch_delta, 0)
        result = evaluate_refined_eager_criterion(parsed, memory)
        self.assertTrue(result.passed, result.failure_reasons)

    def test_eager_nonzero_residual_of_either_sign_fails(self) -> None:
        parsed = attribution(alloc_lifetime(geometry().output_bytes))
        for device_after in (260, 240):
            with self.subTest(device_after=device_after):
                memory = MemoryDeltaEvidence(
                    100, 90, 200, 150, 300, device_after
                )
                self.assertNotEqual(memory.non_pytorch_delta, 0)
                result = evaluate_refined_eager_criterion(parsed, memory)
                self.assertFalse(result.passed)
                self.assertIn(
                    "persistent_non_pytorch_delta_nonzero",
                    result.failure_reasons,
                )

    def test_missing_device_evidence_never_passes(self) -> None:
        parsed = attribution(alloc_lifetime(geometry().output_bytes))
        memory = MemoryDeltaEvidence(100, 100, 200, 200, None, None)
        result = evaluate_refined_eager_criterion(parsed, memory)
        self.assertFalse(result.passed)
        self.assertIn("device_used_delta_unavailable", result.failure_reasons)
        self.assertIn("non_pytorch_delta_unavailable", result.failure_reasons)

    def test_graph_requires_exact_zero_and_rejects_fixed_output(self) -> None:
        parsed = attribution(alloc_lifetime(geometry().output_bytes))
        result = evaluate_strict_graph_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertEqual(
            result.criterion_id, "phase3_graph_zero_allocation_v1"
        )
        self.assertIn("graph_allocation_event_detected", result.failure_reasons)

    def test_graph_empty_trace_passes_but_negative_delta_fails(self) -> None:
        trace: Trace = []
        parsed = attribution(trace)
        passing = evaluate_strict_graph_criterion(parsed, zero_memory())
        self.assertTrue(passing.passed, passing.failure_reasons)
        self.assertTrue(passing.strict_graph_zero_events)

        negative = MemoryDeltaEvidence(100, 90, 200, 150, 300, 250)
        failing = evaluate_strict_graph_criterion(parsed, negative)
        self.assertFalse(failing.passed)
        self.assertIn("graph_allocated_delta_nonzero", failing.failure_reasons)
        self.assertIn("graph_reserved_delta_nonzero", failing.failure_reasons)
        self.assertIn("graph_device_used_delta_nonzero", failing.failure_reasons)


class AllocationHistoryIntegrityTests(unittest.TestCase):
    def test_ring_saturation_fails_evaluation(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        snapshot = {"device_traces": [trace], "segments": []}
        history = build_history_integrity_evidence(
            snapshot,
            trace,
            max_entries=len(trace),
        )
        self.assertTrue(history.ring_saturated)
        parsed = replace(attribution(trace), history_integrity=history)
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "allocator_history_ring_saturated", result.failure_reasons
        )

    def test_incomplete_python_or_cpp_stack_fails(self) -> None:
        trace = alloc_lifetime(
            geometry().output_bytes,
            python_stack=[],
        )
        parsed = attribution(trace)
        self.assertFalse(parsed.allocations[0].python_stack)
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "allocator_allocation_stack_incomplete", result.failure_reasons
        )

    def test_native_python_c_api_frame_is_not_python_source_proof(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        del trace[0]["python_stack"]
        del trace[0]["cpp_stack"]
        trace[0]["frames"] = [
            {"name": "torch::CapturedTraceback::gather", "filename": "??"},
            {"name": "PyCMethod_New", "filename": "??"},
        ]
        parsed = attribution(trace)
        self.assertFalse(parsed.allocations[0].python_stack)
        self.assertEqual(len(parsed.allocations[0].cpp_stack), 2)
        result = evaluate_refined_eager_criterion(parsed, zero_memory())
        self.assertFalse(result.passed)
        self.assertIn(
            "allocator_allocation_stack_incomplete", result.failure_reasons
        )

    def test_raw_snapshot_digest_mismatch_fails(self) -> None:
        parsed = attribution(alloc_lifetime(geometry().output_bytes))
        self.assertIsNotNone(parsed.history_integrity)
        assert parsed.history_integrity is not None
        bad_history = replace(
            parsed.history_integrity,
            expected_raw_snapshot_sha256="b" * 64,
        )
        result = evaluate_refined_eager_criterion(
            replace(parsed, history_integrity=bad_history), zero_memory()
        )
        self.assertFalse(result.passed)
        self.assertIn(
            "raw_allocator_snapshot_sha256_mismatch",
            result.failure_reasons,
        )

    def test_memory_stats_derive_complete_cumulative_counters(self) -> None:
        keys = {
            "allocation.all.allocated": 10,
            "requested_bytes.all.allocated": 100,
            "allocated_bytes.all.allocated": 512,
            "allocation.all.freed": 7,
            "requested_bytes.all.freed": 70,
            "allocated_bytes.all.freed": 3584,
            "segment.all.allocated": 2,
            "segment.all.freed": 1,
            "num_device_alloc": 2,
            "num_device_free": 1,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }
        after = dict(keys)
        increments = {
            "allocation.all.allocated": 1,
            "requested_bytes.all.allocated": 8192,
            "allocated_bytes.all.allocated": 8192,
            "allocation.all.freed": 1,
            "requested_bytes.all.freed": 8192,
            "allocated_bytes.all.freed": 8192,
        }
        for key, increment in increments.items():
            after[key] += increment
        counters = allocator_counters_from_memory_stats(keys, after)
        self.assertTrue(counters.complete)
        self.assertEqual(counters.allocation_count, 1)
        self.assertEqual(counters.free_count, 1)
        self.assertEqual(counters.requested_bytes, 8192)
        self.assertEqual(counters.freed_block_bytes, 8192)
        self.assertEqual(counters.device_allocation_count, 0)

        decreasing = dict(after)
        decreasing["allocation.all.allocated"] = 9
        incomplete = allocator_counters_from_memory_stats(keys, decreasing)
        self.assertIsNone(incomplete.allocation_count)
        self.assertFalse(incomplete.complete)

    def test_raw_evidence_is_no_replace_and_tamper_detectable(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        snapshot = {"device_traces": [trace], "segments": []}
        snapshot_digest = allocator_snapshot_sha256(snapshot)
        trace_digest = allocator_trace_sha256(trace)
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            files = preserve_allocator_evidence(
                staging,
                snapshot=snapshot,
                trace=trace,
                **raw_sample_arguments(),
                expected_snapshot_sha256=snapshot_digest,
                expected_trace_sha256=trace_digest,
                audit_payload={"passed": False, "timing_reported": False},
            )
            self.assertTrue(
                verify_preserved_allocator_evidence(staging, files)
            )
            self.assertIsNotNone(files.audit_sha256)
            self.assertEqual(len(files.audit_sha256 or ""), 64)
            self.assertTrue((staging / files.audit_file).is_file())
            self.assertTrue((staging / files.audit_sha256_file).is_file())
            audit_document = json.loads(
                (staging / files.audit_file).read_text(encoding="utf-8")
            )
            self.assertEqual(
                audit_document["raw_files"],
                files.to_dict(include_audit_sha256=False),
            )
            original = (staging / files.snapshot_file).read_bytes()
            with self.assertRaises(FileExistsError):
                preserve_allocator_evidence(
                    staging,
                    snapshot=snapshot,
                    trace=trace,
                    **raw_sample_arguments(),
                    expected_snapshot_sha256=snapshot_digest,
                    expected_trace_sha256=trace_digest,
                    audit_payload={"passed": True},
                )
            self.assertEqual(
                (staging / files.snapshot_file).read_bytes(), original
            )
            audit_original = (staging / files.audit_file).read_bytes()
            (staging / files.audit_file).write_bytes(b"{}")
            self.assertFalse(
                verify_preserved_allocator_evidence(staging, files)
            )
            (staging / files.audit_file).write_bytes(audit_original)
            self.assertTrue(
                verify_preserved_allocator_evidence(staging, files)
            )
            (staging / files.trace_file).write_bytes(b"{}")
            self.assertFalse(
                verify_preserved_allocator_evidence(staging, files)
            )

    def test_missing_audit_json_fails_preserved_evidence_verification(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        snapshot = {"device_traces": [trace], "segments": []}
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            files = preserve_allocator_evidence(
                staging,
                snapshot=snapshot,
                trace=trace,
                **raw_sample_arguments(),
                expected_snapshot_sha256=allocator_snapshot_sha256(snapshot),
                expected_trace_sha256=allocator_trace_sha256(trace),
                audit_payload={"evidence_status": "complete"},
            )
            (staging / files.audit_file).unlink()
            self.assertFalse(
                verify_preserved_allocator_evidence(staging, files)
            )

    def test_prewrite_tamper_rejection_leaves_staging_empty(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        snapshot = {"device_traces": [trace], "segments": []}
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            with self.assertRaises(AllocationAttributionError):
                preserve_allocator_evidence(
                    staging,
                    snapshot=snapshot,
                    trace=trace,
                    **raw_sample_arguments(),
                    expected_snapshot_sha256="0" * 64,
                    expected_trace_sha256=allocator_trace_sha256(trace),
                    audit_payload={},
                )
            self.assertEqual(list(staging.iterdir()), [])

    def test_audit_raw_file_reference_mismatch_is_rejected_prewrite(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        snapshot = {"device_traces": [trace], "segments": []}
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            with self.assertRaises(AllocationAttributionError):
                preserve_allocator_evidence(
                    staging,
                    snapshot=snapshot,
                    trace=trace,
                    **raw_sample_arguments(),
                    expected_snapshot_sha256=(
                        allocator_snapshot_sha256(snapshot)
                    ),
                    expected_trace_sha256=allocator_trace_sha256(trace),
                    audit_payload={
                        "raw_files": {"trace_sha256": "0" * 64}
                    },
                )
            self.assertEqual(list(staging.iterdir()), [])

    def test_semantic_validator_rejects_trace_snapshot_divergence(self) -> None:
        snapshot = {"device_traces": [[]], "segments": []}
        trace = alloc_lifetime(geometry(context=128).output_bytes)
        binding = production_binding(execution_mode="cuda_graph")
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            files = preserve_allocator_evidence(
                staging,
                snapshot=snapshot,
                trace=trace,
                **raw_sample_arguments(),
                expected_snapshot_sha256=allocator_snapshot_sha256(snapshot),
                expected_trace_sha256=allocator_trace_sha256(trace),
                audit_payload={
                    "device_index": 0,
                    "max_history_entries": 64,
                    "post_warmup_prepare_used": True,
                    "production_binding": binding.to_dict(),
                    "production_binding_sha256": binding.identity_sha256,
                },
            )
            result = validate_preserved_allocator_evidence_semantically(
                staging,
                files,
                production_binding=binding,
            )
            self.assertFalse(result.passed)
            self.assertTrue(result.failure_reasons)

    def test_symlink_staging_directory_is_rejected(self) -> None:
        trace = alloc_lifetime(geometry().output_bytes)
        snapshot = {"device_traces": [trace], "segments": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(AllocationAttributionError):
                preserve_allocator_evidence(
                    link,
                    snapshot=snapshot,
                    trace=trace,
                    **raw_sample_arguments(),
                    expected_snapshot_sha256=(
                        allocator_snapshot_sha256(snapshot)
                    ),
                    expected_trace_sha256=allocator_trace_sha256(trace),
                    audit_payload={},
                )


class OperationWitnessHardeningTests(unittest.TestCase):
    def test_same_shaped_measured_payload_mismatch_is_detected(self) -> None:
        evidence = operation_witness_evidence()
        assert evidence.measured_output is not None
        tampered = replace(
            evidence,
            measured_output=replace(
                evidence.measured_output, sha256="f" * 64
            ),
        )
        self.assertIn(
            "operation_witness_measured_output_mismatch",
            tampered.validation_errors(production_binding()),
        )

    def test_toy_logits_and_fake_cache_geometry_cannot_pass(self) -> None:
        binding = production_binding()
        evidence = operation_witness_evidence(binding)
        toy_output = replace(
            evidence.reference_output, shape=(binding.batch, 1, 16)
        )
        output_errors = replace(
            evidence,
            reference_output=toy_output,
            measured_output=toy_output,
        ).validation_errors(binding)
        self.assertIn(
            "operation_witness_output_geometry_mismatch", output_errors
        )

        fake_shape = (1, binding.batch, 8, binding.cache_capacity, 128)
        fake_before = replace(
            evidence.reference_before,
            key_shape=fake_shape,
            value_shape=fake_shape,
        )
        cache_errors = replace(
            evidence,
            reference_before=fake_before,
            measured_before=fake_before,
        ).validation_errors(binding)
        self.assertIn(
            "operation_witness_reference_before_cache_shape_mismatch",
            cache_errors,
        )
        self.assertIn(
            "operation_witness_reference_before_cache_layout_mismatch",
            cache_errors,
        )

    def test_cache_prefix_pointer_stride_dtype_and_sentinel_are_rederived(
        self,
    ) -> None:
        binding = production_binding()
        evidence = operation_witness_evidence(binding)
        bad_after = replace(
            evidence.measured_after,
            key_strides=(1, 1, 1, 1, 1),
            key_dtype="torch.float16",
            key_data_ptr=0x3000,
            historical_prefix_sha256="e" * 64,
            destination_slot_is_sentinel=True,
        )
        errors = replace(
            evidence, measured_after=bad_after
        ).validation_errors(binding)
        self.assertIn(
            "operation_witness_measured_after_cache_strides_mismatch", errors
        )
        self.assertIn(
            "operation_witness_measured_after_cache_dtype_mismatch", errors
        )
        self.assertIn("operation_witness_cache_pointers_changed", errors)
        self.assertIn("operation_witness_historical_prefix_changed", errors)
        self.assertIn(
            "operation_witness_measured_destination_not_written", errors
        )
        bad_before = replace(
            evidence.reference_before,
            destination_slot_sha256="d" * 64,
        )
        sentinel_errors = replace(
            evidence,
            reference_before=bad_before,
            measured_before=bad_before,
        ).validation_errors(binding)
        self.assertIn(
            "operation_witness_reference_destination_sentinel_mismatch",
            sentinel_errors,
        )

    def test_lane_specific_active_length_is_enforced(self) -> None:
        fixed = production_binding()
        fixed_evidence = operation_witness_evidence(fixed)
        self.assertEqual(fixed_evidence.validation_errors(fixed), ())
        self.assertEqual(
            fixed_evidence.reference_after.active_length,
            fixed.historical_context,
        )
        growing = production_binding(
            run_id="phase3-remediation-growing-witness",
            runner_kind="growing_context",
            execution_mode="eager",
            decode_step=5,
            split_k_raw_inputs=split_raw_inputs(),
        )
        growing_evidence = operation_witness_evidence(growing)
        self.assertEqual(growing_evidence.validation_errors(growing), ())
        self.assertEqual(
            growing_evidence.reference_after.active_length,
            growing.attended_context,
        )


class _FakeDevice:
    type = "cuda"
    index = 0

    def __str__(self) -> str:
        return "cuda:0"


class _FakeMemoryHistory:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.enabled_calls: list[str | None] = []
        self.recording = False
        self.snapshot_error: Exception | None = None
        self.disable_error: Exception | None = None

    def _record_memory_history(self, **kwargs: Any) -> None:
        enabled = kwargs.get("enabled")
        self.enabled_calls.append(enabled)
        if enabled is None:
            if self.disable_error is not None:
                raise self.disable_error
            self.recording = False
        else:
            self.recording = True

    def _snapshot(self, **kwargs: Any) -> dict[str, Any]:
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return self.snapshot


class _FakeCuda:
    def __init__(self) -> None:
        self.memory = _FakeMemoryHistory(
            {"device_traces": [[]], "segments": []}
        )
        self.stats = {
            "allocation.all.allocated": 0,
            "requested_bytes.all.allocated": 0,
            "allocated_bytes.all.allocated": 0,
            "allocation.all.freed": 0,
            "requested_bytes.all.freed": 0,
            "allocated_bytes.all.freed": 0,
            "segment.all.allocated": 0,
            "segment.all.freed": 0,
            "num_device_alloc": 0,
            "num_device_free": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }
        self.synchronize_calls = 0
        self.fail_synchronize_call: int | None = None
        self.memory_stats_calls = 0
        self.fail_memory_stats_call: int | None = None
        self.malformed_memory_stats_call: int | None = None
        self.memory_allocated_calls = 0
        self.fail_memory_allocated_call: int | None = None

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, index: int) -> Any:
        if index != 0:
            raise RuntimeError("unexpected fake CUDA device")
        return type(
            "FakeCudaProperties",
            (),
            {"uuid": "GPU-test-uuid", "total_memory": 8192},
        )()

    def synchronize(self, **kwargs: Any) -> None:
        self.synchronize_calls += 1
        if self.synchronize_calls == self.fail_synchronize_call:
            raise RuntimeError("injected synchronize failure")
        return None

    def memory_allocated(self, **kwargs: Any) -> int:
        self.memory_allocated_calls += 1
        if self.memory_allocated_calls == self.fail_memory_allocated_call:
            raise RuntimeError("injected memory-accounting failure")
        return 1024

    def memory_reserved(self, **kwargs: Any) -> int:
        return 2048

    def mem_get_info(self, **kwargs: Any) -> tuple[int, int]:
        return (6144, 8192)

    def memory_stats(self, **kwargs: Any) -> Any:
        self.memory_stats_calls += 1
        if self.memory_stats_calls == self.fail_memory_stats_call:
            raise RuntimeError("injected memory-stats failure")
        if self.memory_stats_calls == self.malformed_memory_stats_call:
            return []
        return dict(self.stats)


class _FakeVersion:
    cuda = "13.0"


class _FakeTorch:
    __version__ = "2.12.1+cu130"
    version = _FakeVersion()

    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def device(self, value: Any) -> _FakeDevice:
        return _FakeDevice()


class _FakeOutput:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (1, 1, 128_256),
        dtype: str = "torch.bfloat16",
        device: str = "cuda:0",
        payload: bytes = b"fake-output",
        finite: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.payload = payload
        self.finite = finite


class _FakeOperationHarness:
    def __init__(self, binding: ProductionAllocationBinding) -> None:
        self.binding = binding
        self.active_length = binding.historical_context
        self.destination_label = b"fake-cache-sentinel"
        self.destination_is_sentinel = True
        self.operation_calls = 0
        self.prepare_calls = 0
        self.next_shape: tuple[int, ...] = (
            binding.batch,
            1,
            128_256,
        )
        self.next_payload = b"fake-output"

    def prepare(self) -> None:
        self.prepare_calls += 1
        self.active_length = self.binding.historical_context
        self.destination_label = b"fake-cache-sentinel"
        self.destination_is_sentinel = True

    def operation(self) -> _FakeOutput:
        self.operation_calls += 1
        self.active_length = (
            self.binding.historical_context
            if self.binding.runner_kind == "fixed_l"
            else self.binding.attended_context
        )
        self.destination_label = b"fake-cache-written"
        self.destination_is_sentinel = False
        return _FakeOutput(
            shape=self.next_shape,
            payload=self.next_payload,
        )

    def capture_cache_state(self) -> OperationCacheStateWitness:
        shape = (
            32,
            self.binding.batch,
            8,
            self.binding.cache_capacity,
            128,
        )
        strides = (
            self.binding.batch * 8 * self.binding.cache_capacity * 128,
            8 * self.binding.cache_capacity * 128,
            self.binding.cache_capacity * 128,
            128,
            1,
        )
        return OperationCacheStateWitness(
            active_length=self.active_length,
            key_shape=shape,
            value_shape=shape,
            key_strides=strides,
            value_strides=strides,
            key_dtype="torch.bfloat16",
            value_dtype="torch.bfloat16",
            key_device="cuda:0",
            value_device="cuda:0",
            key_data_ptr=0x1000,
            value_data_ptr=0x2000,
            historical_prefix_sha256=hashlib.sha256(
                b"fake-cache-prefix"
            ).hexdigest(),
            destination_slot_sha256=hashlib.sha256(
                self.destination_label
            ).hexdigest()
            if not self.destination_is_sentinel
            else (
                attribution_module._phase3_zero_destination_sentinel_sha256(
                    self.binding
                )
            ),
            destination_slot_is_sentinel=(
                self.destination_is_sentinel
            ),
            layout_fingerprint=self.binding.cache_layout_fingerprint,
        )

    def capture_output(self, value: Any) -> OperationOutputWitness:
        shape, dtype, device = attribution_module._operation_output_metadata(
            value
        )
        return OperationOutputWitness(
            sha256=hashlib.sha256(value.payload).hexdigest(),
            shape=shape,
            dtype=dtype,
            device=device,
            finite=value.finite,
        )

    @property
    def callbacks(self) -> OperationWitnessCallbacks:
        return OperationWitnessCallbacks(
            capture_cache_state=self.capture_cache_state,
            capture_output=self.capture_output,
        )


class AllocationCollectorTests(unittest.TestCase):
    def _collect_graph_success(
        self, fake: _FakeTorch, staging: Path
    ) -> attribution_module.CollectedAllocationAudit:
        binding = production_binding(execution_mode="cuda_graph")
        harness = _FakeOperationHarness(binding)
        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            return collect_cuda_allocation_attribution(
                harness.operation,
                production_binding=binding,
                staging_directory=staging,
                operation_witness=harness.callbacks,
                prepare_operation=harness.prepare,
                warmup_iterations=3,
                max_entries=100_000,
            )
        finally:
            attribution_module._TORCH = previous

    def _collect_expected_failure(
        self,
        fake: _FakeTorch,
        staging: Path,
        *,
        expected_stage: str,
        operation: Any = None,
        warmup_operation: Any = None,
    ) -> tuple[dict[str, Any], FailedAllocatorAuditFiles]:
        binding = production_binding(execution_mode="cuda_graph")
        harness = _FakeOperationHarness(binding)
        selected_operation = (
            harness.operation if operation is None else operation
        )
        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with self.assertRaises(AllocationAttributionError):
                collect_cuda_allocation_attribution(
                    selected_operation,
                    production_binding=binding,
                    staging_directory=staging,
                    operation_witness=harness.callbacks,
                    warmup_operation=warmup_operation,
                    prepare_operation=harness.prepare,
                    warmup_iterations=3,
                    max_entries=100_000,
                )
        finally:
            attribution_module._TORCH = previous
        payload, files = failed_audit_files(staging)
        self.assertEqual(payload["failure_stage"], expected_stage)
        self.assertTrue(
            verify_preserved_failed_allocator_audit(staging, files)
        )
        partial_keys = {
            item.evidence_key for item in files.partial_files
        }
        snapshot_present = "snapshot" in partial_keys
        self.assertIs(
            payload["raw_snapshot_available"], snapshot_present
        )
        self.assertIs(
            payload["raw_snapshot_preserved"], snapshot_present
        )
        return payload, files

    def test_growing_context_requires_and_records_post_warmup_prepare(
        self,
    ) -> None:
        binding = production_binding(
            run_id="phase3-remediation-growing-test",
            runner_kind="growing_context",
            execution_mode="eager",
            batch=1,
            starting_context=128,
            decode_step=5,
            process_replicate=1,
            split_k_raw_inputs=split_raw_inputs(),
        )
        self.assertEqual(
            binding.point_id,
            "growing_context-b1-l128-eager-r1",
        )
        self.assertEqual(binding.historical_context, 133)
        self.assertEqual(binding.attended_context, 134)
        self.assertEqual(binding.geometry.context, 134)
        self.assertEqual(binding.cache_capacity, 144)
        fake = _FakeTorch()
        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(AllocationAttributionError):
                    collect_cuda_allocation_attribution(
                        lambda: None,
                        production_binding=binding,
                        staging_directory=Path(temporary),
                        operation_witness=_FakeOperationHarness(
                            binding
                        ).callbacks,
                    )
                payload, files = failed_audit_files(Path(temporary))
                self.assertEqual(payload["failure_stage"], "input_validation")
                self.assertFalse(payload["prepare_present"])
                self.assertFalse(payload["prepare_attempted"])
                self.assertFalse(payload["prepare_completed"])
                self.assertTrue(
                    verify_preserved_failed_allocator_audit(
                        Path(temporary), files
                    )
                )

            order: list[str] = []
            harness = _FakeOperationHarness(binding)

            def operation() -> _FakeOutput:
                order.append("operation")
                return harness.operation()

            def warmup() -> _FakeOutput:
                order.append("warmup")
                return harness.operation()

            def prepare() -> None:
                order.append("prepare")
                harness.prepare()

            with tempfile.TemporaryDirectory() as temporary:
                result = collect_cuda_allocation_attribution(
                    operation,
                    production_binding=binding,
                    staging_directory=Path(temporary),
                    operation_witness=harness.callbacks,
                    warmup_operation=warmup,
                    prepare_operation=prepare,
                    warmup_iterations=3,
                    max_entries=100_000,
                )
                self.assertEqual(
                    order,
                    [
                        "warmup",
                        "warmup",
                        "warmup",
                        "prepare",
                        "operation",
                        "prepare",
                        "operation",
                    ],
                )
                self.assertTrue(result.prepare_present)
                self.assertTrue(result.prepare_attempted)
                self.assertTrue(result.prepare_completed)
                self.assertEqual(result.prepare_attempt_count, 2)
                self.assertEqual(result.prepare_completion_count, 2)
                self.assertFalse(result.criterion.passed)
                self.assertIn(
                    "allocation_policy_count_bound_failed:"
                    "phase3_decision_0013_embedding_hidden_v1",
                    result.criterion.failure_reasons,
                )
        finally:
            attribution_module._TORCH = previous

    def test_growing_prepare_failure_is_checksum_bound(self) -> None:
        binding = production_binding(
            run_id="phase3-remediation-growing-prepare-failure",
            runner_kind="growing_context",
            execution_mode="eager",
            batch=1,
            starting_context=128,
            decode_step=0,
            process_replicate=1,
            split_k_raw_inputs=split_raw_inputs(),
        )

        def fail_prepare() -> None:
            raise RuntimeError("injected growing prepare failure")

        fake = _FakeTorch()
        harness = _FakeOperationHarness(binding)
        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary)
                with self.assertRaises(AllocationAttributionError):
                    collect_cuda_allocation_attribution(
                        lambda: None,
                        production_binding=binding,
                        staging_directory=staging,
                        operation_witness=harness.callbacks,
                        warmup_operation=lambda: None,
                        prepare_operation=fail_prepare,
                        warmup_iterations=3,
                        max_entries=100_000,
                    )
                payload, files = failed_audit_files(staging)
                self.assertEqual(
                    payload["failure_stage"], "operation_witness_prepare"
                )
                self.assertTrue(payload["prepare_present"])
                self.assertTrue(payload["prepare_attempted"])
                self.assertFalse(payload["prepare_completed"])
                self.assertEqual(payload["prepare_attempt_count"], 1)
                self.assertEqual(payload["prepare_completion_count"], 0)
                self.assertTrue(
                    verify_preserved_failed_allocator_audit(
                        staging, files
                    )
                )
        finally:
            attribution_module._TORCH = previous

    def test_second_prepare_failure_records_attempt_without_completion(
        self,
    ) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        harness = _FakeOperationHarness(binding)

        def fail_second_prepare() -> None:
            harness.prepare()
            if harness.prepare_calls == 2:
                raise RuntimeError("injected second prepare failure")

        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary)
                with self.assertRaises(AllocationAttributionError):
                    collect_cuda_allocation_attribution(
                        harness.operation,
                        production_binding=binding,
                        staging_directory=staging,
                        operation_witness=harness.callbacks,
                        prepare_operation=fail_second_prepare,
                        warmup_iterations=3,
                        max_entries=100_000,
                    )
                payload, files = failed_audit_files(staging)
                self.assertEqual(
                    payload["failure_stage"], "post_witness_prepare"
                )
                self.assertTrue(payload["prepare_present"])
                self.assertTrue(payload["prepare_attempted"])
                self.assertFalse(payload["prepare_completed"])
                self.assertEqual(payload["prepare_attempt_count"], 2)
                self.assertEqual(payload["prepare_completion_count"], 1)
                self.assertTrue(
                    verify_preserved_failed_allocator_audit(staging, files)
                )
        finally:
            attribution_module._TORCH = previous

    def test_post_binding_input_and_runtime_failures_are_preserved(self) -> None:
        cases = (
            (
                "input_validation",
                lambda fake: None,
                {"max_entries": 8},
            ),
            (
                "runtime_initialization",
                lambda fake: setattr(fake, "__version__", "tampered"),
                {},
            ),
            (
                "allocator_api_validation",
                lambda fake: setattr(fake.cuda.memory, "_snapshot", None),
                {},
            ),
        )
        for expected_stage, configure, overrides in cases:
            with self.subTest(stage=expected_stage):
                fake = _FakeTorch()
                configure(fake)
                binding = production_binding(execution_mode="cuda_graph")
                harness = _FakeOperationHarness(binding)
                previous = attribution_module._TORCH
                try:
                    attribution_module._TORCH = fake
                    with tempfile.TemporaryDirectory() as temporary:
                        staging = Path(temporary)
                        kwargs = {
                            "warmup_iterations": 3,
                            "max_entries": 100_000,
                        }
                        kwargs.update(overrides)
                        with self.assertRaises(AllocationAttributionError):
                            collect_cuda_allocation_attribution(
                                harness.operation,
                                production_binding=binding,
                                staging_directory=staging,
                                operation_witness=harness.callbacks,
                                prepare_operation=harness.prepare,
                                **kwargs,
                            )
                        payload, files = failed_audit_files(staging)
                        self.assertEqual(
                            payload["failure_stage"], expected_stage
                        )
                        self.assertTrue(
                            verify_preserved_failed_allocator_audit(
                                staging, files
                            )
                        )
                finally:
                    attribution_module._TORCH = previous

    def test_collector_warms_then_collects_and_preserves_graph_evidence(self) -> None:
        fake = _FakeTorch()
        previous = attribution_module._TORCH
        calls = 0

        def operation() -> _FakeOutput:
            nonlocal calls
            calls += 1
            return harness.operation()

        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                binding = production_binding(execution_mode="cuda_graph")
                harness = _FakeOperationHarness(binding)
                result = collect_cuda_allocation_attribution(
                    operation,
                    production_binding=binding,
                    staging_directory=Path(temporary),
                    operation_witness=harness.callbacks,
                    prepare_operation=harness.prepare,
                    warmup_iterations=3,
                    max_entries=100_000,
                )
                self.assertEqual(calls, 5)
                self.assertTrue(result.criterion.passed)
                self.assertEqual(
                    result.criterion.criterion_id,
                    "phase3_graph_zero_allocation_v1",
                )
                self.assertFalse(
                    result.to_dict()[
                        "instrumented_duration_reported_as_timing"
                    ]
                )
                self.assertTrue(
                    verify_preserved_allocator_evidence(
                        Path(temporary), result.raw_files
                    )
                )
                semantic = validate_preserved_allocator_evidence_semantically(
                    Path(temporary),
                    result.raw_files,
                    production_binding=binding,
                )
                self.assertTrue(semantic.passed, semantic.failure_reasons)
                self.assertTrue(
                    (Path(temporary) / "allocation_audit.json").is_file()
                )

                audit_path = Path(temporary) / result.raw_files.audit_file
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit["criterion"]["passed"] = False
                audit_bytes = json.dumps(
                    audit,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                audit_digest = hashlib.sha256(audit_bytes).hexdigest()
                audit_path.write_bytes(audit_bytes)
                (
                    Path(temporary) / result.raw_files.audit_sha256_file
                ).write_text(
                    f"{audit_digest}  {result.raw_files.audit_file}\n",
                    encoding="ascii",
                )
                rewritten_files = replace(
                    result.raw_files, audit_sha256=audit_digest
                )
                replayed = validate_preserved_allocator_evidence_semantically(
                    Path(temporary),
                    rewritten_files,
                    production_binding=binding,
                )
                self.assertTrue(replayed.passed, replayed.failure_reasons)
            self.assertEqual(
                fake.cuda.memory.enabled_calls, ["all", None]
            )
        finally:
            attribution_module._TORCH = previous

    def test_output_callback_cuda_allocation_remains_in_raw_trace(self) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        harness = _FakeOperationHarness(binding)
        callback_recorder_states: list[bool] = []

        def capture_output(value: Any) -> OperationOutputWitness:
            callback_recorder_states.append(fake.cuda.memory.recording)
            if fake.cuda.memory.recording:
                fake.cuda.memory.snapshot["device_traces"][0].extend(
                    alloc_lifetime(
                        768,
                        address=0xA000,
                        stream=0,
                        allocated_block_size=1024,
                        python_stack=[
                            {
                                "name": "_allocation_audit_capture_output",
                                "filename": (
                                    "src/kvbench/runtime/"
                                    "allocation_attribution.py"
                                ),
                                "line": 1,
                            }
                        ],
                        cpp_stack=[
                            {
                                "name": "at::empty",
                                "filename": "aten/Empty.cpp",
                                "line": 1,
                            }
                        ],
                    )
                )
                fake.cuda.stats.update(
                    {
                        "allocation.all.allocated": 1,
                        "requested_bytes.all.allocated": 768,
                        "allocated_bytes.all.allocated": 1024,
                        "allocation.all.freed": 1,
                        "requested_bytes.all.freed": 768,
                        "allocated_bytes.all.freed": 1024,
                    }
                )
            return harness.capture_output(value)

        callbacks = OperationWitnessCallbacks(
            capture_cache_state=harness.capture_cache_state,
            capture_output=capture_output,
        )
        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary)
                result = collect_cuda_allocation_attribution(
                    harness.operation,
                    production_binding=binding,
                    staging_directory=staging,
                    operation_witness=callbacks,
                    prepare_operation=harness.prepare,
                    warmup_iterations=3,
                    max_entries=100_000,
                )
                self.assertEqual(callback_recorder_states, [False, True])
                self.assertEqual(
                    result.attribution.action_counts.get("alloc"), 1
                )
                self.assertEqual(len(result.attribution.allocations), 1)
                self.assertEqual(
                    result.attribution.allocations[0].event_class,
                    AllocationClass.AUDIT_INSTRUMENTATION,
                )
                self.assertFalse(result.criterion.passed)
                self.assertIn(
                    "graph_allocation_event_detected",
                    result.criterion.failure_reasons,
                )
                semantic = validate_preserved_allocator_evidence_semantically(
                    staging,
                    result.raw_files,
                    production_binding=binding,
                )
                self.assertEqual(semantic.failure_reasons, ())
                self.assertIsNotNone(semantic.criterion)
                assert semantic.criterion is not None
                self.assertFalse(semantic.criterion.passed)
        finally:
            attribution_module._TORCH = previous

    def test_noop_operation_cannot_pass_from_declarative_binding(self) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        harness = _FakeOperationHarness(binding)

        def noop() -> _FakeOutput:
            return _FakeOutput(shape=(binding.batch, 1, 16))

        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                result = collect_cuda_allocation_attribution(
                    noop,
                    production_binding=binding,
                    staging_directory=Path(temporary),
                    operation_witness=harness.callbacks,
                    warmup_operation=noop,
                    prepare_operation=harness.prepare,
                    warmup_iterations=3,
                    max_entries=100_000,
                )
                self.assertFalse(result.criterion.passed)
                self.assertIn(
                    "operation_witness_reference_destination_not_written",
                    result.attribution.integrity_errors,
                )
                self.assertIn(
                    "operation_witness_reference_destination_unchanged",
                    result.attribution.integrity_errors,
                )
        finally:
            attribution_module._TORCH = previous

    def test_measured_output_tamper_fails_witness_replay(self) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        harness = _FakeOperationHarness(binding)
        audit_calls = 0

        def operation() -> _FakeOutput:
            nonlocal audit_calls
            audit_calls += 1
            output = harness.operation()
            if audit_calls == 2:
                return _FakeOutput(shape=(binding.batch, 2, 16))
            return output

        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary)
                result = collect_cuda_allocation_attribution(
                    operation,
                    production_binding=binding,
                    staging_directory=staging,
                    operation_witness=harness.callbacks,
                    warmup_operation=harness.operation,
                    prepare_operation=harness.prepare,
                    warmup_iterations=3,
                    max_entries=100_000,
                )
                self.assertFalse(result.criterion.passed)
                self.assertIn(
                    "operation_witness_measured_output_mismatch",
                    result.attribution.integrity_errors,
                )
                semantic = (
                    validate_preserved_allocator_evidence_semantically(
                        staging,
                        result.raw_files,
                        production_binding=binding,
                    )
                )
                self.assertFalse(semantic.passed)
                self.assertEqual(semantic.failure_reasons, ())
                self.assertIsNotNone(semantic.criterion)
                assert semantic.criterion is not None
                self.assertIsNotNone(semantic.attribution)
                assert semantic.attribution is not None
                self.assertIn(
                    "operation_witness_measured_output_mismatch",
                    semantic.attribution.integrity_errors,
                )
        finally:
            attribution_module._TORCH = previous

    def test_same_shape_wrong_recorded_output_checksum_fails(self) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        harness = _FakeOperationHarness(binding)
        audited_calls = 0

        def operation() -> _FakeOutput:
            nonlocal audited_calls
            audited_calls += 1
            output = harness.operation()
            if audited_calls == 2:
                output.payload = b"different-same-shaped-output"
            return output

        previous = attribution_module._TORCH
        try:
            attribution_module._TORCH = fake
            with tempfile.TemporaryDirectory() as temporary:
                result = collect_cuda_allocation_attribution(
                    operation,
                    production_binding=binding,
                    staging_directory=Path(temporary),
                    operation_witness=harness.callbacks,
                    warmup_operation=harness.operation,
                    prepare_operation=harness.prepare,
                    warmup_iterations=3,
                    max_entries=100_000,
                )
                self.assertFalse(result.criterion.passed)
                self.assertEqual(
                    result.operation_witness.reference_output.shape,
                    result.operation_witness.measured_output.shape,
                )
                self.assertNotEqual(
                    result.operation_witness.reference_output.sha256,
                    result.operation_witness.measured_output.sha256,
                )
                self.assertIn(
                    "operation_witness_measured_output_mismatch",
                    result.attribution.integrity_errors,
                )
        finally:
            attribution_module._TORCH = previous

    def test_semantic_replay_rejects_noncanonical_checksum_rebound_raw_json(
        self,
    ) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            result = self._collect_graph_success(fake, staging)
            files = rewrite_bound_raw_evidence(
                staging,
                result.raw_files,
                file_field="trace_file",
                digest_field="trace_sha256",
                payload=b"[\n]\n",
            )
            self.assertTrue(
                verify_preserved_allocator_evidence(staging, files)
            )
            semantic = validate_preserved_allocator_evidence_semantically(
                staging,
                files,
                production_binding=binding,
            )
            self.assertFalse(semantic.passed)
            self.assertTrue(
                any(
                    "not canonical JSON" in reason
                    for reason in semantic.failure_reasons
                )
            )

    def test_semantic_replay_rejects_checksum_rebound_envelope_tamper(
        self,
    ) -> None:
        cases = (
            ("run_kind", lambda audit: audit.update(run_kind="timing")),
            (
                "execution_mode",
                lambda audit: audit.update(execution_mode="eager"),
            ),
            ("device", lambda audit: audit.update(device="cuda:1")),
            ("device_index", lambda audit: audit.update(device_index=1)),
            (
                "torch_version",
                lambda audit: audit.update(torch_version="tampered"),
            ),
            (
                "cuda_runtime_version",
                lambda audit: audit.update(cuda_runtime_version="12.0"),
            ),
            (
                "backend_identity",
                lambda audit: audit.update(backend_identity="tampered"),
            ),
            (
                "warmup_iterations",
                lambda audit: audit.update(warmup_iterations=2),
            ),
            (
                "max_history_entries",
                lambda audit: audit.update(max_history_entries=8),
            ),
            (
                "reversed_timestamps",
                lambda audit: audit.update(
                    collection_started_ns=audit["collection_finished_ns"] + 1
                ),
            ),
            (
                "timing_flag",
                lambda audit: audit.update(profiler_timing_reported=True),
            ),
            (
                "recorder_configuration",
                lambda audit: audit["recorder_configuration"].update(
                    stacks="python"
                ),
            ),
            (
                "operation_key",
                lambda audit: audit["operation_key"].update(
                    run_id="phase3-remediation-other-run"
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary)
                fake = _FakeTorch()
                binding = production_binding(execution_mode="cuda_graph")
                result = self._collect_graph_success(fake, staging)
                files = rewrite_audit_payload(
                    staging, result.raw_files, mutate
                )
                self.assertTrue(
                    verify_preserved_allocator_evidence(staging, files)
                )
                semantic = validate_preserved_allocator_evidence_semantically(
                    staging,
                    files,
                    production_binding=binding,
                )
                self.assertFalse(semantic.passed)
                self.assertTrue(semantic.failure_reasons)

    def test_semantic_replay_reads_each_hash_verified_file_once(self) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            result = self._collect_graph_success(fake, staging)
            original = attribution_module._read_no_follow
            reads: dict[str, int] = {}

            def read_once(directory_fd: int, name: str) -> bytes:
                reads[name] = reads.get(name, 0) + 1
                payload = original(directory_fd, name)
                if name == result.raw_files.trace_file:
                    (staging / name).write_bytes(b"[{}]")
                return payload

            with patch.object(
                attribution_module,
                "_read_no_follow",
                side_effect=read_once,
            ):
                semantic = validate_preserved_allocator_evidence_semantically(
                    staging,
                    result.raw_files,
                    production_binding=binding,
                )
            self.assertTrue(semantic.passed, semantic.failure_reasons)
            self.assertEqual(set(reads.values()), {1})
            self.assertEqual(len(reads), 9)
            self.assertFalse(
                verify_preserved_allocator_evidence(
                    staging, result.raw_files
                )
            )

    def test_accounting_device_uuid_total_and_unknown_fields_fail_closed(
        self,
    ) -> None:
        mutations = (
            ("device_index", lambda sample: sample.update(device_index=1)),
            ("gpu_uuid", lambda sample: sample.update(gpu_uuid="GPU-other")),
            (
                "device_total_bytes",
                lambda sample: sample.update(
                    device_total_bytes=16_384,
                    device_used_bytes=(
                        16_384 - sample["device_free_bytes"]
                    ),
                ),
            ),
            ("unknown_field", lambda sample: sample.update(unexpected=True)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary)
                fake = _FakeTorch()
                binding = production_binding(execution_mode="cuda_graph")
                result = self._collect_graph_success(fake, staging)
                accounting_path = (
                    staging
                    / result.raw_files.memory_accounting_after_file
                )
                sample = json.loads(
                    accounting_path.read_text(encoding="utf-8")
                )
                mutate(sample)
                files = rewrite_bound_raw_evidence(
                    staging,
                    result.raw_files,
                    file_field="memory_accounting_after_file",
                    digest_field="memory_accounting_after_sha256",
                    payload=canonical_json_bytes(sample),
                )
                self.assertTrue(
                    verify_preserved_allocator_evidence(staging, files)
                )
                semantic = validate_preserved_allocator_evidence_semantically(
                    staging,
                    files,
                    production_binding=binding,
                )
                self.assertFalse(semantic.passed)
                self.assertTrue(
                    any(
                        "accounting" in reason
                        for reason in semantic.failure_reasons
                    ),
                    semantic.failure_reasons,
                )

    def test_semantic_replay_rederives_allocator_counters_from_raw_stats(
        self,
    ) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            result = self._collect_graph_success(fake, staging)
            changed_stats = zero_memory_stats()
            changed_stats["allocation.all.allocated"] = 1
            files = rewrite_bound_raw_evidence(
                staging,
                result.raw_files,
                file_field="memory_stats_after_file",
                digest_field="memory_stats_after_sha256",
                payload=canonical_json_bytes(changed_stats),
            )
            self.assertTrue(
                verify_preserved_allocator_evidence(staging, files)
            )
            semantic = validate_preserved_allocator_evidence_semantically(
                staging,
                files,
                production_binding=binding,
            )
            self.assertFalse(semantic.passed)
            self.assertIsNotNone(semantic.attribution)
            self.assertIsNotNone(semantic.criterion)
            assert semantic.criterion is not None
            self.assertFalse(semantic.criterion.passed)
            self.assertIn(
                "serialized_attribution_derivation_mismatch",
                semantic.failure_reasons,
            )
            self.assertIn(
                "serialized_criterion_derivation_mismatch",
                semantic.failure_reasons,
            )

    def test_semantic_replay_rederives_device_memory_accounting(self) -> None:
        fake = _FakeTorch()
        binding = production_binding(execution_mode="cuda_graph")
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            result = self._collect_graph_success(fake, staging)
            changed_accounting = json.loads(
                (
                    staging
                    / result.raw_files.memory_accounting_after_file
                ).read_text(encoding="utf-8")
            )
            changed_accounting.update(
                {
                    "allocated_bytes": (
                        changed_accounting["allocated_bytes"] + 512
                    ),
                    "reserved_bytes": (
                        changed_accounting["reserved_bytes"] + 512
                    ),
                    "device_free_bytes": (
                        changed_accounting["device_free_bytes"] - 512
                    ),
                    "device_used_bytes": (
                        changed_accounting["device_used_bytes"] + 512
                    ),
                }
            )
            files = rewrite_bound_raw_evidence(
                staging,
                result.raw_files,
                file_field="memory_accounting_after_file",
                digest_field="memory_accounting_after_sha256",
                payload=canonical_json_bytes(changed_accounting),
            )
            self.assertTrue(
                verify_preserved_allocator_evidence(staging, files)
            )
            semantic = validate_preserved_allocator_evidence_semantically(
                staging,
                files,
                production_binding=binding,
            )
            self.assertFalse(semantic.passed)
            self.assertIsNotNone(semantic.memory)
            assert semantic.memory is not None
            self.assertEqual(semantic.memory.allocated_delta, 512)
            self.assertEqual(semantic.memory.device_used_delta, 512)
            self.assertEqual(semantic.memory.non_pytorch_delta, 0)
            self.assertIn(
                "serialized_memory_derivation_mismatch",
                semantic.failure_reasons,
            )
            self.assertIn(
                "serialized_criterion_derivation_mismatch",
                semantic.failure_reasons,
            )

    def test_all_collection_failure_stages_preserve_truthful_partial_evidence(
        self,
    ) -> None:
        def fail_warmup() -> None:
            raise RuntimeError("injected warmup failure")

        early_cases: tuple[
            tuple[
                str,
                Any,
                Any,
                frozenset[str],
            ],
            ...,
        ] = (
            ("warmup", lambda fake: None, fail_warmup, frozenset()),
            (
                "pre_operation_memory_accounting",
                lambda fake: setattr(
                    fake.cuda, "fail_memory_allocated_call", 1
                ),
                None,
                frozenset({"operation_witness_untimed"}),
            ),
            (
                "pre_operation_memory_stats",
                lambda fake: setattr(
                    fake.cuda, "fail_memory_stats_call", 1
                ),
                None,
                frozenset(
                    {
                        "operation_witness_untimed",
                        "memory_accounting_before",
                    }
                ),
            ),
            (
                "allocator_history_disable",
                lambda fake: setattr(
                    fake.cuda.memory,
                    "disable_error",
                    RuntimeError("injected disable failure"),
                ),
                None,
                frozenset(
                    {
                        "memory_accounting_before",
                        "memory_stats_before",
                        "memory_stats_after",
                        "memory_accounting_after",
                        "snapshot",
                        "operation_witness_untimed",
                    }
                ),
            ),
            (
                "allocator_trace_parse",
                lambda fake: setattr(
                    fake.cuda.memory,
                    "snapshot",
                    {"device_traces": "malformed", "segments": []},
                ),
                None,
                frozenset(
                    {
                        "memory_accounting_before",
                        "memory_stats_before",
                        "memory_stats_after",
                        "memory_accounting_after",
                        "snapshot",
                        "operation_witness_untimed",
                        "operation_witness",
                    }
                ),
            ),
        )
        for expected_stage, configure, warmup, expected_keys in early_cases:
            with self.subTest(stage=expected_stage):
                fake = _FakeTorch()
                configure(fake)
                with tempfile.TemporaryDirectory() as temporary:
                    payload, files = self._collect_expected_failure(
                        fake,
                        Path(temporary),
                        expected_stage=expected_stage,
                        warmup_operation=warmup,
                    )
                    self.assertEqual(
                        {
                            item.evidence_key
                            for item in files.partial_files
                        },
                        set(expected_keys),
                    )
                    self.assertEqual(
                        payload["partial_files"],
                        [
                            item.to_dict()
                            for item in files.partial_files
                        ],
                    )

    def test_derived_and_preservation_failures_retain_all_raw_partials(
        self,
    ) -> None:
        derived_cases = (
            ("allocation_attribution", "attribute_allocator_trace"),
            ("criterion_evaluation", "evaluate_strict_graph_criterion"),
            ("evidence_preservation", "preserve_allocator_evidence"),
        )
        expected_keys = {
            "memory_accounting_before",
            "memory_stats_before",
            "memory_stats_after",
            "memory_accounting_after",
            "snapshot",
            "trace",
            "operation_witness_untimed",
            "operation_witness",
        }
        for expected_stage, function_name in derived_cases:
            with self.subTest(stage=expected_stage):
                fake = _FakeTorch()
                with tempfile.TemporaryDirectory() as temporary:
                    with patch.object(
                        attribution_module,
                        function_name,
                        side_effect=RuntimeError(
                            f"injected {expected_stage} failure"
                        ),
                    ):
                        _, files = self._collect_expected_failure(
                            fake,
                            Path(temporary),
                            expected_stage=expected_stage,
                        )
                    self.assertEqual(
                        {
                            item.evidence_key
                            for item in files.partial_files
                        },
                        expected_keys,
                    )

    def test_failed_audit_recomputes_embedded_rules_digest(self) -> None:
        fake = _FakeTorch()
        fake.cuda.fail_memory_allocated_call = 1
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            payload, files = self._collect_expected_failure(
                fake,
                staging,
                expected_stage="pre_operation_memory_accounting",
            )
            payload["attribution_rules"]["policy_authority"] = "tampered"
            tampered_bytes = canonical_json_bytes(payload)
            tampered_digest = hashlib.sha256(tampered_bytes).hexdigest()
            (staging / files.audit_file).write_bytes(tampered_bytes)
            (staging / files.audit_sha256_file).write_text(
                f"{tampered_digest}  {files.audit_file}\n",
                encoding="ascii",
            )
            rebound = replace(files, audit_sha256=tampered_digest)
            self.assertFalse(
                verify_preserved_failed_allocator_audit(staging, rebound)
            )

    def test_post_operation_failures_preserve_checksum_bound_failed_audit(self) -> None:
        cases = (
            (
                "post_operation_sync",
                lambda fake: setattr(
                    fake.cuda, "fail_synchronize_call", 6
                ),
            ),
            (
                "post_operation_memory_stats",
                lambda fake: setattr(
                    fake.cuda, "fail_memory_stats_call", 2
                ),
            ),
            (
                "post_operation_memory_stats",
                lambda fake: setattr(
                    fake.cuda, "malformed_memory_stats_call", 2
                ),
            ),
            (
                "allocator_snapshot",
                lambda fake: setattr(
                    fake.cuda.memory,
                    "snapshot_error",
                    RuntimeError("injected snapshot failure"),
                ),
            ),
        )
        for expected_stage, configure in cases:
            with self.subTest(expected_stage=expected_stage):
                fake = _FakeTorch()
                configure(fake)
                with tempfile.TemporaryDirectory() as temporary:
                    staging = Path(temporary)
                    payload, files = self._collect_expected_failure(
                        fake,
                        staging,
                        expected_stage=expected_stage,
                    )
                    self.assertEqual(payload["evidence_status"], "failed")
                    self.assertFalse(payload["raw_snapshot_available"])
                    self.assertFalse(
                        payload[
                            "instrumented_duration_reported_as_timing"
                        ]
                    )
                    self.assertFalse(
                        (staging / "allocation_audit.json").exists()
                    )
                    with self.assertRaises(FileExistsError):
                        replacement_payload = dict(payload)
                        replacement_payload["failure_message"] = (
                            "replacement"
                        )
                        preserve_failed_allocator_audit(
                            staging,
                            audit_payload=replacement_payload,
                        )
                    (staging / "allocation_audit_failed.json").write_bytes(
                        b"{}"
                    )
                    self.assertFalse(
                        verify_preserved_failed_allocator_audit(
                            staging, files
                        )
                    )
                self.assertEqual(
                    fake.cuda.memory.enabled_calls, ["all", None]
                )


if __name__ == "__main__":
    unittest.main()
