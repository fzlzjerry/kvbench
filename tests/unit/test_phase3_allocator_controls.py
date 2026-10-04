"""CPU-only adversarial tests for paired Phase 3 allocator controls."""

from __future__ import annotations

import hashlib
import json
import unittest

from kvbench.runtime.allocation_attribution import (
    PHASE3_ALLOCATION_WARMUP_ITERATIONS,
    PHASE3_BACKEND_IDENTITY_SHA256,
    PHASE3_CUDA_RUNTIME_VERSION,
    PHASE3_DEVICE,
    PHASE3_DEVICE_INDEX,
    PHASE3_OUTPUT_DTYPE,
    PHASE3_RECORDER_CONFIGURATION,
    PHASE3_TORCH_VERSION,
    RawMemoryAccountingSample,
    allocator_snapshot_sha256,
    allocator_trace_sha256,
    cuda_allocator_rounded_minimum,
)
from kvbench.runtime.phase3_allocator_controls import (
    GQA_ALLOCATOR_CONTROL_SCHEMA,
    INTERNAL_SPLIT_K_CPP_FRAMES,
    MHA_ALLOCATOR_CONTROL_SCHEMA,
    PHASE3_ALLOCATOR_CONTROL_SCALE,
    PUBLIC_FLASH_CPP_FRAME,
    AllocatorControlGeometry,
    AllocatorControlGraphBinding,
    AllocatorControlTensorObservation,
    Phase3AllocatorControlError,
    Phase3AllocatorControlReplay,
    Phase3AllocatorControlObservation,
    canonical_phase3_allocator_control_bytes,
    parse_phase3_allocator_control_bytes,
    replay_phase3_allocator_control,
    verify_phase3_paired_allocator_controls,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.schema import GraphMode, RunnerKind
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)


GQA_DISPATCH_RAW = b'{"traceEvents":[{"name":"gqa"}]}'
MHA_DISPATCH_RAW = b'{"traceEvents":[{"name":"mha_control"}]}'
GPU_UUID = "GPU-phase3-allocator-control-test"
QUERY_POINTER = 0x100000000
GQA_KEY_STORAGE_POINTER = 0x200000000
GQA_VALUE_STORAGE_POINTER = 0x400000000
MHA_KEY_STORAGE_POINTER = 0x600000000
MHA_VALUE_STORAGE_POINTER = 0x800000000
GRAPH_GQA_OUTPUT_POINTER = 0xA00000000
GRAPH_MHA_OUTPUT_POINTER = 0xC00000000
FLASH_FORWARD_CPP_FRAME = INTERNAL_SPLIT_K_CPP_FRAMES[1]
EAGER_BASE_SIZES = (8, 16, 8192, 128)
EAGER_SPLIT_SIZES = (32768, 256)
STAT_KEYS = (
    "allocation.all.allocated",
    "requested_bytes.all.allocated",
    "allocated_bytes.all.allocated",
    "allocation.all.freed",
    "requested_bytes.all.freed",
    "allocated_bytes.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_device_alloc",
    "num_device_free",
    "num_alloc_retries",
    "num_ooms",
)


def operation_key(*, graph: bool = False) -> Phase3AuditOperationKey:
    graph_mode = GraphMode.CUDA_GRAPH if graph else GraphMode.EAGER
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
        run_id=(
            "phase3-allocator-control-graph"
            if graph
            else "phase3-allocator-control-eager"
        ),
        point=point,
        decode_step=0,
        cache_layout_fingerprint="1" * 64,
        execution_git_sha="2" * 40,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[PHASE3_FIXED_PLAN_PATH],
        hardware_identity_sha256="3" * 64,
        software_identity_sha256="4" * 64,
        model_identity_sha256="5" * 64,
        backend_identity_sha256=PHASE3_BACKEND_IDENTITY_SHA256,
        source_identity_sha256="6" * 64,
    )


def tensor(
    shape: tuple[int, ...],
    *,
    pointer: int,
    content_sha256: str | None = None,
    stride: tuple[int, ...] | None = None,
    storage_bytes: int | None = None,
    storage_offset: int = 0,
) -> AllocatorControlTensorObservation:
    canonical_stride: list[int] = []
    running = 1
    for dimension in reversed(shape):
        canonical_stride.append(running)
        running *= dimension
    selected_stride = (
        tuple(reversed(canonical_stride)) if stride is None else stride
    )
    logical_bytes = running * 2
    selected_storage_bytes = (
        logical_bytes if storage_bytes is None else storage_bytes
    )
    contiguous = True
    expected = 1
    for dimension, observed in zip(
        reversed(shape), reversed(selected_stride), strict=True
    ):
        if dimension != 1 and observed != expected:
            contiguous = False
        expected *= dimension
    return AllocatorControlTensorObservation(
        shape=shape,
        stride=selected_stride,
        dtype=PHASE3_OUTPUT_DTYPE,
        device=PHASE3_DEVICE,
        element_size=2,
        logical_bytes=logical_bytes,
        storage_bytes=selected_storage_bytes,
        storage_offset=storage_offset,
        data_ptr=pointer + storage_offset * 2,
        storage_data_ptr=pointer,
        is_contiguous=contiguous,
        content_sha256=content_sha256,
    )


def allocation_events(
    sizes: list[int],
    *,
    cpp_frame: str = PUBLIC_FLASH_CPP_FRAME,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for index, size in enumerate(sizes):
        address = 0x100000 + index * 0x200000
        events.extend(
            (
                {
                    "action": "alloc",
                    "addr": address,
                    "size": size,
                    "stream": 7,
                    "allocated_block_size": (
                        cuda_allocator_rounded_minimum(size)
                    ),
                    "frames": [
                        {
                            "name": "allocator_control_operation",
                            "filename": "tests/control.py",
                            "line": 10,
                        },
                        {
                            "name": cpp_frame,
                            "filename": "flash_api.cpp",
                            "line": 20,
                        },
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
    return events


def split_allocation_events(
    sizes: tuple[int, ...],
    *,
    set_params_frame: str = INTERNAL_SPLIT_K_CPP_FRAMES[0],
    forward_frame: str = FLASH_FORWARD_CPP_FRAME,
) -> list[dict[str, object]]:
    events = allocation_events(list(sizes), cpp_frame=forward_frame)
    for event in events:
        if event["action"] != "alloc":
            continue
        frames = event["frames"]
        if not isinstance(frames, list):
            raise AssertionError("synthetic allocator frames are malformed")
        frames.insert(
            1,
            {
                "name": set_params_frame,
                "filename": "flash_api.cpp",
                "line": 19,
            },
        )
    return events


def eager_trace(
    *,
    base_sizes: tuple[int, ...] = EAGER_BASE_SIZES,
    split_sizes: tuple[int, ...] = EAGER_SPLIT_SIZES,
    base_frame: str = FLASH_FORWARD_CPP_FRAME,
    split_set_params_frame: str = INTERNAL_SPLIT_K_CPP_FRAMES[0],
    split_forward_frame: str = FLASH_FORWARD_CPP_FRAME,
) -> list[dict[str, object]]:
    return allocation_events(list(base_sizes), cpp_frame=base_frame) + (
        split_allocation_events(
            split_sizes,
            set_params_frame=split_set_params_frame,
            forward_frame=split_forward_frame,
        )
    )


def segment_events() -> list[dict[str, object]]:
    return [
        {
            "action": "segment_alloc",
            "addr": 0x900000,
            "size": 2 * 1024 * 1024,
            "stream": 7,
        },
        {
            "action": "segment_free",
            "addr": 0x900000,
            "size": 2 * 1024 * 1024,
            "stream": 7,
        },
    ]


def stats_for(
    trace: list[dict[str, object]],
) -> tuple[dict[str, int], dict[str, int]]:
    allocs = [item for item in trace if item["action"] == "alloc"]
    frees = [
        item for item in trace if item["action"] == "free_completed"
    ]
    segments_alloc = sum(
        item["action"] == "segment_alloc" for item in trace
    )
    segments_free = sum(
        item["action"] == "segment_free" for item in trace
    )
    block_bytes = sum(int(item["allocated_block_size"]) for item in allocs)
    delta = {
        "allocation.all.allocated": len(allocs),
        "requested_bytes.all.allocated": sum(
            int(item["size"]) for item in allocs
        ),
        "allocated_bytes.all.allocated": block_bytes,
        "allocation.all.freed": len(frees),
        "requested_bytes.all.freed": sum(
            int(item["size"]) for item in frees
        ),
        "allocated_bytes.all.freed": block_bytes,
        "segment.all.allocated": segments_alloc,
        "segment.all.freed": segments_free,
        "num_device_alloc": segments_alloc,
        "num_device_free": segments_free,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }
    before = {key: 100 for key in STAT_KEYS}
    after = {key: before[key] + delta[key] for key in STAT_KEYS}
    return before, after


def accounting(
    key: Phase3AuditOperationKey, *, sample_role: str, timestamp: int
) -> RawMemoryAccountingSample:
    return RawMemoryAccountingSample(
        schema_version="kvbench-phase3-memory-accounting-2.0.0",
        operation_fingerprint_sha256=key.operation_fingerprint_sha256,
        sample_role=sample_role,
        timestamp_ns=timestamp,
        device=PHASE3_DEVICE,
        device_index=PHASE3_DEVICE_INDEX,
        gpu_uuid=GPU_UUID,
        allocated_bytes=1024,
        reserved_bytes=2048,
        device_free_bytes=8_000_000,
        device_total_bytes=10_000_000,
    )


def observation_bytes(
    *,
    role: str,
    key: Phase3AuditOperationKey,
    trace: list[dict[str, object]],
    query_pointer: int = QUERY_POINTER,
    query_sha256: str = "a" * 64,
    dispatch_raw: bytes | None = None,
) -> bytes:
    kv_heads = 8 if role == "gqa" else 32
    dispatch = (
        GQA_DISPATCH_RAW if role == "gqa" else MHA_DISPATCH_RAW
    ) if dispatch_raw is None else dispatch_raw
    stats_before, stats_after = stats_for(trace)
    snapshot: dict[str, object] = {"device_traces": [trace]}
    graph = key.dispatch_execution_mode == "cuda_graph_replay"
    if role == "gqa":
        kv_storage_bytes = (
            32 * key.batch_size * 8 * key.capacity * 128 * 2
        )
        kv_stride = (
            8 * key.capacity * 128,
            key.capacity * 128,
            128,
            1,
        )
        key_pointer = GQA_KEY_STORAGE_POINTER
        value_pointer = GQA_VALUE_STORAGE_POINTER
    else:
        kv_storage_bytes = (
            key.batch_size * 32 * key.attended_context * 128 * 2
        )
        kv_stride = None
        key_pointer = MHA_KEY_STORAGE_POINTER
        value_pointer = MHA_VALUE_STORAGE_POINTER
    output_bytes = key.batch_size * 32 * 128 * 2
    output_allocations = [
        item
        for item in trace
        if item.get("action") == "alloc"
        and item.get("size") == output_bytes
    ]
    output_pointer = (
        GRAPH_GQA_OUTPUT_POINTER
        if graph and role == "gqa"
        else GRAPH_MHA_OUTPUT_POINTER
        if graph
        else int(output_allocations[0]["addr"])
        if output_allocations
        else GRAPH_GQA_OUTPUT_POINTER
        if role == "gqa"
        else GRAPH_MHA_OUTPUT_POINTER
    )
    control = Phase3AllocatorControlObservation(
        schema_version=(
            GQA_ALLOCATOR_CONTROL_SCHEMA
            if role == "gqa"
            else MHA_ALLOCATOR_CONTROL_SCHEMA
        ),
        role=role,  # type: ignore[arg-type]
        operation_key=key,
        operation_fingerprint_sha256=key.operation_fingerprint_sha256,
        geometry=AllocatorControlGeometry(
            batch=key.batch_size,
            query_heads=32,
            kv_heads=kv_heads,
            context=key.attended_context,
            head_dim=128,
            query_length=1,
            dtype=PHASE3_OUTPUT_DTYPE,
            dtype_bytes=2,
            is_causal=False,
            scale=float(PHASE3_ALLOCATOR_CONTROL_SCALE),
            dropout_p=0.0,
            enable_gqa=True,
            execution_mode=key.dispatch_execution_mode,
        ),
        backend_identity_sha256=PHASE3_BACKEND_IDENTITY_SHA256,
        dispatch_trace_sha256=hashlib.sha256(dispatch).hexdigest(),
        dispatch_trace_size_bytes=len(dispatch),
        query=tensor(
            (key.batch_size, 32, 1, 128),
            pointer=query_pointer,
            content_sha256=query_sha256,
        ),
        key=tensor(
            (key.batch_size, kv_heads, key.attended_context, 128),
            pointer=key_pointer,
            stride=kv_stride,
            storage_bytes=kv_storage_bytes,
            content_sha256="b" * 64,
        ),
        value=tensor(
            (key.batch_size, kv_heads, key.attended_context, 128),
            pointer=value_pointer,
            stride=kv_stride,
            storage_bytes=kv_storage_bytes,
            content_sha256="c" * 64,
        ),
        query_after=tensor(
            (key.batch_size, 32, 1, 128),
            pointer=query_pointer,
            content_sha256=query_sha256,
        ),
        key_after=tensor(
            (key.batch_size, kv_heads, key.attended_context, 128),
            pointer=key_pointer,
            stride=kv_stride,
            storage_bytes=kv_storage_bytes,
            content_sha256="b" * 64,
        ),
        value_after=tensor(
            (key.batch_size, kv_heads, key.attended_context, 128),
            pointer=value_pointer,
            stride=kv_stride,
            storage_bytes=kv_storage_bytes,
            content_sha256="c" * 64,
        ),
        output=tensor(
            (key.batch_size, 32, 1, 128), pointer=output_pointer
        ),
        graph_binding=AllocatorControlGraphBinding(
            captured=graph,
            output_data_ptr=output_pointer if graph else None,
            capture_stream_id=99 if graph else None,
        ),
        warmup_iterations=PHASE3_ALLOCATION_WARMUP_ITERATIONS,
        recorder_configuration=dict(PHASE3_RECORDER_CONFIGURATION),
        allocator_snapshot=snapshot,
        allocator_history=tuple(trace),
        allocator_snapshot_sha256=allocator_snapshot_sha256(snapshot),
        allocator_history_sha256=allocator_trace_sha256(trace),
        memory_stats_before=stats_before,
        memory_stats_after=stats_after,
        accounting_before=accounting(key, sample_role="before", timestamp=2),
        accounting_after=accounting(key, sample_role="after", timestamp=3),
        runtime={
            "torch_version": PHASE3_TORCH_VERSION,
            "cuda_runtime_version": PHASE3_CUDA_RUNTIME_VERSION,
            "device": PHASE3_DEVICE,
            "device_index": PHASE3_DEVICE_INDEX,
            "gpu_uuid": GPU_UUID,
        },
        collection_started_ns=1,
        collection_finished_ns=4,
    )
    return control.canonical_bytes()


def successful_trace() -> list[dict[str, object]]:
    # One exact instance of every allocation seen in the isolated layer.
    return eager_trace()


class AllocatorControlCanonicalSchemaTests(unittest.TestCase):
    def test_canonical_round_trip_and_noncanonical_rejection(self) -> None:
        key = operation_key()
        raw = observation_bytes(role="gqa", key=key, trace=successful_trace())
        parsed = parse_phase3_allocator_control_bytes(raw)
        self.assertEqual(parsed.canonical_bytes(), raw)
        payload = json.loads(raw)
        self.assertEqual(canonical_phase3_allocator_control_bytes(payload), raw)
        pretty = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "not canonical"
        ):
            parse_phase3_allocator_control_bytes(pretty)

    def test_duplicate_key_and_digest_tamper_are_rejected(self) -> None:
        duplicate = b'{"role":"gqa","role":"mha_control"}'
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "duplicate key"
        ):
            parse_phase3_allocator_control_bytes(duplicate)

        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        payload["allocator_history"][0]["size"] += 1
        tampered = canonical_phase3_allocator_control_bytes(payload)
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "history digest mismatch"
        ):
            parse_phase3_allocator_control_bytes(tampered)

    def test_role_geometry_mismatch_is_rejected(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        payload["geometry"]["kv_heads"] = 32
        payload["tensor_observations"]["key"] = json.loads(
            json.dumps(
                tensor((1, 32, 129, 128), pointer=0xB000).to_dict()
            )
        )
        payload["tensor_observations"]["value"] = json.loads(
            json.dumps(
                tensor((1, 32, 129, 128), pointer=0xC000).to_dict()
            )
        )
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "geometry differs"
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_gqa_expanded_backing_is_rejected(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        expanded_storage_bytes = (
            32 * key.batch_size * 32 * key.capacity * 128 * 2
        )
        payload["tensor_observations"]["key"]["storage_bytes"] = (
            expanded_storage_bytes
        )
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "bounded native-KV"
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_noncanonical_query_layout_is_rejected(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        query = payload["tensor_observations"]["query"]
        query["stride"][1] = 127
        query["is_contiguous"] = False
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "query layout is not canonical"
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_output_accepts_only_singleton_equivalent_contiguous_stride(
        self,
    ) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        output = payload["output_metadata"]
        output["stride"][2] = 4096
        parsed = parse_phase3_allocator_control_bytes(
            canonical_phase3_allocator_control_bytes(payload)
        )
        self.assertEqual(parsed.output.stride, (4096, 128, 4096, 1))
        self.assertTrue(parsed.output.is_contiguous)

        noncontiguous = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        bad_output = noncontiguous["output_metadata"]
        bad_output["stride"][1] = 127
        bad_output["is_contiguous"] = False
        with self.assertRaisesRegex(
            Phase3AllocatorControlError,
            "output layout is not canonical",
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(noncontiguous)
            )

    def test_tensor_storage_alias_is_rejected(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        query = payload["tensor_observations"]["query"]
        output = payload["output_metadata"]
        output["data_ptr"] = query["data_ptr"]
        output["storage_data_ptr"] = query["storage_data_ptr"]
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "tensor storage aliases"
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_data_pointer_must_match_storage_offset(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        payload["tensor_observations"]["key"]["data_ptr"] += 2
        with self.assertRaisesRegex(
            Phase3AllocatorControlError,
            "data pointer differs from storage offset",
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_gqa_key_value_cache_layers_must_match(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        value = payload["tensor_observations"]["value"]
        per_layer_elements = key.batch_size * 8 * key.capacity * 128
        value["storage_offset"] = per_layer_elements
        value["data_ptr"] = (
            value["storage_data_ptr"] + per_layer_elements * 2
        )
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "cache layers differ"
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_post_operation_qkv_content_mutation_is_rejected(self) -> None:
        key = operation_key()
        for name, digest in (
            ("query", "d" * 64),
            ("key", "e" * 64),
            ("value", "f" * 64),
        ):
            with self.subTest(tensor=name):
                payload = json.loads(
                    observation_bytes(
                        role="gqa",
                        key=key,
                        trace=successful_trace(),
                    )
                )
                payload["post_operation_tensor_observations"][name][
                    "content_sha256"
                ] = digest
                with self.assertRaisesRegex(
                    Phase3AllocatorControlError,
                    f"mutated {name} storage, metadata, or content",
                ):
                    parse_phase3_allocator_control_bytes(
                        canonical_phase3_allocator_control_bytes(payload)
                    )

    def test_post_operation_storage_identity_change_is_rejected(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        post_key = payload["post_operation_tensor_observations"]["key"]
        post_key["data_ptr"] += 0x10000000
        post_key["storage_data_ptr"] += 0x10000000
        with self.assertRaisesRegex(
            Phase3AllocatorControlError,
            "mutated key storage, metadata, or content",
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_post_operation_tensor_metadata_change_is_rejected(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        payload["post_operation_tensor_observations"]["value"][
            "dtype"
        ] = "torch.float16"
        with self.assertRaisesRegex(
            Phase3AllocatorControlError,
            "post-operation allocator-control value tensor differs from geometry",
        ):
            parse_phase3_allocator_control_bytes(
                canonical_phase3_allocator_control_bytes(payload)
            )

    def test_every_input_observation_requires_content_identity(self) -> None:
        key = operation_key()
        for phase in (
            "tensor_observations",
            "post_operation_tensor_observations",
        ):
            for name in ("query", "key", "value"):
                with self.subTest(phase=phase, tensor=name):
                    payload = json.loads(
                        observation_bytes(
                            role="gqa", key=key, trace=successful_trace()
                        )
                    )
                    payload[phase][name]["content_sha256"] = None
                    with self.assertRaisesRegex(
                        Phase3AllocatorControlError,
                        "input tensor lacks content identity",
                    ):
                        parse_phase3_allocator_control_bytes(
                            canonical_phase3_allocator_control_bytes(payload)
                        )


class AllocatorControlSemanticReplayTests(unittest.TestCase):
    def _replay_eager(
        self, trace: list[dict[str, object]]
    ) -> Phase3AllocatorControlReplay:
        key = operation_key()
        return replay_phase3_allocator_control(
            observation_bytes(role="gqa", key=key, trace=trace),
            expected_operation_key=key,
            dispatch_trace_raw=GQA_DISPATCH_RAW,
        )

    def test_missing_fixed_output_or_lse_fails(self) -> None:
        cases = (
            (8192, "fixed_attention_output"),
            (128, "fixed_flash_lse"),
        )
        for missing_size, formula in cases:
            with self.subTest(formula=formula):
                replay = self._replay_eager(
                    eager_trace(
                        base_sizes=tuple(
                            size
                            for size in EAGER_BASE_SIZES
                            if size != missing_size
                        )
                    )
                )
                self.assertFalse(replay.passed)
                self.assertIn(
                    f"eager_expected_allocation_missing:{formula}",
                    replay.failure_reasons,
                )

    def test_missing_or_duplicate_fixed_scalar_fails(self) -> None:
        cases = (
            (8, "flash_fixed_scalar_8"),
            (16, "flash_fixed_scalar_16"),
        )
        for size, formula in cases:
            with self.subTest(formula=formula, mutation="missing"):
                missing = self._replay_eager(
                    eager_trace(
                        base_sizes=tuple(
                            item for item in EAGER_BASE_SIZES if item != size
                        )
                    )
                )
                self.assertIn(
                    f"eager_expected_allocation_missing:{formula}",
                    missing.failure_reasons,
                )
            with self.subTest(formula=formula, mutation="duplicate"):
                duplicate = self._replay_eager(
                    eager_trace(base_sizes=(*EAGER_BASE_SIZES, size))
                )
                self.assertIn(
                    f"eager_expected_allocation_duplicate:{formula}",
                    duplicate.failure_reasons,
                )

    def test_missing_split_pair_or_member_fails(self) -> None:
        cases = (
            ((), "eager_split_k_pair_missing"),
            ((32768,), "eager_split_k_pair_incomplete"),
        )
        for split_sizes, reason in cases:
            with self.subTest(reason=reason):
                replay = self._replay_eager(
                    eager_trace(split_sizes=split_sizes)
                )
                self.assertFalse(replay.passed)
                self.assertIn(reason, replay.failure_reasons)

    def test_duplicate_known_allocation_fails(self) -> None:
        replay = self._replay_eager(
            eager_trace(base_sizes=(*EAGER_BASE_SIZES, 8192))
        )
        self.assertFalse(replay.passed)
        self.assertIn(
            "eager_expected_allocation_duplicate:fixed_attention_output",
            replay.failure_reasons,
        )
        self.assertIn(
            "eager_allocation_set_cardinality_mismatch",
            replay.failure_reasons,
        )

    def test_misleading_stack_substrings_do_not_match(self) -> None:
        replay = self._replay_eager(
            eager_trace(
                base_frame=f"wrapper::{FLASH_FORWARD_CPP_FRAME}",
                split_set_params_frame=(
                    f"wrapper::{INTERNAL_SPLIT_K_CPP_FRAMES[0]}"
                ),
                split_forward_frame=f"wrapper::{FLASH_FORWARD_CPP_FRAME}",
            )
        )
        self.assertFalse(replay.passed)
        self.assertIn("unknown_allocation_detected", replay.failure_reasons)
        self.assertIn(
            "eager_expected_allocation_missing:fixed_attention_output",
            replay.failure_reasons,
        )

    def test_output_metadata_binds_to_output_allocation(self) -> None:
        key = operation_key()
        payload = json.loads(
            observation_bytes(role="gqa", key=key, trace=successful_trace())
        )
        payload["output_metadata"]["data_ptr"] = 0xE00000000
        payload["output_metadata"]["storage_data_ptr"] = 0xE00000000
        replay = replay_phase3_allocator_control(
            canonical_phase3_allocator_control_bytes(payload),
            expected_operation_key=key,
            dispatch_trace_raw=GQA_DISPATCH_RAW,
        )
        self.assertFalse(replay.passed)
        self.assertIn(
            "fixed_attention_output_pointer_mismatch",
            replay.failure_reasons,
        )

    def test_expanded_kv_allocations_are_positive_failure_evidence(self) -> None:
        key = operation_key()
        expanded_single = 1 * 32 * key.attended_context * 128 * 2
        for size in (expanded_single, 2 * expanded_single):
            with self.subTest(requested_bytes=size):
                replay = replay_phase3_allocator_control(
                    observation_bytes(
                        role="gqa",
                        key=key,
                        trace=allocation_events([size]),
                    ),
                    expected_operation_key=key,
                    dispatch_trace_raw=GQA_DISPATCH_RAW,
                )
                self.assertFalse(replay.passed)
                self.assertIn(
                    "expanded_kv_allocation_detected",
                    replay.failure_reasons,
                )

    def test_context_scaled_unknown_remains_fail_closed(self) -> None:
        key = operation_key()
        context_plane = 1 * key.attended_context * 128 * 2
        replay = replay_phase3_allocator_control(
            observation_bytes(
                role="gqa",
                key=key,
                trace=allocation_events(
                    [context_plane], cpp_frame="at::native::empty_cuda"
                ),
            ),
            expected_operation_key=key,
            dispatch_trace_raw=GQA_DISPATCH_RAW,
        )
        self.assertFalse(replay.passed)
        self.assertIn(
            "context_scaled_unknown_allocation_detected",
            replay.failure_reasons,
        )

    def test_generic_unknown_remains_fail_closed(self) -> None:
        key = operation_key()
        replay = replay_phase3_allocator_control(
            observation_bytes(
                role="gqa",
                key=key,
                trace=allocation_events(
                    [1234], cpp_frame="at::native::empty_cuda"
                ),
            ),
            expected_operation_key=key,
            dispatch_trace_raw=GQA_DISPATCH_RAW,
        )
        self.assertFalse(replay.passed)
        self.assertIn("unknown_allocation_detected", replay.failure_reasons)

    def test_segment_allocation_fails_even_with_zero_net_memory(self) -> None:
        key = operation_key()
        replay = replay_phase3_allocator_control(
            observation_bytes(role="gqa", key=key, trace=segment_events()),
            expected_operation_key=key,
            dispatch_trace_raw=GQA_DISPATCH_RAW,
        )
        self.assertFalse(replay.passed)
        self.assertIn(
            "segment_alloc_or_free_detected", replay.failure_reasons
        )
        self.assertIn(
            "device_allocation_or_free_detected_or_unavailable",
            replay.failure_reasons,
        )

    def test_graph_replay_rejects_every_allocation_event(self) -> None:
        key = operation_key(graph=True)
        replay = replay_phase3_allocator_control(
            observation_bytes(
                role="gqa", key=key, trace=successful_trace()
            ),
            expected_operation_key=key,
            dispatch_trace_raw=GQA_DISPATCH_RAW,
        )
        self.assertFalse(replay.passed)
        self.assertIn(
            "graph_allocation_event_detected", replay.failure_reasons
        )

    def test_graph_replay_rejects_segment_events_without_alloc_events(self) -> None:
        key = operation_key(graph=True)
        replay = replay_phase3_allocator_control(
            observation_bytes(
                role="gqa",
                key=key,
                trace=segment_events(),
            ),
            expected_operation_key=key,
            dispatch_trace_raw=GQA_DISPATCH_RAW,
        )
        self.assertFalse(replay.passed)
        self.assertIn("segment_alloc_or_free_detected", replay.failure_reasons)
        self.assertIn(
            "graph_allocator_counters_nonzero_or_unavailable", replay.failure_reasons
        )


    def test_graph_pair_with_zero_allocator_events_passes(self) -> None:
        key = operation_key(graph=True)
        result = verify_phase3_paired_allocator_controls(
            gqa_raw=observation_bytes(role="gqa", key=key, trace=[]),
            mha_control_raw=observation_bytes(
                role="mha_control", key=key, trace=[]
            ),
            operation_key=key,
            gqa_dispatch_trace_raw=GQA_DISPATCH_RAW,
            mha_dispatch_trace_raw=MHA_DISPATCH_RAW,
        )
        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.split_k_pair_multiplicity, ())


class PairedAllocatorControlVerificationTests(unittest.TestCase):
    def test_query_identity_or_content_mismatch_fails_pair(self) -> None:
        key = operation_key()
        result = verify_phase3_paired_allocator_controls(
            gqa_raw=observation_bytes(
                role="gqa", key=key, trace=successful_trace()
            ),
            mha_control_raw=observation_bytes(
                role="mha_control",
                key=key,
                trace=successful_trace(),
                query_pointer=QUERY_POINTER + 0x10000000,
                query_sha256="b" * 64,
            ),
            operation_key=key,
            gqa_dispatch_trace_raw=GQA_DISPATCH_RAW,
            mha_dispatch_trace_raw=MHA_DISPATCH_RAW,
        )
        self.assertFalse(result.passed)
        self.assertIn(
            "allocator_control_query_identity_or_content_mismatch",
            result.failure_reasons,
        )

    def test_dispatch_trace_digest_binding_rejects_substitution(self) -> None:
        key = operation_key()
        with self.assertRaisesRegex(
            Phase3AllocatorControlError, "supplied dispatch trace"
        ):
            verify_phase3_paired_allocator_controls(
                gqa_raw=observation_bytes(
                    role="gqa", key=key, trace=successful_trace()
                ),
                mha_control_raw=observation_bytes(
                    role="mha_control", key=key, trace=successful_trace()
                ),
                operation_key=key,
                gqa_dispatch_trace_raw=b"substituted trace",
                mha_dispatch_trace_raw=MHA_DISPATCH_RAW,
            )

    def test_evidence_backed_split_k_pair_passes(self) -> None:
        key = operation_key()
        result = verify_phase3_paired_allocator_controls(
            gqa_raw=observation_bytes(
                role="gqa", key=key, trace=successful_trace()
            ),
            mha_control_raw=observation_bytes(
                role="mha_control", key=key, trace=successful_trace()
            ),
            operation_key=key,
            gqa_dispatch_trace_raw=GQA_DISPATCH_RAW,
            mha_dispatch_trace_raw=MHA_DISPATCH_RAW,
        )
        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.split_k_pair_multiplicity, ((2, 1),))
        self.assertTrue(result.gqa.passed)
        self.assertTrue(result.mha_control.passed)
        self.assertEqual(
            tuple(fact.formula_id for fact in result.gqa.allocation_facts),
            (
                "flash_fixed_scalar_8",
                "flash_fixed_scalar_16",
                "fixed_attention_output",
                "fixed_flash_lse",
                "flash_split_k_output_accumulator",
                "flash_split_k_lse",
            ),
        )

    def test_formula_multiplicity_mismatch_fails(self) -> None:
        key = operation_key()
        result = verify_phase3_paired_allocator_controls(
            gqa_raw=observation_bytes(
                role="gqa", key=key, trace=successful_trace()
            ),
            mha_control_raw=observation_bytes(
                role="mha_control",
                key=key,
                trace=eager_trace(
                    split_sizes=(*EAGER_SPLIT_SIZES, 32768)
                ),
            ),
            operation_key=key,
            gqa_dispatch_trace_raw=GQA_DISPATCH_RAW,
            mha_dispatch_trace_raw=MHA_DISPATCH_RAW,
        )
        self.assertFalse(result.passed)
        self.assertIn(
            "mha_control:eager_split_k_pair_duplicate",
            result.failure_reasons,
        )

    def test_geometry_specific_split_counts_pass_independently(self) -> None:
        key = operation_key()
        result = verify_phase3_paired_allocator_controls(
            gqa_raw=observation_bytes(
                role="gqa",
                key=key,
                trace=eager_trace(split_sizes=(180224, 1408)),
            ),
            mha_control_raw=observation_bytes(
                role="mha_control",
                key=key,
                trace=eager_trace(split_sizes=(81920, 640)),
            ),
            operation_key=key,
            gqa_dispatch_trace_raw=GQA_DISPATCH_RAW,
            mha_dispatch_trace_raw=MHA_DISPATCH_RAW,
        )
        self.assertTrue(result.passed, result.failure_reasons)
        self.assertEqual(result.split_k_pair_multiplicity, ((11, 1),))
        self.assertEqual(
            tuple(
                (fact.formula_id, fact.num_splits)
                for fact in result.mha_control.split_k_facts
            ),
            (
                ("flash_split_k_output_accumulator", 5),
                ("flash_split_k_lse", 5),
            ),
        )


if __name__ == "__main__":
    unittest.main()
