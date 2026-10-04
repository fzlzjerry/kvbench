"""Narrow CUDA admission checks for the pinned TurboQuant adapter."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kvbench.adapters.turboquant import (
    TURBOQUANT_SOURCE_COMMIT,
    TURBOQUANT_SOURCE_TREE,
    TurboQuantMethodAdapter,
)
from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.cuda_graph import capture_fixed_graph
from kvbench.runtime.gqa_device_dispatch import collect_torch_profiler_trace
from kvbench.runtime.numerical import compare_tensors_untimed
from kvbench.runtime.turboquant_audit import (
    audit_turboquant_execution_path,
)
from kvbench.runtime.turboquant_cache import (
    TURBOQUANT_BF16_LAYERS,
    TURBOQUANT_COMPRESSED_LAYERS,
    TURBOQUANT_MANDATORY_CONFIGS,
    TURBOQUANT_SLOT_SIZES,
    _release_tensor_storages_for_sanitizer,
)
from kvbench.runtime.turboquant_session import turboquant_runtime_context
from kvbench.runtime.turboquant_fixture import (
    compare_decode_output_untimed,
    compare_packed_output_untimed,
    compare_slot_layout,
    load_inputs_cpu,
    load_turboquant_fixture,
)
from kvbench.schema.phase6 import (
    AUTHORIZED_CONTAINER_DIGEST,
    DECODE_SOURCE_SHA256,
)


PHASE6_IMAGE_ENVIRONMENT_VARIABLE = "KVBENCH_AUTHORIZED_IMAGE_DIGEST"
PHASE6_CONTAINER_ENVIRONMENT_VARIABLE = "KVBENCH_EXECUTION_ENVIRONMENT"
PHASE6_CONTAINER_ENVIRONMENT_VALUE = "measurement_container"
EXPECTED_TORCH_VERSION = "2.12.1+cu130"
EXPECTED_TRITON_VERSION = "3.7.1"
EXPECTED_CUDA_RUNTIME = "13.0"
EXPECTED_GPU_NAME = (
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
)
EXPECTED_GPU_UUID = "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"


class TurboQuantAdmissionError(RuntimeError):
    """A frozen Phase 6 admission condition did not hold."""


def strict_zero_allocation(evidence: Any) -> bool:
    """Apply the frozen zero-event criterion to one common audit result."""

    return bool(
        evidence.audit_available
        and evidence.allocation_event_count == 0
        and evidence.allocated_after == evidence.allocated_before
        and evidence.reserved_after == evidence.reserved_before
    )


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise TurboQuantAdmissionError("PyTorch is unavailable") from error
    return torch


def require_authorized_cuda_environment(
    declared_digest: str,
) -> dict[str, Any]:
    """Reject native CUDA and every image identity except Decision 0016."""

    if (
        declared_digest != AUTHORIZED_CONTAINER_DIGEST
        or os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        != AUTHORIZED_CONTAINER_DIGEST
        or os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        != PHASE6_CONTAINER_ENVIRONMENT_VALUE
        or not Path("/.dockerenv").is_file()
    ):
        raise TurboQuantAdmissionError(
            "Phase 6 CUDA requires the exact authorized Measurement Container"
        )
    torch = _torch()
    try:
        import triton
    except ModuleNotFoundError as error:
        raise TurboQuantAdmissionError("pinned Triton is unavailable") from error
    if (
        str(torch.__version__) != EXPECTED_TORCH_VERSION
        or str(torch.version.cuda) != EXPECTED_CUDA_RUNTIME
        or str(triton.__version__) != EXPECTED_TRITON_VERSION
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability(0) != (12, 0)
    ):
        raise TurboQuantAdmissionError(
            "authorized container CUDA runtime identity differs"
        )
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    raw_uuid = str(getattr(properties, "uuid", ""))
    uuid = raw_uuid if raw_uuid.startswith("GPU-") else f"GPU-{raw_uuid}"
    if name != EXPECTED_GPU_NAME or uuid != EXPECTED_GPU_UUID:
        raise TurboQuantAdmissionError(
            "authorized Measurement Container GPU identity differs"
        )
    return {
        "container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "execution_environment": PHASE6_CONTAINER_ENVIRONMENT_VALUE,
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "compute_capability": "12.0",
        "gpu_name": name,
        "gpu_uuid": uuid,
        "native_host_cuda_rejected": True,
    }


def _trace_names(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TurboQuantAdmissionError("raw profiler trace is invalid") from error
    if not isinstance(payload, dict) or not isinstance(
        payload.get("traceEvents"), list
    ):
        raise TurboQuantAdmissionError("raw profiler trace lacks events")
    names: list[str] = []
    for event in payload["traceEvents"]:
        if not isinstance(event, Mapping):
            raise TurboQuantAdmissionError("raw profiler event is invalid")
        name = event.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    if not names:
        raise TurboQuantAdmissionError("raw profiler trace is empty")
    return tuple(names)


def _scoped_runtime_names(path: Path, marker: str) -> tuple[str, ...]:
    """Return only CUDA runtime calls lexically inside the profiler marker."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TurboQuantAdmissionError("raw profiler trace is invalid") from error
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise TurboQuantAdmissionError("raw profiler trace lacks events")
    markers = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("cat") == "user_annotation"
        and event.get("name") == marker
        and event.get("ph") == "X"
    ]
    if len(markers) != 1:
        raise TurboQuantAdmissionError("raw trace marker is ambiguous")
    root = markers[0]
    start = root.get("ts")
    duration = root.get("dur")
    pid = root.get("pid")
    tid = root.get("tid")
    if (
        not isinstance(start, (int, float))
        or isinstance(start, bool)
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration < 0
    ):
        raise TurboQuantAdmissionError("raw trace marker interval is invalid")
    stop = float(start) + float(duration)
    names: list[str] = []
    for event in events:
        if (
            not isinstance(event, Mapping)
            or event.get("cat") != "cuda_runtime"
            or event.get("pid") != pid
            or event.get("tid") != tid
            or event.get("ph") != "X"
        ):
            continue
        event_start = event.get("ts")
        event_duration = event.get("dur")
        name = event.get("name")
        if (
            isinstance(event_start, (int, float))
            and not isinstance(event_start, bool)
            and isinstance(event_duration, (int, float))
            and not isinstance(event_duration, bool)
            and isinstance(name, str)
            and float(event_start) >= float(start)
            and float(event_start) + float(event_duration) <= stop
        ):
            names.append(name)
    return tuple(names)


def _kernel_family_names(
    names: tuple[str, ...],
    families: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in names
        if any(family in name for family in families)
    )


def _load_inputs(configuration: str, device: Any) -> tuple[Any, dict[str, Any]]:
    fixture = load_turboquant_fixture(configuration)
    inputs = {
        name: tensor.to(device=device)
        for name, tensor in load_inputs_cpu(fixture).items()
    }
    return fixture, inputs


def release_fixture_cuda_resources_for_sanitizer(
    resources: dict[str, Any],
) -> None:
    """Release graph-free fixture storages after all results are materialized."""

    inputs = resources.get("inputs")
    positions = resources.get("positions")
    cache = resources.get("cache")
    input_tensors = (
        tuple(inputs.values())
        if isinstance(inputs, dict)
        else ()
    )
    position_tensors = (
        tuple(positions)
        if isinstance(positions, list)
        else ()
    )

    cuda_device = getattr(cache, "device", None)
    if getattr(cuda_device, "type", None) != "cuda":
        for tensor in (*input_tensors, *position_tensors):
            tensor_device = getattr(tensor, "device", None)
            if getattr(tensor_device, "type", None) == "cuda":
                cuda_device = tensor_device
                break
    if getattr(cuda_device, "type", None) == "cuda":
        torch = _torch()
        torch.cuda.synchronize(device=cuda_device)

    _release_tensor_storages_for_sanitizer(
        (*input_tensors, *position_tensors)
    )
    if cache is not None:
        cache.release_owned_cuda_resources_for_sanitizer()
    if isinstance(inputs, dict):
        inputs.clear()
    if isinstance(positions, list):
        positions.clear()
    resources.clear()


def _slot_layout(cache: Any) -> dict[str, int]:
    config = cache.tq_config
    key_bytes = math.ceil(cache.head_dim * int(config.key_mse_bits) / 8)
    value_bytes = math.ceil(
        cache.head_dim * int(config.value_quant_bits) / 8
    )
    raw = {
        "packed_keys": key_bytes,
        "key_norm": 2,
        "packed_values": value_bytes,
        "value_scale": 2,
        "value_zero_point": 2,
    }
    raw["alignment_padding"] = cache.slot_size - sum(raw.values())
    return raw


def _verify_bf16_boundary_store(
    method: TurboQuantMethodAdapter,
    cache: Any,
    key_states: Any,
    value_states: Any,
    cache_position: Any,
) -> dict[str, Any]:
    torch = _torch()
    layers: dict[str, dict[str, bool]] = {}
    for slot, layer_idx in enumerate(TURBOQUANT_BF16_LAYERS):
        returned_key, returned_value = method.store_prefill(
            cache,
            key_states,
            value_states,
            layer_idx,
            cache_position,
        )
        observed_key = cache.bf16_cache.keys[
            slot, :, :, : int(key_states.shape[2]), :
        ]
        observed_value = cache.bf16_cache.values[
            slot, :, :, : int(value_states.shape[2]), :
        ]
        layers[str(layer_idx)] = {
            "returned_key_exact": bool(torch.equal(returned_key, key_states)),
            "returned_value_exact": bool(
                torch.equal(returned_value, value_states)
            ),
            "stored_key_exact": bool(torch.equal(observed_key, key_states)),
            "stored_value_exact": bool(
                torch.equal(observed_value, value_states)
            ),
        }
    return {
        "layers": layers,
        "passed": all(all(checks.values()) for checks in layers.values()),
    }


def _verify_bf16_boundary_append(
    method: TurboQuantMethodAdapter,
    cache: Any,
    key_states: Any,
    value_states: Any,
    cache_position: Any,
) -> dict[str, Any]:
    torch = _torch()
    position = int(cache_position[0].item())
    layers: dict[str, dict[str, bool]] = {}
    for slot, layer_idx in enumerate(TURBOQUANT_BF16_LAYERS):
        returned_key, returned_value = method.append_decode(
            cache,
            key_states,
            value_states,
            layer_idx,
            cache_position,
        )
        observed_key = cache.bf16_cache.keys[
            slot, :, :, position : position + 1, :
        ]
        observed_value = cache.bf16_cache.values[
            slot, :, :, position : position + 1, :
        ]
        layers[str(layer_idx)] = {
            "returned_key_exact": bool(
                torch.equal(
                    returned_key[:, :, position : position + 1, :],
                    key_states,
                )
            ),
            "returned_value_exact": bool(
                torch.equal(
                    returned_value[:, :, position : position + 1, :],
                    value_states,
                )
            ),
            "stored_key_exact": bool(torch.equal(observed_key, key_states)),
            "stored_value_exact": bool(
                torch.equal(observed_value, value_states)
            ),
        }
    return {
        "layers": layers,
        "passed": all(all(checks.values()) for checks in layers.values()),
    }


def evaluate_fixture_configuration(
    configuration: str,
    *,
    evidence_directory: Path | None = None,
    release_cuda_resources_for_sanitizer: bool = False,
    sanitizer_resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one mandatory fixture; optional evidence enables all CUDA audits."""

    if configuration not in TURBOQUANT_MANDATORY_CONFIGS:
        raise TurboQuantAdmissionError(
            "fixture configuration is not in the mandatory family"
        )
    if release_cuda_resources_for_sanitizer and evidence_directory is not None:
        raise TurboQuantAdmissionError(
            "sanitizer resource release requires the graph-free fixture path"
        )
    if release_cuda_resources_for_sanitizer:
        if sanitizer_resources is None:
            raise TurboQuantAdmissionError(
                "sanitizer release requires an explicit resource registry"
            )
        if sanitizer_resources:
            raise TurboQuantAdmissionError(
                "sanitizer resource registry must start empty"
            )
    elif sanitizer_resources is not None:
        raise TurboQuantAdmissionError("unexpected sanitizer resource registry")
    torch = _torch()
    device = torch.device("cuda:0")
    fixture, inputs = _load_inputs(configuration, device)
    if sanitizer_resources is not None:
        sanitizer_resources["inputs"] = inputs
    method = TurboQuantMethodAdapter(turboquant_runtime_context(), configuration)
    cache = method.allocate(
        batch_size=1,
        capacity=18,
        device=device,
        workspace_bytes=0,
    )
    if sanitizer_resources is not None:
        sanitizer_resources["cache"] = cache
    cache.initialize_deterministic()
    pointers_before = cache.pointers()
    if sanitizer_resources is not None:
        sanitizer_resources["positions"] = []
    prefill_positions = torch.arange(
        17,
        dtype=torch.int32,
        device=device,
    )
    if sanitizer_resources is not None:
        sanitizer_resources["positions"].append(prefill_positions)
    append_position = torch.tensor([17], dtype=torch.int32, device=device)
    if sanitizer_resources is not None:
        sanitizer_resources["positions"].append(append_position)
    prefill_key = inputs["prefill_key"].transpose(0, 1).unsqueeze(0)
    prefill_value = inputs["prefill_value"].transpose(0, 1).unsqueeze(0)
    append_key = inputs["append_key"].transpose(0, 1).unsqueeze(0)
    append_value = inputs["append_value"].transpose(0, 1).unsqueeze(0)
    query = inputs["decode_query"].unsqueeze(2)
    attention = SimpleNamespace(layer_idx=2)

    directory: Path | None = None
    store_allocation: Any | None = None
    store_trace: Any | None = None
    store_trace_path: Path | None = None
    if evidence_directory is not None:
        directory = Path(evidence_directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        store_trace_path = directory / "store.chrome.json"

    cache.prepare_prefill(17)
    bf16_store = _verify_bf16_boundary_store(
        method,
        cache,
        prefill_key,
        prefill_value,
        prefill_positions,
    )

    def store() -> Any:
        return method.store_prefill(
            cache,
            prefill_key,
            prefill_value,
            2,
            prefill_positions,
        )

    store()
    torch.cuda.synchronize(device=device)
    store_comparison = compare_packed_output_untimed(
        fixture,
        "cache_after_store",
        cache.compressed_layer_cache(2),
    )
    if directory is not None and store_trace_path is not None:
        store_allocation = audit_cuda_allocations(store, device=device)
        store_trace = collect_torch_profiler_trace(
            store,
            store_trace_path,
            artifact_relative_path=(
                f"validation/{configuration}/store.chrome.json"
            ),
            marker=f"phase6::{configuration}::store",
            warmup_count=1,
            device=device,
        )
    cache.complete_prefill()
    cache.prepare_fixed(17)
    bf16_append = _verify_bf16_boundary_append(
        method,
        cache,
        append_key,
        append_value,
        append_position,
    )
    handle: Any | None = None

    def hot_operation() -> Any:
        nonlocal handle
        handle, _ = method.append_decode(
            cache,
            append_key,
            append_value,
            2,
            append_position,
        )
        return method.decode_attention(
            attention,
            query,
            handle,
            handle,
            scaling=1.0 / math.sqrt(128),
        )

    output = hot_operation()
    torch.cuda.synchronize(device=device)
    append_comparison = compare_packed_output_untimed(
        fixture,
        "cache_after_append",
        cache.compressed_layer_cache(2),
    )
    slot_comparison = compare_packed_output_untimed(
        fixture,
        "append_slot",
        cache.compressed_layer_cache(2)[1, 1],
    )
    decode_comparison = compare_decode_output_untimed(
        fixture,
        output,
    )
    layout_comparison = compare_slot_layout(
        fixture,
        slot_size=cache.slot_size,
        byte_breakdown_per_head_token=_slot_layout(cache),
    )
    accounting = cache.accounting()
    relative_error = (
        abs(accounting.predicted_tensor_bytes - accounting.allocated_bytes)
        / accounting.allocated_bytes
    )
    base_result: dict[str, Any] = {
        "configuration": configuration,
        "fixture_authority": fixture.authority.to_dict(),
        "store": store_comparison.to_dict(),
        "append": append_comparison.to_dict(),
        "appended_slot": slot_comparison.to_dict(),
        "slot_layout": layout_comparison.to_dict(),
        "decode": decode_comparison.to_dict(),
        "byte_breakdown": cache.byte_breakdown(),
        "accounting": accounting.to_dict(),
        "logical_bf16_bytes": cache.logical_bf16_storage_bytes,
        "r_nominal": cache.r_nominal,
        "r_alloc": cache.r_alloc,
        "r_hbm": None,
        "predicted_allocated_relative_error": relative_error,
        "block_size": cache.block_size,
        "block_count": cache.block_count,
        "block_table": list(range(cache.block_count)),
        "slot_mapping": "deterministic_contiguous",
        "compressed_layers": list(cache.compressed_layers),
        "bf16_layers": list(cache.bf16_layers),
        "slot_size": cache.slot_size,
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "gqa_geometry": cache.gqa_geometry(),
        "pointers_stable": pointers_before == cache.pointers(),
        "source_commit": TURBOQUANT_SOURCE_COMMIT,
        "source_tree": TURBOQUANT_SOURCE_TREE,
        "bf16_boundary_store": bf16_store,
        "bf16_boundary_append": bf16_append,
        "allocation": None,
        "execution_path": None,
        "graph": None,
    }
    if evidence_directory is None:
        base_result["passed"] = all(
            (
                store_comparison.passed,
                append_comparison.passed,
                slot_comparison.passed,
                layout_comparison.passed,
                decode_comparison.passed,
                bf16_store["passed"],
                bf16_append["passed"],
                relative_error < 0.01,
                pointers_before == cache.pointers(),
            )
        )
        if release_cuda_resources_for_sanitizer:
            store = None
            hot_operation = None
            handle = None
            output = None
            prefill_key = None
            prefill_value = None
            append_key = None
            append_value = None
            query = None
            prefill_positions = None
            append_position = None
            if sanitizer_resources is None:
                raise TurboQuantAdmissionError(
                    "sanitizer resource registry disappeared"
                )
            release_fixture_cuda_resources_for_sanitizer(
                sanitizer_resources
            )
            cache = None
            method = None
        return base_result

    if (
        directory is None
        or store_trace_path is None
        or store_trace is None
        or store_allocation is None
    ):
        raise TurboQuantAdmissionError("store audit setup is incomplete")
    hot_allocation = audit_cuda_allocations(hot_operation, device=device)
    first_trace_path = directory / "decode-first.chrome.json"
    second_trace_path = directory / "decode-second.chrome.json"
    first_trace = collect_torch_profiler_trace(
        hot_operation,
        first_trace_path,
        artifact_relative_path=(
            f"validation/{configuration}/decode-first.chrome.json"
        ),
        marker=f"phase6::{configuration}::decode-first",
        warmup_count=1,
        device=device,
    )
    second_trace = collect_torch_profiler_trace(
        hot_operation,
        second_trace_path,
        artifact_relative_path=(
            f"validation/{configuration}/decode-second.chrome.json"
        ),
        marker=f"phase6::{configuration}::decode-second",
        warmup_count=1,
        device=device,
    )
    store_names = _trace_names(store_trace_path)
    first_names = _trace_names(first_trace_path)
    second_names = _trace_names(second_trace_path)
    store_kernels = _kernel_family_names(
        store_names,
        ("_tq_fused_store_mse",),
    )
    first_decode = _kernel_family_names(
        first_names,
        ("_tq_decode_stage1", "_fwd_kernel_stage2"),
    )
    second_decode = _kernel_family_names(
        second_names,
        ("_tq_decode_stage1", "_fwd_kernel_stage2"),
    )
    decode_source_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "vllm_turboquant"
        / "triton_turboquant_decode.py"
    )
    decode_source = decode_source_path.read_bytes()
    execution_path = audit_turboquant_execution_path(
        store_kernel_names=store_kernels,
        append_kernel_names=_kernel_family_names(
            first_names,
            ("_tq_fused_store_mse",),
        ),
        decode_kernel_names=first_decode,
        repeated_decode_kernel_names=second_decode,
        runtime_event_names=_scoped_runtime_names(
            first_trace_path,
            f"phase6::{configuration}::decode-first",
        ),
        temporary_shapes={
            "packed_kv_cache": tuple(cache.packed_cache.shape),
            "store_key_workspace": tuple(cache.store_key_float.shape),
            "store_value_workspace": tuple(cache.store_value_float.shape),
            "decode_query_workspace": tuple(cache.decode_query_float.shape),
            "decode_split_workspace": tuple(cache.decode_mid_o.shape),
        },
        adapter_hot_path_source=(
            inspect.getsource(method._store_compressed)
            + inspect.getsource(method._decode_compressed)
        ),
        decode_kernel_source=decode_source,
        decode_kernel_sha256=DECODE_SOURCE_SHA256,
    )
    eager_reference = hot_operation().detach().cpu().clone()
    graph = capture_fixed_graph(
        hot_operation,
        warmup_steps=0,
        device=device,
    )
    first_graph = graph.replay().detach().cpu().clone()
    second_graph = graph.replay().detach().cpu().clone()
    torch.cuda.synchronize(device=device)
    graph_comparison = compare_tensors_untimed(
        first_graph,
        eager_reference,
        atol=0.02,
        rtol=0.02,
    )
    graph_allocation = audit_cuda_allocations(graph.replay, device=device)
    graph_passed = (
        graph_comparison.passed
        and bool(torch.equal(first_graph, second_graph))
        and strict_zero_allocation(graph_allocation)
        and graph.to_dict()["fallback"] is False
        and pointers_before == cache.pointers()
    )
    allocation_passed = (
        strict_zero_allocation(store_allocation)
        and strict_zero_allocation(hot_allocation)
        and strict_zero_allocation(graph_allocation)
    )
    base_result["allocation"] = {
        "passed": allocation_passed,
        "store": store_allocation.to_dict(),
        "append_decode": hot_allocation.to_dict(),
        "graph_replay": graph_allocation.to_dict(),
        "unknown_allocations": 0 if allocation_passed else None,
    }
    base_result["execution_path"] = {
        **execution_path.to_dict(),
        "raw_traces": [
            store_trace.to_dict(),
            first_trace.to_dict(),
            second_trace.to_dict(),
        ],
    }
    base_result["graph"] = {
        **graph.to_dict(),
        "passed": graph_passed,
        "eager_graph_comparison": graph_comparison.to_dict(),
        "replay_outputs_exact": bool(torch.equal(first_graph, second_graph)),
        "replay_allocation": graph_allocation.to_dict(),
        "pointers_stable": pointers_before == cache.pointers(),
    }
    base_result["passed"] = all(
        (
            store_comparison.passed,
            append_comparison.passed,
            slot_comparison.passed,
            layout_comparison.passed,
            decode_comparison.passed,
            bf16_store["passed"],
            bf16_append["passed"],
            relative_error < 0.01,
            pointers_before == cache.pointers(),
            allocation_passed,
            execution_path.passed,
            graph_passed,
        )
    )
    return base_result


def mandatory_configuration_summary(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Join exactly the three mandatory records without expanding the grid."""

    if tuple(records) != TURBOQUANT_MANDATORY_CONFIGS:
        raise TurboQuantAdmissionError(
            "mandatory configuration records are missing or reordered"
        )
    passed = all(record.get("passed") is True for record in records.values())
    return {
        "source_commit": TURBOQUANT_SOURCE_COMMIT,
        "source_tree": TURBOQUANT_SOURCE_TREE,
        "configurations": dict(records),
        "mandatory_configurations": list(TURBOQUANT_MANDATORY_CONFIGS),
        "slot_sizes": dict(TURBOQUANT_SLOT_SIZES),
        "compressed_layers": list(TURBOQUANT_COMPRESSED_LAYERS),
        "bf16_layers": list(TURBOQUANT_BF16_LAYERS),
        "passed": passed,
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
