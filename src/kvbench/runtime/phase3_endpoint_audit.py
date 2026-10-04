"""Concrete Phase 3 endpoint session for dispatch and allocation admission.

This module deliberately contains one endpoint-specific session and one builder.
It reuses the frozen endpoint, cache, graph, allocator-witness, and operation-key
primitives; it is not a generic session framework.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
from typing import Any

from kvbench.adapters import (
    KVCacheMethod,
    MethodRuntimeContext,
    build_method_adapter,
)
from kvbench.runtime.backend import BACKEND_IDENTITY
from kvbench.runtime.allocation import (
    MemorySnapshot,
    capture_cuda_memory_snapshot,
)
from kvbench.runtime.allocation_attribution import (
    OperationCacheStateWitness,
    OperationWitnessCallbacks,
    capture_output_witness_d2h,
)
from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
from kvbench.runtime.cuda_graph import CapturedFixedGraph, capture_fixed_graph
from kvbench.runtime.model_loader import (
    LoadedFrozenModel,
    MODEL_ID,
    MODEL_REVISION,
    validate_loaded_frozen_model_receipt,
)
from kvbench.runtime.numerical import (
    NumericalComparison,
    compare_tensors_untimed,
    tensor_sha256_untimed,
)
from kvbench.runtime.phase3_audit_operation import (
    Phase3AuditOperationKey,
    validate_phase3_audit_operation_set,
)
from kvbench.runtime.static_cache import BF16StaticCache
from kvbench.schema import GraphMode, MethodConfig, RunnerKind


PHASE3_ENDPOINT_SESSION_SCHEMA_VERSION = (
    "kvbench-phase3-endpoint-session-1.0.0"
)
PHASE3_ENDPOINT_WARMUP_OPERATIONS = 16
_LAYERS = 32
_QUERY_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = 128
_DTYPE_BYTES = 2
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class Phase3EndpointAuditError(RuntimeError):
    """The concrete endpoint session failed closed before timing."""


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise Phase3EndpointAuditError("PyTorch is unavailable") from error


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise Phase3EndpointAuditError(f"{label} is not a SHA-256")
    return value


def _cpu_row_bytes(value: Any) -> bytes:
    """Copy one contiguous cache row to CPU without a CUDA temporary."""

    torch = _torch()
    row = value.detach().to(device="cpu", non_blocking=False, copy=True)
    return row.contiguous().view(torch.uint8).numpy().tobytes(order="C")


def _cache_region_component_sha256(
    value: Any,
    *,
    start: int,
    length: int,
) -> str:
    shape = (
        int(value.shape[0]),
        int(value.shape[1]),
        int(value.shape[2]),
        length,
        int(value.shape[4]),
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    stop = start + length
    for layer in range(shape[0]):
        for batch in range(shape[1]):
            for head in range(shape[2]):
                digest.update(
                    _cpu_row_bytes(value[layer, batch, head, start:stop, :])
                )
    return digest.hexdigest()


def _cache_pair_sha256(
    cache: BF16StaticCache,
    *,
    start: int,
    length: int,
) -> str:
    key = _cache_region_component_sha256(
        cache.keys,
        start=start,
        length=length,
    )
    value = _cache_region_component_sha256(
        cache.values,
        start=start,
        length=length,
    )
    return hashlib.sha256(f"{key}:{value}".encode("ascii")).hexdigest()


def _zero_destination_sha256(batch_size: int) -> str:
    shape = (_LAYERS, batch_size, _KV_HEADS, 1, _HEAD_DIM)
    header = json.dumps(
        {"shape": list(shape), "dtype": "torch.bfloat16"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    component = hashlib.sha256(
        header
        + b"\0"
        + b"\0" * (_LAYERS * batch_size * _KV_HEADS * _HEAD_DIM * _DTYPE_BYTES)
    ).hexdigest()
    return hashlib.sha256(f"{component}:{component}".encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class EndpointAuditCall:
    """Exact callables supplied to the existing allocator collector."""

    operation_key: Phase3AuditOperationKey
    operation: Callable[[], Any]
    warmup_operation: Callable[[], Any]
    prepare_operation: Callable[[], None]
    operation_witness: OperationWitnessCallbacks

    def __post_init__(self) -> None:
        if type(self.operation_key) is not Phase3AuditOperationKey or any(
            not callable(value)
            for value in (
                self.operation,
                self.warmup_operation,
                self.prepare_operation,
            )
        ):
            raise Phase3EndpointAuditError("endpoint audit call is invalid")


class Phase3EndpointSession:
    """One concrete endpoint/cache shared by audit and measurement."""

    def __init__(
        self,
        *,
        loaded: LoadedFrozenModel,
        operation_keys: tuple[Phase3AuditOperationKey, ...],
        prefix_input_ids: Any,
        decode_input_ids: Any,
        endpoint: BF16DecodeEndpoint,
        cache: BF16StaticCache,
        method: KVCacheMethod,
        adapter_config_fingerprint: str,
        model_memory: MemorySnapshot,
        cache_memory: MemorySnapshot,
        fixed_operation: Callable[[], Any] | None,
        graph: CapturedFixedGraph | None,
        graph_evidence: Mapping[str, object] | None,
        eager_graph_comparison: NumericalComparison | None,
        growing_operations: tuple[Callable[[], Any], ...],
        reset_growing: Callable[[], None] | None,
        fixed_destination_backup: tuple[Any, Any] | None,
        prefix_sha256: str,
    ) -> None:
        self.loaded = loaded
        self.operation_keys = operation_keys
        self.prefix_input_ids = prefix_input_ids
        self.decode_input_ids = decode_input_ids
        self.endpoint = endpoint
        self.cache = cache
        self.method = method
        self.adapter_config_fingerprint = _require_sha256(
            adapter_config_fingerprint,
            "adapter config fingerprint",
        )
        self.model_memory = model_memory
        self.cache_memory = cache_memory
        self.graph_evidence = (
            None if graph_evidence is None else dict(graph_evidence)
        )
        self.eager_graph_comparison = eager_graph_comparison
        self.receipt_sha256 = _require_sha256(
            loaded.receipt.receipt_sha256,
            "loaded model receipt",
        )
        self.cache_pointers = cache.pointers()
        self._fixed_operation = fixed_operation
        self._graph = graph
        self._growing_operations = growing_operations
        self._reset_growing = reset_growing
        self._fixed_destination_backup = fixed_destination_backup
        self._prefix_sha256 = prefix_sha256
        self._history_chain_sha256 = prefix_sha256
        self._audit_references: dict[int, tuple[str, str]] = {}
        self._audit_outputs: dict[int, tuple[str, bool]] = {}
        self._next_audit_step = 0
        self._state = "warmed"
        self._measurement_started = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def graph(self) -> CapturedFixedGraph | None:
        return self._graph

    @property
    def expected_audit_count(self) -> int:
        return len(self.operation_keys)

    @property
    def historical_prefix_sha256(self) -> str:
        return self._prefix_sha256

    @property
    def cache_device(self) -> Any:
        return self.cache.device

    @property
    def active_context(self) -> int:
        return int(self.cache.active_context)

    def current_cache_pointers(self) -> dict[str, int]:
        return self.cache.pointers()

    def method_cache_accounting(self) -> dict[str, int]:
        accounting = self.cache.accounting().to_dict()
        if self.method.allocated_bytes(self.cache) != accounting["allocated_bytes"]:
            raise Phase3EndpointAuditError("method and cache bytes differ")
        return accounting

    def method_byte_breakdown(self) -> dict[str, int]:
        return dict(sorted(self.method.byte_breakdown(self.cache).items()))

    def cache_layout_fingerprint(self) -> str:
        return self.cache.layout_fingerprint()

    def gqa_cache_geometry(self) -> dict[str, Any]:
        from kvbench.runtime.gqa_audit import audit_cache_geometry

        return audit_cache_geometry(
            self.cache,
            num_query_heads=_QUERY_HEADS,
        )

    def current_historical_prefix_sha256(self) -> str:
        first = self.operation_keys[0]
        return _cache_pair_sha256(
            self.cache,
            start=0,
            length=first.historical_context - first.decode_step,
        )

    def audit_output(self, step: int) -> tuple[str, bool]:
        try:
            return self._audit_outputs[step]
        except KeyError as error:
            raise Phase3EndpointAuditError(
                "endpoint audit output is unavailable"
            ) from error

    def gqa_cache_views(self, step: int) -> tuple[Any, Any]:
        key = self._operation_key(step)
        stop = key.attended_context
        return (
            self.cache.keys[0, :, :, :stop, :],
            self.cache.values[0, :, :, :stop, :],
        )

    def _operation_key(self, step: int) -> Phase3AuditOperationKey:
        if type(step) is not int or step < 0 or step >= len(self.operation_keys):
            raise Phase3EndpointAuditError("endpoint session step is invalid")
        return self.operation_keys[step]

    def _destination_views(self, step: int) -> tuple[Any, Any]:
        key = self._operation_key(step)
        position = key.historical_context
        return (
            self.cache.keys[:, :, :, position : position + 1, :],
            self.cache.values[:, :, :, position : position + 1, :],
        )

    def _execute_step(self, step: int) -> Any:
        key = self._operation_key(step)
        if key.runner_kind is RunnerKind.FIXED_L:
            if key.graph_mode is GraphMode.CUDA_GRAPH:
                if self._graph is None:
                    raise Phase3EndpointAuditError("retained graph is absent")
                return self._graph.replay()
            if self._fixed_operation is None:
                raise Phase3EndpointAuditError("fixed eager callable is absent")
            return self._fixed_operation()
        return self._growing_operations[step]()

    def _prepare_step(self, step: int) -> None:
        key = self._operation_key(step)
        if key.runner_kind is RunnerKind.FIXED_L:
            destination = self._destination_views(0)
            destination[0].zero_()
            destination[1].zero_()
            return
        if self._reset_growing is None:
            raise Phase3EndpointAuditError("growing reset is absent")
        self._reset_growing()
        for prior in range(step):
            self._growing_operations[prior]()
        destination = self._destination_views(step)
        destination[0].zero_()
        destination[1].zero_()

    def _capture_cache_state(self, step: int) -> OperationCacheStateWitness:
        key = self._operation_key(step)
        destination_sha256 = _cache_pair_sha256(
            self.cache,
            start=key.historical_context,
            length=1,
        )
        sentinel = _zero_destination_sha256(key.batch_size)
        return OperationCacheStateWitness(
            active_length=int(self.cache.active_context),
            key_shape=tuple(int(item) for item in self.cache.keys.shape),
            value_shape=tuple(int(item) for item in self.cache.values.shape),
            key_strides=tuple(int(item) for item in self.cache.keys.stride()),
            value_strides=tuple(int(item) for item in self.cache.values.stride()),
            key_dtype=str(self.cache.keys.dtype),
            value_dtype=str(self.cache.values.dtype),
            key_device=str(self.cache.keys.device),
            value_device=str(self.cache.values.device),
            key_data_ptr=int(self.cache.keys.data_ptr()),
            value_data_ptr=int(self.cache.values.data_ptr()),
            historical_prefix_sha256=self._history_chain_sha256,
            destination_slot_sha256=destination_sha256,
            destination_slot_is_sentinel=destination_sha256 == sentinel,
            layout_fingerprint=self.cache.layout_fingerprint(),
        )

    def audit_call(self, step: int) -> EndpointAuditCall:
        """Return exact audit callables; no audit action enters normal timing."""

        if self._state not in {"warmed", "auditing"}:
            raise Phase3EndpointAuditError("session is not accepting audits")
        if step != self._next_audit_step:
            raise Phase3EndpointAuditError("audits must be ordered and complete")
        self._state = "auditing"

        def prepare() -> None:
            self._prepare_step(step)

        def operation() -> Any:
            return self._execute_step(step)

        def warmup() -> Any:
            prepare()
            return operation()

        return EndpointAuditCall(
            operation_key=self._operation_key(step),
            operation=operation,
            warmup_operation=warmup,
            prepare_operation=prepare,
            operation_witness=OperationWitnessCallbacks(
                capture_cache_state=lambda: self._capture_cache_state(step),
                capture_output=capture_output_witness_d2h,
            ),
        )

    def record_audit(
        self,
        step: int,
        *,
        dispatch_audit_sha256: str,
        allocation_audit_sha256: str,
        destination_slot_sha256: str,
        output_sha256: str,
        output_finite: bool,
        locally_verified: bool,
    ) -> None:
        """Gate timing locally; the coordinator still replays raw bytes."""

        if self._state != "auditing" or step != self._next_audit_step:
            raise Phase3EndpointAuditError("audit result is out of order")
        if locally_verified is not True:
            self._state = "failed"
            raise Phase3EndpointAuditError("endpoint audit did not verify")
        self._audit_references[step] = (
            _require_sha256(dispatch_audit_sha256, "dispatch audit"),
            _require_sha256(allocation_audit_sha256, "allocation audit"),
        )
        if type(output_finite) is not bool:
            raise Phase3EndpointAuditError("audit output finite flag is invalid")
        self._audit_outputs[step] = (
            _require_sha256(output_sha256, "audit output"),
            output_finite,
        )
        destination = _require_sha256(
            destination_slot_sha256,
            "destination slot",
        )
        self._history_chain_sha256 = hashlib.sha256(
            f"{self._history_chain_sha256}:{destination}".encode("ascii")
        ).hexdigest()
        self._next_audit_step += 1

    def finish_audits(self, *, release_audit_buffers: Callable[[], None]) -> None:
        """Restore the timing state, release audit storage, and admit timing."""

        if (
            self._state != "auditing"
            or self._next_audit_step != len(self.operation_keys)
            or len(self._audit_references) != len(self.operation_keys)
            or len(self._audit_outputs) != len(self.operation_keys)
        ):
            raise Phase3EndpointAuditError("endpoint audit set is incomplete")
        first = self.operation_keys[0]
        if first.runner_kind is RunnerKind.FIXED_L:
            if self._fixed_destination_backup is None:
                raise Phase3EndpointAuditError("fixed destination backup is absent")
            destination = self._destination_views(0)
            destination[0].copy_(self._fixed_destination_backup[0])
            destination[1].copy_(self._fixed_destination_backup[1])
        else:
            if self._reset_growing is None:
                raise Phase3EndpointAuditError("growing reset is absent")
            self._reset_growing()
        prefix_after = _cache_pair_sha256(
            self.cache,
            start=0,
            length=first.historical_context - first.decode_step,
        )
        if prefix_after != self._prefix_sha256:
            self._state = "failed"
            raise Phase3EndpointAuditError("cache prefix changed during audit")
        validate_loaded_frozen_model_receipt(self.loaded)
        if (
            self.loaded.receipt.receipt_sha256 != self.receipt_sha256
            or self.cache.pointers() != self.cache_pointers
        ):
            self._state = "failed"
            raise Phase3EndpointAuditError("session identity changed during audit")
        release_audit_buffers()
        self._fixed_destination_backup = None
        self._state = "ready"

    def fixed_measurement_callable(self) -> Callable[[], Any]:
        """Return the exact eager callable or retained graph after admission."""

        first = self.operation_keys[0]
        if self._state != "ready" or first.runner_kind is not RunnerKind.FIXED_L:
            raise Phase3EndpointAuditError("fixed timing is not admitted")
        self._measurement_started = True
        self._state = "measuring"
        if first.graph_mode is GraphMode.CUDA_GRAPH:
            if self._graph is None:
                raise Phase3EndpointAuditError("retained graph is absent")
            return self._graph.replay
        if self._fixed_operation is None:
            raise Phase3EndpointAuditError("fixed eager callable is absent")
        return self._fixed_operation

    def growing_measurement_callables(self) -> tuple[Callable[[], Any], ...]:
        """Return the exact ordered trajectory after reset/prefill admission."""

        first = self.operation_keys[0]
        if (
            self._state != "ready"
            or first.runner_kind is not RunnerKind.GROWING_CONTEXT
        ):
            raise Phase3EndpointAuditError("growing timing is not admitted")
        self._measurement_started = True
        self._state = "measuring"
        return self._growing_operations

    def mark_measured(self) -> None:
        if self._state != "measuring" or not self._measurement_started:
            raise Phase3EndpointAuditError("measurement did not start")
        self._state = "measured"

    def provenance_payload(self) -> Mapping[str, object]:
        if self._state not in {"ready", "measuring", "measured"}:
            raise Phase3EndpointAuditError("session provenance is not complete")
        return {
            "schema_version": PHASE3_ENDPOINT_SESSION_SCHEMA_VERSION,
            "method_name": self.method.name,
            "adapter_version": self.method.adapter_version,
            "adapter_config_fingerprint": self.adapter_config_fingerprint,
            "receipt_sha256": self.receipt_sha256,
            "cache_pointers": dict(self.cache_pointers),
            "cache_layout_fingerprint": self.cache.layout_fingerprint(),
            "operation_fingerprints": [
                key.operation_fingerprint_sha256 for key in self.operation_keys
            ],
            "dispatch_audit_sha256": [
                self._audit_references[step][0]
                for step in range(len(self.operation_keys))
            ],
            "allocation_audit_sha256": [
                self._audit_references[step][1]
                for step in range(len(self.operation_keys))
            ],
            "audit_output_sha256": [
                self._audit_outputs[step][0]
                for step in range(len(self.operation_keys))
            ],
            "audit_output_finite": [
                self._audit_outputs[step][1]
                for step in range(len(self.operation_keys))
            ],
            "graph_retained": self._graph is not None,
            "prefix_sha256": self._prefix_sha256,
            "history_chain_sha256": self._history_chain_sha256,
        }


def build_phase3_endpoint_session(
    *,
    loaded: LoadedFrozenModel,
    operation_keys: tuple[Phase3AuditOperationKey, ...],
    prefix_input_ids: Any,
    decode_input_ids: Any,
    method_config: MethodConfig | str = "bf16",
) -> Phase3EndpointSession:
    """Build, prefill, warm exactly once, and retain one production session."""

    validate_loaded_frozen_model_receipt(loaded)
    keys = validate_phase3_audit_operation_set(operation_keys)
    first = keys[0]
    method = build_method_adapter(
        method_config,
        MethodRuntimeContext(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            backend_id=str(BACKEND_IDENTITY["backend_id"]),
            backend_fingerprint=first.backend_identity_sha256,
            num_layers=_LAYERS,
            num_query_heads=_QUERY_HEADS,
            num_kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
        ),
    )

    expected_steps = 1 if first.runner_kind is RunnerKind.FIXED_L else 16
    if len(keys) != expected_steps:
        raise Phase3EndpointAuditError("operation set has the wrong run length")
    if (
        int(prefix_input_ids.ndim) != 2
        or int(decode_input_ids.ndim) != 2
        or int(prefix_input_ids.shape[0]) != first.batch_size
        or int(prefix_input_ids.shape[1])
        != first.historical_context - first.decode_step
        or int(decode_input_ids.shape[0]) != first.batch_size
        or int(decode_input_ids.shape[1]) != expected_steps
        or prefix_input_ids.device != decode_input_ids.device
    ):
        raise Phase3EndpointAuditError("endpoint session fixtures differ from plan")
    torch = _torch()
    workspace_bytes = (
        _LAYERS
        * first.batch_size
        * (_QUERY_HEADS + _KV_HEADS)
        * (_HEAD_DIM // 2)
        * _DTYPE_BYTES
    )
    model_memory = capture_cuda_memory_snapshot(
        "model_baseline",
        device=prefix_input_ids.device,
    )
    cache = method.allocate(
        batch_size=first.batch_size,
        capacity=first.capacity,
        device=prefix_input_ids.device,
        workspace_bytes=workspace_bytes,
    )
    if type(cache) is not BF16StaticCache:
        raise Phase3EndpointAuditError("Phase 3 requires BF16 static cache state")
    cache_memory = capture_cuda_memory_snapshot(
        "post_cache_allocation",
        device=prefix_input_ids.device,
    )
    if cache.layout_fingerprint() != first.cache_layout_fingerprint:
        raise Phase3EndpointAuditError(
            "endpoint cache differs from the operation key"
        )
    cache.initialize_deterministic()
    adapter_config_fingerprint = method.config_fingerprint(
        cache.layout_fingerprint()
    )
    endpoint = BF16DecodeEndpoint(loaded.model, cache, method)
    starting_context = int(prefix_input_ids.shape[1])
    positions = tuple(
        torch.tensor(
            [starting_context + step],
            dtype=torch.long,
            device=prefix_input_ids.device,
        )
        for step in range(expected_steps)
    )
    rope = tuple(
        endpoint.prepare_position_embeddings(position.unsqueeze(0))
        for position in positions
    )
    tokens = tuple(
        decode_input_ids[:, step : step + 1]
        for step in range(expected_steps)
    )

    def reset_growing() -> None:
        cache.reset_active_length(0)
        endpoint.prefill(prefix_input_ids)
        cache.prepare_growing(starting_context, expected_steps)

    fixed_operation: Callable[[], Any] | None = None
    growing_operations: tuple[Callable[[], Any], ...] = ()
    graph: CapturedFixedGraph | None = None
    graph_evidence: dict[str, object] | None = None
    eager_graph_comparison: NumericalComparison | None = None
    backup: tuple[Any, Any] | None = None
    endpoint.prefill(prefix_input_ids)
    if first.runner_kind is RunnerKind.FIXED_L:
        cache.prepare_fixed(starting_context)

        def fixed_step() -> Any:
            return endpoint.decode(tokens[0], positions[0], rope[0])

        fixed_operation = fixed_step
        last_warmup_output: Any | None = None
        for _ in range(PHASE3_ENDPOINT_WARMUP_OPERATIONS):
            last_warmup_output = fixed_step()
        if last_warmup_output is None:
            raise Phase3EndpointAuditError("fixed warmup produced no output")
        destination = (
            cache.keys[:, :, :, starting_context : starting_context + 1, :],
            cache.values[:, :, :, starting_context : starting_context + 1, :],
        )
        backup = (destination[0].clone(), destination[1].clone())
        if first.graph_mode is GraphMode.CUDA_GRAPH:
            torch.cuda.synchronize(device=cache.device)
            eager_reference = last_warmup_output.detach().cpu().clone()
            graph = capture_fixed_graph(
                fixed_step,
                warmup_steps=0,
                device=cache.device,
            )
            graph.replay()
            torch.cuda.synchronize(device=cache.device)
            first_replay = graph.output.detach().cpu().clone()
            graph.replay()
            torch.cuda.synchronize(device=cache.device)
            second_replay = graph.output.detach().cpu().clone()
            eager_graph_comparison = compare_tensors_untimed(
                first_replay,
                eager_reference,
                atol=0.02,
                rtol=0.02,
            )
            graph_evidence = {
                **graph.to_dict(),
                "consecutive_replay_outputs_exact": bool(
                    torch.equal(first_replay, second_replay)
                ),
                "first_replay_checksum": tensor_sha256_untimed(first_replay),
                "second_replay_checksum": tensor_sha256_untimed(second_replay),
            }
            destination[0].copy_(backup[0])
            destination[1].copy_(backup[1])
    else:
        cache.prepare_growing(starting_context, expected_steps)
        operations: list[Callable[[], Any]] = []
        for step in range(expected_steps):

            def growing_step(step: int = step) -> Any:
                cache.select_growing_step(step)
                output = endpoint.decode(tokens[step], positions[step], rope[step])
                cache.finish_growing_step()
                return output

            operations.append(growing_step)
        growing_operations = tuple(operations)
        for operation in growing_operations:
            operation()
        reset_growing()
    prefix_sha256 = _cache_pair_sha256(
        cache,
        start=0,
        length=starting_context,
    )
    return Phase3EndpointSession(
        loaded=loaded,
        operation_keys=keys,
        prefix_input_ids=prefix_input_ids,
        decode_input_ids=decode_input_ids,
        endpoint=endpoint,
        cache=cache,
        method=method,
        adapter_config_fingerprint=adapter_config_fingerprint,
        model_memory=model_memory,
        cache_memory=cache_memory,
        fixed_operation=fixed_operation,
        graph=graph,
        graph_evidence=graph_evidence,
        eager_graph_comparison=eager_graph_comparison,
        growing_operations=growing_operations,
        reset_growing=(
            reset_growing
            if first.runner_kind is RunnerKind.GROWING_CONTEXT
            else None
        ),
        fixed_destination_backup=backup,
        prefix_sha256=prefix_sha256,
    )
