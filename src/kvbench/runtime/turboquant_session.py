"""Structural runner seam plus one concrete Phase 6 TurboQuant session."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kvbench.adapters import MethodRuntimeContext, build_method_adapter
from kvbench.adapters.turboquant import TurboQuantMethodAdapter
from kvbench.runtime.allocation import MemorySnapshot, capture_cuda_memory_snapshot
from kvbench.runtime.backend import BACKEND_IDENTITY
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
from kvbench.runtime.turboquant_cache import (
    TURBOQUANT_MANDATORY_CONFIGS,
    TurboQuantStaticCache,
)
from kvbench.schema import (
    GraphMode,
    MeasurementScope,
    RunnerKind,
    canonical_json_bytes,
    sha256_hex,
)


PHASE6_ENDPOINT_WARMUP_OPERATIONS = 3
_LAYERS = 32
_QUERY_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = 128
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class EndpointSessionError(RuntimeError):
    """An endpoint session is invalid or has not passed admission."""


@runtime_checkable
class EndpointOperationKey(Protocol):
    runner_kind: RunnerKind
    graph_mode: GraphMode
    historical_context: int
    attended_context: int
    batch_size: int


@runtime_checkable
class MeasurementEndpointSession(Protocol):
    operation_keys: tuple[EndpointOperationKey, ...]
    adapter_config_fingerprint: str
    model_memory: Any
    cache_memory: Any
    graph_evidence: Mapping[str, object] | None
    eager_graph_comparison: Any | None

    @property
    def state(self) -> str: ...

    @property
    def historical_prefix_sha256(self) -> str: ...

    @property
    def cache_device(self) -> Any: ...

    @property
    def active_context(self) -> int: ...

    def current_cache_pointers(self) -> dict[str, int]: ...

    def current_historical_prefix_sha256(self) -> str: ...

    def method_cache_accounting(self) -> dict[str, int]: ...

    def method_byte_breakdown(self) -> dict[str, int]: ...

    def cache_layout_fingerprint(self) -> str: ...

    def gqa_cache_geometry(self) -> dict[str, Any]: ...

    def audit_output(self, step: int) -> tuple[str, bool]: ...

    def fixed_measurement_callable(self) -> Callable[[], Any]: ...

    def growing_measurement_callables(
        self,
    ) -> tuple[Callable[[], Any], ...]: ...

    def mark_measured(self) -> None: ...


def require_endpoint_session(value: object) -> MeasurementEndpointSession:
    """Fail closed unless the existing runner contract is fully present."""

    if not isinstance(value, MeasurementEndpointSession):
        raise EndpointSessionError(
            "runner requires a structurally complete endpoint session"
        )
    if not value.operation_keys:
        raise EndpointSessionError("endpoint session has no operation keys")
    return value


def session_measurement_scope(
    session: MeasurementEndpointSession,
) -> MeasurementScope:
    """Retain native Phase 3 scope; require an exact enum for Phase 6."""

    value = getattr(
        session,
        "measurement_scope",
        MeasurementScope.NATIVE_HOST_ADMISSION,
    )
    try:
        return MeasurementScope(value)
    except (TypeError, ValueError) as error:
        raise EndpointSessionError(
            "endpoint session measurement scope is invalid"
        ) from error


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise EndpointSessionError("PyTorch is unavailable") from error


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise EndpointSessionError(f"{label} is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class TurboQuantOperationKey:
    """One exact operation consumed by the unchanged common runners."""

    configuration: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    historical_context: int
    attended_context: int
    batch_size: int
    capacity: int
    decode_step: int
    operation_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if self.configuration not in TURBOQUANT_MANDATORY_CONFIGS:
            raise ValueError("operation configuration is not mandatory")
        if (
            self.batch_size != 1
            or type(self.historical_context) is not int
            or self.historical_context <= 0
            or self.attended_context != self.historical_context + 1
            or type(self.capacity) is not int
            or self.capacity < self.attended_context
            or type(self.decode_step) is not int
            or self.decode_step < 0
        ):
            raise ValueError("Phase 6 operation geometry is invalid")
        if self.runner_kind is RunnerKind.FIXED_L:
            if self.decode_step != 0:
                raise ValueError("fixed-L operation has one scratch step")
        elif (
            self.runner_kind is not RunnerKind.GROWING_CONTEXT
            or self.graph_mode is not GraphMode.EAGER
        ):
            raise ValueError("growing Phase 6 operation must remain eager")
        _require_sha256(
            self.operation_fingerprint_sha256,
            "operation fingerprint",
        )

    @classmethod
    def create(
        cls,
        *,
        configuration: str,
        runner_kind: RunnerKind,
        graph_mode: GraphMode,
        starting_context: int,
        capacity: int,
        decode_step: int,
    ) -> "TurboQuantOperationKey":
        historical = starting_context + (
            decode_step
            if runner_kind is RunnerKind.GROWING_CONTEXT
            else 0
        )
        payload = {
            "schema_version": (
                "kvbench-phase6-turboquant-operation-key-1.0.0"
            ),
            "configuration": configuration,
            "runner_kind": runner_kind.value,
            "graph_mode": graph_mode.value,
            "historical_context": historical,
            "attended_context": historical + 1,
            "batch_size": 1,
            "capacity": capacity,
            "decode_step": decode_step,
        }
        return cls(
            configuration=configuration,
            runner_kind=runner_kind,
            graph_mode=graph_mode,
            historical_context=historical,
            attended_context=historical + 1,
            batch_size=1,
            capacity=capacity,
            decode_step=decode_step,
            operation_fingerprint_sha256=sha256_hex(
                canonical_json_bytes(payload)
            ),
        )


def phase6_backend_fingerprint() -> str:
    """Bind unchanged Flash and the three carried TurboQuant kernel files."""

    source_root = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "vllm_turboquant"
    )
    payload = {
        "schema_version": "kvbench-phase6-backend-fingerprint-1.0.0",
        "flash_backend": BACKEND_IDENTITY,
        "turboquant_sources": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in (
                "triton_turboquant_store.py",
                "triton_turboquant_decode.py",
                "triton_decode_attention.py",
            )
        },
    }
    return sha256_hex(canonical_json_bytes(payload))


def turboquant_runtime_context() -> MethodRuntimeContext:
    """Return the one frozen full-model context used by every Phase 6 path."""

    return MethodRuntimeContext(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        backend_id="pytorch_flash_turboquant",
        backend_fingerprint=phase6_backend_fingerprint(),
        num_layers=_LAYERS,
        num_query_heads=_QUERY_HEADS,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
    )


def build_turboquant_operation_keys(
    *,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    starting_context: int,
    output_steps: int,
) -> tuple[TurboQuantOperationKey, ...]:
    """Build exactly one bounded fixed point or the four-step growing point."""

    if (
        configuration not in TURBOQUANT_MANDATORY_CONFIGS
        or type(starting_context) is not int
        or starting_context <= 0
        or type(output_steps) is not int
        or output_steps <= 0
    ):
        raise ValueError("Phase 6 operation set is invalid")
    if runner_kind is RunnerKind.FIXED_L and output_steps == 1:
        capacity = starting_context + 1
        count = 1
    elif (
        runner_kind is RunnerKind.GROWING_CONTEXT
        and graph_mode is GraphMode.EAGER
        and output_steps == 4
    ):
        capacity = starting_context + output_steps
        count = output_steps
    else:
        raise ValueError("operation set is outside the bounded Phase 6 form")
    return tuple(
        TurboQuantOperationKey.create(
            configuration=configuration,
            runner_kind=runner_kind,
            graph_mode=graph_mode,
            starting_context=starting_context,
            capacity=capacity,
            decode_step=step,
        )
        for step in range(count)
    )


class TurboQuantEndpointSession:
    """One warmed full-model TurboQuant session shared by audit and runners."""

    measurement_scope = MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION

    def __init__(
        self,
        *,
        loaded: LoadedFrozenModel,
        operation_keys: tuple[TurboQuantOperationKey, ...],
        endpoint: BF16DecodeEndpoint,
        cache: TurboQuantStaticCache,
        method: TurboQuantMethodAdapter,
        adapter_config_fingerprint: str,
        model_memory: MemorySnapshot,
        cache_memory: MemorySnapshot,
        fixed_operation: Callable[[], Any] | None,
        graph: CapturedFixedGraph | None,
        graph_evidence: Mapping[str, object] | None,
        eager_graph_comparison: NumericalComparison | None,
        growing_operations: tuple[Callable[[], Any], ...],
        reset_growing: Callable[[], None] | None,
        warmed_outputs: tuple[tuple[str, bool], ...],
        prefix_sha256: str,
    ) -> None:
        self.loaded = loaded
        self.operation_keys = operation_keys
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
        self._fixed_operation = fixed_operation
        self._graph = graph
        self._growing_operations = growing_operations
        self._reset_growing = reset_growing
        self._warmed_outputs = warmed_outputs
        self._audit_outputs: dict[int, tuple[str, bool]] = {}
        self._prefix_sha256 = _require_sha256(prefix_sha256, "prefix")
        self._pointers = cache.pointers()
        self._receipt_sha256 = _require_sha256(
            loaded.receipt.receipt_sha256,
            "loaded-model receipt",
        )
        self._state = "warmed"

    @property
    def state(self) -> str:
        return self._state

    @property
    def historical_prefix_sha256(self) -> str:
        return self._prefix_sha256

    @property
    def cache_device(self) -> Any:
        return self.cache.device

    @property
    def active_context(self) -> int:
        return int(self.cache.active_context)

    @property
    def graph(self) -> CapturedFixedGraph | None:
        return self._graph

    def current_cache_pointers(self) -> dict[str, int]:
        return self.cache.pointers()

    def current_historical_prefix_sha256(self) -> str:
        return self.cache.history_sha256(
            self.operation_keys[0].historical_context
        )

    def method_cache_accounting(self) -> dict[str, int]:
        accounting = self.cache.accounting().to_dict()
        if self.method.allocated_bytes(self.cache) != accounting["allocated_bytes"]:
            raise EndpointSessionError("method and cache bytes differ")
        return accounting

    def method_byte_breakdown(self) -> dict[str, int]:
        return dict(sorted(self.method.byte_breakdown(self.cache).items()))

    def cache_layout_fingerprint(self) -> str:
        return self.cache.layout_fingerprint()

    def gqa_cache_geometry(self) -> dict[str, Any]:
        return self.cache.gqa_geometry()

    def audit_output(self, step: int) -> tuple[str, bool]:
        try:
            return self._audit_outputs[step]
        except KeyError as error:
            raise EndpointSessionError("audit output is unavailable") from error

    def _operation(self, step: int) -> Callable[[], Any]:
        if type(step) is not int or step < 0 or step >= len(self.operation_keys):
            raise EndpointSessionError("session step is invalid")
        first = self.operation_keys[0]
        if first.runner_kind is RunnerKind.FIXED_L:
            if step != 0 or self._fixed_operation is None:
                raise EndpointSessionError("fixed operation is unavailable")
            if first.graph_mode is GraphMode.CUDA_GRAPH:
                if self._graph is None:
                    raise EndpointSessionError("captured graph is unavailable")
                return self._graph.replay
            return self._fixed_operation
        return self._growing_operations[step]

    def prepare_audit_step(self, step: int) -> None:
        """Restore the exact state immediately before one audit operation."""

        if self._state != "warmed":
            raise EndpointSessionError("session is not accepting audits")
        first = self.operation_keys[0]
        if first.runner_kind is RunnerKind.FIXED_L:
            if step != 0:
                raise EndpointSessionError("fixed session has one audit step")
            return
        if self._reset_growing is None:
            raise EndpointSessionError("growing reset is unavailable")
        self._reset_growing()
        for prior in range(step):
            self._growing_operations[prior]()

    def execute_audit_step(self, step: int) -> Any:
        """Execute one already-warmed operation outside normal timing."""

        return self._operation(step)()

    def admit(
        self,
        *,
        observed_outputs: Sequence[tuple[str, bool]],
        execution_path_passed: bool,
        allocation_passed: bool,
        graph_passed: bool,
    ) -> None:
        """Admit only after the externally recorded common audits pass."""

        verdicts = (
            execution_path_passed,
            allocation_passed,
            graph_passed,
        )
        observed = tuple(observed_outputs)
        if (
            self._state != "warmed"
            or any(type(value) is not bool for value in verdicts)
            or len(observed) != len(self.operation_keys)
        ):
            raise EndpointSessionError("session admission evidence is invalid")
        for index, (digest, finite) in enumerate(observed):
            _require_sha256(digest, "audit output")
            if (
                type(finite) is not bool
                or (digest, finite) != self._warmed_outputs[index]
            ):
                raise EndpointSessionError("post-warmup output is unstable")
        if (
            not all(verdicts)
            or not all(finite for _, finite in observed)
            or (
                self.eager_graph_comparison is not None
                and not self.eager_graph_comparison.passed
            )
        ):
            self._state = "failed"
            raise EndpointSessionError("session audits did not pass")
        validate_loaded_frozen_model_receipt(self.loaded)
        if (
            self.loaded.receipt.receipt_sha256 != self._receipt_sha256
            or self.cache.pointers() != self._pointers
        ):
            raise EndpointSessionError("session identity changed during audit")
        first = self.operation_keys[0]
        if first.runner_kind is RunnerKind.GROWING_CONTEXT:
            if self._reset_growing is None:
                raise EndpointSessionError("growing reset is unavailable")
            self._reset_growing()
        if (
            self.cache.history_sha256(first.historical_context)
            != self._prefix_sha256
        ):
            raise EndpointSessionError("historical prefix changed during audit")
        self._audit_outputs = dict(enumerate(observed))
        self._state = "ready"

    def fixed_measurement_callable(self) -> Callable[[], Any]:
        if (
            self._state != "ready"
            or self.operation_keys[0].runner_kind is not RunnerKind.FIXED_L
        ):
            raise EndpointSessionError("fixed timing is not admitted")
        self._state = "measuring"
        return self._operation(0)

    def growing_measurement_callables(
        self,
    ) -> tuple[Callable[[], Any], ...]:
        if (
            self._state != "ready"
            or self.operation_keys[0].runner_kind
            is not RunnerKind.GROWING_CONTEXT
        ):
            raise EndpointSessionError("growing timing is not admitted")
        self._state = "measuring"
        return self._growing_operations

    def mark_measured(self) -> None:
        if self._state != "measuring":
            raise EndpointSessionError("measurement did not start")
        self._state = "measured"


def build_turboquant_endpoint_session(
    *,
    loaded: LoadedFrozenModel,
    operation_keys: tuple[TurboQuantOperationKey, ...],
    prefix_input_ids: Any,
    decode_input_ids: Any,
) -> TurboQuantEndpointSession:
    """Allocate, prefill, compile, warm, and retain one Phase 6 session."""

    validate_loaded_frozen_model_receipt(loaded)
    if not operation_keys:
        raise EndpointSessionError("operation set is empty")
    first = operation_keys[0]
    expected_steps = (
        1
        if first.runner_kind is RunnerKind.FIXED_L
        else len(operation_keys)
    )
    if operation_keys != build_turboquant_operation_keys(
        configuration=first.configuration,
        runner_kind=first.runner_kind,
        graph_mode=first.graph_mode,
        starting_context=first.historical_context,
        output_steps=expected_steps,
    ):
        raise EndpointSessionError("operation set differs from bounded form")
    if (
        tuple(int(item) for item in prefix_input_ids.shape)
        != (1, first.historical_context)
        or tuple(int(item) for item in decode_input_ids.shape)
        != (1, expected_steps)
        or prefix_input_ids.device != decode_input_ids.device
        or prefix_input_ids.device.type != "cuda"
    ):
        raise EndpointSessionError("endpoint inputs differ from operation set")

    method = build_method_adapter(
        first.configuration,
        turboquant_runtime_context(),
    )
    if type(method) is not TurboQuantMethodAdapter:
        raise EndpointSessionError("factory did not return TurboQuant adapter")
    model_memory = capture_cuda_memory_snapshot(
        "model_baseline",
        device=prefix_input_ids.device,
    )
    cache = method.allocate(
        batch_size=1,
        capacity=first.capacity,
        device=prefix_input_ids.device,
        workspace_bytes=0,
    )
    if type(cache) is not TurboQuantStaticCache:
        raise EndpointSessionError("cache is not TurboQuant static state")
    cache_memory = capture_cuda_memory_snapshot(
        "post_cache_allocation",
        device=prefix_input_ids.device,
    )
    cache.initialize_deterministic()
    endpoint = BF16DecodeEndpoint(loaded.model, cache, method)
    adapter_fingerprint = method.config_fingerprint(
        cache.layout_fingerprint()
    )
    torch = _torch()
    positions = tuple(
        torch.tensor(
            [first.historical_context + step],
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
        cache.prepare_growing(first.historical_context, expected_steps)

    fixed_operation: Callable[[], Any] | None = None
    growing_operations: tuple[Callable[[], Any], ...] = ()
    graph: CapturedFixedGraph | None = None
    graph_evidence: dict[str, object] | None = None
    graph_comparison: NumericalComparison | None = None
    warmed_outputs: list[tuple[str, bool]] = []
    endpoint.prefill(prefix_input_ids)
    if first.runner_kind is RunnerKind.FIXED_L:
        cache.prepare_fixed(first.historical_context)

        def fixed_step() -> Any:
            return endpoint.decode(tokens[0], positions[0], rope[0])

        fixed_operation = fixed_step
        eager_output: Any | None = None
        for _ in range(PHASE6_ENDPOINT_WARMUP_OPERATIONS):
            eager_output = fixed_step()
        if eager_output is None:
            raise EndpointSessionError("fixed warmup produced no output")
        observed = eager_output.detach().cpu()
        if first.graph_mode is GraphMode.CUDA_GRAPH:
            graph = capture_fixed_graph(
                fixed_step,
                warmup_steps=0,
                device=cache.device,
            )
            first_replay = graph.replay().detach().cpu().clone()
            second_replay = graph.replay().detach().cpu().clone()
            torch.cuda.synchronize(device=cache.device)
            graph_comparison = compare_tensors_untimed(
                first_replay,
                observed,
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
            observed = second_replay
        warmed_outputs.append(
            (
                tensor_sha256_untimed(observed),
                bool(torch.isfinite(observed).all().item()),
            )
        )
    else:
        cache.prepare_growing(first.historical_context, expected_steps)
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
            observed = operation().detach().cpu()
            warmed_outputs.append(
                (
                    tensor_sha256_untimed(observed),
                    bool(torch.isfinite(observed).all().item()),
                )
            )
        reset_growing()
    prefix_sha256 = cache.history_sha256(first.historical_context)
    return TurboQuantEndpointSession(
        loaded=loaded,
        operation_keys=operation_keys,
        endpoint=endpoint,
        cache=cache,
        method=method,
        adapter_config_fingerprint=adapter_fingerprint,
        model_memory=model_memory,
        cache_memory=cache_memory,
        fixed_operation=fixed_operation,
        graph=graph,
        graph_evidence=graph_evidence,
        eager_graph_comparison=graph_comparison,
        growing_operations=growing_operations,
        reset_growing=(
            reset_growing
            if first.runner_kind is RunnerKind.GROWING_CONTEXT
            else None
        ),
        warmed_outputs=tuple(warmed_outputs),
        prefix_sha256=prefix_sha256,
    )
