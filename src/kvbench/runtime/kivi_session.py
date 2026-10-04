"""Narrow common-runner session bridge for Phase 8 KIVI admission."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import struct
from typing import Any

from kvbench.adapters import MethodRuntimeContext, build_method_adapter
from kvbench.adapters.kivi import (
    KIVI_DECISION_0018_PATCH_SHA256,
    KIVI_EXTENSION_SHA256,
    KIVI_FIXTURE_ROOT_SHA256,
    KIVI_NEW_PACK_SHA256,
    KIVI_OFFICIAL_BASE_TREE,
    KIVI_OFFICIAL_COMMIT,
    KIVI_PATCHED_TREE,
    KIVIMethodAdapter,
)
from kvbench.config import load_config
from kvbench.runtime.allocation import capture_cuda_memory_snapshot
from kvbench.runtime.backend import BACKEND_IDENTITY
from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
from kvbench.runtime.cuda_graph import CapturedFixedGraph, capture_fixed_graph
from kvbench.runtime.kivi_cache import (
    KIVI_CONFIG_BITS,
    KIVI_HELD_OUT_CONFIG,
    KIVIStaticCache,
)
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
from kvbench.runtime.turboquant_session import (
    EndpointSessionError,
    TurboQuantEndpointSession,
)
from kvbench.schema import (
    GraphMode,
    MeasurementScope,
    MethodConfig,
    MethodName,
    RunnerKind,
    canonical_json_bytes,
    sha256_hex,
)


PHASE8_ENDPOINT_WARMUP_OPERATIONS = 3
PHASE8_DECODE_ATOL = 0.02
PHASE8_DECODE_RTOL = 0.02
_LAYERS = 32
_QUERY_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = 128
_METHOD_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "methods" / "kivi.yaml"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FIXED_GRID = frozenset(
    {
        ("k4v4", GraphMode.EAGER, 128),
        ("k4v4", GraphMode.CUDA_GRAPH, 128),
        ("k2v4", GraphMode.EAGER, 128),
        ("k2v4", GraphMode.CUDA_GRAPH, 128),
        ("k2v2", GraphMode.EAGER, 128),
        ("k2v2", GraphMode.CUDA_GRAPH, 128),
        ("k4v4", GraphMode.EAGER, 4096),
        ("k4v4", GraphMode.CUDA_GRAPH, 4096),
        (KIVI_HELD_OUT_CONFIG, GraphMode.EAGER, 128),
    }
)
_GROWING_GRID = (
    "k4v4",
    RunnerKind.GROWING_CONTEXT,
    GraphMode.EAGER,
    31,
    4,
)


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
class KIVIOperationKey:
    """One operation from only the preregistered Phase 8 admission grid."""

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
        if self.configuration not in KIVI_CONFIG_BITS:
            raise ValueError("operation configuration is not frozen")
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
            raise ValueError("Phase 8 operation geometry is invalid")
        if self.runner_kind is RunnerKind.FIXED_L:
            if (
                self.decode_step != 0
                or self.capacity != self.historical_context + 1
                or (
                    self.configuration,
                    self.graph_mode,
                    self.historical_context,
                )
                not in _FIXED_GRID
            ):
                raise ValueError("fixed operation is outside the Phase 8 grid")
        elif (
            self.runner_kind is not RunnerKind.GROWING_CONTEXT
            or self.configuration != _GROWING_GRID[0]
            or self.graph_mode is not _GROWING_GRID[2]
            or self.capacity != 35
            or self.decode_step >= 4
            or self.historical_context != 31 + self.decode_step
        ):
            raise ValueError("growing operation is outside the Phase 8 grid")
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
    ) -> "KIVIOperationKey":
        historical = starting_context + (
            decode_step
            if runner_kind is RunnerKind.GROWING_CONTEXT
            else 0
        )
        payload = {
            "schema_version": "kvbench-phase8-kivi-operation-key-1.0.0",
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


def build_kivi_operation_keys(
    *,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    starting_context: int,
    output_steps: int,
) -> tuple[KIVIOperationKey, ...]:
    """Build exactly one of the ten bounded Phase 8 admission points."""

    if (
        configuration not in KIVI_CONFIG_BITS
        or not isinstance(runner_kind, RunnerKind)
        or not isinstance(graph_mode, GraphMode)
        or type(starting_context) is not int
        or type(output_steps) is not int
    ):
        raise ValueError("Phase 8 operation set is invalid")
    if (
        runner_kind is RunnerKind.FIXED_L
        and output_steps == 1
        and (configuration, graph_mode, starting_context) in _FIXED_GRID
    ):
        capacity = starting_context + 1
        count = 1
    elif (
        (
            configuration,
            runner_kind,
            graph_mode,
            starting_context,
            output_steps,
        )
        == _GROWING_GRID
    ):
        capacity = starting_context + output_steps
        count = output_steps
    else:
        raise ValueError("operation set is outside the bounded Phase 8 grid")
    return tuple(
        KIVIOperationKey.create(
            configuration=configuration,
            runner_kind=runner_kind,
            graph_mode=graph_mode,
            starting_context=starting_context,
            capacity=capacity,
            decode_step=step,
        )
        for step in range(count)
    )


def phase8_kivi_backend_fingerprint() -> str:
    """Bind the common endpoint, patched authority, and exact CUDA ABI."""

    source_root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": "kvbench-phase8-kivi-backend-fingerprint-1.0.0",
        "prefill_backend": BACKEND_IDENTITY,
        "decode_backend": {
            "implementation": "patched_official_kivi_direct_compressed_decode",
            "official_commit": KIVI_OFFICIAL_COMMIT,
            "official_base_tree": KIVI_OFFICIAL_BASE_TREE,
            "patched_tree": KIVI_PATCHED_TREE,
            "decision_0018_patch_sha256": KIVI_DECISION_0018_PATCH_SHA256,
            "extension_sha256": KIVI_EXTENSION_SHA256,
            "new_pack_sha256": KIVI_NEW_PACK_SHA256,
            "fixture_root_sha256": KIVI_FIXTURE_ROOT_SHA256,
            "cuda_abi": "float16",
            "model_boundary": "bfloat16_to_float16_to_bfloat16",
            "kernel_families": [
                "bgemv2_kernel_outer_dim",
                "bgemv4_kernel_outer_dim",
            ],
        },
        "local_sources": {
            relative: hashlib.sha256(
                (source_root / relative).read_bytes()
            ).hexdigest()
            for relative in (
                "adapters/kivi.py",
                "runtime/bf16_endpoint.py",
                "runtime/kivi_cache.py",
            )
        },
    }
    return sha256_hex(canonical_json_bytes(payload))


def kivi_runtime_context() -> MethodRuntimeContext:
    """Return the frozen full-model context shared by all four KIVI variants."""

    return MethodRuntimeContext(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        backend_id="pytorch_flash_patched_official_kivi",
        backend_fingerprint=phase8_kivi_backend_fingerprint(),
        num_layers=_LAYERS,
        num_query_heads=_QUERY_HEADS,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
    )


def load_frozen_kivi_method_config() -> MethodConfig:
    """Load the exact preregistered KIVI configuration authority."""

    document = load_config(_METHOD_CONFIG_PATH)
    if (
        type(document) is not MethodConfig
        or document.method is not MethodName.KIVI
        or document.method_config_id != "kivi"
        or document.source_lock_id != "kivi"
        or document.source_revision != KIVI_OFFICIAL_COMMIT
        or tuple(variant.variant_id for variant in document.variants)
        != tuple(KIVI_CONFIG_BITS)
    ):
        raise EndpointSessionError(
            "canonical KIVI method configuration authority differs"
        )
    return document


def _logical_prefix_sha256(
    cache: KIVIStaticCache,
    historical_length: int,
) -> str:
    """Hash original logical token order without treating rollover as mutation."""

    if (
        type(historical_length) is not int
        or historical_length <= 0
        or historical_length > cache.capacity
    ):
        raise EndpointSessionError("historical prefix length is invalid")
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "schema_version": "kvbench-phase8-kivi-logical-prefix-1.0.0",
                "configuration": cache.config_name,
                "historical_length": historical_length,
                "layers": cache.num_layers,
            }
        )
    )
    for layer in range(cache.num_layers):
        state = cache.token_index_state(layer)
        for label, parts in (
            (
                "key",
                (
                    state["quantized_key_tokens"],
                    state["residual_key_tokens"],
                ),
            ),
            (
                "value",
                (
                    state["quantized_value_tokens"],
                    state["residual_value_tokens"],
                ),
            ),
        ):
            digest.update(label.encode("ascii"))
            digest.update(struct.pack("<II", layer, historical_length))
            remaining = historical_length
            for part in parts:
                take = min(remaining, int(part.numel()))
                for index in range(take):
                    digest.update(struct.pack("<q", int(part[index])))
                remaining -= take
                if remaining == 0:
                    break
            if remaining:
                raise EndpointSessionError(
                    "KIVI logical prefix ledger is incomplete"
                )
    return digest.hexdigest()


def _historical_prefix_sha256(
    cache: KIVIStaticCache,
    operation_key: KIVIOperationKey,
) -> str:
    logical = _logical_prefix_sha256(
        cache,
        operation_key.historical_context,
    )
    if operation_key.runner_kind is RunnerKind.GROWING_CONTEXT:
        return logical
    digest = hashlib.sha256(logical.encode("ascii"))
    for layer in range(cache.num_layers):
        digest.update(cache.physical_history_checksum(layer).encode("ascii"))
    return digest.hexdigest()


class KIVIEndpointSession(TurboQuantEndpointSession):
    """One warmed KIVI endpoint using the unchanged common session contract."""

    measurement_scope = MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION

    def current_historical_prefix_sha256(self) -> str:
        return _historical_prefix_sha256(
            self.cache,
            self.operation_keys[0],
        )

    def method_cache_accounting(self) -> dict[str, int]:
        accounting = self.cache.accounting().to_dict()
        if self.method.allocated_bytes(self.cache) != accounting["allocated_bytes"]:
            raise EndpointSessionError("method and cache bytes differ")
        accounting.update(
            {
                "active_storage_bytes": self.cache.active_storage_bytes(),
                "logical_bf16_active_bytes": (
                    self.cache.active_logical_bf16_bytes()
                ),
                "logical_bf16_allocated_bytes": (
                    self.cache.logical_bf16_storage_bytes
                ),
            }
        )
        return accounting

    def method_allocation_ratios(self) -> dict[str, float]:
        ratios = self.cache.ratios()
        return {
            "rho_alloc": ratios.rho_alloc,
            "r_alloc": ratios.r_alloc,
            "reciprocal_product": ratios.rho_alloc * ratios.r_alloc,
        }

    def admit(
        self,
        *,
        observed_outputs: Sequence[tuple[str, bool]],
        execution_path_passed: bool,
        allocation_passed: bool,
        graph_passed: bool,
    ) -> None:
        """Use the common state machine with KIVI logical-prefix rollover."""

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
        if self.current_historical_prefix_sha256() != self._prefix_sha256:
            raise EndpointSessionError("historical prefix changed during audit")
        self._audit_outputs = dict(enumerate(observed))
        self._state = "ready"


def build_kivi_endpoint_session(
    *,
    loaded: LoadedFrozenModel,
    operation_keys: tuple[KIVIOperationKey, ...],
    prefix_input_ids: Any,
    decode_input_ids: Any,
) -> KIVIEndpointSession:
    """Allocate, prefill, prepare authority, warm, and retain one KIVI session."""

    validate_loaded_frozen_model_receipt(loaded)
    if not operation_keys:
        raise EndpointSessionError("operation set is empty")
    first = operation_keys[0]
    expected_steps = (
        1
        if first.runner_kind is RunnerKind.FIXED_L
        else len(operation_keys)
    )
    if operation_keys != build_kivi_operation_keys(
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

    method_config = load_frozen_kivi_method_config()
    method = build_method_adapter(
        method_config,
        kivi_runtime_context(),
        variant_id=first.configuration,
    )
    if type(method) is not KIVIMethodAdapter:
        raise EndpointSessionError("factory did not return KIVI adapter")
    # Authority is verified exactly once before prefill, warmup, or capture.
    method.prepare_runtime()
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
    if type(cache) is not KIVIStaticCache:
        raise EndpointSessionError("cache is not KIVI static state")
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
        cache.initialize_deterministic()
        endpoint.prefill(prefix_input_ids)
        cache.prepare_growing(first.historical_context, expected_steps)

    fixed_operation: Callable[[], Any] | None = None
    growing_operations: tuple[Callable[[], Any], ...] = ()
    graph: CapturedFixedGraph | None = None
    graph_evidence: Mapping[str, object] | None = None
    graph_comparison: NumericalComparison | None = None
    warmed_outputs: list[tuple[str, bool]] = []
    endpoint.prefill(prefix_input_ids)
    if first.runner_kind is RunnerKind.FIXED_L:
        cache.prepare_fixed(first.historical_context)

        def fixed_step() -> Any:
            return endpoint.decode(tokens[0], positions[0], rope[0])

        fixed_operation = fixed_step
        eager_output: Any | None = None
        for _ in range(PHASE8_ENDPOINT_WARMUP_OPERATIONS):
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
                atol=PHASE8_DECODE_ATOL,
                rtol=PHASE8_DECODE_RTOL,
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
                return endpoint.decode(tokens[step], positions[step], rope[step])

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
    prefix_sha256 = _historical_prefix_sha256(
        cache,
        first,
    )
    return KIVIEndpointSession(
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
