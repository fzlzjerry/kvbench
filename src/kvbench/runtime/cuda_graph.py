"""Fixed-shape CUDA Graph capture and replay with no eager fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
from typing import Any

from kvbench.adapters import KVCacheMethod


_TORCH: Any | None = None


class GraphCaptureError(RuntimeError):
    """The exact fixed-L endpoint could not be captured."""


class GraphReplayError(RuntimeError):
    """The captured fixed-L graph could not be replayed."""


@dataclass(slots=True)
class CapturedFixedGraph:
    """A captured graph and its pointer-stable output tensor."""

    graph: Any
    output: Any
    output_data_ptr: int
    capture_stream_id: int

    def replay(self) -> Any:
        try:
            self.graph.replay()
        except RuntimeError as error:
            raise GraphReplayError("CUDA Graph replay failed") from error
        return self.output

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "captured": True,
            "output_data_ptr": self.output_data_ptr,
            "capture_stream_id": self.capture_stream_id,
            "fallback": False,
        }


@dataclass(frozen=True, slots=True)
class FullModelGraphValidation:
    """Untimed fixed-L full-model graph correctness/allocation evidence."""

    passed: bool
    prefix_length: int
    graph: dict[str, int | bool]
    eager_replay_comparison: dict[str, bool | float]
    replay_outputs_exact: bool
    replay_copies_independent: bool
    eager_checksum: str
    first_replay_checksum: str
    second_replay_checksum: str
    cache_pointers_stable: bool
    historical_cache_unchanged: bool
    replay_allocation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "prefix_length": self.prefix_length,
            "graph": self.graph,
            "eager_replay_comparison": self.eager_replay_comparison,
            "replay_outputs_exact": self.replay_outputs_exact,
            "replay_copies_independent": self.replay_copies_independent,
            "eager_checksum": self.eager_checksum,
            "first_replay_checksum": self.first_replay_checksum,
            "second_replay_checksum": self.second_replay_checksum,
            "cache_pointers_stable": self.cache_pointers_stable,
            "historical_cache_unchanged": self.historical_cache_unchanged,
            "replay_allocation": self.replay_allocation,
            "timing_collected": False,
            "performance_claim_eligible": False,
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise GraphCaptureError("PyTorch is unavailable") from error
    return _TORCH


def capture_fixed_graph(
    operation: Callable[[], Any],
    *,
    warmup_steps: int = 3,
    device: Any | None = None,
) -> CapturedFixedGraph:
    """Optionally warm, capture once, and retain static output storage."""

    torch = _torch()
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
    ):
        raise ValueError("warmup_steps must be a nonnegative integer")
    selected = torch.device(
        f"cuda:{torch.cuda.current_device()}" if device is None else device
    )
    if selected.type != "cuda":
        raise GraphCaptureError("CUDA Graph capture requires a CUDA device")
    current = torch.cuda.current_stream(device=selected)
    side = torch.cuda.Stream(device=selected)
    side.wait_stream(current)
    output: Any = None
    try:
        with torch.cuda.stream(side):
            for _ in range(warmup_steps):
                output = operation()
        current.wait_stream(side)
        torch.cuda.synchronize(device=selected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side, capture_error_mode="global"):
            output = operation()
        torch.cuda.synchronize(device=selected)
    except RuntimeError as error:
        raise GraphCaptureError("fixed-L CUDA Graph capture failed") from error
    if output is None:
        raise GraphCaptureError("capture produced no static output")
    return CapturedFixedGraph(
        graph=graph,
        output=output,
        output_data_ptr=int(output.data_ptr()),
        capture_stream_id=int(side.cuda_stream),
    )


def validate_full_model_fixed_graph(
    model: Any,
    prefix_input_ids: Any,
    current_input_ids: Any,
    *,
    method: KVCacheMethod | None = None,
    atol: float = 0.02,
    rtol: float = 0.02,
) -> FullModelGraphValidation:
    """Capture and validate one short full-model fixed-L shape without timing."""

    torch = _torch()
    if prefix_input_ids.ndim != 2 or current_input_ids.ndim != 2:
        raise ValueError("full-model graph inputs must be rank two")
    batch = int(prefix_input_ids.shape[0])
    prefix_length = int(prefix_input_ids.shape[1])
    if prefix_length <= 0 or tuple(current_input_ids.shape) != (batch, 1):
        raise ValueError("full-model graph requires [B,L] prefix and [B,1] token")
    if prefix_input_ids.device != current_input_ids.device:
        raise ValueError("full-model graph input devices differ")

    from kvbench.adapters import (
        build_method_adapter,
        declared_bf16_runtime_context,
    )
    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
    from kvbench.runtime.numerical import (
        cache_state_history_sha256_untimed,
        compare_tensors_untimed,
        tensor_sha256_untimed,
    )

    config = model.config
    num_layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // query_heads))
    workspace_bytes = (
        num_layers * batch * (query_heads + kv_heads) * (head_dim // 2) * 2
    )
    adapter = method
    if adapter is None:
        adapter = build_method_adapter(
            "bf16",
            declared_bf16_runtime_context(model),
        )
    cache = adapter.allocate(
        batch_size=batch,
        capacity=prefix_length + 1,
        device=prefix_input_ids.device,
        workspace_bytes=workspace_bytes,
    )
    endpoint = BF16DecodeEndpoint(model, cache, adapter)
    with torch.inference_mode(), forced_flash_execution():
        endpoint.prefill(prefix_input_ids)
        cache.prepare_fixed(prefix_length)
        position = torch.tensor(
            [prefix_length],
            dtype=torch.long,
            device=prefix_input_ids.device,
        )
        embeddings = endpoint.prepare_position_embeddings(position.unsqueeze(0))

        def operation() -> Any:
            return endpoint.decode(current_input_ids, position, embeddings)

        pointers_before = cache.pointers()
        history_before = cache_state_history_sha256_untimed(
            cache,
            historical_length=prefix_length,
        )
        eager_output = operation()
        torch.cuda.synchronize(device=prefix_input_ids.device)
        eager_copy = eager_output.detach().cpu().clone()
        captured = capture_fixed_graph(
            operation,
            warmup_steps=3,
            device=prefix_input_ids.device,
        )
        captured.replay()
        torch.cuda.synchronize(device=prefix_input_ids.device)
        first_copy = captured.output.detach().cpu().clone()
        captured.replay()
        torch.cuda.synchronize(device=prefix_input_ids.device)
        second_copy = captured.output.detach().cpu().clone()
        allocation = audit_cuda_allocations(
            captured.replay,
            device=prefix_input_ids.device,
        )
        pointers_after = cache.pointers()
        history_after = cache_state_history_sha256_untimed(
            cache,
            historical_length=prefix_length,
        )

    comparison = compare_tensors_untimed(
        first_copy,
        eager_copy,
        atol=atol,
        rtol=rtol,
    )
    replay_exact = bool(torch.equal(first_copy, second_copy))
    copies_independent = int(first_copy.data_ptr()) != int(second_copy.data_ptr())
    pointer_stable = pointers_before == pointers_after
    history_unchanged = history_before == history_after
    passed = (
        comparison.passed
        and replay_exact
        and copies_independent
        and pointer_stable
        and history_unchanged
        and allocation.passed
    )
    return FullModelGraphValidation(
        passed=passed,
        prefix_length=prefix_length,
        graph=captured.to_dict(),
        eager_replay_comparison=comparison.to_dict(),
        replay_outputs_exact=replay_exact,
        replay_copies_independent=copies_independent,
        eager_checksum=tensor_sha256_untimed(eager_copy),
        first_replay_checksum=tensor_sha256_untimed(first_copy),
        second_replay_checksum=tensor_sha256_untimed(second_copy),
        cache_pointers_stable=pointer_stable,
        historical_cache_unchanged=history_unchanged,
        replay_allocation=allocation.to_dict(),
    )
