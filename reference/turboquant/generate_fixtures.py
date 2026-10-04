#!/usr/bin/env python3
"""Generate compact fixtures through the pinned official vLLM TurboQuant APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable

from kvbench.errors import ArtifactConflictError
from kvbench.runtime.artifacts import (
    _json_bytes,
    _rename_noreplace,
    _write_exclusive,
    sha256_bytes,
    sha256_file,
)

from reference.turboquant.bootstrap_environment import (
    DEFAULT_SOURCE,
    DEFAULT_VENV,
    REFERENCE_ROOT,
    SOURCE_MANIFEST_PATH,
    ENVIRONMENT_PATH,
    _load_json,
    verify_all,
)
from reference.turboquant.validate_fixtures import validate_reference


DEFAULT_FIXTURE_ROOT = REFERENCE_ROOT / "fixtures"
SEED = 20260724
BATCH_SIZE = 1
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
INITIAL_CONTEXT = 17
APPEND_TOKENS = 1
TOTAL_CONTEXT = INITIAL_CONTEXT + APPEND_TOKENS
BLOCK_SIZE = 16
NUM_BLOCKS = math.ceil(TOTAL_CONTEXT / BLOCK_SIZE)
MAX_NUM_KV_SPLITS = 4
INPUT_DTYPE = "bfloat16"


class FixtureGenerationError(RuntimeError):
    """Raised when official-path fixture generation cannot be certified."""


def _write(root: Path, relative: str, data: bytes) -> Path:
    return _write_exclusive(root, relative, data)


def _write_json(root: Path, relative: str, payload: dict[str, Any]) -> Path:
    return _write(root, relative, _json_bytes(payload))


def _record(path: Path, *, relative: str, dtype: str, shape: list[int]) -> dict[str, Any]:
    return {
        "path": relative,
        "dtype": dtype,
        "shape": shape,
        "nbytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _file_record(path: Path, *, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "nbytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    byte_view = tensor.detach().contiguous().view(-1).view(torch.uint8).cpu()
    return byte_view.numpy().tobytes(order="C")


def _write_tensor(
    root: Path,
    relative: str,
    tensor: Any,
    *,
    dtype: str,
    shape: list[int],
    record_path: str | None = None,
) -> dict[str, Any]:
    path = _write(root, relative, _tensor_bytes(tensor))
    return _record(path, relative=record_path or relative, dtype=dtype, shape=shape)


def _write_ledger(root: Path, ledger_relative: str = "checksums.sha256") -> Path:
    ledger_path = root / ledger_relative
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise FixtureGenerationError(f"symlink forbidden in fixture staging: {path}")
        if not stat.S_ISREG(metadata.st_mode) or path == ledger_path:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}\n")
    if not entries:
        raise FixtureGenerationError(f"cannot create empty checksum ledger: {root}")
    return _write(root, ledger_relative, "".join(entries).encode("utf-8"))


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def publish_staged(staged: Path, target: Path) -> str:
    """Publish once, or verify an identical final set without replacing it."""
    if target.exists() or target.is_symlink():
        if not target.is_dir() or target.is_symlink():
            raise FixtureGenerationError(f"unsafe finalized fixture target: {target}")
        if _tree_digests(staged) != _tree_digests(target):
            raise ArtifactConflictError(
                "finalized fixture set exists and differs; overwrite is forbidden"
            )
        return "verified_existing"
    try:
        _rename_noreplace(staged, target)
    except ArtifactConflictError:
        if target.is_dir() and _tree_digests(staged) == _tree_digests(target):
            return "verified_existing"
        raise
    return "published_new"


def _profile_cuda_kernel_names(call: Callable[[], Any]) -> tuple[Any, list[str]]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        result = call()
        torch.cuda.synchronize()
    names = sorted(
        {
            event.name
            for event in profiler.events()
            if "cuda" in str(event.device_type).lower()
        }
    )
    if not names:
        raise FixtureGenerationError("torch.profiler exposed no CUDA kernel identity")
    return result, names


def _layout(config: Any) -> dict[str, Any]:
    key_data = HEAD_DIM if config.key_fp8 else math.ceil(
        HEAD_DIM * config.key_mse_bits / 8
    )
    key_norm = 0 if config.key_fp8 else 2
    value_data = math.ceil(HEAD_DIM * config.value_quant_bits / 8)
    padding = config.slot_size_aligned - config.slot_size
    breakdown = {
        "packed_keys": key_data,
        "key_norm": key_norm,
        "packed_values": value_data,
        "value_scale": 2,
        "value_zero_point": 2,
        "alignment_padding": padding,
    }
    if sum(breakdown.values()) != config.slot_size_aligned:
        raise FixtureGenerationError("source-derived layout does not sum to slot size")
    value_offset = config.key_packed_size
    metadata_offset = value_offset + value_data
    return {
        "byte_breakdown_per_head_token": breakdown,
        "key_packed_size": config.key_packed_size,
        "value_packed_size": config.value_packed_size,
        "slot_size": config.slot_size,
        "slot_size_aligned": config.slot_size_aligned,
        "page_bytes": BLOCK_SIZE * NUM_KV_HEADS * config.slot_size_aligned,
        "allocated_cache_bytes": (
            NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * config.slot_size_aligned
        ),
        "owned_cache_bytes_after_store": (
            INITIAL_CONTEXT * NUM_KV_HEADS * config.slot_size_aligned
        ),
        "owned_cache_bytes_after_append": (
            TOTAL_CONTEXT * NUM_KV_HEADS * config.slot_size_aligned
        ),
        "storage_shape": [
            NUM_BLOCKS,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            config.slot_size_aligned,
        ],
        "offsets": {
            "packed_key_start": 0,
            "key_norm_start": None if config.key_fp8 else key_data,
            "packed_value_start": value_offset,
            "value_scale_start": metadata_offset,
            "value_zero_point_start": metadata_offset + 2,
            "slot_end": config.slot_size,
            "aligned_slot_end": config.slot_size_aligned,
        },
    }


def _fixture_geometry() -> dict[str, Any]:
    return {
        "batch_size": BATCH_SIZE,
        "num_query_heads": NUM_QUERY_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "initial_context": INITIAL_CONTEXT,
        "append_tokens": APPEND_TOKENS,
        "total_context": TOTAL_CONTEXT,
        "block_size": BLOCK_SIZE,
        "num_blocks": NUM_BLOCKS,
        "max_num_kv_splits": MAX_NUM_KV_SPLITS,
        "seed": SEED,
        "input_dtype": INPUT_DTYPE,
    }


def _cpu_inputs() -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    def sample(shape: tuple[int, ...]) -> Any:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        ).contiguous()

    return {
        "prefill_key": sample((INITIAL_CONTEXT, NUM_KV_HEADS, HEAD_DIM)),
        "prefill_value": sample((INITIAL_CONTEXT, NUM_KV_HEADS, HEAD_DIM)),
        "append_key": sample((APPEND_TOKENS, NUM_KV_HEADS, HEAD_DIM)),
        "append_value": sample((APPEND_TOKENS, NUM_KV_HEADS, HEAD_DIM)),
        "decode_query": sample((BATCH_SIZE, NUM_QUERY_HEADS, HEAD_DIM)),
    }


def _write_inputs(stage: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, tensor in inputs.items():
        relative = f"inputs/{name}.bf16.bin"
        records[name] = _write_tensor(
            stage,
            relative,
            tensor,
            dtype=INPUT_DTYPE,
            shape=list(tensor.shape),
        )
    return records


def _generate_configuration(
    stage: Path,
    config_record: dict[str, Any],
    cpu_inputs: dict[str, Any],
    input_records: dict[str, Any],
    source_manifest: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    import torch
    from vllm.model_executor.layers.quantization.turboquant.centroids import (
        solve_lloyd_max,
    )
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        triton_turboquant_decode_attention,
    )
    from vllm.v1.attention.ops.triton_turboquant_store import (
        triton_turboquant_store,
    )

    cache_dtype = config_record["cache_dtype"]
    config = TurboQuantConfig.from_cache_dtype(cache_dtype, HEAD_DIM)
    if (
        config.key_quant_bits != config_record["key_bits"]
        or config.value_quant_bits != config_record["value_bits"]
        or config.norm_correction != config_record["norm_correction"]
    ):
        raise FixtureGenerationError(f"installed source preset mismatch: {cache_dtype}")
    device = torch.device("cuda:0")
    Pi = _build_hadamard(HEAD_DIM, str(device))
    PiT = Pi.T.contiguous()
    centroids_cpu, midpoints_cpu = solve_lloyd_max(
        HEAD_DIM, config.centroid_bits
    )
    centroids = centroids_cpu.to(device)
    midpoints = midpoints_cpu.to(device)
    inputs = {name: tensor.to(device) for name, tensor in cpu_inputs.items()}
    prefill_slots = torch.arange(INITIAL_CONTEXT, dtype=torch.int32, device=device)
    append_slots = torch.tensor([INITIAL_CONTEXT], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([TOTAL_CONTEXT], dtype=torch.int32, device=device)
    cache_shape = (
        NUM_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        config.slot_size_aligned,
    )

    def store(cache: Any, key: Any, value: Any, slots: Any) -> None:
        triton_turboquant_store(
            key,
            value,
            cache,
            slots,
            PiT,
            midpoints,
            config.key_mse_bits,
            config.key_packed_size,
            config.value_quant_bits,
            config.key_fp8,
        )

    def decode(cache: Any, mid: Any, output: Any, lse: Any) -> Any:
        return triton_turboquant_decode_attention(
            inputs["decode_query"],
            cache,
            block_table,
            seq_lens,
            Pi,
            centroids,
            1.0 / math.sqrt(HEAD_DIM),
            config.key_mse_bits,
            config.key_packed_size,
            config.value_quant_bits,
            config.key_fp8,
            config.norm_correction,
            PiT,
            mid,
            output,
            lse,
            None,
            MAX_NUM_KV_SPLITS,
        )

    # Compile the exact official paths on disposable state before collecting names.
    warm_cache = torch.zeros(cache_shape, dtype=torch.uint8, device=device)
    store(warm_cache, inputs["prefill_key"], inputs["prefill_value"], prefill_slots)
    store(warm_cache, inputs["append_key"], inputs["append_value"], append_slots)
    warm_mid = torch.empty(
        (BATCH_SIZE, NUM_QUERY_HEADS, MAX_NUM_KV_SPLITS, HEAD_DIM + 1),
        dtype=torch.float32,
        device=device,
    )
    warm_output = torch.empty(
        (BATCH_SIZE, NUM_QUERY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    warm_lse = torch.empty(
        (BATCH_SIZE, NUM_QUERY_HEADS), dtype=torch.float32, device=device
    )
    decode(warm_cache, warm_mid, warm_output, warm_lse)
    torch.cuda.synchronize()

    cache = torch.zeros(cache_shape, dtype=torch.uint8, device=device)
    _, store_kernels = _profile_cuda_kernel_names(
        lambda: store(
            cache,
            inputs["prefill_key"],
            inputs["prefill_value"],
            prefill_slots,
        )
    )
    cache_after_store = cache.detach().clone()
    _, append_kernels = _profile_cuda_kernel_names(
        lambda: store(
            cache,
            inputs["append_key"],
            inputs["append_value"],
            append_slots,
        )
    )
    cache_after_append = cache.detach().clone()
    append_slot = cache[1, 1, :, :].detach().clone()
    mid = torch.empty_like(warm_mid)
    output_buffer = torch.empty_like(warm_output)
    lse = torch.empty_like(warm_lse)
    decode_output, decode_kernels = _profile_cuda_kernel_names(
        lambda: decode(cache, mid, output_buffer, lse)
    )
    if not bool(torch.isfinite(decode_output).all().item()):
        raise FixtureGenerationError(f"non-finite decode output: {cache_dtype}")

    expected_store = "_tq_fused_store_fp8" if config.key_fp8 else "_tq_fused_store_mse"
    if expected_store not in store_kernels or expected_store not in append_kernels:
        raise FixtureGenerationError(f"official store kernel absent: {cache_dtype}")
    for expected_decode in ("_tq_decode_stage1", "_fwd_kernel_stage2"):
        if expected_decode not in decode_kernels:
            raise FixtureGenerationError(
                f"official decode kernel absent for {cache_dtype}: {expected_decode}"
            )

    config_root = stage / cache_dtype
    config_root.mkdir(mode=0o700)
    layout = _layout(config)
    outputs = {
        "cache_after_store": _write_tensor(
            config_root,
            "cache_after_store.uint8.bin",
            cache_after_store,
            dtype="uint8",
            shape=list(cache_shape),
        ),
        "cache_after_append": _write_tensor(
            config_root,
            "cache_after_append.uint8.bin",
            cache_after_append,
            dtype="uint8",
            shape=list(cache_shape),
        ),
        "append_slot": _write_tensor(
            config_root,
            "append_slot.uint8.bin",
            append_slot,
            dtype="uint8",
            shape=[NUM_KV_HEADS, config.slot_size_aligned],
        ),
        "decode_output": _write_tensor(
            config_root,
            "decode_output.bf16.bin",
            decode_output,
            dtype="bfloat16",
            shape=[BATCH_SIZE, NUM_QUERY_HEADS, HEAD_DIM],
        ),
    }
    all_names = store_kernels + append_kernels + decode_kernels
    trace = {
        "schema_version": "turboquant-reference-trace-1.0.0",
        "run_kind": "reference_trace",
        "trace_mechanism": "torch.profiler CUDA activities; kernel names retained only",
        "timings_discarded": True,
        "store_kernels": store_kernels,
        "append_kernels": append_kernels,
        "decode_kernels": decode_kernels,
        "full_prefix_dequantization_observed": any(
            "_tq_full_dequant_kv" in name for name in all_names
        ),
        "gqa_materialization_observed": any(
            "repeat_interleave" in name or "repeat_kv" in name for name in all_names
        ),
        "gqa_source_path": "_tq_decode_stage1 maps query head to kv_head by integer group division",
        "backend_fallback": False,
        "performance_claim_eligible": False,
        "trace_limitation": (
            "Kernel-name tracing identifies executed CUDA kernels but is not a proof "
            "of physical memory traffic and retains no timing fields."
        ),
    }
    trace_path = _write_json(config_root, "kernel_trace.json", trace)
    outputs["kernel_trace"] = _file_record(
        trace_path, relative="kernel_trace.json"
    )
    source = source_manifest["source"]
    environment_sha = sha256_file(ENVIRONMENT_PATH)
    manifest = {
        "schema_version": "turboquant-reference-fixture-1.0.0",
        "fixture_id": (
            f"vllm-v0.25.1-sm120-gqa-d128-l17-a1-seed{SEED}-{cache_dtype}"
        ),
        "configuration": config_record,
        "source": {
            "repository": source["repository"],
            "commit": source["commit"],
            "tree": source["tree"],
            "manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        },
        "environment": {
            "manifest_sha256": environment_sha,
            "linux_amd64_container_digest": environment["base_image"][
                "linux_amd64_digest"
            ],
            "runtime": environment["runtime"],
        },
        "geometry": _fixture_geometry(),
        "inputs": input_records,
        "layout": layout,
        "outputs": outputs,
        "operations": {
            "store": "vllm.v1.attention.ops.triton_turboquant_store.triton_turboquant_store",
            "append": "vllm.v1.attention.ops.triton_turboquant_store.triton_turboquant_store",
            "decode": "vllm.v1.attention.ops.triton_turboquant_decode.triton_turboquant_decode_attention",
            "local_algorithm_reimplementation": False,
        },
        "graph": {
            "upstream_declaration": "AttentionCGSupport.UNIFORM_BATCH",
            "reference_graph_smoke": "not_exercised_minimal_direct_api",
            "reason": "A direct capture would require a separate graph harness beyond this reference fixture lane.",
            "phase6_validation_deferred": True,
        },
        "claims": {
            "comparative_latency": False,
            "performance": False,
            "quality": False,
        },
    }
    _write_json(config_root, "manifest.json", manifest)
    _write_ledger(config_root)


def _generate(stage: Path) -> None:
    import torch
    from vllm.model_executor.layers.quantization.turboquant.config import TQ_PRESETS

    if not torch.cuda.is_available():
        raise FixtureGenerationError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise FixtureGenerationError("the frozen fixture requires SM120")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source_manifest = _load_json(SOURCE_MANIFEST_PATH)
    environment = _load_json(ENVIRONMENT_PATH)
    configurations = source_manifest["configurations"]
    expected_names = [item["cache_dtype"] for item in configurations]
    if list(TQ_PRESETS) != expected_names:
        raise FixtureGenerationError(
            f"installed TQ_PRESETS differ: expected {expected_names}, found {list(TQ_PRESETS)}"
        )
    cpu_inputs = _cpu_inputs()
    input_records = _write_inputs(stage, cpu_inputs)
    for config_record in configurations:
        _generate_configuration(
            stage,
            config_record,
            cpu_inputs,
            input_records,
            source_manifest,
            environment,
        )
    fixture_set = {
        "schema_version": "turboquant-reference-fixture-set-1.0.0",
        "source_commit": source_manifest["source"]["commit"],
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "environment_manifest_sha256": sha256_file(ENVIRONMENT_PATH),
        "geometry": _fixture_geometry(),
        "inputs": input_records,
        "configurations": expected_names,
        "mandatory_configurations": [
            item["cache_dtype"]
            for item in configurations
            if item["phase5_role"] == "mandatory"
        ],
        "optional_configurations": [
            item["cache_dtype"]
            for item in configurations
            if item["phase5_role"] == "optional"
        ],
        "determinism": {
            "input_generation": "CPU torch.Generator with fixed seed, then BF16 cast",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms": True,
            "existing_set_policy": "regenerate, validate, compare exact bytes, never overwrite",
        },
        "claims": {
            "comparative_latency": False,
            "performance": False,
            "quality": False,
        },
    }
    _write_json(stage, "fixture_set.json", fixture_set)
    _write_ledger(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    stage: Path | None = None
    try:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise FixtureGenerationError(
                "CUBLAS_WORKSPACE_CONFIG must be frozen to :4096:8"
            )
        verify_all(args.venv, args.source)
        args.fixture_root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=".turboquant-fixtures-",
                dir=args.fixture_root.parent,
            )
        )
        _generate(stage)
        validation = validate_reference(stage)
        action = publish_staged(stage, args.fixture_root)
        if action == "verified_existing":
            shutil.rmtree(stage)
        result = {
            "schema_version": "turboquant-reference-generation-result-1.0.0",
            "status": "pass",
            "action": action,
            "fixture_root": str(args.fixture_root),
            "mandatory_fixture_count": validation["mandatory_fixture_count"],
            "optional_fixture_count": validation["optional_fixture_count"],
            "timing_data_created": False,
            "performance_claim_eligible": False,
        }
    except (
        ArtifactConflictError,
        FixtureGenerationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        print(
            json.dumps(
                {"status": "blocked", "error": type(error).__name__, "message": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
