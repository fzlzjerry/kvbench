#!/usr/bin/env python3
"""Execute and validate the approved Q0 cache-sensitive correctness suite."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "quality-qp1-20260918t170812117127z-170b638c-e8f4a2"
CONTRACT_SHA256 = "b4d6cb35b0b455ca19680a0a39a12016369b667f8e53e2b07525766a05ecedd1"
FREEZE_TAG = "perf-freeze-20260917-83536c37-r1"
QP1_ROOT = "6c24d464a8f7e9fa1b33f7bece2246d776c26045642ab00ed0df7b8f83200b62"
QUALITY_IMAGE = "sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32"
GPU_UUID = "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
APPROVAL = ROOT / "docs/evidence/q0/quality-contract-approval.json"
CONTRACT = ROOT / "configs/quality/quality_contract.yaml"
CONTRACT_BUNDLE = ROOT / "artifacts/quality_contract" / CONTRACT_ID
LOCKED_HASHES = (
    ROOT
    / "artifacts/performance_freeze/qp0-freeze-20260917t115200000000z-83536c37-bf28e4"
    / "locked_hot_path_hashes.json"
)
EXTERNAL_IDENTITIES = LOCKED_HASHES.with_name("external_kernel_identities.json")
CONFIGS = (
    "bf16",
    "tq_4bit_nc",
    "tq_k3v4_nc",
    "tq_3bit_nc",
    "k4v4",
    "k2v4",
    "k2v2",
    "kvq4",
    "kvq3",
    "kvq2",
)
CORE_LENGTHS = (512, 4096, 16384, 24576, 32768, 65536, 130560)
GRAPH_SAMPLES = 100
BATCH_SAMPLES = 100
BATCHES = (1, 4, 8)
TEACHER_TARGETS = 4
GREEDY_TOKENS = 4
LONG_BENCH_CONDITIONING = 16
UNIT_STAGES = tuple(f"core-l{length}" for length in CORE_LENGTHS) + (
    "longbench-suffix",
    "graph-invariance",
    "batch-invariance",
    "cache-dependence",
)
EXPECTED_UNIT_COUNT = len(CONFIGS) * len(UNIT_STAGES)
WORKER_PREFIX = "Q0_WORKER_RESULT="


class Q0Error(RuntimeError):
    """The approved Q0 contract failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Q0Error(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise Q0Error(f"JSON object required: {path}")
    return value


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path: Path, value: Any) -> None:
    write_new(path, canonical_bytes(value))


def verify_approval() -> dict[str, Any]:
    if sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise Q0Error("approved contract bytes differ")
    snapshot = CONTRACT_BUNDLE / "contract_snapshot.yaml"
    if sha256_file(snapshot) != CONTRACT_SHA256:
        raise Q0Error("approved contract snapshot bytes differ")
    receipt = load_json(APPROVAL)
    required = {
        "status": "APPROVED",
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "performance_freeze_tag": FREEZE_TAG,
        "qp1_bundle_root": QP1_ROOT,
        "authorization_scope": ["Q0_CACHE_SENSITIVE_CORRECTNESS"],
        "frozen_contract_bytes_modified": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise Q0Error("Q0 approval receipt differs from approved authority")
    if not isinstance(receipt.get("approved_at_utc"), str):
        raise Q0Error("Q0 approval time is absent")
    return receipt


def verify_locked_paths() -> dict[str, Any]:
    authority = load_json(LOCKED_HASHES)
    files = authority.get("files")
    if not isinstance(files, list) or len(files) != 68:
        raise Q0Error("locked hot-path authority differs")
    observed: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise Q0Error("locked hot-path record differs")
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise Q0Error("locked hot-path identity differs")
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise Q0Error(f"locked hot-path changed: {relative}")
        observed.append({"path": relative, "sha256": actual})
    return {
        "status": "PASS",
        "locked_path_count": len(observed),
        "authority_sha256": sha256_file(LOCKED_HASHES),
        "files_aggregate_sha256": sha256_bytes(canonical_bytes(observed)),
    }


def contract_config_fingerprints() -> dict[str, str]:
    contract = load_json(CONTRACT)["quality_contract"]
    items = contract["configurations"]["items"]
    result = {item["method_config_id"]: item["method_config_fingerprint"] for item in items}
    if tuple(result) != CONFIGS:
        raise Q0Error("quality contract configuration set differs")
    return result


def verify_entry() -> dict[str, Any]:
    receipt = verify_approval()
    protected = verify_locked_paths()
    tag_source = subprocess.run(
        ("git", "rev-list", "-n", "1", FREEZE_TAG),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tag_source != "83536c37433875cda98c36e2848e05692e9407d0":
        raise Q0Error("corrected performance freeze tag target differs")
    image = subprocess.run(
        ("docker", "image", "inspect", QUALITY_IMAGE, "--format", "{{.Id}}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if image != QUALITY_IMAGE:
        raise Q0Error("approved Quality image is unavailable")
    return {
        "status": "PASS",
        "approval": receipt,
        "protected_paths": protected,
        "quality_image_digest": image,
        "performance_source_commit": tag_source,
        "config_fingerprints": contract_config_fingerprints(),
    }


def new_campaign_id(execution_head: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    nonce = os.urandom(4).hex()
    return f"q0-{stamp}-{execution_head[:8]}-{nonce}"


def _read_i32_gzip(path: Path) -> list[int]:
    import array

    values = array.array("i")
    with gzip.open(path, "rb") as source:
        values.frombytes(source.read())
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


def frozen_input_plan() -> dict[str, Any]:
    selected = CONTRACT_BUNDLE / "selected_inputs"
    anchor_manifest = load_json(selected / "ppl/anchor_manifest.json")
    stream = _read_i32_gzip(selected / "ppl/wikitext2_test/token_stream.i32.gz")
    stages = anchor_manifest["datasets"]["wikitext2_test"]["stages"]["full"]
    probes: list[dict[str, Any]] = []
    for length in CORE_LENGTHS:
        source_length = 4096 if length == 512 else length
        rows = stages[str(source_length)]
        if not rows:
            raise Q0Error(f"no frozen Full-PPL anchor for L={source_length}")
        row = rows[0]
        prefix_start = int(row["prefix_start"])
        if length == 512:
            prefix_start = int(row["prefix_end"]) - 512
        prefix = stream[prefix_start : prefix_start + length]
        burn_in_index = prefix_start + length
        burn_in = stream[burn_in_index]
        targets = stream[burn_in_index + 1 : burn_in_index + 1 + TEACHER_TARGETS]
        if len(prefix) != length or len(targets) != TEACHER_TARGETS:
            raise Q0Error("frozen PPL diagnostic input is truncated")
        probes.append(
            {
                "stage": f"core-l{length}",
                "length": length,
                "source_length": source_length,
                "source_anchor_id": row["anchor_id"],
                "prefix_start": prefix_start,
                "prefix_sha256": sha256_bytes(_i32_bytes(prefix)),
                "burn_in_token": burn_in,
                "target_tokens": targets,
                "input_ids": prefix,
            }
        )
    index = load_json(selected / "longbench_e/qasper.index.json")
    record = index["records"][0]
    values = _read_i32_gzip(selected / "longbench_e/qasper.token_ids.i32.gz")
    offset = int(record["offset"])
    length = int(record["length"])
    prompt = values[offset : offset + length]
    if len(prompt) != length or length <= LONG_BENCH_CONDITIONING:
        raise Q0Error("frozen Qasper diagnostic input is invalid")
    return {
        "schema_version": "kvbench-q0-input-plan-1.0.0",
        "core_probes": probes,
        "longbench": {
            "task": "qasper",
            "sample_id": record["sample_id"],
            "prompt_length": length,
            "prompt_sha256": sha256_bytes(_i32_bytes(prompt)),
            "prefill_ids": prompt[:-LONG_BENCH_CONDITIONING],
            "conditioning_ids": prompt[-LONG_BENCH_CONDITIONING:],
        },
        "invariance": {
            "prefix_length": 512,
            "sample_count": 100,
            "batch_sizes": list(BATCHES),
            "tokens": [int((12000 + index * 104729) % 120000 + 1000) for index in range(100)],
        },
    }


def _i32_bytes(values: Sequence[int]) -> bytes:
    import array

    data = array.array("i", (int(value) for value in values))
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def compact_input_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": value["schema_version"],
        "core_probes": [
            {key: item[key] for key in item if key != "input_ids"}
            for item in value["core_probes"]
        ],
        "longbench": {
            key: item
            for key, item in value["longbench"].items()
            if key not in {"prefill_ids", "conditioning_ids"}
        },
        "invariance": dict(value["invariance"]),
    }


def family(configuration: str) -> str:
    if configuration == "bf16":
        return "bf16"
    if configuration.startswith("tq_"):
        return "turboquant"
    if configuration.startswith("kvq"):
        return "kvquant"
    if configuration.startswith("k"):
        return "kivi"
    raise Q0Error(f"unknown Q0 configuration: {configuration}")


def frozen_tolerance(configuration: str) -> tuple[float, float]:
    if family(configuration) == "kivi":
        from kvbench.runtime.kivi_session import PHASE8_DECODE_ATOL, PHASE8_DECODE_RTOL

        return float(PHASE8_DECODE_ATOL), float(PHASE8_DECODE_RTOL)
    if family(configuration) == "kvquant":
        from kvbench.runtime.kvquant_session import PHASE11_DECODE_ATOL, PHASE11_DECODE_RTOL

        return float(PHASE11_DECODE_ATOL), float(PHASE11_DECODE_RTOL)
    return 0.02, 0.02


def _build_method(loaded: Any, configuration: str) -> tuple[Any, int]:
    from kvbench.adapters import build_method_adapter, declared_bf16_runtime_context
    from kvbench.runtime.kivi_session import kivi_runtime_context, load_frozen_kivi_method_config
    from kvbench.runtime.kvquant_session import kvquant_runtime_context, load_frozen_kvquant_method_config
    from kvbench.runtime.turboquant_session import turboquant_runtime_context
    from scripts import phase12_unified_admission as phase12

    selected = family(configuration)
    if selected == "bf16":
        method = build_method_adapter("bf16", declared_bf16_runtime_context(loaded.model))
    elif selected == "turboquant":
        native = {
            "tq_4bit_nc": "turboquant_4bit_nc",
            "tq_k3v4_nc": "turboquant_k3v4_nc",
            "tq_3bit_nc": "turboquant_3bit_nc",
        }[configuration]
        method = build_method_adapter(native, turboquant_runtime_context())
    elif selected == "kivi":
        method = build_method_adapter(
            load_frozen_kivi_method_config(),
            kivi_runtime_context(),
            variant_id=configuration,
        )
        method.prepare_runtime()
    else:
        method = build_method_adapter(
            load_frozen_kvquant_method_config(),
            kvquant_runtime_context(configuration),
            variant_id=configuration,
        )
        method.prepare_runtime()
    phase12._validate_live_method_identity(method, configuration)
    return method, selected


def _allocate_state(
    loaded: Any,
    configuration: str,
    *,
    batch: int,
    capacity: int,
    mode: str,
    prefix_length: int,
    output_steps: int,
) -> tuple[Any, Any, Any, tuple[Any, ...], tuple[Any, ...]]:
    import torch
    from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint

    method, selected = _build_method(loaded, configuration)
    workspace = 32 * batch * (32 + 8) * 64 * 2 if selected == "bf16" else 0
    cache = method.allocate(
        batch_size=batch,
        capacity=capacity,
        device=torch.device("cuda:0"),
        workspace_bytes=workspace,
    )
    if selected == "kvquant":
        method.initialize_cache_untimed(cache)
    else:
        cache.initialize_deterministic()
    endpoint = BF16DecodeEndpoint(loaded.model, cache, method)
    positions = tuple(
        torch.tensor([prefix_length + index], dtype=torch.long, device=cache.device)
        for index in range(output_steps)
    )
    if selected == "kvquant":
        if mode == "fixed":
            cache.bind_fixed_position_tensor_untimed(positions[0], logical_position=prefix_length)
        else:
            cache.bind_growing_position_tensors_untimed(positions, starting_position=prefix_length)
    rope = tuple(endpoint.prepare_position_embeddings(position.unsqueeze(0)) for position in positions)
    return method, cache, endpoint, positions, rope


def _prefill(endpoint: Any, prefix: Any, selected: str) -> None:
    if selected == "kvquant":
        from scripts.phase13_pilot import _chunked_kvquant_prefix_store

        with _chunked_kvquant_prefix_store(endpoint):
            endpoint.prefill(prefix)
    else:
        endpoint.prefill(prefix)


def _finish_step(cache: Any, selected: str) -> None:
    # KIVI and KVQuant commit growing state inside their final-layer decode.
    # BF16 and TurboQuant retain caller-owned commit semantics.
    if selected in {"bf16", "turboquant"}:
        cache.finish_growing_step()


def growing_logits(
    loaded: Any,
    configuration: str,
    prefix_ids: Sequence[int],
    decode_tokens: Sequence[int],
    *,
    batch: int = 1,
) -> tuple[list[Any], dict[str, Any]]:
    import torch

    length = len(prefix_ids) // batch
    if length <= 0 or len(prefix_ids) != batch * length:
        raise Q0Error("growing prefix geometry is invalid")
    method, cache, endpoint, positions, rope = _allocate_state(
        loaded,
        configuration,
        batch=batch,
        capacity=length + len(decode_tokens) // batch,
        mode="growing",
        prefix_length=length,
        output_steps=len(decode_tokens) // batch,
    )
    selected = family(configuration)
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=cache.device).reshape(batch, length)
    tokens = torch.tensor(decode_tokens, dtype=torch.long, device=cache.device).reshape(batch, -1)
    with torch.inference_mode():
        _prefill(endpoint, prefix, selected)
        steps = int(tokens.shape[1])
        cache.prepare_growing(length, steps)
        outputs: list[Any] = []
        for step in range(steps):
            cache.select_growing_step(step)
            output = endpoint.decode(tokens[:, step : step + 1], positions[step], rope[step])
            outputs.append(output.detach().to(device="cpu", dtype=torch.float32, copy=True).squeeze(1))
            _finish_step(cache, selected)
    accounting = cache.accounting()
    evidence = {
        "method": selected,
        "adapter_config_fingerprint": method.config_fingerprint(cache.layout_fingerprint()),
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "batch": batch,
        "prefix_length": length,
        "decode_steps": len(outputs),
        "active_context": int(cache.active_context),
        "allocated_bytes": int(accounting.allocated_bytes),
        "finite": all(bool(torch.isfinite(output).all()) for output in outputs),
        "growing_context": True,
    }
    return outputs, evidence


def fixed_outputs(
    loaded: Any,
    configuration: str,
    prefix_rows: Sequence[Sequence[int]],
    token_groups: Sequence[Sequence[int]],
    *,
    graph: bool,
) -> tuple[list[Any], dict[str, Any]]:
    import torch
    from kvbench.runtime.cuda_graph import capture_fixed_graph

    batch = len(prefix_rows)
    length = len(prefix_rows[0])
    if any(len(row) != length for row in prefix_rows):
        raise Q0Error("fixed prefix rows have unequal length")
    method, cache, endpoint, positions, rope = _allocate_state(
        loaded,
        configuration,
        batch=batch,
        capacity=length + 1,
        mode="fixed",
        prefix_length=length,
        output_steps=1,
    )
    selected = family(configuration)
    prefix = torch.tensor(prefix_rows, dtype=torch.long, device=cache.device)
    static_token = torch.zeros((batch, 1), dtype=torch.long, device=cache.device)
    with torch.inference_mode():
        _prefill(endpoint, prefix, selected)
        cache.prepare_fixed(length)

        def operation() -> Any:
            return endpoint.decode(static_token, positions[0], rope[0])

        captured = None
        if graph:
            first = token_groups[0]
            static_token.copy_(torch.tensor(first, dtype=torch.long, device=cache.device).reshape(batch, 1))
            captured = capture_fixed_graph(operation, warmup_steps=0, device=cache.device)
        outputs: list[Any] = []
        for group in token_groups:
            static_token.copy_(torch.tensor(group, dtype=torch.long, device=cache.device).reshape(batch, 1))
            output = captured.replay() if captured is not None else operation()
            outputs.append(output.detach().to(device="cpu", dtype=torch.float32, copy=True).squeeze(1))
        torch.cuda.synchronize(device=cache.device)
    return outputs, {
        "mode": "cuda_graph" if graph else "eager",
        "fixed_l_performance_path": True,
        "growing_context_claim": False,
        "batch": batch,
        "prefix_length": length,
        "sample_groups": len(token_groups),
        "finite": all(bool(torch.isfinite(output).all()) for output in outputs),
        "graph": captured.to_dict() if captured is not None else None,
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "adapter_config_fingerprint": method.config_fingerprint(cache.layout_fingerprint()),
    }


def output_digest(output: Any) -> str:
    return sha256_bytes(output.contiguous().numpy().tobytes())


def tensor_compare(left: Any, right: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    delta = (left - right).abs()
    passed = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    return {
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
    }


def logit_metrics(method_logits: Any, bf16_logits: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    method = method_logits.float().reshape(-1)
    reference = bf16_logits.float().reshape(-1)
    ref_logp = functional.log_softmax(reference, dim=-1)
    method_logp = functional.log_softmax(method, dim=-1)
    ref_p = ref_logp.exp()
    kl = torch.sum(ref_p * (ref_logp - method_logp))
    ref_top5 = torch.topk(reference, 5).indices.tolist()
    method_top5 = torch.topk(method, 5).indices.tolist()
    ref_top1 = int(ref_top5[0])
    rank = int((method > method[ref_top1]).sum().item()) + 1
    return {
        "next_token_kl_bf16_to_method": float(kl.item()),
        "logit_cosine": float(functional.cosine_similarity(method, reference, dim=0).item()),
        "top1_agreement": ref_top1 == int(method_top5[0]),
        "top5_overlap_count": len(set(ref_top5) & set(method_top5)),
        "bf16_top1_rank_in_method": rank,
        "method_top1": int(method_top5[0]),
        "bf16_top1": ref_top1,
    }


def _save_logits(path: Path, outputs: Sequence[Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({f"logits_{index}": value.contiguous() for index, value in enumerate(outputs)}, temporary)
    os.chmod(temporary, 0o444)
    os.link(temporary, path)
    temporary.unlink()


def _load_logits(path: Path) -> list[Any]:
    from safetensors.torch import load_file

    values = load_file(path)
    return [values[key] for key in sorted(values, key=lambda item: int(item.split("_")[-1]))]


def _original_unit_root(campaign: Path, configuration: str, stage: str) -> Path:
    return campaign / "units" / configuration / stage


def _replaceable_harness_failure(campaign: Path, configuration: str, stage: str) -> bool:
    root = _original_unit_root(campaign, configuration, stage)
    if not (root / "COMPLETE").is_file() or not (root / "result.json").is_file():
        return False
    result = load_json(root / "result.json")
    return bool(
        configuration.startswith("kvq")
        and result.get("status") == "FAIL"
        and result.get("error")
        == "config_load_error: KIVI requires one explicit frozen configuration"
    )


def _replacement_unit_root(campaign: Path, configuration: str, stage: str) -> Path:
    return campaign / "continuations" / "kvquant-q0-wrapper-fix" / "units" / configuration / stage


def _unit_root(campaign: Path, configuration: str, stage: str) -> Path:
    replacement = _replacement_unit_root(campaign, configuration, stage)
    if (replacement / "COMPLETE").is_file():
        return replacement
    if _replaceable_harness_failure(campaign, configuration, stage):
        return replacement
    return _original_unit_root(campaign, configuration, stage)


def _unit_complete(campaign: Path, configuration: str, stage: str) -> bool:
    root = _unit_root(campaign, configuration, stage)
    return (root / "COMPLETE").is_file() and (root / "result.json").is_file()


def _finalize_unit(
    campaign: Path,
    configuration: str,
    stage: str,
    result: Mapping[str, Any],
    *,
    logits: Sequence[Any] | None = None,
) -> None:
    original = _original_unit_root(campaign, configuration, stage)
    replacing = _replaceable_harness_failure(campaign, configuration, stage)
    root = _replacement_unit_root(campaign, configuration, stage) if replacing else original
    if root.exists():
        raise Q0Error(f"Q0 unit already exists without completion: {configuration}/{stage}")
    root.mkdir(parents=True)
    if logits is not None:
        _save_logits(root / "bf16_logits.safetensors", logits)
    selected_result = dict(result)
    if replacing:
        selected_result.update(
            {
                "replacement_of": original.relative_to(campaign).as_posix(),
                "replacement_reason": "quality_harness_kvquant_routing_and_growing_lifecycle_errors",
                "original_failure_preserved": True,
            }
        )
    write_json_new(root / "result.json", selected_result)
    payload = {
        "schema_version": "kvbench-q0-unit-complete-1.0.0",
        "configuration": configuration,
        "stage": stage,
        "status": result["status"],
        "result_sha256": sha256_file(root / "result.json"),
        "written_last": True,
    }
    if logits is not None:
        payload["bf16_logits_sha256"] = sha256_file(root / "bf16_logits.safetensors")
    write_json_new(root / "COMPLETE", payload)


def _core_stage(
    loaded: Any,
    campaign: Path,
    configuration: str,
    probe: Mapping[str, Any],
) -> None:
    stage = str(probe["stage"])
    prefix = [int(value) for value in probe["input_ids"]]
    burn_in = int(probe["burn_in_token"])
    targets = [int(value) for value in probe["target_tokens"]]
    decode = [burn_in, *targets[:-1]]
    outputs, evidence = growing_logits(loaded, configuration, prefix, decode)
    finite = bool(evidence["finite"])
    diagnostics: list[dict[str, Any]] = []
    if configuration == "bf16":
        diagnostics = [{"target_token": target, "output_sha256": output_digest(output)} for target, output in zip(targets, outputs, strict=True)]
    else:
        reference = _load_logits(_unit_root(campaign, "bf16", stage) / "bf16_logits.safetensors")
        diagnostics = [
            {"target_token": target, "output_sha256": output_digest(output), **logit_metrics(output, ref)}
            for target, output, ref in zip(targets, outputs, reference, strict=True)
        ]
    greedy: dict[str, Any] | None = None
    if int(probe["length"]) == 512:
        trajectory = [burn_in]
        generated: list[int] = []
        greedy_outputs: list[Any] = []
        # Reconstruct the disposable state for each bounded branch so the next
        # input is the actual preceding greedy token, never a guessed token.
        for _ in range(GREEDY_TOKENS):
            observed, _ = growing_logits(loaded, configuration, prefix, trajectory)
            output = observed[-1]
            greedy_outputs.append(output)
            generated.append(int(output.argmax(dim=-1).item()))
            trajectory.append(generated[-1])
        greedy = {
            "budget": GREEDY_TOKENS,
            "tokens": generated,
            "invalid_tokens": sum(token < 0 or token >= int(output.shape[-1]) for token, output in zip(generated, greedy_outputs, strict=True)),
            "first_divergence_from_bf16": None,
        }
        if configuration != "bf16":
            bf = load_json(_unit_root(campaign, "bf16", stage) / "result.json")["greedy_control"]["tokens"]
            greedy["first_divergence_from_bf16"] = next((index for index, pair in enumerate(zip(generated, bf, strict=True)) if pair[0] != pair[1]), None)
    result = {
        "schema_version": "kvbench-q0-core-result-1.0.0",
        "status": "PASS" if finite else "FAIL",
        "configuration": configuration,
        "stage": stage,
        "prefix_length": int(probe["length"]),
        "source_anchor_id": probe["source_anchor_id"],
        "prefix_sha256": probe["prefix_sha256"],
        "alignment": "prefill_prefix_then_unscored_burn_in_then_teacher_force_previous_target",
        "teacher_forced_target_count": len(targets),
        "diagnostics": diagnostics,
        "greedy_control": greedy,
        "growing_execution": evidence,
        "benchmark_score_computed": False,
        "performance_timing_collected": False,
    }
    _finalize_unit(campaign, configuration, stage, result, logits=outputs if configuration == "bf16" else None)


def _longbench_stage(loaded: Any, campaign: Path, configuration: str, plan: Mapping[str, Any]) -> None:
    stage = "longbench-suffix"
    prefix = [int(value) for value in plan["prefill_ids"]]
    conditioning = [int(value) for value in plan["conditioning_ids"]]
    outputs, evidence = growing_logits(loaded, configuration, prefix, conditioning)
    last = outputs[-1]
    generated = [int(last.argmax(dim=-1).item())]
    # Four-token diagnostic answer, never benchmark-scored.
    answer_inputs = conditioning + generated
    for _ in range(GREEDY_TOKENS - 1):
        answer_outputs, _ = growing_logits(loaded, configuration, prefix, answer_inputs)
        generated.append(int(answer_outputs[-1].argmax(dim=-1).item()))
        answer_inputs.append(generated[-1])
    result = {
        "schema_version": "kvbench-q0-longbench-result-1.0.0",
        "status": "PASS" if evidence["finite"] and len(conditioning) == 16 else "FAIL",
        "configuration": configuration,
        "stage": stage,
        "task": plan["task"],
        "sample_id": plan["sample_id"],
        "prompt_sha256": plan["prompt_sha256"],
        "concatenation_identity": sha256_bytes(_i32_bytes([*prefix, *conditioning])) == plan["prompt_sha256"],
        "conditioning_tokens_decoded": len(conditioning),
        "answer_logits_after_final_conditioning_only": True,
        "answer_diagnostic_tokens": generated,
        "benchmark_score_computed": False,
        "growing_execution": evidence,
    }
    if not result["concatenation_identity"]:
        result["status"] = "FAIL"
    _finalize_unit(campaign, configuration, stage, result)


def _graph_stage(loaded: Any, campaign: Path, configuration: str, plan: Mapping[str, Any]) -> None:
    stage = "graph-invariance"
    prefix = next(item["input_ids"] for item in plan["core_probes"] if item["length"] == 512)
    tokens = [[int(token)] for token in plan["invariance"]["tokens"]]
    eager, eager_evidence = fixed_outputs(loaded, configuration, [prefix], tokens, graph=False)
    graph, graph_evidence = fixed_outputs(loaded, configuration, [prefix], tokens, graph=True)
    atol, rtol = frozen_tolerance(configuration)
    comparisons = [tensor_compare(left, right, atol=atol, rtol=rtol) for left, right in zip(eager, graph, strict=True)]
    result = {
        "schema_version": "kvbench-q0-graph-invariance-1.0.0",
        "status": "PASS" if all(item["passed"] for item in comparisons) and eager_evidence["finite"] and graph_evidence["finite"] else "FAIL",
        "configuration": configuration,
        "stage": stage,
        "sample_count": len(comparisons),
        "selected_token_agreement_count": sum(int(left.argmax()) == int(right.argmax()) for left, right in zip(eager, graph, strict=True)),
        "maximum_abs_error": max(item["max_abs_error"] for item in comparisons),
        "frozen_atol": atol,
        "frozen_rtol": rtol,
        "eager": eager_evidence,
        "graph": graph_evidence,
        "quality_lane": "eager_primary",
        "fixed_l_graph_relationship_only": True,
    }
    _finalize_unit(campaign, configuration, stage, result)


def _batch_stage(loaded: Any, campaign: Path, configuration: str, plan: Mapping[str, Any]) -> None:
    stage = "batch-invariance"
    prefix = next(item["input_ids"] for item in plan["core_probes"] if item["length"] == 512)
    tokens = [int(value) for value in plan["invariance"]["tokens"]]
    references, _ = fixed_outputs(loaded, configuration, [prefix], [[token] for token in tokens], graph=False)
    atol, rtol = frozen_tolerance(configuration)
    batch_results: dict[str, Any] = {}
    all_passed = True
    for batch in (4, 8):
        padded = list(tokens)
        while len(padded) % batch:
            padded.append(tokens[len(padded) % len(tokens)])
        groups = [padded[index : index + batch] for index in range(0, len(padded), batch)]
        outputs, evidence = fixed_outputs(loaded, configuration, [prefix for _ in range(batch)], groups, graph=False)
        flattened = [row for output in outputs for row in output]
        comparisons = [tensor_compare(references[index][0], flattened[index], atol=atol, rtol=rtol) for index in range(BATCH_SAMPLES)]
        passed = all(item["passed"] for item in comparisons) and evidence["finite"]
        all_passed = all_passed and passed
        batch_results[str(batch)] = {
            "passed": passed,
            "logical_sample_count": BATCH_SAMPLES,
            "carrier_slot_count": len(flattened),
            "duplicate_fill_slots": len(flattened) - BATCH_SAMPLES,
            "maximum_abs_error": max(item["max_abs_error"] for item in comparisons),
            "selected_token_agreement_count": sum(int(references[index][0].argmax()) == int(flattened[index].argmax()) for index in range(BATCH_SAMPLES)),
            "evidence": evidence,
        }
    result = {
        "schema_version": "kvbench-q0-batch-invariance-1.0.0",
        "status": "PASS" if all_passed else "FAIL",
        "configuration": configuration,
        "stage": stage,
        "sample_count": BATCH_SAMPLES,
        "batch_sizes": list(BATCHES),
        "equal_length_cases": True,
        "unequal_length_padding_claimed": False,
        "reason_padding_not_claimed": "frozen endpoint has no approved per-row mask contract",
        "frozen_atol": atol,
        "frozen_rtol": rtol,
        "results": batch_results,
    }
    _finalize_unit(campaign, configuration, stage, result)


def _active_packed_tensor(cache: Any, selected: str) -> tuple[str, Any]:
    if selected == "turboquant":
        return "packed_cache", cache.packed_cache
    if selected == "kivi":
        return "packed_key_history", cache.packed_key_history
    if selected == "kvquant":
        return "packed_value_cache", cache.packed_value_cache
    raise Q0Error("BF16 has no compressed packed payload")


def _cache_dependence_stage(loaded: Any, campaign: Path, configuration: str, plan: Mapping[str, Any]) -> None:
    import torch

    stage = "cache-dependence"
    selected = family(configuration)
    if selected == "bf16":
        _finalize_unit(campaign, configuration, stage, {
            "schema_version": "kvbench-q0-cache-dependence-1.0.0",
            "status": "PASS",
            "configuration": configuration,
            "stage": stage,
            "applicability": "not_applicable_uncompressed_baseline",
        })
        return
    prefix = next(item["input_ids"] for item in plan["core_probes"] if item["length"] == 512)
    token = int(plan["invariance"]["tokens"][0])
    method, cache, endpoint, positions, rope = _allocate_state(
        loaded, configuration, batch=1, capacity=513, mode="fixed", prefix_length=512, output_steps=1
    )
    prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=cache.device).reshape(1, 512)
    token_tensor = torch.tensor([[token]], dtype=torch.long, device=cache.device)
    with torch.inference_mode():
        _prefill(endpoint, prefix_tensor, selected)
        cache.prepare_fixed(512)
        baseline_a = endpoint.decode(token_tensor, positions[0], rope[0]).detach().to("cpu", dtype=torch.float32, copy=True)
        baseline_b = endpoint.decode(token_tensor, positions[0], rope[0]).detach().to("cpu", dtype=torch.float32, copy=True)
        storage_name, storage = _active_packed_tensor(cache, selected)
        flat = storage.view(-1)
        width = min(64, int(flat.numel()))
        original = flat[:width].clone()
        flat[:width].bitwise_xor_(1)
        perturbed = endpoint.decode(token_tensor, positions[0], rope[0]).detach().to("cpu", dtype=torch.float32, copy=True)
        flat[:width].copy_(original)
        restored = endpoint.decode(token_tensor, positions[0], rope[0]).detach().to("cpu", dtype=torch.float32, copy=True)
        original_decode = method.decode_attention
        negative_control_failed_closed = False

        def intercepted(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise Q0Error("expected compressed decode interception")

        method.decode_attention = intercepted
        try:
            endpoint.decode(token_tensor, positions[0], rope[0])
        except Q0Error as error:
            negative_control_failed_closed = str(error) == "expected compressed decode interception"
        finally:
            method.decode_attention = original_decode
    deterministic = torch.equal(baseline_a, baseline_b) and torch.equal(baseline_a, restored)
    changed = output_digest(baseline_a) != output_digest(perturbed)
    finite = bool(torch.isfinite(perturbed).all())
    result = {
        "schema_version": "kvbench-q0-cache-dependence-1.0.0",
        "status": "PASS" if deterministic and changed and finite and negative_control_failed_closed else "FAIL",
        "configuration": configuration,
        "stage": stage,
        "operation": f"{selected}.decode_attention",
        "active_storage": storage_name,
        "valid_encoded_words_perturbed": width,
        "metadata_or_shape_changed": False,
        "baseline_repeat_exact": bool(torch.equal(baseline_a, baseline_b)),
        "restored_exact": bool(torch.equal(baseline_a, restored)),
        "perturbed_output_changed": changed,
        "perturbed_output_finite": finite,
        "negative_control_failed_closed": negative_control_failed_closed,
        "fallback_permitted": False,
    }
    _finalize_unit(campaign, configuration, stage, result)


def _failure_unit(campaign: Path, configuration: str, stage: str, error: BaseException) -> None:
    result = {
        "schema_version": "kvbench-q0-unit-failure-1.0.0",
        "status": "FAIL",
        "configuration": configuration,
        "stage": stage,
        "failure_class": "implementation_or_correctness_failure",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "retry_permitted": False,
    }
    _finalize_unit(campaign, configuration, stage, result)


def run_worker(campaign: Path, configuration: str, execution_head: str) -> dict[str, Any]:
    if os.environ.get("KVBENCH_QUALITY_IMAGE_DIGEST") != QUALITY_IMAGE:
        raise Q0Error("Quality image identity environment differs")
    if subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip() != execution_head:
        raise Q0Error("Q0 execution HEAD differs")
    if subprocess.run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=ROOT, check=True, capture_output=True, text=True).stdout:
        raise Q0Error("Q0 execution source is not clean")
    verify_approval()
    verify_locked_paths()
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    plan = frozen_input_plan()
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    failures = 0
    with torch.inference_mode(), forced_flash_execution():
        for probe in plan["core_probes"]:
            stage = str(probe["stage"])
            if _unit_complete(campaign, configuration, stage):
                continue
            try:
                _core_stage(loaded, campaign, configuration, probe)
            except BaseException as error:
                failures += 1
                _failure_unit(campaign, configuration, stage, error)
        for stage, callback, payload in (
            ("longbench-suffix", _longbench_stage, plan["longbench"]),
            ("graph-invariance", _graph_stage, plan),
            ("batch-invariance", _batch_stage, plan),
            ("cache-dependence", _cache_dependence_stage, plan),
        ):
            if _unit_complete(campaign, configuration, stage):
                continue
            try:
                callback(loaded, campaign, configuration, payload)
            except BaseException as error:
                failures += 1
                _failure_unit(campaign, configuration, stage, error)
    return {
        "configuration": configuration,
        "terminal_units": sum(_unit_complete(campaign, configuration, stage) for stage in UNIT_STAGES),
        "failed_units": failures,
    }


def _gpu_idle() -> dict[str, Any]:
    result = subprocess.run(
        ("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Q0Error("GPU process query failed")
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if processes:
        raise Q0Error("foreign GPU process detected before Q0 unit")
    return {"status": "clean", "checked_at_utc": utc_now(), "process_count": 0}


@contextmanager
def gpu_lock() -> Iterable[None]:
    path = Path("/tmp/kvbench-q0-gpu.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _docker_worker_command(campaign: Path, configuration: str, execution_head: str) -> list[str]:
    control = ROOT / "artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac/control"
    kivi_source = control / "kivi-source"
    kivi_extension = control / "kivi-image-source/quant/kivi_gemv.cpython-312-x86_64-linux-gnu.so"
    kvquant_source = control / "kvquant-source"
    kvquant_build = control / "kvquant-build"
    calibration = ROOT / "calibration/kvquant/kvqcal-cdb724c806d64d095c040d2673a987a3"
    model_root = Path("/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct")
    for path in (kivi_source, kivi_extension, kvquant_source, kvquant_build, calibration, model_root):
        if not path.exists():
            raise Q0Error(f"required local Q0 input is absent: {path}")
    container_campaign = f"/home/rockrock/cmu_paper/artifacts/q0/{campaign.name}"
    return [
        "docker", "run", "--rm", "--read-only", "--network=none", "--pid=host",
        "--gpus", f"device={GPU_UUID}",
        "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=16g",
        "--tmpfs", "/root:rw,exec,nosuid,nodev,size=8g",
        "--mount", f"type=bind,src={ROOT},dst=/home/rockrock/cmu_paper,readonly",
        "--mount", f"type=bind,src={campaign},dst={container_campaign}",
        "--mount", f"type=bind,src={kivi_source},dst=/opt/kivi-source,readonly",
        "--mount", f"type=bind,src={kivi_extension},dst=/opt/kvbench/.phase3/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so,readonly",
        "--mount", f"type=bind,src={kvquant_source},dst=/opt/kvquant-source,readonly",
        "--mount", f"type=bind,src={kvquant_build},dst=/opt/kvquant-build,readonly",
        "--mount", f"type=bind,src={calibration},dst=/opt/kvquant-calibration,readonly",
        "--mount", f"type=bind,src={model_root},dst=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct,readonly",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONNOUSERSITE=1",
        "--env", "PYTHONPATH=/opt/kivi-source:/opt/kvbench/.phase3/site-packages:/home/rockrock/cmu_paper/src:/home/rockrock/cmu_paper",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "HF_HUB_DISABLE_TELEMETRY=1", "--env", "TOKENIZERS_PARALLELISM=false",
        "--env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "--env", "TORCH_CUDA_ARCH_LIST=12.0+PTX",
        "--env", "TRITON_CACHE_DIR=/root/.triton",
        "--env", "KVBENCH_KIVI_SOURCE_ROOT=/opt/kivi-source",
        "--env", "KVBENCH_KVQUANT_SOURCE_ROOT=/opt/kvquant-source",
        "--env", "KVBENCH_KVQUANT_CALIBRATION_ROOT=/opt/kvquant-calibration",
        "--env", "KVBENCH_KVQUANT_EXTENSION=/opt/kvquant-build/quant_cuda.cpython-312-x86_64-linux-gnu.so",
        "--env", "KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION=/opt/kvquant-build/quant_cuda.cpython-312-x86_64-linux-gnu.so",
        "--env", "KVBENCH_KVQUANT_EXTENSION_SHA256=b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d",
        "--env", f"KVBENCH_QUALITY_IMAGE_DIGEST={QUALITY_IMAGE}",
        "--env", "KVBENCH_EXECUTION_ENVIRONMENT=quality_container",
        "--workdir", "/home/rockrock/cmu_paper",
        "--entrypoint", "/opt/kvbench/.venv/bin/python3", QUALITY_IMAGE,
        "-m", "scripts.q0_cache_sensitive_correctness", "--run-worker",
        "--campaign", container_campaign, "--configuration", configuration,
        "--execution-head", execution_head,
    ]


def run_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    campaign = campaign.resolve()
    if campaign.exists():
        raise Q0Error("Q0 campaign ID already exists")
    campaign.mkdir(parents=True)
    write_json_new(campaign / "approval.json", load_json(APPROVAL))
    input_plan = frozen_input_plan()
    write_json_new(campaign / "input_plan.json", compact_input_plan(input_plan))
    write_json_new(campaign / "entry.json", verify_entry())
    write_json_new(campaign / "started.json", {
        "schema_version": "kvbench-q0-campaign-start-1.0.0",
        "campaign_id": campaign.name,
        "execution_head": execution_head,
        "started_at_utc": utc_now(),
        "configurations": list(CONFIGS),
        "unit_count": EXPECTED_UNIT_COUNT,
    })
    worker_records: list[dict[str, Any]] = []
    with gpu_lock():
        for configuration in CONFIGS:
            if all(_unit_complete(campaign, configuration, stage) for stage in UNIT_STAGES):
                continue
            _gpu_idle()
            for attempt in range(2):
                attempt_root = campaign / "worker_attempts" / configuration / f"attempt-{attempt:02d}"
                attempt_root.mkdir(parents=True)
                command = _docker_worker_command(campaign, configuration, execution_head)
                result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
                write_new(attempt_root / "stdout.txt", result.stdout.encode())
                write_new(attempt_root / "stderr.txt", result.stderr.encode())
                write_json_new(attempt_root / "command.json", {
                    "schema_version": "kvbench-q0-worker-attempt-1.0.0",
                    "configuration": configuration,
                    "attempt": attempt,
                    "return_code": result.returncode,
                    "quality_image_digest": QUALITY_IMAGE,
                    "command_sha256": sha256_bytes(canonical_bytes(command)),
                })
                terminal = sum(_unit_complete(campaign, configuration, stage) for stage in UNIT_STAGES)
                worker_records.append({"configuration": configuration, "attempt": attempt, "return_code": result.returncode, "terminal_units": terminal})
                if terminal == len(UNIT_STAGES) and result.returncode == 0:
                    break
                if attempt == 0 and result.returncode != 0:
                    continue
                raise Q0Error(f"Q0 worker incomplete: {configuration}")
            _gpu_idle()
    write_json_new(campaign / "worker_index.json", {"records": worker_records})
    return {"campaign_id": campaign.name, "terminal_units": sum(_unit_complete(campaign, config, stage) for config in CONFIGS for stage in UNIT_STAGES)}


def resume_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    campaign = campaign.resolve(strict=True)
    if (campaign / "COMPLETE").exists():
        raise Q0Error("finalized Q0 campaign cannot be resumed")
    replaceable = [
        (configuration, stage)
        for configuration in CONFIGS
        for stage in UNIT_STAGES
        if _replaceable_harness_failure(campaign, configuration, stage)
    ]
    if len(replaceable) != 33:
        raise Q0Error("expected exact 33 preserved KVQuant harness failures")
    pending = [
        (configuration, stage)
        for configuration, stage in replaceable
        if not (_replacement_unit_root(campaign, configuration, stage) / "COMPLETE").exists()
    ]
    continuation = campaign / "continuations" / "kvquant-q0-wrapper-fix"
    continuation.mkdir(parents=True, exist_ok=True)
    authority = {
        "schema_version": "kvbench-q0-harness-continuation-1.0.0",
        "execution_head": execution_head,
        "original_execution_head": load_json(campaign / "started.json")["execution_head"],
        "reason": "kvquant_configuration_routing_and_growing_lifecycle_were_incorrect_in_the_q0_wrapper",
        "replacement_unit_count": 33,
        "timing_or_method_code_changed": False,
        "original_failures_preserved": True,
        "continued_at_utc": utc_now(),
    }
    authority_path = continuation / "authority.json"
    if authority_path.exists():
        existing = load_json(authority_path)
        for key in (
            "schema_version",
            "execution_head",
            "original_execution_head",
            "reason",
            "replacement_unit_count",
            "timing_or_method_code_changed",
            "original_failures_preserved",
        ):
            if existing.get(key) != authority.get(key):
                raise Q0Error(f"Q0 continuation authority mismatch: {key}")
    else:
        write_json_new(authority_path, authority)
    records: list[dict[str, Any]] = []
    with gpu_lock():
        for configuration in ("kvq4", "kvq3", "kvq2"):
            if not any(item_configuration == configuration for item_configuration, _ in pending):
                continue
            _gpu_idle()
            existing_attempts = sorted((campaign / "worker_attempts" / configuration).glob("attempt-*"))
            attempt = len(existing_attempts)
            attempt_root = campaign / "worker_attempts" / configuration / f"attempt-{attempt:02d}"
            attempt_root.mkdir(parents=True)
            command = _docker_worker_command(campaign, configuration, execution_head)
            result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
            write_new(attempt_root / "stdout.txt", result.stdout.encode())
            write_new(attempt_root / "stderr.txt", result.stderr.encode())
            write_json_new(attempt_root / "command.json", {
                "schema_version": "kvbench-q0-worker-attempt-1.0.0",
                "configuration": configuration,
                "attempt": attempt,
                "return_code": result.returncode,
                "quality_image_digest": QUALITY_IMAGE,
                "command_sha256": sha256_bytes(canonical_bytes(command)),
                "continuation": "kvquant-q0-wrapper-fix",
            })
            terminal = sum(_unit_complete(campaign, configuration, stage) for stage in UNIT_STAGES)
            records.append({"configuration": configuration, "attempt": attempt, "return_code": result.returncode, "terminal_units": terminal})
            if result.returncode != 0 or terminal != len(UNIT_STAGES):
                raise Q0Error(f"Q0 continuation worker incomplete: {configuration}")
            _gpu_idle()
    write_json_new(continuation / "worker_index.json", {"records": records})
    return {
        "campaign_id": campaign.name,
        "replacement_units_completed_this_invocation": len(pending),
        "terminal_units": sum(_unit_complete(campaign, config, stage) for config in CONFIGS for stage in UNIT_STAGES),
    }


def _role(path: str) -> str:
    if path == "manifest.json":
        return "manifest"
    if path.endswith("result.json"):
        return "q0_unit_result"
    if path.endswith(".safetensors"):
        return "bf16_reference_logits"
    return "q0_evidence"


def finalize_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    if (campaign / "COMPLETE").exists():
        raise Q0Error("Q0 campaign is already finalized")
    results: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, str]] = []
    for configuration in CONFIGS:
        stages: dict[str, Any] = {}
        for stage in UNIT_STAGES:
            if not _unit_complete(campaign, configuration, stage):
                raise Q0Error(f"Q0 unit incomplete: {configuration}/{stage}")
            result = load_json(_unit_root(campaign, configuration, stage) / "result.json")
            stages[stage] = result
            if result.get("status") != "PASS":
                failed.append({"configuration": configuration, "stage": stage})
        results[configuration] = stages
    verdicts = {
        configuration: {
            "q0": "PASS" if all(item.get("status") == "PASS" for item in stages.values()) else "FAIL",
            "fast_ppl_eligible": all(item.get("status") == "PASS" for item in stages.values()),
            "failed_stages": [stage for stage, item in stages.items() if item.get("status") != "PASS"],
        }
        for configuration, stages in results.items()
    }
    summary = {
        "schema_version": "kvbench-q0-summary-1.0.0",
        "campaign_id": campaign.name,
        "status": "PASS" if not failed else "PARTIAL",
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "quality_image_digest": QUALITY_IMAGE,
        "execution_head": execution_head,
        "planned_units": EXPECTED_UNIT_COUNT,
        "completed_units": EXPECTED_UNIT_COUNT,
        "failed_units": failed,
        "configuration_verdicts": verdicts,
        "fast_ppl_started": False,
        "full_ppl_started": False,
        "longbench_scoring_started": False,
        "performance_rerun": False,
        "quality_state": "Q0_COMPLETE_FAST_PPL_NOT_STARTED",
        "protected_paths": verify_locked_paths(),
        "obsolete_harness_failure_units": [
            {
                "configuration": configuration,
                "stage": stage,
                "original_path": _original_unit_root(campaign, configuration, stage).relative_to(campaign).as_posix(),
                "replacement_path": _replacement_unit_root(campaign, configuration, stage).relative_to(campaign).as_posix(),
            }
            for configuration in ("kvq4", "kvq3", "kvq2")
            for stage in UNIT_STAGES
            if _replaceable_harness_failure(campaign, configuration, stage)
        ],
        "obsolete_intermediate_harness_failure_units": [
            {
                "configuration": configuration,
                "stage": stage,
                "path": path.relative_to(campaign).as_posix(),
                "error": load_json(path)["error"],
            }
            for configuration in ("kvq4", "kvq3", "kvq2")
            for stage in UNIT_STAGES
            for path in [
                campaign
                / "continuations"
                / "kvquant-family-routing-fix"
                / "units"
                / configuration
                / stage
                / "result.json"
            ]
            if path.is_file()
        ],
    }
    write_json_new(campaign / "q0_summary.json", summary)
    manifest = {
        "schema_version": "kvbench-q0-campaign-1.0.0",
        "run_id": campaign.name,
        "status": summary["status"],
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "performance_freeze_tag": FREEZE_TAG,
        "qp1_root": QP1_ROOT,
        "quality_image_digest": QUALITY_IMAGE,
        "execution_head": execution_head,
        "configuration_count": len(CONFIGS),
        "unit_count": EXPECTED_UNIT_COUNT,
        "completed_at_utc": utc_now(),
    }
    write_json_new(campaign / "manifest.json", manifest)
    payload_paths = sorted(path for path in campaign.rglob("*") if path.is_file())
    inventory_files = [
        {"path": path.relative_to(campaign).as_posix(), "role": _role(path.relative_to(campaign).as_posix()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in payload_paths
    ]
    inventory = {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": campaign.name,
        "files": inventory_files,
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    }
    write_json_new(campaign / "artifact_inventory.json", inventory)
    ledger_paths = sorted(path for path in campaign.rglob("*") if path.is_file() and path.name not in {"checksums.sha256", "COMPLETE"})
    ledger = "".join(f"{sha256_file(path)}  {path.relative_to(campaign).as_posix()}\n" for path in ledger_paths).encode()
    write_new(campaign / "checksums.sha256", ledger)
    completion = {
        "schema_version": "kvbench-q0-complete-1.0.0",
        "run_id": campaign.name,
        "status": summary["status"],
        "manifest_sha256": sha256_file(campaign / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(campaign / "artifact_inventory.json"),
        "checksum_ledger_sha256": sha256_file(campaign / "checksums.sha256"),
        "checksum_ledger_path": "checksums.sha256",
        "written_last": True,
    }
    write_json_new(campaign / "COMPLETE", completion)
    for path in sorted(campaign.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    campaign.chmod(0o555)
    from scripts.r2_artifact import validate_local_artifact

    artifact = validate_local_artifact(campaign)
    return {"status": summary["status"], "campaign_id": campaign.name, "root_sha256": artifact.root_sha256, "object_count": len(artifact.files), "verdicts": verdicts}


def validate_campaign(campaign: Path) -> dict[str, Any]:
    from scripts.r2_artifact import validate_local_artifact

    artifact = validate_local_artifact(campaign)
    summary = load_json(campaign / "q0_summary.json")
    if summary.get("contract_sha256") != CONTRACT_SHA256 or summary.get("planned_units") != EXPECTED_UNIT_COUNT or summary.get("completed_units") != EXPECTED_UNIT_COUNT:
        raise Q0Error("Q0 summary authority differs")
    return {"status": "PASS", "q0_status": summary["status"], "root_sha256": artifact.root_sha256, "object_count": len(artifact.files)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-entry", action="store_true")
    action.add_argument("--print-input-plan", action="store_true")
    action.add_argument("--new-campaign-id", action="store_true")
    action.add_argument("--run-worker", action="store_true")
    action.add_argument("--run-campaign", action="store_true")
    action.add_argument("--resume-campaign", action="store_true")
    action.add_argument("--finalize", action="store_true")
    action.add_argument("--validate", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--configuration", choices=CONFIGS)
    parser.add_argument("--execution-head")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_entry:
        print(json.dumps(verify_entry(), sort_keys=True))
        return 0
    if args.print_input_plan:
        print(json.dumps(compact_input_plan(frozen_input_plan()), sort_keys=True))
        return 0
    if args.new_campaign_id:
        if not args.execution_head:
            raise Q0Error("--execution-head is required")
        print(new_campaign_id(args.execution_head))
        return 0
    if args.run_worker:
        if args.campaign is None or args.configuration is None or not args.execution_head:
            raise Q0Error("worker arguments are required")
        result = run_worker(args.campaign, args.configuration, args.execution_head)
        print(WORKER_PREFIX + json.dumps(result, sort_keys=True))
        return 0
    if args.run_campaign:
        if args.campaign is None or not args.execution_head:
            raise Q0Error("campaign arguments are required")
        print(json.dumps(run_campaign(args.campaign, args.execution_head), sort_keys=True))
        return 0
    if args.resume_campaign:
        if args.campaign is None or not args.execution_head:
            raise Q0Error("continuation arguments are required")
        print(json.dumps(resume_campaign(args.campaign, args.execution_head), sort_keys=True))
        return 0
    if args.finalize:
        if args.campaign is None or not args.execution_head:
            raise Q0Error("finalization arguments are required")
        print(json.dumps(finalize_campaign(args.campaign, args.execution_head), sort_keys=True))
        return 0
    if args.validate:
        if args.campaign is None:
            raise Q0Error("--campaign is required")
        print(json.dumps(validate_campaign(args.campaign), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
