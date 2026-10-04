"""Bounded untimed Phase 4 functional smokes through the method adapter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
from typing import Any

from kvbench.adapters import (
    KVCacheMethod,
    MethodRuntimeContext,
    build_method_adapter,
)
from kvbench.config import REPOSITORY_ROOT, load_config
from kvbench.runtime.backend import backend_identity, forced_flash_execution
from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
from kvbench.runtime.gqa_audit import audit_cache_geometry
from kvbench.runtime.method_harness import run_graph_harness
from kvbench.runtime.model_loader import (
    LoadedFrozenModel,
    load_frozen_model,
)
from kvbench.runtime.numerical import (
    cache_history_sha256_untimed,
    tensor_sha256_untimed,
)
from kvbench.runtime.process_supervision import publish_bytes_no_replace
from kvbench.schema import (
    MethodConfig,
    MethodConfigFingerprint,
    canonical_json_bytes,
    sha256_hex,
)


PHASE4_SMOKE_SCHEMA_VERSION = "kvbench-phase4-functional-smoke-1.0.0"
PHASE4_SMOKE_INDEX_SCHEMA_VERSION = (
    "kvbench-phase4-functional-smoke-index-1.0.0"
)
_PREFIX_LENGTH = 128
_GROWING_STEPS = 4
_DEVICE = "cuda:0"
_GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
SmokeScenario = Callable[
    [KVCacheMethod, LoadedFrozenModel, Any],
    dict[str, Any],
]


class Phase4SmokeError(RuntimeError):
    """A bounded functional smoke could not be run or published."""


def _git_head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if _GIT_SHA.fullmatch(value) is None:
        raise Phase4SmokeError("current Git SHA is invalid")
    return value


def _require_clean_worktree() -> None:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        raise Phase4SmokeError("functional smoke requires a clean worktree")


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id(kind: str, git_sha: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ").lower()
    return (
        f"phase4-smoke-{kind}-{timestamp}-{git_sha[:8]}-"
        f"{secrets.token_hex(3)}"
    )


def _workspace_bytes(batch_size: int) -> int:
    return 32 * batch_size * (32 + 8) * 64 * 2


def _deterministic_ids(
    torch: Any,
    *,
    length: int,
    offset: int,
) -> Any:
    values = torch.arange(length, dtype=torch.long, device=_DEVICE)
    return ((values + offset) % 120_000 + 1_000).unsqueeze(0)


def _build_adapter(
    loaded: LoadedFrozenModel,
    backend: dict[str, Any],
) -> tuple[KVCacheMethod, MethodConfigFingerprint]:
    method_config = load_config(REPOSITORY_ROOT / "configs/methods/bf16.yaml")
    if not isinstance(method_config, MethodConfig):
        raise Phase4SmokeError("BF16 method config has the wrong type")
    adapter = build_method_adapter(
        method_config,
        MethodRuntimeContext(
            model_id=loaded.identity.model_id,
            model_revision=loaded.identity.revision,
            backend_id=str(backend["backend_id"]),
            backend_fingerprint=sha256_hex(canonical_json_bytes(backend)),
            num_layers=loaded.identity.num_hidden_layers,
            num_query_heads=loaded.identity.num_attention_heads,
            num_kv_heads=loaded.identity.num_key_value_heads,
            head_dim=loaded.identity.head_dim,
        ),
    )
    return adapter, MethodConfigFingerprint.from_config(method_config, "bf16")


def _allocate(
    method: KVCacheMethod,
    loaded: LoadedFrozenModel,
    *,
    capacity: int,
) -> tuple[Any, BF16DecodeEndpoint]:
    cache = method.allocate(
        batch_size=1,
        capacity=capacity,
        device=_DEVICE,
        workspace_bytes=_workspace_bytes(1),
    )
    return cache, BF16DecodeEndpoint(loaded.model, cache, method)


def _cache_evidence(
    method: KVCacheMethod,
    cache: Any,
) -> dict[str, Any]:
    breakdown = dict(sorted(method.byte_breakdown(cache).items()))
    allocated = method.allocated_bytes(cache)
    geometry = audit_cache_geometry(cache, num_query_heads=32)
    return {
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "adapter_config_fingerprint": method.config_fingerprint(
            cache.layout_fingerprint()
        ),
        "allocated_cache_bytes": allocated,
        "logical_bf16_bytes": method.logical_bf16_bytes(cache),
        "byte_breakdown": breakdown,
        "byte_breakdown_sums_to_allocated": sum(breakdown.values()) == allocated,
        "cache_geometry": geometry,
    }


def _fixed_eager(
    method: KVCacheMethod,
    loaded: LoadedFrozenModel,
    torch: Any,
) -> dict[str, Any]:
    cache, endpoint = _allocate(
        method,
        loaded,
        capacity=_PREFIX_LENGTH + 1,
    )
    prefix = _deterministic_ids(torch, length=_PREFIX_LENGTH, offset=10_000)
    token = _deterministic_ids(torch, length=1, offset=40_000)
    with torch.inference_mode(), forced_flash_execution():
        endpoint.prefill(prefix)
        cache.prepare_fixed(_PREFIX_LENGTH)
        position = torch.tensor(
            [_PREFIX_LENGTH],
            dtype=torch.long,
            device=_DEVICE,
        )
        rope = endpoint.prepare_position_embeddings(position.unsqueeze(0))
        pointers_before = cache.pointers()
        history_before = cache_history_sha256_untimed(
            cache.keys,
            cache.values,
            historical_length=_PREFIX_LENGTH,
        )
        output = endpoint.decode(token, position, rope)
        torch.cuda.synchronize(device=_DEVICE)
        output_finite = bool(torch.isfinite(output).all().item())
        output_sha256 = tensor_sha256_untimed(output)
        pointers_after = cache.pointers()
        history_after = cache_history_sha256_untimed(
            cache.keys,
            cache.values,
            historical_length=_PREFIX_LENGTH,
        )
    cache_evidence = _cache_evidence(method, cache)
    passed = (
        output_finite
        and pointers_before == pointers_after
        and history_before == history_after
        and cache_evidence["byte_breakdown_sums_to_allocated"] is True
        and cache_evidence["cache_geometry"]["uses_kv_head_geometry"] is True
        and cache_evidence["cache_geometry"]["query_head_storage_detected"] is False
    )
    return {
        "passed": passed,
        "runner": "fixed_l",
        "graph_mode": "eager",
        "batch_size": 1,
        "context_length": _PREFIX_LENGTH,
        "output_steps": 1,
        "output_finite": output_finite,
        "output_sha256": output_sha256,
        "cache_pointers_stable": pointers_before == pointers_after,
        "historical_cache_unchanged": history_before == history_after,
        **cache_evidence,
    }


def _fixed_graph(
    method: KVCacheMethod,
    loaded: LoadedFrozenModel,
    torch: Any,
) -> dict[str, Any]:
    accounting_cache, accounting_endpoint = _allocate(
        method,
        loaded,
        capacity=_PREFIX_LENGTH + 1,
    )
    cache_evidence = _cache_evidence(method, accounting_cache)
    del accounting_endpoint, accounting_cache
    gc.collect()
    torch.cuda.empty_cache()
    prefix = _deterministic_ids(torch, length=_PREFIX_LENGTH, offset=10_000)
    token = _deterministic_ids(torch, length=1, offset=40_000)
    validation = run_graph_harness(method, loaded.model, prefix, token)
    return {
        "passed": validation.passed,
        "runner": "fixed_l",
        "graph_mode": "cuda_graph",
        "batch_size": 1,
        "context_length": _PREFIX_LENGTH,
        "output_steps": 1,
        "graph_validation": validation.to_dict(),
        **cache_evidence,
    }


def _growing_eager(
    method: KVCacheMethod,
    loaded: LoadedFrozenModel,
    torch: Any,
) -> dict[str, Any]:
    cache, endpoint = _allocate(
        method,
        loaded,
        capacity=_PREFIX_LENGTH + _GROWING_STEPS,
    )
    prefix = _deterministic_ids(torch, length=_PREFIX_LENGTH, offset=10_000)
    tokens = _deterministic_ids(
        torch,
        length=_GROWING_STEPS,
        offset=50_000,
    )
    positions = tuple(
        torch.tensor(
            [_PREFIX_LENGTH + step],
            dtype=torch.long,
            device=_DEVICE,
        )
        for step in range(_GROWING_STEPS)
    )
    with torch.inference_mode(), forced_flash_execution():
        endpoint.prefill(prefix)
        cache.prepare_growing(_PREFIX_LENGTH, _GROWING_STEPS)
        rope = tuple(
            endpoint.prepare_position_embeddings(position.unsqueeze(0))
            for position in positions
        )
        pointers_before = cache.pointers()
        history_before = cache_history_sha256_untimed(
            cache.keys,
            cache.values,
            historical_length=_PREFIX_LENGTH,
        )
        outputs = []
        for step in range(_GROWING_STEPS):
            cache.select_growing_step(step)
            outputs.append(
                endpoint.decode(
                    tokens[:, step : step + 1],
                    positions[step],
                    rope[step],
                )
            )
            cache.finish_growing_step()
        torch.cuda.synchronize(device=_DEVICE)
        output_finite = all(
            bool(torch.isfinite(output).all().item()) for output in outputs
        )
        output_sha256 = tuple(
            tensor_sha256_untimed(output) for output in outputs
        )
        pointers_after = cache.pointers()
        history_after = cache_history_sha256_untimed(
            cache.keys,
            cache.values,
            historical_length=_PREFIX_LENGTH,
        )
    cache_evidence = _cache_evidence(method, cache)
    passed = (
        output_finite
        and cache.active_context == _PREFIX_LENGTH + _GROWING_STEPS
        and pointers_before == pointers_after
        and history_before == history_after
        and cache_evidence["byte_breakdown_sums_to_allocated"] is True
        and cache_evidence["cache_geometry"]["uses_kv_head_geometry"] is True
        and cache_evidence["cache_geometry"]["query_head_storage_detected"] is False
    )
    return {
        "passed": passed,
        "runner": "growing_context",
        "graph_mode": "eager",
        "batch_size": 1,
        "context_length": _PREFIX_LENGTH,
        "output_steps": _GROWING_STEPS,
        "active_context": cache.active_context,
        "outputs_finite": output_finite,
        "output_sha256": list(output_sha256),
        "cache_pointers_stable": pointers_before == pointers_after,
        "historical_cache_unchanged": history_before == history_after,
        **cache_evidence,
    }


def _publish(
    *,
    run_id: str,
    record: dict[str, Any],
) -> dict[str, str]:
    root = REPOSITORY_ROOT / "artifacts/phase4_smoke"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_directory = root / run_id
    run_directory.mkdir(mode=0o700)
    raw = canonical_json_bytes(record) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    publish_bytes_no_replace(run_directory / "smoke.json", raw)
    ledger = f"{digest}  smoke.json\n".encode("ascii")
    publish_bytes_no_replace(run_directory / "checksums.sha256", ledger)
    return {
        "run_id": run_id,
        "path": (run_directory / "smoke.json").relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "sha256": digest,
        "checksum_ledger_sha256": hashlib.sha256(ledger).hexdigest(),
    }


def run_phase4_functional_smokes() -> dict[str, Any]:
    """Run exactly the three requested functional smokes and publish evidence."""

    _require_clean_worktree()
    git_sha = _git_head()
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise Phase4SmokeError("CUDA is unavailable")
    loaded = load_frozen_model(device=_DEVICE)
    backend = backend_identity()
    method, method_fingerprint = _build_adapter(loaded, backend)
    common = {
        "schema_version": PHASE4_SMOKE_SCHEMA_VERSION,
        "created_at_utc": None,
        "execution_git_sha": git_sha,
        "method_name": method.name,
        "method_config_id": "bf16",
        "method_config_fingerprint": method_fingerprint.sha256,
        "adapter_version": method.adapter_version,
        "model_identity": {
            "model_id": loaded.identity.model_id,
            "revision": loaded.identity.revision,
            "fingerprint": loaded.receipt.frozen_identity_sha256,
            "load_receipt_sha256": loaded.receipt.receipt_sha256,
        },
        "backend_identity": {
            "backend_id": backend["backend_id"],
            "fingerprint": sha256_hex(canonical_json_bytes(backend)),
        },
        "functional_evidence_only": True,
        "timing_collected": False,
        "formal_timing_claim_created": False,
        "formal_performance_data_created": False,
        "independent_process_replicates_collected": False,
        "profiler_executed": False,
        "quality_benchmark_executed": False,
        "quality_status": "unvalidated",
        "quality_execution": "locked",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "measurement_scope": "native_host_admission",
        "full_scan_state": "closed",
        "gates": {
            "g0": "PASS",
            "g1": "PASS",
            "g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
        },
        "blockers": ["B-009", "B-010"],
    }
    scenarios: tuple[tuple[str, SmokeScenario], ...] = (
        ("fixed-l-eager", _fixed_eager),
        ("fixed-l-cuda-graph", _fixed_graph),
        ("growing-context-eager", _growing_eager),
    )
    references: list[dict[str, str]] = []
    all_passed = True
    for kind, scenario in scenarios:
        run_id = _run_id(kind, git_sha)
        created_at = _created_at()
        try:
            result = scenario(method, loaded, torch)
            status = "PASS" if result["passed"] is True else "FAIL"
            record = {
                **common,
                "run_id": run_id,
                "created_at_utc": created_at,
                "status": status,
                "result": result,
                "failure": None,
            }
        except Exception as error:
            status = "FAIL"
            record = {
                **common,
                "run_id": run_id,
                "created_at_utc": created_at,
                "status": status,
                "result": None,
                "failure": {
                    "error_type": type(error).__name__,
                    "reason": str(error)[:1024],
                },
            }
        all_passed = all_passed and status == "PASS"
        references.append(_publish(run_id=run_id, record=record))
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "schema_version": PHASE4_SMOKE_INDEX_SCHEMA_VERSION,
        "execution_git_sha": git_sha,
        "status": "PASS" if all_passed else "FAIL",
        "timing_collected": False,
        "run_references": references,
    }


def main() -> int:
    index = run_phase4_functional_smokes()
    print(json.dumps(index, sort_keys=True, separators=(",", ":")))
    return 0 if index["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
