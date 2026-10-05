#!/usr/bin/env python3
"""Addendum Task 4 worker: synthetic-cache timing for one (configuration, B, L, replicate).

Amendment Section 6 ("synthetic cache, timing only"): K/V drawn from a seeded
normal distribution are written through the method's normal store /
quantization path instead of model prefill; everything after cache
construction is the Section 2 protocol.  Runs inside the authorized
measurement container exactly like timing_worker.py (same mounts, same
environment, execution repository ec534d99 read-only at /home/rockrock/cmu_paper).

Mechanism.  The measurement is the Full Scan worker itself,
scripts.phase13_pilot._run_worker, entered the way
scripts.phase16_full_scan.run_worker enters it (GPU-snapshot overrides,
_configure_reused_phase13, context-label override, geometry authority,
point inputs, session builder).  The single substitution is the session
builder: the Full Scan's _direct_session patches BF16DecodeEndpoint.prefill
with a callback that runs the frozen model prefill; here the callback does to
the cache what BF16DecodeEndpoint.prefill does, without the model:

    cache.prepare_prefill(L)
    for layer: method.store_prefill(cache, K_layer, V_layer, layer, arange(L))
    cache.complete_prefill()

with K/V ~ N(0, 1) in BF16 from a seeded torch.Generator on the GPU.
phase12._build_phase12_session (cache allocation, endpoint and its RoPE
scratch, fixed position and decode token, RoPE tables, prepare_fixed, setup
warmups, CUDA Graph capture, eager-vs-graph and replay checks, pointer
checks) and everything after it in _run_worker (64 warmup replays,
allocation / kernel-path / finiteness audits, 5 x 256 graph replays timed with
host perf_counter_ns and CUDA events, telemetry, GPU-ownership snapshots) are
the unchanged frozen code.

Synthetic content.  One generator per (layer, K|V), seed
SYNTHETIC_SEED + 2 * layer + {0: K, 1: V}, drawing contiguous
[B, 8, n, 128] BF16 chunks of n <= SYNTHETIC_CHUNK_TOKENS tokens in token
order.  Seeds do not depend on the configuration or replicate, so BF16 and
KIVI at the same (B, L) store the same K/V values (before KIVI quantization).
RoPE is not applied: keys are stored as drawn.  Llama RoPE rotates each
(i, i + 64) channel pair by a position-dependent angle and an i.i.d. N(0, 1)
pair is rotation invariant, so post-RoPE synthetic keys would have exactly the
same distribution; decode kernels are data-independent in any case.
  KIVI: chunks go straight into store_prefill (its token ledgers make
  successive calls on one layer append), so the transient is two chunks.
  BF16: BF16StaticCache.update in prefill mode accepts only the whole
  [B, 8, L, 128] slab of a layer, so chunks are copied into one reused
  layer-sized K and V staging pair (2 x B x 8 x L x 128 x 2 bytes, 1.6 GB at
  B = 3, L = 128K) and stored once per layer.
After each KIVI store_prefill the returned handle is reset exactly as
KIVIMethodAdapter.decode_attention's prefill branch resets it (the prefill
attention itself is skipped, as in tests/graph/test_phase8_kivi_graph.py).

Decode inputs.  Decode token and position follow the Full Scan logical-prefix
recipe (scripts/phase16_logical_prefix.py): the frozen logical-prefix
artifact when (B, L) is a Full Scan grid point, otherwise
deterministic_inputs(), the function that artifact validation itself
compares against.  Position = historical context (131071 at label 131072).

Deviations from the Full Scan session path (also recorded in each result):
  D1 cache contents are synthetic (no embedding / transformer prefill, no
     prefill attention, no BF16 prefix-snapshot restore);
  D2 the prefix witness hashes the synthetic-cache specification instead of
     the logical prefix (same override mechanism as _direct_session);
  D3 caching-allocator state before graph capture differs (no prefill
     activations; BF16 staging blocks freed to the cache, not released);
  D4 only for B outside {1, 2, 4, 8, 16} and only with
     --unadmitted-batch-bypass: (a) no Decision 0039 / Phase 16G geometry
     record exists, so phase13's successor-authority membership check is
     satisfied by an entry whose admission fingerprints are None and
     require_admitted_geometry is not called; (b) KIVIStaticCache.__init__'s
     "B in {1,2,4,8,16}" clause is removed in this process by a recorded
     source transform (all other geometry checks kept); (c) the decode token
     comes from deterministic_inputs(), there being no logical-prefix artifact.
Results are timing-only (cache_source = synthetic_timing_only) and are never
pooled with same-work ratios.
"""

from __future__ import annotations

import __future__ as future_flags
import argparse
from contextlib import contextmanager
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import textwrap
import time
import traceback
import types
from typing import Any, Iterator, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import overrides  # noqa: E402
import timing_worker  # noqa: E402  (install_variant: the shared variant switch)

RESULT_PREFIX = "ADDENDUM_SYNTHETIC_TIMING_RESULT="
SCHEMA = "kvbench-addendum-20261005-synthetic-timing-1.0.0"
RUN_SCHEMA = "kvbench-addendum-20261005-synthetic-process-run-1.0.0"
SPEC_SCHEMA = "kvbench-addendum-20261005-synthetic-cache-spec-1.0.0"
CACHE_SOURCE = "synthetic_timing_only"
CONFIGURATIONS = ("bf16", "k4v4")  # amendment Section 6
SYNTHETIC_SEED = 2026100540
SYNTHETIC_CHUNK_TOKENS = 4096
ADMITTED_BATCHES = (1, 2, 4, 8, 16)  # Decision 0039; checked against phase16_full_scan.BATCH_SIZES

_KIVI_BATCH_CLAUSE = re.compile(r"geometry\[1\] not in \{1, 2, 4, 8, 16\}\s*\n\s*or ")


# ---------------------------------------------------------------- synthetic spec

def synthetic_spec(*, configuration: str, batch: int, historical: int) -> dict[str, Any]:
    return {
        "schema_version": SPEC_SCHEMA,
        "cache_source": CACHE_SOURCE,
        "distribution": "normal(mean=0, std=1)",
        "dtype": "bfloat16",
        "generator": "torch.Generator(device=cuda) Philox, Tensor.normal_",
        "seed_rule": f"{SYNTHETIC_SEED} + 2 * layer + (0 for K, 1 for V)",
        "seed_independent_of_configuration_and_replicate": True,
        "chunk_tokens": SYNTHETIC_CHUNK_TOKENS,
        "chunk_shape": "[B, 8, n, 128] contiguous, token order",
        "rope_applied": False,
        "rope_note": "i.i.d. N(0,1) channel pairs are rotation invariant; RoPE would not change the distribution",
        "store_path": "cache.prepare_prefill(L); method.store_prefill per layer; cache.complete_prefill()",
        "model_prefill_executed": False,
        "prefill_attention_executed": False,
        "configuration": configuration,
        "batch_size": batch,
        "historical_context": historical,
    }


def _release_kivi_prefill_handle(handle: Any) -> None:
    """Same reset as KIVIMethodAdapter.decode_attention's prefill `finally`."""
    handle.prefill_key_states = None
    handle.prefill_value_states = None
    handle.prefill = False


def write_synthetic_cache(endpoint: Any, *, historical: int, family: str) -> dict[str, Any]:
    """Fill endpoint.cache through the method's store path (untimed setup)."""
    import torch

    cache, method = endpoint.cache, endpoint.method
    if endpoint.method_requires_pre_rope_key:
        raise RuntimeError("synthetic cache supports only methods without a pre-RoPE key store")
    if family not in ("bf16", "kivi"):
        raise RuntimeError(f"synthetic cache family {family} is outside Task 4")
    batch, heads, dim = int(cache.batch_size), int(endpoint.num_kv_heads), int(endpoint.head_dim)
    layers, device, chunk = int(endpoint.num_layers), cache.device, SYNTHETIC_CHUNK_TOKENS
    torch.cuda.synchronize(device=device)
    allocated_before = int(torch.cuda.memory_allocated(device=device))
    began = time.monotonic()
    cache.prepare_prefill(historical)  # as BF16DecodeEndpoint.prefill
    positions = torch.arange(historical, dtype=torch.long, device=device)  # its cache_position
    flat_key = torch.empty(batch * heads * chunk * dim, dtype=torch.bfloat16, device=device)
    flat_value = torch.empty_like(flat_key)
    layer_key = layer_value = None
    if family == "bf16":
        layer_key = torch.empty((batch, heads, historical, dim), dtype=torch.bfloat16, device=device)
        layer_value = torch.empty_like(layer_key)
    staging_bytes = sum(int(t.numel()) * 2 for t in (flat_key, flat_value, layer_key, layer_value)
                        if t is not None)
    key_generator = torch.Generator(device=device)
    value_generator = torch.Generator(device=device)
    store_calls = 0
    for layer in range(layers):
        key_generator.manual_seed(SYNTHETIC_SEED + 2 * layer)
        value_generator.manual_seed(SYNTHETIC_SEED + 2 * layer + 1)
        for start in range(0, historical, chunk):
            stop = min(historical, start + chunk)
            count = batch * heads * (stop - start) * dim
            key = flat_key[:count].view(batch, heads, stop - start, dim)
            value = flat_value[:count].view(batch, heads, stop - start, dim)
            key.normal_(0.0, 1.0, generator=key_generator)
            value.normal_(0.0, 1.0, generator=value_generator)
            if family == "bf16":
                layer_key[:, :, start:stop, :].copy_(key)
                layer_value[:, :, start:stop, :].copy_(value)
            else:
                handle, _ = method.store_prefill(cache, key, value, layer, positions[start:stop])
                _release_kivi_prefill_handle(handle)
                store_calls += 1
        if family == "bf16":
            method.store_prefill(cache, layer_key, layer_value, layer, positions)
            store_calls += 1
    cache.complete_prefill()
    torch.cuda.synchronize(device=device)
    seconds = time.monotonic() - began
    peak = int(torch.cuda.max_memory_allocated(device=device))
    del flat_key, flat_value, layer_key, layer_value, positions
    return {
        "store_calls": store_calls,
        "layers": layers,
        "tokens_per_layer": historical,
        "active_context_after": int(cache.active_context),
        "construction_seconds": seconds,
        "staging_bytes": staging_bytes,
        "memory_allocated_before_bytes": allocated_before,
        "process_peak_allocated_bytes_after_construction": peak,
        "peak_scope": "torch.cuda.max_memory_allocated since process start (includes model load)",
    }


# ---------------------------------------------------------------- unadmitted batch

def install_kivi_batch_guard_bypass() -> dict[str, Any]:
    """Drop only KIVIStaticCache.__init__'s admitted-batch clause, in this process.

    The constructor is recompiled from its own source with the single clause
    `geometry[1] not in {1, 2, 4, 8, 16} or` removed; H_Q = 32, H_KV = 8,
    D = 128 and every other check stay.  The unmodified source is recompiled
    the same way first and must reproduce the loaded bytecode exactly.
    """
    import kvbench.runtime.kivi_cache as module

    cls = module.KIVIStaticCache
    original = cls.__init__
    if getattr(original, "_addendum_batch_clause_removed", False):
        raise RuntimeError("KIVI batch-clause bypass installed twice")
    source = textwrap.dedent(inspect.getsource(original))
    patched, count = _KIVI_BATCH_CLAUSE.subn("", source)
    if count != 1:
        raise RuntimeError(f"KIVI admitted-batch clause found {count} times; refusing to patch")

    def compiled(text: str, label: str) -> Any:
        namespace: dict[str, Any] = {}
        code = compile(text, f"<addendum synthetic_worker: KIVIStaticCache.__init__ {label}>", "exec",
                       flags=future_flags.annotations.compiler_flag, dont_inherit=True)
        exec(code, module.__dict__, namespace)  # module globals: _torch, KIVI_* constants
        return namespace["__init__"]

    replay = compiled(source, "unmodified replay")
    if (replay.__code__.co_code != original.__code__.co_code
            or replay.__code__.co_names != original.__code__.co_names
            or replay.__code__.co_varnames != original.__code__.co_varnames
            or replay.__code__.co_consts != original.__code__.co_consts):
        raise RuntimeError("recompiled KIVIStaticCache.__init__ does not reproduce the loaded bytecode")
    replacement = compiled(patched, "without the admitted-batch clause")
    replacement.__qualname__ = original.__qualname__
    replacement.__module__ = original.__module__
    replacement._addendum_batch_clause_removed = True
    cls.__init__ = replacement
    return {
        "bypass": "kivi_cache_admitted_batch_clause",
        "file": module.__file__,
        "file_sha256": common.sha256_file(Path(module.__file__)),
        "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_source_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "unmodified_recompile_reproduces_bytecode": True,
        "diff": "".join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True),
                                             "KIVIStaticCache.__init__", "patched", n=1)),
        "note": "kivi_cache.py on disk is unchanged, so layout_fingerprint's implementation_sha256 "
                "names the file, not the in-memory constructor",
    }


def unadmitted_authority(predecessor: Mapping[str, Any], batch: int) -> dict[str, Any]:
    """Phase 13 successor authority extended by B with no admission fingerprints."""
    authority = json.loads(json.dumps(predecessor))
    for family_record in authority["families"].values():
        family_record["batch_sizes"] = sorted(set(family_record["batch_sizes"]) | {batch})
        family_record["addendum_unadmitted_batch"] = batch
        for configuration in family_record["configurations"]:
            key = f"{configuration}/B{batch}"
            family_record["adapter_config_fingerprints_l128"][key] = None
            family_record["cache_layout_fingerprints_l128"][key] = None
    return authority


# ---------------------------------------------------------------- inputs

def logical_inputs(p16: Any, logical_prefix: Any, *, batch: int, label: int, historical: int
                   ) -> tuple[Any, Any, dict[str, Any]]:
    catalog = p16.load_logical_prefix_catalog(common.FAMILY / "logical-prefixes", validate_artifacts=False)
    item = catalog.get((batch, label))
    identity = {"batch_size": batch, "configured_context_label": label, "historical_context": historical}
    if item is not None:
        manifest, tokens, decode = logical_prefix.validate_logical_prefix_artifact(
            Path(str(item["artifact_root"])),
            expected={"batch_size": batch, "configured_context_label": label,
                      "actual_historical_context": historical},
            load_tensors=True)
        meta = {"source": "frozen_logical_prefix_artifact",
                "logical_prefix_id": manifest["logical_prefix_id"],
                "token_checksum": manifest["token_tensor"]["sha256"],
                "decode_token_checksum": manifest["current_decode_token"]["sha256"],
                "artifact_root": str(item["artifact_root"])}
    else:
        tokens, decode, _ = logical_prefix.deterministic_inputs(**identity)
        meta = {"source": "phase16_logical_prefix.deterministic_inputs",
                "logical_prefix_id": logical_prefix.logical_prefix_id(**identity),
                "token_checksum": logical_prefix._tensor_sha256(tokens),
                "decode_token_checksum": logical_prefix._tensor_sha256(decode),
                "artifact_root": None}
    meta.update({"schema_version": logical_prefix.LOGICAL_PREFIX_SCHEMA, "batch_size": batch,
                 "configured_context_label": label, "actual_historical_context": historical,
                 "decode_position": historical, "decode_token_ids": [int(v) for v in decode.reshape(-1)],
                 "prefix_tokens_used_for": "shape/device only (no model prefill)"})
    return tokens, decode, meta


# ---------------------------------------------------------------- worker

def synthetic_builder(*, p16: Any, phase12: Any, phase13: Any, family: str,
                      logical: Mapping[str, Any], sink: dict[str, Any]) -> Any:
    """Replacement for phase13._build_restored_session (cf. phase16_full_scan._direct_session)."""

    def builder(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        import torch
        from kvbench.runtime.backend import forced_flash_execution

        operation = kwargs["operation"]
        spec = synthetic_spec(configuration=operation.configuration, batch=operation.batch_size,
                              historical=operation.historical_context)
        witness = p16._canonical_sha256({
            "schema_version": "kvbench-addendum-20261005-synthetic-prefix-witness-1.0.0",
            "method_config_fingerprint": p16.CONFIG_FINGERPRINTS[operation.configuration],
            "input_recipe_sha256": phase12.PHASE12_INPUT_RECIPE_SHA256,
            "decode_token_checksum": logical["decode_token_checksum"],
            "synthetic_cache": spec,
        })

        def callback(endpoint: Any, input_ids: Any, original: Any) -> None:
            del original  # the model prefill is what Task 4 replaces
            if tuple(input_ids.shape) != (operation.batch_size, operation.historical_context):
                raise RuntimeError("synthetic prefix geometry differs")
            if "construction" in sink:
                raise RuntimeError("synthetic cache construction ran twice")
            sink["construction"] = write_synthetic_cache(
                endpoint, historical=operation.historical_context, family=family)
            if hasattr(endpoint.cache, "history_sha256"):
                endpoint.cache.history_sha256 = types.MethodType(
                    lambda self, historical_length: witness, endpoint.cache)
            return None

        with (torch.inference_mode(), forced_flash_execution(),
              phase13._restored_prefix_hash_overrides(witness),
              phase13._patched_endpoint_prefill(callback)):
            session = phase12._build_phase12_session(
                loaded=kwargs["loaded"], operation_key=operation,
                prefix_input_ids=kwargs["prefix"], decode_input_ids=kwargs["decode"])
        if "construction" not in sink:
            raise RuntimeError("synthetic cache construction did not run")
        phase13._bind_session_prefix_witness(session, witness)
        receipt = {
            "schema_version": "kvbench-addendum-20261005-synthetic-build-receipt-1.0.0",
            "mode": CACHE_SOURCE,
            "cache_source": CACHE_SOURCE,
            "witness_sha256": witness,
            "synthetic_cache_spec": spec,
            "construction": sink["construction"],
            "logical_input": dict(logical),
            "method_config_fingerprint": p16.CONFIG_FINGERPRINTS[operation.configuration],
            "cache_layout_fingerprint": session.cache_layout_fingerprint(),
            "batch_size": operation.batch_size,
            "historical_context": operation.historical_context,
            "active_length": session.active_context,
            "allocation_bytes": session.method_cache_accounting()["allocated_bytes"],
            "validation_output_checksum": session.graph_evidence["second_replay_checksum"],
            "validation_decode_before_graph_capture": True,
            "build_status": "PASS",
            "fresh_target_allocation": True,
            "snapshot_persisted": False,
            "cache_build_outside_timing": True,
        }
        sink["receipt"] = receipt
        return session, receipt

    return builder


@contextmanager
def synthetic_overrides(*, p16: Any, phase12: Any, phase13: Any, family: str, batch: int,
                        historical: int, admitted: bool, tokens: Any, decode: Any,
                        logical: Mapping[str, Any], sink: dict[str, Any]) -> Iterator[None]:
    """phase16_full_scan._worker_overrides with the synthetic builder."""
    p16._configure_reused_phase13()
    original_builder = phase13._build_restored_session
    original_authority = phase13._phase13b_successor_authority
    original_inputs = phase13._point_inputs
    expected_batch, expected_historical = batch, historical

    def point_inputs(*, batch: int, historical: int, device: Any) -> tuple[Any, Any]:
        import torch

        if batch != expected_batch or historical != expected_historical:
            raise RuntimeError("worker logical input geometry differs")
        return (tokens.to(device=device, dtype=torch.long, copy=True),
                decode.to(device=device, dtype=torch.long, copy=True))

    predecessor = original_authority()
    merged = (p16._geometry_authority_for_batch(batch, predecessor=predecessor) if admitted
              else unadmitted_authority(predecessor, batch))
    phase13._build_restored_session = synthetic_builder(
        p16=p16, phase12=phase12, phase13=phase13, family=family, logical=logical, sink=sink)
    phase13._phase13b_successor_authority = lambda: merged
    phase13._point_inputs = point_inputs
    try:
        yield
    finally:
        phase13._build_restored_session = original_builder
        phase13._phase13b_successor_authority = original_authority
        phase13._point_inputs = original_inputs


def run_synthetic_worker(*, run_id: str, record: Mapping[str, Any], run_root: Path,
                         entry: Mapping[str, Any], tokens: Any, decode: Any,
                         logical: Mapping[str, Any], admitted: bool, sink: dict[str, Any]
                         ) -> dict[str, Any]:
    """Both phase16_full_scan.run_worker definitions, with synthetic_overrides."""
    import scripts.phase12_unified_admission as phase12
    import scripts.phase13_pilot as phase13
    import scripts.phase16_full_scan as p16
    from kvbench.schema.phase16g import (
        PHASE16G_GEOMETRY_REPORT_PATH, PHASE16G_GEOMETRY_REPORT_SHA256, PHASE16G_PREFIX_SCHEMA,
        require_admitted_geometry)

    configuration = str(record["method_config_id"])
    batch = int(record["batch_size"])
    label = int(record["context_label"])
    family = phase12._method_family(configuration)
    with p16._worker_snapshot_overrides(run_root):
        with (synthetic_overrides(p16=p16, phase12=phase12, phase13=phase13, family=family,
                                  batch=batch, historical=int(record["historical_context"]),
                                  admitted=admitted, tokens=tokens, decode=decode,
                                  logical=logical, sink=sink),
              p16._worker_context_label_override(label)):
            payload = phase13._run_worker(
                run_id=run_id, configuration=configuration, batch=batch, context_label=label,
                replicate_index=int(record["replicate_index"]), order_index=int(record["order_index"]),
                git_sha=common.EXECUTION_SHA, run_artifact_root=run_root,
                prefix_state_root=run_root / "no-prefix-snapshot", prefix_state_sha256="0" * 64)
        report = p16._strict_json(p16.PHASE16G_REPORT_PATH)
        geometry_key = (require_admitted_geometry(report, configuration=configuration, batch_size=batch)
                        if admitted else None)
        payload.update({
            "schema_version": RUN_SCHEMA,
            "run_kind": "timing",
            "cache_source": CACHE_SOURCE,
            "pilot_only": False,
            "claim_eligibility": "timing_only_synthetic_cache",
            "quality_status": "not_applicable_synthetic_cache",
            "performance_claim_eligible": False,
            "phase16g_geometry_binding": {
                "decision": "0039" if admitted else None,
                "geometry_key": geometry_key,
                "admitted": admitted,
                "report_path": PHASE16G_GEOMETRY_REPORT_PATH,
                "report_sha256": PHASE16G_GEOMETRY_REPORT_SHA256,
                "prefix_schema": PHASE16G_PREFIX_SCHEMA,
            },
            "synthetic_logical_input_binding": dict(logical),
            "prefix_source": dict(entry),
            "r_hbm": None,
        })
        payload["kernel_path_fingerprint"] = p16._canonical_sha256({
            "phase13_worker_kernel_path": payload["kernel_path_fingerprint"],
            "phase16g_geometry": payload["phase16g_geometry_binding"],
        })
    snapshot = payload.get("gpu_process_owned_after_measurement")
    unavailable = bool(isinstance(snapshot, Mapping)
                       and snapshot.get("phase16_snapshot_state") == "query_failed")
    payload["postflight_snapshot_unavailable"] = unavailable
    payload["fitting_eligible"] = not unavailable
    if unavailable:
        payload["gpu_exclusive"] = None
    return payload


def run(args: argparse.Namespace) -> dict:
    variant = json.loads(args.variant)
    configuration, batch, label = args.configuration, args.batch_size, args.context_label
    if configuration not in CONFIGURATIONS:
        raise RuntimeError(f"Task 4 synthetic timing covers {CONFIGURATIONS} only")
    if variant and configuration == "bf16":
        raise RuntimeError("BF16 takes no implementation variant")
    state = timing_worker.install_variant(variant)
    import scripts.phase12_unified_admission as phase12
    import scripts.phase13_pilot as phase13
    import scripts.phase16_full_scan as p16
    import scripts.phase16_logical_prefix as logical_prefix

    if tuple(p16.BATCH_SIZES) != ADMITTED_BATCHES:
        raise RuntimeError("Full Scan admitted batch set differs")
    historical = p16.actual_historical_context(label)
    family = phase12._method_family(configuration)
    admitted = batch in ADMITTED_BATCHES
    gemv_offset = None
    if family == "kivi":
        # bgemv{2,4}_kernel_outer_dim index packed K/V with the int product
        # _batch_idx * OC * IC (_batch_idx < 8B): a real kernel limit, unlike the
        # admitted-batch clause.  At L = 128K it allows B <= 16.
        capacity = historical + 1
        gemv_offset = (8 * batch - 1) * 128 * max((capacity // 32) * 32, capacity - 32)
        if gemv_offset >= 2**31:
            raise RuntimeError(f"KIVI GEMV int32 offset {gemv_offset} overflows at B={batch}, L={historical}")
    bypasses: list[dict[str, Any]] = []
    if not admitted:
        if not args.unadmitted_batch_bypass:
            raise RuntimeError(f"B={batch} is outside the admitted geometry {ADMITTED_BATCHES}; "
                               "Task 4 B_max points need --unadmitted-batch-bypass")
        bypasses.append({"bypass": "phase16g_geometry_admission",
                         "detail": "phase13 successor authority extended with None fingerprints; "
                                   "require_admitted_geometry not called"})
        if family == "kivi":
            bypasses.append(install_kivi_batch_guard_bypass())
    record = {
        "method_config_id": configuration,
        "batch_size": batch,
        "context_label": label,
        "historical_context": historical,
        "replicate_index": args.replicate_index,
        "order_index": args.order_index,
    }
    tokens, decode, logical = logical_inputs(p16, logical_prefix, batch=batch, label=label,
                                             historical=historical)
    entry = {
        "kind": "synthetic_cache",
        "cache_source": CACHE_SOURCE,
        "logical_prefix": dict(logical),
        "restore_mode": "synthetic_store_path",
        "optional_snapshot": None,
        "schema_version": None,
        "snapshot_root": None,
        "state_file_sha256": None,
        "source": "synthetic_normal_kv_through_method_store_path",
        "canonical_scientific_evidence": "none_timing_only",
        "materialized_cache_snapshot_required": False,
    }
    run_root = Path(args.output_dir) / "run"
    run_root.mkdir()
    (run_root / "stage-progress").mkdir()
    common.write_new(run_root / "prefix-entry.json", common.json_text(entry))
    common.write_new(run_root / "worker-record.json", common.json_text(record))
    sink: dict[str, Any] = {}
    payload = run_synthetic_worker(run_id=args.run_id, record=record, run_root=run_root, entry=entry,
                                   tokens=tokens, decode=decode, logical=logical, admitted=admitted,
                                   sink=sink)
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    common.write_new(run_root / "result.json", payload_text + "\n")
    wall_median, wall_batches = common.host_wall_process_median_ms(payload)
    allocated = int(payload["runner"]["cache_accounting"]["allocated_bytes"])
    formula = int(phase13.cache_allocated_bytes(configuration, batch, historical + 1))
    checks = {key: payload.get(key) for key in (
        "finite_output", "no_backend_fallback", "allocation_stable", "kernel_path_stable",
        "gpu_exclusive", "fitting_eligible")}
    checks["cache_bytes_match_formula"] = allocated == formula
    memory = payload["runner"].get("memory_evidence") or {}
    return {
        "schema_version": SCHEMA,
        "status": "completed",
        "addendum_id": common.ADDENDUM_ID,
        "run_id": args.run_id,
        "task": args.task,
        "cache_source": CACHE_SOURCE,
        "method_config_id": configuration,
        "batch_size": batch,
        "context_label": label,
        "historical_context": historical,
        "replicate_index": args.replicate_index,
        "order_index": args.order_index,
        "variant": variant,
        "variant_state": overrides.describe(state),
        "latency_basis": "host_wall",
        "host_wall_process_median_ms": wall_median,
        "host_wall_batch_ms_per_op": wall_batches,
        "tokens_per_second_process": batch * 1000.0 / wall_median,
        "cuda_process_median_ms": payload.get("process_median_ms"),
        "warmup_replays": payload.get("warmup_replays"),
        "measured_steps": payload.get("measured_steps"),
        "measured_batches": payload.get("measured_batches"),
        "checks": checks,
        "kernel_count": payload.get("kernel_count"),
        "output_checksum": payload.get("output_checksum"),
        "sm_clock_min_mhz": payload.get("sm_clock_min_mhz"),
        "sm_clock_max_mhz": payload.get("sm_clock_max_mhz"),
        "prefix_source": entry["source"],
        "synthetic_cache": {"spec": sink["receipt"]["synthetic_cache_spec"],
                            "construction": sink["construction"],
                            "witness_sha256": sink["receipt"]["witness_sha256"]},
        "logical_input": {k: v for k, v in logical.items() if k != "decode_token_ids"},
        "geometry_admission": {"admitted": admitted, "admitted_batches": list(ADMITTED_BATCHES),
                               "bypasses": bypasses, "kivi_gemv_int32_offset_max": gemv_offset},
        "cache_allocated_bytes": allocated,
        "cache_allocated_bytes_formula": formula,
        "memory_observed": {name: {k: (memory.get(name) or {}).get(k) for k in (
            "allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")}
            for name in ("model_baseline", "post_cache_allocation", "post_setup",
                         "timing_before", "timing_after")},
        "deviations_from_full_scan_session": ["D1 synthetic cache contents", "D2 synthetic prefix witness",
                                              "D3 allocator state before capture"]
                                             + ([] if admitted else ["D4 unadmitted batch bypass"]),
        "full_scan_payload_path": "run/result.json",
        "full_scan_payload_sha256": hashlib.sha256((payload_text + "\n").encode()).hexdigest(),
        "execution_git_sha": common.EXECUTION_SHA,
        "addendum_git_sha": args.addendum_commit,
        "container_digest": os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST"),
        "worker_sha256": common.sha256_file(Path(__file__)),
        "timing_worker_sha256": common.sha256_file(Path(timing_worker.__file__)),
        "overrides_sha256": common.sha256_file(Path(overrides.__file__)),
        "common_sha256": common.sha256_file(Path(common.__file__)),
        "finished_at_utc": common.utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--configuration", required=True, choices=CONFIGURATIONS)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-label", type=int, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--order-index", type=int, required=True)
    parser.add_argument("--variant", default="{}")
    parser.add_argument("--unadmitted-batch-bypass", action="store_true",
                        help="allow B outside {1,2,4,8,16} (deviation D4, recorded in the result)")
    parser.add_argument("--addendum-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    code = 0
    try:
        result = run(args)
    except BaseException as error:  # the driver records the failure and continues
        result = {"schema_version": SCHEMA, "status": "failed", "run_id": args.run_id,
                  "cache_source": CACHE_SOURCE,
                  "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "finished_at_utc": common.utc_now()}
        code = 3
    text = json.dumps(result, sort_keys=True, default=str)
    common.write_new(Path(args.output_dir) / "worker_result.json", text + "\n")
    sys.stdout.write(RESULT_PREFIX + text + "\n")
    sys.stdout.flush()
    os._exit(code)  # same exit path as the Full Scan / Phase 15 workers


if __name__ == "__main__":
    main()
