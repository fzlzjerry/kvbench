"""Pilot-only deterministic prefix-state snapshot and restore helpers.

The format contains only historical cache state.  It deliberately excludes
model weights, decode workspaces, Graph objects, and live CUDA pointers.  A
timing worker always allocates a new adapter-owned cache and copies this state
into those caller-owned buffers before Graph capture and before timing.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PREFIX_STATE_SCHEMA = "kvbench-phase13-prefix-state-2.0.0"
PREFIX_STATE_SCHEMA_V3 = "kvbench-phase16g-prefix-state-3.0.0"
PREFIX_STATE_SCHEMAS = frozenset({PREFIX_STATE_SCHEMA, PREFIX_STATE_SCHEMA_V3})
PREFIX_BATCH_REUSE_POLICY = "exact_target_batch_only"
PREFIX_STATE_FILE = "state.safetensors"
PREFIX_STATE_MANIFEST = "manifest.json"
PREFIX_STATE_COMPLETE = "COMPLETE"
PREFIX_TOKEN_IDS_NAME = "__prefix_token_ids"
PREFIX_POSITIONS_NAME = "__prefix_positions"
PREFIX_INPUT_NAMES = frozenset({PREFIX_TOKEN_IDS_NAME, PREFIX_POSITIONS_NAME})
_TENSOR_HASH_CHUNK_BYTES = 64 * 1024 * 1024


class Phase13PrefixStateError(RuntimeError):
    """A prefix snapshot is incomplete, incompatible, or altered."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, message: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Phase13PrefixStateError(message)
    return value


def _tensor_spec(
    tensors: Mapping[str, Any],
    batch_axes: Mapping[str, int | None],
    *,
    include_checksums: bool = False,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, tensor in sorted(tensors.items()):
        item = {
            "name": name,
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype),
            "batch_axis": batch_axes[name],
        }
        if include_checksums:
            item["tensor_sha256"] = _tensor_content_sha256(tensor)
        result.append(item)
    return result


def _tensor_content_sha256(tensor: Any) -> str:
    """Hash tensor values in logical row-major order with bounded host copies."""

    import torch

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        byte_count = int(value.numel()) * int(value.element_size())
        if byte_count <= _TENSOR_HASH_CHUNK_BYTES or value.ndim == 0:
            cpu = value.detach().to(device="cpu", copy=True).contiguous()
            digest.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
            return
        if int(value.shape[0]) <= 1:
            flattened = value.reshape(-1)
            elements = max(1, _TENSOR_HASH_CHUNK_BYTES // int(value.element_size()))
            for start in range(0, int(flattened.numel()), elements):
                update(flattened[start : start + elements])
            return
        for index in range(int(value.shape[0])):
            update(value[index])

    update(tensor)
    return digest.hexdigest()


def _identity_payload(value: Mapping[str, Any], label: str) -> dict[str, str]:
    expected = {"id", "revision"}
    if set(value) != expected or any(
        not isinstance(value[key], str) or not value[key].strip()
        for key in expected
    ):
        raise Phase13PrefixStateError(f"prefix {label} identity differs")
    return {key: str(value[key]) for key in sorted(expected)}


def _require_positive_batch(value: Any, message: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Phase13PrefixStateError(message)
    return value


def _bf16_views(cache: Any, historical: int) -> tuple[dict[str, Any], dict[str, int | None]]:
    tensors = {
        "keys": cache.keys[:, :, :, :historical, :],
        "values": cache.values[:, :, :, :historical, :],
    }
    return tensors, {name: 1 for name in tensors}


def _turboquant_views(
    cache: Any, historical: int
) -> tuple[dict[str, Any], dict[str, int | None]]:
    packed = cache.packed_cache.reshape(
        len(cache.compressed_layers),
        cache.batch_size,
        cache.block_count,
        cache.block_size,
        cache.num_kv_heads,
        cache.slot_size,
    )
    tensors = {
        "packed_cache": packed,
        "bf16_keys": cache.bf16_cache.keys[:, :, :, :historical, :],
        "bf16_values": cache.bf16_cache.values[:, :, :, :historical, :],
    }
    return tensors, {name: 1 for name in tensors}


def _kivi_views(cache: Any) -> tuple[dict[str, Any], dict[str, int | None]]:
    batch_names = (
        "packed_key_history",
        "packed_value_history",
        "key_scales",
        "key_minimums",
        "value_scales",
        "value_minimums",
        "key_residual",
        "value_residual_ring",
    )
    ledger_names = (
        "key_history_token_indices",
        "key_residual_token_indices",
        "value_history_token_indices",
        "value_residual_token_indices",
    )
    tensors = {
        name: getattr(cache, name) for name in (*batch_names, *ledger_names)
    }
    axes = {name: 1 for name in batch_names}
    axes.update({name: None for name in ledger_names})
    return tensors, axes


def _kvquant_views(cache: Any) -> tuple[dict[str, Any], dict[str, int | None]]:
    axis_one = (
        "packed_key_cache",
        "packed_value_cache",
        "value_lookup_cache",
        "key_sparse_values",
        "key_sparse_indices",
        "value_sparse_values",
        "value_sparse_indices",
        "key_active_counts",
        "value_active_counts",
        "sink_key",
        "sink_value",
    )
    axis_zero = (
        "value_store_lower_bounds",
        "value_store_upper_bounds",
    )
    tensors = {name: getattr(cache, name) for name in (*axis_one, *axis_zero)}
    axes = {name: 1 for name in axis_one}
    axes.update({name: 0 for name in axis_zero})
    return tensors, axes


def prefix_state_views(
    cache: Any, *, family: str, historical: int
) -> tuple[dict[str, Any], dict[str, int | None]]:
    """Return only state written by prefill and needed by fixed-L decode."""

    if not isinstance(historical, int) or isinstance(historical, bool) or historical <= 0:
        raise Phase13PrefixStateError("historical context is invalid")
    if historical >= int(cache.capacity):
        raise Phase13PrefixStateError("prefix snapshot requires one scratch slot")
    if family == "bf16":
        return _bf16_views(cache, historical)
    if family == "turboquant":
        return _turboquant_views(cache, historical)
    if family == "kivi":
        return _kivi_views(cache)
    if family == "kvquant":
        return _kvquant_views(cache)
    raise Phase13PrefixStateError("prefix snapshot method family is unsupported")


def _lifecycle(cache: Any, family: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "active_context": int(cache.active_context),
        "mode": str(cache.mode),
    }
    if family == "kivi":
        payload.update(
            {
                "key_history_counts": list(cache._key_history_counts),
                "key_residual_counts": list(cache._key_residual_counts),
                "value_history_counts": list(cache._value_history_counts),
                "value_residual_counts": list(cache._value_residual_counts),
                "value_residual_heads": list(cache._value_residual_heads),
                "fixed_scratch_tokens": list(cache._fixed_scratch_tokens),
            }
        )
    elif family == "kvquant":
        payload["known_key_active_entries"] = cache._known_key_active_entries
    return payload


def _restore_lifecycle(cache: Any, family: str, lifecycle: Mapping[str, Any]) -> None:
    if family == "kivi":
        fields = {
            "_key_history_counts": "key_history_counts",
            "_key_residual_counts": "key_residual_counts",
            "_value_history_counts": "value_history_counts",
            "_value_residual_counts": "value_residual_counts",
            "_value_residual_heads": "value_residual_heads",
            "_fixed_scratch_tokens": "fixed_scratch_tokens",
        }
        for attribute, key in fields.items():
            values = lifecycle.get(key)
            if (
                not isinstance(values, list)
                or len(values) != int(cache.num_layers)
                or any(not isinstance(item, int) or isinstance(item, bool) for item in values)
            ):
                raise Phase13PrefixStateError("KIVI prefix lifecycle differs")
            getattr(cache, attribute)[:] = values
    elif family == "kvquant":
        value = lifecycle.get("known_key_active_entries")
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise Phase13PrefixStateError("KVQuant prefix lifecycle differs")
        cache._known_key_active_entries = value


def save_prefix_state(
    *,
    cache: Any,
    family: str,
    configuration: str,
    historical: int,
    source_batch: int,
    method_config_fingerprint: str,
    output: Path,
    authority: Mapping[str, Any],
    schema_version: str = PREFIX_STATE_SCHEMA,
    prefix_token_ids: Any | None = None,
    prefix_positions: Any | None = None,
    model_identity: Mapping[str, Any] | None = None,
    tokenizer_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one safe-format snapshot after a direct untimed prefill."""

    from safetensors.torch import save_file

    if output.exists() or output.is_symlink():
        raise Phase13PrefixStateError("prefix state output already exists")
    if int(cache.batch_size) != source_batch or int(cache.active_context) != historical:
        raise Phase13PrefixStateError("direct prefix cache geometry differs")
    if str(cache.mode) != "ready":
        raise Phase13PrefixStateError("direct prefix cache is not ready")
    source_batch = _require_positive_batch(
        source_batch,
        "direct prefix batch differs",
    )
    if schema_version not in PREFIX_STATE_SCHEMAS:
        raise Phase13PrefixStateError("direct prefix schema differs")
    if schema_version == PREFIX_STATE_SCHEMA and source_batch not in {1, 4, 8}:
        raise Phase13PrefixStateError("direct prefix batch differs")
    fingerprint = _require_sha256(
        method_config_fingerprint,
        "prefix method fingerprint differs",
    )
    layout_fingerprint = _require_sha256(
        cache.layout_fingerprint(),
        "prefix layout fingerprint differs",
    )
    tensors, axes = prefix_state_views(cache, family=family, historical=historical)
    cpu_tensors = {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in tensors.items()
    }
    input_tensors: dict[str, Any] = {}
    input_axes: dict[str, int | None] = {}
    normalized_model_identity: dict[str, str] | None = None
    normalized_tokenizer_identity: dict[str, str] | None = None
    if schema_version == PREFIX_STATE_SCHEMA_V3:
        if prefix_token_ids is None or prefix_positions is None:
            raise Phase13PrefixStateError("prefix input tensors are absent")
        if (
            tuple(prefix_token_ids.shape) != (source_batch, historical)
            or tuple(prefix_positions.shape) != (source_batch, historical)
        ):
            raise Phase13PrefixStateError("prefix input batch geometry differs")
        if str(prefix_token_ids.dtype) != "torch.int64" or str(
            prefix_positions.dtype
        ) != "torch.int64":
            raise Phase13PrefixStateError("prefix input dtype differs")
        if model_identity is None or tokenizer_identity is None:
            raise Phase13PrefixStateError("prefix model or tokenizer identity is absent")
        normalized_model_identity = _identity_payload(model_identity, "model")
        normalized_tokenizer_identity = _identity_payload(
            tokenizer_identity,
            "tokenizer",
        )
        input_tensors = {
            PREFIX_TOKEN_IDS_NAME: prefix_token_ids.detach()
            .to(device="cpu", copy=True)
            .contiguous(),
            PREFIX_POSITIONS_NAME: prefix_positions.detach()
            .to(device="cpu", copy=True)
            .contiguous(),
        }
        input_axes = {name: 0 for name in input_tensors}
    output.mkdir(parents=False)
    state_path = output / PREFIX_STATE_FILE
    save_file({**cpu_tensors, **input_tensors}, str(state_path))
    state_sha256 = _sha256_file(state_path)
    lifecycle = _lifecycle(cache, family)
    manifest = {
        "schema_version": schema_version,
        "configuration": configuration,
        "family": family,
        "historical_context": historical,
        "capacity": int(cache.capacity),
        "source_batch": source_batch,
        "snapshot_key": {
            "configuration": configuration,
            "batch_size": source_batch,
            "historical_context": historical,
        },
        "batch_reuse_policy": PREFIX_BATCH_REUSE_POLICY,
        "method_config_fingerprint": fingerprint,
        "source_layout_fingerprint": layout_fingerprint,
        "state_file": PREFIX_STATE_FILE,
        "state_file_bytes": state_path.stat().st_size,
        "state_file_sha256": state_sha256,
        "tensors": _tensor_spec(
            cpu_tensors,
            axes,
            include_checksums=schema_version == PREFIX_STATE_SCHEMA_V3,
        ),
        "lifecycle": lifecycle,
        "authority": dict(authority),
        "runtime_prefix_sharing": False,
        "fresh_target_allocation_required": True,
        "restore_outside_timing": True,
    }
    if schema_version == PREFIX_STATE_SCHEMA_V3:
        manifest.update(
            {
                "batch_size": source_batch,
                "input_tensors": _tensor_spec(
                    input_tensors,
                    input_axes,
                    include_checksums=True,
                ),
                "model_identity": normalized_model_identity,
                "model_identity_sha256": hashlib.sha256(
                    _canonical_bytes(normalized_model_identity or {})
                ).hexdigest(),
                "tokenizer_identity": normalized_tokenizer_identity,
                "tokenizer_identity_sha256": hashlib.sha256(
                    _canonical_bytes(normalized_tokenizer_identity or {})
                ).hexdigest(),
                "prefix_inputs_stored": True,
                "prefix_loader_decides_admission": False,
            }
        )
    (output / PREFIX_STATE_MANIFEST).write_bytes(_canonical_bytes(manifest))
    (output / PREFIX_STATE_COMPLETE).write_text(
        state_sha256 + "\n", encoding="ascii"
    )
    return manifest


def validate_prefix_state(
    root: Path,
    *,
    configuration: str | None = None,
    family: str | None = None,
    batch: int | None = None,
    historical: int | None = None,
    method_config_fingerprint: str | None = None,
    model_identity: Mapping[str, Any] | None = None,
    tokenizer_identity: Mapping[str, Any] | None = None,
    verify_state_bytes: bool = True,
) -> dict[str, Any]:
    """Validate the manifest and, once per catalog, its complete state bytes."""

    if not root.is_dir() or root.is_symlink():
        raise Phase13PrefixStateError("prefix state root is invalid")
    expected_names = {PREFIX_STATE_FILE, PREFIX_STATE_MANIFEST, PREFIX_STATE_COMPLETE}
    observed_names = {item.name for item in root.iterdir()}
    if observed_names != expected_names:
        raise Phase13PrefixStateError("prefix state file set differs")
    try:
        manifest = json.loads((root / PREFIX_STATE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13PrefixStateError("prefix state manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") not in PREFIX_STATE_SCHEMAS
    ):
        raise Phase13PrefixStateError("prefix state schema differs")
    schema_version = str(manifest["schema_version"])
    if configuration is not None and manifest.get("configuration") != configuration:
        raise Phase13PrefixStateError("prefix state configuration differs")
    if family is not None and manifest.get("family") != family:
        raise Phase13PrefixStateError("prefix state family differs")
    if historical is not None and manifest.get("historical_context") != historical:
        raise Phase13PrefixStateError("prefix state context differs")
    state_path = root / PREFIX_STATE_FILE
    if (
        state_path.stat().st_size != manifest.get("state_file_bytes")
        or (root / PREFIX_STATE_COMPLETE).read_text(encoding="ascii")
        != str(manifest.get("state_file_sha256")) + "\n"
    ):
        raise Phase13PrefixStateError("prefix state size or COMPLETE differs")
    if verify_state_bytes and _sha256_file(state_path) != manifest.get("state_file_sha256"):
        raise Phase13PrefixStateError("prefix state checksum differs")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise Phase13PrefixStateError("prefix state tensor inventory differs")
    names = [item.get("name") for item in tensors if isinstance(item, dict)]
    if len(names) != len(tensors) or len(set(names)) != len(names):
        raise Phase13PrefixStateError("prefix state tensor names differ")
    source_batch = _require_positive_batch(
        manifest.get("source_batch"),
        "prefix state source batch differs",
    )
    if schema_version == PREFIX_STATE_SCHEMA and source_batch not in {1, 4, 8}:
        raise Phase13PrefixStateError("prefix state source batch differs")
    if batch is not None and source_batch != batch:
        raise Phase13PrefixStateError("prefix state exact batch differs")
    if manifest.get("batch_reuse_policy") != PREFIX_BATCH_REUSE_POLICY:
        raise Phase13PrefixStateError("prefix state batch reuse policy differs")
    fingerprint = _require_sha256(
        manifest.get("method_config_fingerprint"),
        "prefix state method fingerprint differs",
    )
    if (
        method_config_fingerprint is not None
        and fingerprint != method_config_fingerprint
    ):
        raise Phase13PrefixStateError("prefix state method fingerprint differs")
    _require_sha256(
        manifest.get("source_layout_fingerprint"),
        "prefix state layout fingerprint differs",
    )
    snapshot_key = manifest.get("snapshot_key")
    if snapshot_key != {
        "configuration": manifest.get("configuration"),
        "batch_size": source_batch,
        "historical_context": manifest.get("historical_context"),
    }:
        raise Phase13PrefixStateError("prefix state snapshot key differs")
    capacity = manifest.get("capacity")
    manifest_historical = manifest.get("historical_context")
    if (
        not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or not isinstance(manifest_historical, int)
        or isinstance(manifest_historical, bool)
        or manifest_historical <= 0
        or capacity <= manifest_historical
    ):
        raise Phase13PrefixStateError("prefix state capacity differs")
    for item in tensors:
        shape = item.get("shape")
        axis = item.get("batch_axis")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or (axis is not None and (not isinstance(axis, int) or axis < 0 or axis >= len(shape)))
            or (axis is not None and shape[axis] != source_batch)
        ):
            raise Phase13PrefixStateError("prefix state batch geometry differs")
        if schema_version == PREFIX_STATE_SCHEMA_V3:
            _require_sha256(
                item.get("tensor_sha256"),
                "prefix state tensor checksum differs",
            )
    if schema_version == PREFIX_STATE_SCHEMA_V3:
        if manifest.get("batch_size") != source_batch:
            raise Phase13PrefixStateError("prefix state manifest batch differs")
        input_tensors = manifest.get("input_tensors")
        if (
            not isinstance(input_tensors, list)
            or len(input_tensors) != 2
            or {item.get("name") for item in input_tensors if isinstance(item, dict)}
            != PREFIX_INPUT_NAMES
        ):
            raise Phase13PrefixStateError("prefix state input tensor inventory differs")
        for item in input_tensors:
            if (
                not isinstance(item, dict)
                or item.get("shape") != [source_batch, manifest_historical]
                or item.get("dtype") != "torch.int64"
                or item.get("batch_axis") != 0
            ):
                raise Phase13PrefixStateError("prefix state input batch geometry differs")
            _require_sha256(
                item.get("tensor_sha256"),
                "prefix state input checksum differs",
            )
        stored_model = _identity_payload(
            manifest.get("model_identity", {}),
            "model",
        )
        stored_tokenizer = _identity_payload(
            manifest.get("tokenizer_identity", {}),
            "tokenizer",
        )
        if (
            manifest.get("model_identity_sha256")
            != hashlib.sha256(_canonical_bytes(stored_model)).hexdigest()
            or manifest.get("tokenizer_identity_sha256")
            != hashlib.sha256(_canonical_bytes(stored_tokenizer)).hexdigest()
        ):
            raise Phase13PrefixStateError("prefix state identity checksum differs")
        if model_identity is not None and stored_model != _identity_payload(
            model_identity,
            "model",
        ):
            raise Phase13PrefixStateError("prefix model identity differs")
        if tokenizer_identity is not None and stored_tokenizer != _identity_payload(
            tokenizer_identity,
            "tokenizer",
        ):
            raise Phase13PrefixStateError("prefix tokenizer identity differs")
        if (
            manifest.get("prefix_inputs_stored") is not True
            or manifest.get("prefix_loader_decides_admission") is not False
        ):
            raise Phase13PrefixStateError("prefix state responsibility split differs")
        try:
            from safetensors import safe_open

            with safe_open(str(state_path), framework="pt", device="cpu") as handle:
                observed = set(handle.keys())
                if observed != set(names) | PREFIX_INPUT_NAMES:
                    raise Phase13PrefixStateError(
                        "prefix state safe tensor inventory differs"
                    )
                for item in [*tensors, *input_tensors]:
                    tensor = handle.get_tensor(str(item["name"]))
                    if (
                        [int(value) for value in tensor.shape] != item.get("shape")
                        or str(tensor.dtype) != item.get("dtype")
                        or (
                            verify_state_bytes
                            and _tensor_content_sha256(tensor)
                            != item.get("tensor_sha256")
                        )
                    ):
                        raise Phase13PrefixStateError(
                            "prefix state safe tensor metadata or checksum differs"
                        )
        except OSError as error:
            raise Phase13PrefixStateError("prefix state safe tensor is invalid") from error
    lifecycle = manifest.get("lifecycle")
    if not isinstance(manifest.get("authority"), dict) or not isinstance(
        lifecycle, dict
    ):
        raise Phase13PrefixStateError("prefix state authority or lifecycle differs")
    if lifecycle.get("active_context") != manifest_historical or lifecycle.get(
        "mode"
    ) != "ready":
        raise Phase13PrefixStateError("prefix state lifecycle differs")
    return manifest


def restored_prefix_witness(manifest: Mapping[str, Any], *, target_batch: int) -> str:
    if (
        manifest.get("batch_reuse_policy") != PREFIX_BATCH_REUSE_POLICY
        or manifest.get("source_batch") != target_batch
    ):
        raise Phase13PrefixStateError("prefix witness exact batch differs")
    payload = {
        "schema_version": "kvbench-phase13-restored-prefix-witness-2.0.0",
        "state_file_sha256": manifest.get("state_file_sha256"),
        "configuration": manifest.get("configuration"),
        "historical_context": manifest.get("historical_context"),
        "source_batch": manifest.get("source_batch"),
        "target_batch": target_batch,
        "batch_reuse_policy": PREFIX_BATCH_REUSE_POLICY,
        "batch_selection": "exact_batch_geometry",
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def restore_prefix_state(
    *,
    cache: Any,
    family: str,
    configuration: str,
    historical: int,
    root: Path,
    expected_state_sha256: str,
    expected_method_config_fingerprint: str,
    expected_prefix_token_ids: Any | None = None,
    expected_prefix_positions: Any | None = None,
    expected_model_identity: Mapping[str, Any] | None = None,
    expected_tokenizer_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore into newly allocated caller-owned cache tensors, outside timing."""

    from safetensors import safe_open

    target_batch = int(cache.batch_size)
    manifest = validate_prefix_state(
        root,
        configuration=configuration,
        family=family,
        batch=target_batch,
        historical=historical,
        method_config_fingerprint=expected_method_config_fingerprint,
        model_identity=expected_model_identity,
        tokenizer_identity=expected_tokenizer_identity,
        verify_state_bytes=False,
    )
    if manifest.get("state_file_sha256") != expected_state_sha256:
        raise Phase13PrefixStateError("prefix catalog binding differs")
    source_batch = manifest.get("source_batch")
    if (
        not isinstance(source_batch, int)
        or isinstance(source_batch, bool)
        or target_batch <= 0
        or source_batch <= 0
        or target_batch != source_batch
        or int(cache.capacity) != manifest.get("capacity")
        or cache.layout_fingerprint()
        != manifest.get("source_layout_fingerprint")
    ):
        raise Phase13PrefixStateError("prefix restore geometry differs")
    schema_version = manifest.get("schema_version")
    if schema_version == PREFIX_STATE_SCHEMA and target_batch not in {1, 4, 8}:
        raise Phase13PrefixStateError("prefix restore legacy geometry differs")
    if schema_version == PREFIX_STATE_SCHEMA_V3 and (
        expected_prefix_token_ids is None
        or expected_prefix_positions is None
        or expected_model_identity is None
        or expected_tokenizer_identity is None
    ):
        raise Phase13PrefixStateError("prefix restore v3 identity inputs are absent")
    target_tensors, target_axes = prefix_state_views(
        cache, family=family, historical=historical
    )
    specs = {item["name"]: item for item in manifest["tensors"]}
    if set(specs) != set(target_tensors):
        raise Phase13PrefixStateError("prefix restore tensor inventory differs")
    for name, target in sorted(target_tensors.items()):
        spec = specs[name]
        if (
            target_axes[name] != spec.get("batch_axis")
            or str(target.dtype) != spec.get("dtype")
            or [int(item) for item in target.shape] != spec.get("shape")
        ):
            raise Phase13PrefixStateError(
                "prefix target tensor metadata differs"
            )
    cache.prepare_prefill(historical)
    restored_tensor_checksums_exact = True
    with safe_open(str(root / PREFIX_STATE_FILE), framework="pt", device="cpu") as handle:
        expected_inventory = set(target_tensors)
        if schema_version == PREFIX_STATE_SCHEMA_V3:
            expected_inventory |= PREFIX_INPUT_NAMES
        if set(handle.keys()) != expected_inventory:
            raise Phase13PrefixStateError("safe tensor inventory differs")
        if schema_version == PREFIX_STATE_SCHEMA_V3:
            expected_inputs = {
                PREFIX_TOKEN_IDS_NAME: expected_prefix_token_ids,
                PREFIX_POSITIONS_NAME: expected_prefix_positions,
            }
            input_specs = {
                item["name"]: item for item in manifest["input_tensors"]
            }
            for name, expected in sorted(expected_inputs.items()):
                source = handle.get_tensor(name)
                spec = input_specs[name]
                expected_cpu = (
                    expected.detach().to(device="cpu", copy=True).contiguous()
                )
                if (
                    tuple(source.shape) != tuple(expected_cpu.shape)
                    or source.dtype != expected_cpu.dtype
                    or not source.equal(expected_cpu)
                    or _tensor_content_sha256(expected_cpu)
                    != spec.get("tensor_sha256")
                ):
                    raise Phase13PrefixStateError(
                        "prefix restore input tensor differs"
                    )
        for name in sorted(target_tensors):
            target = target_tensors[name]
            source = handle.get_tensor(name)
            spec = specs[name]
            axis = target_axes[name]
            if axis != spec.get("batch_axis") or str(source.dtype) != spec.get("dtype"):
                raise Phase13PrefixStateError("prefix tensor metadata differs")
            if [int(item) for item in source.shape] != spec.get("shape"):
                raise Phase13PrefixStateError("prefix source shape differs")
            if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
                raise Phase13PrefixStateError("prefix target shape or dtype differs")
            target.copy_(source, non_blocking=False)
            if schema_version == PREFIX_STATE_SCHEMA_V3:
                restored_tensor_checksums_exact = bool(
                    restored_tensor_checksums_exact
                    and _tensor_content_sha256(target)
                    == spec.get("tensor_sha256")
                )
    if not restored_tensor_checksums_exact:
        raise Phase13PrefixStateError("restored prefix tensor checksum differs")
    _restore_lifecycle(cache, family, manifest["lifecycle"])
    cache.complete_prefill()
    if int(cache.active_context) != historical or str(cache.mode) != "ready":
        raise Phase13PrefixStateError("restored cache lifecycle differs")
    if getattr(cache.device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device=cache.device)
    return {
        "state_file_sha256": expected_state_sha256,
        "source_batch": source_batch,
        "target_batch": target_batch,
        "batch_reuse_policy": PREFIX_BATCH_REUSE_POLICY,
        "witness_sha256": restored_prefix_witness(
            manifest, target_batch=target_batch
        ),
        "fresh_target_allocation": True,
        "runtime_prefix_sharing": False,
        "restore_outside_timing": True,
        "schema_version": schema_version,
        "stored_input_tensors_exact": schema_version == PREFIX_STATE_SCHEMA_V3,
        "restored_tensor_checksums_exact": restored_tensor_checksums_exact,
    }
