"""Compact logical-prefix evidence for the Phase 16 Full Scan.

The artifact binds only deterministic token/position inputs.  It never owns or
serializes a method cache, graph pool, workspace, model weight, or compiler
output.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from preflight.run_preflight import json_bytes, write_exclusive


LOGICAL_PREFIX_SCHEMA = "kvbench-phase16-logical-prefix-1.0.0"
LOGICAL_PREFIX_CATALOG_SCHEMA = "kvbench-phase16-logical-prefix-catalog-1.0.0"
INPUT_GENERATOR_VERSION = "kvbench-phase13-pilot-input-1.0.0"
INPUT_SEED = 12_000
VOCABULARY_SIZE = 128_256
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
TOKENIZER_ID = MODEL_ID
TOKENIZER_REVISION = MODEL_REVISION
TOKEN_FILE = "token_ids.safetensors"
MANIFEST_FILE = "logical_prefix_manifest.json"
LEDGER_FILE = "checksums.sha256"
COMPLETE_FILE = "COMPLETE"
ARTIFACT_FILES = frozenset({MANIFEST_FILE, TOKEN_FILE, LEDGER_FILE, COMPLETE_FILE})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class Phase16LogicalPrefixError(RuntimeError):
    """Logical-prefix evidence failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().to(device="cpu", copy=True).contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_geometry(
    *, batch_size: int, configured_context_label: int, historical_context: int
) -> None:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(configured_context_label, int)
        or isinstance(configured_context_label, bool)
        or configured_context_label <= 0
        or not isinstance(historical_context, int)
        or isinstance(historical_context, bool)
        or historical_context <= 0
        or historical_context > 131_071
        or (
            configured_context_label == 131_072
            and historical_context != 131_071
        )
        or (
            configured_context_label != 131_072
            and historical_context != configured_context_label
        )
    ):
        raise Phase16LogicalPrefixError("logical prefix geometry differs")


def logical_prefix_identity(
    *, batch_size: int, configured_context_label: int, historical_context: int
) -> dict[str, Any]:
    _require_geometry(
        batch_size=batch_size,
        configured_context_label=configured_context_label,
        historical_context=historical_context,
    )
    return {
        "schema_version": LOGICAL_PREFIX_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "batch_size": batch_size,
        "configured_context_label": configured_context_label,
        "actual_historical_context": historical_context,
        "total_attended_length": historical_context + 1,
        "input_generator_version": INPUT_GENERATOR_VERSION,
        "input_seed": INPUT_SEED,
    }


def logical_prefix_id(
    *, batch_size: int, configured_context_label: int, historical_context: int
) -> str:
    identity = logical_prefix_identity(
        batch_size=batch_size,
        configured_context_label=configured_context_label,
        historical_context=historical_context,
    )
    return (
        f"logical-prefix-b{batch_size}-l{configured_context_label}-"
        f"{_canonical_sha256(identity)[:12]}"
    )


def deterministic_inputs(
    *, batch_size: int, configured_context_label: int, historical_context: int
) -> tuple[Any, Any, Any]:
    """Return CPU int32 prefix/decode IDs and deterministic int64 positions."""

    import torch

    _require_geometry(
        batch_size=batch_size,
        configured_context_label=configured_context_label,
        historical_context=historical_context,
    )
    prefix = (
        torch.arange(batch_size * historical_context, dtype=torch.int64)
        .reshape(batch_size, historical_context)
        .add(INPUT_SEED)
        .remainder(120_000)
        .add(1_000)
        .to(dtype=torch.int32)
        .contiguous()
    )
    decode = (
        torch.arange(batch_size, dtype=torch.int64)
        .reshape(batch_size, 1)
        .add(INPUT_SEED + historical_context + 257)
        .remainder(120_000)
        .add(1_000)
        .to(dtype=torch.int32)
        .contiguous()
    )
    positions = (
        torch.arange(historical_context, dtype=torch.int64)
        .reshape(1, historical_context)
        .expand(batch_size, historical_context)
        .contiguous()
    )
    if (
        int(prefix.min()) < 0
        or int(decode.min()) < 0
        or int(prefix.max()) >= VOCABULARY_SIZE
        or int(decode.max()) >= VOCABULARY_SIZE
    ):
        raise Phase16LogicalPrefixError("logical token ID exceeds vocabulary")
    return prefix, decode, positions


def create_logical_prefix_artifact(
    output: Path,
    *,
    batch_size: int,
    configured_context_label: int,
    historical_context: int,
) -> dict[str, Any]:
    """Create one immutable compact input artifact with COMPLETE written last."""

    from safetensors.torch import save_file

    if output.exists() or output.is_symlink():
        raise Phase16LogicalPrefixError("logical prefix output already exists")
    output.mkdir(parents=False, exist_ok=False)
    prefix, decode, positions = deterministic_inputs(
        batch_size=batch_size,
        configured_context_label=configured_context_label,
        historical_context=historical_context,
    )
    artifact_id = logical_prefix_id(
        batch_size=batch_size,
        configured_context_label=configured_context_label,
        historical_context=historical_context,
    )
    token_path = output / TOKEN_FILE
    save_file(
        {"prefix_token_ids": prefix, "current_decode_token_ids": decode},
        str(token_path),
    )
    _fsync_file(token_path)
    manifest = {
        **logical_prefix_identity(
            batch_size=batch_size,
            configured_context_label=configured_context_label,
            historical_context=historical_context,
        ),
        "logical_prefix_id": artifact_id,
        "token_tensor": {
            "name": "prefix_token_ids",
            "shape": [batch_size, historical_context],
            "dtype": "torch.int32",
            "sha256": _tensor_sha256(prefix),
        },
        "current_decode_token": {
            "name": "current_decode_token_ids",
            "shape": [batch_size, 1],
            "dtype": "torch.int32",
            "sha256": _tensor_sha256(decode),
        },
        "position_ids": {
            "stored": False,
            "construction": "repeat(arange(actual_historical_context), batch_size)",
            "shape": [batch_size, historical_context],
            "dtype": "torch.int64",
            "sha256": _tensor_sha256(positions),
        },
        "token_file": TOKEN_FILE,
        "token_file_bytes": token_path.stat().st_size,
        "token_file_sha256": _sha256_file(token_path),
        "cache_snapshot_stored": False,
        "graph_pool_stored": False,
        "workspace_stored": False,
        "model_weights_stored": False,
    }
    manifest_path = output / MANIFEST_FILE
    write_exclusive(manifest_path, json_bytes(manifest))
    _fsync_file(manifest_path)
    ledger = (
        f"{_sha256_file(manifest_path)}  {MANIFEST_FILE}\n"
        f"{_sha256_file(token_path)}  {TOKEN_FILE}\n"
    ).encode("ascii")
    write_exclusive(output / LEDGER_FILE, ledger)
    _fsync_file(output / LEDGER_FILE)
    complete = {
        "schema_version": "kvbench-phase16-logical-prefix-complete-1.0.0",
        "logical_prefix_id": artifact_id,
        "status": "COMPLETE",
        "manifest_sha256": _sha256_file(manifest_path),
        "token_file_sha256": _sha256_file(token_path),
        "checksum_ledger_sha256": _sha256_file(output / LEDGER_FILE),
        "written_last": True,
    }
    write_exclusive(output / COMPLETE_FILE, json_bytes(complete))
    _fsync_file(output / COMPLETE_FILE)
    _fsync_directory(output)
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return validate_logical_prefix_artifact(output, load_tensors=False)[0]


def _parse_ledger(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeError) as error:
        raise Phase16LogicalPrefixError("logical prefix ledger is unreadable") from error
    if not raw.endswith(b"\n"):
        raise Phase16LogicalPrefixError("logical prefix ledger is not canonical")
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or _SHA256_RE.fullmatch(parts[0]) is None
            or parts[1] in result
        ):
            raise Phase16LogicalPrefixError("logical prefix ledger differs")
        result[parts[1]] = parts[0]
    if set(result) != {MANIFEST_FILE, TOKEN_FILE}:
        raise Phase16LogicalPrefixError("logical prefix ledger coverage differs")
    return result


def validate_logical_prefix_artifact(
    root: Path,
    *,
    expected: Mapping[str, Any] | None = None,
    load_tensors: bool = True,
) -> tuple[dict[str, Any], Any | None, Any | None]:
    """Validate controls, identities, tensor metadata, and exact token bytes."""

    from safetensors.torch import load_file

    if root.is_symlink() or not root.is_dir():
        raise Phase16LogicalPrefixError("logical prefix root differs")
    if root.stat().st_mode & _WRITE_BITS:
        raise Phase16LogicalPrefixError("logical prefix root remains writable")
    observed_names = {path.name for path in root.iterdir()}
    if observed_names != ARTIFACT_FILES or any(
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & _WRITE_BITS
        for path in root.iterdir()
    ):
        raise Phase16LogicalPrefixError("logical prefix artifact inventory differs")
    try:
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        complete = json.loads((root / COMPLETE_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase16LogicalPrefixError("logical prefix controls are invalid") from error
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise Phase16LogicalPrefixError("logical prefix controls differ")
    if manifest.get("schema_version") != LOGICAL_PREFIX_SCHEMA:
        raise Phase16LogicalPrefixError("logical prefix schema differs")
    identity = logical_prefix_identity(
        batch_size=manifest.get("batch_size"),
        configured_context_label=manifest.get("configured_context_label"),
        historical_context=manifest.get("actual_historical_context"),
    )
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise Phase16LogicalPrefixError("logical prefix identity differs")
    artifact_id = logical_prefix_id(
        batch_size=int(manifest["batch_size"]),
        configured_context_label=int(manifest["configured_context_label"]),
        historical_context=int(manifest["actual_historical_context"]),
    )
    if manifest.get("logical_prefix_id") != artifact_id or root.name != artifact_id:
        raise Phase16LogicalPrefixError("logical prefix ID differs")
    if expected is not None and any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise Phase16LogicalPrefixError("requested logical prefix identity differs")
    ledger = _parse_ledger(root / LEDGER_FILE)
    if any(_sha256_file(root / relative) != digest for relative, digest in ledger.items()):
        raise Phase16LogicalPrefixError("logical prefix checksum differs")
    if (
        complete.get("logical_prefix_id") != artifact_id
        or complete.get("status") != "COMPLETE"
        or complete.get("written_last") is not True
        or complete.get("manifest_sha256") != _sha256_file(root / MANIFEST_FILE)
        or complete.get("token_file_sha256") != _sha256_file(root / TOKEN_FILE)
        or complete.get("checksum_ledger_sha256") != _sha256_file(root / LEDGER_FILE)
        or manifest.get("token_file_sha256") != _sha256_file(root / TOKEN_FILE)
        or manifest.get("token_file_bytes") != (root / TOKEN_FILE).stat().st_size
        or manifest.get("cache_snapshot_stored") is not False
    ):
        raise Phase16LogicalPrefixError("logical prefix completion binding differs")
    if not load_tensors:
        return manifest, None, None
    try:
        tensors = load_file(str(root / TOKEN_FILE), device="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise Phase16LogicalPrefixError("logical token file is invalid") from error
    if set(tensors) != {"prefix_token_ids", "current_decode_token_ids"}:
        raise Phase16LogicalPrefixError("logical token inventory differs")
    prefix = tensors["prefix_token_ids"]
    decode = tensors["current_decode_token_ids"]
    batch = int(manifest["batch_size"])
    historical = int(manifest["actual_historical_context"])
    if (
        tuple(prefix.shape) != (batch, historical)
        or tuple(decode.shape) != (batch, 1)
        or str(prefix.dtype) != "torch.int32"
        or str(decode.dtype) != "torch.int32"
        or manifest.get("token_tensor", {}).get("sha256") != _tensor_sha256(prefix)
        or manifest.get("current_decode_token", {}).get("sha256")
        != _tensor_sha256(decode)
    ):
        raise Phase16LogicalPrefixError("logical token tensor differs")
    expected_prefix, expected_decode, positions = deterministic_inputs(
        batch_size=batch,
        configured_context_label=int(manifest["configured_context_label"]),
        historical_context=historical,
    )
    if (
        not prefix.equal(expected_prefix)
        or not decode.equal(expected_decode)
        or manifest.get("position_ids", {}).get("sha256")
        != _tensor_sha256(positions)
    ):
        raise Phase16LogicalPrefixError("logical input reconstruction differs")
    return manifest, prefix, decode


def artifact_size_bytes(root: Path) -> int:
    validate_logical_prefix_artifact(root, load_tensors=False)
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())
