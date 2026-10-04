"""Exact offline model/tokenizer identity verification and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

from kvbench.runtime.backend import (
    ATTENTION_IMPLEMENTATION,
    BackendUnsupportedError,
    register_transformers_attention,
)


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
DEFAULT_SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/"
    f"snapshots/{MODEL_REVISION}"
)
TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_HASHES = {
    "LICENSE": "64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369",
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "generation_config.json": "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}
EXPECTED_SIZES = {
    "LICENSE": 7_627,
    "config.json": 855,
    "generation_config.json": 184,
    "model-00001-of-00004.safetensors": 4_976_698_672,
    "model-00002-of-00004.safetensors": 4_999_802_720,
    "model-00003-of-00004.safetensors": 4_915_916_176,
    "model-00004-of-00004.safetensors": 1_168_138_808,
    "model.safetensors.index.json": 23_950,
    "special_tokens_map.json": 296,
    "tokenizer.json": 9_085_657,
    "tokenizer_config.json": 55_351,
}


class ModelIdentityError(RuntimeError):
    """The local checkpoint differs from the frozen identity."""


class ModelAccessError(RuntimeError):
    """The exact gated checkpoint is not locally accessible."""


_LOADED_FROZEN_MODEL_SEAL = object()
_FROZEN_MODEL_LOAD_RECEIPT_SEAL = object()
PARAMETER_BINDING_KIND = (
    "object_storage_pointer_version_no_live_content_hash"
)
_MODEL_LOADER_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenModelIdentity:
    """Fully verified identity used by performance and later quality work."""

    model_id: str
    revision: str
    snapshot_path: str
    file_hashes: dict[str, str]
    architecture: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    weight_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "snapshot_path": self.snapshot_path,
            "file_hashes": dict(sorted(self.file_hashes.items())),
            "architecture": self.architecture,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "weight_dtype": self.weight_dtype,
        }


@dataclass(frozen=True, slots=True)
class FrozenModelLoadReceipt:
    """Runtime identity for one verified load; not a live weight-content hash."""

    schema_version: str
    parameter_binding_kind: str
    frozen_identity_sha256: str
    snapshot_file_ledger_sha256: str
    loader_source_sha256: str
    model_object_id: int
    tokenizer_object_id: int
    model_class_module: str
    model_class_name: str
    tokenizer_class_module: str
    tokenizer_class_name: str
    tokenizer_runtime_sha256: str
    parameter_runtime_sha256: str
    parameter_tensor_count: int
    parameter_element_count: int
    receipt_sha256: str
    _model_ref: Any = field(repr=False, compare=False)
    _tokenizer_ref: Any = field(repr=False, compare=False)
    _parameter_names: tuple[str, ...] = field(repr=False, compare=False)
    _parameter_refs: tuple[Any, ...] = field(repr=False, compare=False)
    _storage_refs: tuple[Any, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    SCHEMA_VERSION = "kvbench-frozen-model-load-receipt-2.0.0"

    def __post_init__(self) -> None:
        if self._seal is not _FROZEN_MODEL_LOAD_RECEIPT_SEAL:
            raise ModelIdentityError("frozen model load receipt is not factory sealed")
        if self.schema_version != self.SCHEMA_VERSION:
            raise ModelIdentityError("frozen model load receipt schema differs")
        if self.parameter_binding_kind != PARAMETER_BINDING_KIND:
            raise ModelIdentityError(
                "frozen model parameter binding kind differs"
            )
        digests = (
            self.frozen_identity_sha256,
            self.snapshot_file_ledger_sha256,
            self.loader_source_sha256,
            self.tokenizer_runtime_sha256,
            self.parameter_runtime_sha256,
            self.receipt_sha256,
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ModelIdentityError("frozen model load receipt digest is invalid")
        if (
            self.model_object_id <= 0
            or self.tokenizer_object_id <= 0
            or self._model_ref is None
            or self._tokenizer_ref is None
            or id(self._model_ref) != self.model_object_id
            or id(self._tokenizer_ref) != self.tokenizer_object_id
            or self.parameter_tensor_count <= 0
            or self.parameter_element_count <= 0
            or not self.model_class_module
            or not self.model_class_name
            or not self.tokenizer_class_module
            or not self.tokenizer_class_name
        ):
            raise ModelIdentityError("frozen model load receipt identity is invalid")
        if (
            len(self._parameter_names) != self.parameter_tensor_count
            or len(self._parameter_refs) != self.parameter_tensor_count
            or len(self._storage_refs) != self.parameter_tensor_count
            or tuple(sorted(self._parameter_names)) != self._parameter_names
            or len(set(self._parameter_names)) != self.parameter_tensor_count
            or any(parameter is None for parameter in self._parameter_refs)
            or any(storage is None for storage in self._storage_refs)
        ):
            raise ModelIdentityError(
                "frozen model load receipt strong references are invalid"
            )
        if self.receipt_sha256 != _receipt_sha256(self._payload()):
            raise ModelIdentityError("frozen model load receipt fingerprint differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parameter_binding_kind": self.parameter_binding_kind,
            "frozen_identity_sha256": self.frozen_identity_sha256,
            "snapshot_file_ledger_sha256": self.snapshot_file_ledger_sha256,
            "loader_source_sha256": self.loader_source_sha256,
            "model_object_id": self.model_object_id,
            "tokenizer_object_id": self.tokenizer_object_id,
            "model_class_module": self.model_class_module,
            "model_class_name": self.model_class_name,
            "tokenizer_class_module": self.tokenizer_class_module,
            "tokenizer_class_name": self.tokenizer_class_name,
            "tokenizer_runtime_sha256": self.tokenizer_runtime_sha256,
            "parameter_runtime_sha256": self.parameter_runtime_sha256,
            "parameter_tensor_count": self.parameter_tensor_count,
            "parameter_element_count": self.parameter_element_count,
        }


@dataclass(frozen=True, slots=True)
class LoadedFrozenModel:
    """Loaded exact checkpoint and tokenizer plus verified identity."""

    model: Any
    tokenizer: Any
    identity: FrozenModelIdentity
    receipt: FrozenModelLoadReceipt
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _LOADED_FROZEN_MODEL_SEAL:
            raise ModelIdentityError("loaded frozen model is not factory sealed")
        validate_loaded_frozen_model_receipt(self)


def _receipt_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _capture_parameter_runtime_identity(
    model: Any,
) -> tuple[
    str,
    int,
    int,
    tuple[str, ...],
    tuple[Any, ...],
    tuple[Any, ...],
]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise ModelIdentityError("loaded model has no named parameter inventory")
    try:
        parameters = tuple(named_parameters())
    except (RuntimeError, TypeError, ValueError) as error:
        raise ModelIdentityError("loaded model parameter inventory is unreadable") from error
    entries: list[tuple[dict[str, Any], Any, Any]] = []
    names: set[str] = set()
    total_elements = 0
    for name, parameter in parameters:
        if type(name) is not str or not name or name in names:
            raise ModelIdentityError("loaded model parameter names are invalid")
        names.add(name)
        try:
            storage = parameter.untyped_storage()
            record = {
                "name": name,
                "parameter_object_id": id(parameter),
                "shape": [int(value) for value in parameter.shape],
                "stride": [int(value) for value in parameter.stride()],
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "numel": int(parameter.numel()),
                "requires_grad": parameter.requires_grad,
                "data_ptr": int(parameter.data_ptr()),
                "storage_data_ptr": int(storage.data_ptr()),
                "storage_handle": int(storage._cdata),
                "storage_nbytes": int(storage.nbytes()),
                "storage_offset": int(parameter.storage_offset()),
                "element_size": int(parameter.element_size()),
                "version": int(parameter._version),
            }
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise ModelIdentityError(
                f"loaded model parameter is unreadable: {name}"
            ) from error
        if (
            record["numel"] <= 0
            or record["data_ptr"] <= 0
            or record["storage_data_ptr"] <= 0
            or record["storage_handle"] <= 0
            or record["storage_nbytes"] <= 0
            or record["storage_offset"] < 0
            or record["element_size"] <= 0
            or record["version"] < 0
            or record["requires_grad"] is not False
            or record["data_ptr"]
            != record["storage_data_ptr"]
            + record["storage_offset"] * record["element_size"]
        ):
            raise ModelIdentityError(
                f"loaded model parameter runtime identity is invalid: {name}"
            )
        total_elements += int(record["numel"])
        entries.append((record, parameter, storage))
    entries.sort(key=lambda value: str(value[0]["name"]))
    if not entries or total_elements <= 0:
        raise ModelIdentityError("loaded model parameter inventory is empty")
    records = tuple(entry[0] for entry in entries)
    parameter_names = tuple(str(record["name"]) for record in records)
    parameter_refs = tuple(entry[1] for entry in entries)
    storage_refs = tuple(entry[2] for entry in entries)
    return (
        _receipt_sha256(records),
        len(records),
        total_elements,
        parameter_names,
        parameter_refs,
        storage_refs,
    )


def _canonical_tokenizer_runtime_value(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ModelIdentityError(
                "tokenizer runtime configuration contains a non-finite value"
            )
        return value
    if isinstance(value, dict):
        if any(type(key) not in {int, str} for key in value):
            raise ModelIdentityError(
                "tokenizer runtime configuration has an unsupported key"
            )
        return {
            (
                f"str:{key}"
                if type(key) is str
                else f"int:{key}"
            ): _canonical_tokenizer_runtime_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: (type(pair[0]).__name__, str(pair[0])),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_tokenizer_runtime_value(item) for item in value]
    token_fields = (
        "content",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    )
    if all(hasattr(value, field_name) for field_name in token_fields):
        return {
            "class_module": value.__class__.__module__,
            "class_name": value.__class__.__name__,
            **{
                field_name: _canonical_tokenizer_runtime_value(
                    getattr(value, field_name)
                )
                for field_name in token_fields
            },
        }
    raise ModelIdentityError(
        "tokenizer runtime configuration contains an unsupported value"
    )


def _capture_tokenizer_runtime_identity(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        raise ModelIdentityError(
            "loaded tokenizer backend serialization is unavailable"
        )
    try:
        backend_json = to_str()
    except (RuntimeError, TypeError, ValueError) as error:
        raise ModelIdentityError(
            "loaded tokenizer backend serialization failed"
        ) from error
    if type(backend_json) is not str or not backend_json:
        raise ModelIdentityError(
            "loaded tokenizer backend serialization is invalid"
        )
    configuration = {
        "backend_tokenizer_sha256": hashlib.sha256(
            backend_json.encode("utf-8")
        ).hexdigest(),
        "special_tokens_map_extended": _canonical_tokenizer_runtime_value(
            getattr(tokenizer, "special_tokens_map_extended", {})
        ),
        "added_tokens_decoder": _canonical_tokenizer_runtime_value(
            getattr(tokenizer, "added_tokens_decoder", {})
        ),
        "all_special_ids": _canonical_tokenizer_runtime_value(
            tuple(getattr(tokenizer, "all_special_ids", ()))
        ),
        "chat_template": _canonical_tokenizer_runtime_value(
            getattr(tokenizer, "chat_template", None)
        ),
        "model_max_length": _canonical_tokenizer_runtime_value(
            getattr(tokenizer, "model_max_length", None)
        ),
        "padding_side": _canonical_tokenizer_runtime_value(
            getattr(tokenizer, "padding_side", None)
        ),
        "truncation_side": _canonical_tokenizer_runtime_value(
            getattr(tokenizer, "truncation_side", None)
        ),
    }
    return _receipt_sha256(configuration)


def validate_loaded_frozen_model_receipt(loaded: LoadedFrozenModel) -> None:
    """Revalidate factory origin and live object/storage/version identity."""

    if type(loaded) is not LoadedFrozenModel:
        raise ModelIdentityError("loaded frozen model has the wrong type")
    if loaded._seal is not _LOADED_FROZEN_MODEL_SEAL:
        raise ModelIdentityError("loaded frozen model seal differs")
    receipt = loaded.receipt
    if (
        type(receipt) is not FrozenModelLoadReceipt
        or receipt._seal is not _FROZEN_MODEL_LOAD_RECEIPT_SEAL
    ):
        raise ModelIdentityError("loaded frozen model receipt seal differs")
    receipt.__post_init__()
    current_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current_source_sha != _MODEL_LOADER_SOURCE_SHA256_AT_IMPORT:
        raise ModelIdentityError(
            "model loader source changed after module import"
        )
    (
        parameter_sha,
        parameter_count,
        parameter_elements,
        parameter_names,
        parameter_refs,
        storage_refs,
    ) = _capture_parameter_runtime_identity(loaded.model)
    if (
        receipt._model_ref is not loaded.model
        or receipt._tokenizer_ref is not loaded.tokenizer
        or parameter_names != receipt._parameter_names
        or len(parameter_refs) != len(receipt._parameter_refs)
        or len(storage_refs) != len(receipt._storage_refs)
        or any(
            observed is not expected
            for observed, expected in zip(
                parameter_refs,
                receipt._parameter_refs,
                strict=True,
            )
        )
        or any(
            observed is not expected
            for observed, expected in zip(
                storage_refs,
                receipt._storage_refs,
                strict=True,
            )
        )
    ):
        raise ModelIdentityError(
            "loaded model strong object or storage identity changed"
        )
    observed = (
        _receipt_sha256(loaded.identity.to_dict()),
        _receipt_sha256(dict(sorted(loaded.identity.file_hashes.items()))),
        current_source_sha,
        id(loaded.model),
        id(loaded.tokenizer),
        loaded.model.__class__.__module__,
        loaded.model.__class__.__name__,
        loaded.tokenizer.__class__.__module__,
        loaded.tokenizer.__class__.__name__,
        _capture_tokenizer_runtime_identity(loaded.tokenizer),
        parameter_sha,
        parameter_count,
        parameter_elements,
    )
    expected = (
        receipt.frozen_identity_sha256,
        receipt.snapshot_file_ledger_sha256,
        receipt.loader_source_sha256,
        receipt.model_object_id,
        receipt.tokenizer_object_id,
        receipt.model_class_module,
        receipt.model_class_name,
        receipt.tokenizer_class_module,
        receipt.tokenizer_class_name,
        receipt.tokenizer_runtime_sha256,
        receipt.parameter_runtime_sha256,
        receipt.parameter_tensor_count,
        receipt.parameter_element_count,
    )
    if observed != expected:
        raise ModelIdentityError(
            "loaded frozen model receipt no longer matches live objects"
        )

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelIdentityError(f"invalid frozen JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ModelIdentityError(f"frozen JSON artifact is not an object: {path.name}")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ModelIdentityError(f"frozen model field mismatch: {label}")


def verify_frozen_snapshot(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
) -> FrozenModelIdentity:
    """Hash every required byte and validate the frozen Llama geometry."""

    snapshot = Path(snapshot_path).expanduser().resolve(strict=False)
    if not snapshot.is_dir():
        raise ModelAccessError("the exact frozen model snapshot is absent")
    if snapshot.name != MODEL_REVISION:
        raise ModelIdentityError("snapshot directory is not the frozen revision")
    observed_entries = {path.name for path in snapshot.iterdir()}
    expected_entries = set(EXPECTED_HASHES)
    if observed_entries != expected_entries:
        missing = sorted(expected_entries - observed_entries)
        unexpected = sorted(observed_entries - expected_entries)
        raise ModelIdentityError(
            "frozen snapshot entry set differs: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    model_cache_root = snapshot.parent.parent
    if any(model_cache_root.rglob("*.incomplete")):
        raise ModelIdentityError("incomplete download artifact exists in model cache")
    observed: dict[str, str] = {}
    for name, expected_hash in EXPECTED_HASHES.items():
        path = snapshot / name
        if not path.is_file():
            raise ModelAccessError(f"required frozen model artifact is absent: {name}")
        if path.stat().st_size != EXPECTED_SIZES[name]:
            raise ModelIdentityError(f"byte size mismatch for frozen artifact: {name}")
        digest = sha256_file(path)
        if digest != expected_hash:
            raise ModelIdentityError(f"SHA-256 mismatch for frozen artifact: {name}")
        observed[name] = digest

    config = _load_json(snapshot / "config.json")
    _require_equal(config.get("architectures"), ["LlamaForCausalLM"], "architectures")
    _require_equal(config.get("model_type"), "llama", "model_type")
    _require_equal(config.get("hidden_size"), 4096, "hidden_size")
    _require_equal(config.get("num_hidden_layers"), 32, "num_hidden_layers")
    _require_equal(config.get("num_attention_heads"), 32, "num_attention_heads")
    _require_equal(config.get("num_key_value_heads"), 8, "num_key_value_heads")
    _require_equal(config.get("max_position_embeddings"), 131072, "max_position_embeddings")
    _require_equal(config.get("rope_theta"), 500000.0, "rope_theta")
    _require_equal(config.get("torch_dtype"), "bfloat16", "torch_dtype")
    _require_equal(config.get("sliding_window"), None, "sliding_window")
    rope = config.get("rope_scaling")
    if not isinstance(rope, dict):
        raise ModelIdentityError("rope_scaling is not a mapping")
    for key, expected in {
        "rope_type": "llama3",
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    }.items():
        _require_equal(rope.get(key), expected, f"rope_scaling.{key}")

    index = _load_json(snapshot / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelIdentityError("weight index has no weight_map")
    declared_shards = {
        value for value in weight_map.values() if isinstance(value, str)
    }
    expected_shards = {
        name for name in EXPECTED_HASHES if name.endswith(".safetensors")
    }
    _require_equal(declared_shards, expected_shards, "weight shards")
    return FrozenModelIdentity(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        snapshot_path=str(snapshot),
        file_hashes=observed,
        architecture="LlamaForCausalLM",
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=131072,
        weight_dtype="bfloat16",
    )


def load_frozen_model(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
    *,
    device: str = "cuda:0",
) -> LoadedFrozenModel:
    """Load only the hash-verified local snapshot with all network paths closed."""

    identity = verify_frozen_snapshot(snapshot_path)
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise ModelAccessError(
            "the locked Phase 3 model-loading dependency target is unavailable"
        ) from error
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ModelIdentityError("Transformers version differs from 4.57.6")
    register_transformers_attention()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    snapshot = identity.snapshot_path
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ModelAccessError("exact local model/tokenizer loading failed") from error
    model.to(device=torch.device(device), dtype=torch.bfloat16)
    model.eval()
    model.requires_grad_(False)
    if model.__class__.__name__ != "LlamaForCausalLM":
        raise ModelIdentityError("loaded architecture is not LlamaForCausalLM")
    if tokenizer.__class__.__name__ != "PreTrainedTokenizerFast":
        raise ModelIdentityError(
            "loaded tokenizer class is not PreTrainedTokenizerFast"
        )
    config = model.config
    for name, expected in {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 131072,
    }.items():
        _require_equal(getattr(config, name, None), expected, name)
    _require_equal(
        config._attn_implementation,
        ATTENTION_IMPLEMENTATION,
        "attention implementation",
    )
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ModelIdentityError("loaded model parameter dtype is not uniformly BF16")
    if len(tokenizer) != 128256:
        raise ModelIdentityError("loaded tokenizer vocabulary size differs")
    current_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current_source_sha != _MODEL_LOADER_SOURCE_SHA256_AT_IMPORT:
        raise ModelIdentityError(
            "model loader source changed after module import"
        )
    (
        parameter_sha,
        parameter_count,
        parameter_elements,
        parameter_names,
        parameter_refs,
        storage_refs,
    ) = _capture_parameter_runtime_identity(model)
    values = {
        "schema_version": FrozenModelLoadReceipt.SCHEMA_VERSION,
        "parameter_binding_kind": PARAMETER_BINDING_KIND,
        "frozen_identity_sha256": _receipt_sha256(identity.to_dict()),
        "snapshot_file_ledger_sha256": _receipt_sha256(
            dict(sorted(identity.file_hashes.items()))
        ),
        "loader_source_sha256": current_source_sha,
        "model_object_id": id(model),
        "tokenizer_object_id": id(tokenizer),
        "model_class_module": model.__class__.__module__,
        "model_class_name": model.__class__.__name__,
        "tokenizer_class_module": tokenizer.__class__.__module__,
        "tokenizer_class_name": tokenizer.__class__.__name__,
        "tokenizer_runtime_sha256": (
            _capture_tokenizer_runtime_identity(tokenizer)
        ),
        "parameter_runtime_sha256": parameter_sha,
        "parameter_tensor_count": parameter_count,
        "parameter_element_count": parameter_elements,
    }
    receipt = FrozenModelLoadReceipt(
        **values,
        receipt_sha256=_receipt_sha256(values),
        _model_ref=model,
        _tokenizer_ref=tokenizer,
        _parameter_names=parameter_names,
        _parameter_refs=parameter_refs,
        _storage_refs=storage_refs,
        _seal=_FROZEN_MODEL_LOAD_RECEIPT_SEAL,
    )
    return LoadedFrozenModel(
        model=model,
        tokenizer=tokenizer,
        identity=identity,
        receipt=receipt,
        _seal=_LOADED_FROZEN_MODEL_SEAL,
    )


def require_local_model() -> FrozenModelIdentity:
    """Validate the default offline snapshot without loading framework objects."""

    try:
        return verify_frozen_snapshot(DEFAULT_SNAPSHOT)
    except BackendUnsupportedError:
        raise
