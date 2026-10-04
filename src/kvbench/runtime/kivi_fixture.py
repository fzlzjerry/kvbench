"""Checksum-bound access to the immutable Phase 7 KIVI fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Final

from kvbench.runtime.kivi_cache import KIVI_CONFIG_BITS
from kvbench.schema.phase8 import Phase7LegacyAllocationRatio


KIVI_OFFICIAL_COMMIT: Final[str] = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
KIVI_BASE_TREE: Final[str] = "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b"
KIVI_PATCHED_TREE: Final[str] = "b617493dea5aff1a754cd27ad6be12ac512b2aee"
KIVI_DECISION_0018_PATCH_SHA256: Final[str] = (
    "c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d"
)
KIVI_EXTENSION_SHA256: Final[str] = (
    "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9"
)
KIVI_FIXTURE_ROOT_DIGEST: Final[str] = (
    "abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302"
)
KIVI_FIXTURE_R2_URI: Final[str] = (
    "r2://kvbench-artifacts/kvbench/sha256/"
    f"{KIVI_FIXTURE_ROOT_DIGEST}/"
)
KIVI_FIXTURE_CONFIGS: Final[tuple[str, ...]] = (
    "k4v4",
    "k2v4",
    "k2v2",
    "k4v2",
)
KIVI_FIXTURE_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[3] / "reference" / "kivi" / "fixtures"
)
_EXPECTED_SET_SHA256: Final[str] = (
    "ae3e83eb65e3cd5b407de81320f38a5eac0c35158500cd42212e215bdb1f584e"
)


class KIVIFixtureError(RuntimeError):
    """The frozen fixture authority or a tensor payload is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KIVIFixtureError(f"invalid KIVI fixture JSON: {path}") from error
    if type(value) is not dict:
        raise KIVIFixtureError(f"KIVI fixture root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise KIVIFixtureError(f"cannot read KIVI fixture file: {path}") from error


def _require_manifest_member(
    root: Path, member: dict[str, Any], *, label: str
) -> Path:
    path_value = member.get("path")
    digest = member.get("sha256")
    nbytes = member.get("nbytes")
    if (
        type(path_value) is not str
        or Path(path_value).is_absolute()
        or ".." in Path(path_value).parts
        or type(digest) is not str
        or len(digest) != 64
        or type(nbytes) is not int
        or nbytes < 0
    ):
        raise KIVIFixtureError(f"{label} manifest member is malformed")
    path = root / path_value
    try:
        observed_size = path.stat().st_size
    except OSError as error:
        raise KIVIFixtureError(f"{label} manifest member is missing") from error
    if observed_size != nbytes or _sha256(path) != digest:
        raise KIVIFixtureError(f"{label} manifest member checksum differs")
    return path


@dataclass(frozen=True, slots=True)
class KIVIFixture:
    """One verified configuration and its immutable JSON tensor records."""

    config_name: str
    manifest_path: Path
    fixture_path: Path
    manifest: dict[str, Any]
    payload: dict[str, Any]

    @property
    def k_bits(self) -> int:
        return KIVI_CONFIG_BITS[self.config_name][0]

    @property
    def v_bits(self) -> int:
        return KIVI_CONFIG_BITS[self.config_name][1]

    def tensor_record(self, dotted_path: str) -> dict[str, Any] | None:
        value: Any = self.payload
        for component in dotted_path.split("."):
            if type(value) is not dict or component not in value:
                raise KIVIFixtureError(
                    f"fixture tensor path is absent: {dotted_path}"
                )
            value = value[component]
        if value is None:
            return None
        if type(value) is not dict:
            raise KIVIFixtureError(
                f"fixture tensor path is not a record: {dotted_path}"
            )
        validate_tensor_record(value)
        return value

    def legacy_allocation_records(self) -> tuple[dict[str, Any], ...]:
        records = self.payload.get("byte_accounting")
        if type(records) is not list:
            raise KIVIFixtureError("fixture byte_accounting is malformed")
        normalized: list[dict[str, Any]] = []
        for record in records:
            if type(record) is not dict:
                raise KIVIFixtureError("fixture byte record is malformed")
            legacy = Phase7LegacyAllocationRatio.from_phase7_r_alloc(
                float(record.get("r_alloc"))
            )
            item = dict(record)
            item.pop("r_alloc", None)
            item["rho_alloc_legacy"] = legacy.rho_alloc_legacy
            item["canonical_r_alloc"] = legacy.canonical_r_alloc
            normalized.append(item)
        return tuple(normalized)


def load_kivi_fixture(
    config_name: str, *, fixture_root: Path = KIVI_FIXTURE_ROOT
) -> KIVIFixture:
    """Load one fixture only after its complete set and member hashes match."""

    if config_name not in KIVI_FIXTURE_CONFIGS:
        raise KIVIFixtureError("unsupported KIVI fixture configuration")
    set_path = fixture_root / "fixture_set.json"
    if fixture_root == KIVI_FIXTURE_ROOT and _sha256(set_path) != _EXPECTED_SET_SHA256:
        raise KIVIFixtureError("KIVI fixture-set checksum differs")
    fixture_set = _load_json(set_path)
    expected_set_identity = (
        fixture_set.get("source_commit"),
        fixture_set.get("base_tree"),
        fixture_set.get("patched_tree"),
        fixture_set.get("extension_sha256"),
        tuple(fixture_set.get("configurations", ())),
    )
    if expected_set_identity != (
        KIVI_OFFICIAL_COMMIT,
        KIVI_BASE_TREE,
        KIVI_PATCHED_TREE,
        KIVI_EXTENSION_SHA256,
        KIVI_FIXTURE_CONFIGS,
    ):
        raise KIVIFixtureError("KIVI fixture-set authority differs")
    entries = fixture_set.get("variant_manifests")
    if type(entries) is not list:
        raise KIVIFixtureError("KIVI variant manifest list is malformed")
    entry = next(
        (
            candidate
            for candidate in entries
            if type(candidate) is dict
            and candidate.get("variant") == config_name
        ),
        None,
    )
    if entry is None:
        raise KIVIFixtureError("KIVI variant manifest is absent")
    manifest_path = _require_manifest_member(
        fixture_root, entry, label=f"{config_name} variant"
    )
    manifest = _load_json(manifest_path)
    variant = manifest.get("variant")
    if (
        type(variant) is not dict
        or (
            variant.get("id"),
            variant.get("k_bits"),
            variant.get("v_bits"),
        )
        != (config_name, *KIVI_CONFIG_BITS[config_name])
        or manifest.get("source_commit") != KIVI_OFFICIAL_COMMIT
        or manifest.get("patched_tree") != KIVI_PATCHED_TREE
        or manifest.get("extension_sha256") != KIVI_EXTENSION_SHA256
    ):
        raise KIVIFixtureError("KIVI variant authority differs")
    fixture_path = _require_manifest_member(
        manifest_path.parent,
        manifest.get("fixture", {}),
        label=f"{config_name} fixture",
    )
    payload = _load_json(fixture_path)
    source = payload.get("source")
    configuration = payload.get("configuration")
    geometry = payload.get("geometry")
    if (
        type(source) is not dict
        or type(configuration) is not dict
        or type(geometry) is not dict
        or (
            source.get("commit"),
            source.get("base_tree"),
            source.get("patched_tree"),
            source.get("authority"),
        )
        != (
            KIVI_OFFICIAL_COMMIT,
            KIVI_BASE_TREE,
            KIVI_PATCHED_TREE,
            "checksum_bound_patched_official_source",
        )
        or (
            configuration.get("k_bits"),
            configuration.get("v_bits"),
            configuration.get("group_size"),
            configuration.get("residual_length"),
        )
        != (*KIVI_CONFIG_BITS[config_name], 32, 32)
        or (
            geometry.get("batch_size"),
            geometry.get("num_query_heads"),
            geometry.get("num_kv_heads"),
            geometry.get("head_dim"),
            geometry.get("input_dtype"),
            geometry.get("upstream_cuda_execution_dtype"),
        )
        != (1, 32, 8, 128, "bfloat16", "float16")
    ):
        raise KIVIFixtureError("KIVI fixture semantics differ")
    return KIVIFixture(
        config_name=config_name,
        manifest_path=manifest_path,
        fixture_path=fixture_path,
        manifest=manifest,
        payload=payload,
    )


def validate_tensor_record(record: dict[str, Any]) -> None:
    payload_hex = record.get("payload_hex")
    payload_sha256 = record.get("payload_sha256")
    shape = record.get("shape")
    logical_nbytes = record.get("logical_nbytes")
    storage_nbytes = record.get("storage_nbytes")
    dtype = record.get("dtype")
    if (
        type(payload_hex) is not str
        or type(payload_sha256) is not str
        or type(shape) is not list
        or any(type(item) is not int or item < 0 for item in shape)
        or type(logical_nbytes) is not int
        or type(storage_nbytes) is not int
        or dtype not in {"bfloat16", "float16", "int32"}
    ):
        raise KIVIFixtureError("fixture tensor record is malformed")
    try:
        payload = bytes.fromhex(payload_hex)
    except ValueError as error:
        raise KIVIFixtureError("fixture tensor payload is not hexadecimal") from error
    if (
        hashlib.sha256(payload).hexdigest() != payload_sha256
        or len(payload) != logical_nbytes
        or logical_nbytes != storage_nbytes
    ):
        raise KIVIFixtureError("fixture tensor payload checksum or size differs")


def tensor_from_record(record: dict[str, Any], *, device: Any) -> Any:
    """Materialize a verified fixture tensor outside measured execution."""

    validate_tensor_record(record)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:  # pragma: no cover - environment
        raise KIVIFixtureError("PyTorch is required to load fixture tensors") from error
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "int32": torch.int32,
    }[record["dtype"]]
    payload = bytearray.fromhex(record["payload_hex"])
    tensor = torch.frombuffer(payload, dtype=dtype).reshape(tuple(record["shape"]))
    # Clone before the bytearray leaves scope. Transfer is admission setup,
    # never part of a measured decode region.
    return tensor.clone().to(device=device)


def validate_all_kivi_fixtures(
    *, fixture_root: Path = KIVI_FIXTURE_ROOT
) -> tuple[KIVIFixture, ...]:
    return tuple(
        load_kivi_fixture(config, fixture_root=fixture_root)
        for config in KIVI_FIXTURE_CONFIGS
    )
