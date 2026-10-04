"""Fail-closed access to the frozen Phase 5 TurboQuant fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

from kvbench.runtime.numerical import compare_tensors_untimed
from kvbench.schema.phase6 import (
    FIXTURE_ROOT_LEDGER_SHA256,
    FIXTURE_SET_SHA256,
    MANDATORY_CONFIG_SLOT_SIZES,
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_TREE,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference" / "turboquant"
DEFAULT_FIXTURE_ROOT = REFERENCE_ROOT / "fixtures"
SOURCE_MANIFEST_PATH = REFERENCE_ROOT / "source_manifest.json"
ENVIRONMENT_MANIFEST_PATH = REFERENCE_ROOT / "environment.json"
SOURCE_MANIFEST_SHA256 = (
    "2817315b99eac1388973fd59b9cf73f795fd81172496ef066efefeda5e2235da"
)
ENVIRONMENT_MANIFEST_SHA256 = (
    "b8a44c6769a17eb3c1de6e3ce129563bca2d338a7702b94e9256d443b89fcdb4"
)
DECODE_ATOL = 0.02
DECODE_RTOL = 0.02
MANDATORY_CONFIGURATIONS = tuple(MANDATORY_CONFIG_SLOT_SIZES)
_EXPECTED_INPUT_SHA256 = {
    "append_key": "8fc4f43caf1b9340449fe210a898c6286a1f5084edcd16a57cdb664d4671e984",
    "append_value": "2c6e62a917106eb96d957daab3679b14c40575fb9feae7a8360b9e4fd15b0a30",
    "decode_query": "55d8a35c7743d2888dab6041d1a5a2ac6b51cd63290fc76de61b792163e96577",
    "prefill_key": "ae4bb4a4c578f8695e89527bc2e004b870acad87efbba8e95a26a5ae15e7da8f",
    "prefill_value": "4280ddbb91a20b1d1ce48f34d4d143fb7fba31391249fc1561d4b6cd4bc89858",
}
_EXPECTED_GEOMETRY: Mapping[str, int | str] = {
    "append_tokens": 1,
    "batch_size": 1,
    "block_size": 16,
    "head_dim": 128,
    "initial_context": 17,
    "input_dtype": "bfloat16",
    "max_num_kv_splits": 4,
    "num_blocks": 2,
    "num_kv_heads": 8,
    "num_query_heads": 32,
    "seed": 20260724,
    "total_context": 18,
}
_EXPECTED_LAYOUTS: Mapping[str, Mapping[str, int]] = {
    "turboquant_4bit_nc": {
        "alignment_padding": 0,
        "key_norm": 2,
        "packed_keys": 64,
        "packed_values": 64,
        "value_scale": 2,
        "value_zero_point": 2,
    },
    "turboquant_k3v4_nc": {
        "alignment_padding": 0,
        "key_norm": 2,
        "packed_keys": 48,
        "packed_values": 64,
        "value_scale": 2,
        "value_zero_point": 2,
    },
    "turboquant_3bit_nc": {
        "alignment_padding": 0,
        "key_norm": 2,
        "packed_keys": 48,
        "packed_values": 48,
        "value_scale": 2,
        "value_zero_point": 2,
    },
}
_TORCH: Any | None = None


class TurboQuantFixtureError(RuntimeError):
    """Raised when frozen fixture authority or conformance is violated."""


@dataclass(frozen=True, slots=True)
class FixtureTensorRecord:
    """One checksum-bound raw fixture tensor."""

    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FixtureAuthority:
    """Exact immutable authority shared by the mandatory fixtures."""

    fixture_root: Path
    fixture_set_sha256: str
    root_ledger_sha256: str
    source_manifest_sha256: str
    environment_manifest_sha256: str
    source_commit: str
    source_tree: str

    def to_dict(self) -> dict[str, str]:
        return {
            "fixture_root": str(self.fixture_root),
            "fixture_set_sha256": self.fixture_set_sha256,
            "root_ledger_sha256": self.root_ledger_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "environment_manifest_sha256": self.environment_manifest_sha256,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
        }


@dataclass(frozen=True, slots=True)
class TurboQuantFixture:
    """Parsed frozen inputs and expected outputs for one configuration."""

    configuration: str
    authority: FixtureAuthority
    geometry: Mapping[str, int | str]
    inputs: Mapping[str, FixtureTensorRecord]
    outputs: Mapping[str, FixtureTensorRecord]
    slot_size: int
    byte_breakdown_per_head_token: Mapping[str, int]

    def input_record(self, name: str) -> FixtureTensorRecord:
        try:
            return self.inputs[name]
        except KeyError as error:
            raise TurboQuantFixtureError(
                f"unknown TurboQuant fixture input: {name}"
            ) from error

    def output_record(self, name: str) -> FixtureTensorRecord:
        try:
            return self.outputs[name]
        except KeyError as error:
            raise TurboQuantFixtureError(
                f"unknown TurboQuant fixture output: {name}"
            ) from error


@dataclass(frozen=True, slots=True)
class ExactPackedComparison:
    """Untimed exact-byte result for one packed cache artifact."""

    passed: bool
    output_name: str
    shape_matches: bool
    dtype_matches: bool
    expected_nbytes: int
    observed_nbytes: int
    expected_sha256: str
    observed_sha256: str

    def to_dict(self) -> dict[str, bool | int | str]:
        return {
            "passed": self.passed,
            "output_name": self.output_name,
            "shape_matches": self.shape_matches,
            "dtype_matches": self.dtype_matches,
            "expected_nbytes": self.expected_nbytes,
            "observed_nbytes": self.observed_nbytes,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
        }


@dataclass(frozen=True, slots=True)
class DecodeFixtureComparison:
    """Untimed finite/tolerance result for the frozen decode output."""

    passed: bool
    finite: bool
    shape_matches: bool
    dtype_matches: bool
    max_absolute_error: float | None
    max_relative_error: float | None
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, bool | float | None]:
        return {
            "passed": self.passed,
            "finite": self.finite,
            "shape_matches": self.shape_matches,
            "dtype_matches": self.dtype_matches,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass(frozen=True, slots=True)
class SlotLayoutComparison:
    """Exact source-derived per-head/token layout comparison."""

    passed: bool
    expected_slot_size: int
    observed_slot_size: int
    expected_breakdown: Mapping[str, int]
    observed_breakdown: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "expected_slot_size": self.expected_slot_size,
            "observed_slot_size": self.observed_slot_size,
            "expected_breakdown": dict(self.expected_breakdown),
            "observed_breakdown": dict(self.observed_breakdown),
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise TurboQuantFixtureError(
                "PyTorch is required to load TurboQuant tensor fixtures"
            ) from error
    return _TORCH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise TurboQuantFixtureError(f"missing fixture file: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise TurboQuantFixtureError(f"unsafe fixture file: {path}")
    return path


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise TurboQuantFixtureError(f"unsafe fixture path: {relative!r}")
    return _require_file(root.joinpath(*pure.parts))


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise TurboQuantFixtureError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> None:
    raise TurboQuantFixtureError(f"non-finite JSON value: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            _require_file(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TurboQuantFixtureError(f"invalid fixture JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TurboQuantFixtureError(f"fixture JSON root is not an object: {path}")
    return payload


def _validate_reference(fixture_root: Path) -> None:
    try:
        validator = importlib.import_module(
            "reference.turboquant.validate_fixtures"
        )
        result = validator.validate_reference(fixture_root)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise TurboQuantFixtureError(
            f"frozen fixture validation failed: {error}"
        ) from error
    if (
        result.get("status") != "pass"
        or result.get("mandatory_fixture_count") != 3
        or result.get("timing_data_present") is not False
        or result.get("performance_claim_eligible") is not False
    ):
        raise TurboQuantFixtureError("frozen fixture validation result differs")


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise TurboQuantFixtureError(f"invalid tensor shape: {name}")
    return tuple(value)


def _tensor_record(
    *,
    name: str,
    base: Path,
    value: Any,
) -> FixtureTensorRecord:
    if not isinstance(value, dict):
        raise TurboQuantFixtureError(f"missing tensor record: {name}")
    shape = _shape(value.get("shape"), name)
    dtype = value.get("dtype")
    nbytes = value.get("nbytes")
    digest = value.get("sha256")
    item_size = {"bfloat16": 2, "uint8": 1}.get(dtype)
    expected_nbytes = item_size
    if expected_nbytes is None:
        raise TurboQuantFixtureError(f"unsupported fixture dtype: {name}")
    for dimension in shape:
        expected_nbytes *= dimension
    if (
        type(nbytes) is not int
        or nbytes != expected_nbytes
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise TurboQuantFixtureError(f"invalid tensor identity: {name}")
    path = _safe_file(base, str(value.get("path")))
    if path.stat().st_size != nbytes or _sha256(path) != digest:
        raise TurboQuantFixtureError(f"tensor identity mismatch: {name}")
    return FixtureTensorRecord(
        name=name,
        path=path,
        dtype=dtype,
        shape=shape,
        nbytes=nbytes,
        sha256=digest,
    )


def _load_authority(
    fixture_root: Path,
) -> tuple[FixtureAuthority, dict[str, Any], Mapping[str, Mapping[str, Any]]]:
    if not fixture_root.is_dir() or fixture_root.is_symlink():
        raise TurboQuantFixtureError("fixture root is missing or unsafe")
    _validate_reference(fixture_root)
    if _sha256(_safe_file(fixture_root, "fixture_set.json")) != FIXTURE_SET_SHA256:
        raise TurboQuantFixtureError("fixture-set identity differs")
    if (
        _sha256(_safe_file(fixture_root, "checksums.sha256"))
        != FIXTURE_ROOT_LEDGER_SHA256
    ):
        raise TurboQuantFixtureError("fixture root-ledger identity differs")
    if _sha256(SOURCE_MANIFEST_PATH) != SOURCE_MANIFEST_SHA256:
        raise TurboQuantFixtureError("source-manifest identity differs")
    if _sha256(ENVIRONMENT_MANIFEST_PATH) != ENVIRONMENT_MANIFEST_SHA256:
        raise TurboQuantFixtureError("environment-manifest identity differs")

    fixture_set = _load_json(fixture_root / "fixture_set.json")
    source_manifest = _load_json(SOURCE_MANIFEST_PATH)
    source = source_manifest.get("source")
    if not isinstance(source, dict) or (
        source.get("commit"),
        source.get("tree"),
    ) != (PINNED_SOURCE_COMMIT, PINNED_SOURCE_TREE):
        raise TurboQuantFixtureError("pinned source authority differs")
    if (
        fixture_set.get("source_commit") != PINNED_SOURCE_COMMIT
        or fixture_set.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or fixture_set.get("environment_manifest_sha256")
        != ENVIRONMENT_MANIFEST_SHA256
        or fixture_set.get("geometry") != dict(_EXPECTED_GEOMETRY)
        or fixture_set.get("mandatory_configurations")
        != list(MANDATORY_CONFIGURATIONS)
    ):
        raise TurboQuantFixtureError("frozen fixture-set contract differs")
    inputs = fixture_set.get("inputs")
    if not isinstance(inputs, dict) or {
        name: value.get("sha256")
        for name, value in inputs.items()
        if isinstance(value, dict)
    } != _EXPECTED_INPUT_SHA256:
        raise TurboQuantFixtureError("frozen input checksums differ")

    configuration_values = source_manifest.get("configurations")
    if not isinstance(configuration_values, list):
        raise TurboQuantFixtureError("source configurations are absent")
    configurations = {
        value["cache_dtype"]: value
        for value in configuration_values
        if isinstance(value, dict) and isinstance(value.get("cache_dtype"), str)
    }
    if not all(name in configurations for name in MANDATORY_CONFIGURATIONS):
        raise TurboQuantFixtureError("mandatory source configurations differ")
    authority = FixtureAuthority(
        fixture_root=fixture_root,
        fixture_set_sha256=FIXTURE_SET_SHA256,
        root_ledger_sha256=FIXTURE_ROOT_LEDGER_SHA256,
        source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        environment_manifest_sha256=ENVIRONMENT_MANIFEST_SHA256,
        source_commit=PINNED_SOURCE_COMMIT,
        source_tree=PINNED_SOURCE_TREE,
    )
    return authority, fixture_set, configurations


def _load_configuration(
    *,
    authority: FixtureAuthority,
    fixture_set: Mapping[str, Any],
    source_configuration: Mapping[str, Any],
    configuration: str,
) -> TurboQuantFixture:
    config_root = authority.fixture_root / configuration
    manifest = _load_json(config_root / "manifest.json")
    source = manifest.get("source")
    if (
        manifest.get("configuration") != dict(source_configuration)
        or manifest.get("geometry") != dict(_EXPECTED_GEOMETRY)
        or not isinstance(source, dict)
        or (
            source.get("commit"),
            source.get("tree"),
            source.get("manifest_sha256"),
        )
        != (
            PINNED_SOURCE_COMMIT,
            PINNED_SOURCE_TREE,
            SOURCE_MANIFEST_SHA256,
        )
    ):
        raise TurboQuantFixtureError(
            f"fixture authority differs: {configuration}"
        )
    input_values = fixture_set["inputs"]
    inputs = {
        name: _tensor_record(
            name=name,
            base=authority.fixture_root,
            value=value,
        )
        for name, value in input_values.items()
    }
    layout = manifest.get("layout")
    slot_size = MANDATORY_CONFIG_SLOT_SIZES[configuration]
    expected_breakdown = _EXPECTED_LAYOUTS[configuration]
    if not isinstance(layout, dict) or (
        layout.get("slot_size"),
        layout.get("slot_size_aligned"),
        layout.get("byte_breakdown_per_head_token"),
        layout.get("storage_shape"),
    ) != (
        slot_size,
        slot_size,
        dict(expected_breakdown),
        [2, 16, 8, slot_size],
    ):
        raise TurboQuantFixtureError(
            f"fixture slot layout differs: {configuration}"
        )
    output_values = manifest.get("outputs")
    if not isinstance(output_values, dict):
        raise TurboQuantFixtureError(f"fixture outputs absent: {configuration}")
    outputs = {
        name: _tensor_record(
            name=name,
            base=config_root,
            value=output_values.get(name),
        )
        for name in (
            "cache_after_store",
            "cache_after_append",
            "append_slot",
            "decode_output",
        )
    }
    if (
        outputs["cache_after_store"].shape != (2, 16, 8, slot_size)
        or outputs["cache_after_append"].shape != (2, 16, 8, slot_size)
        or outputs["append_slot"].shape != (8, slot_size)
        or outputs["decode_output"].shape != (1, 32, 128)
    ):
        raise TurboQuantFixtureError(
            f"fixture output geometry differs: {configuration}"
        )
    return TurboQuantFixture(
        configuration=configuration,
        authority=authority,
        geometry=dict(_EXPECTED_GEOMETRY),
        inputs=inputs,
        outputs=outputs,
        slot_size=slot_size,
        byte_breakdown_per_head_token=dict(expected_breakdown),
    )


def load_mandatory_fixtures(
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
) -> tuple[TurboQuantFixture, ...]:
    """Validate frozen authority and load all mandatory fixture records."""

    authority, fixture_set, configurations = _load_authority(fixture_root)
    return tuple(
        _load_configuration(
            authority=authority,
            fixture_set=fixture_set,
            source_configuration=configurations[configuration],
            configuration=configuration,
        )
        for configuration in MANDATORY_CONFIGURATIONS
    )


def load_turboquant_fixture(
    configuration: str,
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
) -> TurboQuantFixture:
    """Load one mandatory fixture after validating the complete root."""

    if configuration not in MANDATORY_CONFIGURATIONS:
        raise TurboQuantFixtureError(
            f"unsupported mandatory fixture configuration: {configuration}"
        )
    return next(
        fixture
        for fixture in load_mandatory_fixtures(fixture_root)
        if fixture.configuration == configuration
    )


def load_tensor_cpu(record: FixtureTensorRecord) -> Any:
    """Load one checksum-verified raw tensor on CPU."""

    raw = _require_file(record.path).read_bytes()
    if len(raw) != record.nbytes or hashlib.sha256(raw).hexdigest() != record.sha256:
        raise TurboQuantFixtureError(
            f"fixture tensor changed after validation: {record.name}"
        )
    torch = _torch()
    dtype = torch.bfloat16 if record.dtype == "bfloat16" else torch.uint8
    return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(
        record.shape
    )


def load_inputs_cpu(fixture: TurboQuantFixture) -> dict[str, Any]:
    """Load every frozen input tensor on CPU."""

    return {
        name: load_tensor_cpu(record)
        for name, record in fixture.inputs.items()
    }


def _tensor_bytes_untimed(tensor: Any) -> bytes:
    torch = _torch()
    byte_view = tensor.detach().contiguous().view(-1).view(torch.uint8).cpu()
    return byte_view.numpy().tobytes(order="C")


def compare_packed_output_untimed(
    fixture: TurboQuantFixture,
    output_name: str,
    observed: Any,
) -> ExactPackedComparison:
    """Compare packed store, append, or appended-slot bytes exactly."""

    if output_name not in {
        "cache_after_store",
        "cache_after_append",
        "append_slot",
    }:
        raise TurboQuantFixtureError(
            f"not a packed fixture output: {output_name}"
        )
    record = fixture.output_record(output_name)
    observed_raw = _tensor_bytes_untimed(observed)
    expected_raw = _require_file(record.path).read_bytes()
    shape_matches = tuple(observed.shape) == record.shape
    dtype_matches = str(observed.dtype) == "torch.uint8"
    return ExactPackedComparison(
        passed=(
            shape_matches
            and dtype_matches
            and len(observed_raw) == record.nbytes
            and observed_raw == expected_raw
        ),
        output_name=output_name,
        shape_matches=shape_matches,
        dtype_matches=dtype_matches,
        expected_nbytes=record.nbytes,
        observed_nbytes=len(observed_raw),
        expected_sha256=record.sha256,
        observed_sha256=hashlib.sha256(observed_raw).hexdigest(),
    )


def compare_decode_output_untimed(
    fixture: TurboQuantFixture,
    observed: Any,
) -> DecodeFixtureComparison:
    """Compare decode output with the predeclared Phase 6 tolerance."""

    record = fixture.output_record("decode_output")
    shape_matches = tuple(observed.shape) == record.shape
    dtype_matches = str(observed.dtype) == "torch.bfloat16"
    if not shape_matches:
        return DecodeFixtureComparison(
            passed=False,
            finite=False,
            shape_matches=False,
            dtype_matches=dtype_matches,
            max_absolute_error=None,
            max_relative_error=None,
            atol=DECODE_ATOL,
            rtol=DECODE_RTOL,
        )
    comparison = compare_tensors_untimed(
        observed.detach().cpu(),
        load_tensor_cpu(record),
        atol=DECODE_ATOL,
        rtol=DECODE_RTOL,
    )
    return DecodeFixtureComparison(
        passed=dtype_matches and comparison.passed,
        finite=comparison.finite,
        shape_matches=True,
        dtype_matches=dtype_matches,
        max_absolute_error=comparison.max_absolute_error,
        max_relative_error=comparison.max_relative_error,
        atol=comparison.atol,
        rtol=comparison.rtol,
    )


def compare_slot_layout(
    fixture: TurboQuantFixture,
    *,
    slot_size: int,
    byte_breakdown_per_head_token: Mapping[str, int],
) -> SlotLayoutComparison:
    """Compare source-derived slot size and component bytes exactly."""

    expected = dict(fixture.byte_breakdown_per_head_token)
    observed = dict(byte_breakdown_per_head_token)
    return SlotLayoutComparison(
        passed=slot_size == fixture.slot_size and observed == expected,
        expected_slot_size=fixture.slot_size,
        observed_slot_size=slot_size,
        expected_breakdown=expected,
        observed_breakdown=observed,
    )
