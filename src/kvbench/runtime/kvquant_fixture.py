"""Read-only access to the corrected Phase 11P-R KVQuant fixture oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from kvbench.runtime.numerical import NumericalComparison, compare_tensors_untimed


REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
KVQUANT_FIXTURE_ROOT: Final[Path] = (
    REPOSITORY_ROOT / "reference" / "kvquant_phase11pr" / "fixtures"
)
KVQUANT_HISTORICAL_FIXTURE_ROOT: Final[Path] = (
    REPOSITORY_ROOT / "reference" / "kvquant" / "fixtures"
)

KVQUANT_FIXTURE_ID: Final[str] = "kvqref-2e0a0e9022c50cbc6fb497d88cae973e"
KVQUANT_FIXTURE_ROOT_SHA256: Final[str] = (
    "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
)
KVQUANT_HISTORICAL_FIXTURE_ID: Final[str] = (
    "kvqref-a50af6511c314b6394e58a7f81ceefb8"
)
KVQUANT_HISTORICAL_ROOT_SHA256: Final[str] = (
    "32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab"
)
KVQUANT_CALIBRATION_ID: Final[str] = (
    "kvqcal-cdb724c806d64d095c040d2673a987a3"
)
KVQUANT_CALIBRATION_ROOT_SHA256: Final[str] = (
    "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
)
KVQUANT_METHOD_IDENTIFIER: Final[str] = "kvquant_gqa_upstream_patch_v1"
KVQUANT_EXECUTION_SOURCE_IDENTIFIER: Final[str] = (
    "kvquant_gqa_graphsafe_kvq3_v2"
)
KVQUANT_UPSTREAM_BASE_COMMIT: Final[str] = (
    "57a238357f0ffe50084670fcd5781c9848f80ea2"
)
KVQUANT_UPSTREAM_BASE_TREE: Final[str] = (
    "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
)
KVQUANT_DECISION_0021_PATCH_SHA256: Final[str] = (
    "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6"
)
KVQUANT_PHASE10_PATCHED_COMMIT: Final[str] = (
    "4ad80bc8c942d0a05516d2be8f8d443a77a05900"
)
KVQUANT_PHASE10_PATCHED_TREE: Final[str] = (
    "c4f1490c9c0c4ec46099f1e95c092516df2adb4e"
)
KVQUANT_AGGREGATE_PATCH_SHA256: Final[str] = (
    "23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551"
)
KVQUANT_CORRECTED_COMMIT: Final[str] = (
    "0d9df350bd1788284e1ce76a8bf6e886beca5efa"
)
KVQUANT_CORRECTED_TREE: Final[str] = (
    "a85cf7bf093982a4bf89c33d4e6794d9a85f846d"
)
KVQUANT_EXTENSION_SHA256: Final[str] = (
    "46c41aad8f56d58608d4c1273bd3a72fd36c8f69f9ca2c5a046f0c811631bf51"
)

KVQUANT_FAMILIES: Final[tuple[str, ...]] = ("kvq4", "kvq3", "kvq2")
KVQUANT_CASES: Final[Mapping[str, int]] = {
    "key_zero_value_fixed12": 0,
    "key_few_value_fixed12": 6,
    "key_cap_value_fixed12": 12,
}
KVQUANT_BIT_WIDTHS: Final[Mapping[str, int]] = {
    "kvq4": 4,
    "kvq3": 3,
    "kvq2": 2,
}
KVQUANT_TENSOR_FILES: Final[tuple[str, ...]] = (
    "append_state.safetensors",
    "decode_output.safetensors",
    "dense_payload.safetensors",
    "inputs.safetensors",
    "metadata.safetensors",
    "sink.safetensors",
    "sparse_indices.safetensors",
    "sparse_values.safetensors",
    "store_state.safetensors",
)

KVQUANT_DECODE_ATOL: Final[float] = 0.01
KVQUANT_DECODE_RTOL: Final[float] = 0.01
KVQUANT_KEY_LOGITS_ATOL: Final[float] = 0.25
KVQUANT_KEY_LOGITS_RTOL: Final[float] = 0.01
_SUPPORTED_DTYPES: Final[frozenset[str]] = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int32",
        "int64",
    }
)


class KVQuantFixtureError(RuntimeError):
    """The corrected fixture authority or one tensor payload is invalid."""


@dataclass(frozen=True, slots=True)
class KVQuantFixtureAuthority:
    """Checksum-bound authority shared by the mixed-provenance fixtures."""

    fixture_root: Path
    fixture_id: str
    root_sha256: str
    historical_root_sha256: str
    calibration_id: str
    calibration_root_sha256: str
    method_identifier: str
    execution_source_identifier: str
    upstream_base_commit: str
    upstream_base_tree: str
    decision_0021_patch_sha256: str
    aggregate_patch_sha256: str
    corrected_commit: str
    corrected_tree: str
    extension_sha256: str


@dataclass(frozen=True, slots=True)
class KVQuantTensorRecord:
    """One tensor record bound to an immutable safetensors member."""

    file_name: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    logical_nbytes: int
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class KVQuantFixture:
    """One verified family/case fixture; tensor loading is untimed."""

    authority: KVQuantFixtureAuthority
    family: str
    case_name: str
    bit_width: int
    key_active_count: int
    fixture_id: str
    fixture_root: Path
    manifest: Mapping[str, Any]
    byte_breakdown: Mapping[str, Any]
    tensor_records: Mapping[str, Mapping[str, KVQuantTensorRecord]]

    def tensor_record(self, file_name: str, tensor_name: str) -> KVQuantTensorRecord:
        try:
            return self.tensor_records[file_name][tensor_name]
        except KeyError as error:
            raise KVQuantFixtureError(
                f"unknown fixture tensor: {file_name}:{tensor_name}"
            ) from error


@dataclass(frozen=True, slots=True)
class KVQuantExactComparison:
    """Untimed exact comparison for one adapter-owned tensor."""

    passed: bool
    file_name: str
    tensor_name: str
    shape_matches: bool
    dtype_matches: bool
    expected_sha256: str
    observed_sha256: str


@dataclass(frozen=True, slots=True)
class KVQuantToleranceComparison:
    """Untimed shape/dtype/finite/tolerance comparison."""

    passed: bool
    shape_matches: bool
    dtype_matches: bool
    comparison: NumericalComparison | None
    atol: float
    rtol: float


def _validator() -> Any:
    try:
        return importlib.import_module(
            "reference.kvquant_phase11pr.validate_corrected_bundle"
        )
    except ImportError as error:
        raise KVQuantFixtureError(
            "corrected fixture validator is unavailable"
        ) from error


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KVQuantFixtureError(f"invalid fixture JSON: {path}") from error
    if not isinstance(payload, dict):
        raise KVQuantFixtureError(f"fixture JSON is not an object: {path}")
    return payload


def _torch_dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _tensor_payload_untimed(tensor: Any) -> bytes:
    torch = importlib.import_module("torch")
    contiguous = (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .to(device="cpu", copy=True)
    )
    byte_count = int(contiguous.numel())
    return bytes(contiguous.untyped_storage())[:byte_count]


def _load_tensor_file_cpu(
    path: Path,
    *,
    expected_fixture_id: str,
) -> Mapping[str, Any]:
    try:
        safetensors = importlib.import_module("safetensors")
    except ModuleNotFoundError:
        try:
            return _validator()._read_safetensors(path, expected_fixture_id)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise KVQuantFixtureError(
                f"cannot load safetensors member: {path}"
            ) from error

    safe_open = getattr(safetensors, "safe_open", None)
    if safe_open is None:
        raise KVQuantFixtureError("safetensors.safe_open is unavailable")
    result: dict[str, Any] = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if handle.metadata() != {
                "fixture_id": expected_fixture_id,
                "format": "kvbench-phase10-kvquant-reference",
            }:
                raise KVQuantFixtureError(
                    f"safetensors metadata mismatch: {path}"
                )
            for name in sorted(handle.keys()):
                result[name] = handle.get_tensor(name).clone()
    except (OSError, RuntimeError, ValueError) as error:
        raise KVQuantFixtureError(
            f"cannot load safetensors member: {path}"
        ) from error
    return result


def _tensor_record(
    file_name: str,
    tensor_name: str,
    value: Any,
) -> KVQuantTensorRecord:
    if not isinstance(value, dict) or set(value) != {
        "dtype",
        "shape",
        "logical_nbytes",
        "payload_sha256",
    }:
        raise KVQuantFixtureError(
            f"malformed tensor record: {file_name}:{tensor_name}"
        )
    dtype = value["dtype"]
    shape = value["shape"]
    logical_nbytes = value["logical_nbytes"]
    payload_sha256 = value["payload_sha256"]
    if (
        dtype not in _SUPPORTED_DTYPES
        or not isinstance(shape, list)
        or any(type(item) is not int or item < 0 for item in shape)
        or type(logical_nbytes) is not int
        or logical_nbytes < 0
        or not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
    ):
        raise KVQuantFixtureError(
            f"invalid tensor identity: {file_name}:{tensor_name}"
        )
    return KVQuantTensorRecord(
        file_name=file_name,
        name=tensor_name,
        dtype=dtype,
        shape=tuple(shape),
        logical_nbytes=logical_nbytes,
        payload_sha256=payload_sha256,
    )


def _validate_tensor_file(
    fixture_root: Path,
    file_name: str,
    *,
    fixture_id: str,
    declared: Any,
) -> Mapping[str, KVQuantTensorRecord]:
    if not isinstance(declared, dict):
        raise KVQuantFixtureError(f"missing tensor records: {file_name}")
    tensors = _load_tensor_file_cpu(
        fixture_root / file_name,
        expected_fixture_id=fixture_id,
    )
    if set(tensors) != set(declared):
        raise KVQuantFixtureError(f"tensor name set mismatch: {file_name}")
    records: dict[str, KVQuantTensorRecord] = {}
    for name, tensor in tensors.items():
        record = _tensor_record(file_name, name, declared[name])
        payload = _tensor_payload_untimed(tensor)
        if (
            tuple(tensor.shape) != record.shape
            or _torch_dtype_name(tensor) != record.dtype
            or len(payload) != record.logical_nbytes
            or hashlib.sha256(payload).hexdigest() != record.payload_sha256
        ):
            raise KVQuantFixtureError(
                f"tensor identity mismatch: {file_name}:{name}"
            )
        records[name] = record
    return records


def _fixture(
    authority: KVQuantFixtureAuthority,
    family: str,
    case_name: str,
) -> KVQuantFixture:
    root = authority.fixture_root / family / case_name
    manifest = _load_json(root / "fixture_manifest.json")
    bit_width = KVQUANT_BIT_WIDTHS[family]
    key_count = KVQUANT_CASES[case_name]
    fixture_id = (
        KVQUANT_FIXTURE_ID
        if family == "kvq3"
        else KVQUANT_HISTORICAL_FIXTURE_ID
    )
    if (
        manifest.get("fixture_id") != fixture_id
        or manifest.get("family") != family
        or manifest.get("case") != case_name
        or manifest.get("bit_width") != bit_width
        or manifest.get("status") != "PASS"
    ):
        raise KVQuantFixtureError(
            f"fixture identity mismatch: {family}/{case_name}"
        )

    calibration = manifest.get("calibration")
    sparse = manifest.get("sparse_contract")
    numerical = manifest.get("numerical_control")
    semantics = manifest.get("semantics")
    if (
        not isinstance(calibration, dict)
        or calibration.get("calibration_id") != KVQUANT_CALIBRATION_ID
        or calibration.get("root_sha256")
        != KVQUANT_CALIBRATION_ROOT_SHA256
        or calibration.get("fisher_regenerated") is not False
        or calibration.get("quantizer_regenerated") is not False
        or not isinstance(sparse, dict)
        or (
            sparse.get("key_active_count"),
            sparse.get("key_capacity"),
            sparse.get("value_active_count_non_sink"),
            sparse.get("value_active_count_sink"),
            sparse.get("value_capacity"),
            sparse.get("outlier_value_dtype"),
            sparse.get("outlier_index_dtype"),
        )
        != (key_count, 12, 12, 0, 12, "float32", "int32")
        or not isinstance(numerical, dict)
        or (
            numerical.get("decode_atol"),
            numerical.get("decode_rtol"),
            numerical.get("key_logits_atol"),
            numerical.get("key_logits_rtol"),
        )
        != (
            KVQUANT_DECODE_ATOL,
            KVQUANT_DECODE_RTOL,
            KVQUANT_KEY_LOGITS_ATOL,
            KVQUANT_KEY_LOGITS_RTOL,
        )
        or not isinstance(semantics, dict)
        or semantics.get("quantized_key") != "pre_rope_k_proj_output"
        or semantics.get("sink_key_stored")
        != "post_rope_attention_ready_fp16"
        or semantics.get("implementation_head_expansion") is not False
    ):
        raise KVQuantFixtureError(
            f"fixture numerical contract mismatch: {family}/{case_name}"
        )

    source = manifest.get("source")
    expected_source = (
        (
            "0025",
            KVQUANT_AGGREGATE_PATCH_SHA256,
            KVQUANT_CORRECTED_COMMIT,
            KVQUANT_CORRECTED_TREE,
        )
        if family == "kvq3"
        else (
            "0021",
            KVQUANT_DECISION_0021_PATCH_SHA256,
            KVQUANT_PHASE10_PATCHED_COMMIT,
            KVQUANT_PHASE10_PATCHED_TREE,
        )
    )
    if (
        not isinstance(source, dict)
        or source.get("method_identifier") != KVQUANT_METHOD_IDENTIFIER
        or source.get("contract_decision") != "0023"
        or (
            source.get("decision"),
            source.get("patch_sha256"),
            source.get("patched_commit"),
            source.get("patched_tree"),
        )
        != expected_source
    ):
        raise KVQuantFixtureError(
            f"fixture source provenance mismatch: {family}/{case_name}"
        )

    declared = manifest.get("tensor_records")
    if not isinstance(declared, dict) or set(declared) != set(
        KVQUANT_TENSOR_FILES
    ):
        raise KVQuantFixtureError(
            f"fixture tensor file set mismatch: {family}/{case_name}"
        )
    records = {
        file_name: _validate_tensor_file(
            root,
            file_name,
            fixture_id=fixture_id,
            declared=declared[file_name],
        )
        for file_name in KVQUANT_TENSOR_FILES
    }
    return KVQuantFixture(
        authority=authority,
        family=family,
        case_name=case_name,
        bit_width=bit_width,
        key_active_count=key_count,
        fixture_id=fixture_id,
        fixture_root=root,
        manifest=manifest,
        byte_breakdown=_load_json(root / "byte_breakdown.json"),
        tensor_records=records,
    )


def load_all_kvquant_fixtures(
    fixture_root: Path = KVQUANT_FIXTURE_ROOT,
    historical_fixture_root: Path = KVQUANT_HISTORICAL_FIXTURE_ROOT,
) -> tuple[KVQuantFixture, ...]:
    """Validate the complete root, then expose all nine fixtures."""

    try:
        result = _validator().validate_corrected_bundle(
            fixture_root,
            historical_fixture_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise KVQuantFixtureError(
            f"corrected fixture root validation failed: {error}"
        ) from error
    if (
        result.get("status") != "PASS"
        or result.get("fixture_id") != KVQUANT_FIXTURE_ID
        or result.get("local_root_sha256") != KVQUANT_FIXTURE_ROOT_SHA256
        or result.get("old_root_sha256")
        != KVQUANT_HISTORICAL_ROOT_SHA256
        or result.get("fixture_count") != 9
        or result.get("regenerated_family") != "kvq3"
        or result.get("scalar_control") != "PASS"
        or result.get("calibration_changed") is not False
    ):
        raise KVQuantFixtureError("corrected fixture validation result differs")
    authority = KVQuantFixtureAuthority(
        fixture_root=fixture_root,
        fixture_id=KVQUANT_FIXTURE_ID,
        root_sha256=KVQUANT_FIXTURE_ROOT_SHA256,
        historical_root_sha256=KVQUANT_HISTORICAL_ROOT_SHA256,
        calibration_id=KVQUANT_CALIBRATION_ID,
        calibration_root_sha256=KVQUANT_CALIBRATION_ROOT_SHA256,
        method_identifier=KVQUANT_METHOD_IDENTIFIER,
        execution_source_identifier=KVQUANT_EXECUTION_SOURCE_IDENTIFIER,
        upstream_base_commit=KVQUANT_UPSTREAM_BASE_COMMIT,
        upstream_base_tree=KVQUANT_UPSTREAM_BASE_TREE,
        decision_0021_patch_sha256=KVQUANT_DECISION_0021_PATCH_SHA256,
        aggregate_patch_sha256=KVQUANT_AGGREGATE_PATCH_SHA256,
        corrected_commit=KVQUANT_CORRECTED_COMMIT,
        corrected_tree=KVQUANT_CORRECTED_TREE,
        extension_sha256=KVQUANT_EXTENSION_SHA256,
    )
    return tuple(
        _fixture(authority, family, case_name)
        for family in KVQUANT_FAMILIES
        for case_name in KVQUANT_CASES
    )


def load_kvquant_fixture(
    family: str,
    case_name: str,
    *,
    fixture_root: Path = KVQUANT_FIXTURE_ROOT,
    historical_fixture_root: Path = KVQUANT_HISTORICAL_FIXTURE_ROOT,
) -> KVQuantFixture:
    """Load one fixture only after validating the complete mixed root."""

    if family not in KVQUANT_FAMILIES or case_name not in KVQUANT_CASES:
        raise KVQuantFixtureError("unsupported KVQuant fixture family/case")
    return next(
        fixture
        for fixture in load_all_kvquant_fixtures(
            fixture_root,
            historical_fixture_root,
        )
        if fixture.family == family and fixture.case_name == case_name
    )


def load_fixture_tensor_file_untimed(
    fixture: KVQuantFixture,
    file_name: str,
    *,
    device: Any = "cpu",
) -> Mapping[str, Any]:
    """Load one verified member outside measured execution."""

    if file_name not in fixture.tensor_records:
        raise KVQuantFixtureError(f"unknown fixture tensor file: {file_name}")
    tensors = _load_tensor_file_cpu(
        fixture.fixture_root / file_name,
        expected_fixture_id=fixture.fixture_id,
    )
    expected = fixture.tensor_records[file_name]
    if set(tensors) != set(expected):
        raise KVQuantFixtureError(f"loaded tensor name set mismatch: {file_name}")
    result: dict[str, Any] = {}
    for name, tensor in tensors.items():
        record = expected[name]
        payload = _tensor_payload_untimed(tensor)
        if (
            tuple(tensor.shape) != record.shape
            or _torch_dtype_name(tensor) != record.dtype
            or len(payload) != record.logical_nbytes
            or hashlib.sha256(payload).hexdigest() != record.payload_sha256
        ):
            raise KVQuantFixtureError(
                f"loaded tensor identity mismatch: {file_name}:{name}"
            )
        result[name] = tensor.to(device=device)
    return result


def compare_exact_fixture_tensor_untimed(
    fixture: KVQuantFixture,
    file_name: str,
    tensor_name: str,
    observed: Any,
) -> KVQuantExactComparison:
    """Compare one adapter-owned tensor exactly after measured execution."""

    record = fixture.tensor_record(file_name, tensor_name)
    payload = _tensor_payload_untimed(observed)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    shape_matches = tuple(observed.shape) == record.shape
    dtype_matches = _torch_dtype_name(observed) == record.dtype
    return KVQuantExactComparison(
        passed=(
            shape_matches
            and dtype_matches
            and len(payload) == record.logical_nbytes
            and observed_sha256 == record.payload_sha256
        ),
        file_name=file_name,
        tensor_name=tensor_name,
        shape_matches=shape_matches,
        dtype_matches=dtype_matches,
        expected_sha256=record.payload_sha256,
        observed_sha256=observed_sha256,
    )


def _compare_tolerance_untimed(
    fixture: KVQuantFixture,
    observed: Any,
    *,
    tensor_name: str,
    atol: float,
    rtol: float,
) -> KVQuantToleranceComparison:
    record = fixture.tensor_record("decode_output.safetensors", tensor_name)
    shape_matches = tuple(observed.shape) == record.shape
    dtype_matches = _torch_dtype_name(observed) == record.dtype
    if not shape_matches:
        return KVQuantToleranceComparison(
            passed=False,
            shape_matches=False,
            dtype_matches=dtype_matches,
            comparison=None,
            atol=atol,
            rtol=rtol,
        )
    expected = load_fixture_tensor_file_untimed(
        fixture,
        "decode_output.safetensors",
    )[tensor_name]
    comparison = compare_tensors_untimed(
        observed.detach().cpu(),
        expected,
        atol=atol,
        rtol=rtol,
    )
    return KVQuantToleranceComparison(
        passed=dtype_matches and comparison.passed,
        shape_matches=True,
        dtype_matches=dtype_matches,
        comparison=comparison,
        atol=atol,
        rtol=rtol,
    )


def compare_decode_output_untimed(
    fixture: KVQuantFixture,
    observed: Any,
) -> KVQuantToleranceComparison:
    """Compare decode output with the frozen 0.01/0.01 tolerance."""

    return _compare_tolerance_untimed(
        fixture,
        observed,
        tensor_name="source_decode_output",
        atol=KVQUANT_DECODE_ATOL,
        rtol=KVQUANT_DECODE_RTOL,
    )


def compare_key_logits_untimed(
    fixture: KVQuantFixture,
    observed: Any,
) -> KVQuantToleranceComparison:
    """Compare non-sink Key logits with the frozen 0.25/0.01 tolerance."""

    return _compare_tolerance_untimed(
        fixture,
        observed,
        tensor_name="source_nonsink_key_logits",
        atol=KVQUANT_KEY_LOGITS_ATOL,
        rtol=KVQUANT_KEY_LOGITS_RTOL,
    )
