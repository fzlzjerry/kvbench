"""Strict compact reference to the sole completed Phase 9 calibration."""

from __future__ import annotations

import dataclasses

from kvbench.schema.base import (
    StrictModel,
    require_git_sha,
    require_identifier,
    require_relative_path,
    require_sha256,
)


@dataclasses.dataclass(frozen=True, slots=True)
class KVQuantCalibrationReference(StrictModel):
    method_identifier: str
    calibration_id: str
    local_bundle_path: str
    local_compact_manifest_path: str
    source_base_commit: str
    source_base_tree: str
    patch_sha256: str
    patched_commit: str
    patched_tree: str
    model_revision: str
    tokenizer_revision: str
    dataset_revision: str
    dataset_conversion_revision: str
    dataset_root: str
    token_tensor_root: str
    fisher_root: str
    kvq4_sha256: str
    kvq3_sha256: str
    kvq2_sha256: str
    sink_tokens: int
    key_outlier_cap: int
    value_outlier_cap: int
    outlier_value_dtype: str
    outlier_index_dtype: str
    metadata_dtype: str
    durable_r2_uri: str
    calibration_root_digest: str

    def __post_init__(self) -> None:
        require_identifier(
            self.method_identifier,
            field_name="method_identifier",
        )
        require_identifier(self.calibration_id, field_name="calibration_id")
        require_relative_path(
            self.local_bundle_path,
            field_name="local_bundle_path",
        )
        require_relative_path(
            self.local_compact_manifest_path,
            field_name="local_compact_manifest_path",
        )
        for name, value in (
            ("source_base_commit", self.source_base_commit),
            ("source_base_tree", self.source_base_tree),
            ("patched_commit", self.patched_commit),
            ("patched_tree", self.patched_tree),
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("dataset_revision", self.dataset_revision),
            ("dataset_conversion_revision", self.dataset_conversion_revision),
        ):
            require_git_sha(value)
        for name, value in (
            ("patch_sha256", self.patch_sha256),
            ("dataset_root", self.dataset_root),
            ("token_tensor_root", self.token_tensor_root),
            ("fisher_root", self.fisher_root),
            ("kvq4_sha256", self.kvq4_sha256),
            ("kvq3_sha256", self.kvq3_sha256),
            ("kvq2_sha256", self.kvq2_sha256),
            ("calibration_root_digest", self.calibration_root_digest),
        ):
            require_sha256(value, field_name=name)
        if self.method_identifier != "kvquant_gqa_upstream_patch_v1":
            raise ValueError("KVQuant calibration method identifier drifted")
        if (
            self.source_base_commit,
            self.source_base_tree,
            self.patch_sha256,
            self.patched_commit,
            self.patched_tree,
            self.model_revision,
            self.tokenizer_revision,
            self.dataset_revision,
            self.dataset_conversion_revision,
        ) != (
            "57a238357f0ffe50084670fcd5781c9848f80ea2",
            "094e0f736f77ee327e5350cbd1eefb1c936aa77b",
            (
                "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c"
                "61895598d1d482d6"
            ),
            "4ad80bc8c942d0a05516d2be8f8d443a77a05900",
            "c4f1490c9c0c4ec46099f1e95c092516df2adb4e",
            "0e9e39f249a16976918f6564b8830bc894c89659",
            "0e9e39f249a16976918f6564b8830bc894c89659",
            "b08601e04326c79dfdd32d625aee71d232d685c3",
            "3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e",
        ):
            raise ValueError("KVQuant calibration authority drifted")
        if self.local_bundle_path != (
            f"calibration/kvquant/{self.calibration_id}"
        ):
            raise ValueError("KVQuant local bundle path is not calibration-bound")
        if (
            self.local_compact_manifest_path
            != "docs/evidence/phase9/calibration-manifest.json"
        ):
            raise ValueError("KVQuant compact manifest path drifted")
        if (
            self.sink_tokens,
            self.key_outlier_cap,
            self.value_outlier_cap,
        ) != (5, 12, 12):
            raise ValueError("KVQuant sink or fixed sparse cap drifted")
        if (
            self.outlier_value_dtype,
            self.outlier_index_dtype,
            self.metadata_dtype,
        ) != ("float32", "int32", "float32"):
            raise ValueError("KVQuant calibration metadata dtype drifted")
        expected_uri = (
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{self.calibration_root_digest}/"
        )
        if self.durable_r2_uri != expected_uri:
            raise ValueError("KVQuant durable R2 URI is not root-bound")
