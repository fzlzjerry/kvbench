"""Strict lifecycle manifest for the single Phase 9 KVQuant calibration."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

from kvbench.schema.base import (
    RunStatus,
    StrictModel,
    require_git_sha,
    require_identifier,
    require_oci_digest,
    require_run_id,
    require_schema,
    require_sha256,
)
from kvbench.schema.phase6 import _validate_lifecycle


METHOD_IDENTIFIER = "kvquant_gqa_upstream_patch_v1"
UPSTREAM_REPOSITORY = "https://github.com/SqueezeAILab/KVQuant.git"
UPSTREAM_BASE_COMMIT = "57a238357f0ffe50084670fcd5781c9848f80ea2"
UPSTREAM_BASE_TREE = "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
PATCH_SHA256 = "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6"
PATCHED_COMMIT = "4ad80bc8c942d0a05516d2be8f8d443a77a05900"
PATCHED_TREE = "c4f1490c9c0c4ec46099f1e95c092516df2adb4e"
CALIBRATION_IMAGE_DIGEST = (
    "sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d"
)
CALIBRATION_DOCKERFILE_SHA256 = (
    "e3ac0933c21c986bed2ca169c8983f6d1e6412e02bed42a282f9c604fd9c4de5"
)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_SNAPSHOT_MANIFEST_SHA256 = (
    "ab9f6a32a41934c9e49881db68022827b6aca35f4f644627c77e3420978d1336"
)
DATASET_REPOSITORY = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONVERSION_REVISION = "3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e"
DATASET_CONTENT_SHA256 = (
    "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
)


@dataclasses.dataclass(frozen=True, slots=True)
class Phase9CalibrationManifest(StrictModel):
    """Append-only envelope for one exact offline calibration attempt."""

    schema_version: str
    artifact_schema_version: str
    run_id: str
    status: RunStatus
    created_at_utc: str
    started_at_utc: str | None
    finished_at_utc: str | None
    phase: Literal["phase9_kvquant_calibration"]
    run_kind: Literal["offline_calibration"]
    claim_class: Literal["none"]
    git_sha: str
    git_dirty: bool
    container_digest: str
    dockerfile_sha256: str
    method_identifier: str
    upstream_repository: str
    upstream_base_commit: str
    upstream_base_tree: str
    patch_sha256: str
    patched_commit: str
    patched_tree: str
    decision: Literal["0021"]
    source_reconstruction_command: str
    source_reconstruction_passed: bool
    reconstructed_source_checksum_result: Literal["PASS"]
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    model_snapshot_manifest_sha256: str
    dataset_repository: str
    dataset_revision: str
    dataset_conversion_revision: str
    dataset_content_sha256: str
    dataset_split: Literal["train"]
    number_of_examples: int
    sequence_length: int
    random_seed: int
    attempt_sequence: int
    calibration_contract_sha256: str
    command_argv: tuple[str, ...]
    performance_measurement: bool
    profiler_execution: bool
    quality_evaluation: bool
    quality_execution: Literal["locked"]
    performance_data_frozen: bool
    full_scan_state: Literal["closed"]
    phase10_started: bool
    inventory_path: str | None
    failure_reason: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase9-calibration-manifest-1.0.0"
    ARTIFACT_SCHEMA_VERSION: ClassVar[str] = "kvbench-artifacts-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_schema(
            self.artifact_schema_version,
            self.ARTIFACT_SCHEMA_VERSION,
        )
        require_run_id(self.run_id)
        require_git_sha(self.git_sha)
        require_oci_digest(self.container_digest)
        require_sha256(self.dockerfile_sha256, field_name="dockerfile_sha256")
        require_sha256(
            self.model_snapshot_manifest_sha256,
            field_name="model_snapshot_manifest_sha256",
        )
        require_sha256(
            self.dataset_content_sha256,
            field_name="dataset_content_sha256",
        )
        require_sha256(
            self.calibration_contract_sha256,
            field_name="calibration_contract_sha256",
        )
        require_identifier(self.method_identifier, field_name="method_identifier")

        if self.git_dirty:
            raise ValueError("Phase 9 calibration requires a clean execution tree")
        if self.container_digest != CALIBRATION_IMAGE_DIGEST:
            raise ValueError("Phase 9 calibration image identity drifted")
        if self.dockerfile_sha256 != CALIBRATION_DOCKERFILE_SHA256:
            raise ValueError("Phase 9 calibration Dockerfile identity drifted")
        if (
            self.method_identifier,
            self.upstream_repository,
            self.upstream_base_commit,
            self.upstream_base_tree,
            self.patch_sha256,
            self.patched_commit,
            self.patched_tree,
        ) != (
            METHOD_IDENTIFIER,
            UPSTREAM_REPOSITORY,
            UPSTREAM_BASE_COMMIT,
            UPSTREAM_BASE_TREE,
            PATCH_SHA256,
            PATCHED_COMMIT,
            PATCHED_TREE,
        ):
            raise ValueError("Phase 9 patched-upstream authority drifted")
        if not self.source_reconstruction_passed:
            raise ValueError("Phase 9 requires exact source reconstruction")
        if (
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.tokenizer_revision,
            self.model_snapshot_manifest_sha256,
        ) != (
            MODEL_ID,
            MODEL_REVISION,
            MODEL_ID,
            MODEL_REVISION,
            MODEL_SNAPSHOT_MANIFEST_SHA256,
        ):
            raise ValueError("Phase 9 model or tokenizer authority drifted")
        if (
            self.dataset_repository,
            self.dataset_revision,
            self.dataset_conversion_revision,
            self.dataset_content_sha256,
        ) != (
            DATASET_REPOSITORY,
            DATASET_REVISION,
            DATASET_CONVERSION_REVISION,
            DATASET_CONTENT_SHA256,
        ):
            raise ValueError("Phase 9 WikiText-2 train authority drifted")
        if (
            self.number_of_examples != 16
            or self.sequence_length != 2048
            or self.random_seed != 20260721
            or self.attempt_sequence != 1
        ):
            raise ValueError("Phase 9 sample, shape, seed, or attempt drifted")
        if self.command_argv != ("make", "calibrate-kvquant"):
            raise ValueError("Phase 9 lifecycle command must remain exact")
        if any(
            (
                self.performance_measurement,
                self.profiler_execution,
                self.quality_evaluation,
                self.performance_data_frozen,
                self.phase10_started,
            )
        ):
            raise ValueError("Phase 9 calibration crossed a forbidden boundary")

        _validate_lifecycle(
            status=self.status,
            created_at_utc=self.created_at_utc,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            inventory_path=self.inventory_path,
            failure_reason=self.failure_reason,
        )


def parse_phase9_calibration_manifest(
    payload: dict[str, object],
) -> Phase9CalibrationManifest:
    return Phase9CalibrationManifest.from_dict(payload)
