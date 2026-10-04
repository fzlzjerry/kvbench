"""Strict loading for JSON-compatible YAML configuration documents."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeAlias

from kvbench.errors import ConfigLoadError, SchemaValidationError
from kvbench.schema import (
    DocumentType,
    ExperimentConfig,
    HardwareManifest,
    MethodConfig,
    MethodName,
    ModelIdentity,
    ModelIdentityV2,
    Phase3AdmissionPlan,
    ResolutionState,
    StrictModel,
    canonical_json_bytes,
    sha256_hex,
)
from kvbench.schema.phase3 import (
    PHASE3_BF16_CONFIG_FINGERPRINT,
    PHASE3_E00_MANIFEST_SHA256,
    PHASE3_E00_RUN_ID,
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_HARDWARE_FINGERPRINT,
    PHASE3_HARDWARE_ID,
    PHASE3_MODEL_FINGERPRINT,
    PHASE3_PLAN_FINGERPRINTS,
)


MAX_CONFIG_BYTES = 2 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ConfigDocument: TypeAlias = (
    HardwareManifest
    | ModelIdentity
    | ModelIdentityV2
    | MethodConfig
    | ExperimentConfig
    | Phase3AdmissionPlan
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigLoadError("duplicate object key is not allowed")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ConfigLoadError("non-finite JSON numbers are not allowed")


def load_json_compatible_yaml(path: Path) -> dict[str, Any]:
    """Load the documented dependency-free YAML 1.2 JSON subset."""

    try:
        size = path.stat().st_size
    except OSError as error:
        raise ConfigLoadError("configuration path is not readable") from error
    if size > MAX_CONFIG_BYTES:
        raise ConfigLoadError("configuration exceeds the 2 MiB safety limit")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigLoadError("configuration must be readable UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except ConfigLoadError:
        raise
    except json.JSONDecodeError as error:
        raise ConfigLoadError(
            f"configuration is not valid JSON-compatible YAML at line {error.lineno}"
        ) from error
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ConfigLoadError("configuration root must be an object")
    return value


def parse_config(value: dict[str, Any]) -> ConfigDocument:
    """Dispatch a strict configuration document by its declared type."""

    raw_type = value.get("document_type")
    try:
        document_type = DocumentType(raw_type)
    except (TypeError, ValueError) as error:
        raise SchemaValidationError(
            "missing or invalid document_type", path=("document_type",)
        ) from error
    schema_version = value.get("schema_version")
    versioned_models: dict[
        tuple[DocumentType, object], type[StrictModel]
    ] = {
        (DocumentType.HARDWARE, HardwareManifest.SCHEMA_VERSION): HardwareManifest,
        (DocumentType.MODEL, ModelIdentity.SCHEMA_VERSION): ModelIdentity,
        (DocumentType.MODEL, ModelIdentityV2.SCHEMA_VERSION): ModelIdentityV2,
        (DocumentType.METHOD, MethodConfig.SCHEMA_VERSION): MethodConfig,
        (DocumentType.EXPERIMENT, ExperimentConfig.SCHEMA_VERSION): ExperimentConfig,
        (
            DocumentType.EXPERIMENT,
            Phase3AdmissionPlan.SCHEMA_VERSION,
        ): Phase3AdmissionPlan,
    }
    model = versioned_models.get((document_type, schema_version))
    if model is None:
        raise SchemaValidationError(
            "unsupported schema_version for document_type",
            path=("schema_version",),
        )
    return model.from_dict(value)


def load_config(path: str | Path) -> ConfigDocument:
    return parse_config(load_json_compatible_yaml(Path(path)))


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentBundle:
    """A plan and all referenced typed identities, loaded as one contract."""

    plan_path: Path
    plan: ExperimentConfig | Phase3AdmissionPlan
    hardware: HardwareManifest
    model: ModelIdentity | ModelIdentityV2
    methods: tuple[MethodConfig, ...]
    canonical_fingerprints: tuple[tuple[str, str], ...]
    blockers: tuple[str, ...]
    retained_open_blockers: tuple[str, ...] = ()

    @property
    def execution_ready(self) -> bool:
        if isinstance(self.plan, Phase3AdmissionPlan):
            # Decision 0007 authorizes only this exact native-host admission
            # exception.  The formal B-010/E02 blockers remain visible in
            # all_blockers and in the method fingerprint; they are not closed.
            return (
                self.blockers == ("B-010", "E02")
                and self.retained_open_blockers == ("B-009", "B-010")
            )
        return not self.blockers

    @property
    def all_blockers(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.blockers) | set(self.retained_open_blockers)))


def _resolve_reference(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=False)
    resolved_root = root.resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):
        raise ConfigLoadError("configuration reference escapes repository root")
    return candidate


def _load_reference(
    root: Path,
    relative: str,
    expected_sha256: str | None,
) -> tuple[ConfigDocument, str]:
    document = load_config(_resolve_reference(root, relative))
    fingerprint = sha256_hex(canonical_json_bytes(document))
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise SchemaValidationError(
            "referenced configuration fingerprint does not match",
            path=(relative,),
        )
    return document, fingerprint


def load_experiment_bundle(
    plan_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> ExperimentBundle:
    """Load a plan and verify every typed repository-relative reference."""

    root = Path(repository_root).resolve(strict=True)
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    resolved_plan_path = path.resolve(strict=True)
    if not resolved_plan_path.is_relative_to(root):
        raise ConfigLoadError("experiment plan must be tracked inside the repository")
    plan_document = load_config(resolved_plan_path)
    if not isinstance(plan_document, (ExperimentConfig, Phase3AdmissionPlan)):
        raise SchemaValidationError("--plan must reference an experiment document")

    hardware_document, hardware_sha = _load_reference(
        root, plan_document.hardware.path, plan_document.hardware.sha256
    )
    if not isinstance(hardware_document, HardwareManifest):
        raise SchemaValidationError("hardware reference has the wrong document type")
    model_document, model_sha = _load_reference(
        root, plan_document.model.path, plan_document.model.sha256
    )
    if not isinstance(model_document, (ModelIdentity, ModelIdentityV2)):
        raise SchemaValidationError("model reference has the wrong document type")

    methods: list[MethodConfig] = []
    fingerprints: list[tuple[str, str]] = [
        (plan_document.hardware.path, hardware_sha),
        (plan_document.model.path, model_sha),
    ]
    blockers = set(plan_document.resolution.blockers)
    blockers.update(plan_document.software_environment.resolution.blockers)
    if isinstance(plan_document, ExperimentConfig):
        blockers.update(plan_document.admission.blockers)
        blockers.update(plan_document.grid.resolution.blockers)
        blockers.update(plan_document.measurement.resolution.blockers)
    blockers.update(hardware_document.resolution.blockers)
    blockers.update(model_document.resolution.blockers)
    for selection in plan_document.methods:
        loaded, fingerprint = _load_reference(
            root, selection.config.path, selection.config.sha256
        )
        if not isinstance(loaded, MethodConfig):
            raise SchemaValidationError("method reference has the wrong document type")
        known_variants = {variant.variant_id for variant in loaded.variants}
        missing = sorted(set(selection.variants) - known_variants)
        if missing:
            raise SchemaValidationError(
                "selected variant is absent from its method config",
                path=(selection.config.path, "variants"),
            )
        methods.append(loaded)
        fingerprints.append((selection.config.path, fingerprint))
        blockers.update(loaded.resolution.blockers)
        blockers.update(loaded.implementation.resolution.blockers)
        for variant in loaded.variants:
            if variant.variant_id in selection.variants:
                blockers.update(variant.resolution.blockers)

    for reference in (
        plan_document.experiment_contract,
        plan_document.measurement_protocol,
    ):
        referenced_path = _resolve_reference(root, reference.path)
        try:
            raw = referenced_path.read_bytes()
        except OSError as error:
            raise ConfigLoadError("referenced contract is not readable") from error
        digest = sha256_hex(raw)
        if reference.sha256 is not None and digest != reference.sha256:
            raise SchemaValidationError(
                "referenced contract fingerprint does not match",
                path=(reference.path,),
            )
        fingerprints.append((reference.path, digest))

    if (
        isinstance(plan_document, ExperimentConfig)
        and plan_document.resolution.status is ResolutionState.RESOLVED
        and blockers
    ):
        raise SchemaValidationError("resolved plan cannot retain admission blockers")
    return ExperimentBundle(
        plan_path=resolved_plan_path,
        plan=plan_document,
        hardware=hardware_document,
        model=model_document,
        methods=tuple(methods),
        canonical_fingerprints=tuple(sorted(fingerprints)),
        blockers=tuple(sorted(blockers)),
    )


def load_phase3_admission_bundle(
    plan_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> ExperimentBundle:
    """Load a Phase 3 plan and all references without admitting execution."""

    bundle = load_experiment_bundle(plan_path, repository_root=repository_root)
    if not isinstance(bundle.plan, Phase3AdmissionPlan):
        raise SchemaValidationError("--plan must reference a Phase 3 admission plan")
    root = Path(repository_root).resolve(strict=True)
    relative_plan_path = bundle.plan_path.relative_to(root).as_posix()
    if relative_plan_path not in {
        PHASE3_FIXED_PLAN_PATH,
        PHASE3_GROWING_PLAN_PATH,
    }:
        raise SchemaValidationError("Phase 3 plan path is outside the frozen pair")
    expected_plan_fingerprint = PHASE3_PLAN_FINGERPRINTS[relative_plan_path]
    if bundle.plan.fingerprint() != expected_plan_fingerprint:
        raise SchemaValidationError("Phase 3 plan fingerprint does not match the frozen plan")
    if not isinstance(bundle.model, ModelIdentityV2):
        raise SchemaValidationError("Phase 3 requires the exact v2 model identity")
    if bundle.model.fingerprint() != PHASE3_MODEL_FINGERPRINT:
        raise SchemaValidationError("Phase 3 model reference is not the frozen checkpoint")
    hardware = bundle.hardware
    hardware_identity = (
        hardware.hardware_id,
        hardware.fingerprint(),
        hardware.expected_gpu_family,
        hardware.expected_gpu_count,
        hardware.tensor_parallel_size,
        hardware.exclusive_gpu,
        hardware.e00_run_id,
        hardware.e00_manifest_sha256,
        hardware.g0_status,
        hardware.container_parity_status,
        hardware.container_parity_blocker,
    )
    expected_hardware_identity = (
        PHASE3_HARDWARE_ID,
        PHASE3_HARDWARE_FINGERPRINT,
        "RTX_PRO_6000_BLACKWELL",
        1,
        1,
        True,
        PHASE3_E00_RUN_ID,
        PHASE3_E00_MANIFEST_SHA256,
        "PASS",
        "not_evaluated",
        "B-010",
    )
    if hardware_identity != expected_hardware_identity:
        raise SchemaValidationError("Phase 3 hardware is not the certified native host")
    if len(bundle.methods) != 1:
        raise SchemaValidationError("Phase 3 requires exactly one BF16 method config")
    method = bundle.methods[0]
    method_identity = (
        method.method_config_id,
        method.method,
        tuple(variant.variant_id for variant in method.variants),
    )
    if method_identity != ("bf16", MethodName.BF16, ("bf16",)):
        raise SchemaValidationError("Phase 3 method reference is not exact BF16")
    fingerprints = dict(bundle.canonical_fingerprints)
    if (
        fingerprints.get("configs/methods/bf16.yaml")
        != PHASE3_BF16_CONFIG_FINGERPRINT
    ):
        raise SchemaValidationError("Phase 3 BF16 config fingerprint is not frozen")
    return dataclasses.replace(
        bundle,
        retained_open_blockers=bundle.plan.admission.retained_open_blockers,
    )


def validate_all_example_configs(
    *, repository_root: str | Path = REPOSITORY_ROOT
) -> tuple[tuple[str, str], ...]:
    """Validate every versioned example and return stable fingerprints."""

    root = Path(repository_root).resolve(strict=True)
    paths = sorted((root / "configs").glob("*/*.yaml"))
    if not paths:
        raise ConfigLoadError("no example configurations were found")
    results: list[tuple[str, str]] = []
    for path in paths:
        document = load_config(path)
        if isinstance(document, (ExperimentConfig, Phase3AdmissionPlan)):
            load_experiment_bundle(path, repository_root=root)
        results.append((path.relative_to(root).as_posix(), document.fingerprint()))
    return tuple(results)
