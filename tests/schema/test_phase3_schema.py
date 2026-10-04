"""Strict Phase 3 schema, provenance-join, and frozen-grid tests."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

from kvbench.config import (
    load_config,
    load_json_compatible_yaml,
    load_phase3_admission_bundle,
    parse_config,
)
from kvbench.errors import SchemaValidationError
from kvbench.schema import (
    BF16BackendIdentity,
    BF16CacheIdentity,
    FROZEN_PHASE3_POINT_IDS,
    FROZEN_PHASE3_STABILITY_POINT_IDS,
    G1_CRITERIA,
    GateDisposition,
    ModelIdentityV2,
    Phase3AdmissionPlan,
    Phase3CommandSpec,
    Phase3G1AdmissionReport,
    Phase3RunManifest,
    Phase3WorkerResult,
    RunManifest,
    RunStatus,
    SourceDigest,
    derive_cache_layout_fingerprint,
    derive_phase3_point_fingerprint,
    expand_phase3_process_points,
    parse_run_manifest,
)
from kvbench.schema.phase3 import (
    PHASE3_BF16_VARIANT_FINGERPRINT,
    PHASE3_CONTRACT_FINGERPRINT,
    PHASE3_DRIVER_VERSION,
    PHASE3_E00_MANIFEST_SHA256,
    PHASE3_E00_RUN_ID,
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GPU_FULL_NAME,
    PHASE3_GPU_UUID,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_HARDWARE_FINGERPRINT,
    PHASE3_HARDWARE_ID,
    PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT,
    PHASE3_PCI_BUS_ID,
    PHASE3_PCI_DEVICE_ID,
    PHASE3_PLAN_FINGERPRINTS,
    PHASE3_PYTHON_EXECUTABLE,
    PHASE3_REPOSITORY_ROOT,
    PHASE3_SOFTWARE_ENVIRONMENT_ID,
    PHASE3_SOFTWARE_FINGERPRINT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "configs/models/primary_gqa_model.yaml"
FIXED_PLAN = REPOSITORY_ROOT / PHASE3_FIXED_PLAN_PATH
GROWING_PLAN = REPOSITORY_ROOT / PHASE3_GROWING_PLAN_PATH
ZERO_SHA256 = "0" * 64
ONE_SHA256 = "1" * 64
TWO_SHA256 = "2" * 64
THREE_SHA256 = "3" * 64
ZERO_GIT_SHA = "0" * 40
CREATED_AT = "2026-07-22T00:00:00Z"


def quality_payload() -> dict[str, object]:
    return {
        "schema_version": "kvbench.quality-status.v1",
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "quality_execution": "locked",
        "performance_data_frozen": False,
    }


def backend_payload() -> dict[str, object]:
    sources = {
        "include/ATen/native/transformers/cuda/flash_attn/flash_api.h": (
            "1474aa79d8aa6ce39984dbc3c0aad9dba283ab819f034370e5cfb70980524ee7"
        ),
        "lib/libtorch_cuda.so": (
            "b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984"
        ),
        "nn/attention/__init__.py": (
            "56e10b6f965cc050db782dd4dc472097c9b02ec5b5fe3ab2c8b04055c0b0bbe0"
        ),
        "nn/attention/varlen.py": (
            "2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea"
        ),
        "nn/functional.py": (
            "27493186ee22f811b553e31d9c804d4d46716d1be62d034d731537f66f27ef19"
        ),
    }
    return {
        "schema_version": "kvbench.phase3-bf16-backend.v1",
        "backend_id": "torch_sdpa_flash_gqa",
        "torch_version": "2.12.1+cu130",
        "torch_git_sha": "7269437d655783a26cba32aa88195b741ff496aa",
        "cuda_runtime_version": "13.0",
        "cudnn_version": "9.20.0",
        "triton_version": "3.7.1",
        "flash_generation": "FA2",
        "flash_version": "2.5.7",
        "dispatch_api": "torch.nn.functional.scaled_dot_product_attention",
        "selected_backend": "flash_attention",
        "enable_gqa": True,
        "compile_mode": "disabled",
        "source_artifacts": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(sources.items())
        ],
    }


def cache_payload(*, capacity: int = 129, batch_size: int = 1) -> dict[str, object]:
    implementation_sha256 = ZERO_SHA256
    storage_bytes = 2 * 32 * batch_size * 8 * capacity * 128 * 2
    fingerprint = derive_cache_layout_fingerprint(
        num_layers=32,
        batch_size=batch_size,
        num_kv_heads=8,
        capacity=capacity,
        head_dim=128,
        device="cuda:0",
        workspace_bytes=0,
        implementation_sha256=implementation_sha256,
    )
    return {
        "schema_version": "kvbench.phase3-bf16-cache.v1",
        "layout_name": "layers_batch_kv_heads_context_head_dim",
        "dtype": "bfloat16",
        "num_layers": 32,
        "batch_size": batch_size,
        "num_kv_heads": 8,
        "capacity": capacity,
        "head_dim": 128,
        "tensor_storage_bytes": storage_bytes,
        "padding_bytes": 0,
        "workspace_bytes": 0,
        "device": "cuda:0",
        "implementation_sha256": implementation_sha256,
        "layout_fingerprint": fingerprint,
    }


def created_phase3_manifest(run_id: str = "phase3-schema-fixture") -> dict[str, object]:
    model = load_config(MODEL_PATH)
    assert isinstance(model, ModelIdentityV2)
    backend = BF16BackendIdentity.from_dict(backend_payload())
    point_id = "fixed_l-b1-l128-eager-r1"
    plan_fingerprint = PHASE3_PLAN_FINGERPRINTS[PHASE3_FIXED_PLAN_PATH]
    return {
        "schema_version": "kvbench-phase3-run-manifest-1.0.0",
        "artifact_schema_version": "kvbench-artifacts-1.0.0",
        "run_id": run_id,
        "status": "created",
        "created_at_utc": CREATED_AT,
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_kind": "phase3_admission",
        "runner_kind": "fixed_l",
        "graph_mode": "eager",
        "claim_class": "none",
        "measurement_scope": "native_host_admission",
        "performance_claim_eligible": False,
        "plan_source": {
            "kind": "path",
            "path": PHASE3_FIXED_PLAN_PATH,
            "canonical_inline_json": None,
            "sha256": plan_fingerprint,
        },
        "plan_fingerprint": plan_fingerprint,
        "point_id": point_id,
        "point_fingerprint": derive_phase3_point_fingerprint(point_id),
        "git_sha": ZERO_GIT_SHA,
        "git_dirty": False,
        "container_digest": None,
        "hardware_id": PHASE3_HARDWARE_ID,
        "hardware_fingerprint": PHASE3_HARDWARE_FINGERPRINT,
        "native_g0_status": "PASS",
        "e00_run_id": PHASE3_E00_RUN_ID,
        "e00_manifest_sha256": PHASE3_E00_MANIFEST_SHA256,
        "blocker_b010": "OPEN",
        "gpu_uuid": PHASE3_GPU_UUID,
        "gpu_full_name": PHASE3_GPU_FULL_NAME,
        "pci_bus_id": PHASE3_PCI_BUS_ID,
        "pci_device_id": PHASE3_PCI_DEVICE_ID,
        "driver_version": PHASE3_DRIVER_VERSION,
        "software_environment_id": PHASE3_SOFTWARE_ENVIRONMENT_ID,
        "software_fingerprint": PHASE3_SOFTWARE_FINGERPRINT,
        "model_identity": model.to_dict(),
        "model_fingerprint": model.fingerprint(),
        "method": "bf16",
        "method_config_id": "bf16",
        "method_config_fingerprint": {
            "schema_version": "kvbench.method-fingerprint.v1",
            "method": "bf16",
            "variant_id": "bf16",
            "canonicalization": "kvbench-json-v1",
            "algorithm": "sha256",
            "sha256": PHASE3_BF16_VARIANT_FINGERPRINT,
            "execution_ready": False,
        },
        "contract_fingerprint": PHASE3_CONTRACT_FINGERPRINT,
        "measurement_protocol_fingerprint": (
            PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT
        ),
        "backend_identity": backend.to_dict(),
        "backend_fingerprint": backend.fingerprint(),
        "cache_identity": cache_payload(),
        "batch_size": 1,
        "context_length": 128,
        "output_steps": 1,
        "warmup_count": 16,
        "measured_count": 32,
        "measured_batches": 5,
        "count_unit": "decode_operations",
        "random_seed": 20260722,
        "process_replicate": 1,
        "quality": quality_payload(),
        "command": {
            "schema_version": "kvbench-phase3-command-1.0.0",
            "argv": [
                PHASE3_PYTHON_EXECUTABLE,
                "-m",
                "kvbench",
                "phase3-worker",
                "--plan",
                PHASE3_FIXED_PLAN_PATH,
                "--point-id",
                point_id,
                "--replicate",
                "1",
                "--run-id",
                run_id,
            ],
            "working_directory": PHASE3_REPOSITORY_ROOT,
            "environment_sha256": ZERO_SHA256,
            "dry_run": False,
        },
        "inventory_path": None,
        "failure_reason": None,
    }


def legacy_v1_manifest() -> dict[str, object]:
    plan_path = "configs/plans/smoke.yaml"
    return {
        "schema_version": "kvbench-run-manifest-1.0.0",
        "artifact_schema_version": "kvbench-artifacts-1.0.0",
        "run_id": "legacy-v1-fixture",
        "status": "created",
        "created_at_utc": CREATED_AT,
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_kind": "synthetic",
        "runner_kind": "fixed_l",
        "graph_mode": "eager",
        "claim_class": "none",
        "plan_source": {
            "kind": "path",
            "path": plan_path,
            "canonical_inline_json": None,
            "sha256": ONE_SHA256,
        },
        "git_sha": ZERO_GIT_SHA,
        "git_dirty": True,
        "container_digest": None,
        "hardware_id": "synthetic-hardware",
        "hardware_fingerprint": ZERO_SHA256,
        "software_environment_id": "synthetic-software",
        "software_fingerprint": ONE_SHA256,
        "model_id": "synthetic/model",
        "model_revision": "unresolved-synthetic-revision",
        "model_fingerprint": TWO_SHA256,
        "method": "bf16",
        "method_config_id": "bf16-placeholder",
        "method_config_fingerprint": {
            "schema_version": "kvbench.method-fingerprint.v1",
            "method": "bf16",
            "variant_id": "bf16",
            "canonicalization": "kvbench-json-v1",
            "algorithm": "sha256",
            "sha256": ZERO_SHA256,
            "execution_ready": False,
        },
        "contract_fingerprint": THREE_SHA256,
        "attention_backend": None,
        "cache_layout": None,
        "random_seed": 20260722,
        "process_replicate": 1,
        "quality": {
            "schema_version": "kvbench.quality-status.v1",
            "quality_status": "not_applicable",
            "claim_eligibility": "none",
            "quality_execution": "locked",
            "performance_data_frozen": False,
        },
        "command": {
            "schema_version": "kvbench-command-1.0.0",
            "argv": ["kvbench", "run", "--plan", plan_path, "--dry-run"],
            "dry_run": True,
        },
        "inventory_path": None,
        "failure_reason": None,
    }


def _expected_criterion_points(criterion: str) -> tuple[str, ...]:
    if criterion == "fixed_l_runner":
        return tuple(
            point for point in FROZEN_PHASE3_POINT_IDS if point.startswith("fixed_l-")
        )
    if criterion == "growing_context_runner":
        return tuple(
            point
            for point in FROZEN_PHASE3_POINT_IDS
            if point.startswith("growing_context-")
        )
    if criterion == "eager_lane":
        return tuple(point for point in FROZEN_PHASE3_POINT_IDS if "-eager-" in point)
    if criterion in {
        "cuda_graph_capture_and_replay",
        "eager_graph_numerical_agreement",
        "graph_replay_no_allocation",
    }:
        return tuple(
            point for point in FROZEN_PHASE3_POINT_IDS if "-cuda_graph-" in point
        )
    if criterion in {
        "independent_process_replicates",
        "stability_threshold",
    }:
        return FROZEN_PHASE3_STABILITY_POINT_IDS
    return FROZEN_PHASE3_POINT_IDS


def passing_g1_report() -> dict[str, object]:
    evidence: list[dict[str, object]] = []
    run_id_by_point: dict[str, str] = {}
    for index, point_id in enumerate(FROZEN_PHASE3_POINT_IDS):
        run_id = f"phase3-evidence-{index:02d}"
        run_id_by_point[point_id] = run_id
        plan_path = (
            PHASE3_GROWING_PLAN_PATH
            if point_id.startswith("growing_context-")
            else PHASE3_FIXED_PLAN_PATH
        )
        evidence.append(
            {
                "run_id": run_id,
                "point_id": point_id,
                "point_fingerprint": derive_phase3_point_fingerprint(point_id),
                "plan_path": plan_path,
                "plan_fingerprint": PHASE3_PLAN_FINGERPRINTS[plan_path],
                "status": "completed",
                "manifest_sha256": ZERO_SHA256,
                "artifact_inventory_sha256": ONE_SHA256,
                "checksum_ledger_sha256": TWO_SHA256,
                "checksum_valid": True,
            }
        )
    criteria = [
        {
            "criterion": criterion,
            "disposition": "PASS",
            "evidence_run_ids": [
                run_id_by_point[point_id]
                for point_id in _expected_criterion_points(criterion)
            ],
            "reason": None,
        }
        for criterion in G1_CRITERIA
    ]
    stability_summaries: list[dict[str, object]] = []
    for graph_mode in ("eager", "cuda_graph"):
        point_ids = tuple(
            point
            for point in FROZEN_PHASE3_STABILITY_POINT_IDS
            if f"-{graph_mode}-" in point
        )
        stability_summaries.append(
            {
                "graph_mode": graph_mode,
                "point_ids": list(point_ids),
                "evidence_run_ids": [run_id_by_point[point] for point in point_ids],
                "process_replicates": 3,
                "process_median_host_wall_ms": [1.0, 1.0, 1.0],
                "median_host_wall_ms": 1.0,
                "minimum_host_wall_ms": 1.0,
                "maximum_host_wall_ms": 1.0,
                "coefficient_of_variation_percent": 0.0,
                "temperature_min_c": 40.0,
                "temperature_max_c": 52.0,
                "sm_clock_min_mhz": 1800,
                "sm_clock_max_mhz": 2100,
                "power_min_w": 100.0,
                "power_max_w": 250.0,
                "summary_artifact_path": f"stability/{graph_mode}.json",
                "summary_artifact_sha256": THREE_SHA256,
            }
        )
    return {
        "schema_version": "kvbench-phase3-g1-admission-report-1.0.0",
        "generated_at_utc": CREATED_AT,
        "git_sha": ZERO_GIT_SHA,
        "status": "PASS",
        "g0": "PASS",
        "g1": "PASS",
        "g2": "NOT_EVALUATED",
        "g3": "NOT_EVALUATED",
        "g4": "NOT_EVALUATED",
        "g5": "NOT_EVALUATED",
        "full_scan_state": "closed",
        "quality": quality_payload(),
        "quality_benchmark_executed": False,
        "quality_only_dependencies_installed": False,
        "measurement_scope": "native_host_admission",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "blocker_b009": "OPEN",
        "blocker_b010": "OPEN",
        "expected_process_count": 20,
        "plan_sources": [
            {"path": path, "sha256": fingerprint}
            for path, fingerprint in PHASE3_PLAN_FINGERPRINTS.items()
        ],
        "run_evidence": evidence,
        "stability_summaries": stability_summaries,
        "criteria": criteria,
        "all_artifacts_checksum_valid": True,
        "formal_paper_claim_generated": False,
    }


class ModelIdentityV2Tests(unittest.TestCase):
    def test_exact_model_identity_and_all_artifacts_parse(self) -> None:
        model = load_config(MODEL_PATH)
        self.assertIsInstance(model, ModelIdentityV2)
        assert isinstance(model, ModelIdentityV2)
        self.assertEqual(model.parameter_count, 8_030_261_248)
        self.assertEqual(len(model.artifacts), 11)
        self.assertEqual(model.geometry.num_query_heads, 32)
        self.assertEqual(model.geometry.num_kv_heads, 8)
        self.assertEqual(model.rope.rope_type, "llama3")
        self.assertTrue(model.local_snapshot_path.startswith("/root/.cache/"))

    def test_model_byte_or_semantic_substitution_is_rejected(self) -> None:
        raw = load_json_compatible_yaml(MODEL_PATH)
        mutations = (
            lambda item: item.__setitem__("model_id", "meta-llama/Llama-3.1-8B"),
            lambda item: item.__setitem__("local_snapshot_path", "/tmp/substitute"),
            lambda item: item["rope"].__setitem__("theta", 10000.0),
            lambda item: item["artifacts"][0].__setitem__(
                "sha256", ZERO_SHA256
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(raw)
                mutation(changed)
                with self.assertRaises(SchemaValidationError):
                    parse_config(changed)


class FrozenPlanTests(unittest.TestCase):
    def test_exact_plans_expand_to_twenty_unique_processes(self) -> None:
        fixed = load_phase3_admission_bundle(FIXED_PLAN)
        growing = load_phase3_admission_bundle(GROWING_PLAN)
        self.assertIsInstance(fixed.plan, Phase3AdmissionPlan)
        self.assertIsInstance(growing.plan, Phase3AdmissionPlan)
        fixed_points = expand_phase3_process_points(fixed.plan)
        growing_points = expand_phase3_process_points(growing.plan)
        self.assertEqual(len(fixed_points), 16)
        self.assertEqual(len(growing_points), 4)
        self.assertEqual(
            tuple(point.point_id for point in (*fixed_points, *growing_points)),
            FROZEN_PHASE3_POINT_IDS,
        )
        stability = tuple(
            point.point_id
            for point in fixed_points
            if point.stability_member
        )
        self.assertEqual(stability, FROZEN_PHASE3_STABILITY_POINT_IDS)

    def test_bundle_exposes_execution_and_retained_governance_blockers(self) -> None:
        bundle = load_phase3_admission_bundle(FIXED_PLAN)
        self.assertTrue(bundle.execution_ready)
        self.assertEqual(bundle.retained_open_blockers, ("B-009", "B-010"))
        self.assertEqual(bundle.blockers, ("B-010", "E02"))
        self.assertTrue({"B-009", "B-010", "E02"}.issubset(
            set(bundle.all_blockers)
        ))
        self.assertIsInstance(bundle.model, ModelIdentityV2)
        self.assertEqual(bundle.hardware.g0_status, "PASS")
        self.assertEqual(bundle.methods[0].method.value, "bf16")

    def test_seed_software_grid_quality_and_blocker_mutations_are_rejected(self) -> None:
        raw = load_json_compatible_yaml(FIXED_PLAN)
        mutations = (
            lambda item: item["measurement"].__setitem__("seed", 1),
            lambda item: item["software_environment"].__setitem__(
                "python_version", "3.12.4"
            ),
            lambda item: item["software_environment"].__setitem__(
                "container_image", "replacement"
            ),
            lambda item: item["grid"].__setitem__("context_lengths", [128, 8192]),
            lambda item: item.__setitem__("performance_claim_eligible", True),
            lambda item: item["admission"].__setitem__(
                "retained_open_blockers", ["B-009"]
            ),
            lambda item: item["quality"].__setitem__(
                "performance_data_frozen", True
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(raw)
                mutation(changed)
                with self.assertRaises(SchemaValidationError):
                    parse_config(changed)

    def test_growing_context_cuda_graph_is_rejected(self) -> None:
        raw = load_json_compatible_yaml(GROWING_PLAN)
        raw["graph_modes"] = ["eager", "cuda_graph"]
        with self.assertRaises(SchemaValidationError):
            parse_config(raw)


class Phase3ManifestTests(unittest.TestCase):
    def test_manifest_dispatch_is_versioned_and_exactly_bound(self) -> None:
        raw = created_phase3_manifest()
        parsed = parse_run_manifest(raw)
        self.assertIsInstance(parsed, Phase3RunManifest)
        assert isinstance(parsed, Phase3RunManifest)
        self.assertEqual(parsed.plan_fingerprint, raw["plan_fingerprint"])
        self.assertEqual(parsed.cache_identity.capacity, 129)
        self.assertEqual(parsed.pci_bus_id, PHASE3_PCI_BUS_ID)

    def test_manifest_rejects_tuple_plan_native_identity_and_argv_mutations(self) -> None:
        raw = created_phase3_manifest()
        mutations = (
            lambda item: item.__setitem__("context_length", 4096),
            lambda item: item.__setitem__("plan_fingerprint", ZERO_SHA256),
            lambda item: item.__setitem__("point_fingerprint", ZERO_SHA256),
            lambda item: item.__setitem__("gpu_uuid", "GPU-substitute"),
            lambda item: item.__setitem__("native_g0_status", "BLOCKED"),
            lambda item: item["command"]["argv"].__setitem__(4, "--point-id"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(raw)
                mutation(changed)
                with self.assertRaises(SchemaValidationError):
                    parse_run_manifest(changed)

    def test_legacy_v1_remains_valid_but_rejects_phase3_semantics(self) -> None:
        raw = legacy_v1_manifest()
        self.assertIsInstance(parse_run_manifest(raw), RunManifest)

        phase3_kind = copy.deepcopy(raw)
        phase3_kind["run_kind"] = "phase3_admission"
        with self.assertRaises(SchemaValidationError):
            parse_run_manifest(phase3_kind)

        phase3_failure = copy.deepcopy(raw)
        phase3_failure.update(
            {
                "status": "allocation_failed",
                "started_at_utc": "2026-07-22T00:00:01Z",
                "finished_at_utc": "2026-07-22T00:00:02Z",
                "inventory_path": "artifact_inventory.json",
                "failure_reason": "fixture",
            }
        )
        with self.assertRaises(SchemaValidationError):
            parse_run_manifest(phase3_failure)

    def test_cache_fingerprint_is_derived_not_self_asserted(self) -> None:
        cache = BF16CacheIdentity.from_dict(cache_payload())
        self.assertEqual(cache.device, "cuda:0")
        tampered = cache_payload()
        tampered["layout_fingerprint"] = ZERO_SHA256
        with self.assertRaises(SchemaValidationError):
            BF16CacheIdentity.from_dict(tampered)
        query_head_storage = cache_payload()
        query_head_storage["num_kv_heads"] = 32
        with self.assertRaises(SchemaValidationError):
            BF16CacheIdentity.from_dict(query_head_storage)

    def test_backend_command_and_worker_result_are_strict(self) -> None:
        backend = BF16BackendIdentity.from_dict(backend_payload())
        self.assertEqual(len(backend.source_artifacts), 5)
        self.assertTrue(
            all(isinstance(item, SourceDigest) for item in backend.source_artifacts)
        )
        command = Phase3CommandSpec.from_dict(created_phase3_manifest()["command"])
        self.assertFalse(command.dry_run)

        worker_payload = {
            "schema_version": "kvbench-phase3-worker-result-1.0.0",
            "run_id": "worker-result",
            "point_id": "fixed_l-b1-l128-eager-r1",
            "runner_kind": "fixed_l",
            "count_unit": "decode_operations",
            "status": "completed",
            "expected_operations": 160,
            "completed_operations": 160,
            "failed_operations": 0,
            "output_checksum": ZERO_SHA256,
            "failure_reason": None,
        }
        worker = Phase3WorkerResult.from_dict(worker_payload)
        self.assertEqual(worker.status, RunStatus.COMPLETED)
        for key, value in (
            ("expected_operations", 159),
            ("completed_operations", 159),
            ("runner_kind", "growing_context"),
            ("count_unit", "trajectories"),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(worker_payload)
                changed[key] = value
                with self.assertRaises(SchemaValidationError):
                    Phase3WorkerResult.from_dict(changed)

    def test_required_phase3_failure_statuses_are_terminal(self) -> None:
        values = {
            "model_identity_unresolved",
            "model_access_blocked",
            "backend_unsupported",
            "allocation_failed",
            "state_drift_detected",
            "gqa_materialization_detected",
            "gqa_dispatch_unverified",
            "gqa_nonmaterialization_unproven",
            "graph_replay_failed",
        }
        self.assertTrue(values.issubset({status.value for status in RunStatus}))
        self.assertTrue(all(RunStatus(value).is_terminal for value in values))


class G1ReportTests(unittest.TestCase):
    def test_blocked_report_retains_all_governance_states(self) -> None:
        report = Phase3G1AdmissionReport.from_dict(
            {
                "schema_version": "kvbench-phase3-g1-admission-report-1.0.0",
                "generated_at_utc": CREATED_AT,
                "git_sha": ZERO_GIT_SHA,
                "status": "BLOCKED",
                "g0": "PASS",
                "g1": "BLOCKED",
                "g2": "NOT_EVALUATED",
                "g3": "NOT_EVALUATED",
                "g4": "NOT_EVALUATED",
                "g5": "NOT_EVALUATED",
                "full_scan_state": "closed",
                "quality": quality_payload(),
                "quality_benchmark_executed": False,
                "quality_only_dependencies_installed": False,
                "measurement_scope": "native_host_admission",
                "performance_claim_eligible": False,
                "performance_data_frozen": False,
                "blocker_b009": "OPEN",
                "blocker_b010": "OPEN",
                "expected_process_count": 20,
                "plan_sources": [
                    {"path": path, "sha256": fingerprint}
                    for path, fingerprint in PHASE3_PLAN_FINGERPRINTS.items()
                ],
                "run_evidence": [],
                "stability_summaries": [],
                "criteria": [
                    {
                        "criterion": criterion,
                        "disposition": "BLOCKED",
                        "evidence_run_ids": [],
                        "reason": "fixture blocker",
                    }
                    for criterion in G1_CRITERIA
                ],
                "all_artifacts_checksum_valid": False,
                "formal_paper_claim_generated": False,
            }
        )
        self.assertEqual(report.g5, GateDisposition.NOT_EVALUATED)
        self.assertEqual(len(report.criteria), 20)
        self.assertEqual(G1_CRITERIA.count("fixed_l_runner"), 1)

    def test_exact_passing_report_is_accepted(self) -> None:
        report = Phase3G1AdmissionReport.from_dict(passing_g1_report())
        self.assertEqual(report.g1, GateDisposition.PASS)
        self.assertEqual(
            tuple(item.point_id for item in report.run_evidence),
            FROZEN_PHASE3_POINT_IDS,
        )
        self.assertEqual(len(report.stability_summaries), 2)

    def test_pass_rejects_order_coverage_and_stability_loopholes(self) -> None:
        mutations = (
            lambda item: item["run_evidence"].__setitem__(
                slice(0, 2), list(reversed(item["run_evidence"][:2]))
            ),
            lambda item: item["criteria"][7].__setitem__(
                "evidence_run_ids",
                [entry["run_id"] for entry in item["run_evidence"]],
            ),
            lambda item: item["stability_summaries"][0].__setitem__(
                "coefficient_of_variation_percent", 3.1
            ),
            lambda item: item["stability_summaries"][0].__setitem__(
                "median_host_wall_ms", 1.1
            ),
            lambda item: item.__setitem__("stability_summaries", []),
            lambda item: item["plan_sources"][0].__setitem__(
                "sha256", ZERO_SHA256
            ),
            lambda item: item.__setitem__("status", "PARTIAL"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = passing_g1_report()
                mutation(changed)
                with self.assertRaises(SchemaValidationError):
                    Phase3G1AdmissionReport.from_dict(changed)


if __name__ == "__main__":
    unittest.main()
