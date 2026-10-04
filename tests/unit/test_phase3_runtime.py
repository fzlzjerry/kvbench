"""Focused CPU controls for the Phase 3 runtime core."""

from __future__ import annotations

from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from kvbench.config import REPOSITORY_ROOT
from kvbench.runtime import phase3_coordinator
from kvbench.runtime.allocation import MemorySnapshot, NormalTimingMemoryEvidence
from kvbench.runtime.backend import (
    BackendFallbackError,
    backend_identity,
    flash_attention_forward,
)
from kvbench.runtime.bf16_endpoint import rotate_half_in_place
from kvbench.runtime.model_loader import (
    EXPECTED_HASHES,
    EXPECTED_SIZES,
    ModelAccessError,
    verify_frozen_snapshot,
)
from kvbench.runtime.phase3_coordinator import (
    COMMAND_FINGERPRINT_ENV,
    RAW_AUDIT_ROOT_ENV,
    READY_NOT_OBSERVED_V2,
    SENSITIVE_ENV_FRAGMENTS,
    SENSITIVE_ENV_KEY_EXEMPTIONS,
    _run_point,
    _worker_argv,
    _worker_environment,
)
from kvbench.runtime.process_supervision import command_fingerprint
from kvbench.runtime.numerical import (
    compare_tensors_untimed,
    small_attention_reference,
    tensor_sha256_untimed,
)
from kvbench.runtime.static_cache import (
    BF16StaticCache,
    CacheBoundsError,
    cache_accounting_for_geometry,
    layout_fingerprint_for_geometry,
)
from kvbench.runtime.telemetry import (
    TelemetrySnapshot,
    telemetry_sampling_interval_seconds,
)
from kvbench.schema import (
    BF16BackendIdentity,
    RunStatus,
    canonical_json_bytes,
    sha256_hex,
)


class Phase3StaticCacheTests(unittest.TestCase):
    def make_cache(self, *, capacity: int = 5) -> BF16StaticCache:
        return BF16StaticCache(
            num_layers=2,
            batch_size=2,
            num_kv_heads=2,
            capacity=capacity,
            head_dim=4,
            device="cpu",
        )

    def prefill(self, cache: BF16StaticCache, length: int) -> None:
        cache.prepare_prefill(length)
        for layer in range(cache.num_layers):
            base = torch.arange(
                cache.batch_size * cache.num_kv_heads * length * cache.head_dim,
                dtype=torch.bfloat16,
            ).reshape(
                cache.batch_size,
                cache.num_kv_heads,
                length,
                cache.head_dim,
            )
            cache.update(base + layer, base + layer + 1, layer)
        cache.complete_prefill()

    def test_shape_bytes_and_pure_fingerprint_match(self) -> None:
        cache = self.make_cache()
        self.assertEqual(tuple(cache.keys.shape), (2, 2, 2, 5, 4))
        self.assertEqual(cache.predicted_tensor_bytes, 640)
        self.assertEqual(cache.tensor_storage_bytes, 640)
        self.assertEqual(cache.accounting().padding_bytes, 0)
        pure = cache_accounting_for_geometry(
            num_layers=2,
            batch_size=2,
            num_kv_heads=2,
            capacity=5,
            head_dim=4,
        )
        self.assertEqual(pure["predicted_tensor_bytes"], 640)
        self.assertEqual(
            cache.layout_fingerprint(),
            layout_fingerprint_for_geometry(
                num_layers=2,
                batch_size=2,
                num_kv_heads=2,
                capacity=5,
                head_dim=4,
                device="cpu",
            ),
        )

    def test_fixed_scratch_preserves_historical_storage(self) -> None:
        cache = self.make_cache()
        self.prefill(cache, 4)
        historical_keys = cache.keys[:, :, :, :4, :].clone()
        historical_values = cache.values[:, :, :, :4, :].clone()
        pointers = cache.pointers()
        cache.prepare_fixed(4)
        for iteration in range(5):
            key = torch.full((2, 2, 1, 4), iteration, dtype=torch.bfloat16)
            value = torch.full((2, 2, 1, 4), iteration + 1, dtype=torch.bfloat16)
            for layer in range(2):
                attended_key, attended_value = cache.update(key, value, layer)
                self.assertEqual(tuple(attended_key.shape), (2, 2, 5, 4))
                self.assertEqual(tuple(attended_value.shape), (2, 2, 5, 4))
        self.assertTrue(torch.equal(cache.keys[:, :, :, :4, :], historical_keys))
        self.assertTrue(
            torch.equal(cache.values[:, :, :, :4, :], historical_values)
        )
        self.assertEqual(cache.active_context, 4)
        self.assertEqual(cache.pointers(), pointers)

    def test_growing_progression_and_bounds(self) -> None:
        cache = self.make_cache(capacity=4)
        self.prefill(cache, 2)
        cache.prepare_growing(2, 2)
        for step in range(2):
            cache.select_growing_step(step)
            key = torch.full((2, 2, 1, 4), step + 5, dtype=torch.bfloat16)
            for layer in range(2):
                attended, _ = cache.update(key, key, layer)
                self.assertEqual(int(attended.shape[-2]), 3 + step)
            cache.finish_growing_step()
            self.assertEqual(cache.active_context, 3 + step)
        with self.assertRaises(CacheBoundsError):
            cache.select_growing_step(2)
        cache.reset_active_length(0)
        self.assertEqual(cache.active_context, 0)

    def test_capacity_rejection_is_prewrite(self) -> None:
        cache = self.make_cache(capacity=3)
        self.prefill(cache, 2)
        original = cache.keys.clone()
        with self.assertRaises(CacheBoundsError):
            cache.prepare_growing(2, 2)
        self.assertTrue(torch.equal(cache.keys, original))


class Phase3NumericalTests(unittest.TestCase):
    def test_cat_free_half_rotation_matches_explicit_formula(self) -> None:
        original = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0]]]],
            dtype=torch.bfloat16,
        )
        states = original.clone()
        cos = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.bfloat16)
        sin = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]], dtype=torch.bfloat16)
        scratch = torch.empty_like(states[..., :2])
        expected = torch.empty_like(states)
        expected[..., :2] = (
            original[..., :2] * cos.unsqueeze(1)[..., :2]
            - original[..., 2:] * sin.unsqueeze(1)[..., :2]
        )
        expected[..., 2:] = (
            original[..., 2:] * cos.unsqueeze(1)[..., 2:]
            + original[..., :2] * sin.unsqueeze(1)[..., 2:]
        )
        rotate_half_in_place(states, cos, sin, scratch)
        self.assertTrue(torch.equal(states, expected))

    def test_small_gqa_reference_causal_and_finite(self) -> None:
        torch.manual_seed(7)
        query = torch.randn((2, 4, 3, 8), dtype=torch.bfloat16)
        key = torch.randn((2, 2, 3, 8), dtype=torch.bfloat16)
        value = torch.randn((2, 2, 3, 8), dtype=torch.bfloat16)
        reference = small_attention_reference(
            query,
            key,
            value,
            is_causal=True,
            scale=8**-0.5,
        )
        self.assertEqual(tuple(reference.shape), (2, 4, 3, 8))
        comparison = compare_tensors_untimed(
            reference,
            reference.clone(),
            atol=0.02,
            rtol=0.02,
        )
        self.assertTrue(comparison.passed)
        self.assertTrue(comparison.finite)

    def test_checksum_binds_shape_and_dtype(self) -> None:
        value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self.assertEqual(
            tensor_sha256_untimed(value),
            tensor_sha256_untimed(value.clone()),
        )
        self.assertNotEqual(
            tensor_sha256_untimed(value),
            tensor_sha256_untimed(value.reshape(4, 2)),
        )


class Phase3IdentityAndBackendTests(unittest.TestCase):
    def test_frozen_snapshot_contract_has_exact_eleven_sized_entries(self) -> None:
        self.assertEqual(len(EXPECTED_HASHES), 11)
        self.assertEqual(set(EXPECTED_HASHES), set(EXPECTED_SIZES))
        self.assertEqual(EXPECTED_SIZES["LICENSE"], 7_627)
        self.assertEqual(
            EXPECTED_HASHES["LICENSE"],
            "64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369",
        )

    def test_backend_identity_is_verified_and_schema_consumable(self) -> None:
        identity = backend_identity()
        self.assertEqual(identity["torch_version"], "2.12.1+cu130")
        self.assertEqual(identity["selected_backend"], "flash_attention")
        self.assertIs(identity["enable_gqa"], True)
        self.assertEqual(len(identity["source_artifacts"]), 5)
        parsed = BF16BackendIdentity.from_dict(identity)
        self.assertEqual(parsed.backend_id, "torch_sdpa_flash_gqa")

    def test_backend_fails_outside_forced_context(self) -> None:
        query = torch.zeros((1, 4, 1, 8), dtype=torch.bfloat16)
        key = torch.zeros((1, 2, 1, 8), dtype=torch.bfloat16)
        with self.assertRaises(BackendFallbackError):
            flash_attention_forward(
                object(),
                query,
                key,
                key,
                None,
                8**-0.5,
            )

    def test_missing_exact_snapshot_fails_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-the-frozen-revision"
            with self.assertRaises(ModelAccessError):
                verify_frozen_snapshot(missing)


class Phase3EvidenceSerializationTests(unittest.TestCase):
    def _captured_coordinator_run(
        self,
        *,
        readiness_observed: bool,
        termination_succeeds: bool = True,
    ) -> tuple[
        dict[str, object],
        mock.Mock,
        mock.Mock | None,
        dict[str, str],
    ]:
        run_id = "phase3-unit-coordinator"
        plan_path = "config/phase3_fixed_plan.json"
        point = SimpleNamespace(
            point_id="fixed_l-b1-l128-eager-r1",
            process_replicate=1,
            to_dict=lambda: {"point_id": "fixed_l-b1-l128-eager-r1"},
        )
        plan = SimpleNamespace(to_dict=lambda: {"schema_version": "unit-plan"})
        bundle = SimpleNamespace(
            plan=plan,
            canonical_fingerprints=(),
            all_blockers=(),
        )
        base_environment = {
            "LANG": "C",
            "PATH": "/usr/bin:/bin",
            "KVBENCH_PHASE3_AUDIT_READY": "/tmp/unit-ready",
        }
        writes: dict[str, object] = {}
        self_outer = self

        class CapturingRun:
            def start(self) -> None:
                return None

            def write_json(self, relative: str, payload: object) -> None:
                writes[relative] = copy.deepcopy(payload)

            def write_bytes(self, relative: str, payload: bytes) -> None:
                writes[relative] = bytes(payload)

            def finalize(self, manifest: object) -> Path:
                writes["_final_manifest"] = manifest
                return REPOSITORY_ROOT / "artifacts" / "phase3" / run_id

        class CapturingStore:
            def create(self, created_run_id: str, manifest: object) -> CapturingRun:
                self_outer.assertEqual(created_run_id, run_id)
                writes["_initial_manifest"] = manifest
                return CapturingRun()

        failure_reason = (
            "Phase3CoordinatorError: worker release audit failed closed"
            if readiness_observed
            else "Phase3CoordinatorError: synthetic pre-registration failure"
        )
        failed_result = SimpleNamespace(
            status=RunStatus.ABORTED,
            failure_reason=failure_reason,
            to_dict=lambda: {
                "schema_version": "unit-worker-result",
                "status": RunStatus.ABORTED.value,
                "failure_reason": failure_reason,
            },
        )
        initial_manifest = object()
        terminal_manifest = object()
        initial_manifest_mock = mock.Mock(return_value=initial_manifest)
        process_snapshot_mock = mock.Mock(
            side_effect=(
                [
                    {"before": True},
                    {"release": True},
                    {"after": True},
                ]
                if readiness_observed
                else [
                    phase3_coordinator.Phase3CoordinatorError(
                        "synthetic pre-registration failure"
                    ),
                    {"after": True},
                ]
            )
        )
        common_patches = {
            "_utc_now": mock.Mock(return_value="2026-07-23T00:00:00Z"),
            "_pin_phase3_execution_sources": mock.Mock(
                return_value=phase3_coordinator.Phase3ExecutionSourcePin(
                    execution_git_sha="1" * 40,
                    source_bytes_by_path=tuple(
                        (relative, f"unit:{relative}".encode("utf-8"))
                        for relative in (
                            phase3_coordinator.PHASE3_EXECUTION_SOURCE_PATHS
                        )
                    ),
                    source_identity_sha256=(
                        phase3_coordinator.phase3_source_identity_sha256(
                            {
                                relative: sha256_hex(
                                    f"unit:{relative}".encode("utf-8")
                                )
                                for relative in (
                                    phase3_coordinator.REQUIRED_SUT_SOURCES
                                )
                            }
                        )
                    ),
                    execution_source_identity_sha256=(
                        phase3_coordinator._phase3_execution_source_identity_sha256(
                            {
                                relative: sha256_hex(
                                    f"unit:{relative}".encode("utf-8")
                                )
                                for relative in phase3_coordinator.PHASE3_EXECUTION_SOURCE_PATHS
                            }
                        )
                    ),
                )
            ),
            "_expected_phase3_raw_audit_operations": mock.Mock(
                return_value=()
            ),
            "_revalidate_phase3_execution_sources": mock.Mock(),
            "_validate_cache_source_join": mock.Mock(),
            "_resolved_phase3_worker_result": mock.Mock(return_value=failed_result),
            "_worker_environment": mock.Mock(
                return_value=dict(base_environment)
            ),
            "_cache_identity": mock.Mock(return_value=object()),
            "_initial_manifest": initial_manifest_mock,
            "phase3_artifact_store": mock.Mock(return_value=CapturingStore()),
            "_process_snapshot": process_snapshot_mock,
            "_snapshot_clean": mock.Mock(return_value=True),
            "_failed_result": mock.Mock(return_value=failed_result),
            "_terminal_manifest": mock.Mock(return_value=terminal_manifest),
            "validate_run_directory": mock.Mock(
                return_value=SimpleNamespace(valid=True, complete=True)
            ),
        }
        register_spawn_mock: mock.Mock | None = None
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.multiple(phase3_coordinator, **common_patches)
            )
            if readiness_observed:
                outcome = {
                    "disposition": "owned_worker_failure",
                    "reason": "registered worker exited before evidence_flushed",
                    "returncode": -15,
                    "observed_stages": ["supervisor_reaped"],
                    "missing_worker_stages": [
                        "worker_started",
                        "cuda_context_created",
                        "measurement_started",
                        "measurement_finished",
                        "evidence_flushed",
                        "worker_exiting",
                    ],
                    "evidence_flushed": False,
                    "worker_exiting_observed": False,
                    "full_handshake_observed": False,
                    "exclusivity_passed": True,
                }

                class FakeRegistry:
                    def __init__(self) -> None:
                        self.reaped = False

                    def refresh_handshake_directory(self, directory: Path) -> None:
                        return None

                    def note_exit_observed(self) -> None:
                        return None

                    def record_supervisor_reaped(
                        self,
                        returncode: int,
                        *,
                        recorded_at_utc: str,
                    ) -> object:
                        self.reaped = True
                        return object()

                    def terminal_outcome(self) -> object:
                        return SimpleNamespace(to_dict=lambda: dict(outcome))

                    def to_evidence(self) -> dict[str, object]:
                        return {
                            "schema_version": (
                                "kvbench-phase3-process-registry-3.0.0"
                            ),
                            "owned_completion_policy": (
                                "zero_exit_after_durable_evidence_flush_"
                                "worker_exiting_optional"
                            ),
                            "worker_exiting_required_for_owned_completion": False,
                            "handshake_events": [],
                            "outcome": dict(outcome),
                        }

                ready = {
                    "schema_version": "kvbench-phase3-worker-ready-1.0.0",
                    "pid": 4242,
                    "process_start_time_ticks": 123456,
                    "cuda_imported": False,
                }
                process = SimpleNamespace(pid=4242, returncode=None)
                registry = FakeRegistry()
                register_spawn_mock = mock.Mock(return_value=registry)
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator.subprocess,
                        "Popen",
                        return_value=process,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator,
                        "_pidfd_open",
                        return_value=(False, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator,
                        "read_process_identity",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator.RunOwnedProcessRegistry,
                        "register_spawn",
                        register_spawn_mock,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator,
                        "_wait_for_ready",
                        return_value=ready,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator,
                        "_registry_snapshot_verdict",
                        return_value={"passed": False},
                    )
                )
                def terminate_registered(
                    observed_process: object,
                    observed_registry: object,
                    *,
                    handshake_directory: Path | None = None,
                ) -> tuple[int, object]:
                    observed_registry.note_exit_observed()
                    event = observed_registry.record_supervisor_reaped(
                        -15,
                        recorded_at_utc="2026-07-23T00:00:00Z",
                    )
                    observed_process.returncode = -15
                    return -15, event

                termination_effect: object = terminate_registered
                if not termination_succeeds:
                    termination_effect = phase3_coordinator.Phase3CoordinatorError(
                        "synthetic registered-worker termination timeout"
                    )
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator,
                        "_terminate_registered_worker",
                        side_effect=termination_effect,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        phase3_coordinator,
                        "write_handshake_event",
                    )
                )
            writes["_run_result"] = _run_point(
                bundle=bundle,
                plan_path=plan_path,
                point=point,
                run_id=run_id,
                git_sha="1" * 40,
                backend=object(),
                live_hardware={"gpu_uuid": "GPU-unit"},
            )
        return (
            writes,
            initial_manifest_mock,
            register_spawn_mock,
            base_environment,
        )

    def test_coordinator_retains_sentinel_and_pre_injection_digest(self) -> None:
        writes, initial_manifest_mock, _, base_environment = (
            self._captured_coordinator_run(readiness_observed=False)
        )
        environment_sha = sha256_hex(canonical_json_bytes(base_environment))
        self.assertEqual(
            initial_manifest_mock.call_args.kwargs["environment_sha256"],
            environment_sha,
        )
        expected_fingerprint = command_fingerprint(
            _worker_argv(
                "config/phase3_fixed_plan.json",
                SimpleNamespace(
                    point_id="fixed_l-b1-l128-eager-r1",
                    process_replicate=1,
                ),
                "phase3-unit-coordinator",
            ),
            working_directory=str(REPOSITORY_ROOT),
            environment_sha256=environment_sha,
        )
        self.assertEqual(
            writes["environment/worker_environment.json"],
            {
                **base_environment,
                COMMAND_FINGERPRINT_ENV: expected_fingerprint,
            },
        )
        self.assertEqual(
            writes["environment/process.ready.json"],
            READY_NOT_OBSERVED_V2,
        )

    def test_observed_readiness_overwrites_sentinel_and_registry_uses_digest(
        self,
    ) -> None:
        writes, initial_manifest_mock, register_spawn_mock, base_environment = (
            self._captured_coordinator_run(readiness_observed=True)
        )
        environment_sha = sha256_hex(canonical_json_bytes(base_environment))
        expected_fingerprint = command_fingerprint(
            _worker_argv(
                "config/phase3_fixed_plan.json",
                SimpleNamespace(
                    point_id="fixed_l-b1-l128-eager-r1",
                    process_replicate=1,
                ),
                "phase3-unit-coordinator",
            ),
            working_directory=str(REPOSITORY_ROOT),
            environment_sha256=environment_sha,
        )
        self.assertEqual(
            initial_manifest_mock.call_args.kwargs["environment_sha256"],
            environment_sha,
        )
        self.assertIsNotNone(register_spawn_mock)
        assert register_spawn_mock is not None
        self.assertEqual(
            register_spawn_mock.call_args.kwargs[
                "expected_command_fingerprint"
            ],
            expected_fingerprint,
        )
        self.assertEqual(
            writes["environment/process.ready.json"],
            {
                "schema_version": "kvbench-phase3-worker-ready-1.0.0",
                "pid": 4242,
                "process_start_time_ticks": 123456,
                "cuda_imported": False,
            },
        )
        handshake = writes["environment/process.handshake.json"]
        self.assertEqual(
            handshake["schema_version"],
            "kvbench-phase3-worker-handshake-3.0.0",
        )
        self.assertFalse(
            handshake["worker_exiting_required_for_owned_completion"]
        )
        self.assertTrue(
            handshake[
                "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed"
            ]
        )
        process_audit = writes["validation/process_audit_outcome.json"]
        self.assertEqual(
            process_audit["schema_version"],
            "kvbench-phase3-process-audit-3.0.0",
        )
        self.assertFalse(
            process_audit["worker_exiting_required_for_owned_completion"]
        )
        self.assertIn(
            "validation/execution_source_pin.before_spawn.json",
            writes,
        )
        self.assertIn(
            "validation/execution_source_pin.after_worker_exit.json",
            writes,
        )
        termination = writes["validation/worker_termination.json"]
        self.assertTrue(termination["resolved"])
        self.assertEqual(
            termination["disposition"],
            "registered_terminated_reaped",
        )
        self.assertTrue(
            termination["source_revalidation_attempted_after_resolution"]
        )
        self.assertTrue(termination["source_revalidated_after_resolution"])
        self.assertIsNone(termination["pidfd_closed_after_resolution"])
        self.assertTrue(writes["_run_result"]["worker_termination_resolved"])

    def test_unresolved_termination_is_preserved_and_not_revalidated(self) -> None:
        writes, _, _, _ = self._captured_coordinator_run(
            readiness_observed=True,
            termination_succeeds=False,
        )

        termination = writes["validation/worker_termination.json"]
        self.assertFalse(termination["resolved"])
        self.assertEqual(
            termination["disposition"],
            "registered_termination_unresolved",
        )
        self.assertIn("termination timeout", termination["failure_reason"])
        self.assertFalse(
            termination["source_revalidation_attempted_after_resolution"]
        )
        self.assertFalse(termination["source_revalidated_after_resolution"])
        self.assertNotIn(
            "validation/execution_source_pin.after_worker_exit.json",
            writes,
        )
        self.assertFalse(writes["_run_result"]["worker_termination_resolved"])

    def test_campaign_stops_before_next_point_after_unresolved_worker(self) -> None:
        points = (
            SimpleNamespace(point_id="fixed_l-b1-l128-eager-r1"),
            SimpleNamespace(point_id="fixed_l-b2-l128-eager-r1"),
        )
        plan = SimpleNamespace(
            expected_process_count=2,
            fingerprint=lambda: "a" * 64,
        )
        bundle = SimpleNamespace(
            execution_ready=True,
            plan_path=REPOSITORY_ROOT / "configs/plans/phase3-unit.yaml",
            plan=plan,
        )
        recorder = SimpleNamespace(
            finalize=mock.Mock(
                return_value=(
                    REPOSITORY_ROOT
                    / "artifacts"
                    / "phase3-campaigns"
                    / "phase3-unit-campaign"
                )
            )
        )
        first_result = {
            "run_id": "phase3-unit-first",
            "point_id": points[0].point_id,
            "status": RunStatus.ABORTED.value,
            "run_dir": "artifacts/phase3/phase3-unit-first",
            "checksum_valid": True,
            "worker_termination_resolved": False,
            "timing_collected": False,
        }
        run_point_mock = mock.Mock(return_value=first_result)
        with mock.patch.multiple(
            phase3_coordinator,
            _validate_entry_evidence=mock.Mock(),
            _git_identity=mock.Mock(return_value=("1" * 40, False)),
            _live_hardware=mock.Mock(return_value={"gpu_uuid": "GPU-unit"}),
            load_phase3_admission_bundle=mock.Mock(return_value=bundle),
            expand_phase3_process_points=mock.Mock(return_value=points),
            _backend_identity_stdlib=mock.Mock(return_value=object()),
            _utc_now=mock.Mock(return_value="2026-07-23T00:00:00Z"),
            _run_point=run_point_mock,
        ), mock.patch.object(
            phase3_coordinator.Phase3CampaignRecorder,
            "create",
            return_value=recorder,
        ):
            with self.assertRaises(
                phase3_coordinator.Phase3WorkerTerminationUnresolved
            ):
                phase3_coordinator.run_phase3_campaign(bundle.plan_path)

        run_point_mock.assert_called_once()
        finalized = recorder.finalize.call_args.args[0]
        self.assertEqual(finalized["attempted_process_count"], 1)
        self.assertEqual(
            finalized["unattempted_point_ids"],
            [points[1].point_id],
        )
        self.assertTrue(finalized["unexpected_campaign_abort"])
        self.assertEqual(finalized["runs"], [first_result])

    def test_initial_manifest_uses_supplied_environment_digest(self) -> None:
        plan_path = "configs/plans/phase3_bf16_fixed_l.yaml"
        bundle = phase3_coordinator.load_phase3_admission_bundle(
            REPOSITORY_ROOT / plan_path
        )
        point = phase3_coordinator.expand_phase3_process_points(bundle.plan)[0]
        environment_sha = "a" * 64
        backend = SimpleNamespace(
            backend_id="torch_sdpa_flash_gqa",
            to_dict=lambda: {"schema_version": "unit-backend"},
            fingerprint=lambda: "b" * 64,
        )
        cache = SimpleNamespace(
            layout_fingerprint="c" * 64,
            to_dict=lambda: {"schema_version": "unit-cache"},
        )
        with mock.patch.object(
            phase3_coordinator.Phase3RunManifest,
            "from_dict",
            side_effect=lambda payload: payload,
        ):
            payload = phase3_coordinator._initial_manifest(
                bundle=bundle,
                plan_path=plan_path,
                point=point,
                run_id="phase3-unit-manifest",
                created_at="2026-07-23T00:00:00Z",
                git_sha="1" * 40,
                environment_sha256=environment_sha,
                backend=backend,
                cache=cache,
            )
        self.assertEqual(
            payload["command"]["environment_sha256"],
            environment_sha,
        )
        self.assertEqual(
            len(payload["adapter_config_fingerprint"]),
            64,
        )

    def test_sanitized_worker_environment_is_constructible_and_secret_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = _worker_environment(Path(directory))

        self.assertEqual(environment["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(
            environment[RAW_AUDIT_ROOT_ENV],
            str(Path(directory) / "raw-audits"),
        )
        self.assertNotEqual(
            environment[RAW_AUDIT_ROOT_ENV],
            environment["KVBENCH_PHASE3_IPC_PATH"],
        )
        self.assertNotIn("HF_TOKEN", environment)
        self.assertFalse(
            any(
                fragment in key.lower()
                for key in environment
                if key not in SENSITIVE_ENV_KEY_EXEMPTIONS
                for fragment in SENSITIVE_ENV_FRAGMENTS
            )
        )

    def test_normal_timing_memory_evidence_keeps_audit_separate(self) -> None:
        def snapshot(label: str, allocated: int, peak: int) -> MemorySnapshot:
            return MemorySnapshot(
                label=label,
                host_timestamp_ns=1,
                allocated_bytes=allocated,
                reserved_bytes=allocated + 10,
                peak_allocated_bytes=peak,
                peak_reserved_bytes=peak + 10,
            )

        evidence = NormalTimingMemoryEvidence(
            model_baseline=snapshot("model_baseline", 100, 100),
            post_cache_allocation=snapshot("post_cache_allocation", 150, 150),
            post_setup=snapshot("post_setup", 175, 200),
            timing_before=snapshot("normal_timing_before", 175, 175),
            timing_after=snapshot("normal_timing_after", 180, 210),
            timing_executed=True,
        ).to_dict()
        self.assertEqual(evidence["timing_allocated_delta_bytes"], 5)
        self.assertEqual(evidence["timing_peak_allocated_bytes"], 210)
        self.assertTrue(evidence["instrumented_audit_separate"])
        self.assertTrue(evidence["timing_executed"])
        self.assertFalse(evidence["peak_reset_inside_measured_boundary"])
        self.assertFalse(evidence["profiler_duration_reported"])

    def test_telemetry_interval_is_raw_and_monotonic(self) -> None:
        def snapshot(host_ns: int) -> TelemetrySnapshot:
            return TelemetrySnapshot(
                timestamp="2026/07/22 00:00:00.000",
                collected_at_utc="2026-07-22T00:00:00+00:00",
                host_query_started_ns=host_ns - 5,
                host_query_finished_ns=host_ns + 5,
                host_monotonic_ns=host_ns,
                gpu_name="GPU",
                gpu_uuid="GPU-0000",
                power_watts=100.0,
                temperature_celsius=40.0,
                sm_clock_mhz=1000.0,
                memory_clock_mhz=2000.0,
                vram_used_mib=3000.0,
                ecc_mode="Enabled",
            )

        before = snapshot(1_000_000_000)
        after = snapshot(2_500_000_000)
        self.assertEqual(telemetry_sampling_interval_seconds(before, after), 1.5)
        serialized = before.to_dict()
        self.assertTrue(serialized["raw_snapshot"])
        self.assertFalse(serialized["stability_inference"])


if __name__ == "__main__":
    unittest.main()
