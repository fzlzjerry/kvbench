#!/usr/bin/python3
"""Dependency-free, read-only repository governance validation.

These checks never execute CUDA, timing, profiler, model, or quality code and
never install packages. They protect the certified E00 evidence/environment
while preserving the completed Phase 2 audit boundary and validating the
approved Phase 3 implementation separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tomllib
import typing
from collections.abc import Callable, Sequence
from typing import Any


REPOSITORY_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_SCRIPT_ROOT
SRC = ROOT / "src"
PHASE2_ENTRY_COMMIT = "fb164b5ea96031ca40b21f4b8436a49a3bb5b8d2"
PHASE2_FINAL_COMMIT = "7c36d130565acef0883acb638c6b6c731b3f32ad"
PHASE4_ENTRY_COMMIT = "89189297992947b7a8b79252add551c9321e5f33"
PHASE5_ENTRY_COMMIT = "9eeabe787060e84c20cd7f88da8f7bca68eae1d4"
PHASE6A_ENTRY_COMMIT = "a25a76a052a918428e8eb56cdfde63470cf6a152"
PHASE6_ENTRY_COMMIT = "e06f638f4b913f9bd1be2975a478657f5bf2338e"
PHASE7_ENTRY_COMMIT = "0974bbc98f8f941b09800786591108292dc4e0dd"
PHASE8_ENTRY_COMMIT = "8d6d766a34a15bd40bd42cc47c5482b0dd052cc0"
PHASE9P_ENTRY_COMMIT = "f2c6475f09cdf6e9660552eb23c91b03e386aa59"
PHASE9P_FINAL_COMMIT = "1b3a98160ba4760007ca861c1a280def698b2027"
PHASE9_ENTRY_COMMIT = "b4d253724717076188a38032d6d6204fdf15e191"
PHASE10_ENTRY_COMMIT = "a873fc93754fa86bfb757fce476388897bee8dca"
PHASE11PR_ENTRY_COMMIT = "1cb2c95be61a328f88a031ae4ce91784dddec736"
PHASE11_ENTRY_COMMIT = "72f1897af78b738cc8c74fd335a8957a8e8f5d6c"
PHASE11D_ENTRY_COMMIT = "69e99389b548e82e65e027cc0ea7b86c9fbe43dd"
PHASE11R_ENTRY_COMMIT = "f0f02364a556da70e67b3107a0c0afad5f75eae9"
PHASE12E_ENTRY_COMMIT = "7c7af7cd1efe4a8befa36ceaedb11e2b47733276"
PHASE11DQ23_ENTRY_COMMIT = "2bc6aaa1d05b08d50f4c01bbc0b2863dd8689fe1"
PHASE11RQ23_ENTRY_COMMIT = "d99920e5dd7ea94bce7c98b4301bd035c073dfea"
PHASE12_ENTRY_COMMIT = "845a9293877121187a383c2c7aeab67912c856bd"
PHASE13_ENTRY_COMMIT = "7379e808ff687b10bf18c56364ae1c545cd00fe4"
PHASE13B_ENTRY_COMMIT = "c853acf65048b957a713f67b05ee560b845cd37f"
PHASE13R_ENTRY_COMMIT = "3dcd075d987db2452793408f8dd2b8f97f87530b"
PHASE13F_ENTRY_COMMIT = "50d7820e1feb986acecd78d1b58a3fff6e0bd713"
PHASE13R2_ENTRY_COMMIT = "bdfcb5ba41f150b1f56ddc8ac3515cd5a0b7bde3"
PHASE13T_ENTRY_COMMIT = "fb6a1fedc0cc01abbfbd974005f7567fbd291fd7"
PHASE13PR_ENTRY_COMMIT = "3e022662364fdbd129d4ca327e3eae8078a04ea1"
PHASE13PB_ENTRY_COMMIT = "5ff41722edfcdabe0ec96dc226f0546b008c6438"
PHASE13PC_ENTRY_COMMIT = "91b984069e3305428db612a3515fdc483ec990ed"
PHASE13RQ4_ENTRY_COMMIT = "a127b0d12f1815f9e51f21767686d4536b9b35e3"
PHASE13D_ENTRY_COMMIT = "345d805126587f318e3180ef75356ec38600e9f0"
PHASE14_ENTRY_COMMIT = "7b14043c0563607d92a9dc03d0069144db65508d"
PHASE15_ENTRY_COMMIT = "68cd50b912d5d91ffe57710b7e36714f4fe43f22"
PHASE16G_ENTRY_COMMIT = "a0c294f6c4f5dbb2f11953e42605a3bd1668359f"
QUALITY_COMMIT = "a7b8285dd8ed2fb598efbb3312e9f55064a0ee64"
ENVIRONMENT_COMMIT = "ea176994921c793789ebbd9d42515ce20ae4baee"
EVIDENCE_COMMIT = "fb164b5ea96031ca40b21f4b8436a49a3bb5b8d2"
PLAN_COMMIT = "9aadba6e4a0a6028ff0415e60feaccbd1fc9253f"

QUALITY_PROTOCOL_HASHES = {
    "CODEX_QUALITY_EVALUATION_ADDENDUM.md": (
        "62a8978e04732caff101487275d8b22f14358254538a7b377db2153597a1f332"
    ),
    "CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md": (
        "b6178566f239ca6ae598b477754f2ebb9d34d0f44c4fd25593b7ea58aa844620"
    ),
}
E00_RUNS = {
    "e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d": {
        "status": "FAIL",
        "manifest_sha256": (
            "0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035"
        ),
        "ledger_sha256": (
            "8716fc317747e7e9b5c06017cb8e5339df610c5a89d0d7fbee82ad07fbc68b52"
        ),
    },
    "e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32": {
        "status": "PASS",
        "manifest_sha256": (
            "d054df714bb5eea1f114bf10a03a2879f56ec8d17d3b07e24fe6efcaba6b7aca"
        ),
        "ledger_sha256": (
            "5a610162163979aca97beb2b7b0b480befb85d0b4e63b77c26ec46c36864eca8"
        ),
    },
}
PHASE2_CONFIG_PATHS = (
    "configs/hardware/rtx_pro_6000.yaml",
    "configs/models/primary_gqa_model.yaml",
    "configs/methods/bf16.yaml",
    "configs/methods/turboquant.yaml",
    "configs/methods/kivi.yaml",
    "configs/methods/kvquant.yaml",
    "configs/plans/smoke.yaml",
    "configs/plans/pilot.yaml",
    "configs/plans/graph_ab.yaml",
    "configs/plans/profiler_subset.yaml",
    "configs/plans/full_scan.yaml",
)
PHASE3_CONFIG_PATHS = (
    "configs/plans/phase3_bf16_fixed_l.yaml",
    "configs/plans/phase3_bf16_growing.yaml",
)
CONFIG_PATHS = (*PHASE2_CONFIG_PATHS, *PHASE3_CONFIG_PATHS)
PHASE2_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "Makefile",
        "README.md",
        "pyproject.toml",
        "analysis/README.md",
        "artifacts/README.md",
        "calibration/README.md",
        "reference/README.md",
        *PHASE2_CONFIG_PATHS,
        "docs/experiment_contract.md",
        "docs/measurement_protocol.md",
        "docs/method_notes/README.md",
        "docs/decisions/0006-phase2-schema-tooling-and-artifact-boundary.md",
        "docs/phase2_implementation_plan.md",
        "docs/phase_reports/phase2.md",
        "docs/artifact_policy.md",
        "docs/status.md",
        "docs/blockers.md",
        "docs/risk_register.md",
        "docs/tasks.md",
        "src/kvbench/__init__.py",
        "src/kvbench/__main__.py",
        "src/kvbench/errors.py",
        "src/kvbench/config.py",
        "src/kvbench/validation.py",
        "src/kvbench/cli.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/base.py",
        "src/kvbench/schema/config.py",
        "src/kvbench/schema/result.py",
        "src/kvbench/runtime/__init__.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/command.py",
        "scripts/validate_phase2.py",
        "tests/schema/__init__.py",
        "tests/schema/test_config_schema.py",
        "tests/unit/test_phase2_artifacts.py",
        "tests/unit/test_phase2_cli.py",
        "tests/unit/test_phase2_governance.py",
    }
)
PHASE3_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "Makefile",
        "configs/models/primary_gqa_model.yaml",
        "configs/methods/bf16.yaml",
        "configs/plans/phase3_bf16_fixed_l.yaml",
        "configs/plans/phase3_bf16_growing.yaml",
        "docs/blockers.md",
        "docs/decisions/0007-phase3-primary-model-and-bf16-backend.md",
        "docs/decisions/0008-phase3-gqa-evidence-taxonomy.md",
        "docs/decisions/0009-phase3-eager-allocation-criterion.md",
        "docs/decisions/0010-phase3-audit-to-measurement-session.md",
        "docs/decisions/0011-phase3-run-session-and-control-join.md",
        "docs/decisions/0012-phase3-minimum-remediation-scope.md",
        "docs/decisions/0013-phase3-source-backed-eager-allocation-catalog.md",
        "docs/decisions/0014-phase3-b015-raw-audit-bounds-and-split-controls.md",
        "docs/decisions/0015-phase3-graph-trace-asynchronous-marker-scope.md",
        "docs/evidence/phase3/backend-identity.md",
        "docs/evidence/phase3/g1-admission.json",
        "docs/evidence/phase3/g1-remediation-admission.json",
        "docs/evidence/phase3/model-identity.md",
        "docs/experiment_contract.md",
        "docs/measurement_protocol.md",
        "docs/plans/phase3-bf16-baseline.md",
        "docs/phase_reports/phase3.md",
        "docs/phase_reports/phase3-remediation.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "preflight/requirements-phase3.txt",
        "scripts/bootstrap_phase3.py",
        "scripts/validate_phase2.py",
        "src/kvbench/cli.py",
        "src/kvbench/config.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/base.py",
        "src/kvbench/schema/phase3.py",
        "src/kvbench/schema/result.py",
        "src/kvbench/runtime/__init__.py",
        "src/kvbench/runtime/allocation.py",
        "src/kvbench/runtime/allocation_attribution.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/backend.py",
        "src/kvbench/runtime/bf16_endpoint.py",
        "src/kvbench/runtime/command.py",
        "src/kvbench/runtime/cuda_graph.py",
        "src/kvbench/runtime/fixed_l_runner.py",
        "src/kvbench/runtime/gqa_audit.py",
        "src/kvbench/runtime/gqa_device_dispatch.py",
        "src/kvbench/runtime/gqa_taxonomy.py",
        "src/kvbench/runtime/growing_context_runner.py",
        "src/kvbench/runtime/model_loader.py",
        "src/kvbench/runtime/numerical.py",
        "src/kvbench/runtime/phase3_allocator_controls.py",
        "src/kvbench/runtime/phase3_coordinator.py",
        "src/kvbench/runtime/phase3_campaign.py",
        "src/kvbench/runtime/phase3_audit_operation.py",
        "src/kvbench/runtime/phase3_endpoint_audit.py",
        "src/kvbench/runtime/phase3_raw_audit_evidence.py",
        "src/kvbench/runtime/phase3_report.py",
        "src/kvbench/runtime/phase3_report_publication.py",
        "src/kvbench/runtime/phase3_worker.py",
        "src/kvbench/runtime/phase3_worker_channels.py",
        "src/kvbench/runtime/process_supervision.py",
        "src/kvbench/runtime/static_cache.py",
        "src/kvbench/runtime/telemetry.py",
        "src/kvbench/runtime/timing.py",
        "tests/schema/test_config_schema.py",
        "tests/schema/test_phase3_schema.py",
        "tests/unit/test_model_loader_receipt.py",
        "tests/unit/test_phase3_artifacts.py",
        "tests/unit/test_phase3_cli.py",
        "tests/unit/test_phase3_governance.py",
        "tests/unit/test_phase3_campaign.py",
        "tests/unit/test_phase3_allocator_controls.py",
        "tests/unit/test_phase3_endpoint_audit.py",
        "tests/unit/test_phase3_audit_operation.py",
        "tests/unit/test_phase3_report.py",
        "tests/unit/test_phase3_raw_audit_evidence.py",
        "tests/unit/test_phase3_raw_audit_ipc.py",
        "tests/unit/test_phase3_runtime.py",
        "tests/unit/test_allocation_attribution.py",
        "tests/unit/test_gqa_device_dispatch.py",
        "tests/unit/test_gqa_taxonomy.py",
        "tests/unit/test_phase3_gqa_device_dispatch_geometry.py",
        "tests/unit/test_phase3_report_publication.py",
        "tests/unit/test_process_supervision.py",
        "tests/cuda/test_phase3_allocation_attribution.py",
        "tests/cuda/test_phase3_endpoint_audit.py",
        "tests/cuda/test_phase3_gqa_device_dispatch.py",
        "tests/cuda/test_phase3_process_supervision.py",
        "tests/cuda/test_phase3_runtime_cuda.py",
        "tests/cuda/test_phase3_full_model.py",
        "tests/graph/test_phase3_full_model_graph.py",
        "tests/graph/test_phase3_runtime_graph.py",
    }
)
PHASE4_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/plans/phase4-common-adapter.md",
        "docs/phase_reports/phase4.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "docs/evidence/phase4/method-admission.json",
        "docs/evidence/phase4/smoke-index.json",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/__init__.py",
        "src/kvbench/adapters/base.py",
        "src/kvbench/adapters/bf16.py",
        "src/kvbench/adapters/factory.py",
        "src/kvbench/runtime/bf16_endpoint.py",
        "src/kvbench/runtime/cuda_graph.py",
        "src/kvbench/runtime/fixed_l_runner.py",
        "src/kvbench/runtime/growing_context_runner.py",
        "src/kvbench/runtime/method_harness.py",
        "src/kvbench/runtime/numerical.py",
        "src/kvbench/runtime/phase3_coordinator.py",
        "src/kvbench/runtime/phase3_endpoint_audit.py",
        "src/kvbench/runtime/phase3_worker.py",
        "src/kvbench/runtime/phase4_smoke.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/method_admission.py",
        "src/kvbench/schema/phase3.py",
        "tests/schema/test_method_admission.py",
        "tests/cuda/test_phase3_full_model.py",
        "tests/cuda/test_phase4_adapter_cuda.py",
        "tests/graph/test_phase3_full_model_graph.py",
        "tests/graph/test_phase4_adapter_graph.py",
        "tests/unit/test_phase3_endpoint_audit.py",
        "tests/unit/test_phase3_runtime.py",
        "tests/unit/test_phase4_adapter.py",
        "tests/unit/test_phase4_governance.py",
    }
)

PHASE5_FIXTURE_CONFIGURATIONS = (
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
)
PHASE5_FIXTURE_INPUTS = (
    "append_key.bf16.bin",
    "append_value.bf16.bin",
    "decode_query.bf16.bin",
    "prefill_key.bf16.bin",
    "prefill_value.bf16.bin",
)
PHASE5_FIXTURE_FILES = (
    "append_slot.uint8.bin",
    "cache_after_append.uint8.bin",
    "cache_after_store.uint8.bin",
    "checksums.sha256",
    "decode_output.bf16.bin",
    "kernel_trace.json",
    "manifest.json",
)
PHASE5_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "Makefile",
        "docker/reference-turboquant.Dockerfile",
        "docs/blockers.md",
        "docs/method_notes/turboquant.md",
        "docs/plans/phase5-turboquant-reference.md",
        "docs/phase_reports/phase5-turboquant-reference.md",
        "docs/phase_reports/phase6-turboquant-measurement-blocked.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "reference/turboquant/README.md",
        "reference/turboquant/bootstrap_environment.py",
        "reference/turboquant/environment.json",
        "reference/turboquant/fixtures/checksums.sha256",
        "reference/turboquant/fixtures/fixture_set.json",
        "reference/turboquant/generate_fixtures.py",
        "reference/turboquant/python-freeze.txt",
        "reference/turboquant/source_manifest.json",
        "reference/turboquant/validate_fixtures.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase5_turboquant_reference.py",
        "third_party/LOCK.json",
        "third_party/NOTICE.md",
        *(
            f"reference/turboquant/fixtures/inputs/{name}"
            for name in PHASE5_FIXTURE_INPUTS
        ),
        *(
            f"reference/turboquant/fixtures/{configuration}/{name}"
            for configuration in PHASE5_FIXTURE_CONFIGURATIONS
            for name in PHASE5_FIXTURE_FILES
        ),
    }
)
PHASE6A_ALLOWED_PATHS = frozenset(
    {
        ".dockerignore",
        ".gitignore",
        "Makefile",
        "docker/measurement.Dockerfile",
        "docs/blockers.md",
        "docs/decisions/0016-measurement-container-execution-authority.md",
        "docs/phase_reports/phase6a-measurement-container-and-r2-blocked.md",
        "docs/phase_reports/phase6a-measurement-container-and-r2.md",
        "docs/plans/phase6a-measurement-container-and-r2.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "preflight/e00_manifest.schema.json",
        "preflight/measurement-container-system-packages.expected.json",
        "preflight/measurement-container-system-packages.lock.json",
        "preflight/process_query.py",
        "preflight/run_preflight.py",
        "scripts/phase6a_bf16_parity.py",
        "scripts/r2_artifact.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_measurement_container.py",
        "tests/unit/test_phase6a_bf16_parity.py",
        "tests/unit/test_phase6a_governance.py",
        "tests/unit/test_preflight_unit.py",
        "tests/unit/test_r2_artifact.py",
    }
)
PHASE6A_E00_ALLOWED_PATHS = frozenset(
    {
        "preflight/e00_manifest.schema.json",
        "preflight/measurement-container-system-packages.expected.json",
        "preflight/measurement-container-system-packages.lock.json",
        "preflight/process_query.py",
        "preflight/run_preflight.py",
        "tests/unit/test_preflight_unit.py",
    }
)
PHASE6_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "configs/methods/turboquant.yaml",
        "docs/blockers.md",
        "docs/evidence/phase6/r2-admission-outer-publication.json",
        "docs/evidence/phase6/r2-admission-publication.json",
        "docs/evidence/phase6/r2-publication.json",
        "docs/evidence/phase6/turboquant-method-admission.json",
        "docs/phase_reports/phase6-turboquant-measurement-adapter.md",
        "docs/phase_reports/phase6-turboquant-measurement-adapter-pass.md",
        "docs/plans/phase6-turboquant-measurement-adapter.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase6_r2_outer_bundle.py",
        "scripts/phase6_turboquant_admission.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/__init__.py",
        "src/kvbench/adapters/factory.py",
        "src/kvbench/adapters/turboquant.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/cuda_graph.py",
        "src/kvbench/runtime/fixed_l_runner.py",
        "src/kvbench/runtime/growing_context_runner.py",
        "src/kvbench/runtime/numerical.py",
        "src/kvbench/runtime/phase3_coordinator.py",
        "src/kvbench/runtime/turboquant_admission.py",
        "src/kvbench/runtime/turboquant_audit.py",
        "src/kvbench/runtime/turboquant_cache.py",
        "src/kvbench/runtime/turboquant_fixture.py",
        "src/kvbench/runtime/turboquant_session.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/base.py",
        "src/kvbench/schema/method_admission.py",
        "src/kvbench/schema/phase6.py",
        "src/kvbench/third_party/__init__.py",
        "src/kvbench/third_party/vllm_turboquant/__init__.py",
        "src/kvbench/third_party/vllm_turboquant/centroids.py",
        "src/kvbench/third_party/vllm_turboquant/compat.py",
        "src/kvbench/third_party/vllm_turboquant/config.py",
        "src/kvbench/third_party/vllm_turboquant/provenance.json",
        "src/kvbench/third_party/vllm_turboquant/triton_decode_attention.py",
        "src/kvbench/third_party/vllm_turboquant/triton_turboquant_decode.py",
        "src/kvbench/third_party/vllm_turboquant/triton_turboquant_store.py",
        "tests/cuda/phase6_turboquant_sanitizer_probe.py",
        "tests/cuda/test_phase6_turboquant_cuda.py",
        "tests/graph/test_phase6_turboquant_graph.py",
        "tests/schema/test_config_schema.py",
        "tests/schema/test_phase6_schema.py",
        "tests/unit/test_phase4_adapter.py",
        "tests/unit/test_phase5_turboquant_reference.py",
        "tests/unit/test_phase6a_governance.py",
        "tests/unit/test_process_supervision.py",
        "tests/unit/test_phase6_artifacts.py",
        "tests/unit/test_phase6_governance.py",
        "tests/unit/test_phase6_r2_outer_bundle.py",
        "tests/unit/test_phase6_turboquant_adapter.py",
        "tests/unit/test_phase6_turboquant_fixture.py",
        "tests/unit/test_phase6_turboquant_session.py",
    }
)
PHASE7_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docker/reference-kivi.Dockerfile",
        "docs/blockers.md",
        "docs/decisions/0017-kivi-source-authority-and-gqa-materialization.md",
        "docs/decisions/0018-kivi-b019-native-gqa-patch-authority.md",
        "docs/evidence/phase7/kivi-b019-remediation.json",
        "docs/evidence/phase7/kivi-reference-r2-publication.json",
        "docs/evidence/phase7/kivi-source-audit.json",
        "docs/method_notes/kivi.md",
        "docs/phase_reports/phase7-kivi-b019-remediation.md",
        "docs/phase_reports/phase7-kivi-reference.md",
        "docs/phase_reports/phase7-kivi-reference-blocked.md",
        "docs/plans/phase7-kivi-b019-remediation.md",
        "docs/plans/phase7-kivi-reference.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "reference/kivi/README.md",
        "reference/kivi/build_manifest.json",
        "reference/kivi/environment.json",
        "reference/kivi/fixtures/checksums.sha256",
        "reference/kivi/fixtures/fixture_set.json",
        "reference/kivi/fixtures/k2v2/fixture.json",
        "reference/kivi/fixtures/k2v2/manifest.json",
        "reference/kivi/fixtures/k2v4/fixture.json",
        "reference/kivi/fixtures/k2v4/manifest.json",
        "reference/kivi/fixtures/k4v2/fixture.json",
        "reference/kivi/fixtures/k4v2/manifest.json",
        "reference/kivi/fixtures/k4v4/fixture.json",
        "reference/kivi/fixtures/k4v4/manifest.json",
        "reference/kivi/generate_fixtures.py",
        "reference/kivi/python-freeze.txt",
        "reference/kivi/source_manifest.json",
        "reference/kivi/validate_fixtures.py",
        "scripts/validate_kivi_b019_patch.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_measurement_container.py",
        "tests/unit/test_phase7_kivi_b019_remediation.py",
        "tests/unit/test_phase7_kivi_reference.py",
        "tests/unit/test_phase7_kivi_source_audit.py",
        "third_party/LOCK.json",
        "third_party/NOTICE.md",
        "third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch",
        "third_party/patches/kivi/manifest.json",
    }
)

PHASE8_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0019-phase7-allocation-ratio-terminology-erratum.md",
        "docs/evidence/phase8/kivi-method-admission.json",
        "docs/evidence/phase8/kivi-method-admission.sha256",
        "docs/evidence/phase8/r2-admission-outer-publication.json",
        "docs/evidence/phase8/r2-admission-publication.json",
        "docs/method_notes/kivi.md",
        "docs/phase_reports/phase8-kivi-measurement-adapter.md",
        "docs/plans/phase8-kivi-measurement-adapter.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase8_kivi_admission.py",
        "scripts/phase8_r2_outer_bundle.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/__init__.py",
        "src/kvbench/adapters/factory.py",
        "src/kvbench/adapters/kivi.py",
        "src/kvbench/runtime/allocation.py",
        "src/kvbench/runtime/allocation_attribution.py",
        "src/kvbench/runtime/kivi_admission.py",
        "src/kvbench/runtime/kivi_allocation.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/kivi_cache.py",
        "src/kvbench/runtime/kivi_fixture.py",
        "src/kvbench/runtime/numerical.py",
        "src/kvbench/runtime/kivi_session.py",
        "src/kvbench/runtime/process_supervision.py",
        "src/kvbench/schema/phase8.py",
        "tests/cuda/phase8_kivi_sanitizer_probe.py",
        "tests/cuda/test_phase8_kivi_cuda.py",
        "tests/graph/test_phase8_kivi_graph.py",
        "tests/unit/test_phase7_kivi_b019_remediation.py",
        "tests/unit/test_phase7_kivi_reference.py",
        "tests/unit/test_phase7_kivi_source_audit.py",
        "tests/unit/test_phase8_governance.py",
        "tests/unit/test_phase8_artifacts.py",
        "tests/unit/test_phase8_kivi_adapter.py",
        "tests/unit/test_phase8_kivi_admission.py",
        "tests/unit/test_phase8_kivi_admission_driver.py",
        "tests/unit/test_phase8_kivi_allocation.py",
        "tests/unit/test_phase8_kivi_cache.py",
        "tests/unit/test_phase8_kivi_fixture.py",
        "tests/unit/test_phase8_kivi_schema.py",
        "tests/unit/test_phase8_kivi_session.py",
        "tests/unit/test_phase8_make_targets.py",
        "tests/unit/test_phase8_process_supervision.py",
        "tests/unit/test_phase8_r2_outer_bundle.py",
        "tests/unit/test_phase8_ratio_terminology.py",
    }
)

PHASE9P_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/decisions/0020-kvquant-upstream-gqa-patch.md",
        "docs/evidence/phase9p/patch-manifest.json",
        "docs/evidence/phase9p/test-report.json",
        "docs/method_notes/kvquant.md",
        "docs/phase_reports/phase9p-kvquant-upstream-gqa-patch.md",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase7_kivi_b019_remediation.py",
        "tests/unit/test_phase9p_governance.py",
        "third_party/LOCK.json",
        "third_party/NOTICE.md",
    }
)

KVQUANT_PATCH_CUSTODY_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/decisions/0021-kvquant-patch-main-repository-custody.md",
        "docs/decisions/0022-phase9-blocked-report-custody.md",
        "docs/evidence/phase9/blocked-report-custody.json",
        "docs/method_notes/kvquant.md",
        "docs/phase_reports/phase9-kvquant-calibration-blocked.md",
        "scripts/validate_kvquant_gqa_patch.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase9p_patch_custody.py",
        "third_party/LOCK.json",
        "third_party/NOTICE.md",
        "third_party/patches/kvquant/0001-llama31-native-gqa.patch",
        "third_party/patches/kvquant/manifest.json",
    }
)

PHASE9_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "Makefile",
        "configs/methods/kvquant.yaml",
        "docker/calibration-kvquant.Dockerfile",
        "docker/calibration-kvquant.Dockerfile.dockerignore",
        "docker/calibration-kvquant.image.json",
        "docker/calibration-kvquant.python-freeze.txt",
        "docker/calibration-kvquant.requirements.txt",
        "docs/blockers.md",
        "docs/evidence/phase9/calibration-checksums.sha256",
        "docs/evidence/phase9/calibration-manifest.json",
        "docs/evidence/phase9/dataset-selection.json",
        "docs/evidence/phase9/layer-stats-summary.json",
        "docs/evidence/phase9/model-snapshot-manifest.json",
        "docs/evidence/phase9/r2-publication.json",
        "docs/evidence/phase9/reproducibility.json",
        "docs/method_notes/kvquant.md",
        "docs/phase_reports/phase9-kvquant-calibration.md",
        "docs/plans/phase9-kvquant-calibration.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase9_kvquant_calibration.py",
        "scripts/phase9_kvquant_worker.py",
        "scripts/r2_artifact.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_r2_artifact.py",
        "src/kvbench/runtime/__init__.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/config.py",
        "src/kvbench/schema/phase9.py",
        "src/kvbench/schema/phase9_config.py",
        "tests/schema/test_config_schema.py",
        "tests/schema/test_phase9_schema.py",
        "tests/unit/test_phase9_calibration.py",
        "tests/unit/test_phase9_governance.py",
        "tests/unit/test_phase9_scope.py",
        "tests/unit/test_phase9p_governance.py",
    }
)

PHASE10_FIXTURE_FAMILIES = ("kvq4", "kvq3", "kvq2")
PHASE10_FIXTURE_CASES = (
    "key_zero_value_fixed12",
    "key_few_value_fixed12",
    "key_cap_value_fixed12",
)
PHASE10_FIXTURE_MEMBERS = (
    "fixture_manifest.json",
    "inputs.safetensors",
    "dense_payload.safetensors",
    "metadata.safetensors",
    "sparse_values.safetensors",
    "sparse_indices.safetensors",
    "sink.safetensors",
    "store_state.safetensors",
    "append_state.safetensors",
    "decode_output.safetensors",
    "byte_breakdown.json",
    "checksums.sha256",
)
PHASE10_FIXTURE_PATHS = frozenset(
    f"reference/kvquant/fixtures/{family}/{case}/{member}"
    for family in PHASE10_FIXTURE_FAMILIES
    for case in PHASE10_FIXTURE_CASES
    for member in PHASE10_FIXTURE_MEMBERS
)
PHASE10_SAFE_TENSOR_PATHS = frozenset(
    path for path in PHASE10_FIXTURE_PATHS if path.endswith(".safetensors")
)
PHASE10_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/decisions/0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md",
        "docs/evidence/phase10/blocked-report-custody.json",
        "docs/evidence/phase10/cuda-validation.json",
        "docs/evidence/phase10/r2-publication.json",
        "docs/phase_reports/phase10-kvquant-reference-blocked.md",
        "docs/phase_reports/phase10-kvquant-reference.md",
        "docs/plans/phase10-kvquant-reference.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "docker/reference-kvquant.Dockerfile",
        "reference/kvquant/README.md",
        "reference/kvquant/build_manifest.json",
        "reference/kvquant/calibration_manifest.json",
        "reference/kvquant/environment.json",
        "reference/kvquant/fixtures/COMPLETE",
        "reference/kvquant/fixtures/artifact_inventory.json",
        "reference/kvquant/fixtures/checksums.sha256",
        "reference/kvquant/fixtures/manifest.json",
        "reference/kvquant/fixtures/reference_trace.json",
        "reference/kvquant/generate_fixtures.py",
        "reference/kvquant/source_manifest.json",
        "reference/kvquant/validate_fixtures.py",
        "scripts/validate_phase2.py",
        "tests/cuda/phase10_kvquant_sanitizer_probe.py",
        "tests/unit/test_measurement_container.py",
        "tests/unit/test_phase9_governance.py",
        "tests/unit/test_phase9p_governance.py",
        "tests/unit/test_phase10_kvquant_reference.py",
        "tests/unit/test_phase10_scope.py",
    }
    | PHASE10_FIXTURE_PATHS
)

PHASE11P_ALLOWED_PATHS = frozenset(
    {
        "docs/decisions/0024-kvquant-graph-safe-caller-owned-cuda-apis.md",
    }
)
PHASE11PR_FIXTURE_PATHS = frozenset(
    f"reference/kvquant_phase11pr/fixtures/{family}/{case}/{member}"
    for family in PHASE10_FIXTURE_FAMILIES
    for case in PHASE10_FIXTURE_CASES
    for member in PHASE10_FIXTURE_MEMBERS
)
PHASE11PR_SAFE_TENSOR_PATHS = frozenset(
    path
    for path in PHASE11PR_FIXTURE_PATHS
    if path.endswith(".safetensors")
)
PHASE11PR_FIXTURE_ROOT_PATHS = frozenset(
    {
        "reference/kvquant_phase11pr/fixtures/COMPLETE",
        "reference/kvquant_phase11pr/fixtures/artifact_inventory.json",
        "reference/kvquant_phase11pr/fixtures/checksums.sha256",
        "reference/kvquant_phase11pr/fixtures/manifest.json",
        "reference/kvquant_phase11pr/fixtures/reference_trace.json",
        "reference/kvquant_phase11pr/fixtures/reuse_proof.json",
        (
            "reference/kvquant_phase11pr/fixtures/authority/"
            "build_manifest.json"
        ),
        (
            "reference/kvquant_phase11pr/fixtures/authority/"
            "calibration_manifest.json"
        ),
        (
            "reference/kvquant_phase11pr/fixtures/authority/"
            "environment.json"
        ),
        (
            "reference/kvquant_phase11pr/fixtures/authority/"
            "source_manifest.json"
        ),
    }
)
PHASE11PR_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/decisions/0025-kvquant-deterministic-kvq3-value-pack.md",
        "docs/evidence/phase11pr/cuda-validation.json",
        "docs/evidence/phase11pr/r2-publication.json",
        "docs/phase_reports/phase11p-r-kvq3-value-pack.md",
        "reference/kvquant_phase11pr/generate_corrected_bundle.py",
        "reference/kvquant_phase11pr/validate_corrected_bundle.py",
        "scripts/validate_kvquant_graphsafe_patch.py",
        "scripts/validate_phase2.py",
        "tests/cuda/phase11pr_kvq3_pack_validation.py",
        "tests/unit/test_phase9p_patch_custody.py",
        "tests/unit/test_phase11pr_scope.py",
        (
            "third_party/patches/kvquant/"
            "0002-graphsafe-kvq3-deterministic.patch"
        ),
        (
            "third_party/patches/kvquant/"
            "graphsafe-kvq3-manifest.json"
        ),
    }
    | PHASE11PR_FIXTURE_PATHS
    | PHASE11PR_FIXTURE_ROOT_PATHS
)
PHASE11_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/decisions/0026-kvquant-pre-rope-adapter-boundary.md",
        "docs/evidence/phase11/kvquant-method-admission.json",
        "docs/evidence/phase11/kvquant-method-admission.sha256",
        "docs/evidence/phase11/r2-admission-outer-publish.stderr.txt",
        "docs/evidence/phase11/r2-admission-outer-publish.stdout.json",
        "docs/evidence/phase11/r2-admission-outer-publication.json",
        "docs/evidence/phase11/r2-admission-outer-verify.stderr.txt",
        "docs/evidence/phase11/r2-admission-outer-verify.stdout.json",
        "docs/evidence/phase11/r2-admission-publish.stderr.txt",
        "docs/evidence/phase11/r2-admission-publish.stdout.json",
        "docs/evidence/phase11/r2-admission-publication.json",
        "docs/evidence/phase11/r2-admission-verify.stderr.txt",
        "docs/evidence/phase11/r2-admission-verify.stdout.json",
        "docs/method_notes/kvquant.md",
        "docs/phase_reports/phase11-kvquant-measurement-adapter.md",
        "docs/plans/phase11-kvquant-measurement-adapter.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase11_kvquant_admission.py",
        "scripts/phase11_r2_outer_bundle.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/__init__.py",
        "src/kvbench/adapters/base.py",
        "src/kvbench/adapters/factory.py",
        "src/kvbench/adapters/kvquant.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/allocation_attribution.py",
        "src/kvbench/runtime/bf16_endpoint.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_fixture.py",
        "src/kvbench/runtime/kvquant_session.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/phase11.py",
        "tests/cuda/phase11_kvquant_sanitizer_probe.py",
        "tests/cuda/test_phase11_kvquant_cuda.py",
        "tests/graph/test_phase11_kvquant_graph.py",
        "tests/unit/test_phase11_kvquant_adapter.py",
        "tests/unit/test_phase11_kvquant_admission.py",
        "tests/unit/test_phase11_kvquant_admission_driver.py",
        "tests/unit/test_phase11_kvquant_cache.py",
        "tests/unit/test_phase11_kvquant_factory.py",
        "tests/unit/test_phase11_kvquant_fixture.py",
        "tests/unit/test_phase11_kvquant_session.py",
        "tests/unit/test_phase11_make_targets.py",
        "tests/unit/test_phase11_r2_outer_bundle.py",
        "tests/unit/test_phase11_scope.py",
        "tests/unit/test_phase6_governance.py",
        "tests/unit/test_phase9_governance.py",
        "tests/unit/test_phase9p_governance.py",
        "tests/unit/test_phase9p_patch_custody.py",
        "tests/unit/test_phase10_kvquant_reference.py",
        "tests/unit/test_phase11pr_scope.py",
    }
)
PHASE11D_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        (
            "docs/decisions/"
            "0027-kvquant-deterministic-long-context-value-decode.md"
        ),
        "docs/evidence/phase11d/cuda-validation.json",
        (
            "docs/phase_reports/"
            "phase11d-kvquant-deterministic-long-context-cuda.md"
        ),
        "scripts/phase11d_kvquant_validation.py",
        "scripts/validate_kvquant_long_context_patch.py",
        "scripts/validate_phase2.py",
        (
            "tests/cuda/"
            "phase11d_kvquant_long_context_validation.py"
        ),
        "tests/unit/test_phase11d_scope.py",
        "tests/unit/test_phase9p_patch_custody.py",
        (
            "third_party/patches/kvquant/"
            "0003-deterministic-long-context-value-decode.patch"
        ),
        (
            "third_party/patches/kvquant/"
            "deterministic-long-context-manifest.json"
        ),
    }
)
PHASE11R_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase11/kvquant-method-admission.json",
        "docs/evidence/phase11/kvquant-method-admission.sha256",
        "docs/evidence/phase11/r2-admission-outer-publish.stderr.txt",
        "docs/evidence/phase11/r2-admission-outer-publish.stdout.json",
        "docs/evidence/phase11/r2-admission-outer-publication.json",
        "docs/evidence/phase11/r2-admission-outer-verify.stderr.txt",
        "docs/evidence/phase11/r2-admission-outer-verify.stdout.json",
        "docs/evidence/phase11/r2-admission-publish.stderr.txt",
        "docs/evidence/phase11/r2-admission-publish.stdout.json",
        "docs/evidence/phase11/r2-admission-publication.json",
        "docs/evidence/phase11/r2-admission-verify.stderr.txt",
        "docs/evidence/phase11/r2-admission-verify.stdout.json",
        "docs/method_notes/kvquant.md",
        (
            "docs/phase_reports/"
            "phase11r-kvquant-measurement-adapter.md"
        ),
        "docs/plans/phase11r-kvquant-admission-rerun.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase11_kvquant_admission.py",
        "scripts/phase11_r2_outer_bundle.py",
        "scripts/validate_kvquant_long_context_patch.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/kvquant.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_session.py",
        "src/kvbench/schema/phase11.py",
        "tests/cuda/phase11_kvquant_sanitizer_probe.py",
        "tests/cuda/test_phase11_kvquant_cuda.py",
        "tests/graph/test_phase11_kvquant_graph.py",
        "tests/unit/test_phase11_kvquant_adapter.py",
        "tests/unit/test_phase11_kvquant_admission.py",
        "tests/unit/test_phase11_kvquant_admission_driver.py",
        "tests/unit/test_phase11_kvquant_cache.py",
        "tests/unit/test_phase11_kvquant_factory.py",
        "tests/unit/test_phase11_kvquant_session.py",
        "tests/unit/test_phase11_make_targets.py",
        "tests/unit/test_phase11_r2_outer_bundle.py",
        "tests/unit/test_phase11r_scope.py",
    }
)
PHASE12E_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        (
            "docs/decisions/"
            "0028-phase12e-kivi-historical-source-validation.md"
        ),
        "scripts/validate_phase2.py",
        "src/kvbench/runtime/kivi_admission.py",
        "tests/unit/test_phase8_kivi_admission.py",
        (
            "tests/unit/"
            "test_phase12e_kivi_historical_authority.py"
        ),
        "tests/unit/test_phase12e_scope.py",
    }
)
PHASE11DQ23_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        (
            "docs/decisions/"
            "0029-kvquant-deterministic-long-context-q3-q2-value-decode.md"
        ),
        "docs/evidence/phase11dq23/cuda-validation.json",
        (
            "docs/phase_reports/"
            "phase11dq23-kvquant-deterministic-long-context-q3-q2.md"
        ),
        "scripts/phase11dq23_kvquant_validation.py",
        "scripts/validate_kvquant_q23_long_context_patch.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/kvquant.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_session.py",
        "tests/cuda/phase11_kvquant_sanitizer_probe.py",
        (
            "tests/cuda/"
            "phase11dq23_kvquant_long_context_validation.py"
        ),
        "tests/cuda/test_phase11_kvquant_cuda.py",
        "tests/graph/test_phase11_kvquant_graph.py",
        "tests/unit/test_phase11_kvquant_adapter.py",
        "tests/unit/test_phase11_kvquant_cache.py",
        "tests/unit/test_phase11_kvquant_session.py",
        "tests/unit/test_phase11dq23_scope.py",
        "tests/unit/test_phase12e_scope.py",
        "tests/unit/test_phase9p_patch_custody.py",
        (
            "third_party/patches/kvquant/"
            "0004-deterministic-long-context-q3-q2-value-decode.patch"
        ),
        (
            "third_party/patches/kvquant/"
            "deterministic-long-context-q3-q2-manifest.json"
        ),
    }
)
PHASE11RQ23_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase11rq23/kvquant-method-admission.json",
        "docs/evidence/phase11rq23/kvquant-method-admission.sha256",
        (
            "docs/evidence/phase11rq23/"
            "r2-admission-outer-publish.stderr.txt"
        ),
        (
            "docs/evidence/phase11rq23/"
            "r2-admission-outer-publish.stdout.json"
        ),
        (
            "docs/evidence/phase11rq23/"
            "r2-admission-outer-publication.json"
        ),
        (
            "docs/evidence/phase11rq23/"
            "r2-admission-outer-verify.stderr.txt"
        ),
        (
            "docs/evidence/phase11rq23/"
            "r2-admission-outer-verify.stdout.json"
        ),
        "docs/evidence/phase11rq23/r2-admission-publish.stderr.txt",
        "docs/evidence/phase11rq23/r2-admission-publish.stdout.json",
        "docs/evidence/phase11rq23/r2-admission-publication.json",
        "docs/evidence/phase11rq23/r2-admission-verify.stderr.txt",
        "docs/evidence/phase11rq23/r2-admission-verify.stdout.json",
        "docs/method_notes/kvquant.md",
        (
            "docs/phase_reports/"
            "phase11rq23-kvquant-measurement-adapter.md"
        ),
        "docs/plans/phase11rq23-kvquant-admission-rerun.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase11_kvquant_admission.py",
        "scripts/phase11_r2_outer_bundle.py",
        "scripts/validate_phase2.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/phase11.py",
        "src/kvbench/runtime/artifacts.py",
        "tests/cuda/phase11_kvquant_sanitizer_probe.py",
        "tests/unit/test_phase11rq23_kvquant_admission.py",
        "tests/unit/test_phase11rq23_kvquant_admission_driver.py",
        "tests/unit/test_phase11rq23_make_targets.py",
        "tests/unit/test_phase11rq23_r2_outer_bundle.py",
        "tests/unit/test_phase11rq23_r2_outer_profile.py",
        "tests/unit/test_phase11rq23_schema.py",
        "tests/unit/test_phase11rq23_scope.py",
    }
)
PHASE12_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase12/r2-publication.json",
        "docs/evidence/phase12/unified-admission.json",
        "docs/phase_reports/phase12-unified-admission.md",
        "docs/plans/phase12-unified-admission.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase12_unified_admission.py",
        "scripts/validate_phase2.py",
        "src/kvbench/schema/phase12.py",
        "tests/unit/test_phase12_artifact_lifecycle.py",
        "tests/unit/test_phase12_schema.py",
        "tests/unit/test_phase12_scope.py",
        "tests/unit/test_phase12_unified_admission.py",
    }
)
PHASE13_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase13/pilot_qc.json",
        "docs/evidence/phase13/r2-publication.json",
        "docs/phase_reports/phase13-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/plans/phase13-pilot-execution-order.json",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase12_unified_admission.py",
        "scripts/phase13_pilot.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase12_unified_admission.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13_scope.py",
    }
)
PHASE13B_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0030-compressed-static-cache-batch-geometry.md",
        "docs/evidence/phase13b/cuda-validation.json",
        "docs/evidence/phase13b/kivi-method-admission.json",
        "docs/evidence/phase13b/kvquant-method-admission.json",
        "docs/evidence/phase13b/r2-publication.json",
        "docs/evidence/phase13b/turboquant-method-admission.json",
        "docs/phase_reports/phase13b-compressed-batch-geometry.md",
        "docs/plans/phase13b-compressed-batch-geometry.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase12_unified_admission.py",
        "scripts/phase11_kvquant_admission.py",
        "scripts/phase13b_compressed_batch_admission.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/kivi.py",
        "src/kvbench/adapters/kvquant.py",
        "src/kvbench/adapters/turboquant.py",
        "src/kvbench/runtime/artifacts.py",
        "src/kvbench/runtime/kivi_admission.py",
        "src/kvbench/runtime/kivi_cache.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_session.py",
        "src/kvbench/runtime/turboquant_cache.py",
        "src/kvbench/schema/__init__.py",
        "src/kvbench/schema/phase13b.py",
        "tests/cuda/phase11_kvquant_sanitizer_probe.py",
        "tests/cuda/phase13b_batch_sanitizer_probe.py",
        "tests/cuda/test_phase11_kvquant_cuda.py",
        "tests/cuda/test_phase13b_batch_geometry_cuda.py",
        "tests/graph/test_phase11_kvquant_graph.py",
        "tests/unit/test_phase11_kvquant_adapter.py",
        "tests/unit/test_phase11_kvquant_cache.py",
        "tests/unit/test_phase11_kvquant_session.py",
        "tests/unit/test_phase11rq23_kvquant_admission_driver.py",
        "tests/unit/test_phase12_unified_admission.py",
        "tests/unit/test_phase12e_kivi_historical_authority.py",
        "tests/unit/test_phase8_kivi_admission.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13b_batch_geometry.py",
        "tests/unit/test_phase13b_scope.py",
        "tests/unit/test_phase7_kivi_b019_remediation.py",
        "tests/unit/test_phase7_kivi_source_audit.py",
    }
)
PHASE13R_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase13r/pilot_qc.json",
        "docs/evidence/phase13r/r2-publication.json",
        "docs/phase_reports/phase13r-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13_scope.py",
    }
)
PHASE13F_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0031-phase13-end-to-end-prefix-feasibility.md",
        "docs/evidence/phase13f/feasibility-summary.json",
        "docs/evidence/phase13f/r2-publication.json",
        "docs/phase_reports/phase13f-prefix-feasibility.md",
        "docs/plans/phase13f-prefix-feasibility.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13f_feasibility.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13f_feasibility.py",
    }
)
PHASE13R2_ALLOWED_PATHS = frozenset(
    {
        "docs/blockers.md",
        "docs/evidence/phase13r2/pilot_qc.json",
        "docs/evidence/phase13r2/r2-publication.json",
        "docs/phase_reports/phase13r2-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
    }
)
PHASE13T_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0032-phase13-stage-aware-supervision.md",
        "docs/evidence/phase13t/r2-publication.json",
        "docs/evidence/phase13t/timeout-validation.json",
        "docs/evidence/phase13tr/pilot_qc.json",
        "docs/evidence/phase13tr/r2-publication.json",
        "docs/phase_reports/phase13t-supervisor-timeout.md",
        "docs/phase_reports/phase13tr-pilot-scan.md",
        "docs/plans/phase13t-supervisor-timeout.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13t_timeout.py",
        "scripts/validate_phase2.py",
        "src/kvbench/runtime/process_supervision.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13t_timeout.py",
    }
)
PHASE13PR_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0033-phase13-untimed-prefix-state-reuse.md",
        "docs/evidence/phase13/pilot_qc.json",
        "docs/evidence/phase13/r2-publication.json",
        "docs/phase_reports/phase13-pilot-scan.md",
        "docs/plans/phase13-pilot-scan.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13_prefix_state.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13pr_prefix_state.py",
    }
)
PHASE13PB_ALLOWED_PATHS = frozenset(
    {
        "docs/decisions/0034-phase13-batch-exact-prefix-state.md",
        "docs/evidence/phase13pb/prefix-equivalence.json",
        "docs/evidence/phase13pb/r2-publication.json",
        "docs/phase_reports/phase13pb-batch-exact-prefix-state.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13_prefix_state.py",
        "scripts/phase13pb_prefix_remediation.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13pr_prefix_state.py",
    }
)
PHASE13PC_ALLOWED_PATHS = frozenset(
    {
        "docs/blockers.md",
        "docs/decisions/0035-phase13-equivalence-cuda-context-isolation.md",
        "docs/evidence/phase13pc/isolation-validation.json",
        "docs/evidence/phase13pc/r2-publication.json",
        "docs/phase_reports/phase13pc-cuda-context-isolation.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_pilot.py",
        "scripts/phase13pc_cuda_context_isolation.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13pc_cuda_context_isolation.py",
    }
)
PHASE13RQ4_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0036-kvquant-q4-value-decode-workspace-geometry.md",
        "docs/decisions/0037-phase13-persistent-chunked-prefix-construction.md",
        "docs/decisions/0038-phase13-prefix-child-allocator.md",
        "docs/evidence/phase13rq4/cuda-validation.json",
        "docs/evidence/phase13rq4/kvquant-q4-method-admission.json",
        "docs/evidence/phase13rq4/r2-publication.json",
        "docs/evidence/phase13rq4/unified-admission.json",
        "docs/evidence/phase13rq4/workspace-boundaries.json",
        "docs/evidence/phase13-successor/r2-publication.json",
        "docs/phase_reports/phase13r-kvquant-q4-workspace.md",
        "docs/phase_reports/phase13-successor-pilot-scan.md",
        "docs/plans/phase13-pilot-execution-order.json",
        "docs/plans/phase13-pilot-scan.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase11_kvquant_admission.py",
        "scripts/phase12_unified_admission.py",
        "scripts/phase13_pilot.py",
        "scripts/phase13r_q4_workspace.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/kvquant.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_session.py",
        "src/kvbench/runtime/numerical.py",
        "tests/cuda/phase11_kvquant_sanitizer_probe.py",
        "tests/cuda/phase13r_q4_workspace_sanitizer_probe.py",
        "tests/cuda/test_phase11_kvquant_cuda.py",
        "tests/graph/test_phase11_kvquant_graph.py",
        "tests/unit/test_phase11_kvquant_cache.py",
        "tests/unit/test_phase11_kvquant_session.py",
        "tests/unit/test_phase13r_q4_workspace.py",
        "tests/unit/test_phase13_pilot.py",
        "tests/unit/test_phase13pr_prefix_state.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13rq4_scope.py",
    }
)
PHASE13D_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase13d/r2-publication.json",
        "docs/phase_reports/phase13d-continuation.md",
        "docs/phase_reports/phase13d-knee-densification.md",
        "docs/plans/phase13d-candidate-table.json",
        "docs/plans/phase13d-execution-order.json",
        "docs/plans/phase13d-knee-densification.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13d_knee_densification.py",
        "scripts/phase13d_continuation.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase13_scope.py",
        "tests/unit/test_phase13d_continuation.py",
        "tests/unit/test_phase13d_knee_densification.py",
    }
)
PHASE14_ALLOWED_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase14/analysis-closure-r2-publication.json",
        "docs/evidence/phase14/analysis-closure.json",
        "docs/evidence/phase14/r2-publication.json",
        "docs/phase_reports/phase14-analysis-closure.md",
        "docs/phase_reports/phase14-graph-ab.md",
        "docs/plans/phase14-graph-ab-execution-order.json",
        "docs/plans/phase14-graph-ab.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase14_analysis_closure.py",
        "scripts/phase14_graph_ab.py",
        "scripts/r2_artifact.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase14_analysis_closure.py",
        "tests/unit/test_phase14_graph_ab.py",
        "tests/unit/test_phase14_scope.py",
        "tests/unit/test_r2_artifact.py",
    }
)
PHASE15_ALLOWED_PATHS = frozenset(
    {
        "docs/evidence/phase15/ncu-metric-map.json",
        "docs/evidence/phase15/r2-publication-attempt0.json",
        "docs/evidence/phase15/r2-publication-attempt1.json",
        "docs/evidence/phase15/r2-publication-attempt2.json",
        "docs/evidence/phase15/r2-publication-attempt3.json",
        "docs/evidence/phase15/r2-publication.json",
        "docs/phase_reports/phase15-profiler-subset.md",
        "docs/plans/phase15-profiler-selection.json",
        "docs/plans/phase15-profiler-subset.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase15_profiler_subset.py",
        "scripts/r2_artifact.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase15_profiler_subset.py",
        "tests/unit/test_r2_artifact.py",
    }
)
PHASE16G_ALLOWED_PATHS = frozenset(
    {
        "docs/decisions/0039-full-scan-batch-geometry-admission.md",
        "docs/evidence/phase16g/batch-geometry-admission.json",
        "docs/evidence/phase16g/r2-publication.json",
        "docs/phase_reports/phase16g-batch-geometry-admission.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase13_prefix_state.py",
        "scripts/phase12_unified_admission.py",
        "scripts/phase13_pilot.py",
        "scripts/phase16g_batch_geometry_admission.py",
        "scripts/validate_phase2.py",
        "src/kvbench/runtime/kivi_admission.py",
        "src/kvbench/runtime/kivi_cache.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_session.py",
        "src/kvbench/runtime/turboquant_cache.py",
        "src/kvbench/schema/phase16g.py",
        "tests/cuda/phase16g_batch_sanitizer_probe.py",
        "tests/unit/test_phase16g_batch_geometry.py",
        "tests/unit/test_phase13b_batch_geometry.py",
    }
)


RAW_RESULT_SUFFIXES = {
    ".bin",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".gguf",
    ".ncu-rep",
    ".nsys-rep",
    ".parquet",
    ".safetensors",
    ".pt",
    ".pth",
}
PINNED_TURBOQUANT_SOURCE_PATHS = frozenset(
    {
        "src/kvbench/third_party/vllm_turboquant/centroids.py",
        "src/kvbench/third_party/vllm_turboquant/config.py",
        "src/kvbench/third_party/vllm_turboquant/triton_decode_attention.py",
        "src/kvbench/third_party/vllm_turboquant/triton_turboquant_decode.py",
        "src/kvbench/third_party/vllm_turboquant/triton_turboquant_store.py",
    }
)
PINNED_TURBOQUANT_ANNOTATION_EXEMPTIONS = (
    PINNED_TURBOQUANT_SOURCE_PATHS
    | frozenset(
        {"src/kvbench/third_party/vllm_turboquant/__init__.py"}
    )
)

BANNED_IMPORTS = {
    "datasets",
    "evaluate",
    "lm_eval",
    "longbench",
    "ruler",
    "triton",
    "vllm",
}
PHASE3_EXTERNAL_IMPORTS = {
    "src/kvbench/runtime/allocation.py": {"torch"},
    "src/kvbench/runtime/backend.py": {"torch"},
    "src/kvbench/runtime/bf16_endpoint.py": {"torch"},
    "src/kvbench/runtime/cuda_graph.py": {"torch"},
    "src/kvbench/runtime/fixed_l_runner.py": {"torch"},
    "src/kvbench/runtime/gqa_audit.py": {"torch"},
    "src/kvbench/runtime/growing_context_runner.py": {"torch"},
    "src/kvbench/runtime/model_loader.py": {"torch", "transformers"},
    "src/kvbench/runtime/numerical.py": {"torch"},
    "src/kvbench/runtime/static_cache.py": {"torch"},
    "src/kvbench/runtime/turboquant_admission.py": {"torch", "triton"},
    "src/kvbench/runtime/timing.py": {"torch"},
    "src/kvbench/runtime/kivi_admission.py": {"torch"},
    "src/kvbench/runtime/kivi_cache.py": {"torch"},
    "src/kvbench/runtime/kivi_fixture.py": {"torch"},
    "src/kvbench/runtime/kivi_session.py": {"torch"},
    "src/kvbench/adapters/kivi.py": {"torch"},
    "src/kvbench/third_party/vllm_turboquant/centroids.py": {"torch"},
    "src/kvbench/third_party/vllm_turboquant/compat.py": {"torch"},
    "src/kvbench/third_party/vllm_turboquant/config.py": {"vllm"},
    "src/kvbench/third_party/vllm_turboquant/triton_decode_attention.py": {
        "triton"
    },
    "src/kvbench/third_party/vllm_turboquant/triton_turboquant_decode.py": {
        "torch",
        "triton",
    },
    "src/kvbench/third_party/vllm_turboquant/triton_turboquant_store.py": {
        "torch",
        "triton",
    },
}
HOT_PATH_FUNCTIONS = {
    "src/kvbench/adapters/bf16.py": {
        "store_prefill",
        "append_decode",
        "decode_attention",
    },
    "src/kvbench/adapters/turboquant.py": {
        "_store_compressed",
        "store_prefill",
        "append_decode",
        "_decode_compressed",
        "decode_attention",
    },
    "src/kvbench/adapters/kivi.py": {
        "_commit_token",
        "_layer_context",
        "_quantize_into",
        "_store_historical_k",
        "_store_historical_v",
        "store_prefill",
        "append_decode",
        "_decode_compressed",
        "decode_attention",
        "launch_into",
    },
    "src/kvbench/runtime/backend.py": {
        "flash_attention_forward",
    },
    "src/kvbench/runtime/static_cache.py": {"update"},
    "src/kvbench/runtime/kivi_cache.py": {"update"},
    "src/kvbench/runtime/bf16_endpoint.py": {
        "rotate_half_in_place",
        "_attention",
        "_base_forward",
        "decode",
    },
    "src/kvbench/runtime/cuda_graph.py": {"replay"},
    "src/kvbench/runtime/phase3_endpoint_audit.py": {
        "fixed_step",
        "growing_step",
    },
    "src/kvbench/runtime/growing_context_runner.py": {
        "measured_step",
    },
}
PHASE8_HOT_PATH_SOURCES = frozenset(
    {
        "src/kvbench/adapters/kivi.py",
        "src/kvbench/runtime/kivi_cache.py",
    }
)
HOT_PATH_BANNED_CALLS = {
    "cat",
    "cpu",
    "expand",
    "item",
    "numpy",
    "repeat_interleave",
    "repeat_kv",
    "synchronize",
    "tolist",
}
E00_PROTECTED_PATHS = (
    "preflight/README.md",
    "preflight/__init__.py",
    "preflight/audit_checkpoint.py",
    "preflight/e00_cuda/binding.cpp",
    "preflight/e00_cuda/build.py",
    "preflight/e00_cuda/xor_kernel.cu",
    "preflight/e00_cuda/xor_kernel.h",
    "preflight/e00_manifest.schema.json",
    "preflight/process_query.py",
    "preflight/python_integrity_probe.py",
    "preflight/python_probe.py",
    "preflight/requirements-e00.txt",
    "preflight/run_preflight.py",
    "preflight/system-packages.lock.json",
    "scripts/preflight.sh",
    "tests/allocation/test_e00_allocation.py",
    "tests/cuda/e00_runtime_probe.py",
    "tests/cuda/e00_sanitizer_probe.py",
    "tests/golden/test_e00_numerical.py",
    "tests/graph_capture/test_e00_graph.py",
    "tests/unit/test_preflight_unit.py",
)
METHOD_NAMES = {"bf16", "turboquant", "kivi", "kvquant"}
SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REQUIREMENT = re.compile(
    r"([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([0-9a-f]{64})"
)


class ValidationFailure(RuntimeError):
    """A check could not be completed safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def run(
    argv: Sequence[str],
    *,
    cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=environment,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    return run(("/usr/bin/git", *argv), environment=environment)


def git_output(argv: Sequence[str]) -> str:
    result = git(argv)
    if result.returncode != 0:
        raise ValidationFailure(
            f"git command failed with exit status {result.returncode}: "
            + " ".join(argv)
        )
    return result.stdout


def git_paths(argv: Sequence[str]) -> set[str]:
    return {item for item in git_output(argv).split("\0") if item}


def report(name: str, errors: Sequence[str], *, note: str | None = None) -> int:
    if errors:
        print(f"[phase2:{name}] FAIL ({len(errors)} error(s))", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    suffix = f"; {note}" if note else ""
    print(f"[phase2:{name}] PASS{suffix}")
    return 0


def historical_phase2_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE2_ENTRY_COMMIT,
            PHASE2_FINAL_COMMIT,
            "--",
        )
    )


def current_phase3_paths() -> set[str]:
    changed = git_paths(
        ("diff", "--name-only", "-z", PHASE2_FINAL_COMMIT, "--")
    )
    untracked = git_paths(
        ("ls-files", "--others", "--exclude-standard", "-z", "--")
    )
    return changed | untracked

def historical_phase3_paths() -> set[str]:
    return git_paths(
        (
            "diff", "--name-only", "-z", PHASE2_FINAL_COMMIT, PHASE4_ENTRY_COMMIT, "--"
        )
    )


def historical_phase4_paths() -> set[str]:
    return git_paths(
        (
            "diff", "--name-only", "-z", PHASE4_ENTRY_COMMIT, PHASE5_ENTRY_COMMIT, "--"
        )
    )


def historical_phase5_paths() -> set[str]:
    return git_paths(
        (
            "diff", "--name-only", "-z", PHASE5_ENTRY_COMMIT, PHASE6A_ENTRY_COMMIT, "--"
        )
    )


def historical_phase6a_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE6A_ENTRY_COMMIT,
            PHASE6_ENTRY_COMMIT,
            "--",
        )
    )


def historical_phase6_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE6_ENTRY_COMMIT,
            PHASE7_ENTRY_COMMIT,
            "--",
        )
    )


def historical_phase7_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE7_ENTRY_COMMIT,
            PHASE8_ENTRY_COMMIT,
            "--",
        )
    )


def historical_phase8_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE8_ENTRY_COMMIT,
            PHASE9P_ENTRY_COMMIT,
            "--",
        )
    )


def historical_phase9p_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE9P_ENTRY_COMMIT,
            PHASE9P_FINAL_COMMIT,
            "--",
        )
    )


def current_kvquant_patch_custody_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE9P_FINAL_COMMIT,
            PHASE9_ENTRY_COMMIT,
            "--",
        )
    )


def historical_phase9_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE9_ENTRY_COMMIT,
            PHASE10_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase10_paths() -> set[str]:
    """Return the frozen Phase 10 plus Phase 11P decision segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE10_ENTRY_COMMIT,
            PHASE11PR_ENTRY_COMMIT,
            "--",
        )
    )


def historical_phase11pr_paths() -> set[str]:
    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE11PR_ENTRY_COMMIT,
            PHASE11_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase11pr_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 11P-R segment."""

    return historical_phase11pr_paths()


def historical_phase11_paths() -> set[str]:
    """Return the completed Phase 11 Adapter implementation segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE11_ENTRY_COMMIT,
            PHASE11D_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase11_paths() -> set[str]:
    """Compatibility view of the completed Phase 11 Adapter segment."""

    return historical_phase11_paths()


def historical_phase11d_paths() -> set[str]:
    """Return the completed deterministic long-context CUDA remediation."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE11D_ENTRY_COMMIT,
            PHASE11R_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase11d_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 11D segment."""

    return historical_phase11d_paths()


def historical_phase11r_paths() -> set[str]:
    """Return the completed Phase 11R admission-rerun segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE11R_ENTRY_COMMIT,
            PHASE12E_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase11r_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 11R segment."""

    return historical_phase11r_paths()


def historical_phase12e_paths() -> set[str]:
    """Return the completed Phase 12E validator-only remediation."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE12E_ENTRY_COMMIT,
            PHASE11DQ23_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase12e_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 12E segment."""

    return historical_phase12e_paths()


def historical_phase11dq23_paths() -> set[str]:
    """Return the completed deterministic q3/q2 CUDA remediation."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE11DQ23_ENTRY_COMMIT,
            PHASE11RQ23_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase11dq23_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 11D-Q23 segment."""

    return historical_phase11dq23_paths()


def historical_phase11rq23_paths() -> set[str]:
    """Return the completed current-source KVQuant re-admission segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE11RQ23_ENTRY_COMMIT,
            PHASE12_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase11rq23_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 11R-Q23 segment."""

    return historical_phase11rq23_paths()


def historical_phase12_paths() -> set[str]:
    """Return the completed, frozen Phase 12R segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE12_ENTRY_COMMIT,
            PHASE13_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase12_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 12R segment."""

    return historical_phase12_paths()


def current_phase13_paths() -> set[str]:
    """Compatibility view of the completed, stopped Phase 13 segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13_ENTRY_COMMIT,
            PHASE13B_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13b_paths() -> set[str]:
    """Compatibility view of the completed Phase 13B segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13B_ENTRY_COMMIT,
            PHASE13R_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13r_paths() -> set[str]:
    """Compatibility view of the completed, stopped Phase 13R segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13R_ENTRY_COMMIT,
            PHASE13F_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13f_paths() -> set[str]:
    """Compatibility view of the completed Phase 13F segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13F_ENTRY_COMMIT,
            PHASE13R2_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13r2_paths() -> set[str]:
    """Compatibility view of the completed, stopped Phase 13R2 segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13R2_ENTRY_COMMIT,
            PHASE13T_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13t_paths() -> set[str]:
    """Compatibility view of the completed Phase 13T-R segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13T_ENTRY_COMMIT,
            PHASE13PR_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13pr_paths() -> set[str]:
    """Compatibility view of the completed Phase 13P-R/P-D segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13PR_ENTRY_COMMIT,
            PHASE13PB_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13pb_paths() -> set[str]:
    """Compatibility view of the completed Phase 13P-B segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13PB_ENTRY_COMMIT,
            PHASE13PC_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13pc_paths() -> set[str]:
    """Compatibility view of the completed Phase 13P-C segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13PC_ENTRY_COMMIT,
            PHASE13RQ4_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13rq4_paths() -> set[str]:
    """Return the frozen q4 remediation and successor-Pilot segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13RQ4_ENTRY_COMMIT,
            PHASE13D_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase13d_paths() -> set[str]:
    """Compatibility view of the completed Phase 13D segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE13D_ENTRY_COMMIT,
            PHASE14_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase14_paths() -> set[str]:
    """Compatibility view of the completed Phase 14 segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE14_ENTRY_COMMIT,
            PHASE15_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase15_paths() -> set[str]:
    """Compatibility view of the completed Phase 15 segment."""

    return git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE15_ENTRY_COMMIT,
            PHASE16G_ENTRY_COMMIT,
            "--",
        )
    )


def current_phase16g_paths() -> set[str]:
    """Return tracked and untracked Phase 16G geometry changes."""

    changed = git_paths(
        ("diff", "--name-only", "-z", PHASE16G_ENTRY_COMMIT, "--")
    )
    untracked = git_paths(
        ("ls-files", "--others", "--exclude-standard", "-z", "--")
    )
    return changed | untracked


def phase12_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 12R allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE12_ALLOWED_PATHS
    )


def phase13_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13 allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13_ALLOWED_PATHS
    )


def phase13b_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13B allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13B_ALLOWED_PATHS
    )


def phase13r_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13R allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13R_ALLOWED_PATHS
    )


def phase13f_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13F allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13F_ALLOWED_PATHS
    )


def phase13r2_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13R2 allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13R2_ALLOWED_PATHS
    )


def phase13t_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13T allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13T_ALLOWED_PATHS
    )


def phase13pr_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13P-R allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13PR_ALLOWED_PATHS
    )


def phase13pb_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13P-B allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13PB_ALLOWED_PATHS
    )


def phase13pc_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13P-C allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13PC_ALLOWED_PATHS
    )


def phase13rq4_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13R q4 allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13RQ4_ALLOWED_PATHS
    )


def phase13d_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 13D allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE13D_ALLOWED_PATHS
    )


def phase14_path_is_allowed(relative: str) -> bool:
    """Accept only canonical exact paths in the Phase 14 allowlist."""

    try:
        candidate = PurePosixPath(relative)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(relative, str)
        and relative == candidate.as_posix()
        and relative not in {"", "."}
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\\" not in relative
        and relative in PHASE14_ALLOWED_PATHS
    )


def current_phase9_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 9 segment."""

    return historical_phase9_paths()


def current_phase9p_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 9P segment."""

    return historical_phase9p_paths()


def current_phase8_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 8 segment."""

    return historical_phase8_paths()


def current_phase7_paths() -> set[str]:
    """Compatibility view of the completed, frozen Phase 7 segment."""

    return historical_phase7_paths()


def current_phase6_paths() -> set[str]:
    """Compatibility view spanning completed Phase 6 and Phase 7."""

    return historical_phase6_paths() | historical_phase7_paths()


def current_phase6a_paths() -> set[str]:
    """Compatibility alias for callers predating the Phase 6 boundary."""

    return historical_phase6a_paths()


def current_phase5_paths() -> set[str]:
    """Compatibility alias for callers predating the Phase 6A boundary."""

    return (
        historical_phase5_paths()
        | historical_phase6a_paths()
        | current_phase6_paths()
    )


def changed_paths() -> set[str]:
    return current_phase12_paths()


def repository_python_paths() -> list[Path]:
    paths: set[Path] = set()
    if SRC.is_dir():
        paths.update(SRC.rglob("*.py"))
    paths.add(ROOT / "scripts" / "validate_phase2.py")
    paths.add(ROOT / "scripts" / "r2_artifact.py")
    paths.add(ROOT / "scripts" / "phase6_turboquant_admission.py")
    paths.add(ROOT / "scripts" / "phase6a_bf16_parity.py")
    paths.add(ROOT / "scripts" / "phase8_kivi_admission.py")
    paths.add(ROOT / "scripts" / "phase8_r2_outer_bundle.py")
    paths.add(ROOT / "scripts" / "validate_kvquant_gqa_patch.py")
    paths.add(ROOT / "scripts" / "validate_kvquant_graphsafe_patch.py")
    paths.add(ROOT / "scripts" / "phase11d_kvquant_validation.py")
    paths.add(ROOT / "scripts" / "validate_kvquant_long_context_patch.py")
    paths.add(ROOT / "scripts" / "phase11dq23_kvquant_validation.py")
    paths.add(
        ROOT / "scripts" / "validate_kvquant_q23_long_context_patch.py"
    )
    paths.add(ROOT / "scripts" / "phase9_kvquant_calibration.py")
    paths.add(ROOT / "scripts" / "phase9_kvquant_worker.py")
    paths.add(ROOT / "scripts" / "phase14_graph_ab.py")
    phase10_reference = ROOT / "reference" / "kvquant"
    if phase10_reference.is_dir():
        paths.update(phase10_reference.glob("*.py"))
    phase11pr_reference = ROOT / "reference" / "kvquant_phase11pr"
    if phase11pr_reference.is_dir():
        paths.update(phase11pr_reference.glob("*.py"))
    schema_tests = ROOT / "tests" / "schema"
    if schema_tests.is_dir():
        paths.update(schema_tests.rglob("*.py"))
    unit_tests = ROOT / "tests" / "unit"
    if unit_tests.is_dir():
        paths.update(unit_tests.glob("test_phase2_*.py"))
        paths.update(unit_tests.glob("test_phase3_*.py"))
        paths.update(unit_tests.glob("test_phase4_*.py"))
        paths.update(unit_tests.glob("test_phase5_*.py"))
        paths.update(unit_tests.glob("test_phase6a_*.py"))
        paths.update(unit_tests.glob("test_phase6_*.py"))
        paths.update(unit_tests.glob("test_phase7_*.py"))
        paths.update(unit_tests.glob("test_phase8_*.py"))
        paths.update(unit_tests.glob("test_phase9p_*.py"))
        paths.update(unit_tests.glob("test_phase9_*.py"))
        paths.update(unit_tests.glob("test_phase10_*.py"))
        paths.update(unit_tests.glob("test_phase11pr_*.py"))
        paths.update(unit_tests.glob("test_phase11_*.py"))
        paths.add(unit_tests / "test_phase11d_scope.py")
        paths.add(unit_tests / "test_phase11r_scope.py")
        paths.add(unit_tests / "test_phase14_graph_ab.py")
        paths.add(unit_tests / "test_phase14_scope.py")
        paths.update(unit_tests.glob("test_phase12e_*.py"))
        paths.update(
            unit_tests / name
            for name in (
                "test_allocation_attribution.py",
                "test_gqa_device_dispatch.py",
                "test_gqa_taxonomy.py",
                "test_measurement_container.py",
                "test_preflight_unit.py",
                "test_r2_artifact.py",
                "test_process_supervision.py",
            )
        )
    cuda_tests = ROOT / "tests" / "cuda"
    if cuda_tests.is_dir():
        paths.update(cuda_tests.glob("test_phase3_*.py"))
        paths.update(cuda_tests.glob("test_phase6_*.py"))
        paths.update(cuda_tests.glob("test_phase8_*.py"))
        paths.update(cuda_tests.glob("test_phase11_*.py"))
        sanitizer_probe = cuda_tests / "phase6_turboquant_sanitizer_probe.py"
        if sanitizer_probe.is_file():
            paths.add(sanitizer_probe)
        phase8_sanitizer_probe = cuda_tests / "phase8_kivi_sanitizer_probe.py"
        if phase8_sanitizer_probe.is_file():
            paths.add(phase8_sanitizer_probe)
        phase10_sanitizer_probe = (
            cuda_tests / "phase10_kvquant_sanitizer_probe.py"
        )
        if phase10_sanitizer_probe.is_file():
            paths.add(phase10_sanitizer_probe)
        phase11pr_validation = (
            cuda_tests / "phase11pr_kvq3_pack_validation.py"
        )
        if phase11pr_validation.is_file():
            paths.add(phase11pr_validation)
        phase11_sanitizer_probe = (
            cuda_tests / "phase11_kvquant_sanitizer_probe.py"
        )
        if phase11_sanitizer_probe.is_file():
            paths.add(phase11_sanitizer_probe)
        phase11d_validation = (
            cuda_tests / "phase11d_kvquant_long_context_validation.py"
        )
        if phase11d_validation.is_file():
            paths.add(phase11d_validation)
    graph_tests = ROOT / "tests" / "graph"
    if graph_tests.is_dir():
        paths.update(graph_tests.glob("test_phase3_*.py"))
        paths.update(graph_tests.glob("test_phase6_*.py"))
        paths.update(graph_tests.glob("test_phase8_*.py"))
        paths.update(graph_tests.glob("test_phase11_*.py"))
    return sorted(path for path in paths if path.is_file())


def check_format() -> int:
    errors: list[str] = []
    text_suffixes = {".md", ".py", ".toml", ".yaml", ".json", ".txt"}
    candidates = {
        ROOT / relative
        for relative in (
            changed_paths() | current_phase13_paths() | current_phase14_paths()
        )
        if Path(relative).suffix in text_suffixes
        or relative in {".gitignore", "Makefile"}
    }
    candidates.update(repository_python_paths())
    for path in sorted(candidates):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 Phase 2 text file: {relative}")
            continue
        if b"\r" in data:
            errors.append(f"CR/CRLF line ending is not canonical: {relative}")
        if data and not data.endswith(b"\n"):
            errors.append(f"missing terminal newline: {relative}")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                errors.append(f"trailing whitespace: {relative}:{line_number}")
            if relative.endswith(".py") and "\t" in line:
                errors.append(f"tab indentation in Python: {relative}:{line_number}")
    return report(
        "format",
        errors,
        note="UTF-8/newline/whitespace check (no formatter dependency)",
    )


def check_lint() -> int:
    errors: list[str] = []
    paths = repository_python_paths()
    if not any(path.is_relative_to(SRC) for path in paths):
        errors.append("src/kvbench Python package is missing")
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
            compile(tree, relative, "exec", dont_inherit=True)
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"AST compilation failed: {relative}: {error}")
            continue
        defines_typed_code = any(
            isinstance(node, (ast.AnnAssign, ast.ClassDef, ast.FunctionDef))
            for node in tree.body
        )
        if (
            defines_typed_code
            and relative not in PINNED_TURBOQUANT_SOURCE_PATHS
            and (
                path.is_relative_to(SRC)
                or path.name == "validate_phase2.py"
            )
        ):
            has_future_annotations = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in tree.body
            )
            if not has_future_annotations:
                errors.append(f"missing future annotations import: {relative}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                errors.append(f"wildcard import: {relative}:{node.lineno}")
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                errors.append(f"bare except: {relative}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = [*node.args.defaults, *node.args.kw_defaults]
                if any(
                    isinstance(default, (ast.Dict, ast.List, ast.Set))
                    for default in defaults
                ):
                    errors.append(
                        f"mutable function default: {relative}:{node.lineno}"
                    )
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            imported: set[str] = set()
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])
            allowed_external = PHASE3_EXTERNAL_IMPORTS.get(relative, set())
            banned = (imported & BANNED_IMPORTS) - allowed_external
            if banned:
                errors.append(
                    f"out-of-scope import {sorted(banned)!r}: "
                    f"{relative}:{node.lineno}"
                )
            external = {
                name
                for name in imported
                if name not in sys.stdlib_module_names and name != "kvbench"
            }
            undeclared = external - allowed_external
            if undeclared and (
                path.is_relative_to(SRC) or path.name == "validate_phase2.py"
            ):
                errors.append(
                    f"undeclared non-stdlib import {sorted(undeclared)!r}: "
                    f"{relative}:{node.lineno}"
                )
    return report(
        "lint",
        errors,
        note="AST compilation and dependency/scope lint",
    )


def _call_leaf(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _contains_forbidden_expand_reshape(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if _call_leaf(node) not in {"reshape", "view"}:
        return False
    function = node.func
    if not isinstance(function, ast.Attribute):
        return False
    return any(
        isinstance(inner, ast.Call) and _call_leaf(inner) == "expand"
        for inner in ast.walk(function.value)
    )


def _function_definitions(tree: ast.AST) -> dict[str, list[ast.AST]]:
    definitions: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)
    return definitions


def _hot_node_errors(
    relative: str,
    function_name: str,
    node: ast.AST,
) -> list[str]:
    errors: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            leaf = _call_leaf(child)
            if leaf in HOT_PATH_BANNED_CALLS:
                errors.append(
                    f"forbidden hot-path call {leaf}: "
                    f"{relative}:{getattr(child, 'lineno', 0)}:"
                    f"{function_name}"
                )
            if leaf in {
                "run",
                "Popen",
                "collect_telemetry",
                "print",
                "debug",
                "info",
                "warning",
                "error",
                "critical",
            }:
                errors.append(
                    f"logging, telemetry, or subprocess call in hot path: "
                    f"{relative}:{getattr(child, 'lineno', 0)}:"
                    f"{function_name}"
                )
            if _contains_forbidden_expand_reshape(child):
                errors.append(
                    f"expand-plus-reshape in hot path: "
                    f"{relative}:{getattr(child, 'lineno', 0)}:"
                    f"{function_name}"
                )
        if isinstance(child, ast.Name) and child.id == "DynamicCache":
            errors.append(
                f"DynamicCache in hot path: {relative}:"
                f"{getattr(child, 'lineno', 0)}:{function_name}"
            )
    return errors


def check_hot_path() -> int:
    errors: list[str] = []
    trees: dict[str, ast.AST] = {}
    for relative, expected_names in HOT_PATH_FUNCTIONS.items():
        path = ROOT / relative
        if not path.is_file():
            if relative in PHASE8_HOT_PATH_SOURCES:
                continue
            errors.append(f"missing Phase 3 SUT source: {relative}")
            continue
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=relative,
            )
        except (SyntaxError, UnicodeError) as error:
            errors.append(
                f"cannot audit Phase 3 SUT source {relative}: {error}"
            )
            continue
        trees[relative] = tree
        definitions = _function_definitions(tree)
        for name in sorted(expected_names):
            matches = definitions.get(name, [])
            if len(matches) != 1:
                errors.append(
                    f"expected exactly one audited function "
                    f"{relative}:{name}, found {len(matches)}"
                )
                continue
            errors.extend(
                _hot_node_errors(relative, name, matches[0])
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_leaf(node) in {
                "cat",
                "repeat_interleave",
                "repeat_kv",
            }:
                errors.append(
                    f"forbidden GQA/cache operation in SUT source: "
                    f"{relative}:{getattr(node, 'lineno', 0)}"
                )
            if isinstance(node, ast.Name) and node.id == "DynamicCache":
                errors.append(
                    f"DynamicCache in SUT source: {relative}:"
                    f"{getattr(node, 'lineno', 0)}"
                )
            if _contains_forbidden_expand_reshape(node):
                errors.append(
                    f"expand-plus-reshape in SUT source: {relative}:"
                    f"{getattr(node, 'lineno', 0)}"
                )
    timing_path = ROOT / "src/kvbench/runtime/timing.py"
    if not timing_path.is_file():
        errors.append("missing Phase 3 timing source")
    else:
        try:
            timing_tree = ast.parse(
                timing_path.read_text(encoding="utf-8"),
                filename="src/kvbench/runtime/timing.py",
            )
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"cannot audit Phase 3 timing source: {error}")
        else:
            definitions = _function_definitions(timing_tree)
            loop_specs = {
                "measure_fixed_batches": "count",
                "measure_growing_trajectory": "steps",
            }
            for name, range_name in loop_specs.items():
                matches = definitions.get(name, [])
                if len(matches) != 1:
                    errors.append(
                        f"expected exactly one timing function: {name}"
                    )
                    continue
                function = matches[0]
                synchronize_calls = sum(
                    1
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and _call_leaf(node) == "synchronize"
                )
                if synchronize_calls != 2:
                    errors.append(
                        f"{name} must contain exactly start/end "
                        f"synchronization boundaries"
                    )
                operation_loops = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Call)
                    and _call_leaf(node.iter) == "range"
                    and len(node.iter.args) == 1
                    and isinstance(node.iter.args[0], ast.Name)
                    and node.iter.args[0].id == range_name
                ]
                if len(operation_loops) != 1:
                    errors.append(
                        f"{name} must retain one exact measured "
                        f"operation loop"
                    )
                    continue
                loop_wrapper = ast.Module(
                    body=operation_loops[0].body,
                    type_ignores=[],
                )
                errors.extend(
                    _hot_node_errors(
                        "src/kvbench/runtime/timing.py",
                        f"{name}:measured_loop",
                        loop_wrapper,
                    )
                )
    return report(
        "hot-path",
        errors,
        note="measured decode and GQA/cache AST audit",
    )


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def callable_annotation_errors(
    qualified_name: str,
    value: Callable[..., Any],
) -> list[str]:
    errors: list[str] = []
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as error:
        return [f"cannot inspect {qualified_name}: {error}"]
    for parameter in signature.parameters.values():
        if (
            parameter.name not in {"self", "cls"}
            and parameter.annotation is inspect.Signature.empty
        ):
            errors.append(
                f"missing parameter annotation: "
                f"{qualified_name}.{parameter.name}"
            )
    if signature.return_annotation is inspect.Signature.empty:
        errors.append(f"missing return annotation: {qualified_name}")
    try:
        typing.get_type_hints(value, include_extras=True)
    except Exception as error:  # annotation resolution is the check itself
        errors.append(
            f"unresolvable annotations: {qualified_name}: "
            f"{type(error).__name__}"
        )
    return errors


def check_annotations() -> int:
    errors: list[str] = []
    if not SRC.is_dir():
        return report("annotations", ["src/kvbench package is missing"])
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    phase3_child = os.environ.get(
        "KVBENCH_PHASE3_ANNOTATION_CHILD"
    ) == "1"
    modules: list[Any] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in PINNED_TURBOQUANT_ANNOTATION_EXEMPTIONS:
            continue
        is_phase3_external = relative in PHASE3_EXTERNAL_IMPORTS
        if phase3_child != is_phase3_external:
            continue
        name = module_name(path)
        if not name:
            continue
        try:
            modules.append(importlib.import_module(name))
        except Exception as error:
            errors.append(
                f"module import failed: {name}: {type(error).__name__}"
            )
    for module in modules:
        try:
            typing.get_type_hints(module, include_extras=True)
        except Exception as error:
            errors.append(
                f"module annotations do not resolve: {module.__name__}: "
                f"{type(error).__name__}"
            )
        for name, value in vars(module).items():
            if inspect.isfunction(value) and value.__module__ == module.__name__:
                errors.extend(
                    callable_annotation_errors(
                        f"{module.__name__}.{name}", value
                    )
                )
            if not inspect.isclass(value) or value.__module__ != module.__name__:
                continue
            try:
                typing.get_type_hints(value, include_extras=True)
            except Exception as error:
                errors.append(
                    f"class annotations do not resolve: "
                    f"{module.__name__}.{name}: {type(error).__name__}"
                )
            for member_name, member in vars(value).items():
                if member_name.startswith("__") and member_name.endswith("__"):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    for accessor in (member.fget, member.fset, member.fdel):
                        if accessor is not None:
                            errors.extend(
                                callable_annotation_errors(
                                    f"{module.__name__}.{name}."
                                    f"{accessor.__name__}",
                                    accessor,
                                )
                            )
                    continue
                if (
                    inspect.isfunction(member)
                    and member.__module__ == module.__name__
                ):
                    errors.extend(
                        callable_annotation_errors(
                            f"{module.__name__}.{name}.{member_name}",
                            member,
                        )
                    )
    return report(
        "annotations",
        errors,
        note=(
            "runtime annotation resolution, not third-party static "
            "type analysis"
        ),
    )


def check_phase3_annotations() -> int:
    errors: list[str] = []
    python = ROOT / ".venv" / "bin" / "python"
    site = ROOT / ".phase3" / "site-packages"
    if not python.is_file():
        errors.append("certified Phase 3 base interpreter is missing")
    if not site.is_dir():
        errors.append("isolated Phase 3 dependency target is missing")
    if errors:
        return report("phase3-annotations", errors)
    environment = {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "KVBENCH_PHASE3_ANNOTATION_CHILD": "1",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{site}:{SRC}",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = run(
        (str(python), str(Path(__file__).resolve()), "annotations"),
        environment=environment,
    )
    if result.returncode != 0:
        errors.append(
            "Phase 3 runtime annotation resolution failed"
        )
        detail = (result.stderr or result.stdout).strip()
        if detail:
            errors.append(detail.replace("\n", " | "))
    return report(
        "phase3-annotations",
        errors,
        note="isolated runtime annotation resolution",
    )


def phase2_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def cli_config_error(relative: str) -> str | None:
    result = run(
        (
            "/usr/bin/python3",
            "-m",
            "kvbench",
            "validate-config",
            relative,
        ),
        environment=phase2_environment(),
    )
    if result.returncode != 0:
        return (
            f"CLI validation failed for {relative} "
            f"(exit {result.returncode})"
        )
    return None


def check_configs() -> int:
    errors: list[str] = []
    for relative in CONFIG_PATHS:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing example configuration: {relative}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            errors.append(
                "configuration is not the declared JSON-compatible "
                f"YAML subset: {relative}"
            )
            continue
        error = cli_config_error(relative)
        if error:
            errors.append(error)
    return report("configs", errors)


def parse_ledger(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return {}, [
            f"cannot read checksum ledger {path}: {type(error).__name__}"
        ]
    for line_number, line in enumerate(lines, start=1):
        if "  " not in line:
            errors.append(
                f"malformed checksum ledger line {line_number}: {path}"
            )
            continue
        digest, relative = line.split("  ", 1)
        pure = PurePosixPath(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(
                f"malformed checksum digest at line {line_number}: {path}"
            )
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in entries
        ):
            errors.append(
                f"unsafe or duplicate ledger path at line "
                f"{line_number}: {path}"
            )
            continue
        entries[relative] = digest
    return entries, errors


def validate_e00_run(
    run_id: str,
    expected: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    run_root = ROOT / "docs" / "evidence" / "e00" / run_id
    manifest_path = run_root / "manifest.json"
    ledger_path = run_root / "checksums.sha256"
    complete_path = run_root / "COMPLETE"
    for required in (manifest_path, ledger_path, complete_path):
        if not required.is_file() or required.is_symlink():
            errors.append(
                f"missing or unsafe E00 file: {required.relative_to(ROOT)}"
            )
    if errors:
        return errors
    if sha256(manifest_path) != expected["manifest_sha256"]:
        errors.append(f"immutable E00 manifest hash mismatch: {run_id}")
    if sha256(ledger_path) != expected["ledger_sha256"]:
        errors.append(f"immutable E00 ledger hash mismatch: {run_id}")
    entries, ledger_errors = parse_ledger(ledger_path)
    errors.extend(ledger_errors)
    actual_payloads: set[str] = set()
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            errors.append(
                f"symlink found in E00 evidence: {path.relative_to(ROOT)}"
            )
            continue
        if path.is_file():
            relative = path.relative_to(run_root).as_posix()
            if relative not in {"checksums.sha256", "COMPLETE"}:
                actual_payloads.add(relative)
            if path.stat().st_mode & write_bits:
                errors.append(
                    f"writable E00 evidence file: {path.relative_to(ROOT)}"
                )
        elif path.is_dir() and path.stat().st_mode & write_bits:
            errors.append(
                f"writable E00 evidence directory: {path.relative_to(ROOT)}"
            )
    if run_root.stat().st_mode & write_bits:
        errors.append(
            f"writable E00 run directory: {run_root.relative_to(ROOT)}"
        )
    if set(entries) != actual_payloads:
        errors.append(f"checksum ledger coverage mismatch: {run_id}")
    for relative, expected_hash in entries.items():
        target = run_root / relative
        if not target.is_file() or target.is_symlink():
            errors.append(
                f"checksum target missing or unsafe: {run_id}/{relative}"
            )
        elif sha256(target) != expected_hash:
            errors.append(f"checksum mismatch: {run_id}/{relative}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        errors.append(
            f"invalid E00 JSON: {run_id}: {type(error).__name__}"
        )
        return errors
    if manifest.get("run", {}).get("id") != run_id:
        errors.append(f"manifest run ID mismatch: {run_id}")
    if manifest.get("run", {}).get("status") != expected["status"]:
        errors.append(f"manifest status mismatch: {run_id}")
    if manifest.get("run", {}).get("benchmark_timing_collected") is not False:
        errors.append(f"E00 manifest does not deny benchmark timing: {run_id}")
    expected_complete = {
        "run_id": run_id,
        "status": expected["status"],
        "manifest_sha256": expected["manifest_sha256"],
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": expected["ledger_sha256"],
        "written_last": True,
    }
    for key, value in expected_complete.items():
        if complete.get(key) != value:
            errors.append(
                f"completion marker field mismatch: {run_id}/{key}"
            )
    return errors


def commit_is_ancestor(commit: str) -> bool:
    result = git(("merge-base", "--is-ancestor", commit, "HEAD"))
    return result.returncode == 0


def freeze_markers() -> list[Path]:
    markers: list[Path] = []
    for base, directories, files in os.walk(ROOT):
        base_path = Path(base)
        directories[:] = [
            item
            for item in directories
            if item not in {".git", ".phase3", ".venv", "__pycache__"}
        ]
        markers.extend(
            base_path / item
            for item in files
            if item == "PERFORMANCE_DATA_FROZEN"
        )
    return markers


def validate_performance_freeze_markers(markers: list[Path]) -> list[str]:
    """Keep performance-freeze completion separate from quality approval."""

    if not markers:
        return []
    expected_path = ROOT / "PERFORMANCE_DATA_FROZEN"
    if markers != [expected_path]:
        return ["unexpected PERFORMANCE_DATA_FROZEN marker location or count"]
    try:
        marker = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid PERFORMANCE_DATA_FROZEN marker: {error}"]
    errors: list[str] = []
    expected = {
        "schema_version": "kvbench-performance-data-frozen-marker-1.0.0",
        "status": "PASS",
        "performance_freeze_complete": True,
        "quality_contract_approved": False,
        "quality_execution": "LOCKED",
        "quality_evaluation_executed": False,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            errors.append(f"PERFORMANCE_DATA_FROZEN field mismatch: {key}")
    receipt_relative = marker.get("correction_receipt")
    if not isinstance(receipt_relative, str):
        errors.append("PERFORMANCE_DATA_FROZEN correction receipt is missing")
        return errors
    receipt_path = ROOT / receipt_relative
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid performance-freeze correction receipt: {error}")
        return errors
    binding = receipt.get("active_freeze_binding", {})
    for key in ("tag_name", "tag_object_id", "target_commit"):
        if marker.get(f"active_freeze_{key}") != binding.get(key):
            errors.append(f"performance-freeze correction binding mismatch: {key}")
    if marker.get("freeze_bundle_root_sha256") != receipt.get("freeze_bundle", {}).get("root_sha256"):
        errors.append("performance-freeze bundle root mismatch")
    if marker.get("manifest_sha256") != receipt.get("freeze_bundle", {}).get("manifest_sha256"):
        errors.append("performance-freeze manifest SHA mismatch")
    if receipt.get("quality", {}).get("contract_approved") is not False:
        errors.append("performance-freeze receipt incorrectly approves quality")
    tag_name = binding.get("tag_name")
    if isinstance(tag_name, str):
        tag_object = git(("rev-parse", tag_name))
        tag_target = git(("rev-parse", f"{tag_name}^{{}}"))
        if tag_object.returncode != 0 or tag_object.stdout.strip() != binding.get("tag_object_id"):
            errors.append("active performance-freeze tag object mismatch")
        if tag_target.returncode != 0 or tag_target.stdout.strip() != binding.get("target_commit"):
            errors.append("active performance-freeze tag target mismatch")
    else:
        errors.append("active performance-freeze tag is missing")
    return errors


def check_provenance() -> int:
    errors: list[str] = []
    head = git_output(("rev-parse", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        errors.append("current Git HEAD is not a full SHA-1 identity")
    for commit, label in (
        (QUALITY_COMMIT, "quality protocol"),
        (ENVIRONMENT_COMMIT, "environment lock"),
        (EVIDENCE_COMMIT, "successful E00 evidence"),
        (PLAN_COMMIT, "Phase 2 plan"),
    ):
        if not commit_is_ancestor(commit):
            errors.append(
                f"required {label} commit is not an ancestor of HEAD"
            )
    for relative, expected_hash in QUALITY_PROTOCOL_HASHES.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected_hash:
            errors.append(
                f"quality protocol byte lock mismatch: {relative}"
            )
    for run_id, expected in E00_RUNS.items():
        errors.extend(validate_e00_run(run_id, expected))
    errors.extend(validate_performance_freeze_markers(freeze_markers()))
    status_path = ROOT / "docs" / "status.md"
    if not status_path.is_file():
        errors.append("docs/status.md is missing")
    else:
        status_text = status_path.read_text(encoding="utf-8").lower()
        normalized_status = " ".join(status_text.split())
        if "quality execution: locked" not in status_text:
            errors.append(
                "status does not record quality execution as LOCKED"
            )
        if (
            "full scan remains closed" not in normalized_status
            and "full-scan admission: closed" not in normalized_status
        ):
            errors.append("status does not record the full scan as CLOSED")
    return report("provenance", errors)


def make_target_block(text: str, target: str) -> str | None:
    lines = text.splitlines()
    start: int | None = None
    target_pattern = re.compile(
        r"^[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)*\s*:"
    )
    for index, line in enumerate(lines):
        if line == f"{target}:" or line.startswith(f"{target}: "):
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if (
            line
            and not line[0].isspace()
            and target_pattern.match(line)
        ):
            end = index
            break
    while end > start and not lines[end - 1]:
        end -= 1
    return "\n".join(lines[start:end]) + "\n"


def validate_phase3_campaign_and_report_roots(artifacts: Path) -> list[str]:
    errors: list[str] = []
    campaigns = artifacts / "phase3_campaigns"
    reports = artifacts / "phase3_reports"
    if campaigns.exists() or campaigns.is_symlink():
        if campaigns.is_symlink() or not campaigns.is_dir():
            errors.append("Phase 3 campaign root is unsafe")
        else:
            from kvbench.runtime.phase3_campaign import (
                validate_phase3_campaign_directory,
            )

            for child in sorted(campaigns.iterdir()):
                if child.is_symlink() or not child.is_dir():
                    errors.append(
                        f"unsafe Phase 3 campaign child: {child.name}"
                    )
                    continue
                validation = validate_phase3_campaign_directory(child)
                if not validation.get("valid"):
                    errors.append(
                        f"invalid Phase 3 campaign: {child.name}"
                    )
    if reports.exists() or reports.is_symlink():
        if reports.is_symlink() or not reports.is_dir():
            errors.append("Phase 3 report root is unsafe")
        else:
            from kvbench.runtime.phase3_report import (
                validate_phase3_g1_report_directory,
            )
            from kvbench.runtime.phase3_report_publication import (
                validate_failed_report_attempt,
            )

            for child in sorted(reports.iterdir()):
                if child.name == ".kvbench-report-staging":
                    if child.is_symlink() or not child.is_dir():
                        errors.append("unsafe Phase 3 report staging root")
                    elif any(child.iterdir()):
                        errors.append("nonempty Phase 3 report staging root")
                    continue
                if child.name == ".kvbench-report-reservations":
                    if child.is_symlink() or not child.is_dir():
                        errors.append("unsafe Phase 3 report reservation root")
                        continue
                    for reservation in sorted(child.iterdir()):
                        metadata = reservation.lstat()
                        if (
                            stat.S_ISLNK(metadata.st_mode)
                            or not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_mode
                            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                        ):
                            errors.append(
                                f"unsafe Phase 3 report reservation: {reservation.name}"
                            )
                    continue
                if child.name == ".kvbench-report-failed":
                    if child.is_symlink() or not child.is_dir():
                        errors.append("unsafe Phase 3 failed-report root")
                        continue
                    for attempt in sorted(child.iterdir()):
                        if (
                            attempt.is_symlink()
                            or not attempt.is_dir()
                            or not validate_failed_report_attempt(attempt).get("valid")
                        ):
                            errors.append(
                                f"invalid Phase 3 failed report: {attempt.name}"
                            )
                    continue
                if child.is_symlink() or not child.is_dir():
                    errors.append(
                        f"unsafe Phase 3 report child: {child.name}"
                    )
                    continue
                validation = validate_phase3_g1_report_directory(child)
                if not validation.get("valid"):
                    errors.append(
                        f"invalid Phase 3 report: {child.name}"
                    )
    return errors


APPROVED_ARTIFACT_ROOT_NAMES = frozenset(
    {
        "README.md",
        "phase3",
        "phase3_campaigns",
        "phase3_reports",
        "phase4_smoke",
        "phase6",
        "phase6_r2_outer",
        "phase6a",
    }
)

PHASE7_APPROVED_ARTIFACT_ROOT_NAMES = frozenset(
    {"phase7_kivi_reference"}
)

PHASE8_APPROVED_ARTIFACT_ROOT_NAMES = frozenset(
    {"phase8", "phase8_r2_outer"}
)

PHASE11_APPROVED_ARTIFACT_ROOT_NAMES = frozenset(
    {"phase11", "phase11_r2_outer"}
)
PHASE3_BACKUP_ARTIFACT_ROOT_NAMES = frozenset(
    {
        "cloud_backup_catalog",
        "historical_failure_custody",
        "phase3_r2_custody",
        "phase3_r2_outer",
        "residual_evidence_custody",
    }
)
PHASE13_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13"})
PHASE13B_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13b"})
PHASE13F_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13f"})
PHASE13T_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13t"})
PHASE13PB_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13pb"})
PHASE13PC_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13pc"})
PHASE13RQ4_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13rq4"})
PHASE13D_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase13d"})
PHASE14_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase14"})
PHASE15_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase15"})
PHASE16G_APPROVED_ARTIFACT_ROOT_NAMES = frozenset({"phase16g"})
PHASE13_PREFIX_APPROVED_ARTIFACT_ROOT_NAMES = frozenset(
    {"phase13_prefix_catalogs"}
)
PHASE12_BLOCKED_ARTIFACT_ROOT_NAMES = frozenset({"phase12"})
PHASE12_STOPPED_CAMPAIGN_ID = (
    "phase12-20260730t000000000000z-2bc6aaa1-abcdef"
)
PHASE12_CAMPAIGN_ID_RE = re.compile(
    r"phase12-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}"
)
PHASE12_BLOCKED_STAGING_DIRECTORIES = frozenset(
    {
        ".kvbench-reservations",
        (
            ".kvbench-reservations/"
            f"{PHASE12_STOPPED_CAMPAIGN_ID}"
        ),
        ".kvbench-staging",
        (
            ".kvbench-staging/"
            f"{PHASE12_STOPPED_CAMPAIGN_ID}."
            "df41252db7c860d9755c4843.staging"
        ),
        (
            ".kvbench-staging/"
            f"{PHASE12_STOPPED_CAMPAIGN_ID}."
            "df41252db7c860d9755c4843.staging/unified"
        ),
    }
)
PHASE12_BLOCKED_STAGING_FILE_SHA256S = {
    (
        ".kvbench-staging/"
        f"{PHASE12_STOPPED_CAMPAIGN_ID}."
        "df41252db7c860d9755c4843.staging/campaign-reservation.json"
    ): "d123ba56f1341176dab82e8b4eba108117053c018f11e7ed2ec94a1f8a3e16a1",
    (
        ".kvbench-staging/"
        f"{PHASE12_STOPPED_CAMPAIGN_ID}."
        "df41252db7c860d9755c4843.staging/unified/entry-authority.json"
    ): "033f883fbfd001e515d345cb8f1c9e0568b6c60b6e3fd4bef2dacf51b1672e6c",
    (
        ".kvbench-staging/"
        f"{PHASE12_STOPPED_CAMPAIGN_ID}."
        "df41252db7c860d9755c4843.staging/unified/entry-g1-g4.json"
    ): "27e336995b26aac74e954432dc2786d6704b223f5a77bf2beb6fe34a648d65e9",
}


def _phase12_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _import_repository_script(name: str) -> Any:
    """Import one exact repository script for Phase 12 semantic replay."""

    if re.fullmatch(r"scripts\.[a-z0-9_]+", name) is None:
        raise ValueError("repository script module name is invalid")
    root_text = str(REPOSITORY_SCRIPT_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    expected = (
        REPOSITORY_SCRIPT_ROOT.joinpath(*name.split("."))
        .with_suffix(".py")
        .resolve()
    )
    if (
        not isinstance(module_file, str)
        or Path(module_file).resolve() != expected
        or expected.is_symlink()
        or not expected.is_file()
    ):
        raise ValueError("repository script resolved outside the exact root")
    return module


def _validate_phase12_failed_campaign(
    campaign: Path,
    *,
    campaign_id: str,
) -> None:
    """Replay the terminal failed-campaign contract without admitting a retry."""

    phase12 = _import_repository_script(
        "scripts.phase12_unified_admission"
    )

    phase12.validate_local_artifact(campaign, environ={})
    manifest = _phase12_json_object(campaign / "manifest.json")
    failure = _phase12_json_object(campaign / "failure.json")
    reservation = _phase12_json_object(
        campaign / "campaign-reservation.json"
    )
    execution_git_sha = manifest.get("execution_git_sha")
    preserved_run_ids = manifest.get("preserved_run_ids")
    if (
        manifest.get("schema_version")
        != "kvbench-phase12-failed-campaign-bundle-1.0.0"
        or manifest.get("run_id") != campaign_id
        or manifest.get("campaign_id") != campaign_id
        or manifest.get("status") != "failed"
        or not isinstance(execution_git_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", execution_git_sha) is None
        or manifest.get("authorized_container_digest")
        != phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST
        or not isinstance(preserved_run_ids, list)
        or len(preserved_run_ids) != len(set(preserved_run_ids))
        or preserved_run_ids != sorted(preserved_run_ids)
        or any(
            not isinstance(run_id, str)
            or not run_id.startswith(f"{campaign_id}-")
            for run_id in preserved_run_ids
        )
        or manifest.get("failure_path") != "failure.json"
        or manifest.get("failure_sha256")
        != phase12.sha256_file(campaign / "failure.json")
        or manifest.get("append_only") is not True
        or manifest.get("resume_permitted") is not False
        or manifest.get("selective_reruns") != 0
        or manifest.get("global_g5") != "NOT_EVALUATED"
        or manifest.get("pilot") != "NOT_READY"
        or manifest.get("full_scan") != "CLOSED"
        or manifest.get("quality_execution") != "LOCKED"
        or manifest.get("performance_data_frozen") is not False
        or manifest.get("speedup_calculated") is not False
        or manifest.get("r_hbm") is not None
        or failure.get("schema_version")
        != "kvbench-phase12-terminal-failure-1.0.0"
        or failure.get("campaign_id") != campaign_id
        or failure.get("execution_git_sha") != execution_git_sha
        or type(failure.get("failure_code")) is not int
        or not 1 <= failure["failure_code"] <= 255
        or failure.get("preserved_run_ids") != preserved_run_ids
        or failure.get("resume_permitted") is not False
        or failure.get("selective_rerun_permitted") is not False
        or failure.get("pilot") != "NOT_READY"
        or failure.get("full_scan") != "CLOSED"
        or failure.get("quality_execution") != "LOCKED"
        or failure.get("performance_data_frozen") is not False
        or failure.get("speedup_calculated") is not False
        or reservation.get("schema_version")
        != "kvbench-phase12-campaign-reservation-1.0.0"
        or reservation.get("campaign_id") != campaign_id
        or reservation.get("execution_git_sha") != execution_git_sha
        or reservation.get("append_only") is not True
        or reservation.get("reuse_permitted") is not False
    ):
        raise ValueError("failed Phase 12 campaign semantics differ")


def _validate_phase12_final_campaign(campaign: Path) -> None:
    """Dispatch one immutable campaign to its exact semantic validator."""

    phase12 = _import_repository_script(
        "scripts.phase12_unified_admission"
    )
    r2_artifact = _import_repository_script("scripts.r2_artifact")

    campaign_id = campaign.name
    try:
        manifest = _phase12_json_object(campaign / "manifest.json")
        schema = manifest.get("schema_version")
        if schema == "kvbench-phase12-campaign-bundle-1.0.0":
            result = phase12.validate_phase12_campaign(campaign)
            if (
                result.get("campaign_id") != campaign_id
                or result.get("status") not in {"PASS", "UNSTABLE"}
            ):
                raise ValueError("Phase 12 campaign verdict differs")
        elif schema == "kvbench-phase12-failed-campaign-bundle-1.0.0":
            _validate_phase12_failed_campaign(
                campaign,
                campaign_id=campaign_id,
            )
        else:
            raise ValueError("Phase 12 campaign schema is unrecognized")
    except (
        r2_artifact.ArtifactValidationError,
        json.JSONDecodeError,
        OSError,
        phase12.Phase12UnifiedAdmissionError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Phase 12 campaign validation failed: {campaign_id}"
        ) from error


def _phase12_active_path_is_allowed(
    relative: str,
    *,
    campaign_id: str,
) -> bool:
    """Accept only files and prefixes written by the one Phase 12 campaign."""

    parts = PurePosixPath(relative).parts
    if relative in {
        "campaign-reservation.json",
        "unified",
        "unified/entry-authority.json",
        "unified/entry-g1-g4.json",
        "unified/local-admission.json",
        "unified/campaign-result.json",
        "validation",
        "runs",
        "admission",
    }:
        return True
    if len(parts) >= 2 and parts[0] == "validation":
        target = parts[1]
        return bool(
            target in {"test-cuda", "test-graph"}
            and (
                len(parts) == 2
                or (
                    len(parts) == 3
                    and parts[2]
                    in {
                        "command.stdout.txt",
                        "command.stderr.txt",
                        "command.supervision.json",
                        "command.gpu-before.json",
                        "command.gpu-after.json",
                        "verdict.json",
                    }
                )
            )
        )
    if len(parts) >= 2 and parts[0] == "runs":
        run_id = parts[1]
        return bool(
            run_id.startswith(f"{campaign_id}-")
            and (
                len(parts) == 2
                or (
                    len(parts) == 3
                    and parts[2]
                    in {
                        "started.json",
                        "worker.stdout.txt",
                        "worker.stderr.txt",
                        "worker.supervision.json",
                        "worker.gpu-before.json",
                        "worker.gpu-after.json",
                        "failure.json",
                        "result.json",
                        "manifest.json",
                        "kernel-path.before.raw.dot",
                        "kernel-path.before.normalized.dot",
                        "kernel-path.after.raw.dot",
                        "kernel-path.after.normalized.dot",
                    }
                )
            )
        )
    if len(parts) >= 2 and parts[0] == "admission":
        return bool(
            parts[1]
            in {
                "bf16",
                "tq_4bit_nc",
                "tq_k3v4_nc",
                "tq_3bit_nc",
                "k4v4",
                "k2v4",
                "k2v2",
                "kvq4",
                "kvq3",
                "kvq2",
            }
            and (
                len(parts) == 2
                or (len(parts) == 3 and parts[2] == "report.json")
            )
        )
    return False


def _validate_phase12_active_test_evidence(
    stage: Path,
    *,
    target: str,
) -> None:
    """Validate any append-only prefix of one container-test evidence set."""

    phase12 = _import_repository_script(
        "scripts.phase12_unified_admission"
    )
    root = stage / "validation" / target
    if not root.exists():
        return
    ordered = (
        "command.stdout.txt",
        "command.stderr.txt",
        "command.supervision.json",
        "command.gpu-before.json",
        "command.gpu-after.json",
        "verdict.json",
    )
    present = {path.name for path in root.iterdir()}
    if present not in {
        frozenset(ordered[:count]) for count in range(len(ordered) + 1)
    }:
        raise ValueError("container-test evidence is not append-only")
    if "command.supervision.json" in present:
        phase12._validate_supervision_payload(
            _phase12_json_object(root / "command.supervision.json")
        )
    for name in ("command.gpu-before.json", "command.gpu-after.json"):
        if name in present:
            phase12._validate_idle_snapshot_payload(
                _phase12_json_object(root / name)
            )
    if "verdict.json" in present:
        verdict = _phase12_json_object(root / "verdict.json")
        if (
            set(verdict)
            != {
                "schema_version",
                "target",
                "authorized_container_digest",
                "passed",
                "cuda_executed_on_native_host",
            }
            or verdict.get("schema_version")
            != "kvbench-phase12-container-test-verdict-1.0.0"
            or verdict.get("target") != target
            or verdict.get("authorized_container_digest")
            != phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST
            or type(verdict.get("passed")) is not bool
            or verdict.get("cuda_executed_on_native_host") is not False
        ):
            raise ValueError("container-test verdict differs")


def _validate_phase12_active_stage(
    stage: Path,
    *,
    campaign_id: str,
) -> None:
    """Validate the sole in-flight stage without treating it as finalized."""

    phase12 = _import_repository_script(
        "scripts.phase12_unified_admission"
    )
    reservation = _phase12_json_object(
        stage / "campaign-reservation.json"
    )
    execution_git_sha = reservation.get("execution_git_sha")
    expected_reservation_keys = {
        "schema_version",
        "campaign_id",
        "execution_git_sha",
        "created_at_utc",
        "staging_directory",
        "reservation_directory",
        "append_only",
        "reuse_permitted",
    }
    if (
        set(reservation) != expected_reservation_keys
        or reservation.get("schema_version")
        != "kvbench-phase12-campaign-reservation-1.0.0"
        or reservation.get("campaign_id") != campaign_id
        or not isinstance(execution_git_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", execution_git_sha) is None
        or execution_git_sha[:8] != campaign_id.rsplit("-", 2)[1]
        or not isinstance(reservation.get("created_at_utc"), str)
        or not reservation["created_at_utc"].endswith("Z")
        or reservation.get("staging_directory") != stage.name
        or reservation.get("reservation_directory") != campaign_id
        or reservation.get("append_only") is not True
        or reservation.get("reuse_permitted") is not False
    ):
        raise ValueError("active Phase 12 reservation differs")
    entry_authority = _phase12_json_object(
        stage / "unified" / "entry-authority.json"
    )
    expected_authority = phase12._expected_entry_authority(
        campaign_id=campaign_id,
        execution_git_sha=execution_git_sha,
        repo_root=ROOT,
    )
    if entry_authority != expected_authority:
        raise ValueError("active Phase 12 authority differs")
    entry_gates = _phase12_json_object(
        stage / "unified" / "entry-g1-g4.json"
    )
    if entry_gates != phase12._expected_entry_g1_g4():
        raise ValueError("active Phase 12 G1-G4 evidence differs")
    for path in stage.rglob("*"):
        relative = path.relative_to(stage).as_posix()
        if not _phase12_active_path_is_allowed(
            relative,
            campaign_id=campaign_id,
        ):
            raise ValueError(
                f"active Phase 12 path is unrecognized: {relative}"
            )
    for target in ("test-cuda", "test-graph"):
        _validate_phase12_active_test_evidence(stage, target=target)
    if (stage / "validation" / "test-graph").exists() and not (
        stage / "validation" / "test-cuda" / "verdict.json"
    ).is_file():
        raise ValueError("test-graph exists before completed test-cuda")

    run_roots = stage / "runs"
    observed_slots: set[tuple[int, int]] = set()
    if run_roots.is_dir():
        for run_root in sorted(run_roots.iterdir()):
            run_id = run_root.name
            if (
                not run_root.is_dir()
                or phase12._RUN_ID_RE.fullmatch(run_id) is None
                or not run_id.startswith(f"{campaign_id}-")
            ):
                raise ValueError("active Phase 12 run identity differs")
            started = _phase12_json_object(run_root / "started.json")
            replicate_index = started.get("replicate_index")
            order_index = started.get("order_index")
            configuration = started.get("method_config_id")
            if (
                started.get("schema_version")
                != "kvbench-phase12-run-start-1.0.0"
                or started.get("run_id") != run_id
                or started.get("campaign_id") != campaign_id
                or type(replicate_index) is not int
                or type(order_index) is not int
                or not 0 <= replicate_index < len(
                    phase12.PHASE12_RANDOMIZATION_SEEDS
                )
                or not 0 <= order_index < len(phase12.MAIN_CONFIG_IDS)
                or started.get("seed")
                != phase12.PHASE12_RANDOMIZATION_SEEDS[replicate_index]
                or configuration
                != phase12.PHASE12_RANDOMIZED_ORDERS[replicate_index][
                    order_index
                ]
                or (replicate_index, order_index) in observed_slots
            ):
                raise ValueError("active Phase 12 run start differs")
            observed_slots.add((replicate_index, order_index))


def validate_phase12_blocked_artifact_root() -> list[str]:
    """Freeze stopped staging and validate only new append-only campaigns."""

    root = ROOT / "artifacts" / "phase12"
    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or not root.is_dir():
        return ["historical Phase 12 blocked artifact root is unsafe"]
    errors: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            return [
                "historical Phase 12 blocked artifact contains a symlink: "
                f"{relative}"
            ]
        if not path.is_dir() and not path.is_file():
            return [
                "historical Phase 12 blocked artifact contains an unsafe "
                f"entry: {relative}"
            ]
    stopped_stage_prefix = next(
        relative
        for relative in PHASE12_BLOCKED_STAGING_DIRECTORIES
        if relative.startswith(".kvbench-staging/")
        and relative.count("/") == 1
    )
    observed_stopped_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
        and (
            path.relative_to(root).as_posix()
            in {
                ".kvbench-reservations",
                (
                    ".kvbench-reservations/"
                    f"{PHASE12_STOPPED_CAMPAIGN_ID}"
                ),
                ".kvbench-staging",
            }
            or path.relative_to(root).as_posix() == stopped_stage_prefix
            or path.relative_to(root).as_posix().startswith(
                f"{stopped_stage_prefix}/"
            )
        )
    }
    observed_stopped_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix().startswith(
            f"{stopped_stage_prefix}/"
        )
    }
    if observed_stopped_directories != PHASE12_BLOCKED_STAGING_DIRECTORIES:
        errors.append(
            "historical Phase 12 blocked artifact directories differ"
        )
    if observed_stopped_files != set(
        PHASE12_BLOCKED_STAGING_FILE_SHA256S
    ):
        errors.append("historical Phase 12 blocked artifact files differ")
    for relative in PHASE12_BLOCKED_STAGING_DIRECTORIES:
        if not (root / relative).is_dir():
            errors.append(
                "historical Phase 12 blocked artifact directory differs: "
                f"{relative}"
            )
    for relative, expected in PHASE12_BLOCKED_STAGING_FILE_SHA256S.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(
                "historical Phase 12 blocked artifact checksum differs: "
                f"{relative}"
            )
    staging = root / ".kvbench-staging"
    stopped_stage = stopped_stage_prefix.split("/", 1)[1]
    staging_entries = (
        {
            path.name: path
            for path in staging.iterdir()
        }
        if staging.is_dir()
        else {}
    )
    active_entries = {
        name: path
        for name, path in staging_entries.items()
        if name != stopped_stage
    }
    active_campaign_id: str | None = None
    active_stage: Path | None = None
    if len(active_entries) > 1:
        errors.append("Phase 12 staging contains multiple active campaigns")
    elif active_entries:
        active_name, candidate = next(iter(active_entries.items()))
        match = re.fullmatch(
            (
                rf"({PHASE12_CAMPAIGN_ID_RE.pattern})"
                r"\.([0-9a-f]{24})\.staging"
            ),
            active_name,
        )
        if (
            match is None
            or not candidate.is_dir()
            or candidate.is_symlink()
            or match.group(1) == PHASE12_STOPPED_CAMPAIGN_ID
        ):
            errors.append("Phase 12 staging contains an unrecognized campaign")
        else:
            active_campaign_id = match.group(1)
            active_stage = candidate

    reservations = root / ".kvbench-reservations"
    reservation_ids = (
        {path.name for path in reservations.iterdir()}
        if reservations.is_dir()
        else set()
    )
    final_campaigns = {
        path.name: path
        for path in root.iterdir()
        if path.name not in {".kvbench-reservations", ".kvbench-staging"}
    }
    invalid_ids = sorted(
        campaign_id
        for campaign_id, path in final_campaigns.items()
        if (
            not path.is_dir()
            or PHASE12_CAMPAIGN_ID_RE.fullmatch(campaign_id) is None
            or campaign_id == PHASE12_STOPPED_CAMPAIGN_ID
        )
    )
    if invalid_ids:
        errors.append(
            f"Phase 12 artifact contains invalid campaign roots: {invalid_ids!r}"
        )
    if (
        active_campaign_id is not None
        and active_campaign_id in final_campaigns
    ):
        errors.append("active Phase 12 campaign was already finalized")
    expected_reservations = {
        PHASE12_STOPPED_CAMPAIGN_ID,
        *(
            campaign_id
            for campaign_id in final_campaigns
            if campaign_id not in invalid_ids
        ),
    }
    if active_campaign_id is not None:
        expected_reservations.add(active_campaign_id)
    if reservation_ids != expected_reservations:
        errors.append(
            "Phase 12 reservation/campaign identities differ"
        )
    for campaign_id in sorted(expected_reservations):
        reservation = reservations / campaign_id
        if (
            not reservation.is_dir()
            or reservation.is_symlink()
            or any(reservation.iterdir())
        ):
            errors.append(
                "Phase 12 reservation is unsafe or reused: "
                f"{campaign_id}"
            )
    for campaign_id, campaign in sorted(final_campaigns.items()):
        if campaign_id in invalid_ids:
            continue
        try:
            _validate_phase12_final_campaign(campaign)
        except ValueError:
            errors.append(
                f"Phase 12 campaign semantic validation failed: {campaign_id}"
            )
    if active_campaign_id is not None and active_stage is not None:
        phase12 = _import_repository_script(
            "scripts.phase12_unified_admission"
        )
        try:
            _validate_phase12_active_stage(
                active_stage,
                campaign_id=active_campaign_id,
            )
        except (
            json.JSONDecodeError,
            OSError,
            phase12.Phase12UnifiedAdmissionError,
            TypeError,
            ValueError,
        ):
            errors.append(
                "Phase 12 active staging semantic validation failed: "
                f"{active_campaign_id}"
            )
    return errors


def validate_phase3_artifact_root() -> list[str]:
    errors: list[str] = []
    artifacts = ROOT / "artifacts"
    phase3 = artifacts / "phase3"
    forbidden = (
        artifacts / "quality",
        artifacts / "profiler",
        ROOT / "docs" / "evidence" / "quality",
        ROOT / "paper-results",
        ROOT / "paper_results",
        ROOT / "results",
    )
    for path in forbidden:
        if path.exists() or path.is_symlink():
            errors.append(
                f"forbidden Phase 3 output path exists: "
                f"{path.relative_to(ROOT)}"
            )
    if artifacts.is_symlink():
        return [*errors, "artifact root is a symlink"]
    if artifacts.exists():
        unexpected = sorted(
            path.name
            for path in artifacts.iterdir()
            if path.name not in (
                APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE7_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE8_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE11_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE3_BACKUP_ARTIFACT_ROOT_NAMES
                | PHASE13_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13B_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13F_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13T_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13PB_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13PC_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13RQ4_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13D_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE14_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE15_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE16G_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE13_PREFIX_APPROVED_ARTIFACT_ROOT_NAMES
                | PHASE12_BLOCKED_ARTIFACT_ROOT_NAMES
            )
        )
        if unexpected:
            errors.append(
                f"unapproved artifact roots: {unexpected!r}"
            )
        for name in sorted(PHASE3_BACKUP_ARTIFACT_ROOT_NAMES):
            backup = artifacts / name
            if backup.exists() or backup.is_symlink():
                if backup.is_symlink() or not backup.is_dir():
                    errors.append(
                        f"unsafe Phase 3 backup artifact root: {name}"
                    )
    errors.extend(validate_phase12_blocked_artifact_root())
    errors.extend(validate_phase3_campaign_and_report_roots(artifacts))
    if not phase3.exists() and not phase3.is_symlink():
        return errors
    if phase3.is_symlink() or not phase3.is_dir():
        return [*errors, "Phase 3 artifact root is unsafe"]
    for control_name in (
        ".kvbench-staging",
        ".kvbench-reservations",
    ):
        control = phase3 / control_name
        if control.is_symlink() or (
            control.exists() and not control.is_dir()
        ):
            errors.append(
                f"Phase 3 control path is unsafe: {control_name}"
            )
    staging = phase3 / ".kvbench-staging"
    if staging.is_dir() and any(staging.iterdir()):
        errors.append("Phase 3 contains incomplete staging runs")
    for child in sorted(phase3.iterdir()):
        if child.name in {
            ".kvbench-staging",
            ".kvbench-reservations",
        }:
            continue
        if child.is_symlink() or not child.is_dir():
            errors.append(
                f"unsafe Phase 3 artifact child: {child.name}"
            )
            continue
        result = run(
            (
                "/usr/bin/python3",
                "-m",
                "kvbench",
                "validate-run",
                str(child),
            ),
            environment=phase2_environment(),
        )
        if result.returncode != 0:
            errors.append(
                f"invalid or incomplete Phase 3 run: {child.name}"
            )
    return errors


def validate_phase6a_artifact_root() -> list[str]:
    """Keep ignored Phase 6A outputs inside three exact append-only lanes."""

    root = ROOT / "artifacts" / "phase6a"
    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or not root.is_dir():
        return ["Phase 6A artifact root is unsafe"]
    allowed = {"container_g0", "bf16_parity", "r2_acceptance"}
    errors: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name not in allowed:
            errors.append(
                f"unapproved Phase 6A artifact root: {child.name}"
            )
        elif child.is_symlink() or not child.is_dir():
            errors.append(
                f"unsafe Phase 6A artifact root: {child.name}"
            )
    return errors


def validate_phase6_artifact_root() -> list[str]:
    """Require every ignored Phase 6 run to retain complete checksums."""

    root = ROOT / "artifacts" / "phase6"
    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or not root.is_dir():
        return ["Phase 6 artifact root is unsafe"]
    errors: list[str] = []
    controls = {".kvbench-staging", ".kvbench-reservations"}
    for control_name in sorted(controls):
        control = root / control_name
        if control.is_symlink() or (
            control.exists() and not control.is_dir()
        ):
            errors.append(f"Phase 6 control path is unsafe: {control_name}")
    staging = root / ".kvbench-staging"
    if staging.is_dir() and any(staging.iterdir()):
        errors.append("Phase 6 contains incomplete staging runs")
    for child in sorted(root.iterdir()):
        if child.name in controls:
            continue
        if child.is_symlink() or not child.is_dir():
            errors.append(f"unsafe Phase 6 artifact child: {child.name}")
            continue
        result = run(
            (
                "/usr/bin/python3",
                "-m",
                "kvbench",
                "validate-run",
                str(child),
            ),
            environment=phase2_environment(),
        )
        if result.returncode != 0:
            errors.append(
                f"invalid or incomplete Phase 6 run: {child.name}"
            )
    return errors


def validate_phase9_calibration_root() -> list[str]:
    """Structurally constrain the ignored append-only Phase 9 bundle root."""

    root = ROOT / "calibration" / "kvquant"
    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or not root.is_dir():
        return ["Phase 9 calibration root is unsafe"]
    errors: list[str] = []
    controls = {".kvbench-staging", ".kvbench-reservations"}
    for control_name in sorted(controls):
        control = root / control_name
        if control.is_symlink() or (
            control.exists() and not control.is_dir()
        ):
            errors.append(
                f"Phase 9 calibration control path is unsafe: {control_name}"
            )
    staging = root / ".kvbench-staging"
    if staging.is_dir() and any(staging.iterdir()):
        errors.append("Phase 9 contains incomplete calibration staging")
    for child in sorted(root.iterdir()):
        if child.name in controls:
            continue
        if (
            re.fullmatch(r"kvqcal-[0-9a-f]{32}", child.name) is None
            or child.is_symlink()
            or not child.is_dir()
        ):
            errors.append(
                f"unsafe Phase 9 calibration child: {child.name}"
            )
            continue
        for required in (
            "manifest.json",
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ):
            path = child / required
            if path.is_symlink() or not path.is_file():
                errors.append(
                    f"incomplete Phase 9 calibration control: "
                    f"{child.name}/{required}"
                )
        if child.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            errors.append(
                f"completed Phase 9 calibration remains writable: {child.name}"
            )
    return errors


def check_scope() -> int:
    errors: list[str] = []
    if not commit_is_ancestor(PHASE2_FINAL_COMMIT):
        errors.append(
            "the accepted Phase 2 final commit is not an ancestor of HEAD"
        )
    historical = historical_phase2_paths()
    if not commit_is_ancestor(PHASE4_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 4 entry commit is not an ancestor of HEAD"
        )
    historical_unexpected = sorted(
        historical - PHASE2_ALLOWED_PATHS
    )
    if historical_unexpected:
        errors.append(
            "historical files outside the approved Phase 2 plan: "
            f"{historical_unexpected!r}"
        )
    phase3 = historical_phase3_paths()
    phase3_unexpected = sorted(phase3 - PHASE3_ALLOWED_PATHS)
    if phase3_unexpected:
        errors.append(
            f"files outside the approved Phase 3 plan: {phase3_unexpected!r}"
        )
    if not commit_is_ancestor(PHASE5_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 5 entry commit is not an ancestor of HEAD"
        )
    phase4 = historical_phase4_paths()
    phase4_unexpected = sorted(phase4 - PHASE4_ALLOWED_PATHS)
    if phase4_unexpected:
        errors.append(
            f"historical files outside the approved Phase 4 plan: {phase4_unexpected!r}"
        )
    if not commit_is_ancestor(PHASE6A_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 6A entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE6_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 6 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE7_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 7 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE8_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 8 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE9P_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 9P entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE9P_FINAL_COMMIT):
        errors.append(
            "the accepted Phase 9P final commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE9_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 9 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE10_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 10 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE11PR_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 11P-R entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE11_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 11 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE11D_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 11D entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE11R_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 11R entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE12E_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 12E entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE11DQ23_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 11D-Q23 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE11RQ23_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 11R-Q23 entry commit is not an ancestor of HEAD"
        )
    if not commit_is_ancestor(PHASE12_ENTRY_COMMIT):
        errors.append(
            "the accepted Phase 12R entry commit is not an ancestor of HEAD"
        )
    phase5 = historical_phase5_paths()
    phase5_unexpected = sorted(phase5 - PHASE5_ALLOWED_PATHS)
    if phase5_unexpected:
        errors.append(
            f"historical files outside the approved Phase 5 plan: "
            f"{phase5_unexpected!r}"
        )
    phase6a = historical_phase6a_paths()
    phase6a_unexpected = sorted(phase6a - PHASE6A_ALLOWED_PATHS)
    if phase6a_unexpected:
        errors.append(
            "historical files outside the approved Phase 6A plan: "
            f"{phase6a_unexpected!r}"
        )
    phase6 = historical_phase6_paths()
    phase6_unexpected = sorted(phase6 - PHASE6_ALLOWED_PATHS)
    if phase6_unexpected:
        errors.append(
            f"historical files outside the approved Phase 6 plan: "
            f"{phase6_unexpected!r}"
        )
    phase7 = historical_phase7_paths()
    phase7_unexpected = sorted(phase7 - PHASE7_ALLOWED_PATHS)
    if phase7_unexpected:
        errors.append(
            f"historical files outside the approved Phase 7 plan: "
            f"{phase7_unexpected!r}"
        )
    phase8 = historical_phase8_paths()
    phase8_unexpected = sorted(phase8 - PHASE8_ALLOWED_PATHS)
    if phase8_unexpected:
        errors.append(
            f"historical files outside the approved Phase 8 plan: "
            f"{phase8_unexpected!r}"
        )
    phase9p = historical_phase9p_paths()
    phase9p_unexpected = sorted(phase9p - PHASE9P_ALLOWED_PATHS)
    if phase9p_unexpected:
        errors.append(
            f"historical files outside the approved Phase 9P plan: "
            f"{phase9p_unexpected!r}"
        )
    custody = current_kvquant_patch_custody_paths()
    custody_unexpected = sorted(
        custody - KVQUANT_PATCH_CUSTODY_ALLOWED_PATHS
    )
    if custody_unexpected:
        errors.append(
            "files outside the approved KVQuant patch-custody plan: "
            f"{custody_unexpected!r}"
        )
    phase9 = historical_phase9_paths()
    phase9_unexpected = sorted(phase9 - PHASE9_ALLOWED_PATHS)
    if phase9_unexpected:
        errors.append(
            "files outside the approved Phase 9 calibration plan: "
            f"{phase9_unexpected!r}"
        )
    phase10 = current_phase10_paths()
    phase10_unexpected = sorted(
        phase10 - PHASE10_ALLOWED_PATHS - PHASE11P_ALLOWED_PATHS
    )
    if phase10_unexpected:
        errors.append(
            "files outside the frozen Phase 10 and Phase 11P plan: "
            f"{phase10_unexpected!r}"
        )
    phase11pr = historical_phase11pr_paths()
    phase11pr_unexpected = sorted(phase11pr - PHASE11PR_ALLOWED_PATHS)
    if phase11pr_unexpected:
        errors.append(
            "files outside the approved Phase 11P-R correction plan: "
            f"{phase11pr_unexpected!r}"
        )
    phase11 = historical_phase11_paths()
    phase11_unexpected = sorted(phase11 - PHASE11_ALLOWED_PATHS)
    if phase11_unexpected:
        errors.append(
            "historical files outside the approved Phase 11 adapter plan: "
            f"{phase11_unexpected!r}"
        )
    phase11d = historical_phase11d_paths()
    phase11d_unexpected = sorted(phase11d - PHASE11D_ALLOWED_PATHS)
    if phase11d_unexpected:
        errors.append(
            "historical files outside the approved Phase 11D CUDA "
            "remediation plan: "
            f"{phase11d_unexpected!r}"
        )
    phase11r = historical_phase11r_paths()
    phase11r_unexpected = sorted(phase11r - PHASE11R_ALLOWED_PATHS)
    if phase11r_unexpected:
        errors.append(
            "historical files outside the approved Phase 11R "
            "admission-rerun plan: "
            f"{phase11r_unexpected!r}"
        )
    phase12e = historical_phase12e_paths()
    phase12e_unexpected = sorted(phase12e - PHASE12E_ALLOWED_PATHS)
    if phase12e_unexpected:
        errors.append(
            "historical files outside the approved Phase 12E "
            f"historical-authority remediation plan: {phase12e_unexpected!r}"
        )
    phase11dq23 = historical_phase11dq23_paths()
    phase11dq23_unexpected = sorted(
        phase11dq23 - PHASE11DQ23_ALLOWED_PATHS
    )
    if phase11dq23_unexpected:
        errors.append(
            "historical files outside the approved Phase 11D-Q23 CUDA "
            "remediation "
            f"plan: {phase11dq23_unexpected!r}"
        )
    changed = historical_phase11rq23_paths()
    phase11rq23_unexpected = sorted(
        changed - PHASE11RQ23_ALLOWED_PATHS
    )
    if phase11rq23_unexpected:
        errors.append(
            "files outside the approved Phase 11R-Q23 admission-rerun "
            f"plan: {phase11rq23_unexpected!r}"
        )
    phase12 = historical_phase12_paths()
    phase12_unexpected = sorted(phase12 - PHASE12_ALLOWED_PATHS)
    if phase12_unexpected:
        errors.append(
            "files outside the approved Phase 12R unified-admission "
            f"plan: {phase12_unexpected!r}"
        )
    for relative in sorted(phase12):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 12R: "
                f"{relative}"
            )
        if (
            Path(relative).suffix in RAW_RESULT_SUFFIXES
        ):
            errors.append(
                f"forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 12R Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 12R scope: {relative}"
            )
    phase13 = current_phase13_paths()
    phase13_unexpected = sorted(phase13 - PHASE13_ALLOWED_PATHS)
    if phase13_unexpected:
        errors.append(
            "files outside the approved Phase 13 Pilot plan: "
            f"{phase13_unexpected!r}"
        )
    for relative in sorted(phase13):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13 Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13 scope: {relative}"
            )
    phase13b = current_phase13b_paths()
    phase13b_unexpected = sorted(phase13b - PHASE13B_ALLOWED_PATHS)
    if phase13b_unexpected:
        errors.append(
            "files outside the approved Phase 13B remediation: "
            f"{phase13b_unexpected!r}"
        )
    for relative in sorted(phase13b):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13B: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13B Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13B scope: {relative}"
            )
    phase13r = current_phase13r_paths()
    phase13r_unexpected = sorted(phase13r - PHASE13R_ALLOWED_PATHS)
    if phase13r_unexpected:
        errors.append(
            "files outside the approved Phase 13R Pilot rerun: "
            f"{phase13r_unexpected!r}"
        )
    for relative in sorted(phase13r):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13R: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13R Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13R scope: {relative}"
            )
    phase13f = current_phase13f_paths()
    phase13f_unexpected = sorted(phase13f - PHASE13F_ALLOWED_PATHS)
    if phase13f_unexpected:
        errors.append(
            "files outside the approved Phase 13F feasibility remediation: "
            f"{phase13f_unexpected!r}"
        )
    for relative in sorted(phase13f):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13F: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13F Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13F scope: {relative}"
            )
    phase13r2 = current_phase13r2_paths()
    phase13r2_unexpected = sorted(phase13r2 - PHASE13R2_ALLOWED_PATHS)
    if phase13r2_unexpected:
        errors.append(
            "files outside the approved Phase 13R2 Pilot rerun: "
            f"{phase13r2_unexpected!r}"
        )
    for relative in sorted(phase13r2):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13R2: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13R2 Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13R2 scope: {relative}"
            )
    phase13t = current_phase13t_paths()
    phase13t_unexpected = sorted(phase13t - PHASE13T_ALLOWED_PATHS)
    if phase13t_unexpected:
        errors.append(
            "files outside the approved Phase 13T-R remediation and Pilot "
            f"rerun: {phase13t_unexpected!r}"
        )
    for relative in sorted(phase13t):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13T-R: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13T-R Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13T-R scope: {relative}"
            )
    phase13pr = current_phase13pr_paths()
    phase13pr_unexpected = sorted(phase13pr - PHASE13PR_ALLOWED_PATHS)
    if phase13pr_unexpected:
        errors.append(
            "files outside the approved Phase 13P-R prefix remediation and "
            f"Pilot rerun: {phase13pr_unexpected!r}"
        )
    for relative in sorted(phase13pr):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13P-R: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13P-R Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13P-R scope: {relative}"
            )
    phase13pb = current_phase13pb_paths()
    phase13pb_unexpected = sorted(phase13pb - PHASE13PB_ALLOWED_PATHS)
    if phase13pb_unexpected:
        errors.append(
            "files outside the approved Phase 13P-B batch-exact prefix "
            f"remediation: {phase13pb_unexpected!r}"
        )
    for relative in sorted(phase13pb):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13P-B: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13P-B Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13P-B scope: {relative}"
            )
    phase13pc = current_phase13pc_paths()
    phase13pc_unexpected = sorted(phase13pc - PHASE13PC_ALLOWED_PATHS)
    if phase13pc_unexpected:
        errors.append(
            "files outside the approved Phase 13P-C CUDA-context isolation "
            f"remediation: {phase13pc_unexpected!r}"
        )
    for relative in sorted(phase13pc):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13P-C: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13P-C Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13P-C scope: {relative}"
            )
    phase13rq4 = current_phase13rq4_paths()
    phase13rq4_unexpected = sorted(
        phase13rq4 - PHASE13RQ4_ALLOWED_PATHS
    )
    if phase13rq4_unexpected:
        errors.append(
            "files outside the approved Phase 13R q4 workspace remediation: "
            f"{phase13rq4_unexpected!r}"
        )
    for relative in sorted(phase13rq4):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13R q4: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13R q4 Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13R q4 scope: {relative}"
            )
    phase13d = current_phase13d_paths()
    phase13d_unexpected = sorted(phase13d - PHASE13D_ALLOWED_PATHS)
    if phase13d_unexpected:
        errors.append(
            "files outside the approved Phase 13D knee densification: "
            f"{phase13d_unexpected!r}"
        )
    for relative in sorted(phase13d):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 13D: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 13D Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 13D scope: {relative}"
            )
    phase14 = current_phase14_paths()
    phase14_unexpected = sorted(phase14 - PHASE14_ALLOWED_PATHS)
    if phase14_unexpected:
        errors.append(
            "files outside the approved Phase 14 Graph A/B mechanism: "
            f"{phase14_unexpected!r}"
        )
    for relative in sorted(phase14):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 14: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 14 Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 14 scope: {relative}"
            )
    phase15 = current_phase15_paths()
    phase15_unexpected = sorted(phase15 - PHASE15_ALLOWED_PATHS)
    if phase15_unexpected:
        errors.append(
            "files outside the approved Phase 15 profiler subset: "
            f"{phase15_unexpected!r}"
        )
    phase16g = current_phase16g_paths()
    phase16g_unexpected = sorted(phase16g - PHASE16G_ALLOWED_PATHS)
    if phase16g_unexpected:
        errors.append(
            "files outside the approved Phase 16G geometry admission: "
            f"{phase16g_unexpected!r}"
        )
    for relative in sorted(phase16g):
        if relative.startswith("docs/evidence/e00/"):
            errors.append(f"immutable E00 evidence changed: {relative}")
        if relative in QUALITY_PROTOCOL_HASHES:
            errors.append(
                "quality protocol changed during Phase 16G: "
                f"{relative}"
            )
        if Path(relative).suffix in RAW_RESULT_SUFFIXES:
            errors.append(
                "forbidden binary, kernel, model, or profiler artifact "
                f"in Phase 16G Git scope: {relative}"
            )
        if relative.startswith(
            (
                "artifacts/profiler/",
                "artifacts/quality/",
                "paper-results/",
                "paper_results/",
                "results/",
            )
        ):
            errors.append(
                f"forbidden result tree in Phase 16G scope: {relative}"
            )
    e00_changes = git_paths(
        (
            "diff",
            "--name-only",
            "-z",
            PHASE2_FINAL_COMMIT,
            "--",
            *E00_PROTECTED_PATHS,
        )
    )
    unexpected_e00_changes = e00_changes - PHASE6A_E00_ALLOWED_PATHS
    if unexpected_e00_changes:
        errors.append(
            f"certified E00 implementation changed outside the exact "
            f"Phase 6A extension: {sorted(unexpected_e00_changes)!r}"
        )
    entry_makefile = git(
        ("show", f"{PHASE2_FINAL_COMMIT}:Makefile")
    )
    current_path = ROOT / "Makefile"
    if entry_makefile.returncode != 0 or not current_path.is_file():
        errors.append("cannot compare certified Makefile targets")
    else:
        current = current_path.read_text(encoding="utf-8")
        for target in ("preflight", "preflight-unit"):
            if make_target_block(
                current, target
            ) != make_target_block(entry_makefile.stdout, target):
                errors.append(
                    f"certified Makefile target semantics changed: {target}"
                )
    errors.extend(validate_phase3_artifact_root())
    errors.extend(validate_phase6a_artifact_root())
    errors.extend(validate_phase6_artifact_root())
    errors.extend(validate_phase9_calibration_root())
    forbidden_modules = (
        "src/kvbench/methods/kvquant",
        "src/kvbench/runtime/kvquant.py",
    )
    for relative in forbidden_modules:
        if (ROOT / relative).exists():
            errors.append(
                f"deferred KVQuant execution implementation exists: {relative}"
            )
    return report("scope", errors)


def check_immutable() -> int:
    errors: list[str] = []
    diff = git(
        (
            "diff",
            "--quiet",
            "--no-ext-diff",
            PHASE2_ENTRY_COMMIT,
            "--",
            "docs/evidence/e00",
        )
    )
    if diff.returncode != 0:
        errors.append(
            "E00 evidence differs from the Phase 2 entry commit"
        )
    evidence_status = git_output(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "docs/evidence/e00",
        )
    )
    if evidence_status:
        errors.append(
            "tracked or untracked changes exist under E00 evidence"
        )
    flags = git_output(
        ("ls-files", "-v", "--", "docs/evidence/e00")
    )
    if any(not line.startswith("H ") for line in flags.splitlines()):
        errors.append(
            "E00 evidence has a non-ordinary Git index flag"
        )
    for run_id, expected in E00_RUNS.items():
        errors.extend(validate_e00_run(run_id, expected))
    return report("immutable", errors)


def parse_requirements(
    path: Path,
) -> tuple[dict[str, str], list[str]]:
    locked: dict[str, str] = {}
    errors: list[str] = []
    directives = (
        "--index-url ",
        "--extra-index-url ",
        "--only-binary=",
    )
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--"):
            if not stripped.startswith(directives):
                errors.append(
                    f"unexpected requirements directive at "
                    f"line {line_number}"
                )
            continue
        match = REQUIREMENT.fullmatch(stripped)
        if match is None:
            errors.append(
                f"unhashed or malformed requirement at line {line_number}"
            )
            continue
        name = normalize_distribution(match.group(1))
        if name in locked:
            errors.append(f"duplicate locked distribution: {name}")
        locked[name] = match.group(2)
    return locked, errors


def installed_venv() -> tuple[dict[str, str], str | None]:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        return {}, "certified .venv interpreter is missing"
    code = (
        "import importlib.metadata as m,json,re;"
        "norm=lambda s:re.sub(r'[-_.]+','-',s).lower();"
        "print(json.dumps({norm(d.metadata['Name']):d.version "
        "for d in m.distributions()},sort_keys=True))"
    )
    result = run(
        (str(python), "-I", "-c", code),
        environment={
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if result.returncode != 0:
        return {}, (
            f"cannot inspect certified .venv "
            f"(exit {result.returncode})"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}, "certified .venv distribution inventory is not JSON"
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in payload.items()
    ):
        return {}, (
            "certified .venv distribution inventory has invalid types"
        )
    return payload, None


def git_blob(commit: str, relative: str) -> bytes | None:
    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    result = subprocess.run(
        ["/usr/bin/git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        env=environment,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout if result.returncode == 0 else None


def check_package_lock() -> int:
    errors: list[str] = []
    requirements_path = (
        ROOT / "preflight" / "requirements-e00.txt"
    )
    system_lock_path = (
        ROOT / "preflight" / "system-packages.lock.json"
    )
    for relative in (
        "preflight/requirements-e00.txt",
        "preflight/system-packages.lock.json",
    ):
        path = ROOT / relative
        current = path.read_bytes() if path.is_file() else None
        if current is None or current != git_blob(
            ENVIRONMENT_COMMIT, relative
        ):
            errors.append(
                f"E00 environment-lock bytes changed: {relative}"
            )
    if not requirements_path.is_file():
        errors.append("E00 Python requirements lock is missing")
        locked: dict[str, str] = {}
    else:
        locked, parse_errors = parse_requirements(
            requirements_path
        )
        errors.extend(parse_errors)
        if len(locked) != 35:
            errors.append(
                "E00 Python lock is not the accepted "
                "35-distribution closure"
            )
    installed, installed_error = installed_venv()
    if installed_error:
        errors.append(installed_error)
    elif installed != locked:
        errors.append(
            "certified .venv distributions differ from the exact E00 lock"
        )
    pip_check = run(
        (
            str(ROOT / ".venv" / "bin" / "python"),
            "-I",
            "-m",
            "pip",
            "check",
        ),
        environment={
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if pip_check.returncode != 0:
        errors.append(
            "pip check failed in the certified E00 environment"
        )
    try:
        system_lock = json.loads(
            system_lock_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        system_lock = {}
        errors.append("E00 system package lock is invalid JSON")
    if system_lock:
        if system_lock.get("schema_version") != 1:
            errors.append(
                "unexpected E00 system package lock schema version"
            )
        packages = system_lock.get("dpkg_packages")
        tools = system_lock.get("tools")
        if not isinstance(packages, list) or not isinstance(tools, list):
            errors.append(
                "E00 system package/tool lock entries are malformed"
            )
        else:
            package_names = [
                item.get("name")
                for item in packages
                if isinstance(item, dict)
            ]
            tool_names = [
                item.get("name")
                for item in tools
                if isinstance(item, dict)
            ]
            if len(package_names) != len(set(package_names)):
                errors.append(
                    "duplicate E00 system package lock name"
                )
            if len(tool_names) != len(set(tool_names)):
                errors.append("duplicate E00 tool lock name")
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                invocation = Path(
                    str(tool.get("invocation_path", ""))
                )
                try:
                    resolved = invocation.resolve(strict=True)
                except OSError:
                    errors.append(
                        f"locked E00 tool is missing: {tool.get('name')}"
                    )
                    continue
                if str(resolved) != tool.get("resolved_path"):
                    errors.append(
                        f"locked E00 tool path mismatch: "
                        f"{tool.get('name')}"
                    )
                if (
                    not resolved.is_file()
                    or sha256(resolved) != tool.get("sha256")
                ):
                    errors.append(
                        f"locked E00 tool hash mismatch: "
                        f"{tool.get('name')}"
                    )
            if package_names:
                dollar = chr(36)
                dpkg_format = (
                    dollar
                    + "{Package}\t"
                    + dollar
                    + "{Version}\t"
                    + dollar
                    + "{Architecture}\n"
                )
                query = run(
                    (
                        "/usr/bin/dpkg-query",
                        "-W",
                        f"-f={dpkg_format}",
                        *package_names,
                    ),
                    environment={"LC_ALL": "C", "LANG": "C"},
                )
                observed: dict[str, tuple[str, str]] = {}
                if query.returncode == 0:
                    for line in query.stdout.splitlines():
                        parts = line.split("\t")
                        if len(parts) == 3:
                            observed[parts[0]] = (
                                parts[1],
                                parts[2],
                            )
                else:
                    errors.append(
                        "cannot query locked E00 system packages"
                    )
                for package in packages:
                    expected = (
                        package.get("version"),
                        package.get("architecture"),
                    )
                    if observed.get(package.get("name")) != expected:
                        errors.append(
                            f"locked system package mismatch: "
                            f"{package.get('name')}"
                        )
    pyproject_path = ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        errors.append("pyproject.toml is missing")
    else:
        try:
            pyproject = tomllib.loads(
                pyproject_path.read_text(encoding="utf-8")
            )
        except (UnicodeError, tomllib.TOMLDecodeError):
            pyproject = {}
            errors.append("pyproject.toml is invalid")
        project = (
            pyproject.get("project", {})
            if isinstance(pyproject, dict)
            else {}
        )
        dependencies = (
            project.get("dependencies", [])
            if isinstance(project, dict)
            else []
        )
        optional = (
            project.get("optional-dependencies", {})
            if isinstance(project, dict)
            else {}
        )
        if dependencies:
            errors.append(
                "Phase 2 pyproject declares runtime dependencies"
            )
        if optional:
            errors.append(
                "Phase 2 pyproject declares optional dependency groups"
            )
    return report("package-lock", errors)


def check_phase3_package_lock() -> int:
    errors: list[str] = []
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TZ": "UTC",
    }
    result = run(
        (
            str(ROOT / ".venv" / "bin" / "python"),
            "scripts/bootstrap_phase3.py",
            "verify",
        ),
        environment=environment,
    )
    payload: dict[str, Any] = {}
    if result.returncode != 0:
        errors.append(
            "Phase 3 dependency verification failed "
            f"(exit {result.returncode})"
        )
    else:
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            errors.append(
                "Phase 3 dependency verifier output is not JSON"
            )
        else:
            if isinstance(parsed, dict):
                payload = parsed
            else:
                errors.append(
                    "Phase 3 dependency verifier output is not an object"
                )
    if payload and payload.get("status") != "pass":
        errors.append(
            "Phase 3 dependency verifier did not report pass"
        )
    if payload and payload.get("target") != str(
        ROOT / ".phase3" / "site-packages"
    ):
        errors.append(
            "Phase 3 dependency target identity mismatch"
        )
    return report("phase3-package-lock", errors)


def check_method() -> int:
    method = os.environ.get("KVBENCH_METHOD", "")
    errors: list[str] = []
    if method not in METHOD_NAMES:
        errors.append(
            "METHOD must be one of bf16, turboquant, kivi, or kvquant"
        )
    else:
        error = cli_config_error(
            f"configs/methods/{method}.yaml"
        )
        if error:
            errors.append(error)
    return report("method", errors)


def check_run_id() -> int:
    run_id = os.environ.get("KVBENCH_RUN_ID", "")
    errors: list[str] = []
    if SAFE_IDENTIFIER.fullmatch(run_id) is None or ".." in run_id:
        errors.append("RUN_ID is missing or unsafe")
    return report("run-id", errors)


CHECKS: dict[str, Callable[[], int]] = {
    "format": check_format,
    "lint": check_lint,
    "hot-path": check_hot_path,
    "annotations": check_annotations,
    "phase3-annotations": check_phase3_annotations,
    "configs": check_configs,
    "provenance": check_provenance,
    "scope": check_scope,
    "immutable": check_immutable,
    "package-lock": check_package_lock,
    "phase3-package-lock": check_phase3_package_lock,
    "method": check_method,
    "run-id": check_run_id,
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=(*CHECKS, "all"))
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(
        sys.argv[1:] if argv is None else argv
    )
    if arguments.check != "all":
        try:
            return CHECKS[arguments.check]()
        except ValidationFailure as error:
            return report(arguments.check, [str(error)])
    result = 0
    for name in (
        "format",
        "lint",
        "hot-path",
        "annotations",
        "phase3-annotations",
        "configs",
        "provenance",
        "scope",
        "immutable",
        "package-lock",
        "phase3-package-lock",
    ):
        try:
            result |= CHECKS[name]()
        except ValidationFailure as error:
            result |= report(name, [str(error)])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
