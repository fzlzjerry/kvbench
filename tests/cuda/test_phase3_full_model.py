"""Untimed exact-checkpoint numerical controls for the Phase 3 BF16 SUT."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from kvbench.adapters import (
    build_method_adapter,
    declared_bf16_runtime_context,
)

from kvbench.runtime.backend import backend_identity, forced_flash_execution
from kvbench.runtime.gqa_device_dispatch import REQUIRED_SUT_SOURCES
from kvbench.runtime.model_loader import load_frozen_model
from kvbench.runtime.numerical import (
    tensor_sha256_untimed,
    validate_full_model_reference,
)
from kvbench.runtime.phase3_coordinator import (
    _cache_identity,
    _expected_phase3_raw_audit_operations,
)
from kvbench.runtime.phase3_raw_audit_evidence import (
    PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND,
    RAW_AUDIT_STATUS_COMPLETED,
    REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS,
    Phase3RawAuditOperationRecord,
)
from kvbench.runtime.phase3_worker import (
    _phase3_raw_audit_producer_bindings,
)
from kvbench.schema import GraphMode, RunnerKind
from kvbench.schema.phase3 import (
    BF16BackendIdentity,
    Phase3ProcessPoint,
)


def collect_exact_endpoint_audit(
    loaded: object,
    *,
    graph_mode: GraphMode,
) -> tuple[object, Phase3RawAuditOperationRecord, Path]:
    """Run one exact B=1, L=128 endpoint audit without normal timing."""

    repository_root = Path(__file__).resolve().parents[2]
    point = Phase3ProcessPoint(
        point_id=f"fixed_l-b1-l128-{graph_mode.value}-r1",
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=graph_mode,
        batch_size=1,
        context_length=128,
        output_steps=1,
        process_replicate=1,
        stability_member=False,
    )
    source_hashes = {
        relative: hashlib.sha256(
            (repository_root / relative).read_bytes()
        ).hexdigest()
        for relative in REQUIRED_SUT_SOURCES
    }
    cache = _cache_identity(
        point,
        implementation_sha256=source_hashes[
            "src/kvbench/runtime/static_cache.py"
        ],
    )
    backend = BF16BackendIdentity.from_dict(backend_identity())
    operations = _expected_phase3_raw_audit_operations(
        point=point,
        run_id=(
            "phase3-remediation-endpoint-"
            f"{graph_mode.value}-control"
        ),
        git_sha="5" * 40,
        cache=cache,
        backend=backend,
        source_sha256_by_path=source_hashes,
    )
    device = torch.device("cuda:0")
    prefix = torch.arange(
        1_000,
        1_128,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    decode = torch.tensor([[6_000]], dtype=torch.long, device=device)
    evidence_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "kvbench-phase3-endpoint-audit-"
                f"{graph_mode.value}-"
            ),
            dir="/tmp",
        )
    )
    with torch.inference_mode(), forced_flash_execution():
        session, bindings = _phase3_raw_audit_producer_bindings(
            expected_operations=operations,
            torch=torch,
            device=device,
            loaded=loaded,
            point=point,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
        operation, producer = bindings[0]
        record = producer(operation, evidence_root)
    if record.status != RAW_AUDIT_STATUS_COMPLETED:
        details = []
        for declared in record.files:
            if "error" in declared.kind:
                details.append(
                    (evidence_root / declared.path).read_text(
                        encoding="utf-8"
                    ).strip()
                )
        raise AssertionError(
            "exact endpoint audit failed; "
            f"evidence={evidence_root}; details={details}"
        )
    return session, record, evidence_root


def retained_fixed_output_checksums_untimed(
    session: object,
    *,
    repetitions: int = 160,
) -> tuple[str, ...]:
    """Repeat the admitted callable without invoking the timing harness."""

    checksums: list[str] = []
    with torch.inference_mode(), forced_flash_execution():
        operation = session.fixed_measurement_callable()
        for _ in range(repetitions):
            checksums.append(tensor_sha256_untimed(operation()))
    session.mark_measured()
    return tuple(checksums)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3FullModelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        loaded = load_frozen_model(device="cuda:0")
        cls.loaded = loaded
        cls.model = loaded.model
        cls.tokenizer = loaded.tokenizer
        cls.identity = loaded.identity
        cls.adapter = build_method_adapter(
            "bf16",
            declared_bf16_runtime_context(cls.model),
        )

    def test_exact_model_fixed_and_growing_match_dynamic_cache_reference(
        self,
    ) -> None:
        self.assertEqual(self.model.__class__.__name__, "LlamaForCausalLM")
        self.assertEqual(
            self.tokenizer.__class__.__name__,
            "PreTrainedTokenizerFast",
        )
        self.assertEqual(len(self.identity.file_hashes), 11)
        prefix = torch.arange(
            1_000,
            1_008,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        decode = torch.arange(
            2_000,
            2_003,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        result = validate_full_model_reference(
            self.model,
            prefix,
            decode,
            method=self.adapter,
        )
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.reference_cache_type, "DynamicCache")
        self.assertTrue(result.reference_implementation_restored)
        self.assertTrue(result.fixed_repeat_exact)
        self.assertTrue(result.fixed_historical_cache_unchanged)
        self.assertEqual(len(result.fixed_steps), 3)
        self.assertEqual(len(result.growing_steps), 3)
        for evidence in (*result.fixed_steps, *result.growing_steps):
            self.assertTrue(evidence.comparison.passed, evidence.to_dict())
            self.assertTrue(evidence.comparison.finite, evidence.to_dict())
        serialized = result.to_dict()
        self.assertFalse(serialized["timing_collected"])
        self.assertFalse(serialized["performance_claim_eligible"])

    def test_exact_endpoint_eager_audit_admits_without_normal_timing(
        self,
    ) -> None:
        session, record, evidence_root = collect_exact_endpoint_audit(
            self.loaded,
            graph_mode=GraphMode.EAGER,
        )
        print(f"preserved_endpoint_audit={evidence_root}")
        self.assertEqual(session.state, "ready")
        self.assertEqual(session.method.name, "bf16")
        self.assertEqual(len(session.adapter_config_fingerprint), 64)
        self.assertEqual(
            {item.kind for item in record.files},
            set(REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS)
            | {PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND},
        )
        self.assertFalse(session.provenance_payload()["graph_retained"])
        audit_checksum, audit_finite = session.audit_output(0)
        observed = retained_fixed_output_checksums_untimed(session)
        self.assertTrue(audit_finite)
        self.assertEqual(set(observed), {audit_checksum})



if __name__ == "__main__":
    unittest.main()
