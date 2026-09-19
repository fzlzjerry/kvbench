from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import qp1_quality_contract as qp1


class QP1QualityContractTests(unittest.TestCase):
    def test_anchor_alignment_and_boundary_policy(self) -> None:
        spans = [{"document_id": "d", "start": 0, "end": 10000}]
        anchors = qp1.choose_anchors(spans, 4096, 128, 16, 7)
        self.assertEqual(len(anchors), 16)
        for anchor in anchors:
            self.assertEqual(anchor["prefix_end"], anchor["burn_in_index"])
            self.assertEqual(anchor["target_start"], anchor["burn_in_index"] + 1)
            self.assertLessEqual(anchor["target_end_exclusive"], 10000)

    def test_deterministic_gzip_and_resume_unit(self) -> None:
        data = b"fixture\n"
        self.assertEqual(qp1.gzip_bytes(data), qp1.gzip_bytes(data))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "done.jsonl.gz"
            qp1.write(path, qp1.gzip_bytes(data))
            before = qp1.sha256_file(path)
            self.assertEqual(gzip.decompress(path.read_bytes()), data)
            self.assertEqual(qp1.sha256_file(path), before)

    def test_contract_constants_are_frozen(self) -> None:
        self.assertEqual(qp1.CONFIG_IDS, ["bf16", "tq_4bit_nc", "tq_k3v4_nc",
                         "tq_3bit_nc", "k4v4", "k2v4", "k2v2", "kvq4", "kvq3", "kvq2"])
        self.assertEqual(qp1.SEED, 20260722)
        self.assertEqual(qp1.MODEL_MAX, 131072)
        self.assertEqual(len(qp1.TASKS), 13)

    def test_final_fixture_token_hash_and_conditioning_split(self) -> None:
        root = qp1.ROOT / "artifacts/quality_contract" / qp1.CONTRACT_ID
        index_path = root / "selected_inputs/longbench_e/qasper.index.json"
        if not index_path.exists():
            self.skipTest("final QP-1 fixture not materialized")
        index = json.loads(index_path.read_text())
        raw = gzip.decompress(
            (root / "selected_inputs/longbench_e/qasper.token_ids.i32.gz").read_bytes()
        )
        values = np.frombuffer(raw, dtype="<i4")
        first = index["records"][0]
        sample = values[first["offset"]:first["offset"] + first["length"]]
        self.assertEqual(qp1.sha256_bytes(sample.tobytes()), first["token_ids_sha256"])
        self.assertEqual(sample.tolist(), sample[:-16].tolist() + sample[-16:].tolist())

    def test_final_contract_is_approval_pending_and_score_free(self) -> None:
        root = qp1.ROOT / "artifacts/quality_contract" / qp1.CONTRACT_ID
        if not root.exists():
            self.skipTest("final QP-1 fixture not materialized")
        result = qp1.validate(root)
        self.assertEqual(result["quality_execution"], "LOCKED")
        self.assertFalse(result["quality_scores_computed"])


if __name__ == "__main__":
    unittest.main()
