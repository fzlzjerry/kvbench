"""Full execution-path, allocation, and graph checks for Phase 6."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    evaluate_fixture_configuration,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_cache import TURBOQUANT_MANDATORY_CONFIGS
from kvbench.schema.phase6 import AUTHORIZED_CONTAINER_DIGEST


def _authorized_environment_declared() -> bool:
    return (
        Path("/.dockerenv").is_file()
        and os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        == AUTHORIZED_CONTAINER_DIGEST
        and os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        == PHASE6_CONTAINER_ENVIRONMENT_VALUE
    )


@unittest.skipUnless(
    _authorized_environment_declared(),
    "Phase 6 CUDA Graph is authorized only in the Measurement Container",
)
class Phase6TurboQuantGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = require_authorized_cuda_environment(
            AUTHORIZED_CONTAINER_DIGEST
        )

    def test_all_mandatory_configs_pass_common_graph_and_audits(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase6-graph-test-"
        ) as temporary:
            root = Path(temporary)
            for configuration in TURBOQUANT_MANDATORY_CONFIGS:
                with self.subTest(configuration=configuration):
                    result = evaluate_fixture_configuration(
                        configuration,
                        evidence_directory=root / configuration,
                    )
                    self.assertTrue(result["passed"], result)
                    self.assertTrue(result["execution_path"]["passed"])
                    self.assertTrue(result["allocation"]["passed"])
                    self.assertEqual(
                        result["allocation"]["unknown_allocations"],
                        0,
                    )
                    self.assertTrue(result["graph"]["captured"])
                    self.assertTrue(result["graph"]["passed"])
                    self.assertTrue(
                        result["graph"]["eager_graph_comparison"]["passed"]
                    )
                    self.assertTrue(
                        result["graph"]["replay_outputs_exact"]
                    )
                    self.assertTrue(
                        result["graph"]["replay_allocation"]["passed"]
                    )
                    self.assertFalse(result["graph"]["fallback"])
                    self.assertTrue(result["graph"]["pointers_stable"])


if __name__ == "__main__":
    unittest.main()
