from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interactive import LiveArm2DOFConfig, run_multi_target_deployment


class MultiTargetDeploymentTests(unittest.TestCase):
    def test_deployment_sequence_records_metrics_for_reached_target(self) -> None:
        rows = run_multi_target_deployment(
            (("home", (1.8, 0.0)),),
            method="fuzzy_base",
            mode="fuzzy",
            config=LiveArm2DOFConfig(target=(1.8, 0.0)),
            max_steps_per_target=5,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.target_id, "home")
        self.assertTrue(row.done)
        self.assertEqual(row.steps, 0)
        self.assertEqual(row.start_step, 0)
        self.assertEqual(row.end_step, 0)
        self.assertLessEqual(row.final_distance, 1e-12)

    def test_deployment_sequence_rejects_empty_targets(self) -> None:
        with self.assertRaises(ValueError):
            run_multi_target_deployment(
                (),
                method="fuzzy_base",
                mode="fuzzy",
            )

    def test_deployment_sequence_rejects_non_positive_step_budget(self) -> None:
        with self.assertRaises(ValueError):
            run_multi_target_deployment(
                (("home", (1.8, 0.0)),),
                method="fuzzy_base",
                mode="fuzzy",
                max_steps_per_target=0,
            )


if __name__ == "__main__":
    unittest.main()
