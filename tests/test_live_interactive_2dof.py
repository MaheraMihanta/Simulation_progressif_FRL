from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interactive import LiveArm2DOFConfig, LiveArm2DOFSimulation
from rl import FuzzyDynamicStateEncoder, FuzzyResidualQLearningConfig


class LiveInteractive2DOFTests(unittest.TestCase):
    def test_pid_live_step_returns_finite_summary(self) -> None:
        sim = LiveArm2DOFSimulation(mode="pid")

        summary = sim.step()

        self.assertEqual(summary["mode"], "pid")
        self.assertEqual(summary["step"], 1)
        self.assertTrue(np.isfinite(summary["distance"]))
        self.assertTrue(np.isfinite(summary["torque_norm"]))

    def test_target_can_be_changed_without_resetting_robot(self) -> None:
        sim = LiveArm2DOFSimulation(mode="fuzzy")
        sim.step()
        old_step = sim.summary()["step"]

        sim.set_target((0.85, 0.85))
        summary = sim.summary()

        self.assertEqual(summary["step"], old_step)
        np.testing.assert_allclose(summary["target"], [0.85, 0.85])
        self.assertEqual(summary["message"], "target=(0.850, 0.850)")

    def test_unreachable_target_is_rejected(self) -> None:
        sim = LiveArm2DOFSimulation()

        with self.assertRaises(ValueError):
            sim.set_target((5.0, 5.0))

    def test_disturbance_is_applied_and_decays(self) -> None:
        sim = LiveArm2DOFSimulation(
            config=LiveArm2DOFConfig(disturbance_decay=0.5),
        )
        sim.apply_disturbance((10.0, -4.0))

        sim.step()
        external_torque = sim.summary()["external_torque"]

        np.testing.assert_allclose(external_torque, [5.0, -2.0])

    def test_safe_residual_mode_disables_stale_residual(self) -> None:
        encoder = FuzzyDynamicStateEncoder()
        learning_config = FuzzyResidualQLearningConfig(max_steps_per_episode=10)
        q_value = np.zeros((encoder.n_rules, 9), dtype=float)
        q_value[:, 1] = 1.0
        sim = LiveArm2DOFSimulation(
            config=LiveArm2DOFConfig(
                safety_patience=1,
                safety_min_progress=10.0,
            ),
            mode="fuzzy_rl_safe",
            q_value=q_value,
            encoder=encoder,
            learning_config=learning_config,
        )

        first = sim.step()
        second = sim.step()

        self.assertTrue(first["residual_disabled"])
        self.assertEqual(first["residual_switch_step"], 1)
        self.assertTrue(second["residual_disabled"])
        self.assertEqual(second["action_index"], 0)


if __name__ == "__main__":
    unittest.main()
