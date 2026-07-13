from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs import Arm2DOFDynamicEnvConfig
from rl import (
    FuzzyDynamicStateEncoder,
    FuzzyResidualQLearningConfig,
    FuzzyResidualSafetyConfig,
    aggregate_fuzzy_q_values,
    rollout_fuzzy_residual_q_policy,
    train_fuzzy_residual_q_learning,
)


class FuzzyResidualQLearningTests(unittest.TestCase):
    def test_encoder_returns_normalized_sparse_rule_activations(self) -> None:
        encoder = FuzzyDynamicStateEncoder(
            error_scale=(1.0, 1.0),
            velocity_scale=(2.0, 2.0),
        )

        indices, weights = encoder.active_rules([0.25, -0.5], [0.0, 1.0])

        self.assertEqual(encoder.n_rules, 81)
        self.assertGreaterEqual(indices.size, 1)
        self.assertLessEqual(indices.size, 16)
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_aggregate_fuzzy_q_values_uses_rule_weights(self) -> None:
        q_value = np.array(
            [
                [0.0, 2.0],
                [4.0, 0.0],
                [9.0, 9.0],
            ],
            dtype=float,
        )

        values = aggregate_fuzzy_q_values(
            q_value,
            np.asarray([0, 1]),
            np.asarray([0.25, 0.75]),
        )

        np.testing.assert_allclose(values, [3.0, 0.5])

    def test_training_returns_rule_q_table_and_episode_traces(self) -> None:
        env_config = Arm2DOFDynamicEnvConfig(max_steps=20)
        encoder = FuzzyDynamicStateEncoder()
        learning_config = FuzzyResidualQLearningConfig(
            episodes=3,
            max_steps_per_episode=5,
            seed=5,
        )

        result = train_fuzzy_residual_q_learning(
            env_config,
            encoder=encoder,
            config=learning_config,
        )

        self.assertEqual(result.q_value.shape, (encoder.n_rules, 9))
        self.assertEqual(result.rule_policy.shape, (encoder.n_rules,))
        self.assertEqual(result.episode_returns.shape, (3,))
        self.assertEqual(result.episode_lengths.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result.q_value)))

    def test_zero_q_table_keeps_fuzzy_computed_torque_controller_stable(self) -> None:
        env_config = Arm2DOFDynamicEnvConfig(
            dt=0.01,
            max_torque=(60.0, 35.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=500,
        )
        encoder = FuzzyDynamicStateEncoder()
        learning_config = FuzzyResidualQLearningConfig(
            max_steps_per_episode=500,
            residual_acceleration_scale=(1.5, 1.5),
        )
        q_value = np.zeros((encoder.n_rules, 9), dtype=float)

        rollout = rollout_fuzzy_residual_q_policy(
            env_config,
            q_value,
            encoder,
            config=learning_config,
        )

        self.assertTrue(rollout.done)
        self.assertLessEqual(float(rollout.distance_history[-1]), env_config.target_tolerance)
        self.assertLessEqual(float(rollout.speed_history[-1]), env_config.speed_tolerance)

    def test_safety_supervisor_falls_back_to_base_action(self) -> None:
        env_config = Arm2DOFDynamicEnvConfig(
            dt=0.01,
            max_torque=(60.0, 35.0),
            max_steps=10,
        )
        encoder = FuzzyDynamicStateEncoder()
        learning_config = FuzzyResidualQLearningConfig(max_steps_per_episode=10)
        q_value = np.zeros((encoder.n_rules, 9), dtype=float)
        q_value[:, 1] = 1.0

        rollout = rollout_fuzzy_residual_q_policy(
            env_config,
            q_value,
            encoder,
            config=learning_config,
            safety_config=FuzzyResidualSafetyConfig(
                patience=1,
                min_progress=10.0,
            ),
        )

        self.assertTrue(rollout.residual_disabled)
        self.assertEqual(rollout.residual_switch_step, 1)
        self.assertGreaterEqual(len(rollout.action_indices), 2)
        self.assertEqual(rollout.action_indices[0], 1)
        self.assertTrue(all(action == 0 for action in rollout.action_indices[1:]))


if __name__ == "__main__":
    unittest.main()
