from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs import Arm3DOFEnv, Arm3DOFEnvConfig, Arm3DOFDynamicEnvConfig
from rl import (
    FuzzyDynamicStateEncoder3DOF,
    FuzzyResidualQLearning3DOFConfig,
    aggregate_fuzzy_q_values_3dof,
    rollout_fuzzy_residual_q_policy_3dof,
    train_fuzzy_residual_q_learning_3dof,
)


class EnvironmentAndFuzzyResidual3DOFTests(unittest.TestCase):
    def test_environment_reports_3d_target_error_and_distance(self) -> None:
        env = Arm3DOFEnv()
        observation = env.reset(q=[0.0, 0.0, 0.0], target=[0.95, 0.55, 0.50])

        self.assertIn("error", observation)
        self.assertIn("distance", observation)
        self.assertEqual(observation["error"].shape, (3,))
        self.assertGreater(float(observation["distance"]), 0.0)

    def test_environment_step_updates_three_joint_state(self) -> None:
        env = Arm3DOFEnv(Arm3DOFEnvConfig(dt=0.1, max_joint_speed=(1.0, 1.0, 1.0)))
        observation = env.reset(q=[0.0, 0.0, 0.0])
        next_observation, reward, done, info = env.step([10.0, 0.0, 0.0])

        self.assertLess(reward, 0.0)
        self.assertFalse(done)
        self.assertFalse(bool(info["truncated"]))
        np.testing.assert_allclose(next_observation["q"], [0.1, 0.0, 0.0], atol=1e-12)
        self.assertLess(float(next_observation["distance"]), float(observation["distance"]))

    def test_encoder_returns_normalized_sparse_rule_activations(self) -> None:
        encoder = FuzzyDynamicStateEncoder3DOF(
            error_scale=(1.0, 1.0, 1.0),
            velocity_scale=(2.0, 2.0, 2.0),
        )

        indices, weights = encoder.active_rules([0.25, -0.5, 0.1], [0.0, 1.0, -0.5])

        self.assertEqual(encoder.n_rules, 729)
        self.assertGreaterEqual(indices.size, 1)
        self.assertLessEqual(indices.size, 64)
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

        values = aggregate_fuzzy_q_values_3dof(
            q_value,
            np.asarray([0, 1]),
            np.asarray([0.25, 0.75]),
        )

        np.testing.assert_allclose(values, [3.0, 0.5])

    def test_training_returns_rule_q_table_and_episode_traces(self) -> None:
        env_config = Arm3DOFDynamicEnvConfig(max_steps=20)
        encoder = FuzzyDynamicStateEncoder3DOF()
        learning_config = FuzzyResidualQLearning3DOFConfig(
            episodes=2,
            max_steps_per_episode=5,
            seed=5,
        )

        result = train_fuzzy_residual_q_learning_3dof(
            env_config,
            encoder=encoder,
            config=learning_config,
        )

        self.assertEqual(result.q_value.shape, (encoder.n_rules, 27))
        self.assertEqual(result.rule_policy.shape, (encoder.n_rules,))
        self.assertEqual(result.episode_returns.shape, (2,))
        self.assertEqual(result.episode_lengths.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.q_value)))

    def test_zero_q_table_keeps_fuzzy_computed_torque_controller_stable(self) -> None:
        env_config = Arm3DOFDynamicEnvConfig(
            dt=0.01,
            max_torque=(45.0, 60.0, 35.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=650,
        )
        encoder = FuzzyDynamicStateEncoder3DOF()
        learning_config = FuzzyResidualQLearning3DOFConfig(max_steps_per_episode=650)
        q_value = np.zeros((encoder.n_rules, 27), dtype=float)

        rollout = rollout_fuzzy_residual_q_policy_3dof(
            env_config,
            q_value,
            encoder,
            config=learning_config,
        )

        self.assertTrue(rollout.done)
        self.assertLessEqual(float(rollout.distance_history[-1]), env_config.target_tolerance)
        self.assertLessEqual(float(rollout.speed_history[-1]), env_config.speed_tolerance)


if __name__ == "__main__":
    unittest.main()
