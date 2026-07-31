from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs import Arm4DOFEnv, Arm4DOFEnvConfig, Arm4DOFDynamicEnvConfig
from rl import (
    PIDResidualQLearning4DOFConfig,
    PIDResidualStateEncoder4DOF,
    residual_acceleration_actions_4dof,
    rollout_pid_residual_q_policy_4dof,
    train_pid_residual_q_learning_4dof,
)


class PIDResidualQLearning4DOFTests(unittest.TestCase):
    def test_environment_reports_3d_target_error_and_distance(self) -> None:
        env = Arm4DOFEnv()
        observation = env.reset(q=[0.0, 0.0, 0.0, 0.0], target=[1.15, 0.45, 0.55])

        self.assertIn("error", observation)
        self.assertIn("distance", observation)
        self.assertEqual(observation["error"].shape, (3,))
        self.assertGreater(float(observation["distance"]), 0.0)

    def test_environment_step_updates_four_joint_state(self) -> None:
        env = Arm4DOFEnv(
            Arm4DOFEnvConfig(dt=0.1, max_joint_speed=(1.0, 1.0, 1.0, 1.0))
        )
        observation = env.reset(q=[0.0, 0.0, 0.0, 0.0])
        next_observation, reward, done, info = env.step([10.0, 0.0, 0.0, 0.0])

        self.assertLess(reward, 0.0)
        self.assertFalse(done)
        self.assertFalse(bool(info["truncated"]))
        np.testing.assert_allclose(
            next_observation["q"],
            [0.1, 0.0, 0.0, 0.0],
            atol=1e-12,
        )
        self.assertLess(float(next_observation["distance"]), float(observation["distance"]))

    def test_encoder_returns_compact_state_index(self) -> None:
        encoder = PIDResidualStateEncoder4DOF(
            joint_error_deadband=(0.1, 0.1, 0.1, 0.1),
            speed_bins=(0.2, 1.0),
        )

        index = encoder.encode([0.25, -0.5, 0.0, 0.1], [0.0, 0.8, -0.1, 0.0])

        self.assertEqual(encoder.n_states, 243)
        self.assertGreaterEqual(index, 0)
        self.assertLess(index, encoder.n_states)

    def test_residual_actions_are_axis_aligned(self) -> None:
        actions = residual_acceleration_actions_4dof((0.3, 0.4, 0.5, 0.6))

        self.assertEqual(actions.shape, (9, 4))
        np.testing.assert_allclose(actions[0], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(actions[1], [0.3, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(actions[-1], [0.0, 0.0, 0.0, -0.6])

    def test_training_returns_state_q_table_and_episode_traces(self) -> None:
        env_config = Arm4DOFDynamicEnvConfig(max_steps=20)
        encoder = PIDResidualStateEncoder4DOF()
        learning_config = PIDResidualQLearning4DOFConfig(
            episodes=2,
            max_steps_per_episode=5,
            seed=7,
        )

        result = train_pid_residual_q_learning_4dof(
            env_config,
            encoder=encoder,
            config=learning_config,
        )

        self.assertEqual(result.q_value.shape, (encoder.n_states, 9))
        self.assertEqual(result.state_policy.shape, (encoder.n_states,))
        self.assertEqual(result.episode_returns.shape, (2,))
        self.assertEqual(result.episode_lengths.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.q_value)))

    def test_zero_q_table_keeps_adaptive_pid_controller_stable(self) -> None:
        env_config = Arm4DOFDynamicEnvConfig(
            dt=0.01,
            max_torque=(55.0, 85.0, 60.0, 35.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=900,
        )
        encoder = PIDResidualStateEncoder4DOF()
        learning_config = PIDResidualQLearning4DOFConfig(max_steps_per_episode=900)
        q_value = np.zeros((encoder.n_states, 9), dtype=float)

        rollout = rollout_pid_residual_q_policy_4dof(
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
