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
    DynamicArmStateDiscretizer,
    DynamicResidualQLearningConfig,
    residual_acceleration_actions,
    rollout_dynamic_residual_policy,
    train_dynamic_residual_q_learning,
)


class DynamicResidualQLearningTests(unittest.TestCase):
    def test_discretizer_encodes_error_and_velocity_inside_state_space(self) -> None:
        discretizer = DynamicArmStateDiscretizer(
            error_bins=(5, 7),
            velocity_bins=(3, 5),
        )

        state = discretizer.encode([0.2, -0.4], [1.0, -1.0])
        clipped_state = discretizer.encode([100.0, -100.0], [100.0, -100.0])

        self.assertGreaterEqual(state, 0)
        self.assertLess(state, discretizer.n_states)
        self.assertGreaterEqual(clipped_state, 0)
        self.assertLess(clipped_state, discretizer.n_states)
        self.assertEqual(discretizer.grid_shape, (5, 7, 3, 5))

    def test_residual_action_table_scales_each_joint(self) -> None:
        actions = residual_acceleration_actions((2.0, 3.0))

        self.assertEqual(actions.shape, (9, 2))
        np.testing.assert_allclose(actions[0], [0.0, 0.0])
        np.testing.assert_allclose(actions[-1], [2.0, 3.0])

    def test_training_returns_q_table_and_episode_traces(self) -> None:
        env_config = Arm2DOFDynamicEnvConfig(max_steps=20)
        discretizer = DynamicArmStateDiscretizer(
            error_bins=(4, 4),
            velocity_bins=(3, 3),
        )
        learning_config = DynamicResidualQLearningConfig(
            episodes=3,
            max_steps_per_episode=5,
            seed=2,
        )

        result = train_dynamic_residual_q_learning(
            env_config,
            discretizer=discretizer,
            config=learning_config,
        )

        self.assertEqual(result.q_value.shape, (discretizer.n_states, 9))
        self.assertEqual(result.policy.shape, (discretizer.n_states,))
        self.assertEqual(result.episode_returns.shape, (3,))
        self.assertEqual(result.episode_lengths.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result.q_value)))

    def test_zero_residual_policy_keeps_computed_torque_pid_stable(self) -> None:
        env_config = Arm2DOFDynamicEnvConfig(
            dt=0.01,
            max_torque=(60.0, 35.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=300,
        )
        discretizer = DynamicArmStateDiscretizer()
        learning_config = DynamicResidualQLearningConfig(
            max_steps_per_episode=300,
            residual_acceleration_scale=(2.0, 2.0),
        )
        zero_policy = np.zeros(discretizer.n_states, dtype=int)

        rollout = rollout_dynamic_residual_policy(
            env_config,
            zero_policy,
            discretizer,
            config=learning_config,
        )

        self.assertTrue(rollout.done)
        self.assertLessEqual(float(rollout.distance_history[-1]), env_config.target_tolerance)
        self.assertLessEqual(float(rollout.speed_history[-1]), env_config.speed_tolerance)


if __name__ == "__main__":
    unittest.main()
