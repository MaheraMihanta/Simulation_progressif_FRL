from __future__ import annotations

from math import pi
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl import (
    DiscreteArm2DOFMDP,
    DiscreteArm2DOFMDPConfig,
    QLearningConfig,
    epsilon_at_episode,
    rollout_policy,
    train_q_learning,
)
from robot import forward_kinematics


def make_small_mdp() -> DiscreteArm2DOFMDP:
    target_q = np.array([0.0, 2.0 * pi / 5.0])
    target = tuple(forward_kinematics(target_q))
    return DiscreteArm2DOFMDP(
        DiscreteArm2DOFMDPConfig(
            target=target,
            bins_per_joint=(11, 11),
            target_tolerance=1e-8,
            gamma=0.9,
            goal_reward=5.0,
        )
    )


class QLearningTests(unittest.TestCase):
    def test_epsilon_schedule_decays_to_floor(self) -> None:
        config = QLearningConfig(
            epsilon_start=1.0,
            epsilon_end=0.2,
            epsilon_decay=0.5,
        )

        self.assertAlmostEqual(epsilon_at_episode(config, 0), 1.0)
        self.assertAlmostEqual(epsilon_at_episode(config, 1), 0.5)
        self.assertAlmostEqual(epsilon_at_episode(config, 4), 0.2)

    def test_q_learning_returns_table_and_training_traces(self) -> None:
        mdp = make_small_mdp()
        start_state = mdp.nearest_state([0.0, 0.0])
        result = train_q_learning(
            mdp,
            start_state=start_state,
            config=QLearningConfig(
                episodes=50,
                max_steps_per_episode=20,
                alpha=0.6,
                epsilon_decay=0.98,
                seed=3,
            ),
        )

        self.assertEqual(result.q_value.shape, (mdp.n_states, mdp.n_actions))
        self.assertEqual(result.policy.shape, (mdp.n_states,))
        self.assertEqual(result.episode_returns.shape, (50,))
        self.assertEqual(result.episode_lengths.shape, (50,))
        self.assertEqual(result.episode_success.shape, (50,))
        self.assertTrue(np.all(np.isfinite(result.q_value)))

    def test_q_learning_learns_policy_reaching_target(self) -> None:
        mdp = make_small_mdp()
        start_state = mdp.nearest_state([0.0, 0.0])
        result = train_q_learning(
            mdp,
            start_state=start_state,
            config=QLearningConfig(
                episodes=300,
                max_steps_per_episode=30,
                alpha=0.6,
                gamma=0.9,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.985,
                seed=3,
            ),
        )

        rollout = rollout_policy(mdp, result.policy, start_state, max_steps=20)

        self.assertTrue(rollout.done)
        self.assertLessEqual(len(rollout.actions), 3)
        self.assertGreaterEqual(float(np.mean(result.episode_success[-50:])), 0.8)


if __name__ == "__main__":
    unittest.main()
