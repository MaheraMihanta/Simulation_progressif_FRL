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
    discounted_return,
    evaluate_policy,
    rollout_policy,
    value_iteration,
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


class RLBellmanTests(unittest.TestCase):
    def test_mdp_defines_discrete_states_actions_and_terminal_state(self) -> None:
        mdp = make_small_mdp()
        target_state = mdp.nearest_state([0.0, 2.0 * pi / 5.0])

        self.assertEqual(mdp.n_states, 121)
        self.assertEqual(mdp.n_actions, 9)
        self.assertTrue(mdp.is_terminal(target_state))

    def test_transition_updates_state_and_returns_reward(self) -> None:
        mdp = make_small_mdp()
        start_state = mdp.nearest_state([0.0, 0.0])
        action = mdp.action_names.index("q2+")
        next_state, reward, done = mdp.step(start_state, action)

        self.assertNotEqual(next_state, start_state)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(done)

    def test_discounted_return_matches_definition(self) -> None:
        value = discounted_return([1.0, 1.0, 1.0], gamma=0.5)

        self.assertAlmostEqual(value, 1.75)

    def test_policy_evaluation_returns_state_values(self) -> None:
        mdp = make_small_mdp()
        random_policy = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
        result = evaluate_policy(mdp, random_policy, theta=1e-6)

        self.assertEqual(result.value.shape, (mdp.n_states,))
        self.assertTrue(np.all(np.isfinite(result.value)))
        self.assertGreater(result.iterations, 0)

    def test_value_iteration_policy_reaches_target(self) -> None:
        mdp = make_small_mdp()
        start_state = mdp.nearest_state([0.0, 0.0])
        result = value_iteration(mdp, theta=1e-8)
        rollout = rollout_policy(mdp, result.policy, start_state, max_steps=20)

        self.assertTrue(rollout.done)
        self.assertLessEqual(len(rollout.actions), 3)
        self.assertAlmostEqual(
            discounted_return(rollout.rewards, mdp.config.gamma),
            result.value[start_state],
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
