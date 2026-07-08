from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from controllers import FuzzyVelocityController, PIDController
from envs import Arm2DOFEnv, Arm2DOFEnvConfig
from robot import inverse_kinematics


class EnvironmentAndPIDTests(unittest.TestCase):
    def test_environment_reports_target_error_and_distance(self) -> None:
        env = Arm2DOFEnv()
        observation = env.reset(q=[0.0, 0.0], target=[1.1, 0.55])

        self.assertIn("error", observation)
        self.assertIn("distance", observation)
        self.assertGreater(float(observation["distance"]), 0.0)

    def test_environment_step_updates_joint_state(self) -> None:
        env = Arm2DOFEnv(Arm2DOFEnvConfig(dt=0.1, max_joint_speed=1.0))
        observation = env.reset(q=[0.0, 0.0])
        next_observation, reward, done, info = env.step([10.0, 0.0])

        self.assertLess(reward, 0.0)
        self.assertFalse(done)
        self.assertFalse(bool(info["truncated"]))
        np.testing.assert_allclose(next_observation["q"], [0.1, 0.0], atol=1e-12)
        self.assertLess(float(next_observation["distance"]), float(observation["distance"]))

    def test_pid_output_direction(self) -> None:
        pid = PIDController(kp=2.0, kd=0.0, output_limits=(-1.0, 1.0))
        output = pid.compute([1.0, -1.0], [0.0, 0.0], dt=0.1)
        np.testing.assert_allclose(output, [1.0, -1.0], atol=1e-12)

    def test_pid_reaches_static_target(self) -> None:
        config = Arm2DOFEnvConfig(
            target=(1.1, 0.55),
            dt=0.05,
            max_joint_speed=2.0,
            target_tolerance=1e-2,
            max_steps=400,
        )
        env = Arm2DOFEnv(config)
        observation = env.reset(q=[0.0, 0.0])
        desired_q = inverse_kinematics(config.target, config.arm_config.link_lengths)
        pid = PIDController(
            kp=[4.0, 4.0],
            ki=[0.0, 0.0],
            kd=[0.15, 0.15],
            output_limits=(-config.max_joint_speed, config.max_joint_speed),
        )

        done = False
        for _ in range(config.max_steps):
            action = pid.compute(desired_q, observation["q"], config.dt)
            observation, reward, done, info = env.step(action)
            if done:
                break

        self.assertTrue(done)
        self.assertLessEqual(float(observation["distance"]), config.target_tolerance)

    def test_fuzzy_output_direction(self) -> None:
        controller = FuzzyVelocityController(
            error_scale=1.0,
            output_scale=1.0,
            output_limits=(-1.0, 1.0),
        )
        output = controller.compute([1.0, -1.0], [0.0, 0.0], dt=0.1)

        self.assertGreater(output[0], 0.0)
        self.assertLess(output[1], 0.0)

    def test_fuzzy_reaches_static_target(self) -> None:
        config = Arm2DOFEnvConfig(
            target=(1.1, 0.55),
            dt=0.05,
            max_joint_speed=2.0,
            target_tolerance=1e-2,
            max_steps=400,
        )
        env = Arm2DOFEnv(config)
        observation = env.reset(q=[0.0, 0.0])
        desired_q = inverse_kinematics(
            config.target,
            config.arm_config.link_lengths,
            elbow="up",
        )
        controller = FuzzyVelocityController(
            error_scale=[0.45, 0.75],
            derivative_scale=[4.0, 4.0],
            output_scale=config.max_joint_speed,
            output_limits=(-config.max_joint_speed, config.max_joint_speed),
        )

        done = False
        for _ in range(config.max_steps):
            action = controller.compute(desired_q, observation["q"], config.dt)
            observation, reward, done, info = env.step(action)
            if done:
                break

        self.assertTrue(done)
        self.assertLessEqual(float(observation["distance"]), config.target_tolerance)


if __name__ == "__main__":
    unittest.main()
