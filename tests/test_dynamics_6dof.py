from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from controllers import FuzzyGainScheduledPIDController, PIDController
from envs import Arm6DOFDynamicEnv, Arm6DOFDynamicEnvConfig
from robot import (
    gravity_torque_6dof,
    inverse_dynamics_torque_6dof,
    inverse_kinematics_6dof,
    joint_acceleration_6dof,
    mass_matrix_6dof,
)


class Dynamics6DOFTests(unittest.TestCase):
    def test_mass_matrix_is_symmetric_positive_definite(self) -> None:
        matrix = mass_matrix_6dof([0.3, 0.4, -0.8, 0.2, -0.1, 0.15])

        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0.0))

    def test_inverse_and_forward_dynamics_are_consistent(self) -> None:
        q = np.array([0.3, 0.2, -0.6, 0.4, -0.2, 0.15])
        q_dot = np.array([0.1, 0.4, -0.2, 0.3, -0.1, 0.05])
        q_ddot = np.array([0.7, 1.2, -0.7, 0.5, -0.3, 0.2])

        torque = inverse_dynamics_torque_6dof(q, q_dot, q_ddot)
        recovered = joint_acceleration_6dof(q, q_dot, torque)

        np.testing.assert_allclose(recovered, q_ddot, atol=1e-10)

    def test_gravity_compensation_holds_static_pose(self) -> None:
        env = Arm6DOFDynamicEnv(Arm6DOFDynamicEnvConfig(dt=0.01))
        observation = env.reset(
            q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            q_dot=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        torque = gravity_torque_6dof(observation["q"])

        next_observation, reward, done, info = env.step(torque)

        np.testing.assert_allclose(
            next_observation["q"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            next_observation["q_dot"],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            atol=1e-12,
        )
        self.assertLess(reward, 0.0)
        self.assertFalse(done)
        self.assertFalse(bool(info["truncated"]))

    def test_pid_computed_torque_reaches_static_target(self) -> None:
        config = Arm6DOFDynamicEnvConfig(
            target=(1.25, 0.45, 0.60),
            dt=0.01,
            max_torque=(65.0, 95.0, 70.0, 45.0, 30.0, 25.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=1400,
        )
        env = Arm6DOFDynamicEnv(config)
        observation = env.reset(
            q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            q_dot=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        desired_q = inverse_kinematics_6dof(
            config.target,
            config.arm_config.link_lengths,
            elbow="up",
            terminal_pitch=0.0,
            joint_limits=config.arm_config.joint_limits,
        )
        pid = PIDController(
            kp=[32.0, 48.0, 38.0, 26.0, 18.0, 14.0],
            ki=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            kd=[8.0, 11.0, 9.0, 6.0, 4.5, 3.5],
            size=6,
            output_limits=(-55.0, 55.0),
        )

        done = False
        for _ in range(config.max_steps):
            desired_q_ddot = pid.compute(desired_q, observation["q"], config.dt)
            torque = inverse_dynamics_torque_6dof(
                observation["q"],
                observation["q_dot"],
                desired_q_ddot,
                config.dynamics_config,
            )
            observation, reward, done, info = env.step(torque)
            if done:
                break

        self.assertTrue(done)
        self.assertLessEqual(float(observation["distance"]), config.target_tolerance)
        self.assertLessEqual(float(observation["speed"]), config.speed_tolerance)

    def test_fuzzy_gain_pid_reaches_static_target(self) -> None:
        config = Arm6DOFDynamicEnvConfig(
            target=(1.25, 0.45, 0.60),
            dt=0.01,
            max_torque=(65.0, 95.0, 70.0, 45.0, 30.0, 25.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=1400,
        )
        env = Arm6DOFDynamicEnv(config)
        observation = env.reset(
            q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            q_dot=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        desired_q = inverse_kinematics_6dof(
            config.target,
            config.arm_config.link_lengths,
            elbow="up",
            terminal_pitch=0.0,
            joint_limits=config.arm_config.joint_limits,
        )
        controller = FuzzyGainScheduledPIDController(
            kp=[32.0, 48.0, 38.0, 26.0, 18.0, 14.0],
            ki=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            kd=[8.0, 11.0, 9.0, 6.0, 4.5, 3.5],
            size=6,
            error_scale=[0.35, 0.45, 0.55, 0.55, 0.45, 0.40],
            derivative_scale=[4.0, 5.0, 5.0, 5.0, 4.0, 4.0],
            output_limits=(-55.0, 55.0),
        )

        done = False
        for _ in range(config.max_steps):
            desired_q_ddot = controller.compute(
                desired_q,
                observation["q"],
                config.dt,
            )
            torque = inverse_dynamics_torque_6dof(
                observation["q"],
                observation["q_dot"],
                desired_q_ddot,
                config.dynamics_config,
            )
            observation, reward, done, info = env.step(torque)
            if done:
                break

        self.assertTrue(done)
        self.assertLessEqual(float(observation["distance"]), config.target_tolerance)
        self.assertLessEqual(float(observation["speed"]), config.speed_tolerance)


if __name__ == "__main__":
    unittest.main()
