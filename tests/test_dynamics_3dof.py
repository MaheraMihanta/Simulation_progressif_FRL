from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from controllers import FuzzyAccelerationController, PIDController
from envs import Arm3DOFDynamicEnv, Arm3DOFDynamicEnvConfig
from robot import (
    gravity_torque_3dof,
    inverse_dynamics_torque_3dof,
    inverse_kinematics_3dof,
    joint_acceleration_3dof,
    mass_matrix_3dof,
)


class Dynamics3DOFTests(unittest.TestCase):
    def test_mass_matrix_is_symmetric_positive_definite(self) -> None:
        matrix = mass_matrix_3dof([0.3, 0.4, -0.8])

        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0.0))

    def test_inverse_and_forward_dynamics_are_consistent(self) -> None:
        q = np.array([0.3, 0.2, -0.6])
        q_dot = np.array([0.1, 0.4, -0.2])
        q_ddot = np.array([0.7, 1.2, -0.7])

        torque = inverse_dynamics_torque_3dof(q, q_dot, q_ddot)
        recovered = joint_acceleration_3dof(q, q_dot, torque)

        np.testing.assert_allclose(recovered, q_ddot, atol=1e-10)

    def test_gravity_compensation_holds_static_pose(self) -> None:
        env = Arm3DOFDynamicEnv(Arm3DOFDynamicEnvConfig(dt=0.01))
        observation = env.reset(q=[0.0, 0.0, 0.0], q_dot=[0.0, 0.0, 0.0])
        torque = gravity_torque_3dof(observation["q"])

        next_observation, reward, done, info = env.step(torque)

        np.testing.assert_allclose(next_observation["q"], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(next_observation["q_dot"], [0.0, 0.0, 0.0], atol=1e-12)
        self.assertLess(reward, 0.0)
        self.assertFalse(done)
        self.assertFalse(bool(info["truncated"]))

    def test_pid_computed_torque_reaches_static_target(self) -> None:
        config = Arm3DOFDynamicEnvConfig(
            target=(0.95, 0.55, 0.50),
            dt=0.01,
            max_torque=(45.0, 60.0, 35.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=900,
        )
        env = Arm3DOFDynamicEnv(config)
        observation = env.reset(q=[0.0, 0.0, 0.0], q_dot=[0.0, 0.0, 0.0])
        desired_q = inverse_kinematics_3dof(
            config.target,
            config.arm_config.link_lengths,
            elbow="up",
        )
        pid = PIDController(
            kp=[25.0, 35.0, 30.0],
            ki=[0.0, 0.0, 0.0],
            kd=[6.0, 8.0, 7.0],
            size=3,
            output_limits=(-40.0, 40.0),
        )

        done = False
        for _ in range(config.max_steps):
            desired_q_ddot = pid.compute(desired_q, observation["q"], config.dt)
            torque = inverse_dynamics_torque_3dof(
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

    def test_fuzzy_computed_torque_reaches_static_target(self) -> None:
        config = Arm3DOFDynamicEnvConfig(
            target=(0.95, 0.55, 0.50),
            dt=0.01,
            max_torque=(45.0, 60.0, 35.0),
            target_tolerance=1e-2,
            speed_tolerance=8e-2,
            max_steps=900,
        )
        env = Arm3DOFDynamicEnv(config)
        observation = env.reset(q=[0.0, 0.0, 0.0], q_dot=[0.0, 0.0, 0.0])
        desired_q = inverse_kinematics_3dof(
            config.target,
            config.arm_config.link_lengths,
            elbow="up",
        )
        controller = FuzzyAccelerationController(
            error_scale=[0.35, 0.35, 0.65],
            derivative_scale=[3.5, 4.0, 4.0],
            output_scale=[22.0, 35.0, 30.0],
            size=3,
            output_limits=(-40.0, 40.0),
        )

        done = False
        for _ in range(config.max_steps):
            desired_q_ddot = controller.compute(
                desired_q,
                observation["q"],
                config.dt,
            )
            torque = inverse_dynamics_torque_3dof(
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
