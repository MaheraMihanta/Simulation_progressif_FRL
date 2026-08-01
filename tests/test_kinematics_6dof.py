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

from robot import (
    Arm6DOFConfig,
    clip_to_joint_limits_6dof,
    forward_kinematics_6dof,
    inverse_kinematics_6dof,
    is_reachable_6dof,
    jacobian_6dof,
    joint_positions_6dof,
)


class Kinematics6DOFTests(unittest.TestCase):
    def test_forward_kinematics_at_zero_angles(self) -> None:
        position = forward_kinematics_6dof(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            (1.0, 0.8, 0.55, 0.35, 0.25),
        )
        np.testing.assert_allclose(position, [2.95, 0.0, 0.0], atol=1e-12)

    def test_joint_positions_shape(self) -> None:
        positions = joint_positions_6dof(
            [0.2, -0.4, 0.3, -0.2, 0.1, -0.1],
            (1.0, 0.8, 0.55, 0.35, 0.25),
        )
        self.assertEqual(positions.shape, (6, 3))
        np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_inverse_kinematics_reaches_spatial_target(self) -> None:
        target = np.array([1.25, 0.45, 0.60])
        q = inverse_kinematics_6dof(
            target,
            (1.0, 0.8, 0.55, 0.35, 0.25),
            elbow="up",
            terminal_pitch=0.0,
        )
        reached = forward_kinematics_6dof(q, (1.0, 0.8, 0.55, 0.35, 0.25))
        np.testing.assert_allclose(reached, target, atol=1e-9)

    def test_inverse_kinematics_respects_distal_folds(self) -> None:
        target = np.array([1.25, 0.45, 0.60])
        wrist_fold = -pi / 9.0
        terminal_fold = pi / 6.0

        q = inverse_kinematics_6dof(
            target,
            (1.0, 0.8, 0.55, 0.35, 0.25),
            elbow="up",
            terminal_pitch=0.0,
            wrist_fold=wrist_fold,
            terminal_fold=terminal_fold,
        )

        np.testing.assert_allclose(q[4], wrist_fold, atol=1e-12)
        np.testing.assert_allclose(q[5], terminal_fold, atol=1e-12)
        reached = forward_kinematics_6dof(q, (1.0, 0.8, 0.55, 0.35, 0.25))
        np.testing.assert_allclose(reached, target, atol=1e-9)

    def test_unreachable_target_is_rejected(self) -> None:
        self.assertFalse(
            is_reachable_6dof([3.5, 0.0, 0.0], (1.0, 0.8, 0.55, 0.35, 0.25))
        )
        with self.assertRaises(ValueError):
            inverse_kinematics_6dof(
                [3.5, 0.0, 0.0],
                (1.0, 0.8, 0.55, 0.35, 0.25),
            )

    def test_jacobian_matches_finite_difference(self) -> None:
        q = np.array([0.5, 0.4, -0.7, 0.3, -0.2, 0.15])
        eps = 1e-7
        numerical = np.column_stack(
            [
                (forward_kinematics_6dof(q + np.eye(6)[index] * eps)
                 - forward_kinematics_6dof(q - np.eye(6)[index] * eps))
                / (2.0 * eps)
                for index in range(6)
            ]
        )
        np.testing.assert_allclose(jacobian_6dof(q), numerical, atol=1e-8)

    def test_joint_limit_clipping(self) -> None:
        config = Arm6DOFConfig(
            joint_limits=(
                (-0.5, 0.5),
                (-pi, pi),
                (-1.0, 1.0),
                (-0.7, 0.7),
                (-0.3, 0.3),
                (-0.2, 0.2),
            )
        )
        clipped = clip_to_joint_limits_6dof(
            [1.2, -4.0, 2.0, -2.0, 0.8, -0.4],
            config.joint_limits,
        )
        np.testing.assert_allclose(
            clipped,
            [0.5, -pi, 1.0, -0.7, 0.3, -0.2],
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
