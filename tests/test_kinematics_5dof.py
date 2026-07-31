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
    Arm5DOFConfig,
    clip_to_joint_limits_5dof,
    forward_kinematics_5dof,
    inverse_kinematics_5dof,
    is_reachable_5dof,
    jacobian_5dof,
    joint_positions_5dof,
)


class Kinematics5DOFTests(unittest.TestCase):
    def test_forward_kinematics_at_zero_angles(self) -> None:
        position = forward_kinematics_5dof(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            (1.0, 0.8, 0.55, 0.35),
        )
        np.testing.assert_allclose(position, [2.7, 0.0, 0.0], atol=1e-12)

    def test_joint_positions_shape(self) -> None:
        positions = joint_positions_5dof(
            [0.2, -0.4, 0.3, -0.2, 0.1],
            (1.0, 0.8, 0.55, 0.35),
        )
        self.assertEqual(positions.shape, (5, 3))
        np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_inverse_kinematics_reaches_spatial_target(self) -> None:
        target = np.array([1.25, 0.45, 0.60])
        q = inverse_kinematics_5dof(
            target,
            (1.0, 0.8, 0.55, 0.35),
            elbow="up",
            terminal_pitch=0.0,
        )
        reached = forward_kinematics_5dof(q, (1.0, 0.8, 0.55, 0.35))
        np.testing.assert_allclose(reached, target, atol=1e-9)

    def test_inverse_kinematics_respects_terminal_fold(self) -> None:
        target = np.array([1.25, 0.45, 0.60])
        terminal_fold = pi / 6.0

        q = inverse_kinematics_5dof(
            target,
            (1.0, 0.8, 0.55, 0.35),
            elbow="up",
            terminal_pitch=0.0,
            terminal_fold=terminal_fold,
        )

        np.testing.assert_allclose(q[4], terminal_fold, atol=1e-12)
        reached = forward_kinematics_5dof(q, (1.0, 0.8, 0.55, 0.35))
        np.testing.assert_allclose(reached, target, atol=1e-9)

    def test_unreachable_target_is_rejected(self) -> None:
        self.assertFalse(is_reachable_5dof([3.2, 0.0, 0.0], (1.0, 0.8, 0.55, 0.35)))
        with self.assertRaises(ValueError):
            inverse_kinematics_5dof([3.2, 0.0, 0.0], (1.0, 0.8, 0.55, 0.35))

    def test_jacobian_matches_finite_difference(self) -> None:
        q = np.array([0.5, 0.4, -0.7, 0.3, -0.2])
        eps = 1e-7
        numerical = np.column_stack(
            [
                (
                    forward_kinematics_5dof(q + [eps, 0.0, 0.0, 0.0, 0.0])
                    - forward_kinematics_5dof(q - [eps, 0.0, 0.0, 0.0, 0.0])
                )
                / (2.0 * eps),
                (
                    forward_kinematics_5dof(q + [0.0, eps, 0.0, 0.0, 0.0])
                    - forward_kinematics_5dof(q - [0.0, eps, 0.0, 0.0, 0.0])
                )
                / (2.0 * eps),
                (
                    forward_kinematics_5dof(q + [0.0, 0.0, eps, 0.0, 0.0])
                    - forward_kinematics_5dof(q - [0.0, 0.0, eps, 0.0, 0.0])
                )
                / (2.0 * eps),
                (
                    forward_kinematics_5dof(q + [0.0, 0.0, 0.0, eps, 0.0])
                    - forward_kinematics_5dof(q - [0.0, 0.0, 0.0, eps, 0.0])
                )
                / (2.0 * eps),
                (
                    forward_kinematics_5dof(q + [0.0, 0.0, 0.0, 0.0, eps])
                    - forward_kinematics_5dof(q - [0.0, 0.0, 0.0, 0.0, eps])
                )
                / (2.0 * eps),
            ]
        )
        np.testing.assert_allclose(jacobian_5dof(q), numerical, atol=1e-8)

    def test_joint_limit_clipping(self) -> None:
        config = Arm5DOFConfig(
            joint_limits=(
                (-0.5, 0.5),
                (-pi, pi),
                (-1.0, 1.0),
                (-0.7, 0.7),
                (-0.3, 0.3),
            )
        )
        clipped = clip_to_joint_limits_5dof(
            [1.2, -4.0, 2.0, -2.0, 0.8],
            config.joint_limits,
        )
        np.testing.assert_allclose(clipped, [0.5, -pi, 1.0, -0.7, 0.3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
