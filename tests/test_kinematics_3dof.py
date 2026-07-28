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
    Arm3DOFConfig,
    clip_to_joint_limits_3dof,
    forward_kinematics_3dof,
    inverse_kinematics_3dof,
    is_reachable_3dof,
    jacobian_3dof,
    joint_positions_3dof,
)


class Kinematics3DOFTests(unittest.TestCase):
    def test_forward_kinematics_at_zero_angles(self) -> None:
        position = forward_kinematics_3dof([0.0, 0.0, 0.0], (1.0, 0.8))
        np.testing.assert_allclose(position, [1.8, 0.0, 0.0], atol=1e-12)

    def test_joint_positions_shape(self) -> None:
        positions = joint_positions_3dof([0.2, -0.4, 0.3], (1.0, 0.8))
        self.assertEqual(positions.shape, (3, 3))
        np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_inverse_kinematics_reaches_spatial_target(self) -> None:
        target = np.array([0.95, 0.55, 0.50])
        q = inverse_kinematics_3dof(target, (1.0, 0.8), elbow="up")
        reached = forward_kinematics_3dof(q, (1.0, 0.8))
        np.testing.assert_allclose(reached, target, atol=1e-9)

    def test_unreachable_target_is_rejected(self) -> None:
        self.assertFalse(is_reachable_3dof([3.0, 0.0, 0.0], (1.0, 0.8)))
        with self.assertRaises(ValueError):
            inverse_kinematics_3dof([3.0, 0.0, 0.0], (1.0, 0.8))

    def test_jacobian_matches_finite_difference(self) -> None:
        q = np.array([0.5, 0.4, -0.7])
        eps = 1e-7
        numerical = np.column_stack(
            [
                (
                    forward_kinematics_3dof(q + [eps, 0.0, 0.0])
                    - forward_kinematics_3dof(q - [eps, 0.0, 0.0])
                )
                / (2.0 * eps),
                (
                    forward_kinematics_3dof(q + [0.0, eps, 0.0])
                    - forward_kinematics_3dof(q - [0.0, eps, 0.0])
                )
                / (2.0 * eps),
                (
                    forward_kinematics_3dof(q + [0.0, 0.0, eps])
                    - forward_kinematics_3dof(q - [0.0, 0.0, eps])
                )
                / (2.0 * eps),
            ]
        )
        np.testing.assert_allclose(jacobian_3dof(q), numerical, atol=1e-8)

    def test_joint_limit_clipping(self) -> None:
        config = Arm3DOFConfig(
            joint_limits=((-0.5, 0.5), (-pi, pi), (-1.0, 1.0))
        )
        clipped = clip_to_joint_limits_3dof([1.2, -4.0, 2.0], config.joint_limits)
        np.testing.assert_allclose(clipped, [0.5, -pi, 1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
