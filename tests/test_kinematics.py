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
    Arm2DOFConfig,
    clip_to_joint_limits,
    forward_kinematics,
    inverse_kinematics,
    is_reachable,
    jacobian,
    joint_positions,
)


class Kinematics2DOFTests(unittest.TestCase):
    def test_forward_kinematics_at_zero_angles(self) -> None:
        position = forward_kinematics([0.0, 0.0], (1.0, 0.8))
        np.testing.assert_allclose(position, [1.8, 0.0], atol=1e-12)

    def test_joint_positions_shape(self) -> None:
        positions = joint_positions([0.2, -0.4], (1.0, 0.8))
        self.assertEqual(positions.shape, (3, 2))
        np.testing.assert_allclose(positions[0], [0.0, 0.0], atol=1e-12)

    def test_inverse_kinematics_reaches_target(self) -> None:
        target = np.array([1.1, 0.55])
        q = inverse_kinematics(target, (1.0, 0.8), elbow="up")
        reached = forward_kinematics(q, (1.0, 0.8))
        np.testing.assert_allclose(reached, target, atol=1e-9)

    def test_unreachable_target_is_rejected(self) -> None:
        self.assertFalse(is_reachable([3.0, 0.0], (1.0, 0.8)))
        with self.assertRaises(ValueError):
            inverse_kinematics([3.0, 0.0], (1.0, 0.8))

    def test_jacobian_matches_finite_difference(self) -> None:
        q = np.array([0.6, -0.7])
        eps = 1e-7
        numerical = np.column_stack(
            [
                (
                    forward_kinematics(q + [eps, 0.0])
                    - forward_kinematics(q - [eps, 0.0])
                )
                / (2.0 * eps),
                (
                    forward_kinematics(q + [0.0, eps])
                    - forward_kinematics(q - [0.0, eps])
                )
                / (2.0 * eps),
            ]
        )
        np.testing.assert_allclose(jacobian(q), numerical, atol=1e-8)

    def test_joint_limit_clipping(self) -> None:
        config = Arm2DOFConfig(joint_limits=((-0.5, 0.5), (-pi, pi)))
        clipped = clip_to_joint_limits([1.2, -4.0], config.joint_limits)
        np.testing.assert_allclose(clipped, [0.5, -pi], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

