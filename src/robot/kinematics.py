"""Kinematics for a planar two-degree-of-freedom robotic arm."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, pi, sin, sqrt
from typing import Literal

import numpy as np


ArrayLike2 = tuple[float, float] | list[float] | np.ndarray
ElbowMode = Literal["up", "down"]


@dataclass(frozen=True)
class Arm2DOFConfig:
    """Physical and geometric constants for the first 2-DOF model."""

    link_lengths: tuple[float, float] = (1.0, 0.8)
    joint_limits: tuple[tuple[float, float], tuple[float, float]] = (
        (-pi, pi),
        (-pi, pi),
    )

    def __post_init__(self) -> None:
        l1, l2 = self.link_lengths
        if l1 <= 0.0 or l2 <= 0.0:
            raise ValueError("Link lengths must be strictly positive.")
        if len(self.joint_limits) != 2:
            raise ValueError("A 2-DOF arm needs exactly two joint limits.")
        for lower, upper in self.joint_limits:
            if lower >= upper:
                raise ValueError("Each joint limit must be ordered as (min, max).")


def _as_vector2(values: ArrayLike2, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"{name} must contain exactly two values.")
    return vector


def joint_positions(
    q: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> np.ndarray:
    """Return base, elbow and end-effector positions as a (3, 2) array."""

    q1, q2 = _as_vector2(q, "q")
    l1, l2 = link_lengths

    base = np.array([0.0, 0.0])
    elbow = np.array([l1 * cos(q1), l1 * sin(q1)])
    end_effector = elbow + np.array(
        [l2 * cos(q1 + q2), l2 * sin(q1 + q2)]
    )
    return np.vstack([base, elbow, end_effector])


def forward_kinematics(
    q: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> np.ndarray:
    """Return the end-effector position for joint angles q."""

    return joint_positions(q, link_lengths)[-1]


def jacobian(
    q: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> np.ndarray:
    """Return the 2x2 geometric Jacobian of the end-effector."""

    q1, q2 = _as_vector2(q, "q")
    l1, l2 = link_lengths
    q12 = q1 + q2

    return np.array(
        [
            [-l1 * sin(q1) - l2 * sin(q12), -l2 * sin(q12)],
            [l1 * cos(q1) + l2 * cos(q12), l2 * cos(q12)],
        ],
        dtype=float,
    )


def workspace_radius(
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> tuple[float, float]:
    """Return the minimum and maximum reachable radius."""

    l1, l2 = link_lengths
    return abs(l1 - l2), l1 + l2


def is_reachable(
    target: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    tolerance: float = 1e-9,
) -> bool:
    """Return True if the target is inside the annular workspace."""

    x, y = _as_vector2(target, "target")
    radius = sqrt(x * x + y * y)
    r_min, r_max = workspace_radius(link_lengths)
    return (r_min - tolerance) <= radius <= (r_max + tolerance)


def inverse_kinematics(
    target: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    elbow: ElbowMode = "down",
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Return one analytical inverse-kinematics solution for a reachable target."""

    if elbow not in ("up", "down"):
        raise ValueError("elbow must be either 'up' or 'down'.")

    x, y = _as_vector2(target, "target")
    l1, l2 = link_lengths
    r2 = x * x + y * y

    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if c2 < -1.0 - tolerance or c2 > 1.0 + tolerance:
        raise ValueError("Target is outside the reachable workspace.")

    c2 = float(np.clip(c2, -1.0, 1.0))
    s2_abs = sqrt(max(0.0, 1.0 - c2 * c2))
    s2 = s2_abs if elbow == "up" else -s2_abs

    q2 = atan2(s2, c2)
    q1 = atan2(y, x) - atan2(l2 * s2, l1 + l2 * c2)
    return np.array([q1, q2], dtype=float)


def clip_to_joint_limits(
    q: ArrayLike2,
    joint_limits: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Clip joint angles to the configured limits."""

    angles = _as_vector2(q, "q")
    lower = np.array([limit[0] for limit in joint_limits], dtype=float)
    upper = np.array([limit[1] for limit in joint_limits], dtype=float)
    return np.clip(angles, lower, upper)


def elbow_angle_from_target(
    target: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> float:
    """Return the absolute elbow angle used by the law of cosines."""

    x, y = _as_vector2(target, "target")
    l1, l2 = link_lengths
    r2 = x * x + y * y
    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if c2 < -1.0 or c2 > 1.0:
        raise ValueError("Target is outside the reachable workspace.")
    return acos(float(np.clip(c2, -1.0, 1.0)))

