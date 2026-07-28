"""Kinematics for a spatial 3-DOF arm with yaw base and planar 2R arm."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from typing import Literal

import numpy as np


ArrayLike3 = tuple[float, float, float] | list[float] | np.ndarray
ElbowMode = Literal["up", "down"]


@dataclass(frozen=True)
class Arm3DOFConfig:
    """Geometric constants for the spatial 3-DOF model.

    The first joint is a yaw rotation around the vertical z axis. The second
    and third joints form the same two-link arm as the planar model, but in the
    vertical radial-z plane selected by the base yaw.
    """

    link_lengths: tuple[float, float] = (1.0, 0.8)
    joint_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-pi, pi),
        (-pi, pi),
        (-pi, pi),
    )

    def __post_init__(self) -> None:
        l1, l2 = self.link_lengths
        if l1 <= 0.0 or l2 <= 0.0:
            raise ValueError("Link lengths must be strictly positive.")
        if len(self.joint_limits) != 3:
            raise ValueError("A 3-DOF arm needs exactly three joint limits.")
        for lower, upper in self.joint_limits:
            if lower >= upper:
                raise ValueError("Each joint limit must be ordered as (min, max).")


def _as_vector3(values: ArrayLike3, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return vector


def _as_target3(values: ArrayLike3, name: str = "target") -> np.ndarray:
    return _as_vector3(values, name)


def joint_positions_3dof(
    q: ArrayLike3,
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> np.ndarray:
    """Return base, elbow and end-effector positions as a (3, 3) array."""

    yaw, shoulder, elbow = _as_vector3(q, "q")
    l1, l2 = link_lengths

    c0, s0 = cos(yaw), sin(yaw)
    rho_elbow = l1 * cos(shoulder)
    z_elbow = l1 * sin(shoulder)
    rho_end = rho_elbow + l2 * cos(shoulder + elbow)
    z_end = z_elbow + l2 * sin(shoulder + elbow)

    base = np.array([0.0, 0.0, 0.0], dtype=float)
    elbow_position = np.array([rho_elbow * c0, rho_elbow * s0, z_elbow], dtype=float)
    end_effector = np.array([rho_end * c0, rho_end * s0, z_end], dtype=float)
    return np.vstack([base, elbow_position, end_effector])


def forward_kinematics_3dof(
    q: ArrayLike3,
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> np.ndarray:
    """Return the 3D end-effector position for joint angles q."""

    return joint_positions_3dof(q, link_lengths)[-1]


def jacobian_3dof(
    q: ArrayLike3,
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> np.ndarray:
    """Return the 3x3 geometric Jacobian of the end-effector."""

    yaw, shoulder, elbow = _as_vector3(q, "q")
    l1, l2 = link_lengths
    q12 = shoulder + elbow
    c0, s0 = cos(yaw), sin(yaw)
    rho = l1 * cos(shoulder) + l2 * cos(q12)
    drho_dq1 = -l1 * sin(shoulder) - l2 * sin(q12)
    drho_dq2 = -l2 * sin(q12)
    dz_dq1 = rho
    dz_dq2 = l2 * cos(q12)

    return np.array(
        [
            [-rho * s0, drho_dq1 * c0, drho_dq2 * c0],
            [rho * c0, drho_dq1 * s0, drho_dq2 * s0],
            [0.0, dz_dq1, dz_dq2],
        ],
        dtype=float,
    )


def workspace_radius_3dof(
    link_lengths: tuple[float, float] = (1.0, 0.8),
) -> tuple[float, float]:
    """Return the minimum and maximum radial distance from the shoulder."""

    l1, l2 = link_lengths
    return abs(l1 - l2), l1 + l2


def is_reachable_3dof(
    target: ArrayLike3,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    tolerance: float = 1e-9,
) -> bool:
    """Return True if the target is inside the spherical shell workspace."""

    x, y, z = _as_target3(target)
    radius = sqrt(x * x + y * y + z * z)
    r_min, r_max = workspace_radius_3dof(link_lengths)
    return (r_min - tolerance) <= radius <= (r_max + tolerance)


def inverse_kinematics_3dof(
    target: ArrayLike3,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    elbow: ElbowMode = "down",
    tolerance: float = 1e-9,
    yaw_at_axis: float = 0.0,
) -> np.ndarray:
    """Return one analytical IK solution for a reachable 3D target."""

    if elbow not in ("up", "down"):
        raise ValueError("elbow must be either 'up' or 'down'.")

    x, y, z = _as_target3(target)
    rho = sqrt(x * x + y * y)
    yaw = yaw_at_axis if rho <= tolerance else atan2(y, x)
    l1, l2 = link_lengths
    r2 = rho * rho + z * z

    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if c2 < -1.0 - tolerance or c2 > 1.0 + tolerance:
        raise ValueError("Target is outside the reachable workspace.")

    c2 = float(np.clip(c2, -1.0, 1.0))
    s2_abs = sqrt(max(0.0, 1.0 - c2 * c2))
    s2 = s2_abs if elbow == "up" else -s2_abs

    q2 = atan2(s2, c2)
    q1 = atan2(z, rho) - atan2(l2 * s2, l1 + l2 * c2)
    return np.array([yaw, q1, q2], dtype=float)


def clip_to_joint_limits_3dof(
    q: ArrayLike3,
    joint_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Clip joint angles to the configured limits."""

    angles = _as_vector3(q, "q")
    lower = np.array([limit[0] for limit in joint_limits], dtype=float)
    upper = np.array([limit[1] for limit in joint_limits], dtype=float)
    return np.clip(angles, lower, upper)
