"""Kinematics for a spatial 4-DOF arm with yaw base and planar 3R chain."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from typing import Literal

import numpy as np


ArrayLike3 = tuple[float, float, float] | list[float] | np.ndarray
ArrayLike4 = tuple[float, float, float, float] | list[float] | np.ndarray
ElbowMode = Literal["up", "down"]


@dataclass(frozen=True)
class Arm4DOFConfig:
    """Geometric constants for the spatial 4-DOF model.

    The first joint is a yaw rotation around the vertical z axis. The last
    three joints form a serial 3R arm in the selected radial-z plane.
    """

    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55)
    joint_limits: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = (
        (-pi, pi),
        (-pi, pi),
        (-pi, pi),
        (-pi, pi),
    )

    def __post_init__(self) -> None:
        lengths = np.asarray(self.link_lengths, dtype=float)
        if lengths.shape != (3,) or np.any(lengths <= 0.0):
            raise ValueError("link_lengths must contain three positive values.")
        if len(self.joint_limits) != 4:
            raise ValueError("A 4-DOF arm needs exactly four joint limits.")
        for lower, upper in self.joint_limits:
            if lower >= upper:
                raise ValueError("Each joint limit must be ordered as (min, max).")


def _as_vector3(values: ArrayLike3, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return vector


def _as_vector4(values: ArrayLike4, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must contain exactly four values.")
    return vector


def _wrap_to_pi(angle: float) -> float:
    return float(atan2(sin(angle), cos(angle)))


def joint_positions_4dof(
    q: ArrayLike4,
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
) -> np.ndarray:
    """Return base, intermediate joints and end-effector as a (4, 3) array."""

    angles = _as_vector4(q, "q")
    yaw = float(angles[0])
    planar = np.cumsum(angles[1:])
    lengths = np.asarray(link_lengths, dtype=float)
    if lengths.shape != (3,):
        raise ValueError("link_lengths must contain exactly three values.")

    c0, s0 = cos(yaw), sin(yaw)
    radial = 0.0
    z_position = 0.0
    positions = [np.array([0.0, 0.0, 0.0], dtype=float)]
    for length, theta in zip(lengths, planar):
        radial += float(length) * cos(float(theta))
        z_position += float(length) * sin(float(theta))
        positions.append(
            np.array([radial * c0, radial * s0, z_position], dtype=float)
        )
    return np.vstack(positions)


def forward_kinematics_4dof(
    q: ArrayLike4,
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
) -> np.ndarray:
    """Return the 3D end-effector position for joint angles q."""

    return joint_positions_4dof(q, link_lengths)[-1]


def jacobian_4dof(
    q: ArrayLike4,
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
) -> np.ndarray:
    """Return the 3x4 geometric Jacobian of the end-effector."""

    angles = _as_vector4(q, "q")
    yaw = float(angles[0])
    planar = np.cumsum(angles[1:])
    lengths = np.asarray(link_lengths, dtype=float)
    if lengths.shape != (3,):
        raise ValueError("link_lengths must contain exactly three values.")

    c0, s0 = cos(yaw), sin(yaw)
    radial = float(np.sum(lengths * np.cos(planar)))
    matrix = np.zeros((3, 4), dtype=float)
    matrix[:, 0] = np.array([-radial * s0, radial * c0, 0.0], dtype=float)

    for joint_index in range(3):
        dradial = -float(
            np.sum(lengths[joint_index:] * np.sin(planar[joint_index:]))
        )
        dz = float(np.sum(lengths[joint_index:] * np.cos(planar[joint_index:])))
        matrix[:, joint_index + 1] = np.array(
            [dradial * c0, dradial * s0, dz],
            dtype=float,
        )
    return matrix


def workspace_radius_4dof(
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
) -> tuple[float, float]:
    """Return the minimum and maximum radial distance from the shoulder."""

    lengths = np.asarray(link_lengths, dtype=float)
    if lengths.shape != (3,):
        raise ValueError("link_lengths must contain exactly three values.")
    r_max = float(np.sum(lengths))
    longest = float(np.max(lengths))
    r_min = max(0.0, longest - (r_max - longest))
    return r_min, r_max


def is_reachable_4dof(
    target: ArrayLike3,
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
    tolerance: float = 1e-9,
) -> bool:
    """Return True if the target is inside the spherical shell workspace."""

    x, y, z = _as_vector3(target, "target")
    radius = sqrt(x * x + y * y + z * z)
    r_min, r_max = workspace_radius_4dof(link_lengths)
    return (r_min - tolerance) <= radius <= (r_max + tolerance)


def _candidate_terminal_pitches(rho: float, z: float) -> list[float]:
    line_angle = atan2(z, rho)
    candidates = [line_angle, 0.0]
    step = pi / 180.0
    for index in range(1, 181):
        candidates.append(line_angle + index * step)
        candidates.append(line_angle - index * step)
    return [_wrap_to_pi(angle) for angle in candidates]


def _within_joint_limits(
    q: np.ndarray,
    joint_limits: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    tolerance: float,
) -> bool:
    lower = np.array([limit[0] for limit in joint_limits], dtype=float)
    upper = np.array([limit[1] for limit in joint_limits], dtype=float)
    return bool(np.all(q >= lower - tolerance) and np.all(q <= upper + tolerance))


def inverse_kinematics_4dof(
    target: ArrayLike3,
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
    elbow: ElbowMode = "down",
    terminal_pitch: float | None = None,
    tolerance: float = 1e-9,
    yaw_at_axis: float = 0.0,
    joint_limits: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    | None = None,
) -> np.ndarray:
    """Return one analytical IK solution for a reachable 3D target.

    The redundant degree of freedom is resolved by choosing a terminal pitch
    for the planar 3R chain. If no pitch is supplied, nearby pitches are scanned
    and the smallest-norm valid posture is returned.
    """

    if elbow not in ("up", "down"):
        raise ValueError("elbow must be either 'up' or 'down'.")

    x, y, z = _as_vector3(target, "target")
    lengths = np.asarray(link_lengths, dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("link_lengths must contain three positive values.")
    if not is_reachable_4dof(target, tuple(lengths), tolerance=tolerance):
        raise ValueError("Target is outside the reachable workspace.")

    rho = sqrt(x * x + y * y)
    yaw = yaw_at_axis if rho <= tolerance else atan2(y, x)
    l1, l2, l3 = [float(value) for value in lengths]
    pitches = (
        [_wrap_to_pi(float(terminal_pitch))]
        if terminal_pitch is not None
        else _candidate_terminal_pitches(rho, z)
    )
    sign = 1.0 if elbow == "up" else -1.0
    best_q: np.ndarray | None = None
    best_score = float("inf")
    line_angle = atan2(z, rho)

    for pitch in pitches:
        wrist_rho = rho - l3 * cos(pitch)
        wrist_z = z - l3 * sin(pitch)
        wrist_radius2 = wrist_rho * wrist_rho + wrist_z * wrist_z
        c2 = (wrist_radius2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        if c2 < -1.0 - tolerance or c2 > 1.0 + tolerance:
            continue

        c2 = float(np.clip(c2, -1.0, 1.0))
        s2 = sign * sqrt(max(0.0, 1.0 - c2 * c2))
        q2 = atan2(s2, c2)
        q1 = atan2(wrist_z, wrist_rho) - atan2(l2 * s2, l1 + l2 * c2)
        q3 = _wrap_to_pi(pitch - q1 - q2)
        q = np.array(
            [
                _wrap_to_pi(yaw),
                _wrap_to_pi(q1),
                _wrap_to_pi(q2),
                q3,
            ],
            dtype=float,
        )
        if joint_limits is not None and not _within_joint_limits(q, joint_limits, tolerance):
            continue

        reached = forward_kinematics_4dof(q, tuple(lengths))
        if np.linalg.norm(reached - np.asarray(target, dtype=float)) > 5e-7:
            continue

        if terminal_pitch is not None:
            return q

        score = abs(_wrap_to_pi(pitch - line_angle)) + 0.02 * float(np.linalg.norm(q))
        if score < best_score:
            best_score = score
            best_q = q

    if best_q is None:
        raise ValueError("No 4-DOF inverse kinematics solution found.")
    return best_q


def clip_to_joint_limits_4dof(
    q: ArrayLike4,
    joint_limits: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> np.ndarray:
    """Clip joint angles to the configured limits."""

    angles = _as_vector4(q, "q")
    lower = np.array([limit[0] for limit in joint_limits], dtype=float)
    upper = np.array([limit[1] for limit in joint_limits], dtype=float)
    return np.clip(angles, lower, upper)
