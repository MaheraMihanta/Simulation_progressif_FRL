from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot import forward_kinematics_6dof, inverse_kinematics_6dof, jacobian_6dof


SUPPORTED_TRAJECTORIES: tuple[str, ...] = (
    "multi_sine",
    "point_to_point",
    "cartesian_loop",
    "cartesian_point_to_point",
)


@dataclass(frozen=True)
class TrajectorySample:
    q: np.ndarray
    q_dot: np.ndarray
    q_ddot: np.ndarray
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None


class JointTrajectory(Protocol):
    def sample(self, t: float) -> TrajectorySample:
        ...


def _as_vector(values: np.ndarray | tuple[float, ...], dof: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (dof,):
        raise ValueError(f"{name} must have shape ({dof},), got {vector.shape}")
    return vector


@dataclass
class MultiSineTrajectory:
    """Smooth joint trajectory used for nominal tracking and robustness tests."""

    start: np.ndarray
    amplitude: np.ndarray
    frequency: np.ndarray
    phase: np.ndarray
    smooth_start: float = 1.0

    @classmethod
    def nominal(cls, start: np.ndarray) -> "MultiSineTrajectory":
        dof = start.size
        base_amplitude = np.array([0.22, 0.14, 0.14, 0.18, 0.12, 0.16], dtype=float)
        base_frequency = np.array([0.08, 0.11, 0.07, 0.10, 0.13, 0.09], dtype=float)
        base_phase = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5], dtype=float)
        return cls(
            start=np.asarray(start, dtype=float),
            amplitude=base_amplitude[:dof],
            frequency=base_frequency[:dof],
            phase=base_phase[:dof],
        )

    def __post_init__(self) -> None:
        dof = np.asarray(self.start).size
        self.start = _as_vector(self.start, dof, "start")
        self.amplitude = _as_vector(self.amplitude, dof, "amplitude")
        self.frequency = _as_vector(self.frequency, dof, "frequency")
        self.phase = _as_vector(self.phase, dof, "phase")
        if self.smooth_start < 0.0:
            raise ValueError("smooth_start must be non-negative")

    def sample(self, t: float) -> TrajectorySample:
        omega = 2.0 * math.pi * self.frequency
        angle = omega * t + self.phase
        initial_angle = self.phase
        offset = self.amplitude * (np.sin(angle) - np.sin(initial_angle))
        offset_dot = self.amplitude * omega * np.cos(angle)
        offset_ddot = -self.amplitude * omega**2 * np.sin(angle)

        blend, blend_dot, blend_ddot = _smooth_blend(t, self.smooth_start)
        q = self.start + blend * offset
        q_dot = blend_dot * offset + blend * offset_dot
        q_ddot = blend_ddot * offset + 2.0 * blend_dot * offset_dot + blend * offset_ddot
        return TrajectorySample(q=q, q_dot=q_dot, q_ddot=q_ddot)


@dataclass
class PointToPointTrajectory:
    """Quintic joint-space motion from start to goal."""

    start: np.ndarray
    goal: np.ndarray
    duration: float

    @classmethod
    def from_center_offset(
        cls,
        center: np.ndarray,
        offset: np.ndarray | None = None,
        duration: float = 8.0,
    ) -> "PointToPointTrajectory":
        center = np.asarray(center, dtype=float)
        if offset is None:
            offset = np.array([0.35, -0.18, 0.20, -0.22, 0.16, -0.14], dtype=float)
        offset = np.asarray(offset, dtype=float)[: center.size]
        return cls(start=center, goal=center + offset, duration=duration)

    def __post_init__(self) -> None:
        dof = np.asarray(self.start).size
        self.start = _as_vector(self.start, dof, "start")
        self.goal = _as_vector(self.goal, dof, "goal")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")

    def sample(self, t: float) -> TrajectorySample:
        u = float(np.clip(t / self.duration, 0.0, 1.0))
        delta = self.goal - self.start
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        s_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / self.duration
        s_ddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / self.duration**2
        return TrajectorySample(
            q=self.start + s * delta,
            q_dot=s_dot * delta,
            q_ddot=s_ddot * delta,
        )


@dataclass
class CartesianLoopTrajectory:
    """Smooth 3D end-effector loop converted to a 6-DOF joint reference by IK."""

    start_q: np.ndarray
    center: np.ndarray
    amplitude: np.ndarray
    frequency: np.ndarray
    phase: np.ndarray
    duration: float
    smooth_start: float = 1.0
    link_lengths: tuple[float, float, float, float, float] = (
        1.0,
        0.8,
        0.55,
        0.35,
        0.25,
    )
    joint_limits: tuple[tuple[float, float], ...] = ((-math.pi, math.pi),) * 6

    @classmethod
    def nominal(cls, start_q: np.ndarray, duration: float) -> "CartesianLoopTrajectory":
        start_q = np.asarray(start_q, dtype=float)
        return cls(
            start_q=start_q,
            center=_nominal_cartesian_center(start_q),
            amplitude=np.array([0.06, 0.05, 0.05], dtype=float),
            frequency=np.array([0.10, 0.07, 0.13], dtype=float),
            phase=np.array([0.0, 1.2, 2.1], dtype=float),
            duration=duration,
        )

    def __post_init__(self) -> None:
        self.start_q = _as_vector(self.start_q, 6, "start_q")
        self.center = _as_cartesian_vector(self.center, "center")
        self.amplitude = _as_cartesian_vector(self.amplitude, "amplitude")
        self.frequency = _as_cartesian_vector(self.frequency, "frequency")
        self.phase = _as_cartesian_vector(self.phase, "phase")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.smooth_start < 0.0:
            raise ValueError("smooth_start must be non-negative")

    def sample(self, t: float) -> TrajectorySample:
        omega = 2.0 * math.pi * self.frequency
        angle = omega * t + self.phase
        offset = self.amplitude * (np.sin(angle) - np.sin(self.phase))
        offset_dot = self.amplitude * omega * np.cos(angle)
        offset_ddot = -self.amplitude * omega**2 * np.sin(angle)

        start_position = forward_kinematics_6dof(self.start_q, self.link_lengths)
        loop_position = self.center + offset
        delta = loop_position - start_position
        blend, blend_dot, blend_ddot = _smooth_blend(t, min(self.smooth_start, self.duration))
        position = start_position + blend * delta
        velocity = blend_dot * delta + blend * offset_dot
        acceleration = blend_ddot * delta + 2.0 * blend_dot * offset_dot + blend * offset_ddot
        q = _cartesian_ik(position, self.link_lengths, self.joint_limits)
        q_dot = _cartesian_velocity_to_joint_velocity(q, velocity, self.link_lengths)
        return TrajectorySample(
            q=q,
            q_dot=q_dot,
            q_ddot=np.zeros(6, dtype=float),
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )


@dataclass
class CartesianPointToPointTrajectory:
    """Quintic 3D end-effector transfer converted to a 6-DOF joint reference."""

    start_q: np.ndarray
    goal_position: np.ndarray
    duration: float
    link_lengths: tuple[float, float, float, float, float] = (
        1.0,
        0.8,
        0.55,
        0.35,
        0.25,
    )
    joint_limits: tuple[tuple[float, float], ...] = ((-math.pi, math.pi),) * 6

    @classmethod
    def nominal(
        cls,
        start_q: np.ndarray,
        duration: float,
    ) -> "CartesianPointToPointTrajectory":
        start_q = np.asarray(start_q, dtype=float)
        return cls(
            start_q=start_q,
            goal_position=_nominal_cartesian_goal(start_q),
            duration=duration * 0.85,
        )

    def __post_init__(self) -> None:
        self.start_q = _as_vector(self.start_q, 6, "start_q")
        self.goal_position = _as_cartesian_vector(self.goal_position, "goal_position")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")

    def sample(self, t: float) -> TrajectorySample:
        u = float(np.clip(t / self.duration, 0.0, 1.0))
        start_position = forward_kinematics_6dof(self.start_q, self.link_lengths)
        delta = self.goal_position - start_position
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        s_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / self.duration
        s_ddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / self.duration**2
        position = start_position + s * delta
        velocity = s_dot * delta
        acceleration = s_ddot * delta
        q = _cartesian_ik(position, self.link_lengths, self.joint_limits)
        q_dot = _cartesian_velocity_to_joint_velocity(q, velocity, self.link_lengths)
        return TrajectorySample(
            q=q,
            q_dot=q_dot,
            q_ddot=np.zeros(6, dtype=float),
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )


def make_trajectory(name: str, center: np.ndarray, duration: float) -> JointTrajectory:
    if name == "multi_sine":
        return MultiSineTrajectory.nominal(center)
    if name == "point_to_point":
        return PointToPointTrajectory.from_center_offset(center, duration=duration * 0.85)
    if name == "cartesian_loop":
        return CartesianLoopTrajectory.nominal(center, duration=duration)
    if name == "cartesian_point_to_point":
        return CartesianPointToPointTrajectory.nominal(center, duration=duration)
    raise ValueError(f"Unsupported trajectory: {name}")


def _as_cartesian_vector(values: np.ndarray | tuple[float, ...], name: str) -> np.ndarray:
    return _as_vector(values, 3, name)


def _cartesian_ik(
    position: np.ndarray,
    link_lengths: tuple[float, float, float, float, float],
    joint_limits: tuple[tuple[float, float], ...],
) -> np.ndarray:
    return inverse_kinematics_6dof(
        position,
        link_lengths=link_lengths,
        joint_limits=joint_limits,  # type: ignore[arg-type]
        elbow="down",
    )


def _cartesian_velocity_to_joint_velocity(
    q: np.ndarray,
    velocity: np.ndarray,
    link_lengths: tuple[float, float, float, float, float],
) -> np.ndarray:
    jacobian = jacobian_6dof(q, link_lengths)
    return np.linalg.pinv(jacobian, rcond=1e-3) @ velocity


def _nominal_cartesian_center(start_q: np.ndarray) -> np.ndarray:
    start_position = forward_kinematics_6dof(start_q)
    inward, tangent = _cartesian_inward_tangent(start_position)
    return start_position + 0.38 * inward + 0.12 * tangent + np.array([0.0, 0.0, 0.22])


def _nominal_cartesian_goal(start_q: np.ndarray) -> np.ndarray:
    start_position = forward_kinematics_6dof(start_q)
    inward, tangent = _cartesian_inward_tangent(start_position)
    return start_position + 0.50 * inward + 0.16 * tangent + np.array([0.0, 0.0, 0.28])


def _cartesian_inward_tangent(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(position[:2], dtype=float)
    radius = float(np.linalg.norm(xy))
    if radius <= 1e-9:
        radial = np.array([1.0, 0.0], dtype=float)
    else:
        radial = xy / radius
    inward = np.array([-radial[0], -radial[1], 0.0], dtype=float)
    tangent = np.array([-radial[1], radial[0], 0.0], dtype=float)
    return inward, tangent


def _smooth_blend(t: float, duration: float) -> tuple[float, float, float]:
    if duration == 0.0 or t >= duration:
        return 1.0, 0.0, 0.0
    if t <= 0.0:
        return 0.0, 0.0, 0.0

    u = t / duration
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    blend_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    blend_ddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration**2
    return blend, blend_dot, blend_ddot
