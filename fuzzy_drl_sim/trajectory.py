from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    q: np.ndarray
    q_dot: np.ndarray
    q_ddot: np.ndarray


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


def make_trajectory(name: str, center: np.ndarray, duration: float) -> JointTrajectory:
    if name == "multi_sine":
        return MultiSineTrajectory.nominal(center)
    if name == "point_to_point":
        return PointToPointTrajectory.from_center_offset(center, duration=duration * 0.85)
    raise ValueError(f"Unsupported trajectory: {name}")


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
