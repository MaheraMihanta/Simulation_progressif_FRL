from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fuzzy import FuzzySupervisor


@dataclass(frozen=True)
class ControllerOutput:
    target_position: np.ndarray
    correction: np.ndarray
    info: dict[str, Any]


def _vector(values: tuple[float, ...] | np.ndarray, dof: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (dof,):
        raise ValueError(f"{name} must have shape ({dof},), got {vector.shape}")
    return vector


class PIDController:
    """Position-target correction baseline around the desired trajectory."""

    def __init__(
        self,
        dof: int,
        kp: tuple[float, ...] | np.ndarray | None = None,
        ki: tuple[float, ...] | np.ndarray | None = None,
        kd: tuple[float, ...] | np.ndarray | None = None,
        correction_limit: tuple[float, ...] | np.ndarray | None = None,
        joint_lower: tuple[float, ...] | np.ndarray | None = None,
        joint_upper: tuple[float, ...] | np.ndarray | None = None,
        derivative_filter_alpha: float = 0.70,
        target_filter_alpha: float = 0.25,
    ) -> None:
        self.dof = dof
        self.kp = _vector(kp if kp is not None else (0.35,) * dof, dof, "kp")
        self.ki = _vector(ki if ki is not None else (0.0,) * dof, dof, "ki")
        self.kd = _vector(kd if kd is not None else (0.02,) * dof, dof, "kd")
        self.correction_limit = _vector(
            correction_limit if correction_limit is not None else (0.12,) * dof,
            dof,
            "correction_limit",
        )
        self.joint_lower = None if joint_lower is None else _vector(joint_lower, dof, "joint_lower")
        self.joint_upper = None if joint_upper is None else _vector(joint_upper, dof, "joint_upper")
        if not 0.0 <= derivative_filter_alpha < 1.0:
            raise ValueError("derivative_filter_alpha must be in [0, 1)")
        if not 0.0 <= target_filter_alpha < 1.0:
            raise ValueError("target_filter_alpha must be in [0, 1)")
        self.derivative_filter_alpha = derivative_filter_alpha
        self.target_filter_alpha = target_filter_alpha
        self.integral = np.zeros(dof, dtype=float)
        self.previous_error = np.zeros(dof, dtype=float)
        self.filtered_error_rate = np.zeros(dof, dtype=float)
        self.previous_target: np.ndarray | None = None

    def reset(self) -> None:
        self.integral[:] = 0.0
        self.previous_error[:] = 0.0
        self.filtered_error_rate[:] = 0.0
        self.previous_target = None

    def compute(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        q_ref_dot: np.ndarray,
        dt: float,
    ) -> ControllerOutput:
        error = q_ref - q
        error_rate = self._filtered_rate(q_ref_dot - q_dot)
        self.integral = np.clip(self.integral + error * dt, -0.35, 0.35)
        correction = self.kp * error + self.ki * self.integral + self.kd * error_rate
        correction = np.clip(correction, -self.correction_limit, self.correction_limit)
        target = self._smooth_target(self._clip_target(q_ref + correction))
        self.previous_error = error.copy()
        return ControllerOutput(
            target_position=target,
            correction=correction,
            info={
                "controller": "pid",
                "error_norm": float(np.linalg.norm(error)),
                "correction_norm": float(np.linalg.norm(correction)),
            },
        )

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        if self.joint_lower is not None:
            target = np.maximum(target, self.joint_lower)
        if self.joint_upper is not None:
            target = np.minimum(target, self.joint_upper)
        return target

    def _filtered_rate(self, raw_error_rate: np.ndarray) -> np.ndarray:
        alpha = self.derivative_filter_alpha
        self.filtered_error_rate = alpha * self.filtered_error_rate + (1.0 - alpha) * raw_error_rate
        return self.filtered_error_rate

    def _smooth_target(self, raw_target: np.ndarray) -> np.ndarray:
        if self.previous_target is None:
            self.previous_target = raw_target.copy()
            return raw_target
        alpha = self.target_filter_alpha
        target = alpha * self.previous_target + (1.0 - alpha) * raw_target
        self.previous_target = target.copy()
        return target


class ReferenceController(PIDController):
    """Directly sends the desired joint trajectory to CoppeliaSim."""

    def compute(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        q_ref_dot: np.ndarray,
        dt: float,
    ) -> ControllerOutput:
        target = self._smooth_target(self._clip_target(q_ref))
        error = q_ref - q
        return ControllerOutput(
            target_position=target,
            correction=np.zeros(self.dof, dtype=float),
            info={
                "controller": "reference",
                "error_norm": float(np.linalg.norm(error)),
                "correction_norm": 0.0,
            },
        )


class FuzzyGuidedPIDController(PIDController):
    """PID baseline whose gains and correction bounds are adapted by fuzzy rules."""

    def __init__(self, *args: Any, supervisor: FuzzySupervisor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.supervisor = supervisor or FuzzySupervisor()
        self.previous_correction = np.zeros(self.dof, dtype=float)

    def reset(self) -> None:
        super().reset()
        self.previous_correction[:] = 0.0

    def compute(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        q_ref_dot: np.ndarray,
        dt: float,
    ) -> ControllerOutput:
        error = q_ref - q
        error_rate = self._filtered_rate(q_ref_dot - q_dot)
        fuzzy = self.supervisor.evaluate(error, error_rate, self.previous_correction)

        self.integral = np.clip(self.integral + error * dt, -0.35, 0.35)
        kp = self.kp * fuzzy.kp_multiplier
        kd = self.kd * fuzzy.kd_multiplier
        limit = self.correction_limit * fuzzy.action_limit_multiplier

        correction = kp * error + self.ki * self.integral + kd * error_rate
        correction = np.clip(correction, -limit, limit)
        target = self._smooth_target(self._clip_target(q_ref + correction))

        smoothness = float(np.linalg.norm(correction - self.previous_correction))
        self.previous_error = error.copy()
        self.previous_correction = correction.copy()

        return ControllerOutput(
            target_position=target,
            correction=correction,
            info={
                "controller": "fuzzy-pid",
                "error_norm": float(np.linalg.norm(error)),
                "correction_norm": float(np.linalg.norm(correction)),
                "smoothness": smoothness,
                "fuzzy_severity": fuzzy.severity,
                "fuzzy_exploration_scale": fuzzy.exploration_scale,
                "fuzzy_expert_blend": fuzzy.expert_blend,
                "fuzzy_kp_multiplier": fuzzy.kp_multiplier,
                "fuzzy_kd_multiplier": fuzzy.kd_multiplier,
                "reward_error_weight": fuzzy.reward_error_weight,
                "reward_velocity_weight": fuzzy.reward_velocity_weight,
                "reward_effort_weight": fuzzy.reward_effort_weight,
                "reward_smoothness_weight": fuzzy.reward_smoothness_weight,
            },
        )
