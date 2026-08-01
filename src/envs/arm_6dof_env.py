"""Kinematic environment for the spatial 6-DOF arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from robot import Arm6DOF, Arm6DOFConfig, is_reachable_6dof
from robot.kinematics_6dof import ArrayLike3, ArrayLike6


Limit6 = float | Sequence[float] | np.ndarray


def _limit_vector6(value: Limit6, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        vector = np.full(6, float(vector), dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must be a scalar or a vector of size 6.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return vector


@dataclass(frozen=True)
class Arm6DOFEnvConfig:
    """Configuration for the kinematic 6-DOF control environment."""

    arm_config: Arm6DOFConfig = field(default_factory=Arm6DOFConfig)
    target: tuple[float, float, float] = (1.25, 0.45, 0.60)
    dt: float = 0.05
    max_joint_speed: Limit6 = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    target_tolerance: float = 1e-2
    action_penalty: float = 1e-3
    max_steps: int = 700

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        _limit_vector6(self.max_joint_speed, "max_joint_speed")
        if self.target_tolerance <= 0.0:
            raise ValueError("target_tolerance must be strictly positive.")
        if self.action_penalty < 0.0:
            raise ValueError("action_penalty must be non-negative.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be strictly positive.")


class Arm6DOFEnv:
    """Kinematic environment with 3D target, error, reward and step logic."""

    def __init__(self, config: Arm6DOFEnvConfig | None = None) -> None:
        self.config = config or Arm6DOFEnvConfig()
        self.arm = Arm6DOF(config=self.config.arm_config)
        self.target = np.zeros(3, dtype=float)
        self.steps = 0
        self.set_target(self.config.target)

    def set_target(self, target: ArrayLike3) -> None:
        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (3,):
            raise ValueError("target must contain exactly three values.")
        if not is_reachable_6dof(target_array, self.config.arm_config.link_lengths):
            raise ValueError("target is outside the reachable workspace.")
        self.target = target_array

    def reset(
        self,
        q: ArrayLike6 | None = None,
        target: ArrayLike3 | None = None,
    ) -> dict[str, np.ndarray | float]:
        if target is not None:
            self.set_target(target)
        self.steps = 0
        self.arm.reset(q=q)
        return self.observe()

    def observe(self) -> dict[str, np.ndarray | float]:
        observation = self.arm.observe(target=self.target)
        error = observation["error"]
        observation["distance"] = float(np.linalg.norm(error))
        observation["step"] = float(self.steps)
        return observation

    def step(
        self,
        action: ArrayLike6,
    ) -> tuple[dict[str, np.ndarray | float], float, bool, dict[str, object]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (6,):
            raise ValueError("action must contain exactly six values.")

        speed_limit = _limit_vector6(self.config.max_joint_speed, "max_joint_speed")
        clipped_action = np.clip(action_array, -speed_limit, speed_limit)

        old_q = self.arm.q.copy()
        self.arm.set_joint_angles(old_q + clipped_action * self.config.dt)
        self.arm.q_dot = (self.arm.q - old_q) / self.config.dt
        self.steps += 1

        observation = self.observe()
        distance = float(observation["distance"])
        effort = float(np.linalg.norm(clipped_action))
        reward = -distance - self.config.action_penalty * effort
        done = distance <= self.config.target_tolerance
        truncated = self.steps >= self.config.max_steps and not done

        info: dict[str, object] = {
            "distance": distance,
            "effort": effort,
            "action": clipped_action.copy(),
            "truncated": truncated,
        }
        return observation, reward, done, info
