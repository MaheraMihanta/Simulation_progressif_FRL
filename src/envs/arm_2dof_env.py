"""A small kinematic environment for the planar 2-DOF arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot import Arm2DOF, Arm2DOFConfig, is_reachable
from robot.kinematics import ArrayLike2


@dataclass(frozen=True)
class Arm2DOFEnvConfig:
    """Configuration for the first kinematic control environment."""

    arm_config: Arm2DOFConfig = field(default_factory=Arm2DOFConfig)
    target: tuple[float, float] = (1.1, 0.55)
    dt: float = 0.05
    max_joint_speed: float = 2.0
    target_tolerance: float = 1e-2
    action_penalty: float = 1e-3
    max_steps: int = 400

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        if self.max_joint_speed <= 0.0:
            raise ValueError("max_joint_speed must be strictly positive.")
        if self.target_tolerance <= 0.0:
            raise ValueError("target_tolerance must be strictly positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be strictly positive.")


class Arm2DOFEnv:
    """Kinematic environment with target, error, reward and step logic."""

    def __init__(self, config: Arm2DOFEnvConfig | None = None) -> None:
        self.config = config or Arm2DOFEnvConfig()
        self.arm = Arm2DOF(config=self.config.arm_config)
        self.target = np.zeros(2, dtype=float)
        self.steps = 0
        self.set_target(self.config.target)

    def set_target(self, target: ArrayLike2) -> None:
        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (2,):
            raise ValueError("target must contain exactly two values.")
        if not is_reachable(target_array, self.config.arm_config.link_lengths):
            raise ValueError("target is outside the reachable workspace.")
        self.target = target_array

    def reset(
        self,
        q: ArrayLike2 | None = None,
        target: ArrayLike2 | None = None,
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
        action: ArrayLike2,
    ) -> tuple[dict[str, np.ndarray | float], float, bool, dict[str, object]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (2,):
            raise ValueError("action must contain exactly two values.")

        clipped_action = np.clip(
            action_array,
            -self.config.max_joint_speed,
            self.config.max_joint_speed,
        )

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

