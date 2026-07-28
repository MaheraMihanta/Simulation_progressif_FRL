"""Torque-driven dynamic environment for the spatial 3-DOF arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from robot import Arm3DOF, Arm3DOFConfig, is_reachable_3dof
from robot.dynamics_3dof import Arm3DOFDynamicsConfig, joint_acceleration_3dof
from robot.kinematics_3dof import ArrayLike3, clip_to_joint_limits_3dof


TorqueLimit3 = float | Sequence[float] | np.ndarray


def _limit_vector3(value: TorqueLimit3, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        vector = np.full(3, float(vector), dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a scalar or a vector of size 3.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return vector


@dataclass(frozen=True)
class Arm3DOFDynamicEnvConfig:
    """Configuration for torque-based dynamic 3-DOF control experiments."""

    dynamics_config: Arm3DOFDynamicsConfig = field(
        default_factory=Arm3DOFDynamicsConfig
    )
    target: tuple[float, float, float] = (0.95, 0.55, 0.50)
    dt: float = 0.01
    max_torque: TorqueLimit3 = (25.0, 60.0, 35.0)
    max_joint_speed: TorqueLimit3 = (6.0, 8.0, 8.0)
    target_tolerance: float = 1e-2
    speed_tolerance: float = 8e-2
    action_penalty: float = 1e-4
    speed_penalty: float = 1e-3
    max_steps: int = 3000

    @property
    def arm_config(self) -> Arm3DOFConfig:
        return self.dynamics_config.arm_config

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        _limit_vector3(self.max_torque, "max_torque")
        _limit_vector3(self.max_joint_speed, "max_joint_speed")
        if self.target_tolerance <= 0.0:
            raise ValueError("target_tolerance must be strictly positive.")
        if self.speed_tolerance <= 0.0:
            raise ValueError("speed_tolerance must be strictly positive.")
        if self.action_penalty < 0.0:
            raise ValueError("action_penalty must be non-negative.")
        if self.speed_penalty < 0.0:
            raise ValueError("speed_penalty must be non-negative.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be strictly positive.")


class Arm3DOFDynamicEnv:
    """Environment where actions are 3 joint torques and physics drives motion."""

    def __init__(self, config: Arm3DOFDynamicEnvConfig | None = None) -> None:
        self.config = config or Arm3DOFDynamicEnvConfig()
        self.arm = Arm3DOF(config=self.config.arm_config)
        self.target = np.zeros(3, dtype=float)
        self.steps = 0
        self.last_q_ddot = np.zeros(3, dtype=float)
        self.set_target(self.config.target)

    def set_target(self, target: ArrayLike3) -> None:
        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (3,):
            raise ValueError("target must contain exactly three values.")
        if not is_reachable_3dof(target_array, self.config.arm_config.link_lengths):
            raise ValueError("target is outside the reachable workspace.")
        self.target = target_array

    def reset(
        self,
        q: ArrayLike3 | None = None,
        q_dot: ArrayLike3 | None = None,
        target: ArrayLike3 | None = None,
    ) -> dict[str, np.ndarray | float]:
        if target is not None:
            self.set_target(target)
        self.steps = 0
        self.last_q_ddot = np.zeros(3, dtype=float)
        self.arm.reset(q=q, q_dot=q_dot)
        return self.observe()

    def observe(self) -> dict[str, np.ndarray | float]:
        observation = self.arm.observe(target=self.target)
        error = observation["error"]
        observation["distance"] = float(np.linalg.norm(error))
        observation["speed"] = float(np.linalg.norm(observation["q_dot"]))
        observation["q_ddot"] = self.last_q_ddot.copy()
        observation["step"] = float(self.steps)
        return observation

    def step(
        self,
        action: ArrayLike3,
        external_torque: ArrayLike3 | None = None,
    ) -> tuple[dict[str, np.ndarray | float], float, bool, dict[str, object]]:
        torque_command = np.asarray(action, dtype=float)
        if torque_command.shape != (3,):
            raise ValueError("action must contain exactly three values.")

        torque_limit = _limit_vector3(self.config.max_torque, "max_torque")
        speed_limit = _limit_vector3(self.config.max_joint_speed, "max_joint_speed")
        clipped_torque = np.clip(torque_command, -torque_limit, torque_limit)

        old_q = self.arm.q.copy()
        q_ddot = joint_acceleration_3dof(
            old_q,
            self.arm.q_dot,
            clipped_torque,
            self.config.dynamics_config,
            external_torque=external_torque,
        )

        q_dot_next = self.arm.q_dot + q_ddot * self.config.dt
        q_dot_next = np.clip(q_dot_next, -speed_limit, speed_limit)
        q_next_unclipped = old_q + q_dot_next * self.config.dt
        q_next = clip_to_joint_limits_3dof(
            q_next_unclipped,
            self.config.arm_config.joint_limits,
        )

        limited = ~np.isclose(q_next, q_next_unclipped, atol=1e-12, rtol=0.0)
        if np.any(limited):
            q_dot_next = (q_next - old_q) / self.config.dt
            q_dot_next[limited] = 0.0

        self.arm.q = q_next
        self.arm.q_dot = q_dot_next
        self.last_q_ddot = q_ddot
        self.steps += 1

        observation = self.observe()
        distance = float(observation["distance"])
        speed = float(observation["speed"])
        effort = float(np.linalg.norm(clipped_torque))
        reward = (
            -distance
            - self.config.action_penalty * effort
            - self.config.speed_penalty * speed
        )
        done = (
            distance <= self.config.target_tolerance
            and speed <= self.config.speed_tolerance
        )
        truncated = self.steps >= self.config.max_steps and not done

        info: dict[str, object] = {
            "distance": distance,
            "speed": speed,
            "effort": effort,
            "action": clipped_torque.copy(),
            "q_ddot": q_ddot.copy(),
            "external_torque": (
                None if external_torque is None else np.asarray(external_torque, dtype=float)
            ),
            "truncated": truncated,
        }
        return observation, reward, done, info
