"""State container for the spatial 5-DOF robotic arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics_5dof import (
    Arm5DOFConfig,
    ArrayLike3,
    ArrayLike5,
    clip_to_joint_limits_5dof,
    forward_kinematics_5dof,
    joint_positions_5dof,
)


@dataclass
class Arm5DOF:
    """Kinematic arm model with yaw base and four vertical-plane joints."""

    config: Arm5DOFConfig = field(default_factory=Arm5DOFConfig)
    q: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    q_dot: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))

    def __post_init__(self) -> None:
        self.q = clip_to_joint_limits_5dof(self.q, self.config.joint_limits)
        self.q_dot = np.asarray(self.q_dot, dtype=float)
        if self.q_dot.shape != (5,):
            raise ValueError("q_dot must contain exactly five values.")

    @property
    def link_lengths(self) -> tuple[float, float, float, float]:
        return self.config.link_lengths

    def reset(
        self,
        q: ArrayLike5 | None = None,
        q_dot: ArrayLike5 | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset the state and return the first observation."""

        self.q = clip_to_joint_limits_5dof(
            np.zeros(5, dtype=float) if q is None else q,
            self.config.joint_limits,
        )
        self.q_dot = (
            np.zeros(5, dtype=float)
            if q_dot is None
            else np.asarray(q_dot, dtype=float)
        )
        if self.q_dot.shape != (5,):
            raise ValueError("q_dot must contain exactly five values.")
        return self.observe()

    def set_joint_angles(self, q: ArrayLike5) -> dict[str, np.ndarray]:
        """Set joint angles, respecting joint limits."""

        self.q = clip_to_joint_limits_5dof(q, self.config.joint_limits)
        return self.observe()

    def apply_joint_delta(self, delta_q: ArrayLike5) -> dict[str, np.ndarray]:
        """Apply a small kinematic action to the current joint angles."""

        delta = np.asarray(delta_q, dtype=float)
        if delta.shape != (5,):
            raise ValueError("delta_q must contain exactly five values.")
        self.q = clip_to_joint_limits_5dof(self.q + delta, self.config.joint_limits)
        return self.observe()

    def end_effector_position(self) -> np.ndarray:
        return forward_kinematics_5dof(self.q, self.link_lengths)

    def joint_positions(self) -> np.ndarray:
        return joint_positions_5dof(self.q, self.link_lengths)

    def observe(self, target: ArrayLike3 | None = None) -> dict[str, np.ndarray]:
        """Return the current state as arrays ready for controllers or RL."""

        end_effector = self.end_effector_position()
        observation: dict[str, np.ndarray] = {
            "q": self.q.copy(),
            "q_dot": self.q_dot.copy(),
            "end_effector": end_effector,
        }
        if target is not None:
            target_array = np.asarray(target, dtype=float)
            if target_array.shape != (3,):
                raise ValueError("target must contain exactly three values.")
            observation["target"] = target_array.copy()
            observation["error"] = target_array - end_effector
        return observation
