"""Small state container for the first planar 2-DOF robotic arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import (
    Arm2DOFConfig,
    ArrayLike2,
    clip_to_joint_limits,
    forward_kinematics,
    joint_positions,
)


@dataclass
class Arm2DOF:
    """Kinematic arm model used before adding dynamics and controllers."""

    config: Arm2DOFConfig = field(default_factory=Arm2DOFConfig)
    q: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    q_dot: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def __post_init__(self) -> None:
        self.q = clip_to_joint_limits(self.q, self.config.joint_limits)
        self.q_dot = np.asarray(self.q_dot, dtype=float)
        if self.q_dot.shape != (2,):
            raise ValueError("q_dot must contain exactly two values.")

    @property
    def link_lengths(self) -> tuple[float, float]:
        return self.config.link_lengths

    def reset(
        self,
        q: ArrayLike2 | None = None,
        q_dot: ArrayLike2 | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset the state and return the first observation."""

        self.q = clip_to_joint_limits(
            np.zeros(2, dtype=float) if q is None else q,
            self.config.joint_limits,
        )
        self.q_dot = (
            np.zeros(2, dtype=float)
            if q_dot is None
            else np.asarray(q_dot, dtype=float)
        )
        if self.q_dot.shape != (2,):
            raise ValueError("q_dot must contain exactly two values.")
        return self.observe()

    def set_joint_angles(self, q: ArrayLike2) -> dict[str, np.ndarray]:
        """Set joint angles, respecting joint limits."""

        self.q = clip_to_joint_limits(q, self.config.joint_limits)
        return self.observe()

    def apply_joint_delta(self, delta_q: ArrayLike2) -> dict[str, np.ndarray]:
        """Apply a small kinematic action to the current joint angles."""

        delta = np.asarray(delta_q, dtype=float)
        if delta.shape != (2,):
            raise ValueError("delta_q must contain exactly two values.")
        self.q = clip_to_joint_limits(self.q + delta, self.config.joint_limits)
        return self.observe()

    def end_effector_position(self) -> np.ndarray:
        return forward_kinematics(self.q, self.link_lengths)

    def joint_positions(self) -> np.ndarray:
        return joint_positions(self.q, self.link_lengths)

    def observe(self, target: ArrayLike2 | None = None) -> dict[str, np.ndarray]:
        """Return the current state as arrays ready for controllers or RL."""

        end_effector = self.end_effector_position()
        observation: dict[str, np.ndarray] = {
            "q": self.q.copy(),
            "q_dot": self.q_dot.copy(),
            "end_effector": end_effector,
        }
        if target is not None:
            target_array = np.asarray(target, dtype=float)
            if target_array.shape != (2,):
                raise ValueError("target must contain exactly two values.")
            observation["target"] = target_array.copy()
            observation["error"] = target_array - end_effector
        return observation

