from __future__ import annotations

import numpy as np

from .config import RobotConfig, SimulationConfig
from .state import ArmState


class OfflineArmEnv:
    """Small deterministic joint plant for tests when CoppeliaSim is closed."""

    def __init__(
        self,
        robot_config: RobotConfig,
        simulation_config: SimulationConfig,
        q0: np.ndarray | None = None,
        time_constant: float = 0.18,
    ) -> None:
        self.robot_config = robot_config
        self.simulation_config = simulation_config
        self.q = np.zeros(robot_config.dof, dtype=float) if q0 is None else np.asarray(q0, dtype=float)
        self.q_dot = np.zeros(robot_config.dof, dtype=float)
        self.time_constant = time_constant
        self.running = False

    def start(self) -> None:
        self.running = True

    def reset(self, q0: np.ndarray | None = None) -> ArmState:
        if q0 is not None:
            self.q = np.asarray(q0, dtype=float).copy()
        self.q_dot = np.zeros_like(self.q)
        return self.read_state()

    def read_state(self) -> ArmState:
        return ArmState(q=self.q.copy(), q_dot=self.q_dot.copy())

    def step(self, target_position: np.ndarray) -> ArmState:
        if not self.running:
            raise RuntimeError("OfflineArmEnv.start() must be called before step()")
        dt = self.simulation_config.dt
        lower = np.asarray(self.robot_config.joint_lower_limits, dtype=float)
        upper = np.asarray(self.robot_config.joint_upper_limits, dtype=float)
        max_velocity = np.asarray(self.robot_config.max_joint_velocity, dtype=float)
        target = np.clip(np.asarray(target_position, dtype=float), lower, upper)
        desired_velocity = (target - self.q) / max(self.time_constant, dt)
        self.q_dot = np.clip(desired_velocity, -max_velocity, max_velocity)
        self.q = np.clip(self.q + self.q_dot * dt, lower, upper)
        return self.read_state()

    def stop(self) -> None:
        self.running = False
