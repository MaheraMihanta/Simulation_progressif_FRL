from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import RobotConfig, SimulationConfig
from .fuzzy import FuzzyOutput, FuzzySupervisor
from .state import ArmState
from .trajectory import JointTrajectory, make_trajectory


class ArmBackend(Protocol):
    def start(self) -> None:
        ...

    def reset(self, q0: np.ndarray | None = None) -> ArmState:
        ...

    def read_state(self) -> ArmState:
        ...

    def step(self, target_position: np.ndarray) -> ArmState:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class RLStepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float | int | bool]


class FuzzyGuidedTrackingTask:
    """Gym-like tracking task ready to wrap with SAC/TD3 later.

    Action convention: a policy outputs one normalized correction per joint in
    [-1, 1]. The task scales it by `RobotConfig.max_position_correction`, adds it
    around q_ref, sends that joint-position target to the backend, then computes
    the fuzzy-shaped reward.
    """

    def __init__(
        self,
        backend: ArmBackend,
        robot_config: RobotConfig,
        simulation_config: SimulationConfig,
        trajectory_name: str = "multi_sine",
        supervisor: FuzzySupervisor | None = None,
    ) -> None:
        self.backend = backend
        self.robot_config = robot_config
        self.simulation_config = simulation_config
        self.trajectory_name = trajectory_name
        self.supervisor = supervisor or FuzzySupervisor()
        self.trajectory: JointTrajectory | None = None
        self.t = 0.0
        self.previous_action = np.zeros(robot_config.dof, dtype=float)

    @property
    def action_size(self) -> int:
        return self.robot_config.dof

    @property
    def observation_size(self) -> int:
        return self.robot_config.dof * 6 + 3

    def reset(self) -> np.ndarray:
        self.backend.start()
        state = self.backend.reset()
        self.trajectory = make_trajectory(self.trajectory_name, state.q, self.simulation_config.duration)
        self.t = 0.0
        self.previous_action[:] = 0.0
        reference = self.trajectory.sample(self.t)
        fuzzy = self.supervisor.evaluate(reference.q - state.q, reference.q_dot - state.q_dot, self.previous_action)
        return self._make_observation(state, reference.q, reference.q_dot, fuzzy)

    def step(self, normalized_action: np.ndarray) -> RLStepResult:
        if self.trajectory is None:
            raise RuntimeError("FuzzyGuidedTrackingTask.reset() must be called before step()")

        action = np.clip(np.asarray(normalized_action, dtype=float), -1.0, 1.0)
        if action.shape != (self.robot_config.dof,):
            raise ValueError(f"Action must have shape ({self.robot_config.dof},), got {action.shape}")

        reference = self.trajectory.sample(self.t)
        correction_limit = np.asarray(self.robot_config.max_position_correction, dtype=float)
        correction = action * correction_limit
        raw_target = reference.q + correction
        target = self._clip_target(raw_target)
        next_state = self.backend.step(target)

        next_t = self.t + self.simulation_config.dt
        next_reference = self.trajectory.sample(next_t)
        error = next_reference.q - next_state.q
        error_rate = next_reference.q_dot - next_state.q_dot
        fuzzy = self.supervisor.evaluate(error, error_rate, correction)
        reward = self._reward(error, error_rate, correction, raw_target, fuzzy)
        violation_count = self._constraint_violations(next_state.q)

        self.t = next_t
        truncated = self.t >= self.simulation_config.duration
        self.previous_action = correction.copy()
        observation = self._make_observation(next_state, next_reference.q, next_reference.q_dot, fuzzy)
        return RLStepResult(
            observation=observation,
            reward=reward,
            terminated=False,
            truncated=truncated,
            info={
                "time": self.t,
                "error_norm": float(np.linalg.norm(error)),
                "action_norm": float(np.linalg.norm(correction)),
                "fuzzy_severity": fuzzy.severity,
                "fuzzy_exploration_scale": fuzzy.exploration_scale,
                "constraint_violations": violation_count,
                "has_constraint_violation": violation_count > 0,
            },
        )

    def close(self) -> None:
        self.backend.stop()

    def _make_observation(
        self,
        state: ArmState,
        q_ref: np.ndarray,
        q_ref_dot: np.ndarray,
        fuzzy: FuzzyOutput,
    ) -> np.ndarray:
        error = q_ref - state.q
        error_rate = q_ref_dot - state.q_dot
        indicators = np.array(
            [
                fuzzy.severity,
                fuzzy.exploration_scale,
                np.linalg.norm(self.previous_action),
            ],
            dtype=float,
        )
        return np.concatenate([state.q, state.q_dot, q_ref, q_ref_dot, error, error_rate, indicators])

    def _reward(
        self,
        error: np.ndarray,
        error_rate: np.ndarray,
        correction: np.ndarray,
        raw_target: np.ndarray,
        fuzzy: FuzzyOutput,
    ) -> float:
        smoothness = correction - self.previous_action
        reward = -fuzzy.reward_error_weight * float(np.mean(error**2))
        reward -= fuzzy.reward_velocity_weight * float(np.mean(error_rate**2))
        reward -= fuzzy.reward_effort_weight * float(np.mean(correction**2))
        reward -= fuzzy.reward_smoothness_weight * float(np.mean(smoothness**2))
        reward -= 0.5 * self._constraint_violations(raw_target)
        return reward

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.robot_config.joint_lower_limits, dtype=float)
        upper = np.asarray(self.robot_config.joint_upper_limits, dtype=float)
        return np.clip(target, lower, upper)

    def _constraint_violations(self, q: np.ndarray) -> int:
        lower = np.asarray(self.robot_config.joint_lower_limits, dtype=float)
        upper = np.asarray(self.robot_config.joint_upper_limits, dtype=float)
        return int(np.count_nonzero((q < lower) | (q > upper)))
