from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import RobotConfig, SimulationConfig
from .controllers import FuzzyGuidedPIDController
from .fuzzy import FuzzyOutput, FuzzySupervisor
from .state import ArmState
from .trajectory import JointTrajectory, make_trajectory

from robot import forward_kinematics_6dof


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

    Action convention: a policy outputs one normalized value per joint in
    [-1, 1]. In residual mode, this value is a bounded correction around a
    fuzzy-PID expert target. In direct mode, it is the full correction around
    q_ref. The residual mode matches the PID-flou + RL pattern used by the
    tabular controllers.
    """

    def __init__(
        self,
        backend: ArmBackend,
        robot_config: RobotConfig,
        simulation_config: SimulationConfig,
        trajectory_name: str = "multi_sine",
        supervisor: FuzzySupervisor | None = None,
        action_mode: str = "residual",
        residual_scale: float = 0.35,
        initial_q: np.ndarray | None = None,
    ) -> None:
        if action_mode not in {"direct", "residual"}:
            raise ValueError("action_mode must be 'direct' or 'residual'")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        self.backend = backend
        self.robot_config = robot_config
        self.simulation_config = simulation_config
        self.trajectory_name = trajectory_name
        self.supervisor = supervisor or FuzzySupervisor()
        self.action_mode = action_mode
        self.residual_scale = residual_scale
        self.initial_q = None if initial_q is None else np.asarray(initial_q, dtype=float)
        if self.initial_q is not None and self.initial_q.shape != (robot_config.dof,):
            raise ValueError(
                f"initial_q must have shape ({robot_config.dof},), got {self.initial_q.shape}"
            )
        self.expert_controller = (
            FuzzyGuidedPIDController(
                dof=robot_config.dof,
                correction_limit=robot_config.max_position_correction,
                joint_lower=robot_config.joint_lower_limits,
                joint_upper=robot_config.joint_upper_limits,
                supervisor=self.supervisor,
            )
            if action_mode == "residual"
            else None
        )
        self.trajectory: JointTrajectory | None = None
        self.current_state: ArmState | None = None
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
        state = self.backend.reset(self.initial_q)
        self.trajectory = make_trajectory(self.trajectory_name, state.q, self.simulation_config.duration)
        self.current_state = state
        self.t = 0.0
        self.previous_action[:] = 0.0
        if self.expert_controller is not None:
            self.expert_controller.reset()
        reference = self.trajectory.sample(self.t)
        fuzzy = self.supervisor.evaluate(reference.q - state.q, reference.q_dot - state.q_dot, self.previous_action)
        return self._make_observation(state, reference.q, reference.q_dot, fuzzy)

    def step(self, normalized_action: np.ndarray) -> RLStepResult:
        if self.trajectory is None:
            raise RuntimeError("FuzzyGuidedTrackingTask.reset() must be called before step()")
        if self.current_state is None:
            raise RuntimeError("FuzzyGuidedTrackingTask.reset() must be called before step()")

        action = np.clip(np.asarray(normalized_action, dtype=float), -1.0, 1.0)
        if action.shape != (self.robot_config.dof,):
            raise ValueError(f"Action must have shape ({self.robot_config.dof},), got {action.shape}")

        reference = self.trajectory.sample(self.t)
        correction_limit = np.asarray(self.robot_config.max_position_correction, dtype=float)
        if self.expert_controller is None:
            residual = action * correction_limit
            expert_correction = np.zeros(self.robot_config.dof, dtype=float)
            expert_target = reference.q
        else:
            expert_output = self.expert_controller.compute(
                self.current_state.q,
                self.current_state.q_dot,
                reference.q,
                reference.q_dot,
                self.simulation_config.dt,
            )
            residual = action * correction_limit * self.residual_scale
            expert_correction = expert_output.correction
            expert_target = expert_output.target_position
        raw_target = expert_target + residual
        target = self._clip_target(raw_target)
        next_state = self.backend.step(target)

        next_t = self.t + self.simulation_config.dt
        next_reference = self.trajectory.sample(next_t)
        error = next_reference.q - next_state.q
        error_rate = next_reference.q_dot - next_state.q_dot
        cartesian_error_norm = self._cartesian_error_norm(next_state, next_reference)
        correction = target - reference.q
        policy_effort = residual if self.expert_controller is not None else correction
        fuzzy = self.supervisor.evaluate(error, error_rate, policy_effort)
        reward = self._reward(error, error_rate, policy_effort, raw_target, fuzzy)
        violation_count = self._constraint_violations(next_state.q)

        self.t = next_t
        truncated = self.t >= self.simulation_config.duration
        self.current_state = next_state
        self.previous_action = policy_effort.copy()
        observation = self._make_observation(next_state, next_reference.q, next_reference.q_dot, fuzzy)
        return RLStepResult(
            observation=observation,
            reward=reward,
            terminated=False,
            truncated=truncated,
            info={
                "time": self.t,
                "error_norm": float(np.linalg.norm(error)),
                "cartesian_error_norm": cartesian_error_norm,
                "action_norm": float(np.linalg.norm(correction)),
                "residual_norm": float(np.linalg.norm(residual)),
                "expert_correction_norm": float(np.linalg.norm(expert_correction)),
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

    def _cartesian_error_norm(
        self,
        state: ArmState,
        reference,
    ) -> float:
        if reference.position is None:
            return 0.0
        if state.tip_position is not None:
            position = state.tip_position
        elif self.robot_config.dof == 6:
            position = forward_kinematics_6dof(state.q)
        else:
            return 0.0
        return float(np.linalg.norm(reference.position - position))
