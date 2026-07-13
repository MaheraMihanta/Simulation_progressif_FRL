"""Live interactive dynamics loop for the planar 2-DOF arm."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Sequence

import numpy as np

from controllers import FuzzyAccelerationController, PIDController
from envs import Arm2DOFDynamicEnv, Arm2DOFDynamicEnvConfig
from robot import inverse_dynamics_torque, inverse_kinematics, is_reachable
from rl import (
    RESIDUAL_ACTION_NAMES,
    FuzzyDynamicStateEncoder,
    FuzzyResidualQLearningConfig,
    aggregate_fuzzy_q_values,
    residual_acceleration_actions,
)


ControllerMode = Literal["pid", "fuzzy", "fuzzy_rl_safe"]
CONTROLLER_MODES: tuple[ControllerMode, ...] = ("pid", "fuzzy", "fuzzy_rl_safe")


@dataclass(frozen=True)
class LiveArm2DOFConfig:
    """Configuration for live Python interaction with the 2-DOF arm."""

    target: tuple[float, float] = (1.1, 0.55)
    dt: float = 0.01
    max_torque: tuple[float, float] = (60.0, 35.0)
    max_joint_speed: tuple[float, float] = (8.0, 8.0)
    target_tolerance: float = 1e-2
    speed_tolerance: float = 8e-2
    max_steps: int = 10_000_000
    history_length: int = 700
    initial_q: tuple[float, float] = (0.0, 0.0)
    initial_q_dot: tuple[float, float] = (0.0, 0.0)
    disturbance_decay: float = 0.90
    safety_patience: int = 100
    safety_min_progress: float = 1e-4

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        if self.history_length <= 1:
            raise ValueError("history_length must be greater than one.")
        if not 0.0 <= self.disturbance_decay < 1.0:
            raise ValueError("disturbance_decay must satisfy 0 <= value < 1.")
        if self.safety_patience <= 0:
            raise ValueError("safety_patience must be strictly positive.")
        if self.safety_min_progress < 0.0:
            raise ValueError("safety_min_progress must be non-negative.")


class LiveArm2DOFSimulation:
    """Reusable live loop shared by the Matplotlib UI and future adapters."""

    def __init__(
        self,
        config: LiveArm2DOFConfig | None = None,
        mode: ControllerMode = "fuzzy",
        q_value: np.ndarray | None = None,
        encoder: FuzzyDynamicStateEncoder | None = None,
        learning_config: FuzzyResidualQLearningConfig | None = None,
    ) -> None:
        self.config = config or LiveArm2DOFConfig()
        self.env_config = Arm2DOFDynamicEnvConfig(
            target=self.config.target,
            dt=self.config.dt,
            max_torque=self.config.max_torque,
            max_joint_speed=self.config.max_joint_speed,
            target_tolerance=self.config.target_tolerance,
            speed_tolerance=self.config.speed_tolerance,
            max_steps=self.config.max_steps,
        )
        self.env = Arm2DOFDynamicEnv(self.env_config)
        self.encoder = encoder or FuzzyDynamicStateEncoder()
        self.learning_config = learning_config or FuzzyResidualQLearningConfig(
            max_steps_per_episode=550,
        )
        self.residual_actions = residual_acceleration_actions(
            self.learning_config.residual_acceleration_scale,
        )
        self.q_value = self._prepare_q_value(q_value)
        self.pid_controller = self._make_pid_controller()
        self.fuzzy_controller = self._make_fuzzy_controller()
        self.mode: ControllerMode = "fuzzy"
        self.external_torque = np.zeros(2, dtype=float)
        self.last_torque = np.zeros(2, dtype=float)
        self.last_residual_acceleration = np.zeros(2, dtype=float)
        self.last_action_index = 0
        self.last_action_name = RESIDUAL_ACTION_NAMES[0]
        self.last_dominant_rule: int | None = None
        self.last_reward = 0.0
        self.status_message = "ready"
        self.total_steps = 0
        self.residual_disabled = False
        self.residual_switch_step: int | None = None
        self._best_distance = float("inf")
        self._stale_steps = 0

        self.observation = self.env.reset(
            q=self.config.initial_q,
            q_dot=self.config.initial_q_dot,
        )
        self.desired_q = self._desired_q_for_current_target()
        self._init_history()
        self.set_mode(mode)
        self._record_current_state(reward=0.0)

    @property
    def target(self) -> np.ndarray:
        """Return the current Cartesian target."""

        return self.env.target.copy()

    @property
    def link_lengths(self) -> tuple[float, float]:
        """Return the arm geometry used by the live simulation."""

        return self.env_config.arm_config.link_lengths

    def set_mode(self, mode: ControllerMode) -> None:
        """Switch the active controller without resetting the robot state."""

        if mode not in CONTROLLER_MODES:
            raise ValueError(f"unknown controller mode: {mode}")
        self.mode = mode
        self.pid_controller.reset()
        self.fuzzy_controller.reset()
        self._reset_residual_safety()
        self.status_message = f"mode={mode}"

    def set_target(self, target: Sequence[float] | np.ndarray) -> None:
        """Move the live target while keeping the current robot state."""

        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (2,):
            raise ValueError("target must contain exactly two values.")
        if not is_reachable(target_array, self.link_lengths):
            raise ValueError("target is outside the reachable workspace.")
        self.env.set_target(target_array)
        self.desired_q = self._desired_q_for_current_target()
        self.pid_controller.reset()
        self.fuzzy_controller.reset()
        self._reset_residual_safety()
        self.status_message = (
            f"target=({target_array[0]:.3f}, {target_array[1]:.3f})"
        )

    def reset_robot(self, clear_history: bool = True) -> None:
        """Reset the arm state and keep the current target."""

        self.observation = self.env.reset(
            q=self.config.initial_q,
            q_dot=self.config.initial_q_dot,
            target=self.env.target,
        )
        self.pid_controller.reset()
        self.fuzzy_controller.reset()
        self.external_torque[:] = 0.0
        self.last_torque[:] = 0.0
        self.last_residual_acceleration[:] = 0.0
        self.last_action_index = 0
        self.last_action_name = RESIDUAL_ACTION_NAMES[0]
        self.last_dominant_rule = None
        self.last_reward = 0.0
        self.total_steps = 0
        self._reset_residual_safety()
        if clear_history:
            self._init_history()
            self._record_current_state(reward=0.0)
        self.status_message = "reset"

    def apply_disturbance(self, torque: Sequence[float] | np.ndarray) -> None:
        """Add a decaying external joint-torque disturbance."""

        torque_array = np.asarray(torque, dtype=float)
        if torque_array.shape != (2,):
            raise ValueError("disturbance torque must contain exactly two values.")
        self.external_torque += torque_array
        self.status_message = (
            f"disturbance=({torque_array[0]:.2f}, {torque_array[1]:.2f})"
        )

    def step(self) -> dict[str, object]:
        """Advance the live simulation by one control step."""

        torque, residual_acceleration, action_index, dominant_rule = self._control()
        external_torque = (
            self.external_torque.copy()
            if float(np.linalg.norm(self.external_torque)) > 1e-12
            else None
        )
        self.observation, _, done, info = self.env.step(
            torque,
            external_torque=external_torque,
        )
        self.external_torque *= self.config.disturbance_decay
        if float(np.linalg.norm(self.external_torque)) <= 1e-9:
            self.external_torque[:] = 0.0

        self.total_steps += 1
        self.last_torque = np.asarray(info["action"], dtype=float)
        self.last_residual_acceleration = residual_acceleration.copy()
        self.last_action_index = int(action_index)
        self.last_action_name = RESIDUAL_ACTION_NAMES[action_index]
        self.last_dominant_rule = dominant_rule
        self.last_reward = self._live_reward()

        if self.mode == "fuzzy_rl_safe":
            self._update_residual_safety(float(self.observation["distance"]))

        if done:
            self.status_message = "target reached"
        elif bool(info.get("truncated", False)):
            self.status_message = "environment truncated"

        self._record_current_state(reward=self.last_reward)
        return self.summary()

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Return bounded live histories as NumPy arrays."""

        return {
            "q": np.asarray(self.q_history, dtype=float),
            "q_dot": np.asarray(self.q_dot_history, dtype=float),
            "end_effector": np.asarray(self.ee_history, dtype=float),
            "distance": np.asarray(self.distance_history, dtype=float),
            "speed": np.asarray(self.speed_history, dtype=float),
            "torque": np.asarray(self.torque_history, dtype=float),
            "residual": np.asarray(self.residual_history, dtype=float),
            "reward": np.asarray(self.reward_history, dtype=float),
            "action_index": np.asarray(self.action_index_history, dtype=int),
        }

    def summary(self) -> dict[str, object]:
        """Return the compact state needed by live displays and tests."""

        return {
            "mode": self.mode,
            "step": self.total_steps,
            "target": self.target,
            "desired_q": self.desired_q.copy(),
            "q": self.observation["q"].copy(),
            "q_dot": self.observation["q_dot"].copy(),
            "end_effector": self.observation["end_effector"].copy(),
            "distance": float(self.observation["distance"]),
            "speed": float(self.observation["speed"]),
            "torque": self.last_torque.copy(),
            "torque_norm": float(np.linalg.norm(self.last_torque)),
            "residual_acceleration": self.last_residual_acceleration.copy(),
            "action_index": self.last_action_index,
            "action_name": self.last_action_name,
            "dominant_rule": self.last_dominant_rule,
            "residual_disabled": self.residual_disabled,
            "residual_switch_step": self.residual_switch_step,
            "external_torque": self.external_torque.copy(),
            "reward": self.last_reward,
            "message": self.status_message,
        }

    def _prepare_q_value(self, q_value: np.ndarray | None) -> np.ndarray:
        action_count = len(self.residual_actions)
        if q_value is None:
            return np.zeros((self.encoder.n_rules, action_count), dtype=float)
        table = np.asarray(q_value, dtype=float)
        expected_shape = (self.encoder.n_rules, action_count)
        if table.shape != expected_shape:
            raise ValueError(f"q_value must have shape {expected_shape}.")
        return table.copy()

    def _make_pid_controller(self) -> PIDController:
        return PIDController(
            kp=[35.0, 30.0],
            ki=[0.0, 0.0],
            kd=[8.0, 7.0],
            output_limits=(-35.0, 35.0),
        )

    def _make_fuzzy_controller(self) -> FuzzyAccelerationController:
        cfg = self.learning_config
        return FuzzyAccelerationController(
            error_scale=cfg.fuzzy_error_scale,
            derivative_scale=cfg.fuzzy_derivative_scale,
            output_scale=cfg.fuzzy_output_scale,
            output_limits=cfg.fuzzy_output_limits,
        )

    def _desired_q_for_current_target(self) -> np.ndarray:
        return inverse_kinematics(
            self.env.target,
            self.link_lengths,
            elbow="up",
        )

    def _control(self) -> tuple[np.ndarray, np.ndarray, int, int | None]:
        observation = self.observation
        if self.mode == "pid":
            acceleration = self.pid_controller.compute(
                self.desired_q,
                observation["q"],
                self.config.dt,
            )
            residual_acceleration = np.zeros(2, dtype=float)
            action_index = 0
            dominant_rule = None
        else:
            acceleration = self.fuzzy_controller.compute(
                self.desired_q,
                observation["q"],
                self.config.dt,
            )
            residual_acceleration = np.zeros(2, dtype=float)
            action_index = 0
            dominant_rule = None
            if self.mode == "fuzzy_rl_safe":
                rule_indices, rule_weights = self.encoder.active_rules_from_observation(
                    observation,
                    self.desired_q,
                )
                dominant_rule = int(rule_indices[np.argmax(rule_weights)])
                action_values = aggregate_fuzzy_q_values(
                    self.q_value,
                    rule_indices,
                    rule_weights,
                )
                action_index = 0 if self.residual_disabled else int(np.argmax(action_values))
                residual_acceleration = self.residual_actions[action_index]
                acceleration = acceleration + residual_acceleration

        torque = inverse_dynamics_torque(
            observation["q"],
            observation["q_dot"],
            acceleration,
            self.env_config.dynamics_config,
        )
        return torque, residual_acceleration, action_index, dominant_rule

    def _live_reward(self) -> float:
        distance = float(self.observation["distance"])
        speed = float(self.observation["speed"])
        torque_norm = float(np.linalg.norm(self.last_torque))
        residual_norm = float(np.linalg.norm(self.last_residual_acceleration))
        return float(
            -distance
            - 0.02 * speed
            - 3e-4 * torque_norm
            - 0.012 * residual_norm
        )

    def _reset_residual_safety(self) -> None:
        self.residual_disabled = False
        self.residual_switch_step = None
        self._best_distance = float(self.observation["distance"])
        self._stale_steps = 0

    def _update_residual_safety(self, distance: float) -> None:
        if distance < self._best_distance - self.config.safety_min_progress:
            self._best_distance = distance
            self._stale_steps = 0
        else:
            self._stale_steps += 1
        if (
            not self.residual_disabled
            and self._stale_steps >= self.config.safety_patience
        ):
            self.residual_disabled = True
            self.residual_switch_step = self.total_steps
            self.status_message = "residual disabled"

    def _init_history(self) -> None:
        maxlen = self.config.history_length
        self.q_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.q_dot_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.ee_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.distance_history: Deque[float] = deque(maxlen=maxlen)
        self.speed_history: Deque[float] = deque(maxlen=maxlen)
        self.torque_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.residual_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.reward_history: Deque[float] = deque(maxlen=maxlen)
        self.action_index_history: Deque[int] = deque(maxlen=maxlen)

    def _record_current_state(self, reward: float) -> None:
        self.q_history.append(self.observation["q"].copy())
        self.q_dot_history.append(self.observation["q_dot"].copy())
        self.ee_history.append(self.observation["end_effector"].copy())
        self.distance_history.append(float(self.observation["distance"]))
        self.speed_history.append(float(self.observation["speed"]))
        self.torque_history.append(self.last_torque.copy())
        self.residual_history.append(self.last_residual_acceleration.copy())
        self.reward_history.append(float(reward))
        self.action_index_history.append(int(self.last_action_index))
