"""Residual Q-learning on top of fuzzy gain-scheduled PID for the 6-DOF arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from controllers import FuzzyGainScheduledPIDController
from envs import Arm6DOFDynamicEnv, Arm6DOFDynamicEnvConfig
from robot import inverse_dynamics_torque_6dof, inverse_kinematics_6dof
from robot.kinematics_6dof import ArrayLike6
from .residual_actions import (
    axis_aligned_residual_action_directions,
    axis_aligned_residual_action_names,
    factorized_residual_action_directions,
    factorized_residual_action_label,
    factorized_residual_action_names,
    factorized_residual_action_vector,
)


PID_RESIDUAL_ACTION_DIRECTIONS_6DOF = axis_aligned_residual_action_directions(6)
PID_RESIDUAL_ACTION_NAMES_6DOF = axis_aligned_residual_action_names(6)
PID_FACTORIZED_RESIDUAL_ACTION_DIRECTIONS_6DOF = (
    factorized_residual_action_directions(6)
)
PID_FACTORIZED_RESIDUAL_ACTION_NAMES_6DOF = factorized_residual_action_names(6)


def _as_positive_vector6(values: float | Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 0:
        vector = np.full(6, float(vector), dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must be a scalar or a vector of size 6.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return vector


def residual_acceleration_actions_6dof(
    scale: float | Sequence[float],
) -> np.ndarray:
    """Return the residual acceleration vectors used as RL actions."""

    return PID_RESIDUAL_ACTION_DIRECTIONS_6DOF * _as_positive_vector6(
        scale,
        "residual_acceleration_scale",
    )


def factorized_residual_acceleration_action_6dof(
    local_action_indices: Sequence[int] | np.ndarray,
    scale: float | Sequence[float],
) -> np.ndarray:
    """Decode one local residual decision per joint into a 6-DOF action."""

    command = factorized_residual_action_vector(local_action_indices, scale)
    if command.shape != (6,):
        raise ValueError("local_action_indices must contain exactly six values.")
    return command


def pid_residual_6dof_epsilon_at_episode(
    config: "PIDResidualQLearning6DOFConfig",
    episode_index: int,
) -> float:
    """Return exponentially decayed epsilon for 6-DOF residual learning."""

    if episode_index < 0:
        raise ValueError("episode_index must be non-negative.")
    return float(
        max(
            config.epsilon_end,
            config.epsilon_start * (config.epsilon_decay**episode_index),
        )
    )


@dataclass(frozen=True)
class PIDResidualStateEncoder6DOF:
    """Encode joint-error signs and speed level into a compact tabular state."""

    joint_error_deadband: float | Sequence[float] = (
        0.04,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
    )
    speed_bins: tuple[float, float] = (0.12, 0.75)
    _deadband_vector: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_deadband_vector",
            _as_positive_vector6(self.joint_error_deadband, "joint_error_deadband"),
        )
        if len(self.speed_bins) != 2 or not 0.0 < self.speed_bins[0] < self.speed_bins[1]:
            raise ValueError("speed_bins must contain two increasing positive values.")

    @property
    def grid_shape(self) -> tuple[int, int, int, int, int, int, int]:
        return (3, 3, 3, 3, 3, 3, 3)

    @property
    def n_states(self) -> int:
        return int(np.prod(self.grid_shape))

    def encode(self, joint_error: ArrayLike6, q_dot: ArrayLike6) -> int:
        error = np.asarray(joint_error, dtype=float)
        velocity = np.asarray(q_dot, dtype=float)
        if error.shape != (6,):
            raise ValueError("joint_error must contain exactly six values.")
        if velocity.shape != (6,):
            raise ValueError("q_dot must contain exactly six values.")

        error_terms = np.ones(6, dtype=int)
        error_terms[error < -self._deadband_vector] = 0
        error_terms[error > self._deadband_vector] = 2

        speed = float(np.linalg.norm(velocity))
        if speed < self.speed_bins[0]:
            speed_term = 0
        elif speed < self.speed_bins[1]:
            speed_term = 1
        else:
            speed_term = 2
        return int(np.ravel_multi_index((*error_terms, speed_term), self.grid_shape))


@dataclass(frozen=True)
class PIDResidualQLearning6DOFConfig:
    """Hyperparameters for residual Q-learning on the 6-DOF arm."""

    episodes: int = 55
    max_steps_per_episode: int = 1000
    alpha: float = 0.25
    gamma: float = 0.97
    epsilon_start: float = 0.70
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.965
    residual_acceleration_scale: float | Sequence[float] = (
        0.30,
        0.40,
        0.40,
        0.30,
        0.25,
        0.20,
    )
    residual_mode: Literal["acceleration", "torque"] = "acceleration"
    pid_kp: float | Sequence[float] = (32.0, 48.0, 38.0, 26.0, 18.0, 14.0)
    pid_ki: float | Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pid_kd: float | Sequence[float] = (8.0, 11.0, 9.0, 6.0, 4.5, 3.5)
    pid_error_scale: float | Sequence[float] = (
        0.35,
        0.45,
        0.55,
        0.55,
        0.45,
        0.40,
    )
    pid_derivative_scale: float | Sequence[float] = (
        4.0,
        5.0,
        5.0,
        5.0,
        4.0,
        4.0,
    )
    pid_output_limits: tuple[float, float] = (-55.0, 55.0)
    initial_q_value: float = -0.1
    distance_weight: float = 1.0
    speed_weight: float = 0.02
    torque_weight: float = 2.5e-4
    residual_weight: float = 0.02
    progress_weight: float = 8.0
    goal_reward: float = 12.0
    external_torque: None | Sequence[float] = None
    start_q: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    start_q_dot: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    seed: int | None = 43

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be strictly positive.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be strictly positive.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma <= 1.")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1."
            )
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must satisfy 0 < epsilon_decay <= 1.")
        _as_positive_vector6(
            self.residual_acceleration_scale,
            "residual_acceleration_scale",
        )
        if self.residual_mode not in ("acceleration", "torque"):
            raise ValueError("residual_mode must be either 'acceleration' or 'torque'.")
        _as_positive_vector6(self.pid_error_scale, "pid_error_scale")
        _as_positive_vector6(self.pid_derivative_scale, "pid_derivative_scale")
        if self.pid_output_limits[0] >= self.pid_output_limits[1]:
            raise ValueError("pid_output_limits must be ordered as (min, max).")
        if self.external_torque is not None:
            torque = np.asarray(self.external_torque, dtype=float)
            if torque.shape != (6,):
                raise ValueError("external_torque must contain exactly six values.")
        for name in (
            "distance_weight",
            "speed_weight",
            "torque_weight",
            "residual_weight",
            "progress_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class PIDResidualQLearning6DOFResult:
    """Training result for 6-DOF residual Q-learning."""

    q_value: np.ndarray
    state_policy: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    episode_success: np.ndarray
    epsilon_history: np.ndarray
    desired_q: np.ndarray
    encoder: PIDResidualStateEncoder6DOF


@dataclass(frozen=True)
class PIDFactorizedResidualQLearning6DOFResult:
    """Training result for factorized 6-DOF residual Q-learning."""

    q_value: np.ndarray
    state_policy: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    episode_success: np.ndarray
    epsilon_history: np.ndarray
    desired_q: np.ndarray
    encoder: PIDResidualStateEncoder6DOF


@dataclass(frozen=True)
class PIDResidualSafety6DOFConfig:
    """Fallback supervisor for residual actions during 6-DOF policy rollout."""

    patience: int = 80
    min_progress: float = 1e-4

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be strictly positive.")
        if self.min_progress < 0.0:
            raise ValueError("min_progress must be non-negative.")


@dataclass(frozen=True)
class PIDResidual6DOFRollout:
    """Rollout generated from a 6-DOF residual Q table."""

    q_history: np.ndarray
    ee_history: np.ndarray
    distance_history: np.ndarray
    speed_history: np.ndarray
    torque_history: np.ndarray
    action_indices: list[int]
    state_indices: list[int]
    rewards: list[float]
    done: bool
    truncated: bool
    residual_disabled: bool = False
    residual_switch_step: int | None = None


@dataclass(frozen=True)
class PIDFactorizedResidual6DOFRollout:
    """Rollout generated from a factorized 6-DOF residual Q table."""

    q_history: np.ndarray
    ee_history: np.ndarray
    distance_history: np.ndarray
    speed_history: np.ndarray
    torque_history: np.ndarray
    local_action_indices: list[np.ndarray]
    action_labels: list[str]
    state_indices: list[int]
    rewards: list[float]
    done: bool
    truncated: bool
    residual_disabled: bool = False
    residual_switch_step: int | None = None


def _desired_q(env_config: Arm6DOFDynamicEnvConfig) -> np.ndarray:
    return inverse_kinematics_6dof(
        env_config.target,
        env_config.arm_config.link_lengths,
        elbow="up",
        terminal_pitch=0.0,
        joint_limits=env_config.arm_config.joint_limits,
    )


def _make_controller(
    config: PIDResidualQLearning6DOFConfig,
) -> FuzzyGainScheduledPIDController:
    return FuzzyGainScheduledPIDController(
        kp=config.pid_kp,
        ki=config.pid_ki,
        kd=config.pid_kd,
        size=6,
        error_scale=config.pid_error_scale,
        derivative_scale=config.pid_derivative_scale,
        output_limits=config.pid_output_limits,
    )


def _external_torque(
    config: PIDResidualQLearning6DOFConfig,
) -> np.ndarray | None:
    if config.external_torque is None:
        return None
    return np.asarray(config.external_torque, dtype=float)


def _torque_with_residual(
    q: np.ndarray,
    q_dot: np.ndarray,
    base_acceleration: np.ndarray,
    residual_command: np.ndarray,
    env_config: Arm6DOFDynamicEnvConfig,
    config: PIDResidualQLearning6DOFConfig,
) -> np.ndarray:
    if config.residual_mode == "acceleration":
        return inverse_dynamics_torque_6dof(
            q,
            q_dot,
            base_acceleration + residual_command,
            env_config.dynamics_config,
        )
    base_torque = inverse_dynamics_torque_6dof(
        q,
        q_dot,
        base_acceleration,
        env_config.dynamics_config,
    )
    return base_torque + residual_command


def _reward(
    previous_distance: float,
    distance: float,
    speed: float,
    effort: float,
    residual_norm: float,
    done: bool,
    config: PIDResidualQLearning6DOFConfig,
) -> float:
    reward = (
        -config.distance_weight * distance
        - config.speed_weight * speed
        - config.torque_weight * effort
        - config.residual_weight * residual_norm
        + config.progress_weight * (previous_distance - distance)
    )
    if done:
        reward += config.goal_reward
    return float(reward)


def _random_factorized_action(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(
        PID_FACTORIZED_RESIDUAL_ACTION_DIRECTIONS_6DOF.shape[1],
        size=6,
        dtype=int,
    )


def _greedy_factorized_action(q_values_for_state: np.ndarray) -> np.ndarray:
    if q_values_for_state.shape != (6, 3):
        raise ValueError("q_values_for_state must have shape (6, 3).")
    return np.argmax(q_values_for_state, axis=1).astype(int, copy=False)


def train_pid_residual_q_learning_6dof(
    env_config: Arm6DOFDynamicEnvConfig,
    encoder: PIDResidualStateEncoder6DOF | None = None,
    config: PIDResidualQLearning6DOFConfig | None = None,
) -> PIDResidualQLearning6DOFResult:
    """Train a tabular residual policy on top of fuzzy gain-scheduled PID."""

    cfg = config or PIDResidualQLearning6DOFConfig()
    state_encoder = encoder or PIDResidualStateEncoder6DOF()
    actions = residual_acceleration_actions_6dof(cfg.residual_acceleration_scale)
    rng = np.random.default_rng(cfg.seed)
    desired_q = _desired_q(env_config)
    q_value = np.full(
        (state_encoder.n_states, len(actions)),
        cfg.initial_q_value,
        dtype=float,
    )
    episode_returns = np.zeros(cfg.episodes, dtype=float)
    episode_lengths = np.zeros(cfg.episodes, dtype=int)
    episode_success = np.zeros(cfg.episodes, dtype=bool)
    epsilon_history = np.zeros(cfg.episodes, dtype=float)

    env = Arm6DOFDynamicEnv(env_config)
    for episode in range(cfg.episodes):
        epsilon = pid_residual_6dof_epsilon_at_episode(cfg, episode)
        epsilon_history[episode] = epsilon
        controller = _make_controller(cfg)
        observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
        state = state_encoder.encode(desired_q - observation["q"], observation["q_dot"])
        previous_distance = float(observation["distance"])
        total_reward = 0.0
        done = False
        step_count = 0

        for step_count in range(1, cfg.max_steps_per_episode + 1):
            if rng.random() < epsilon:
                action = int(rng.integers(len(actions)))
            else:
                action = int(np.argmax(q_value[state]))

            base_acceleration = controller.compute(
                desired_q,
                observation["q"],
                env_config.dt,
            )
            residual_command = actions[action]
            torque = _torque_with_residual(
                observation["q"],
                observation["q_dot"],
                base_acceleration,
                residual_command,
                env_config,
                cfg,
            )
            next_observation, _, done, info = env.step(
                torque,
                external_torque=_external_torque(cfg),
            )
            distance = float(next_observation["distance"])
            speed = float(next_observation["speed"])
            effort = float(info["effort"])
            reward = _reward(
                previous_distance,
                distance,
                speed,
                effort,
                float(np.linalg.norm(residual_command)),
                done,
                cfg,
            )
            next_state = state_encoder.encode(
                desired_q - next_observation["q"],
                next_observation["q_dot"],
            )
            target = reward if done else reward + cfg.gamma * float(np.max(q_value[next_state]))
            q_value[state, action] += cfg.alpha * (target - q_value[state, action])

            total_reward += reward
            observation = next_observation
            state = next_state
            previous_distance = distance
            if done:
                break

        episode_returns[episode] = total_reward
        episode_lengths[episode] = step_count
        episode_success[episode] = done

    return PIDResidualQLearning6DOFResult(
        q_value=q_value,
        state_policy=np.argmax(q_value, axis=1).astype(int, copy=False),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_success=episode_success,
        epsilon_history=epsilon_history,
        desired_q=desired_q,
        encoder=state_encoder,
    )


def train_pid_factorized_residual_q_learning_6dof(
    env_config: Arm6DOFDynamicEnvConfig,
    encoder: PIDResidualStateEncoder6DOF | None = None,
    config: PIDResidualQLearning6DOFConfig | None = None,
) -> PIDFactorizedResidualQLearning6DOFResult:
    """Train additive per-joint residual Q tables on top of fuzzy PID."""

    cfg = config or PIDResidualQLearning6DOFConfig()
    state_encoder = encoder or PIDResidualStateEncoder6DOF()
    rng = np.random.default_rng(cfg.seed)
    desired_q = _desired_q(env_config)
    local_action_count = PID_FACTORIZED_RESIDUAL_ACTION_DIRECTIONS_6DOF.shape[1]
    q_value = np.full(
        (state_encoder.n_states, 6, local_action_count),
        cfg.initial_q_value,
        dtype=float,
    )
    episode_returns = np.zeros(cfg.episodes, dtype=float)
    episode_lengths = np.zeros(cfg.episodes, dtype=int)
    episode_success = np.zeros(cfg.episodes, dtype=bool)
    epsilon_history = np.zeros(cfg.episodes, dtype=float)

    env = Arm6DOFDynamicEnv(env_config)
    for episode in range(cfg.episodes):
        epsilon = pid_residual_6dof_epsilon_at_episode(cfg, episode)
        epsilon_history[episode] = epsilon
        controller = _make_controller(cfg)
        observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
        state = state_encoder.encode(desired_q - observation["q"], observation["q_dot"])
        previous_distance = float(observation["distance"])
        total_reward = 0.0
        done = False
        step_count = 0

        for step_count in range(1, cfg.max_steps_per_episode + 1):
            if rng.random() < epsilon:
                local_actions = _random_factorized_action(rng)
            else:
                local_actions = _greedy_factorized_action(q_value[state])

            base_acceleration = controller.compute(
                desired_q,
                observation["q"],
                env_config.dt,
            )
            residual_command = factorized_residual_acceleration_action_6dof(
                local_actions,
                cfg.residual_acceleration_scale,
            )
            torque = _torque_with_residual(
                observation["q"],
                observation["q_dot"],
                base_acceleration,
                residual_command,
                env_config,
                cfg,
            )
            next_observation, _, done, info = env.step(
                torque,
                external_torque=_external_torque(cfg),
            )
            distance = float(next_observation["distance"])
            speed = float(next_observation["speed"])
            effort = float(info["effort"])
            reward = _reward(
                previous_distance,
                distance,
                speed,
                effort,
                float(np.linalg.norm(residual_command)),
                done,
                cfg,
            )
            next_state = state_encoder.encode(
                desired_q - next_observation["q"],
                next_observation["q_dot"],
            )

            joint_indices = np.arange(6)
            current_sum = float(np.sum(q_value[state, joint_indices, local_actions]))
            next_sum = float(np.sum(np.max(q_value[next_state], axis=1)))
            target = reward if done else reward + cfg.gamma * next_sum
            td_error = target - current_sum
            q_value[state, joint_indices, local_actions] += cfg.alpha * td_error / 6.0

            total_reward += reward
            observation = next_observation
            state = next_state
            previous_distance = distance
            if done:
                break

        episode_returns[episode] = total_reward
        episode_lengths[episode] = step_count
        episode_success[episode] = done

    return PIDFactorizedResidualQLearning6DOFResult(
        q_value=q_value,
        state_policy=np.argmax(q_value, axis=2).astype(int, copy=False),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_success=episode_success,
        epsilon_history=epsilon_history,
        desired_q=desired_q,
        encoder=state_encoder,
    )


def rollout_pid_residual_q_policy_6dof(
    env_config: Arm6DOFDynamicEnvConfig,
    q_value: np.ndarray,
    encoder: PIDResidualStateEncoder6DOF,
    config: PIDResidualQLearning6DOFConfig | None = None,
    desired_q: ArrayLike6 | None = None,
    safety_config: PIDResidualSafety6DOFConfig | None = None,
) -> PIDResidual6DOFRollout:
    """Run one greedy rollout from a 6-DOF residual Q table."""

    cfg = config or PIDResidualQLearning6DOFConfig()
    actions = residual_acceleration_actions_6dof(cfg.residual_acceleration_scale)
    q_table = np.asarray(q_value, dtype=float)
    if q_table.shape != (encoder.n_states, len(actions)):
        raise ValueError("q_value has an invalid shape.")
    goal_q = _desired_q(env_config) if desired_q is None else np.asarray(desired_q, dtype=float)
    if goal_q.shape != (6,):
        raise ValueError("desired_q must contain exactly six values.")

    env = Arm6DOFDynamicEnv(env_config)
    controller = _make_controller(cfg)
    observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
    state = encoder.encode(goal_q - observation["q"], observation["q_dot"])
    previous_distance = float(observation["distance"])

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    speed_history = [float(observation["speed"])]
    torque_history: list[np.ndarray] = []
    action_indices: list[int] = []
    state_indices: list[int] = []
    rewards: list[float] = []
    done = False
    info: dict[str, object] = {"truncated": False}
    best_distance = previous_distance
    stale_steps = 0
    residual_disabled = False
    residual_switch_step: int | None = None

    for _ in range(cfg.max_steps_per_episode):
        action = 0 if residual_disabled else int(np.argmax(q_table[state]))
        base_acceleration = controller.compute(goal_q, observation["q"], env_config.dt)
        residual_command = actions[action]
        torque = _torque_with_residual(
            observation["q"],
            observation["q_dot"],
            base_acceleration,
            residual_command,
            env_config,
            cfg,
        )
        observation, _, done, info = env.step(
            torque,
            external_torque=_external_torque(cfg),
        )
        distance = float(observation["distance"])
        speed = float(observation["speed"])
        effort = float(info["effort"])
        reward = _reward(
            previous_distance,
            distance,
            speed,
            effort,
            float(np.linalg.norm(residual_command)),
            done,
            cfg,
        )

        q_history.append(observation["q"].copy())
        ee_history.append(observation["end_effector"].copy())
        distance_history.append(distance)
        speed_history.append(speed)
        torque_history.append(info["action"].copy())
        action_indices.append(action)
        state_indices.append(state)
        rewards.append(reward)

        if safety_config is not None:
            if distance < best_distance - safety_config.min_progress:
                best_distance = distance
                stale_steps = 0
            else:
                stale_steps += 1
            if not residual_disabled and stale_steps >= safety_config.patience:
                residual_disabled = True
                residual_switch_step = len(action_indices)

        state = encoder.encode(goal_q - observation["q"], observation["q_dot"])
        previous_distance = distance
        if done:
            break

    return PIDResidual6DOFRollout(
        q_history=np.asarray(q_history),
        ee_history=np.asarray(ee_history),
        distance_history=np.asarray(distance_history),
        speed_history=np.asarray(speed_history),
        torque_history=np.asarray(torque_history),
        action_indices=action_indices,
        state_indices=state_indices,
        rewards=rewards,
        done=done,
        truncated=bool(info.get("truncated", False)),
        residual_disabled=residual_disabled,
        residual_switch_step=residual_switch_step,
    )


def rollout_pid_factorized_residual_q_policy_6dof(
    env_config: Arm6DOFDynamicEnvConfig,
    q_value: np.ndarray,
    encoder: PIDResidualStateEncoder6DOF,
    config: PIDResidualQLearning6DOFConfig | None = None,
    desired_q: ArrayLike6 | None = None,
    safety_config: PIDResidualSafety6DOFConfig | None = None,
) -> PIDFactorizedResidual6DOFRollout:
    """Run one greedy rollout from factorized 6-DOF residual Q tables."""

    cfg = config or PIDResidualQLearning6DOFConfig()
    q_table = np.asarray(q_value, dtype=float)
    expected_shape = (
        encoder.n_states,
        6,
        PID_FACTORIZED_RESIDUAL_ACTION_DIRECTIONS_6DOF.shape[1],
    )
    if q_table.shape != expected_shape:
        raise ValueError("q_value has an invalid shape.")
    goal_q = _desired_q(env_config) if desired_q is None else np.asarray(desired_q, dtype=float)
    if goal_q.shape != (6,):
        raise ValueError("desired_q must contain exactly six values.")

    env = Arm6DOFDynamicEnv(env_config)
    controller = _make_controller(cfg)
    observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
    state = encoder.encode(goal_q - observation["q"], observation["q_dot"])
    previous_distance = float(observation["distance"])

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    speed_history = [float(observation["speed"])]
    torque_history: list[np.ndarray] = []
    local_action_indices: list[np.ndarray] = []
    action_labels: list[str] = []
    state_indices: list[int] = []
    rewards: list[float] = []
    done = False
    info: dict[str, object] = {"truncated": False}
    best_distance = previous_distance
    stale_steps = 0
    residual_disabled = False
    residual_switch_step: int | None = None

    for _ in range(cfg.max_steps_per_episode):
        if residual_disabled:
            local_actions = np.zeros(6, dtype=int)
        else:
            local_actions = _greedy_factorized_action(q_table[state])
        base_acceleration = controller.compute(goal_q, observation["q"], env_config.dt)
        residual_command = factorized_residual_acceleration_action_6dof(
            local_actions,
            cfg.residual_acceleration_scale,
        )
        torque = _torque_with_residual(
            observation["q"],
            observation["q_dot"],
            base_acceleration,
            residual_command,
            env_config,
            cfg,
        )
        observation, _, done, info = env.step(
            torque,
            external_torque=_external_torque(cfg),
        )
        distance = float(observation["distance"])
        speed = float(observation["speed"])
        effort = float(info["effort"])
        reward = _reward(
            previous_distance,
            distance,
            speed,
            effort,
            float(np.linalg.norm(residual_command)),
            done,
            cfg,
        )

        q_history.append(observation["q"].copy())
        ee_history.append(observation["end_effector"].copy())
        distance_history.append(distance)
        speed_history.append(speed)
        torque_history.append(info["action"].copy())
        local_action_indices.append(local_actions.copy())
        action_labels.append(factorized_residual_action_label(local_actions))
        state_indices.append(state)
        rewards.append(reward)

        if safety_config is not None:
            if distance < best_distance - safety_config.min_progress:
                best_distance = distance
                stale_steps = 0
            else:
                stale_steps += 1
            if not residual_disabled and stale_steps >= safety_config.patience:
                residual_disabled = True
                residual_switch_step = len(local_action_indices)

        state = encoder.encode(goal_q - observation["q"], observation["q_dot"])
        previous_distance = distance
        if done:
            break

    return PIDFactorizedResidual6DOFRollout(
        q_history=np.asarray(q_history),
        ee_history=np.asarray(ee_history),
        distance_history=np.asarray(distance_history),
        speed_history=np.asarray(speed_history),
        torque_history=np.asarray(torque_history),
        local_action_indices=local_action_indices,
        action_labels=action_labels,
        state_indices=state_indices,
        rewards=rewards,
        done=done,
        truncated=bool(info.get("truncated", False)),
        residual_disabled=residual_disabled,
        residual_switch_step=residual_switch_step,
    )


__all__ = [
    "PID_FACTORIZED_RESIDUAL_ACTION_DIRECTIONS_6DOF",
    "PID_FACTORIZED_RESIDUAL_ACTION_NAMES_6DOF",
    "PID_RESIDUAL_ACTION_DIRECTIONS_6DOF",
    "PID_RESIDUAL_ACTION_NAMES_6DOF",
    "PIDFactorizedResidual6DOFRollout",
    "PIDFactorizedResidualQLearning6DOFResult",
    "PIDResidual6DOFRollout",
    "PIDResidualQLearning6DOFConfig",
    "PIDResidualQLearning6DOFResult",
    "PIDResidualSafety6DOFConfig",
    "PIDResidualStateEncoder6DOF",
    "factorized_residual_acceleration_action_6dof",
    "pid_residual_6dof_epsilon_at_episode",
    "residual_acceleration_actions_6dof",
    "rollout_pid_factorized_residual_q_policy_6dof",
    "rollout_pid_residual_q_policy_6dof",
    "train_pid_factorized_residual_q_learning_6dof",
    "train_pid_residual_q_learning_6dof",
]
