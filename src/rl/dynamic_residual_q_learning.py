"""Residual tabular Q-learning on the dynamic 2-DOF arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from controllers import PIDController
from envs import Arm2DOFDynamicEnv, Arm2DOFDynamicEnvConfig
from robot import inverse_dynamics_torque, inverse_kinematics
from robot.kinematics import ArrayLike2


RESIDUAL_ACTION_DIRECTIONS = np.array(
    [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ],
    dtype=float,
)

RESIDUAL_ACTION_NAMES = (
    "base",
    "q1_res-",
    "q1_res+",
    "q2_res-",
    "q2_res+",
    "q1_res-/q2_res-",
    "q1_res-/q2_res+",
    "q1_res+/q2_res-",
    "q1_res+/q2_res+",
)


def _as_positive_vector2(values: float | Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 0:
        vector = np.full(2, float(vector), dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"{name} must be a scalar or a vector of size 2.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return vector


def _validate_limits(
    limits: tuple[tuple[float, float], tuple[float, float]],
    name: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(limits) != 2:
        raise ValueError(f"{name} must contain two intervals.")
    for lower, upper in limits:
        if lower >= upper:
            raise ValueError(f"{name} intervals must be ordered as (min, max).")
    return limits


@dataclass(frozen=True)
class DynamicArmStateDiscretizer:
    """Discretize joint error and velocity for tabular dynamic RL."""

    error_bins: tuple[int, int] = (15, 15)
    velocity_bins: tuple[int, int] = (7, 7)
    error_limits: tuple[tuple[float, float], tuple[float, float]] = (
        (-np.pi, np.pi),
        (-np.pi, np.pi),
    )
    velocity_limits: tuple[tuple[float, float], tuple[float, float]] = (
        (-6.0, 6.0),
        (-6.0, 6.0),
    )
    _error_edges: tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)
    _velocity_edges: tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.error_bins) != 2 or len(self.velocity_bins) != 2:
            raise ValueError("error_bins and velocity_bins must contain two values.")
        if any(count < 2 for count in self.error_bins + self.velocity_bins):
            raise ValueError("all discretization axes must have at least two bins.")

        error_limits = _validate_limits(self.error_limits, "error_limits")
        velocity_limits = _validate_limits(self.velocity_limits, "velocity_limits")
        object.__setattr__(
            self,
            "_error_edges",
            tuple(
                np.linspace(lower, upper, count + 1)
                for (lower, upper), count in zip(error_limits, self.error_bins)
            ),
        )
        object.__setattr__(
            self,
            "_velocity_edges",
            tuple(
                np.linspace(lower, upper, count + 1)
                for (lower, upper), count in zip(velocity_limits, self.velocity_bins)
            ),
        )

    @property
    def grid_shape(self) -> tuple[int, int, int, int]:
        return self.error_bins + self.velocity_bins

    @property
    def n_states(self) -> int:
        return int(np.prod(self.grid_shape))

    def indices(self, joint_error: ArrayLike2, q_dot: ArrayLike2) -> tuple[int, int, int, int]:
        error = np.asarray(joint_error, dtype=float)
        velocity = np.asarray(q_dot, dtype=float)
        if error.shape != (2,):
            raise ValueError("joint_error must contain exactly two values.")
        if velocity.shape != (2,):
            raise ValueError("q_dot must contain exactly two values.")

        indices: list[int] = []
        for value, edges in zip(error, self._error_edges):
            index = int(np.searchsorted(edges, value, side="right") - 1)
            indices.append(int(np.clip(index, 0, len(edges) - 2)))
        for value, edges in zip(velocity, self._velocity_edges):
            index = int(np.searchsorted(edges, value, side="right") - 1)
            indices.append(int(np.clip(index, 0, len(edges) - 2)))
        return tuple(indices)  # type: ignore[return-value]

    def encode(self, joint_error: ArrayLike2, q_dot: ArrayLike2) -> int:
        """Return the flat tabular state index."""

        return int(np.ravel_multi_index(self.indices(joint_error, q_dot), self.grid_shape))

    def encode_observation(
        self,
        observation: dict[str, np.ndarray | float],
        desired_q: ArrayLike2,
    ) -> int:
        """Encode an environment observation relative to a desired posture."""

        return self.encode(np.asarray(desired_q, dtype=float) - observation["q"], observation["q_dot"])


@dataclass(frozen=True)
class DynamicResidualQLearningConfig:
    """Hyperparameters for residual Q-learning on the dynamic arm."""

    episodes: int = 180
    max_steps_per_episode: int = 450
    alpha: float = 0.45
    gamma: float = 0.97
    epsilon_start: float = 0.8
    epsilon_end: float = 0.04
    epsilon_decay: float = 0.985
    residual_acceleration_scale: float | Sequence[float] = (2.0, 2.0)
    pid_kp: float | Sequence[float] = (35.0, 30.0)
    pid_ki: float | Sequence[float] = (0.0, 0.0)
    pid_kd: float | Sequence[float] = (8.0, 7.0)
    pid_output_limits: tuple[float, float] = (-35.0, 35.0)
    distance_weight: float = 1.0
    speed_weight: float = 0.02
    torque_weight: float = 3e-4
    residual_weight: float = 0.01
    progress_weight: float = 8.0
    goal_reward: float = 12.0
    start_q: tuple[float, float] = (0.0, 0.0)
    start_q_dot: tuple[float, float] = (0.0, 0.0)
    seed: int | None = 11

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
        _as_positive_vector2(self.residual_acceleration_scale, "residual_acceleration_scale")
        if self.pid_output_limits[0] >= self.pid_output_limits[1]:
            raise ValueError("pid_output_limits must be ordered as (min, max).")
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
class DynamicResidualQLearningResult:
    """Training result for dynamic residual Q-learning."""

    q_value: np.ndarray
    policy: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    episode_success: np.ndarray
    epsilon_history: np.ndarray
    desired_q: np.ndarray
    discretizer: DynamicArmStateDiscretizer


@dataclass(frozen=True)
class DynamicResidualRollout:
    """Rollout generated by a residual Q-learning policy."""

    q_history: np.ndarray
    ee_history: np.ndarray
    distance_history: np.ndarray
    speed_history: np.ndarray
    torque_history: np.ndarray
    action_indices: list[int]
    rewards: list[float]
    done: bool
    truncated: bool


def residual_acceleration_actions(
    scale: float | Sequence[float],
) -> np.ndarray:
    """Return the discrete residual acceleration vectors used as RL actions."""

    return RESIDUAL_ACTION_DIRECTIONS * _as_positive_vector2(
        scale,
        "residual_acceleration_scale",
    )


def epsilon_at_episode(
    config: DynamicResidualQLearningConfig,
    episode_index: int,
) -> float:
    """Return exponentially decayed epsilon for dynamic residual Q-learning."""

    if episode_index < 0:
        raise ValueError("episode_index must be non-negative.")
    return float(
        max(
            config.epsilon_end,
            config.epsilon_start * (config.epsilon_decay**episode_index),
        )
    )


def _make_pid(config: DynamicResidualQLearningConfig) -> PIDController:
    return PIDController(
        kp=config.pid_kp,
        ki=config.pid_ki,
        kd=config.pid_kd,
        output_limits=config.pid_output_limits,
    )


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    maximum = float(np.max(values))
    candidates = np.flatnonzero(np.isclose(values, maximum))
    return int(rng.choice(candidates))


def _reward(
    previous_distance: float,
    distance: float,
    speed: float,
    effort: float,
    residual_norm: float,
    done: bool,
    config: DynamicResidualQLearningConfig,
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


def _desired_q(env_config: Arm2DOFDynamicEnvConfig) -> np.ndarray:
    return inverse_kinematics(
        env_config.target,
        env_config.arm_config.link_lengths,
        elbow="up",
    )


def train_dynamic_residual_q_learning(
    env_config: Arm2DOFDynamicEnvConfig,
    discretizer: DynamicArmStateDiscretizer | None = None,
    config: DynamicResidualQLearningConfig | None = None,
) -> DynamicResidualQLearningResult:
    """Train a residual Q-table around a computed-torque PID controller."""

    cfg = config or DynamicResidualQLearningConfig()
    disc = discretizer or DynamicArmStateDiscretizer()
    actions = residual_acceleration_actions(cfg.residual_acceleration_scale)
    rng = np.random.default_rng(cfg.seed)
    desired_q = _desired_q(env_config)

    q_value = np.zeros((disc.n_states, len(actions)), dtype=float)
    episode_returns = np.zeros(cfg.episodes, dtype=float)
    episode_lengths = np.zeros(cfg.episodes, dtype=int)
    episode_success = np.zeros(cfg.episodes, dtype=bool)
    epsilon_history = np.zeros(cfg.episodes, dtype=float)

    env = Arm2DOFDynamicEnv(env_config)
    for episode in range(cfg.episodes):
        epsilon = epsilon_at_episode(cfg, episode)
        epsilon_history[episode] = epsilon
        pid = _make_pid(cfg)
        observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
        state = disc.encode_observation(observation, desired_q)
        previous_distance = float(observation["distance"])
        total_reward = 0.0
        done = False
        step_count = 0

        for step_count in range(1, cfg.max_steps_per_episode + 1):
            if rng.random() < epsilon:
                action = int(rng.integers(len(actions)))
            else:
                action = _random_argmax(q_value[state], rng)

            base_acceleration = pid.compute(desired_q, observation["q"], env_config.dt)
            residual_acceleration = actions[action]
            desired_acceleration = base_acceleration + residual_acceleration
            torque = inverse_dynamics_torque(
                observation["q"],
                observation["q_dot"],
                desired_acceleration,
                env_config.dynamics_config,
            )
            next_observation, _, done, info = env.step(torque)

            distance = float(next_observation["distance"])
            speed = float(next_observation["speed"])
            effort = float(info["effort"])
            residual_norm = float(np.linalg.norm(residual_acceleration))
            reward = _reward(
                previous_distance,
                distance,
                speed,
                effort,
                residual_norm,
                done,
                cfg,
            )
            next_state = disc.encode_observation(next_observation, desired_q)
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

    policy = np.argmax(q_value, axis=1).astype(int, copy=False)
    return DynamicResidualQLearningResult(
        q_value=q_value,
        policy=policy,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_success=episode_success,
        epsilon_history=epsilon_history,
        desired_q=desired_q,
        discretizer=disc,
    )


def rollout_dynamic_residual_policy(
    env_config: Arm2DOFDynamicEnvConfig,
    policy: np.ndarray,
    discretizer: DynamicArmStateDiscretizer,
    config: DynamicResidualQLearningConfig | None = None,
    desired_q: ArrayLike2 | None = None,
) -> DynamicResidualRollout:
    """Run one greedy rollout of a residual Q-learning policy."""

    cfg = config or DynamicResidualQLearningConfig()
    actions = residual_acceleration_actions(cfg.residual_acceleration_scale)
    policy_array = np.asarray(policy, dtype=int)
    if policy_array.shape != (discretizer.n_states,):
        raise ValueError("policy has an invalid number of states.")
    if np.any(policy_array < 0) or np.any(policy_array >= len(actions)):
        raise ValueError("policy contains an invalid action index.")

    goal_q = _desired_q(env_config) if desired_q is None else np.asarray(desired_q, dtype=float)
    if goal_q.shape != (2,):
        raise ValueError("desired_q must contain exactly two values.")

    env = Arm2DOFDynamicEnv(env_config)
    pid = _make_pid(cfg)
    observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
    state = discretizer.encode_observation(observation, goal_q)
    previous_distance = float(observation["distance"])

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    speed_history = [float(observation["speed"])]
    torque_history: list[np.ndarray] = []
    action_indices: list[int] = []
    rewards: list[float] = []
    done = False
    info: dict[str, object] = {"truncated": False}

    for _ in range(cfg.max_steps_per_episode):
        action = int(policy_array[state])
        base_acceleration = pid.compute(goal_q, observation["q"], env_config.dt)
        residual_acceleration = actions[action]
        torque = inverse_dynamics_torque(
            observation["q"],
            observation["q_dot"],
            base_acceleration + residual_acceleration,
            env_config.dynamics_config,
        )
        observation, _, done, info = env.step(torque)
        distance = float(observation["distance"])
        speed = float(observation["speed"])
        effort = float(info["effort"])
        reward = _reward(
            previous_distance,
            distance,
            speed,
            effort,
            float(np.linalg.norm(residual_acceleration)),
            done,
            cfg,
        )

        q_history.append(observation["q"].copy())
        ee_history.append(observation["end_effector"].copy())
        distance_history.append(distance)
        speed_history.append(speed)
        torque_history.append(info["action"].copy())
        action_indices.append(action)
        rewards.append(reward)
        state = discretizer.encode_observation(observation, goal_q)
        previous_distance = distance
        if done:
            break

    return DynamicResidualRollout(
        q_history=np.asarray(q_history),
        ee_history=np.asarray(ee_history),
        distance_history=np.asarray(distance_history),
        speed_history=np.asarray(speed_history),
        torque_history=np.asarray(torque_history),
        action_indices=action_indices,
        rewards=rewards,
        done=done,
        truncated=bool(info.get("truncated", False)),
    )
