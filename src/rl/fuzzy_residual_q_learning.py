"""Fuzzy residual Q-learning on the dynamic 2-DOF arm.

The fuzzy part provides both the stabilizing base controller and a linguistic
state abstraction. Q-learning then learns residual acceleration actions on top
of the fuzzy computed-torque command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

import numpy as np

from controllers import FuzzyAccelerationController
from envs import Arm2DOFDynamicEnv, Arm2DOFDynamicEnvConfig
from robot import inverse_dynamics_torque, inverse_kinematics
from robot.kinematics import ArrayLike2

from .dynamic_residual_q_learning import (
    RESIDUAL_ACTION_DIRECTIONS,
    RESIDUAL_ACTION_NAMES,
    residual_acceleration_actions,
)


FUZZY_TERMS = ("negative", "zero", "positive")
FUZZY_STATE_VARIABLES = ("error_q1", "error_q2", "q1_dot", "q2_dot")


def _as_positive_vector2(values: float | Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 0:
        vector = np.full(2, float(vector), dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"{name} must be a scalar or a vector of size 2.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return vector


def _fuzzy_memberships(value: float) -> np.ndarray:
    value = float(np.clip(value, -1.0, 1.0))
    return np.array(
        [
            max(0.0, -value),
            max(0.0, 1.0 - abs(value)),
            max(0.0, value),
        ],
        dtype=float,
    )


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    maximum = float(np.max(values))
    candidates = np.flatnonzero(np.isclose(values, maximum))
    return int(rng.choice(candidates))


def _desired_q(env_config: Arm2DOFDynamicEnvConfig) -> np.ndarray:
    return inverse_kinematics(
        env_config.target,
        env_config.arm_config.link_lengths,
        elbow="up",
    )


def _reward(
    previous_distance: float,
    distance: float,
    speed: float,
    effort: float,
    residual_norm: float,
    done: bool,
    config: FuzzyResidualQLearningConfig,
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


@dataclass(frozen=True)
class FuzzyDynamicStateEncoder:
    """Encode dynamic observations as activated fuzzy linguistic rules."""

    error_scale: float | Sequence[float] = (0.9, 1.2)
    velocity_scale: float | Sequence[float] = (6.0, 6.0)
    min_activation: float = 1e-12
    _error_scale_vector: np.ndarray = field(init=False, repr=False)
    _velocity_scale_vector: np.ndarray = field(init=False, repr=False)
    _rule_terms: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_error_scale_vector",
            _as_positive_vector2(self.error_scale, "error_scale"),
        )
        object.__setattr__(
            self,
            "_velocity_scale_vector",
            _as_positive_vector2(self.velocity_scale, "velocity_scale"),
        )
        if self.min_activation < 0.0:
            raise ValueError("min_activation must be non-negative.")
        object.__setattr__(
            self,
            "_rule_terms",
            np.asarray(list(product(range(len(FUZZY_TERMS)), repeat=4)), dtype=int),
        )

    @property
    def grid_shape(self) -> tuple[int, int, int, int]:
        return (3, 3, 3, 3)

    @property
    def n_rules(self) -> int:
        return int(np.prod(self.grid_shape))

    def memberships(self, joint_error: ArrayLike2, q_dot: ArrayLike2) -> np.ndarray:
        """Return fuzzy memberships for error and velocity variables."""

        error = np.asarray(joint_error, dtype=float)
        velocity = np.asarray(q_dot, dtype=float)
        if error.shape != (2,):
            raise ValueError("joint_error must contain exactly two values.")
        if velocity.shape != (2,):
            raise ValueError("q_dot must contain exactly two values.")

        normalized = np.concatenate(
            [
                error / self._error_scale_vector,
                velocity / self._velocity_scale_vector,
            ]
        )
        return np.vstack([_fuzzy_memberships(value) for value in normalized])

    def rule_activations(
        self,
        joint_error: ArrayLike2,
        q_dot: ArrayLike2,
    ) -> np.ndarray:
        """Return normalized activation weights for all fuzzy rules."""

        memberships = self.memberships(joint_error, q_dot)
        activations = np.prod(
            memberships[np.arange(4)[:, None], self._rule_terms.T],
            axis=0,
        )
        total = float(np.sum(activations))
        if total <= 0.0:
            return np.full(self.n_rules, 1.0 / self.n_rules, dtype=float)
        return activations / total

    def active_rules(
        self,
        joint_error: ArrayLike2,
        q_dot: ArrayLike2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return indices and normalized weights of active fuzzy rules."""

        activations = self.rule_activations(joint_error, q_dot)
        indices = np.flatnonzero(activations > self.min_activation)
        if indices.size == 0:
            index = int(np.argmax(activations))
            return np.asarray([index], dtype=int), np.asarray([1.0], dtype=float)
        weights = activations[indices]
        weights = weights / float(np.sum(weights))
        return indices.astype(int, copy=False), weights.astype(float, copy=False)

    def active_rules_from_observation(
        self,
        observation: dict[str, np.ndarray | float],
        desired_q: ArrayLike2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fuzzy rules activated by an environment observation."""

        joint_error = np.asarray(desired_q, dtype=float) - observation["q"]
        return self.active_rules(joint_error, observation["q_dot"])

    def dominant_rule(self, joint_error: ArrayLike2, q_dot: ArrayLike2) -> int:
        """Return the most activated fuzzy rule index."""

        return int(np.argmax(self.rule_activations(joint_error, q_dot)))

    def rule_terms(self, rule_index: int) -> tuple[str, str, str, str]:
        """Return linguistic terms for one fuzzy rule."""

        if rule_index < 0 or rule_index >= self.n_rules:
            raise ValueError("rule_index is outside the fuzzy rule base.")
        return tuple(FUZZY_TERMS[index] for index in self._rule_terms[rule_index])


@dataclass(frozen=True)
class FuzzyResidualQLearningConfig:
    """Hyperparameters for fuzzy residual Q-learning."""

    episodes: int = 220
    max_steps_per_episode: int = 550
    alpha: float = 0.35
    gamma: float = 0.97
    epsilon_start: float = 0.75
    epsilon_end: float = 0.04
    epsilon_decay: float = 0.985
    residual_acceleration_scale: float | Sequence[float] = (1.5, 1.5)
    fuzzy_error_scale: float | Sequence[float] = (0.30, 0.60)
    fuzzy_derivative_scale: float | Sequence[float] = (4.0, 4.0)
    fuzzy_output_scale: float | Sequence[float] = (35.0, 30.0)
    fuzzy_output_limits: tuple[float, float] = (-35.0, 35.0)
    initial_q_value: float = 0.0
    distance_weight: float = 1.0
    speed_weight: float = 0.02
    torque_weight: float = 3e-4
    residual_weight: float = 0.012
    progress_weight: float = 8.0
    goal_reward: float = 12.0
    start_q: tuple[float, float] = (0.0, 0.0)
    start_q_dot: tuple[float, float] = (0.0, 0.0)
    seed: int | None = 17

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
        _as_positive_vector2(self.fuzzy_error_scale, "fuzzy_error_scale")
        _as_positive_vector2(self.fuzzy_derivative_scale, "fuzzy_derivative_scale")
        _as_positive_vector2(self.fuzzy_output_scale, "fuzzy_output_scale")
        if self.fuzzy_output_limits[0] >= self.fuzzy_output_limits[1]:
            raise ValueError("fuzzy_output_limits must be ordered as (min, max).")
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
class FuzzyResidualQLearningResult:
    """Training result for fuzzy residual Q-learning."""

    q_value: np.ndarray
    rule_policy: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    episode_success: np.ndarray
    epsilon_history: np.ndarray
    desired_q: np.ndarray
    encoder: FuzzyDynamicStateEncoder


@dataclass(frozen=True)
class FuzzyResidualRollout:
    """Rollout generated from a fuzzy residual Q table."""

    q_history: np.ndarray
    ee_history: np.ndarray
    distance_history: np.ndarray
    speed_history: np.ndarray
    torque_history: np.ndarray
    action_indices: list[int]
    dominant_rule_indices: list[int]
    rewards: list[float]
    done: bool
    truncated: bool


def fuzzy_residual_epsilon_at_episode(
    config: FuzzyResidualQLearningConfig,
    episode_index: int,
) -> float:
    """Return exponentially decayed epsilon for fuzzy residual Q-learning."""

    if episode_index < 0:
        raise ValueError("episode_index must be non-negative.")
    return float(
        max(
            config.epsilon_end,
            config.epsilon_start * (config.epsilon_decay**episode_index),
        )
    )


def aggregate_fuzzy_q_values(
    q_value: np.ndarray,
    rule_indices: np.ndarray,
    rule_weights: np.ndarray,
) -> np.ndarray:
    """Return action values obtained by fuzzy aggregation of active rules."""

    if q_value.ndim != 2:
        raise ValueError("q_value must be a 2D array.")
    indices = np.asarray(rule_indices, dtype=int)
    weights = np.asarray(rule_weights, dtype=float)
    if indices.ndim != 1 or weights.ndim != 1 or indices.shape != weights.shape:
        raise ValueError("rule_indices and rule_weights must be matching 1D arrays.")
    if np.any(indices < 0) or np.any(indices >= q_value.shape[0]):
        raise ValueError("rule_indices contains an invalid rule index.")
    if np.any(weights < 0.0):
        raise ValueError("rule_weights must be non-negative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("at least one rule weight must be positive.")
    normalized = weights / total
    return normalized @ q_value[indices]


def _make_fuzzy_controller(
    config: FuzzyResidualQLearningConfig,
) -> FuzzyAccelerationController:
    return FuzzyAccelerationController(
        error_scale=config.fuzzy_error_scale,
        derivative_scale=config.fuzzy_derivative_scale,
        output_scale=config.fuzzy_output_scale,
        output_limits=config.fuzzy_output_limits,
    )


def train_fuzzy_residual_q_learning(
    env_config: Arm2DOFDynamicEnvConfig,
    encoder: FuzzyDynamicStateEncoder | None = None,
    config: FuzzyResidualQLearningConfig | None = None,
) -> FuzzyResidualQLearningResult:
    """Train residual Q-learning with fuzzy state aggregation."""

    cfg = config or FuzzyResidualQLearningConfig()
    fuzzy_encoder = encoder or FuzzyDynamicStateEncoder()
    actions = residual_acceleration_actions(cfg.residual_acceleration_scale)
    rng = np.random.default_rng(cfg.seed)
    desired_q = _desired_q(env_config)

    q_value = np.full(
        (fuzzy_encoder.n_rules, len(actions)),
        cfg.initial_q_value,
        dtype=float,
    )
    episode_returns = np.zeros(cfg.episodes, dtype=float)
    episode_lengths = np.zeros(cfg.episodes, dtype=int)
    episode_success = np.zeros(cfg.episodes, dtype=bool)
    epsilon_history = np.zeros(cfg.episodes, dtype=float)

    env = Arm2DOFDynamicEnv(env_config)
    for episode in range(cfg.episodes):
        epsilon = fuzzy_residual_epsilon_at_episode(cfg, episode)
        epsilon_history[episode] = epsilon
        controller = _make_fuzzy_controller(cfg)
        observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
        rule_indices, rule_weights = fuzzy_encoder.active_rules_from_observation(
            observation,
            desired_q,
        )
        previous_distance = float(observation["distance"])
        total_reward = 0.0
        done = False
        step_count = 0

        for step_count in range(1, cfg.max_steps_per_episode + 1):
            if rng.random() < epsilon:
                action = int(rng.integers(len(actions)))
            else:
                action_values = aggregate_fuzzy_q_values(
                    q_value,
                    rule_indices,
                    rule_weights,
                )
                action = _random_argmax(action_values, rng)

            base_acceleration = controller.compute(desired_q, observation["q"], env_config.dt)
            residual_acceleration = actions[action]
            torque = inverse_dynamics_torque(
                observation["q"],
                observation["q_dot"],
                base_acceleration + residual_acceleration,
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
            next_rule_indices, next_rule_weights = fuzzy_encoder.active_rules_from_observation(
                next_observation,
                desired_q,
            )
            current_value = float(
                np.dot(rule_weights, q_value[rule_indices, action])
            )
            if done:
                target = reward
            else:
                next_action_values = aggregate_fuzzy_q_values(
                    q_value,
                    next_rule_indices,
                    next_rule_weights,
                )
                target = reward + cfg.gamma * float(np.max(next_action_values))
            td_error = target - current_value
            q_value[rule_indices, action] += cfg.alpha * rule_weights * td_error

            total_reward += reward
            observation = next_observation
            rule_indices = next_rule_indices
            rule_weights = next_rule_weights
            previous_distance = distance
            if done:
                break

        episode_returns[episode] = total_reward
        episode_lengths[episode] = step_count
        episode_success[episode] = done

    rule_policy = np.argmax(q_value, axis=1).astype(int, copy=False)
    return FuzzyResidualQLearningResult(
        q_value=q_value,
        rule_policy=rule_policy,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_success=episode_success,
        epsilon_history=epsilon_history,
        desired_q=desired_q,
        encoder=fuzzy_encoder,
    )


def rollout_fuzzy_residual_q_policy(
    env_config: Arm2DOFDynamicEnvConfig,
    q_value: np.ndarray,
    encoder: FuzzyDynamicStateEncoder,
    config: FuzzyResidualQLearningConfig | None = None,
    desired_q: ArrayLike2 | None = None,
) -> FuzzyResidualRollout:
    """Run one greedy rollout from a fuzzy residual Q table."""

    cfg = config or FuzzyResidualQLearningConfig()
    actions = residual_acceleration_actions(cfg.residual_acceleration_scale)
    q_table = np.asarray(q_value, dtype=float)
    if q_table.shape != (encoder.n_rules, len(actions)):
        raise ValueError("q_value has an invalid shape.")

    goal_q = _desired_q(env_config) if desired_q is None else np.asarray(desired_q, dtype=float)
    if goal_q.shape != (2,):
        raise ValueError("desired_q must contain exactly two values.")

    env = Arm2DOFDynamicEnv(env_config)
    controller = _make_fuzzy_controller(cfg)
    observation = env.reset(q=cfg.start_q, q_dot=cfg.start_q_dot)
    rule_indices, rule_weights = encoder.active_rules_from_observation(
        observation,
        goal_q,
    )
    previous_distance = float(observation["distance"])

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    speed_history = [float(observation["speed"])]
    torque_history: list[np.ndarray] = []
    action_indices: list[int] = []
    dominant_rule_indices: list[int] = []
    rewards: list[float] = []
    done = False
    info: dict[str, object] = {"truncated": False}

    for _ in range(cfg.max_steps_per_episode):
        action_values = aggregate_fuzzy_q_values(q_table, rule_indices, rule_weights)
        action = int(np.argmax(action_values))
        base_acceleration = controller.compute(goal_q, observation["q"], env_config.dt)
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
        dominant_rule_indices.append(int(rule_indices[np.argmax(rule_weights)]))
        rewards.append(reward)
        rule_indices, rule_weights = encoder.active_rules_from_observation(
            observation,
            goal_q,
        )
        previous_distance = distance
        if done:
            break

    return FuzzyResidualRollout(
        q_history=np.asarray(q_history),
        ee_history=np.asarray(ee_history),
        distance_history=np.asarray(distance_history),
        speed_history=np.asarray(speed_history),
        torque_history=np.asarray(torque_history),
        action_indices=action_indices,
        dominant_rule_indices=dominant_rule_indices,
        rewards=rewards,
        done=done,
        truncated=bool(info.get("truncated", False)),
    )


__all__ = [
    "FUZZY_STATE_VARIABLES",
    "FUZZY_TERMS",
    "FuzzyDynamicStateEncoder",
    "FuzzyResidualQLearningConfig",
    "FuzzyResidualQLearningResult",
    "FuzzyResidualRollout",
    "RESIDUAL_ACTION_DIRECTIONS",
    "RESIDUAL_ACTION_NAMES",
    "aggregate_fuzzy_q_values",
    "fuzzy_residual_epsilon_at_episode",
    "rollout_fuzzy_residual_q_policy",
    "train_fuzzy_residual_q_learning",
]
