"""Tabular Q-learning for finite robotic-arm MDPs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .discrete_arm_mdp import DiscreteArm2DOFMDP


@dataclass(frozen=True)
class QLearningConfig:
    """Hyperparameters for tabular Q-learning."""

    episodes: int = 5_000
    max_steps_per_episode: int = 80
    alpha: float = 0.55
    gamma: float | None = None
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.996
    random_start_states: bool = False
    initial_q_value: float = 0.0
    seed: int | None = 7

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be strictly positive.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be strictly positive.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1.")
        if self.gamma is not None and not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma <= 1.")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1."
            )
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must satisfy 0 < epsilon_decay <= 1.")


@dataclass(frozen=True)
class QLearningResult:
    """Learned Q table and episode-level training traces."""

    q_value: np.ndarray
    policy: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    episode_success: np.ndarray
    epsilon_history: np.ndarray


def epsilon_at_episode(config: QLearningConfig, episode_index: int) -> float:
    """Return exponentially decayed epsilon for a zero-based episode index."""

    if episode_index < 0:
        raise ValueError("episode_index must be non-negative.")
    return float(
        max(
            config.epsilon_end,
            config.epsilon_start * (config.epsilon_decay**episode_index),
        )
    )


def greedy_policy_from_q(
    q_value: np.ndarray,
    terminal_states: np.ndarray | None = None,
) -> np.ndarray:
    """Return a deterministic greedy policy from a Q table."""

    if q_value.ndim != 2:
        raise ValueError("q_value must be a 2D array.")
    policy = np.argmax(q_value, axis=1)
    if terminal_states is not None:
        terminal_mask = np.asarray(terminal_states, dtype=bool)
        if terminal_mask.shape != (q_value.shape[0],):
            raise ValueError("terminal_states has an invalid shape.")
        policy[terminal_mask] = 0
    return policy.astype(int, copy=False)


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    maximum = float(np.max(values))
    candidates = np.flatnonzero(np.isclose(values, maximum))
    return int(rng.choice(candidates))


def _epsilon_greedy_action(
    q_value: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_value.shape[1]))
    return _random_argmax(q_value[state], rng)


def train_q_learning(
    mdp: DiscreteArm2DOFMDP,
    start_state: int | None = None,
    config: QLearningConfig | None = None,
) -> QLearningResult:
    """Learn Q(s, a) through sampled interaction with a finite MDP."""

    cfg = config or QLearningConfig()
    gamma = mdp.config.gamma if cfg.gamma is None else cfg.gamma
    rng = np.random.default_rng(cfg.seed)

    if start_state is not None:
        if start_state < 0 or start_state >= mdp.n_states:
            raise ValueError("start_state is outside the discrete state space.")
        if mdp.is_terminal(start_state):
            raise ValueError("start_state must be non-terminal.")

    non_terminal_states = np.flatnonzero(~mdp.terminal_states)
    if non_terminal_states.size == 0:
        raise ValueError("mdp must contain at least one non-terminal state.")

    q_value = np.full(
        (mdp.n_states, mdp.n_actions),
        cfg.initial_q_value,
        dtype=float,
    )
    q_value[mdp.terminal_states] = 0.0

    episode_returns = np.zeros(cfg.episodes, dtype=float)
    episode_lengths = np.zeros(cfg.episodes, dtype=int)
    episode_success = np.zeros(cfg.episodes, dtype=bool)
    epsilon_history = np.zeros(cfg.episodes, dtype=float)

    for episode in range(cfg.episodes):
        epsilon = epsilon_at_episode(cfg, episode)
        epsilon_history[episode] = epsilon
        if cfg.random_start_states or start_state is None:
            state = int(rng.choice(non_terminal_states))
        else:
            state = int(start_state)

        total_reward = 0.0
        done = False
        step_count = 0

        for step_count in range(1, cfg.max_steps_per_episode + 1):
            action = _epsilon_greedy_action(q_value, state, epsilon, rng)
            next_state, reward, done = mdp.step(state, action)
            continuation = 0.0 if done else gamma * float(np.max(q_value[next_state]))
            target = reward + continuation
            q_value[state, action] += cfg.alpha * (target - q_value[state, action])

            total_reward += reward
            state = next_state
            if done:
                break

        episode_returns[episode] = total_reward
        episode_lengths[episode] = step_count
        episode_success[episode] = done

    q_value[mdp.terminal_states] = 0.0
    policy = greedy_policy_from_q(q_value, mdp.terminal_states)
    return QLearningResult(
        q_value=q_value,
        policy=policy,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_success=episode_success,
        epsilon_history=epsilon_history,
    )
