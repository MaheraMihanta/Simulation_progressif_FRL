"""Discrete MDP formulation for the planar 2-DOF arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np

from robot import Arm2DOFConfig, forward_kinematics, is_reachable
from robot.kinematics import ArrayLike2


ACTION_DELTAS = np.array(
    [
        [0, 0],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=int,
)

ACTION_NAMES = (
    "stay",
    "q1-",
    "q1+",
    "q2-",
    "q2+",
    "q1-/q2-",
    "q1-/q2+",
    "q1+/q2-",
    "q1+/q2+",
)


@dataclass(frozen=True)
class DiscreteArm2DOFMDPConfig:
    """Configuration of the first tabular RL problem."""

    arm_config: Arm2DOFConfig = field(
        default_factory=lambda: Arm2DOFConfig(
            joint_limits=((-pi, pi), (-pi, pi)),
        )
    )
    target: tuple[float, float] = (1.1, 0.55)
    bins_per_joint: tuple[int, int] = (31, 31)
    target_tolerance: float = 0.09
    gamma: float = 0.95
    distance_weight: float = 1.0
    action_penalty: float = 0.02
    goal_reward: float = 10.0

    def __post_init__(self) -> None:
        if len(self.bins_per_joint) != 2:
            raise ValueError("bins_per_joint must contain exactly two values.")
        if any(count < 3 for count in self.bins_per_joint):
            raise ValueError("each joint must have at least three discrete bins.")
        if self.target_tolerance <= 0.0:
            raise ValueError("target_tolerance must be strictly positive.")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1.")
        if self.action_penalty < 0.0:
            raise ValueError("action_penalty must be non-negative.")
        if not is_reachable(self.target, self.arm_config.link_lengths):
            raise ValueError("target is outside the reachable workspace.")


class DiscreteArm2DOFMDP:
    """Finite MDP used to demonstrate elementary RL notions."""

    action_deltas = ACTION_DELTAS
    action_names = ACTION_NAMES

    def __init__(self, config: DiscreteArm2DOFMDPConfig | None = None) -> None:
        self.config = config or DiscreteArm2DOFMDPConfig()
        self.q1_values = np.linspace(
            self.config.arm_config.joint_limits[0][0],
            self.config.arm_config.joint_limits[0][1],
            self.config.bins_per_joint[0],
        )
        self.q2_values = np.linspace(
            self.config.arm_config.joint_limits[1][0],
            self.config.arm_config.joint_limits[1][1],
            self.config.bins_per_joint[1],
        )
        self.grid_shape = self.config.bins_per_joint
        self.n_states = self.grid_shape[0] * self.grid_shape[1]
        self.n_actions = len(self.action_names)
        self.target = np.asarray(self.config.target, dtype=float)
        self.q_step = np.array(
            [
                self.q1_values[1] - self.q1_values[0],
                self.q2_values[1] - self.q2_values[0],
            ],
            dtype=float,
        )

        self.end_effector_positions = np.vstack(
            [
                forward_kinematics(
                    self.q_for_state(state),
                    self.config.arm_config.link_lengths,
                )
                for state in range(self.n_states)
            ]
        )
        self.distances = np.linalg.norm(
            self.end_effector_positions - self.target,
            axis=1,
        )
        self.terminal_states = self.distances <= self.config.target_tolerance
        if not np.any(self.terminal_states):
            raise ValueError(
                "no discrete state reaches the target; increase target_tolerance "
                "or bins_per_joint."
            )
        self.next_states, self.rewards, self.done = self._build_transition_table()

    def state_index(self, q1_index: int, q2_index: int) -> int:
        q1_index = int(np.clip(q1_index, 0, self.grid_shape[0] - 1))
        q2_index = int(np.clip(q2_index, 0, self.grid_shape[1] - 1))
        return q1_index * self.grid_shape[1] + q2_index

    def state_indices(self, state: int) -> tuple[int, int]:
        if state < 0 or state >= self.n_states:
            raise ValueError("state is outside the discrete state space.")
        return divmod(int(state), self.grid_shape[1])

    def q_for_state(self, state: int) -> np.ndarray:
        q1_index, q2_index = self.state_indices(state)
        return np.array(
            [self.q1_values[q1_index], self.q2_values[q2_index]],
            dtype=float,
        )

    def nearest_state(self, q: ArrayLike2) -> int:
        q_array = np.asarray(q, dtype=float)
        if q_array.shape != (2,):
            raise ValueError("q must contain exactly two values.")
        q1_index = int(np.argmin(np.abs(self.q1_values - q_array[0])))
        q2_index = int(np.argmin(np.abs(self.q2_values - q_array[1])))
        return self.state_index(q1_index, q2_index)

    def is_terminal(self, state: int) -> bool:
        return bool(self.terminal_states[state])

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        if action < 0 or action >= self.n_actions:
            raise ValueError("action is outside the discrete action space.")
        if not hasattr(self, "next_states"):
            return self._compute_transition(state, action)
        return (
            int(self.next_states[state, action]),
            float(self.rewards[state, action]),
            bool(self.done[state, action]),
        )

    def _compute_transition(self, state: int, action: int) -> tuple[int, float, bool]:
        if self.is_terminal(state):
            return state, 0.0, True

        q1_index, q2_index = self.state_indices(state)
        delta = self.action_deltas[action]
        next_state = self.state_index(q1_index + delta[0], q2_index + delta[1])
        done = self.is_terminal(next_state)

        effort = float(np.linalg.norm(delta * self.q_step))
        reward = (
            -self.config.distance_weight * float(self.distances[next_state])
            -self.config.action_penalty * effort
        )
        if done:
            reward += self.config.goal_reward
        return next_state, float(reward), done

    def _build_transition_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_states = np.zeros((self.n_states, self.n_actions), dtype=int)
        rewards = np.zeros((self.n_states, self.n_actions), dtype=float)
        done = np.zeros((self.n_states, self.n_actions), dtype=bool)

        for state in range(self.n_states):
            for action in range(self.n_actions):
                next_state, reward, is_done = self._compute_transition(state, action)
                next_states[state, action] = next_state
                rewards[state, action] = reward
                done[state, action] = is_done
        return next_states, rewards, done

    def transition_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.next_states.copy(), self.rewards.copy(), self.done.copy()

    def state_summary(self, state: int) -> dict[str, object]:
        q = self.q_for_state(state)
        end_effector = self.end_effector_positions[state]
        return {
            "state": int(state),
            "q": q,
            "end_effector": end_effector,
            "target": self.target.copy(),
            "distance": float(self.distances[state]),
            "terminal": self.is_terminal(state),
        }
