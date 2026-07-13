"""Elementary RL tools for the robotic-arm simulations."""

from .discrete_arm_mdp import (
    ACTION_DELTAS,
    ACTION_NAMES,
    DiscreteArm2DOFMDP,
    DiscreteArm2DOFMDPConfig,
)
from .dynamic_programming import (
    PolicyEvaluationResult,
    Rollout,
    ValueIterationResult,
    bellman_action_values,
    discounted_return,
    evaluate_policy,
    rollout_policy,
    undiscounted_return,
    value_iteration,
)
from .dynamic_residual_q_learning import (
    RESIDUAL_ACTION_DIRECTIONS,
    RESIDUAL_ACTION_NAMES,
    DynamicArmStateDiscretizer,
    DynamicResidualQLearningConfig,
    DynamicResidualQLearningResult,
    DynamicResidualRollout,
    residual_acceleration_actions,
    rollout_dynamic_residual_policy,
    train_dynamic_residual_q_learning,
)
from .q_learning import (
    QLearningConfig,
    QLearningResult,
    epsilon_at_episode,
    greedy_policy_from_q,
    train_q_learning,
)

__all__ = [
    "ACTION_DELTAS",
    "ACTION_NAMES",
    "DiscreteArm2DOFMDP",
    "DiscreteArm2DOFMDPConfig",
    "DynamicArmStateDiscretizer",
    "DynamicResidualQLearningConfig",
    "DynamicResidualQLearningResult",
    "DynamicResidualRollout",
    "PolicyEvaluationResult",
    "QLearningConfig",
    "QLearningResult",
    "RESIDUAL_ACTION_DIRECTIONS",
    "RESIDUAL_ACTION_NAMES",
    "Rollout",
    "ValueIterationResult",
    "bellman_action_values",
    "discounted_return",
    "epsilon_at_episode",
    "evaluate_policy",
    "greedy_policy_from_q",
    "residual_acceleration_actions",
    "rollout_dynamic_residual_policy",
    "rollout_policy",
    "train_dynamic_residual_q_learning",
    "train_q_learning",
    "undiscounted_return",
    "value_iteration",
]
