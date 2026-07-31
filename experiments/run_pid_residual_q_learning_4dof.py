"""Train residual Q-learning on top of fuzzy gain-scheduled PID for 4-DOF."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

matplotlib.use("Agg")

from envs import Arm4DOFDynamicEnvConfig
from rl import (
    PID_RESIDUAL_ACTION_NAMES_4DOF,
    PIDResidualQLearning4DOFConfig,
    PIDResidualSafety4DOFConfig,
    PIDResidualStateEncoder4DOF,
    residual_acceleration_actions_4dof,
    rollout_pid_residual_q_policy_4dof,
    train_pid_residual_q_learning_4dof,
)
from visualization import plot_control_simulation_4dof


def _moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    effective_window = min(max(1, window), values.size)
    kernel = np.ones(effective_window, dtype=float) / effective_window
    averaged = np.convolve(values.astype(float), kernel, mode="valid")
    x = np.arange(effective_window - 1, values.size)
    return x, averaged


def _save_learning_plot(
    output_path: Path,
    returns: np.ndarray,
    success: np.ndarray,
    learned_rollout,
    baseline_rollout,
    tolerance: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_return, ax_success, ax_distance, ax_action = axes.ravel()

    x_return, mean_return = _moving_average(returns, window=8)
    ax_return.plot(x_return + 1, mean_return, label="return moyen")
    ax_return.grid(True, alpha=0.3)
    ax_return.set_title("Apprentissage residuel")
    ax_return.set_xlabel("episode")
    ax_return.set_ylabel("return")
    ax_return.legend()

    x_success, success_rate = _moving_average(success.astype(float), window=8)
    ax_success.plot(x_success + 1, success_rate, color="tab:green", label="succes")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.grid(True, alpha=0.3)
    ax_success.set_title("Taux de succes")
    ax_success.set_xlabel("episode")
    ax_success.set_ylabel("succes")
    ax_success.legend()

    ax_distance.plot(
        baseline_rollout.distance_history,
        linestyle="--",
        label="PID adapte",
    )
    ax_distance.plot(learned_rollout.distance_history, label="PID adapte + RL")
    ax_distance.axhline(tolerance, linestyle=":", color="tab:green", label="tolerance")
    ax_distance.grid(True, alpha=0.3)
    ax_distance.set_title("Distance cible")
    ax_distance.set_xlabel("iteration")
    ax_distance.set_ylabel("distance")
    ax_distance.legend()

    if learned_rollout.action_indices:
        ax_action.step(
            np.arange(len(learned_rollout.action_indices)),
            learned_rollout.action_indices,
            where="post",
            label="action RL",
        )
    ax_action.grid(True, alpha=0.3)
    ax_action.set_title("Action residuelle")
    ax_action.set_xlabel("iteration")
    ax_action.set_ylabel("index action")
    ax_action.legend()

    fig.suptitle("Q-learning residuel sur PID adapte - bras 4DDL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    env_config = Arm4DOFDynamicEnvConfig(
        target=(1.15, 0.45, 0.55),
        dt=0.01,
        max_torque=(55.0, 85.0, 60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=900,
    )
    encoder = PIDResidualStateEncoder4DOF(
        joint_error_deadband=(0.04, 0.05, 0.05, 0.05),
        speed_bins=(0.12, 0.75),
    )
    learning_config = PIDResidualQLearning4DOFConfig(
        episodes=45,
        max_steps_per_episode=900,
        alpha=0.25,
        gamma=0.97,
        epsilon_start=0.70,
        epsilon_end=0.05,
        epsilon_decay=0.965,
        residual_acceleration_scale=(0.30, 0.40, 0.40, 0.30),
        seed=31,
    )

    result = train_pid_residual_q_learning_4dof(
        env_config,
        encoder=encoder,
        config=learning_config,
    )
    learned_rollout = rollout_pid_residual_q_policy_4dof(
        env_config,
        result.q_value,
        encoder,
        config=learning_config,
        desired_q=result.desired_q,
        safety_config=PIDResidualSafety4DOFConfig(patience=80, min_progress=1e-4),
    )
    zero_q_value = np.zeros_like(result.q_value)
    baseline_rollout = rollout_pid_residual_q_policy_4dof(
        env_config,
        zero_q_value,
        encoder,
        config=learning_config,
        desired_q=result.desired_q,
    )

    output_path = ROOT / "results" / "figures" / "step_25_pid_residual_q_learning_4dof.png"
    learning_output_path = (
        ROOT / "results" / "figures" / "step_25_pid_residual_q_learning_4dof_learning.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_control_simulation_4dof(
        learned_rollout.q_history,
        learned_rollout.ee_history,
        learned_rollout.distance_history,
        learned_rollout.torque_history,
        env_config.target,
        link_lengths=env_config.arm_config.link_lengths,
        tolerance=env_config.target_tolerance,
        title="PID adapte + RL residuel - bras 4DDL spatial",
        action_ylabel="N.m",
    )
    fig.savefig(output_path, dpi=150)
    _save_learning_plot(
        learning_output_path,
        result.episode_returns,
        result.episode_success,
        learned_rollout,
        baseline_rollout,
        env_config.target_tolerance,
    )

    last_window = min(15, learning_config.episodes)
    success_rate = float(np.mean(result.episode_success[-last_window:]))
    mean_episode_length = float(np.mean(result.episode_lengths[-last_window:]))
    mean_return = float(np.mean(result.episode_returns[-last_window:]))
    learned_mean_torque = float(np.mean(np.linalg.norm(learned_rollout.torque_history, axis=1)))
    baseline_mean_torque = float(np.mean(np.linalg.norm(baseline_rollout.torque_history, axis=1)))
    learned_actions = [
        PID_RESIDUAL_ACTION_NAMES_4DOF[action] for action in learned_rollout.action_indices
    ]
    residual_actions = residual_acceleration_actions_4dof(
        learning_config.residual_acceleration_scale
    )

    print(f"state_count={encoder.n_states}")
    print(f"action_count={len(PID_RESIDUAL_ACTION_NAMES_4DOF)}")
    print(f"episodes={learning_config.episodes}")
    print(f"epsilon_final={result.epsilon_history[-1]:.3f}")
    print(f"residual_actions={np.array2string(residual_actions, precision=3)}")
    print(f"desired_joint_angles_rad={np.array2string(result.desired_q, precision=6)}")
    print(f"success_rate_last_{last_window}={success_rate:.3f}")
    print(f"mean_episode_length_last_{last_window}={mean_episode_length:.3f}")
    print(f"mean_return_last_{last_window}={mean_return:.12e}")
    print(f"learned_done={learned_rollout.done}")
    print(f"learned_truncated={learned_rollout.truncated}")
    print(f"learned_steps={len(learned_rollout.action_indices)}")
    print(f"learned_final_distance={learned_rollout.distance_history[-1]:.12e}")
    print(f"learned_final_speed={learned_rollout.speed_history[-1]:.12e}")
    print(f"learned_mean_torque_norm={learned_mean_torque:.12e}")
    print(f"learned_residual_disabled={learned_rollout.residual_disabled}")
    print(f"learned_residual_switch_step={learned_rollout.residual_switch_step}")
    print(f"baseline_done={baseline_rollout.done}")
    print(f"baseline_steps={len(baseline_rollout.action_indices)}")
    print(f"baseline_final_distance={baseline_rollout.distance_history[-1]:.12e}")
    print(f"baseline_final_speed={baseline_rollout.speed_history[-1]:.12e}")
    print(f"baseline_mean_torque_norm={baseline_mean_torque:.12e}")
    print(f"learned_unique_actions={sorted(set(learned_actions))}")
    print(f"figure={output_path}")
    print(f"learning_figure={learning_output_path}")

    return 0 if learned_rollout.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
