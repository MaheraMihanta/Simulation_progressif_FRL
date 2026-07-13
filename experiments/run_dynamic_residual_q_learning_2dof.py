"""Train residual Q-learning on the dynamic 2-DOF arm."""

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

from matplotlib.patches import Circle

from envs import Arm2DOFDynamicEnvConfig
from rl import (
    RESIDUAL_ACTION_NAMES,
    DynamicArmStateDiscretizer,
    DynamicResidualQLearningConfig,
    residual_acceleration_actions,
    rollout_dynamic_residual_policy,
    train_dynamic_residual_q_learning,
)
from robot import joint_positions, workspace_radius


def _moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    effective_window = min(max(1, window), values.size)
    kernel = np.ones(effective_window, dtype=float) / effective_window
    averaged = np.convolve(values.astype(float), kernel, mode="valid")
    x = np.arange(effective_window - 1, values.size)
    return x, averaged


def _plot_trajectory(ax, learned_rollout, baseline_rollout, target, link_lengths) -> None:
    ax.plot(
        baseline_rollout.ee_history[:, 0],
        baseline_rollout.ee_history[:, 1],
        linestyle="--",
        linewidth=1.8,
        label="PID dynamique",
    )
    ax.plot(
        learned_rollout.ee_history[:, 0],
        learned_rollout.ee_history[:, 1],
        linewidth=2.2,
        label="PID + residu Q",
    )
    ax.scatter([target[0]], [target[1]], marker="x", s=100, label="cible")
    ax.scatter(
        [learned_rollout.ee_history[0, 0]],
        [learned_rollout.ee_history[0, 1]],
        marker="o",
        s=60,
        label="depart",
    )
    ax.scatter(
        [learned_rollout.ee_history[-1, 0]],
        [learned_rollout.ee_history[-1, 1]],
        marker="s",
        s=60,
        label="final",
    )

    indices = np.linspace(
        0,
        len(learned_rollout.q_history) - 1,
        min(7, len(learned_rollout.q_history)),
        dtype=int,
    )
    for index in indices:
        positions = joint_positions(learned_rollout.q_history[index], link_lengths)
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            color="tab:gray",
            alpha=0.22 if index != indices[-1] else 0.9,
            linewidth=1.4 if index != indices[-1] else 3.0,
            markersize=5,
        )

    _, r_max = workspace_radius(link_lengths)
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            r_max,
            fill=False,
            linestyle="--",
            linewidth=1,
            alpha=0.3,
        )
    )
    reach = sum(link_lengths)
    margin = 0.15 * reach
    ax.set_xlim(-reach - margin, reach + margin)
    ax.set_ylim(-reach - margin, reach + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Trajectoire dynamique")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")


def _plot_distance(ax, learned_rollout, baseline_rollout, tolerance: float) -> None:
    ax.plot(
        baseline_rollout.distance_history,
        linestyle="--",
        label="distance PID",
    )
    ax.plot(learned_rollout.distance_history, label="distance PID + Q")
    ax.axhline(tolerance, linestyle=":", linewidth=1, color="tab:green", label="tolerance")
    ax.grid(True, alpha=0.3)
    ax.set_title("Erreur de position")
    ax.set_xlabel("iteration")
    ax.set_ylabel("distance")
    ax.legend()


def _plot_torque_and_actions(ax, learned_rollout, baseline_rollout) -> None:
    learned_torque_norm = np.linalg.norm(learned_rollout.torque_history, axis=1)
    baseline_torque_norm = np.linalg.norm(baseline_rollout.torque_history, axis=1)
    ax.plot(baseline_torque_norm, linestyle="--", label="norme couple PID")
    ax.plot(learned_torque_norm, label="norme couple PID + Q")
    ax.set_title("Couple moteur et action RL")
    ax.set_xlabel("iteration")
    ax.set_ylabel("N.m")
    ax.grid(True, alpha=0.3)

    ax_action = ax.twinx()
    if learned_rollout.action_indices:
        ax_action.step(
            np.arange(len(learned_rollout.action_indices)),
            learned_rollout.action_indices,
            where="post",
            color="tab:green",
            alpha=0.45,
            label="action RL",
        )
    ax_action.set_ylabel("action")
    ax_action.set_yticks(range(len(RESIDUAL_ACTION_NAMES)))

    lines, labels = ax.get_legend_handles_labels()
    action_lines, action_labels = ax_action.get_legend_handles_labels()
    ax.legend(lines + action_lines, labels + action_labels, loc="upper right")


def _plot_learning(ax, returns: np.ndarray, success: np.ndarray) -> None:
    x_return, mean_return = _moving_average(returns, window=20)
    x_success, success_rate = _moving_average(success.astype(float), window=20)
    ax.plot(x_return + 1, mean_return, color="tab:blue", label="return moyen")
    ax.set_title("Apprentissage Q residuel")
    ax.set_xlabel("episode")
    ax.set_ylabel("return moyen")
    ax.grid(True, alpha=0.3)

    ax_success = ax.twinx()
    ax_success.plot(
        x_success + 1,
        success_rate,
        color="tab:green",
        alpha=0.85,
        label="taux de succes",
    )
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.set_ylabel("succes")

    lines, labels = ax.get_legend_handles_labels()
    success_lines, success_labels = ax_success.get_legend_handles_labels()
    ax.legend(lines + success_lines, labels + success_labels, loc="lower right")


def main() -> int:
    env_config = Arm2DOFDynamicEnvConfig(
        target=(1.1, 0.55),
        dt=0.01,
        max_torque=(60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=450,
    )
    discretizer = DynamicArmStateDiscretizer(
        error_bins=(15, 15),
        velocity_bins=(7, 7),
        velocity_limits=((-6.0, 6.0), (-6.0, 6.0)),
    )
    learning_config = DynamicResidualQLearningConfig(
        episodes=180,
        max_steps_per_episode=450,
        alpha=0.45,
        gamma=0.97,
        epsilon_start=0.8,
        epsilon_end=0.04,
        epsilon_decay=0.985,
        residual_acceleration_scale=(2.0, 2.0),
        seed=11,
    )

    result = train_dynamic_residual_q_learning(
        env_config,
        discretizer=discretizer,
        config=learning_config,
    )
    learned_rollout = rollout_dynamic_residual_policy(
        env_config,
        result.policy,
        discretizer,
        config=learning_config,
        desired_q=result.desired_q,
    )
    zero_policy = np.zeros(discretizer.n_states, dtype=int)
    baseline_rollout = rollout_dynamic_residual_policy(
        env_config,
        zero_policy,
        discretizer,
        config=learning_config,
        desired_q=result.desired_q,
    )

    output_path = ROOT / "results" / "figures" / "step_09_dynamic_residual_q_learning_2dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _plot_trajectory(
        axes[0, 0],
        learned_rollout,
        baseline_rollout,
        env_config.target,
        env_config.arm_config.link_lengths,
    )
    _plot_distance(
        axes[0, 1],
        learned_rollout,
        baseline_rollout,
        env_config.target_tolerance,
    )
    _plot_torque_and_actions(axes[1, 0], learned_rollout, baseline_rollout)
    _plot_learning(
        axes[1, 1],
        result.episode_returns,
        result.episode_success,
    )
    fig.suptitle("Q-learning dynamique residuel - bras 2DDL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    last_window = min(60, learning_config.episodes)
    success_rate = float(np.mean(result.episode_success[-last_window:]))
    mean_episode_length = float(np.mean(result.episode_lengths[-last_window:]))
    mean_return = float(np.mean(result.episode_returns[-last_window:]))
    learned_mean_torque = float(np.mean(np.linalg.norm(learned_rollout.torque_history, axis=1)))
    baseline_mean_torque = float(np.mean(np.linalg.norm(baseline_rollout.torque_history, axis=1)))
    learned_actions = [RESIDUAL_ACTION_NAMES[action] for action in learned_rollout.action_indices]
    unique_actions = sorted(set(learned_actions))
    residual_actions = residual_acceleration_actions(
        learning_config.residual_acceleration_scale
    )

    print(f"state_count={discretizer.n_states}")
    print(f"action_count={len(RESIDUAL_ACTION_NAMES)}")
    print(f"episodes={learning_config.episodes}")
    print(f"alpha={learning_config.alpha:.3f}")
    print(f"gamma={learning_config.gamma:.3f}")
    print(f"epsilon_start={learning_config.epsilon_start:.3f}")
    print(f"epsilon_end={learning_config.epsilon_end:.3f}")
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
    print(f"baseline_done={baseline_rollout.done}")
    print(f"baseline_steps={len(baseline_rollout.action_indices)}")
    print(f"baseline_final_distance={baseline_rollout.distance_history[-1]:.12e}")
    print(f"baseline_final_speed={baseline_rollout.speed_history[-1]:.12e}")
    print(f"baseline_mean_torque_norm={baseline_mean_torque:.12e}")
    print(f"learned_unique_actions={unique_actions}")
    print(f"figure={output_path}")

    return 0 if learned_rollout.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
