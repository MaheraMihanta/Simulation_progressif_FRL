"""Train tabular Q-learning on the discrete 2-DOF arm MDP."""

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

from rl import (
    DiscreteArm2DOFMDP,
    DiscreteArm2DOFMDPConfig,
    QLearningConfig,
    discounted_return,
    rollout_policy,
    train_q_learning,
    undiscounted_return,
    value_iteration,
)
from robot import joint_positions, workspace_radius


def _moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or values.size <= 1:
        return np.arange(values.size), values.astype(float, copy=False)
    effective_window = min(window, values.size)
    kernel = np.ones(effective_window, dtype=float) / effective_window
    averaged = np.convolve(values.astype(float), kernel, mode="valid")
    x = np.arange(effective_window - 1, values.size)
    return x, averaged


def _plot_q_heatmap(
    ax,
    mdp: DiscreteArm2DOFMDP,
    q_value: np.ndarray,
    start_state: int,
) -> None:
    q_max_grid = np.max(q_value, axis=1).reshape(mdp.grid_shape).T
    extent = [
        mdp.q1_values[0],
        mdp.q1_values[-1],
        mdp.q2_values[0],
        mdp.q2_values[-1],
    ]
    image = ax.imshow(
        q_max_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    terminal_q = np.asarray(
        [mdp.q_for_state(state) for state in np.flatnonzero(mdp.terminal_states)]
    )
    start_q = mdp.q_for_state(start_state)
    ax.scatter([start_q[0]], [start_q[1]], c="white", edgecolors="black", label="start")
    ax.scatter(terminal_q[:, 0], terminal_q[:, 1], c="red", marker="x", s=80, label="target state")
    ax.set_title("Learned max Q(s, a)")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.legend(loc="upper right")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_policy(
    ax,
    mdp: DiscreteArm2DOFMDP,
    policy: np.ndarray,
    start_state: int,
) -> None:
    skip = 2
    q1_points: list[float] = []
    q2_points: list[float] = []
    u: list[int] = []
    v: list[int] = []

    for q1_index in range(0, mdp.grid_shape[0], skip):
        for q2_index in range(0, mdp.grid_shape[1], skip):
            state = mdp.state_index(q1_index, q2_index)
            if mdp.is_terminal(state):
                continue
            delta = mdp.action_deltas[int(policy[state])]
            q1_points.append(float(mdp.q1_values[q1_index]))
            q2_points.append(float(mdp.q2_values[q2_index]))
            u.append(int(delta[0]))
            v.append(int(delta[1]))

    ax.quiver(q1_points, q2_points, u, v, angles="xy", scale_units="xy", scale=2.5)
    terminal_q = np.asarray(
        [mdp.q_for_state(state) for state in np.flatnonzero(mdp.terminal_states)]
    )
    start_q = mdp.q_for_state(start_state)
    ax.scatter([start_q[0]], [start_q[1]], c="white", edgecolors="black", label="start")
    ax.scatter(terminal_q[:, 0], terminal_q[:, 1], c="red", marker="x", s=80, label="target state")
    ax.set_xlim(mdp.q1_values[0], mdp.q1_values[-1])
    ax.set_ylim(mdp.q2_values[0], mdp.q2_values[-1])
    ax.grid(True, alpha=0.25)
    ax.set_title("Greedy policy from Q-learning")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.legend(loc="upper right")


def _plot_cartesian_rollout(
    ax,
    mdp: DiscreteArm2DOFMDP,
    states: list[int],
) -> None:
    q_history = np.asarray([mdp.q_for_state(state) for state in states])
    ee_history = np.asarray([mdp.end_effector_positions[state] for state in states])

    ax.plot(ee_history[:, 0], ee_history[:, 1], linewidth=2, label="trajectory")
    ax.scatter([mdp.target[0]], [mdp.target[1]], marker="x", s=100, label="target")
    ax.scatter([ee_history[0, 0]], [ee_history[0, 1]], marker="o", s=60, label="start")
    ax.scatter([ee_history[-1, 0]], [ee_history[-1, 1]], marker="s", s=60, label="final")

    indices = np.linspace(0, len(q_history) - 1, min(6, len(q_history)), dtype=int)
    for index in indices:
        positions = joint_positions(q_history[index], mdp.config.arm_config.link_lengths)
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            color="tab:gray",
            alpha=0.25 if index != indices[-1] else 0.9,
            linewidth=1.5 if index != indices[-1] else 3.0,
            markersize=5,
        )

    _, r_max = workspace_radius(mdp.config.arm_config.link_lengths)
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
    reach = sum(mdp.config.arm_config.link_lengths)
    margin = 0.15 * reach
    ax.set_xlim(-reach - margin, reach + margin)
    ax.set_ylim(-reach - margin, reach + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Greedy rollout in Cartesian space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")


def _plot_learning_curves(ax, episode_returns: np.ndarray, episode_success: np.ndarray) -> None:
    x_return, mean_return = _moving_average(episode_returns, window=100)
    x_success, success_rate = _moving_average(episode_success.astype(float), window=100)

    ax.plot(x_return + 1, mean_return, color="tab:blue", label="mean return")
    ax.set_title("Q-learning training")
    ax.set_xlabel("episode")
    ax.set_ylabel("mean return")
    ax.grid(True, alpha=0.3)

    ax_success = ax.twinx()
    ax_success.plot(
        x_success + 1,
        success_rate,
        color="tab:green",
        alpha=0.85,
        label="success rate",
    )
    ax_success.set_ylabel("success rate")
    ax_success.set_ylim(-0.05, 1.05)

    lines, labels = ax.get_legend_handles_labels()
    success_lines, success_labels = ax_success.get_legend_handles_labels()
    ax.legend(lines + success_lines, labels + success_labels, loc="lower right")


def main() -> int:
    mdp = DiscreteArm2DOFMDP(
        DiscreteArm2DOFMDPConfig(
            target=(1.1, 0.55),
            bins_per_joint=(31, 31),
            target_tolerance=0.09,
            gamma=0.95,
        )
    )
    start_state = mdp.nearest_state([0.0, 0.0])
    learning_config = QLearningConfig(
        episodes=5_000,
        max_steps_per_episode=80,
        alpha=0.55,
        gamma=mdp.config.gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.996,
        random_start_states=False,
        seed=7,
    )

    learned = train_q_learning(mdp, start_state=start_state, config=learning_config)
    rollout = rollout_policy(mdp, learned.policy, start_state, max_steps=200)
    optimal = value_iteration(mdp, theta=1e-6)
    optimal_rollout = rollout_policy(mdp, optimal.policy, start_state, max_steps=200)

    raw_return = undiscounted_return(rollout.rewards)
    gamma_return = discounted_return(rollout.rewards, mdp.config.gamma)
    optimal_gamma_return = discounted_return(optimal_rollout.rewards, mdp.config.gamma)
    final_state = rollout.states[-1]
    action_names = [mdp.action_names[action] for action in rollout.actions]
    last_window = min(200, learning_config.episodes)
    success_rate = float(np.mean(learned.episode_success[-last_window:]))
    mean_return = float(np.mean(learned.episode_returns[-last_window:]))

    output_path = ROOT / "results" / "figures" / "step_08_q_learning_2dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _plot_q_heatmap(axes[0, 0], mdp, learned.q_value, start_state)
    _plot_policy(axes[0, 1], mdp, learned.policy, start_state)
    _plot_cartesian_rollout(axes[1, 0], mdp, rollout.states)
    _plot_learning_curves(
        axes[1, 1],
        learned.episode_returns,
        learned.episode_success,
    )
    fig.suptitle("Q-learning tabulaire - bras 2DDL discret")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"state_count={mdp.n_states}")
    print(f"action_count={mdp.n_actions}")
    print(f"episodes={learning_config.episodes}")
    print(f"alpha={learning_config.alpha:.3f}")
    print(f"gamma={mdp.config.gamma:.3f}")
    print(f"epsilon_start={learning_config.epsilon_start:.3f}")
    print(f"epsilon_end={learning_config.epsilon_end:.3f}")
    print(f"epsilon_final={learned.epsilon_history[-1]:.3f}")
    print(f"start_state={start_state}")
    print(f"q_start_max={np.max(learned.q_value[start_state]):.12e}")
    print(f"optimal_value_start={optimal.value[start_state]:.12e}")
    print(f"rollout_steps={len(rollout.actions)}")
    print(f"optimal_rollout_steps={len(optimal_rollout.actions)}")
    print(f"done={rollout.done}")
    print(f"return={raw_return:.12e}")
    print(f"discounted_return={gamma_return:.12e}")
    print(f"optimal_discounted_return={optimal_gamma_return:.12e}")
    print(f"success_rate_last_{last_window}={success_rate:.3f}")
    print(f"mean_return_last_{last_window}={mean_return:.12e}")
    print(f"actions={action_names}")
    print(f"final_summary={mdp.state_summary(final_state)}")
    print(f"figure={output_path}")

    return 0 if rollout.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
