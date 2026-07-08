"""Demonstrate elementary RL concepts on a discrete 2-DOF arm MDP."""

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
    discounted_return,
    evaluate_policy,
    rollout_policy,
    undiscounted_return,
    value_iteration,
)
from robot import joint_positions, workspace_radius


def _plot_value_heatmap(ax, mdp: DiscreteArm2DOFMDP, value: np.ndarray, start_state: int) -> None:
    value_grid = value.reshape(mdp.grid_shape).T
    extent = [
        mdp.q1_values[0],
        mdp.q1_values[-1],
        mdp.q2_values[0],
        mdp.q2_values[-1],
    ]
    image = ax.imshow(
        value_grid,
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
    ax.set_title("State value V*")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.legend(loc="upper right")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_policy(ax, mdp: DiscreteArm2DOFMDP, policy: np.ndarray, start_state: int) -> None:
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
    ax.set_title("Policy pi*(s)")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.legend(loc="upper right")


def _plot_cartesian_rollout(ax, mdp: DiscreteArm2DOFMDP, states: list[int]) -> None:
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
    ax.set_title("Rollout in Cartesian space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")


def _plot_returns(
    ax,
    mdp: DiscreteArm2DOFMDP,
    states: list[int],
    rewards: list[float],
    raw_return: float,
    gamma_return: float,
) -> None:
    distances = np.asarray([mdp.distances[state] for state in states])
    ax.plot(distances, marker="o", label="distance")
    if rewards:
        ax.bar(
            np.arange(len(rewards)),
            rewards,
            alpha=0.35,
            label="reward",
        )
    ax.axhline(mdp.config.target_tolerance, linestyle="--", linewidth=1, label="tolerance")
    ax.text(
        0.02,
        0.05,
        f"return = {raw_return:.3f}\ndiscounted = {gamma_return:.3f}",
        transform=ax.transAxes,
        va="bottom",
    )
    ax.grid(True, alpha=0.3)
    ax.set_title("Reward, return and discounted return")
    ax.set_xlabel("step")
    ax.legend(loc="upper right")


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

    random_policy = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
    random_evaluation = evaluate_policy(mdp, random_policy, theta=1e-6)
    optimal = value_iteration(mdp, theta=1e-6)
    rollout = rollout_policy(mdp, optimal.policy, start_state, max_steps=200)

    raw_return = undiscounted_return(rollout.rewards)
    gamma_return = discounted_return(rollout.rewards, mdp.config.gamma)
    final_state = rollout.states[-1]
    action_names = [mdp.action_names[action] for action in rollout.actions]

    output_path = ROOT / "results" / "figures" / "step_04_rl_bellman_2dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _plot_value_heatmap(axes[0, 0], mdp, optimal.value, start_state)
    _plot_policy(axes[0, 1], mdp, optimal.policy, start_state)
    _plot_cartesian_rollout(axes[1, 0], mdp, rollout.states)
    _plot_returns(
        axes[1, 1],
        mdp,
        rollout.states,
        rollout.rewards,
        raw_return,
        gamma_return,
    )
    fig.suptitle("RL elementaire - Bellman sur bras 2DDL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"state_count={mdp.n_states}")
    print(f"action_count={mdp.n_actions}")
    print(f"gamma={mdp.config.gamma:.3f}")
    print(f"start_state={start_state}")
    print(f"start_summary={mdp.state_summary(start_state)}")
    print(f"random_policy_value_start={random_evaluation.value[start_state]:.12e}")
    print(f"optimal_value_start={optimal.value[start_state]:.12e}")
    print(f"value_iteration_iterations={optimal.iterations}")
    print(f"rollout_steps={len(rollout.actions)}")
    print(f"done={rollout.done}")
    print(f"return={raw_return:.12e}")
    print(f"discounted_return={gamma_return:.12e}")
    print(f"actions={action_names}")
    print(f"final_summary={mdp.state_summary(final_state)}")
    print(f"figure={output_path}")

    return 0 if rollout.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
