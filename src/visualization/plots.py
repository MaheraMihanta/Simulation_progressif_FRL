"""Plot helpers for the planar robotic arm."""

from __future__ import annotations

import numpy as np

from robot.kinematics import ArrayLike2, joint_positions, workspace_radius


def plot_arm(
    q: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    target: ArrayLike2 | None = None,
    ax=None,
    title: str | None = None,
    show_workspace: bool = True,
):
    """Plot the 2-DOF arm and return `(figure, axes)`."""

    import matplotlib.pyplot as plt

    positions = joint_positions(q, link_lengths)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.plot(positions[:, 0], positions[:, 1], "-o", linewidth=3, markersize=8)
    ax.scatter([0.0], [0.0], s=80, marker="s", label="base")

    if target is not None:
        target_array = np.asarray(target, dtype=float)
        ax.scatter(
            [target_array[0]],
            [target_array[1]],
            s=100,
            marker="x",
            label="cible",
        )

    if show_workspace:
        _, r_max = workspace_radius(link_lengths)
        circle = plt.Circle(
            (0.0, 0.0),
            r_max,
            fill=False,
            linestyle="--",
            linewidth=1,
            alpha=0.35,
        )
        ax.add_patch(circle)

    reach = sum(link_lengths)
    margin = 0.15 * reach
    ax.set_xlim(-reach - margin, reach + margin)
    ax.set_ylim(-reach - margin, reach + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_control_simulation(
    q_history: np.ndarray,
    ee_history: np.ndarray,
    distance_history: np.ndarray,
    action_history: np.ndarray,
    target: ArrayLike2,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    tolerance: float | None = None,
    title: str = "Simulation bras 2DDL",
    snapshot_count: int = 7,
    action_ylabel: str = "rad/s",
):
    """Plot trajectory, arm snapshots, error and command histories."""

    import matplotlib.pyplot as plt

    target_array = np.asarray(target, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_traj, ax_dist, ax_q, ax_action = axes.ravel()

    ax_traj.plot(
        ee_history[:, 0],
        ee_history[:, 1],
        linewidth=2,
        label="trajectoire effecteur",
    )
    ax_traj.scatter(
        [target_array[0]],
        [target_array[1]],
        marker="x",
        s=100,
        label="cible",
    )

    if len(q_history):
        indices = np.linspace(
            0,
            len(q_history) - 1,
            min(snapshot_count, len(q_history)),
            dtype=int,
        )
        for index in indices:
            positions = joint_positions(q_history[index], link_lengths)
            alpha = 0.25 if index != indices[-1] else 0.95
            linewidth = 1.5 if index != indices[-1] else 3.0
            ax_traj.plot(
                positions[:, 0],
                positions[:, 1],
                "-o",
                color="tab:gray" if index != indices[-1] else "tab:red",
                alpha=alpha,
                linewidth=linewidth,
                markersize=5,
            )

    _, r_max = workspace_radius(link_lengths)
    workspace = plt.Circle(
        (0.0, 0.0),
        r_max,
        fill=False,
        linestyle="--",
        linewidth=1,
        alpha=0.3,
    )
    ax_traj.add_patch(workspace)
    reach = sum(link_lengths)
    margin = 0.15 * reach
    ax_traj.set_xlim(-reach - margin, reach + margin)
    ax_traj.set_ylim(-reach - margin, reach + margin)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title("Trajectoire et poses successives")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.legend(loc="upper right")

    ax_dist.plot(distance_history, color="tab:blue", label="distance cible")
    if tolerance is not None:
        ax_dist.axhline(
            tolerance,
            linestyle="--",
            linewidth=1,
            color="tab:green",
            label="tolerance",
        )
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_title("Erreur de position")
    ax_dist.set_xlabel("iteration")
    ax_dist.set_ylabel("distance")
    ax_dist.legend()

    ax_q.plot(q_history[:, 0], label="q1")
    ax_q.plot(q_history[:, 1], label="q2")
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title("Angles articulaires")
    ax_q.set_xlabel("iteration")
    ax_q.set_ylabel("rad")
    ax_q.legend()

    if action_history.size:
        ax_action.plot(action_history[:, 0], label="commande q1")
        ax_action.plot(action_history[:, 1], label="commande q2")
        ax_action.plot(
            np.linalg.norm(action_history, axis=1),
            label="norme commande",
            alpha=0.75,
        )
        ax_action.legend()
    ax_action.grid(True, alpha=0.3)
    ax_action.set_title("Commande articulaire")
    ax_action.set_xlabel("iteration")
    ax_action.set_ylabel(action_ylabel)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
