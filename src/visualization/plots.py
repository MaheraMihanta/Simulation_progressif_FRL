"""Plot helpers for the planar robotic arm."""

from __future__ import annotations

import numpy as np

from robot.kinematics import ArrayLike2, joint_positions, workspace_radius
from robot.kinematics_3dof import ArrayLike3, joint_positions_3dof, workspace_radius_3dof
from robot.kinematics_4dof import ArrayLike4, joint_positions_4dof, workspace_radius_4dof
from robot.kinematics_5dof import ArrayLike5, joint_positions_5dof, workspace_radius_5dof
from robot.kinematics_6dof import ArrayLike6, joint_positions_6dof, workspace_radius_6dof


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


def _set_equal_3d_axes(ax, reach: float, margin_ratio: float = 0.15) -> None:
    margin = margin_ratio * reach
    limit = reach + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _draw_workspace_sphere(ax, radius: float) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 18)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.35, alpha=0.12, color="tab:gray")


def plot_arm_3dof(
    q: ArrayLike3,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    target: ArrayLike3 | None = None,
    ax=None,
    title: str | None = None,
    show_workspace: bool = True,
):
    """Plot the spatial 3-DOF arm and return `(figure, axes)`."""

    import matplotlib.pyplot as plt

    positions = joint_positions_3dof(q, link_lengths)
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "-o",
        linewidth=3,
        markersize=8,
        label="bras",
    )
    ax.scatter([0.0], [0.0], [0.0], s=80, marker="s", label="base")

    if target is not None:
        target_array = np.asarray(target, dtype=float)
        ax.scatter(
            [target_array[0]],
            [target_array[1]],
            [target_array[2]],
            s=100,
            marker="x",
            label="cible",
        )

    _, r_max = workspace_radius_3dof(link_lengths)
    if show_workspace:
        _draw_workspace_sphere(ax, r_max)

    _set_equal_3d_axes(ax, r_max)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_control_simulation_3dof(
    q_history: np.ndarray,
    ee_history: np.ndarray,
    distance_history: np.ndarray,
    action_history: np.ndarray,
    target: ArrayLike3,
    link_lengths: tuple[float, float] = (1.0, 0.8),
    tolerance: float | None = None,
    title: str = "Simulation bras 3DDL",
    snapshot_count: int = 7,
    action_ylabel: str = "rad/s",
):
    """Plot 3D trajectory, arm snapshots, error and command histories."""

    import matplotlib.pyplot as plt

    target_array = np.asarray(target, dtype=float)
    fig = plt.figure(figsize=(13, 9))
    ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
    ax_dist = fig.add_subplot(2, 2, 2)
    ax_q = fig.add_subplot(2, 2, 3)
    ax_action = fig.add_subplot(2, 2, 4)
    axes = np.asarray([[ax_traj, ax_dist], [ax_q, ax_action]], dtype=object)

    ax_traj.plot(
        ee_history[:, 0],
        ee_history[:, 1],
        ee_history[:, 2],
        linewidth=2,
        label="trajectoire effecteur",
    )
    ax_traj.scatter(
        [target_array[0]],
        [target_array[1]],
        [target_array[2]],
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
            positions = joint_positions_3dof(q_history[index], link_lengths)
            alpha = 0.25 if index != indices[-1] else 0.95
            linewidth = 1.3 if index != indices[-1] else 3.0
            ax_traj.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                "-o",
                color="tab:gray" if index != indices[-1] else "tab:red",
                alpha=alpha,
                linewidth=linewidth,
                markersize=5,
            )

    _, r_max = workspace_radius_3dof(link_lengths)
    _draw_workspace_sphere(ax_traj, r_max)
    _set_equal_3d_axes(ax_traj, r_max)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title("Trajectoire et poses successives")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
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
    ax_dist.set_title("Erreur de position 3D")
    ax_dist.set_xlabel("iteration")
    ax_dist.set_ylabel("distance")
    ax_dist.legend()

    ax_q.plot(q_history[:, 0], label="q0 base")
    ax_q.plot(q_history[:, 1], label="q1 epaule")
    ax_q.plot(q_history[:, 2], label="q2 coude")
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title("Angles articulaires")
    ax_q.set_xlabel("iteration")
    ax_q.set_ylabel("rad")
    ax_q.legend()

    if action_history.size:
        ax_action.plot(action_history[:, 0], label="commande q0")
        ax_action.plot(action_history[:, 1], label="commande q1")
        ax_action.plot(action_history[:, 2], label="commande q2")
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


def plot_arm_4dof(
    q: ArrayLike4,
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
    target: ArrayLike3 | None = None,
    ax=None,
    title: str | None = None,
    show_workspace: bool = True,
):
    """Plot the spatial 4-DOF arm and return `(figure, axes)`."""

    import matplotlib.pyplot as plt

    positions = joint_positions_4dof(q, link_lengths)
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "-o",
        linewidth=3,
        markersize=7,
        label="bras",
    )
    ax.scatter([0.0], [0.0], [0.0], s=80, marker="s", label="base")

    if target is not None:
        target_array = np.asarray(target, dtype=float)
        ax.scatter(
            [target_array[0]],
            [target_array[1]],
            [target_array[2]],
            s=100,
            marker="x",
            label="cible",
        )

    _, r_max = workspace_radius_4dof(link_lengths)
    if show_workspace:
        _draw_workspace_sphere(ax, r_max)

    _set_equal_3d_axes(ax, r_max)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_control_simulation_4dof(
    q_history: np.ndarray,
    ee_history: np.ndarray,
    distance_history: np.ndarray,
    action_history: np.ndarray,
    target: ArrayLike3,
    link_lengths: tuple[float, float, float] = (1.0, 0.8, 0.55),
    tolerance: float | None = None,
    title: str = "Simulation bras 4DDL",
    snapshot_count: int = 7,
    action_ylabel: str = "rad/s",
    gain_history: np.ndarray | None = None,
):
    """Plot 3D trajectory, arm snapshots, error, commands and optional gains."""

    import matplotlib.pyplot as plt

    target_array = np.asarray(target, dtype=float)
    if gain_history is None:
        fig = plt.figure(figsize=(13, 9))
        ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
        ax_dist = fig.add_subplot(2, 2, 2)
        ax_q = fig.add_subplot(2, 2, 3)
        ax_action = fig.add_subplot(2, 2, 4)
        axes = np.asarray([[ax_traj, ax_dist], [ax_q, ax_action]], dtype=object)
    else:
        fig = plt.figure(figsize=(14, 11))
        ax_traj = fig.add_subplot(3, 2, 1, projection="3d")
        ax_dist = fig.add_subplot(3, 2, 2)
        ax_q = fig.add_subplot(3, 2, 3)
        ax_action = fig.add_subplot(3, 2, 4)
        ax_gain = fig.add_subplot(3, 1, 3)
        axes = np.asarray(
            [[ax_traj, ax_dist], [ax_q, ax_action], [ax_gain, None]],
            dtype=object,
        )

    ax_traj.plot(
        ee_history[:, 0],
        ee_history[:, 1],
        ee_history[:, 2],
        linewidth=2,
        label="trajectoire effecteur",
    )
    ax_traj.scatter(
        [target_array[0]],
        [target_array[1]],
        [target_array[2]],
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
            positions = joint_positions_4dof(q_history[index], link_lengths)
            alpha = 0.25 if index != indices[-1] else 0.95
            linewidth = 1.3 if index != indices[-1] else 3.0
            ax_traj.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                "-o",
                color="tab:gray" if index != indices[-1] else "tab:red",
                alpha=alpha,
                linewidth=linewidth,
                markersize=5,
            )

    _, r_max = workspace_radius_4dof(link_lengths)
    _draw_workspace_sphere(ax_traj, r_max)
    _set_equal_3d_axes(ax_traj, r_max)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title("Trajectoire et poses successives")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
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
    ax_dist.set_title("Erreur de position 3D")
    ax_dist.set_xlabel("iteration")
    ax_dist.set_ylabel("distance")
    ax_dist.legend()

    labels = ("q0 base", "q1 epaule", "q2 coude", "q3 poignet")
    for index, label in enumerate(labels):
        ax_q.plot(q_history[:, index], label=label)
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title("Angles articulaires")
    ax_q.set_xlabel("iteration")
    ax_q.set_ylabel("rad")
    ax_q.legend()

    if action_history.size:
        for index in range(4):
            ax_action.plot(action_history[:, index], label=f"commande q{index}")
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

    if gain_history is not None:
        kp_mean = np.mean(gain_history[:, 0, :], axis=1)
        ki_mean = np.mean(gain_history[:, 1, :], axis=1)
        kd_mean = np.mean(gain_history[:, 2, :], axis=1)
        ax_gain.plot(kp_mean, label="Kp moyen")
        ax_gain.plot(ki_mean, label="Ki moyen")
        ax_gain.plot(kd_mean, label="Kd moyen")
        ax_gain.grid(True, alpha=0.3)
        ax_gain.set_title("Adaptation moyenne des gains PID")
        ax_gain.set_xlabel("iteration")
        ax_gain.set_ylabel("gain")
        ax_gain.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_arm_5dof(
    q: ArrayLike5,
    link_lengths: tuple[float, float, float, float] = (1.0, 0.8, 0.55, 0.35),
    target: ArrayLike3 | None = None,
    ax=None,
    title: str | None = None,
    show_workspace: bool = True,
):
    """Plot the spatial 5-DOF arm and return `(figure, axes)`."""

    import matplotlib.pyplot as plt

    positions = joint_positions_5dof(q, link_lengths)
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "-o",
        linewidth=3,
        markersize=7,
        label="bras",
    )
    ax.scatter([0.0], [0.0], [0.0], s=80, marker="s", label="base")

    if target is not None:
        target_array = np.asarray(target, dtype=float)
        ax.scatter(
            [target_array[0]],
            [target_array[1]],
            [target_array[2]],
            s=100,
            marker="x",
            label="cible",
        )

    _, r_max = workspace_radius_5dof(link_lengths)
    if show_workspace:
        _draw_workspace_sphere(ax, r_max)

    _set_equal_3d_axes(ax, r_max)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_control_simulation_5dof(
    q_history: np.ndarray,
    ee_history: np.ndarray,
    distance_history: np.ndarray,
    action_history: np.ndarray,
    target: ArrayLike3,
    link_lengths: tuple[float, float, float, float] = (1.0, 0.8, 0.55, 0.35),
    tolerance: float | None = None,
    title: str = "Simulation bras 5DDL",
    snapshot_count: int = 7,
    action_ylabel: str = "rad/s",
    gain_history: np.ndarray | None = None,
):
    """Plot 3D trajectory, arm snapshots, error, commands and optional gains."""

    import matplotlib.pyplot as plt

    target_array = np.asarray(target, dtype=float)
    if gain_history is None:
        fig = plt.figure(figsize=(13, 9))
        ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
        ax_dist = fig.add_subplot(2, 2, 2)
        ax_q = fig.add_subplot(2, 2, 3)
        ax_action = fig.add_subplot(2, 2, 4)
        axes = np.asarray([[ax_traj, ax_dist], [ax_q, ax_action]], dtype=object)
    else:
        fig = plt.figure(figsize=(14, 11))
        ax_traj = fig.add_subplot(3, 2, 1, projection="3d")
        ax_dist = fig.add_subplot(3, 2, 2)
        ax_q = fig.add_subplot(3, 2, 3)
        ax_action = fig.add_subplot(3, 2, 4)
        ax_gain = fig.add_subplot(3, 1, 3)
        axes = np.asarray(
            [[ax_traj, ax_dist], [ax_q, ax_action], [ax_gain, None]],
            dtype=object,
        )

    ax_traj.plot(
        ee_history[:, 0],
        ee_history[:, 1],
        ee_history[:, 2],
        linewidth=2,
        label="trajectoire effecteur",
    )
    ax_traj.scatter(
        [target_array[0]],
        [target_array[1]],
        [target_array[2]],
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
            positions = joint_positions_5dof(q_history[index], link_lengths)
            alpha = 0.25 if index != indices[-1] else 0.95
            linewidth = 1.3 if index != indices[-1] else 3.0
            ax_traj.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                "-o",
                color="tab:gray" if index != indices[-1] else "tab:red",
                alpha=alpha,
                linewidth=linewidth,
                markersize=5,
            )

    _, r_max = workspace_radius_5dof(link_lengths)
    _draw_workspace_sphere(ax_traj, r_max)
    _set_equal_3d_axes(ax_traj, r_max)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title("Trajectoire et poses successives")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
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
    ax_dist.set_title("Erreur de position 3D")
    ax_dist.set_xlabel("iteration")
    ax_dist.set_ylabel("distance")
    ax_dist.legend()

    labels = ("q0 base", "q1 epaule", "q2 coude", "q3 poignet", "q4 poignet")
    for index, label in enumerate(labels):
        ax_q.plot(q_history[:, index], label=label)
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title("Angles articulaires")
    ax_q.set_xlabel("iteration")
    ax_q.set_ylabel("rad")
    ax_q.legend()

    if action_history.size:
        for index in range(5):
            ax_action.plot(action_history[:, index], label=f"commande q{index}")
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

    if gain_history is not None:
        kp_mean = np.mean(gain_history[:, 0, :], axis=1)
        ki_mean = np.mean(gain_history[:, 1, :], axis=1)
        kd_mean = np.mean(gain_history[:, 2, :], axis=1)
        ax_gain.plot(kp_mean, label="Kp moyen")
        ax_gain.plot(ki_mean, label="Ki moyen")
        ax_gain.plot(kd_mean, label="Kd moyen")
        ax_gain.grid(True, alpha=0.3)
        ax_gain.set_title("Adaptation moyenne des gains PID")
        ax_gain.set_xlabel("iteration")
        ax_gain.set_ylabel("gain")
        ax_gain.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_arm_6dof(
    q: ArrayLike6,
    link_lengths: tuple[float, float, float, float, float] = (
        1.0,
        0.8,
        0.55,
        0.35,
        0.25,
    ),
    target: ArrayLike3 | None = None,
    ax=None,
    title: str | None = None,
    show_workspace: bool = True,
):
    """Plot the spatial 6-DOF arm and return `(figure, axes)`."""

    import matplotlib.pyplot as plt

    positions = joint_positions_6dof(q, link_lengths)
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "-o",
        linewidth=3,
        markersize=7,
        label="bras",
    )
    ax.scatter([0.0], [0.0], [0.0], s=80, marker="s", label="base")

    if target is not None:
        target_array = np.asarray(target, dtype=float)
        ax.scatter(
            [target_array[0]],
            [target_array[1]],
            [target_array[2]],
            s=100,
            marker="x",
            label="cible",
        )

    _, r_max = workspace_radius_6dof(link_lengths)
    if show_workspace:
        _draw_workspace_sphere(ax, r_max)

    _set_equal_3d_axes(ax, r_max)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_control_simulation_6dof(
    q_history: np.ndarray,
    ee_history: np.ndarray,
    distance_history: np.ndarray,
    action_history: np.ndarray,
    target: ArrayLike3,
    link_lengths: tuple[float, float, float, float, float] = (
        1.0,
        0.8,
        0.55,
        0.35,
        0.25,
    ),
    tolerance: float | None = None,
    title: str = "Simulation bras 6DDL",
    snapshot_count: int = 7,
    action_ylabel: str = "rad/s",
    gain_history: np.ndarray | None = None,
):
    """Plot 3D trajectory, arm snapshots, error, commands and optional gains."""

    import matplotlib.pyplot as plt

    target_array = np.asarray(target, dtype=float)
    if gain_history is None:
        fig = plt.figure(figsize=(13, 9))
        ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
        ax_dist = fig.add_subplot(2, 2, 2)
        ax_q = fig.add_subplot(2, 2, 3)
        ax_action = fig.add_subplot(2, 2, 4)
        axes = np.asarray([[ax_traj, ax_dist], [ax_q, ax_action]], dtype=object)
    else:
        fig = plt.figure(figsize=(14, 11))
        ax_traj = fig.add_subplot(3, 2, 1, projection="3d")
        ax_dist = fig.add_subplot(3, 2, 2)
        ax_q = fig.add_subplot(3, 2, 3)
        ax_action = fig.add_subplot(3, 2, 4)
        ax_gain = fig.add_subplot(3, 1, 3)
        axes = np.asarray(
            [[ax_traj, ax_dist], [ax_q, ax_action], [ax_gain, None]],
            dtype=object,
        )

    ax_traj.plot(
        ee_history[:, 0],
        ee_history[:, 1],
        ee_history[:, 2],
        linewidth=2,
        label="trajectoire effecteur",
    )
    ax_traj.scatter(
        [target_array[0]],
        [target_array[1]],
        [target_array[2]],
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
            positions = joint_positions_6dof(q_history[index], link_lengths)
            alpha = 0.25 if index != indices[-1] else 0.95
            linewidth = 1.3 if index != indices[-1] else 3.0
            ax_traj.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                "-o",
                color="tab:gray" if index != indices[-1] else "tab:red",
                alpha=alpha,
                linewidth=linewidth,
                markersize=5,
            )

    _, r_max = workspace_radius_6dof(link_lengths)
    _draw_workspace_sphere(ax_traj, r_max)
    _set_equal_3d_axes(ax_traj, r_max)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_title("Trajectoire et poses successives")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
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
    ax_dist.set_title("Erreur de position 3D")
    ax_dist.set_xlabel("iteration")
    ax_dist.set_ylabel("distance")
    ax_dist.legend()

    labels = (
        "q0 base",
        "q1 epaule",
        "q2 coude",
        "q3 poignet",
        "q4 poignet",
        "q5 terminal",
    )
    for index, label in enumerate(labels):
        ax_q.plot(q_history[:, index], label=label)
    ax_q.grid(True, alpha=0.3)
    ax_q.set_title("Angles articulaires")
    ax_q.set_xlabel("iteration")
    ax_q.set_ylabel("rad")
    ax_q.legend()

    if action_history.size:
        for index in range(6):
            ax_action.plot(action_history[:, index], label=f"commande q{index}")
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

    if gain_history is not None:
        kp_mean = np.mean(gain_history[:, 0, :], axis=1)
        ki_mean = np.mean(gain_history[:, 1, :], axis=1)
        kd_mean = np.mean(gain_history[:, 2, :], axis=1)
        ax_gain.plot(kp_mean, label="Kp moyen")
        ax_gain.plot(ki_mean, label="Ki moyen")
        ax_gain.plot(kd_mean, label="Kd moyen")
        ax_gain.grid(True, alpha=0.3)
        ax_gain.set_title("Adaptation moyenne des gains PID")
        ax_gain.set_xlabel("iteration")
        ax_gain.set_ylabel("gain")
        ax_gain.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
