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

