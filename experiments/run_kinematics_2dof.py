"""Run a first reproducible 2-DOF kinematics check."""

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

from robot import Arm2DOF, Arm2DOFConfig, forward_kinematics, inverse_kinematics
from visualization import plot_arm


def main() -> int:
    config = Arm2DOFConfig(link_lengths=(1.0, 0.8))
    target = np.array([1.1, 0.55], dtype=float)

    q = inverse_kinematics(target, config.link_lengths, elbow="up")
    arm = Arm2DOF(config=config)
    arm.set_joint_angles(q)

    end_effector = forward_kinematics(q, config.link_lengths)
    error = float(np.linalg.norm(end_effector - target))

    output_path = ROOT / "results" / "figures" / "step_01_kinematics_2dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_arm(
        arm.q,
        config.link_lengths,
        target=target,
        title="Bras 2DDL - cinematique inverse",
    )
    ax.text(
        0.02,
        0.02,
        f"erreur = {error:.2e}",
        transform=ax.transAxes,
        va="bottom",
    )
    fig.savefig(output_path, dpi=150)

    print(f"joint_angles_rad={np.array2string(q, precision=6)}")
    print(f"target={np.array2string(target, precision=6)}")
    print(f"end_effector={np.array2string(end_effector, precision=6)}")
    print(f"error={error:.12e}")
    print(f"figure={output_path}")

    return 0 if error < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())

