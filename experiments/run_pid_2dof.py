"""Run the first PID control experiment on the 2-DOF kinematic arm."""

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

from controllers import PIDController
from envs import Arm2DOFEnv, Arm2DOFEnvConfig
from robot import inverse_kinematics, joint_positions


def main() -> int:
    config = Arm2DOFEnvConfig(
        target=(1.1, 0.55),
        dt=0.05,
        max_joint_speed=2.0,
        target_tolerance=1e-2,
        max_steps=400,
    )
    env = Arm2DOFEnv(config)
    observation = env.reset(q=[0.0, 0.0])

    desired_q = inverse_kinematics(
        config.target,
        config.arm_config.link_lengths,
        elbow="up",
    )
    controller = PIDController(
        kp=[4.0, 4.0],
        ki=[0.0, 0.0],
        kd=[0.15, 0.15],
        output_limits=(-config.max_joint_speed, config.max_joint_speed),
    )

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    action_history = []
    done = False
    info = {}

    for _ in range(config.max_steps):
        action = controller.compute(desired_q, observation["q"], config.dt)
        observation, reward, done, info = env.step(action)
        q_history.append(observation["q"].copy())
        ee_history.append(observation["end_effector"].copy())
        distance_history.append(float(observation["distance"]))
        action_history.append(action.copy())
        if done:
            break

    q_history_array = np.asarray(q_history)
    ee_history_array = np.asarray(ee_history)
    distance_history_array = np.asarray(distance_history)
    action_history_array = np.asarray(action_history)

    output_path = ROOT / "results" / "figures" / "step_02_pid_2dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(ee_history_array[:, 0], ee_history_array[:, 1], linewidth=2)
    axes[0].scatter([config.target[0]], [config.target[1]], marker="x", s=90)
    final_positions = joint_positions(env.arm.q, config.arm_config.link_lengths)
    axes[0].plot(
        final_positions[:, 0],
        final_positions[:, 1],
        "-o",
        linewidth=3,
        markersize=7,
    )
    reach = sum(config.arm_config.link_lengths)
    margin = 0.15 * reach
    axes[0].set_xlim(-reach - margin, reach + margin)
    axes[0].set_ylim(-reach - margin, reach + margin)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Trajectoire effecteur")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].plot(distance_history_array, label="distance cible")
    if action_history_array.size:
        axes[1].plot(
            np.linalg.norm(action_history_array, axis=1),
            label="norme action",
            alpha=0.8,
        )
    axes[1].axhline(config.target_tolerance, linestyle="--", linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Convergence PID")
    axes[1].set_xlabel("iteration")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    final_distance = float(distance_history_array[-1])
    print(f"steps={len(distance_history_array) - 1}")
    print(f"done={done}")
    print(f"truncated={info.get('truncated', False)}")
    print(f"desired_joint_angles_rad={np.array2string(desired_q, precision=6)}")
    print(f"final_joint_angles_rad={np.array2string(q_history_array[-1], precision=6)}")
    print(f"final_distance={final_distance:.12e}")
    print(f"figure={output_path}")

    return 0 if final_distance <= config.target_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())

