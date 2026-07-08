"""Run the first fuzzy control experiment on the 2-DOF kinematic arm."""

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

from controllers import FuzzyVelocityController
from envs import Arm2DOFEnv, Arm2DOFEnvConfig
from robot import inverse_kinematics
from visualization import plot_control_simulation


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
    controller = FuzzyVelocityController(
        error_scale=[0.45, 0.75],
        derivative_scale=[4.0, 4.0],
        output_scale=config.max_joint_speed,
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

    output_path = ROOT / "results" / "figures" / "step_03_fuzzy_2dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_control_simulation(
        q_history_array,
        ee_history_array,
        distance_history_array,
        action_history_array,
        config.target,
        link_lengths=config.arm_config.link_lengths,
        tolerance=config.target_tolerance,
        title="Controle flou - bras 2DDL",
    )
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
