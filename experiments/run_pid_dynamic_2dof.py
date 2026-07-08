"""Run PID computed-torque control on the dynamic 2-DOF arm."""

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
from envs import Arm2DOFDynamicEnv, Arm2DOFDynamicEnvConfig
from robot import inverse_dynamics_torque, inverse_kinematics
from visualization import plot_control_simulation


def main() -> int:
    config = Arm2DOFDynamicEnvConfig(
        target=(1.1, 0.55),
        dt=0.01,
        max_torque=(60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=2500,
    )
    env = Arm2DOFDynamicEnv(config)
    observation = env.reset(q=[0.0, 0.0], q_dot=[0.0, 0.0])

    desired_q = inverse_kinematics(
        config.target,
        config.arm_config.link_lengths,
        elbow="up",
    )
    controller = PIDController(
        kp=[35.0, 30.0],
        ki=[0.0, 0.0],
        kd=[8.0, 7.0],
        output_limits=(-35.0, 35.0),
    )

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    speed_history = [float(observation["speed"])]
    torque_history = []
    done = False
    info = {}

    for _ in range(config.max_steps):
        desired_q_ddot = controller.compute(desired_q, observation["q"], config.dt)
        torque = inverse_dynamics_torque(
            observation["q"],
            observation["q_dot"],
            desired_q_ddot,
            config.dynamics_config,
        )
        observation, reward, done, info = env.step(torque)
        q_history.append(observation["q"].copy())
        ee_history.append(observation["end_effector"].copy())
        distance_history.append(float(observation["distance"]))
        speed_history.append(float(observation["speed"]))
        torque_history.append(info["action"].copy())
        if done:
            break

    q_history_array = np.asarray(q_history)
    ee_history_array = np.asarray(ee_history)
    distance_history_array = np.asarray(distance_history)
    speed_history_array = np.asarray(speed_history)
    torque_history_array = np.asarray(torque_history)

    output_path = ROOT / "results" / "figures" / "step_06_pid_dynamic_2dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_control_simulation(
        q_history_array,
        ee_history_array,
        distance_history_array,
        torque_history_array,
        config.target,
        link_lengths=config.arm_config.link_lengths,
        tolerance=config.target_tolerance,
        title="PID dynamique couple calcule - bras 2DDL",
        action_ylabel="N.m",
    )
    fig.savefig(output_path, dpi=150)

    final_distance = float(distance_history_array[-1])
    final_speed = float(speed_history_array[-1])
    mean_torque = (
        float(np.mean(np.linalg.norm(torque_history_array, axis=1)))
        if torque_history_array.size
        else 0.0
    )
    print(f"steps={len(distance_history_array) - 1}")
    print(f"done={done}")
    print(f"truncated={info.get('truncated', False)}")
    print(f"desired_joint_angles_rad={np.array2string(desired_q, precision=6)}")
    print(f"final_joint_angles_rad={np.array2string(q_history_array[-1], precision=6)}")
    print(f"final_distance={final_distance:.12e}")
    print(f"final_speed={final_speed:.12e}")
    print(f"mean_torque_norm={mean_torque:.12e}")
    print(f"figure={output_path}")

    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
