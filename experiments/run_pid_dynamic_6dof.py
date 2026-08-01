"""Run computed-torque PID control on the dynamic 6-DOF arm."""

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
from envs import Arm6DOFDynamicEnv, Arm6DOFDynamicEnvConfig
from robot import inverse_dynamics_torque_6dof, inverse_kinematics_6dof
from visualization import plot_control_simulation_6dof


def main() -> int:
    config = Arm6DOFDynamicEnvConfig(
        target=(1.25, 0.45, 0.60),
        dt=0.01,
        max_torque=(65.0, 95.0, 70.0, 45.0, 30.0, 25.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=4000,
    )
    env = Arm6DOFDynamicEnv(config)
    observation = env.reset(
        q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        q_dot=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    desired_q = inverse_kinematics_6dof(
        config.target,
        config.arm_config.link_lengths,
        elbow="up",
        terminal_pitch=0.0,
        joint_limits=config.arm_config.joint_limits,
    )
    controller = PIDController(
        kp=[32.0, 48.0, 38.0, 26.0, 18.0, 14.0],
        ki=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        kd=[8.0, 11.0, 9.0, 6.0, 4.5, 3.5],
        size=6,
        output_limits=(-55.0, 55.0),
    )

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    torque_history = []
    done = False
    info = {}

    for _ in range(config.max_steps):
        desired_q_ddot = controller.compute(desired_q, observation["q"], config.dt)
        torque = inverse_dynamics_torque_6dof(
            observation["q"],
            observation["q_dot"],
            desired_q_ddot,
            config.dynamics_config,
        )
        observation, reward, done, info = env.step(torque)
        q_history.append(observation["q"].copy())
        ee_history.append(observation["end_effector"].copy())
        distance_history.append(float(observation["distance"]))
        torque_history.append(info["action"].copy())
        if done:
            break

    q_history_array = np.asarray(q_history)
    ee_history_array = np.asarray(ee_history)
    distance_history_array = np.asarray(distance_history)
    torque_history_array = np.asarray(torque_history)

    output_path = ROOT / "results" / "figures" / "step_36_pid_dynamic_6dof.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_control_simulation_6dof(
        q_history_array,
        ee_history_array,
        distance_history_array,
        torque_history_array,
        config.target,
        link_lengths=config.arm_config.link_lengths,
        tolerance=config.target_tolerance,
        title="Controle PID dynamique - bras 6DDL spatial",
        action_ylabel="N.m",
    )
    fig.savefig(output_path, dpi=150)

    final_distance = float(distance_history_array[-1])
    mean_torque = float(np.mean(np.linalg.norm(torque_history_array, axis=1)))
    print(f"steps={len(distance_history_array) - 1}")
    print(f"done={done}")
    print(f"truncated={info.get('truncated', False)}")
    print(f"desired_joint_angles_rad={np.array2string(desired_q, precision=6)}")
    print(f"final_joint_angles_rad={np.array2string(q_history_array[-1], precision=6)}")
    print(f"final_distance={final_distance:.12e}")
    print(f"final_speed={float(observation['speed']):.12e}")
    print(f"mean_torque_norm={mean_torque:.12e}")
    print(f"figure={output_path}")

    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
