"""Run fuzzy gain-scheduled computed-torque PID on the dynamic 4-DOF arm."""

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

from controllers import FuzzyGainScheduledPIDController
from envs import Arm4DOFDynamicEnv, Arm4DOFDynamicEnvConfig
from robot import inverse_dynamics_torque_4dof, inverse_kinematics_4dof
from visualization import plot_control_simulation_4dof


def main() -> int:
    config = Arm4DOFDynamicEnvConfig(
        target=(1.15, 0.45, 0.55),
        dt=0.01,
        max_torque=(55.0, 85.0, 60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=3500,
    )
    env = Arm4DOFDynamicEnv(config)
    observation = env.reset(q=[0.0, 0.0, 0.0, 0.0], q_dot=[0.0, 0.0, 0.0, 0.0])

    desired_q = inverse_kinematics_4dof(
        config.target,
        config.arm_config.link_lengths,
        elbow="up",
        terminal_pitch=0.0,
        joint_limits=config.arm_config.joint_limits,
    )
    controller = FuzzyGainScheduledPIDController(
        kp=[28.0, 42.0, 34.0, 22.0],
        ki=[0.0, 0.0, 0.0, 0.0],
        kd=[7.0, 10.0, 8.0, 5.0],
        size=4,
        error_scale=[0.35, 0.45, 0.55, 0.55],
        derivative_scale=[4.0, 5.0, 5.0, 5.0],
        output_limits=(-45.0, 45.0),
    )

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    torque_history = []
    gain_history = []
    done = False
    info = {}

    for _ in range(config.max_steps):
        desired_q_ddot = controller.compute(desired_q, observation["q"], config.dt)
        gain_history.append(
            np.vstack(
                [
                    controller.last_kp_vector,
                    controller.last_ki_vector,
                    controller.last_kd_vector,
                ]
            )
        )
        torque = inverse_dynamics_torque_4dof(
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
    gain_history_array = np.asarray(gain_history)

    output_path = (
        ROOT / "results" / "figures" / "step_24_pid_fuzzy_gain_dynamic_4dof.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_control_simulation_4dof(
        q_history_array,
        ee_history_array,
        distance_history_array,
        torque_history_array,
        config.target,
        link_lengths=config.arm_config.link_lengths,
        tolerance=config.target_tolerance,
        title="PID a gains flous dynamiques - bras 4DDL spatial",
        action_ylabel="N.m",
        gain_history=gain_history_array,
    )
    fig.savefig(output_path, dpi=150)

    final_distance = float(distance_history_array[-1])
    mean_torque = float(np.mean(np.linalg.norm(torque_history_array, axis=1)))
    print("local_fuzzy_rules_per_joint=9")
    print("global_joint_rule_base=not_used")
    print(f"steps={len(distance_history_array) - 1}")
    print(f"done={done}")
    print(f"truncated={info.get('truncated', False)}")
    print(f"desired_joint_angles_rad={np.array2string(desired_q, precision=6)}")
    print(f"final_joint_angles_rad={np.array2string(q_history_array[-1], precision=6)}")
    print(f"final_distance={final_distance:.12e}")
    print(f"final_speed={float(observation['speed']):.12e}")
    print(f"mean_torque_norm={mean_torque:.12e}")
    print(f"mean_kp={np.array2string(np.mean(gain_history_array[:, 0, :], axis=0), precision=6)}")
    print(f"mean_kd={np.array2string(np.mean(gain_history_array[:, 2, :], axis=0), precision=6)}")
    print(f"figure={output_path}")

    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
