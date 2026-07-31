"""Learn a bounded torque residual for a 5-DOF arm under external disturbance."""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
from envs import Arm5DOFDynamicEnv, Arm5DOFDynamicEnvConfig
from rl import PID_RESIDUAL_ACTION_NAMES_5DOF, residual_acceleration_actions_5dof
from robot import inverse_dynamics_torque_5dof, inverse_kinematics_5dof


EXTERNAL_TORQUE = (0.0, -4.0, 0.0, 0.0, 0.0)
RESIDUAL_TORQUE_SCALE = (0.1, 4.0, 2.0, 1.5, 1.0)
EPISODES = 26
MAX_STEPS = 550


@dataclass(frozen=True)
class FixedResidualRollout:
    q_history: np.ndarray
    ee_history: np.ndarray
    distance_history: np.ndarray
    speed_history: np.ndarray
    torque_history: np.ndarray
    rewards: np.ndarray
    action_index: int
    done: bool
    truncated: bool


@dataclass(frozen=True)
class BanditTrainingResult:
    action_values: np.ndarray
    action_counts: np.ndarray
    episode_returns: np.ndarray
    episode_success: np.ndarray
    episode_actions: np.ndarray
    epsilon_history: np.ndarray


def _make_config() -> Arm5DOFDynamicEnvConfig:
    return Arm5DOFDynamicEnvConfig(
        target=(1.25, 0.45, 0.60),
        dt=0.01,
        max_torque=(65.0, 95.0, 70.0, 45.0, 30.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=MAX_STEPS,
    )


def _make_controller() -> FuzzyGainScheduledPIDController:
    return FuzzyGainScheduledPIDController(
        kp=[32.0, 48.0, 38.0, 26.0, 18.0],
        ki=[0.0, 0.0, 0.0, 0.0, 0.0],
        kd=[8.0, 11.0, 9.0, 6.0, 4.5],
        size=5,
        error_scale=[0.35, 0.45, 0.55, 0.55, 0.45],
        derivative_scale=[4.0, 5.0, 5.0, 5.0, 4.0],
        output_limits=(-55.0, 55.0),
    )


def _desired_q(config: Arm5DOFDynamicEnvConfig) -> np.ndarray:
    return inverse_kinematics_5dof(
        config.target,
        config.arm_config.link_lengths,
        elbow="up",
        terminal_pitch=0.0,
        joint_limits=config.arm_config.joint_limits,
    )


def _step_reward(
    previous_distance: float,
    distance: float,
    speed: float,
    torque_norm: float,
    done: bool,
) -> float:
    reward = (
        -distance
        - 0.02 * speed
        - 2e-4 * torque_norm
        + 10.0 * (previous_distance - distance)
    )
    if done:
        reward += 25.0
    return float(reward)


def _rollout_fixed_residual(action_index: int) -> FixedResidualRollout:
    config = _make_config()
    env = Arm5DOFDynamicEnv(config)
    controller = _make_controller()
    residual_actions = residual_acceleration_actions_5dof(RESIDUAL_TORQUE_SCALE)
    residual_torque = residual_actions[action_index]
    desired_q = _desired_q(config)
    observation = env.reset(
        q=[0.0, 0.0, 0.0, 0.0, 0.0],
        q_dot=[0.0, 0.0, 0.0, 0.0, 0.0],
    )

    q_history = [observation["q"].copy()]
    ee_history = [observation["end_effector"].copy()]
    distance_history = [float(observation["distance"])]
    speed_history = [float(observation["speed"])]
    torque_history: list[np.ndarray] = []
    rewards: list[float] = []
    done = False
    info: dict[str, object] = {"truncated": False}
    previous_distance = float(observation["distance"])

    for _ in range(config.max_steps):
        base_acceleration = controller.compute(desired_q, observation["q"], config.dt)
        base_torque = inverse_dynamics_torque_5dof(
            observation["q"],
            observation["q_dot"],
            base_acceleration,
            config.dynamics_config,
        )
        torque = base_torque + residual_torque
        observation, _, done, info = env.step(
            torque,
            external_torque=EXTERNAL_TORQUE,
        )
        distance = float(observation["distance"])
        speed = float(observation["speed"])
        clipped_torque = info["action"].copy()
        reward = _step_reward(
            previous_distance,
            distance,
            speed,
            float(np.linalg.norm(clipped_torque)),
            done,
        )

        q_history.append(observation["q"].copy())
        ee_history.append(observation["end_effector"].copy())
        distance_history.append(distance)
        speed_history.append(speed)
        torque_history.append(clipped_torque)
        rewards.append(reward)
        previous_distance = distance
        if done:
            break

    return FixedResidualRollout(
        q_history=np.asarray(q_history),
        ee_history=np.asarray(ee_history),
        distance_history=np.asarray(distance_history),
        speed_history=np.asarray(speed_history),
        torque_history=np.asarray(torque_history),
        rewards=np.asarray(rewards),
        action_index=action_index,
        done=done,
        truncated=bool(info.get("truncated", False)),
    )


def _episode_score(rollout: FixedResidualRollout) -> float:
    final_distance = float(rollout.distance_history[-1])
    mean_torque = float(np.mean(np.linalg.norm(rollout.torque_history, axis=1)))
    score = float(np.sum(rollout.rewards))
    score += 75.0 if rollout.done else 0.0
    score -= 80.0 * final_distance
    score -= 0.01 * len(rollout.rewards)
    score -= 0.01 * mean_torque
    return score


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    maximum = float(np.max(values))
    candidates = np.flatnonzero(np.isclose(values, maximum))
    return int(rng.choice(candidates))


def _epsilon_at_episode(episode: int) -> float:
    return float(max(0.05, 0.55 * (0.88**episode)))


def _train_bandit(seed: int = 67) -> BanditTrainingResult:
    rng = np.random.default_rng(seed)
    n_actions = len(PID_RESIDUAL_ACTION_NAMES_5DOF)
    action_values = np.zeros(n_actions, dtype=float)
    action_counts = np.zeros(n_actions, dtype=int)
    episode_returns = np.zeros(EPISODES, dtype=float)
    episode_success = np.zeros(EPISODES, dtype=bool)
    episode_actions = np.zeros(EPISODES, dtype=int)
    epsilon_history = np.zeros(EPISODES, dtype=float)

    for episode in range(EPISODES):
        epsilon = _epsilon_at_episode(episode)
        epsilon_history[episode] = epsilon
        if episode < n_actions:
            action = episode
        elif rng.random() < epsilon:
            action = int(rng.integers(n_actions))
        else:
            action = _random_argmax(action_values, rng)

        rollout = _rollout_fixed_residual(action)
        score = _episode_score(rollout)
        action_counts[action] += 1
        action_values[action] += (score - action_values[action]) / action_counts[action]
        episode_returns[episode] = score
        episode_success[episode] = rollout.done
        episode_actions[episode] = action

    return BanditTrainingResult(
        action_values=action_values,
        action_counts=action_counts,
        episode_returns=episode_returns,
        episode_success=episode_success,
        episode_actions=episode_actions,
        epsilon_history=epsilon_history,
    )


def _write_csv(
    path: Path,
    learned_rollout: FixedResidualRollout,
    baseline_rollout: FixedResidualRollout,
    result: BanditTrainingResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "controller",
                "action",
                "done",
                "truncated",
                "steps",
                "final_distance",
                "final_speed",
                "mean_torque_norm",
            ]
        )
        for name, rollout in (
            ("PID_adapte_RL_residuel_couple", learned_rollout),
            ("PID_adapte_seul", baseline_rollout),
        ):
            writer.writerow(
                [
                    name,
                    PID_RESIDUAL_ACTION_NAMES_5DOF[rollout.action_index],
                    rollout.done,
                    rollout.truncated,
                    len(rollout.rewards),
                    f"{rollout.distance_history[-1]:.12e}",
                    f"{rollout.speed_history[-1]:.12e}",
                    f"{np.mean(np.linalg.norm(rollout.torque_history, axis=1)):.12e}",
                ]
            )
        writer.writerow([])
        writer.writerow(["action", "q_value", "count"])
        for index, value in enumerate(result.action_values):
            writer.writerow(
                [
                    PID_RESIDUAL_ACTION_NAMES_5DOF[index],
                    f"{value:.12e}",
                    int(result.action_counts[index]),
                ]
            )


def _write_markdown(
    path: Path,
    learned_rollout: FixedResidualRollout,
    baseline_rollout: FixedResidualRollout,
    result: BanditTrainingResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_action = int(np.argmax(result.action_values))
    lines = [
        "# RL residuel 5DDL sous perturbation externe",
        "",
        f"Couple externe applique : `{EXTERNAL_TORQUE}` N.m.",
        f"Mode residuel : couple moteur borne, action apprise `{PID_RESIDUAL_ACTION_NAMES_5DOF[best_action]}`.",
        "",
        "| Controleur | Action | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rollout in (
        ("PID adapte + RL residuel", learned_rollout),
        ("PID adapte seul", baseline_rollout),
    ):
        lines.append(
            "| "
            f"{name} | {PID_RESIDUAL_ACTION_NAMES_5DOF[rollout.action_index]} | "
            f"{rollout.done} | {len(rollout.rewards)} | "
            f"{rollout.distance_history[-1]:.4e} | {rollout.speed_history[-1]:.4e} | "
            f"{np.mean(np.linalg.norm(rollout.torque_history, axis=1)):.4e} |"
        )
    lines.extend(
        [
            "",
            "Valeurs Q finales du bandit residuel :",
            "",
            "| Action | Valeur Q | Essais |",
            "|---|---:|---:|",
        ]
    )
    for index, value in enumerate(result.action_values):
        lines.append(
            f"| {PID_RESIDUAL_ACTION_NAMES_5DOF[index]} | "
            f"{value:.4e} | {int(result.action_counts[index])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation : le PID adapte seul garde une erreur statique sous",
            "perturbation constante. Le residu RL en couple apprend l'action qui",
            "annule le biais moteur dominant et ramene la trajectoire dans la",
            "tolerance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_plot(
    path: Path,
    result: BanditTrainingResult,
    learned_rollout: FixedResidualRollout,
    baseline_rollout: FixedResidualRollout,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_learning, ax_distance, ax_values, ax_torque = axes.ravel()

    ax_learning.plot(result.episode_returns, label="score episode")
    ax_learning_success = ax_learning.twinx()
    ax_learning_success.step(
        np.arange(EPISODES),
        result.episode_success.astype(float),
        where="post",
        color="tab:green",
        alpha=0.65,
        label="succes",
    )
    ax_learning.set_title("Apprentissage du residu de couple")
    ax_learning.set_xlabel("episode")
    ax_learning.set_ylabel("score")
    ax_learning_success.set_ylabel("succes")
    ax_learning.grid(True, alpha=0.3)
    lines, labels = ax_learning.get_legend_handles_labels()
    success_lines, success_labels = ax_learning_success.get_legend_handles_labels()
    ax_learning.legend(lines + success_lines, labels + success_labels, loc="lower right")

    ax_distance.plot(
        baseline_rollout.distance_history,
        linestyle="--",
        label="PID adapte seul",
    )
    ax_distance.plot(learned_rollout.distance_history, label="PID adapte + RL")
    ax_distance.axhline(1e-2, linestyle=":", color="tab:green", label="tolerance")
    ax_distance.set_title("Distance cible sous perturbation")
    ax_distance.set_xlabel("iteration")
    ax_distance.set_ylabel("distance")
    ax_distance.grid(True, alpha=0.3)
    ax_distance.legend()

    x = np.arange(len(PID_RESIDUAL_ACTION_NAMES_5DOF))
    ax_values.bar(x, result.action_values)
    ax_values.set_xticks(x)
    ax_values.set_xticklabels(PID_RESIDUAL_ACTION_NAMES_5DOF, rotation=30, ha="right")
    ax_values.set_title("Valeurs Q des actions residuelles")
    ax_values.set_ylabel("valeur")
    ax_values.grid(True, axis="y", alpha=0.3)

    ax_torque.plot(
        np.linalg.norm(baseline_rollout.torque_history, axis=1),
        linestyle="--",
        label="PID adapte seul",
    )
    ax_torque.plot(
        np.linalg.norm(learned_rollout.torque_history, axis=1),
        label="PID adapte + RL",
    )
    ax_torque.set_title("Norme du couple moteur")
    ax_torque.set_xlabel("iteration")
    ax_torque.set_ylabel("N.m")
    ax_torque.grid(True, alpha=0.3)
    ax_torque.legend()

    fig.suptitle("RL residuel borne sous perturbation externe - bras 5DDL")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    result = _train_bandit()
    best_action = int(np.argmax(result.action_values))
    learned_rollout = _rollout_fixed_residual(best_action)
    baseline_rollout = _rollout_fixed_residual(0)

    table_path = ROOT / "results" / "tables" / "step_32_pid_residual_disturbance_5dof.csv"
    markdown_path = ROOT / "results" / "tables" / "step_32_pid_residual_disturbance_5dof.md"
    figure_path = ROOT / "results" / "figures" / "step_32_pid_residual_disturbance_5dof.png"
    _write_csv(table_path, learned_rollout, baseline_rollout, result)
    _write_markdown(markdown_path, learned_rollout, baseline_rollout, result)
    _save_plot(figure_path, result, learned_rollout, baseline_rollout)

    print(f"external_torque={EXTERNAL_TORQUE}")
    print("residual_mode=torque_bandit")
    print(f"episodes={EPISODES}")
    print(f"best_action={PID_RESIDUAL_ACTION_NAMES_5DOF[best_action]}")
    print(f"best_action_value={result.action_values[best_action]:.12e}")
    print(f"success_rate={np.mean(result.episode_success):.3f}")
    print(f"learned_done={learned_rollout.done}")
    print(f"learned_steps={len(learned_rollout.rewards)}")
    print(f"learned_final_distance={learned_rollout.distance_history[-1]:.12e}")
    print(f"learned_final_speed={learned_rollout.speed_history[-1]:.12e}")
    print(f"baseline_done={baseline_rollout.done}")
    print(f"baseline_steps={len(baseline_rollout.rewards)}")
    print(f"baseline_final_distance={baseline_rollout.distance_history[-1]:.12e}")
    print(f"baseline_final_speed={baseline_rollout.speed_history[-1]:.12e}")
    print(f"table={table_path}")
    print(f"markdown={markdown_path}")
    print(f"figure={figure_path}")
    return 0 if learned_rollout.done and not baseline_rollout.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
