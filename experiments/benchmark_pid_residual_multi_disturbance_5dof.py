"""Evaluate axis-aligned residual limits under multi-joint 5-DOF disturbances."""

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


MAX_STEPS = 550
RESIDUAL_TORQUE_SCALE = (0.1, 4.0, 3.0, 1.5, 1.0)


@dataclass(frozen=True)
class DisturbanceScenario:
    name: str
    external_torque: tuple[float, float, float, float, float]
    multi_axis_residual: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class FixedResidualRollout:
    scenario: str
    controller: str
    action: str
    external_torque: tuple[float, float, float, float, float]
    residual_torque: np.ndarray
    q_history: np.ndarray
    ee_history: np.ndarray
    distance_history: np.ndarray
    speed_history: np.ndarray
    torque_history: np.ndarray
    rewards: np.ndarray
    done: bool
    truncated: bool

    @property
    def steps(self) -> int:
        return int(self.rewards.size)

    @property
    def final_distance(self) -> float:
        return float(self.distance_history[-1])

    @property
    def final_speed(self) -> float:
        return float(self.speed_history[-1])

    @property
    def mean_torque_norm(self) -> float:
        return float(np.mean(np.linalg.norm(self.torque_history, axis=1)))


SCENARIOS = (
    DisturbanceScenario(
        name="single_q1",
        external_torque=(0.0, -4.0, 0.0, 0.0, 0.0),
        multi_axis_residual=(0.0, 4.0, 0.0, 0.0, 0.0),
    ),
    DisturbanceScenario(
        name="single_q2",
        external_torque=(0.0, 0.0, -3.0, 0.0, 0.0),
        multi_axis_residual=(0.0, 0.0, 3.0, 0.0, 0.0),
    ),
    DisturbanceScenario(
        name="multi_q1_q2",
        external_torque=(0.0, -4.0, -3.0, 0.0, 0.0),
        multi_axis_residual=(0.0, 4.0, 3.0, 0.0, 0.0),
    ),
    DisturbanceScenario(
        name="multi_q1_q3",
        external_torque=(0.0, -4.0, 0.0, -1.0, 0.0),
        multi_axis_residual=(0.0, 4.0, 0.0, 1.0, 0.0),
    ),
)


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


def _rollout_fixed_residual(
    scenario: DisturbanceScenario,
    controller_name: str,
    action_name: str,
    residual_torque: np.ndarray,
) -> FixedResidualRollout:
    config = _make_config()
    env = Arm5DOFDynamicEnv(config)
    controller = _make_controller()
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
            external_torque=scenario.external_torque,
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
        scenario=scenario.name,
        controller=controller_name,
        action=action_name,
        external_torque=scenario.external_torque,
        residual_torque=residual_torque.copy(),
        q_history=np.asarray(q_history),
        ee_history=np.asarray(ee_history),
        distance_history=np.asarray(distance_history),
        speed_history=np.asarray(speed_history),
        torque_history=np.asarray(torque_history),
        rewards=np.asarray(rewards),
        done=done,
        truncated=bool(info.get("truncated", False)),
    )


def _episode_score(rollout: FixedResidualRollout) -> float:
    score = float(np.sum(rollout.rewards))
    score += 75.0 if rollout.done else 0.0
    score -= 80.0 * rollout.final_distance
    score -= 0.01 * rollout.steps
    score -= 0.01 * rollout.mean_torque_norm
    return score


def _evaluate_scenario(
    scenario: DisturbanceScenario,
) -> tuple[list[FixedResidualRollout], FixedResidualRollout]:
    residual_actions = residual_acceleration_actions_5dof(RESIDUAL_TORQUE_SCALE)
    action_rollouts = [
        _rollout_fixed_residual(
            scenario,
            "action_axis_alignee",
            PID_RESIDUAL_ACTION_NAMES_5DOF[index],
            residual_actions[index],
        )
        for index in range(len(PID_RESIDUAL_ACTION_NAMES_5DOF))
    ]
    multi_axis_rollout = _rollout_fixed_residual(
        scenario,
        "compensation_multi_axes",
        "multi_axis_reference",
        np.asarray(scenario.multi_axis_residual, dtype=float),
    )
    return action_rollouts, multi_axis_rollout


def _write_csv(path: Path, rows: list[FixedResidualRollout]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "controller",
                "action",
                "external_torque",
                "residual_torque",
                "done",
                "truncated",
                "steps",
                "final_distance",
                "final_speed",
                "mean_torque_norm",
                "score",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.scenario,
                    row.controller,
                    row.action,
                    tuple(float(value) for value in row.external_torque),
                    tuple(float(value) for value in row.residual_torque),
                    row.done,
                    row.truncated,
                    row.steps,
                    f"{row.final_distance:.12e}",
                    f"{row.final_speed:.12e}",
                    f"{row.mean_torque_norm:.12e}",
                    f"{_episode_score(row):.12e}",
                ]
            )


def _write_markdown(
    path: Path,
    summaries: list[tuple[DisturbanceScenario, FixedResidualRollout, FixedResidualRollout, FixedResidualRollout]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Perturbations multiples 5DDL",
        "",
        "| Scenario | Perturbation | Controleur | Action | Succes | Pas | Distance finale | Couple moyen |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, baseline, best_axis, multi_axis in summaries:
        for row in (baseline, best_axis, multi_axis):
            lines.append(
                "| "
                f"{scenario.name} | `{scenario.external_torque}` | {row.controller} | "
                f"{row.action} | {row.done} | {row.steps} | "
                f"{row.final_distance:.4e} | {row.mean_torque_norm:.4e} |"
            )
    lines.extend(
        [
            "",
            "Interpretation : les perturbations mono-articulaires sont corrigees",
            "par une action axis-alignee unique. Les perturbations simultanees",
            "restent hors tolerance avec une seule action, alors qu'une compensation",
            "multi-axes de reference reussit. L'espace d'actions `1 + 2n` est donc",
            "suffisant pour des biais dominants isoles, mais limite pour plusieurs",
            "biais independants appliques en meme temps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_plot(
    path: Path,
    summaries: list[tuple[DisturbanceScenario, FixedResidualRollout, FixedResidualRollout, FixedResidualRollout]],
) -> None:
    import matplotlib.pyplot as plt

    labels = [scenario.name for scenario, _, _, _ in summaries]
    controllers = ("PID seul", "meilleure action", "multi-axes")
    x = np.arange(len(labels))
    width = 0.24

    grouped_rows = [
        (baseline, best_axis, multi_axis)
        for _, baseline, best_axis, multi_axis in summaries
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_distance, ax_steps, ax_torque, ax_success = axes.ravel()

    for offset, controller_index, label in zip(
        (-width, 0.0, width),
        range(3),
        controllers,
    ):
        rows = [group[controller_index] for group in grouped_rows]
        ax_distance.bar(
            x + offset,
            [row.final_distance for row in rows],
            width,
            label=label,
        )
        ax_steps.bar(x + offset, [row.steps for row in rows], width, label=label)
        ax_torque.bar(
            x + offset,
            [row.mean_torque_norm for row in rows],
            width,
            label=label,
        )
        ax_success.bar(
            x + offset,
            [1.0 if row.done else 0.0 for row in rows],
            width,
            label=label,
        )

    ax_distance.axhline(1e-2, linestyle=":", color="tab:green", label="tolerance")
    ax_distance.set_yscale("log")
    ax_distance.set_title("Distance finale")
    ax_distance.set_ylabel("m")
    ax_steps.set_title("Nombre de pas")
    ax_steps.set_ylabel("pas")
    ax_torque.set_title("Couple moyen")
    ax_torque.set_ylabel("N.m")
    ax_success.set_title("Succes")
    ax_success.set_ylim(0.0, 1.15)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Limites des actions residuelles axis-alignees - bras 5DDL")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    all_rows: list[FixedResidualRollout] = []
    summaries: list[
        tuple[
            DisturbanceScenario,
            FixedResidualRollout,
            FixedResidualRollout,
            FixedResidualRollout,
        ]
    ] = []
    for scenario in SCENARIOS:
        action_rollouts, multi_axis_rollout = _evaluate_scenario(scenario)
        baseline = action_rollouts[0]
        best_axis = max(action_rollouts, key=_episode_score)
        all_rows.extend(action_rollouts)
        all_rows.append(multi_axis_rollout)
        summaries.append((scenario, baseline, best_axis, multi_axis_rollout))

    table_path = ROOT / "results" / "tables" / "step_34_multi_disturbance_5dof.csv"
    markdown_path = ROOT / "results" / "tables" / "step_34_multi_disturbance_5dof.md"
    figure_path = ROOT / "results" / "figures" / "step_34_multi_disturbance_5dof.png"
    _write_csv(table_path, all_rows)
    _write_markdown(markdown_path, summaries)
    _save_plot(figure_path, summaries)

    axis_success = sum(best_axis.done for _, _, best_axis, _ in summaries)
    multi_axis_success = sum(multi_axis.done for _, _, _, multi_axis in summaries)
    print(f"scenario_count={len(summaries)}")
    print(f"axis_aligned_best_success={axis_success}/{len(summaries)}")
    print(f"multi_axis_reference_success={multi_axis_success}/{len(summaries)}")
    for scenario, baseline, best_axis, multi_axis in summaries:
        print(
            f"{scenario.name}: baseline_done={baseline.done}, "
            f"best_axis={best_axis.action}, best_axis_done={best_axis.done}, "
            f"best_axis_distance={best_axis.final_distance:.12e}, "
            f"multi_axis_done={multi_axis.done}, "
            f"multi_axis_distance={multi_axis.final_distance:.12e}"
        )
    print(f"table={table_path}")
    print(f"markdown={markdown_path}")
    print(f"figure={figure_path}")
    return 0 if multi_axis_success == len(summaries) and axis_success < len(summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
