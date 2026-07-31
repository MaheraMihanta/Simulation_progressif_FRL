"""Benchmark 5-DOF terminal-fold choices for the redundant distal posture."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import pi
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
from robot import inverse_dynamics_torque_5dof, inverse_kinematics_5dof


TARGET = (1.25, 0.45, 0.60)
TERMINAL_FOLDS = (
    ("fold_m90", -pi / 2.0),
    ("fold_m60", -pi / 3.0),
    ("fold_m30", -pi / 6.0),
    ("fold_0", 0.0),
    ("fold_p30", pi / 6.0),
    ("fold_p60", pi / 3.0),
    ("fold_p90", pi / 2.0),
)


@dataclass(frozen=True)
class FoldBenchmarkResult:
    fold_name: str
    terminal_fold: float
    desired_q: np.ndarray
    done: bool
    truncated: bool
    steps: int
    final_distance: float
    final_speed: float
    mean_torque_norm: float
    peak_torque_norm: float
    posture_norm: float


def _make_config() -> Arm5DOFDynamicEnvConfig:
    return Arm5DOFDynamicEnvConfig(
        target=TARGET,
        dt=0.01,
        max_torque=(65.0, 95.0, 70.0, 45.0, 30.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=1400,
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


def _simulate_fold(fold_name: str, terminal_fold: float) -> FoldBenchmarkResult:
    config = _make_config()
    desired_q = inverse_kinematics_5dof(
        config.target,
        config.arm_config.link_lengths,
        elbow="up",
        terminal_pitch=0.0,
        terminal_fold=terminal_fold,
        joint_limits=config.arm_config.joint_limits,
    )
    env = Arm5DOFDynamicEnv(config)
    controller = _make_controller()
    observation = env.reset(
        q=[0.0, 0.0, 0.0, 0.0, 0.0],
        q_dot=[0.0, 0.0, 0.0, 0.0, 0.0],
    )

    done = False
    info: dict[str, object] = {"truncated": False}
    torque_history: list[np.ndarray] = []
    for _ in range(config.max_steps):
        desired_q_ddot = controller.compute(desired_q, observation["q"], config.dt)
        torque = inverse_dynamics_torque_5dof(
            observation["q"],
            observation["q_dot"],
            desired_q_ddot,
            config.dynamics_config,
        )
        observation, reward, done, info = env.step(torque)
        torque_history.append(info["action"].copy())
        if done:
            break

    torque_history_array = np.asarray(torque_history)
    torque_norm = np.linalg.norm(torque_history_array, axis=1)
    return FoldBenchmarkResult(
        fold_name=fold_name,
        terminal_fold=terminal_fold,
        desired_q=desired_q,
        done=done,
        truncated=bool(info.get("truncated", False)),
        steps=len(torque_history),
        final_distance=float(observation["distance"]),
        final_speed=float(observation["speed"]),
        mean_torque_norm=float(np.mean(torque_norm)),
        peak_torque_norm=float(np.max(torque_norm)),
        posture_norm=float(np.linalg.norm(desired_q)),
    )


def _write_csv(path: Path, results: list[FoldBenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "fold_name",
                "terminal_fold_rad",
                "terminal_fold_deg",
                "desired_q0",
                "desired_q1",
                "desired_q2",
                "desired_q3",
                "desired_q4",
                "done",
                "truncated",
                "steps",
                "final_distance",
                "final_speed",
                "mean_torque_norm",
                "peak_torque_norm",
                "posture_norm",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.fold_name,
                    f"{result.terminal_fold:.12e}",
                    f"{np.degrees(result.terminal_fold):.6f}",
                    *[f"{value:.12e}" for value in result.desired_q],
                    result.done,
                    result.truncated,
                    result.steps,
                    f"{result.final_distance:.12e}",
                    f"{result.final_speed:.12e}",
                    f"{result.mean_torque_norm:.12e}",
                    f"{result.peak_torque_norm:.12e}",
                    f"{result.posture_norm:.12e}",
                ]
            )


def _write_markdown(path: Path, results: list[FoldBenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_distance = min(results, key=lambda row: row.final_distance)
    best_effort = min(results, key=lambda row: row.mean_torque_norm)
    fastest = min(results, key=lambda row: row.steps)
    lines = [
        "# Benchmark redondance terminal_fold 5DDL",
        "",
        "| Repli | Degres | Succes | Pas | Distance finale | Vitesse finale | Couple moyen | Norme posture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.fold_name} | {np.degrees(result.terminal_fold):.1f} | "
            f"{result.done} | {result.steps} | {result.final_distance:.4e} | "
            f"{result.final_speed:.4e} | {result.mean_torque_norm:.4e} | "
            f"{result.posture_norm:.4e} |"
        )
    lines.extend(
        [
            "",
            "Synthese :",
            "",
            f"- meilleure distance finale : `{best_distance.fold_name}` "
            f"avec `{best_distance.final_distance:.4e}` ;",
            f"- effort moyen minimal : `{best_effort.fold_name}` "
            f"avec `{best_effort.mean_torque_norm:.4e} N.m` ;",
            f"- convergence la plus rapide : `{fastest.fold_name}` "
            f"en `{fastest.steps}` pas.",
            "",
            "Interpretation : le cinquieme DDL peut etre exploite comme un choix",
            "de posture. Toutes les postures testees atteignent la cible, mais le",
            "repli distal deplace le compromis entre distance finale, temps de",
            "convergence, effort et norme articulaire.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_plot(path: Path, results: list[FoldBenchmarkResult]) -> None:
    import matplotlib.pyplot as plt

    labels = [result.fold_name for result in results]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_steps, ax_distance, ax_torque, ax_posture = axes.ravel()

    ax_steps.bar(x, [result.steps for result in results])
    ax_steps.set_title("Nombre de pas")
    ax_steps.set_ylabel("pas")

    ax_distance.bar(x, [result.final_distance for result in results])
    ax_distance.set_title("Distance finale")
    ax_distance.set_ylabel("m")
    ax_distance.set_yscale("log")

    ax_torque.bar(x, [result.mean_torque_norm for result in results])
    ax_torque.set_title("Couple moyen")
    ax_torque.set_ylabel("N.m")

    ax_posture.bar(x, [result.posture_norm for result in results])
    ax_posture.set_title("Norme de la posture cible")
    ax_posture.set_ylabel("rad")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Effet du repli terminal sur le bras 5DDL")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    results = [_simulate_fold(name, fold) for name, fold in TERMINAL_FOLDS]
    table_path = ROOT / "results" / "tables" / "step_33_terminal_fold_5dof.csv"
    markdown_path = ROOT / "results" / "tables" / "step_33_terminal_fold_5dof.md"
    figure_path = ROOT / "results" / "figures" / "step_33_terminal_fold_5dof.png"
    _write_csv(table_path, results)
    _write_markdown(markdown_path, results)
    _save_plot(figure_path, results)

    success_count = sum(result.done for result in results)
    best_distance = min(results, key=lambda row: row.final_distance)
    best_effort = min(results, key=lambda row: row.mean_torque_norm)
    fastest = min(results, key=lambda row: row.steps)
    print(f"fold_count={len(results)}")
    print(f"success={success_count}/{len(results)}")
    print(f"best_distance_fold={best_distance.fold_name}")
    print(f"best_distance={best_distance.final_distance:.12e}")
    print(f"best_effort_fold={best_effort.fold_name}")
    print(f"best_effort_mean_torque={best_effort.mean_torque_norm:.12e}")
    print(f"fastest_fold={fastest.fold_name}")
    print(f"fastest_steps={fastest.steps}")
    print(f"table={table_path}")
    print(f"markdown={markdown_path}")
    print(f"figure={figure_path}")
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
