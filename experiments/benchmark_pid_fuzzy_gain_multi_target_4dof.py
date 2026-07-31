"""Benchmark PID and fuzzy gain-scheduled PID on several 4-DOF targets."""

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

from controllers import FuzzyGainScheduledPIDController, PIDController
from envs import Arm4DOFDynamicEnv, Arm4DOFDynamicEnvConfig
from robot import inverse_dynamics_torque_4dof, inverse_kinematics_4dof


TARGETS = (
    ("T1_reference", (1.15, 0.45, 0.55)),
    ("T2_lateral_bas", (1.45, -0.35, 0.35)),
    ("T3_haut_diagonal", (0.85, 0.85, 0.75)),
    ("T4_avant_droit", (1.25, 0.75, -0.20)),
    ("T5_arriere_haut", (0.70, -0.95, 0.60)),
)


@dataclass(frozen=True)
class BenchmarkResult:
    target_name: str
    target: tuple[float, float, float]
    controller: str
    done: bool
    truncated: bool
    steps: int
    final_distance: float
    final_speed: float
    mean_torque_norm: float
    peak_torque_norm: float


def _make_controller(kind: str):
    if kind == "PID":
        return PIDController(
            kp=[28.0, 42.0, 34.0, 22.0],
            ki=[0.0, 0.0, 0.0, 0.0],
            kd=[7.0, 10.0, 8.0, 5.0],
            size=4,
            output_limits=(-45.0, 45.0),
        )
    if kind == "PID_gains_flous":
        return FuzzyGainScheduledPIDController(
            kp=[28.0, 42.0, 34.0, 22.0],
            ki=[0.0, 0.0, 0.0, 0.0],
            kd=[7.0, 10.0, 8.0, 5.0],
            size=4,
            error_scale=[0.35, 0.45, 0.55, 0.55],
            derivative_scale=[4.0, 5.0, 5.0, 5.0],
            output_limits=(-45.0, 45.0),
        )
    raise ValueError(f"Unknown controller kind: {kind}")


def _simulate_target(
    target_name: str,
    target: tuple[float, float, float],
    controller_kind: str,
) -> BenchmarkResult:
    config = Arm4DOFDynamicEnvConfig(
        target=target,
        dt=0.01,
        max_torque=(55.0, 85.0, 60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=1400,
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
    controller = _make_controller(controller_kind)

    done = False
    info: dict[str, object] = {"truncated": False}
    torque_history: list[np.ndarray] = []

    for _ in range(config.max_steps):
        desired_q_ddot = controller.compute(desired_q, observation["q"], config.dt)
        torque = inverse_dynamics_torque_4dof(
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
    return BenchmarkResult(
        target_name=target_name,
        target=target,
        controller=controller_kind,
        done=done,
        truncated=bool(info.get("truncated", False)),
        steps=len(torque_history),
        final_distance=float(observation["distance"]),
        final_speed=float(observation["speed"]),
        mean_torque_norm=float(np.mean(torque_norm)),
        peak_torque_norm=float(np.max(torque_norm)),
    )


def _write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_name",
                "target_x",
                "target_y",
                "target_z",
                "controller",
                "done",
                "truncated",
                "steps",
                "final_distance",
                "final_speed",
                "mean_torque_norm",
                "peak_torque_norm",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.target_name,
                    *result.target,
                    result.controller,
                    result.done,
                    result.truncated,
                    result.steps,
                    f"{result.final_distance:.12e}",
                    f"{result.final_speed:.12e}",
                    f"{result.mean_torque_norm:.12e}",
                    f"{result.peak_torque_norm:.12e}",
                ]
            )


def _write_markdown(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark multi-cibles 4DDL",
        "",
        "| Cible | Controleur | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.target_name} | {result.controller} | {result.done} | "
            f"{result.steps} | {result.final_distance:.4e} | "
            f"{result.final_speed:.4e} | {result.mean_torque_norm:.4e} |"
        )
    lines.extend(
        [
            "",
            "Interpretation courte : le PID a gains flous conserve la meme",
            "architecture de commande que le PID dynamique, mais module localement",
            "`Kp`, `Ki` et `Kd` sans base floue globale.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_plot(path: Path, results: list[BenchmarkResult]) -> None:
    import matplotlib.pyplot as plt

    target_names = [name for name, _ in TARGETS]
    controllers = ("PID", "PID_gains_flous")
    x = np.arange(len(target_names))
    width = 0.36

    values = {
        controller: [result for result in results if result.controller == controller]
        for controller in controllers
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_steps, ax_distance, ax_torque, ax_speed = axes.ravel()
    for offset, controller in zip((-width / 2, width / 2), controllers):
        rows = values[controller]
        label = "PID" if controller == "PID" else "PID gains flous"
        ax_steps.bar(x + offset, [row.steps for row in rows], width, label=label)
        ax_distance.bar(
            x + offset,
            [row.final_distance for row in rows],
            width,
            label=label,
        )
        ax_torque.bar(
            x + offset,
            [row.mean_torque_norm for row in rows],
            width,
            label=label,
        )
        ax_speed.bar(
            x + offset,
            [row.final_speed for row in rows],
            width,
            label=label,
        )

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(target_names, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    ax_steps.set_title("Nombre de pas")
    ax_steps.set_ylabel("pas")
    ax_distance.set_title("Distance finale")
    ax_distance.set_ylabel("m")
    ax_distance.set_yscale("log")
    ax_torque.set_title("Couple moyen")
    ax_torque.set_ylabel("N.m")
    ax_speed.set_title("Vitesse finale")
    ax_speed.set_ylabel("rad/s")
    fig.suptitle("Comparaison multi-cibles PID vs PID a gains flous - 4DDL")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    results: list[BenchmarkResult] = []
    for target_name, target in TARGETS:
        for controller_kind in ("PID", "PID_gains_flous"):
            results.append(_simulate_target(target_name, target, controller_kind))

    table_path = ROOT / "results" / "tables" / "step_26_pid_vs_fuzzy_gain_4dof.csv"
    markdown_path = ROOT / "results" / "tables" / "step_26_pid_vs_fuzzy_gain_4dof.md"
    figure_path = ROOT / "results" / "figures" / "step_26_pid_vs_fuzzy_gain_4dof.png"
    _write_csv(table_path, results)
    _write_markdown(markdown_path, results)
    _save_plot(figure_path, results)

    pid_rows = [result for result in results if result.controller == "PID"]
    fuzzy_rows = [result for result in results if result.controller == "PID_gains_flous"]
    pid_success = sum(result.done for result in pid_rows)
    fuzzy_success = sum(result.done for result in fuzzy_rows)
    print(f"target_count={len(TARGETS)}")
    print(f"pid_success={pid_success}/{len(pid_rows)}")
    print(f"fuzzy_gain_pid_success={fuzzy_success}/{len(fuzzy_rows)}")
    print(f"pid_mean_steps={np.mean([row.steps for row in pid_rows]):.3f}")
    print(f"fuzzy_gain_pid_mean_steps={np.mean([row.steps for row in fuzzy_rows]):.3f}")
    print(
        "pid_mean_final_distance="
        f"{np.mean([row.final_distance for row in pid_rows]):.12e}"
    )
    print(
        "fuzzy_gain_pid_mean_final_distance="
        f"{np.mean([row.final_distance for row in fuzzy_rows]):.12e}"
    )
    print(f"table={table_path}")
    print(f"markdown={markdown_path}")
    print(f"figure={figure_path}")
    return 0 if pid_success == len(pid_rows) and fuzzy_success == len(fuzzy_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
