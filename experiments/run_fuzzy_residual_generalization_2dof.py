"""Evaluate fuzzy residual Q-learning generalization across target positions."""

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

from envs import Arm2DOFDynamicEnvConfig
from rl import (
    FuzzyDynamicStateEncoder,
    FuzzyResidualQLearningConfig,
    rollout_fuzzy_residual_q_policy,
    train_fuzzy_residual_q_learning,
)


@dataclass(frozen=True)
class EvaluationRow:
    """One method evaluation on one Cartesian target."""

    target_id: str
    target_x: float
    target_y: float
    method: str
    done: bool
    steps: int
    final_distance: float
    final_speed: float
    mean_torque_norm: float
    total_reward: float


TARGETS = (
    ("T1_train", (1.10, 0.55)),
    ("T2_diag", (0.85, 0.85)),
    ("T3_low", (1.25, 0.25)),
    ("T4_high", (0.65, 1.05)),
    ("T5_far", (1.35, 0.45)),
)


def _make_env_config(target: tuple[float, float]) -> Arm2DOFDynamicEnvConfig:
    return Arm2DOFDynamicEnvConfig(
        target=target,
        dt=0.01,
        max_torque=(60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=550,
    )


def _metrics(target_id: str, target: tuple[float, float], method: str, rollout) -> EvaluationRow:
    mean_torque = (
        float(np.mean(np.linalg.norm(rollout.torque_history, axis=1)))
        if rollout.torque_history.size
        else 0.0
    )
    return EvaluationRow(
        target_id=target_id,
        target_x=float(target[0]),
        target_y=float(target[1]),
        method=method,
        done=bool(rollout.done),
        steps=len(rollout.action_indices),
        final_distance=float(rollout.distance_history[-1]),
        final_speed=float(rollout.speed_history[-1]),
        mean_torque_norm=mean_torque,
        total_reward=float(np.sum(rollout.rewards)),
    )


def _write_csv(rows: list[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvaluationRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _target_pairs(rows: list[EvaluationRow]) -> dict[str, dict[str, EvaluationRow]]:
    grouped: dict[str, dict[str, EvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.target_id, {})[row.method] = row
    return grouped


def _write_markdown(rows: list[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _target_pairs(rows)
    lines = [
        "# Step 11 - Generalisation flou/RL",
        "",
        "| Cible | Methode | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target_id, methods in grouped.items():
        for method in ("fuzzy_base", "fuzzy_rl"):
            row = methods[method]
            lines.append(
                "| "
                f"{target_id} ({row.target_x:.2f}, {row.target_y:.2f}) | "
                f"{method} | {int(row.done)} | {row.steps} | "
                f"{row.final_distance:.6f} | {row.final_speed:.6f} | "
                f"{row.mean_torque_norm:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Ecarts flou/RL - flou seul",
            "",
            "| Cible | Delta pas | Delta distance | Delta couple | Interpretation courte |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for target_id, methods in grouped.items():
        base = methods["fuzzy_base"]
        learned = methods["fuzzy_rl"]
        delta_steps = learned.steps - base.steps
        delta_distance = learned.final_distance - base.final_distance
        delta_torque = learned.mean_torque_norm - base.mean_torque_norm
        if not learned.done:
            interpretation = "degradation: la politique apprise ne converge pas"
        elif delta_steps < 0 and delta_torque > 0.0:
            interpretation = "plus rapide, mais plus couteux en effort"
        elif delta_steps < 0:
            interpretation = "plus rapide"
        elif delta_steps == 0:
            interpretation = "comportement proche de la base floue"
        else:
            interpretation = "moins rapide que la base floue"
        lines.append(
            "| "
            f"{target_id} | {delta_steps:+d} | {delta_distance:+.6f} | "
            f"{delta_torque:+.6f} | {interpretation} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(rows: list[EvaluationRow], path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _target_pairs(rows)
    target_ids = list(grouped)
    x = np.arange(len(target_ids), dtype=float)
    width = 0.36

    base_rows = [grouped[target]["fuzzy_base"] for target in target_ids]
    learned_rows = [grouped[target]["fuzzy_rl"] for target in target_ids]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_steps, ax_distance, ax_torque, ax_success = axes.ravel()

    ax_steps.bar(x - width / 2, [row.steps for row in base_rows], width, label="flou")
    ax_steps.bar(x + width / 2, [row.steps for row in learned_rows], width, label="flou + Q")
    ax_steps.set_title("Temps de convergence")
    ax_steps.set_ylabel("pas")
    ax_steps.set_xticks(x, target_ids, rotation=20)
    ax_steps.grid(True, axis="y", alpha=0.3)
    ax_steps.legend()

    ax_distance.bar(
        x - width / 2,
        [row.final_distance for row in base_rows],
        width,
        label="flou",
    )
    ax_distance.bar(
        x + width / 2,
        [row.final_distance for row in learned_rows],
        width,
        label="flou + Q",
    )
    ax_distance.axhline(1e-2, linestyle=":", color="tab:green", linewidth=1, label="tolerance")
    ax_distance.set_title("Erreur finale")
    ax_distance.set_ylabel("distance")
    ax_distance.set_xticks(x, target_ids, rotation=20)
    ax_distance.grid(True, axis="y", alpha=0.3)
    ax_distance.legend()

    ax_torque.bar(
        x - width / 2,
        [row.mean_torque_norm for row in base_rows],
        width,
        label="flou",
    )
    ax_torque.bar(
        x + width / 2,
        [row.mean_torque_norm for row in learned_rows],
        width,
        label="flou + Q",
    )
    ax_torque.set_title("Effort moyen")
    ax_torque.set_ylabel("N.m")
    ax_torque.set_xticks(x, target_ids, rotation=20)
    ax_torque.grid(True, axis="y", alpha=0.3)
    ax_torque.legend()

    ax_success.bar(
        x - width / 2,
        [int(row.done) for row in base_rows],
        width,
        label="flou",
    )
    ax_success.bar(
        x + width / 2,
        [int(row.done) for row in learned_rows],
        width,
        label="flou + Q",
    )
    ax_success.set_title("Succes")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.set_ylabel("0/1")
    ax_success.set_xticks(x, target_ids, rotation=20)
    ax_success.grid(True, axis="y", alpha=0.3)
    ax_success.legend()

    fig.suptitle("Generalisation du Q-learning residuel flou")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    encoder = FuzzyDynamicStateEncoder(
        error_scale=(0.9, 1.2),
        velocity_scale=(6.0, 6.0),
    )
    learning_config = FuzzyResidualQLearningConfig(
        episodes=220,
        max_steps_per_episode=550,
        alpha=0.35,
        gamma=0.97,
        epsilon_start=0.75,
        epsilon_end=0.04,
        epsilon_decay=0.985,
        residual_acceleration_scale=(1.5, 1.5),
        initial_q_value=-0.5,
        seed=17,
    )
    train_target_id, train_target = TARGETS[0]
    train_env_config = _make_env_config(train_target)

    result = train_fuzzy_residual_q_learning(
        train_env_config,
        encoder=encoder,
        config=learning_config,
    )

    rows: list[EvaluationRow] = []
    zero_q_value = np.zeros_like(result.q_value)
    for target_id, target in TARGETS:
        env_config = _make_env_config(target)
        base_rollout = rollout_fuzzy_residual_q_policy(
            env_config,
            zero_q_value,
            encoder,
            config=learning_config,
        )
        learned_rollout = rollout_fuzzy_residual_q_policy(
            env_config,
            result.q_value,
            encoder,
            config=learning_config,
        )
        rows.append(_metrics(target_id, target, "fuzzy_base", base_rollout))
        rows.append(_metrics(target_id, target, "fuzzy_rl", learned_rollout))

    figure_path = ROOT / "results" / "figures" / "step_11_fuzzy_residual_generalization_2dof.png"
    csv_path = ROOT / "results" / "tables" / "step_11_fuzzy_residual_generalization_2dof.csv"
    markdown_path = ROOT / "results" / "tables" / "step_11_fuzzy_residual_generalization_2dof.md"
    _plot_summary(rows, figure_path)
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)

    grouped = _target_pairs(rows)
    base_successes = sum(int(grouped[target]["fuzzy_base"].done) for target in grouped)
    learned_successes = sum(int(grouped[target]["fuzzy_rl"].done) for target in grouped)
    step_deltas = [
        grouped[target]["fuzzy_rl"].steps - grouped[target]["fuzzy_base"].steps
        for target in grouped
    ]
    torque_deltas = [
        grouped[target]["fuzzy_rl"].mean_torque_norm
        - grouped[target]["fuzzy_base"].mean_torque_norm
        for target in grouped
    ]
    last_window = min(60, learning_config.episodes)
    success_rate = float(np.mean(result.episode_success[-last_window:]))

    print(f"train_target={train_target_id}:{train_target}")
    print(f"target_count={len(TARGETS)}")
    print(f"fuzzy_rule_count={encoder.n_rules}")
    print(f"episodes={learning_config.episodes}")
    print(f"success_rate_last_{last_window}={success_rate:.3f}")
    print(f"baseline_successes={base_successes}/{len(TARGETS)}")
    print(f"learned_successes={learned_successes}/{len(TARGETS)}")
    print(f"mean_step_delta={float(np.mean(step_deltas)):.3f}")
    print(f"mean_torque_delta={float(np.mean(torque_deltas)):.12e}")
    print(f"figure={figure_path}")
    print(f"csv={csv_path}")
    print(f"markdown={markdown_path}")

    return 0 if learned_successes >= base_successes - 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
