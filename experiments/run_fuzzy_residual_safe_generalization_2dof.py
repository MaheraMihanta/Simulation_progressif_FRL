"""Evaluate safe fuzzy residual Q-learning across target positions."""

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
    FuzzyResidualSafetyConfig,
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
    residual_disabled: bool
    residual_switch_step: int | None


TARGETS = (
    ("T1_train", (1.10, 0.55)),
    ("T2_diag", (0.85, 0.85)),
    ("T3_low", (1.25, 0.25)),
    ("T4_high", (0.65, 1.05)),
    ("T5_far", (1.35, 0.45)),
)

METHOD_ORDER = ("fuzzy_base", "fuzzy_rl", "fuzzy_rl_safe")
METHOD_LABELS = {
    "fuzzy_base": "flou",
    "fuzzy_rl": "flou + Q",
    "fuzzy_rl_safe": "flou + Q securise",
}


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
        residual_disabled=bool(rollout.residual_disabled),
        residual_switch_step=rollout.residual_switch_step,
    )


def _target_groups(rows: list[EvaluationRow]) -> dict[str, dict[str, EvaluationRow]]:
    grouped: dict[str, dict[str, EvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.target_id, {})[row.method] = row
    return grouped


def _write_csv(rows: list[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvaluationRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _interpret(base: EvaluationRow, row: EvaluationRow) -> str:
    delta_steps = row.steps - base.steps
    delta_torque = row.mean_torque_norm - base.mean_torque_norm
    if not row.done:
        return "degradation: pas de convergence"
    if row.residual_disabled:
        if delta_steps < 0:
            return "residu coupe, convergence plus rapide que le flou seul"
        return "residu coupe par securite"
    if delta_steps < 0 and delta_torque > 0.0:
        return "plus rapide, effort plus eleve"
    if delta_steps < 0:
        return "plus rapide"
    return "proche ou moins rapide que le flou seul"


def _write_markdown(rows: list[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _target_groups(rows)
    lines = [
        "# Step 12 - Generalisation flou/RL securisee",
        "",
        "| Cible | Methode | Succes | Pas | Distance finale | Couple moyen | Coupure residu |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for target_id, methods in grouped.items():
        for method in METHOD_ORDER:
            row = methods[method]
            switch = "-" if row.residual_switch_step is None else str(row.residual_switch_step)
            lines.append(
                "| "
                f"{target_id} ({row.target_x:.2f}, {row.target_y:.2f}) | "
                f"{METHOD_LABELS[method]} | {int(row.done)} | {row.steps} | "
                f"{row.final_distance:.6f} | {row.mean_torque_norm:.6f} | "
                f"{switch} |"
            )

    lines.extend(
        [
            "",
            "## Ecarts par rapport au flou seul",
            "",
            "| Cible | Methode | Delta pas | Delta distance | Delta couple | Interpretation |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for target_id, methods in grouped.items():
        base = methods["fuzzy_base"]
        for method in ("fuzzy_rl", "fuzzy_rl_safe"):
            row = methods[method]
            lines.append(
                "| "
                f"{target_id} | {METHOD_LABELS[method]} | "
                f"{row.steps - base.steps:+d} | "
                f"{row.final_distance - base.final_distance:+.6f} | "
                f"{row.mean_torque_norm - base.mean_torque_norm:+.6f} | "
                f"{_interpret(base, row)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(rows: list[EvaluationRow], path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _target_groups(rows)
    target_ids = list(grouped)
    x = np.arange(len(target_ids), dtype=float)
    width = 0.25
    offsets = {
        "fuzzy_base": -width,
        "fuzzy_rl": 0.0,
        "fuzzy_rl_safe": width,
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_steps, ax_distance, ax_torque, ax_success = axes.ravel()

    for method in METHOD_ORDER:
        rows_for_method = [grouped[target][method] for target in target_ids]
        label = METHOD_LABELS[method]
        offset = offsets[method]
        ax_steps.bar(
            x + offset,
            [row.steps for row in rows_for_method],
            width,
            label=label,
        )
        ax_distance.bar(
            x + offset,
            [row.final_distance for row in rows_for_method],
            width,
            label=label,
        )
        ax_torque.bar(
            x + offset,
            [row.mean_torque_norm for row in rows_for_method],
            width,
            label=label,
        )
        ax_success.bar(
            x + offset,
            [int(row.done) for row in rows_for_method],
            width,
            label=label,
        )

    ax_steps.set_title("Temps de convergence")
    ax_steps.set_ylabel("pas")
    ax_steps.set_xticks(x, target_ids, rotation=20)
    ax_steps.grid(True, axis="y", alpha=0.3)
    ax_steps.legend()

    ax_distance.axhline(1e-2, linestyle=":", color="tab:green", linewidth=1, label="tolerance")
    ax_distance.set_title("Erreur finale")
    ax_distance.set_ylabel("distance")
    ax_distance.set_xticks(x, target_ids, rotation=20)
    ax_distance.grid(True, axis="y", alpha=0.3)
    ax_distance.legend()

    ax_torque.set_title("Effort moyen")
    ax_torque.set_ylabel("N.m")
    ax_torque.set_xticks(x, target_ids, rotation=20)
    ax_torque.grid(True, axis="y", alpha=0.3)
    ax_torque.legend()

    ax_success.set_title("Succes")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.set_ylabel("0/1")
    ax_success.set_xticks(x, target_ids, rotation=20)
    ax_success.grid(True, axis="y", alpha=0.3)
    ax_success.legend()

    fig.suptitle("Generalisation securisee du Q-learning residuel flou")
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
    safety_config = FuzzyResidualSafetyConfig(
        patience=100,
        min_progress=1e-4,
    )
    train_target_id, train_target = TARGETS[0]
    result = train_fuzzy_residual_q_learning(
        _make_env_config(train_target),
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
        raw_rollout = rollout_fuzzy_residual_q_policy(
            env_config,
            result.q_value,
            encoder,
            config=learning_config,
        )
        safe_rollout = rollout_fuzzy_residual_q_policy(
            env_config,
            result.q_value,
            encoder,
            config=learning_config,
            safety_config=safety_config,
        )
        rows.append(_metrics(target_id, target, "fuzzy_base", base_rollout))
        rows.append(_metrics(target_id, target, "fuzzy_rl", raw_rollout))
        rows.append(_metrics(target_id, target, "fuzzy_rl_safe", safe_rollout))

    figure_path = ROOT / "results" / "figures" / "step_12_fuzzy_residual_safe_generalization_2dof.png"
    csv_path = ROOT / "results" / "tables" / "step_12_fuzzy_residual_safe_generalization_2dof.csv"
    markdown_path = ROOT / "results" / "tables" / "step_12_fuzzy_residual_safe_generalization_2dof.md"
    _plot_summary(rows, figure_path)
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)

    grouped = _target_groups(rows)
    base_successes = sum(int(grouped[target]["fuzzy_base"].done) for target in grouped)
    raw_successes = sum(int(grouped[target]["fuzzy_rl"].done) for target in grouped)
    safe_successes = sum(int(grouped[target]["fuzzy_rl_safe"].done) for target in grouped)
    safe_step_deltas = [
        grouped[target]["fuzzy_rl_safe"].steps - grouped[target]["fuzzy_base"].steps
        for target in grouped
    ]
    safe_torque_deltas = [
        grouped[target]["fuzzy_rl_safe"].mean_torque_norm
        - grouped[target]["fuzzy_base"].mean_torque_norm
        for target in grouped
    ]
    switches = [
        grouped[target]["fuzzy_rl_safe"].residual_switch_step
        for target in grouped
        if grouped[target]["fuzzy_rl_safe"].residual_switch_step is not None
    ]

    print(f"train_target={train_target_id}:{train_target}")
    print(f"target_count={len(TARGETS)}")
    print(f"fuzzy_rule_count={encoder.n_rules}")
    print(f"episodes={learning_config.episodes}")
    print(f"safety_patience={safety_config.patience}")
    print(f"safety_min_progress={safety_config.min_progress:.12e}")
    print(f"baseline_successes={base_successes}/{len(TARGETS)}")
    print(f"raw_successes={raw_successes}/{len(TARGETS)}")
    print(f"safe_successes={safe_successes}/{len(TARGETS)}")
    print(f"safe_mean_step_delta={float(np.mean(safe_step_deltas)):.3f}")
    print(f"safe_mean_torque_delta={float(np.mean(safe_torque_deltas)):.12e}")
    print(f"safe_switches={switches}")
    print(f"figure={figure_path}")
    print(f"csv={csv_path}")
    print(f"markdown={markdown_path}")

    return 0 if safe_successes >= base_successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
