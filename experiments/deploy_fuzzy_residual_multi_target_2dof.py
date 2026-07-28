"""Deploy a saved fuzzy residual Q-learning policy on multiple 2-DOF targets."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
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
from interactive import (
    LiveArm2DOFConfig,
    MultiTargetDeploymentRow,
    run_multi_target_deployment,
)
from rl import (
    FuzzyDynamicStateEncoder,
    FuzzyResidualQLearningConfig,
    FuzzyResidualSafetyConfig,
    load_fuzzy_residual_policy,
    save_fuzzy_residual_policy,
    train_fuzzy_residual_q_learning,
)


TARGETS = (
    ("D1_train", (1.10, 0.55)),
    ("D2_diag", (0.85, 0.85)),
    ("D3_low", (1.25, 0.25)),
    ("D4_high", (0.65, 1.05)),
    ("D5_far", (1.35, 0.45)),
)

METHOD_ORDER = ("fuzzy_base", "fuzzy_rl_safe_deployed")
METHOD_LABELS = {
    "fuzzy_base": "flou seul",
    "fuzzy_rl_safe_deployed": "flou + Q securise deploye",
}


def _make_env_config(target: tuple[float, float]) -> Arm2DOFDynamicEnvConfig:
    return Arm2DOFDynamicEnvConfig(
        target=target,
        dt=0.01,
        max_torque=(60.0, 35.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=700,
    )


def _make_live_config(
    target: tuple[float, float],
    safety_config: FuzzyResidualSafetyConfig,
) -> LiveArm2DOFConfig:
    return LiveArm2DOFConfig(
        target=target,
        dt=0.01,
        max_torque=(60.0, 35.0),
        max_joint_speed=(8.0, 8.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        safety_patience=safety_config.patience,
        safety_min_progress=safety_config.min_progress,
    )


def _write_csv(rows: list[MultiTargetDeploymentRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MultiTargetDeploymentRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _target_groups(
    rows: list[MultiTargetDeploymentRow],
) -> dict[str, dict[str, MultiTargetDeploymentRow]]:
    grouped: dict[str, dict[str, MultiTargetDeploymentRow]] = {}
    for row in rows:
        grouped.setdefault(row.target_id, {})[row.method] = row
    return grouped


def _write_markdown(rows: list[MultiTargetDeploymentRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _target_groups(rows)
    lines = [
        "# Step 15 - Deploiement multi-cibles 2 DDL",
        "",
        "La politique flou/RL est entrainee une fois, sauvegardee dans un artefact",
        "`npz`, rechargee, puis deployee sur une sequence de cibles sans remise a",
        "zero du robot entre deux cibles.",
        "",
        "| Cible | Methode | Succes | Pas | Distance finale | Vitesse finale | Couple moyen | Coupure residu |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for target_id, _ in TARGETS:
        methods = grouped[target_id]
        for method in METHOD_ORDER:
            row = methods[method]
            switch = "-" if row.residual_switch_step is None else str(row.residual_switch_step)
            lines.append(
                "| "
                f"{target_id} ({row.target_x:.2f}, {row.target_y:.2f}) | "
                f"{METHOD_LABELS[method]} | {int(row.done)} | {row.steps} | "
                f"{row.final_distance:.6f} | {row.final_speed:.6f} | "
                f"{row.mean_torque_norm:.6f} | {switch} |"
            )

    lines.extend(
        [
            "",
            "## Ecarts du controleur deploye par rapport au flou seul",
            "",
            "| Cible | Delta pas | Delta distance | Delta couple | Interpretation |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for target_id, _ in TARGETS:
        methods = grouped[target_id]
        base = methods["fuzzy_base"]
        deployed = methods["fuzzy_rl_safe_deployed"]
        delta_steps = deployed.steps - base.steps
        delta_distance = deployed.final_distance - base.final_distance
        delta_torque = deployed.mean_torque_norm - base.mean_torque_norm
        if not deployed.done:
            interpretation = "deploiement non convergent"
        elif deployed.residual_disabled:
            interpretation = "residu coupe par supervision"
        elif delta_steps < 0 and delta_torque > 0.0:
            interpretation = "plus rapide, effort plus eleve"
        elif delta_steps < 0:
            interpretation = "plus rapide"
        else:
            interpretation = "proche ou moins rapide que le flou seul"
        lines.append(
            "| "
            f"{target_id} | {delta_steps:+d} | {delta_distance:+.6f} | "
            f"{delta_torque:+.6f} | {interpretation} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(rows: list[MultiTargetDeploymentRow], path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _target_groups(rows)
    target_ids = [target_id for target_id, _ in TARGETS]
    x = np.arange(len(target_ids), dtype=float)
    width = 0.36
    offsets = {
        "fuzzy_base": -width / 2.0,
        "fuzzy_rl_safe_deployed": width / 2.0,
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_steps, ax_distance, ax_torque, ax_success = axes.ravel()

    for method in METHOD_ORDER:
        method_rows = [grouped[target_id][method] for target_id in target_ids]
        ax_steps.bar(
            x + offsets[method],
            [row.steps for row in method_rows],
            width,
            label=METHOD_LABELS[method],
        )
        ax_distance.bar(
            x + offsets[method],
            [row.final_distance for row in method_rows],
            width,
            label=METHOD_LABELS[method],
        )
        ax_torque.bar(
            x + offsets[method],
            [row.mean_torque_norm for row in method_rows],
            width,
            label=METHOD_LABELS[method],
        )
        ax_success.bar(
            x + offsets[method],
            [int(row.done) for row in method_rows],
            width,
            label=METHOD_LABELS[method],
        )

    ax_steps.set_title("Temps de convergence par cible")
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

    fig.suptitle("Deploiement multi-cibles du controleur flou/RL securise")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_manifest(
    rows: list[MultiTargetDeploymentRow],
    policy_path: Path,
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _target_groups(rows)
    summary = {}
    for method in METHOD_ORDER:
        method_rows = [grouped[target_id][method] for target_id, _ in TARGETS]
        summary[method] = {
            "successes": sum(int(row.done) for row in method_rows),
            "target_count": len(method_rows),
            "total_steps": sum(row.steps for row in method_rows),
            "mean_final_distance": float(
                np.mean([row.final_distance for row in method_rows])
            ),
            "mean_torque_norm": float(
                np.mean([row.mean_torque_norm for row in method_rows])
            ),
            "residual_switches": [
                {
                    "target_id": row.target_id,
                    "step": row.residual_switch_step,
                }
                for row in method_rows
                if row.residual_switch_step is not None
            ],
        }

    manifest = {
        "experiment": "step_15_multi_target_deployment_2dof",
        "policy_path": str(policy_path.relative_to(ROOT)),
        "target_sequence": [
            {"id": target_id, "x": target[0], "y": target[1]}
            for target_id, target in TARGETS
        ],
        "methods": METHOD_LABELS,
        "summary": summary,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    encoder = FuzzyDynamicStateEncoder(
        error_scale=(0.9, 1.2),
        velocity_scale=(6.0, 6.0),
    )
    learning_config = FuzzyResidualQLearningConfig(
        episodes=220,
        max_steps_per_episode=700,
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

    policy_path = ROOT / "results" / "policies" / "step_15_fuzzy_residual_policy_2dof.npz"
    save_fuzzy_residual_policy(
        policy_path,
        result,
        learning_config,
        safety_config,
        train_target_id,
        train_target,
        TARGETS,
    )
    package = load_fuzzy_residual_policy(policy_path)

    live_config = _make_live_config(train_target, package.safety_config)
    baseline_rows = run_multi_target_deployment(
        TARGETS,
        method="fuzzy_base",
        mode="fuzzy",
        config=live_config,
        learning_config=package.learning_config,
        max_steps_per_target=package.learning_config.max_steps_per_episode,
    )
    deployed_rows = run_multi_target_deployment(
        TARGETS,
        method="fuzzy_rl_safe_deployed",
        mode="fuzzy_rl_safe",
        config=live_config,
        q_value=package.q_value,
        encoder=package.encoder,
        learning_config=package.learning_config,
        max_steps_per_target=package.learning_config.max_steps_per_episode,
    )
    rows = baseline_rows + deployed_rows

    figure_path = ROOT / "results" / "figures" / "step_15_multi_target_deployment_2dof.png"
    csv_path = ROOT / "results" / "deployments" / "step_15_multi_target_deployment_2dof.csv"
    markdown_path = ROOT / "results" / "deployments" / "step_15_multi_target_deployment_2dof.md"
    manifest_path = ROOT / "results" / "deployments" / "step_15_multi_target_deployment_2dof.json"
    _plot_summary(rows, figure_path)
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)
    _write_manifest(rows, policy_path, manifest_path)

    grouped = _target_groups(rows)
    base_successes = sum(
        int(grouped[target_id]["fuzzy_base"].done) for target_id, _ in TARGETS
    )
    deployed_successes = sum(
        int(grouped[target_id]["fuzzy_rl_safe_deployed"].done)
        for target_id, _ in TARGETS
    )
    base_total_steps = sum(
        grouped[target_id]["fuzzy_base"].steps for target_id, _ in TARGETS
    )
    deployed_total_steps = sum(
        grouped[target_id]["fuzzy_rl_safe_deployed"].steps
        for target_id, _ in TARGETS
    )
    switches = [
        grouped[target_id]["fuzzy_rl_safe_deployed"].residual_switch_step
        for target_id, _ in TARGETS
        if grouped[target_id]["fuzzy_rl_safe_deployed"].residual_switch_step
        is not None
    ]

    print(f"policy={policy_path}")
    print(f"target_count={len(TARGETS)}")
    print(f"fuzzy_rule_count={package.encoder.n_rules}")
    print(f"episodes={package.learning_config.episodes}")
    print(f"baseline_successes={base_successes}/{len(TARGETS)}")
    print(f"deployed_successes={deployed_successes}/{len(TARGETS)}")
    print(f"baseline_total_steps={base_total_steps}")
    print(f"deployed_total_steps={deployed_total_steps}")
    print(f"step_delta={deployed_total_steps - base_total_steps:+d}")
    print(f"residual_switches={switches}")
    print(f"figure={figure_path}")
    print(f"csv={csv_path}")
    print(f"markdown={markdown_path}")
    print(f"manifest={manifest_path}")

    return 0 if deployed_successes == len(TARGETS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
