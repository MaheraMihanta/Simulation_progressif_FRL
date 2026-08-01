"""Train factorized residual Q-learning under changing 6-DOF disturbances."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

matplotlib.use("Agg")

from envs import Arm6DOFDynamicEnvConfig
from rl import (
    PID_FACTORIZED_RESIDUAL_ACTION_NAMES_6DOF,
    PIDResidualQLearning6DOFConfig,
    PIDResidualSafety6DOFConfig,
    PIDResidualStateEncoder6DOF,
    rollout_pid_factorized_residual_q_policy_6dof,
    train_pid_factorized_residual_q_learning_6dof,
)
from visualization import plot_control_simulation_6dof


DISTURBANCE_PROFILES = (
    ("single_q1", (0.0, -4.0, 0.0, 0.0, 0.0, 0.0)),
    ("multi_q1_q2", (0.0, -4.0, -3.0, 0.0, 0.0, 0.0)),
    ("multi_q1_q3_q5", (0.0, -4.0, 0.0, -1.0, 0.0, -0.8)),
)
DISTURBANCE_SCHEDULE = tuple(profile[1] for profile in DISTURBANCE_PROFILES)
DISTURBANCE_SEGMENT_STEPS = 120


def _moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    effective_window = min(max(1, window), values.size)
    kernel = np.ones(effective_window, dtype=float) / effective_window
    averaged = np.convolve(values.astype(float), kernel, mode="valid")
    x = np.arange(effective_window - 1, values.size)
    return x, averaged


def _dominant_actions(action_labels: list[str], limit: int = 8) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for label in action_labels:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]


def _save_learning_plot(
    output_path: Path,
    returns: np.ndarray,
    success: np.ndarray,
    learned_rollout,
    baseline_rollout,
    tolerance: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_return, ax_success, ax_distance, ax_action = axes.ravel()

    x_return, mean_return = _moving_average(returns, window=8)
    ax_return.plot(x_return + 1, mean_return, label="return moyen")
    ax_return.grid(True, alpha=0.3)
    ax_return.set_title("Apprentissage factorise")
    ax_return.set_xlabel("episode")
    ax_return.set_ylabel("return")
    ax_return.legend()

    x_success, success_rate = _moving_average(success.astype(float), window=8)
    ax_success.plot(x_success + 1, success_rate, color="tab:green", label="succes")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.grid(True, alpha=0.3)
    ax_success.set_title("Taux de succes")
    ax_success.set_xlabel("episode")
    ax_success.set_ylabel("succes")
    ax_success.legend()

    ax_distance.plot(
        baseline_rollout.distance_history,
        linestyle="--",
        label="PID adapte",
    )
    ax_distance.plot(learned_rollout.distance_history, label="PID adapte + RL")
    ax_distance.axhline(tolerance, linestyle=":", color="tab:green", label="tolerance")
    for boundary in range(
        DISTURBANCE_SEGMENT_STEPS,
        len(learned_rollout.distance_history),
        DISTURBANCE_SEGMENT_STEPS,
    ):
        ax_distance.axvline(boundary, color="tab:gray", alpha=0.25, linewidth=1)
    ax_distance.grid(True, alpha=0.3)
    ax_distance.set_title("Distance cible sous perturbation changeante")
    ax_distance.set_xlabel("iteration")
    ax_distance.set_ylabel("distance")
    ax_distance.legend()

    if learned_rollout.local_action_indices:
        actions = np.asarray(learned_rollout.local_action_indices)
        for joint in range(actions.shape[1]):
            ax_action.step(
                np.arange(actions.shape[0]),
                actions[:, joint],
                where="post",
                label=f"q{joint}",
                alpha=0.75,
            )
    ax_action.grid(True, alpha=0.3)
    ax_action.set_title("Actions locales factorisees")
    ax_action.set_xlabel("iteration")
    ax_action.set_ylabel("0 base, 1 res+, 2 res-")
    ax_action.legend(ncol=2)

    fig.suptitle("Q-learning factorise sous perturbations changeantes - bras 6DDL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    success_rate: float,
    dominant_actions: list[tuple[str, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Q-learning factorise 6DDL sous perturbations changeantes",
        "",
        "Profil de perturbation par segments de "
        f"`{DISTURBANCE_SEGMENT_STEPS}` pas :",
        "",
    ]
    for index, (name, torque) in enumerate(DISTURBANCE_PROFILES, start=1):
        lines.append(f"- segment {index} / `{name}` : `{torque}`")
    lines.extend(
        [
            "",
            "| Profil | Controleur | Succes | Pas | Distance finale | Vitesse finale | Couple moyen |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['profile']} | {row['controller']} | {row['done']} | "
            f"{row['steps']} | {float(row['final_distance']):.4e} | "
            f"{float(row['final_speed']):.4e} | "
            f"{float(row['mean_torque_norm']):.4e} |"
        )
    lines.extend(
        [
            "",
            f"Taux de succes des 15 derniers episodes : `{success_rate:.3f}`.",
            "",
            "Actions apprises dominantes :",
            "",
        ]
    )
    for label, count in dominant_actions:
        lines.append(f"- `{label}` : `{count}` pas")
    lines.extend(
        [
            "",
            "Interpretation : cette experience retire le prior explicite du benchmark",
            "factorise. La politique doit choisir les signes locaux a partir de",
            "l'erreur articulaire et de la vitesse. Les perturbations changent",
            "d'un episode a l'autre pendant l'apprentissage, puis la politique est",
            "testee sur chaque profil et sur un profil temporel changeant.",
            "",
            "Resultat actuel : le probleme n'est pas encore resolu par cette",
            "variante tabulaire compacte. La politique reduit parfois la distance",
            "finale sur des perturbations multi-axes statiques, mais elle garde des",
            "vitesses finales trop elevees et ne produit aucun episode reussi dans",
            "la fenetre finale. Sur le profil temporel changeant, elle ne depasse",
            "pas le PID adapte. Cette etape sert donc de diagnostic avant de passer",
            "a une politique continue plus expressive.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rollout_for_profile(env_config, q_value, encoder, learning_config, desired_q, torque):
    rollout_config = replace(
        learning_config,
        external_torque=torque,
        external_torque_episode_schedule=None,
        external_torque_schedule=None,
    )
    return rollout_pid_factorized_residual_q_policy_6dof(
        env_config,
        q_value,
        encoder,
        config=rollout_config,
        desired_q=desired_q,
    )


def _rollout_for_changing_profile(env_config, q_value, encoder, learning_config, desired_q):
    rollout_config = replace(
        learning_config,
        external_torque=None,
        external_torque_episode_schedule=None,
        external_torque_schedule=DISTURBANCE_SCHEDULE,
        external_torque_segment_steps=DISTURBANCE_SEGMENT_STEPS,
    )
    return rollout_pid_factorized_residual_q_policy_6dof(
        env_config,
        q_value,
        encoder,
        config=rollout_config,
        desired_q=desired_q,
        safety_config=PIDResidualSafety6DOFConfig(patience=130, min_progress=5e-5),
    )


def _row_from_rollout(profile: str, controller: str, rollout) -> dict[str, object]:
    return {
        "profile": profile,
        "controller": controller,
        "done": rollout.done,
        "steps": len(rollout.action_labels),
        "final_distance": float(rollout.distance_history[-1]),
        "final_speed": float(rollout.speed_history[-1]),
        "mean_torque_norm": float(np.mean(np.linalg.norm(rollout.torque_history, axis=1))),
        "residual_disabled": rollout.residual_disabled,
        "residual_switch_step": rollout.residual_switch_step,
    }


def main() -> int:
    env_config = Arm6DOFDynamicEnvConfig(
        target=(1.25, 0.45, 0.60),
        dt=0.01,
        max_torque=(65.0, 95.0, 70.0, 45.0, 30.0, 25.0),
        target_tolerance=1e-2,
        speed_tolerance=8e-2,
        max_steps=420,
    )
    encoder = PIDResidualStateEncoder6DOF(
        joint_error_deadband=(0.035, 0.04, 0.04, 0.04, 0.04, 0.04),
        speed_bins=(0.12, 0.75),
    )
    learning_config = PIDResidualQLearning6DOFConfig(
        episodes=50,
        max_steps_per_episode=420,
        alpha=0.35,
        gamma=0.97,
        epsilon_start=0.95,
        epsilon_end=0.08,
        epsilon_decay=0.955,
        residual_acceleration_scale=(0.1, 4.0, 3.0, 1.0, 1.0, 0.8),
        residual_mode="torque",
        distance_weight=1.0,
        speed_weight=0.02,
        torque_weight=2.0e-4,
        residual_weight=0.005,
        progress_weight=10.0,
        goal_reward=25.0,
        external_torque_episode_schedule=DISTURBANCE_SCHEDULE,
        external_torque_segment_steps=DISTURBANCE_SEGMENT_STEPS,
        seed=53,
    )

    result = train_pid_factorized_residual_q_learning_6dof(
        env_config,
        encoder=encoder,
        config=learning_config,
    )
    zero_q_value = np.zeros_like(result.q_value)

    rows: list[dict[str, object]] = []
    baseline_success = 0
    learned_success = 0
    for profile_name, torque in DISTURBANCE_PROFILES:
        baseline = _rollout_for_profile(
            env_config,
            zero_q_value,
            encoder,
            learning_config,
            result.desired_q,
            torque,
        )
        learned = _rollout_for_profile(
            env_config,
            result.q_value,
            encoder,
            learning_config,
            result.desired_q,
            torque,
        )
        baseline_success += int(baseline.done)
        learned_success += int(learned.done)
        rows.append(_row_from_rollout(profile_name, "pid_adapte", baseline))
        rows.append(_row_from_rollout(profile_name, "pid_adapte_q_factorise", learned))

    learned_rollout = _rollout_for_changing_profile(
        env_config,
        result.q_value,
        encoder,
        learning_config,
        result.desired_q,
    )
    baseline_rollout = _rollout_for_changing_profile(
        env_config,
        zero_q_value,
        encoder,
        learning_config,
        result.desired_q,
    )
    rows.append(_row_from_rollout("changing_schedule", "pid_adapte", baseline_rollout))
    rows.append(
        _row_from_rollout(
            "changing_schedule",
            "pid_adapte_q_factorise",
            learned_rollout,
        )
    )

    figure_path = (
        ROOT
        / "results"
        / "figures"
        / "step_39_factorized_q_learning_changing_disturbance_6dof.png"
    )
    learning_path = (
        ROOT
        / "results"
        / "figures"
        / "step_39_factorized_q_learning_changing_disturbance_6dof_learning.png"
    )
    table_path = (
        ROOT
        / "results"
        / "tables"
        / "step_39_factorized_q_learning_changing_disturbance_6dof.csv"
    )
    markdown_path = (
        ROOT
        / "results"
        / "tables"
        / "step_39_factorized_q_learning_changing_disturbance_6dof.md"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_control_simulation_6dof(
        learned_rollout.q_history,
        learned_rollout.ee_history,
        learned_rollout.distance_history,
        learned_rollout.torque_history,
        env_config.target,
        link_lengths=env_config.arm_config.link_lengths,
        tolerance=env_config.target_tolerance,
        title="PID adapte + Q-learning factorise - perturbations changeantes 6DDL",
        action_ylabel="N.m",
    )
    fig.savefig(figure_path, dpi=150)
    _save_learning_plot(
        learning_path,
        result.episode_returns,
        result.episode_success,
        learned_rollout,
        baseline_rollout,
        env_config.target_tolerance,
    )

    last_window = min(15, learning_config.episodes)
    success_rate = float(np.mean(result.episode_success[-last_window:]))
    dominant_actions = _dominant_actions(learned_rollout.action_labels)
    _write_summary_csv(table_path, rows)
    _write_markdown(markdown_path, rows, success_rate, dominant_actions)

    print(f"state_count={encoder.n_states}")
    print(f"local_action_names={PID_FACTORIZED_RESIDUAL_ACTION_NAMES_6DOF}")
    print(f"episodes={learning_config.episodes}")
    print(f"epsilon_final={result.epsilon_history[-1]:.3f}")
    print(f"success_rate_last_{last_window}={success_rate:.3f}")
    print(f"baseline_static_success={baseline_success}/{len(DISTURBANCE_PROFILES)}")
    print(f"learned_static_success={learned_success}/{len(DISTURBANCE_PROFILES)}")
    print(f"baseline_done={baseline_rollout.done}")
    print(f"baseline_steps={len(baseline_rollout.action_labels)}")
    print(f"baseline_final_distance={baseline_rollout.distance_history[-1]:.12e}")
    print(f"baseline_final_speed={baseline_rollout.speed_history[-1]:.12e}")
    print(f"learned_done={learned_rollout.done}")
    print(f"learned_steps={len(learned_rollout.action_labels)}")
    print(f"learned_final_distance={learned_rollout.distance_history[-1]:.12e}")
    print(f"learned_final_speed={learned_rollout.speed_history[-1]:.12e}")
    print(f"learned_residual_disabled={learned_rollout.residual_disabled}")
    print(f"learned_residual_switch_step={learned_rollout.residual_switch_step}")
    print(f"dominant_learned_actions={dominant_actions}")
    print(f"table={table_path}")
    print(f"markdown={markdown_path}")
    print(f"figure={figure_path}")
    print(f"learning_figure={learning_path}")

    improved_static = learned_success > baseline_success
    improved_changing = learned_rollout.distance_history[-1] < baseline_rollout.distance_history[-1]
    print(f"improved_static={improved_static}")
    print(f"improved_changing={improved_changing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
