from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig, RobotConfig, SimulationConfig
from .controllers import ControllerOutput, FuzzyGuidedPIDController, PIDController, ReferenceController
from .coppelia_env import CoppeliaArmEnv
from .logging_utils import save_csv, save_json, save_plots
from .metrics import TrackingMetrics, compute_tracking_metrics
from .offline_env import OfflineArmEnv
from .scenarios import ScenarioConfig, smooth_step
from .state import ArmState
from .trajectory import make_trajectory


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    csv_path: Path
    summary_path: Path
    plot_paths: list[Path]
    metrics: TrackingMetrics


def run_tracking_experiment(config: ExperimentConfig) -> ExperimentResult:
    config.validate()
    robot = config.robot
    sim_cfg = config.simulation
    scenario = config.scenario
    rng = np.random.default_rng(scenario.seed)
    run_dir = _make_run_dir(sim_cfg.output_dir, config.controller_name, config.dry_run, scenario.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = OfflineArmEnv(robot, sim_cfg) if config.dry_run else CoppeliaArmEnv(robot, sim_cfg)
    rows: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    try:
        env.start()
        initial_state = env.read_state()
        trajectory = make_trajectory(config.trajectory_name, initial_state.q, sim_cfg.duration)
        controller = _build_controller(config.controller_name, robot)
        controller.reset()
        state_history: list[ArmState] = []

        steps = int(np.floor(sim_cfg.duration / sim_cfg.dt))
        for index in range(steps + 1):
            t = index * sim_cfg.dt
            state = env.read_state()
            state_history.append(state)
            max_history = scenario.observation_delay_steps + 1
            if len(state_history) > max_history:
                state_history.pop(0)
            delayed_state = state_history[0]
            observed_state = _apply_observation_effects(delayed_state, scenario, rng, robot.dof)
            reference = _apply_reference_effects(trajectory.sample(t), scenario, robot.dof, t)
            output = controller.compute(
                observed_state.q,
                observed_state.q_dot,
                reference.q,
                reference.q_dot,
                sim_cfg.dt,
            )
            env.step(output.target_position)
            rows.append(
                _make_row(
                    t,
                    state.q,
                    state.q_dot,
                    observed_state.q,
                    observed_state.q_dot,
                    reference.q,
                    reference.q_dot,
                    output,
                )
            )
    finally:
        env.stop()

    arrays = _rows_to_arrays(rows, robot.dof)
    metrics = compute_tracking_metrics(
        arrays["time"],
        arrays["q"],
        arrays["q_ref"],
        arrays["correction"],
        np.asarray(robot.joint_lower_limits, dtype=float),
        np.asarray(robot.joint_upper_limits, dtype=float),
    )

    csv_path = run_dir / "tracking_log.csv"
    summary_path = run_dir / "summary.json"
    save_csv(csv_path, rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": scenario.name,
        "scenario_config": {
            "name": scenario.name,
            "measurement_noise_std": scenario.measurement_noise_std,
            "velocity_noise_std": scenario.velocity_noise_std,
            "observation_delay_steps": scenario.observation_delay_steps,
            "reference_step_time": scenario.reference_step_time,
            "reference_step_ramp": scenario.reference_step_ramp,
            "seed": scenario.seed,
        },
        "backend": "offline" if config.dry_run else "coppeliasim",
        "controller": config.controller_name,
        "trajectory": config.trajectory_name,
        "dt": sim_cfg.dt,
        "duration": sim_cfg.duration,
        "robot": robot.name,
        "metrics": metrics.to_dict(),
    }
    save_json(summary_path, summary)

    if sim_cfg.make_plots:
        error_norm = np.linalg.norm(arrays["q_ref"] - arrays["q"], axis=1)
        plot_paths = save_plots(run_dir, arrays["time"], arrays["q"], arrays["q_ref"], error_norm)

    return ExperimentResult(
        run_dir=run_dir,
        csv_path=csv_path,
        summary_path=summary_path,
        plot_paths=plot_paths,
        metrics=metrics,
    )


def _build_controller(name: str, robot: RobotConfig) -> PIDController:
    common = {
        "dof": robot.dof,
        "correction_limit": robot.max_position_correction,
        "joint_lower": robot.joint_lower_limits,
        "joint_upper": robot.joint_upper_limits,
    }
    if name == "pid":
        return PIDController(**common)
    if name == "fuzzy-pid":
        return FuzzyGuidedPIDController(**common)
    if name == "reference":
        return ReferenceController(**common)
    raise ValueError(f"Unsupported controller: {name}")


def _make_run_dir(base_dir: Path, controller_name: str, dry_run: bool, scenario_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = "offline" if dry_run else "coppelia"
    return base_dir / f"{stamp}_{backend}_{scenario_name}_{controller_name}"


def _apply_observation_effects(
    state: ArmState,
    scenario: ScenarioConfig,
    rng: np.random.Generator,
    dof: int,
) -> ArmState:
    q = state.q.copy()
    q_dot = state.q_dot.copy()
    if scenario.measurement_noise_std > 0.0:
        q += rng.normal(0.0, scenario.measurement_noise_std, size=dof)
    if scenario.velocity_noise_std > 0.0:
        q_dot += rng.normal(0.0, scenario.velocity_noise_std, size=dof)
    return ArmState(q=q, q_dot=q_dot, tip_position=state.tip_position)


def _apply_reference_effects(reference: Any, scenario: ScenarioConfig, dof: int, t: float) -> Any:
    blend, blend_dot, blend_ddot = smooth_step(t, scenario.reference_step_time, scenario.reference_step_ramp)
    if blend == 0.0 and blend_dot == 0.0 and blend_ddot == 0.0:
        return reference
    offset = scenario.reference_offset(dof)
    return type(reference)(
        q=reference.q + blend * offset,
        q_dot=reference.q_dot + blend_dot * offset,
        q_ddot=reference.q_ddot + blend_ddot * offset,
    )


def _make_row(
    t: float,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_observed: np.ndarray,
    q_dot_observed: np.ndarray,
    q_ref: np.ndarray,
    q_ref_dot: np.ndarray,
    output: ControllerOutput,
) -> dict[str, Any]:
    row: dict[str, Any] = {"time": t}
    for idx, value in enumerate(q, start=1):
        row[f"q{idx}"] = float(value)
    for idx, value in enumerate(q_dot, start=1):
        row[f"qdot{idx}"] = float(value)
    for idx, value in enumerate(q_observed, start=1):
        row[f"q_obs{idx}"] = float(value)
    for idx, value in enumerate(q_dot_observed, start=1):
        row[f"qdot_obs{idx}"] = float(value)
    for idx, value in enumerate(q_ref, start=1):
        row[f"q_ref{idx}"] = float(value)
    for idx, value in enumerate(q_ref_dot, start=1):
        row[f"qdot_ref{idx}"] = float(value)
    error = q_ref - q
    for idx, value in enumerate(error, start=1):
        row[f"error{idx}"] = float(value)
    for idx, value in enumerate(output.correction, start=1):
        row[f"correction{idx}"] = float(value)
    for idx, value in enumerate(output.target_position, start=1):
        row[f"target{idx}"] = float(value)
    for key, value in output.info.items():
        if isinstance(value, (int, float, str)):
            row[key] = value
    return row


def _rows_to_arrays(rows: list[dict[str, Any]], dof: int) -> dict[str, np.ndarray]:
    time = np.array([row["time"] for row in rows], dtype=float)
    q = np.array([[row[f"q{idx}"] for idx in range(1, dof + 1)] for row in rows], dtype=float)
    q_ref = np.array([[row[f"q_ref{idx}"] for idx in range(1, dof + 1)] for row in rows], dtype=float)
    correction = np.array(
        [[row[f"correction{idx}"] for idx in range(1, dof + 1)] for row in rows],
        dtype=float,
    )
    return {"time": time, "q": q, "q_ref": q_ref, "correction": correction}
