from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from fuzzy_drl_sim import RobotConfig, SimulationConfig
from fuzzy_drl_sim.coppelia_env import CoppeliaArmEnv, CoppeliaConnectionError
from fuzzy_drl_sim.fuzzy import FuzzySupervisor
from fuzzy_drl_sim.logging_utils import save_csv, save_json, save_plots
from fuzzy_drl_sim.metrics import compute_tracking_metrics
from fuzzy_drl_sim.offline_env import OfflineArmEnv
from fuzzy_drl_sim.rl_task import FuzzyGuidedTrackingTask


PolicyFn = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the FRL/DRL tracking task with simple non-trained policies.",
    )
    parser.add_argument("--policy", choices=["zero", "random", "proportional", "fuzzy_expert"], default="fuzzy_expert")
    parser.add_argument("--trajectory", choices=["multi_sine", "point_to_point"], default="multi_sine")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results_frl_eval"))
    parser.add_argument("--dry-run", action="store_true", help="Use the offline plant instead of CoppeliaSim.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot = RobotConfig()
    simulation = SimulationConfig(
        dt=args.dt,
        duration=args.duration,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )
    backend = OfflineArmEnv(robot, simulation) if args.dry_run else CoppeliaArmEnv(robot, simulation)
    task = FuzzyGuidedTrackingTask(backend, robot, simulation, trajectory_name=args.trajectory)
    policy = _make_policy(args.policy, robot, args.seed)

    try:
        result = _run_task(task, policy, robot, simulation, args)
    except CoppeliaConnectionError as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        task.close()

    print(f"Run directory : {result['run_dir']}")
    print(f"CSV log       : {result['csv_path']}")
    print(f"Summary       : {result['summary_path']}")
    for plot_path in result["plot_paths"]:
        print(f"Plot          : {plot_path}")
    print("Metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")
    return 0


def _run_task(
    task: FuzzyGuidedTrackingTask,
    policy: PolicyFn,
    robot: RobotConfig,
    simulation: SimulationConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    run_dir = _make_run_dir(simulation.output_dir, args.policy, args.dry_run)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    observation = task.reset()
    total_reward = 0.0
    step_count = 0
    truncated = False

    while not truncated:
        action = policy(observation)
        step = task.step(action)
        observation = step.observation
        total_reward += step.reward
        step_count += 1
        rows.append(
            _make_row(
                task.t,
                observation,
                action,
                step.reward,
                robot.dof,
                args.policy,
                step.info,
                np.asarray(robot.max_position_correction, dtype=float),
            )
        )
        truncated = step.truncated

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
        "backend": "offline" if args.dry_run else "coppeliasim",
        "policy": args.policy,
        "trajectory": args.trajectory,
        "dt": simulation.dt,
        "duration": simulation.duration,
        "total_reward": total_reward,
        "mean_reward": total_reward / max(step_count, 1),
        "metrics": metrics.to_dict(),
    }
    save_json(summary_path, summary)

    plot_paths: list[Path] = []
    if simulation.make_plots:
        error_norm = np.linalg.norm(arrays["q_ref"] - arrays["q"], axis=1)
        plot_paths = save_plots(run_dir, arrays["time"], arrays["q"], arrays["q_ref"], error_norm)

    return {
        "run_dir": run_dir,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "plot_paths": plot_paths,
        "metrics": metrics.to_dict(),
    }


def _make_policy(name: str, robot: RobotConfig, seed: int) -> PolicyFn:
    rng = np.random.default_rng(seed)
    correction_limit = np.asarray(robot.max_position_correction, dtype=float)
    supervisor = FuzzySupervisor()
    previous_correction = np.zeros(robot.dof, dtype=float)

    def zero(observation: np.ndarray) -> np.ndarray:
        return np.zeros(robot.dof, dtype=float)

    def random(observation: np.ndarray) -> np.ndarray:
        return np.clip(rng.normal(0.0, 0.25, size=robot.dof), -1.0, 1.0)

    def proportional(observation: np.ndarray) -> np.ndarray:
        error, _error_rate = _extract_error(observation, robot.dof)
        return np.clip(0.35 * error / correction_limit, -1.0, 1.0)

    def fuzzy_expert(observation: np.ndarray) -> np.ndarray:
        nonlocal previous_correction
        error, error_rate = _extract_error(observation, robot.dof)
        fuzzy = supervisor.evaluate(error, error_rate, previous_correction)
        raw_correction = 0.35 * fuzzy.kp_multiplier * error + 0.02 * fuzzy.kd_multiplier * error_rate
        limit = correction_limit * fuzzy.action_limit_multiplier
        correction = np.clip(raw_correction, -limit, limit)
        previous_correction = correction.copy()
        return np.clip(correction / correction_limit, -1.0, 1.0)

    policies = {
        "zero": zero,
        "random": random,
        "proportional": proportional,
        "fuzzy_expert": fuzzy_expert,
    }
    return policies[name]


def _extract_error(observation: np.ndarray, dof: int) -> tuple[np.ndarray, np.ndarray]:
    error = observation[4 * dof : 5 * dof]
    error_rate = observation[5 * dof : 6 * dof]
    return error, error_rate


def _make_row(
    t: float,
    observation: np.ndarray,
    action: np.ndarray,
    reward: float,
    dof: int,
    policy_name: str,
    info: dict[str, float | int | bool],
    correction_limit: np.ndarray,
) -> dict[str, float | str]:
    q = observation[0:dof]
    q_dot = observation[dof : 2 * dof]
    q_ref = observation[2 * dof : 3 * dof]
    q_ref_dot = observation[3 * dof : 4 * dof]
    error = q_ref - q
    row: dict[str, float | str] = {"time": float(t), "policy": policy_name, "reward": float(reward)}
    for idx, value in enumerate(q, start=1):
        row[f"q{idx}"] = float(value)
    for idx, value in enumerate(q_dot, start=1):
        row[f"qdot{idx}"] = float(value)
    for idx, value in enumerate(q_ref, start=1):
        row[f"q_ref{idx}"] = float(value)
    for idx, value in enumerate(q_ref_dot, start=1):
        row[f"qdot_ref{idx}"] = float(value)
    for idx, value in enumerate(error, start=1):
        row[f"error{idx}"] = float(value)
    for idx, value in enumerate(action * correction_limit, start=1):
        row[f"correction{idx}"] = float(value)
    for idx, value in enumerate(action, start=1):
        row[f"action{idx}"] = float(value)
    for key, value in info.items():
        if isinstance(value, (int, float, bool)):
            row[key] = float(value)
    return row


def _rows_to_arrays(rows: list[dict[str, float | str]], dof: int) -> dict[str, np.ndarray]:
    time = np.array([row["time"] for row in rows], dtype=float)
    q = np.array([[row[f"q{idx}"] for idx in range(1, dof + 1)] for row in rows], dtype=float)
    q_ref = np.array([[row[f"q_ref{idx}"] for idx in range(1, dof + 1)] for row in rows], dtype=float)
    correction = np.array(
        [[row[f"correction{idx}"] for idx in range(1, dof + 1)] for row in rows],
        dtype=float,
    )
    return {"time": time, "q": q, "q_ref": q_ref, "correction": correction}


def _make_run_dir(base_dir: Path, policy_name: str, dry_run: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = "offline" if dry_run else "coppelia"
    return base_dir / f"{stamp}_{backend}_{policy_name}"


if __name__ == "__main__":
    raise SystemExit(main())
