"""Compare joint and Cartesian tracking on 6-DOF task trajectories."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzy_drl_sim import (
    ExperimentConfig,
    RobotConfig,
    SimulationConfig,
    available_scenarios,
    run_tracking_experiment,
    scenario_from_name,
)
from fuzzy_drl_sim.coppelia_env import CoppeliaConnectionError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 6-DOF joint tracking and Cartesian end-effector tracking.",
    )
    parser.add_argument("--mode", choices=["offline", "coppelia", "both"], default="offline")
    parser.add_argument(
        "--controller",
        choices=["reference", "pid", "fuzzy-pid"],
        default="fuzzy-pid",
    )
    parser.add_argument(
        "--trajectory",
        choices=["cartesian_loop", "cartesian_point_to_point"],
        default="cartesian_loop",
    )
    parser.add_argument("--scenario", choices=available_scenarios(), default="nominal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cartesian_6dof"))
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    return parser.parse_args()


def _run_case(args: argparse.Namespace, dry_run: bool):
    scenario = replace(scenario_from_name(args.scenario, args.duration), seed=args.seed)
    config = ExperimentConfig(
        robot=RobotConfig(),
        simulation=SimulationConfig(
            dt=args.dt,
            duration=args.duration,
            output_dir=args.output_dir,
            make_plots=not args.no_plots,
        ),
        trajectory_name=args.trajectory,
        controller_name=args.controller,
        scenario=scenario,
        dry_run=dry_run,
    )
    return run_tracking_experiment(config)


def _print_result(label: str, result) -> None:
    metrics = result.metrics.to_dict()
    cartesian = result.cartesian_metrics or {}
    print(f"[{label}]")
    print(f"Run directory       : {result.run_dir}")
    print(f"CSV log             : {result.csv_path}")
    print(f"Summary             : {result.summary_path}")
    print(f"joint_rmse          : {metrics['joint_rmse']:.6e}")
    print(f"joint_final_error   : {metrics['final_error_norm']:.6e}")
    if cartesian:
        print(f"cartesian_rmse      : {cartesian['cartesian_rmse']:.6e}")
        print(f"cartesian_final_err : {cartesian['cartesian_final_error']:.6e}")
    for plot_path in result.plot_paths:
        print(f"Plot                : {plot_path}")


def main() -> int:
    args = parse_args()
    dry_run_values = {
        "offline": [True],
        "coppelia": [False],
        "both": [True, False],
    }[args.mode]

    for dry_run in dry_run_values:
        label = "offline-python" if dry_run else "coppeliasim"
        try:
            result = _run_case(args, dry_run=dry_run)
        except CoppeliaConnectionError as exc:
            print(exc, file=sys.stderr)
            print(
                "Astuce: ouvrez bras_manipulateur_niryoOne.ttt et l'add-on ZeroMQ, "
                "ou lancez avec --mode offline pour verifier la trajectoire cartesienne.",
                file=sys.stderr,
            )
            return 2
        _print_result(label, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
