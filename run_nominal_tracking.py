from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

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
        description="Run a nominal 6-DOF trajectory-tracking experiment.",
    )
    parser.add_argument("--controller", choices=["reference", "pid", "fuzzy-pid"], default="fuzzy-pid")
    parser.add_argument("--trajectory", choices=["multi_sine", "point_to_point"], default="multi_sine")
    parser.add_argument("--scenario", choices=available_scenarios(), default="nominal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true", help="Use the offline plant instead of CoppeliaSim.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        dry_run=args.dry_run,
    )
    try:
        result = run_tracking_experiment(config)
    except CoppeliaConnectionError as exc:
        print(exc, file=sys.stderr)
        print(
            "Astuce: pour verifier le pipeline sans simulateur, lancez "
            "`python run_nominal_tracking.py --dry-run`.",
            file=sys.stderr,
        )
        return 2

    print(f"Run directory : {result.run_dir}")
    print(f"CSV log       : {result.csv_path}")
    print(f"Summary       : {result.summary_path}")
    for plot_path in result.plot_paths:
        print(f"Plot          : {plot_path}")
    print("Metrics:")
    for key, value in result.metrics.to_dict().items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
