from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

from fuzzy_drl_sim import ExperimentConfig, RobotConfig, SimulationConfig, available_scenarios
from fuzzy_drl_sim.coppelia_env import CoppeliaConnectionError
from fuzzy_drl_sim.experiment import run_tracking_experiment
from fuzzy_drl_sim.scenarios import scenario_from_name


DEFAULT_CONTROLLERS = ("reference", "pid", "fuzzy-pid")
DEFAULT_SCENARIOS = ("nominal", "sensor_noise", "observation_delay", "trajectory_step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a repeatable tracking validation campaign.")
    parser.add_argument("--controllers", nargs="+", choices=DEFAULT_CONTROLLERS, default=list(DEFAULT_CONTROLLERS))
    parser.add_argument("--scenarios", nargs="+", choices=available_scenarios(), default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--trajectory", choices=["multi_sine", "point_to_point"], default="multi_sine")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results_campaign"))
    parser.add_argument("--dry-run", action="store_true", help="Use the offline plant instead of CoppeliaSim.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-run PNG plot generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scenario_name in args.scenarios:
        scenario = replace(scenario_from_name(scenario_name, args.duration), seed=args.seed)
        for controller_name in args.controllers:
            config = ExperimentConfig(
                robot=RobotConfig(),
                simulation=SimulationConfig(
                    dt=args.dt,
                    duration=args.duration,
                    output_dir=args.output_dir,
                    make_plots=not args.no_plots,
                ),
                trajectory_name=args.trajectory,
                controller_name=controller_name,
                scenario=scenario,
                dry_run=args.dry_run,
            )
            print(f"Running {scenario_name} / {controller_name} ...")
            try:
                result = run_tracking_experiment(config)
            except CoppeliaConnectionError as exc:
                print(exc, file=sys.stderr)
                return 2
            metrics = result.metrics.to_dict()
            row = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "backend": "offline" if args.dry_run else "coppeliasim",
                "scenario": scenario_name,
                "controller": controller_name,
                "trajectory": args.trajectory,
                "run_dir": str(result.run_dir),
                **metrics,
            }
            rows.append(row)
            print(
                "  rmse={joint_rmse:.6f}, max={joint_max_abs_error:.6f}, "
                "flip={correction_sign_flip_ratio:.6f}, hf={high_frequency_error_index:.6f}".format(**metrics)
            )

    _write_outputs(args.output_dir, rows, args)
    return 0


def _write_outputs(output_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{stamp}_campaign_summary.csv"
    json_path = output_dir / f"{stamp}_campaign_summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "controllers": args.controllers,
            "scenarios": args.scenarios,
            "trajectory": args.trajectory,
            "duration": args.duration,
            "dt": args.dt,
            "seed": args.seed,
            "dry_run": args.dry_run,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Campaign CSV  : {csv_path}")
    print(f"Campaign JSON : {json_path}")


if __name__ == "__main__":
    raise SystemExit(main())
