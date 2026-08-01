from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzy_drl_sim import (
    ExperimentConfig,
    FuzzyGuidedTrackingTask,
    RobotConfig,
    SimulationConfig,
    run_tracking_experiment,
)
from fuzzy_drl_sim.offline_env import OfflineArmEnv
from fuzzy_drl_sim.trajectory import SUPPORTED_TRAJECTORIES, make_trajectory
from robot import forward_kinematics_6dof


class CartesianTrajectory6DOFTests(unittest.TestCase):
    def test_cartesian_loop_sample_matches_forward_kinematics(self) -> None:
        trajectory = make_trajectory("cartesian_loop", np.zeros(6, dtype=float), duration=4.0)

        sample = trajectory.sample(1.5)

        self.assertEqual(sample.q.shape, (6,))
        self.assertEqual(sample.q_dot.shape, (6,))
        self.assertIsNotNone(sample.position)
        self.assertIsNotNone(sample.velocity)
        self.assertTrue(np.all(np.isfinite(sample.q)))
        reached = forward_kinematics_6dof(sample.q)
        np.testing.assert_allclose(reached, sample.position, atol=1e-7)

    def test_config_accepts_cartesian_trajectory_names(self) -> None:
        self.assertIn("cartesian_loop", SUPPORTED_TRAJECTORIES)
        self.assertIn("cartesian_point_to_point", SUPPORTED_TRAJECTORIES)

        ExperimentConfig(trajectory_name="cartesian_loop", dry_run=True).validate()
        ExperimentConfig(trajectory_name="cartesian_point_to_point", dry_run=True).validate()

    def test_offline_experiment_records_cartesian_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                robot=RobotConfig(),
                simulation=SimulationConfig(
                    dt=0.1,
                    duration=0.4,
                    output_dir=Path(tmpdir),
                    make_plots=False,
                ),
                trajectory_name="cartesian_point_to_point",
                controller_name="reference",
                dry_run=True,
            )

            result = run_tracking_experiment(config)

            self.assertIsNotNone(result.cartesian_metrics)
            assert result.cartesian_metrics is not None
            self.assertIn("cartesian_rmse", result.cartesian_metrics)
            with result.csv_path.open(newline="", encoding="utf-8") as handle:
                first_row = next(csv.DictReader(handle))
            self.assertIn("x_ref", first_row)
            self.assertIn("cartesian_error_norm", first_row)

    def test_residual_task_zero_action_uses_fuzzy_pid_expert(self) -> None:
        robot = RobotConfig()
        simulation = SimulationConfig(dt=0.05, duration=0.4, make_plots=False)
        backend = OfflineArmEnv(robot, simulation)
        task = FuzzyGuidedTrackingTask(
            backend,
            robot,
            simulation,
            trajectory_name="cartesian_loop",
            action_mode="residual",
        )

        try:
            task.reset()
            task.step(np.zeros(robot.dof, dtype=float))
            result = task.step(np.zeros(robot.dof, dtype=float))
        finally:
            task.close()

        self.assertEqual(result.info["constraint_violations"], 0)
        self.assertEqual(result.info["residual_norm"], 0.0)
        self.assertGreater(result.info["expert_correction_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
