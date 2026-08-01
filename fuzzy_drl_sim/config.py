from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

from .scenarios import ScenarioConfig


DEFAULT_JOINT_PATHS: tuple[str, ...] = (
    "/NiryoOne/Joint",
    "/NiryoOne/Joint/Link/Joint",
    "/NiryoOne/Joint/Link/Joint/Link/Joint",
    "/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint",
    "/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint",
    "/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint",
)


@dataclass(frozen=True)
class RobotConfig:
    """Robot-specific settings used by both CoppeliaSim and offline runs."""

    name: str = "NiryoOne"
    root_path: str = "/NiryoOne"
    joint_paths: tuple[str, ...] = DEFAULT_JOINT_PATHS
    tip_path: str | None = None
    joint_lower_limits: tuple[float, ...] = (-math.pi,) * 6
    joint_upper_limits: tuple[float, ...] = (math.pi,) * 6
    max_joint_velocity: tuple[float, ...] = (1.2,) * 6
    max_position_correction: tuple[float, ...] = (0.12,) * 6

    @property
    def dof(self) -> int:
        return len(self.joint_paths)

    def validate(self) -> None:
        expected = self.dof
        vector_fields = {
            "joint_lower_limits": self.joint_lower_limits,
            "joint_upper_limits": self.joint_upper_limits,
            "max_joint_velocity": self.max_joint_velocity,
            "max_position_correction": self.max_position_correction,
        }
        for name, values in vector_fields.items():
            if len(values) != expected:
                raise ValueError(f"{name} must contain {expected} values")


@dataclass(frozen=True)
class SimulationConfig:
    """Timing and output configuration for one tracking experiment."""

    dt: float = 0.05
    duration: float = 12.0
    settling_steps: int = 10
    output_dir: Path = Path("results")
    synchronous_stepping: bool = True
    make_plots: bool = True

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.settling_steps < 0:
            raise ValueError("settling_steps must be zero or positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Full experiment configuration."""

    robot: RobotConfig = RobotConfig()
    simulation: SimulationConfig = SimulationConfig()
    trajectory_name: str = "multi_sine"
    controller_name: str = "fuzzy-pid"
    scenario: ScenarioConfig = ScenarioConfig()
    dry_run: bool = False

    def validate(self) -> None:
        self.robot.validate()
        self.simulation.validate()
        self.scenario.validate(self.robot.dof, self.simulation.duration)
        if self.controller_name not in {"reference", "pid", "fuzzy-pid"}:
            raise ValueError("controller_name must be 'reference', 'pid' or 'fuzzy-pid'")
        if self.trajectory_name not in {"multi_sine", "point_to_point"}:
            raise ValueError("trajectory_name must be 'multi_sine' or 'point_to_point'")
