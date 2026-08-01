from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_REFERENCE_OFFSET: tuple[float, ...] = (0.08, -0.05, 0.06, -0.04, 0.04, -0.05)


@dataclass(frozen=True)
class ScenarioConfig:
    """Repeatable perturbation settings applied around one tracking experiment."""

    name: str = "nominal"
    measurement_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    observation_delay_steps: int = 0
    reference_step_time: float | None = None
    reference_step_ramp: float = 0.8
    reference_step_offset: tuple[float, ...] = DEFAULT_REFERENCE_OFFSET
    seed: int = 7

    def validate(self, dof: int, duration: float) -> None:
        if self.measurement_noise_std < 0.0:
            raise ValueError("measurement_noise_std must be non-negative")
        if self.velocity_noise_std < 0.0:
            raise ValueError("velocity_noise_std must be non-negative")
        if self.observation_delay_steps < 0:
            raise ValueError("observation_delay_steps must be non-negative")
        if self.reference_step_ramp <= 0.0:
            raise ValueError("reference_step_ramp must be positive")
        if self.reference_step_time is not None and not 0.0 <= self.reference_step_time <= duration:
            raise ValueError("reference_step_time must lie inside the experiment duration")
        if len(self.reference_step_offset) < dof:
            raise ValueError(f"reference_step_offset must contain at least {dof} values")

    def reference_offset(self, dof: int) -> np.ndarray:
        return np.asarray(self.reference_step_offset[:dof], dtype=float)


def scenario_from_name(name: str, duration: float) -> ScenarioConfig:
    if name == "nominal":
        return ScenarioConfig(name=name)
    if name == "sensor_noise":
        return ScenarioConfig(
            name=name,
            measurement_noise_std=0.003,
            velocity_noise_std=0.015,
        )
    if name == "observation_delay":
        return ScenarioConfig(
            name=name,
            observation_delay_steps=3,
        )
    if name == "trajectory_step":
        return ScenarioConfig(
            name=name,
            reference_step_time=duration * 0.55,
        )
    if name == "combined_uncertainty":
        return ScenarioConfig(
            name=name,
            measurement_noise_std=0.002,
            velocity_noise_std=0.012,
            observation_delay_steps=2,
            reference_step_time=duration * 0.55,
        )
    raise ValueError(f"Unsupported scenario: {name}")


def available_scenarios() -> tuple[str, ...]:
    return (
        "nominal",
        "sensor_noise",
        "observation_delay",
        "trajectory_step",
        "combined_uncertainty",
    )


def smooth_step(t: float, start: float | None, ramp: float) -> tuple[float, float, float]:
    if start is None or t <= start:
        return 0.0, 0.0, 0.0
    u = min((t - start) / ramp, 1.0)
    if u >= 1.0:
        return 1.0, 0.0, 0.0
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    blend_dot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / ramp
    blend_ddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / ramp**2
    return blend, blend_dot, blend_ddot
