"""Vector PID controller used as the first classical baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


Gain = float | Sequence[float] | np.ndarray


def _gain_vector(value: Gain, size: int, name: str) -> np.ndarray:
    gain = np.asarray(value, dtype=float)
    if gain.ndim == 0:
        return np.full(size, float(gain), dtype=float)
    if gain.shape != (size,):
        raise ValueError(f"{name} must be a scalar or a vector of size {size}.")
    return gain.copy()


@dataclass
class PIDController:
    """Simple vector PID controller with optional symmetric output limits."""

    kp: Gain
    ki: Gain = 0.0
    kd: Gain = 0.0
    size: int = 2
    output_limits: tuple[float, float] | None = None
    integral: np.ndarray = field(init=False)
    previous_error: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be strictly positive.")
        self.kp_vector = _gain_vector(self.kp, self.size, "kp")
        self.ki_vector = _gain_vector(self.ki, self.size, "ki")
        self.kd_vector = _gain_vector(self.kd, self.size, "kd")
        if self.output_limits is not None:
            lower, upper = self.output_limits
            if lower >= upper:
                raise ValueError("output_limits must be ordered as (min, max).")
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros(self.size, dtype=float)
        self.previous_error = None

    def compute(
        self,
        setpoint: Sequence[float] | np.ndarray,
        measurement: Sequence[float] | np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")

        setpoint_array = np.asarray(setpoint, dtype=float)
        measurement_array = np.asarray(measurement, dtype=float)
        if setpoint_array.shape != (self.size,):
            raise ValueError("setpoint has an invalid shape.")
        if measurement_array.shape != (self.size,):
            raise ValueError("measurement has an invalid shape.")

        error = setpoint_array - measurement_array
        self.integral += error * dt

        if self.previous_error is None:
            derivative = np.zeros(self.size, dtype=float)
        else:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error.copy()

        output = (
            self.kp_vector * error
            + self.ki_vector * self.integral
            + self.kd_vector * derivative
        )
        if self.output_limits is not None:
            lower, upper = self.output_limits
            output = np.clip(output, lower, upper)
        return output

