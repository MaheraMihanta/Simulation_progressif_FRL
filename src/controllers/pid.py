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


def _magnitude_memberships(value: float) -> np.ndarray:
    normalized = float(np.clip(abs(value), 0.0, 1.0))
    return np.array(
        [
            max(0.0, 1.0 - 2.0 * normalized),
            max(0.0, 1.0 - abs(2.0 * normalized - 1.0)),
            max(0.0, 2.0 * normalized - 1.0),
        ],
        dtype=float,
    )


def _infer_gain_scale(
    error_level: float,
    derivative_level: float,
    table: np.ndarray,
) -> float:
    error_mu = _magnitude_memberships(error_level)
    derivative_mu = _magnitude_memberships(derivative_level)
    weights = np.outer(error_mu, derivative_mu)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(table[1, 1])
    return float(np.sum(weights * table) / total)


@dataclass
class FuzzyGainScheduledPIDController(PIDController):
    """PID controller whose gains are adapted by per-joint fuzzy scheduling.

    Each joint uses a compact 3x3 rule table over error magnitude and error
    derivative magnitude. This keeps the fuzzy part linear in the number of
    joints instead of building a Cartesian rule base over all joints.
    """

    error_scale: Gain = 1.0
    derivative_scale: Gain = 4.0
    kp_scale_table: np.ndarray | None = None
    ki_scale_table: np.ndarray | None = None
    kd_scale_table: np.ndarray | None = None
    last_kp_vector: np.ndarray = field(init=False)
    last_ki_vector: np.ndarray = field(init=False)
    last_kd_vector: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.error_scale_vector = _gain_vector(
            self.error_scale,
            self.size,
            "error_scale",
        )
        self.derivative_scale_vector = _gain_vector(
            self.derivative_scale,
            self.size,
            "derivative_scale",
        )
        if np.any(self.error_scale_vector <= 0.0):
            raise ValueError("error_scale values must be strictly positive.")
        if np.any(self.derivative_scale_vector <= 0.0):
            raise ValueError("derivative_scale values must be strictly positive.")

        self.kp_scale_matrix = self._scale_matrix(
            self.kp_scale_table,
            np.array(
                [
                    [0.75, 0.85, 1.00],
                    [1.05, 1.15, 1.30],
                    [1.45, 1.65, 1.85],
                ],
                dtype=float,
            ),
            "kp_scale_table",
        )
        self.ki_scale_matrix = self._scale_matrix(
            self.ki_scale_table,
            np.array(
                [
                    [1.10, 0.90, 0.65],
                    [0.75, 0.55, 0.35],
                    [0.35, 0.20, 0.10],
                ],
                dtype=float,
            ),
            "ki_scale_table",
        )
        self.kd_scale_matrix = self._scale_matrix(
            self.kd_scale_table,
            np.array(
                [
                    [0.75, 1.00, 1.25],
                    [0.90, 1.20, 1.45],
                    [1.05, 1.45, 1.70],
                ],
                dtype=float,
            ),
            "kd_scale_table",
        )
        self.last_kp_vector = self.kp_vector.copy()
        self.last_ki_vector = self.ki_vector.copy()
        self.last_kd_vector = self.kd_vector.copy()

    @staticmethod
    def _scale_matrix(
        value: np.ndarray | None,
        default: np.ndarray,
        name: str,
    ) -> np.ndarray:
        matrix = default if value is None else np.asarray(value, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3).")
        if np.any(matrix < 0.0):
            raise ValueError(f"{name} values must be non-negative.")
        return matrix.copy()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "kp_vector"):
            self.last_kp_vector = self.kp_vector.copy()
            self.last_ki_vector = self.ki_vector.copy()
            self.last_kd_vector = self.kd_vector.copy()

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

        error_level = np.clip(error / self.error_scale_vector, -1.0, 1.0)
        derivative_level = np.clip(
            derivative / self.derivative_scale_vector,
            -1.0,
            1.0,
        )
        kp_scale = np.array(
            [
                _infer_gain_scale(error_level[index], derivative_level[index], self.kp_scale_matrix)
                for index in range(self.size)
            ],
            dtype=float,
        )
        ki_scale = np.array(
            [
                _infer_gain_scale(error_level[index], derivative_level[index], self.ki_scale_matrix)
                for index in range(self.size)
            ],
            dtype=float,
        )
        kd_scale = np.array(
            [
                _infer_gain_scale(error_level[index], derivative_level[index], self.kd_scale_matrix)
                for index in range(self.size)
            ],
            dtype=float,
        )

        self.last_kp_vector = self.kp_vector * kp_scale
        self.last_ki_vector = self.ki_vector * ki_scale
        self.last_kd_vector = self.kd_vector * kd_scale

        output = (
            self.last_kp_vector * error
            + self.last_ki_vector * self.integral
            + self.last_kd_vector * derivative
        )
        if self.output_limits is not None:
            lower, upper = self.output_limits
            output = np.clip(output, lower, upper)
        return output
