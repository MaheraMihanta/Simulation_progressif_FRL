"""Dependency-free fuzzy velocity controller for the 2-DOF arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


Scale = float | Sequence[float] | np.ndarray

_TERMS = ("negative", "zero", "positive")
_RULE_OUTPUTS = {
    ("negative", "negative"): -1.0,
    ("negative", "zero"): -0.8,
    ("negative", "positive"): -0.4,
    ("zero", "negative"): -0.35,
    ("zero", "zero"): 0.0,
    ("zero", "positive"): 0.35,
    ("positive", "negative"): 0.4,
    ("positive", "zero"): 0.8,
    ("positive", "positive"): 1.0,
}


def _scale_vector(value: Scale, size: int, name: str) -> np.ndarray:
    scale = np.asarray(value, dtype=float)
    if scale.ndim == 0:
        vector = np.full(size, float(scale), dtype=float)
    elif scale.shape == (size,):
        vector = scale.copy()
    else:
        raise ValueError(f"{name} must be a scalar or a vector of size {size}.")
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return vector


def _memberships(value: float) -> dict[str, float]:
    value = float(np.clip(value, -1.0, 1.0))
    return {
        "negative": max(0.0, -value),
        "zero": max(0.0, 1.0 - abs(value)),
        "positive": max(0.0, value),
    }


def _infer_output(error_norm: float, derivative_norm: float) -> float:
    error_mu = _memberships(error_norm)
    derivative_mu = _memberships(derivative_norm)
    weighted_sum = 0.0
    total_weight = 0.0

    for error_term in _TERMS:
        for derivative_term in _TERMS:
            weight = error_mu[error_term] * derivative_mu[derivative_term]
            if weight <= 0.0:
                continue
            weighted_sum += weight * _RULE_OUTPUTS[(error_term, derivative_term)]
            total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, -1.0, 1.0))


@dataclass
class FuzzyVelocityController:
    """Mamdani-like fuzzy controller returning joint velocity commands."""

    error_scale: Scale = 0.6
    derivative_scale: Scale = 4.0
    output_scale: Scale = 2.0
    size: int = 2
    output_limits: tuple[float, float] | None = None
    previous_error: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be strictly positive.")
        self.error_scale_vector = _scale_vector(
            self.error_scale,
            self.size,
            "error_scale",
        )
        self.derivative_scale_vector = _scale_vector(
            self.derivative_scale,
            self.size,
            "derivative_scale",
        )
        self.output_scale_vector = _scale_vector(
            self.output_scale,
            self.size,
            "output_scale",
        )
        if self.output_limits is not None:
            lower, upper = self.output_limits
            if lower >= upper:
                raise ValueError("output_limits must be ordered as (min, max).")
        self.reset()

    def reset(self) -> None:
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
        if self.previous_error is None:
            derivative = np.zeros(self.size, dtype=float)
        else:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error.copy()

        error_norm = np.clip(error / self.error_scale_vector, -1.0, 1.0)
        derivative_norm = np.clip(
            derivative / self.derivative_scale_vector,
            -1.0,
            1.0,
        )

        output_norm = np.array(
            [
                _infer_output(error_norm[index], derivative_norm[index])
                for index in range(self.size)
            ],
            dtype=float,
        )
        output = output_norm * self.output_scale_vector
        if self.output_limits is not None:
            lower, upper = self.output_limits
            output = np.clip(output, lower, upper)
        return output
