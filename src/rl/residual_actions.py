"""Generic bounded residual action sets for joint-space RL controllers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def axis_aligned_residual_action_directions(size: int) -> np.ndarray:
    """Return zero plus positive/negative unit actions for each joint."""

    if size <= 0:
        raise ValueError("size must be strictly positive.")
    return np.vstack(
        [
            np.zeros(size, dtype=float),
            np.eye(size, dtype=float),
            -np.eye(size, dtype=float),
        ]
    )


def axis_aligned_residual_action_names(size: int) -> tuple[str, ...]:
    """Return readable names for axis-aligned residual actions."""

    if size <= 0:
        raise ValueError("size must be strictly positive.")
    positive = tuple(f"q{index}_res+" for index in range(size))
    negative = tuple(f"q{index}_res-" for index in range(size))
    return ("base", *positive, *negative)


def axis_aligned_residual_actions(
    scale: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return bounded residual actions scaled per joint."""

    scale_array = np.asarray(scale, dtype=float)
    if scale_array.ndim == 0:
        if float(scale_array) <= 0.0:
            raise ValueError("scale must be strictly positive.")
        scale_array = np.full(1, float(scale_array), dtype=float)
    if scale_array.ndim != 1 or scale_array.size == 0:
        raise ValueError("scale must be a scalar or a non-empty 1D vector.")
    if np.any(scale_array <= 0.0):
        raise ValueError("scale values must be strictly positive.")
    return axis_aligned_residual_action_directions(scale_array.size) * scale_array


__all__ = [
    "axis_aligned_residual_action_directions",
    "axis_aligned_residual_action_names",
    "axis_aligned_residual_actions",
]
