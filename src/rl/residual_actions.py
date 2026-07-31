"""Generic bounded residual action sets for joint-space RL controllers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


FACTORIZED_RESIDUAL_LOCAL_DIRECTIONS = np.array([0.0, 1.0, -1.0], dtype=float)
FACTORIZED_RESIDUAL_LOCAL_ACTION_NAMES = ("base", "res+", "res-")


def _positive_scale_array(
    scale: float | Sequence[float] | np.ndarray,
    size: int | None = None,
) -> np.ndarray:
    scale_array = np.asarray(scale, dtype=float)
    if scale_array.ndim == 0:
        if float(scale_array) <= 0.0:
            raise ValueError("scale must be strictly positive.")
        if size is None:
            scale_array = np.full(1, float(scale_array), dtype=float)
        else:
            scale_array = np.full(size, float(scale_array), dtype=float)
    if scale_array.ndim != 1 or scale_array.size == 0:
        raise ValueError("scale must be a scalar or a non-empty 1D vector.")
    if size is not None and scale_array.size != size:
        raise ValueError(f"scale must contain exactly {size} values.")
    if np.any(scale_array <= 0.0):
        raise ValueError("scale values must be strictly positive.")
    return scale_array


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

    scale_array = _positive_scale_array(scale)
    return axis_aligned_residual_action_directions(scale_array.size) * scale_array


def factorized_residual_action_directions(size: int) -> np.ndarray:
    """Return local residual directions for each joint without Cartesian product."""

    if size <= 0:
        raise ValueError("size must be strictly positive.")
    return np.tile(FACTORIZED_RESIDUAL_LOCAL_DIRECTIONS, (size, 1))


def factorized_residual_action_names(size: int) -> tuple[tuple[str, ...], ...]:
    """Return readable per-joint names for factorized residual actions."""

    if size <= 0:
        raise ValueError("size must be strictly positive.")
    return tuple(
        (
            f"q{joint}_base",
            f"q{joint}_res+",
            f"q{joint}_res-",
        )
        for joint in range(size)
    )


def factorized_residual_action_vector(
    local_action_indices: Sequence[int] | np.ndarray,
    scale: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Decode one local action per joint into a bounded residual vector."""

    indices = np.asarray(local_action_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("local_action_indices must be a non-empty 1D vector.")
    if np.any(indices < 0) or np.any(indices >= FACTORIZED_RESIDUAL_LOCAL_DIRECTIONS.size):
        raise ValueError("local_action_indices values must be in [0, 2].")
    scale_array = _positive_scale_array(scale, size=int(indices.size))
    return FACTORIZED_RESIDUAL_LOCAL_DIRECTIONS[indices] * scale_array


def factorized_residual_action_label(
    local_action_indices: Sequence[int] | np.ndarray,
) -> str:
    """Return a compact label for a factorized residual action vector."""

    indices = np.asarray(local_action_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("local_action_indices must be a non-empty 1D vector.")
    if np.any(indices < 0) or np.any(indices >= FACTORIZED_RESIDUAL_LOCAL_DIRECTIONS.size):
        raise ValueError("local_action_indices values must be in [0, 2].")

    parts = [
        f"q{joint}_{FACTORIZED_RESIDUAL_LOCAL_ACTION_NAMES[action]}"
        for joint, action in enumerate(indices)
        if action != 0
    ]
    return "base" if not parts else ",".join(parts)


__all__ = [
    "FACTORIZED_RESIDUAL_LOCAL_ACTION_NAMES",
    "FACTORIZED_RESIDUAL_LOCAL_DIRECTIONS",
    "axis_aligned_residual_action_directions",
    "axis_aligned_residual_action_names",
    "axis_aligned_residual_actions",
    "factorized_residual_action_directions",
    "factorized_residual_action_label",
    "factorized_residual_action_names",
    "factorized_residual_action_vector",
]
