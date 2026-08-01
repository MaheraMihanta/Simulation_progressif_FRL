from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class TrackingMetrics:
    joint_rmse: float
    joint_max_abs_error: float
    final_error_norm: float
    control_energy: float
    action_smoothness: float
    constraint_violations: int
    settling_time: float | None
    mean_error_norm: float
    error_norm_std: float
    correction_sign_flip_ratio: float
    high_frequency_error_index: float

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def compute_tracking_metrics(
    time: np.ndarray,
    q: np.ndarray,
    q_ref: np.ndarray,
    correction: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    tolerance: float = 0.035,
) -> TrackingMetrics:
    error = q_ref - q
    error_norm = np.linalg.norm(error, axis=1)
    joint_rmse = float(np.sqrt(np.mean(error**2)))
    joint_max_abs_error = float(np.max(np.abs(error)))
    final_error_norm = float(error_norm[-1])
    mean_error_norm = float(np.mean(error_norm))
    error_norm_std = float(np.std(error_norm))
    dt = float(np.mean(np.diff(time))) if time.size > 1 else 0.0
    control_energy = float(np.sum(correction**2) * dt)
    action_smoothness = float(np.sum(np.diff(correction, axis=0) ** 2))
    correction_sign_flip_ratio = _sign_flip_ratio(correction)
    high_frequency_error_index = _high_frequency_index(error)
    lower_violation = q < joint_lower.reshape(1, -1)
    upper_violation = q > joint_upper.reshape(1, -1)
    constraint_violations = int(np.count_nonzero(lower_violation | upper_violation))
    settling_time = _settling_time(time, error_norm, tolerance)
    return TrackingMetrics(
        joint_rmse=joint_rmse,
        joint_max_abs_error=joint_max_abs_error,
        final_error_norm=final_error_norm,
        control_energy=control_energy,
        action_smoothness=action_smoothness,
        constraint_violations=constraint_violations,
        settling_time=settling_time,
        mean_error_norm=mean_error_norm,
        error_norm_std=error_norm_std,
        correction_sign_flip_ratio=correction_sign_flip_ratio,
        high_frequency_error_index=high_frequency_error_index,
    )


def _settling_time(time: np.ndarray, error_norm: np.ndarray, tolerance: float) -> float | None:
    for idx, value in enumerate(error_norm):
        if value <= tolerance and np.all(error_norm[idx:] <= tolerance):
            return float(time[idx])
    return None


def _sign_flip_ratio(correction: np.ndarray) -> float:
    if correction.shape[0] < 2:
        return 0.0
    signs = np.sign(correction)
    active = (signs[1:] != 0.0) & (signs[:-1] != 0.0)
    if not np.any(active):
        return 0.0
    flips = (signs[1:] != signs[:-1]) & active
    return float(np.count_nonzero(flips) / np.count_nonzero(active))


def _high_frequency_index(error: np.ndarray) -> float:
    if error.shape[0] < 3:
        return 0.0
    second_difference = np.diff(error, n=2, axis=0)
    return float(np.sqrt(np.mean(second_difference**2)))
