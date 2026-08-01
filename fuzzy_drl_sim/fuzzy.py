from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FuzzyOutput:
    severity: float
    exploration_scale: float
    expert_blend: float
    kp_multiplier: float
    kd_multiplier: float
    action_limit_multiplier: float
    reward_error_weight: float
    reward_velocity_weight: float
    reward_effort_weight: float
    reward_smoothness_weight: float


def _ramp(x: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("high must be greater than low")
    return float(np.clip((x - low) / (high - low), 0.0, 1.0))


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class FuzzySupervisor:
    """Interpretable fuzzy layer for reward and controller adaptation.

    The current runner uses the gain and action-limit outputs for a fuzzy-guided
    PID baseline. The reward and exploration outputs are already logged so a DRL
    policy can later consume the same supervisor without changing the simulator.
    """

    def __init__(
        self,
        error_scale: float = 0.20,
        error_rate_scale: float = 0.45,
        effort_scale: float = 0.18,
    ) -> None:
        self.error_scale = error_scale
        self.error_rate_scale = error_rate_scale
        self.effort_scale = effort_scale

    def evaluate(
        self,
        error: np.ndarray,
        error_rate: np.ndarray,
        effort: np.ndarray,
    ) -> FuzzyOutput:
        dof_scale = max(np.sqrt(error.size), 1.0)
        error_level = np.linalg.norm(error) / (self.error_scale * dof_scale)
        rate_level = np.linalg.norm(error_rate) / (self.error_rate_scale * dof_scale)
        effort_level = np.linalg.norm(effort) / (self.effort_scale * dof_scale)

        error_high = _smoothstep(_ramp(error_level, 0.35, 1.15))
        rate_high = _smoothstep(_ramp(rate_level, 0.25, 1.05))
        effort_high = _smoothstep(_ramp(effort_level, 0.40, 1.20))

        tracking_pressure = np.clip(0.65 * error_high + 0.35 * rate_high, 0.0, 1.0)
        severity = float(np.clip(0.80 * tracking_pressure + 0.20 * effort_high, 0.0, 1.0))

        return FuzzyOutput(
            severity=severity,
            exploration_scale=float(0.03 + 0.22 * severity),
            expert_blend=float(np.clip(0.20 + 0.70 * tracking_pressure - 0.15 * effort_high, 0.10, 0.90)),
            kp_multiplier=float(np.clip(0.80 + 0.85 * tracking_pressure - 0.25 * effort_high, 0.55, 1.75)),
            kd_multiplier=float(np.clip(0.90 + 0.55 * rate_high, 0.75, 1.55)),
            action_limit_multiplier=float(np.clip(0.85 + 0.50 * tracking_pressure - 0.30 * effort_high, 0.55, 1.35)),
            reward_error_weight=float(1.0 + 2.0 * error_high),
            reward_velocity_weight=float(0.3 + 1.2 * rate_high),
            reward_effort_weight=float(0.02 + 0.20 * effort_high),
            reward_smoothness_weight=float(0.03 + 0.25 * effort_high),
        )
