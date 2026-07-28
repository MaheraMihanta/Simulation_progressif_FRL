"""Run deployed controllers on a sequence of Cartesian targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rl import FuzzyDynamicStateEncoder, FuzzyResidualQLearningConfig

from .live_arm_2dof import ControllerMode, LiveArm2DOFConfig, LiveArm2DOFSimulation


TargetSpec = tuple[str, tuple[float, float]]


@dataclass(frozen=True)
class MultiTargetDeploymentRow:
    """Metrics collected for one target during a deployed target sequence."""

    target_id: str
    target_x: float
    target_y: float
    method: str
    done: bool
    steps: int
    final_distance: float
    final_speed: float
    mean_torque_norm: float
    total_reward: float
    residual_disabled: bool
    residual_switch_step: int | None
    start_step: int
    end_step: int


def _normalize_targets(targets: Sequence[TargetSpec]) -> tuple[TargetSpec, ...]:
    if not targets:
        raise ValueError("at least one deployment target is required.")
    normalized: list[TargetSpec] = []
    for target_id, target in targets:
        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (2,):
            raise ValueError("each deployment target must contain exactly two values.")
        normalized.append((str(target_id), (float(target_array[0]), float(target_array[1]))))
    return tuple(normalized)


def _is_target_reached(
    summary: dict[str, object],
    config: LiveArm2DOFConfig,
) -> bool:
    return (
        float(summary["distance"]) <= config.target_tolerance
        and float(summary["speed"]) <= config.speed_tolerance
    )


def run_multi_target_deployment(
    targets: Sequence[TargetSpec],
    *,
    method: str,
    mode: ControllerMode,
    config: LiveArm2DOFConfig | None = None,
    q_value: np.ndarray | None = None,
    encoder: FuzzyDynamicStateEncoder | None = None,
    learning_config: FuzzyResidualQLearningConfig | None = None,
    max_steps_per_target: int = 550,
) -> list[MultiTargetDeploymentRow]:
    """Deploy one controller through a target sequence without resetting the arm."""

    if max_steps_per_target <= 0:
        raise ValueError("max_steps_per_target must be strictly positive.")

    normalized_targets = _normalize_targets(targets)
    live_config = config or LiveArm2DOFConfig(target=normalized_targets[0][1])
    sim = LiveArm2DOFSimulation(
        config=live_config,
        mode=mode,
        q_value=q_value,
        encoder=encoder,
        learning_config=learning_config,
    )

    rows: list[MultiTargetDeploymentRow] = []
    for target_id, target in normalized_targets:
        sim.set_target(target)
        start_step = int(sim.summary()["step"])
        torque_norms: list[float] = []
        rewards: list[float] = []

        summary = sim.summary()
        done = _is_target_reached(summary, live_config)
        for _ in range(max_steps_per_target):
            if done:
                break
            summary = sim.step()
            torque_norms.append(float(summary["torque_norm"]))
            rewards.append(float(summary["reward"]))
            done = _is_target_reached(summary, live_config)

        end_step = int(summary["step"])
        switch_step = summary["residual_switch_step"]
        local_switch_step = None if switch_step is None else int(switch_step) - start_step
        rows.append(
            MultiTargetDeploymentRow(
                target_id=target_id,
                target_x=float(target[0]),
                target_y=float(target[1]),
                method=method,
                done=bool(done),
                steps=end_step - start_step,
                final_distance=float(summary["distance"]),
                final_speed=float(summary["speed"]),
                mean_torque_norm=(
                    float(np.mean(torque_norms)) if torque_norms else 0.0
                ),
                total_reward=float(np.sum(rewards)) if rewards else 0.0,
                residual_disabled=bool(summary["residual_disabled"]),
                residual_switch_step=local_switch_step,
                start_step=start_step,
                end_step=end_step,
            )
        )

    return rows


__all__ = [
    "MultiTargetDeploymentRow",
    "TargetSpec",
    "run_multi_target_deployment",
]
