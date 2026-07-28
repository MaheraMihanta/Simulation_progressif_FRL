"""Persistence helpers for deployable fuzzy residual Q-learning policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dynamic_residual_q_learning import residual_acceleration_actions
from .fuzzy_residual_q_learning import (
    FuzzyDynamicStateEncoder,
    FuzzyResidualQLearningConfig,
    FuzzyResidualQLearningResult,
    FuzzyResidualSafetyConfig,
)


POLICY_TYPE = "fuzzy_residual_q_learning_2dof"
FORMAT_VERSION = 1
TargetSpec = tuple[str, tuple[float, float]]


@dataclass(frozen=True)
class FuzzyResidualPolicyPackage:
    """Loaded fuzzy residual policy and the configuration needed to deploy it."""

    q_value: np.ndarray
    rule_policy: np.ndarray
    desired_q: np.ndarray
    metadata: dict[str, Any]
    encoder: FuzzyDynamicStateEncoder
    learning_config: FuzzyResidualQLearningConfig
    safety_config: FuzzyResidualSafetyConfig


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _target_entry(target_id: str, target: Sequence[float]) -> dict[str, object]:
    target_array = np.asarray(target, dtype=float)
    if target_array.shape != (2,):
        raise ValueError("each deployment target must contain exactly two values.")
    return {
        "id": str(target_id),
        "x": float(target_array[0]),
        "y": float(target_array[1]),
    }


def _target_entries(targets: Sequence[TargetSpec]) -> list[dict[str, object]]:
    return [_target_entry(target_id, target) for target_id, target in targets]


def _encoder_metadata(encoder: FuzzyDynamicStateEncoder) -> dict[str, object]:
    return {
        "error_scale": encoder._error_scale_vector.tolist(),
        "velocity_scale": encoder._velocity_scale_vector.tolist(),
        "min_activation": float(encoder.min_activation),
    }


def save_fuzzy_residual_policy(
    path: str | Path,
    result: FuzzyResidualQLearningResult,
    learning_config: FuzzyResidualQLearningConfig,
    safety_config: FuzzyResidualSafetyConfig,
    train_target_id: str,
    train_target: Sequence[float],
    deployment_targets: Sequence[TargetSpec],
) -> Path:
    """Save a trained fuzzy residual Q table as a deployable policy package."""

    policy_path = Path(path)
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "policy_type": POLICY_TYPE,
        "format_version": FORMAT_VERSION,
        "train_target": _target_entry(train_target_id, train_target),
        "deployment_targets": _target_entries(deployment_targets),
        "encoder": _encoder_metadata(result.encoder),
        "learning_config": _json_ready(asdict(learning_config)),
        "safety_config": _json_ready(asdict(safety_config)),
        "fuzzy_rule_count": int(result.encoder.n_rules),
        "action_count": int(result.q_value.shape[1]),
        "success_rate_last_60": float(
            np.mean(result.episode_success[-min(60, len(result.episode_success)) :])
        ),
    }

    np.savez_compressed(
        policy_path,
        q_value=np.asarray(result.q_value, dtype=float),
        rule_policy=np.asarray(result.rule_policy, dtype=int),
        desired_q=np.asarray(result.desired_q, dtype=float),
        episode_returns=np.asarray(result.episode_returns, dtype=float),
        episode_lengths=np.asarray(result.episode_lengths, dtype=int),
        episode_success=np.asarray(result.episode_success, dtype=bool),
        epsilon_history=np.asarray(result.epsilon_history, dtype=float),
        metadata_json=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
    )
    return policy_path


def load_fuzzy_residual_policy(path: str | Path) -> FuzzyResidualPolicyPackage:
    """Load and validate a fuzzy residual policy package."""

    policy_path = Path(path)
    with np.load(policy_path) as archive:
        q_value = np.asarray(archive["q_value"], dtype=float)
        rule_policy = np.asarray(archive["rule_policy"], dtype=int)
        desired_q = np.asarray(archive["desired_q"], dtype=float)
        metadata = json.loads(str(archive["metadata_json"].item()))

    if metadata.get("policy_type") != POLICY_TYPE:
        raise ValueError("unsupported fuzzy residual policy type.")
    if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported fuzzy residual policy format version.")

    encoder = FuzzyDynamicStateEncoder(**metadata["encoder"])
    learning_config = FuzzyResidualQLearningConfig(**metadata["learning_config"])
    safety_config = FuzzyResidualSafetyConfig(**metadata["safety_config"])
    action_count = len(
        residual_acceleration_actions(learning_config.residual_acceleration_scale)
    )

    expected_q_shape = (encoder.n_rules, action_count)
    if q_value.shape != expected_q_shape:
        raise ValueError(f"q_value must have shape {expected_q_shape}.")
    if rule_policy.shape != (encoder.n_rules,):
        raise ValueError("rule_policy has an invalid shape.")
    if desired_q.shape != (2,):
        raise ValueError("desired_q must contain exactly two values.")

    return FuzzyResidualPolicyPackage(
        q_value=q_value,
        rule_policy=rule_policy,
        desired_q=desired_q,
        metadata=metadata,
        encoder=encoder,
        learning_config=learning_config,
        safety_config=safety_config,
    )


__all__ = [
    "FORMAT_VERSION",
    "POLICY_TYPE",
    "FuzzyResidualPolicyPackage",
    "TargetSpec",
    "load_fuzzy_residual_policy",
    "save_fuzzy_residual_policy",
]
