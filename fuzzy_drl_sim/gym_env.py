from __future__ import annotations

from typing import Any

import numpy as np

from .config import RobotConfig, SimulationConfig
from .rl_task import ArmBackend, FuzzyGuidedTrackingTask


def make_fuzzy_guided_gym_env(
    backend: ArmBackend,
    robot_config: RobotConfig,
    simulation_config: SimulationConfig,
    trajectory_name: str = "multi_sine",
    action_mode: str = "residual",
    residual_scale: float = 0.35,
    initial_q: np.ndarray | None = None,
):
    """Wrap `FuzzyGuidedTrackingTask` as a Gymnasium environment.

    The import is intentionally local so the rest of the project remains usable
    without optional DRL dependencies.
    """

    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as exc:
        raise RuntimeError(
            "gymnasium is required for SAC/TD3 training. Install it with "
            "`pip install gymnasium stable-baselines3`."
        ) from exc

    class FuzzyGuidedGymEnv(gym.Env):
        metadata: dict[str, list[str]] = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.task = FuzzyGuidedTrackingTask(
                backend,
                robot_config,
                simulation_config,
                trajectory_name=trajectory_name,
                action_mode=action_mode,
                residual_scale=residual_scale,
                initial_q=initial_q,
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.task.action_size,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.task.observation_size,),
                dtype=np.float32,
            )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            observation = self.task.reset()
            return observation.astype(np.float32, copy=False), {}

        def step(
            self,
            action: np.ndarray,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            result = self.task.step(action)
            return (
                result.observation.astype(np.float32, copy=False),
                float(result.reward),
                bool(result.terminated),
                bool(result.truncated),
                dict(result.info),
            )

        def close(self) -> None:
            self.task.close()

    return FuzzyGuidedGymEnv()


__all__ = ["make_fuzzy_guided_gym_env"]
