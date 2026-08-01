from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmState:
    """Joint state returned by simulation backends."""

    q: np.ndarray
    q_dot: np.ndarray
    tip_position: np.ndarray | None = None
