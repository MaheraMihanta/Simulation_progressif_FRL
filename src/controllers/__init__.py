"""Controllers for the robotic-arm simulations."""

from .fuzzy import FuzzyAccelerationController, FuzzyVelocityController
from .pid import PIDController

__all__ = [
    "FuzzyAccelerationController",
    "FuzzyVelocityController",
    "PIDController",
]
