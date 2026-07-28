"""Dynamics for a spatial 3-DOF yaw-plus-planar robotic arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .kinematics_3dof import Arm3DOFConfig, ArrayLike3


@dataclass(frozen=True)
class Arm3DOFDynamicsConfig:
    """Physical constants for the 3-DOF manipulator.

    The shoulder and elbow use the rigid planar 2R dynamics in the vertical
    plane. The yaw axis adds a posture-dependent inertia around the vertical
    axis, which is enough for the current simulation and controller comparison
    stage without pretending to be a full CAD-grade rigid-body model.
    """

    arm_config: Arm3DOFConfig = field(default_factory=Arm3DOFConfig)
    link_masses: tuple[float, float] = (1.0, 0.8)
    center_of_mass_ratios: tuple[float, float] = (0.5, 0.5)
    link_inertias: tuple[float, float] | None = None
    base_yaw_inertia: float = 0.05
    gravity: float = 9.81
    viscous_friction: tuple[float, float, float] = (0.05, 0.08, 0.06)

    def __post_init__(self) -> None:
        masses = np.asarray(self.link_masses, dtype=float)
        if masses.shape != (2,) or np.any(masses <= 0.0):
            raise ValueError("link_masses must contain two positive values.")

        ratios = np.asarray(self.center_of_mass_ratios, dtype=float)
        if ratios.shape != (2,) or np.any(ratios <= 0.0) or np.any(ratios > 1.0):
            raise ValueError(
                "center_of_mass_ratios must contain two values in ]0, 1]."
            )

        if self.link_inertias is not None:
            inertias = np.asarray(self.link_inertias, dtype=float)
            if inertias.shape != (2,) or np.any(inertias <= 0.0):
                raise ValueError("link_inertias must contain two positive values.")

        if self.base_yaw_inertia <= 0.0:
            raise ValueError("base_yaw_inertia must be strictly positive.")
        if self.gravity < 0.0:
            raise ValueError("gravity must be positive or zero.")

        friction = np.asarray(self.viscous_friction, dtype=float)
        if friction.shape != (3,) or np.any(friction < 0.0):
            raise ValueError("viscous_friction must contain three non-negative values.")


def _as_vector3(values: ArrayLike3, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return vector


def _link_parameters(
    config: Arm3DOFDynamicsConfig,
) -> tuple[float, float, float, float, float, float, float, float]:
    l1, l2 = config.arm_config.link_lengths
    m1, m2 = config.link_masses
    r1_ratio, r2_ratio = config.center_of_mass_ratios
    r1 = r1_ratio * l1
    r2 = r2_ratio * l2

    if config.link_inertias is None:
        i1 = m1 * l1 * l1 / 12.0
        i2 = m2 * l2 * l2 / 12.0
    else:
        i1, i2 = config.link_inertias

    return l1, l2, m1, m2, r1, r2, i1, i2


def _planar_mass_terms(
    q_shoulder_elbow: Sequence[float] | np.ndarray,
    config: Arm3DOFDynamicsConfig,
) -> np.ndarray:
    _, q2 = np.asarray(q_shoulder_elbow, dtype=float)
    l1, _, m1, m2, r1, r2, i1, i2 = _link_parameters(config)
    cos_q2 = float(np.cos(q2))

    m11 = (
        i1
        + i2
        + m1 * r1 * r1
        + m2 * (l1 * l1 + r2 * r2 + 2.0 * l1 * r2 * cos_q2)
    )
    m12 = i2 + m2 * (r2 * r2 + l1 * r2 * cos_q2)
    m22 = i2 + m2 * r2 * r2
    return np.array([[m11, m12], [m12, m22]], dtype=float)


def yaw_inertia_and_gradient(
    q: ArrayLike3,
    config: Arm3DOFDynamicsConfig | None = None,
) -> tuple[float, np.ndarray]:
    """Return base yaw inertia and its gradient over q."""

    cfg = config or Arm3DOFDynamicsConfig()
    _, shoulder, elbow = _as_vector3(q, "q")
    l1, _, m1, m2, r1, r2, _, _ = _link_parameters(cfg)
    q12 = shoulder + elbow

    rho1 = r1 * np.cos(shoulder)
    rho2 = l1 * np.cos(shoulder) + r2 * np.cos(q12)
    drho1_dq1 = -r1 * np.sin(shoulder)
    drho2_dq1 = -l1 * np.sin(shoulder) - r2 * np.sin(q12)
    drho2_dq2 = -r2 * np.sin(q12)

    inertia = cfg.base_yaw_inertia + m1 * rho1 * rho1 + m2 * rho2 * rho2
    gradient = np.array(
        [
            0.0,
            2.0 * m1 * rho1 * drho1_dq1 + 2.0 * m2 * rho2 * drho2_dq1,
            2.0 * m2 * rho2 * drho2_dq2,
        ],
        dtype=float,
    )
    return float(inertia), gradient


def mass_matrix_3dof(
    q: ArrayLike3,
    config: Arm3DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return the joint-space inertia matrix M(q)."""

    cfg = config or Arm3DOFDynamicsConfig()
    q_array = _as_vector3(q, "q")
    yaw_inertia, _ = yaw_inertia_and_gradient(q_array, cfg)
    matrix = np.zeros((3, 3), dtype=float)
    matrix[0, 0] = yaw_inertia
    matrix[1:, 1:] = _planar_mass_terms(q_array[1:], cfg)
    return matrix


def coriolis_centrifugal_torque_3dof(
    q: ArrayLike3,
    q_dot: ArrayLike3,
    config: Arm3DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return Coriolis and centrifugal terms C(q, q_dot) q_dot."""

    cfg = config or Arm3DOFDynamicsConfig()
    q_array = _as_vector3(q, "q")
    velocity = _as_vector3(q_dot, "q_dot")
    _, shoulder, elbow = q_array
    yaw_rate, shoulder_rate, elbow_rate = velocity
    l1, _, _, m2, _, r2, _, _ = _link_parameters(cfg)

    _, yaw_gradient = yaw_inertia_and_gradient(q_array, cfg)
    yaw_inertia_dot = float(np.dot(yaw_gradient, velocity))

    coupling = m2 * l1 * r2 * float(np.sin(elbow))
    planar = np.array(
        [
            -coupling * (2.0 * shoulder_rate * elbow_rate + elbow_rate * elbow_rate),
            coupling * shoulder_rate * shoulder_rate,
        ],
        dtype=float,
    )

    return np.array(
        [
            yaw_inertia_dot * yaw_rate,
            planar[0] - 0.5 * yaw_gradient[1] * yaw_rate * yaw_rate,
            planar[1] - 0.5 * yaw_gradient[2] * yaw_rate * yaw_rate,
        ],
        dtype=float,
    )


def gravity_torque_3dof(
    q: ArrayLike3,
    config: Arm3DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return gravity compensation torques for the current posture."""

    cfg = config or Arm3DOFDynamicsConfig()
    _, shoulder, elbow = _as_vector3(q, "q")
    l1, _, m1, m2, r1, r2, _, _ = _link_parameters(cfg)
    g = cfg.gravity

    tau_shoulder = (m1 * r1 + m2 * l1) * g * np.cos(shoulder)
    tau_shoulder += m2 * r2 * g * np.cos(shoulder + elbow)
    tau_elbow = m2 * r2 * g * np.cos(shoulder + elbow)
    return np.array([0.0, tau_shoulder, tau_elbow], dtype=float)


def viscous_friction_torque_3dof(
    q_dot: ArrayLike3,
    config: Arm3DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return viscous friction torques opposing joint motion."""

    cfg = config or Arm3DOFDynamicsConfig()
    return np.asarray(cfg.viscous_friction, dtype=float) * _as_vector3(
        q_dot,
        "q_dot",
    )


def inverse_dynamics_torque_3dof(
    q: ArrayLike3,
    q_dot: ArrayLike3,
    q_ddot: ArrayLike3,
    config: Arm3DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return tau needed to produce q_ddot from the current state."""

    cfg = config or Arm3DOFDynamicsConfig()
    acceleration = _as_vector3(q_ddot, "q_ddot")
    return (
        mass_matrix_3dof(q, cfg) @ acceleration
        + coriolis_centrifugal_torque_3dof(q, q_dot, cfg)
        + gravity_torque_3dof(q, cfg)
        + viscous_friction_torque_3dof(q_dot, cfg)
    )


def joint_acceleration_3dof(
    q: ArrayLike3,
    q_dot: ArrayLike3,
    tau: ArrayLike3,
    config: Arm3DOFDynamicsConfig | None = None,
    external_torque: ArrayLike3 | None = None,
) -> np.ndarray:
    """Return q_ddot obtained from the manipulator dynamic equation."""

    cfg = config or Arm3DOFDynamicsConfig()
    torque = _as_vector3(tau, "tau")
    if external_torque is not None:
        torque = torque + _as_vector3(external_torque, "external_torque")

    passive = (
        coriolis_centrifugal_torque_3dof(q, q_dot, cfg)
        + gravity_torque_3dof(q, cfg)
        + viscous_friction_torque_3dof(q_dot, cfg)
    )
    return np.linalg.solve(mass_matrix_3dof(q, cfg), torque - passive)
