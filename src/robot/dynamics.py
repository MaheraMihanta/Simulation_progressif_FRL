"""Dynamics for a planar two-degree-of-freedom robotic arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import Arm2DOFConfig, ArrayLike2


@dataclass(frozen=True)
class Arm2DOFDynamicsConfig:
    """Physical constants for a rigid planar 2-DOF manipulator."""

    arm_config: Arm2DOFConfig = field(default_factory=Arm2DOFConfig)
    link_masses: tuple[float, float] = (1.0, 0.8)
    center_of_mass_ratios: tuple[float, float] = (0.5, 0.5)
    link_inertias: tuple[float, float] | None = None
    gravity: float = 9.81
    viscous_friction: tuple[float, float] = (0.08, 0.06)

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

        if self.gravity < 0.0:
            raise ValueError("gravity must be positive or zero.")

        friction = np.asarray(self.viscous_friction, dtype=float)
        if friction.shape != (2,) or np.any(friction < 0.0):
            raise ValueError("viscous_friction must contain two non-negative values.")


def _as_vector2(values: ArrayLike2, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"{name} must contain exactly two values.")
    return vector


def _link_parameters(
    config: Arm2DOFDynamicsConfig,
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


def mass_matrix(
    q: ArrayLike2,
    config: Arm2DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return the joint-space inertia matrix M(q)."""

    cfg = config or Arm2DOFDynamicsConfig()
    _, q2 = _as_vector2(q, "q")
    l1, _, m1, m2, r1, r2, i1, i2 = _link_parameters(cfg)
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


def coriolis_centrifugal_torque(
    q: ArrayLike2,
    q_dot: ArrayLike2,
    config: Arm2DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return Coriolis and centrifugal terms C(q, q_dot) q_dot."""

    cfg = config or Arm2DOFDynamicsConfig()
    _, q2 = _as_vector2(q, "q")
    dq1, dq2 = _as_vector2(q_dot, "q_dot")
    l1, _, _, m2, _, r2, _, _ = _link_parameters(cfg)
    coupling = m2 * l1 * r2 * float(np.sin(q2))

    return np.array(
        [
            -coupling * (2.0 * dq1 * dq2 + dq2 * dq2),
            coupling * dq1 * dq1,
        ],
        dtype=float,
    )


def gravity_torque(
    q: ArrayLike2,
    config: Arm2DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return gravity compensation torques for the current posture."""

    cfg = config or Arm2DOFDynamicsConfig()
    q1, q2 = _as_vector2(q, "q")
    l1, _, m1, m2, r1, r2, _, _ = _link_parameters(cfg)
    g = cfg.gravity

    tau1 = (m1 * r1 + m2 * l1) * g * np.cos(q1)
    tau1 += m2 * r2 * g * np.cos(q1 + q2)
    tau2 = m2 * r2 * g * np.cos(q1 + q2)
    return np.array([tau1, tau2], dtype=float)


def viscous_friction_torque(
    q_dot: ArrayLike2,
    config: Arm2DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return viscous friction torques opposing joint motion."""

    cfg = config or Arm2DOFDynamicsConfig()
    return np.asarray(cfg.viscous_friction, dtype=float) * _as_vector2(
        q_dot,
        "q_dot",
    )


def inverse_dynamics_torque(
    q: ArrayLike2,
    q_dot: ArrayLike2,
    q_ddot: ArrayLike2,
    config: Arm2DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return tau needed to produce q_ddot from the current state."""

    cfg = config or Arm2DOFDynamicsConfig()
    acceleration = _as_vector2(q_ddot, "q_ddot")
    return (
        mass_matrix(q, cfg) @ acceleration
        + coriolis_centrifugal_torque(q, q_dot, cfg)
        + gravity_torque(q, cfg)
        + viscous_friction_torque(q_dot, cfg)
    )


def joint_acceleration(
    q: ArrayLike2,
    q_dot: ArrayLike2,
    tau: ArrayLike2,
    config: Arm2DOFDynamicsConfig | None = None,
    external_torque: ArrayLike2 | None = None,
) -> np.ndarray:
    """Return q_ddot obtained from the manipulator dynamic equation."""

    cfg = config or Arm2DOFDynamicsConfig()
    torque = _as_vector2(tau, "tau")
    if external_torque is not None:
        torque = torque + _as_vector2(external_torque, "external_torque")

    passive = (
        coriolis_centrifugal_torque(q, q_dot, cfg)
        + gravity_torque(q, cfg)
        + viscous_friction_torque(q_dot, cfg)
    )
    return np.linalg.solve(mass_matrix(q, cfg), torque - passive)
