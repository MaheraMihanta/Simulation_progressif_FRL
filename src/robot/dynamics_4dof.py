"""Dynamics for a spatial 4-DOF yaw-plus-planar robotic arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics_4dof import Arm4DOFConfig, ArrayLike4


@dataclass(frozen=True)
class Arm4DOFDynamicsConfig:
    """Physical constants for the 4-DOF manipulator.

    The dynamic model keeps the same modelling level as the 3-DOF simulator:
    a yaw inertia around the vertical axis and a rigid planar chain in the
    radial-z plane. The planar terms are assembled generically for 3 links.
    """

    arm_config: Arm4DOFConfig = field(default_factory=Arm4DOFConfig)
    link_masses: tuple[float, float, float] = (1.0, 0.8, 0.55)
    center_of_mass_ratios: tuple[float, float, float] = (0.5, 0.5, 0.5)
    link_inertias: tuple[float, float, float] | None = None
    base_yaw_inertia: float = 0.06
    gravity: float = 9.81
    viscous_friction: tuple[float, float, float, float] = (0.05, 0.08, 0.06, 0.04)

    def __post_init__(self) -> None:
        masses = np.asarray(self.link_masses, dtype=float)
        if masses.shape != (3,) or np.any(masses <= 0.0):
            raise ValueError("link_masses must contain three positive values.")

        ratios = np.asarray(self.center_of_mass_ratios, dtype=float)
        if ratios.shape != (3,) or np.any(ratios <= 0.0) or np.any(ratios > 1.0):
            raise ValueError(
                "center_of_mass_ratios must contain three values in ]0, 1]."
            )

        if self.link_inertias is not None:
            inertias = np.asarray(self.link_inertias, dtype=float)
            if inertias.shape != (3,) or np.any(inertias <= 0.0):
                raise ValueError("link_inertias must contain three positive values.")

        if self.base_yaw_inertia <= 0.0:
            raise ValueError("base_yaw_inertia must be strictly positive.")
        if self.gravity < 0.0:
            raise ValueError("gravity must be positive or zero.")

        friction = np.asarray(self.viscous_friction, dtype=float)
        if friction.shape != (4,) or np.any(friction < 0.0):
            raise ValueError("viscous_friction must contain four non-negative values.")


def _as_vector4(values: ArrayLike4, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must contain exactly four values.")
    return vector


def _link_parameters(
    config: Arm4DOFDynamicsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(config.arm_config.link_lengths, dtype=float)
    masses = np.asarray(config.link_masses, dtype=float)
    ratios = np.asarray(config.center_of_mass_ratios, dtype=float)
    com_lengths = ratios * lengths
    if config.link_inertias is None:
        inertias = masses * lengths * lengths / 12.0
    else:
        inertias = np.asarray(config.link_inertias, dtype=float)
    return lengths, masses, com_lengths, inertias


def _com_radial_positions_and_jacobians(
    q_planar: np.ndarray,
    config: Arm4DOFDynamicsConfig,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    lengths, _, com_lengths, _ = _link_parameters(config)
    theta = np.cumsum(q_planar)
    radial_positions = np.zeros(3, dtype=float)
    radial_jacobians: list[np.ndarray] = []
    z_jacobians: list[np.ndarray] = []

    for link_index in range(3):
        radial = 0.0
        for segment in range(link_index):
            radial += lengths[segment] * np.cos(theta[segment])
        radial += com_lengths[link_index] * np.cos(theta[link_index])
        radial_positions[link_index] = radial

        j_radial = np.zeros(3, dtype=float)
        j_z = np.zeros(3, dtype=float)
        for joint_index in range(link_index + 1):
            for segment in range(joint_index, link_index):
                j_radial[joint_index] -= lengths[segment] * np.sin(theta[segment])
                j_z[joint_index] += lengths[segment] * np.cos(theta[segment])
            j_radial[joint_index] -= com_lengths[link_index] * np.sin(
                theta[link_index]
            )
            j_z[joint_index] += com_lengths[link_index] * np.cos(theta[link_index])
        radial_jacobians.append(j_radial)
        z_jacobians.append(j_z)

    return radial_positions, radial_jacobians, z_jacobians


def _planar_mass_matrix(
    q_planar: np.ndarray,
    config: Arm4DOFDynamicsConfig,
) -> np.ndarray:
    _, masses, _, inertias = _link_parameters(config)
    _, radial_jacobians, z_jacobians = _com_radial_positions_and_jacobians(
        q_planar,
        config,
    )
    matrix = np.zeros((3, 3), dtype=float)
    for link_index in range(3):
        translational = np.outer(
            radial_jacobians[link_index],
            radial_jacobians[link_index],
        ) + np.outer(z_jacobians[link_index], z_jacobians[link_index])
        angular = np.zeros((3, 3), dtype=float)
        active = np.zeros(3, dtype=float)
        active[: link_index + 1] = 1.0
        angular += np.outer(active, active)
        matrix += masses[link_index] * translational + inertias[link_index] * angular
    return matrix


def yaw_inertia_and_gradient_4dof(
    q: ArrayLike4,
    config: Arm4DOFDynamicsConfig | None = None,
) -> tuple[float, np.ndarray]:
    """Return base yaw inertia and its gradient over q."""

    cfg = config or Arm4DOFDynamicsConfig()
    q_array = _as_vector4(q, "q")
    q_planar = q_array[1:]
    _, masses, _, _ = _link_parameters(cfg)
    radial_positions, radial_jacobians, _ = _com_radial_positions_and_jacobians(
        q_planar,
        cfg,
    )

    inertia = cfg.base_yaw_inertia + float(np.sum(masses * radial_positions**2))
    gradient = np.zeros(4, dtype=float)
    for link_index in range(3):
        gradient[1:] += (
            2.0
            * masses[link_index]
            * radial_positions[link_index]
            * radial_jacobians[link_index]
        )
    return inertia, gradient


def mass_matrix_4dof(
    q: ArrayLike4,
    config: Arm4DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return the joint-space inertia matrix M(q)."""

    cfg = config or Arm4DOFDynamicsConfig()
    q_array = _as_vector4(q, "q")
    yaw_inertia, _ = yaw_inertia_and_gradient_4dof(q_array, cfg)
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0] = yaw_inertia
    matrix[1:, 1:] = _planar_mass_matrix(q_array[1:], cfg)
    return matrix


def _mass_matrix_derivatives(
    q: np.ndarray,
    config: Arm4DOFDynamicsConfig,
    epsilon: float = 1e-6,
) -> np.ndarray:
    derivatives = np.zeros((4, 4, 4), dtype=float)
    for index in range(1, 4):
        delta = np.zeros(4, dtype=float)
        delta[index] = epsilon
        derivatives[:, :, index] = (
            mass_matrix_4dof(q + delta, config)
            - mass_matrix_4dof(q - delta, config)
        ) / (2.0 * epsilon)
    return derivatives


def coriolis_centrifugal_torque_4dof(
    q: ArrayLike4,
    q_dot: ArrayLike4,
    config: Arm4DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return Coriolis and centrifugal terms C(q, q_dot) q_dot.

    The current 4-DOF study keeps the dominant yaw/posture coupling and omits
    the smaller planar 3R Coriolis terms. This keeps the first 4-DOF extension
    fast enough for controller comparisons while preserving consistency between
    inverse and forward dynamics.
    """

    cfg = config or Arm4DOFDynamicsConfig()
    q_array = _as_vector4(q, "q")
    velocity = _as_vector4(q_dot, "q_dot")
    torque = np.zeros(4, dtype=float)
    yaw_rate = float(velocity[0])
    _, yaw_gradient = yaw_inertia_and_gradient_4dof(q_array, cfg)
    yaw_inertia_dot = float(np.dot(yaw_gradient, velocity))
    torque[0] = yaw_inertia_dot * yaw_rate
    torque[1:] = -0.5 * yaw_gradient[1:] * yaw_rate * yaw_rate
    return torque


def gravity_torque_4dof(
    q: ArrayLike4,
    config: Arm4DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return gravity compensation torques for the current posture."""

    cfg = config or Arm4DOFDynamicsConfig()
    q_array = _as_vector4(q, "q")
    q_planar = q_array[1:]
    _, masses, _, _ = _link_parameters(cfg)
    _, _, z_jacobians = _com_radial_positions_and_jacobians(q_planar, cfg)

    torque = np.zeros(4, dtype=float)
    for link_index in range(3):
        torque[1:] += masses[link_index] * cfg.gravity * z_jacobians[link_index]
    return torque


def viscous_friction_torque_4dof(
    q_dot: ArrayLike4,
    config: Arm4DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return viscous friction torques opposing joint motion."""

    cfg = config or Arm4DOFDynamicsConfig()
    return np.asarray(cfg.viscous_friction, dtype=float) * _as_vector4(
        q_dot,
        "q_dot",
    )


def inverse_dynamics_torque_4dof(
    q: ArrayLike4,
    q_dot: ArrayLike4,
    q_ddot: ArrayLike4,
    config: Arm4DOFDynamicsConfig | None = None,
) -> np.ndarray:
    """Return tau needed to produce q_ddot from the current state."""

    cfg = config or Arm4DOFDynamicsConfig()
    acceleration = _as_vector4(q_ddot, "q_ddot")
    return (
        mass_matrix_4dof(q, cfg) @ acceleration
        + coriolis_centrifugal_torque_4dof(q, q_dot, cfg)
        + gravity_torque_4dof(q, cfg)
        + viscous_friction_torque_4dof(q_dot, cfg)
    )


def joint_acceleration_4dof(
    q: ArrayLike4,
    q_dot: ArrayLike4,
    tau: ArrayLike4,
    config: Arm4DOFDynamicsConfig | None = None,
    external_torque: ArrayLike4 | None = None,
) -> np.ndarray:
    """Return q_ddot obtained from the manipulator dynamic equation."""

    cfg = config or Arm4DOFDynamicsConfig()
    torque = _as_vector4(tau, "tau")
    if external_torque is not None:
        torque = torque + _as_vector4(external_torque, "external_torque")

    passive = (
        coriolis_centrifugal_torque_4dof(q, q_dot, cfg)
        + gravity_torque_4dof(q, cfg)
        + viscous_friction_torque_4dof(q_dot, cfg)
    )
    return np.linalg.solve(mass_matrix_4dof(q, cfg), torque - passive)
