"""Robot models used by the FRL simulations."""

from .arm_2dof import Arm2DOF
from .arm_3dof import Arm3DOF
from .dynamics import (
    Arm2DOFDynamicsConfig,
    coriolis_centrifugal_torque,
    gravity_torque,
    inverse_dynamics_torque,
    joint_acceleration,
    mass_matrix,
    viscous_friction_torque,
)
from .dynamics_3dof import (
    Arm3DOFDynamicsConfig,
    coriolis_centrifugal_torque_3dof,
    gravity_torque_3dof,
    inverse_dynamics_torque_3dof,
    joint_acceleration_3dof,
    mass_matrix_3dof,
    viscous_friction_torque_3dof,
    yaw_inertia_and_gradient,
)
from .kinematics import (
    Arm2DOFConfig,
    clip_to_joint_limits,
    forward_kinematics,
    inverse_kinematics,
    is_reachable,
    jacobian,
    joint_positions,
    workspace_radius,
)
from .kinematics_3dof import (
    Arm3DOFConfig,
    clip_to_joint_limits_3dof,
    forward_kinematics_3dof,
    inverse_kinematics_3dof,
    is_reachable_3dof,
    jacobian_3dof,
    joint_positions_3dof,
    workspace_radius_3dof,
)

__all__ = [
    "Arm2DOF",
    "Arm2DOFConfig",
    "Arm2DOFDynamicsConfig",
    "Arm3DOF",
    "Arm3DOFConfig",
    "Arm3DOFDynamicsConfig",
    "clip_to_joint_limits",
    "clip_to_joint_limits_3dof",
    "coriolis_centrifugal_torque",
    "coriolis_centrifugal_torque_3dof",
    "forward_kinematics",
    "forward_kinematics_3dof",
    "gravity_torque",
    "gravity_torque_3dof",
    "inverse_dynamics_torque",
    "inverse_dynamics_torque_3dof",
    "inverse_kinematics",
    "inverse_kinematics_3dof",
    "is_reachable",
    "is_reachable_3dof",
    "jacobian",
    "jacobian_3dof",
    "joint_acceleration",
    "joint_acceleration_3dof",
    "joint_positions",
    "joint_positions_3dof",
    "mass_matrix",
    "mass_matrix_3dof",
    "viscous_friction_torque",
    "viscous_friction_torque_3dof",
    "workspace_radius",
    "workspace_radius_3dof",
    "yaw_inertia_and_gradient",
]
