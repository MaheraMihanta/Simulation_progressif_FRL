from __future__ import annotations

import time

import numpy as np

from .config import RobotConfig, SimulationConfig
from .state import ArmState


class CoppeliaConnectionError(RuntimeError):
    """Raised when the ZeroMQ remote API cannot reach the running scene."""


class CoppeliaArmEnv:
    """Synchronous Python-CoppeliaSim joint-position environment."""

    def __init__(self, robot_config: RobotConfig, simulation_config: SimulationConfig) -> None:
        self.robot_config = robot_config
        self.simulation_config = simulation_config
        self.client = None
        self.sim = None
        self.joint_handles: list[int] = []
        self.tip_handle: int | None = None
        self.running = False

    def connect(self) -> None:
        if self.sim is not None:
            return
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient

            self.client = RemoteAPIClient()
            self.sim = self.client.require("sim")
            self._configure_timestep()
            self.joint_handles = self._resolve_joint_handles()
            if self.robot_config.tip_path:
                self.tip_handle = self.sim.getObject(self.robot_config.tip_path)
        except Exception as exc:
            raise CoppeliaConnectionError(
                "Impossible de se connecter a CoppeliaSim. Ouvrez la scene .ttt, "
                "verifiez que l'add-on ZeroMQ remote API est actif, puis relancez."
            ) from exc

    def start(self) -> None:
        if self.running:
            return
        self.connect()
        assert self.sim is not None
        if self.simulation_config.synchronous_stepping:
            self.sim.setStepping(True)
        self.sim.startSimulation()
        self.running = True
        for _ in range(self.simulation_config.settling_steps):
            self.step(self.read_state().q)

    def reset(self, q0: np.ndarray | None = None) -> ArmState:
        if q0 is not None:
            for handle, value in zip(self.joint_handles, q0):
                self.sim.setJointTargetPosition(handle, float(value))
            if self.running:
                for _ in range(self.simulation_config.settling_steps):
                    if self.simulation_config.synchronous_stepping:
                        self.sim.step()
                    else:
                        time.sleep(self.simulation_config.dt)
        return self.read_state()

    def read_state(self) -> ArmState:
        self.connect()
        assert self.sim is not None
        q = np.array([self.sim.getJointPosition(handle) for handle in self.joint_handles], dtype=float)
        q_dot = np.array([self.sim.getJointVelocity(handle) for handle in self.joint_handles], dtype=float)
        tip_position = None
        if self.tip_handle is not None:
            tip_position = np.array(self.sim.getObjectPosition(self.tip_handle, self.sim.handle_world), dtype=float)
        return ArmState(q=q, q_dot=q_dot, tip_position=tip_position)

    def step(self, target_position: np.ndarray) -> ArmState:
        if not self.running:
            raise RuntimeError("CoppeliaArmEnv.start() must be called before step()")
        assert self.sim is not None
        target = np.asarray(target_position, dtype=float)
        lower = np.asarray(self.robot_config.joint_lower_limits, dtype=float)
        upper = np.asarray(self.robot_config.joint_upper_limits, dtype=float)
        target = np.clip(target, lower, upper)
        for handle, value in zip(self.joint_handles, target):
            self.sim.setJointTargetPosition(handle, float(value))
        if self.simulation_config.synchronous_stepping:
            self.sim.step()
        else:
            time.sleep(self.simulation_config.dt)
        return self.read_state()

    def stop(self) -> None:
        if self.sim is None:
            return
        try:
            self.sim.stopSimulation()
        finally:
            self.running = False

    def _configure_timestep(self) -> None:
        assert self.sim is not None
        try:
            self.sim.setFloatParam(self.sim.floatparam_simulation_time_step, self.simulation_config.dt)
        except Exception:
            # Some CoppeliaSim builds lock the time step from the scene settings.
            pass

    def _resolve_joint_handles(self) -> list[int]:
        assert self.sim is not None
        handles: list[int] = []
        for path in self.robot_config.joint_paths:
            handles.append(self.sim.getObject(path))
        if len(handles) != self.robot_config.dof:
            raise CoppeliaConnectionError("Nombre de joints detectes incoherent")
        return handles
