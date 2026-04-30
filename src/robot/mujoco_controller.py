from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import logging

from ..base import (
    BaseController,
    ControlMode,
    ControlCommand,
    ControlTarget,
    PIDController,
    PIDGains,
)
from .mujoco_model import MuJoCoModel
from ..mavlink.control_mapper import ControlTargetType


logger = logging.getLogger(__name__)


class MuJoCoController(BaseController):
    def __init__(self, model: Optional[MuJoCoModel] = None):
        super().__init__()
        self._model = model or MuJoCoModel()
        
        self._joint_to_actuator: Dict[str, int] = {}
        self._actuator_to_joint: Dict[int, str] = {}
        
        self._init_actuator_mappings()
    
    def _init_actuator_mappings(self) -> None:
        joint_names = self._model.joint_names
        
        for i, joint_name in enumerate(joint_names):
            self._joint_to_actuator[joint_name] = i
            self._actuator_to_joint[i] = joint_name
        
        logger.info(f"Initialized {len(self._joint_to_actuator)} actuator mappings")
    
    @property
    def model(self) -> MuJoCoModel:
        return self._model
    
    def load_model(self, model_path: str) -> bool:
        success = self._model.load_model(model_path)
        if success:
            self._init_actuator_mappings()
        return success
    
    def apply_control(self, command: ControlCommand) -> None:
        self._last_command = command
        
        for joint_name, target in command.targets.items():
            if not self._model.has_joint(joint_name):
                logger.warning(f"Joint not found: {joint_name}")
                continue
            
            self.set_control_mode(joint_name, self._convert_mode(target.mode))
            self._apply_target(joint_name, target)
        
        for body_name, force in command.forces.items():
            if self._model.has_body(body_name):
                self._model.apply_body_force(body_name, force)
            else:
                logger.warning(f"Body not found: {body_name}")
    
    def _convert_mode(self, target_mode: ControlMode) -> ControlMode:
        return target_mode
    
    def _apply_target(self, joint_name: str, target: ControlTarget) -> None:
        mode = self._control_modes.get(joint_name, ControlMode.TORQUE)
        
        value = target.value * target.scale + target.offset
        
        if mode == ControlMode.POSITION:
            self._targets[joint_name] = value
            self._set_position_control(joint_name, value)
        
        elif mode == ControlMode.VELOCITY:
            self._targets[joint_name] = value
            self._set_velocity_control(joint_name, value)
        
        elif mode == ControlMode.TORQUE:
            self._targets[joint_name] = value
            self._set_torque_control(joint_name, value)
        
        elif mode == ControlMode.FORCE:
            self._targets[joint_name] = value
            self._set_force_control(joint_name, value)
        
        elif mode == ControlMode.PID:
            self._targets[joint_name] = value
            self._apply_pid_control(joint_name, value)
        
        elif mode == ControlMode.CUSTOM:
            if joint_name in self._custom_handlers:
                try:
                    self._custom_handlers[joint_name](value)
                except Exception as e:
                    logger.error(f"Custom handler error for {joint_name}: {e}")
    
    def _set_position_control(self, joint_name: str, position: float) -> None:
        actuator_idx = self._joint_to_actuator.get(joint_name)
        if actuator_idx is not None:
            self._model.set_control(actuator_idx, position)
    
    def _set_velocity_control(self, joint_name: str, velocity: float) -> None:
        actuator_idx = self._joint_to_actuator.get(joint_name)
        if actuator_idx is not None:
            self._model.set_control(actuator_idx, velocity)
    
    def _set_torque_control(self, joint_name: str, torque: float) -> None:
        actuator_idx = self._joint_to_actuator.get(joint_name)
        if actuator_idx is not None:
            self._model.set_control(actuator_idx, torque)
    
    def _set_force_control(self, joint_name: str, force: float) -> None:
        actuator_idx = self._joint_to_actuator.get(joint_name)
        if actuator_idx is not None:
            self._model.set_control(actuator_idx, force)
    
    def _apply_pid_control(self, joint_name: str, target_position: float) -> None:
        import time
        
        current_position = self._model.get_joint_qpos(joint_name)
        if current_position is None:
            return
        
        if joint_name not in self._pid_controllers:
            self._pid_controllers[joint_name] = PIDController()
        
        pid = self._pid_controllers[joint_name]
        torque = pid.compute(target_position, current_position, time.time())
        
        self._set_torque_control(joint_name, torque)
    
    def get_control_output(self, joint_name: str) -> float:
        actuator_idx = self._joint_to_actuator.get(joint_name)
        if actuator_idx is not None:
            return self._model.get_control(actuator_idx)
        return 0.0
    
    def get_current_state(self, joint_name: str) -> Dict[str, float]:
        state = {
            "position": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "target": self._targets.get(joint_name, 0.0),
            "control_output": self.get_control_output(joint_name),
        }
        
        pos = self._model.get_joint_qpos(joint_name)
        if pos is not None:
            state["position"] = pos
        
        vel = self._model.get_joint_qvel(joint_name)
        if vel is not None:
            state["velocity"] = vel
        
        return state
    
    def get_all_joint_states(self) -> Dict[str, Dict[str, float]]:
        states = {}
        for joint_name in self._model.joint_names:
            states[joint_name] = self.get_current_state(joint_name)
        return states
    
    def apply_target_type(self,
                           target_name: str,
                           target_type: ControlTargetType,
                           value: float) -> None:
        if target_type in [
            ControlTargetType.JOINT_POSITION,
            ControlTargetType.JOINT_VELOCITY,
            ControlTargetType.JOINT_TORQUE,
        ]:
            if self._model.has_joint(target_name):
                if target_type == ControlTargetType.JOINT_POSITION:
                    self.set_control_mode(target_name, ControlMode.POSITION)
                    self._set_position_control(target_name, value)
                elif target_type == ControlTargetType.JOINT_VELOCITY:
                    self.set_control_mode(target_name, ControlMode.VELOCITY)
                    self._set_velocity_control(target_name, value)
                elif target_type == ControlTargetType.JOINT_TORQUE:
                    self.set_control_mode(target_name, ControlMode.TORQUE)
                    self._set_torque_control(target_name, value)
        
        elif target_type == ControlTargetType.BODY_FORCE:
            if self._model.has_body(target_name):
                force = np.array([value, 0.0, 0.0])
                self._model.apply_body_force(target_name, force)
        
        elif target_type == ControlTargetType.BODY_TORQUE:
            if self._model.has_body(target_name):
                force = np.array([0.0, 0.0, 0.0, value, 0.0, 0.0])
                self._model.apply_body_force(target_name, force)
    
    def step(self, n_steps: int = 1) -> None:
        self._update_pid_controllers()
        self._model.step(n_steps)
    
    def _update_pid_controllers(self) -> None:
        import time
        
        for joint_name, mode in self._control_modes.items():
            if mode == ControlMode.PID and joint_name in self._targets:
                current_position = self._model.get_joint_qpos(joint_name)
                if current_position is not None and joint_name in self._pid_controllers:
                    target = self._targets[joint_name]
                    pid = self._pid_controllers[joint_name]
                    torque = pid.compute(target, current_position, time.time())
                    self._set_torque_control(joint_name, torque)
    
    def reset(self) -> None:
        super().reset()
        self._model.reset()
    
    @property
    def joint_names(self) -> List[str]:
        return self._model.joint_names
    
    @property
    def body_names(self) -> List[str]:
        return self._model.body_names
    
    @property
    def sensor_names(self) -> List[str]:
        return self._model.sensor_names
    
    def get_robot_state(self):
        return self._model.get_state()
    
    def get_timestep(self) -> float:
        return self._model.get_timestep()
