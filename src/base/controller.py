from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import numpy as np


class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    FORCE = "force"
    PID = "pid"
    CUSTOM = "custom"


@dataclass
class ControlTarget:
    joint_name: str
    mode: ControlMode
    value: float
    scale: float = 1.0
    offset: float = 0.0
    gains: Dict[str, float] = field(default_factory=dict)


@dataclass
class ControlCommand:
    timestamp: float
    targets: Dict[str, ControlTarget] = field(default_factory=dict)
    forces: Dict[str, np.ndarray] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    i_max: float = 0.0
    output_max: float = 0.0


class PIDController:
    def __init__(self, gains: Optional[PIDGains] = None):
        self.gains = gains or PIDGains()
        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._last_time: float = 0.0
    
    def compute(self, target: float, current: float, current_time: Optional[float] = None) -> float:
        import time
        
        if current_time is None:
            current_time = time.time()
        
        error = target - current
        
        dt = current_time - self._last_time if self._last_time > 0 else 0.001
        
        if dt <= 0:
            dt = 0.001
        
        p_term = self.gains.kp * error
        
        if self.gains.ki > 0:
            self._integral += error * dt
            if self.gains.i_max > 0:
                self._integral = np.clip(self._integral, -self.gains.i_max, self.gains.i_max)
        i_term = self.gains.ki * self._integral
        
        if self._last_time > 0 and dt > 0:
            d_term = self.gains.kd * (error - self._last_error) / dt
        else:
            d_term = 0.0
        
        output = p_term + i_term + d_term
        
        if self.gains.output_max > 0:
            output = np.clip(output, -self.gains.output_max, self.gains.output_max)
        
        self._last_error = error
        self._last_time = current_time
        
        return output
    
    def reset(self) -> None:
        import time
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()
    
    def set_gains(self, kp: Optional[float] = None, 
                   ki: Optional[float] = None, 
                   kd: Optional[float] = None,
                   i_max: Optional[float] = None,
                   output_max: Optional[float] = None) -> None:
        if kp is not None:
            self.gains.kp = kp
        if ki is not None:
            self.gains.ki = ki
        if kd is not None:
            self.gains.kd = kd
        if i_max is not None:
            self.gains.i_max = i_max
        if output_max is not None:
            self.gains.output_max = output_max


class BaseController(ABC):
    def __init__(self):
        self._control_modes: Dict[str, ControlMode] = {}
        self._pid_controllers: Dict[str, PIDController] = {}
        self._targets: Dict[str, float] = {}
        self._custom_handlers: Dict[str, Callable] = {}
        
        self._last_command: Optional[ControlCommand] = None
    
    @abstractmethod
    def apply_control(self, command: ControlCommand) -> None:
        pass
    
    @abstractmethod
    def get_control_output(self, joint_name: str) -> float:
        pass
    
    @abstractmethod
    def get_current_state(self, joint_name: str) -> Dict[str, float]:
        pass
    
    def set_control_mode(self, joint_name: str, mode: ControlMode) -> None:
        self._control_modes[joint_name] = mode
        
        if mode == ControlMode.PID and joint_name not in self._pid_controllers:
            self._pid_controllers[joint_name] = PIDController()
    
    def get_control_mode(self, joint_name: str) -> Optional[ControlMode]:
        return self._control_modes.get(joint_name)
    
    def set_pid_gains(self, joint_name: str, 
                       kp: Optional[float] = None,
                       ki: Optional[float] = None,
                       kd: Optional[float] = None,
                       **kwargs) -> None:
        if joint_name not in self._pid_controllers:
            self._pid_controllers[joint_name] = PIDController()
        
        self._pid_controllers[joint_name].set_gains(kp=kp, ki=ki, kd=kd, **kwargs)
    
    def set_target(self, joint_name: str, value: float, mode: Optional[ControlMode] = None) -> None:
        self._targets[joint_name] = value
        if mode:
            self.set_control_mode(joint_name, mode)
    
    def get_target(self, joint_name: str) -> Optional[float]:
        return self._targets.get(joint_name)
    
    def register_custom_handler(self, control_type: str, handler: Callable) -> None:
        self._custom_handlers[control_type] = handler
    
    def unregister_custom_handler(self, control_type: str) -> None:
        if control_type in self._custom_handlers:
            del self._custom_handlers[control_type]
    
    def create_command(self, 
                       joint_targets: Optional[Dict[str, Tuple[ControlMode, float]]] = None,
                       body_forces: Optional[Dict[str, np.ndarray]] = None,
                       **kwargs) -> ControlCommand:
        import time
        
        command = ControlCommand(timestamp=time.time())
        
        if joint_targets:
            for joint_name, (mode, value) in joint_targets.items():
                command.targets[joint_name] = ControlTarget(
                    joint_name=joint_name,
                    mode=mode,
                    value=value
                )
        
        if body_forces:
            command.forces = body_forces.copy()
        
        command.custom_data = kwargs
        
        return command
    
    @property
    def last_command(self) -> Optional[ControlCommand]:
        return self._last_command
    
    @property
    def controlled_joints(self) -> List[str]:
        return list(self._control_modes.keys())
    
    def reset(self) -> None:
        self._last_command = None
        self._targets.clear()
        
        for pid in self._pid_controllers.values():
            pid.reset()
