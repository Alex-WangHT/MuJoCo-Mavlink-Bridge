from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class JointInfo:
    name: str
    index: int
    joint_type: str
    qpos_idx: int
    qvel_idx: int
    dof: int = 1
    range: Tuple[float, float] = (-np.inf, np.inf)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyInfo:
    name: str
    index: int
    parent_name: Optional[str]
    body_id: int
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorInfo:
    name: str
    index: int
    sensor_type: str
    data_dim: int
    data_start: int
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    time: float
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]
    joint_accelerations: Dict[str, float]
    joint_torques: Dict[str, float]
    body_positions: Dict[str, np.ndarray]
    body_velocities: Dict[str, np.ndarray]
    sensor_data: Dict[str, np.ndarray]
    custom_data: Dict[str, Any] = field(default_factory=dict)


class BaseRobotModel(ABC):
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._is_initialized = False
        
        self._joints: Dict[str, JointInfo] = {}
        self._bodies: Dict[str, BodyInfo] = {}
        self._sensors: Dict[str, SensorInfo] = {}
        
        self._joint_names: List[str] = []
        self._body_names: List[str] = []
        self._sensor_names: List[str] = []
        
        if model_path:
            self.load_model(model_path)
        else:
            self._create_default_model()
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        pass
    
    @abstractmethod
    def _create_default_model(self) -> bool:
        pass
    
    @abstractmethod
    def step(self, n_steps: int = 1) -> None:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def get_timestep(self) -> float:
        pass
    
    @abstractmethod
    def get_state(self) -> RobotState:
        pass
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    @property
    def joints(self) -> Dict[str, JointInfo]:
        return self._joints.copy()
    
    @property
    def joint_names(self) -> List[str]:
        return self._joint_names.copy()
    
    @property
    def bodies(self) -> Dict[str, BodyInfo]:
        return self._bodies.copy()
    
    @property
    def body_names(self) -> List[str]:
        return self._body_names.copy()
    
    @property
    def sensors(self) -> Dict[str, SensorInfo]:
        return self._sensors.copy()
    
    @property
    def sensor_names(self) -> List[str]:
        return self._sensor_names.copy()
    
    def get_joint_info(self, joint_name: str) -> Optional[JointInfo]:
        return self._joints.get(joint_name)
    
    def get_body_info(self, body_name: str) -> Optional[BodyInfo]:
        return self._bodies.get(body_name)
    
    def get_sensor_info(self, sensor_name: str) -> Optional[SensorInfo]:
        return self._sensors.get(sensor_name)
    
    def has_joint(self, joint_name: str) -> bool:
        return joint_name in self._joints
    
    def has_body(self, body_name: str) -> bool:
        return body_name in self._bodies
    
    def has_sensor(self, sensor_name: str) -> bool:
        return sensor_name in self._sensors
    
    def _add_joint(self, joint_info: JointInfo) -> None:
        self._joints[joint_info.name] = joint_info
        if joint_info.name not in self._joint_names:
            self._joint_names.append(joint_info.name)
    
    def _add_body(self, body_info: BodyInfo) -> None:
        self._bodies[body_info.name] = body_info
        if body_info.name not in self._body_names:
            self._body_names.append(body_info.name)
    
    def _add_sensor(self, sensor_info: SensorInfo) -> None:
        self._sensors[sensor_info.name] = sensor_info
        if sensor_info.name not in self._sensor_names:
            self._sensor_names.append(sensor_info.name)
    
    def _clear_model(self) -> None:
        self._joints.clear()
        self._bodies.clear()
        self._sensors.clear()
        self._joint_names.clear()
        self._body_names.clear()
        self._sensor_names.clear()
        self._is_initialized = False
