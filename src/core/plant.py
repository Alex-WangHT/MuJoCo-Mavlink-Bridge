from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class ControlVector:
    values: np.ndarray
    names: Optional[List[str]] = None
    
    def __post_init__(self):
        if isinstance(self.values, list):
            self.values = np.array(self.values, dtype=np.float64)
        
        if self.names is None:
            self.names = [f"u{i}" for i in range(len(self.values))]
    
    def __len__(self) -> int:
        return len(self.values)
    
    def __getitem__(self, idx: int) -> float:
        return float(self.values[idx])
    
    def __setitem__(self, idx: int, value: float):
        self.values[idx] = value
    
    def to_list(self) -> List[float]:
        return self.values.tolist()
    
    def to_dict(self) -> Dict[str, float]:
        if self.names:
            return {name: float(val) for name, val in zip(self.names, self.values)}
        return {f"u{i}": float(val) for i, val in enumerate(self.values)}


@dataclass
class StateVector:
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    body_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    body_velocities: Dict[str, np.ndarray] = field(default_factory=dict)
    sensor_data: Dict[str, np.ndarray] = field(default_factory=dict)
    time: float = 0.0
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_joint_position(self, name: str) -> Optional[float]:
        return self.joint_positions.get(name)
    
    def get_joint_velocity(self, name: str) -> Optional[float]:
        return self.joint_velocities.get(name)
    
    def get_body_position(self, name: str) -> Optional[np.ndarray]:
        return self.body_positions.get(name)
    
    def get_body_velocity(self, name: str) -> Optional[np.ndarray]:
        return self.body_velocities.get(name)
    
    def get_sensor_data(self, name: str) -> Optional[np.ndarray]:
        return self.sensor_data.get(name)
    
    @property
    def joint_names(self) -> List[str]:
        return list(self.joint_positions.keys())
    
    @property
    def body_names(self) -> List[str]:
        return list(self.body_positions.keys())
    
    @property
    def sensor_names(self) -> List[str]:
        return list(self.sensor_data.keys())


class Plant(ABC):
    @property
    @abstractmethod
    def control_dim(self) -> int:
        pass
    
    @property
    @abstractmethod
    def control_names(self) -> List[str]:
        pass
    
    @property
    @abstractmethod
    def timestep(self) -> float:
        pass
    
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        pass
    
    @abstractmethod
    def set_control(self, control: ControlVector) -> None:
        pass
    
    @abstractmethod
    def set_control_by_index(self, index: int, value: float) -> None:
        pass
    
    @abstractmethod
    def set_control_by_name(self, name: str, value: float) -> None:
        pass
    
    @abstractmethod
    def get_control(self) -> ControlVector:
        pass
    
    @abstractmethod
    def get_state(self) -> StateVector:
        pass
    
    @abstractmethod
    def step(self, n_steps: int = 1) -> None:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @property
    def state(self) -> StateVector:
        return self.get_state()
    
    def set_control_array(self, values: np.ndarray) -> None:
        self.set_control(ControlVector(values=values))
    
    def set_control_list(self, values: List[float]) -> None:
        self.set_control(ControlVector(values=np.array(values, dtype=np.float64)))
