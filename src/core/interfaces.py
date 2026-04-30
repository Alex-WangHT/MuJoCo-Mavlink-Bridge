from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum
import numpy as np

from .plant import Plant, StateVector, ControlVector


class ControlSource(Enum):
    HIL_ACTUATOR_CONTROLS = "hil_actuator_controls"
    RC_CHANNELS = "rc_channels"
    MANUAL_CONTROL = "manual_control"
    CUSTOM = "custom"


class StateTarget(Enum):
    HIL_STATE = "hil_state"
    HIL_STATE_QUATERNION = "hil_state_quaternion"
    HIL_GPS = "hil_gps"
    HIL_SENSOR = "hil_sensor"
    CUSTOM = "custom"


@dataclass
class ControlMappingEntry:
    mavlink_source: ControlSource
    mavlink_index: int
    plant_control_name: str
    plant_control_index: int
    scale: float = 1.0
    offset: float = 0.0
    range_min: float = -np.inf
    range_max: float = np.inf
    transform: Optional[Callable[[float], float]] = None
    enabled: bool = True
    
    def apply(self, raw_value: float) -> float:
        if self.transform:
            value = self.transform(raw_value)
        else:
            value = raw_value * self.scale + self.offset
        
        return np.clip(value, self.range_min, self.range_max)


@dataclass
class StateMappingEntry:
    plant_state_source: str
    plant_state_type: str
    mavlink_target: StateTarget
    mavlink_field: str
    scale: float = 1.0
    offset: float = 0.0
    enabled: bool = True


class ControlMapping:
    def __init__(self):
        self._entries: List[ControlMappingEntry] = []
        self._mavlink_to_entry: Dict[Tuple[ControlSource, int], ControlMappingEntry] = {}
        self._name_to_entry: Dict[str, ControlMappingEntry] = {}
    
    def add_mapping(self,
                     mavlink_source: Union[ControlSource, str],
                     mavlink_index: int,
                     plant_control_name: str,
                     plant_control_index: int,
                     scale: float = 1.0,
                     offset: float = 0.0,
                     range_min: float = -np.inf,
                     range_max: float = np.inf,
                     transform: Optional[Callable[[float], float]] = None,
                     enabled: bool = True) -> None:
        if isinstance(mavlink_source, str):
            mavlink_source = ControlSource(mavlink_source)
        
        entry = ControlMappingEntry(
            mavlink_source=mavlink_source,
            mavlink_index=mavlink_index,
            plant_control_name=plant_control_name,
            plant_control_index=plant_control_index,
            scale=scale,
            offset=offset,
            range_min=range_min,
            range_max=range_max,
            transform=transform,
            enabled=enabled
        )
        
        self._entries.append(entry)
        self._mavlink_to_entry[(mavlink_source, mavlink_index)] = entry
        self._name_to_entry[plant_control_name] = entry
    
    def get_mapping(self, mavlink_source: ControlSource, mavlink_index: int) -> Optional[ControlMappingEntry]:
        return self._mavlink_to_entry.get((mavlink_source, mavlink_index))
    
    def get_mapping_by_name(self, name: str) -> Optional[ControlMappingEntry]:
        return self._name_to_entry.get(name)
    
    def map_controls(self,
                      mavlink_source: ControlSource,
                      raw_controls: List[float]) -> Dict[int, float]:
        result = {}
        
        for i, raw_value in enumerate(raw_controls):
            entry = self.get_mapping(mavlink_source, i)
            if entry and entry.enabled:
                value = entry.apply(raw_value)
                result[entry.plant_control_index] = value
        
        return result
    
    def map_single_control(self,
                            mavlink_source: ControlSource,
                            mavlink_index: int,
                            raw_value: float) -> Optional[Tuple[int, float]]:
        entry = self.get_mapping(mavlink_source, mavlink_index)
        if entry and entry.enabled:
            value = entry.apply(raw_value)
            return (entry.plant_control_index, value)
        return None
    
    def create_default_joint_mappings(self,
                                        control_names: List[str],
                                        mavlink_source: ControlSource = ControlSource.HIL_ACTUATOR_CONTROLS,
                                        start_index: int = 0,
                                        scale: float = 1.0,
                                        offset: float = 0.0) -> None:
        for i, name in enumerate(control_names):
            self.add_mapping(
                mavlink_source=mavlink_source,
                mavlink_index=start_index + i,
                plant_control_name=name,
                plant_control_index=i,
                scale=scale,
                offset=offset
            )
    
    def enable_all(self) -> None:
        for entry in self._entries:
            entry.enabled = True
    
    def disable_all(self) -> None:
        for entry in self._entries:
            entry.enabled = False
    
    def clear(self) -> None:
        self._entries.clear()
        self._mavlink_to_entry.clear()
        self._name_to_entry.clear()
    
    @property
    def entries(self) -> List[ControlMappingEntry]:
        return self._entries.copy()
    
    @property
    def enabled_entries(self) -> List[ControlMappingEntry]:
        return [e for e in self._entries if e.enabled]


class StateMapping:
    def __init__(self):
        self._entries: List[StateMappingEntry] = []
    
    def add_mapping(self,
                     plant_state_source: str,
                     plant_state_type: str,
                     mavlink_target: Union[StateTarget, str],
                     mavlink_field: str,
                     scale: float = 1.0,
                     offset: float = 0.0,
                     enabled: bool = True) -> None:
        if isinstance(mavlink_target, str):
            mavlink_target = StateTarget(mavlink_target)
        
        entry = StateMappingEntry(
            plant_state_source=plant_state_source,
            plant_state_type=plant_state_type,
            mavlink_target=mavlink_target,
            mavlink_field=mavlink_field,
            scale=scale,
            offset=offset,
            enabled=enabled
        )
        
        self._entries.append(entry)
    
    def get_mappings_for_target(self, target: StateTarget) -> List[StateMappingEntry]:
        return [e for e in self._entries if e.mavlink_target == target and e.enabled]
    
    def create_default_joint_state_mappings(self,
                                              joint_names: List[str],
                                              mavlink_target: StateTarget = StateTarget.HIL_ACTUATOR_CONTROLS) -> None:
        for i, name in enumerate(joint_names):
            self.add_mapping(
                plant_state_source=name,
                plant_state_type="joint_position",
                mavlink_target=mavlink_target,
                mavlink_field=f"controls[{i}]"
            )
    
    def clear(self) -> None:
        self._entries.clear()
    
    @property
    def entries(self) -> List[StateMappingEntry]:
        return self._entries.copy()


class MavlinkInterface(ABC):
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass
    
    @abstractmethod
    def start(self) -> None:
        pass
    
    @abstractmethod
    def stop(self) -> None:
        pass
    
    @abstractmethod
    def receive_controls(self) -> Optional[Dict[ControlSource, List[float]]]:
        pass
    
    @abstractmethod
    def send_state(self, state: StateVector, mapping: StateMapping) -> bool:
        pass
    
    @abstractmethod
    def send_hil_actuator_controls(self, controls: List[float]) -> bool:
        pass
    
    @abstractmethod
    def send_hil_state_quaternion(self,
                                    attitude: List[float],
                                    rollspeed: float,
                                    pitchspeed: float,
                                    yawspeed: float,
                                    lat: int,
                                    lon: int,
                                    alt: int,
                                    vx: float,
                                    vy: float,
                                    vz: float,
                                    xacc: float,
                                    yacc: float,
                                    zacc: float) -> bool:
        pass
    
    @abstractmethod
    def send_heartbeat(self) -> bool:
        pass
