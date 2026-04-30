import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .core import (
    Plant,
    StateVector,
    ControlVector,
    MavlinkInterface,
    ControlMapping,
    StateMapping,
    ControlSource,
    StateTarget,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    mavlink_host: str = "0.0.0.0"
    mavlink_port: int = 14540
    real_time_factor: float = 1.0
    timestep: Optional[float] = None
    model_path: Optional[str] = None
    model_xml: Optional[str] = None
    enable_telemetry: bool = True
    telemetry_interval: float = 0.1
    heartbeat_interval: float = 1.0


class Simulator:
    def __init__(self,
                 config: Optional[SimulatorConfig] = None,
                 plant: Optional[Plant] = None,
                 mavlink_interface: Optional[MavlinkInterface] = None,
                 control_mapping: Optional[ControlMapping] = None,
                 state_mapping: Optional[StateMapping] = None):
        self._config = config or SimulatorConfig()
        
        self._plant = plant
        self._mavlink_interface = mavlink_interface
        self._control_mapping = control_mapping or ControlMapping()
        self._state_mapping = state_mapping or StateMapping()
        
        self._is_running = False
        self._simulation_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._stats = {
            "steps": 0,
            "controls_received": 0,
            "states_sent": 0,
            "heartbeats_sent": 0,
            "start_time": 0.0,
        }
        
        self._last_telemetry_time: float = 0.0
        self._last_heartbeat_time: float = 0.0
        
        self._on_state_update: Optional[Callable[[StateVector], None]] = None
        self._on_control_received: Optional[Callable[[Dict[ControlSource, List[float]]], None]] = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        if self._plant is None:
            from .implementations import MuJoCoPlant
            self._plant = MuJoCoPlant(
                model_path=self._config.model_path,
                model_xml=self._config.model_xml
            )
        
        if self._mavlink_interface is None:
            from .implementations import MavlinkUDPInterface
            self._mavlink_interface = MavlinkUDPInterface(
                host=self._config.mavlink_host,
                port=self._config.mavlink_port
            )
        
        if len(self._control_mapping.entries) == 0:
            self._control_mapping.create_default_joint_mappings(
                control_names=self._plant.control_names,
                mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS
            )
            logger.info(f"Created default control mappings for {len(self._plant.control_names)} controls")
        
        logger.info("Simulator initialized")
    
    @property
    def plant(self) -> Plant:
        return self._plant
    
    @property
    def mavlink_interface(self) -> MavlinkInterface:
        return self._mavlink_interface
    
    @property
    def control_mapping(self) -> ControlMapping:
        return self._control_mapping
    
    @property
    def state_mapping(self) -> StateMapping:
        return self._state_mapping
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def config(self) -> SimulatorConfig:
        return self._config
    
    def connect(self) -> bool:
        if self._mavlink_interface.is_connected:
            return True
        
        success = self._mavlink_interface.connect()
        if success:
            logger.info("MAVLink interface connected")
        return success
    
    def disconnect(self) -> None:
        self._mavlink_interface.disconnect()
        logger.info("MAVLink interface disconnected")
    
    def start(self) -> None:
        if self._is_running:
            return
        
        if not self._mavlink_interface.is_connected:
            if not self.connect():
                raise RuntimeError("Failed to connect MAVLink interface")
        
        self._mavlink_interface.start()
        
        self._is_running = True
        self._stats["start_time"] = time.time()
        self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._simulation_thread.start()
        
        logger.info("Simulator started")
    
    def stop(self) -> None:
        self._is_running = False
        
        if self._simulation_thread:
            self._simulation_thread.join(timeout=2.0)
            self._simulation_thread = None
        
        self._mavlink_interface.stop()
        
        logger.info("Simulator stopped")
    
    def _simulation_loop(self) -> None:
        timestep = self._config.timestep or self._plant.timestep
        real_timestep = timestep / self._config.real_time_factor
        
        last_time = time.time()
        
        while self._is_running:
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed >= real_timestep:
                self._step()
                last_time = current_time
            
            time.sleep(0.0001)
    
    def _step(self) -> None:
        self._receive_and_apply_controls()
        
        self._plant.step(n_steps=1)
        
        state = self._plant.get_state()
        
        if self._on_state_update:
            self._on_state_update(state)
        
        self._send_state_if_needed(state)
        
        self._send_heartbeat_if_needed()
        
        with self._lock:
            self._stats["steps"] += 1
    
    def _receive_and_apply_controls(self) -> None:
        controls = self._mavlink_interface.receive_controls()
        
        if controls is None:
            return
        
        if self._on_control_received:
            self._on_control_received(controls)
        
        for source, raw_values in controls.items():
            mapped_controls = self._control_mapping.map_controls(source, raw_values)
            
            for index, value in mapped_controls.items():
                self._plant.set_control_by_index(index, value)
        
        with self._lock:
            self._stats["controls_received"] += 1
    
    def _send_state_if_needed(self, state: StateVector) -> None:
        if not self._config.enable_telemetry:
            return
        
        current_time = time.time()
        if current_time - self._last_telemetry_time < self._config.telemetry_interval:
            return
        
        self._last_telemetry_time = current_time
        
        joint_positions = state.joint_positions
        controls = []
        for name in self._plant.control_names:
            pos = joint_positions.get(name, 0.0)
            controls.append(pos)
        
        success = self._mavlink_interface.send_hil_actuator_controls(controls)
        
        if success:
            with self._lock:
                self._stats["states_sent"] += 1
    
    def _send_heartbeat_if_needed(self) -> None:
        current_time = time.time()
        if current_time - self._last_heartbeat_time < self._config.heartbeat_interval:
            return
        
        self._last_heartbeat_time = current_time
        
        success = self._mavlink_interface.send_heartbeat()
        
        if success:
            with self._lock:
                self._stats["heartbeats_sent"] += 1
    
    def reset(self) -> None:
        self._plant.reset()
        
        with self._lock:
            self._stats = {
                "steps": 0,
                "controls_received": 0,
                "states_sent": 0,
                "heartbeats_sent": 0,
                "start_time": time.time(),
            }
        
        logger.info("Simulator reset")
    
    def get_state(self) -> StateVector:
        return self._plant.get_state()
    
    def set_control(self, control: ControlVector) -> None:
        self._plant.set_control(control)
    
    def set_control_by_index(self, index: int, value: float) -> None:
        self._plant.set_control_by_index(index, value)
    
    def set_control_by_name(self, name: str, value: float) -> None:
        self._plant.set_control_by_name(name, value)
    
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._stats["start_time"] if self._stats["start_time"] > 0 else 0.0
            return {
                "steps": self._stats["steps"],
                "controls_received": self._stats["controls_received"],
                "states_sent": self._stats["states_sent"],
                "heartbeats_sent": self._stats["heartbeats_sent"],
                "elapsed_time": elapsed,
                "steps_per_second": self._stats["steps"] / elapsed if elapsed > 0 else 0.0,
                "real_time_factor": self._config.real_time_factor,
                "is_running": self._is_running,
                "is_connected": self._mavlink_interface.is_connected,
            }
    
    def set_on_state_update(self, callback: Callable[[StateVector], None]) -> None:
        self._on_state_update = callback
    
    def set_on_control_received(self, callback: Callable[[Dict[ControlSource, List[float]]], None]) -> None:
        self._on_control_received = callback
    
    def add_control_mapping(self,
                             mavlink_source: ControlSource,
                             mavlink_index: int,
                             plant_control_name: str,
                             plant_control_index: int,
                             scale: float = 1.0,
                             offset: float = 0.0,
                             range_min: float = -float('inf'),
                             range_max: float = float('inf')) -> None:
        self._control_mapping.add_mapping(
            mavlink_source=mavlink_source,
            mavlink_index=mavlink_index,
            plant_control_name=plant_control_name,
            plant_control_index=plant_control_index,
            scale=scale,
            offset=offset,
            range_min=range_min,
            range_max=range_max
        )
    
    def clear_control_mappings(self) -> None:
        self._control_mapping.clear()
