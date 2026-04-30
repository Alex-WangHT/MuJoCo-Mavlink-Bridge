import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np

from ..base import (
    BaseRobotModel,
    BaseController,
    BaseMavlinkHandler,
    MavlinkMessage,
    MessageType,
    ControlMode,
    ControlCommand,
    RobotState,
)
from ..mavlink import (
    MavlinkUDPBridge,
    ControlMapper,
    ControlTargetType,
    MavlinkControl,
    MappedControlResult,
)
from ..robot import (
    MuJoCoModel,
    MuJoCoController,
)
from .config import SimulatorConfig, MavlinkConfig


logger = logging.getLogger(__name__)


@dataclass
class SimulatorStats:
    steps: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    start_time: float = 0.0
    elapsed_time: float = 0.0
    
    @property
    def steps_per_second(self) -> float:
        if self.elapsed_time > 0:
            return self.steps / self.elapsed_time
        return 0.0


class Simulator:
    def __init__(self,
                 config: Optional[SimulatorConfig] = None,
                 model: Optional[BaseRobotModel] = None,
                 controller: Optional[BaseController] = None,
                 mavlink_bridge: Optional[BaseMavlinkHandler] = None,
                 control_mapper: Optional[ControlMapper] = None):
        self._config = config or SimulatorConfig()
        
        self._model = model
        self._controller = controller
        self._mavlink_bridge = mavlink_bridge
        self._control_mapper = control_mapper or ControlMapper()
        
        self._is_running = False
        self._simulation_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._stats = SimulatorStats()
        
        self._on_state_update: Optional[Callable[[RobotState], None]] = None
        self._on_message_received: Optional[Callable[[MavlinkMessage], None]] = None
        self._on_control_applied: Optional[Callable[[ControlCommand], None]] = None
        
        self._last_telemetry_time: float = 0.0
        self._last_heartbeat_time: float = 0.0
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        if self._model is None:
            if self._config.model_path:
                self._model = MuJoCoModel(model_path=self._config.model_path)
            elif self._config.model_xml:
                self._model = MuJoCoModel(model_xml=self._config.model_xml)
            else:
                self._model = MuJoCoModel()
        
        if self._controller is None:
            if isinstance(self._model, MuJoCoModel):
                self._controller = MuJoCoController(model=self._model)
            else:
                from ..base import BaseController
                self._controller = BaseController()
        
        if self._mavlink_bridge is None:
            mav_config = self._config.mavlink
            self._mavlink_bridge = MavlinkUDPBridge(
                host=mav_config.host,
                port=mav_config.port,
                source_system=mav_config.source_system,
                source_component=mav_config.source_component,
            )
        
        self._setup_default_mappings()
        self._setup_message_handlers()
        
        logger.info("Simulator components initialized")
    
    def _setup_default_mappings(self) -> None:
        joint_names = self._model.joint_names
        
        self._control_mapper.create_default_joint_mappings(
            joint_names=joint_names,
            control_type=ControlTargetType.JOINT_TORQUE,
            start_index=0
        )
        
        logger.info(f"Default control mappings created for %d joints", len(joint_names))
    
    def _setup_message_handlers(self) -> None:
        if self._mavlink_bridge:
            self._mavlink_bridge.register_handler(
                MessageType.HIL_ACTUATOR_CONTROLS,
                self._handle_hil_actuator_controls
            )
            
            self._mavlink_bridge.register_handler(
                MessageType.HEARTBEAT,
                self._handle_heartbeat
            )
    
    def _handle_hil_actuator_controls(self, msg: MavlinkMessage) -> None:
        with self._lock:
            self._stats.messages_received += 1
        
        if self._on_message_received:
            self._on_message_received(msg)
        
        controls = msg.get("controls", [])
        
        mapped_controls = self._control_mapper.map_controls(controls)
        
        command = ControlCommand(timestamp=time.time())
        
        for result in mapped_controls:
            if result.target_type in [
                ControlTargetType.JOINT_POSITION,
                ControlTargetType.JOINT_VELOCITY,
                ControlTargetType.JOINT_TORQUE,
                ControlTargetType.CUSTOM,
            ]:
                command.targets[result.target_name] = ControlTarget(
                    joint_name=result.target_name,
                    mode=self._control_mapper.to_control_mode(result.target_type),
                    value=result.value,
                )
            
            if result.target_type in [ControlTargetType.BODY_FORCE, ControlTargetType.BODY_TORQUE]:
                force = np.zeros(6)
                if result.target_type == ControlTargetType.BODY_FORCE:
                    force[0] = result.value
                else:
                    force[3] = result.value
                command.forces[result.target_name] = force
        
        if self._controller:
            self._controller.apply_control(command)
        
        if self._on_control_applied:
            self._on_control_applied(command)
    
    def _handle_heartbeat(self, msg: MavlinkMessage) -> None:
        with self._lock:
            self._stats.messages_received += 1
        
        if self._on_message_received:
            self._on_message_received(msg)
        
        logger.debug("Received heartbeat from sysid=%d, compid=%d", msg.sysid, msg.compid)
    
    def add_control_mapping(self,
                           mavlink_index: int,
                           target_name: str,
                           target_type: ControlTargetType,
                           scale: float = 1.0,
                           offset: float = 0.0,
                           range_min: float = -np.inf,
                           range_max: float = np.inf,
                           **kwargs) -> None:
        self._control_mapper.add_mapping(
            mavlink_index=mavlink_index,
            target_name=target_name,
            target_type=target_type,
            scale=scale,
            offset=offset,
            range_min=range_min,
            range_max=range_max,
            **kwargs
        )
    
    def remove_control_mapping(self, mavlink_index: int) -> bool:
        return self._control_mapper.remove_mapping(mavlink_index)
    
    def clear_control_mappings(self) -> None:
        self._control_mapper.clear_mappings()
    
    def connect(self) -> bool:
        if self._mavlink_bridge:
            return self._mavlink_bridge.connect()
        return False
    
    def disconnect(self) -> None:
        if self._mavlink_bridge:
            self._mavlink_bridge.disconnect()
    
    def start(self) -> None:
        if self._is_running:
            return
        
        if not self._mavlink_bridge or not self._mavlink_bridge.is_connected:
            self.connect()
        
        if self._mavlink_bridge:
            self._mavlink_bridge.start()
        
        self._is_running = True
        self._stats.start_time = time.time()
        self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._simulation_thread.start()
        
        logger.info("Simulator started")
    
    def stop(self) -> None:
        self._is_running = False
        
        if self._simulation_thread:
            self._simulation_thread.join(timeout=2.0)
            self._simulation_thread = None
        
        if self._mavlink_bridge:
            self._mavlink_bridge.stop()
        
        logger.info("Simulator stopped")
    
    def _simulation_loop(self) -> None:
        timestep = self._config.simulation.timestep or self._model.get_timestep()
        real_timestep = timestep / self._config.simulation.real_time_factor
        
        last_time = time.time()
        
        while self._is_running:
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed >= real_timestep:
                self._step_simulation()
                last_time = current_time
            
            time.sleep(0.0001)
    
    def _step_simulation(self) -> None:
        if self._controller:
            self._controller.step(n_steps=1)
        
        with self._lock:
            self._stats.steps += 1
        
        state = self._model.get_state()
        
        if self._on_state_update:
            self._on_state_update(state)
        
        self._send_telemetry(state)
        
        self._send_heartbeat_if_needed()
    
    def _send_telemetry(self, state: RobotState) -> None:
        if not self._config.enable_telemetry:
            return
        
        current_time = time.time()
        if current_time - self._last_telemetry_time < self._config.telemetry_interval:
            return
        
        self._last_telemetry_time = current_time
        
        if self._mavlink_bridge and self._mavlink_bridge.is_connected:
            controls = []
            for i in range(16):
                mapping = self._control_mapper.get_mapping(i)
                if mapping and mapping.target_name in state.joint_positions:
                    pos = state.joint_positions[mapping.target_name]
                    controls.append((pos - mapping.offset) / mapping.scale if mapping.scale != 0 else 0.0)
                else:
                    controls.append(0.0)
            
            self._mavlink_bridge.send_hil_actuator_controls(controls)
            
            with self._lock:
                self._stats.messages_sent += 1
    
    def _send_heartbeat_if_needed(self) -> None:
        current_time = time.time()
        if current_time - self._last_heartbeat_time < self._config.mavlink.heartbeat_interval:
            return
        
        self._last_heartbeat_time = current_time
        
        if self._mavlink_bridge and self._mavlink_bridge.is_connected:
            self._mavlink_bridge.send_heartbeat()
            with self._lock:
                self._stats.messages_sent += 1
    
    def reset(self) -> None:
        if self._model:
            self._model.reset()
        if self._controller:
            self._controller.reset()
        
        with self._lock:
            self._stats = SimulatorStats()
            self._stats.start_time = time.time()
        
        logger.info("Simulator reset")
    
    def get_state(self) -> RobotState:
        return self._model.get_state()
    
    def get_joint_state(self, joint_name: str) -> Optional[Dict[str, float]]:
        if self._controller:
            return self._controller.get_current_state(joint_name)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._stats.start_time if self._stats.start_time > 0 else 0.0
            stats_dict = {
                "steps": self._stats.steps,
                "messages_received": self._stats.messages_received,
                "messages_sent": self._stats.messages_sent,
                "elapsed_time": elapsed,
                "steps_per_second": self._stats.steps / elapsed if elapsed > 0 else 0.0,
                "real_time_factor": self._config.simulation.real_time_factor,
                "is_running": self._is_running,
                "is_connected": self._mavlink_bridge.is_connected if self._mavlink_bridge else False,
            }
            return stats_dict
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def model(self) -> BaseRobotModel:
        return self._model
    
    @property
    def controller(self) -> BaseController:
        return self._controller
    
    @property
    def mavlink_bridge(self) -> BaseMavlinkHandler:
        return self._mavlink_bridge
    
    @property
    def control_mapper(self) -> ControlMapper:
        return self._control_mapper
    
    @property
    def config(self) -> SimulatorConfig:
        return self._config
    
    def set_on_state_update(self, callback: Callable[[RobotState], None]) -> None:
        self._on_state_update = callback
    
    def set_on_message_received(self, callback: Callable[[MavlinkMessage], None]) -> None:
        self._on_message_received = callback
    
    def set_on_control_applied(self, callback: Callable[[ControlCommand], None]) -> None:
        self._on_control_applied = callback
