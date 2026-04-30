import threading
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

from .mavlink_bridge import MavlinkBridge, MavlinkMessage, PYMAVLINK_AVAILABLE
from .mujoco_controller import MuJoCoController, ControlMode, RobotState, JointState


@dataclass
class ControlMapping:
    mavlink_index: int
    joint_name: str
    control_type: str
    scale: float = 1.0
    offset: float = 0.0


class Simulator:
    def __init__(self,
                 mavlink_connection: str = "udp:0.0.0.0:14540",
                 mujoco_model_path: Optional[str] = None,
                 real_time_factor: float = 1.0):
        self.mavlink_connection = mavlink_connection
        self.mujoco_model_path = mujoco_model_path
        self.real_time_factor = real_time_factor
        
        self.mavlink_bridge: Optional[MavlinkBridge] = None
        self.mujoco_controller: Optional[MuJoCoController] = None
        
        self.is_running = False
        self.simulation_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.control_mappings: Dict[int, ControlMapping] = {}
        self.joint_to_mavlink: Dict[str, int] = {}
        
        self.on_state_update: Optional[Callable[[RobotState], None]] = None
        self.on_message_received: Optional[Callable[[MavlinkMessage], None]] = None
        
        self._stats = {
            "steps": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "start_time": 0.0
        }
        
        self._initialize()
    
    def _initialize(self):
        self.mujoco_controller = MuJoCoController(self.mujoco_model_path)
        self.mavlink_bridge = MavlinkBridge(
            connection_string=self.mavlink_connection
        )
        
        self._setup_default_mappings()
        self._setup_message_handlers()
        
        print("Simulator initialized")
    
    def _setup_default_mappings(self):
        joint_names = self.mujoco_controller.joint_names
        
        for i, joint_name in enumerate(joint_names):
            mapping = ControlMapping(
                mavlink_index=i,
                joint_name=joint_name,
                control_type=ControlMode.TORQUE,
                scale=1.0,
                offset=0.0
            )
            self.control_mappings[i] = mapping
            self.joint_to_mavlink[joint_name] = i
            self.mujoco_controller.set_control_mode(joint_name, ControlMode.TORQUE)
        
        print(f"Default mappings: {len(self.control_mappings)} joints")
    
    def _setup_message_handlers(self):
        if self.mavlink_bridge:
            self.mavlink_bridge.register_handler(
                "HIL_ACTUATOR_CONTROLS",
                self._handle_hil_actuator_controls
            )
            self.mavlink_bridge.register_handler(
                "MAVLINK1",
                self._handle_raw_mavlink1
            )
            self.mavlink_bridge.register_handler(
                "MAVLINK2",
                self._handle_raw_mavlink2
            )
    
    def _handle_hil_actuator_controls(self, msg: MavlinkMessage):
        with self.lock:
            self._stats["messages_received"] += 1
        
        if self.on_message_received:
            self.on_message_received(msg)
        
        controls = msg.data.get("controls", [])
        
        for i, control_value in enumerate(controls):
            if i in self.control_mappings:
                mapping = self.control_mappings[i]
                value = control_value * mapping.scale + mapping.offset
                
                joint_name = mapping.joint_name
                
                if mapping.control_type == ControlMode.POSITION:
                    self.mujoco_controller.set_joint_position(joint_name, value)
                elif mapping.control_type == ControlMode.VELOCITY:
                    self.mujoco_controller.set_joint_velocity(joint_name, value)
                elif mapping.control_type == ControlMode.TORQUE:
                    self.mujoco_controller.set_joint_torque(joint_name, value)
    
    def _handle_raw_mavlink1(self, msg: MavlinkMessage):
        with self.lock:
            self._stats["messages_received"] += 1
        
        if self.on_message_received:
            self.on_message_received(msg)
        
        raw_data = msg.data.get("raw", b"")
        if len(raw_data) >= 8:
            pass
    
    def _handle_raw_mavlink2(self, msg: MavlinkMessage):
        with self.lock:
            self._stats["messages_received"] += 1
        
        if self.on_message_received:
            self.on_message_received(msg)
    
    def add_control_mapping(self,
                            mavlink_index: int,
                            joint_name: str,
                            control_type: str = ControlMode.TORQUE,
                            scale: float = 1.0,
                            offset: float = 0.0):
        if joint_name not in self.mujoco_controller.joint_indices:
            raise ValueError(f"Joint {joint_name} not found")
        
        mapping = ControlMapping(
            mavlink_index=mavlink_index,
            joint_name=joint_name,
            control_type=control_type,
            scale=scale,
            offset=offset
        )
        
        self.control_mappings[mavlink_index] = mapping
        self.joint_to_mavlink[joint_name] = mavlink_index
        self.mujoco_controller.set_control_mode(joint_name, control_type)
    
    def set_pid_gains(self, joint_name: str, kp: float, ki: float, kd: float):
        self.mujoco_controller.set_pid_gains(joint_name, kp, ki, kd)
    
    def _simulation_loop(self):
        timestep = self.mujoco_controller.get_timestep()
        real_timestep = timestep / self.real_time_factor
        
        last_time = time.time()
        self._stats["start_time"] = last_time
        
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed >= real_timestep:
                self.mujoco_controller.step()
                
                with self.lock:
                    self._stats["steps"] += 1
                
                robot_state = self.mujoco_controller.get_robot_state()
                
                if self.on_state_update:
                    self.on_state_update(robot_state)
                
                self._send_state_to_mavlink(robot_state)
                
                last_time = current_time
            
            time.sleep(0.0001)
    
    def _send_state_to_mavlink(self, state: RobotState):
        if not self.mavlink_bridge or not self.mavlink_bridge.is_connected:
            return
        
        joint_states = state.joint_states
        controls = []
        
        for i in range(16):
            if i in self.control_mappings:
                mapping = self.control_mappings[i]
                joint_name = mapping.joint_name
                
                if joint_name in joint_states:
                    js = joint_states[joint_name]
                    value = (js.qpos - mapping.offset) / mapping.scale if mapping.scale != 0 else 0.0
                    controls.append(value)
                else:
                    controls.append(0.0)
            else:
                controls.append(0.0)
        
        success = self.mavlink_bridge.send_hil_actuator_controls(controls)
        if success:
            with self.lock:
                self._stats["messages_sent"] += 1
        
        quaternion = [1.0, 0.0, 0.0, 0.0]
        self.mavlink_bridge.send_hil_state_quaternion(
            attitude_quaternion=quaternion,
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
            lat=0,
            lon=0,
            alt=0,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            xacc=0.0,
            yacc=0.0,
            zacc=0.0
        )
    
    def start(self):
        if self.is_running:
            return
        
        if self.mavlink_bridge:
            self.mavlink_bridge.start()
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        print("Simulator started")
    
    def stop(self):
        self.is_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        
        if self.mavlink_bridge:
            self.mavlink_bridge.stop()
        
        print("Simulator stopped")
    
    def reset(self):
        self.mujoco_controller.reset()
        
        with self.lock:
            self._stats["steps"] = 0
            self._stats["messages_received"] = 0
            self._stats["messages_sent"] = 0
            self._stats["start_time"] = time.time()
    
    def get_robot_state(self) -> RobotState:
        return self.mujoco_controller.get_robot_state()
    
    def get_joint_state(self, joint_name: str) -> Optional[JointState]:
        return self.mujoco_controller.get_joint_state(joint_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self._stats["start_time"] if self._stats["start_time"] > 0 else 0.0
            steps_per_second = self._stats["steps"] / elapsed if elapsed > 0 else 0.0
            
            return {
                "steps": self._stats["steps"],
                "messages_received": self._stats["messages_received"],
                "messages_sent": self._stats["messages_sent"],
                "elapsed_time": elapsed,
                "steps_per_second": steps_per_second,
                "real_time_factor": self.real_time_factor,
                "is_running": self.is_running
            }
    
    def send_heartbeat(self):
        if self.mavlink_bridge:
            self.mavlink_bridge.send_heartbeat()
