import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import time

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco not installed. Using mock implementation.")


@dataclass
class JointState:
    name: str
    qpos: float
    qvel: float
    qacc: float
    qfrc_applied: float


@dataclass
class RobotState:
    time: float
    joint_states: Dict[str, JointState]
    body_positions: Dict[str, np.ndarray]
    body_velocities: Dict[str, np.ndarray]
    sensor_data: Dict[str, Any]


class ControlMode:
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    PID = "pid"


class PIDController:
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
    
    def compute(self, target: float, current: float) -> float:
        error = target - current
        now = time.time()
        dt = now - self.last_time
        
        if dt > 0:
            derivative = (error - self.last_error) / dt
            self.integral += error * dt
        else:
            derivative = 0.0
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.last_error = error
        self.last_time = now
        
        return output
    
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()


class MuJoCoController:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.data = None
        self.is_initialized = False
        
        self.joint_names: List[str] = []
        self.joint_indices: Dict[str, int] = {}
        self.actuator_names: List[str] = []
        self.actuator_indices: Dict[str, int] = {}
        self.body_names: List[str] = []
        self.body_indices: Dict[str, int] = {}
        self.sensor_names: List[str] = []
        self.sensor_indices: Dict[str, int] = {}
        
        self.control_modes: Dict[str, str] = {}
        self.pid_controllers: Dict[str, PIDController] = {}
        self.target_positions: Dict[str, float] = {}
        self.target_velocities: Dict[str, float] = {}
        self.target_torques: Dict[str, float] = {}
        
        self._setup_mujoco()
    
    def _setup_mujoco(self):
        if MUJOCO_AVAILABLE:
            if self.model_path:
                try:
                    self.model = mujoco.MjModel.from_xml_path(self.model_path)
                    self.data = mujoco.MjData(self.model)
                    self.is_initialized = True
                    self._extract_model_info()
                    print(f"Loaded MuJoCo model from: {self.model_path}")
                except Exception as e:
                    print(f"Failed to load MuJoCo model: {e}")
                    self._create_simple_model()
            else:
                self._create_simple_model()
        else:
            self._create_mock_model()
    
    def _create_simple_model(self):
        if not MUJOCO_AVAILABLE:
            return
        
        xml = """
<mujoco model="simple_robot">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light name="top" pos="0 0 3"/>
    <body name="base" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom name="link1" type="cylinder" size="0.1 0.3" pos="0 0 0.15" rgba="0.8 0.3 0.3 1"/>
      <body name="link2" pos="0 0 0.3">
        <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="link2_geom" type="cylinder" size="0.08 0.25" pos="0 0 0.125" rgba="0.3 0.8 0.3 1"/>
        <body name="end_effector" pos="0 0 0.25">
          <geom name="ee" type="sphere" size="0.05" rgba="0.3 0.3 0.8 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="1.0"/>
    <motor name="motor2" joint="joint2" gear="1.0"/>
  </actuator>
  <sensor>
    <jointpos name="joint1_pos" joint="joint1"/>
    <jointpos name="joint2_pos" joint="joint2"/>
    <jointvel name="joint1_vel" joint="joint1"/>
    <jointvel name="joint2_vel" joint="joint2"/>
  </sensor>
</mujoco>
"""
        try:
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            self.is_initialized = True
            self._extract_model_info()
            print("Created simple MuJoCo model")
        except Exception as e:
            print(f"Failed to create simple MuJoCo model: {e}")
            self._create_mock_model()
    
    def _create_mock_model(self):
        self.is_initialized = True
        self.joint_names = ["joint1", "joint2"]
        self.joint_indices = {"joint1": 0, "joint2": 1}
        self.actuator_names = ["motor1", "motor2"]
        self.actuator_indices = {"motor1": 0, "motor2": 1}
        self.body_names = ["base", "link2", "end_effector"]
        self.body_indices = {"base": 0, "link2": 1, "end_effector": 2}
        self.sensor_names = ["joint1_pos", "joint2_pos", "joint1_vel", "joint2_vel"]
        self.sensor_indices = {"joint1_pos": 0, "joint2_pos": 1, "joint1_vel": 2, "joint2_vel": 3}
        
        self._mock_qpos = np.zeros(2)
        self._mock_qvel = np.zeros(2)
        self._mock_ctrl = np.zeros(2)
        self._mock_time = 0.0
        
        print("Created mock MuJoCo model")
    
    def _extract_model_info(self):
        if not MUJOCO_AVAILABLE or not self.model:
            return
        
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_names.append(name)
                self.joint_indices[name] = i
        
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_names.append(name)
                self.actuator_indices[name] = i
        
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self.body_names.append(name)
                self.body_indices[name] = i
        
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self.sensor_names.append(name)
                self.sensor_indices[name] = i
        
        print(f"Model info: {len(self.joint_names)} joints, {len(self.actuator_names)} actuators, {len(self.sensor_names)} sensors")
    
    def set_control_mode(self, joint_name: str, mode: str):
        if joint_name not in self.joint_indices:
            raise ValueError(f"Joint {joint_name} not found")
        
        self.control_modes[joint_name] = mode
        
        if mode == ControlMode.PID and joint_name not in self.pid_controllers:
            self.pid_controllers[joint_name] = PIDController(kp=10.0, ki=0.1, kd=1.0)
    
    def set_pid_gains(self, joint_name: str, kp: float, ki: float, kd: float):
        if joint_name not in self.pid_controllers:
            self.pid_controllers[joint_name] = PIDController()
        
        pid = self.pid_controllers[joint_name]
        pid.kp = kp
        pid.ki = ki
        pid.kd = kd
    
    def set_joint_position(self, joint_name: str, position: float):
        self.target_positions[joint_name] = position
        if joint_name not in self.control_modes:
            self.set_control_mode(joint_name, ControlMode.POSITION)
    
    def set_joint_velocity(self, joint_name: str, velocity: float):
        self.target_velocities[joint_name] = velocity
        if joint_name not in self.control_modes:
            self.set_control_mode(joint_name, ControlMode.VELOCITY)
    
    def set_joint_torque(self, joint_name: str, torque: float):
        self.target_torques[joint_name] = torque
        if joint_name not in self.control_modes:
            self.set_control_mode(joint_name, ControlMode.TORQUE)
    
    def set_all_joint_positions(self, positions: Dict[str, float]):
        for joint_name, position in positions.items():
            self.set_joint_position(joint_name, position)
    
    def set_all_joint_torques(self, torques: Dict[str, float]):
        for joint_name, torque in torques.items():
            self.set_joint_torque(joint_name, torque)
    
    def apply_force_to_body(self, body_name: str, force: np.ndarray, point: Optional[np.ndarray] = None):
        if not MUJOCO_AVAILABLE or not self.is_initialized:
            return
        
        if body_name not in self.body_indices:
            raise ValueError(f"Body {body_name} not found")
        
        body_idx = self.body_indices[body_name]
        
        if point is None:
            point = np.zeros(3)
        
        if MUJOCO_AVAILABLE and self.data:
            self.data.xfrc_applied[body_idx, :3] = force
            if len(force) >= 6:
                self.data.xfrc_applied[body_idx, 3:] = force[3:]
    
    def _update_control_signals(self):
        if not self.is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self.data:
            for joint_name, mode in self.control_modes.items():
                if joint_name not in self.joint_indices:
                    continue
                
                joint_idx = self.joint_indices[joint_name]
                
                actuator_idx = None
                for act_name, idx in self.actuator_indices.items():
                    if joint_name in act_name:
                        actuator_idx = idx
                        break
                
                if actuator_idx is None:
                    continue
                
                if mode == ControlMode.POSITION and joint_name in self.target_positions:
                    target = self.target_positions[joint_name]
                    self.data.ctrl[actuator_idx] = target
                
                elif mode == ControlMode.VELOCITY and joint_name in self.target_velocities:
                    target = self.target_velocities[joint_name]
                    self.data.ctrl[actuator_idx] = target
                
                elif mode == ControlMode.TORQUE and joint_name in self.target_torques:
                    target = self.target_torques[joint_name]
                    self.data.ctrl[actuator_idx] = target
                
                elif mode == ControlMode.PID:
                    if joint_name in self.target_positions and joint_name in self.pid_controllers:
                        current = self.data.qpos[joint_idx]
                        target = self.target_positions[joint_name]
                        pid = self.pid_controllers[joint_name]
                        torque = pid.compute(target, current)
                        self.data.ctrl[actuator_idx] = torque
        else:
            for joint_name, mode in self.control_modes.items():
                if joint_name not in self.joint_indices:
                    continue
                
                joint_idx = self.joint_indices[joint_name]
                
                if mode == ControlMode.TORQUE and joint_name in self.target_torques:
                    torque = self.target_torques[joint_name]
                    self._mock_qvel[joint_idx] += torque * 0.01
                    self._mock_qpos[joint_idx] += self._mock_qvel[joint_idx] * 0.01
                
                elif mode == ControlMode.POSITION and joint_name in self.target_positions:
                    target = self.target_positions[joint_name]
                    self._mock_qpos[joint_idx] += (target - self._mock_qpos[joint_idx]) * 0.1
                
                elif mode == ControlMode.VELOCITY and joint_name in self.target_velocities:
                    target = self.target_velocities[joint_name]
                    self._mock_qvel[joint_idx] += (target - self._mock_qvel[joint_idx]) * 0.1
                    self._mock_qpos[joint_idx] += self._mock_qvel[joint_idx] * 0.01
    
    def step(self, n_steps: int = 1):
        if not self.is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self.model and self.data:
            for _ in range(n_steps):
                self._update_control_signals()
                mujoco.mj_step(self.model, self.data)
        else:
            for _ in range(n_steps):
                self._update_control_signals()
                self._mock_time += 0.01
    
    def get_joint_state(self, joint_name: str) -> Optional[JointState]:
        if joint_name not in self.joint_indices:
            return None
        
        joint_idx = self.joint_indices[joint_name]
        
        if MUJOCO_AVAILABLE and self.data:
            qpos = self.data.qpos[joint_idx] if joint_idx < len(self.data.qpos) else 0.0
            qvel = self.data.qvel[joint_idx] if joint_idx < len(self.data.qvel) else 0.0
            qacc = self.data.qacc[joint_idx] if joint_idx < len(self.data.qacc) else 0.0
            qfrc = self.data.qfrc_applied[joint_idx] if joint_idx < len(self.data.qfrc_applied) else 0.0
        else:
            qpos = self._mock_qpos[joint_idx]
            qvel = self._mock_qvel[joint_idx]
            qacc = 0.0
            qfrc = 0.0
        
        return JointState(
            name=joint_name,
            qpos=float(qpos),
            qvel=float(qvel),
            qacc=float(qacc),
            qfrc_applied=float(qfrc)
        )
    
    def get_all_joint_states(self) -> Dict[str, JointState]:
        states = {}
        for joint_name in self.joint_names:
            state = self.get_joint_state(joint_name)
            if state:
                states[joint_name] = state
        return states
    
    def get_body_position(self, body_name: str) -> Optional[np.ndarray]:
        if body_name not in self.body_indices:
            return None
        
        if MUJOCO_AVAILABLE and self.data:
            body_idx = self.body_indices[body_name]
            return self.data.xpos[body_idx].copy()
        else:
            return np.zeros(3)
    
    def get_body_velocity(self, body_name: str) -> Optional[np.ndarray]:
        if body_name not in self.body_indices:
            return None
        
        if MUJOCO_AVAILABLE and self.data:
            body_idx = self.body_indices[body_name]
            vel = np.zeros(6)
            vel[:3] = self.data.cvel[body_idx, 3:]
            vel[3:] = self.data.cvel[body_idx, :3]
            return vel
        else:
            return np.zeros(6)
    
    def get_sensor_data(self, sensor_name: str) -> Optional[Any]:
        if sensor_name not in self.sensor_indices:
            return None
        
        if MUJOCO_AVAILABLE and self.data:
            sensor_idx = self.sensor_indices[sensor_name]
            sensor = self.model.sensor(sensor_idx)
            start = sensor.adr
            end = start + sensor.dim
            return self.data.sensordata[start:end].copy()
        else:
            return np.array([0.0])
    
    def get_all_sensor_data(self) -> Dict[str, Any]:
        data = {}
        for sensor_name in self.sensor_names:
            value = self.get_sensor_data(sensor_name)
            if value is not None:
                data[sensor_name] = value
        return data
    
    def get_robot_state(self) -> RobotState:
        if MUJOCO_AVAILABLE and self.data:
            sim_time = self.data.time
        else:
            sim_time = self._mock_time
        
        joint_states = self.get_all_joint_states()
        
        body_positions = {}
        for body_name in self.body_names:
            pos = self.get_body_position(body_name)
            if pos is not None:
                body_positions[body_name] = pos
        
        body_velocities = {}
        for body_name in self.body_names:
            vel = self.get_body_velocity(body_name)
            if vel is not None:
                body_velocities[body_name] = vel
        
        sensor_data = self.get_all_sensor_data()
        
        return RobotState(
            time=sim_time,
            joint_states=joint_states,
            body_positions=body_positions,
            body_velocities=body_velocities,
            sensor_data=sensor_data
        )
    
    def reset(self):
        if not self.is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self.model and self.data:
            mujoco.mj_resetData(self.model, self.data)
        
        for pid in self.pid_controllers.values():
            pid.reset()
        
        self.target_positions.clear()
        self.target_velocities.clear()
        self.target_torques.clear()
        
        if not MUJOCO_AVAILABLE:
            self._mock_qpos = np.zeros(2)
            self._mock_qvel = np.zeros(2)
            self._mock_ctrl = np.zeros(2)
            self._mock_time = 0.0
    
    def get_timestep(self) -> float:
        if MUJOCO_AVAILABLE and self.model:
            return float(self.model.opt.timestep)
        return 0.01
