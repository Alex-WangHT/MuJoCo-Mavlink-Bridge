from typing import Dict, List, Optional, Any
import numpy as np
import logging

from ..core import Plant, StateVector, ControlVector

logger = logging.getLogger(__name__)

DEFAULT_ROBOT_XML = """
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
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.warning("mujoco not installed. Using mock implementation.")


class MuJoCoPlant(Plant):
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_xml: Optional[str] = None):
        self._model_path = model_path
        self._model_xml = model_xml
        
        self._model = None
        self._data = None
        self._is_initialized = False
        
        self._joint_names: List[str] = []
        self._joint_to_actuator: Dict[str, int] = {}
        self._actuator_to_joint: Dict[int, str] = {}
        self._body_names: List[str] = []
        self._sensor_names: List[str] = []
        
        self._control_dim = 0
        self._control_names: List[str] = []
        self._timestep = 0.01
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        if MUJOCO_AVAILABLE:
            self._initialize_mujoco()
        else:
            self._initialize_mock()
    
    def _initialize_mujoco(self) -> None:
        try:
            if self._model_path:
                logger.info(f"Loading MuJoCo model from: {self._model_path}")
                self._model = mujoco.MjModel.from_xml_path(self._model_path)
            else:
                xml = self._model_xml or DEFAULT_ROBOT_XML
                logger.info("Creating default MuJoCo model")
                self._model = mujoco.MjModel.from_xml_string(xml)
            
            self._data = mujoco.MjData(self._model)
            self._timestep = float(self._model.opt.timestep)
            self._extract_model_info()
            self._is_initialized = True
            logger.info(f"MuJoCo model initialized: {self._control_dim} controls, {len(self._joint_names)} joints")
            
        except Exception as e:
            logger.error(f"Failed to initialize MuJoCo: {e}")
            self._initialize_mock()
    
    def _initialize_mock(self) -> None:
        logger.info("Initializing mock MuJoCo plant")
        
        self._joint_names = ["joint1", "joint2"]
        self._body_names = ["base", "link2", "end_effector"]
        self._sensor_names = ["joint1_pos", "joint2_pos", "joint1_vel", "joint2_vel"]
        
        self._control_dim = 2
        self._control_names = ["joint1", "joint2"]
        
        self._mock_qpos = np.zeros(2)
        self._mock_qvel = np.zeros(2)
        self._mock_ctrl = np.zeros(2)
        self._mock_time = 0.0
        
        self._mock_body_positions = {
            "base": np.zeros(3),
            "link2": np.array([0.0, 0.0, 0.3]),
            "end_effector": np.array([0.0, 0.0, 0.55])
        }
        
        self._is_initialized = True
        logger.info(f"Mock plant initialized: {self._control_dim} controls")
    
    def _extract_model_info(self) -> None:
        if not MUJOCO_AVAILABLE or not self._model:
            return
        
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._joint_names.append(name)
        
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._control_names.append(name)
                
                for joint_name in self._joint_names:
                    if joint_name in name:
                        self._joint_to_actuator[joint_name] = i
                        self._actuator_to_joint[i] = joint_name
                        break
        
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self._body_names.append(name)
        
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self._sensor_names.append(name)
        
        self._control_dim = self._model.nu
        logger.info(f"Model info: {self._control_dim} actuators, {len(self._joint_names)} joints")
    
    @property
    def control_dim(self) -> int:
        return self._control_dim
    
    @property
    def control_names(self) -> List[str]:
        return self._control_names.copy()
    
    @property
    def joint_names(self) -> List[str]:
        return self._joint_names.copy()
    
    @property
    def body_names(self) -> List[str]:
        return self._body_names.copy()
    
    @property
    def sensor_names(self) -> List[str]:
        return self._sensor_names.copy()
    
    @property
    def timestep(self) -> float:
        return self._timestep
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def set_control(self, control: ControlVector) -> None:
        if not self._is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self._data is not None:
            for i in range(min(len(control), len(self._data.ctrl))):
                self._data.ctrl[i] = control.values[i]
        else:
            for i in range(min(len(control), len(self._mock_ctrl))):
                self._mock_ctrl[i] = control.values[i]
    
    def set_control_by_index(self, index: int, value: float) -> None:
        if not self._is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self._data is not None:
            if 0 <= index < len(self._data.ctrl):
                self._data.ctrl[index] = value
        else:
            if 0 <= index < len(self._mock_ctrl):
                self._mock_ctrl[index] = value
    
    def set_control_by_name(self, name: str, value: float) -> None:
        if not self._is_initialized:
            return
        
        if name in self._control_names:
            index = self._control_names.index(name)
            self.set_control_by_index(index, value)
        elif name in self._joint_to_actuator:
            index = self._joint_to_actuator[name]
            self.set_control_by_index(index, value)
    
    def get_control(self) -> ControlVector:
        if not self._is_initialized:
            return ControlVector(values=np.array([]))
        
        if MUJOCO_AVAILABLE and self._data is not None:
            values = self._data.ctrl.copy()
        else:
            values = self._mock_ctrl.copy()
        
        return ControlVector(values=values, names=self._control_names)
    
    def get_state(self) -> StateVector:
        state = StateVector()
        
        if MUJOCO_AVAILABLE and self._data is not None:
            state.time = float(self._data.time)
            
            for i, joint_name in enumerate(self._joint_names):
                jnt_idx = i
                qpos_idx = self._model.jnt_qposadr[jnt_idx] if jnt_idx < len(self._model.jnt_qposadr) else 0
                qvel_idx = self._model.jnt_dofadr[jnt_idx] if jnt_idx < len(self._model.jnt_dofadr) else 0
                
                if qpos_idx < len(self._data.qpos):
                    state.joint_positions[joint_name] = float(self._data.qpos[qpos_idx])
                if qvel_idx < len(self._data.qvel):
                    state.joint_velocities[joint_name] = float(self._data.qvel[qvel_idx])
            
            for i, body_name in enumerate(self._body_names):
                body_idx = i
                if body_idx < len(self._data.xpos):
                    state.body_positions[body_name] = self._data.xpos[body_idx].copy()
                if body_idx < len(self._data.cvel):
                    vel = np.zeros(6)
                    vel[:3] = self._data.cvel[body_idx, 3:]
                    vel[3:] = self._data.cvel[body_idx, :3]
                    state.body_velocities[body_name] = vel
            
            for i, sensor_name in enumerate(self._sensor_names):
                sensor_idx = i
                if sensor_idx < len(self._model.sensor_adr):
                    start = self._model.sensor_adr[sensor_idx]
                    dim = self._model.sensor_dim[sensor_idx]
                    if start + dim <= len(self._data.sensordata):
                        state.sensor_data[sensor_name] = self._data.sensordata[start:start+dim].copy()
        
        else:
            state.time = self._mock_time
            
            for i, joint_name in enumerate(self._joint_names):
                if i < len(self._mock_qpos):
                    state.joint_positions[joint_name] = float(self._mock_qpos[i])
                if i < len(self._mock_qvel):
                    state.joint_velocities[joint_name] = float(self._mock_qvel[i])
            
            state.body_positions = {k: v.copy() for k, v in self._mock_body_positions.items()}
            for name in self._body_names:
                state.body_velocities[name] = np.zeros(6)
            
            for i, sensor_name in enumerate(self._sensor_names):
                if "pos" in sensor_name.lower():
                    joint_idx = 0 if "1" in sensor_name else 1
                    state.sensor_data[sensor_name] = np.array([self._mock_qpos[joint_idx]])
                elif "vel" in sensor_name.lower():
                    joint_idx = 0 if "1" in sensor_name else 1
                    state.sensor_data[sensor_name] = np.array([self._mock_qvel[joint_idx]])
        
        return state
    
    def step(self, n_steps: int = 1) -> None:
        if not self._is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self._model and self._data:
            for _ in range(n_steps):
                mujoco.mj_step(self._model, self._data)
        else:
            for _ in range(n_steps):
                self._mock_step()
    
    def _mock_step(self) -> None:
        dt = self._timestep
        self._mock_time += dt
        
        for i in range(len(self._mock_ctrl)):
            torque = self._mock_ctrl[i]
            self._mock_qvel[i] += torque * dt
            self._mock_qpos[i] += self._mock_qvel[i] * dt
            
            damping = 0.99
            self._mock_qvel[i] *= damping
    
    def reset(self) -> None:
        if not self._is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self._model and self._data:
            mujoco.mj_resetData(self._model, self._data)
        else:
            self._mock_qpos = np.zeros(len(self._mock_qpos))
            self._mock_qvel = np.zeros(len(self._mock_qvel))
            self._mock_ctrl = np.zeros(len(self._mock_ctrl))
            self._mock_time = 0.0
        
        logger.info("Plant reset")
    
    @property
    def mj_model(self):
        return self._model
    
    @property
    def mj_data(self):
        return self._data
