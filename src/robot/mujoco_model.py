from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

from ..base import (
    BaseRobotModel,
    JointInfo,
    BodyInfo,
    SensorInfo,
    RobotState,
)

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logging.warning("mujoco not installed. Using mock implementation.")


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


class MuJoCoModel(BaseRobotModel):
    def __init__(self, model_path: Optional[str] = None, model_xml: Optional[str] = None):
        self._model = None
        self._data = None
        self._model_xml = model_xml
        
        super().__init__(model_path)
    
    def load_model(self, model_path: str) -> bool:
        if not MUJOCO_AVAILABLE:
            logger.warning("MuJoCo not available, using mock model")
            return self._create_mock_model()
        
        try:
            self._clear_model()
            logger.info(f"Loading MuJoCo model from: {model_path}")
            self._model = mujoco.MjModel.from_xml_path(model_path)
            self._data = mujoco.MjData(self._model)
            self._extract_model_info()
            self._is_initialized = True
            logger.info(f"Model loaded: {len(self._joint_names)} joints, {len(self._body_names)} bodies")
            return True
        except Exception as e:
            logger.error(f"Failed to load MuJoCo model: {e}")
            return self._create_default_model()
    
    def _create_default_model(self) -> bool:
        if not MUJOCO_AVAILABLE:
            return self._create_mock_model()
        
        try:
            self._clear_model()
            xml = self._model_xml or DEFAULT_ROBOT_XML
            logger.info("Creating default MuJoCo model")
            self._model = mujoco.MjModel.from_xml_string(xml)
            self._data = mujoco.MjData(self._model)
            self._extract_model_info()
            self._is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to create default MuJoCo model: {e}")
            return self._create_mock_model()
    
    def _create_mock_model(self) -> bool:
        self._clear_model()
        logger.info("Creating mock robot model")
        
        self._add_joint(JointInfo(
            name="joint1",
            index=0,
            joint_type="hinge",
            qpos_idx=0,
            qvel_idx=0,
            range=(-3.14, 3.14)
        ))
        
        self._add_joint(JointInfo(
            name="joint2",
            index=1,
            joint_type="hinge",
            qpos_idx=1,
            qvel_idx=1,
            range=(-1.57, 1.57)
        ))
        
        self._add_body(BodyInfo(
            name="base",
            index=0,
            parent_name=None,
            body_id=0
        ))
        
        self._add_body(BodyInfo(
            name="link2",
            index=1,
            parent_name="base",
            body_id=1
        ))
        
        self._add_body(BodyInfo(
            name="end_effector",
            index=2,
            parent_name="link2",
            body_id=2
        ))
        
        self._add_sensor(SensorInfo(
            name="joint1_pos",
            index=0,
            sensor_type="jointpos",
            data_dim=1,
            data_start=0
        ))
        
        self._add_sensor(SensorInfo(
            name="joint2_pos",
            index=1,
            sensor_type="jointpos",
            data_dim=1,
            data_start=1
        ))
        
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
        return True
    
    def _extract_model_info(self) -> None:
        if not MUJOCO_AVAILABLE or not self._model:
            return
        
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                joint_type = "unknown"
                jnt_type = self._model.jnt_type[i]
                if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                    joint_type = "hinge"
                elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    joint_type = "slide"
                elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                    joint_type = "ball"
                elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                    joint_type = "free"
                
                qpos_idx = self._model.jnt_qposadr[i]
                qvel_idx = self._model.jnt_dofadr[i]
                
                jnt_range = self._model.jnt_range[i] if i < len(self._model.jnt_range) else [-np.inf, np.inf]
                
                self._add_joint(JointInfo(
                    name=name,
                    index=i,
                    joint_type=joint_type,
                    qpos_idx=qpos_idx,
                    qvel_idx=qvel_idx,
                    range=(float(jnt_range[0]), float(jnt_range[1]))
                ))
        
        for i in range(self._model.nu):
            pass
        
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                parent_id = self._model.body_parentid[i]
                parent_name = None
                if parent_id >= 0:
                    parent_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                
                self._add_body(BodyInfo(
                    name=name,
                    index=i,
                    parent_name=parent_name,
                    body_id=i
                ))
        
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                sensor_type = "unknown"
                if i < len(self._model.sensor_type):
                    sens_type = self._model.sensor_type[i]
                    if sens_type == mujoco.mjtSensor.mjSENS_JOINTPOS:
                        sensor_type = "jointpos"
                    elif sens_type == mujoco.mjtSensor.mjSENS_JOINTVEL:
                        sensor_type = "jointvel"
                    elif sens_type == mujoco.mjtSensor.mjSENS_TOUCH:
                        sensor_type = "touch"
                    elif sens_type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
                        sensor_type = "accelerometer"
                    elif sens_type == mujoco.mjtSensor.mjSENS_GYRO:
                        sensor_type = "gyro"
                
                data_start = self._model.sensor_adr[i] if i < len(self._model.sensor_adr) else 0
                data_dim = self._model.sensor_dim[i] if i < len(self._model.sensor_dim) else 1
                
                self._add_sensor(SensorInfo(
                    name=name,
                    index=i,
                    sensor_type=sensor_type,
                    data_dim=data_dim,
                    data_start=data_start
                ))
        
        logger.info(f"Extracted model info: {len(self._joint_names)} joints, {len(self._body_names)} bodies, {len(self._sensor_names)} sensors")
    
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
        dt = 0.01
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
    
    def get_timestep(self) -> float:
        if MUJOCO_AVAILABLE and self._model:
            return float(self._model.opt.timestep)
        return 0.01
    
    def get_state(self) -> RobotState:
        joint_positions = {}
        joint_velocities = {}
        joint_accelerations = {}
        joint_torques = {}
        
        if MUJOCO_AVAILABLE and self._data:
            for joint_name, joint_info in self._joints.items():
                qpos_idx = joint_info.qpos_idx
                qvel_idx = joint_info.qvel_idx
                
                pos = self._data.qpos[qpos_idx] if qpos_idx < len(self._data.qpos) else 0.0
                vel = self._data.qvel[qvel_idx] if qvel_idx < len(self._data.qvel) else 0.0
                acc = self._data.qacc[qvel_idx] if qvel_idx < len(self._data.qacc) else 0.0
                trq = self._data.qfrc_applied[qvel_idx] if qvel_idx < len(self._data.qfrc_applied) else 0.0
                
                joint_positions[joint_name] = float(pos)
                joint_velocities[joint_name] = float(vel)
                joint_accelerations[joint_name] = float(acc)
                joint_torques[joint_name] = float(trq)
            
            sim_time = self._data.time
        else:
            for i, joint_name in enumerate(self._joint_names):
                joint_positions[joint_name] = float(self._mock_qpos[i])
                joint_velocities[joint_name] = float(self._mock_qvel[i])
                joint_accelerations[joint_name] = 0.0
                joint_torques[joint_name] = float(self._mock_ctrl[i])
            
            sim_time = self._mock_time
        
        body_positions = {}
        body_velocities = {}
        
        if MUJOCO_AVAILABLE and self._data:
            for body_name in self._body_names:
                body_info = self._bodies.get(body_name)
                if body_info:
                    body_idx = body_info.body_id
                    pos = self._data.xpos[body_idx].copy()
                    body_positions[body_name] = pos
                    
                    vel = np.zeros(6)
                    if body_idx < len(self._data.cvel):
                        vel[:3] = self._data.cvel[body_idx, 3:]
                        vel[3:] = self._data.cvel[body_idx, :3]
                    body_velocities[body_name] = vel
        else:
            body_positions = self._mock_body_positions.copy()
            for name in self._body_names:
                body_velocities[name] = np.zeros(6)
        
        sensor_data = {}
        
        if MUJOCO_AVAILABLE and self._data:
            for sensor_name, sensor_info in self._sensors.items():
                start = sensor_info.data_start
                dim = sensor_info.data_dim
                if start + dim <= len(self._data.sensordata):
                    sensor_data[sensor_name] = self._data.sensordata[start:start+dim].copy()
                else:
                    sensor_data[sensor_name] = np.zeros(dim)
        else:
            for i, sensor_name in enumerate(self._sensor_names):
                if "pos" in sensor_name.lower():
                    joint_idx = i // 2 if i < 4 else 0
                    sensor_data[sensor_name] = np.array([self._mock_qpos[joint_idx % 2]])
                elif "vel" in sensor_name.lower():
                    joint_idx = i // 2 if i < 4 else 0
                    sensor_data[sensor_name] = np.array([self._mock_qvel[joint_idx % 2]])
                else:
                    sensor_data[sensor_name] = np.zeros(1)
        
        return RobotState(
            time=sim_time,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_accelerations=joint_accelerations,
            joint_torques=joint_torques,
            body_positions=body_positions,
            body_velocities=body_velocities,
            sensor_data=sensor_data
        )
    
    def get_joint_qpos(self, joint_name: str) -> Optional[float]:
        joint_info = self._joints.get(joint_name)
        if not joint_info:
            return None
        
        if MUJOCO_AVAILABLE and self._data:
            qpos_idx = joint_info.qpos_idx
            if qpos_idx < len(self._data.qpos):
                return float(self._data.qpos[qpos_idx])
        else:
            idx = joint_info.index
            if idx < len(self._mock_qpos):
                return float(self._mock_qpos[idx])
        
        return None
    
    def get_joint_qvel(self, joint_name: str) -> Optional[float]:
        joint_info = self._joints.get(joint_name)
        if not joint_info:
            return None
        
        if MUJOCO_AVAILABLE and self._data:
            qvel_idx = joint_info.qvel_idx
            if qvel_idx < len(self._data.qvel):
                return float(self._data.qvel[qvel_idx])
        else:
            idx = joint_info.index
            if idx < len(self._mock_qvel):
                return float(self._mock_qvel[idx])
        
        return None
    
    def set_control(self, index: int, value: float) -> None:
        if MUJOCO_AVAILABLE and self._data:
            if index < len(self._data.ctrl):
                self._data.ctrl[index] = value
        else:
            if index < len(self._mock_ctrl):
                self._mock_ctrl[index] = value
    
    def get_control(self, index: int) -> float:
        if MUJOCO_AVAILABLE and self._data:
            if index < len(self._data.ctrl):
                return float(self._data.ctrl[index])
        else:
            if index < len(self._mock_ctrl):
                return float(self._mock_ctrl[index])
        return 0.0
    
    def apply_body_force(self, body_name: str, force: np.ndarray, point: Optional[np.ndarray] = None) -> None:
        if not MUJOCO_AVAILABLE or not self._data:
            return
        
        body_info = self._bodies.get(body_name)
        if not body_info:
            return
        
        body_idx = body_info.body_id
        
        if point is None:
            point = np.zeros(3)
        
        if body_idx < len(self._data.xfrc_applied):
            self._data.xfrc_applied[body_idx, :3] = force
            if len(force) >= 6:
                self._data.xfrc_applied[body_idx, 3:] = force[3:]
    
    def get_body_position(self, body_name: str) -> Optional[np.ndarray]:
        if MUJOCO_AVAILABLE and self._data:
            body_info = self._bodies.get(body_name)
            if body_info:
                body_idx = body_info.body_id
                if body_idx < len(self._data.xpos):
                    return self._data.xpos[body_idx].copy()
        else:
            return self._mock_body_positions.get(body_name)
        
        return None
    
    @property
    def mj_model(self):
        return self._model
    
    @property
    def mj_data(self):
        return self._data
