"""
MuJoCo-MAVLink Bridge - MuJoCo被控对象类

本模块定义了MuJoCoPlant类，作为框架的核心被控对象：

架构说明：
    ┌─────────────────────────────────────────────────────────────┐
    │                      MuJoCoPlant                              │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │                    输入 (控制量 u)                      │   │
    │  │  - set_control(): 设置所有控制量                         │   │
    │  │  - set_control_by_index(): 按索引设置单个控制量          │   │
    │  │  - set_control_by_name(): 按名称设置单个控制量           │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                            │                                  │
    │                            ▼                                  │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │                    MuJoCo物理仿真                       │   │
    │  │  - model: MuJoCo模型定义 (MjModel)                     │   │
    │  │  - data:  MuJoCo状态数据 (MjData)                       │   │
    │  │  - step(): 执行仿真步进                                  │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                            │                                  │
    │                            ▼                                  │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │                    输出 (状态 x)                         │   │
    │  │  - get_state(): 获取完整状态向量                         │   │
    │  │    包含：关节位置/速度、刚体位置/速度、传感器数据          │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

扩展方式：
    子类可以继承MuJoCoPlant并重写以下方法：
    - _initialize_model(): 自定义模型初始化
    - step(): 自定义仿真步进逻辑
    - get_state(): 自定义状态提取
    - set_control(): 自定义控制量应用
"""

from typing import Dict, List, Optional, Any
import numpy as np
import logging

from .types import StateVector, ControlVector

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


class MuJoCoPlant:
    """
    MuJoCo被控对象类。
    
    作为框架的核心被控对象(Plant)，提供以下能力：
    1. 接收控制量输入（关节力矩、力等）
    2. 执行MuJoCo物理仿真
    3. 输出状态向量（关节位置/速度、刚体姿态等）
    
    这是一个具体的父类实现，子类可以通过继承扩展功能。
    
    属性:
        _model_path: Optional[str] - 模型文件路径
        _model_xml: Optional[str] - 模型XML字符串
        _model: Optional[mujoco.MjModel] - MuJoCo模型对象
        _data: Optional[mujoco.MjData] - MuJoCo数据对象
        _is_initialized: bool - 是否已初始化
        _joint_names: List[str] - 关节名称列表
        _body_names: List[str] - 刚体名称列表
        _sensor_names: List[str] - 传感器名称列表
        _control_dim: int - 控制量维度
        _control_names: List[str] - 控制量名称列表
        _timestep: float - 仿真时间步长
    
    示例:
        >>> # 使用默认模型
        >>> plant = MuJoCoPlant()
        >>> plant.control_dim
        2
        
        >>> # 使用自定义模型
        >>> plant = MuJoCoPlant(model_path="my_robot.xml")
        
        >>> # 基本操作
        >>> plant.set_control_by_index(0, 1.0)  # 设置第0个控制量
        >>> plant.step()                          # 执行一步仿真
        >>> state = plant.get_state()            # 获取状态
        >>> state.joint_positions
        {'joint1': 0.001, 'joint2': 0.0}
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_xml: Optional[str] = None):
        """
        初始化MuJoCoPlant。
        
        可以通过以下方式之一加载模型：
        1. model_path: 从文件加载XML模型
        2. model_xml: 直接使用XML字符串
        3. 都不提供: 使用内置的默认两关节机器人模型
        
        如果mujoco库未安装，会自动回退到mock实现（简单的物理模拟）。
        
        Args:
            model_path: Optional[str] - MuJoCo模型XML文件路径
            model_xml: Optional[str] - MuJoCo模型XML字符串
        """
        self._model_path = model_path
        """模型文件路径"""
        
        self._model_xml = model_xml
        """模型XML字符串"""
        
        self._model = None
        """MuJoCo模型对象 (MjModel)"""
        
        self._data = None
        """MuJoCo数据对象 (MjData)"""
        
        self._is_initialized = False
        """ 是否已初始化标志"""
        
        self._joint_names: List[str] = []
        """关节名称列表"""
        
        self._joint_to_actuator: Dict[str, int] = {}
        """关节名称到执行器索引的映射"""
        
        self._actuator_to_joint: Dict[int, str] = {}
        """执行器索引到关节名称的映射"""
        
        self._body_names: List[str] = []
        """刚体名称列表"""
        
        self._sensor_names: List[str] = []
        """传感器名称列表"""
        
        self._control_dim = 0
        """控制量维度（执行器数量）"""
        
        self._control_names: List[str] = []
        """控制量名称列表"""
        
        self._timestep = 0.01
        """仿真时间步长（秒）"""
        
        self._mock_qpos: np.ndarray = np.array([])
        """Mock模式下的关节位置"""
        
        self._mock_qvel: np.ndarray = np.array([])
        """Mock模式下的关节速度"""
        
        self._mock_ctrl: np.ndarray = np.array([])
        """Mock模式下的控制量"""
        
        self._mock_time: float = 0.0
        """Mock模式下的仿真时间"""
        
        self._mock_body_positions: Dict[str, np.ndarray] = {}
        """Mock模式下的刚体位置"""
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        初始化模型。
        
        根据mujoco库是否可用，选择不同的初始化方式：
        - 可用：使用真实的MuJoCo初始化
        - 不可用：使用mock模拟实现
        
        子类可以重写此方法以自定义模型初始化逻辑。
        """
        if MUJOCO_AVAILABLE:
            self._initialize_mujoco()
        else:
            self._initialize_mock()
    
    def _initialize_mujoco(self) -> None:
        """
        使用真实MuJoCo库初始化模型。
        
        加载流程：
        1. 优先从model_path加载文件
        2. 如果没有model_path，使用model_xml字符串
        3. 如果都没有，使用默认的DEFAULT_ROBOT_XML
        
        初始化完成后提取模型信息（关节、执行器、刚体、传感器等）。
        """
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
        """
        使用mock模拟初始化（无mujoco库时）。
        
        创建一个简单的两关节机器人模拟，用于测试和开发。
        包含基本的物理模拟（力矩→速度→位置）和阻尼。
        """
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
        """
        从MuJoCo模型中提取信息。
        
        提取的信息包括：
        - 关节名称列表
        - 执行器名称和索引映射
        - 刚体名称列表
        - 传感器名称列表
        """
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
        """
        控制量维度（执行器数量）。
        
        Returns:
            int - 控制量维度
        """
        return self._control_dim
    
    @property
    def control_names(self) -> List[str]:
        """
        控制量名称列表的副本。
        
        Returns:
            List[str] - 控制量名称列表
        """
        return self._control_names.copy()
    
    @property
    def joint_names(self) -> List[str]:
        """
        关节名称列表的副本。
        
        Returns:
            List[str] - 关节名称列表
        """
        return self._joint_names.copy()
    
    @property
    def body_names(self) -> List[str]:
        """
        刚体名称列表的副本。
        
        Returns:
            List[str] - 刚体名称列表
        """
        return self._body_names.copy()
    
    @property
    def sensor_names(self) -> List[str]:
        """
        传感器名称列表的副本。
        
        Returns:
            List[str] - 传感器名称列表
        """
        return self._sensor_names.copy()
    
    @property
    def timestep(self) -> float:
        """
        仿真时间步长（秒）。
        
        Returns:
            float - 时间步长
        """
        return self._timestep
    
    @property
    def is_initialized(self) -> bool:
        """
        是否已初始化。
        
        Returns:
            bool - True表示已成功初始化
        """
        return self._is_initialized
    
    def set_control(self, control: ControlVector) -> None:
        """
        设置所有控制量。
        
        将ControlVector中的值应用到所有执行器。
        超出维度的部分会被忽略，不足的部分保持原值。
        
        Args:
            control: ControlVector - 控制量向量
            
        示例:
            >>> control = ControlVector(values=[1.0, 0.5], names=["motor1", "motor2"])
            >>> plant.set_control(control)
        """
        if not self._is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self._data is not None:
            for i in range(min(len(control), len(self._data.ctrl))):
                self._data.ctrl[i] = control.values[i]
        else:
            for i in range(min(len(control), len(self._mock_ctrl))):
                self._mock_ctrl[i] = control.values[i]
    
    def set_control_by_index(self, index: int, value: float) -> None:
        """
        按索引设置单个控制量。
        
        Args:
            index: int - 控制量索引（从0开始）
            value: float - 控制量值
            
        示例:
            >>> plant.set_control_by_index(0, 1.0)  # 设置第0个执行器
        """
        if not self._is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self._data is not None:
            if 0 <= index < len(self._data.ctrl):
                self._data.ctrl[index] = value
        else:
            if 0 <= index < len(self._mock_ctrl):
                self._mock_ctrl[index] = value
    
    def set_control_by_name(self, name: str, value: float) -> None:
        """
        按名称设置单个控制量。
        
        会尝试以下两种方式查找：
        1. 直接匹配控制量名称（执行器名称）
        2. 匹配关节名称（通过关节到执行器的映射）
        
        Args:
            name: str - 控制量名称或关节名称
            value: float - 控制量值
            
        示例:
            >>> plant.set_control_by_name("motor1", 1.0)  # 按执行器名
            >>> plant.set_control_by_name("joint1", 1.0)  # 按关节名
        """
        if not self._is_initialized:
            return
        
        if name in self._control_names:
            index = self._control_names.index(name)
            self.set_control_by_index(index, value)
        elif name in self._joint_to_actuator:
            index = self._joint_to_actuator[name]
            self.set_control_by_index(index, value)
    
    def get_control(self) -> ControlVector:
        """
        获取当前控制量。
        
        Returns:
            ControlVector - 当前控制量向量
            
        示例:
            >>> control = plant.get_control()
            >>> control.values
            array([1.0, 0.5])
        """
        if not self._is_initialized:
            return ControlVector(values=np.array([]))
        
        if MUJOCO_AVAILABLE and self._data is not None:
            values = self._data.ctrl.copy()
        else:
            values = self._mock_ctrl.copy()
        
        return ControlVector(values=values, names=self._control_names)
    
    def get_state(self) -> StateVector:
        """
        获取当前状态向量。
        
        状态向量包含：
        - 关节位置（joint_positions）
        - 关节速度（joint_velocities）
        - 刚体位置（body_positions）
        - 刚体速度（body_velocities）- 6维：[vx, vy, vz, wx, wy, wz]
        - 传感器数据（sensor_data）
        - 仿真时间（time）
        
        子类可以重写此方法以自定义状态提取逻辑。
        
        Returns:
            StateVector - 完整的状态向量
            
        示例:
            >>> state = plant.get_state()
            >>> state.joint_positions
            {'joint1': 0.0, 'joint2': 0.0}
            >>> state.time
            0.01
        """
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
        """
        执行仿真步进。
        
        根据当前控制量更新物理状态。
        
        子类可以重写此方法以自定义仿真步进逻辑。
        
        Args:
            n_steps: int - 执行的步数，默认1步
            
        示例:
            >>> plant.step()           # 执行1步
            >>> plant.step(n_steps=10)  # 执行10步
        """
        if not self._is_initialized:
            return
        
        if MUJOCO_AVAILABLE and self._model and self._data:
            for _ in range(n_steps):
                mujoco.mj_step(self._model, self._data)
        else:
            for _ in range(n_steps):
                self._mock_step()
    
    def _mock_step(self) -> None:
        """
        Mock模式下的仿真步进。
        
        实现简单的物理模拟：
        - 力矩(torque) = 控制量
        - 速度 += 力矩 * dt
        - 位置 += 速度 * dt
        - 速度 *= 阻尼系数(0.99)
        
        这是一个简化的模型，仅用于测试。
        """
        dt = self._timestep
        self._mock_time += dt
        
        for i in range(len(self._mock_ctrl)):
            torque = self._mock_ctrl[i]
            self._mock_qvel[i] += torque * dt
            self._mock_qpos[i] += self._mock_qvel[i] * dt
            
            damping = 0.99
            self._mock_qvel[i] *= damping
    
    def reset(self) -> None:
        """
        重置仿真状态。
        
        将所有状态变量恢复到初始值（位置=0，速度=0，控制量=0，时间=0）。
        
        示例:
            >>> plant.reset()
        """
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
        """
        获取原始MuJoCo模型对象（如果可用）。
        
        Returns:
            Optional[mujoco.MjModel] - MuJoCo模型对象，mock模式下返回None
        """
        return self._model
    
    @property
    def mj_data(self):
        """
        获取原始MuJoCo数据对象（如果可用）。
        
        Returns:
            Optional[mujoco.MjData] - MuJoCo数据对象，mock模式下返回None
        """
        return self._data
    
    @property
    def state(self) -> StateVector:
        """
        获取当前状态（get_state()的快捷方式）。
        
        Returns:
            StateVector - 当前状态向量
        """
        return self.get_state()
    
    def set_control_array(self, values: np.ndarray) -> None:
        """
        用numpy数组设置控制量。
        
        Args:
            values: np.ndarray - 控制量数组
            
        示例:
            >>> plant.set_control_array(np.array([1.0, 0.5]))
        """
        self.set_control(ControlVector(values=values))
    
    def set_control_list(self, values: List[float]) -> None:
        """
        用Python列表设置控制量。
        
        Args:
            values: List[float] - 控制量列表
            
        示例:
            >>> plant.set_control_list([1.0, 0.5])
        """
        self.set_control(ControlVector(values=np.array(values, dtype=np.float64)))
