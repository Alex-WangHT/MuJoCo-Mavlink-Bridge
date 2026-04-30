"""
MuJoCo-MAVLink Bridge - 主仿真器

本模块定义了Simulator类，协调Plant和MAVLink接口的交互：

架构说明：
    ┌─────────────────────────────────────────────────────────────────┐
    │                          Simulator                                 │
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐ │
    │  │                    主循环 (Real-time)                         │ │
    │  │                                                                 │ │
    │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │ │
    │  │  │ 接收   │───►│ 应用   │───►│ 仿真   │───►│ 发送   │  │ │
    │  │  │控制量  │    │控制量  │    │步进   │    │状态   │  │ │
    │  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │ │
    │  │       │              │              │              │         │ │
    │  │       ▼              ▼              ▼              ▼         │ │
    │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │ │
    │  │  │Mavlink  │    │Control  │    │MuJoCo   │    │Mavlink  │  │ │
    │  │  │Interface│    │Mapping  │    │Plant    │    │Interface│  │ │
    │  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │ │
    │  └─────────────────────────────────────────────────────────────┘ │
    │                                                                     │
    │  时序说明：                                                          │
    │  1. receive_controls()   - 从MAVLink接收控制量                      │
    │  2. control_mapping.map_controls() - 映射MAVLink值到Plant控制量    │
    │  3. plant.set_control()  - 应用控制量到Plant                        │
    │  4. plant.step()         - 执行仿真步进                              │
    │  5. plant.get_state()    - 获取状态                                  │
    │  6. mavlink.send_state() - 发送状态反馈                              │
    └─────────────────────────────────────────────────────────────────┘
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .core import (
    MuJoCoPlant,
    MavlinkUDPInterface,
    ControlMapping,
    StateVector,
    ControlVector,
    ControlSource,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """
    仿真器配置类。
    
    定义了Simulator的运行参数。
    
    属性:
        mavlink_host: str - MAVLink监听主机地址
        mavlink_port: int - MAVLink监听端口
        real_time_factor: float - 实时因子，1.0表示实时，>1.0表示加速
        timestep: Optional[float] - 仿真时间步长，None表示使用Plant默认值
        model_path: Optional[str] - MuJoCo模型文件路径
        model_xml: Optional[str] - MuJoCo模型XML字符串
        enable_telemetry: bool - 是否启用遥测（状态发送）
        telemetry_interval: float - 遥测发送间隔（秒）
        heartbeat_interval: float - 心跳发送间隔（秒）
    """
    
    mavlink_host: str = "0.0.0.0"
    """MAVLink监听主机地址"""
    
    mavlink_port: int = 14540
    """MAVLink监听端口"""
    
    real_time_factor: float = 1.0
    """实时因子，1.0表示实时"""
    
    timestep: Optional[float] = None
    """仿真时间步长"""
    
    model_path: Optional[str] = None
    """MuJoCo模型文件路径"""
    
    model_xml: Optional[str] = None
    """MuJoCo模型XML字符串"""
    
    enable_telemetry: bool = True
    """是否启用遥测"""
    
    telemetry_interval: float = 0.1
    """遥测发送间隔（秒）"""
    
    heartbeat_interval: float = 1.0
    """心跳发送间隔（秒）"""


class Simulator:
    """
    主仿真器类。
    
    协调Plant和MAVLink接口，实现完整的硬件在环(HIL)仿真闭环。
    
    核心流程：
        MAVLink (外部)
            │
            ▼ 控制量输入 (u)
        ┌─────────┐
        │ Control │
        │ Mapping │
        └─────────┘
            │
            ▼
        ┌─────────┐
        │  Plant  │ ──► step()
        │ (MuJoCo)│
        └─────────┘
            │
            ▼ 状态输出 (x)
        MAVLink (外部)
    
    属性:
        _config: SimulatorConfig - 仿真器配置
        _plant: MuJoCoPlant - 被控对象实例
        _mavlink_interface: MavlinkUDPInterface - MAVLink通信接口
        _control_mapping: ControlMapping - 控制量映射表
        _is_running: bool - 是否运行中
        _stats: Dict[str, Any] - 运行统计数据
    
    示例:
        >>> # 使用默认配置
        >>> sim = Simulator()
        >>> sim.start()
        >>> # ... 仿真运行中 ...
        >>> sim.stop()
        
        >>> # 使用自定义配置
        >>> config = SimulatorConfig(
        ...     mavlink_port=14550,
        ...     real_time_factor=2.0,  # 2倍速
        ...     model_path="my_robot.xml"
        ... )
        >>> sim = Simulator(config=config)
    """
    
    def __init__(self,
                 config: Optional[SimulatorConfig] = None,
                 plant: Optional[MuJoCoPlant] = None,
                 mavlink_interface: Optional[MavlinkUDPInterface] = None,
                 control_mapping: Optional[ControlMapping] = None):
        """
        初始化仿真器。
        
        可以通过参数注入自定义的Plant、MAVLink接口或控制映射，
        也可以让Simulator根据配置自动创建。
        
        Args:
            config: Optional[SimulatorConfig] - 仿真器配置，None则使用默认值
            plant: Optional[MuJoCoPlant] - 自定义Plant实例，None则自动创建
            mavlink_interface: Optional[MavlinkUDPInterface] - 自定义MAVLink接口，None则自动创建
            control_mapping: Optional[ControlMapping] - 自定义控制映射，None则自动创建
        """
        self._config = config or SimulatorConfig()
        """仿真器配置"""
        
        self._plant = plant
        """被控对象实例"""
        
        self._mavlink_interface = mavlink_interface
        """MAVLink通信接口"""
        
        self._control_mapping = control_mapping or ControlMapping()
        """控制量映射表"""
        
        self._is_running = False
        """ 是否运行中标志"""
        
        self._simulation_thread: Optional[threading.Thread] = None
        """仿真线程"""
        
        self._lock = threading.Lock()
        """线程锁"""
        
        self._stats = {
            "steps": 0,
            "controls_received": 0,
            "states_sent": 0,
            "heartbeats_sent": 0,
            "start_time": 0.0,
        }
        """运行统计数据"""
        
        self._last_telemetry_time: float = 0.0
        """上次发送遥测的时间"""
        
        self._last_heartbeat_time: float = 0.0
        """上次发送心跳的时间"""
        
        self._on_state_update: Optional[Callable[[StateVector], None]] = None
        """状态更新回调函数"""
        
        self._on_control_received: Optional[Callable[[Dict[ControlSource, List[float]]], None]] = None
        """控制量接收回调函数"""
        
        self._initialize()
    
    def _initialize(self) -> None:
        """
        初始化仿真器组件。
        
        如果用户没有注入自定义组件，则根据配置自动创建：
        1. 创建MuJoCoPlant
        2. 创建MavlinkUDPInterface
        3. 创建默认控制映射
        """
        if self._plant is None:
            self._plant = MuJoCoPlant(
                model_path=self._config.model_path,
                model_xml=self._config.model_xml
            )
        
        if self._mavlink_interface is None:
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
    def plant(self) -> MuJoCoPlant:
        """
        获取被控对象实例。
        
        Returns:
            MuJoCoPlant - 被控对象实例
        """
        return self._plant
    
    @property
    def mavlink_interface(self) -> MavlinkUDPInterface:
        """
        获取MAVLink通信接口实例。
        
        Returns:
            MavlinkUDPInterface - MAVLink接口实例
        """
        return self._mavlink_interface
    
    @property
    def control_mapping(self) -> ControlMapping:
        """
        获取控制量映射表。
        
        Returns:
            ControlMapping - 控制映射实例
        """
        return self._control_mapping
    
    @property
    def is_running(self) -> bool:
        """
        检查仿真器是否运行中。
        
        Returns:
            bool - True表示仿真线程正在运行
        """
        return self._is_running
    
    @property
    def config(self) -> SimulatorConfig:
        """
        获取仿真器配置。
        
        Returns:
            SimulatorConfig - 配置实例
        """
        return self._config
    
    def connect(self) -> bool:
        """
        连接MAVLink接口。
        
        Returns:
            bool - 连接成功返回True
        """
        if self._mavlink_interface.is_connected:
            return True
        
        success = self._mavlink_interface.connect()
        if success:
            logger.info("MAVLink interface connected")
        return success
    
    def disconnect(self) -> None:
        """
        断开MAVLink接口。
        """
        self._mavlink_interface.disconnect()
        logger.info("MAVLink interface disconnected")
    
    def start(self) -> None:
        """
        启动仿真器。
        
        执行以下操作：
        1. 如果未连接，自动连接MAVLink接口
        2. 启动MAVLink接收线程
        3. 启动仿真主线程
        
        Raises:
            RuntimeError: MAVLink连接失败时抛出异常
        """
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
        """
        停止仿真器。
        
        停止所有线程并断开连接。
        """
        self._is_running = False
        
        if self._simulation_thread:
            self._simulation_thread.join(timeout=2.0)
            self._simulation_thread = None
        
        self._mavlink_interface.stop()
        
        logger.info("Simulator stopped")
    
    def _simulation_loop(self) -> None:
        """
        仿真主循环（在线程中运行）。
        
        执行实时仿真，按照timestep和real_time_factor控制仿真速度。
        
        每一步执行：
        1. 接收并应用控制量
        2. 执行仿真步进
        3. 获取状态并触发回调
        4. 按间隔发送遥测和心跳
        """
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
        """
        执行单个仿真步。
        
        完整的闭环流程：
        1. 从MAVLink接收控制量
        2. 通过映射表转换为Plant控制量
        3. 应用控制量到Plant
        4. 执行仿真步进
        5. 获取状态
        6. 触发状态更新回调
        7. 按间隔发送状态反馈
        8. 按间隔发送心跳
        """
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
        """
        接收并应用控制量。
        
        流程：
        1. 从MAVLink接收控制消息
        2. 通过ControlMapping映射到Plant控制量
        3. 应用到Plant
        """
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
        """
        按间隔发送状态遥测。
        
        Args:
            state: StateVector - 当前状态向量
        """
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
        """
        按间隔发送心跳消息。
        """
        current_time = time.time()
        if current_time - self._last_heartbeat_time < self._config.heartbeat_interval:
            return
        
        self._last_heartbeat_time = current_time
        
        success = self._mavlink_interface.send_heartbeat()
        
        if success:
            with self._lock:
                self._stats["heartbeats_sent"] += 1
    
    def reset(self) -> None:
        """
        重置仿真器。
        
        重置Plant状态和统计数据。
        """
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
        """
        获取当前状态。
        
        Returns:
            StateVector - 当前状态向量
        """
        return self._plant.get_state()
    
    def set_control(self, control: ControlVector) -> None:
        """
        设置控制量。
        
        Args:
            control: ControlVector - 控制量向量
        """
        self._plant.set_control(control)
    
    def set_control_by_index(self, index: int, value: float) -> None:
        """
        按索引设置单个控制量。
        
        Args:
            index: int - 控制量索引
            value: float - 控制量值
        """
        self._plant.set_control_by_index(index, value)
    
    def set_control_by_name(self, name: str, value: float) -> None:
        """
        按名称设置单个控制量。
        
        Args:
            name: str - 控制量名称或关节名称
            value: float - 控制量值
        """
        self._plant.set_control_by_name(name, value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取运行统计数据。
        
        返回的统计数据包括：
        - steps: 仿真步数
        - controls_received: 接收控制量次数
        - states_sent: 发送状态次数
        - heartbeats_sent: 发送心跳次数
        - elapsed_time: 运行时间（秒）
        - steps_per_second: 平均每秒钟步数
        - real_time_factor: 实时因子
        - is_running: 是否运行中
        - is_connected: MAVLink是否连接
        
        Returns:
            Dict[str, Any] - 统计数据字典
        """
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
        """
        设置状态更新回调函数。
        
        每次仿真步进后会调用此回调。
        
        Args:
            callback: Callable[[StateVector], None] - 回调函数，
                接收当前状态向量作为参数
        """
        self._on_state_update = callback
    
    def set_on_control_received(self, callback: Callable[[Dict[ControlSource, List[float]]], None]) -> None:
        """
        设置控制量接收回调函数。
        
        每次接收到新的MAVLink控制消息时会调用此回调。
        
        Args:
            callback: Callable[[Dict[ControlSource, List[float]]], None] - 回调函数，
                接收控制消息字典作为参数
        """
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
        """
        添加控制量映射。
        
        Args:
            mavlink_source: ControlSource - MAVLink消息来源
            mavlink_index: int - MAVLink消息中的索引
            plant_control_name: str - Plant控制量名称
            plant_control_index: int - Plant控制量索引
            scale: float - 缩放因子
            offset: float - 偏移量
            range_min: float - 下限
            range_max: float - 上限
        """
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
        """
        清空所有控制量映射。
        """
        self._control_mapping.clear()
