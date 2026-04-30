"""
MuJoCo-MAVLink Bridge - 核心数据类型

本模块定义了框架中使用的核心数据结构：
- ControlVector: 控制量向量，作为Plant的输入
- StateVector: 状态向量，作为Plant的输出
- ControlSource: MAVLink控制消息来源类型
- ControlMapping: 控制量映射（MAVLink → Plant控制量）

架构说明：
    MAVLink (外部) ──控制量u──► Plant (仿真器) ──状态x──► MAVLink (外部)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import numpy as np


@dataclass
class ControlVector:
    """
    控制量向量类。
    
    表示Plant（被控对象）的输入控制量。每个控制量对应Plant的一个执行器。
    例如：关节力矩、力、速度等。
    
    属性:
        values: np.ndarray - 控制量数值数组，形状为(control_dim,)
        names: Optional[List[str]] - 控制量名称列表，长度与values相同
    
    示例:
        >>> # 创建2个控制量的向量
        >>> control = ControlVector(values=[1.0, 0.5], names=["joint1", "joint2"])
        >>> len(control)
        2
        >>> control[0]
        1.0
    """
    
    values: np.ndarray
    """控制量数值数组"""
    
    names: Optional[List[str]] = None
    """控制量名称列表，可选"""
    
    def __post_init__(self):
        """
        初始化后处理。
        
        自动将list类型的values转换为np.ndarray，
        并在未提供names时生成默认名称。
        """
        if isinstance(self.values, list):
            self.values = np.array(self.values, dtype=np.float64)
        
        if self.names is None:
            self.names = [f"u{i}" for i in range(len(self.values))]
    
    def __len__(self) -> int:
        """
        返回控制量数量。
        
        Returns:
            int - 控制量维度
        """
        return len(self.values)
    
    def __getitem__(self, idx: int) -> float:
        """
        按索引获取控制量值。
        
        Args:
            idx: int - 控制量索引
            
        Returns:
            float - 控制量值
        """
        return float(self.values[idx])
    
    def __setitem__(self, idx: int, value: float):
        """
        按索引设置控制量值。
        
        Args:
            idx: int - 控制量索引
            value: float - 控制量值
        """
        self.values[idx] = value
    
    def to_list(self) -> List[float]:
        """
        转换为Python列表。
        
        Returns:
            List[float] - 控制量数值列表
        """
        return self.values.tolist()
    
    def to_dict(self) -> Dict[str, float]:
        """
        转换为字典（名称到值的映射）。
        
        Returns:
            Dict[str, float] - 控制名称到值的映射字典
        """
        if self.names:
            return {name: float(val) for name, val in zip(self.names, self.values)}
        return {f"u{i}": float(val) for i, val in enumerate(self.values)}


@dataclass
class StateVector:
    """
    状态向量类。
    
    表示Plant（被控对象）的输出状态。包含关节、刚体、传感器等各种状态信息。
    
    属性:
        joint_positions: Dict[str, float] - 关节位置（角度/位移）
        joint_velocities: Dict[str, float] - 关节速度
        body_positions: Dict[str, np.ndarray] - 刚体位姿（世界坐标系下位置）
        body_velocities: Dict[str, np.ndarray] - 刚体速度（线速度+角速度，6维）
        sensor_data: Dict[str, np.ndarray] - 传感器原始数据
        time: float - 仿真时间（秒）
        custom_data: Dict[str, Any] - 自定义扩展数据
    
    示例:
        >>> state = StateVector()
        >>> state.joint_positions["joint1"] = 0.5
        >>> state.get_joint_position("joint1")
        0.5
    """
    
    joint_positions: Dict[str, float] = field(default_factory=dict)
    """关节位置字典，键为关节名称，值为位置值"""
    
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    """关节速度字典"""
    
    body_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    """刚体位姿字典，值为(3,)形状的位置数组"""
    
    body_velocities: Dict[str, np.ndarray] = field(default_factory=dict)
    """刚体速度字典，值为(6,)形状的数组[vx, vy, vz, wx, wy, wz]"""
    
    sensor_data: Dict[str, np.ndarray] = field(default_factory=dict)
    """传感器数据字典"""
    
    time: float = 0.0
    """当前仿真时间"""
    
    custom_data: Dict[str, Any] = field(default_factory=dict)
    """自定义扩展数据字段"""
    
    def get_joint_position(self, name: str) -> Optional[float]:
        """
        获取指定关节的位置。
        
        Args:
            name: str - 关节名称
            
        Returns:
            Optional[float] - 关节位置，不存在时返回None
        """
        return self.joint_positions.get(name)
    
    def get_joint_velocity(self, name: str) -> Optional[float]:
        """
        获取指定关节的速度。
        
        Args:
            name: str - 关节名称
            
        Returns:
            Optional[float] - 关节速度，不存在时返回None
        """
        return self.joint_velocities.get(name)
    
    def get_body_position(self, name: str) -> Optional[np.ndarray]:
        """
        获取指定刚体的位置。
        
        Args:
            name: str - 刚体名称
            
        Returns:
            Optional[np.ndarray] - 位置数组(3,)，不存在时返回None
        """
        return self.body_positions.get(name)
    
    def get_body_velocity(self, name: str) -> Optional[np.ndarray]:
        """
        获取指定刚体的速度。
        
        Args:
            name: str - 刚体名称
            
        Returns:
            Optional[np.ndarray] - 速度数组(6,)，不存在时返回None
        """
        return self.body_velocities.get(name)
    
    def get_sensor_data(self, name: str) -> Optional[np.ndarray]:
        """
        获取指定传感器的数据。
        
        Args:
            name: str - 传感器名称
            
        Returns:
            Optional[np.ndarray] - 传感器数据，不存在时返回None
        """
        return self.sensor_data.get(name)
    
    @property
    def joint_names(self) -> List[str]:
        """
        获取所有关节名称列表。
        
        Returns:
            List[str] - 关节名称列表
        """
        return list(self.joint_positions.keys())
    
    @property
    def body_names(self) -> List[str]:
        """
        获取所有刚体名称列表。
        
        Returns:
            List[str] - 刚体名称列表
        """
        return list(self.body_positions.keys())
    
    @property
    def sensor_names(self) -> List[str]:
        """
        获取所有传感器名称列表。
        
        Returns:
            List[str] - 传感器名称列表
        """
        return list(self.sensor_data.keys())


class ControlSource(Enum):
    """
    MAVLink控制消息来源类型。
    
    定义了可以接收控制量的MAVLink消息类型。
    
    成员:
        HIL_ACTUATOR_CONTROLS: 硬件在环执行器控制消息（最常用）
        RC_CHANNELS: 遥控器通道消息
        MANUAL_CONTROL: 手动控制消息
        CUSTOM: 自定义消息类型
    """
    
    HIL_ACTUATOR_CONTROLS = "hil_actuator_controls"
    """硬件在环执行器控制消息（MAVLink MSG ID 93）"""
    
    RC_CHANNELS = "rc_channels"
    """遥控器通道消息"""
    
    MANUAL_CONTROL = "manual_control"
    """手动控制消息"""
    
    CUSTOM = "custom"
    """自定义消息类型"""


class StateTarget(Enum):
    """
    MAVLink状态消息目标类型。
    
    定义了可以发送状态的MAVLink消息类型。
    
    成员:
        HIL_STATE: 硬件在环状态消息
        HIL_STATE_QUATERNION: 四元数格式的状态消息
        HIL_GPS: GPS状态消息
        HIL_SENSOR: 传感器消息
        CUSTOM: 自定义消息类型
    """
    
    HIL_STATE = "hil_state"
    """硬件在环状态消息"""
    
    HIL_STATE_QUATERNION = "hil_state_quaternion"
    """四元数格式的状态消息"""
    
    HIL_GPS = "hil_gps"
    """GPS状态消息"""
    
    HIL_SENSOR = "hil_sensor"
    """传感器消息"""
    
    CUSTOM = "custom"
    """自定义消息类型"""


@dataclass
class ControlMappingEntry:
    """
    控制量映射条目。
    
    定义单个MAVLink控制通道到Plant控制量的映射关系。
    支持缩放、偏移、限幅和自定义变换。
    
    属性:
        mavlink_source: ControlSource - MAVLink消息来源类型
        mavlink_index: int - MAVLink消息中控制通道的索引
        plant_control_name: str - Plant控制量的名称
        plant_control_index: int - Plant控制量的索引
        scale: float - 缩放因子（默认1.0）
        offset: float - 偏移量（默认0.0）
        range_min: float - 下限（默认-∞）
        range_max: float - 上限（默认+∞）
        transform: Optional[Callable[[float], float]] - 自定义变换函数
        enabled: bool - 是否启用此映射
        
    示例:
        >>> # 映射：MAVLink通道0 → Plant控制量"joint1"，缩放2.0，限幅[-1, 1]
        >>> entry = ControlMappingEntry(
        ...     mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
        ...     mavlink_index=0,
        ...     plant_control_name="joint1",
        ...     plant_control_index=0,
        ...     scale=2.0,
        ...     range_min=-1.0,
        ...     range_max=1.0
        ... )
        >>> entry.apply(0.5)
        1.0  # 0.5 * 2.0 = 1.0
    """
    
    mavlink_source: ControlSource
    """MAVLink消息来源类型"""
    
    mavlink_index: int
    """MAVLink消息中控制通道的索引"""
    
    plant_control_name: str
    """Plant控制量的名称"""
    
    plant_control_index: int
    """Plant控制量的索引"""
    
    scale: float = 1.0
    """缩放因子：value = raw_value * scale + offset"""
    
    offset: float = 0.0
    """偏移量"""
    
    range_min: float = -np.inf
    """输出下限"""
    
    range_max: float = np.inf
    """输出上限"""
    
    transform: Optional[Callable[[float], float]] = None
    """自定义变换函数，如果提供则忽略scale和offset"""
    
    enabled: bool = True
    """是否启用此映射"""
    
    def apply(self, raw_value: float) -> float:
        """
        应用映射变换到原始值。
        
        处理流程：
        1. 如果有transform函数，使用transform(raw_value)
        2. 否则使用：raw_value * scale + offset
        3. 最后限幅到[range_min, range_max]
        
        Args:
            raw_value: float - MAVLink消息中的原始控制值
            
        Returns:
            float - 变换后的值，已限幅
        """
        if self.transform:
            value = self.transform(raw_value)
        else:
            value = raw_value * self.scale + self.offset
        
        return float(np.clip(value, self.range_min, self.range_max))


class ControlMapping:
    """
    控制量映射管理器。
    
    管理一组ControlMappingEntry，实现从MAVLink消息到Plant控制量的批量映射。
    
    架构说明：
        MAVLink消息 (controls[0..15])
            │
            ├──► 映射表查询
            │       ├── controls[0] → plant_control[0]
            │       ├── controls[1] → plant_control[2]
            │       └── ...
            │
            ▼
        Plant控制量输入
    
    属性:
        _entries: List[ControlMappingEntry] - 映射条目列表
        _mavlink_to_entry: Dict[Tuple[ControlSource, int], ControlMappingEntry] - 快速查找索引
        _name_to_entry: Dict[str, ControlMappingEntry] - 按名称查找索引
    
    示例:
        >>> mapping = ControlMapping()
        >>> # 为2个控制量创建默认映射
        >>> mapping.create_default_joint_mappings(["joint1", "joint2"])
        >>> # 映射MAVLink控制值
        >>> result = mapping.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [1.0, 0.5])
        >>> result
        {0: 1.0, 1: 0.5}
    """
    
    def __init__(self):
        """
        初始化空的控制映射管理器。
        """
        self._entries: List[ControlMappingEntry] = []
        """映射条目列表"""
        
        self._mavlink_to_entry: Dict[Tuple[ControlSource, int], ControlMappingEntry] = {}
        """(source, index)到条目的快速查找字典"""
        
        self._name_to_entry: Dict[str, ControlMappingEntry] = {}
        """名称到条目的快速查找字典"""
    
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
        """
        添加一个控制量映射。
        
        Args:
            mavlink_source: MAVLink消息来源（ControlSource或字符串）
            mavlink_index: MAVLink消息中的通道索引
            plant_control_name: Plant控制量名称
            plant_control_index: Plant控制量索引
            scale: 缩放因子
            offset: 偏移量
            range_min: 下限
            range_max: 上限
            transform: 自定义变换函数
            enabled: 是否启用
        """
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
        """
        根据MAVLink来源和索引获取映射条目。
        
        Args:
            mavlink_source: ControlSource - 消息来源类型
            mavlink_index: int - 通道索引
            
        Returns:
            Optional[ControlMappingEntry] - 映射条目，不存在时返回None
        """
        return self._mavlink_to_entry.get((mavlink_source, mavlink_index))
    
    def get_mapping_by_name(self, name: str) -> Optional[ControlMappingEntry]:
        """
        根据控制量名称获取映射条目。
        
        Args:
            name: str - 控制量名称
            
        Returns:
            Optional[ControlMappingEntry] - 映射条目，不存在时返回None
        """
        return self._name_to_entry.get(name)
    
    def map_controls(self,
                      mavlink_source: ControlSource,
                      raw_controls: List[float]) -> Dict[int, float]:
        """
        将MAVLink原始控制值映射为Plant控制量索引到值的字典。
        
        对每个raw_controls中的值，查找对应的映射条目并应用变换。
        
        Args:
            mavlink_source: ControlSource - 消息来源类型
            raw_controls: List[float] - 原始控制值列表
            
        Returns:
            Dict[int, float] - Plant控制量索引到变换后值的字典
            
        示例:
            >>> # 假设映射：[0→0, 1→1]
            >>> mapping.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [1.0, 0.5])
            {0: 1.0, 1: 0.5}
        """
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
        """
        映射单个控制值。
        
        Args:
            mavlink_source: ControlSource - 消息来源类型
            mavlink_index: int - 通道索引
            raw_value: float - 原始值
            
        Returns:
            Optional[Tuple[int, float]] - (控制量索引, 变换后的值)，映射不存在时返回None
        """
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
        """
        为一组控制名称创建默认的一一映射。
        
        映射关系：
            MAVLink[start_index] → control_names[0]
            MAVLink[start_index+1] → control_names[1]
            ...
        
        Args:
            control_names: List[str] - 控制量名称列表
            mavlink_source: ControlSource - 消息来源类型
            start_index: int - MAVLink起始索引
            scale: float - 统一缩放因子
            offset: float - 统一偏移量
        """
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
        """
        启用所有映射条目。
        """
        for entry in self._entries:
            entry.enabled = True
    
    def disable_all(self) -> None:
        """
        禁用所有映射条目。
        """
        for entry in self._entries:
            entry.enabled = False
    
    def clear(self) -> None:
        """
        清空所有映射条目。
        """
        self._entries.clear()
        self._mavlink_to_entry.clear()
        self._name_to_entry.clear()
    
    @property
    def entries(self) -> List[ControlMappingEntry]:
        """
        获取所有映射条目副本。
        
        Returns:
            List[ControlMappingEntry] - 映射条目列表的副本
        """
        return self._entries.copy()
    
    @property
    def enabled_entries(self) -> List[ControlMappingEntry]:
        """
        获取所有启用的映射条目。
        
        Returns:
            List[ControlMappingEntry] - 启用的映射条目列表
        """
        return [e for e in self._entries if e.enabled]
