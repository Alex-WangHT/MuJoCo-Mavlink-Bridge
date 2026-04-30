"""
MuJoCo-MAVLink Bridge - MAVLink UDP通信接口

本模块定义了MavlinkUDPInterface类，用于处理MAVLink协议的UDP通信：

架构说明：
    ┌─────────────────────────────────────────────────────────────────┐
    │                    MavlinkUDPInterface                             │
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐ │
    │  │                    接收 (MAVLink → 控制量)                      │ │
    │  │  - receive_controls(): 接收并解析MAVLink控制消息                │ │
    │  │    支持消息类型：                                                │ │
    │  │    - HIL_ACTUATOR_CONTROLS (硬件在环执行器控制)                 │ │
    │  │    - RC_CHANNELS_OVERRIDE (遥控器通道覆盖)                       │ │
    │  │    - MANUAL_CONTROL (手动控制)                                  │ │
    │  └─────────────────────────────────────────────────────────────┘ │
    │                              │                                       │
    │                              ▼                                       │
    │  ┌─────────────────────────────────────────────────────────────┐ │
    │  │                    发送 (状态 → MAVLink)                        │ │
    │  │  - send_hil_actuator_controls(): 发送执行器控制量（用于状态反馈）│ │
    │  │  - send_hil_state_quaternion(): 发送四元数格式的状态消息         │ │
    │  │  - send_heartbeat(): 发送心跳消息                               │ │
    │  └─────────────────────────────────────────────────────────────┘ │
    │                                                                     │
    │  底层实现：                                                          │
    │  - 优先使用pymavlink库（完整的MAVLink协议支持）                      │
    │  - 回退到原始UDP socket（简化实现，用于测试）                         │
    └─────────────────────────────────────────────────────────────────┘

扩展方式：
    子类可以继承MavlinkUDPInterface并重写以下方法：
    - connect(): 自定义连接逻辑
    - receive_controls(): 自定义消息接收逻辑
    - send_heartbeat(): 自定义心跳逻辑
"""

import socket
import threading
import time
import struct
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .types import ControlSource, StateTarget, StateVector, ControlVector

logger = logging.getLogger(__name__)

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    logger.warning("pymavlink not installed. Using simplified implementation.")


class MavlinkUDPInterface:
    """
    MAVLink UDP通信接口类。
    
    提供与外部MAVLink系统的双向通信能力：
    1. 接收：从外部接收MAVLink控制消息，解析为控制量
    2. 发送：将Plant状态反馈发送到外部MAVLink系统
    
    这是一个具体的父类实现，子类可以通过继承扩展功能。
    
    属性:
        _host: str - 监听主机地址
        _port: int - 监听端口号
        _source_system: int - 源系统ID（MAVLink system ID）
        _source_component: int - 源组件ID（MAVLink component ID）
        _target_system: int - 目标系统ID
        _target_component: int - 目标组件ID
        _mavutil_connection: 可选 - pymavlink连接对象
        _socket: 可选 - 原始UDP socket对象
        _target_addr: 可选 - 目标地址(host, port)
        _is_connected: bool - 是否已连接
        _is_running: bool - 是否运行中
        _receive_thread: 可选 - 接收线程
        _sequence: int - 消息序列号
        _stats: Dict[str, int] - 统计数据
        _last_messages: Dict[ControlSource, List[float]] - 最近接收的消息
        _callbacks: Dict[ControlSource, List[callable]] - 回调函数列表
    
    示例:
        >>> # 创建接口
        >>> mavlink = MavlinkUDPInterface(host="0.0.0.0", port=14540)
        >>> mavlink.connect()
        >>> mavlink.start()
        
        >>> # 接收控制量
        >>> controls = mavlink.receive_controls()
        >>> if controls:
        ...     for source, values in controls.items():
        ...         print(f"Received {source}: {values}")
        
        >>> # 发送状态反馈
        >>> mavlink.send_hil_actuator_controls([0.1, 0.2, 0.0, ...])
        >>> mavlink.send_heartbeat()
    """
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 14540,
                 source_system: int = 1,
                 source_component: int = 1,
                 target_system: int = 1,
                 target_component: int = 1):
        """
        初始化MAVLink UDP接口。
        
        MAVLink连接参数说明：
        - System ID: 标识一个MAVLink系统（如一架无人机）
        - Component ID: 标识系统中的一个组件（如飞控、GCS等）
        
        常用组件ID：
        - 0: MAV_COMP_ID_ALL
        - 1: MAV_COMP_ID_AUTOPILOT1 (飞控)
        - 12: MAV_COMP_ID_GIMBAL (云台)
        - 19: MAV_COMP_ID_GPS (GPS)
        
        Args:
            host: str - 监听主机地址，"0.0.0.0"表示监听所有接口
            port: int - 监听端口号，默认14540（MAVLink HIL常用端口）
            source_system: int - 源系统ID (1-255)
            source_component: int - 源组件ID (0-255)
            target_system: int - 目标系统ID (1-255)
            target_component: int - 目标组件ID (0-255)
        """
        self._host = host
        """监听主机地址"""
        
        self._port = port
        """监听端口号"""
        
        self._source_system = source_system
        """源系统ID"""
        
        self._source_component = source_component
        """源组件ID"""
        
        self._target_system = target_system
        """目标系统ID"""
        
        self._target_component = target_component
        """目标组件ID"""
        
        self._mavutil_connection = None
        """pymavlink连接对象"""
        
        self._socket = None
        """原始UDP socket对象"""
        
        self._target_addr: Optional[Tuple[str, int]] = None
        """目标地址元组 (host, port)"""
        
        self._is_connected = False
        """ 是否已连接标志"""
        
        self._is_running = False
        """ 是否运行中标志"""
        
        self._receive_thread: Optional[threading.Thread] = None
        """接收线程"""
        
        self._lock = threading.Lock()
        """线程锁"""
        
        self._sequence = 0
        """消息序列号"""
        
        self._stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
        }
        """通信统计数据"""
        
        self._last_messages: Dict[ControlSource, List[float]] = {}
        """最近接收的消息缓存"""
        
        self._callbacks: Dict[ControlSource, List[callable]] = {}
        """回调函数字典，键为ControlSource，值为回调函数列表"""
    
    @property
    def is_connected(self) -> bool:
        """
        是否已连接。
        
        Returns:
            bool - True表示已成功建立连接
        """
        return self._is_connected
    
    @property
    def is_running(self) -> bool:
        """
        是否运行中。
        
        Returns:
            bool - True表示接收线程正在运行
        """
        return self._is_running
    
    def connect(self) -> bool:
        """
        建立MAVLink连接。
        
        会自动选择最佳的连接方式：
        1. 优先使用pymavlink库（完整协议支持）
        2. 如果pymavlink不可用，回退到原始UDP socket
        
        子类可以重写此方法以自定义连接逻辑。
        
        Returns:
            bool - 连接成功返回True，失败返回False
            
        示例:
            >>> mavlink = MavlinkUDPInterface()
            >>> if mavlink.connect():
            ...     print("Connected!")
            ... else:
            ...     print("Connection failed!")
        """
        if PYMAVLINK_AVAILABLE:
            return self._connect_pymavlink()
        return self._connect_raw_udp()
    
    def _connect_pymavlink(self) -> bool:
        """
        使用pymavlink库建立连接。
        
        pymavlink提供完整的MAVLink协议支持，包括：
        - 消息解析
        - 校验和验证
        - 心跳管理
        
        Returns:
            bool - 连接成功返回True
        """
        try:
            conn_str = f"udp:{self._host}:{self._port}"
            logger.info(f"Connecting via pymavlink: {conn_str}")
            
            self._mavutil_connection = mavutil.mavlink_connection(
                conn_str,
                source_system=self._source_system,
                source_component=self._source_component
            )
            
            self._is_connected = True
            logger.info("Connected via pymavlink")
            return True
            
        except Exception as e:
            logger.warning(f"pymavlink connection failed: {e}, falling back to raw UDP")
            return self._connect_raw_udp()
    
    def _connect_raw_udp(self) -> bool:
        """
        使用原始UDP socket建立连接。
        
        这是一个简化实现，不依赖pymavlink库，适用于：
        - 开发测试
        - pymavlink不可用的环境
        
        注意：原始UDP实现可能不支持完整的MAVLink协议。
        
        Returns:
            bool - 连接成功返回True
        """
        try:
            logger.info(f"Connecting via raw UDP: {self._host}:{self._port}")
            
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(0.1)
            
            if self._host == "0.0.0.0":
                self._socket.bind(("0.0.0.0", self._port))
                logger.info(f"UDP server bound to 0.0.0.0:{self._port}")
            else:
                self._target_addr = (self._host, self._port)
                logger.info(f"UDP client configured for {self._host}:{self._port}")
            
            self._is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Raw UDP connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """
        断开MAVLink连接。
        
        会关闭所有打开的连接和socket。
        """
        self._is_connected = False
        
        if self._mavutil_connection:
            self._mavutil_connection = None
        
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        
        logger.info("MAVLink interface disconnected")
    
    def start(self) -> None:
        """
        启动接收线程。
        
        启动后台线程持续接收MAVLink消息。
        必须在connect()之后调用。
        
        Raises:
            RuntimeError: 未连接时调用会抛出异常
        """
        if not self._is_connected:
            raise RuntimeError("Cannot start: not connected")
        
        if self._is_running:
            return
        
        self._is_running = True
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
        logger.info("MAVLink interface started")
    
    def stop(self) -> None:
        """
        停止接收线程。
        """
        self._is_running = False
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None
        
        logger.info("MAVLink interface stopped")
    
    def _receive_loop(self) -> None:
        """
        接收消息的主循环（在线程中运行）。
        
        持续调用receive_controls()接收消息，并：
        1. 缓存最近接收的消息
        2. 触发注册的回调函数
        """
        while self._is_running:
            try:
                controls = self.receive_controls()
                if controls:
                    for source, values in controls.items():
                        with self._lock:
                            self._last_messages[source] = values
                        
                        if source in self._callbacks:
                            for callback in self._callbacks[source]:
                                try:
                                    callback(values)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                
                time.sleep(0.001)
                
            except Exception as e:
                logger.debug(f"Receive loop error: {e}")
                time.sleep(0.01)
    
    def receive_controls(self) -> Optional[Dict[ControlSource, List[float]]]:
        """
        接收并解析MAVLink控制消息。
        
        非阻塞调用，如果没有消息则返回None。
        
        支持解析的消息类型：
        - HIL_ACTUATOR_CONTROLS: 硬件在环执行器控制（16个float）
        - RC_CHANNELS_OVERRIDE: 遥控器通道覆盖（4个归一化通道）
        - MANUAL_CONTROL: 手动控制（4个归一化轴）
        
        子类可以重写此方法以自定义消息解析逻辑。
        
        Returns:
            Optional[Dict[ControlSource, List[float]]] - 解析结果字典，
                键为ControlSource，值为控制量列表；无消息时返回None
                
        示例:
            >>> controls = mavlink.receive_controls()
            >>> if controls:
            ...     for source, values in controls.items():
            ...         print(f"Source: {source}, Values: {values[:4]}")
        """
        if not self._is_connected:
            return None
        
        result = {}
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                msg = self._mavutil_connection.recv_match(blocking=False)
                if msg:
                    msg_type = msg.get_type()
                    
                    if msg_type == "HIL_ACTUATOR_CONTROLS":
                        controls = list(msg.controls)
                        result[ControlSource.HIL_ACTUATOR_CONTROLS] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                    
                    elif msg_type == "RC_CHANNELS_OVERRIDE":
                        controls = [
                            msg.chan1_raw / 1000.0 - 1.0,
                            msg.chan2_raw / 1000.0 - 1.0,
                            msg.chan3_raw / 1000.0 - 1.0,
                            msg.chan4_raw / 1000.0 - 1.0,
                        ]
                        result[ControlSource.RC_CHANNELS] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                    
                    elif msg_type == "MANUAL_CONTROL":
                        controls = [
                            msg.x / 1000.0,
                            msg.y / 1000.0,
                            msg.z / 1000.0,
                            msg.r / 1000.0,
                        ]
                        result[ControlSource.MANUAL_CONTROL] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                
            except Exception as e:
                logger.debug(f"pymavlink receive error: {e}")
        
        elif self._socket:
            try:
                ready = self._poll_socket(0.01)
                if ready:
                    data, addr = self._socket.recvfrom(2048)
                    
                    if self._target_addr is None:
                        self._target_addr = addr
                    
                    controls = self._parse_raw_mavlink(data)
                    if controls:
                        result[ControlSource.HIL_ACTUATOR_CONTROLS] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                            
            except socket.timeout:
                pass
            except Exception as e:
                logger.debug(f"Raw UDP receive error: {e}")
        
        return result if result else None
    
    def _poll_socket(self, timeout: float) -> bool:
        """
        检查socket是否有数据可读。
        
        Args:
            timeout: float - 超时时间（秒）
            
        Returns:
            bool - True表示有数据可读
        """
        import select
        
        if not self._socket:
            return False
        
        ready, _, _ = select.select([self._socket], [], [], timeout)
        return len(ready) > 0
    
    def _parse_raw_mavlink(self, data: bytes) -> Optional[List[float]]:
        """
        解析原始MAVLink二进制数据（简化实现）。
        
        注意：这是一个简化实现，完整解析需要pymavlink。
        
        Args:
            data: bytes - 原始字节数据
            
        Returns:
            Optional[List[float]] - 解析的控制量列表，解析失败返回None
        """
        if len(data) < 8:
            return None
        
        magic = data[0]
        
        if magic == 0xFD:
            pass
        elif magic == 0xFE:
            pass
        
        return None
    
    def send_state(self, state: StateVector, mapping: Optional[Any] = None) -> bool:
        """
        发送状态向量。
        
        从StateVector中提取关节位置，作为控制量反馈发送。
        
        Args:
            state: StateVector - 状态向量
            mapping: Optional[Any] - 映射配置（保留参数）
            
        Returns:
            bool - 发送成功返回True
        """
        controls = []
        
        for name in state.joint_positions.keys():
            pos = state.joint_positions.get(name, 0.0)
            controls.append(pos)
        
        while len(controls) < 16:
            controls.append(0.0)
        controls = controls[:16]
        
        return self.send_hil_actuator_controls(controls)
    
    def send_hil_actuator_controls(self, controls: List[float]) -> bool:
        """
        发送HIL_ACTUATOR_CONTROLS消息。
        
        这条消息用于：
        1. 从GCS向仿真器发送控制量（输入方向）
        2. 从仿真器向GCS反馈状态（输出方向，本框架用法）
        
        消息格式：
        - time_usec: uint64 - 时间戳（微秒）
        - controls: float[16] - 16个控制量
        - flags: uint8 - 标志位
        - mode: uint8 - 模式
        
        Args:
            controls: List[float] - 控制量列表，最多16个，不足则补0
            
        Returns:
            bool - 发送成功返回True
            
        示例:
            >>> # 发送2个关节的位置反馈
            >>> mavlink.send_hil_actuator_controls([0.5, 0.3])
        """
        if not self._is_connected:
            return False
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                while len(controls) < 16:
                    controls.append(0.0)
                controls = controls[:16]
                
                self._mavutil_connection.mav.hil_actuator_controls_send(
                    int(time.time() * 1e6),
                    controls,
                    0,
                    0
                )
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return True
                
            except Exception as e:
                logger.warning(f"pymavlink send error: {e}")
                return False
        
        elif self._socket and self._target_addr:
            try:
                msg = self._create_hil_actuator_controls_message(controls)
                sent = self._socket.sendto(msg, self._target_addr)
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return sent == len(msg)
                
            except Exception as e:
                logger.debug(f"Raw UDP send error: {e}")
                return False
        
        return False
    
    def _create_hil_actuator_controls_message(self, controls: List[float]) -> bytes:
        """
        创建原始HIL_ACTUATOR_CONTROLS消息字节流。
        
        MAVLink v2消息格式：
        - magic: 0xFD (1字节)
        - length: 载荷长度 (1字节)
        - incompat_flags: 不兼容标志 (1字节)
        - compat_flags: 兼容标志 (1字节)
        - seq: 序列号 (1字节)
        - sysid: 系统ID (1字节)
        - compid: 组件ID (1字节)
        - msgid: 消息ID (3字节，小端)
        - payload: 载荷数据
        - checksum: 校验和 (2字节)
        
        HIL_ACTUATOR_CONTROLS消息ID: 93
        载荷格式: time_usec(8) + controls[16](64) + flags(1) + mode(8)
        
        Args:
            controls: List[float] - 控制量列表
            
        Returns:
            bytes - 完整的MAVLink消息字节流
        """
        data = bytearray()
        
        data.append(0xFD)
        data.append(64)
        data.append(0)
        data.append(0)
        
        with self._lock:
            data.append(self._sequence & 0xFF)
            self._sequence = (self._sequence + 1) & 0xFF
        
        data.append(self._source_system)
        data.append(self._source_component)
        
        msg_id = 93
        data.append(msg_id & 0xFF)
        data.append((msg_id >> 8) & 0xFF)
        data.append((msg_id >> 16) & 0xFF)
        
        time_usec = int(time.time() * 1e6)
        data.extend(time_usec.to_bytes(8, byteorder='little'))
        
        for i in range(16):
            val = controls[i] if i < len(controls) else 0.0
            float_bytes = struct.pack('<f', val)
            data.extend(float_bytes)
        
        data.append(0)
        data.extend((0).to_bytes(8, byteorder='little'))
        
        checksum = self._calculate_checksum(bytes(data[1:]))
        data.extend(checksum.to_bytes(2, byteorder='little'))
        
        return bytes(data)
    
    def _calculate_checksum(self, data: bytes) -> int:
        """
        计算MAVLink校验和（X.25 CRC-16）。
        
        算法说明：
        - 使用CRC-16/XMODEM算法
        - 初始值: 0xFFFF
        - 多项式: 0x1021
        
        Args:
            data: bytes - 需要计算校验和的数据（从magic字节之后开始）
            
        Returns:
            int - 16位校验和值
        """
        checksum = 0xFFFF
        for byte in data:
            tmp = byte ^ (checksum & 0xFF)
            tmp ^= (tmp << 4) & 0xFF
            checksum = (checksum >> 8) ^ (tmp << 8) ^ ((tmp << 3) & 0xFFFF) ^ (tmp >> 4)
            checksum = checksum & 0xFFFF
        return checksum
    
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
        """
        发送HIL_STATE_QUATERNION消息。
        
        用于发送完整的飞行器状态，包括：
        - 姿态（四元数）
        - 角速度
        - GPS位置
        - 速度
        - 加速度
        
        消息格式：
        - time_usec: uint64 - 时间戳
        - attitude_quaternion: float[4] - 四元数 [w, x, y, z]
        - rollspeed: float - 滚转角速度 (rad/s)
        - pitchspeed: float - 俯仰角速度 (rad/s)
        - yawspeed: float - 偏航角速度 (rad/s)
        - lat: int32 - 纬度 (1e7度)
        - lon: int32 - 经度 (1e7度)
        - alt: int32 - 高度 (毫米)
        - vx/vy/vz: float - 速度 (m/s)
        - xacc/yacc/zacc: float - 加速度 (m/s²)
        
        Args:
            attitude: List[float] - 四元数 [w, x, y, z]
            rollspeed: float - 滚转角速度
            pitchspeed: float - 俯仰角速度
            yawspeed: float - 偏航角速度
            lat: int - 纬度 (1e7度)
            lon: int - 经度 (1e7度)
            alt: int - 高度 (毫米)
            vx: float - X方向速度
            vy: float - Y方向速度
            vz: float - Z方向速度
            xacc: float - X方向加速度
            yacc: float - Y方向加速度
            zacc: float - Z方向加速度
            
        Returns:
            bool - 发送成功返回True
        """
        if not self._is_connected:
            return False
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                while len(attitude) < 4:
                    attitude.append(0.0)
                attitude = attitude[:4]
                
                self._mavutil_connection.mav.hil_state_quaternion_send(
                    int(time.time() * 1e6),
                    attitude,
                    rollspeed,
                    pitchspeed,
                    yawspeed,
                    lat,
                    lon,
                    alt,
                    vx,
                    vy,
                    vz,
                    xacc,
                    yacc,
                    zacc
                )
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return True
                
            except Exception as e:
                logger.warning(f"pymavlink send error: {e}")
                return False
        
        return False
    
    def send_heartbeat(self) -> bool:
        """
        发送HEARTBEAT消息。
        
        心跳消息用于告知GCS本系统仍在运行。
        通常建议以1Hz的频率发送。
        
        消息格式：
        - type: uint8 - 系统类型
        - autopilot: uint8 - 飞控类型
        - base_mode: uint8 - 基础模式
        - custom_mode: uint32 - 自定义模式
        - system_status: uint8 - 系统状态
        
        常用值：
        - type: 2 = MAV_TYPE_QUADROTOR (四旋翼)
        - autopilot: 12 = MAV_AUTOPILOT_INVALID (无效/仿真)
        - system_status: 4 = MAV_STATE_ACTIVE (运行中)
        
        子类可以重写此方法以自定义心跳内容。
        
        Returns:
            bool - 发送成功返回True
        """
        if not self._is_connected:
            return False
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                self._mavutil_connection.mav.heartbeat_send(
                    2,
                    12,
                    0,
                    0,
                    4
                )
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return True
                
            except Exception as e:
                logger.warning(f"pymavlink heartbeat error: {e}")
                return False
        
        elif self._socket and self._target_addr:
            try:
                msg = self._create_heartbeat_message()
                sent = self._socket.sendto(msg, self._target_addr)
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return sent == len(msg)
                
            except Exception as e:
                logger.debug(f"Raw UDP heartbeat error: {e}")
                return False
        
        return False
    
    def _create_heartbeat_message(self) -> bytes:
        """
        创建原始HEARTBEAT消息字节流。
        
        HEARTBEAT消息ID: 0
        载荷格式: custom_mode(4) + type(1) + autopilot(1) + 
                  base_mode(1) + system_status(1) + mavlink_version(1)
        
        Returns:
            bytes - 完整的MAVLink消息字节流
        """
        data = bytearray()
        
        data.append(0xFD)
        data.append(9)
        data.append(0)
        data.append(0)
        
        with self._lock:
            data.append(self._sequence & 0xFF)
            self._sequence = (self._sequence + 1) & 0xFF
        
        data.append(self._source_system)
        data.append(self._source_component)
        
        msg_id = 0
        data.append(msg_id & 0xFF)
        data.append((msg_id >> 8) & 0xFF)
        data.append((msg_id >> 16) & 0xFF)
        
        data.extend((0).to_bytes(4, byteorder='little'))
        data.append(2)
        data.append(12)
        data.append(0)
        data.append(4)
        data.append(3)
        
        checksum = self._calculate_checksum(bytes(data[1:]))
        data.extend(checksum.to_bytes(2, byteorder='little'))
        
        return bytes(data)
    
    def get_last_controls(self, source: ControlSource) -> Optional[List[float]]:
        """
        获取最近接收的控制量。
        
        Args:
            source: ControlSource - 消息来源类型
            
        Returns:
            Optional[List[float]] - 最近的控制量列表，不存在时返回None
        """
        with self._lock:
            return self._last_messages.get(source)
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取通信统计数据。
        
        返回的统计数据包括：
        - messages_received: 接收消息数
        - messages_sent: 发送消息数
        - errors: 错误次数
        
        Returns:
            Dict[str, int] - 统计数据字典的副本
        """
        with self._lock:
            return self._stats.copy()
    
    def register_callback(self, source: ControlSource, callback: callable) -> None:
        """
        注册控制消息回调函数。
        
        当接收到指定类型的消息时，会自动调用回调函数。
        
        Args:
            source: ControlSource - 消息来源类型
            callback: callable - 回调函数，签名为 callback(values: List[float]) -> None
            
        示例:
            >>> def on_controls(values):
            ...     print(f"Received controls: {values}")
            >>> mavlink.register_callback(ControlSource.HIL_ACTUATOR_CONTROLS, on_controls)
        """
        if source not in self._callbacks:
            self._callbacks[source] = []
        self._callbacks[source].append(callback)
    
    def unregister_callback(self, source: ControlSource, callback: callable) -> None:
        """
        注销控制消息回调函数。
        
        Args:
            source: ControlSource - 消息来源类型
            callback: callable - 要注销的回调函数
        """
        if source in self._callbacks:
            if callback in self._callbacks[source]:
                self._callbacks[source].remove(callback)
