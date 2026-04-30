"""
测试 Simulator 类和核心数据类型

Simulator 是主仿真器类，协调 Plant 和 MAVLink 接口的交互。

核心功能：
- connect(): 连接 MAVLink 接口
- start(): 启动仿真
- stop(): 停止仿真
- step(): 执行仿真步进
- reset(): 重置仿真
- get_state(): 获取状态
- set_control(): 设置控制量

架构：
    MAVLink (外部)
        |
        ▼ 控制量输入 (u)
    ┌─────────────────────────────────┐
    │           Simulator              │
    │  ┌─────────┐  ┌─────────┐       │
    │  │ Control │  │ MuJoCo  │       │
    │  │ Mapping │◄─►│  Plant  │       │
    │  └─────────┘  └─────────┘       │
    └─────────────────────────────────┘
        |
        ▼ 状态输出 (x)
    MAVLink (外部)

输出示例：
    [OUTPUT] Simulator initialized: plant.control_dim=2
    [OUTPUT] SimulatorConfig: mavlink_port=18888, real_time_factor=100.0
    [OUTPUT] ControlVector: values=[1.0, 0.5], names=['joint1', 'joint2']
    [OUTPUT] StateVector: time=0.0, joint_positions={'joint1': 0.0, 'joint2': 0.0}
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestSimulatorConfig:
    """
    测试 SimulatorConfig 配置类
    """
    
    def test_default_config(self, simulator_config):
        """
        测试默认配置
        
        验证：
        1. 默认配置正确
        2. 所有属性可访问
        
        输出：
            [TEST] default_config:
            [TEST]   mavlink_host: 127.0.0.1
            [TEST]   mavlink_port: 18888
            [TEST]   real_time_factor: 100.0
            [TEST]   enable_telemetry: True
            [TEST]   telemetry_interval: 0.1
            [TEST]   heartbeat_interval: 1.0
        """
        config = simulator_config
        
        print(f"\n  [TEST] test_default_config:")
        print(f"    mavlink_host: {config.mavlink_host}")
        print(f"    mavlink_port: {config.mavlink_port}")
        print(f"    real_time_factor: {config.real_time_factor}")
        print(f"    enable_telemetry: {config.enable_telemetry}")
        print(f"    telemetry_interval: {config.telemetry_interval}")
        print(f"    heartbeat_interval: {config.heartbeat_interval}")
        
        assert isinstance(config.mavlink_port, int)
        assert isinstance(config.real_time_factor, float)
        assert isinstance(config.enable_telemetry, bool)
    
    def test_custom_config(self):
        """
        测试自定义配置
        
        验证：
        1. 自定义参数正确应用
        
        输出：
            [TEST] custom_config:
            [TEST]   mavlink_port: 17777
            [TEST]   real_time_factor: 200.0
            [TEST]   enable_telemetry: False
        """
        from src import SimulatorConfig
        
        print(f"\n  [TEST] test_custom_config:")
        
        config = SimulatorConfig(
            mavlink_port=17777,
            real_time_factor=200.0,
            enable_telemetry=False
        )
        
        print(f"    mavlink_port: {config.mavlink_port}")
        print(f"    real_time_factor: {config.real_time_factor}")
        print(f"    enable_telemetry: {config.enable_telemetry}")
        
        assert config.mavlink_port == 17777
        assert config.real_time_factor == 200.0
        assert config.enable_telemetry is False


class TestSimulatorInitialization:
    """
    测试 Simulator 初始化
    """
    
    def test_default_creation(self, simulator):
        """
        测试默认参数创建
        
        验证：
        1. Simulator 正确初始化
        2. 内部组件（plant, mavlink_interface, control_mapping）正确创建
        
        输出：
            [TEST] default_creation:
            [TEST]   is_running: False
            [TEST]   plant.control_dim: 2
            [TEST]   plant.control_names: ['joint1', 'joint2']
            [TEST]   control_mapping entries: 2
            [TEST]   mavlink_interface.is_connected: False
        """
        sim = simulator
        
        print(f"\n  [TEST] test_default_creation:")
        print(f"    is_running: {sim.is_running}")
        print(f"    plant.control_dim: {sim.plant.control_dim}")
        print(f"    plant.control_names: {sim.plant.control_names}")
        print(f"    control_mapping entries: {len(sim.control_mapping.entries)}")
        print(f"    mavlink_interface.is_connected: {sim.mavlink_interface.is_connected}")
        
        assert sim.is_running is False
        assert sim.plant.control_dim > 0
        assert len(sim.control_mapping.entries) > 0
        assert sim.mavlink_interface.is_connected is False
    
    def test_properties(self, simulator):
        """
        测试属性访问
        
        验证：
        1. 所有属性正确返回
        
        输出：
            [TEST] properties:
            [TEST]   plant is not None: True
            [TEST]   mavlink_interface is not None: True
            [TEST]   control_mapping is not None: True
            [TEST]   config is not None: True
            [TEST]   is_running: False
        """
        sim = simulator
        
        print(f"\n  [TEST] test_properties:")
        print(f"    plant is not None: {sim.plant is not None}")
        print(f"    mavlink_interface is not None: {sim.mavlink_interface is not None}")
        print(f"    control_mapping is not None: {sim.control_mapping is not None}")
        print(f"    config is not None: {sim.config is not None}")
        print(f"    is_running: {sim.is_running}")
        
        assert sim.plant is not None
        assert sim.mavlink_interface is not None
        assert sim.control_mapping is not None
        assert sim.config is not None
        assert sim.is_running is False


class TestSimulatorControl:
    """
    测试 Simulator 控制量设置
    """
    
    def test_set_control_by_index(self, simulator):
        """
        测试 set_control_by_index() - 按索引设置控制量
        
        验证：
        1. 控制量正确设置到 Plant
        
        输出：
            [TEST] set_control_by_index:
            [TEST]   plant.control_dim: 2
            [TEST]   set_control_by_index(0, 1.0)
            [TEST]   set_control_by_index(1, 0.5)
            [TEST]   get_control values: [1.0, 0.5]
        """
        sim = simulator
        
        print(f"\n  [TEST] test_set_control_by_index:")
        print(f"    plant.control_dim: {sim.plant.control_dim}")
        
        print(f"    set_control_by_index(0, 1.0)")
        sim.set_control_by_index(0, 1.0)
        
        print(f"    set_control_by_index(1, 0.5)")
        sim.set_control_by_index(1, 0.5)
        
        control = sim.plant.get_control()
        print(f"    get_control values: {control.to_list()}")
        
        assert control[0] == 1.0
        assert control[1] == 0.5
    
    def test_set_control_by_name(self, simulator):
        """
        测试 set_control_by_name() - 按名称设置控制量
        
        验证：
        1. 按名称设置控制量
        
        输出：
            [TEST] set_control_by_name:
            [TEST]   control_names: ['joint1', 'joint2']
            [TEST]   set_control_by_name('joint1', 0.8)
            [TEST]   get_control values: [0.8, 0.0]
        """
        sim = simulator
        
        print(f"\n  [TEST] test_set_control_by_name:")
        print(f"    control_names: {sim.plant.control_names}")
        
        if sim.plant.control_names:
            first_name = sim.plant.control_names[0]
            print(f"    set_control_by_name('{first_name}', 0.8)")
            sim.set_control_by_name(first_name, 0.8)
            
            control = sim.plant.get_control()
            print(f"    get_control values: {control.to_list()}")
            
            assert control[0] == 0.8
    
    def test_set_control_vector(self, simulator):
        """
        测试 set_control() - 使用 ControlVector 设置
        
        验证：
        1. ControlVector 正确应用
        
        输出：
            [TEST] set_control_vector:
            [TEST]   set_control with values: [0.3, 0.7]
            [TEST]   get_control values: [0.3, 0.7]
        """
        from src import ControlVector
        import numpy as np
        
        sim = simulator
        
        print(f"\n  [TEST] test_set_control_vector:")
        
        values = [0.3, 0.7]
        print(f"    set_control with values: {values}")
        
        cv = ControlVector(values=np.array(values), names=sim.plant.control_names[:2])
        sim.set_control(cv)
        
        control = sim.plant.get_control()
        print(f"    get_control values: {control.to_list()}")
        
        assert control[0] == 0.3
        assert control[1] == 0.7


class TestSimulatorState:
    """
    测试 Simulator 状态获取
    """
    
    def test_get_state(self, simulator):
        """
        测试 get_state() - 获取状态向量
        
        验证：
        1. 返回正确的 StateVector 类型
        2. 包含关节位置、速度等信息
        
        输出：
            [TEST] get_state:
            [TEST]   time: 0.0
            [TEST]   joint_positions: {'joint1': 0.0, 'joint2': 0.0}
            [TEST]   joint_velocities: {'joint1': 0.0, 'joint2': 0.0}
            [TEST]   body_positions count: 3
        """
        sim = simulator
        
        print(f"\n  [TEST] test_get_state:")
        
        state = sim.get_state()
        
        print(f"    time: {state.time}")
        print(f"    joint_positions: {state.joint_positions}")
        print(f"    joint_velocities: {state.joint_velocities}")
        print(f"    body_positions count: {len(state.body_positions)}")
        
        assert state is not None
        assert isinstance(state.time, float)
        assert isinstance(state.joint_positions, dict)
        assert isinstance(state.joint_velocities, dict)


class TestSimulatorReset:
    """
    测试 Simulator 重置功能
    """
    
    def test_reset(self, simulator):
        """
        测试 reset() - 重置仿真
        
        验证：
        1. 执行仿真后重置
        2. 状态恢复到初始值
        
        输出：
            [TEST] reset:
            [TEST]   set_control(0, 1.0)
            [TEST]   step 100 times...
            [TEST]   before reset:
            [TEST]     time: 1.0
            [TEST]   after reset:
            [TEST]     time: 0.0
        """
        sim = simulator
        
        print(f"\n  [TEST] test_reset:")
        
        print(f"    set_control(0, 1.0)")
        sim.set_control_by_index(0, 1.0)
        
        print(f"    step 100 times...")
        for _ in range(100):
            sim.plant.step(n_steps=1)
        
        state_before_reset = sim.get_state()
        print(f"    before reset:")
        print(f"      time: {state_before_reset.time}")
        
        sim.reset()
        
        state_after_reset = sim.get_state()
        print(f"    after reset:")
        print(f"      time: {state_after_reset.time}")
        
        assert state_after_reset.time == 0.0


class TestSimulatorStatistics:
    """
    测试 Simulator 统计功能
    """
    
    def test_get_statistics_initial(self, simulator):
        """
        测试初始统计数据
        
        验证：
        1. 初始统计数据正确
        
        输出：
            [TEST] get_statistics_initial:
            [TEST]   steps: 0
            [TEST]   controls_received: 0
            [TEST]   states_sent: 0
            [TEST]   heartbeats_sent: 0
            [TEST]   is_running: False
            [TEST]   is_connected: False
        """
        sim = simulator
        
        print(f"\n  [TEST] test_get_statistics_initial:")
        
        stats = sim.get_statistics()
        print(f"    steps: {stats['steps']}")
        print(f"    controls_received: {stats['controls_received']}")
        print(f"    states_sent: {stats['states_sent']}")
        print(f"    heartbeats_sent: {stats['heartbeats_sent']}")
        print(f"    is_running: {stats['is_running']}")
        print(f"    is_connected: {stats['is_connected']}")
        
        assert stats["steps"] == 0
        assert stats["controls_received"] == 0
        assert stats["is_running"] is False
        assert stats["is_connected"] is False


class TestSimulatorExtensible:
    """
    测试 Simulator 可扩展性 - 作为父类继承
    """
    
    def test_inheritance(self):
        """
        测试继承 Simulator 并扩展功能
        
        验证：
        1. 可以继承 Simulator
        2. 可以重写方法
        3. 可以添加自定义方法
        
        输出：
            [TEST] inheritance:
            [TEST]   creating CustomSimulator instance...
            [TEST]   CustomSimulator.plant.control_dim: 2
            [TEST]   CustomSimulator.is_running: False
            [TEST]   CustomSimulator.custom_method() called
        """
        from src import Simulator, SimulatorConfig
        
        class CustomSimulator(Simulator):
            """
            自定义仿真器，继承 Simulator
            
            展示如何扩展父类功能：
            - 添加自定义方法
            - 重写现有方法
            """
            
            def __init__(self):
                config = SimulatorConfig(
                    mavlink_port=16666,
                    real_time_factor=200.0
                )
                super().__init__(config=config)
                self.custom_counter = 0
            
            def custom_method(self):
                """自定义方法"""
                self.custom_counter += 1
                return f"Custom method called, counter={self.custom_counter}"
        
        print(f"\n  [TEST] test_inheritance:")
        print(f"    creating CustomSimulator instance...")
        
        custom_sim = CustomSimulator()
        
        print(f"    CustomSimulator.plant.control_dim: {custom_sim.plant.control_dim}")
        print(f"    CustomSimulator.is_running: {custom_sim.is_running}")
        
        result = custom_sim.custom_method()
        print(f"    CustomSimulator.custom_method() called")
        
        assert custom_sim.custom_counter == 1
        assert custom_sim.plant.control_dim > 0
        
        custom_sim.stop()


class TestControlVectorBasics:
    """
    测试 ControlVector 数据类型
    """
    
    def test_creation(self, control_vector):
        """
        测试 ControlVector 创建
        
        验证：
        1. 正确创建 ControlVector
        2. 可访问 values 和 names
        
        输出：
            [TEST] creation:
            [TEST]   values: [1.0, 0.5, -0.3, 0.0]
            [TEST]   names: ['joint1', 'joint2', 'joint3', 'joint4']
            [TEST]   len: 4
        """
        cv = control_vector
        
        print(f"\n  [TEST] test_ControlVector_creation:")
        print(f"    values: {cv.to_list()}")
        print(f"    names: {cv.names}")
        print(f"    len: {len(cv)}")
        
        assert len(cv) == 4
        assert cv.names is not None
    
    def test_index_access(self, control_vector):
        """
        测试索引访问
        
        验证：
        1. 可以通过索引访问值
        
        输出：
            [TEST] index_access:
            [TEST]   cv[0]: 1.0
            [TEST]   cv[1]: 0.5
            [TEST]   cv[2]: -0.3
        """
        cv = control_vector
        
        print(f"\n  [TEST] test_ControlVector_index_access:")
        print(f"    cv[0]: {cv[0]}")
        print(f"    cv[1]: {cv[1]}")
        print(f"    cv[2]: {cv[2]}")
        
        assert cv[0] == 1.0
        assert cv[1] == 0.5
        assert cv[2] == -0.3
    
    def test_to_list(self, control_vector):
        """
        测试 to_list() 方法
        
        验证：
        1. 返回正确的列表
        
        输出：
            [TEST] to_list:
            [TEST]   to_list(): [1.0, 0.5, -0.3, 0.0]
        """
        cv = control_vector
        
        print(f"\n  [TEST] test_ControlVector_to_list:")
        
        values = cv.to_list()
        print(f"    to_list(): {values}")
        
        assert values == [1.0, 0.5, -0.3, 0.0]


class TestStateVectorBasics:
    """
    测试 StateVector 数据类型
    """
    
    def test_creation(self, state_vector):
        """
        测试 StateVector 创建
        
        验证：
        1. 正确创建 StateVector
        2. 可访问所有属性
        
        输出：
            [TEST] creation:
            [TEST]   time: 0.5
            [TEST]   joint_positions: {'joint1': 0.5, 'joint2': 1.0}
            [TEST]   joint_velocities: {'joint1': 0.1, 'joint2': -0.2}
            [TEST]   joint_names: ['joint1', 'joint2']
        """
        sv = state_vector
        
        print(f"\n  [TEST] test_StateVector_creation:")
        print(f"    time: {sv.time}")
        print(f"    joint_positions: {sv.joint_positions}")
        print(f"    joint_velocities: {sv.joint_velocities}")
        print(f"    joint_names: {sv.joint_names}")
        
        assert sv.time == 0.5
        assert "joint1" in sv.joint_positions
        assert "joint2" in sv.joint_positions
    
    def test_get_joint_position(self, state_vector):
        """
        测试 get_joint_position() 方法
        
        验证：
        1. 正确返回关节位置
        
        输出：
            [TEST] get_joint_position:
            [TEST]   get_joint_position('joint1'): 0.5
            [TEST]   get_joint_position('joint2'): 1.0
        """
        sv = state_vector
        
        print(f"\n  [TEST] test_StateVector_get_joint_position:")
        
        pos1 = sv.get_joint_position("joint1")
        pos2 = sv.get_joint_position("joint2")
        
        print(f"    get_joint_position('joint1'): {pos1}")
        print(f"    get_joint_position('joint2'): {pos2}")
        
        assert pos1 == 0.5
        assert pos2 == 1.0
    
    def test_get_joint_velocity(self, state_vector):
        """
        测试 get_joint_velocity() 方法
        
        验证：
        1. 正确返回关节速度
        
        输出：
            [TEST] get_joint_velocity:
            [TEST]   get_joint_velocity('joint1'): 0.1
            [TEST]   get_joint_velocity('joint2'): -0.2
        """
        sv = state_vector
        
        print(f"\n  [TEST] test_StateVector_get_joint_velocity:")
        
        vel1 = sv.get_joint_velocity("joint1")
        vel2 = sv.get_joint_velocity("joint2")
        
        print(f"    get_joint_velocity('joint1'): {vel1}")
        print(f"    get_joint_velocity('joint2'): {vel2}")
        
        assert vel1 == 0.1
        assert vel2 == -0.2


class TestEnums:
    """
    测试枚举类型
    """
    
    def test_control_source_enum(self, control_source_enum):
        """
        测试 ControlSource 枚举
        
        验证：
        1. 枚举值正确
        
        输出：
            [TEST] control_source_enum:
            [TEST]   HIL_ACTUATOR_CONTROLS: hil_actuator_controls
            [TEST]   RC_CHANNELS: rc_channels
            [TEST]   MANUAL_CONTROL: manual_control
            [TEST]   CUSTOM: custom
        """
        ControlSource = control_source_enum
        
        print(f"\n  [TEST] test_ControlSource_enum:")
        print(f"    HIL_ACTUATOR_CONTROLS: {ControlSource.HIL_ACTUATOR_CONTROLS.value}")
        print(f"    RC_CHANNELS: {ControlSource.RC_CHANNELS.value}")
        print(f"    MANUAL_CONTROL: {ControlSource.MANUAL_CONTROL.value}")
        print(f"    CUSTOM: {ControlSource.CUSTOM.value}")
        
        assert ControlSource.HIL_ACTUATOR_CONTROLS.value == "hil_actuator_controls"
        assert ControlSource.RC_CHANNELS.value == "rc_channels"
        assert ControlSource.MANUAL_CONTROL.value == "manual_control"
        assert ControlSource.CUSTOM.value == "custom"
    
    def test_state_target_enum(self, state_target_enum):
        """
        测试 StateTarget 枚举
        
        验证：
        1. 枚举值正确
        
        输出：
            [TEST] state_target_enum:
            [TEST]   HIL_STATE: hil_state
            [TEST]   HIL_STATE_QUATERNION: hil_state_quaternion
            [TEST]   HIL_GPS: hil_gps
            [TEST]   HIL_SENSOR: hil_sensor
            [TEST]   CUSTOM: custom
        """
        StateTarget = state_target_enum
        
        print(f"\n  [TEST] test_StateTarget_enum:")
        print(f"    HIL_STATE: {StateTarget.HIL_STATE.value}")
        print(f"    HIL_STATE_QUATERNION: {StateTarget.HIL_STATE_QUATERNION.value}")
        print(f"    HIL_GPS: {StateTarget.HIL_GPS.value}")
        print(f"    HIL_SENSOR: {StateTarget.HIL_SENSOR.value}")
        print(f"    CUSTOM: {StateTarget.CUSTOM.value}")
        
        assert StateTarget.HIL_STATE.value == "hil_state"
        assert StateTarget.HIL_STATE_QUATERNION.value == "hil_state_quaternion"
        assert StateTarget.HIL_GPS.value == "hil_gps"
        assert StateTarget.HIL_SENSOR.value == "hil_sensor"
        assert StateTarget.CUSTOM.value == "custom"
