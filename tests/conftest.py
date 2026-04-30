"""
pytest fixtures 配置 - 适配新架构

新架构的核心类：
- ControlVector, StateVector (数据类型)
- ControlSource, StateTarget (枚举)
- ControlMappingEntry, ControlMapping (控制映射)
- MuJoCoPlant (被控对象 - 可继承作为父类)
- MavlinkUDPInterface (MAVLink接口 - 可继承作为父类)
- Simulator, SimulatorConfig (仿真器)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def control_vector():
    """
    Fixture: 创建一个测试用的 ControlVector 实例
    
    ControlVector 用于表示 Plant 的控制量输入。
    
    输出示例:
        [OUTPUT] ControlVector: values=[1.0, 0.5, -0.3, 0.0], names=['joint1', 'joint2', 'joint3', 'joint4']
    """
    from src import ControlVector
    import numpy as np
    
    cv = ControlVector(
        values=np.array([1.0, 0.5, -0.3, 0.0]),
        names=['joint1', 'joint2', 'joint3', 'joint4']
    )
    
    print(f"\n  [FIXTURE] ControlVector created:")
    print(f"    values: {cv.to_list()}")
    print(f"    names: {cv.names}")
    
    yield cv


@pytest.fixture
def state_vector():
    """
    Fixture: 创建一个测试用的 StateVector 实例
    
    StateVector 用于表示 Plant 的状态输出。
    
    输出示例:
        [OUTPUT] StateVector: time=0.5, joints=['joint1', 'joint2']
    """
    from src import StateVector
    import numpy as np
    
    sv = StateVector(
        time=0.5,
        joint_positions={'joint1': 0.5, 'joint2': 1.0},
        joint_velocities={'joint1': 0.1, 'joint2': -0.2},
        body_positions={'base': np.zeros(3), 'end_effector': np.array([0.0, 0.0, 0.5])},
        body_velocities={'base': np.zeros(6), 'end_effector': np.zeros(6)},
        sensor_data={}
    )
    
    print(f"\n  [FIXTURE] StateVector created:")
    print(f"    time: {sv.time}")
    print(f"    joint_positions: {sv.joint_positions}")
    
    yield sv


@pytest.fixture
def control_mapping():
    """
    Fixture: 创建一个测试用的 ControlMapping 实例
    
    ControlMapping 用于将 MAVLink 消息映射到 Plant 控制量。
    
    输出示例:
        [OUTPUT] ControlMapping created with 4 entries
    """
    from src import ControlMapping, ControlSource
    
    cm = ControlMapping()
    
    cm.create_default_joint_mappings(
        control_names=['joint1', 'joint2', 'joint3', 'joint4'],
        mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
        start_index=0
    )
    
    print(f"\n  [FIXTURE] ControlMapping created:")
    print(f"    entries count: {len(cm.entries)}")
    for entry in cm.entries:
        print(f"      [{entry.mavlink_index}] {entry.plant_control_name} (scale={entry.scale})")
    
    yield cm
    
    cm.clear()


@pytest.fixture
def mujoco_plant():
    """
    Fixture: 创建一个测试用的 MuJoCoPlant 实例
    
    MuJoCoPlant 是核心被控对象类，可继承作为父类扩展。
    
    输出示例:
        [OUTPUT] MuJoCoPlant created: control_dim=2, joint_names=['joint1', 'joint2']
    """
    from src import MuJoCoPlant
    
    plant = MuJoCoPlant()
    
    print(f"\n  [FIXTURE] MuJoCoPlant created:")
    print(f"    control_dim: {plant.control_dim}")
    print(f"    control_names: {plant.control_names}")
    print(f"    joint_names: {plant.joint_names}")
    print(f"    timestep: {plant.timestep}")
    print(f"    is_initialized: {plant.is_initialized}")
    
    yield plant
    
    plant.reset()


@pytest.fixture
def mavlink_interface():
    """
    Fixture: 创建一个测试用的 MavlinkUDPInterface 实例
    
    MavlinkUDPInterface 是核心通信类，可继承作为父类扩展。
    
    输出示例:
        [OUTPUT] MavlinkUDPInterface created: host=127.0.0.1, port=19999
    """
    from src import MavlinkUDPInterface
    
    mavlink = MavlinkUDPInterface(
        host="127.0.0.1",
        port=19999,
        source_system=1,
        source_component=1
    )
    
    print(f"\n  [FIXTURE] MavlinkUDPInterface created:")
    print(f"    host: 127.0.0.1")
    print(f"    port: 19999")
    print(f"    is_connected: {mavlink.is_connected}")
    print(f"    is_running: {mavlink.is_running}")
    
    yield mavlink
    
    if mavlink.is_connected:
        mavlink.disconnect()


@pytest.fixture
def simulator_config():
    """
    Fixture: 创建一个测试用的 SimulatorConfig 实例
    
    输出示例:
        [OUTPUT] SimulatorConfig created: mavlink_port=18888, real_time_factor=100.0
    """
    from src import SimulatorConfig
    
    config = SimulatorConfig(
        mavlink_host="127.0.0.1",
        mavlink_port=18888,
        real_time_factor=100.0,
        enable_telemetry=True
    )
    
    print(f"\n  [FIXTURE] SimulatorConfig created:")
    print(f"    mavlink_host: {config.mavlink_host}")
    print(f"    mavlink_port: {config.mavlink_port}")
    print(f"    real_time_factor: {config.real_time_factor}")
    
    yield config


@pytest.fixture
def simulator(simulator_config):
    """
    Fixture: 创建一个测试用的 Simulator 实例
    
    Simulator 协调 Plant 和 MAVLink 接口。
    
    输出示例:
        [OUTPUT] Simulator created: plant.control_dim=2, mapping_entries=2
    """
    from src import Simulator
    
    sim = Simulator(config=simulator_config)
    
    print(f"\n  [FIXTURE] Simulator created:")
    print(f"    plant.control_dim: {sim.plant.control_dim}")
    print(f"    control_mapping entries: {len(sim.control_mapping.entries)}")
    print(f"    is_running: {sim.is_running}")
    
    yield sim
    
    if sim.is_running:
        sim.stop()


@pytest.fixture
def control_source_enum():
    """
    Fixture: 返回 ControlSource 枚举
    
    ControlSource 定义 MAVLink 控制消息来源类型。
    """
    from src import ControlSource
    
    print(f"\n  [FIXTURE] ControlSource enum:")
    print(f"    HIL_ACTUATOR_CONTROLS = {ControlSource.HIL_ACTUATOR_CONTROLS.value}")
    print(f"    RC_CHANNELS = {ControlSource.RC_CHANNELS.value}")
    print(f"    MANUAL_CONTROL = {ControlSource.MANUAL_CONTROL.value}")
    print(f"    CUSTOM = {ControlSource.CUSTOM.value}")
    
    yield ControlSource


@pytest.fixture
def state_target_enum():
    """
    Fixture: 返回 StateTarget 枚举
    
    StateTarget 定义 MAVLink 状态消息目标类型。
    """
    from src import StateTarget
    
    print(f"\n  [FIXTURE] StateTarget enum:")
    print(f"    HIL_STATE = {StateTarget.HIL_STATE.value}")
    print(f"    HIL_STATE_QUATERNION = {StateTarget.HIL_STATE_QUATERNION.value}")
    print(f"    HIL_GPS = {StateTarget.HIL_GPS.value}")
    print(f"    HIL_SENSOR = {StateTarget.HIL_SENSOR.value}")
    print(f"    CUSTOM = {StateTarget.CUSTOM.value}")
    
    yield StateTarget
