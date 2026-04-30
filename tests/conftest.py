import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def basic_mavlink_message():
    from src import MavlinkMessage, MessageType
    return MavlinkMessage(
        msg_type=MessageType.HIL_ACTUATOR_CONTROLS,
        msg_name="HIL_ACTUATOR_CONTROLS",
        sysid=1,
        compid=1,
        seq=42,
        data={
            "controls": [0.5, 0.3, -0.2, 0.0] + [0.0] * 12,
            "mode": 0,
            "flags": 0
        }
    )


@pytest.fixture
def mujoco_model():
    from src import MuJoCoModel
    model = MuJoCoModel()
    yield model


@pytest.fixture
def mujoco_controller():
    from src import MuJoCoController, MuJoCoModel
    model = MuJoCoModel()
    controller = MuJoCoController(model=model)
    yield controller
    controller.reset()


@pytest.fixture
def simulator_config():
    from src import SimulatorConfig, MavlinkConfig, SimulationConfig
    mavlink_config = MavlinkConfig(
        host="127.0.0.1",
        port=19999,
        source_system=1,
        source_component=1,
    )
    simulation_config = SimulationConfig(
        real_time_factor=100.0,
    )
    return SimulatorConfig(
        mavlink=mavlink_config,
        simulation=simulation_config,
    )


@pytest.fixture
def simulator(simulator_config):
    from src import Simulator
    sim = Simulator(config=simulator_config)
    yield sim
    if sim.is_running:
        sim.stop()


@pytest.fixture
def control_mode_enum():
    from src import ControlMode
    return ControlMode


@pytest.fixture
def control_target_type_enum():
    from src import ControlTargetType
    return ControlTargetType


@pytest.fixture
def pid_controller():
    from src import PIDController, PIDGains
    gains = PIDGains(kp=10.0, ki=0.1, kd=1.0)
    return PIDController(gains=gains)


@pytest.fixture
def control_mapper():
    from src import ControlMapper
    mapper = ControlMapper()
    yield mapper
    mapper.clear_mappings()


@pytest.fixture
def mavlink_bridge():
    from src import MavlinkUDPBridge
    bridge = MavlinkUDPBridge(
        host="127.0.0.1",
        port=18888,
        source_system=1,
        source_component=1,
    )
    yield bridge
    if bridge.is_connected:
        bridge.disconnect()


@pytest.fixture
def robot_state():
    from src import RobotState
    import numpy as np
    return RobotState(
        time=0.0,
        joint_positions={"joint1": 0.0, "joint2": 0.0},
        joint_velocities={"joint1": 0.0, "joint2": 0.0},
        joint_accelerations={"joint1": 0.0, "joint2": 0.0},
        joint_torques={"joint1": 0.0, "joint2": 0.0},
        body_positions={"base": np.zeros(3), "end_effector": np.array([0.0, 0.0, 0.5])},
        body_velocities={"base": np.zeros(6), "end_effector": np.zeros(6)},
        sensor_data={}
    )
