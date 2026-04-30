import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_mavlink_message():
    from src import MavlinkMessage
    return MavlinkMessage(
        msg_type="HIL_ACTUATOR_CONTROLS",
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
def mujoco_controller():
    from src import MuJoCoController
    controller = MuJoCoController()
    yield controller
    controller.reset()


@pytest.fixture
def simulator():
    from src import Simulator
    sim = Simulator(
        mavlink_connection="udp:127.0.0.1:19999",
        real_time_factor=10.0
    )
    yield sim
    if sim.is_running:
        sim.stop()


@pytest.fixture
def control_mode_enum():
    from src import ControlMode
    return ControlMode


@pytest.fixture
def pid_controller():
    from src import PIDController
    return PIDController(kp=10.0, ki=0.1, kd=1.0)
