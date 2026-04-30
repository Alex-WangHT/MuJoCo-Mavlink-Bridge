import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestJointInfo:
    def test_joint_info_creation(self):
        from src import JointInfo
        
        info = JointInfo(
            name="test_joint",
            index=0,
            joint_type="hinge",
            qpos_idx=0,
            qvel_idx=0,
            range=(-3.14, 3.14)
        )
        
        assert info.name == "test_joint"
        assert info.index == 0
        assert info.joint_type == "hinge"
        assert info.qpos_idx == 0
        assert info.qvel_idx == 0
        assert info.range == (-3.14, 3.14)

    def test_joint_info_defaults(self):
        from src import JointInfo
        
        info = JointInfo(
            name="test",
            index=0,
            joint_type="hinge",
            qpos_idx=0,
            qvel_idx=0
        )
        
        assert info.dof == 1
        assert info.range == (-np.inf, np.inf)
        assert info.properties == {}


class TestBodyInfo:
    def test_body_info_creation(self):
        from src import BodyInfo
        
        info = BodyInfo(
            name="base",
            index=0,
            parent_name=None,
            body_id=0
        )
        
        assert info.name == "base"
        assert info.index == 0
        assert info.parent_name is None
        assert info.body_id == 0

    def test_body_info_with_parent(self):
        from src import BodyInfo
        
        info = BodyInfo(
            name="link2",
            index=1,
            parent_name="base",
            body_id=1
        )
        
        assert info.parent_name == "base"


class TestSensorInfo:
    def test_sensor_info_creation(self):
        from src import SensorInfo
        
        info = SensorInfo(
            name="joint1_pos",
            index=0,
            sensor_type="jointpos",
            data_dim=1,
            data_start=0
        )
        
        assert info.name == "joint1_pos"
        assert info.sensor_type == "jointpos"
        assert info.data_dim == 1
        assert info.data_start == 0


class TestRobotState:
    def test_robot_state_creation(self, robot_state):
        state = robot_state
        
        assert state.time == 0.0
        assert "joint1" in state.joint_positions
        assert "joint2" in state.joint_velocities
        assert "base" in state.body_positions

    def test_robot_state_custom_data(self):
        from src import RobotState
        import numpy as np
        
        state = RobotState(
            time=1.5,
            joint_positions={},
            joint_velocities={},
            joint_accelerations={},
            joint_torques={},
            body_positions={},
            body_velocities={},
            sensor_data={},
            custom_data={"key": "value", "number": 42}
        )
        
        assert state.custom_data["key"] == "value"
        assert state.custom_data["number"] == 42


class TestControlModeEnum:
    def test_control_mode_values(self, control_mode_enum):
        mode = control_mode_enum
        
        assert mode.POSITION.value == "position"
        assert mode.VELOCITY.value == "velocity"
        assert mode.TORQUE.value == "torque"
        assert mode.FORCE.value == "force"
        assert mode.PID.value == "pid"
        assert mode.CUSTOM.value == "custom"


class TestControlTargetTypeEnum:
    def test_control_target_type_values(self, control_target_type_enum):
        target_type = control_target_type_enum
        
        assert target_type.JOINT_POSITION.value == "joint_position"
        assert target_type.JOINT_VELOCITY.value == "joint_velocity"
        assert target_type.JOINT_TORQUE.value == "joint_torque"
        assert target_type.BODY_FORCE.value == "body_force"
        assert target_type.BODY_TORQUE.value == "body_torque"


class TestPIDGains:
    def test_pid_gains_creation(self):
        from src import PIDGains
        
        gains = PIDGains(kp=10.0, ki=1.0, kd=0.5, i_max=100.0, output_max=50.0)
        
        assert gains.kp == 10.0
        assert gains.ki == 1.0
        assert gains.kd == 0.5
        assert gains.i_max == 100.0
        assert gains.output_max == 50.0

    def test_pid_gains_defaults(self):
        from src import PIDGains
        
        gains = PIDGains()
        
        assert gains.kp == 0.0
        assert gains.ki == 0.0
        assert gains.kd == 0.0
        assert gains.i_max == 0.0
        assert gains.output_max == 0.0


class TestPIDController:
    def test_pid_controller_creation(self, pid_controller):
        pid = pid_controller
        
        assert pid.gains.kp == 10.0
        assert pid.gains.ki == 0.1
        assert pid.gains.kd == 1.0
        assert pid._integral == 0.0
        assert pid._last_error == 0.0

    def test_pid_proportional_only(self):
        from src import PIDController, PIDGains
        
        gains = PIDGains(kp=2.0, ki=0.0, kd=0.0)
        pid = PIDController(gains=gains)
        
        output = pid.compute(target=5.0, current=3.0)
        
        assert output == 4.0

    def test_pid_integral_action(self):
        from src import PIDController, PIDGains
        
        gains = PIDGains(kp=0.0, ki=0.5, kd=0.0)
        pid = PIDController(gains=gains)
        
        pid.compute(target=4.0, current=2.0)
        output = pid.compute(target=4.0, current=2.0)
        
        assert pid._integral > 0

    def test_pid_derivative_action(self):
        from src import PIDController, PIDGains
        
        gains = PIDGains(kp=0.0, ki=0.0, kd=0.5)
        pid = PIDController(gains=gains)
        
        pid.compute(target=0.0, current=2.0)
        time.sleep(0.01)
        output = pid.compute(target=0.0, current=1.0)
        
        assert output != 0.0

    def test_pid_reset(self, pid_controller):
        pid = pid_controller
        
        pid.compute(target=5.0, current=0.0)
        assert pid._integral > 0
        
        pid.reset()
        
        assert pid._integral == 0.0
        assert pid._last_error == 0.0

    def test_pid_set_gains(self, pid_controller):
        pid = pid_controller
        
        pid.set_gains(kp=100.0, ki=10.0, kd=5.0, i_max=1000.0, output_max=100.0)
        
        assert pid.gains.kp == 100.0
        assert pid.gains.ki == 10.0
        assert pid.gains.kd == 5.0
        assert pid.gains.i_max == 1000.0
        assert pid.gains.output_max == 100.0


class TestControlCommand:
    def test_control_command_creation(self):
        from src import ControlCommand, ControlTarget, ControlMode
        import time
        
        cmd = ControlCommand(timestamp=time.time())
        
        assert cmd.timestamp > 0
        assert cmd.targets == {}
        assert cmd.forces == {}
        assert cmd.custom_data == {}

    def test_control_command_with_targets(self):
        from src import ControlCommand, ControlTarget, ControlMode
        
        target = ControlTarget(
            joint_name="joint1",
            mode=ControlMode.TORQUE,
            value=5.0,
            scale=1.0,
            offset=0.0
        )
        
        cmd = ControlCommand(
            timestamp=0.0,
            targets={"joint1": target}
        )
        
        assert "joint1" in cmd.targets
        assert cmd.targets["joint1"].value == 5.0


class TestControlTarget:
    def test_control_target_creation(self):
        from src import ControlTarget, ControlMode
        
        target = ControlTarget(
            joint_name="test_joint",
            mode=ControlMode.POSITION,
            value=1.57,
            scale=2.0,
            offset=0.5
        )
        
        assert target.joint_name == "test_joint"
        assert target.mode == ControlMode.POSITION
        assert target.value == 1.57
        assert target.scale == 2.0
        assert target.offset == 0.5


class TestMessageTypeEnum:
    def test_message_type_values(self):
        from src import MessageType
        
        assert MessageType.HIL_ACTUATOR_CONTROLS.value == 93
        assert MessageType.HIL_STATE_QUATERNION.value == 115
        assert MessageType.HEARTBEAT.value == 0
        assert MessageType.CUSTOM.value == -1


class TestMavlinkMessage:
    def test_mavlink_message_creation(self, basic_mavlink_message):
        msg = basic_mavlink_message
        
        from src import MessageType
        
        assert msg.msg_type == MessageType.HIL_ACTUATOR_CONTROLS
        assert msg.msg_name == "HIL_ACTUATOR_CONTROLS"
        assert msg.sysid == 1
        assert msg.compid == 1
        assert msg.seq == 42
        assert "controls" in msg.data

    def test_mavlink_message_get_method(self, basic_mavlink_message):
        msg = basic_mavlink_message
        
        controls = msg.get("controls")
        assert controls[0] == 0.5
        
        missing = msg.get("nonexistent", "default")
        assert missing == "default"

    def test_mavlink_message_dict_interface(self, basic_mavlink_message):
        msg = basic_mavlink_message
        
        assert "controls" in msg
        assert msg["controls"][0] == 0.5


class TestBaseRobotModelAbstract:
    def test_base_robot_model_is_abstract(self):
        from src import BaseRobotModel
        import inspect
        
        assert inspect.isabstract(BaseRobotModel)

    def test_base_robot_model_has_abstract_methods(self):
        from src import BaseRobotModel
        
        abstract_methods = {m for m in dir(BaseRobotModel) if not m.startswith('_')}
        
        assert 'load_model' in dir(BaseRobotModel)
        assert 'step' in dir(BaseRobotModel)
        assert 'reset' in dir(BaseRobotModel)
        assert 'get_state' in dir(BaseRobotModel)


class TestBaseControllerAbstract:
    def test_base_controller_has_methods(self):
        from src import BaseController
        
        assert 'apply_control' in dir(BaseController)
        assert 'get_control_output' in dir(BaseController)
        assert 'get_current_state' in dir(BaseController)
        assert 'set_control_mode' in dir(BaseController)
        assert 'set_pid_gains' in dir(BaseController)
        assert 'reset' in dir(BaseController)


class TestBaseMavlinkHandlerAbstract:
    def test_base_mavlink_handler_is_abstract(self):
        from src import BaseMavlinkHandler
        import inspect
        
        assert inspect.isabstract(BaseMavlinkHandler)

    def test_base_mavlink_handler_abstract_methods(self):
        from src import BaseMavlinkHandler
        
        assert 'connect' in dir(BaseMavlinkHandler)
        assert 'disconnect' in dir(BaseMavlinkHandler)
        assert 'send_message' in dir(BaseMavlinkHandler)
        assert 'receive_message' in dir(BaseMavlinkHandler)
        assert 'start' in dir(BaseMavlinkHandler)
        assert 'stop' in dir(BaseMavlinkHandler)
