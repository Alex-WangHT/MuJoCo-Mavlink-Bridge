import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestJointState:
    def test_joint_state_creation(self):
        from src import JointState
        js = JointState(
            name="joint1",
            qpos=1.5,
            qvel=0.5,
            qacc=0.1,
            qfrc_applied=2.0
        )
        assert js.name == "joint1"
        assert js.qpos == 1.5
        assert js.qvel == 0.5
        assert js.qacc == 0.1
        assert js.qfrc_applied == 2.0

    def test_joint_state_defaults(self):
        from src import JointState
        from dataclasses import fields
        js = JointState(name="test")
        
        for field in fields(js):
            if field.name != "name":
                assert getattr(js, field.name) == 0.0


class TestControlMode:
    def test_control_mode_values(self, control_mode_enum):
        assert control_mode_enum.POSITION == "position"
        assert control_mode_enum.VELOCITY == "velocity"
        assert control_mode_enum.TORQUE == "torque"
        assert control_mode_enum.PID == "pid"


class TestPIDController:
    def test_pid_initialization(self, pid_controller):
        from src import PIDController
        pid = pid_controller
        
        assert pid.kp == 10.0
        assert pid.ki == 0.1
        assert pid.kd == 1.0
        assert pid.integral == 0.0
        assert pid.last_error == 0.0

    def test_pid_compute_proportional(self):
        from src import PIDController
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
        
        output = pid.compute(target=5.0, current=3.0)
        
        assert output == 4.0

    def test_pid_compute_integral(self):
        from src import PIDController
        pid = PIDController(kp=0.0, ki=0.5, kd=0.0)
        
        pid.compute(target=4.0, current=2.0)
        output = pid.compute(target=4.0, current=2.0)
        
        assert pid.integral > 0

    def test_pid_compute_derivative(self):
        from src import PIDController
        pid = PIDController(kp=0.0, ki=0.0, kd=0.5)
        
        pid.compute(target=0.0, current=2.0)
        time.sleep(0.01)
        output = pid.compute(target=0.0, current=1.0)
        
        assert output != 0.0

    def test_pid_reset(self):
        from src import PIDController
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
        
        pid.compute(target=5.0, current=0.0)
        assert pid.integral != 0.0
        
        pid.reset()
        assert pid.integral == 0.0
        assert pid.last_error == 0.0


class TestMuJoCoControllerInitialization:
    def test_controller_creation(self, mujoco_controller):
        controller = mujoco_controller
        assert controller.is_initialized is True
        assert len(controller.joint_names) > 0
        assert len(controller.actuator_names) > 0

    def test_joint_indices(self, mujoco_controller):
        controller = mujoco_controller
        for name, idx in controller.joint_indices.items():
            assert name in controller.joint_names
            assert isinstance(idx, int)

    def test_actuator_indices(self, mujoco_controller):
        controller = mujoco_controller
        for name, idx in controller.actuator_indices.items():
            assert name in controller.actuator_names
            assert isinstance(idx, int)


class TestMuJoCoControllerControlModes:
    def test_set_control_mode(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0] if controller.joint_names else "joint1"
        
        controller.set_control_mode(joint_name, "position")
        assert controller.control_modes[joint_name] == "position"
        
        controller.set_control_mode(joint_name, "torque")
        assert controller.control_modes[joint_name] == "torque"

    def test_set_control_mode_invalid_joint(self, mujoco_controller):
        controller = mujoco_controller
        
        with pytest.raises(ValueError):
            controller.set_control_mode("non_existent_joint", "position")

    def test_set_pid_gains(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0] if controller.joint_names else "joint1"
        
        controller.set_pid_gains(joint_name, kp=100.0, ki=10.0, kd=5.0)
        
        assert joint_name in controller.pid_controllers
        pid = controller.pid_controllers[joint_name]
        assert pid.kp == 100.0
        assert pid.ki == 10.0
        assert pid.kd == 5.0


class TestMuJoCoControllerJointControl:
    def test_set_joint_position(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0] if controller.joint_names else "joint1"
        
        controller.set_joint_position(joint_name, 1.57)
        
        assert joint_name in controller.target_positions
        assert controller.target_positions[joint_name] == 1.57
        assert controller.control_modes.get(joint_name) == "position"

    def test_set_joint_velocity(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0] if controller.joint_names else "joint1"
        
        controller.set_joint_velocity(joint_name, 2.0)
        
        assert joint_name in controller.target_velocities
        assert controller.target_velocities[joint_name] == 2.0
        assert controller.control_modes.get(joint_name) == "velocity"

    def test_set_joint_torque(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0] if controller.joint_names else "joint1"
        
        controller.set_joint_torque(joint_name, 5.0)
        
        assert joint_name in controller.target_torques
        assert controller.target_torques[joint_name] == 5.0
        assert controller.control_modes.get(joint_name) == "torque"

    def test_set_all_joint_positions(self, mujoco_controller):
        controller = mujoco_controller
        
        positions = {}
        for i, name in enumerate(controller.joint_names[:2]):
            positions[name] = float(i) * 0.5
        
        controller.set_all_joint_positions(positions)
        
        for name, pos in positions.items():
            assert controller.target_positions[name] == pos

    def test_set_all_joint_torques(self, mujoco_controller):
        controller = mujoco_controller
        
        torques = {}
        for i, name in enumerate(controller.joint_names[:2]):
            torques[name] = float(i + 1)
        
        controller.set_all_joint_torques(torques)
        
        for name, torque in torques.items():
            assert controller.target_torques[name] == torque


class TestMuJoCoControllerSimulation:
    def test_step_function(self, mujoco_controller):
        controller = mujoco_controller
        
        initial_state = controller.get_joint_state(controller.joint_names[0])
        
        controller.step(n_steps=10)
        
        assert True

    def test_get_joint_state(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0]
        
        state = controller.get_joint_state(joint_name)
        
        assert state is not None
        assert state.name == joint_name
        assert isinstance(state.qpos, float)
        assert isinstance(state.qvel, float)

    def test_get_joint_state_invalid(self, mujoco_controller):
        controller = mujoco_controller
        
        state = controller.get_joint_state("non_existent")
        
        assert state is None

    def test_get_all_joint_states(self, mujoco_controller):
        controller = mujoco_controller
        
        states = controller.get_all_joint_states()
        
        assert len(states) == len(controller.joint_names)
        for name, state in states.items():
            assert name in controller.joint_names
            assert state.name == name

    def test_reset_function(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, 10.0)
        controller.step(n_steps=100)
        
        controller.reset()
        
        assert len(controller.target_positions) == 0
        assert len(controller.target_velocities) == 0
        assert len(controller.target_torques) == 0

    def test_get_timestep(self, mujoco_controller):
        controller = mujoco_controller
        
        timestep = controller.get_timestep()
        
        assert timestep > 0
        assert isinstance(timestep, float)


class TestMuJoCoControllerBodyAndSensor:
    def test_get_body_position(self, mujoco_controller):
        controller = mujoco_controller
        
        if controller.body_names:
            body_name = controller.body_names[0]
            pos = controller.get_body_position(body_name)
            
            assert pos is not None
            assert len(pos) == 3

    def test_get_body_position_invalid(self, mujoco_controller):
        controller = mujoco_controller
        
        pos = controller.get_body_position("non_existent_body")
        
        assert pos is None

    def test_get_body_velocity(self, mujoco_controller):
        controller = mujoco_controller
        
        if controller.body_names:
            body_name = controller.body_names[0]
            vel = controller.get_body_velocity(body_name)
            
            assert vel is not None
            assert len(vel) == 6

    def test_get_sensor_data(self, mujoco_controller):
        controller = mujoco_controller
        
        if controller.sensor_names:
            sensor_name = controller.sensor_names[0]
            data = controller.get_sensor_data(sensor_name)
            
            assert data is not None

    def test_get_sensor_data_invalid(self, mujoco_controller):
        controller = mujoco_controller
        
        data = controller.get_sensor_data("non_existent_sensor")
        
        assert data is None

    def test_get_all_sensor_data(self, mujoco_controller):
        controller = mujoco_controller
        
        data = controller.get_all_sensor_data()
        
        assert isinstance(data, dict)
        for name, value in data.items():
            assert name in controller.sensor_names


class TestMuJoCoControllerRobotState:
    def test_get_robot_state(self, mujoco_controller):
        controller = mujoco_controller
        
        state = controller.get_robot_state()
        
        assert state is not None
        assert isinstance(state.time, float)
        assert isinstance(state.joint_states, dict)
        assert isinstance(state.body_positions, dict)
        assert isinstance(state.body_velocities, dict)
        assert isinstance(state.sensor_data, dict)

    def test_robot_state_joint_states(self, mujoco_controller):
        controller = mujoco_controller
        
        state = controller.get_robot_state()
        
        for name, js in state.joint_states.items():
            assert name in controller.joint_names
            assert js.name == name
            assert isinstance(js.qpos, float)
            assert isinstance(js.qvel, float)


class TestMuJoCoControllerForceApplication:
    def test_apply_force_to_body(self, mujoco_controller):
        controller = mujoco_controller
        
        if controller.body_names:
            body_name = controller.body_names[0]
            force = np.array([0.0, 0.0, 1.0])
            
            controller.apply_force_to_body(body_name, force)
            
            assert True

    def test_apply_force_to_body_invalid(self, mujoco_controller):
        controller = mujoco_controller
        
        with pytest.raises(ValueError):
            controller.apply_force_to_body(
                "non_existent",
                np.array([0.0, 0.0, 1.0])
            )

    def test_apply_force_with_point(self, mujoco_controller):
        controller = mujoco_controller
        
        if controller.body_names:
            body_name = controller.body_names[0]
            force = np.array([1.0, 0.0, 0.0])
            point = np.array([0.1, 0.0, 0.0])
            
            controller.apply_force_to_body(body_name, force, point)
            
            assert True


class TestMuJoCoControllerEdgeCases:
    def test_empty_joint_names(self):
        from src import MuJoCoController
        from unittest.mock import patch, MagicMock
        
        with patch('src.mujoco_controller.MUJOCO_AVAILABLE', False):
            controller = MuJoCoController()
            
            assert controller.is_initialized is True
            assert len(controller.joint_names) == 2

    def test_step_zero_steps(self, mujoco_controller):
        controller = mujoco_controller
        
        controller.step(n_steps=0)
        
        assert True

    def test_step_large_steps(self, mujoco_controller):
        controller = mujoco_controller
        
        controller.step(n_steps=1000)
        
        assert True

    def test_multiple_resets(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0]
        
        for _ in range(5):
            controller.set_joint_torque(joint_name, 5.0)
            controller.step(n_steps=10)
            controller.reset()
        
        assert len(controller.target_torques) == 0

    def test_pid_mode_activation(self, mujoco_controller):
        controller = mujoco_controller
        joint_name = controller.joint_names[0]
        
        controller.set_control_mode(joint_name, "pid")
        controller.target_positions[joint_name] = 1.0
        
        controller.step(n_steps=10)
        
        assert joint_name in controller.pid_controllers
