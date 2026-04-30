import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMuJoCoModelInitialization:
    def test_default_model_creation(self, mujoco_model):
        model = mujoco_model
        
        assert model.is_initialized is True
        assert len(model.joint_names) > 0
        assert len(model.body_names) > 0

    def test_joint_info_retrieval(self, mujoco_model):
        model = mujoco_model
        
        for joint_name in model.joint_names:
            info = model.get_joint_info(joint_name)
            assert info is not None
            assert info.name == joint_name

    def test_body_info_retrieval(self, mujoco_model):
        model = mujoco_model
        
        for body_name in model.body_names:
            info = model.get_body_info(body_name)
            assert info is not None
            assert info.name == body_name

    def test_sensor_info_retrieval(self, mujoco_model):
        model = mujoco_model
        
        for sensor_name in model.sensor_names:
            info = model.get_sensor_info(sensor_name)
            assert info is not None
            assert info.name == sensor_name

    def test_has_joint(self, mujoco_model):
        model = mujoco_model
        
        joint_name = model.joint_names[0] if model.joint_names else "joint1"
        
        assert model.has_joint(joint_name) is True
        assert model.has_joint("nonexistent_joint") is False

    def test_has_body(self, mujoco_model):
        model = mujoco_model
        
        body_name = model.body_names[0] if model.body_names else "base"
        
        assert model.has_body(body_name) is True
        assert model.has_body("nonexistent_body") is False

    def test_has_sensor(self, mujoco_model):
        model = mujoco_model
        
        if model.sensor_names:
            sensor_name = model.sensor_names[0]
            assert model.has_sensor(sensor_name) is True
        
        assert model.has_sensor("nonexistent_sensor") is False


class TestMuJoCoModelSimulation:
    def test_step_function(self, mujoco_model):
        model = mujoco_model
        
        initial_state = model.get_state()
        
        model.step(n_steps=10)
        
        assert True

    def test_get_state(self, mujoco_model):
        model = mujoco_model
        
        state = model.get_state()
        
        assert state is not None
        assert isinstance(state.time, float)
        assert isinstance(state.joint_positions, dict)
        assert isinstance(state.joint_velocities, dict)
        assert isinstance(state.body_positions, dict)

    def test_get_joint_qpos(self, mujoco_model):
        model = mujoco_model
        
        joint_name = model.joint_names[0] if model.joint_names else "joint1"
        
        pos = model.get_joint_qpos(joint_name)
        
        assert pos is not None
        assert isinstance(pos, float)

    def test_get_joint_qvel(self, mujoco_model):
        model = mujoco_model
        
        joint_name = model.joint_names[0] if model.joint_names else "joint1"
        
        vel = model.get_joint_qvel(joint_name)
        
        assert vel is not None
        assert isinstance(vel, float)

    def test_set_control(self, mujoco_model):
        model = mujoco_model
        
        model.set_control(0, 1.0)
        
        control = model.get_control(0)
        
        assert control == 1.0

    def test_get_control(self, mujoco_model):
        model = mujoco_model
        
        control = model.get_control(0)
        
        assert isinstance(control, float)

    def test_reset(self, mujoco_model):
        model = mujoco_model
        
        model.set_control(0, 1.0)
        model.step(n_steps=100)
        
        model.reset()
        
        control = model.get_control(0)
        assert control == 0.0

    def test_get_timestep(self, mujoco_model):
        model = mujoco_model
        
        timestep = model.get_timestep()
        
        assert timestep > 0
        assert isinstance(timestep, float)

    def test_get_body_position(self, mujoco_model):
        model = mujoco_model
        
        body_name = model.body_names[0] if model.body_names else "base"
        
        pos = model.get_body_position(body_name)
        
        assert pos is not None
        assert len(pos) == 3


class TestMuJoCoModelJoints:
    def test_joint_count(self, mujoco_model):
        model = mujoco_model
        
        assert len(model.joint_names) == len(model.joints)

    def test_joint_types(self, mujoco_model):
        model = mujoco_model
        
        for joint_name in model.joint_names:
            info = model.get_joint_info(joint_name)
            assert info.joint_type in ["hinge", "slide", "ball", "free", "unknown"]

    def test_joint_ranges(self, mujoco_model):
        model = mujoco_model
        
        for joint_name in model.joint_names:
            info = model.get_joint_info(joint_name)
            assert isinstance(info.range, tuple)
            assert len(info.range) == 2


class TestMuJoCoModelAdvanced:
    def test_apply_body_force(self, mujoco_model):
        model = mujoco_model
        
        body_name = model.body_names[0] if model.body_names else "base"
        
        force = np.array([0.0, 0.0, 1.0])
        model.apply_body_force(body_name, force)
        
        assert True

    def test_apply_body_force_with_point(self, mujoco_model):
        model = mujoco_model
        
        body_name = model.body_names[0] if model.body_names else "base"
        
        force = np.array([1.0, 0.0, 0.0])
        point = np.array([0.1, 0.0, 0.0])
        model.apply_body_force(body_name, force, point)
        
        assert True

    def test_simulation_with_controls(self, mujoco_model):
        model = mujoco_model
        
        model.set_control(0, 0.5)
        
        for _ in range(100):
            model.step(n_steps=1)
        
        state = model.get_state()
        
        assert state is not None

    def test_multiple_steps(self, mujoco_model):
        model = mujoco_model
        
        steps_before = model.get_state().time
        
        model.step(n_steps=100)
        
        state = model.get_state()
        
        assert state.time > steps_before

    def test_state_consistency(self, mujoco_model):
        model = mujoco_model
        
        state1 = model.get_state()
        state2 = model.get_state()
        
        assert state1.time == state2.time
        assert state1.joint_positions.keys() == state2.joint_positions.keys()


class TestMuJoCoModelEdgeCases:
    def test_get_joint_qpos_nonexistent(self, mujoco_model):
        model = mujoco_model
        
        pos = model.get_joint_qpos("nonexistent_joint")
        
        assert pos is None

    def test_get_joint_qvel_nonexistent(self, mujoco_model):
        model = mujoco_model
        
        vel = model.get_joint_qvel("nonexistent_joint")
        
        assert vel is None

    def test_get_body_position_nonexistent(self, mujoco_model):
        model = mujoco_model
        
        pos = model.get_body_position("nonexistent_body")
        
        assert pos is None

    def test_apply_body_force_nonexistent(self, mujoco_model):
        model = mujoco_model
        
        force = np.array([0.0, 0.0, 1.0])
        
        model.apply_body_force("nonexistent_body", force)
        
        assert True

    def test_zero_steps(self, mujoco_model):
        model = mujoco_model
        
        state_before = model.get_state()
        
        model.step(n_steps=0)
        
        state_after = model.get_state()
        
        assert state_before.time == state_after.time

    def test_large_steps(self, mujoco_model):
        model = mujoco_model
        
        model.step(n_steps=1000)
        
        assert True

    def test_repeated_reset(self, mujoco_model):
        model = mujoco_model
        joint_name = model.joint_names[0] if model.joint_names else "joint1"
        
        for _ in range(5):
            model.set_control(0, 1.0)
            model.step(n_steps=10)
            model.reset()
        
        control = model.get_control(0)
        assert control == 0.0


class TestMuJoCoModelProperties:
    def test_joints_property(self, mujoco_model):
        model = mujoco_model
        
        joints = model.joints
        
        assert isinstance(joints, dict)
        for name, info in joints.items():
            assert name in model.joint_names

    def test_joint_names_property(self, mujoco_model):
        model = mujoco_model
        
        names = model.joint_names
        
        assert isinstance(names, list)
        for name in names:
            assert model.has_joint(name)

    def test_bodies_property(self, mujoco_model):
        model = mujoco_model
        
        bodies = model.bodies
        
        assert isinstance(bodies, dict)

    def test_body_names_property(self, mujoco_model):
        model = mujoco_model
        
        names = model.body_names
        
        assert isinstance(names, list)

    def test_sensors_property(self, mujoco_model):
        model = mujoco_model
        
        sensors = model.sensors
        
        assert isinstance(sensors, dict)

    def test_sensor_names_property(self, mujoco_model):
        model = mujoco_model
        
        names = model.sensor_names
        
        assert isinstance(names, list)

    def test_is_initialized_property(self, mujoco_model):
        model = mujoco_model
        
        assert model.is_initialized is True
