import pytest
import numpy as np
import time
import threading
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestControlMapping:
    def test_control_mapping_creation(self):
        from src import ControlMapping
        mapping = ControlMapping(
            mavlink_index=0,
            joint_name="joint1",
            control_type="torque",
            scale=2.0,
            offset=0.5
        )
        
        assert mapping.mavlink_index == 0
        assert mapping.joint_name == "joint1"
        assert mapping.control_type == "torque"
        assert mapping.scale == 2.0
        assert mapping.offset == 0.5

    def test_control_mapping_defaults(self):
        from src import ControlMapping
        mapping = ControlMapping(
            mavlink_index=0,
            joint_name="test",
            control_type="position"
        )
        
        assert mapping.scale == 1.0
        assert mapping.offset == 0.0


class TestSimulatorInitialization:
    def test_simulator_creation(self, simulator):
        sim = simulator
        
        assert sim.mavlink_bridge is not None
        assert sim.mujoco_controller is not None
        assert sim.is_running is False
        assert sim.real_time_factor == 10.0

    def test_default_control_mappings(self, simulator):
        sim = simulator
        
        assert len(sim.control_mappings) > 0
        assert len(sim.joint_to_mavlink) > 0
        
        for idx, mapping in sim.control_mappings.items():
            assert mapping.mavlink_index == idx
            assert mapping.joint_name in sim.mujoco_controller.joint_names

    def test_connection_string(self, simulator):
        sim = simulator
        
        assert "127.0.0.1" in sim.mavlink_connection
        assert "19999" in sim.mavlink_connection


class TestSimulatorControlMappings:
    def test_add_control_mapping(self, simulator):
        sim = simulator
        joint_name = sim.mujoco_controller.joint_names[0]
        
        sim.add_control_mapping(
            mavlink_index=5,
            joint_name=joint_name,
            control_type="position",
            scale=0.5,
            offset=1.0
        )
        
        assert 5 in sim.control_mappings
        mapping = sim.control_mappings[5]
        assert mapping.joint_name == joint_name
        assert mapping.control_type == "position"
        assert mapping.scale == 0.5
        assert mapping.offset == 1.0

    def test_add_control_mapping_invalid_joint(self, simulator):
        sim = simulator
        
        with pytest.raises(ValueError):
            sim.add_control_mapping(
                mavlink_index=0,
                joint_name="non_existent_joint",
                control_type="torque"
            )

    def test_set_pid_gains(self, simulator):
        sim = simulator
        joint_name = sim.mujoco_controller.joint_names[0]
        
        sim.set_pid_gains(joint_name, kp=50.0, ki=5.0, kd=2.0)
        
        assert joint_name in sim.mujoco_controller.pid_controllers
        pid = sim.mujoco_controller.pid_controllers[joint_name]
        assert pid.kp == 50.0
        assert pid.ki == 5.0
        assert pid.kd == 2.0


class TestSimulatorMessageHandling:
    def test_handle_hil_actuator_controls(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [0.5, 0.3, -0.2] + [0.0] * 13,
                "mode": 0,
                "flags": 0
            }
        )
        
        initial_messages = sim._stats["messages_received"]
        sim._handle_hil_actuator_controls(msg)
        
        assert sim._stats["messages_received"] == initial_messages + 1

    def test_handle_raw_mavlink1(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        
        msg = MavlinkMessage(
            msg_type="MAVLINK1",
            sysid=1,
            compid=1,
            seq=0,
            data={"raw": b"\x01\x02\x03", "addr": ("127.0.0.1", 14550)}
        )
        
        initial_messages = sim._stats["messages_received"]
        sim._handle_raw_mavlink1(msg)
        
        assert sim._stats["messages_received"] == initial_messages + 1

    def test_handle_raw_mavlink2(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        
        msg = MavlinkMessage(
            msg_type="MAVLINK2",
            sysid=1,
            compid=1,
            seq=0,
            data={"raw": b"\xFDtest", "addr": ("127.0.0.1", 14550)}
        )
        
        initial_messages = sim._stats["messages_received"]
        sim._handle_raw_mavlink2(msg)
        
        assert sim._stats["messages_received"] == initial_messages + 1

    def test_on_message_received_callback(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        received_messages = []
        
        def callback(msg):
            received_messages.append(msg)
        
        sim.on_message_received = callback
        
        msg = MavlinkMessage(
            msg_type="TEST_MESSAGE",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        sim.on_message_received(msg)
        
        assert len(received_messages) == 1
        assert received_messages[0].msg_type == "TEST_MESSAGE"


class TestSimulatorStateUpdates:
    def test_get_robot_state(self, simulator):
        sim = simulator
        
        state = sim.get_robot_state()
        
        assert state is not None
        assert isinstance(state.time, float)
        assert len(state.joint_states) == len(sim.mujoco_controller.joint_names)

    def test_get_joint_state(self, simulator):
        sim = simulator
        joint_name = sim.mujoco_controller.joint_names[0]
        
        state = sim.get_joint_state(joint_name)
        
        assert state is not None
        assert state.name == joint_name

    def test_on_state_update_callback(self, simulator):
        sim = simulator
        state_updates = []
        
        def callback(state):
            state_updates.append(state)
        
        sim.on_state_update = callback
        
        sim.mujoco_controller.step(n_steps=5)
        state = sim.get_robot_state()
        sim.on_state_update(state)
        
        assert len(state_updates) == 1


class TestSimulatorLifecycle:
    def test_start_stop(self, simulator):
        sim = simulator
        
        assert sim.is_running is False
        
        sim.start()
        assert sim.is_running is True
        
        time.sleep(0.1)
        
        sim.stop()
        assert sim.is_running is False

    def test_double_start(self, simulator):
        sim = simulator
        
        sim.start()
        first_thread = sim.simulation_thread
        
        sim.start()
        assert sim.simulation_thread == first_thread
        
        sim.stop()

    def test_stop_without_start(self, simulator):
        sim = simulator
        
        sim.stop()
        
        assert sim.is_running is False

    def test_reset(self, simulator):
        sim = simulator
        
        sim.start()
        time.sleep(0.1)
        sim.stop()
        
        initial_steps = sim._stats["steps"]
        sim.reset()
        
        assert sim._stats["steps"] == 0
        assert sim._stats["messages_received"] == 0
        assert sim._stats["messages_sent"] == 0


class TestSimulatorStatistics:
    def test_initial_statistics(self, simulator):
        sim = simulator
        
        stats = sim.get_statistics()
        
        assert stats["steps"] == 0
        assert stats["messages_received"] == 0
        assert stats["messages_sent"] == 0
        assert stats["is_running"] is False

    def test_statistics_during_simulation(self, simulator):
        sim = simulator
        
        sim.start()
        time.sleep(0.2)
        
        stats = sim.get_statistics()
        
        assert stats["is_running"] is True
        assert stats["steps"] > 0
        assert stats["elapsed_time"] > 0
        
        sim.stop()

    def test_steps_per_second(self, simulator):
        sim = simulator
        sim.real_time_factor = 100.0
        
        sim.start()
        time.sleep(0.2)
        
        stats = sim.get_statistics()
        
        assert stats["steps_per_second"] > 0
        
        sim.stop()


class TestSimulatorMAVLinkIntegration:
    def test_send_heartbeat(self, simulator):
        sim = simulator
        
        sim.send_heartbeat()
        
        assert True

    def test_send_state_to_mavlink(self, simulator):
        sim = simulator
        
        state = sim.get_robot_state()
        sim._send_state_to_mavlink(state)
        
        assert True

    def test_control_mapping_application(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        joint_name = sim.mujoco_controller.joint_names[0]
        
        sim.add_control_mapping(
            mavlink_index=0,
            joint_name=joint_name,
            control_type="torque",
            scale=1.0,
            offset=0.0
        )
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [1.0] + [0.0] * 15,
                "mode": 0,
                "flags": 0
            }
        )
        
        sim._handle_hil_actuator_controls(msg)
        
        assert joint_name in sim.mujoco_controller.target_torques
        assert sim.mujoco_controller.target_torques[joint_name] == 1.0


class TestSimulatorEdgeCases:
    def test_empty_message_handlers(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        sim.mavlink_bridge.message_handlers.clear()
        
        msg = MavlinkMessage(
            msg_type="UNKNOWN",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        sim._handle_hil_actuator_controls(msg)
        
        assert True

    def test_control_mapping_scale_and_offset(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        joint_name = sim.mujoco_controller.joint_names[0]
        
        sim.add_control_mapping(
            mavlink_index=0,
            joint_name=joint_name,
            control_type="position",
            scale=2.0,
            offset=1.0
        )
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [0.5] + [0.0] * 15,
                "mode": 0,
                "flags": 0
            }
        )
        
        sim._handle_hil_actuator_controls(msg)
        
        expected = 0.5 * 2.0 + 1.0
        assert sim.mujoco_controller.target_positions[joint_name] == expected

    def test_multiple_control_mappings(self, simulator):
        from src import MavlinkMessage
        
        sim = simulator
        
        for i, joint_name in enumerate(sim.mujoco_controller.joint_names[:2]):
            sim.add_control_mapping(
                mavlink_index=i,
                joint_name=joint_name,
                control_type="torque",
                scale=1.0,
                offset=0.0
            )
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [1.0, -1.0] + [0.0] * 14,
                "mode": 0,
                "flags": 0
            }
        )
        
        sim._handle_hil_actuator_controls(msg)
        
        joint_names = sim.mujoco_controller.joint_names[:2]
        assert sim.mujoco_controller.target_torques[joint_names[0]] == 1.0
        assert sim.mujoco_controller.target_torques[joint_names[1]] == -1.0

    def test_simulation_loop_timing(self, simulator):
        sim = simulator
        sim.real_time_factor = 1000.0
        
        sim.start()
        time.sleep(0.2)
        
        stats = sim.get_statistics()
        
        assert stats["steps"] > 10
        
        sim.stop()

    def test_concurrent_access(self, simulator):
        sim = simulator
        errors = []
        
        def access_stats():
            try:
                for _ in range(100):
                    sim.get_statistics()
            except Exception as e:
                errors.append(e)
        
        def access_state():
            try:
                for _ in range(100):
                    sim.get_robot_state()
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=access_stats),
            threading.Thread(target=access_state),
            threading.Thread(target=access_stats),
            threading.Thread(target=access_state)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
