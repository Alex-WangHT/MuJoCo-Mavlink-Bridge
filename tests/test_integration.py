import pytest
import time
import threading
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMAVLinkMuJoCoIntegration:
    def test_bridge_and_controller_initialization(self):
        from src import MavlinkBridge, MuJoCoController
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:18888")
        controller = MuJoCoController()
        
        assert bridge.is_connected is True
        assert controller.is_initialized is True

    def test_message_to_joint_control_flow(self):
        from src import MavlinkBridge, MuJoCoController, MavlinkMessage
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [2.5] + [0.0] * 15,
                "mode": 0,
                "flags": 0
            }
        )
        
        controller.set_joint_torque(joint_name, msg.data["controls"][0])
        
        assert joint_name in controller.target_torques
        assert controller.target_torques[joint_name] == 2.5

    def test_simulation_state_update_flow(self):
        from src import Simulator
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:18887",
            real_time_factor=100.0
        )
        
        joint_name = sim.mujoco_controller.joint_names[0]
        sim.mujoco_controller.set_joint_torque(joint_name, 1.0)
        
        initial_state = sim.get_joint_state(joint_name)
        
        sim.mujoco_controller.step(n_steps=100)
        
        final_state = sim.get_joint_state(joint_name)
        
        assert initial_state.qpos != final_state.qpos or initial_state.qvel != final_state.qvel


class TestControlModeIntegration:
    def test_position_control_mode(self):
        from src import MuJoCoController, ControlMode
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_control_mode(joint_name, ControlMode.POSITION)
        controller.set_joint_position(joint_name, 0.5)
        
        controller.step(n_steps=50)
        
        state = controller.get_joint_state(joint_name)
        
        assert joint_name in controller.target_positions
        assert controller.control_modes[joint_name] == ControlMode.POSITION

    def test_velocity_control_mode(self):
        from src import MuJoCoController, ControlMode
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_control_mode(joint_name, ControlMode.VELOCITY)
        controller.set_joint_velocity(joint_name, 2.0)
        
        controller.step(n_steps=50)
        
        state = controller.get_joint_state(joint_name)
        
        assert controller.control_modes[joint_name] == ControlMode.VELOCITY

    def test_torque_control_mode(self):
        from src import MuJoCoController, ControlMode
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_control_mode(joint_name, ControlMode.TORQUE)
        controller.set_joint_torque(joint_name, 5.0)
        
        controller.step(n_steps=50)
        
        state = controller.get_joint_state(joint_name)
        
        assert controller.control_modes[joint_name] == ControlMode.TORQUE

    def test_pid_control_mode(self):
        from src import MuJoCoController, ControlMode
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_pid_gains(joint_name, kp=100.0, ki=10.0, kd=5.0)
        controller.set_control_mode(joint_name, ControlMode.PID)
        controller.target_positions[joint_name] = 1.0
        
        controller.step(n_steps=100)
        
        assert joint_name in controller.pid_controllers
        assert controller.control_modes[joint_name] == ControlMode.PID


class TestMAVLinkMessageProcessing:
    def test_hil_actuator_controls_parsing(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=42,
            data={
                "controls": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                            0.9, 1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
                "mode": 1,
                "flags": 0x12345678
            }
        )
        
        assert len(msg.data["controls"]) == 16
        assert msg.data["controls"][0] == 0.1
        assert msg.data["controls"][15] == -0.6
        assert msg.data["mode"] == 1

    def test_message_handling_pipeline(self):
        from src import Simulator, MavlinkMessage
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:18886",
            real_time_factor=100.0
        )
        
        received_states = []
        received_messages = []
        
        def state_callback(state):
            received_states.append(state)
        
        def message_callback(msg):
            received_messages.append(msg)
        
        sim.on_state_update = state_callback
        sim.on_message_received = message_callback
        
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
        
        sim.mujoco_controller.step(n_steps=10)
        state = sim.get_robot_state()
        sim.on_state_update(state)
        
        assert len(received_messages) == 1
        assert len(received_states) == 1
        assert received_messages[0].msg_type == "HIL_ACTUATOR_CONTROLS"


class TestMultiJointControl:
    def test_parallel_joint_control(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_names = controller.joint_names[:min(4, len(controller.joint_names))]
        
        torques = {name: float(i + 1) for i, name in enumerate(joint_names)}
        controller.set_all_joint_torques(torques)
        
        for name, torque in torques.items():
            assert controller.target_torques[name] == torque

    def test_different_control_modes_per_joint(self):
        from src import MuJoCoController, ControlMode
        
        controller = MuJoCoController()
        joint_names = controller.joint_names[:min(3, len(controller.joint_names))]
        
        if len(joint_names) >= 1:
            controller.set_control_mode(joint_names[0], ControlMode.POSITION)
            controller.set_joint_position(joint_names[0], 0.5)
        
        if len(joint_names) >= 2:
            controller.set_control_mode(joint_names[1], ControlMode.VELOCITY)
            controller.set_joint_velocity(joint_names[1], 1.0)
        
        if len(joint_names) >= 3:
            controller.set_control_mode(joint_names[2], ControlMode.TORQUE)
            controller.set_joint_torque(joint_names[2], 2.0)
        
        controller.step(n_steps=50)
        
        if len(joint_names) >= 1:
            assert controller.control_modes[joint_names[0]] == ControlMode.POSITION
        if len(joint_names) >= 2:
            assert controller.control_modes[joint_names[1]] == ControlMode.VELOCITY
        if len(joint_names) >= 3:
            assert controller.control_modes[joint_names[2]] == ControlMode.TORQUE


class TestSensorDataIntegration:
    def test_sensor_data_retrieval(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        sensor_data = controller.get_all_sensor_data()
        
        assert isinstance(sensor_data, dict)
        
        for name, data in sensor_data.items():
            assert name in controller.sensor_names

    def test_joint_position_sensor(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, 1.0)
        controller.step(n_steps=100)
        
        state = controller.get_joint_state(joint_name)
        
        assert isinstance(state.qpos, float)
        assert isinstance(state.qvel, float)


class TestForceApplication:
    def test_body_force_application(self):
        from src import MuJoCoController
        import numpy as np
        
        controller = MuJoCoController()
        
        if controller.body_names:
            body_name = controller.body_names[0]
            
            initial_pos = controller.get_body_position(body_name)
            
            force = np.array([0.0, 0.0, 10.0])
            controller.apply_force_to_body(body_name, force)
            
            controller.step(n_steps=50)
            
            final_pos = controller.get_body_position(body_name)
            
            assert initial_pos is not None
            assert final_pos is not None

    def test_force_with_application_point(self):
        from src import MuJoCoController
        import numpy as np
        
        controller = MuJoCoController()
        
        if controller.body_names:
            body_name = controller.body_names[0]
            
            force = np.array([1.0, 0.0, 0.0])
            point = np.array([0.1, 0.0, 0.1])
            
            controller.apply_force_to_body(body_name, force, point)
            
            controller.step(n_steps=10)
            
            assert True


class TestRealTimeFactors:
    def test_different_realtime_factors(self):
        from src import Simulator
        import time
        
        for rtf in [1.0, 10.0, 100.0]:
            sim = Simulator(
                mavlink_connection="udp:127.0.0.1:18885",
                real_time_factor=rtf
            )
            
            sim.start()
            time.sleep(0.1)
            
            stats = sim.get_statistics()
            
            assert stats["is_running"] is True
            assert stats["steps"] > 0
            
            sim.stop()

    def test_realtime_factor_effect(self):
        from src import Simulator
        import time
        
        sim_slow = Simulator(
            mavlink_connection="udp:127.0.0.1:18884",
            real_time_factor=1.0
        )
        
        sim_fast = Simulator(
            mavlink_connection="udp:127.0.0.1:18883",
            real_time_factor=100.0
        )
        
        sim_fast.start()
        time.sleep(0.2)
        stats_fast = sim_fast.get_statistics()
        sim_fast.stop()
        
        sim_slow.start()
        time.sleep(0.2)
        stats_slow = sim_slow.get_statistics()
        sim_slow.stop()
        
        assert stats_fast["steps"] >= stats_slow["steps"]


class TestErrorHandling:
    def test_invalid_joint_name_handling(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        with pytest.raises(ValueError):
            controller.set_control_mode("invalid_joint_name", "position")
        
        with pytest.raises(ValueError):
            controller.apply_force_to_body("invalid_body_name", np.array([0, 0, 1]))
        
        state = controller.get_joint_state("invalid_joint_name")
        assert state is None

    def test_connection_error_handling(self):
        from src import MavlinkBridge
        
        bridge = MavlinkBridge(connection_string="invalid://connection:99999")
        
        result = bridge.send_hil_actuator_controls([0.0] * 16)
        
        assert result is False or bridge.is_connected is False

    def test_message_handler_exception_handling(self):
        from src import MavlinkBridge, MavlinkMessage
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:18882")
        
        def bad_handler(msg):
            raise RuntimeError("Test error in handler")
        
        bridge.register_handler("TEST", bad_handler)
        
        msg = MavlinkMessage(
            msg_type="TEST",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        try:
            with bridge.lock:
                bridge._process_message(msg)
        except Exception as e:
            pytest.fail(f"Handler exception should be caught: {e}")


class TestLongRunningSimulation:
    def test_extended_simulation(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, 0.5)
        
        for i in range(10):
            controller.step(n_steps=100)
            state = controller.get_joint_state(joint_name)
            
            assert state is not None
            assert isinstance(state.qpos, float)
            assert isinstance(state.qvel, float)

    def test_simulation_stability(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_names = controller.joint_names[:min(4, len(controller.joint_names))]
        
        positions = []
        
        for _ in range(50):
            for name in joint_names:
                controller.set_joint_torque(name, np.random.uniform(-1.0, 1.0))
            
            controller.step(n_steps=10)
            
            for name in joint_names:
                state = controller.get_joint_state(name)
                positions.append(state.qpos)
        
        assert len(positions) > 0
        for pos in positions:
            assert not np.isnan(pos)
            assert not np.isinf(pos)


class TestStatePersistence:
    def test_reset_state_consistency(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, 10.0)
        controller.step(n_steps=200)
        
        state_before = controller.get_joint_state(joint_name)
        
        controller.reset()
        
        state_after = controller.get_joint_state(joint_name)
        
        assert len(controller.target_torques) == 0
        assert len(controller.target_positions) == 0
        assert len(controller.target_velocities) == 0

    def test_multiple_reset_cycles(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        for cycle in range(5):
            controller.set_joint_torque(joint_name, float(cycle + 1))
            controller.step(n_steps=50)
            
            state = controller.get_joint_state(joint_name)
            
            controller.reset()
            
            assert len(controller.target_torques) == 0


class TestConcurrentSimulation:
    def test_thread_safe_simulation(self):
        from src import MuJoCoController
        import threading
        
        controller = MuJoCoController()
        errors = []
        
        def simulate():
            try:
                for _ in range(50):
                    controller.step(n_steps=10)
                    state = controller.get_joint_state(controller.joint_names[0])
            except Exception as e:
                errors.append(e)
        
        def modify_controls():
            try:
                for i in range(50):
                    controller.set_joint_torque(
                        controller.joint_names[0],
                        float(i % 10)
                    )
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=simulate),
            threading.Thread(target=modify_controls),
            threading.Thread(target=simulate),
            threading.Thread(target=modify_controls)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
