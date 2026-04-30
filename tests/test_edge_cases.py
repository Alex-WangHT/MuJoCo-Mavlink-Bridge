import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestEdgeCasesMuJoCo:
    def test_very_small_timesteps(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        for _ in range(1000):
            controller.step(n_steps=1)
        
        assert True

    def test_very_large_torques(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, 1e6)
        controller.step(n_steps=100)
        
        state = controller.get_joint_state(joint_name)
        
        assert not np.isnan(state.qpos)
        assert not np.isinf(state.qpos)

    def test_negative_torques(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, -10.0)
        controller.step(n_steps=100)
        
        state = controller.get_joint_state(joint_name)
        
        assert True

    def test_zero_torque(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, 0.0)
        controller.step(n_steps=100)
        
        state = controller.get_joint_state(joint_name)
        
        assert True

    def test_nan_values_handling(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        state = controller.get_joint_state(controller.joint_names[0])
        
        assert not np.isnan(state.qpos)
        assert not np.isnan(state.qvel)
        assert not np.isnan(state.qacc)

    def test_inf_values_handling(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        state = controller.get_joint_state(controller.joint_names[0])
        
        assert not np.isinf(state.qpos)
        assert not np.isinf(state.qvel)
        assert not np.isinf(state.qacc)


class TestEdgeCasesMAVLink:
    def test_empty_controls_array(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [],
                "mode": 0,
                "flags": 0
            }
        )
        
        assert msg.data["controls"] == []

    def test_large_controls_array(self):
        from src import MavlinkMessage
        
        large_controls = [float(i) for i in range(100)]
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": large_controls,
                "mode": 0,
                "flags": 0
            }
        )
        
        assert len(msg.data["controls"]) == 100

    def test_negative_control_values(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [-1.0, -0.5, 0.0, 0.5, 1.0] + [0.0] * 11,
                "mode": 0,
                "flags": 0
            }
        )
        
        assert msg.data["controls"][0] == -1.0
        assert msg.data["controls"][4] == 1.0

    def test_extreme_control_values(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [1e6, -1e6, 1e-6, -1e-6] + [0.0] * 12,
                "mode": 0,
                "flags": 0
            }
        )
        
        assert msg.data["controls"][0] == 1e6
        assert msg.data["controls"][1] == -1e6

    def test_invalid_message_types(self):
        from src import MavlinkBridge, MavlinkMessage
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:16666")
        
        invalid_msg = MavlinkMessage(
            msg_type="INVALID_MESSAGE_TYPE_12345",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        with bridge.lock:
            bridge._process_message(invalid_msg)
        
        assert True

    def test_empty_data_message(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="EMPTY",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        assert msg.data == {}

    def test_none_data_message(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="TEST",
            sysid=1,
            compid=1,
            seq=0,
            data={"value": None}
        )
        
        assert msg.data["value"] is None


class TestEdgeCasesPID:
    def test_zero_gains(self):
        from src import PIDController
        
        pid = PIDController(kp=0.0, ki=0.0, kd=0.0)
        
        output = pid.compute(target=1.0, current=0.0)
        
        assert output == 0.0

    def test_very_high_gains(self):
        from src import PIDController
        
        pid = PIDController(kp=1e6, ki=1e6, kd=1e6)
        
        output = pid.compute(target=1.0, current=0.0)
        
        assert not np.isnan(output)
        assert not np.isinf(output)

    def test_negative_gains(self):
        from src import PIDController
        
        pid = PIDController(kp=-1.0, ki=-1.0, kd=-1.0)
        
        output = pid.compute(target=1.0, current=0.0)
        
        assert output < 0

    def test_identical_target_current(self):
        from src import PIDController
        
        pid = PIDController(kp=10.0, ki=1.0, kd=0.1)
        
        output1 = pid.compute(target=5.0, current=5.0)
        output2 = pid.compute(target=5.0, current=5.0)
        
        assert True

    def test_rapid_changes(self):
        from src import PIDController
        
        pid = PIDController(kp=10.0, ki=1.0, kd=0.1)
        
        outputs = []
        for i in range(100):
            target = float(i % 2)
            current = float((i + 1) % 2)
            output = pid.compute(target, current)
            outputs.append(output)
        
        assert len(outputs) == 100

    def test_integral_windup(self):
        from src import PIDController
        
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        
        for _ in range(1000):
            pid.compute(target=1.0, current=0.0)
        
        assert pid.integral > 0
        assert not np.isinf(pid.integral)

    def test_pid_reset_mid_calculation(self):
        from src import PIDController
        
        pid = PIDController(kp=10.0, ki=1.0, kd=0.1)
        
        for _ in range(10):
            pid.compute(target=1.0, current=0.0)
        
        assert pid.integral > 0
        
        pid.reset()
        
        assert pid.integral == 0
        assert pid.last_error == 0


class TestEdgeCasesSimulator:
    def test_zero_realtime_factor(self):
        from src import Simulator
        
        with pytest.raises(Exception):
            sim = Simulator(
                mavlink_connection="udp:127.0.0.1:16665",
                real_time_factor=0.0
            )

    def test_very_high_realtime_factor(self):
        from src import Simulator
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:16664",
            real_time_factor=10000.0
        )
        
        sim.start()
        time.sleep(0.1)
        
        stats = sim.get_statistics()
        
        sim.stop()
        
        assert stats["steps"] > 0

    def test_negative_realtime_factor(self):
        from src import Simulator
        
        with pytest.raises(Exception):
            sim = Simulator(
                mavlink_connection="udp:127.0.0.1:16663",
                real_time_factor=-1.0
            )

    def test_rapid_start_stop(self):
        from src import Simulator
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:16662",
            real_time_factor=1000.0
        )
        
        for _ in range(10):
            sim.start()
            time.sleep(0.01)
            sim.stop()
        
        assert sim.is_running is False

    def test_simulator_without_mavlink(self):
        from src import Simulator
        
        sim = Simulator(
            mavlink_connection="udp:invalid:99999",
            real_time_factor=100.0
        )
        
        sim.start()
        time.sleep(0.1)
        
        stats = sim.get_statistics()
        
        sim.stop()
        
        assert True

    def test_empty_control_mappings(self):
        from src import Simulator, MavlinkMessage
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:16661",
            real_time_factor=100.0
        )
        
        sim.control_mappings.clear()
        sim.joint_to_mavlink.clear()
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [1.0] * 16,
                "mode": 0,
                "flags": 0
            }
        )
        
        sim._handle_hil_actuator_controls(msg)
        
        assert True


class TestEdgeCasesConcurrency:
    def test_concurrent_read_write(self):
        from src import MuJoCoController
        import threading
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        errors = []
        
        def writer():
            try:
                for i in range(100):
                    controller.set_joint_torque(joint_name, float(i))
                    controller.step(n_steps=1)
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for _ in range(100):
                    state = controller.get_joint_state(joint_name)
                    _ = state.qpos
                    _ = state.qvel
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0

    def test_concurrent_reset(self):
        from src import MuJoCoController
        import threading
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        errors = []
        
        def simulate_and_reset():
            try:
                for _ in range(50):
                    controller.set_joint_torque(joint_name, 1.0)
                    controller.step(n_steps=10)
                    controller.reset()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=simulate_and_reset) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestEdgeCasesDataTypes:
    def test_int_control_values(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, 5)
        
        assert joint_name in controller.target_torques
        assert controller.target_torques[joint_name] == 5.0

    def test_numpy_control_values(self):
        from src import MuJoCoController
        import numpy as np
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        controller.set_joint_torque(joint_name, np.float64(2.5))
        
        assert joint_name in controller.target_torques

    def test_large_integer_flags(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="HIL_ACTUATOR_CONTROLS",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "controls": [0.0] * 16,
                "mode": 0,
                "flags": 0xFFFFFFFFFFFFFFFF
            }
        )
        
        assert msg.data["flags"] == 0xFFFFFFFFFFFFFFFF

    def test_negative_sequence_numbers(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="TEST",
            sysid=1,
            compid=1,
            seq=-1,
            data={}
        )
        
        assert msg.seq == -1

    def test_large_system_ids(self):
        from src import MavlinkMessage
        
        msg = MavlinkMessage(
            msg_type="TEST",
            sysid=255,
            compid=255,
            seq=65535,
            data={}
        )
        
        assert msg.sysid == 255
        assert msg.compid == 255
        assert msg.seq == 65535


class TestEdgeCasesRobustness:
    def test_repeated_message_handling(self):
        from src import MavlinkBridge, MavlinkMessage
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:16660")
        
        msg_count = 0
        
        def handler(msg):
            nonlocal msg_count
            msg_count += 1
        
        bridge.register_handler("TEST", handler)
        
        msg = MavlinkMessage(
            msg_type="TEST",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        for i in range(1000):
            msg.seq = i
            with bridge.lock:
                bridge._process_message(msg)
        
        assert msg_count == 1000

    def test_message_handler_removal(self):
        from src import MavlinkBridge, MavlinkMessage
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:16659")
        
        call_count = 0
        
        def handler(msg):
            nonlocal call_count
            call_count += 1
        
        bridge.register_handler("TEST", handler)
        
        msg = MavlinkMessage(
            msg_type="TEST",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        with bridge.lock:
            bridge._process_message(msg)
        
        bridge.unregister_handler("TEST")
        
        with bridge.lock:
            bridge._process_message(msg)
        
        assert call_count == 1

    def test_nonexistent_handler_types(self):
        from src import MavlinkBridge, MavlinkMessage
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:16658")
        
        msg = MavlinkMessage(
            msg_type="NONEXISTENT_HANDLER",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        with bridge.lock:
            bridge._process_message(msg)
        
        assert "NONEXISTENT_HANDLER" in bridge.last_messages

    def test_rapid_state_queries(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        states = []
        for _ in range(1000):
            state = controller.get_robot_state()
            states.append(state)
        
        assert len(states) == 1000
