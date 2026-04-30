import pytest
import time
import threading
from unittest.mock import MagicMock, patch, call
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMavlinkMessage:
    def test_message_creation(self, sample_mavlink_message):
        msg = sample_mavlink_message
        assert msg.msg_type == "HIL_ACTUATOR_CONTROLS"
        assert msg.sysid == 1
        assert msg.compid == 1
        assert msg.seq == 42
        assert "controls" in msg.data
        assert msg.data["controls"][0] == 0.5

    def test_message_data_modification(self):
        from src import MavlinkMessage
        msg = MavlinkMessage(
            msg_type="HEARTBEAT",
            sysid=2,
            compid=3,
            seq=100,
            data={"type": 2, "autopilot": 12}
        )
        msg.data["custom_mode"] = 0
        assert msg.data["custom_mode"] == 0
        assert msg.msg_type == "HEARTBEAT"


class TestMavlinkBridge:
    def test_bridge_initialization(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(
            connection_string="udp:127.0.0.1:19998",
            source_system=2,
            source_component=5
        )
        assert bridge.source_system == 2
        assert bridge.source_component == 5
        assert bridge.target_system == 1
        assert bridge.is_running is False

    def test_bridge_connection_string_parsing(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:0.0.0.0:14550")
        assert bridge.connection_string == "udp:0.0.0.0:14550"

    def test_register_handler(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19997")
        
        def handler(msg):
            pass
        
        bridge.register_handler("HIL_ACTUATOR_CONTROLS", handler)
        assert "HIL_ACTUATOR_CONTROLS" in bridge.message_handlers

    def test_unregister_handler(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19996")
        
        def handler(msg):
            pass
        
        bridge.register_handler("HEARTBEAT", handler)
        assert "HEARTBEAT" in bridge.message_handlers
        
        bridge.unregister_handler("HEARTBEAT")
        assert "HEARTBEAT" not in bridge.message_handlers

    def test_get_last_message_none(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19995")
        assert bridge.get_last_message("NON_EXISTENT") is None

    def test_message_handlers_called(self, sample_mavlink_message):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19994")
        
        received_messages = []
        
        def handler(msg):
            received_messages.append(msg)
        
        bridge.register_handler("HIL_ACTUATOR_CONTROLS", handler)
        
        with bridge.lock:
            bridge._process_message(sample_mavlink_message)
        
        assert len(received_messages) == 1
        assert received_messages[0].msg_type == "HIL_ACTUATOR_CONTROLS"

    def test_multiple_handlers_different_types(self, sample_mavlink_message):
        from src import MavlinkBridge, MavlinkMessage
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19993")
        
        hil_received = []
        heartbeat_received = []
        
        def hil_handler(msg):
            hil_received.append(msg)
        
        def heartbeat_handler(msg):
            heartbeat_received.append(msg)
        
        bridge.register_handler("HIL_ACTUATOR_CONTROLS", hil_handler)
        bridge.register_handler("HEARTBEAT", heartbeat_handler)
        
        heartbeat_msg = MavlinkMessage(
            msg_type="HEARTBEAT",
            sysid=1,
            compid=1,
            seq=0,
            data={"type": 2}
        )
        
        with bridge.lock:
            bridge._process_message(sample_mavlink_message)
            bridge._process_message(heartbeat_msg)
        
        assert len(hil_received) == 1
        assert len(heartbeat_received) == 1
        assert hil_received[0].msg_type == "HIL_ACTUATOR_CONTROLS"
        assert heartbeat_received[0].msg_type == "HEARTBEAT"

    def test_handler_exception_handling(self):
        from src import MavlinkBridge, MavlinkMessage
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19992")
        
        def bad_handler(msg):
            raise ValueError("Test error")
        
        bridge.register_handler("TEST", bad_handler)
        
        test_msg = MavlinkMessage(
            msg_type="TEST",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
        
        try:
            with bridge.lock:
                bridge._process_message(test_msg)
        except Exception as e:
            pytest.fail(f"Handler exception should be caught: {e}")

    def test_send_hil_actuator_controls_without_connection(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19991")
        bridge.is_connected = False
        
        result = bridge.send_hil_actuator_controls([0.0] * 16)
        assert result is False

    def test_send_heartbeat_without_connection(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19990")
        bridge.is_connected = False
        
        result = bridge.send_heartbeat()
        assert result is False

    def test_send_hil_state_quaternion_without_connection(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19989")
        bridge.is_connected = False
        
        result = bridge.send_hil_state_quaternion(
            [1.0, 0.0, 0.0, 0.0],
            0.0, 0.0, 0.0,
            0, 0, 0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        )
        assert result is False

    def test_wait_for_message_timeout(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19988")
        
        start_time = time.time()
        result = bridge.wait_for_message("NON_EXISTENT", timeout=0.1)
        elapsed = time.time() - start_time
        
        assert result is None
        assert elapsed < 1.0

    def test_last_messages_storage(self, sample_mavlink_message):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19987")
        
        with bridge.lock:
            bridge._process_message(sample_mavlink_message)
        
        stored = bridge.get_last_message("HIL_ACTUATOR_CONTROLS")
        assert stored is not None
        assert stored.msg_type == "HIL_ACTUATOR_CONTROLS"
        assert stored.seq == 42

    def test_start_stop_behavior(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19986")
        
        assert bridge.is_running is False
        
        bridge.start()
        assert bridge.is_running is True
        
        bridge.stop()
        assert bridge.is_running is False

    def test_double_start(self):
        from src import MavlinkBridge
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:19985")
        
        bridge.start()
        first_thread = bridge.receive_thread
        
        bridge.start()
        assert bridge.receive_thread == first_thread
        
        bridge.stop()
