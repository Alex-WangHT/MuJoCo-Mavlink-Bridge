import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMavlinkUDPBridgeInitialization:
    def test_bridge_creation(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        assert bridge is not None

    def test_bridge_not_connected_initially(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        assert bridge.is_connected is False
        assert bridge.is_running is False


class TestMavlinkUDPBridgeConnection:
    def test_connect(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        result = bridge.connect()
        
        assert result is True
        assert bridge.is_connected is True

    def test_disconnect(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        bridge.connect()
        assert bridge.is_connected is True
        
        bridge.disconnect()
        assert bridge.is_connected is False


class TestMavlinkUDPBridgeMessages:
    def test_create_hil_actuator_controls_message(self, mavlink_bridge):
        from src import MessageType
        
        bridge = mavlink_bridge
        
        controls = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 11
        
        msg = bridge.create_hil_actuator_controls_message(
            controls=controls,
            mode=0,
            flags=0
        )
        
        assert msg.msg_type == MessageType.HIL_ACTUATOR_CONTROLS
        assert msg.msg_name == "HIL_ACTUATOR_CONTROLS"
        assert msg.get("controls") == controls

    def test_create_heartbeat_message(self, mavlink_bridge):
        from src import MessageType
        
        bridge = mavlink_bridge
        
        msg = bridge.create_heartbeat_message(
            type_val=2,
            autopilot=12,
            base_mode=0,
            custom_mode=0,
            system_status=4
        )
        
        assert msg.msg_type == MessageType.HEARTBEAT
        assert msg.msg_name == "HEARTBEAT"

    def test_send_hil_actuator_controls(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        bridge.connect()
        
        controls = [0.0] * 16
        result = bridge.send_hil_actuator_controls(controls)
        
        assert isinstance(result, bool)

    def test_send_heartbeat(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        bridge.connect()
        
        result = bridge.send_heartbeat()
        
        assert isinstance(result, bool)


class TestMavlinkUDPBridgeHandlers:
    def test_register_handler(self, mavlink_bridge):
        from src import MessageType
        
        bridge = mavlink_bridge
        
        def handler(msg):
            pass
        
        bridge.register_handler(MessageType.HIL_ACTUATOR_CONTROLS, handler)
        
        assert MessageType.HIL_ACTUATOR_CONTROLS in bridge.registered_types

    def test_register_named_handler(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        def handler(msg):
            pass
        
        bridge.register_named_handler("CUSTOM_MESSAGE", handler)
        
        assert "CUSTOM_MESSAGE" in bridge.registered_names

    def test_unregister_handler(self, mavlink_bridge):
        from src import MessageType
        
        bridge = mavlink_bridge
        
        def handler(msg):
            pass
        
        bridge.register_handler(MessageType.HEARTBEAT, handler)
        assert MessageType.HEARTBEAT in bridge.registered_types
        
        bridge.unregister_handler(MessageType.HEARTBEAT)
        assert MessageType.HEARTBEAT not in bridge.registered_types

    def test_unregister_named_handler(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        def handler(msg):
            pass
        
        bridge.register_named_handler("TEST", handler)
        assert "TEST" in bridge.registered_names
        
        bridge.unregister_named_handler("TEST")
        assert "TEST" not in bridge.registered_names


class TestMavlinkUDPBridgeStatistics:
    def test_get_statistics(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        stats = bridge.get_statistics()
        
        assert isinstance(stats, dict)
        assert "messages_received" in stats
        assert "messages_sent" in stats
        assert "errors" in stats

    def test_statistics_initial_state(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        stats = bridge.get_statistics()
        
        assert stats["messages_received"] == 0
        assert stats["messages_sent"] == 0
        assert stats["errors"] == 0


class TestMavlinkUDPBridgeLifecycle:
    def test_start_stop(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        bridge.connect()
        
        assert bridge.is_running is False
        
        bridge.start()
        assert bridge.is_running is True
        
        bridge.stop()
        assert bridge.is_running is False

    def test_double_start(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        bridge.connect()
        bridge.start()
        
        bridge.start()
        
        bridge.stop()
        
        assert True

    def test_stop_without_start(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        bridge.stop()
        
        assert bridge.is_running is False


class TestMavlinkMessageParser:
    def test_parse_hil_actuator_controls(self, basic_mavlink_message):
        from src import MavlinkMessageParser
        
        msg = basic_mavlink_message
        
        result = MavlinkMessageParser.parse_hil_actuator_controls(msg)
        
        assert "controls" in result
        assert "mode" in result
        assert "flags" in result

    def test_extract_joint_controls(self, basic_mavlink_message):
        from src import MavlinkMessageParser
        
        msg = basic_mavlink_message
        
        controls = MavlinkMessageParser.extract_joint_controls(msg, num_joints=4)
        
        assert len(controls) == 4
        assert controls[0] == 0.5
        assert controls[1] == 0.3

    def test_get_control_value(self, basic_mavlink_message):
        from src import MavlinkMessageParser
        
        msg = basic_mavlink_message
        
        value = MavlinkMessageParser.get_control_value(msg, index=0, default=0.0)
        
        assert value == 0.5

    def test_get_control_value_default(self, basic_mavlink_message):
        from src import MavlinkMessageParser
        
        msg = basic_mavlink_message
        
        value = MavlinkMessageParser.get_control_value(msg, index=999, default=-1.0)
        
        assert value == -1.0

    def test_parse_heartbeat(self):
        from src import MavlinkMessage, MessageType, MavlinkMessageParser
        
        msg = MavlinkMessage(
            msg_type=MessageType.HEARTBEAT,
            msg_name="HEARTBEAT",
            sysid=1,
            compid=1,
            seq=0,
            data={
                "type": 2,
                "autopilot": 12,
                "base_mode": 0,
                "custom_mode": 0,
                "system_status": 4,
                "mavlink_version": 3
            }
        )
        
        result = MavlinkMessageParser.parse_heartbeat(msg)
        
        assert result["type"] == 2
        assert result["autopilot"] == 12
        assert result["mavlink_version"] == 3

    def test_is_valid_message(self, basic_mavlink_message):
        from src import MavlinkMessageParser
        
        assert MavlinkMessageParser.is_valid_message(basic_mavlink_message) is True


class TestCreateMessageFromType:
    def test_create_message_from_type(self):
        from src import create_message_from_type, MessageType
        
        msg = create_message_from_type(
            msg_type=MessageType.HIL_ACTUATOR_CONTROLS,
            sysid=2,
            compid=3,
            seq=100,
            controls=[0.1, 0.2],
            mode=0
        )
        
        assert msg.msg_type == MessageType.HIL_ACTUATOR_CONTROLS
        assert msg.sysid == 2
        assert msg.compid == 3
        assert msg.seq == 100
        assert msg.get("controls") == [0.1, 0.2]
        assert msg.get("mode") == 0


class TestMavlinkUDPBridgeEdgeCases:
    def test_send_without_connection(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        result = bridge.send_hil_actuator_controls([0.0] * 16)
        
        assert result is False

    def test_send_heartbeat_without_connection(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        result = bridge.send_heartbeat()
        
        assert result is False

    def test_start_without_connection(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        with pytest.raises(RuntimeError):
            bridge.start()

    def test_empty_controls(self, mavlink_bridge):
        from src import MessageType
        
        bridge = mavlink_bridge
        
        msg = bridge.create_hil_actuator_controls_message(
            controls=[],
            mode=0,
            flags=0
        )
        
        assert msg.get("controls") == []

    def test_large_controls_array(self, mavlink_bridge):
        from src import MessageType
        
        bridge = mavlink_bridge
        
        large_controls = [float(i) for i in range(100)]
        
        msg = bridge.create_hil_actuator_controls_message(
            controls=large_controls,
            mode=0,
            flags=0
        )
        
        assert len(msg.get("controls")) == 100

    def test_negative_control_values(self, mavlink_bridge):
        from src import MessageType
        
        bridge = mavlink_bridge
        
        controls = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        msg = bridge.create_hil_actuator_controls_message(
            controls=controls,
            mode=0,
            flags=0
        )
        
        assert msg.get("controls") == controls

    def test_get_last_message_none(self, mavlink_bridge):
        bridge = mavlink_bridge
        
        from src import MessageType
        
        msg = bridge.get_last_message(MessageType.HEARTBEAT)
        
        assert msg is None
