import pytest
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestControlTargetType:
    def test_control_target_type_enum(self, control_target_type_enum):
        ctt = control_target_type_enum
        
        assert ctt.JOINT_POSITION.value == "joint_position"
        assert ctt.JOINT_VELOCITY.value == "joint_velocity"
        assert ctt.JOINT_TORQUE.value == "joint_torque"
        assert ctt.BODY_FORCE.value == "body_force"
        assert ctt.BODY_TORQUE.value == "body_torque"
        assert ctt.CUSTOM.value == "custom"


class TestMavlinkControl:
    def test_mavlink_control_creation(self):
        from src import MavlinkControl, ControlTargetType
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=2.0,
            offset=0.5,
            range_min=-10.0,
            range_max=10.0,
            enabled=True
        )
        
        assert ctrl.mavlink_index == 0
        assert ctrl.target_name == "joint1"
        assert ctrl.target_type == ControlTargetType.JOINT_TORQUE
        assert ctrl.scale == 2.0
        assert ctrl.offset == 0.5
        assert ctrl.range_min == -10.0
        assert ctrl.range_max == 10.0
        assert ctrl.enabled is True

    def test_mavlink_control_defaults(self):
        from src import MavlinkControl, ControlTargetType
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE
        )
        
        assert ctrl.scale == 1.0
        assert ctrl.offset == 0.0
        assert ctrl.range_min == -np.inf
        assert ctrl.range_max == np.inf
        assert ctrl.enabled is True
        assert ctrl.transform is None

    def test_mavlink_control_apply_basic(self):
        from src import MavlinkControl, ControlTargetType
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=1.0,
            offset=0.0
        )
        
        result = ctrl.apply(0.5)
        
        assert result == 0.5

    def test_mavlink_control_apply_with_scale(self):
        from src import MavlinkControl, ControlTargetType
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=2.0,
            offset=0.0
        )
        
        result = ctrl.apply(0.5)
        
        assert result == 1.0

    def test_mavlink_control_apply_with_offset(self):
        from src import MavlinkControl, ControlTargetType
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=1.0,
            offset=1.0
        )
        
        result = ctrl.apply(0.5)
        
        assert result == 1.5

    def test_mavlink_control_apply_with_scale_and_offset(self):
        from src import MavlinkControl, ControlTargetType
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=2.0,
            offset=1.0
        )
        
        result = ctrl.apply(0.5)
        
        assert result == 2.0

    def test_mavlink_control_apply_with_clamping(self):
        from src import MavlinkControl, ControlTargetType
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=1.0,
            offset=0.0,
            range_min=-1.0,
            range_max=1.0
        )
        
        assert ctrl.apply(2.0) == 1.0
        assert ctrl.apply(-2.0) == -1.0
        assert ctrl.apply(0.5) == 0.5

    def test_mavlink_control_apply_with_transform(self):
        from src import MavlinkControl, ControlTargetType
        
        def custom_transform(x):
            return x * x
        
        ctrl = MavlinkControl(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            transform=custom_transform
        )
        
        result = ctrl.apply(3.0)
        
        assert result == 9.0


class TestMappedControlResult:
    def test_mapped_control_result_creation(self):
        from src import MappedControlResult, ControlTargetType
        
        result = MappedControlResult(
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            value=1.5,
            raw_value=0.75,
            mavlink_index=0
        )
        
        assert result.target_name == "joint1"
        assert result.target_type == ControlTargetType.JOINT_TORQUE
        assert result.value == 1.5
        assert result.raw_value == 0.75
        assert result.mavlink_index == 0


class TestControlMapper:
    def test_control_mapper_creation(self, control_mapper):
        mapper = control_mapper
        
        assert len(mapper.mappings) == 0

    def test_add_mapping(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=1.0,
            offset=0.0
        )
        
        assert 0 in mapper.mappings
        assert mapper.get_mapping(0) is not None

    def test_get_mapping(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=5,
            target_name="test_joint",
            target_type=ControlTargetType.JOINT_POSITION
        )
        
        mapping = mapper.get_mapping(5)
        
        assert mapping is not None
        assert mapping.target_name == "test_joint"

    def test_get_mapping_nonexistent(self, control_mapper):
        mapper = control_mapper
        
        mapping = mapper.get_mapping(999)
        
        assert mapping is None

    def test_get_mapping_by_name(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="my_joint",
            target_type=ControlTargetType.JOINT_TORQUE
        )
        
        mapping = mapper.get_mapping_by_name("my_joint")
        
        assert mapping is not None
        assert mapping.mavlink_index == 0

    def test_get_mapping_by_name_nonexistent(self, control_mapper):
        mapper = control_mapper
        
        mapping = mapper.get_mapping_by_name("nonexistent")
        
        assert mapping is None

    def test_remove_mapping(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE
        )
        
        assert 0 in mapper.mappings
        
        result = mapper.remove_mapping(0)
        
        assert result is True
        assert 0 not in mapper.mappings

    def test_remove_mapping_nonexistent(self, control_mapper):
        mapper = control_mapper
        
        result = mapper.remove_mapping(999)
        
        assert result is False

    def test_clear_mappings(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        for i in range(5):
            mapper.add_mapping(
                mavlink_index=i,
                target_name=f"joint{i}",
                target_type=ControlTargetType.JOINT_TORQUE
            )
        
        assert len(mapper.mappings) == 5
        
        mapper.clear_mappings()
        
        assert len(mapper.mappings) == 0

    def test_map_controls(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=2.0
        )
        
        mapper.add_mapping(
            mavlink_index=1,
            target_name="joint2",
            target_type=ControlTargetType.JOINT_POSITION,
            scale=1.0,
            offset=1.0
        )
        
        raw_controls = [0.5, 0.5, 0.0, 0.0]
        
        results = mapper.map_controls(raw_controls)
        
        assert len(results) == 2
        
        for result in results:
            if result.target_name == "joint1":
                assert result.value == 1.0
            elif result.target_name == "joint2":
                assert result.value == 1.5

    def test_map_single_control(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=3,
            target_name="test_joint",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=0.5
        )
        
        result = mapper.map_single_control(3, 2.0)
        
        assert result is not None
        assert result.target_name == "test_joint"
        assert result.value == 1.0

    def test_map_single_control_nonexistent(self, control_mapper):
        mapper = control_mapper
        
        result = mapper.map_single_control(999, 1.0)
        
        assert result is None

    def test_enable_disable_mapping(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            enabled=True
        )
        
        mapper.disable_mapping(0)
        
        mapping = mapper.get_mapping(0)
        assert mapping.enabled is False
        
        results = mapper.map_controls([1.0])
        assert len(results) == 0
        
        mapper.enable_mapping(0)
        mapping = mapper.get_mapping(0)
        assert mapping.enabled is True
        
        results = mapper.map_controls([1.0])
        assert len(results) == 1

    def test_enabled_mappings(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            enabled=True
        )
        
        mapper.add_mapping(
            mavlink_index=1,
            target_name="joint2",
            target_type=ControlTargetType.JOINT_TORQUE,
            enabled=False
        )
        
        assert len(mapper.mappings) == 2
        assert len(mapper.enabled_mappings) == 1

    def test_create_default_joint_mappings(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        joint_names = ["joint1", "joint2", "joint3"]
        
        mapper.create_default_joint_mappings(
            joint_names=joint_names,
            control_type=ControlTargetType.JOINT_TORQUE,
            start_index=0
        )
        
        assert len(mapper.mappings) == 3
        
        for i, name in enumerate(joint_names):
            mapping = mapper.get_mapping(i)
            assert mapping is not None
            assert mapping.target_name == name
            assert mapping.target_type == ControlTargetType.JOINT_TORQUE

    def test_create_default_joint_mappings_with_start_index(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        joint_names = ["joint1", "joint2"]
        
        mapper.create_default_joint_mappings(
            joint_names=joint_names,
            control_type=ControlTargetType.JOINT_POSITION,
            start_index=5
        )
        
        assert 5 in mapper.mappings
        assert 6 in mapper.mappings

    def test_to_control_mode(self, control_mapper):
        from src import ControlTargetType, ControlMode
        
        mapper = control_mapper
        
        assert mapper.to_control_mode(ControlTargetType.JOINT_POSITION) == ControlMode.POSITION
        assert mapper.to_control_mode(ControlTargetType.JOINT_VELOCITY) == ControlMode.VELOCITY
        assert mapper.to_control_mode(ControlTargetType.JOINT_TORQUE) == ControlMode.TORQUE
        assert mapper.to_control_mode(ControlTargetType.CUSTOM) == ControlMode.CUSTOM

    def test_add_custom_transform(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        def custom_sin(x):
            return math.sin(x)
        
        mapper.add_custom_transform("sin", custom_sin)
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            transform="sin"
        )
        
        mapping = mapper.get_mapping(0)
        
        result = mapping.apply(math.pi / 2)
        
        assert abs(result - 1.0) < 1e-6


class TestControlMapperEdgeCases:
    def test_map_controls_empty(self, control_mapper):
        mapper = control_mapper
        
        results = mapper.map_controls([])
        
        assert results == []

    def test_map_controls_no_mappings(self, control_mapper):
        mapper = control_mapper
        
        results = mapper.map_controls([1.0, 2.0, 3.0])
        
        assert results == []

    def test_add_duplicate_mapping(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE
        )
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="different_joint",
            target_type=ControlTargetType.JOINT_POSITION
        )
        
        mapping = mapper.get_mapping(0)
        
        assert mapping.target_name == "different_joint"

    def test_negative_values(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            scale=1.0,
            offset=0.0
        )
        
        result = mapper.map_single_control(0, -1.0)
        
        assert result.value == -1.0

    def test_large_values(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE
        )
        
        result = mapper.map_single_control(0, 1e6)
        
        assert result.value == 1e6

    def test_clamping_extreme_values(self, control_mapper):
        from src import ControlTargetType
        
        mapper = control_mapper
        
        mapper.add_mapping(
            mavlink_index=0,
            target_name="joint1",
            target_type=ControlTargetType.JOINT_TORQUE,
            range_min=-5.0,
            range_max=5.0
        )
        
        assert mapper.map_single_control(0, 10.0).value == 5.0
        assert mapper.map_single_control(0, -10.0).value == -5.0
