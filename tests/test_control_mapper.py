"""
测试 ControlMapping 类 - 控制量映射

ControlMapping 用于将 MAVLink 消息映射到 Plant 控制量。

核心功能：
- add_mapping(): 添加映射关系
- map_controls(): 批量映射控制值
- map_single_control(): 映射单个控制值
- create_default_joint_mappings(): 创建默认映射

输出示例：
    [OUTPUT] ControlMapping: entries=4
    [OUTPUT]   [0] joint1 (scale=2.0, offset=0.0, range=[-1.0, 1.0])
    [OUTPUT]   [1] joint2 (scale=1.0, offset=1.0)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestControlMappingBasics:
    """
    测试 ControlMapping 基础功能
    """
    
    def test_create_default_joint_mappings(self, control_mapping):
        """
        测试 create_default_joint_mappings() - 创建默认映射
        
        验证：
        1. 映射条目数量正确
        2. 每个条目都有正确的 mavlink_index 和 plant_control_name
        
        输出：
            [TEST] create_default_joint_mappings:
            [TEST]   entries count: 4
            [TEST]   entry 0: mavlink_index=0, plant_control_name=joint1
            [TEST]   entry 1: mavlink_index=1, plant_control_name=joint2
            [TEST]   entry 2: mavlink_index=2, plant_control_name=joint3
            [TEST]   entry 3: mavlink_index=3, plant_control_name=joint4
        """
        cm = control_mapping
        
        print(f"\n  [TEST] test_create_default_joint_mappings:")
        print(f"    entries count: {len(cm.entries)}")
        
        for i, entry in enumerate(cm.entries):
            print(f"    entry {i}: mavlink_index={entry.mavlink_index}, plant_control_name={entry.plant_control_name}")
        
        assert len(cm.entries) == 4
        
        for i in range(4):
            entry = cm.get_mapping(cm.entries[0].mavlink_source, i)
            assert entry is not None
            assert entry.plant_control_name == f"joint{i+1}"
    
    def test_add_mapping(self):
        """
        测试 add_mapping() - 添加自定义映射
        
        验证：
        1. 可以添加带缩放、偏移、限幅的映射
        2. 映射可以正确查找
        
        输出：
            [TEST] add_mapping:
            [TEST]   adding: mavlink_index=5 -> joint_test (scale=2.0, offset=1.0, range=[-5.0, 5.0])
            [TEST]   entry found: True
            [TEST]   scale=2.0, offset=1.0, range_min=-5.0, range_max=5.0
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        
        print(f"\n  [TEST] test_add_mapping:")
        print(f"    adding: mavlink_index=5 -> joint_test (scale=2.0, offset=1.0, range=[-5.0, 5.0])")
        
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=5,
            plant_control_name="joint_test",
            plant_control_index=10,
            scale=2.0,
            offset=1.0,
            range_min=-5.0,
            range_max=5.0
        )
        
        entry = cm.get_mapping(ControlSource.HIL_ACTUATOR_CONTROLS, 5)
        
        print(f"    entry found: {entry is not None}")
        if entry:
            print(f"    scale={entry.scale}, offset={entry.offset}, range_min={entry.range_min}, range_max={entry.range_max}")
        
        assert entry is not None
        assert entry.plant_control_name == "joint_test"
        assert entry.plant_control_index == 10
        assert entry.scale == 2.0
        assert entry.offset == 1.0
        assert entry.range_min == -5.0
        assert entry.range_max == 5.0
    
    def test_get_mapping(self, control_mapping):
        """
        测试 get_mapping() - 按索引获取映射
        
        验证：
        1. 存在的索引返回正确的映射
        2. 不存在的索引返回 None
        
        输出：
            [TEST] get_mapping:
            [TEST]   get_mapping(0) found: True (joint1)
            [TEST]   get_mapping(1) found: True (joint2)
            [TEST]   get_mapping(999) found: False
        """
        from src import ControlSource
        
        cm = control_mapping
        
        print(f"\n  [TEST] test_get_mapping:")
        
        entry0 = cm.get_mapping(ControlSource.HIL_ACTUATOR_CONTROLS, 0)
        entry1 = cm.get_mapping(ControlSource.HIL_ACTUATOR_CONTROLS, 1)
        entry_none = cm.get_mapping(ControlSource.HIL_ACTUATOR_CONTROLS, 999)
        
        print(f"    get_mapping(0) found: {entry0 is not None} ({entry0.plant_control_name if entry0 else 'None'})")
        print(f"    get_mapping(1) found: {entry1 is not None} ({entry1.plant_control_name if entry1 else 'None'})")
        print(f"    get_mapping(999) found: {entry_none is not None}")
        
        assert entry0 is not None
        assert entry1 is not None
        assert entry_none is None
    
    def test_get_mapping_by_name(self, control_mapping):
        """
        测试 get_mapping_by_name() - 按名称获取映射
        
        验证：
        1. 存在的名称返回正确的映射
        2. 不存在的名称返回 None
        
        输出：
            [TEST] get_mapping_by_name:
            [TEST]   get_mapping_by_name('joint1') found: True (index 0)
            [TEST]   get_mapping_by_name('joint2') found: True (index 1)
            [TEST]   get_mapping_by_name('nonexistent') found: False
        """
        cm = control_mapping
        
        print(f"\n  [TEST] test_get_mapping_by_name:")
        
        entry1 = cm.get_mapping_by_name("joint1")
        entry2 = cm.get_mapping_by_name("joint2")
        entry_none = cm.get_mapping_by_name("nonexistent")
        
        print(f"    get_mapping_by_name('joint1') found: {entry1 is not None} (index {entry1.mavlink_index if entry1 else 'None'})")
        print(f"    get_mapping_by_name('joint2') found: {entry2 is not None} (index {entry2.mavlink_index if entry2 else 'None'})")
        print(f"    get_mapping_by_name('nonexistent') found: {entry_none is not None}")
        
        assert entry1 is not None
        assert entry2 is not None
        assert entry_none is None


class TestControlMappingApply:
    """
    测试 ControlMapping 映射变换功能
    """
    
    def test_map_controls_basic(self, control_mapping):
        """
        测试 map_controls() - 批量映射控制值
        
        验证：
        1. 原始控制值被正确映射
        2. 只有有映射的索引被返回
        
        输出：
            [TEST] map_controls_basic:
            [TEST]   raw_controls: [0.5, 1.0, 0.3, 0.0]
            [TEST]   mapped: {0: 0.5, 1: 1.0, 2: 0.3, 3: 0.0}
            [TEST]   mapped count: 4
        """
        from src import ControlSource
        
        cm = control_mapping
        
        raw_controls = [0.5, 1.0, 0.3, 0.0]
        
        print(f"\n  [TEST] test_map_controls_basic:")
        print(f"    raw_controls: {raw_controls}")
        
        mapped = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, raw_controls)
        
        print(f"    mapped: {mapped}")
        print(f"    mapped count: {len(mapped)}")
        
        assert len(mapped) == 4
        assert mapped[0] == 0.5
        assert mapped[1] == 1.0
        assert mapped[2] == 0.3
        assert mapped[3] == 0.0
    
    def test_map_controls_with_scale(self):
        """
        测试 map_controls() - 带缩放因子的映射
        
        验证：
        1. 缩放因子正确应用：value = raw * scale + offset
        
        输出：
            [TEST] map_controls_with_scale:
            [TEST]   adding mapping: scale=2.0, offset=0.0
            [TEST]   raw_value=0.5 -> mapped_value=1.0
            [TEST]   raw_value=1.0 -> mapped_value=2.0
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        
        print(f"\n  [TEST] test_map_controls_with_scale:")
        print(f"    adding mapping: scale=2.0, offset=0.0")
        
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=0,
            plant_control_name="test",
            plant_control_index=0,
            scale=2.0,
            offset=0.0
        )
        
        mapped1 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.5])
        mapped2 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [1.0])
        
        print(f"    raw_value=0.5 -> mapped_value={mapped1.get(0)}")
        print(f"    raw_value=1.0 -> mapped_value={mapped2.get(0)}")
        
        assert mapped1[0] == 1.0
        assert mapped2[0] == 2.0
    
    def test_map_controls_with_offset(self):
        """
        测试 map_controls() - 带偏移的映射
        
        验证：
        1. 偏移量正确应用
        
        输出：
            [TEST] map_controls_with_offset:
            [TEST]   adding mapping: scale=1.0, offset=1.0
            [TEST]   raw_value=0.0 -> mapped_value=1.0
            [TEST]   raw_value=0.5 -> mapped_value=1.5
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        
        print(f"\n  [TEST] test_map_controls_with_offset:")
        print(f"    adding mapping: scale=1.0, offset=1.0")
        
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=0,
            plant_control_name="test",
            plant_control_index=0,
            scale=1.0,
            offset=1.0
        )
        
        mapped1 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.0])
        mapped2 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.5])
        
        print(f"    raw_value=0.0 -> mapped_value={mapped1.get(0)}")
        print(f"    raw_value=0.5 -> mapped_value={mapped2.get(0)}")
        
        assert mapped1[0] == 1.0
        assert mapped2[0] == 1.5
    
    def test_map_controls_with_clamping(self):
        """
        测试 map_controls() - 带限幅的映射
        
        验证：
        1. 超出范围的值被限幅
        
        输出：
            [TEST] map_controls_with_clamping:
            [TEST]   adding mapping: scale=1.0, range=[-1.0, 1.0]
            [TEST]   raw_value=2.0 -> mapped_value=1.0 (clamped)
            [TEST]   raw_value=-2.0 -> mapped_value=-1.0 (clamped)
            [TEST]   raw_value=0.5 -> mapped_value=0.5 (no clamp)
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        
        print(f"\n  [TEST] test_map_controls_with_clamping:")
        print(f"    adding mapping: scale=1.0, range=[-1.0, 1.0]")
        
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=0,
            plant_control_name="test",
            plant_control_index=0,
            scale=1.0,
            offset=0.0,
            range_min=-1.0,
            range_max=1.0
        )
        
        mapped1 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [2.0])
        mapped2 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [-2.0])
        mapped3 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.5])
        
        print(f"    raw_value=2.0 -> mapped_value={mapped1.get(0)} (clamped)")
        print(f"    raw_value=-2.0 -> mapped_value={mapped2.get(0)} (clamped)")
        print(f"    raw_value=0.5 -> mapped_value={mapped3.get(0)} (no clamp)")
        
        assert mapped1[0] == 1.0
        assert mapped2[0] == -1.0
        assert mapped3[0] == 0.5
    
    def test_map_controls_with_scale_and_clamping(self):
        """
        测试 map_controls() - 带缩放和限幅的组合
        
        验证：
        1. 先缩放，再限幅
        
        输出：
            [TEST] map_controls_with_scale_and_clamping:
            [TEST]   adding mapping: scale=2.0, range=[-1.0, 1.0]
            [TEST]   raw_value=0.8 -> mapped_value=1.0 (0.8*2.0=1.6, clamped to 1.0)
            [TEST]   raw_value=0.3 -> mapped_value=0.6 (0.3*2.0=0.6, no clamp)
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        
        print(f"\n  [TEST] test_map_controls_with_scale_and_clamping:")
        print(f"    adding mapping: scale=2.0, range=[-1.0, 1.0]")
        
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=0,
            plant_control_name="test",
            plant_control_index=0,
            scale=2.0,
            offset=0.0,
            range_min=-1.0,
            range_max=1.0
        )
        
        mapped1 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.8])
        mapped2 = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.3])
        
        print(f"    raw_value=0.8 -> mapped_value={mapped1.get(0)} (0.8*2.0=1.6, clamped to 1.0)")
        print(f"    raw_value=0.3 -> mapped_value={mapped2.get(0)} (0.3*2.0=0.6, no clamp)")
        
        assert mapped1[0] == 1.0
        assert mapped2[0] == 0.6
    
    def test_map_single_control(self, control_mapping):
        """
        测试 map_single_control() - 映射单个控制值
        
        验证：
        1. 单个值被正确映射
        2. 返回元组 (plant_control_index, value)
        
        输出：
            [TEST] map_single_control:
            [TEST]   mavlink_index=0, raw_value=0.5 -> (0, 0.5)
            [TEST]   mavlink_index=1, raw_value=1.0 -> (1, 1.0)
            [TEST]   mavlink_index=999, raw_value=0.0 -> None
        """
        from src import ControlSource
        
        cm = control_mapping
        
        print(f"\n  [TEST] test_map_single_control:")
        
        result1 = cm.map_single_control(ControlSource.HIL_ACTUATOR_CONTROLS, 0, 0.5)
        result2 = cm.map_single_control(ControlSource.HIL_ACTUATOR_CONTROLS, 1, 1.0)
        result3 = cm.map_single_control(ControlSource.HIL_ACTUATOR_CONTROLS, 999, 0.0)
        
        print(f"    mavlink_index=0, raw_value=0.5 -> {result1}")
        print(f"    mavlink_index=1, raw_value=1.0 -> {result2}")
        print(f"    mavlink_index=999, raw_value=0.0 -> {result3}")
        
        assert result1 == (0, 0.5)
        assert result2 == (1, 1.0)
        assert result3 is None


class TestControlMappingManipulation:
    """
    测试 ControlMapping 映射管理功能
    """
    
    def test_clear(self, control_mapping):
        """
        测试 clear() - 清空所有映射
        
        验证：
        1. 清空后映射数量为0
        
        输出：
            [TEST] clear:
            [TEST]   before clear: entries=4
            [TEST]   after clear: entries=0
        """
        cm = control_mapping
        
        print(f"\n  [TEST] test_clear:")
        print(f"    before clear: entries={len(cm.entries)}")
        
        cm.clear()
        
        print(f"    after clear: entries={len(cm.entries)}")
        
        assert len(cm.entries) == 0
    
    def test_enable_disable_all(self, control_mapping):
        """
        测试 enable_all() / disable_all() - 启用/禁用所有映射
        
        验证：
        1. disable_all 后 enabled_entries 数量为0
        2. enable_all 后 enabled_entries 数量恢复
        
        输出：
            [TEST] enable_disable_all:
            [TEST]   initial enabled_entries: 4
            [TEST]   after disable_all: enabled_entries=0
            [TEST]   after enable_all: enabled_entries=4
        """
        cm = control_mapping
        
        print(f"\n  [TEST] test_enable_disable_all:")
        print(f"    initial enabled_entries: {len(cm.enabled_entries)}")
        
        cm.disable_all()
        print(f"    after disable_all: enabled_entries={len(cm.enabled_entries)}")
        assert len(cm.enabled_entries) == 0
        
        cm.enable_all()
        print(f"    after enable_all: enabled_entries={len(cm.enabled_entries)}")
        assert len(cm.enabled_entries) == 4
    
    def test_entries_property(self, control_mapping):
        """
        测试 entries 属性
        
        验证：
        1. entries 返回所有映射条目的副本
        
        输出：
            [TEST] entries_property:
            [TEST]   entries count: 4
            [TEST]   entry 0: mavlink_index=0, plant_control_name=joint1
            [TEST]   entry 1: mavlink_index=1, plant_control_name=joint2
            [TEST]   entry 2: mavlink_index=2, plant_control_name=joint3
            [TEST]   entry 3: mavlink_index=3, plant_control_name=joint4
        """
        cm = control_mapping
        
        print(f"\n  [TEST] test_entries_property:")
        print(f"    entries count: {len(cm.entries)}")
        
        for i, entry in enumerate(cm.entries):
            print(f"    entry {i}: mavlink_index={entry.mavlink_index}, plant_control_name={entry.plant_control_name}")
        
        assert len(cm.entries) == 4


class TestControlMappingEdgeCases:
    """
    测试 ControlMapping 边界情况
    """
    
    def test_map_controls_empty(self):
        """
        测试空控制值列表
        
        验证：
        1. 空列表返回空字典
        
        输出：
            [TEST] map_controls_empty:
            [TEST]   raw_controls: []
            [TEST]   mapped: {}
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=0,
            plant_control_name="test",
            plant_control_index=0
        )
        
        print(f"\n  [TEST] test_map_controls_empty:")
        print(f"    raw_controls: []")
        
        mapped = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [])
        
        print(f"    mapped: {mapped}")
        
        assert mapped == {}
    
    def test_map_controls_no_mappings(self):
        """
        测试没有映射的情况
        
        验证：
        1. 没有映射时返回空字典
        
        输出：
            [TEST] map_controls_no_mappings:
            [TEST]   raw_controls: [1.0, 2.0, 3.0]
            [TEST]   mapped (no mappings): {}
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        
        raw_controls = [1.0, 2.0, 3.0]
        
        print(f"\n  [TEST] test_map_controls_no_mappings:")
        print(f"    raw_controls: {raw_controls}")
        
        mapped = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, raw_controls)
        
        print(f"    mapped (no mappings): {mapped}")
        
        assert mapped == {}
    
    def test_negative_values(self):
        """
        测试负值映射
        
        验证：
        1. 负值被正确映射和限幅
        
        输出：
            [TEST] negative_values:
            [TEST]   raw_value=-0.5 -> mapped_value=-0.5
            [TEST]   raw_value=-10.0 (range=[-5.0, 5.0]) -> mapped_value=-5.0
        """
        from src import ControlMapping, ControlSource
        
        cm = ControlMapping()
        
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=0,
            plant_control_name="test1",
            plant_control_index=0
        )
        
        cm.add_mapping(
            mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
            mavlink_index=1,
            plant_control_name="test2",
            plant_control_index=1,
            range_min=-5.0,
            range_max=5.0
        )
        
        print(f"\n  [TEST] test_negative_values:")
        
        mapped1 = cm.map_single_control(ControlSource.HIL_ACTUATOR_CONTROLS, 0, -0.5)
        mapped2 = cm.map_single_control(ControlSource.HIL_ACTUATOR_CONTROLS, 1, -10.0)
        
        print(f"    raw_value=-0.5 -> mapped_value={mapped1[1] if mapped1 else None}")
        print(f"    raw_value=-10.0 (range=[-5.0, 5.0]) -> mapped_value={mapped2[1] if mapped2 else None}")
        
        assert mapped1[1] == -0.5
        assert mapped2[1] == -5.0
