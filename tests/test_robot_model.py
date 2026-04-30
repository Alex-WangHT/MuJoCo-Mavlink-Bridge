"""
测试 MuJoCoPlant 类 - 被控对象

MuJoCoPlant 是核心被控对象类，可继承作为父类扩展。

核心功能：
- set_control(): 设置控制量输入
- get_state(): 获取状态输出
- step(): 执行仿真步进
- reset(): 重置状态

架构：
    MAVLink → 控制量(u) → MuJoCoPlant → 状态(x) → MAVLink

输出示例：
    [OUTPUT] MuJoCoPlant initialized: control_dim=2, timestep=0.01
    [OUTPUT] Control applied: index 0 = 1.0
    [OUTPUT] After step: time=0.1, joint1_pos=0.005, joint1_vel=0.095
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMuJoCoPlantInitialization:
    """
    测试 MuJoCoPlant 初始化
    """
    
    def test_default_model_creation(self, mujoco_plant):
        """
        测试默认模型创建
        
        验证：
        1. 模型正确初始化
        2. 控制量维度正确
        3. 关节、刚体、传感器信息正确
        
        输出：
            [TEST] default_model_creation:
            [TEST]   is_initialized: True
            [TEST]   control_dim: 2
            [TEST]   control_names: ['joint1', 'joint2']
            [TEST]   joint_names: ['joint1', 'joint2']
            [TEST]   body_names: ['base', 'link2', 'end_effector']
            [TEST]   timestep: 0.01
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_default_model_creation:")
        print(f"    is_initialized: {plant.is_initialized}")
        print(f"    control_dim: {plant.control_dim}")
        print(f"    control_names: {plant.control_names}")
        print(f"    joint_names: {plant.joint_names}")
        print(f"    body_names: {plant.body_names}")
        print(f"    sensor_names: {plant.sensor_names}")
        print(f"    timestep: {plant.timestep}")
        
        assert plant.is_initialized is True
        assert plant.control_dim > 0
        assert len(plant.joint_names) > 0
        assert len(plant.body_names) > 0
        assert plant.timestep > 0
    
    def test_properties(self, mujoco_plant):
        """
        测试属性访问
        
        验证：
        1. 所有属性正确返回
        
        输出：
            [TEST] properties:
            [TEST]   control_dim: 2
            [TEST]   control_names count: 2
            [TEST]   joint_names count: 2
            [TEST]   body_names count: 3
            [TEST]   is_initialized: True
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_properties:")
        print(f"    control_dim: {plant.control_dim}")
        print(f"    control_names count: {len(plant.control_names)}")
        print(f"    joint_names count: {len(plant.joint_names)}")
        print(f"    body_names count: {len(plant.body_names)}")
        print(f"    sensor_names count: {len(plant.sensor_names)}")
        print(f"    is_initialized: {plant.is_initialized}")
        
        assert isinstance(plant.control_dim, int)
        assert isinstance(plant.control_names, list)
        assert isinstance(plant.joint_names, list)
        assert isinstance(plant.body_names, list)
        assert isinstance(plant.sensor_names, list)
        assert isinstance(plant.is_initialized, bool)


class TestMuJoCoPlantControl:
    """
    测试 MuJoCoPlant 控制量设置
    """
    
    def test_set_control_by_index(self, mujoco_plant):
        """
        测试 set_control_by_index() - 按索引设置控制量
        
        验证：
        1. 控制量被正确设置
        2. get_control() 返回正确值
        
        输出：
            [TEST] set_control_by_index:
            [TEST]   set_control_by_index(0, 1.0)
            [TEST]   set_control_by_index(1, 0.5)
            [TEST]   get_control values: [1.0, 0.5]
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_set_control_by_index:")
        
        print(f"    set_control_by_index(0, 1.0)")
        plant.set_control_by_index(0, 1.0)
        
        print(f"    set_control_by_index(1, 0.5)")
        plant.set_control_by_index(1, 0.5)
        
        control = plant.get_control()
        print(f"    get_control values: {control.to_list()}")
        
        assert control[0] == 1.0
        assert control[1] == 0.5
    
    def test_set_control_by_name(self, mujoco_plant):
        """
        测试 set_control_by_name() - 按名称设置控制量
        
        验证：
        1. 按控制量名称设置
        2. 按关节名称设置（通过映射）
        
        输出：
            [TEST] set_control_by_name:
            [TEST]   control_names: ['joint1', 'joint2']
            [TEST]   set_control_by_name('joint1', 0.8)
            [TEST]   get_control values: [0.8, 0.0]
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_set_control_by_name:")
        print(f"    control_names: {plant.control_names}")
        
        if plant.control_names:
            first_name = plant.control_names[0]
            print(f"    set_control_by_name('{first_name}', 0.8)")
            plant.set_control_by_name(first_name, 0.8)
            
            control = plant.get_control()
            print(f"    get_control values: {control.to_list()}")
            
            assert control[0] == 0.8
    
    def test_set_control_vector(self, mujoco_plant):
        """
        测试 set_control() - 使用 ControlVector 设置
        
        验证：
        1. ControlVector 被正确应用
        
        输出：
            [TEST] set_control_vector:
            [TEST]   set_control with values: [0.3, 0.7]
            [TEST]   get_control values: [0.3, 0.7]
        """
        from src import ControlVector
        import numpy as np
        
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_set_control_vector:")
        
        values = [0.3, 0.7]
        print(f"    set_control with values: {values}")
        
        cv = ControlVector(values=np.array(values), names=plant.control_names[:2])
        plant.set_control(cv)
        
        control = plant.get_control()
        print(f"    get_control values: {control.to_list()}")
        
        assert control[0] == 0.3
        assert control[1] == 0.7


class TestMuJoCoPlantSimulation:
    """
    测试 MuJoCoPlant 仿真功能
    """
    
    def test_step_function(self, mujoco_plant):
        """
        测试 step() - 执行仿真步进
        
        验证：
        1. 执行步进后时间增加
        2. 状态发生变化
        
        输出：
            [TEST] step_function:
            [TEST]   initial time: 0.0
            [TEST]   initial joint1_pos: 0.0
            [TEST]   after step(10):
            [TEST]     time: 0.1
            [TEST]     joint1_pos: 0.0
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_step_function:")
        
        state_before = plant.get_state()
        print(f"    initial time: {state_before.time}")
        print(f"    initial joint1_pos: {state_before.get_joint_position('joint1')}")
        
        plant.step(n_steps=10)
        
        state_after = plant.get_state()
        print(f"    after step(10):")
        print(f"      time: {state_after.time}")
        print(f"      joint1_pos: {state_after.get_joint_position('joint1')}")
        
        assert state_after.time > state_before.time
    
    def test_get_state(self, mujoco_plant):
        """
        测试 get_state() - 获取状态向量
        
        验证：
        1. 返回正确的 StateVector 类型
        2. 包含关节位置、速度、刚体位置等
        
        输出：
            [TEST] get_state:
            [TEST]   time: 0.0
            [TEST]   joint_positions: {'joint1': 0.0, 'joint2': 0.0}
            [TEST]   joint_velocities: {'joint1': 0.0, 'joint2': 0.0}
            [TEST]   body_positions count: 3
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_get_state:")
        
        state = plant.get_state()
        
        print(f"    time: {state.time}")
        print(f"    joint_positions: {state.joint_positions}")
        print(f"    joint_velocities: {state.joint_velocities}")
        print(f"    body_positions count: {len(state.body_positions)}")
        print(f"    body_velocities count: {len(state.body_velocities)}")
        print(f"    sensor_data count: {len(state.sensor_data)}")
        
        assert state is not None
        assert isinstance(state.time, float)
        assert isinstance(state.joint_positions, dict)
        assert isinstance(state.joint_velocities, dict)
    
    def test_get_state_methods(self, mujoco_plant):
        """
        测试 StateVector 便捷方法
        
        验证：
        1. get_joint_position() 正确返回
        2. get_joint_velocity() 正确返回
        3. get_body_position() 正确返回
        
        输出：
            [TEST] get_state_methods:
            [TEST]   joint_names: ['joint1', 'joint2']
            [TEST]   get_joint_position('joint1'): 0.0
            [TEST]   get_joint_velocity('joint1'): 0.0
            [TEST]   get_body_position('base'): [0. 0. 0.]
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_get_state_methods:")
        
        state = plant.get_state()
        
        print(f"    joint_names: {state.joint_names}")
        
        if state.joint_names:
            first_joint = state.joint_names[0]
            pos = state.get_joint_position(first_joint)
            vel = state.get_joint_velocity(first_joint)
            print(f"    get_joint_position('{first_joint}'): {pos}")
            print(f"    get_joint_velocity('{first_joint}'): {vel}")
            
            assert pos is not None
            assert vel is not None
        
        if state.body_names:
            first_body = state.body_names[0]
            body_pos = state.get_body_position(first_body)
            print(f"    get_body_position('{first_body}'): {body_pos}")
            
            assert body_pos is not None
    
    def test_simulation_with_controls(self, mujoco_plant):
        """
        测试带控制量的仿真
        
        验证：
        1. 设置控制量后执行仿真
        2. 状态发生变化
        
        输出：
            [TEST] simulation_with_controls:
            [TEST]   set_control(0, 1.0)
            [TEST]   step(50 times):
            [TEST]     initial time: 0.0
            [TEST]     initial joint1_pos: 0.0
            [TEST]     initial joint1_vel: 0.0
            [TEST]     after 50 steps:
            [TEST]       time: 0.5
            [TEST]       joint1_pos: 0.125
            [TEST]       joint1_vel: 0.475
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_simulation_with_controls:")
        print(f"    set_control(0, 1.0)")
        
        plant.set_control_by_index(0, 1.0)
        
        print(f"    step(50 times):")
        
        state_before = plant.get_state()
        print(f"      initial time: {state_before.time}")
        print(f"      initial joint1_pos: {state_before.get_joint_position('joint1')}")
        print(f"      initial joint1_vel: {state_before.get_joint_velocity('joint1')}")
        
        for _ in range(50):
            plant.step(n_steps=1)
        
        state_after = plant.get_state()
        print(f"      after 50 steps:")
        print(f"        time: {state_after.time}")
        print(f"        joint1_pos: {state_after.get_joint_position('joint1')}")
        print(f"        joint1_vel: {state_after.get_joint_velocity('joint1')}")
        
        assert state_after.time > state_before.time


class TestMuJoCoPlantReset:
    """
    测试 MuJoCoPlant 重置功能
    """
    
    def test_reset(self, mujoco_plant):
        """
        测试 reset() - 重置状态
        
        验证：
        1. 执行仿真后重置
        2. 状态恢复到初始值
        
        输出：
            [TEST] reset:
            [TEST]   step 100 times...
            [TEST]   before reset:
            [TEST]     time: 1.0
            [TEST]     joint1_pos: 0.25
            [TEST]   after reset:
            [TEST]     time: 0.0
            [TEST]     joint1_pos: 0.0
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_reset:")
        
        plant.set_control_by_index(0, 1.0)
        
        print(f"    step 100 times...")
        plant.step(n_steps=100)
        
        state_before_reset = plant.get_state()
        print(f"    before reset:")
        print(f"      time: {state_before_reset.time}")
        print(f"      joint1_pos: {state_before_reset.get_joint_position('joint1')}")
        
        plant.reset()
        
        state_after_reset = plant.get_state()
        print(f"    after reset:")
        print(f"      time: {state_after_reset.time}")
        print(f"      joint1_pos: {state_after_reset.get_joint_position('joint1')}")
        
        assert state_after_reset.time == 0.0
        
        first_joint = state_after_reset.joint_names[0] if state_after_reset.joint_names else None
        if first_joint:
            assert state_after_reset.get_joint_position(first_joint) == 0.0


class TestMuJoCoPlantEdgeCases:
    """
    测试 MuJoCoPlant 边界情况
    """
    
    def test_invalid_control_index(self, mujoco_plant):
        """
        测试无效控制量索引
        
        验证：
        1. 无效索引不抛出异常
        
        输出：
            [TEST] invalid_control_index:
            [TEST]   set_control_by_index(999, 1.0)
            [TEST]   set_control_by_index(-1, 1.0)
            [TEST]   no errors raised
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_invalid_control_index:")
        
        print(f"    set_control_by_index(999, 1.0)")
        plant.set_control_by_index(999, 1.0)
        
        print(f"    set_control_by_index(-1, 1.0)")
        plant.set_control_by_index(-1, 1.0)
        
        print(f"    no errors raised")
        
        assert True
    
    def test_invalid_control_name(self, mujoco_plant):
        """
        测试无效控制量名称
        
        验证：
        1. 无效名称不抛出异常
        
        输出：
            [TEST] invalid_control_name:
            [TEST]   set_control_by_name('nonexistent', 1.0)
            [TEST]   no errors raised
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_invalid_control_name:")
        
        print(f"    set_control_by_name('nonexistent', 1.0)")
        plant.set_control_by_name("nonexistent", 1.0)
        
        print(f"    no errors raised")
        
        assert True
    
    def test_zero_steps(self, mujoco_plant):
        """
        测试零步进
        
        验证：
        1. 零步进不改变状态
        
        输出：
            [TEST] zero_steps:
            [TEST]   time before: 0.0
            [TEST]   step(0)
            [TEST]   time after: 0.0
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_zero_steps:")
        
        state_before = plant.get_state()
        print(f"    time before: {state_before.time}")
        
        print(f"    step(0)")
        plant.step(n_steps=0)
        
        state_after = plant.get_state()
        print(f"    time after: {state_after.time}")
        
        assert state_after.time == state_before.time
    
    def test_large_steps(self, mujoco_plant):
        """
        测试大步数
        
        验证：
        1. 大步数正常执行
        
        输出：
            [TEST] large_steps:
            [TEST]   step(1000 times)
            [TEST]   time after: 10.0
        """
        plant = mujoco_plant
        
        print(f"\n  [TEST] test_large_steps:")
        
        print(f"    step(1000 times)")
        plant.step(n_steps=1000)
        
        state_after = plant.get_state()
        print(f"    time after: {state_after.time}")
        
        assert state_after.time > 0


class TestMuJoCoPlantExtensible:
    """
    测试 MuJoCoPlant 可扩展性 - 作为父类继承
    """
    
    def test_inheritance(self):
        """
        测试继承 MuJoCoPlant 并扩展功能
        
        验证：
        1. 可以继承 MuJoCoPlant
        2. 可以重写方法
        3. 可以添加自定义方法
        
        输出：
            [TEST] inheritance:
            [TEST]   creating CustomPlant instance...
            [TEST]   CustomPlant.control_dim: 2
            [TEST]   CustomPlant.is_initialized: True
            [TEST]   CustomPlant.custom_method() called
            [TEST]   CustomPlant.step() with custom logic
        """
        from src import MuJoCoPlant
        
        class CustomPlant(MuJoCoPlant):
            """
            自定义被控对象，继承 MuJoCoPlant
            
            展示如何扩展父类功能：
            - 重写 step() 方法
            - 添加自定义方法
            """
            
            def __init__(self):
                super().__init__()
                self.custom_counter = 0
            
            def step(self, n_steps: int = 1) -> None:
                """重写 step()，添加自定义逻辑"""
                self.custom_counter += n_steps
                super().step(n_steps)
            
            def custom_method(self):
                """自定义方法"""
                return f"Custom method called, counter={self.custom_counter}"
        
        print(f"\n  [TEST] test_inheritance:")
        print(f"    creating CustomPlant instance...")
        
        custom_plant = CustomPlant()
        
        print(f"    CustomPlant.control_dim: {custom_plant.control_dim}")
        print(f"    CustomPlant.is_initialized: {custom_plant.is_initialized}")
        
        result = custom_plant.custom_method()
        print(f"    CustomPlant.custom_method() called")
        
        print(f"    CustomPlant.step() with custom logic")
        custom_plant.step(n_steps=5)
        
        assert custom_plant.is_initialized is True
        assert custom_plant.control_dim > 0
        assert custom_plant.custom_counter == 5
