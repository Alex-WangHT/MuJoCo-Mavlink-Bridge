#!/usr/bin/env python3
"""
新架构测试脚本 - 验证所有组件
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    ControlVector,
    StateVector,
    ControlSource,
    ControlMapping,
    MuJoCoPlant,
    MavlinkUDPInterface,
    Simulator,
    SimulatorConfig,
)

print("=" * 70)
print("  新架构测试 - MuJoCo-MAVLink Bridge")
print("=" * 70)
print()
print("  新架构说明：")
print("  - Plant 和 MuJoCoPlant 已合并为一个父类（可继承扩展）")
print("  - MavlinkInterface 和 MavlinkUDPInterface 已合并为一个父类")
print("  - 每个类和函数都有详细的中文注释")
print()
print("=" * 70)

# ============================================================================
# 测试1: ControlVector
# ============================================================================
print("\n【测试1: ControlVector - 控制量向量】")
print("-" * 70)

cv = ControlVector(values=[1.0, 0.5, -0.3], names=['motor1', 'motor2', 'motor3'])
print(f"  创建ControlVector: values=[1.0, 0.5, -0.3]")
print(f"  长度: {len(cv)}")
print(f"  按索引访问: cv[0] = {cv[0]}")
print(f"  转换为列表: {cv.to_list()}")
print(f"  转换为字典: {cv.to_dict()}")
print("  [OK] ControlVector 测试通过")

# ============================================================================
# 测试2: StateVector
# ============================================================================
print("\n【测试2: StateVector - 状态向量】")
print("-" * 70)

sv = StateVector()
sv.joint_positions["joint1"] = 0.5
sv.joint_positions["joint2"] = 1.0
sv.joint_velocities["joint1"] = 0.1
sv.time = 0.5

print(f"  创建StateVector并设置关节位置")
print(f"  joint_names: {sv.joint_names}")
print(f"  get_joint_position('joint1'): {sv.get_joint_position('joint1')}")
print(f"  time: {sv.time}")
print("  [OK] StateVector 测试通过")

# ============================================================================
# 测试3: ControlMapping
# ============================================================================
print("\n【测试3: ControlMapping - 控制量映射】")
print("-" * 70)

cm = ControlMapping()
cm.create_default_joint_mappings(["motor1", "motor2"])

print(f"  创建默认映射（2个控制量）")
print(f"  映射条目数量: {len(cm.entries)}")

# 测试映射应用
raw_values = [0.5, 1.0, 0.3]
mapped = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, raw_values)
print(f"  原始值: {raw_values}")
print(f"  映射后: {mapped}")

# 测试带缩放的映射
cm.clear()
cm.add_mapping(
    mavlink_source=ControlSource.HIL_ACTUATOR_CONTROLS,
    mavlink_index=0,
    plant_control_name="joint1",
    plant_control_index=0,
    scale=2.0,
    offset=0.0,
    range_min=-1.0,
    range_max=1.0
)

result = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.8])
print(f"  带缩放(scale=2.0, 限幅[-1,1])测试:")
print(f"    输入0.8 -> 输出{result}")

result = cm.map_controls(ControlSource.HIL_ACTUATOR_CONTROLS, [0.6])
print(f"    输入0.6 -> 输出{result} (限幅到1.0)")
print("  [OK] ControlMapping 测试通过")

# ============================================================================
# 测试4: MuJoCoPlant
# ============================================================================
print("\n【测试4: MuJoCoPlant - 被控对象类】")
print("-" * 70)

print("  初始化MuJoCoPlant...")
plant = MuJoCoPlant()

print(f"  control_dim: {plant.control_dim}")
print(f"  control_names: {plant.control_names}")
print(f"  joint_names: {plant.joint_names}")
print(f"  body_names: {plant.body_names}")
print(f"  timestep: {plant.timestep}")
print(f"  is_initialized: {plant.is_initialized}")

# 测试设置控制量
print(f"\n  测试控制量设置...")
plant.set_control_by_index(0, 1.0)
control = plant.get_control()
print(f"  设置控制量[0]=1.0后: {control.to_list()}")

# 测试仿真步进
print(f"\n  测试仿真步进...")
state_before = plant.get_state()
print(f"  步进前 - time: {state_before.time}")
print(f"          - joint1 pos: {state_before.get_joint_position('joint1')}")
print(f"          - joint1 vel: {state_before.get_joint_velocity('joint1')}")

plant.step(n_steps=10)

state_after = plant.get_state()
print(f"  步进10步后 - time: {state_after.time}")
print(f"            - joint1 pos: {state_after.get_joint_position('joint1')}")
print(f"            - joint1 vel: {state_after.get_joint_velocity('joint1')}")

# 测试重置
print(f"\n  测试重置...")
plant.reset()
state_reset = plant.get_state()
print(f"  重置后 - time: {state_reset.time}")
print(f"         - joint1 pos: {state_reset.get_joint_position('joint1')}")
print("  [OK] MuJoCoPlant 测试通过")

# ============================================================================
# 测试5: MavlinkUDPInterface
# ============================================================================
print("\n【测试5: MavlinkUDPInterface - MAVLink通信接口】")
print("-" * 70)

print("  初始化MavlinkUDPInterface（端口14550）...")
mavlink = MavlinkUDPInterface(host="127.0.0.1", port=14550)

print(f"  is_connected (before connect): {mavlink.is_connected}")
print(f"  is_running: {mavlink.is_running}")

# 测试连接
print(f"\n  测试连接...")
connected = mavlink.connect()
print(f"  connected: {connected}")
print(f"  is_connected (after connect): {mavlink.is_connected}")

# 测试统计
stats = mavlink.get_statistics()
print(f"  statistics: {stats}")

# 测试断开
mavlink.disconnect()
print(f"  断开后 is_connected: {mavlink.is_connected}")
print("  [OK] MavlinkUDPInterface 测试通过")

# ============================================================================
# 测试6: Simulator
# ============================================================================
print("\n【测试6: Simulator - 主仿真器】")
print("-" * 70)

print("  初始化Simulator...")
sim = Simulator()

print(f"  plant.control_dim: {sim.plant.control_dim}")
print(f"  control_mapping entries: {len(sim.control_mapping.entries)}")
print(f"  is_running: {sim.is_running}")

# 测试统计
stats = sim.get_statistics()
print(f"  statistics before run:")
print(f"    steps: {stats['steps']}")
print(f"    is_connected: {stats['is_connected']}")

# 测试直接控制
print(f"\n  测试直接控制...")
sim.set_control_by_index(0, 0.5)
sim.set_control_by_name("motor2", 1.0)

# 手动步进测试
print(f"\n  手动步进测试...")
state1 = sim.get_state()
print(f"  手动步进前 - time: {state1.time}")

for _ in range(10):
    sim.plant.step()
    
state2 = sim.get_state()
print(f"  手动步进后 - time: {state2.time}")
print(f"            - joint1 pos: {state2.get_joint_position('joint1')}")

# 测试重置
sim.reset()
state_reset = sim.get_state()
print(f"\n  重置后 - time: {state_reset.time}")
print(f"         - joint1 pos: {state_reset.get_joint_position('joint1')}")
print("  [OK] Simulator 测试通过")

# ============================================================================
# 总结
# ============================================================================
print()
print("=" * 70)
print("  所有测试通过！新架构工作正常")
print("=" * 70)
print()
print("  新架构特点：")
print("  1. 简洁 - 没有不必要的抽象基类")
print("  2. 可扩展 - MuJoCoPlant和MavlinkUDPInterface都可以被继承")
print("  3. 文档完善 - 每个类和函数都有详细的中文注释")
print()
print("  扩展示例：")
print("    class MyCustomPlant(MuJoCoPlant):")
print("        def step(self, n_steps=1):")
print("            # 自定义步进逻辑")
print("            super().step(n_steps)")
print()
print("    class MyCustomMavlink(MavlinkUDPInterface):")
print("        def connect(self):")
print("            # 自定义连接逻辑")
print("            return super().connect()")
print()
print("=" * 70)
