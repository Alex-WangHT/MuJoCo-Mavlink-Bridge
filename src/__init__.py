"""
MuJoCo-MAVLink Bridge

简洁的硬件在环仿真框架：
- MAVLink → Plant: 控制量输入 (u)
- Plant → MAVLink: 状态反馈 (x)

架构说明：
    ┌──────────┐         ┌──────────┐
    │  MAVLink │         │  Plant   │
    │ (外部)   │         │(MuJoCo)  │
    └────┬─────┘         └────┬─────┘
         │                    │
         │  控制量输入 (u)      │
         ├────────────────────►│
         │                    │
         │  状态反馈 (x)        │
         │◄────────────────────┤
         │                    │

扩展方式：
    - 继承 MuJoCoPlant: 自定义被控对象
    - 继承 MavlinkUDPInterface: 自定义通信接口
"""

from .core import (
    ControlVector,
    StateVector,
    ControlSource,
    StateTarget,
    ControlMappingEntry,
    ControlMapping,
    MuJoCoPlant,
    DEFAULT_ROBOT_XML,
    MavlinkUDPInterface,
)

from .simulator import (
    Simulator,
    SimulatorConfig,
)

__version__ = "1.0.0"

__all__ = [
    "ControlVector",
    "StateVector",
    "ControlSource",
    "StateTarget",
    "ControlMappingEntry",
    "ControlMapping",
    "MuJoCoPlant",
    "DEFAULT_ROBOT_XML",
    "MavlinkUDPInterface",
    "Simulator",
    "SimulatorConfig",
]
