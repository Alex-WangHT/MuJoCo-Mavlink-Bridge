from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ConnectionType(Enum):
    UDP = "udp"
    TCP = "tcp"
    SERIAL = "serial"


@dataclass
class MavlinkConfig:
    connection_type: ConnectionType = ConnectionType.UDP
    host: str = "0.0.0.0"
    port: int = 14540
    source_system: int = 1
    source_component: int = 1
    target_system: int = 1
    target_component: int = 1
    heartbeat_interval: float = 1.0


@dataclass
class SimulationConfig:
    real_time_factor: float = 1.0
    timestep: Optional[float] = None
    max_steps: Optional[int] = None
    enable_sync: bool = True


@dataclass
class SimulatorConfig:
    mavlink: MavlinkConfig = field(default_factory=MavlinkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    model_path: Optional[str] = None
    model_xml: Optional[str] = None
    
    enable_telemetry: bool = True
    telemetry_interval: float = 0.1
    
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulatorConfig":
        mavlink_config = MavlinkConfig()
        if "mavlink" in config_dict:
            mav_dict = config_dict["mavlink"]
            if "connection_type" in mav_dict:
                mavlink_config.connection_type = ConnectionType(mav_dict["connection_type"])
            if "host" in mav_dict:
                mavlink_config.host = mav_dict["host"]
            if "port" in mav_dict:
                mavlink_config.port = mav_dict["port"]
            if "source_system" in mav_dict:
                mavlink_config.source_system = mav_dict["source_system"]
            if "source_component" in mav_dict:
                mavlink_config.source_component = mav_dict["source_component"]
        
        sim_config = SimulationConfig()
        if "simulation" in config_dict:
            sim_dict = config_dict["simulation"]
            if "real_time_factor" in sim_dict:
                sim_config.real_time_factor = sim_dict["real_time_factor"]
            if "timestep" in sim_dict:
                sim_config.timestep = sim_dict["timestep"]
        
        return cls(
            mavlink=mavlink_config,
            simulation=sim_config,
            model_path=config_dict.get("model_path"),
            model_xml=config_dict.get("model_xml"),
            enable_telemetry=config_dict.get("enable_telemetry", True),
            telemetry_interval=config_dict.get("telemetry_interval", 0.1),
            custom_config=config_dict.get("custom_config", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mavlink": {
                "connection_type": self.mavlink.connection_type.value,
                "host": self.mavlink.host,
                "port": self.mavlink.port,
                "source_system": self.mavlink.source_system,
                "source_component": self.mavlink.source_component,
            },
            "simulation": {
                "real_time_factor": self.simulation.real_time_factor,
                "timestep": self.simulation.timestep,
            },
            "model_path": self.model_path,
            "model_xml": self.model_xml,
            "enable_telemetry": self.enable_telemetry,
            "telemetry_interval": self.telemetry_interval,
            "custom_config": self.custom_config,
        }
