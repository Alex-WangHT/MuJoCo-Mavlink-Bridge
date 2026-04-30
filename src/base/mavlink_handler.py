from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import struct


class MessageType(Enum):
    HIL_ACTUATOR_CONTROLS = 93
    HIL_STATE_QUATERNION = 115
    HEARTBEAT = 0
    HIL_GPS = 113
    HIL_SENSOR = 107
    RC_CHANNELS_OVERRIDE = 70
    MANUAL_CONTROL = 69
    SET_ATTITUDE_TARGET = 82
    SET_POSITION_TARGET_LOCAL_NED = 84
    CUSTOM = -1


@dataclass
class MavlinkMessage:
    msg_type: MessageType
    msg_name: str
    sysid: int
    compid: int
    seq: int
    data: Dict[str, Any] = field(default_factory=dict)
    raw_data: bytes = b""
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self.data[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.data


@dataclass
class ControlMapping:
    mavlink_index: int
    target_name: str
    target_type: str
    control_mode: MessageType
    scale: float = 1.0
    offset: float = 0.0
    transform: Optional[Callable[[float], float]] = None


class BaseMavlinkHandler(ABC):
    def __init__(self):
        self._message_handlers: Dict[MessageType, Callable[[MavlinkMessage], None]] = {}
        self._named_handlers: Dict[str, Callable[[MavlinkMessage], None]] = {}
        self._last_messages: Dict[MessageType, MavlinkMessage] = {}
        self._is_connected = False
        self._is_running = False
    
    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass
    
    @abstractmethod
    def send_message(self, msg: MavlinkMessage) -> bool:
        pass
    
    @abstractmethod
    def receive_message(self, timeout: float = 0.1) -> Optional[MavlinkMessage]:
        pass
    
    @abstractmethod
    def start(self) -> None:
        pass
    
    @abstractmethod
    def stop(self) -> None:
        pass
    
    def register_handler(self, 
                         msg_type: MessageType, 
                         handler: Callable[[MavlinkMessage], None]) -> None:
        self._message_handlers[msg_type] = handler
    
    def register_named_handler(self, 
                                msg_name: str, 
                                handler: Callable[[MavlinkMessage], None]) -> None:
        self._named_handlers[msg_name] = handler
    
    def unregister_handler(self, msg_type: MessageType) -> None:
        if msg_type in self._message_handlers:
            del self._message_handlers[msg_type]
    
    def unregister_named_handler(self, msg_name: str) -> None:
        if msg_name in self._named_handlers:
            del self._named_handlers[msg_name]
    
    def _process_message(self, msg: MavlinkMessage) -> None:
        self._last_messages[msg.msg_type] = msg
        
        if msg.msg_type in self._message_handlers:
            try:
                self._message_handlers[msg.msg_type](msg)
            except Exception as e:
                self._handle_error(f"Error in handler for {msg.msg_type}: {e}")
        
        if msg.msg_name in self._named_handlers:
            try:
                self._named_handlers[msg.msg_name](msg)
            except Exception as e:
                self._handle_error(f"Error in named handler for {msg.msg_name}: {e}")
    
    def get_last_message(self, msg_type: MessageType) -> Optional[MavlinkMessage]:
        return self._last_messages.get(msg_type)
    
    def wait_for_message(self, msg_type: MessageType, timeout: float = 5.0) -> Optional[MavlinkMessage]:
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = self.get_last_message(msg_type)
            if msg:
                return msg
            time.sleep(0.01)
        
        return None
    
    @abstractmethod
    def _handle_error(self, error_message: str) -> None:
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def registered_types(self) -> List[MessageType]:
        return list(self._message_handlers.keys())
    
    @property
    def registered_names(self) -> List[str]:
        return list(self._named_handlers.keys())
    
    @abstractmethod
    def create_hil_actuator_controls_message(self,
                                               controls: List[float],
                                               mode: int = 0,
                                               flags: int = 0) -> MavlinkMessage:
        pass
    
    @abstractmethod
    def create_heartbeat_message(self,
                                   type_val: int = 2,
                                   autopilot: int = 12,
                                   base_mode: int = 0,
                                   custom_mode: int = 0,
                                   system_status: int = 4) -> MavlinkMessage:
        pass


class BaseMavlinkUDPServer(BaseMavlinkHandler):
    def __init__(self, host: str = "0.0.0.0", port: int = 14540):
        super().__init__()
        self._host = host
        self._port = port
        self._socket = None
        self._target_addr: Optional[Tuple[str, int]] = None
        self._receive_thread = None
    
    def _handle_error(self, error_message: str) -> None:
        import logging
        logging.error(error_message)
    
    @abstractmethod
    def _parse_message(self, data: bytes, addr: Tuple[str, int]) -> Optional[MavlinkMessage]:
        pass
    
    @abstractmethod
    def _serialize_message(self, msg: MavlinkMessage) -> bytes:
        pass
