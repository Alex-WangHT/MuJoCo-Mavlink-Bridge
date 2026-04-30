import socket
import threading
import time
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
import struct

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    print("Warning: pymavlink not installed. Using simplified implementation.")


@dataclass
class MavlinkMessage:
    msg_type: str
    sysid: int
    compid: int
    seq: int
    data: Dict[str, Any]


class MavlinkBridge:
    def __init__(self, 
                 connection_string: str = "udp:0.0.0.0:14540",
                 source_system: int = 1,
                 source_component: int = 1,
                 target_system: int = 1,
                 target_component: int = 1):
        self.connection_string = connection_string
        self.source_system = source_system
        self.source_component = source_component
        self.target_system = target_system
        self.target_component = target_component
        
        self.mav = None
        self.is_connected = False
        self.is_running = False
        self.receive_thread = None
        self.lock = threading.Lock()
        
        self.message_handlers: Dict[str, Callable[[MavlinkMessage], None]] = {}
        self.last_messages: Dict[str, MavlinkMessage] = {}
        
        if PYMAVLINK_AVAILABLE:
            self._init_pymavlink()
        else:
            self._init_simple_udp()
    
    def _init_pymavlink(self):
        try:
            self.mav = mavutil.mavlink_connection(
                self.connection_string,
                source_system=self.source_system,
                source_component=self.source_component
            )
            self.is_connected = True
            print(f"Connected via pymavlink: {self.connection_string}")
        except Exception as e:
            print(f"Failed to initialize pymavlink: {e}")
            self._init_simple_udp()
    
    def _init_simple_udp(self):
        try:
            parts = self.connection_string.split(':')
            if len(parts) >= 3:
                host = parts[1].replace('//', '') if '//' in parts[1] else parts[1]
                port = int(parts[2])
                self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.udp_socket.settimeout(0.1)
                self.udp_socket.bind((host, port))
                self.is_connected = True
                print(f"Connected via simple UDP: {host}:{port}")
        except Exception as e:
            print(f"Failed to initialize simple UDP: {e}")
            self.is_connected = False
    
    def register_handler(self, msg_type: str, handler: Callable[[MavlinkMessage], None]):
        with self.lock:
            self.message_handlers[msg_type] = handler
    
    def unregister_handler(self, msg_type: str):
        with self.lock:
            if msg_type in self.message_handlers:
                del self.message_handlers[msg_type]
    
    def _convert_mavlink_message(self, msg) -> MavlinkMessage:
        if PYMAVLINK_AVAILABLE and hasattr(msg, 'get_type'):
            msg_type = msg.get_type()
            data = {}
            for field in msg._fieldnames:
                data[field] = getattr(msg, field)
            return MavlinkMessage(
                msg_type=msg_type,
                sysid=msg.get_srcSystem() if hasattr(msg, 'get_srcSystem') else 1,
                compid=msg.get_srcComponent() if hasattr(msg, 'get_srcComponent') else 1,
                seq=msg.get_seq() if hasattr(msg, 'get_seq') else 0,
                data=data
            )
        return MavlinkMessage(
            msg_type="UNKNOWN",
            sysid=1,
            compid=1,
            seq=0,
            data={}
        )
    
    def _process_message(self, msg: MavlinkMessage):
        with self.lock:
            self.last_messages[msg.msg_type] = msg
            if msg.msg_type in self.message_handlers:
                try:
                    self.message_handlers[msg.msg_type](msg)
                except Exception as e:
                    print(f"Error in message handler for {msg.msg_type}: {e}")
    
    def _receive_loop(self):
        while self.is_running:
            try:
                if PYMAVLINK_AVAILABLE and self.mav:
                    msg = self.mav.recv_match(blocking=False)
                    if msg:
                        mav_msg = self._convert_mavlink_message(msg)
                        self._process_message(mav_msg)
                else:
                    try:
                        data, addr = self.udp_socket.recvfrom(1024)
                        self._handle_raw_data(data, addr)
                    except socket.timeout:
                        pass
                    except Exception as e:
                        print(f"UDP receive error: {e}")
                
                time.sleep(0.001)
            except Exception as e:
                print(f"Receive loop error: {e}")
                time.sleep(0.1)
    
    def _handle_raw_data(self, data: bytes, addr: tuple):
        try:
            if len(data) >= 8:
                magic = data[0]
                if magic == 0xFD:
                    msg_type = "MAVLINK2"
                elif magic == 0xFE:
                    msg_type = "MAVLINK1"
                else:
                    msg_type = "UNKNOWN"
                
                mav_msg = MavlinkMessage(
                    msg_type=msg_type,
                    sysid=1,
                    compid=1,
                    seq=0,
                    data={"raw": data, "addr": addr}
                )
                self._process_message(mav_msg)
        except Exception as e:
            print(f"Raw data handling error: {e}")
    
    def start(self):
        if not self.is_connected:
            raise RuntimeError("Not connected to MAVLink")
        
        self.is_running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        print("MAVLink bridge started")
    
    def stop(self):
        self.is_running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
        if hasattr(self, 'udp_socket') and self.udp_socket:
            self.udp_socket.close()
        print("MAVLink bridge stopped")
    
    def send_hil_actuator_controls(self, 
                                     controls: list,
                                     mode: int = 0,
                                     flags: int = 0):
        if not self.is_connected:
            return False
        
        try:
            if PYMAVLINK_AVAILABLE and self.mav:
                while len(controls) < 16:
                    controls.append(0.0)
                controls = controls[:16]
                
                self.mav.mav.hil_actuator_controls_send(
                    int(time.time() * 1e6),
                    controls,
                    mode,
                    flags
                )
                return True
            else:
                print("send_hil_actuator_controls: pymavlink not available")
                return False
        except Exception as e:
            print(f"Send HIL_ACTUATOR_CONTROLS error: {e}")
            return False
    
    def send_hil_state_quaternion(self,
                                   attitude_quaternion: list,
                                   rollspeed: float,
                                   pitchspeed: float,
                                   yawspeed: float,
                                   lat: int,
                                   lon: int,
                                   alt: int,
                                   vx: float,
                                   vy: float,
                                   vz: float,
                                   xacc: float,
                                   yacc: float,
                                   zacc: float):
        if not self.is_connected:
            return False
        
        try:
            if PYMAVLINK_AVAILABLE and self.mav:
                while len(attitude_quaternion) < 4:
                    attitude_quaternion.append(0.0)
                attitude_quaternion = attitude_quaternion[:4]
                
                self.mav.mav.hil_state_quaternion_send(
                    int(time.time() * 1e6),
                    attitude_quaternion,
                    rollspeed,
                    pitchspeed,
                    yawspeed,
                    lat,
                    lon,
                    alt,
                    vx,
                    vy,
                    vz,
                    xacc,
                    yacc,
                    zacc
                )
                return True
            else:
                print("send_hil_state_quaternion: pymavlink not available")
                return False
        except Exception as e:
            print(f"Send HIL_STATE_QUATERNION error: {e}")
            return False
    
    def send_heartbeat(self,
                       type: int = 2,
                       autopilot: int = 12,
                       base_mode: int = 0,
                       custom_mode: int = 0,
                       system_status: int = 4):
        if not self.is_connected:
            return False
        
        try:
            if PYMAVLINK_AVAILABLE and self.mav:
                self.mav.mav.heartbeat_send(
                    type,
                    autopilot,
                    base_mode,
                    custom_mode,
                    system_status
                )
                return True
            else:
                print("send_heartbeat: pymavlink not available")
                return False
        except Exception as e:
            print(f"Send HEARTBEAT error: {e}")
            return False
    
    def wait_for_message(self, msg_type: str, timeout: float = 5.0) -> Optional[MavlinkMessage]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if msg_type in self.last_messages:
                    return self.last_messages[msg_type]
            time.sleep(0.01)
        return None
    
    def get_last_message(self, msg_type: str) -> Optional[MavlinkMessage]:
        with self.lock:
            return self.last_messages.get(msg_type)
