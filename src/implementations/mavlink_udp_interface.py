import socket
import threading
import time
import struct
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..core import MavlinkInterface, ControlSource, StateTarget, StateVector, ControlVector

logger = logging.getLogger(__name__)

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    logger.warning("pymavlink not installed. Using simplified implementation.")


class MavlinkUDPInterface(MavlinkInterface):
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 14540,
                 source_system: int = 1,
                 source_component: int = 1,
                 target_system: int = 1,
                 target_component: int = 1):
        self._host = host
        self._port = port
        self._source_system = source_system
        self._source_component = source_component
        self._target_system = target_system
        self._target_component = target_component
        
        self._mavutil_connection = None
        self._socket = None
        self._target_addr: Optional[Tuple[str, int]] = None
        
        self._is_connected = False
        self._is_running = False
        self._receive_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._sequence = 0
        self._stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
        }
        
        self._last_messages: Dict[ControlSource, List[float]] = {}
        self._callbacks: Dict[ControlSource, List[callable]] = {}
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    def connect(self) -> bool:
        if PYMAVLINK_AVAILABLE:
            return self._connect_pymavlink()
        return self._connect_raw_udp()
    
    def _connect_pymavlink(self) -> bool:
        try:
            conn_str = f"udp:{self._host}:{self._port}"
            logger.info(f"Connecting via pymavlink: {conn_str}")
            
            self._mavutil_connection = mavutil.mavlink_connection(
                conn_str,
                source_system=self._source_system,
                source_component=self._source_component
            )
            
            self._is_connected = True
            logger.info("Connected via pymavlink")
            return True
            
        except Exception as e:
            logger.warning(f"pymavlink connection failed: {e}, falling back to raw UDP")
            return self._connect_raw_udp()
    
    def _connect_raw_udp(self) -> bool:
        try:
            logger.info(f"Connecting via raw UDP: {self._host}:{self._port}")
            
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(0.1)
            
            if self._host == "0.0.0.0":
                self._socket.bind(("0.0.0.0", self._port))
                logger.info(f"UDP server bound to 0.0.0.0:{self._port}")
            else:
                self._target_addr = (self._host, self._port)
                logger.info(f"UDP client configured for {self._host}:{self._port}")
            
            self._is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Raw UDP connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        self._is_connected = False
        
        if self._mavutil_connection:
            self._mavutil_connection = None
        
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        
        logger.info("MAVLink interface disconnected")
    
    def start(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot start: not connected")
        
        if self._is_running:
            return
        
        self._is_running = True
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
        logger.info("MAVLink interface started")
    
    def stop(self) -> None:
        self._is_running = False
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None
        
        logger.info("MAVLink interface stopped")
    
    def _receive_loop(self) -> None:
        while self._is_running:
            try:
                controls = self.receive_controls()
                if controls:
                    for source, values in controls.items():
                        with self._lock:
                            self._last_messages[source] = values
                        
                        if source in self._callbacks:
                            for callback in self._callbacks[source]:
                                try:
                                    callback(values)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                
                time.sleep(0.001)
                
            except Exception as e:
                logger.debug(f"Receive loop error: {e}")
                time.sleep(0.01)
    
    def receive_controls(self) -> Optional[Dict[ControlSource, List[float]]]:
        if not self._is_connected:
            return None
        
        result = {}
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                msg = self._mavutil_connection.recv_match(blocking=False)
                if msg:
                    msg_type = msg.get_type()
                    
                    if msg_type == "HIL_ACTUATOR_CONTROLS":
                        controls = list(msg.controls)
                        result[ControlSource.HIL_ACTUATOR_CONTROLS] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                    
                    elif msg_type == "RC_CHANNELS_OVERRIDE":
                        controls = [
                            msg.chan1_raw / 1000.0 - 1.0,
                            msg.chan2_raw / 1000.0 - 1.0,
                            msg.chan3_raw / 1000.0 - 1.0,
                            msg.chan4_raw / 1000.0 - 1.0,
                        ]
                        result[ControlSource.RC_CHANNELS] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                    
                    elif msg_type == "MANUAL_CONTROL":
                        controls = [
                            msg.x / 1000.0,
                            msg.y / 1000.0,
                            msg.z / 1000.0,
                            msg.r / 1000.0,
                        ]
                        result[ControlSource.MANUAL_CONTROL] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                
            except Exception as e:
                logger.debug(f"pymavlink receive error: {e}")
        
        elif self._socket:
            try:
                ready = self._poll_socket(0.01)
                if ready:
                    data, addr = self._socket.recvfrom(2048)
                    
                    if self._target_addr is None:
                        self._target_addr = addr
                    
                    controls = self._parse_raw_mavlink(data)
                    if controls:
                        result[ControlSource.HIL_ACTUATOR_CONTROLS] = controls
                        with self._lock:
                            self._stats["messages_received"] += 1
                            
            except socket.timeout:
                pass
            except Exception as e:
                logger.debug(f"Raw UDP receive error: {e}")
        
        return result if result else None
    
    def _poll_socket(self, timeout: float) -> bool:
        import select
        
        if not self._socket:
            return False
        
        ready, _, _ = select.select([self._socket], [], [], timeout)
        return len(ready) > 0
    
    def _parse_raw_mavlink(self, data: bytes) -> Optional[List[float]]:
        if len(data) < 8:
            return None
        
        magic = data[0]
        
        if magic == 0xFD:
            pass
        elif magic == 0xFE:
            pass
        
        return None
    
    def send_state(self, state: StateVector, mapping: Optional[Any] = None) -> bool:
        controls = []
        
        for name in state.joint_positions.keys():
            pos = state.joint_positions.get(name, 0.0)
            controls.append(pos)
        
        while len(controls) < 16:
            controls.append(0.0)
        controls = controls[:16]
        
        return self.send_hil_actuator_controls(controls)
    
    def send_hil_actuator_controls(self, controls: List[float]) -> bool:
        if not self._is_connected:
            return False
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                while len(controls) < 16:
                    controls.append(0.0)
                controls = controls[:16]
                
                self._mavutil_connection.mav.hil_actuator_controls_send(
                    int(time.time() * 1e6),
                    controls,
                    0,
                    0
                )
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return True
                
            except Exception as e:
                logger.warning(f"pymavlink send error: {e}")
                return False
        
        elif self._socket and self._target_addr:
            try:
                msg = self._create_hil_actuator_controls_message(controls)
                sent = self._socket.sendto(msg, self._target_addr)
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return sent == len(msg)
                
            except Exception as e:
                logger.debug(f"Raw UDP send error: {e}")
                return False
        
        return False
    
    def _create_hil_actuator_controls_message(self, controls: List[float]) -> bytes:
        data = bytearray()
        
        data.append(0xFD)
        data.append(64)
        data.append(0)
        data.append(0)
        
        with self._lock:
            data.append(self._sequence & 0xFF)
            self._sequence = (self._sequence + 1) & 0xFF
        
        data.append(self._source_system)
        data.append(self._source_component)
        
        msg_id = 93
        data.append(msg_id & 0xFF)
        data.append((msg_id >> 8) & 0xFF)
        data.append((msg_id >> 16) & 0xFF)
        
        time_usec = int(time.time() * 1e6)
        data.extend(time_usec.to_bytes(8, byteorder='little'))
        
        for i in range(16):
            val = controls[i] if i < len(controls) else 0.0
            float_bytes = struct.pack('<f', val)
            data.extend(float_bytes)
        
        data.append(0)
        data.extend((0).to_bytes(8, byteorder='little'))
        
        checksum = self._calculate_checksum(bytes(data[1:]))
        data.extend(checksum.to_bytes(2, byteorder='little'))
        
        return bytes(data)
    
    def _calculate_checksum(self, data: bytes) -> int:
        checksum = 0xFFFF
        for byte in data:
            tmp = byte ^ (checksum & 0xFF)
            tmp ^= (tmp << 4) & 0xFF
            checksum = (checksum >> 8) ^ (tmp << 8) ^ ((tmp << 3) & 0xFFFF) ^ (tmp >> 4)
            checksum = checksum & 0xFFFF
        return checksum
    
    def send_hil_state_quaternion(self,
                                    attitude: List[float],
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
                                    zacc: float) -> bool:
        if not self._is_connected:
            return False
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                while len(attitude) < 4:
                    attitude.append(0.0)
                attitude = attitude[:4]
                
                self._mavutil_connection.mav.hil_state_quaternion_send(
                    int(time.time() * 1e6),
                    attitude,
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
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return True
                
            except Exception as e:
                logger.warning(f"pymavlink send error: {e}")
                return False
        
        return False
    
    def send_heartbeat(self) -> bool:
        if not self._is_connected:
            return False
        
        if PYMAVLINK_AVAILABLE and self._mavutil_connection:
            try:
                self._mavutil_connection.mav.heartbeat_send(
                    2,
                    12,
                    0,
                    0,
                    4
                )
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return True
                
            except Exception as e:
                logger.warning(f"pymavlink heartbeat error: {e}")
                return False
        
        elif self._socket and self._target_addr:
            try:
                msg = self._create_heartbeat_message()
                sent = self._socket.sendto(msg, self._target_addr)
                
                with self._lock:
                    self._stats["messages_sent"] += 1
                return sent == len(msg)
                
            except Exception as e:
                logger.debug(f"Raw UDP heartbeat error: {e}")
                return False
        
        return False
    
    def _create_heartbeat_message(self) -> bytes:
        data = bytearray()
        
        data.append(0xFD)
        data.append(9)
        data.append(0)
        data.append(0)
        
        with self._lock:
            data.append(self._sequence & 0xFF)
            self._sequence = (self._sequence + 1) & 0xFF
        
        data.append(self._source_system)
        data.append(self._source_component)
        
        msg_id = 0
        data.append(msg_id & 0xFF)
        data.append((msg_id >> 8) & 0xFF)
        data.append((msg_id >> 16) & 0xFF)
        
        data.extend((0).to_bytes(4, byteorder='little'))
        data.append(2)
        data.append(12)
        data.append(0)
        data.append(4)
        data.append(3)
        
        checksum = self._calculate_checksum(bytes(data[1:]))
        data.extend(checksum.to_bytes(2, byteorder='little'))
        
        return bytes(data)
    
    def get_last_controls(self, source: ControlSource) -> Optional[List[float]]:
        with self._lock:
            return self._last_messages.get(source)
    
    def get_statistics(self) -> Dict[str, int]:
        with self._lock:
            return self._stats.copy()
    
    def register_callback(self, source: ControlSource, callback: callable) -> None:
        if source not in self._callbacks:
            self._callbacks[source] = []
        self._callbacks[source].append(callback)
    
    def unregister_callback(self, source: ControlSource, callback: callable) -> None:
        if source in self._callbacks:
            if callback in self._callbacks[source]:
                self._callbacks[source].remove(callback)
