#!/usr/bin/env python3
"""
MAVLink Sender Example - Python implementation

This script sends MAVLink messages to control a MuJoCo simulation.
It supports various control modes and waveforms.
"""

import time
import math
import argparse
import sys
import os
from typing import List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src import MavlinkUDPBridge, MessageType, create_message_from_type
    PYMAVLINK_AVAILABLE = True
except ImportError:
    print("Warning: Could not import MAVLink bridge")
    PYMAVLINK_AVAILABLE = False


@dataclass
class SenderConfig:
    host: str = "127.0.0.1"
    port: int = 14540
    rate: int = 50
    mode: str = "sin"
    amplitude: float = 1.0
    frequency: float = 0.5
    num_joints: int = 2
    start_index: int = 0


class WaveformGenerator:
    @staticmethod
    def sine(t: float, amplitude: float, frequency: float, phase: float = 0.0) -> float:
        return amplitude * math.sin(2.0 * math.pi * frequency * t + phase)
    
    @staticmethod
    def cosine(t: float, amplitude: float, frequency: float, phase: float = 0.0) -> float:
        return amplitude * math.cos(2.0 * math.pi * frequency * t + phase)
    
    @staticmethod
    def square(t: float, amplitude: float, frequency: float, phase: float = 0.0) -> float:
        value = math.sin(2.0 * math.pi * frequency * t + phase)
        return amplitude if value >= 0 else -amplitude
    
    @staticmethod
    def triangle(t: float, amplitude: float, frequency: float, phase: float = 0.0) -> float:
        normalized = (t * frequency + phase / (2.0 * math.pi)) % 1.0
        if normalized < 0.25:
            return amplitude * (4.0 * normalized)
        elif normalized < 0.75:
            return amplitude * (2.0 - 4.0 * normalized)
        else:
            return amplitude * (4.0 * normalized - 4.0)
    
    @staticmethod
    def sawtooth(t: float, amplitude: float, frequency: float, phase: float = 0.0) -> float:
        normalized = (t * frequency + phase / (2.0 * math.pi)) % 1.0
        return amplitude * (2.0 * normalized - 1.0)
    
    @staticmethod
    def step(t: float, amplitude: float, *args, **kwargs) -> float:
        return amplitude
    
    @classmethod
    def get_waveform(cls, mode: str):
        waveforms = {
            "sin": cls.sine,
            "sine": cls.sine,
            "cos": cls.cosine,
            "cosine": cls.cosine,
            "square": cls.square,
            "triangle": cls.triangle,
            "sawtooth": cls.sawtooth,
            "step": cls.step,
        }
        return waveforms.get(mode.lower(), cls.sine)


class MavlinkSender:
    def __init__(self, config: SenderConfig):
        self.config = config
        self.bridge: Optional[MavlinkUDPBridge] = None
        self._running = False
        self._message_count = 0
    
    def connect(self) -> bool:
        if not PYMAVLINK_AVAILABLE:
            print("Error: MAVLink not available")
            return False
        
        try:
            print(f"Connecting to {self.config.host}:{self.config.port}...")
            self.bridge = MavlinkUDPBridge(
                host=self.config.host,
                port=self.config.port,
                source_system=1,
                source_component=1,
            )
            success = self.bridge.connect()
            if success:
                print(f"Connected to {self.config.host}:{self.config.port}")
            return success
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.bridge:
            self.bridge.disconnect()
            print("Disconnected")
    
    def send_controls(self, controls: List[float]) -> bool:
        if not self.bridge:
            return False
        
        try:
            success = self.bridge.send_hil_actuator_controls(controls)
            if success:
                self._message_count += 1
            return success
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        if not self.bridge:
            return False
        return self.bridge.send_heartbeat()
    
    def generate_controls(self, t: float) -> List[float]:
        controls = [0.0] * 16
        waveform = WaveformGenerator.get_waveform(self.config.mode)
        
        for i in range(self.config.num_joints):
            idx = self.config.start_index + i
            if idx < 16:
                phase = i * 0.5
                controls[idx] = waveform(
                    t,
                    self.config.amplitude,
                    self.config.frequency,
                    phase
                )
        
        return controls
    
    def run(self, duration: Optional[float] = None) -> None:
        if not self.connect():
            return
        
        self._running = True
        self._message_count = 0
        start_time = time.time()
        last_print_time = start_time
        interval = 1.0 / self.config.rate
        
        print(f"\nStarting sender with mode: {self.config.mode}")
        print(f"Rate: {self.config.rate} Hz, Amplitude: {self.config.amplitude}, Frequency: {self.config.frequency} Hz")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self._running:
                current_time = time.time()
                elapsed = current_time - start_time
                
                if duration and elapsed >= duration:
                    break
                
                t = elapsed
                controls = self.generate_controls(t)
                
                if not self.send_controls(controls):
                    print("Warning: Failed to send controls")
                
                if self._message_count % 100 == 0:
                    if current_time - last_print_time >= 1.0:
                        control_str = ", ".join([f"J{i}: {controls[self.config.start_index + i]:.3f}" 
                                                  for i in range(min(self.config.num_joints, 4))])
                        print(f"Time: {elapsed:.1f}s | {control_str} | Total: {self._message_count}")
                        last_print_time = current_time
                
                next_time = start_time + (self._message_count + 1) * interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._running = False
            total_time = time.time() - start_time
            print(f"\nStatistics:")
            print(f"  Messages sent: {self._message_count}")
            print(f"  Duration: {total_time:.2f}s")
            print(f"  Average rate: {self._message_count / total_time:.2f} Hz")
            self.disconnect()
    
    def stop(self) -> None:
        self._running = False


def run_interactive(config: SenderConfig):
    """Run in interactive mode, allowing manual control input"""
    sender = MavlinkSender(config)
    
    if not sender.connect():
        return
    
    print("\nInteractive Control Mode")
    print("Enter control values (comma-separated) or 'q' to quit")
    print("Example: 0.5, 0.3, -0.2\n")
    
    try:
        while True:
            user_input = input("Controls> ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            try:
                controls = [float(x.strip()) for x in user_input.split(',')]
                while len(controls) < 16:
                    controls.append(0.0)
                controls = controls[:16]
                
                if sender.send_controls(controls):
                    print(f"Sent: {controls[:config.num_joints]}")
                else:
                    print("Failed to send controls")
                    
            except ValueError as e:
                print(f"Invalid input: {e}")
                print("Please enter numbers separated by commas")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sender.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="MAVLink Control Sender - Python implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sine wave control at 0.5 Hz
  python mavlink_sender.py --mode sin --amplitude 1.0 --frequency 0.5
  
  # Step control with specific value
  python mavlink_sender.py --mode step --amplitude 2.5
  
  # Interactive mode
  python mavlink_sender.py --mode interactive
  
  # Control 4 joints starting at index 1
  python mavlink_sender.py --num-joints 4 --start-index 1
        """
    )
    
    parser.add_argument("--host", "-H", type=str, default="127.0.0.1",
                        help="Target IP address (default: 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=14540,
                        help="Target port (default: 14540)")
    parser.add_argument("--rate", "-r", type=int, default=50,
                        help="Send rate in Hz (default: 50)")
    parser.add_argument("--mode", "-m", type=str, default="sin",
                        choices=["sin", "sine", "cos", "cosine", "square", 
                                 "triangle", "sawtooth", "step", "interactive"],
                        help="Control mode (default: sin)")
    parser.add_argument("--amplitude", "-a", type=float, default=1.0,
                        help="Wave amplitude (default: 1.0)")
    parser.add_argument("--frequency", "-f", type=float, default=0.5,
                        help="Wave frequency in Hz (default: 0.5)")
    parser.add_argument("--num-joints", "-n", type=int, default=2,
                        help="Number of joints to control (default: 2)")
    parser.add_argument("--start-index", "-s", type=int, default=0,
                        help="Starting index in controls array (default: 0)")
    parser.add_argument("--duration", "-d", type=float, default=None,
                        help="Duration in seconds (default: infinite)")
    
    args = parser.parse_args()
    
    config = SenderConfig(
        host=args.host,
        port=args.port,
        rate=args.rate,
        mode=args.mode,
        amplitude=args.amplitude,
        frequency=args.frequency,
        num_joints=args.num_joints,
        start_index=args.start_index,
    )
    
    if args.mode == "interactive":
        run_interactive(config)
    else:
        sender = MavlinkSender(config)
        sender.run(duration=args.duration)


if __name__ == "__main__":
    main()
