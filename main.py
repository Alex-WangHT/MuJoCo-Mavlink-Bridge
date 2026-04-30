#!/usr/bin/env python3
"""
MuJoCo-MAVLink Bridge - Main Entry Point

This script runs the simulator that connects MAVLink messages
to MuJoCo physics simulation.
"""

import time
import argparse
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src import (
    Simulator,
    SimulatorConfig,
    MavlinkConfig,
    SimulationConfig,
    ControlTargetType,
    ControlMode,
)


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="MuJoCo-MAVLink Bridge Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py
  
  # Custom connection settings
  python main.py --host 0.0.0.0 --port 14550
  
  # Custom real-time factor for faster simulation
  python main.py --rtf 10.0
  
  # Load custom model
  python main.py --model path/to/robot.xml
  
  # Control mode: position, velocity, torque
  python main.py --control-mode position
        """
    )
    
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0",
                        help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=14540,
                        help="Listen port (default: 14540)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to MuJoCo XML model file")
    parser.add_argument("--rtf", "--real-time-factor", type=float, default=1.0,
                        help="Real-time factor (default: 1.0)")
    parser.add_argument("--control-mode", type=str, default="torque",
                        choices=["position", "velocity", "torque"],
                        help="Default control mode (default: torque)")
    parser.add_argument("--source-system", type=int, default=1,
                        help="MAVLink source system ID (default: 1)")
    parser.add_argument("--source-component", type=int, default=1,
                        help="MAVLink source component ID (default: 1)")
    parser.add_argument("--no-auto-start", action="store_true",
                        help="Don't start simulation automatically")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    return parser


def control_mode_from_string(mode_str: str) -> ControlTargetType:
    mode_map = {
        "position": ControlTargetType.JOINT_POSITION,
        "velocity": ControlTargetType.JOINT_VELOCITY,
        "torque": ControlTargetType.JOINT_TORQUE,
    }
    return mode_map.get(mode_str, ControlTargetType.JOINT_TORQUE)


def create_config(args) -> SimulatorConfig:
    mavlink_config = MavlinkConfig(
        host=args.host,
        port=args.port,
        source_system=args.source_system,
        source_component=args.source_component,
    )
    
    simulation_config = SimulationConfig(
        real_time_factor=args.rtf,
    )
    
    return SimulatorConfig(
        mavlink=mavlink_config,
        simulation=simulation_config,
        model_path=args.model,
    )


def setup_callbacks(simulator: Simulator, verbose: bool = False):
    if verbose:
        def on_state_update(state):
            pass
        
        def on_message_received(msg):
            print(f"Received message: {msg.msg_name} (sysid={msg.sysid}, compid={msg.compid})")
        
        simulator.set_on_message_received(on_message_received)
    
    state_count = [0]
    
    def periodic_state_update(state):
        state_count[0] += 1
        if state_count[0] % 100 == 0:
            stats = simulator.get_statistics()
            joint_info = []
            for joint_name, pos in state.joint_positions.items():
                vel = state.joint_velocities.get(joint_name, 0.0)
                joint_info.append(f"{joint_name}: pos={pos:.3f}, vel={vel:.3f}")
            
            print(f"Step {stats['steps']}: {', '.join(joint_info[:2])}...")
    
    simulator.set_on_state_update(periodic_state_update)


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("MuJoCo-MAVLink Bridge Simulator")
    print("=" * 60)
    print(f"Listening on: {args.host}:{args.port}")
    print(f"Real-time factor: {args.rtf}")
    print(f"Control mode: {args.control_mode}")
    if args.model:
        print(f"Model: {args.model}")
    print("=" * 60)
    
    config = create_config(args)
    
    print("\nInitializing simulator...")
    simulator = Simulator(config=config)
    
    joint_names = simulator.model.joint_names
    print(f"Model has {len(joint_names)} joints: {joint_names}")
    
    control_type = control_mode_from_string(args.control_mode)
    print(f"\nSetting up control mappings (type: {control_type.value})...")
    
    simulator.clear_control_mappings()
    for i, joint_name in enumerate(joint_names):
        simulator.add_control_mapping(
            mavlink_index=i,
            target_name=joint_name,
            target_type=control_type,
            scale=1.0,
            offset=0.0,
        )
        print(f"  MAVLink index {i} -> {joint_name} ({control_type.value})")
    
    setup_callbacks(simulator, args.verbose)
    
    if not args.no_auto_start:
        print("\nConnecting MAVLink bridge...")
        simulator.connect()
        
        print("Starting simulator...")
        print("Press Ctrl+C to stop\n")
        
        simulator.start()
        
        try:
            while True:
                time.sleep(1.0)
                stats = simulator.get_statistics()
                print(f"[Stats] Steps: {stats['steps']}, "
                      f"Msgs Received: {stats['messages_received']}, "
                      f"Msgs Sent: {stats['messages_sent']}, "
                      f"FPS: {stats['steps_per_second']:.1f}")
        except KeyboardInterrupt:
            print("\n\nStopping simulator...")
            simulator.stop()
            print("Simulator stopped.")
    else:
        print("\nSimulation not started (--no-auto-start flag)")
        print("Use simulator.start() to begin")
    
    return simulator


if __name__ == "__main__":
    sim = main()
