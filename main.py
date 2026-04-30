#!/usr/bin/env python3
"""
MuJoCo-MAVLink Bridge - Main Entry Point

Simple architecture:
- MAVLink → Plant: Control inputs (u)
- Plant → MAVLink: State feedback (x)
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
    ControlSource,
    ControlMapping,
    StateVector,
)


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="MuJoCo-MAVLink Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture:
  MAVLink (control inputs u)  ──►  Plant (simulation)
  MAVLink (state feedback x)  ◄──  Plant (simulation)

Examples:
  # Run with default settings
  python main.py
  
  # Custom connection
  python main.py --host 0.0.0.0 --port 14540
  
  # Faster simulation
  python main.py --rtf 10.0
  
  # Load custom model
  python main.py --model path/to/robot.xml
        """
    )
    
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0",
                        help="MAVLink listen host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=14540,
                        help="MAVLink listen port (default: 14540)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to MuJoCo model XML file")
    parser.add_argument("--rtf", "--real-time-factor", type=float, default=1.0,
                        help="Real-time factor (default: 1.0)")
    parser.add_argument("--no-auto-start", action="store_true",
                        help="Don't start automatically")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser


def setup_callbacks(simulator: Simulator, verbose: bool = False):
    state_count = [0]
    
    def on_state_update(state: StateVector):
        state_count[0] += 1
        
        if state_count[0] % 100 == 0:
            stats = simulator.get_statistics()
            joint_info = []
            for name, pos in state.joint_positions.items():
                vel = state.joint_velocities.get(name, 0.0)
                joint_info.append(f"{name}: pos={pos:.3f}, vel={vel:.3f}")
            
            print(f"Step {stats['steps']}: {', '.join(joint_info[:2])}")
    
    simulator.set_on_state_update(on_state_update)
    
    if verbose:
        def on_control_received(controls):
            for source, values in controls.items():
                print(f"Received {source.value}: {values[:4]}")
        
        simulator.set_on_control_received(on_control_received)


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("MuJoCo-MAVLink Bridge")
    print("=" * 60)
    print(f"MAVLink: {args.host}:{args.port}")
    print(f"Real-time factor: {args.rtf}")
    if args.model:
        print(f"Model: {args.model}")
    print("=" * 60)
    print("\nArchitecture:")
    print("  MAVLink ──► Plant  (control inputs)")
    print("  MAVLink ◄── Plant  (state feedback)")
    print("=" * 60)
    
    config = SimulatorConfig(
        mavlink_host=args.host,
        mavlink_port=args.port,
        real_time_factor=args.rtf,
        model_path=args.model,
    )
    
    print("\nInitializing simulator...")
    simulator = Simulator(config=config)
    
    control_names = simulator.plant.control_names
    joint_names = simulator.plant.joint_names
    
    print(f"Plant initialized:")
    print(f"  Control inputs: {control_names}")
    print(f"  Joints: {joint_names}")
    
    print("\nControl mappings (MAVLink index → Plant control):")
    for entry in simulator.control_mapping.entries:
        print(f"  {entry.mavlink_source.value}[{entry.mavlink_index}] → {entry.plant_control_name}")
    
    setup_callbacks(simulator, args.verbose)
    
    if not args.no_auto_start:
        print("\nConnecting MAVLink interface...")
        simulator.connect()
        
        print("Starting simulator...")
        print("Press Ctrl+C to stop\n")
        
        simulator.start()
        
        try:
            while True:
                time.sleep(1.0)
                stats = simulator.get_statistics()
                print(f"[Stats] Steps: {stats['steps']}, "
                      f"Controls received: {stats['controls_received']}, "
                      f"States sent: {stats['states_sent']}, "
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
