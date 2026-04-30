import time
import argparse
from src import Simulator, ControlMode, PYMAVLINK_AVAILABLE, MUJOCO_AVAILABLE


def main():
    parser = argparse.ArgumentParser(description="MuJoCo-MAVLink Bridge Simulator")
    parser.add_argument("--connection", "-c", type=str, default="udp:0.0.0.0:14540",
                        help="MAVLink connection string (default: udp:0.0.0.0:14540)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to MuJoCo XML model file")
    parser.add_argument("--rtf", "-r", type=float, default=1.0,
                        help="Real-time factor (default: 1.0)")
    parser.add_argument("--control-mode", type=str, default="torque",
                        choices=["position", "velocity", "torque", "pid"],
                        help="Default control mode (default: torque)")
    parser.add_argument("--no-run", action="store_true",
                        help="Don't start simulation automatically")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MuJoCo-MAVLink Bridge Simulator")
    print("=" * 60)
    print(f"pymavlink available: {PYMAVLINK_AVAILABLE}")
    print(f"mujoco available: {MUJOCO_AVAILABLE}")
    print(f"Connection: {args.connection}")
    print(f"Model: {args.model or 'default simple robot'}")
    print(f"Real-time factor: {args.rtf}")
    print("=" * 60)
    
    control_mode_map = {
        "position": ControlMode.POSITION,
        "velocity": ControlMode.VELOCITY,
        "torque": ControlMode.TORQUE,
        "pid": ControlMode.PID
    }
    default_control_mode = control_mode_map[args.control_mode]
    
    simulator = Simulator(
        mavlink_connection=args.connection,
        mujoco_model_path=args.model,
        real_time_factor=args.rtf
    )
    
    for mapping in simulator.control_mappings.values():
        mapping.control_type = default_control_mode
        simulator.mujoco_controller.set_control_mode(mapping.joint_name, default_control_mode)
    
    def on_state_update(state):
        stats = simulator.get_statistics()
        if stats["steps"] % 100 == 0:
            joint_info = []
            for joint_name, js in state.joint_states.items():
                joint_info.append(f"{joint_name}: pos={js.qpos:.3f}, vel={js.qvel:.3f}")
            print(f"Step {stats['steps']}: {', '.join(joint_info)}")
    
    def on_message_received(msg):
        print(f"Received message: {msg.msg_type}")
    
    simulator.on_state_update = on_state_update
    simulator.on_message_received = on_message_received
    
    if not args.no_run:
        print("\nStarting simulator...")
        simulator.start()
        
        try:
            while True:
                time.sleep(1.0)
                stats = simulator.get_statistics()
                print(f"\n[Stats] Steps: {stats['steps']}, "
                      f"Msgs Received: {stats['messages_received']}, "
                      f"Msgs Sent: {stats['messages_sent']}, "
                      f"FPS: {stats['steps_per_second']:.1f}")
        except KeyboardInterrupt:
            print("\nStopping simulator...")
            simulator.stop()
    else:
        print("\nSimulation not started (--no-run flag)")
        print("Use simulator.start() to begin")
    
    return simulator


if __name__ == "__main__":
    sim = main()
