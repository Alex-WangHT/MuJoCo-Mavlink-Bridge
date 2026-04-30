import pytest
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestSimulationPerformance:
    def test_simulation_steps_per_second(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        steps = 1000
        start_time = time.perf_counter()
        
        controller.step(n_steps=steps)
        
        elapsed = time.perf_counter() - start_time
        steps_per_second = steps / elapsed
        
        print(f"\nSimulation Performance:")
        print(f"  Steps: {steps}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Steps/sec: {steps_per_second:.2f}")
        
        assert elapsed < 10.0
        assert steps_per_second > 100

    def test_simulation_with_controls(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        steps = 500
        start_time = time.perf_counter()
        
        for i in range(steps):
            controller.set_joint_torque(joint_name, float(i % 10) / 10.0)
            controller.step(n_steps=1)
        
        elapsed = time.perf_counter() - start_time
        steps_per_second = steps / elapsed
        
        print(f"\nSimulation with Controls:")
        print(f"  Steps: {steps}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Steps/sec: {steps_per_second:.2f}")
        
        assert steps_per_second > 50

    def test_state_retrieval_performance(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        
        controller.step(n_steps=100)
        
        iterations = 1000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            state = controller.get_robot_state()
            _ = state.time
            _ = state.joint_states
        
        elapsed = time.perf_counter() - start_time
        retrievals_per_second = iterations / elapsed
        
        print(f"\nState Retrieval Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Retrievals/sec: {retrievals_per_second:.2f}")
        
        assert retrievals_per_second > 1000

    def test_joint_state_retrieval(self):
        from src import MuJoCoController
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        iterations = 10000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            state = controller.get_joint_state(joint_name)
            _ = state.qpos
            _ = state.qvel
        
        elapsed = time.perf_counter() - start_time
        retrievals_per_second = iterations / elapsed
        
        print(f"\nJoint State Retrieval:")
        print(f"  Iterations: {iterations}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Retrievals/sec: {retrievals_per_second:.2f}")
        
        assert retrievals_per_second > 5000


class TestMAVLinkPerformance:
    def test_message_handling_performance(self):
        from src import MavlinkBridge, MavlinkMessage
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:17777")
        
        messages = []
        for i in range(1000):
            msg = MavlinkMessage(
                msg_type="HIL_ACTUATOR_CONTROLS",
                sysid=1,
                compid=1,
                seq=i,
                data={
                    "controls": [float(i) / 1000.0] * 16,
                    "mode": 0,
                    "flags": 0
                }
            )
            messages.append(msg)
        
        received_count = 0
        
        def handler(msg):
            nonlocal received_count
            received_count += 1
        
        bridge.register_handler("HIL_ACTUATOR_CONTROLS", handler)
        
        start_time = time.perf_counter()
        
        for msg in messages:
            with bridge.lock:
                bridge._process_message(msg)
        
        elapsed = time.perf_counter() - start_time
        messages_per_second = len(messages) / elapsed
        
        print(f"\nMessage Handling Performance:")
        print(f"  Messages: {len(messages)}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Messages/sec: {messages_per_second:.2f}")
        
        assert received_count == len(messages)
        assert messages_per_second > 1000

    def test_message_send_performance(self):
        from src import MavlinkBridge
        
        bridge = MavlinkBridge(connection_string="udp:127.0.0.1:17776")
        
        iterations = 100
        controls = [0.0] * 16
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            controls[0] = float(i) / 100.0
            _ = bridge.send_hil_actuator_controls(controls)
        
        elapsed = time.perf_counter() - start_time
        sends_per_second = iterations / elapsed
        
        print(f"\nMessage Send Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Sends/sec: {sends_per_second:.2f}")
        
        assert sends_per_second > 100


class TestPIDPerformance:
    def test_pid_compute_performance(self):
        from src import PIDController
        
        pid = PIDController(kp=10.0, ki=1.0, kd=0.1)
        
        iterations = 100000
        start_time = time.perf_counter()
        
        for i in range(iterations):
            target = 1.0
            current = float(i) / iterations
            _ = pid.compute(target, current)
        
        elapsed = time.perf_counter() - start_time
        computes_per_second = iterations / elapsed
        
        print(f"\nPID Compute Performance:")
        print(f"  Iterations: {iterations}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Computes/sec: {computes_per_second:.2f}")
        
        assert computes_per_second > 100000

    def test_multiple_pid_controllers(self):
        from src import PIDController
        
        num_controllers = 16
        pids = [PIDController(kp=10.0, ki=1.0, kd=0.1) for _ in range(num_controllers)]
        
        iterations = 10000
        start_time = time.perf_counter()
        
        for i in range(iterations):
            for j, pid in enumerate(pids):
                target = float(j)
                current = float(i) / iterations * 10.0
                _ = pid.compute(target, current)
        
        elapsed = time.perf_counter() - start_time
        total_computes = iterations * num_controllers
        computes_per_second = total_computes / elapsed
        
        print(f"\nMultiple PID Controllers:")
        print(f"  Controllers: {num_controllers}")
        print(f"  Iterations: {iterations}")
        print(f"  Total Computes: {total_computes}")
        print(f"  Elapsed: {elapsed:.4f}s")
        print(f"  Computes/sec: {computes_per_second:.2f}")
        
        assert computes_per_second > 50000


class TestSimulatorPerformance:
    def test_simulator_running_performance(self):
        from src import Simulator
        import time
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:17775",
            real_time_factor=1000.0
        )
        
        sim.start()
        time.sleep(0.5)
        
        stats = sim.get_statistics()
        
        print(f"\nSimulator Running Performance:")
        print(f"  Steps: {stats['steps']}")
        print(f"  Elapsed: {stats['elapsed_time']:.4f}s")
        print(f"  Steps/sec: {stats['steps_per_second']:.2f}")
        print(f"  Real-time Factor: {sim.real_time_factor}")
        
        sim.stop()
        
        assert stats["steps"] > 0
        assert stats["steps_per_second"] > 100

    def test_simulator_state_callback_performance(self):
        from src import Simulator
        import time
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:17774",
            real_time_factor=1000.0
        )
        
        callback_count = 0
        
        def callback(state):
            nonlocal callback_count
            callback_count += 1
        
        sim.on_state_update = callback
        
        sim.start()
        time.sleep(0.3)
        
        stats = sim.get_statistics()
        
        print(f"\nSimulator Callback Performance:")
        print(f"  Steps: {stats['steps']}")
        print(f"  Callbacks: {callback_count}")
        print(f"  Elapsed: {stats['elapsed_time']:.4f}s")
        
        sim.stop()
        
        assert callback_count > 0


class TestMemoryPerformance:
    def test_memory_stability_during_simulation(self):
        from src import MuJoCoController
        import gc
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        for i in range(1000):
            controller.set_joint_torque(joint_name, float(i % 10) / 10.0)
            controller.step(n_steps=10)
            state = controller.get_robot_state()
            _ = state.joint_states
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        object_increase = final_objects - initial_objects
        
        print(f"\nMemory Stability:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Increase: {object_increase}")
        
        assert object_increase < 1000

    def test_repeated_reset_memory(self):
        from src import MuJoCoController
        import gc
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        for _ in range(100):
            controller.set_joint_torque(joint_name, 5.0)
            controller.step(n_steps=50)
            controller.reset()
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        object_increase = final_objects - initial_objects
        
        print(f"\nReset Memory Stability:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Increase: {object_increase}")
        
        assert object_increase < 500


class TestScalability:
    def test_many_joints_scalability(self):
        from src import MuJoCoController
        import numpy as np
        
        controller = MuJoCoController()
        num_joints = len(controller.joint_names)
        
        if num_joints < 2:
            pytest.skip("Need at least 2 joints for this test")
        
        print(f"\nJoint Scalability Test:")
        print(f"  Number of joints: {num_joints}")
        
        torques = {name: np.random.uniform(-1.0, 1.0) for name in controller.joint_names}
        
        start_time = time.perf_counter()
        controller.set_all_joint_torques(torques)
        elapsed_set = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        controller.step(n_steps=100)
        elapsed_step = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        states = controller.get_all_joint_states()
        elapsed_get = time.perf_counter() - start_time
        
        print(f"  Set all torques: {elapsed_set*1000:.2f}ms")
        print(f"  Step 100 steps: {elapsed_step*1000:.2f}ms")
        print(f"  Get all states: {elapsed_get*1000:.2f}ms")
        print(f"  States retrieved: {len(states)}")
        
        assert len(states) == num_joints

    def test_control_mapping_scalability(self):
        from src import Simulator
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:17773",
            real_time_factor=100.0
        )
        
        num_mappings = len(sim.control_mappings)
        
        print(f"\nControl Mapping Scalability:")
        print(f"  Number of mappings: {num_mappings}")
        
        for i, (idx, mapping) in enumerate(sim.control_mappings.items()):
            assert mapping.mavlink_index == idx
            assert mapping.joint_name in sim.mujoco_controller.joint_names
        
        print(f"  All mappings validated: {num_mappings}")
        
        assert num_mappings > 0


class TestLatency:
    def test_message_to_actuation_latency(self):
        from src import Simulator, MavlinkMessage
        import time
        
        sim = Simulator(
            mavlink_connection="udp:127.0.0.1:17772",
            real_time_factor=100.0
        )
        
        joint_name = sim.mujoco_controller.joint_names[0]
        
        measurements = []
        
        for i in range(100):
            torque = float(i % 20) / 10.0 - 1.0
            
            msg = MavlinkMessage(
                msg_type="HIL_ACTUATOR_CONTROLS",
                sysid=1,
                compid=1,
                seq=i,
                data={
                    "controls": [torque] + [0.0] * 15,
                    "mode": 0,
                    "flags": 0
                }
            )
            
            start_time = time.perf_counter()
            sim._handle_hil_actuator_controls(msg)
            elapsed = (time.perf_counter() - start_time) * 1e6
            
            measurements.append(elapsed)
            
            assert joint_name in sim.mujoco_controller.target_torques
        
        avg_latency = sum(measurements) / len(measurements)
        max_latency = max(measurements)
        min_latency = min(measurements)
        
        print(f"\nMessage to Actuation Latency:")
        print(f"  Samples: {len(measurements)}")
        print(f"  Average: {avg_latency:.2f}µs")
        print(f"  Min: {min_latency:.2f}µs")
        print(f"  Max: {max_latency:.2f}µs")
        
        assert avg_latency < 1000


class TestBenchmark:
    def test_full_benchmark(self):
        from src import MuJoCoController, PIDController, MavlinkMessage
        import time
        
        print("\n" + "="*60)
        print("Full System Benchmark")
        print("="*60)
        
        controller = MuJoCoController()
        joint_name = controller.joint_names[0]
        pid = PIDController(kp=10.0, ki=1.0, kd=0.1)
        
        print("\n1. Simulation Steps Benchmark")
        steps = 5000
        start = time.perf_counter()
        controller.step(n_steps=steps)
        elapsed = time.perf_counter() - start
        print(f"   {steps} steps in {elapsed:.3f}s -> {steps/elapsed:.1f} steps/sec")
        
        print("\n2. Control Application Benchmark")
        iterations = 1000
        start = time.perf_counter()
        for i in range(iterations):
            controller.set_joint_torque(joint_name, float(i % 10))
            controller.step(n_steps=1)
        elapsed = time.perf_counter() - start
        print(f"   {iterations} control+step in {elapsed:.3f}s -> {iterations/elapsed:.1f} cycles/sec")
        
        print("\n3. PID Compute Benchmark")
        iterations = 100000
        start = time.perf_counter()
        for i in range(iterations):
            _ = pid.compute(1.0, float(i) / iterations)
        elapsed = time.perf_counter() - start
        print(f"   {iterations} PID computes in {elapsed:.3f}s -> {iterations/elapsed:.0f} computes/sec")
        
        print("\n4. State Retrieval Benchmark")
        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            state = controller.get_robot_state()
            _ = state.joint_states
            _ = state.sensor_data
        elapsed = time.perf_counter() - start
        print(f"   {iterations} state retrievals in {elapsed:.3f}s -> {iterations/elapsed:.1f} retrievals/sec")
        
        print("\n" + "="*60)
        print("Benchmark Complete")
        print("="*60)
        
        assert True
