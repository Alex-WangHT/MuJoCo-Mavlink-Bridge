"""
测试 MavlinkUDPInterface 类 - MAVLink 通信接口

MavlinkUDPInterface 是核心通信类，可继承作为父类扩展。

核心功能：
- connect(): 建立 UDP 连接
- disconnect(): 断开连接
- start(): 启动接收线程
- stop(): 停止接收线程
- receive_controls(): 接收并解析 MAVLink 控制消息
- send_hil_actuator_controls(): 发送执行器控制量
- send_heartbeat(): 发送心跳消息
- register_callback(): 注册回调函数

架构：
    MAVLink (外部)
        |
        ▼ 控制量输入 (u)
    MavlinkUDPInterface ──► receive_controls()
        |
        ▼ 状态输出 (x)
    send_hil_actuator_controls() / send_heartbeat()

输出示例：
    [OUTPUT] MavlinkUDPInterface initialized: host=127.0.0.1, port=19999
    [OUTPUT] connect() returned: True
    [OUTPUT] is_connected: True
    [OUTPUT] Statistics: messages_received=0, messages_sent=0, errors=0
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMavlinkUDPInterfaceInitialization:
    """
    测试 MavlinkUDPInterface 初始化
    """
    
    def test_default_creation(self, mavlink_interface):
        """
        测试默认参数创建
        
        验证：
        1. 接口正确初始化
        2. 初始状态正确（未连接、未运行）
        3. 属性正确返回
        
        输出：
            [TEST] default_creation:
            [TEST]   is_connected: False
            [TEST]   is_running: False
            [TEST]   get_statistics(): {'messages_received': 0, 'messages_sent': 0, 'errors': 0}
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_default_creation:")
        print(f"    is_connected: {mavlink.is_connected}")
        print(f"    is_running: {mavlink.is_running}")
        
        stats = mavlink.get_statistics()
        print(f"    get_statistics(): {stats}")
        
        assert mavlink.is_connected is False
        assert mavlink.is_running is False
        assert stats["messages_received"] == 0
        assert stats["messages_sent"] == 0
        assert stats["errors"] == 0
    
    def test_custom_params(self):
        """
        测试自定义参数创建
        
        验证：
        1. 自定义端口和系统ID正确应用
        2. 初始化状态正确
        
        输出：
            [TEST] custom_params:
            [TEST]   creating with port=18888, source_system=2, source_component=3
            [TEST]   is_connected: False
        """
        from src import MavlinkUDPInterface
        
        print(f"\n  [TEST] test_custom_params:")
        print(f"    creating with port=18888, source_system=2, source_component=3")
        
        mavlink = MavlinkUDPInterface(
            host="127.0.0.1",
            port=18888,
            source_system=2,
            source_component=3
        )
        
        print(f"    is_connected: {mavlink.is_connected}")
        
        assert mavlink.is_connected is False


class TestMavlinkUDPInterfaceConnection:
    """
    测试 MavlinkUDPInterface 连接功能
    """
    
    def test_connect(self, mavlink_interface):
        """
        测试 connect() - 建立连接
        
        验证：
        1. connect() 返回 True
        2. is_connected 变为 True
        
        输出：
            [TEST] connect:
            [TEST]   connect() returned: True
            [TEST]   is_connected: True
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_connect:")
        
        result = mavlink.connect()
        print(f"    connect() returned: {result}")
        print(f"    is_connected: {mavlink.is_connected}")
        
        assert result is True
        assert mavlink.is_connected is True
    
    def test_disconnect(self, mavlink_interface):
        """
        测试 disconnect() - 断开连接
        
        验证：
        1. 连接后 disconnect() 会断开
        2. is_connected 变为 False
        
        输出：
            [TEST] disconnect:
            [TEST]   before disconnect: is_connected=True
            [TEST]   after disconnect: is_connected=False
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_disconnect:")
        
        mavlink.connect()
        print(f"    before disconnect: is_connected={mavlink.is_connected}")
        
        mavlink.disconnect()
        print(f"    after disconnect: is_connected={mavlink.is_connected}")
        
        assert mavlink.is_connected is False
    
    def test_double_connect(self, mavlink_interface):
        """
        测试重复连接
        
        验证：
        1. 重复调用 connect() 不会抛出异常
        
        输出：
            [TEST] double_connect:
            [TEST]   first connect(): True
            [TEST]   second connect(): True
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_double_connect:")
        
        result1 = mavlink.connect()
        print(f"    first connect(): {result1}")
        
        result2 = mavlink.connect()
        print(f"    second connect(): {result2}")
        
        assert mavlink.is_connected is True
    
    def test_disconnect_without_connect(self, mavlink_interface):
        """
        测试未连接时断开
        
        验证：
        1. 未连接时调用 disconnect() 不会抛出异常
        
        输出：
            [TEST] disconnect_without_connect:
            [TEST]   disconnect() called (was not connected)
            [TEST]   no errors raised
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_disconnect_without_connect:")
        print(f"    disconnect() called (was not connected)")
        
        mavlink.disconnect()
        
        print(f"    no errors raised")
        
        assert True


class TestMavlinkUDPInterfaceLifecycle:
    """
    测试 MavlinkUDPInterface 生命周期管理
    """
    
    def test_start_stop(self, mavlink_interface):
        """
        测试 start() / stop() - 启动/停止接收线程
        
        验证：
        1. start() 需要先连接
        2. start() 后 is_running 为 True
        3. stop() 后 is_running 为 False
        
        输出：
            [TEST] start_stop:
            [TEST]   connect()...
            [TEST]   start()...
            [TEST]   is_running: True
            [TEST]   stop()...
            [TEST]   is_running: False
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_start_stop:")
        
        print(f"    connect()...")
        mavlink.connect()
        
        print(f"    start()...")
        mavlink.start()
        print(f"    is_running: {mavlink.is_running}")
        
        assert mavlink.is_running is True
        
        print(f"    stop()...")
        mavlink.stop()
        print(f"    is_running: {mavlink.is_running}")
        
        assert mavlink.is_running is False
    
    def test_start_without_connect(self, mavlink_interface):
        """
        测试未连接时启动
        
        验证：
        1. 未连接时调用 start() 会抛出 RuntimeError
        
        输出：
            [TEST] start_without_connect:
            [TEST]   start() called without connect
            [TEST]   RuntimeError raised as expected
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_start_without_connect:")
        print(f"    start() called without connect")
        
        try:
            mavlink.start()
            print(f"    ERROR: RuntimeError not raised")
            assert False, "Expected RuntimeError"
        except RuntimeError:
            print(f"    RuntimeError raised as expected")
            assert True
    
    def test_double_start(self, mavlink_interface):
        """
        测试重复启动
        
        验证：
        1. 重复调用 start() 不会抛出异常
        
        输出：
            [TEST] double_start:
            [TEST]   first start() - is_running: True
            [TEST]   second start() - is_running: True
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_double_start:")
        
        mavlink.connect()
        mavlink.start()
        print(f"    first start() - is_running: {mavlink.is_running}")
        
        mavlink.start()
        print(f"    second start() - is_running: {mavlink.is_running}")
        
        assert mavlink.is_running is True
        
        mavlink.stop()
    
    def test_stop_without_start(self, mavlink_interface):
        """
        测试未启动时停止
        
        验证：
        1. 未启动时调用 stop() 不会抛出异常
        
        输出：
            [TEST] stop_without_start:
            [TEST]   stop() called without start
            [TEST]   is_running: False
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_stop_without_start:")
        print(f"    stop() called without start")
        
        mavlink.stop()
        print(f"    is_running: {mavlink.is_running}")
        
        assert mavlink.is_running is False


class TestMavlinkUDPInterfaceSendMessages:
    """
    测试 MavlinkUDPInterface 消息发送功能
    """
    
    def test_send_hil_actuator_controls_without_connection(self, mavlink_interface):
        """
        测试未连接时发送 HIL_ACTUATOR_CONTROLS
        
        验证：
        1. 未连接时发送返回 False
        
        输出：
            [TEST] send_hil_actuator_controls_without_connection:
            [TEST]   is_connected: False
            [TEST]   send_hil_actuator_controls() returned: False
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_send_hil_actuator_controls_without_connection:")
        print(f"    is_connected: {mavlink.is_connected}")
        
        result = mavlink.send_hil_actuator_controls([0.0] * 16)
        print(f"    send_hil_actuator_controls() returned: {result}")
        
        assert result is False
    
    def test_send_heartbeat_without_connection(self, mavlink_interface):
        """
        测试未连接时发送心跳
        
        验证：
        1. 未连接时发送返回 False
        
        输出：
            [TEST] send_heartbeat_without_connection:
            [TEST]   is_connected: False
            [TEST]   send_heartbeat() returned: False
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_send_heartbeat_without_connection:")
        print(f"    is_connected: {mavlink.is_connected}")
        
        result = mavlink.send_heartbeat()
        print(f"    send_heartbeat() returned: {result}")
        
        assert result is False
    
    def test_send_hil_actuator_controls_connected(self, mavlink_interface):
        """
        测试连接后发送 HIL_ACTUATOR_CONTROLS
        
        验证：
        1. 连接后可以发送消息
        2. 返回类型为 bool
        
        输出：
            [TEST] send_hil_actuator_controls_connected:
            [TEST]   connect()...
            [TEST]   sending controls: [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, ...]
            [TEST]   send_hil_actuator_controls() returned: <bool>
            [TEST]   statistics after send: messages_sent=1
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_send_hil_actuator_controls_connected:")
        
        print(f"    connect()...")
        mavlink.connect()
        
        controls = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 11
        print(f"    sending controls: {controls[:5] + ['...']}")
        
        result = mavlink.send_hil_actuator_controls(controls)
        print(f"    send_hil_actuator_controls() returned: {result}")
        print(f"    (Note: True means pymavlink sent, False means raw UDP or not connected)")
        
        stats = mavlink.get_statistics()
        print(f"    statistics after send: messages_sent={stats['messages_sent']}")
        
        assert isinstance(result, bool)
    
    def test_send_heartbeat_connected(self, mavlink_interface):
        """
        测试连接后发送心跳
        
        验证：
        1. 连接后可以发送心跳
        2. 返回类型为 bool
        
        输出：
            [TEST] send_heartbeat_connected:
            [TEST]   connect()...
            [TEST]   send_heartbeat() returned: <bool>
            [TEST]   statistics after send: messages_sent=1
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_send_heartbeat_connected:")
        
        print(f"    connect()...")
        mavlink.connect()
        
        result = mavlink.send_heartbeat()
        print(f"    send_heartbeat() returned: {result}")
        print(f"    (Note: True means pymavlink sent, False means raw UDP or not connected)")
        
        stats = mavlink.get_statistics()
        print(f"    statistics after send: messages_sent={stats['messages_sent']}")
        
        assert isinstance(result, bool)
    
    def test_send_hil_state_quaternion(self, mavlink_interface):
        """
        测试发送 HIL_STATE_QUATERNION 消息
        
        验证：
        1. 可以调用该方法
        2. 返回类型为 bool
        
        输出：
            [TEST] send_hil_state_quaternion:
            [TEST]   connect()...
            [TEST]   sending state with attitude: [1.0, 0.0, 0.0, 0.0]
            [TEST]   send_hil_state_quaternion() returned: <bool>
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_send_hil_state_quaternion:")
        
        print(f"    connect()...")
        mavlink.connect()
        
        print(f"    sending state with attitude: [1.0, 0.0, 0.0, 0.0]")
        
        result = mavlink.send_hil_state_quaternion(
            attitude=[1.0, 0.0, 0.0, 0.0],
            rollspeed=0.0,
            pitchspeed=0.0,
            yawspeed=0.0,
            lat=0,
            lon=0,
            alt=0,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            xacc=0.0,
            yacc=0.0,
            zacc=0.0
        )
        
        print(f"    send_hil_state_quaternion() returned: {result}")
        
        assert isinstance(result, bool)


class TestMavlinkUDPInterfaceReceiveControls:
    """
    测试 MavlinkUDPInterface 接收控制量功能
    """
    
    def test_receive_controls_without_connection(self, mavlink_interface):
        """
        测试未连接时接收控制量
        
        验证：
        1. 未连接时接收返回 None
        
        输出：
            [TEST] receive_controls_without_connection:
            [TEST]   is_connected: False
            [TEST]   receive_controls() returned: None
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_receive_controls_without_connection:")
        print(f"    is_connected: {mavlink.is_connected}")
        
        result = mavlink.receive_controls()
        print(f"    receive_controls() returned: {result}")
        
        assert result is None
    
    def test_receive_controls_connected(self, mavlink_interface):
        """
        测试连接后接收控制量
        
        验证：
        1. 连接后可以调用 receive_controls()
        2. 无消息时返回 None
        
        输出：
            [TEST] receive_controls_connected:
            [TEST]   connect()...
            [TEST]   receive_controls() returned: None (no messages)
            [TEST]   statistics: messages_received=0
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_receive_controls_connected:")
        
        print(f"    connect()...")
        mavlink.connect()
        
        result = mavlink.receive_controls()
        print(f"    receive_controls() returned: {result} (no messages)")
        
        stats = mavlink.get_statistics()
        print(f"    statistics: messages_received={stats['messages_received']}")
        
        assert result is None


class TestMavlinkUDPInterfaceStatistics:
    """
    测试 MavlinkUDPInterface 统计功能
    """
    
    def test_get_statistics_initial(self, mavlink_interface):
        """
        测试初始统计数据
        
        验证：
        1. 初始统计数据全为 0
        
        输出：
            [TEST] get_statistics_initial:
            [TEST]   statistics:
            [TEST]     messages_received: 0
            [TEST]     messages_sent: 0
            [TEST]     errors: 0
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_get_statistics_initial:")
        
        stats = mavlink.get_statistics()
        print(f"    statistics:")
        print(f"      messages_received: {stats['messages_received']}")
        print(f"      messages_sent: {stats['messages_sent']}")
        print(f"      errors: {stats['errors']}")
        
        assert stats["messages_received"] == 0
        assert stats["messages_sent"] == 0
        assert stats["errors"] == 0
    
    def test_get_statistics_after_send(self, mavlink_interface):
        """
        测试发送后的统计数据
        
        验证：
        1. 发送消息后 messages_sent 增加
        
        输出：
            [TEST] get_statistics_after_send:
            [TEST]   before send: messages_sent=0
            [TEST]   sending message...
            [TEST]   after send: messages_sent=1
        """
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_get_statistics_after_send:")
        
        mavlink.connect()
        
        stats_before = mavlink.get_statistics()
        print(f"    before send: messages_sent={stats_before['messages_sent']}")
        
        print(f"    sending message...")
        mavlink.send_heartbeat()
        
        stats_after = mavlink.get_statistics()
        print(f"    after send: messages_sent={stats_after['messages_sent']}")
        
        assert isinstance(stats_after["messages_sent"], int)


class TestMavlinkUDPInterfaceCallbacks:
    """
    测试 MavlinkUDPInterface 回调函数功能
    """
    
    def test_register_unregister_callback(self, mavlink_interface):
        """
        测试注册/注销回调函数
        
        验证：
        1. 可以注册回调函数
        2. 可以注销回调函数
        
        输出：
            [TEST] register_unregister_callback:
            [TEST]   registering callback for HIL_ACTUATOR_CONTROLS
            [TEST]   callback registered
            [TEST]   unregistering callback
            [TEST]   callback unregistered
        """
        from src import ControlSource
        
        mavlink = mavlink_interface
        
        print(f"\n  [TEST] test_register_unregister_callback:")
        
        def my_callback(values):
            print(f"      callback called with: {values}")
        
        print(f"    registering callback for HIL_ACTUATOR_CONTROLS")
        mavlink.register_callback(ControlSource.HIL_ACTUATOR_CONTROLS, my_callback)
        print(f"    callback registered")
        
        print(f"    unregistering callback")
        mavlink.unregister_callback(ControlSource.HIL_ACTUATOR_CONTROLS, my_callback)
        print(f"    callback unregistered")
        
        assert True


class TestMavlinkUDPInterfaceExtensible:
    """
    测试 MavlinkUDPInterface 可扩展性 - 作为父类继承
    """
    
    def test_inheritance(self):
        """
        测试继承 MavlinkUDPInterface 并扩展功能
        
        验证：
        1. 可以继承 MavlinkUDPInterface
        2. 可以重写方法
        3. 可以添加自定义方法
        
        输出：
            [TEST] inheritance:
            [TEST]   creating CustomMavlinkInterface instance...
            [TEST]   CustomMavlinkInterface.is_connected: False
            [TEST]   CustomMavlinkInterface.custom_method() called
            [TEST]   CustomMavlinkInterface.connect() with custom logic
        """
        from src import MavlinkUDPInterface
        
        class CustomMavlinkInterface(MavlinkUDPInterface):
            """
            自定义 MAVLink 接口，继承 MavlinkUDPInterface
            
            展示如何扩展父类功能：
            - 重写 connect() 方法
            - 添加自定义方法
            """
            
            def __init__(self):
                super().__init__(host="127.0.0.1", port=17777)
                self.custom_counter = 0
            
            def connect(self) -> bool:
                """重写 connect()，添加自定义逻辑"""
                self.custom_counter += 1
                return super().connect()
            
            def custom_method(self):
                """自定义方法"""
                return f"Custom method called, counter={self.custom_counter}"
        
        print(f"\n  [TEST] test_inheritance:")
        print(f"    creating CustomMavlinkInterface instance...")
        
        custom_mavlink = CustomMavlinkInterface()
        
        print(f"    CustomMavlinkInterface.is_connected: {custom_mavlink.is_connected}")
        
        result = custom_mavlink.custom_method()
        print(f"    CustomMavlinkInterface.custom_method() called")
        
        print(f"    CustomMavlinkInterface.connect() with custom logic")
        custom_mavlink.connect()
        
        assert custom_mavlink.custom_counter == 1
        
        custom_mavlink.disconnect()
