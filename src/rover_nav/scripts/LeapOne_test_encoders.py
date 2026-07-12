#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from odrive_can.srv import AxisState
from odrive_can.msg import ControlMessage, ControllerStatus
import time

class EncoderTest(Node):
    def __init__(self):
        super().__init__('encoder_test')
        
        self.num_axes = 6
        self.right_wheels = [0, 1, 2]
        self.left_wheels = [3, 4, 5]
        
        self.axis_clients = []
        self.axis_pubs = []
        self.positions = [0.0] * 6
        
        for i in range(self.num_axes):
            srv_name = f"/odrive_axis{i}/request_axis_state"
            topic_name = f"/odrive_axis{i}/control_message"
            status_topic = f"/odrive_axis{i}/controller_status"
            
            self.axis_clients.append(self.create_client(AxisState, srv_name))
            self.axis_pubs.append(self.create_publisher(ControlMessage, topic_name, 10))
            
            self.create_subscription(
                ControllerStatus,
                status_topic,
                lambda msg, idx=i: self.status_callback(msg, idx),
                10
            )
        
        self.get_logger().info("Encoder test node initialized")
    
    def status_callback(self, msg, axis_idx):
        self.positions[axis_idx] = msg.pos_estimate
    
    def wait_for_services(self):
        for i, client in enumerate(self.axis_clients):
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for service {client.srv_name}...")
    
    def set_closed_loop(self):
        for i, client in enumerate(self.axis_clients):
            req = AxisState.Request()
            req.axis_requested_state = 8
            self.get_logger().info(f"Setting odrive_axis{i} to CLOSED_LOOP_CONTROL...")
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            time.sleep(0.1)
    
    def get_current_positions(self):
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.01)
        return self.positions.copy()
    
    def send_velocity(self, right_vel, left_vel):
        for i in self.right_wheels:
            msg = ControlMessage()
            msg.control_mode = 2
            msg.input_mode = 2
            msg.input_pos = 0.0
            msg.input_vel = right_vel
            msg.input_torque = 0.0
            self.axis_pubs[i].publish(msg)
        
        for i in self.left_wheels:
            msg = ControlMessage()
            msg.control_mode = 2
            msg.input_mode = 2
            msg.input_pos = 0.0
            msg.input_vel = left_vel
            msg.input_torque = 0.0
            self.axis_pubs[i].publish(msg)
    
    def run_test(self, duration=10.0, speed=0.3):
        self.wait_for_services()
        self.set_closed_loop()
        
        start_positions = self.get_current_positions()
        self.get_logger().info(f"Starting positions (turns): {[f'{p:.3f}' for p in start_positions]}")
        self.get_logger().info(f"Running at {speed} turns/sec for {duration} seconds...")
        self.get_logger().info(f"Expected turns: {speed * duration:.1f}")
        
        # Start moving
        self.send_velocity(speed, -speed)
        
        # Run for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            elapsed = time.time() - start_time
            self.get_logger().info(f"Time: {elapsed:.1f}s / {duration}s | Positions: {[f'{p:.3f}' for p in self.positions]}")
            time.sleep(0.5)
        
        # Stop
        self.send_velocity(0.0, 0.0)
        self.get_logger().info("Stopping wheels...")
        time.sleep(0.5)
        
        # Final report
        final_positions = self.get_current_positions()
        self.get_logger().info("=" * 50)
        self.get_logger().info("TEST COMPLETE")
        self.get_logger().info("=" * 50)
        
        expected_turns = speed * duration
        
        self.get_logger().info("RIGHT WHEELS (0,1,2):")
        for i in self.right_wheels:
            actual_turns = final_positions[i] - start_positions[i]
            error_turns = actual_turns - expected_turns
            error_percent = (error_turns / expected_turns) * 100
            self.get_logger().info(
                f"  Axis {i}: {actual_turns:.3f} turns | "
                f"Error: {error_turns:.3f} ({error_percent:.2f}%)"
            )
        
        self.get_logger().info("LEFT WHEELS (3,4,5):")
        for i in self.left_wheels:
            actual_turns = final_positions[i] - start_positions[i]
            expected = -expected_turns
            error_turns = actual_turns - expected
            error_percent = (error_turns / expected_turns) * 100
            self.get_logger().info(
                f"  Axis {i}: {actual_turns:.3f} turns | "
                f"Error: {error_turns:.3f} ({error_percent:.2f}%)"
            )


def main(args=None):
    rclpy.init(args=args)
    node = EncoderTest()
    
    try:
        node.run_test(duration=10.0, speed=0.3)  # 10 sec × 0.3 turns/sec = 3 turns
    except KeyboardInterrupt:
        node.get_logger().info("Test interrupted")
        node.send_velocity(0.0, 0.0)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
