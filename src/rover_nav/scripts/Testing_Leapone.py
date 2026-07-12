#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from odrive_can.srv import AxisState
from odrive_can.msg import ControlMessage
import time

def main(args=None):
    rclpy.init(args=args)
    node = Node("six_wheel_controller")

    num_axes = 6
    velocity = 10.0  # you can change this value
   
    # Create service clients and publishers for each ODrive axis
    clients = []
    pubs = []
    for i in range(num_axes):
        srv_name = f"/odrive_axis{i}/request_axis_state"
        topic_name = f"/odrive_axis{i}/control_message"
        clients.append(node.create_client(AxisState, srv_name))
        pubs.append(node.create_publisher(ControlMessage, topic_name, 10))

    # Wait for all services to be available
    for i, client in enumerate(clients):
        while not client.wait_for_service(timeout_sec=1.0):
            node.get_logger().info(f"Waiting for service /odrive_axis{i}/request_axis_state...")

    # 1️⃣ Request CLOSED_LOOP_CONTROL (state = 8) for all axes
    for i, client in enumerate(clients):
        req = AxisState.Request()
        req.axis_requested_state = 8
        node.get_logger().info(f"Setting odrive_axis{i} to CLOSED_LOOP_CONTROL...")
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        time.sleep(0.1)
    
    # 2️⃣ Publish velocity command to all axes
    msg = ControlMessage()
    msg.control_mode = 2
    msg.input_mode = 1
    msg.input_pos = 0.0
    msg.input_vel = velocity
    msg.input_torque = 0.0

    for i, pub in enumerate(pubs):
        node.get_logger().info(f"Publishing velocity command to odrive_axis{i} (vel = {velocity})...")
        pub.publish(msg)
        time.sleep(0.1)

    node.get_logger().info("✅ All 6 wheels set to CLOSED_LOOP_CONTROL and running at target velocity.")

    rclpy.shutdown()

if __name__ == "__main__":
    main()
