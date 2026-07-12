#!/usr/bin/env python3
"""
Publishes static joint states for the rover wheel joints.
These joints are not controlled by ros2_control, so we publish default values.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class WheelJointPublisher(Node):
    def __init__(self):
        super().__init__('wheel_joint_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        
        # Wheel joint names
        self.joint_names = [
            'L_Rocker_Joint',
            'R_Rocker_Joint',
            'L_Boggie_Joint',
            'R_Boggie_Joint',
            'L_1_Wheel_Joint',
            'L_2_Wheel_Joint',
            'L_3_Wheel_Joint',
            'R_1_Wheel_Joint',
            'R_2_Wheel_Joint',
            'R_3_Wheel_Joint',
            'aux_L_Rocker_joint',
            'aux_R_Rocker_joint',
        ]
        
        # Publish at 100 Hz to keep monitored state timestamps fresh.
        self.timer = self.create_timer(0.01, self.publish_joint_states)
        self.get_logger().info('Wheel joint publisher started')
    
    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [0.0] * len(self.joint_names)
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = []
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WheelJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
