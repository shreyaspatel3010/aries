#!/usr/bin/env python3
"""
Publishes static joint states for joints ros2_control does not drive.

The rover wheels and passive suspension, plus the drill. Anything in the URDF
that nothing publishes leaves MoveIt's planning scene monitor without a
complete robot state, and servo refuses to run until it has one.
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
            # The drill joints are in the URDF but driven by nothing: no
            # ros2_control interface, no hardware. Left unpublished, MoveIt's
            # planning scene monitor never completes a robot state and warns
            # "The complete state of the robot is not yet known" forever, which
            # blocks servo. Zero is the stowed pose and is inside every limit
            # (container's upper bound is exactly 0.0).
            'drill_motor_joint',
            'drill_bit_joint',
            'drill_container_joint',
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
