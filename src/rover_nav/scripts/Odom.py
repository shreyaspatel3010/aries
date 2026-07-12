#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from odrive_can.msg import ControllerStatus
import math

# Constants
WHEEL_CIRCUMFERENCE = 0.697  # meters
WHEELBASE = 0.566  # meters
PUBLISH_RATE = 20  # Hz

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odom_node')
        
        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Velocity state
        self.vx = 0.0
        self.vth = 0.0
        
        # Previous encoder positions (revolutions)
        self.prev_left_pos = [None, None, None]
        self.prev_right_pos = [None, None, None]
        
        # Current encoder positions
        self.current_left_pos = [0.0, 0.0, 0.0]
        self.current_right_pos = [0.0, 0.0, 0.0]
        
        # Timestamps for velocity calculation
        self.last_update_time = self.get_clock().now()
        
        # Subscribe to all 6 wheels
        for i in range(6):
            self.create_subscription(
                ControllerStatus,
                f'/odrive_axis{i}/controller_status',
                lambda msg, axis=i: self.feedback_callback(msg, axis),
                10
            )
        
        # Publisher - ONLY odometry topic, NO TF!
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        
        # Timer for periodic odometry calculation and publishing
        self.create_timer(1.0/PUBLISH_RATE, self.update_odometry)
        
        self.get_logger().info("✅ Odometry node started (TF disabled - EKF will publish TF)")
    
    def feedback_callback(self, msg, axis):
        """Store encoder positions from each wheel"""
        current_pos = msg.pos_estimate  # revolutions
        
        if axis in [0, 1, 2]:  # Right wheels
            self.current_right_pos[axis] = current_pos
        else:  # Left wheels (3, 4, 5)
            self.current_left_pos[axis - 3] = current_pos
    
    def update_odometry(self):
        """Calculate and publish odometry at fixed rate"""
        current_time = self.get_clock().now()
        
        # Calculate average positions for each side
        avg_left_pos = sum(self.current_left_pos) / 3
        avg_right_pos = sum(self.current_right_pos) / 3
        
        # Initialize previous positions on first run
        if self.prev_left_pos[0] is None:
            self.prev_left_pos = [avg_left_pos] * 3
            self.prev_right_pos = [avg_right_pos] * 3
            self.last_update_time = current_time
            return
        
        # Calculate delta in revolutions (since last update)
        delta_left_rev = avg_left_pos - sum(self.prev_left_pos) / 3
        delta_right_rev = avg_right_pos - sum(self.prev_right_pos) / 3
        
        # Update previous positions
        self.prev_left_pos = self.current_left_pos.copy()
        self.prev_right_pos = self.current_right_pos.copy()
        
        # Convert to distance - LEFT WHEELS ARE NEGATED (opposite mounting)
        d_left = -delta_left_rev * WHEEL_CIRCUMFERENCE
        d_right = delta_right_rev * WHEEL_CIRCUMFERENCE
        
        # Skip if no significant movement
        if abs(d_left) < 0.0001 and abs(d_right) < 0.0001:
            self.publish_odom(current_time)
            return
        
        # Calculate movement
        d_center = (d_left + d_right) / 2.0
        delta_theta = (d_right - d_left) / WHEELBASE
        
        # Calculate velocities
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        if dt > 0:
            self.vx = d_center / dt
            self.vth = delta_theta / dt
        
        # Update pose using midpoint method for better accuracy
        theta_mid = self.theta + delta_theta / 2.0
        delta_x = d_center * math.cos(theta_mid)
        delta_y = d_center * math.sin(theta_mid)
        
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Update timestamp
        self.last_update_time = current_time
        
        # Publish
        self.publish_odom(current_time)
    
    def publish_odom(self, current_time):
        """Publish odometry message (NO TF - EKF handles that!)"""
        # Create quaternion from yaw (for the odometry message)
        half_yaw = self.theta * 0.5
        q = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(half_yaw),
            w=math.cos(half_yaw),
        )
        
        # Publish odometry message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        
        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q
        
        # Velocity
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = self.vth
        
        # Pose covariance
        odom.pose.covariance[0] = 0.01   # x
        odom.pose.covariance[7] = 0.01   # y
        odom.pose.covariance[14] = 1e6   # z (not measured)
        odom.pose.covariance[21] = 1e6   # roll (not measured)
        odom.pose.covariance[28] = 1e6   # pitch (not measured)
        odom.pose.covariance[35] = 1e6   # yaw - LARGE (don't trust encoder yaw!)

        # Twist covariance
        odom.twist.covariance[0] = 0.01   # vx
        odom.twist.covariance[7] = 0.1    # vy
        odom.twist.covariance[14] = 1e6   # vz (not measured)
        odom.twist.covariance[21] = 1e6   # roll rate (not measured)
        odom.twist.covariance[28] = 1e6   # pitch rate (not measured)
        odom.twist.covariance[35] = 0.1   # yaw rate
        
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
