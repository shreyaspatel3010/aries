#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


class MockRoverDrive(Node):
    """
    Mock rover backend for rover_drive launches.

    Used when real CAN/ODrive rover hardware is not connected.

    Publishes:
      /cmd_vel
      /odom
      /tf: odom -> base_footprint
      /mock_rover/status

    This lets RViz show rover motion even without physical rover hardware.
    """

    def __init__(self):
        super().__init__("mock_rover_drive")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("status_topic", "/mock_rover/status")

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("publish_tf", True)

        self.declare_parameter("enable_button", 4)       # LB
        self.declare_parameter("linear_axis", 1)
        self.declare_parameter("angular_axis", 0)

        # Your latest correction: forward/reverse fixed.
        self.declare_parameter("invert_linear", False)
        self.declare_parameter("invert_angular", False)

        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("max_linear", 0.70)
        self.declare_parameter("max_angular", 1.70)
        self.declare_parameter("accel_limit", 1.20)
        self.declare_parameter("angular_accel_limit", 3.00)
        self.declare_parameter("publish_rate_hz", 30.0)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.enable_button = int(self.get_parameter("enable_button").value)
        self.linear_axis = int(self.get_parameter("linear_axis").value)
        self.angular_axis = int(self.get_parameter("angular_axis").value)
        self.invert_linear = bool(self.get_parameter("invert_linear").value)
        self.invert_angular = bool(self.get_parameter("invert_angular").value)

        self.deadzone = float(self.get_parameter("deadzone").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.accel_limit = float(self.get_parameter("accel_limit").value)
        self.angular_accel_limit = float(self.get_parameter("angular_accel_limit").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.enabled = False
        self.prev_enabled = False
        self.last_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(Joy, self.joy_topic, self._joy_cb, 10)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self._timer_cb)
        self.create_timer(2.0, self._status_cb)

        self.get_logger().warn(
            "MOCK ROVER DRIVE ACTIVE. No ODrive/CAN hardware is commanded. "
            f"Hold LB/button {self.enable_button} to drive mock rover in RViz. "
            f"Publishing {self.odom_topic} and TF {self.odom_frame}->{self.base_frame}."
        )

    def _axis(self, msg, index):
        return float(msg.axes[index]) if 0 <= index < len(msg.axes) else 0.0

    def _button(self, msg, index):
        return int(msg.buttons[index]) if 0 <= index < len(msg.buttons) else 0

    def _deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * (abs(value) - self.deadzone) / max(1e-6, 1.0 - self.deadzone)

    def _ramp(self, current, target, limit, dt):
        step = abs(limit) * dt
        diff = target - current
        if abs(diff) <= step:
            return target
        return current + math.copysign(step, diff)

    def _yaw_to_quat(self, yaw):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def _joy_cb(self, msg):
        self.enabled = self._button(msg, self.enable_button) == 1

        if self.enabled and not self.prev_enabled:
            self.get_logger().debug("Mock rover enabled by LB")
        if not self.enabled and self.prev_enabled:
            self.get_logger().debug("Mock rover stopped")
        self.prev_enabled = self.enabled

        if not self.enabled:
            self.target_linear = 0.0
            self.target_angular = 0.0
            return

        linear = self._axis(msg, self.linear_axis)
        angular = self._axis(msg, self.angular_axis)

        if self.invert_linear:
            linear = -linear
        if self.invert_angular:
            angular = -angular

        self.target_linear = self._deadzone(linear) * self.max_linear
        self.target_angular = self._deadzone(angular) * self.max_angular

    def _timer_cb(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 0.5:
            dt = 1.0 / max(self.publish_rate_hz, 1.0)

        self.current_linear = self._ramp(
            self.current_linear,
            self.target_linear,
            self.accel_limit,
            dt,
        )
        self.current_angular = self._ramp(
            self.current_angular,
            self.target_angular,
            self.angular_accel_limit,
            dt,
        )

        # Integrate simple planar differential-drive odom.
        self.x += self.current_linear * math.cos(self.yaw) * dt
        self.y += self.current_linear * math.sin(self.yaw) * dt
        self.yaw += self.current_angular * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        qz, qw = self._yaw_to_quat(self.yaw)

        cmd = Twist()
        cmd.linear.x = self.current_linear
        cmd.angular.z = self.current_angular
        self.cmd_pub.publish(cmd)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = self.current_linear
        odom.twist.twist.angular.z = self.current_angular

        # Reasonable covariance for mock odometry.
        odom.pose.covariance[0] = 0.02
        odom.pose.covariance[7] = 0.02
        odom.pose.covariance[35] = 0.05
        odom.twist.covariance[0] = 0.05
        odom.twist.covariance[35] = 0.10

        self.odom_pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = now.to_msg()
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

    def _status_cb(self):
        msg = String()
        msg.data = (
            f"mock_rover_drive active enabled={self.enabled} "
            f"x={self.x:.3f} y={self.y:.3f} yaw={self.yaw:.3f} "
            f"cmd_linear={self.current_linear:.3f} cmd_angular={self.current_angular:.3f} "
            f"odom={self.odom_topic} tf={self.odom_frame}->{self.base_frame}"
        )
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockRoverDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
