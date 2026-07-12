#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class RoverCmdVelJoystick(Node):
    def __init__(self):
        super().__init__("rover_cmd_vel_joystick")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("enable_button", 4)       # LB
        self.declare_parameter("linear_axis", 1)         # left stick vertical
        self.declare_parameter("angular_axis", 0)        # left stick horizontal
        self.declare_parameter("invert_linear", False)
        self.declare_parameter("invert_angular", False)

        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("max_linear", 0.55)
        self.declare_parameter("max_angular", 1.60)
        self.declare_parameter("accel_limit", 1.20)
        self.declare_parameter("angular_accel_limit", 3.00)
        self.declare_parameter("publish_rate_hz", 30.0)

        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

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

        self.was_enabled = False

        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(Joy, self.joy_topic, self._joy_cb, 10)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self._timer_cb)

        self.get_logger().info(
            f"Rover simulation joystick ready. Hold LB/button {self.enable_button}. "
            f"Publishing ONLY {self.cmd_vel_topic}."
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

    def _joy_cb(self, msg):
        enabled = self._button(msg, self.enable_button) == 1

        if enabled and not self.was_enabled:
            self.get_logger().info("LB pressed: rover drive enabled")
        if not enabled and self.was_enabled:
            self.get_logger().info("LB released: rover drive stopped")
        self.was_enabled = enabled

        if enabled:
            linear = self._axis(msg, self.linear_axis)
            angular = self._axis(msg, self.angular_axis)

            if self.invert_linear:
                linear = -linear
            if self.invert_angular:
                angular = -angular

            self.target_linear = self._deadzone(linear) * self.max_linear
            self.target_angular = self._deadzone(angular) * self.max_angular

            self.get_logger().info(
                f"cmd_vel target: linear={self.target_linear:.2f}, angular={self.target_angular:.2f}",
                throttle_duration_sec=1.0,
            )
        else:
            self.target_linear = 0.0
            self.target_angular = 0.0

    def _timer_cb(self):
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

        msg = Twist()
        msg.linear.x = self.current_linear
        msg.angular.z = self.current_angular
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoverCmdVelJoystick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
