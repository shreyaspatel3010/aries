#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


class RoverCmdVelJoystick(Node):
    def __init__(self):
        super().__init__("rover_cmd_vel_joystick")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel/teleop")

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

        # Stop when /joy goes quiet. This is a relay: it publishes the LAST
        # joystick state on a timer, so without this check a dead /joy is
        # indistinguishable from a held stick and the rover keeps driving at
        # whatever it was last told.
        #
        # That also defeats the guard downstream. cmd_vel_odrive_bridge has a
        # command_timeout_s of its own, but it times out on /cmd_vel going
        # SILENT -- and this node keeps it fed with fresh messages carrying a
        # stale command, so the bridge never sees a gap. The staleness has to
        # be caught here, at the point where it is visible.
        #
        # It matters far more now that the pad lives on the base station: with
        # the joystick on the rover, /joy stops only if the driver dies or the
        # pad is unplugged. Over the antenna, an ordinary radio dropout does it,
        # and the rover is 150 m away.
        #
        # 0.35 s matches rebel_servo_teleop_gamepad, drill_joystick and
        # arm_preset_pose_joystick, so every teleop path on the pad gives up at
        # the same moment. joy_node autorepeats at 80 Hz, so 0.35 s is 28 missed
        # messages -- comfortably past jitter, well short of a runaway.
        self.declare_parameter("joy_timeout_sec", 0.35)

        # LB + Y re-arms the drive. The request goes to the ODrive bridge's
        # enable service rather than to the axes directly, so the bridge stays
        # the single owner of the motor commands and the waypoint arbiter is
        # not bypassed. The bridge re-requests CLOSED_LOOP_CONTROL on all six
        # axes, and the vendor CAN node sends kClearErrors ahead of the
        # kSetAxisState frame, so a faulted axis recovers on the same call.
        # An empty enable_service disables the binding.
        self.declare_parameter("enable_service", "/aries_drive/enable")
        self.declare_parameter("enabled_topic", "/aries_drive/enabled")
        self.declare_parameter("reinit_button", 3)              # Y
        self.declare_parameter("reinit_modifier_button", 4)     # LB
        # LB is held by definition during the combo, so the drive gate is open
        # the instant the axes arm. Publish zero for this long afterwards so a
        # deflected stick cannot lurch the rover on re-arm.
        self.declare_parameter("reinit_hold_sec", 1.0)
        # The bridge answers the enable call immediately and arms in the
        # background, so the settle above can only be measured from the moment
        # it reports the drive enabled — which is anywhere from about a second
        # to several, since the same call re-establishes a CAN interface that
        # was unplugged. Zero is held until that report arrives, and this caps
        # the wait for the case where it never does (nothing armed, so nothing
        # can move either).
        self.declare_parameter("reinit_hold_max_sec", 8.0)

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
        self.joy_timeout_sec = max(
            0.0, float(self.get_parameter("joy_timeout_sec").value)
        )

        self.enable_service = str(self.get_parameter("enable_service").value).strip()
        self.enabled_topic = str(self.get_parameter("enabled_topic").value).strip()
        self.reinit_button = int(self.get_parameter("reinit_button").value)
        self.reinit_modifier_button = int(
            self.get_parameter("reinit_modifier_button").value
        )
        self.reinit_hold_sec = float(self.get_parameter("reinit_hold_sec").value)
        self.reinit_hold_max_sec = max(
            self.reinit_hold_sec, float(self.get_parameter("reinit_hold_max_sec").value)
        )

        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0

        self.was_enabled = False

        # Zero, not "now": a node that comes up before the joystick does has
        # never had a command, and must not treat that as a fresh one.
        self._last_joy_at = 0.0
        self._joy_lost = False

        # Initialised before the callbacks that read them are registered.
        self._prev_reinit_combo = False
        self._reinit_hold_until = 0.0
        self._reinit_deadline = 0.0
        self._awaiting_enable = False
        self._reinit_call_pending = False

        self.enable_client = (
            self.create_client(SetBool, self.enable_service)
            if self.enable_service
            else None
        )

        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(Joy, self.joy_topic, self._joy_cb, 10)
        if self.enabled_topic:
            self.create_subscription(
                Bool,
                self.enabled_topic,
                self._enabled_cb,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
        self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self._timer_cb)

        combo = (
            f"Press button {self.reinit_modifier_button}+{self.reinit_button} "
            f"(LB+Y) to re-arm the drive via {self.enable_service}."
            if self.enable_client is not None
            else "Drive re-arm binding disabled (empty enable_service)."
        )
        self.get_logger().info(
            f"Rover joystick ready. Hold LB/button {self.enable_button}. "
            f"Publishing ONLY {self.cmd_vel_topic}. {combo}"
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
        self._last_joy_at = time.monotonic()
        if self._joy_lost:
            self._joy_lost = False
            self.get_logger().info("joystick back: drive re-enabled (hold LB)")

        enabled = self._button(msg, self.enable_button) == 1

        if enabled and not self.was_enabled:
            self.get_logger().info("LB pressed: rover drive enabled")
        if not enabled and self.was_enabled:
            self.get_logger().info("LB released: rover drive stopped")
        self.was_enabled = enabled

        # LB + Y, edge-triggered so holding the pair re-arms exactly once.
        reinit_combo = bool(
            self._button(msg, self.reinit_modifier_button)
            and self._button(msg, self.reinit_button)
        )
        if reinit_combo and not self._prev_reinit_combo:
            self._request_reinit()
        self._prev_reinit_combo = reinit_combo

        if enabled and not reinit_combo:
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

    def _request_reinit(self):
        """LB + Y: ask the ODrive bridge to re-arm every axis."""
        if self.enable_client is None:
            return

        if self._reinit_call_pending:
            self.get_logger().warn("Drive re-arm already in flight — ignoring LB+Y")
            return

        # service_is_ready() rather than wait_for_service(): this runs in the
        # subscription callback, and blocking here would stall the executor
        # that has to deliver the very response being waited on.
        if not self.enable_client.service_is_ready():
            self.get_logger().warn(
                f"LB+Y: {self.enable_service} unavailable — is the ODrive bridge running?"
            )
            return

        # Only once the request is actually going out: nothing can arm if it
        # is not, so holding the output there would be a freeze for no reason.
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        now = time.monotonic()
        self._reinit_hold_until = now + self.reinit_hold_sec
        self._reinit_deadline = now + self.reinit_hold_max_sec
        self._awaiting_enable = bool(self.enabled_topic)

        self.get_logger().warn(f"LB+Y: re-arming the drive via {self.enable_service}")
        self._reinit_call_pending = True
        future = self.enable_client.call_async(SetBool.Request(data=True))
        future.add_done_callback(self._on_reinit_response)

    def _enabled_cb(self, msg):
        """The bridge reported the arm result; start the settle from here."""
        if not self._awaiting_enable or not msg.data:
            return
        self._awaiting_enable = False
        self._reinit_hold_until = time.monotonic() + self.reinit_hold_sec
        self.get_logger().info("LB+Y: drive reported enabled; releasing the zero hold")

    def _on_reinit_response(self, future):
        self._reinit_call_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"LB+Y: drive re-arm request failed: {exc}")
            return
        if response is None:
            self.get_logger().error("LB+Y: drive re-arm returned an empty result")
        elif response.success:
            self.get_logger().info(f"LB+Y: {response.message}")
        else:
            self.get_logger().warn(f"LB+Y: drive re-arm rejected: {response.message}")

    def _timer_cb(self):
        dt = 1.0 / max(self.publish_rate_hz, 1.0)
        now = time.monotonic()

        # Radio dropout, dead joy driver, unplugged pad: all the same thing
        # from here, and all of them mean the last command is no longer an
        # instruction. Publish an explicit zero rather than falling silent --
        # the bridge ramps it down at wheel_accel_rps2 the same way it handles
        # a released LB, and an explicit stop does not depend on any
        # downstream timeout being configured the way we expect.
        if self.joy_timeout_sec > 0.0 and (now - self._last_joy_at) > self.joy_timeout_sec:
            if not self._joy_lost and self._last_joy_at > 0.0:
                self._joy_lost = True
                self.get_logger().warn(
                    f"no /joy for {self.joy_timeout_sec:.2f} s — stopping the rover "
                    f"(link down, or the joy driver died)"
                )
            self.target_linear = 0.0
            self.target_angular = 0.0
            self.current_linear = 0.0
            self.current_angular = 0.0
            self.pub.publish(Twist())
            return

        # The bridge arms asynchronously, so LB alone would let a deflected
        # stick command motion the moment the axes close the loop.
        if self._awaiting_enable and now >= self._reinit_deadline:
            self._awaiting_enable = False
            self.get_logger().warn(
                f"LB+Y: drive never reported enabled within {self.reinit_hold_max_sec:.0f} s; "
                f"releasing the zero hold (the axes did not arm)"
            )
        if self._awaiting_enable or now < self._reinit_hold_until:
            self.target_linear = 0.0
            self.target_angular = 0.0
            self.current_linear = 0.0
            self.current_angular = 0.0
            self.pub.publish(Twist())
            return

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
        if rclpy.ok():
            node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
