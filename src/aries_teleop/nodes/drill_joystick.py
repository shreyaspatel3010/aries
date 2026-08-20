#!/usr/bin/env python3
"""Joystick control for the drill's three axes, gated behind LT.

Mapping (canonical layout, i.e. after joy_layout_normalizer):

    LT + D-pad up/down      feed carriage   drill_motor_joint      position
    LT + D-pad left/right   sample bin      drill_container_joint  position
    LT + left stick up/down auger           drill_bit_joint        velocity

LT rather than LB: LB is the rover's drive enable and its left stick is the
drive command, so LB + left stick would spin the auger and drive the rover at
the same time. LT is only taken by arm_preset_pose_joystick, and only in
combination with the FACE buttons (Y/A/B), so the d-pad and the sticks are free
under it. RB and RT gate the arm, so nothing here can move the arm either.

LB also BLOCKS the drill outright, so LB + LT reaches nothing. Two separate
nodes reading two separate gates would otherwise let the rover drive off with
the auger spinning and the mast down; the arm teleop already yields to LB the
same way, and the drill now matches it. LB wins whichever order they are
pressed in, and blocking behaves exactly like releasing LT: auger to zero, feed
targets latched where they are.

The two feed axes are POSITION-controlled - they are a lead screw and a rail,
and gz drives them through JointPositionController - so holding a direction
integrates a target at a fixed speed and clamps it at the joint limit. The
auger is VELOCITY-controlled through JointController, so the stick is the
speed directly and centring it stops the spindle.

Targets track /joint_states while LT is released, so grabbing the trigger never
snaps a joint back to wherever this node last left it - it resumes from where
the drill actually is. Without that, a drill moved by anything else (a future
autonomous sequence, or the sim being reset) would jump on the first press.

Publishing is likewise gated: a 30 Hz command stream while LT is held, one
final message per topic on release (auger zeroed, feed targets latched), then
silence. An idle drill leaves its command topics free for another publisher.

Nothing consumes these topics on the real rover yet - there is no drill driver.
They are bridged into gz by aries/config/*_gazebo_bridge.yaml.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float64


class DrillJoystick(Node):
    def __init__(self):
        super().__init__("drill_joystick")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("joint_states_topic", "/joint_states")

        self.declare_parameter("motor_cmd_topic", "/aries/drill_motor_joint/cmd_pos")
        self.declare_parameter("container_cmd_topic", "/aries/drill_container_joint/cmd_pos")
        self.declare_parameter("bit_cmd_topic", "/aries/drill_bit_joint/cmd_vel")

        self.declare_parameter("motor_joint", "drill_motor_joint")
        self.declare_parameter("container_joint", "drill_container_joint")

        # LT. joy_layout_normalizer republishes canonical axis 2 as
        # 0.0 released -> 1.0 fully pressed on every driver, so this is a press
        # fraction. Release below press, for hysteresis, exactly as
        # gamepad.yaml does for RT.
        self.declare_parameter("modifier_axis", 2)
        self.declare_parameter("modifier_threshold", 0.5)
        self.declare_parameter("modifier_release", 0.35)

        # LB wins over LT. Holding the rover's drive enable blocks the drill
        # outright, the same way the arm teleop hard-stops and returns on LB
        # ("LB rover mode active: arm blocked"). Without this, LB + LT drives
        # the rover with the auger spinning and the mast down, because the two
        # nodes read different gates and neither knows about the other. Set to
        # -1 to disable the block.
        self.declare_parameter("block_button", 4)

        # D-pad. In the canonical layout axis 7 is +1 UP / -1 DOWN and axis 6 is
        # +1 LEFT / -1 RIGHT (the normalizer synthesises both from SDL buttons
        # 11..14 with exactly those signs). The d-pad is digital, so this is a
        # detection threshold, not a deadzone.
        self.declare_parameter("dpad_vertical_axis", 7)
        self.declare_parameter("dpad_horizontal_axis", 6)
        self.declare_parameter("dpad_threshold", 0.5)

        self.declare_parameter("bit_axis", 1)
        self.declare_parameter("bit_deadzone", 0.20)

        # Keep these in step with drill.xacro's joint limits. They are repeated
        # rather than read from the URDF so this node can run without a robot
        # description, but a mismatch means the teleop stops short of, or
        # commands past, the real stop.
        self.declare_parameter("motor_lower", -0.375)
        self.declare_parameter("motor_upper", 0.185)
        self.declare_parameter("motor_speed", 0.05)

        self.declare_parameter("container_lower", -0.1304)
        self.declare_parameter("container_upper", 0.0)
        self.declare_parameter("container_speed", 0.05)

        # drill_bit_joint's URDF velocity limit is 60 rad/s; half of that is a
        # sane manual ceiling (~285 rpm).
        self.declare_parameter("bit_max_speed", 30.0)

        # Sign conventions, all expressed as "what the positive stick/pad
        # direction does", and all flippable without touching the code:
        #   motor      D-pad UP raises the carriage (+Z).
        #   container  D-pad RIGHT parks the bin forward (+X); LEFT runs it back
        #              under the auger, which is the -0.1304 end of the stroke.
        #   bit        stick UP turns the auger CLOCKWISE seen from above, which
        #              is the cutting direction of a right-hand flight and,
        #              being a rotation about +Z, is a NEGATIVE rate.
        self.declare_parameter("invert_motor", False)
        self.declare_parameter("invert_container", False)
        self.declare_parameter("invert_bit", False)

        self.declare_parameter("publish_rate_hz", 30.0)
        # A stale /joy is treated as LT released: the auger must not keep
        # spinning because the pad's battery died mid-cut.
        self.declare_parameter("joy_timeout_sec", 0.35)

        g = self.get_parameter
        joy_topic = str(g("joy_topic").value)
        joint_states_topic = str(g("joint_states_topic").value)
        self.motor_joint = str(g("motor_joint").value)
        self.container_joint = str(g("container_joint").value)

        self.modifier_axis = int(g("modifier_axis").value)
        self.modifier_threshold = float(g("modifier_threshold").value)
        self.modifier_release = float(g("modifier_release").value)
        self.block_button = int(g("block_button").value)

        self.dpad_vertical_axis = int(g("dpad_vertical_axis").value)
        self.dpad_horizontal_axis = int(g("dpad_horizontal_axis").value)
        self.dpad_threshold = float(g("dpad_threshold").value)

        self.bit_axis = int(g("bit_axis").value)
        self.bit_deadzone = float(g("bit_deadzone").value)

        self.motor_lower = float(g("motor_lower").value)
        self.motor_upper = float(g("motor_upper").value)
        self.motor_speed = float(g("motor_speed").value)
        self.container_lower = float(g("container_lower").value)
        self.container_upper = float(g("container_upper").value)
        self.container_speed = float(g("container_speed").value)
        self.bit_max_speed = float(g("bit_max_speed").value)

        self.motor_sign = -1.0 if bool(g("invert_motor").value) else 1.0
        self.container_sign = -1.0 if bool(g("invert_container").value) else 1.0
        # Stick UP (+1) -> clockwise from above -> negative about +Z.
        self.bit_sign = 1.0 if bool(g("invert_bit").value) else -1.0

        self.publish_rate_hz = float(g("publish_rate_hz").value)
        self.joy_timeout_sec = float(g("joy_timeout_sec").value)

        # Targets stay None until /joint_states or the first command seeds
        # them, so a d-pad press before either can never publish a target
        # invented out of nothing.
        self.motor_target = None
        self.container_target = None
        self.measured = {}

        self.modifier_held = False
        self.blocked = False
        self.was_blocked = False
        self.was_active = False
        self.last_joy_time = 0.0
        self.axis_motor = 0.0
        self.axis_container = 0.0
        self.axis_bit = 0.0

        self.pub_motor = self.create_publisher(Float64, str(g("motor_cmd_topic").value), 10)
        self.pub_container = self.create_publisher(Float64, str(g("container_cmd_topic").value), 10)
        self.pub_bit = self.create_publisher(Float64, str(g("bit_cmd_topic").value), 10)

        self.create_subscription(Joy, joy_topic, self._joy_cb, 10)
        self.create_subscription(JointState, joint_states_topic, self._joint_state_cb, 10)

        self.dt = 1.0 / max(self.publish_rate_hz, 1.0)
        self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f"Drill joystick ready. Hold LT/axis {self.modifier_axis}: "
            f"d-pad up/down = feed, d-pad left/right = bin, "
            f"left stick up/down = auger."
        )

    # -- helpers -----------------------------------------------------------
    def _axis(self, msg, index):
        return float(msg.axes[index]) if 0 <= index < len(msg.axes) else 0.0

    def _button(self, msg, index):
        return int(msg.buttons[index]) if 0 <= index < len(msg.buttons) else 0

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def _deadzone(self, value, dz):
        if abs(value) < dz:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * (abs(value) - dz) / max(1e-6, 1.0 - dz)

    @staticmethod
    def _detent(value, threshold):
        if value >= threshold:
            return 1.0
        if value <= -threshold:
            return -1.0
        return 0.0

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -- callbacks ---------------------------------------------------------
    def _joint_state_cb(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name in (self.motor_joint, self.container_joint):
                self.measured[name] = float(position)

    def _joy_cb(self, msg):
        self.last_joy_time = self._now()

        self.blocked = self._button(msg, self.block_button) == 1
        if self.blocked and not self.was_blocked:
            self.get_logger().info("LB rover mode active: drill blocked")
        self.was_blocked = self.blocked

        trigger = self._axis(msg, self.modifier_axis)
        if self.modifier_held:
            self.modifier_held = trigger > self.modifier_release
        else:
            self.modifier_held = trigger >= self.modifier_threshold

        self.axis_motor = self._detent(self._axis(msg, self.dpad_vertical_axis), self.dpad_threshold)
        # Canonical axis 6 is +1 LEFT, so RIGHT (the +X, park direction) is the
        # negated axis.
        self.axis_container = -self._detent(
            self._axis(msg, self.dpad_horizontal_axis), self.dpad_threshold)
        self.axis_bit = self._deadzone(self._axis(msg, self.bit_axis), self.bit_deadzone)

    # -- output ------------------------------------------------------------
    def _timer_cb(self):
        stale = (self._now() - self.last_joy_time) > self.joy_timeout_sec
        active = self.modifier_held and not stale and not self.blocked

        if not active:
            # Follow the joints while idle so the next press resumes from the
            # real pose rather than from a stale target.
            self.motor_target = self.measured.get(self.motor_joint, self.motor_target)
            self.container_target = self.measured.get(self.container_joint, self.container_target)
            if self.was_active:
                self._publish(bit_rate=0.0)
                reason = "LB held" if self.blocked else "LT released"
                self.get_logger().info(f"{reason}: auger stopped, feed axes latched")
            self.was_active = False
            return

        if not self.was_active:
            # Seed from the measured pose, or from 0.0 (the CAD home) when
            # nothing is publishing joint states at all.
            if self.motor_target is None:
                self.motor_target = self.measured.get(self.motor_joint, 0.0)
            if self.container_target is None:
                self.container_target = self.measured.get(self.container_joint, 0.0)
            self.get_logger().info("LT held: drill teleop active")
        self.was_active = True

        self.motor_target = self._clamp(
            self.motor_target + self.motor_sign * self.axis_motor * self.motor_speed * self.dt,
            self.motor_lower, self.motor_upper)
        self.container_target = self._clamp(
            self.container_target + self.container_sign * self.axis_container * self.container_speed * self.dt,
            self.container_lower, self.container_upper)

        self._publish(bit_rate=self.bit_sign * self.axis_bit * self.bit_max_speed)

    def _publish(self, bit_rate):
        if self.motor_target is not None:
            self.pub_motor.publish(Float64(data=float(self.motor_target)))
        if self.container_target is not None:
            self.pub_container.publish(Float64(data=float(self.container_target)))
        self.pub_bit.publish(Float64(data=float(bit_rate)))


def main(args=None):
    rclpy.init(args=args)
    node = DrillJoystick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # gz's JointController latches the last rate it was given, so an auger
        # left spinning when this node dies keeps spinning. Zero it on the way
        # out, the same way the rover teleop zeroes its Twist. The feed targets
        # are positions and are meant to stay where they are.
        if rclpy.ok():
            node.pub_bit.publish(Float64(data=0.0))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
