#!/usr/bin/env python3
"""Joystick control for the drill's three axes, gated behind LT.

Mapping (canonical layout, i.e. after joy_layout_normalizer):

    LT + D-pad up/down       feed carriage  drill_motor_joint      m/s
    LT + D-pad left/right    sample bin     drill_container_joint  m/s
    LT + left stick up/down  auger          drill_bit_joint        rad/s
    LT + right stick up/down sand box lid   /sand_box/lid/cmd      -1..1

HARDWARE MODEL. Every axis on the real drill is a DC motor - the feed carriage
turns a lead screw, the auger spins on the head, and the sample bin rides a
prismatic linear actuator. Not one of them is a position servo, so not one of
them is commanded a position here. A held button runs a motor, a released
button stops it, and the mechanism stays where the motor left it: neither a
lead screw nor a linear actuator of that kind backdrives, and gz reproduces
that by servo-pinning the two feed joints at whatever rate they were last
given, zero included.

That is why all three topics carry RATES. The previous version commanded the
two feed axes as positions, and a position target is a thing that can be told
the wrong place: gz held it with a force PID, which sagged mg/p_gain = 50 mm
under the carriage's 4.1 kg, and this node re-seeded its target from that
sagging measurement on every LT press - so each press sent the carriage 50 mm
further down its own stroke, on nothing but the trigger. Nothing is seeded now
and nothing latches; released means zero.

LIMIT SWITCHES. drill_motor has one at each end of its travel, bottom and top,
and the bin's actuator has its own at each end of its stroke. They live here
rather than in the URDF because that is where they live on the drill: a switch
cuts the motor at a MEASURED position, and it cuts it in ONE DIRECTION only -
sitting on the bottom switch must still leave the carriage free to come back
up. They trip `limit_margin` short of the URDF joint limits, which stay what
they always were: the mechanical stop the switch protects.

Position for that check comes from /joint_states, and is dead-reckoned from
the commanded rate whenever no measurement has arrived, so the switches still
work with no simulator attached. One caveat on the real rover: publish_wheel_
joints.py publishes the drill joints at a constant 0.0 (MoveIt needs a complete
robot state), so the measurement there is a placeholder that never moves and
the switches would never trip. That is harmless only for as long as it stays
true that no drill driver subscribes to these topics - when one is written, it
owns the switches, and this node's copy becomes the courtesy stop it already
is on every other axis.

LT rather than LB: LB is the rover's drive enable and its left stick is the
drive command, so LB + left stick would spin the auger and drive the rover at
the same time. LT is only taken by arm_preset_pose_joystick, and only in
combination with the FACE buttons (Y/A/B), so the d-pad and the sticks are free
under it. RB and RT gate the arm, so nothing here can move the arm either.

LB also BLOCKS the drill outright, so LB + LT reaches nothing. Two separate
nodes reading two separate gates would otherwise let the rover drive off with
the auger spinning and the mast down; the arm teleop already yields to LB the
same way, and the drill now matches it. LB wins whichever order they are
pressed in, and blocking behaves exactly like releasing LT: every motor off.

Publishing is gated too: a 30 Hz stream of rates while LT is held, then a short
burst of zeros on release (`stop_hold_sec`, so one dropped message cannot leave
a motor running), then silence. An idle drill leaves its command topics free
for another publisher, and silence is safe precisely because zero was the last
thing said.

Nothing consumes these topics on the real rover yet - there is no drill driver.
They are bridged into gz by aries/config/*_gazebo_bridge.yaml.

THE LID IS THE ONE AXIS THAT IS NOT LIKE THE OTHERS, in three ways worth
knowing before touching it.

It goes STRAIGHT TO THE BOARD. The other three publish Float64 rates onto the
gz bridge topics and a driver turns those into PWM; /sand_box/lid/cmd is
std_msgs/Float32 and is subscribed by the Teensy itself. The message type is
part of that contract - publish Float64 there and micro-ROS makes no match at
all, so the topic lists fine on both sides and nothing is ever delivered.

It has NO SIMULATION. There is no lid joint in the URDF and no gz bridge entry,
so this is the one control on the pad that does nothing whatsoever without the
real board attached.

And it CANNOT STOP ITSELF. It is a 360-degree continuous-rotation servo, so the
value is a speed with no resting state: whatever was last said keeps happening.
That is why it is inside the same stop_hold_sec burst as the motors rather than
being published only while the stick is pushed - and why the firmware runs its
own 500 ms watchdog underneath, in case this node is not there to send the zero
at all.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float32, Float64, UInt8


class LimitSwitchedAxis:
    """One DC-motor linear axis: a rate command, cut at each end of travel.

    `lower`/`upper` are the URDF joint limits - the mechanical stops. The
    switches sit `margin` inside them, and only ever block the direction that
    runs further into the stop.

    `has_limits` is whether this axis HAS switches at all. Only the feed
    carriage does: firmware/teensy_drill_sys/include/pins.h maps exactly two,
    both on the feed, and says so - "BOTH SWITCHES ARE ON THE FEED CARRIAGE"
    and "the BIN's two switches are not in this map ... until then the bin is
    dead-reckoned". With no switch and no encoder the bin's `position` here is
    pure dead reckoning from an ASSUMED q = 0 start, and gating on an invented
    number does not protect anything - it just refuses to move. Because the
    bin's `upper` IS 0.0, that assumed start sat exactly on the top stop and
    every command toward it was cut on the first press, which is what the
    "drill_container_joint: top limit switch, motor cut" flood was on
    2026-08-27. Off for the bin, on for the feed.
    """

    def __init__(self, joint, lower, upper, speed, sign, margin, has_limits=True):
        self.joint = joint
        self.lower = float(lower)
        self.upper = float(upper)
        self.speed = float(speed)
        self.sign = float(sign)
        self.margin = float(margin)
        self.has_limits = bool(has_limits)
        # Dead reckoning starts at the CAD home, which is q = 0 on both axes,
        # and is replaced by the first real measurement that arrives.
        self.position = 0.0
        self.measured = False
        self.tripped = 0

    def measure(self, position):
        self.position = float(position)
        self.measured = True

    def rate(self, command, dt):
        """Rate to publish for a -1/0/+1 pad command, plus any NEW trip.

        Returns (rate, trip) where trip is -1 the moment the bottom switch
        stops the motor, +1 for the top one, and 0 otherwise. It reports the
        edge only, so holding a direction against a switch logs once.
        """
        rate = self.sign * float(command) * self.speed

        trip = 0
        # An axis with no switches is never cut here. See the class docstring:
        # its position is dead reckoning, not a measurement, so a cut based on
        # it stops a healthy motor and reports a switch that does not exist.
        if self.has_limits:
            if rate > 0.0 and self.position >= self.upper - self.margin:
                rate, trip = 0.0, 1
            elif rate < 0.0 and self.position <= self.lower + self.margin:
                rate, trip = 0.0, -1

        if not self.measured:
            self.position = max(self.lower, min(self.upper,
                                                self.position + rate * dt))

        new_trip = trip if trip != self.tripped else 0
        self.tripped = trip
        return rate, new_trip


class DrillJoystick(Node):
    def __init__(self):
        super().__init__("drill_joystick")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("joint_states_topic", "/joint_states")

        # All three are rates. See the module docstring: the drill has no
        # position servo on any axis.
        self.declare_parameter("motor_cmd_topic", "/aries/drill_motor_joint/cmd_vel")
        self.declare_parameter("container_cmd_topic", "/aries/drill_container_joint/cmd_vel")
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

        # THE SAND BOX LID. Right stick vertical -- canonical axis 4, the only
        # stick still free under LT (0/1 is the left stick and drives the
        # auger, 2/5 are the triggers, 6/7 the d-pad).
        #
        # Float32, not Float64: this one is subscribed by the Teensy directly.
        # See the note in the module docstring.
        self.declare_parameter("lid_cmd_topic", "/sand_box/lid/cmd")
        self.declare_parameter("lid_axis", 4)
        self.declare_parameter("lid_deadzone", 0.20)
        # Ceiling on the speed sent to the servo, as a fraction of its full
        # rate. 1.0 lets the stick reach full speed; drop it if the lid slams.
        self.declare_parameter("lid_max_speed", 1.0)
        # WHICH DIRECTION OPENS THE LID is a fact about how the servo is
        # mounted, and there is no way to derive it -- the servo has no position
        # feedback and the firmware deliberately does not guess. If stick UP
        # closes the lid, flip this.
        self.declare_parameter("invert_lid", False)

        # Keep these in step with drill.xacro's joint limits: they are the
        # mechanical stops, and the limit switches are placed off them. They
        # are repeated rather than read from the URDF so this node can run
        # without a robot description, but a mismatch means the teleop stops
        # short of, or drives into, the real stop.
        self.declare_parameter("motor_lower", -0.375)
        self.declare_parameter("motor_upper", 0.185)
        self.declare_parameter("motor_speed", 0.05)

        self.declare_parameter("container_lower", -0.1304)
        self.declare_parameter("container_upper", 0.0)
        self.declare_parameter("container_speed", 0.05)

        # How far inside the mechanical stop each limit switch trips. A real
        # switch is placed to cut the motor before anything lands on anything.
        self.declare_parameter("limit_margin", 0.002)

        # WHICH AXES ACTUALLY HAVE SWITCHES. The feed carriage does - two of
        # them, bottom and top, on the pins pins.h calls LIMIT_SWITCH1/2. The
        # sample bin does NOT: the mechanism has a pair for it but they are not
        # in the loom or the firmware pin map, so nothing can read them. See
        # LimitSwitchedAxis' docstring for why gating the bin on dead reckoning
        # blocked it outright instead of protecting it.
        self.declare_parameter("motor_has_limits", False)
        self.declare_parameter("container_has_limits", False)

        # THE REAL SWITCHES, straight off the board. firmware/teensy_drill_sys
        # publishes drill/limits as a UInt8 -- bit0 bottom, bit1 top -- and it
        # is the only feedback the drill has: no axis carries an encoder, so
        # nothing else here is a measurement. Gate on this and never on a
        # position, which can only ever be dead reckoning.
        self.declare_parameter("limits_topic", "/drill/limits")
        # Two missed heartbeats. The board republishes at 2 Hz even when
        # nothing changes, so silence for this long means the board, the agent
        # or the link is gone -- NOT that the switches are open.
        self.declare_parameter("limits_timeout_sec", 1.5)

        # drill_bit_joint's URDF velocity limit is 60 rad/s; half of that is a
        # sane manual ceiling (~285 rpm). The auger is continuous - it has no
        # travel, so it has no limit switches either.
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
        # Zeros are repeated for this long after the drill goes idle. One
        # message is enough in theory; a motor left running because a single
        # message was dropped is not a theory worth testing.
        self.declare_parameter("stop_hold_sec", 0.5)

        g = self.get_parameter
        joy_topic = str(g("joy_topic").value)
        joint_states_topic = str(g("joint_states_topic").value)

        self.modifier_axis = int(g("modifier_axis").value)
        self.modifier_threshold = float(g("modifier_threshold").value)
        self.modifier_release = float(g("modifier_release").value)
        self.block_button = int(g("block_button").value)

        self.dpad_vertical_axis = int(g("dpad_vertical_axis").value)
        self.dpad_horizontal_axis = int(g("dpad_horizontal_axis").value)
        self.dpad_threshold = float(g("dpad_threshold").value)

        self.bit_axis = int(g("bit_axis").value)
        self.bit_deadzone = float(g("bit_deadzone").value)
        self.bit_max_speed = float(g("bit_max_speed").value)
        # Stick UP (+1) -> clockwise from above -> negative about +Z.
        self.bit_sign = 1.0 if bool(g("invert_bit").value) else -1.0

        self.lid_axis = int(g("lid_axis").value)
        self.lid_deadzone = float(g("lid_deadzone").value)
        self.lid_max_speed = float(g("lid_max_speed").value)
        self.lid_sign = -1.0 if bool(g("invert_lid").value) else 1.0

        margin = float(g("limit_margin").value)
        self.motor = LimitSwitchedAxis(
            joint=str(g("motor_joint").value),
            lower=float(g("motor_lower").value),
            upper=float(g("motor_upper").value),
            speed=float(g("motor_speed").value),
            sign=-1.0 if bool(g("invert_motor").value) else 1.0,
            margin=margin,
            has_limits=bool(g("motor_has_limits").value))
        self.container = LimitSwitchedAxis(
            joint=str(g("container_joint").value),
            lower=float(g("container_lower").value),
            upper=float(g("container_upper").value),
            speed=float(g("container_speed").value),
            sign=-1.0 if bool(g("invert_container").value) else 1.0,
            margin=margin,
            has_limits=bool(g("container_has_limits").value))
        self.axes = {self.motor.joint: self.motor,
                     self.container.joint: self.container}

        self.publish_rate_hz = float(g("publish_rate_hz").value)
        self.joy_timeout_sec = float(g("joy_timeout_sec").value)
        self.stop_hold_sec = float(g("stop_hold_sec").value)

        self.modifier_held = False
        self.blocked = False
        self.was_blocked = False
        self.was_active = False
        self.stop_until = 0.0
        self.last_joy_time = 0.0
        self.command_motor = 0.0
        self.command_container = 0.0
        self.command_bit = 0.0
        self.command_lid = 0.0

        self.limits_timeout_sec = float(g("limits_timeout_sec").value)
        self.limit_bottom = False
        self.limit_top = False
        self.limits_time = 0.0
        self.limits_seen = False
        self.switch_tripped = 0

        self.pub_motor = self.create_publisher(Float64, str(g("motor_cmd_topic").value), 10)
        self.pub_container = self.create_publisher(Float64, str(g("container_cmd_topic").value), 10)
        self.pub_bit = self.create_publisher(Float64, str(g("bit_cmd_topic").value), 10)
        self.pub_lid = self.create_publisher(Float32, str(g("lid_cmd_topic").value), 10)

        self.create_subscription(Joy, joy_topic, self._joy_cb, 10)
        self.create_subscription(JointState, joint_states_topic, self._joint_state_cb, 10)
        self.create_subscription(UInt8, str(g("limits_topic").value), self._limits_cb, 10)

        self.dt = 1.0 / max(self.publish_rate_hz, 1.0)
        self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f"Drill joystick ready. Hold LT/axis {self.modifier_axis}: "
            f"d-pad up/down = feed, d-pad left/right = bin, "
            f"left stick up/down = auger, right stick up/down = sand box lid. "
            f"Released = every motor off."
        )

    # -- helpers -----------------------------------------------------------
    def _axis(self, msg, index):
        return float(msg.axes[index]) if 0 <= index < len(msg.axes) else 0.0

    def _button(self, msg, index):
        return int(msg.buttons[index]) if 0 <= index < len(msg.buttons) else 0

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

    def _report(self, axis, trip):
        if trip:
            end = "top" if trip > 0 else "bottom"
            self.get_logger().info(f"{axis.joint}: {end} limit switch, motor cut")

    # -- callbacks ---------------------------------------------------------
    def _joint_state_cb(self, msg):
        for name, position in zip(msg.name, msg.position):
            axis = self.axes.get(name)
            if axis is not None:
                axis.measure(position)

    def _limits_cb(self, msg):
        bits = int(msg.data)
        self.limit_bottom = bool(bits & 0x01)
        self.limit_top = bool(bits & 0x02)
        self.limits_time = self._now()
        if not self.limits_seen:
            self.limits_seen = True
            self.get_logger().info(
                "drill/limits is live: gating the feed on the real switches")

    def _switch_cut(self, rate):
        """Cut a feed rate that drives into a CLOSED switch. Returns (rate, trip).

        The mirror of apply_motor_commands() in the firmware, which gates the
        same way on the same two bits -- deliberately, so the operator sees the
        cut in the log at the moment the board applies it. The firmware is what
        actually protects the mechanism; this is the half the operator can see.

        Stale or absent feedback does NOT block: with no board there is no
        switch to believe, and refusing to move is its own failure. The firmware
        still holds the real gate, and a missing drill/limits is reported rather
        than silently turned into a stop.
        """
        if not self.limits_seen:
            return rate, 0
        if self._now() - self.limits_time > self.limits_timeout_sec:
            return rate, 0
        if rate > 0.0 and self.limit_top:
            return 0.0, 1
        if rate < 0.0 and self.limit_bottom:
            return 0.0, -1
        return rate, 0

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

        self.command_motor = self._detent(
            self._axis(msg, self.dpad_vertical_axis), self.dpad_threshold)
        # Canonical axis 6 is +1 LEFT, so RIGHT (the +X, park direction) is the
        # negated axis.
        self.command_container = -self._detent(
            self._axis(msg, self.dpad_horizontal_axis), self.dpad_threshold)
        self.command_bit = self._deadzone(
            self._axis(msg, self.bit_axis), self.bit_deadzone)
        self.command_lid = self._deadzone(
            self._axis(msg, self.lid_axis), self.lid_deadzone)

    # -- output ------------------------------------------------------------
    def _timer_cb(self):
        now = self._now()
        stale = (now - self.last_joy_time) > self.joy_timeout_sec
        active = self.modifier_held and not stale and not self.blocked

        if not active:
            if self.was_active:
                self.stop_until = now + self.stop_hold_sec
                reason = "LB held" if self.blocked else (
                    "joy stale" if stale else "LT released")
                self.get_logger().info(f"{reason}: drill stopped")
            self.was_active = False
            # Let the axes forget any switch they were sitting on, so the next
            # press against it logs again, and burst zeros for long enough that
            # a dropped message cannot leave a motor running.
            self.motor.tripped = 0
            self.container.tripped = 0
            self.switch_tripped = 0
            if now < self.stop_until:
                self._publish(0.0, 0.0, 0.0, 0.0)
            return

        if not self.was_active:
            self.get_logger().info("LT held: drill teleop active")
        self.was_active = True

        motor_rate, motor_trip = self.motor.rate(self.command_motor, self.dt)
        # The real switches outrank anything the axis dead-reckoned.
        motor_rate, switch_trip = self._switch_cut(motor_rate)
        if switch_trip != self.switch_tripped:
            motor_trip = switch_trip or motor_trip
        self.switch_tripped = switch_trip
        container_rate, container_trip = self.container.rate(
            self.command_container, self.dt)
        self._report(self.motor, motor_trip)
        self._report(self.container, container_trip)

        # Clamped, not just scaled. lid_max_speed is a ceiling and the
        # firmware clamps again anyway, but a value outside -1..1 arriving at a
        # servo is worth stopping at the source where the units are still
        # legible.
        lid_speed = self.lid_sign * self.command_lid * self.lid_max_speed
        lid_speed = max(-1.0, min(1.0, lid_speed))

        self._publish(motor_rate, container_rate,
                      self.bit_sign * self.command_bit * self.bit_max_speed,
                      lid_speed)

    def _publish(self, motor_rate, container_rate, bit_rate, lid_speed):
        self.pub_motor.publish(Float64(data=float(motor_rate)))
        self.pub_container.publish(Float64(data=float(container_rate)))
        self.pub_bit.publish(Float64(data=float(bit_rate)))
        self.pub_lid.publish(Float32(data=float(lid_speed)))


def main(args=None):
    rclpy.init(args=args)
    node = DrillJoystick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # gz latches the last rate it was given on all three drill axes, so a
        # drill left running when this node dies keeps running. Zero every one
        # of them on the way out, the same way the rover teleop zeroes its
        # Twist. The mechanisms hold where they stop; nothing here needs to be
        # told a position to stay at.
        #
        # The LID needs this most and can least rely on it. It is a
        # continuous-rotation servo on the real board, so a lid turning when
        # this node is killed keeps turning -- and a hard kill never reaches
        # this line at all. The firmware's own 500 ms watchdog is the backstop
        # for that case; this is the tidy exit.
        if rclpy.ok():
            node._publish(0.0, 0.0, 0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
