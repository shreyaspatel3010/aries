#!/usr/bin/env python3
"""Turn the drill's rate commands into what the Teensy actually takes.

The workspace has commanded the drill in physical units since drill_joystick.py
was written - m/s for the two linear axes, rad/s for the auger - and nothing has
ever consumed them:

    /aries/drill_bit_joint/cmd_vel        rad/s   the auger
    /aries/drill_motor_joint/cmd_vel      m/s     the feed carriage
    /aries/drill_container_joint/cmd_vel  m/s     the sample bin

The board speaks duty cycle and millimetres:

    motor1/cmd_speed   std_msgs/Int32     -255..255, the auger
    motor2/cmd_speed   std_msgs/Int32     -255..255, the feed carriage
    linact/cext        std_msgs/Float32   signed millimetres, the bin

This node is the map between them, and it is a separate node rather than a
change to either side for the reason aries_load_cells gives for keeping its
calibration in YAML: `rad_per_s_at_full_pwm` and friends are properties of a
motor, a gearbox and a battery state of charge, found by running the mechanism
and watching it. In this file they are an edit and a relaunch - the workspace is
--symlink-install, so not even a rebuild. Compiled into the firmware, every one
of them is a reflash with the rover open.

WHY THE BIN IS DIFFERENT. The two motors take a duty cycle, so their mapping is
arithmetic. The bin's actuator does not: the firmware drives it for a TIME
computed from a requested distance, because the actuator has no encoder and its
only known quantity is the OEM's 15 mm/s free speed. So a rate command becomes a
stream of short timed moves - each message buys the actuator a slice of travel
slightly longer than the gap to the next message, so held-down means continuous
motion and letting go means it coasts to a stop within about one slice.

That overlap is what makes the stream safe. `linact/cext` is self-limiting in a
way `motor1/cmd_speed` is not: if this node dies mid-move the actuator stops
when the current slice expires, with no watchdog needed anywhere.

RATES ABOVE WHAT THE HARDWARE CAN DO ARE NOT AN ERROR. joystick.yaml asks the
bin for 0.05 m/s and the actuator's free speed is 0.015 m/s. The command is
clipped to full scale and the axis simply runs as fast as it runs. Nothing here
tries to make the joystick's numbers true; it reports the clip once, so the
number in joystick.yaml can be brought back to earth if anyone cares.

STOPPING. Three independent things stop this drill, and that is deliberate:

  1. drill_joystick.py sends a burst of zeros on release (stop_hold_sec).
  2. This node's own input watchdog: if a rate topic that was non-zero goes
     silent for input_timeout_s, it publishes a zero itself. That covers the
     joystick node dying mid-spin, which its own burst cannot.
  3. The firmware's MOTOR_COMMAND_TIMEOUT_MS, which covers this node dying, and
     the link dropping.

None of them is redundant: each covers a failure the others cannot see. The
firmware watchdog cannot be defeated by a timer relay here, because this node
only republishes when its input changes or when it is stopping - see the note in
_publish_motor.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, Float64, Int32


class RateToPwm:
    """One DC axis: a rate in physical units in, a signed duty cycle out.

    `full_scale` is the rate the axis reaches at 100 % duty cycle. It is NOT the
    joystick's commanded speed - joystick.yaml's motor_speed is what the
    operator asked for, this is what the motor does - though on an axis nobody
    has measured yet the two are often set the same to start with.
    """

    def __init__(self, full_scale, deadband, min_pwm, invert):
        self.full_scale = float(full_scale)
        self.deadband = float(deadband)
        self.min_pwm = int(min_pwm)
        self.sign = -1.0 if invert else 1.0

    def __call__(self, rate):
        rate = self.sign * float(rate)

        # Below the deadband is a stopped axis, not a very slow one. A duty
        # cycle of 3 % does not turn an auger, it just heats the bridge.
        if abs(rate) < self.deadband:
            return 0, False

        duty = rate / self.full_scale if self.full_scale > 0.0 else 0.0
        clipped = abs(duty) > 1.0
        duty = max(-1.0, min(1.0, duty))

        pwm = int(round(duty * 255.0))

        # Stiction floor. A command that survived the deadband is a command to
        # move, so it must not be rounded down into a duty cycle the mechanism
        # cannot break away at.
        if pwm != 0 and abs(pwm) < self.min_pwm:
            pwm = self.min_pwm if pwm > 0 else -self.min_pwm

        return max(-255, min(255, pwm)), clipped


class DrillDriver(Node):
    def __init__(self):
        super().__init__("drill_driver")

        g = self.get_parameter

        # --- inputs, in physical units --------------------------------------
        self.declare_parameter("bit_cmd_topic", "/aries/drill_bit_joint/cmd_vel")
        self.declare_parameter("motor_cmd_topic", "/aries/drill_motor_joint/cmd_vel")
        self.declare_parameter("container_cmd_topic", "/aries/drill_container_joint/cmd_vel")

        # --- outputs, the firmware's contract --------------------------------
        # Declared in main.cpp with no leading slash under an empty namespace,
        # which resolves to these.
        self.declare_parameter("auger_pwm_topic", "/motor1/cmd_speed")
        self.declare_parameter("feed_pwm_topic", "/motor2/cmd_speed")
        self.declare_parameter("container_cext_topic", "/linact/cext")

        # --- calibration -----------------------------------------------------
        # NONE OF THESE ARE MEASURED. Every full_scale below is currently the
        # matching commanded speed out of joystick.yaml, which is an assumption
        # that the axis reaches exactly what the operator asks for at 100 % duty
        # cycle. It will not be right. To calibrate: command a known rate, time
        # the axis over a known distance, and set full_scale to the rate it
        # actually reached at full PWM. The node says so at startup while they
        # are still at their defaults.
        self.declare_parameter("auger_full_scale_rad_s", 30.0)
        self.declare_parameter("auger_deadband_rad_s", 0.05)
        self.declare_parameter("auger_min_pwm", 40)
        self.declare_parameter("auger_invert", False)

        self.declare_parameter("feed_full_scale_m_s", 0.05)
        self.declare_parameter("feed_deadband_m_s", 1.0e-4)
        self.declare_parameter("feed_min_pwm", 60)
        self.declare_parameter("feed_invert", False)

        # The bin: the OEM free speed the FIRMWARE assumes, which is what makes
        # its distance-to-time conversion true. If this and
        # LinearActuator::m_oem_max_speed ever disagree, every bin move is
        # wrong by the ratio. 15 mm/s, from drill.h.
        self.declare_parameter("container_actuator_mm_s", 15.0)
        self.declare_parameter("container_deadband_m_s", 1.0e-4)
        self.declare_parameter("container_invert", False)
        # Each slice of travel covers this many command periods. Above 1.0 the
        # slices overlap, so a held command is continuous motion rather than a
        # stutter of on/off; the cost is that release coasts for the remainder
        # of the last slice. 1.5 is ~50 ms at 30 Hz.
        self.declare_parameter("container_slice_overlap", 1.5)

        # --- timing ----------------------------------------------------------
        self.declare_parameter("publish_rate_hz", 30.0)
        # A rate topic that was non-zero and has gone quiet for this long is a
        # publisher that died, not an axis at rest: drill_joystick.py always
        # sends its zeros before going silent. Generous against 30 Hz.
        self.declare_parameter("input_timeout_s", 0.4)

        rate_hz = float(g("publish_rate_hz").value)
        self.period = 1.0 / rate_hz if rate_hz > 0.0 else 1.0 / 30.0
        self.input_timeout_s = float(g("input_timeout_s").value)

        self.auger = RateToPwm(
            g("auger_full_scale_rad_s").value, g("auger_deadband_rad_s").value,
            g("auger_min_pwm").value, g("auger_invert").value)
        self.feed = RateToPwm(
            g("feed_full_scale_m_s").value, g("feed_deadband_m_s").value,
            g("feed_min_pwm").value, g("feed_invert").value)

        self.container_mm_s = float(g("container_actuator_mm_s").value)
        self.container_deadband = float(g("container_deadband_m_s").value)
        self.container_sign = -1.0 if g("container_invert").value else 1.0
        self.container_slice_s = self.period * float(g("container_slice_overlap").value)

        # BEST EFFORT on the two motor topics, matching the firmware's
        # subscriptions. They are a continuous stream where the newest value
        # supersedes the last, and a reliable stream that stalls costs the whole
        # serial link - see the QoS note in main.cpp's create_entities().
        #
        # A RELIABLE publisher would still MATCH a best-effort subscriber, so
        # this is about the stream, not about compatibility. linact/cext is
        # reliable on both ends because it is an event, not a sample.
        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.pub_auger = self.create_publisher(
            Int32, str(g("auger_pwm_topic").value), best_effort)
        self.pub_feed = self.create_publisher(
            Int32, str(g("feed_pwm_topic").value), best_effort)
        self.pub_container = self.create_publisher(
            Float32, str(g("container_cext_topic").value), 10)

        self.create_subscription(
            Float64, str(g("bit_cmd_topic").value), self._bit_cb, 10)
        self.create_subscription(
            Float64, str(g("motor_cmd_topic").value), self._motor_cb, 10)
        self.create_subscription(
            Float64, str(g("container_cmd_topic").value), self._container_cb, 10)

        now = self._now()
        self.bit_rate = 0.0
        self.motor_rate = 0.0
        self.container_rate = 0.0
        self.bit_stamp = now
        self.motor_stamp = now
        self.container_stamp = now

        # Last value actually put on the wire, so a steady command is not
        # restated 30 times a second. See _publish_motor.
        self.last_auger_pwm = None
        self.last_feed_pwm = None
        self._container_moving = False

        self._clip_warned = {"auger": False, "feed": False, "container": False}

        self.create_timer(self.period, self._tick)

        if self.container_mm_s <= 0.0:
            self.get_logger().error(
                "container_actuator_mm_s is %.3f - the bin cannot be commanded. "
                "It must match LinearActuator::m_oem_max_speed in the firmware "
                "(15 mm/s), which is what makes the firmware's "
                "distance-to-time conversion true."
                % self.container_mm_s)

        self.get_logger().info(
            "drill_driver: rates -> PWM. auger %.3g rad/s, feed %.3g m/s at full scale; "
            "bin actuator %.3g mm/s in %.0f ms slices. "
            "NONE OF THESE ARE MEASURED - see the calibration note in "
            "config/drill_driver.yaml."
            % (self.auger.full_scale, self.feed.full_scale,
               self.container_mm_s, self.container_slice_s * 1000.0))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _bit_cb(self, msg):
        self.bit_rate = float(msg.data)
        self.bit_stamp = self._now()

    def _motor_cb(self, msg):
        self.motor_rate = float(msg.data)
        self.motor_stamp = self._now()

    def _container_cb(self, msg):
        self.container_rate = float(msg.data)
        self.container_stamp = self._now()

    def _stale(self, stamp, now):
        return (now - stamp) > self.input_timeout_s

    def _warn_clip(self, key, commanded, full_scale, unit):
        if self._clip_warned[key]:
            return
        self._clip_warned[key] = True
        self.get_logger().warning(
            "%s commanded %.4g %s but full scale is %.4g %s - clipped, and the "
            "axis will simply run at its own top speed. Not an error: lower the "
            "matching speed in joystick.yaml if the mismatch matters."
            % (key, abs(commanded), unit, full_scale, unit))

    def _publish_motor(self, pub, pwm, last_attr):
        """Send every cycle while moving; go quiet once stopped.

        A HELD COMMAND IS RESTATED at the full rate, deliberately. The firmware
        stops its motors if nothing arrives for MOTOR_COMMAND_TIMEOUT_MS, so a
        motor that is supposed to keep turning has to keep being asked.

        That is only safe because the stream stops when the operator does. What
        defeated the rover drive bridge's own timeout was a node that replayed
        its last command on a TIMER, independently of whether anything was still
        arriving - so the watchdog downstream never fired. Here the value being
        restated is refreshed from a live input every cycle, and _tick has
        already replaced a stale input with zero before this is called. Let go
        and the zeros stop within input_timeout_s.

        Zero is the one value worth going quiet on: the firmware is already
        stopped, and its watchdog keeps it that way with no help from here.
        """
        if getattr(self, last_attr) == pwm and pwm == 0:
            return
        pub.publish(Int32(data=int(pwm)))
        setattr(self, last_attr, pwm)

    def _tick(self):
        now = self._now()

        bit = 0.0 if self._stale(self.bit_stamp, now) else self.bit_rate
        motor = 0.0 if self._stale(self.motor_stamp, now) else self.motor_rate
        container = 0.0 if self._stale(self.container_stamp, now) else self.container_rate

        auger_pwm, auger_clipped = self.auger(bit)
        feed_pwm, feed_clipped = self.feed(motor)

        if auger_clipped:
            self._warn_clip("auger", bit, self.auger.full_scale, "rad/s")
        if feed_clipped:
            self._warn_clip("feed", motor, self.feed.full_scale, "m/s")

        self._publish_motor(self.pub_auger, auger_pwm, "last_auger_pwm")
        self._publish_motor(self.pub_feed, feed_pwm, "last_feed_pwm")

        self._publish_container(container)

    def _publish_container(self, rate_m_s):
        """A rate becomes the next slice of travel, in signed millimetres.

        Silence here is a stop, and it needs no zero message: the firmware's
        current slice expires on its own. A 0.0 IS sent on the transition to
        stopped, because it reaches LinearActuator::stop_motor() and cuts the
        remainder of the slice rather than letting it run out.
        """
        rate = self.container_sign * rate_m_s

        if abs(rate) < self.container_deadband:
            # Only on the transition. A repeated 0.0 would restart nothing, and
            # the actuator is already stopping on its own as the slice expires;
            # the one 0.0 is what cuts the remainder of it short.
            if self._container_moving:
                self._container_moving = False
                self.pub_container.publish(Float32(data=0.0))
            return

        # How far the actuator can actually travel in one slice. The commanded
        # rate does not appear here beyond its SIGN, and that is the honest
        # thing: the actuator has one speed and no way to be asked for another.
        # Asking for 0.05 m/s from a 0.015 m/s actuator cannot be granted, so a
        # slice is always a full-speed slice and the only question is how long.
        mm = self.container_mm_s * self.container_slice_s
        if rate < 0.0:
            mm = -mm

        if abs(rate_m_s) > (self.container_mm_s * 1e-3):
            self._warn_clip("container", rate_m_s, self.container_mm_s * 1e-3, "m/s")

        self._container_moving = True
        self.pub_container.publish(Float32(data=float(mm)))

    def stop(self):
        """Everything off, on the way out."""
        try:
            self.pub_auger.publish(Int32(data=0))
            self.pub_feed.publish(Int32(data=0))
            self.pub_container.publish(Float32(data=0.0))
        except Exception:
            pass


def main():
    rclpy.init()
    node = DrillDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
