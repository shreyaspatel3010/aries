#!/usr/bin/env python3
"""Drive the three-colour stack light from the rover's own state.

    RED     emergency stop pressed (either switch), or halted
    YELLOW  operating
    GREEN   ready, doing nothing
    off     this node is not running

The light itself hangs off the gripper Teensy, whose firmware
(firmware/teensy_gripper/teensy_gripper.ino) subscribes to a std_msgs/UInt8 and
switches one of three GPIOs: 1 red, 2 yellow, 3 green, 4 all off. That firmware
has been ready for a while; nothing in the workspace was publishing the topic,
so the light was dark whatever the rover did. This node is that publisher, and
it is the ONLY thing that should write the topic - two publishers on a latching
GPIO means the light shows whichever message landed last.

WHAT THE STATES ARE MEASURED FROM. Only the last two can be measured today:

  emergency   Both e-stops physically disconnect ODrive power, so there is no
              contact to read and nothing publishes one. estop_topics is the
              hook for when a sense line exists; until it is filled in, this
              state can never be entered. See the note in stacklight.yaml -
              inferring it from "the ODrives went quiet" is deliberately NOT
              done here, because a CAN unplug looks identical and the light
              would cry emergency at a cable.
  halt        Same: halt_topics is empty until something declares a halt.
  operating   Wheels turning (right_rps/left_rps out of /aries_drive/status)
              or any rate topic in motion_topics non-zero - the drill's three
              axes are there by default, since a spinning auger is the most
              obviously "operating" thing on the robot.
  ready       Powered, reporting, and none of the above.

PRIORITY. States are evaluated in the order listed in `priority` and the first
one that holds wins, so an e-stop pressed while the wheels are turning shows
red rather than yellow. Anything not decided - no drive status yet, or a status
that has gone stale - is `unknown`, which is red by default: a node that cannot
see the rover must not claim it is safe.

PUBLISHING. On every change, plus a refresh every publish_period_s. The refresh
is not redundant. The firmware's stack-light subscription is RELIABLE but not
TRANSIENT_LOCAL, and it is torn down and recreated on every micro-ROS
reconnect (create_entities/destroy_entities in the sketch), which happens
whenever the agent restarts. A late-joining subscriber gets no history, so a
publisher that only spoke on change would leave the light dark - or worse,
showing a colour from before the reconnect - until the state next changed. The
refresh bounds that to publish_period_s. It also re-asserts the colour after a
Teensy reset, which starts with all three GPIOs LOW.

The light is switched off on a clean shutdown rather than left latched: off
means "the stack is not running", which is honest, where a latched green would
be a lie told by a dead node.
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Bool, Float64, String, UInt8


# The firmware's enum, and the reason this node exists at all. Keep in step
# with stacklight_color in teensy_gripper.ino: an unknown value there lights
# red, which is the right way round, but do not rely on it.
COLOR_CODES = {"red": 1, "yellow": 2, "green": 3, "off": 4}


class Stacklight(Node):
    def __init__(self):
        super().__init__("stacklight")

        # Must match the firmware's topic exactly. It is declared there without
        # a leading slash, under an empty namespace, which resolves to
        # /stacklight_subscription.
        self.declare_parameter("topic", "/stacklight_subscription")

        # State -> colour. Every state named in `priority` needs an entry, and
        # so does `unknown`.
        self.declare_parameter(
            "priority",
            ["emergency", "drive_fault", "halt", "operating", "ready"])
        self.declare_parameter("colors", [
            "emergency:red", "drive_fault:red", "halt:red",
            "operating:yellow", "ready:green", "unknown:red",
        ])
        self.declare_parameter("shutdown_color", "off")

        # Inputs. Empty lists mean "nothing publishes this yet", which is the
        # case for both e-stops and for halt - see the module docstring.
        self.declare_parameter("estop_topics", [""])
        self.declare_parameter("halt_topics", [""])
        # An e-stop is a CONDITION, not an event, so its publisher should
        # latch (TRANSIENT_LOCAL) and this must match to receive the value that
        # was published before this node started. igus_rebel's /arm/estop does
        # latch. Set false for a publisher that does not: a TRANSIENT_LOCAL
        # subscriber against a VOLATILE publisher is an incompatible pair and
        # DDS makes NO match at all -- the topic lists fine and never delivers,
        # which on an e-stop is the worst failure in this file.
        self.declare_parameter("estop_latched", True)

        # False if the switch reads true when HEALTHY, which is how a
        # normally-closed e-stop loop is usually wired.
        self.declare_parameter("estop_active_high", True)
        self.declare_parameter("halt_active_high", True)

        self.declare_parameter("drive_status_topic", "/aries_drive/status")
        self.declare_parameter("drive_status_timeout_s", 5.0)
        # /aries_drive/status reports the ramped wheel command, in rev/s. Below
        # this the wheels are standing still as far as a bystander is concerned.
        self.declare_parameter("wheel_motion_rps", 0.01)

        # Any std_msgs/Float64 rate topic. Non-zero on any of them is
        # "operating". The drill's three axes are DC motors commanded as rates
        # by aries_teleop/drill_joystick.py.
        self.declare_parameter("motion_topics", [
            "/aries/drill_motor_joint/cmd_vel",
            "/aries/drill_container_joint/cmd_vel",
            "/aries/drill_bit_joint/cmd_vel",
        ])
        # geometry_msgs/Twist command topics. Non-zero on any of them is also
        # "operating". /cmd_vel covers the simulation, where the wheels are
        # driven by gz's DiffDrive and there is no drive bridge publishing rps.
        self.declare_parameter("twist_topics", [""])
        self.declare_parameter("twist_epsilon", 1e-3)

        # Evidence that the rover is reporting at all, beyond the drive status.
        # sensor_msgs/JointState topics: any message on one is a heartbeat.
        # `ready` is only ever claimed on evidence, and in simulation the drive
        # status does not exist - /joint_states from the gz plugin is what says
        # the robot is alive there. Empty on hardware, where the drive bridge
        # IS the evidence and a joint-state stream from a rover whose ODrives
        # are unpowered would be a green light on a dead drive.
        self.declare_parameter("alive_topics", [""])

        # YELLOW IS TELEOPERATION. Any button pressed or any stick or trigger
        # off centre means a human is driving something, whatever it is - the
        # light says "a person is operating this machine", which is what a
        # bystander needs, rather than trying to enumerate every actuator.
        # Deliberately the whole pad and not the enable buttons: a stick moved
        # with no gate held still means hands on the controls.
        self.declare_parameter("joy_topic", "/joy")
        # Axes are normalised to rest at 0.0 by joy_layout_normalizer, triggers
        # included. Wide enough to ignore a worn stick's centre drift.
        self.declare_parameter("joy_axis_deadzone", 0.15)
        # How long after the last input the light stays yellow. A pad at 80 Hz
        # autorepeats, so this is really "how long after the operator lets go".
        self.declare_parameter("joy_hold_s", 2.0)

        # RED, SOURCE TWO: the drive. Any ODrive reporting an error, a bus
        # voltage on the floor, or going silent after having been present -
        # which is what pulling the drive's power looks like from here.
        self.declare_parameter("odrive_status_topics", [""])
        self.declare_parameter("odrive_min_bus_voltage", 20.0)
        self.declare_parameter("odrive_timeout_s", 3.0)

        self.declare_parameter("motion_epsilon", 1e-6)
        # A rate topic that has gone quiet is not a moving axis: the drill
        # teleop stops publishing entirely once it has sent its zeros.
        self.declare_parameter("motion_timeout_s", 1.0)

        self.declare_parameter("update_rate_hz", 10.0)
        self.declare_parameter("publish_period_s", 1.0)

        g = self.get_parameter
        self.priority = [s for s in g("priority").value if s]
        self.colors = self._parse_colors(g("colors").value)
        self.shutdown_color = str(g("shutdown_color").value)

        self.estop_active_high = bool(g("estop_active_high").value)
        self.halt_active_high = bool(g("halt_active_high").value)
        self.drive_status_timeout_s = float(g("drive_status_timeout_s").value)
        self.wheel_motion_rps = float(g("wheel_motion_rps").value)
        self.motion_epsilon = float(g("motion_epsilon").value)
        self.twist_epsilon = float(g("twist_epsilon").value)
        self.joy_axis_deadzone = float(g("joy_axis_deadzone").value)
        self.joy_hold_s = float(g("joy_hold_s").value)
        self.odrive_min_bus_voltage = float(g("odrive_min_bus_voltage").value)
        self.odrive_timeout_s = float(g("odrive_timeout_s").value)
        self.motion_timeout_s = float(g("motion_timeout_s").value)
        self.publish_period_s = float(g("publish_period_s").value)

        self._validate_colors()

        self.estop = {}
        self.halt = {}
        self.motion = {}
        self.twist = {}
        self.alive = {}
        self.joy_active_at = None
        # topic -> (fault_reason or None, last seen)
        self.odrive = {}
        self.drive_status = None
        self.drive_status_at = None

        self.pub = self.create_publisher(UInt8, str(g("topic").value), 10)

        estop_topics = [t for t in g("estop_topics").value if t]
        halt_topics = [t for t in g("halt_topics").value if t]
        estop_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=(DurabilityPolicy.TRANSIENT_LOCAL
                        if bool(g("estop_latched").value)
                        else DurabilityPolicy.VOLATILE),
        )
        for topic in estop_topics:
            self.estop[topic] = False
            self.create_subscription(
                Bool, topic,
                (lambda t: lambda m: self.estop.__setitem__(t, bool(m.data)))(topic),
                estop_qos)
        for topic in halt_topics:
            self.halt[topic] = False
            self.create_subscription(
                Bool, topic,
                (lambda t: lambda m: self.halt.__setitem__(t, bool(m.data)))(topic),
                estop_qos)
        for topic in [t for t in g("motion_topics").value if t]:
            self.motion[topic] = (0.0, None)
            self.create_subscription(
                Float64, topic,
                (lambda t: lambda m: self.motion.__setitem__(
                    t, (float(m.data), self._now())))(topic), 10)

        for topic in [t for t in g("twist_topics").value if t]:
            self.twist[topic] = (0.0, None)
            self.create_subscription(
                Twist, topic,
                (lambda t: lambda m: self.twist.__setitem__(
                    t, (max(abs(m.linear.x), abs(m.linear.y), abs(m.linear.z),
                            abs(m.angular.x), abs(m.angular.y), abs(m.angular.z)),
                        self._now())))(topic), 10)
        for topic in [t for t in g("alive_topics").value if t]:
            self.alive[topic] = None
            self.create_subscription(
                JointState, topic,
                (lambda t: lambda m: self.alive.__setitem__(t, self._now()))(topic), 10)

        joy_topic = str(g("joy_topic").value)
        if joy_topic:
            self.create_subscription(Joy, joy_topic, self._joy_cb, 10)

        odrive_topics = [t for t in g("odrive_status_topics").value if t]
        if odrive_topics:
            # Imported here, not at module scope: the drive messages are a
            # rover-only dependency and this node has to start on a base
            # station or in simulation, where odrive_can is not installed.
            from odrive_can.msg import ODriveStatus
            for topic in odrive_topics:
                self.odrive[topic] = (None, None)
                self.create_subscription(
                    ODriveStatus, topic,
                    (lambda t: lambda m: self._odrive_cb(t, m))(topic), 10)

        self.create_subscription(
            String, str(g("drive_status_topic").value), self._drive_status_cb, 10)

        self.state = None
        self.color = None
        self.last_published_at = 0.0
        period = 1.0 / max(float(g("update_rate_hz").value), 1.0)
        self.create_timer(period, self._timer_cb)

        unwired = [name for name in ("emergency", "halt")
                   if name in self.priority and not (estop_topics if name == "emergency"
                                                     else halt_topics)]
        self.get_logger().info(
            f"Stacklight ready on {self.pub.topic_name}: "
            + ", ".join(f"{s}={self.colors[s]}" for s in self.priority))
        if unwired:
            self.get_logger().info(
                f"No source wired for {', '.join(unwired)} yet - "
                "fill in estop_topics/halt_topics in stacklight.yaml when the "
                "sense lines exist. Until then that state cannot be reached.")

    # -- setup helpers -----------------------------------------------------
    @staticmethod
    def _parse_colors(entries):
        """`colors` is a list of "state:colour" strings, not a dict: ROS 2
        parameters have no nested-map type, so a dict in YAML arrives as a
        parameter per key and cannot be read back as one value."""
        parsed = {}
        for entry in entries:
            if not entry:
                continue
            state, _, color = str(entry).partition(":")
            parsed[state.strip()] = color.strip()
        return parsed

    def _validate_colors(self):
        """A state with no colour, or a colour the firmware does not know, is a
        config error worth dying on: the alternative is a light that silently
        never changes."""
        for state in list(self.priority) + ["unknown"]:
            if state not in self.colors:
                raise ValueError(
                    f"stacklight: state '{state}' has no entry in `colors`")
        for state, color in self.colors.items():
            if color not in COLOR_CODES:
                raise ValueError(
                    f"stacklight: state '{state}' maps to unknown colour "
                    f"'{color}'; known colours are {sorted(COLOR_CODES)}")
        if self.shutdown_color not in COLOR_CODES:
            raise ValueError(
                f"stacklight: shutdown_color '{self.shutdown_color}' is not a "
                f"known colour")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -- inputs ------------------------------------------------------------
    def _drive_status_cb(self, msg):
        try:
            self.drive_status = json.loads(msg.data)
            self.drive_status_at = self._now()
        except (ValueError, TypeError):
            # Malformed status is no status. Leaving the previous one in place
            # would keep reporting a state nothing is confirming any more.
            self.drive_status = None

    def _joy_cb(self, msg):
        """Any button down, or any axis off centre, is a human operating."""
        active = any(b for b in msg.buttons) or any(
            abs(a) > self.joy_axis_deadzone for a in msg.axes)
        if active:
            self.joy_active_at = self._now()

    def _odrive_cb(self, topic, msg):
        reason = None
        if msg.active_errors:
            # Hex: ODrive documents its error bits that way, and a decimal
            # 33554432 is not something anyone can look up in the field.
            reason = f"error 0x{int(msg.active_errors):X}"
        elif msg.disarm_reason:
            reason = f"disarmed 0x{int(msg.disarm_reason):X}"
        elif float(msg.bus_voltage) < self.odrive_min_bus_voltage:
            reason = f"bus {float(msg.bus_voltage):.1f} V"
        self.odrive[topic] = (reason, self._now())

    # -- state -------------------------------------------------------------
    def _pressed(self, values, active_high):
        return any(v == active_high for v in values.values())

    def _wheels_turning(self):
        if self.drive_status is None:
            return False
        try:
            return (abs(float(self.drive_status.get("right_rps", 0.0))) > self.wheel_motion_rps
                    or abs(float(self.drive_status.get("left_rps", 0.0))) > self.wheel_motion_rps)
        except (TypeError, ValueError):
            return False

    def _axis_running(self):
        now = self._now()
        for rate, stamp in self.motion.values():
            if stamp is None or (now - stamp) > self.motion_timeout_s:
                continue
            if abs(rate) > self.motion_epsilon:
                return True
        for speed, stamp in self.twist.values():
            if stamp is None or (now - stamp) > self.motion_timeout_s:
                continue
            if speed > self.twist_epsilon:
                return True
        return False

    def _drive_status_fresh(self):
        return (self.drive_status is not None
                and self.drive_status_at is not None
                and (self._now() - self.drive_status_at) <= self.drive_status_timeout_s)

    def _drive_fault(self):
        """Why the drive is faulted, or None.

        Silence counts, but only from an axis that HAS been heard from: pulling
        the ODrives' power is exactly a topic that stops, and that is the case
        worth catching. An axis that never reported at all is a rover that was
        never wired for one - simulation, or a base station - and calling that
        a fault would leave the light permanently red for no reason.
        """
        now = self._now()
        for topic, (reason, stamp) in self.odrive.items():
            if stamp is None:
                continue
            if reason is not None:
                return f"{topic.split('/')[1] if '/' in topic else topic}: {reason}"
            if (now - stamp) > self.odrive_timeout_s:
                return (f"{topic.split('/')[1] if '/' in topic else topic}: "
                        f"silent {now - stamp:.1f}s (power?)")
        return None

    def _reporting(self):
        """Evidence the rover is alive: the drive bridge, or any alive_topic."""
        if self._drive_status_fresh():
            return True
        now = self._now()
        return any(stamp is not None and (now - stamp) <= self.drive_status_timeout_s
                   for stamp in self.alive.values())

    def _holds(self, state):
        if state == "emergency":
            return self._pressed(self.estop, self.estop_active_high)
        if state == "halt":
            return self._pressed(self.halt, self.halt_active_high)
        if state == "drive_fault":
            return self._drive_fault() is not None
        if state == "operating":
            if (self.joy_active_at is not None
                    and (self._now() - self.joy_active_at) <= self.joy_hold_s):
                return True
            return self._wheels_turning() or self._axis_running()
        if state == "ready":
            # Only claim ready on evidence. Without something reporting, this
            # node cannot tell a parked rover from a dead one.
            return self._reporting()
        self.get_logger().warn(
            f"stacklight: no rule for state '{state}', skipping",
            once=True)
        return False

    def evaluate(self):
        for state in self.priority:
            if self._holds(state):
                return state
        return "unknown"

    # -- output ------------------------------------------------------------
    def _timer_cb(self):
        state = self.evaluate()
        color = self.colors[state]
        now = self._now()

        changed = color != self.color
        if changed:
            detail = ""
            if state == "drive_fault":
                detail = f" ({self._drive_fault()})"
            self.get_logger().info(
                f"{self.state or 'start'} -> {state}{detail}: {color}")
        if changed or (now - self.last_published_at) >= self.publish_period_s:
            self._publish(color)
        self.state = state
        self.color = color

    def _publish(self, color):
        self.pub.publish(UInt8(data=COLOR_CODES[color]))
        self.last_published_at = self._now()


def main(args=None):
    rclpy.init(args=args)
    node = Stacklight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # The firmware latches the last colour it was given, so a light left
        # showing green by a node that has died says the rover is ready when
        # nothing is watching it. Switch it off on the way out.
        if rclpy.ok():
            node._publish(node.shutdown_color)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
