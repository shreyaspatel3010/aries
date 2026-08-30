#!/usr/bin/env python3
"""Split the science board's telemetry array into named topics.

THE PROBLEM THIS SOLVES. firmware/teensy_science_sys publishes one
std_msgs/Float32MultiArray of ten floats on /science/telemetry and nothing
else - no labels, no units, no per-field validity. The only thing tying index 3
to "ORP in millivolts" is that the firmware's TelemetryIndex enum and this
package's science.yaml happen to be in the same order. Swap two entries in
either and the rover reports the pH probe as soil moisture, both numbers stay
entirely plausible, and nothing anywhere says a word.

So this node does two things: it republishes each index under a name, and it
makes the order a checkable artefact rather than a comment. test_science_-
telemetry.py reads the firmware's enum directly and fails if the two drift.

WHAT IT DOES NOT DO. It converts nothing and calibrates nothing. Every value
here is exactly what the board sent - pH is pH, ORP is millivolts, gas
resistance is ohms. The calibration for these sensors lives on the board (see
protocols.md), unlike the load cells, because most of it is a probe-specific
voltage curve rather than a scale factor, and two of them are calibrated with a
physical potentiometer.

NaN IS A REAL ANSWER AND IS PASSED THROUGH. The board sends NaN for a sensor
that has never been read, whose init failed, or whose read did not land. That
is different from the board being absent, and this node preserves the
difference: NaN on a topic that is still updating means "no reading"; a topic
that has stopped updating, plus a `stale` status, means "no board". Dropping
NaN instead would collapse the two into one silence.

PULL, NOT STREAM. Nothing on the board is sampled unless it is asked for, so
these topics carry the LAST value each sensor was commanded to produce, not a
live measurement. The board republishes them every second regardless. To take a
fresh reading, send the sensor's command - and because remembering that
`ros2 topic pub ... "{data: 32}"` means "read the ORP" is its own kind of trap,
this node offers the same thing by name on /science/read.
"""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String, UInt8


class ScienceTelemetry(Node):
    def __init__(self):
        super().__init__("science_telemetry")

        self.declare_parameter("telemetry_topic", "/science/telemetry")
        self.declare_parameter("sensor_cmd_topic", "/science/sensor_cmd")
        self.declare_parameter("ns", "/science")
        self.declare_parameter("timeout_s", 3.0)
        self.declare_parameter("status_rate_hz", 1.0)
        self.declare_parameter("publish_nan", True)
        self.declare_parameter(
            "fields",
            ["ph", "soil_moisture", "tds", "orp", "soil_temp",
             "air_temp", "humidity", "pressure", "gas_resistance", "co2"])

        g = self.get_parameter
        self.ns = str(g("ns").value).rstrip("/")
        self.timeout_s = float(g("timeout_s").value)
        self.publish_nan = bool(g("publish_nan").value)

        self.fields = [str(n) for n in g("fields").value]
        if not self.fields:
            raise ValueError("aries_science: `fields` is empty - there is "
                             "nothing to name the telemetry array with.")

        # Per-field unit and command id. Declared per name because ROS
        # parameters are scalars and arrays of scalars; a list of dicts in the
        # YAML would silently fail to load.
        self.units = {}
        self.cmds = {}
        for name in self.fields:
            self.declare_parameter(f"field.{name}.unit", "")
            self.declare_parameter(f"field.{name}.cmd", -1)
            self.units[name] = str(g(f"field.{name}.unit").value)
            self.cmds[name] = int(g(f"field.{name}.cmd").value)

        # name -> publisher, and the last value seen for each.
        self.pubs = {n: self.create_publisher(Float32, f"{self.ns}/{n}", 10)
                     for n in self.fields}
        self.latest = {n: float("nan") for n in self.fields}

        self.status_pub = self.create_publisher(String, f"{self.ns}/status", 10)
        self.cmd_pub = self.create_publisher(
            UInt8, str(g("sensor_cmd_topic").value), 10)

        self.create_subscription(
            Float32MultiArray, str(g("telemetry_topic").value),
            self._telemetry_cb, 10)

        # READ BY NAME. `ros2 topic pub --once /science/read std_msgs/String
        # "{data: orp}"` sends the board 32. `<name>:init` sends 31 instead.
        # The numeric protocol still works and is unchanged; this only removes
        # the need to remember it.
        self.create_subscription(String, f"{self.ns}/read", self._read_cb, 10)

        self.last_msg_time = None
        self.msg_count = 0
        self.bad_length_reported = False

        self.create_timer(1.0 / max(float(g("status_rate_hz").value), 0.1),
                          self._status_cb)

        self.get_logger().info(
            f"Science telemetry up: {len(self.fields)} fields on "
            f"{self.ns}/<name>, from {g('telemetry_topic').value}. "
            f"Read one with: ros2 topic pub --once {self.ns}/read "
            f"std_msgs/String \"{{data: ph}}\"")

    # -- helpers -----------------------------------------------------------
    def _owner_of(self, name):
        """The field whose read command fills `name`, or None.

        Walks BACKWARDS from `name` to the nearest field carrying a real
        command id. That works because the array is grouped by sensor: a
        multi-value sensor puts its commanded field first and its extra values
        immediately after.
        """
        for prior in reversed(self.fields[:self.fields.index(name)]):
            if self.cmds[prior] >= 0:
                return prior
        return None

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -- callbacks ---------------------------------------------------------
    def _telemetry_cb(self, msg):
        data = list(msg.data)

        # LENGTH IS THE ONLY INTEGRITY CHECK AVAILABLE. The array carries no
        # names and no version, so a firmware that grew an eleventh field would
        # otherwise be read silently against the old table - every index past
        # the change reporting the wrong sensor.
        #
        # Reported ONCE rather than every second: at 1 Hz a mismatch that
        # logged every message would bury everything else in the terminal, and
        # it is a build-time fact, not an event.
        if len(data) != len(self.fields):
            if not self.bad_length_reported:
                self.bad_length_reported = True
                self.get_logger().error(
                    f"/science/telemetry carried {len(data)} values but "
                    f"science.yaml names {len(self.fields)}. The firmware and "
                    f"this package have drifted apart - NOT republishing, "
                    f"because every index past the difference would be the "
                    f"wrong sensor. Compare TelemetryIndex in "
                    f"firmware/teensy_science_sys/src/main.cpp against "
                    f"`fields` in science.yaml.")
            return

        self.bad_length_reported = False
        self.last_msg_time = self._now()
        self.msg_count += 1

        for name, value in zip(self.fields, data):
            value = float(value)
            self.latest[name] = value
            if math.isnan(value) and not self.publish_nan:
                continue
            self.pubs[name].publish(Float32(data=value))

    def _read_cb(self, msg):
        """Read (or initialise) one sensor by name.

        `orp` reads it; `orp:init` initialises it. Init has to be explicit and
        separate because it is not free - the SCD41's blocks the board for
        500 ms inside its library - and because re-initialising a sensor
        mid-task discards whatever state it had built up.
        """
        text = str(msg.data).strip()
        if not text:
            return

        name, _, action = text.partition(":")
        name = name.strip()
        action = (action.strip() or "read").lower()

        if name not in self.cmds:
            self.get_logger().warn(
                f"{self.ns}/read: no field called '{name}'. "
                f"Known: {', '.join(self.fields)}")
            return

        sensor_id = self.cmds[name]
        if sensor_id < 0:
            # A field with no command of its own is filled as a side effect of
            # an earlier one - the BME688 answers a single read by populating
            # air_temp, humidity, pressure and gas_resistance together. The
            # owner is the nearest PRECEDING field that does have a command,
            # which follows from the array being grouped by sensor. If that
            # ever stops being true, this is the assumption that breaks.
            owner = self._owner_of(name)
            if owner is None:
                self.get_logger().warn(
                    f"{self.ns}/read: '{name}' has no command and no field "
                    f"before it does either - nothing to send.")
                return
            self.get_logger().info(
                f"{self.ns}/read: '{name}' is filled by reading '{owner}'. "
                f"Sending that instead.")
            sensor_id = self.cmds[owner]

        if action not in ("read", "init"):
            self.get_logger().warn(
                f"{self.ns}/read: unknown action '{action}' - "
                f"use '<name>' or '<name>:init'.")
            return

        code = sensor_id * 10 + (1 if action == "init" else 2)
        self.cmd_pub.publish(UInt8(data=code))
        self.get_logger().info(f"{self.ns}/read: {name} {action} -> sent {code:02d}")

    def _status_cb(self):
        """One line describing the board, published whether or not it is there.

        The absence of telemetry is the thing most worth reporting and is the
        one thing a telemetry topic cannot say, so it is said here.
        """
        if self.last_msg_time is None:
            self.status_pub.publish(String(
                data="no telemetry yet - is the science board flashed and its "
                     "micro-ROS agent running?"))
            return

        age = self._now() - self.last_msg_time
        if age > self.timeout_s:
            self.status_pub.publish(String(
                data=f"STALE: no telemetry for {age:.1f}s "
                     f"(board or agent gone; last values are {age:.0f}s old)"))
            return

        read = [n for n in self.fields if not math.isnan(self.latest[n])]
        unread = [n for n in self.fields if math.isnan(self.latest[n])]

        parts = [f"{n}={self.latest[n]:.2f}{self.units[n]}" for n in read]
        summary = " ".join(parts) if parts else "nothing read yet"
        if unread:
            summary += f" | no reading: {', '.join(unread)}"
        self.status_pub.publish(String(data=summary))


def main(args=None):
    rclpy.init(args=args)
    node = ScienceTelemetry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        # Ctrl-C through launch, or any external rclpy.shutdown(). rclpy.spin()
        # raises this rather than returning, and an uncaught one exits 1 -- so
        # every ordinary stop of the stack prints "process has died ... exit
        # code 1" for a node that did exactly what it was told. In a launch
        # this size that is the kind of line people learn to scroll past, which
        # is precisely how a real death goes unnoticed.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
