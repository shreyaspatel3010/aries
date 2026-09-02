#!/usr/bin/env python3
"""Live ST3215 gripper telemetry as an RViz text overlay.

Draws, floating just above the gripper and following the arm:

    ST3215 GRIPPER  OK / DANGER / CUTOFF
    gap  42.1 mm  (cmd 40.0)
    joint  -2.031 rad  -116.4 deg  step 2016
    tcp  0.412 0.031 0.287 m  (base_link)
    load  18 %  0.53 Nm
    current  210 mA  (cutoff 1500)
    voltage  11.9 V  (11.0 - 14.0)
    temp  38 C  (cutoff 70)
    status  0x00 none

The numbers come from the hardware component, which is the only thing that can
read them: it owns the serial port, so nothing else can ask the servo. It
publishes a diagnostic_msgs/DiagnosticArray on /diagnostics and this node turns
one entry of that into markers. Adding a field means adding a key there, not a
parser here - anything unrecognised is simply not drawn.

WHY A SEPARATE NODE AND NOT MARKERS FROM THE C++
The layout is the part that gets fiddled with (which fields, what order, how
big, where it sits), and here that is a parameter edit and a restart rather
than a rebuild of a ros2_control plugin that the arm depends on. /diagnostics
is also useful on its own - rqt_robot_monitor, a checker, a flight log - and
this way it exists whether or not anybody is running RViz.

WHY TEXT MARKERS AND NOT AN OVERLAY PLUGIN
A screen-fixed HUD would need rviz_2d_overlay_plugins, which is not installed
on the rover and is not in Jazzy's default set. A TEXT_VIEW_FACING marker is
stock rviz_default_plugins, always billboards toward the camera, and has the
advantage of sitting on the gripper it describes.

Add the topic (default: gripper_status_markers) as a MarkerArray display. It is
already in aries_moveit/config/gripper.rviz and launch/moveit.rviz.

WHAT "DANGER" AND "CUTOFF" MEAN
Both are decided in the hardware component against the servo's OWN EPROM
protection limits, not against constants chosen here - see the header of
st3215_gripper_system.hpp. This node only colours what it is told:

    OK      green   nothing near a limit
    DANGER  amber   within a margin of a limit the servo will trip on
    CUTOFF  red     already tripped, commands inhibited, or the servo has
                    stopped answering
    STALE   grey    no diagnostics at all for stale_after_s - the gripper
                    hardware is not running, or this is mock/sim
"""

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

import tf2_ros

from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# DiagnosticStatus.level -> (label, colour). Colours are picked to read against
# both the dark and the light RViz backgrounds; pure red on black is hard to
# read, so the CUTOFF red is lifted toward pink.
LEVEL_STYLE = {
    DiagnosticStatus.OK: ("OK", (0.40, 0.95, 0.45)),
    DiagnosticStatus.WARN: ("DANGER", (1.00, 0.72, 0.15)),
    DiagnosticStatus.ERROR: ("CUTOFF", (1.00, 0.35, 0.35)),
    DiagnosticStatus.STALE: ("STALE", (0.65, 0.65, 0.65)),
}
STALE_STYLE = ("NO TELEMETRY", (0.60, 0.60, 0.60))


class GripperStatusOverlay(Node):
    def __init__(self):
        super().__init__("gripper_status_overlay")

        # WHERE THE PANEL HANGS.
        #
        # base_link, the ROVER body, since 2026-08-31. It was
        # arm_gripper_base_link, which made the text ride along with the jaws --
        # readable when working close in, but it swung across the view with
        # every arm move and, at some poses, ended up behind the robot or
        # inside a mesh.
        #
        # Anchoring on the rover also costs fewer TF hops. RViz has to
        # transform this frame into its fixed frame to draw the marker, and on
        # this stack the fixed frame usually IS base_link -- in which case the
        # transform is the identity and cannot fail at all. From the gripper it
        # was a lookup down the whole arm chain, refreshed only as fast as
        # robot_state_publisher republishes /joint_states. See the stamp note
        # in publish_markers() for the other half of that problem.
        #
        # THE OFFSET IS A VIEWING PREFERENCE, not geometry. 0.16 m was chosen to
        # clear the jaws from the gripper frame; on base_link that height is
        # inside the chassis, so the text is lifted to float above the deck
        # instead. Both are parameters -- move it wherever it reads best.
        # 0.026 m cap height, up from 0.016. RViz draws this in WORLD units, so
        # the panel is sized against the rover next to it rather than against
        # the window: at 0.016 it was legible only zoomed right in, which is
        # not how anyone drives. 0.026 is roughly one line per 40 mm of rover
        # and stays readable at the whole-robot view the operator actually
        # uses. The offset went up with it -- a taller block hung at 0.60 m
        # reached down into the chassis and the arm, and text that is half
        # behind a mesh is what reads as "scattered": the panel is depth
        # tested, so occluded lines simply are not drawn.
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("offset_xyz", [0.0, 0.0, 0.85])
        self.declare_parameter("text_height_m", 0.026)
        self.declare_parameter("marker_topic", "gripper_status_markers")
        self.declare_parameter("publish_rate_hz", 5.0)
        # Substring matched against DiagnosticStatus.name. The hardware
        # component publishes "<component>: ST3215 servo", and the component
        # name comes from the URDF, so match on the stable half.
        self.declare_parameter("diagnostic_match", "ST3215 servo")
        # Longer than one dropped sample, short enough that unplugging the
        # servo shows up while the operator is still looking at it.
        self.declare_parameter("stale_after_s", 2.0)
        # TCP pose in the arm's own base frame - the "where is the gripper"
        # half of the question, which /diagnostics cannot answer because the
        # servo knows nothing about the arm it is bolted to.
        self.declare_parameter("show_tcp_pose", True)
        self.declare_parameter("tcp_frame", "gripper_tcp")
        self.declare_parameter("reference_frame", "base_link")

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.offset = [float(v) for v in self.get_parameter("offset_xyz").value]
        self.text_height = float(self.get_parameter("text_height_m").value)
        self.match = str(self.get_parameter("diagnostic_match").value)
        self.stale_after = float(self.get_parameter("stale_after_s").value)
        self.show_tcp = bool(self.get_parameter("show_tcp_pose").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.reference_frame = str(self.get_parameter("reference_frame").value)

        self.status = None
        self.status_stamp = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 1)
        # Default (reliable) QoS, matching the hardware component's publisher.
        # A best-effort subscription would not match a reliable publisher at
        # all, and the panel would stay blank with the topic visibly alive in
        # `ros2 topic hz` - so leave this alone.
        self.create_subscription(DiagnosticArray, "/diagnostics", self.on_diagnostics, 5)

        rate = max(float(self.get_parameter("publish_rate_hz").value), 0.5)
        self.create_timer(1.0 / rate, self.publish_markers)

        self.get_logger().info(
            f"Gripper status overlay up: frame={self.frame_id} "
            f"matching /diagnostics entries containing '{self.match}'")

    # -- input ---------------------------------------------------------
    def on_diagnostics(self, msg):
        """Keep the first entry whose name matches. /diagnostics is shared, so
        everything else on it - and there may be a lot - is ignored."""
        for status in msg.status:
            if self.match in status.name:
                self.status = status
                self.status_stamp = self.get_clock().now()
                return

    def value(self, key, default=None):
        for entry in self.status.values:
            if entry.key == key:
                return entry.value
        return default

    def tcp_pose(self):
        """TCP translation in the reference frame, or None if TF cannot say."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.reference_frame, self.tcp_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.0))
        except tf2_ros.TransformException:
            return None
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    # -- layout --------------------------------------------------------
    def lines(self):
        """(text, colour) for the panel, from whatever keys are present."""
        if self.status is None or self.status_stamp is None:
            return STALE_STYLE[0], STALE_STYLE[1], [
                "ST3215 GRIPPER  NO TELEMETRY",
                "",
                "nothing is publishing gripper diagnostics.",
                "expected on mock or sim hardware; on the rover it",
                "means the ros2_control gripper component is not up.",
            ]

        age = (self.get_clock().now() - self.status_stamp).nanoseconds * 1e-9
        if age > self.stale_after:
            label, colour = STALE_STYLE
            return label, colour, [
                f"ST3215 GRIPPER  STALE {age:.0f}s",
                "",
                "the gripper component stopped publishing.",
                "last message:",
                f"  {self.status.message[:60]}",
            ]

        label, colour = LEVEL_STYLE.get(self.status.level, STALE_STYLE)
        out = [f"ST3215 GRIPPER  {label}"]

        def row(label, value, unit, extra=""):
            # NOT padded into columns any more. The columns were built with
            # spaces ("{label:<8}{value:>7}"), and RViz's font is proportional:
            # a space is far narrower than a digit, so "gap" + 5 spaces and
            # "current" + 1 space do not end at the same x. Every value started
            # at its own offset and the panel read as loose fragments rather
            # than a block -- the alignment was costing exactly what it was
            # supposed to buy.
            #
            # Two spaces between fields instead, so the eye groups each row as
            # one "label value unit" unit. Ragged on purpose beats ragged by
            # accident, and the only real fix -- a monospaced font -- is not
            # something a marker can ask for.
            out.append("  ".join(p for p in (label, f"{value} {unit}".strip(), extra) if p))

        gap = self.value("gap_mm")
        cmd_gap = self.value("command_gap_mm")
        if gap is not None:
            row("gap", gap, "mm", f"(cmd {cmd_gap})" if cmd_gap is not None else "")

        q = self.value("position_rad")
        if q is not None:
            row("joint", q, "rad",
                f"{self.value('position_deg')} deg  step {self.value('position_steps')}")

        if self.show_tcp:
            tcp = self.tcp_pose()
            if tcp is not None:
                row("tcp", f"{tcp[0]:.3f} {tcp[1]:.3f} {tcp[2]:.3f}", "m",
                    f"({self.reference_frame})")

        load = self.value("load_percent")
        if load is not None:
            row("load", load, "%", f"{self.value('effort_nm')} Nm")

        # Each measurement carries the limit it will be cut off at, so the
        # number can be judged without knowing the servo's configuration.
        row("current", self.value("current_ma", "?"), "mA",
            f"(cutoff {self.value('limit_current_ma', '?')})")
        row("voltage", self.value("voltage_v", "?"), "V",
            f"({self.value('limit_voltage_min_v', '?')} - "
            f"{self.value('limit_voltage_max_v', '?')})")
        row("temp", self.value("temperature_c", "?"), "C",
            f"(cutoff {self.value('limit_temperature_c', '?')})")

        status_byte = self.value("status_byte")
        if status_byte is not None:
            row("status", status_byte, "", self.value("status_flags", ""))

        if self.value("squeeze_relax_active") == "true":
            out.append("holding with squeeze-relax (goal backed off)")
        if self.value("comms") == "no reply":
            out.append(
                f"NO REPLY from the servo ({self.value('read_failures')} failed reads)")

        # The reason, last and on its own lines: it is a sentence, not a field,
        # and it is the only thing that matters when it is there.
        if self.status.level != DiagnosticStatus.OK and self.status.message:
            out.append("")
            out.extend(wrap(self.status.message, 46))

        return label, colour, out

    # -- output --------------------------------------------------------
    def publish_markers(self):
        label, colour, lines = self.lines()

        text = Marker()
        text.header.frame_id = self.frame_id
        # STAMP ZERO ON PURPOSE -- "the latest transform you have", not "the
        # transform at this instant".
        #
        # This was self.get_clock().now(). RViz has to transform frame_id into
        # the fixed frame AT THE MARKER'S STAMP, and the newest TF for this
        # chain is only as fresh as robot_state_publisher's last publish from
        # /joint_states. A marker stamped now() is therefore routinely a few
        # milliseconds AHEAD of the newest transform, and tf2 refuses to
        # extrapolate into the future -- so RViz drops the marker and logs a
        # transform error. It comes and goes rather than failing outright
        # because it is a race between this timer and that one, which is
        # exactly what an intermittent TF error on a marker looks like.
        #
        # A zero stamp is the standard answer for a label pinned to a link: the
        # text is telemetry, nothing here needs interpolating to an exact
        # instant, and being one TF cycle stale is invisible.
        text.header.stamp = Time()
        text.ns = "gripper_status"
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = self.offset[0]
        text.pose.position.y = self.offset[1]
        text.pose.position.z = self.offset[2]
        text.pose.orientation.w = 1.0
        # Re-transform every render from the latest TF instead of freezing the
        # marker where it was when it arrived. Keeps the text glued to the
        # gripper while the arm moves, and means a stale stamp can never strand
        # it in mid-air.
        text.frame_locked = True
        # Only z is read for TEXT_VIEW_FACING: it is the cap height of one line.
        text.scale.z = self.text_height
        text.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)
        # RViz's MovableText lays out '\n' in SCREEN space, so the block stays
        # readable from any viewpoint. Stacking one marker per line in world
        # space would collapse into a single line when viewed from above.
        text.text = "\n".join(lines)

        arr = MarkerArray()
        arr.markers.append(text)
        self.marker_pub.publish(arr)


def wrap(text, width):
    """Break a sentence onto lines of at most `width` characters.

    Not textwrap.wrap: the inhibit reasons carry long unbroken paths and
    shell commands that textwrap would either mangle or refuse to split,
    and a line running off the side of the panel is better than a lost one.
    """
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


def main():
    rclpy.init()
    node = GripperStatusOverlay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
