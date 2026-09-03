#!/usr/bin/env python3
"""Live ST3215 gripper telemetry as an RViz text overlay.

Draws, floating just above the gripper and following the arm:

    ST3215 GRIPPER OK / DANGER / CUTOFF
    gap 42.1 mm cmd 40.0
    joint -2.031 rad -116.4 deg
    step 2016 load 18 % 0.53 Nm
    tcp 0.412 0.031 0.287 m
    current 210 mA max 1500
    voltage 11.9 V 11.0-14.0
    temp 38 C max 70
    status 0x00 none

That is the MARKER layout: rows grouped to come out roughly the same width and
every gap a single space, because of how RViz measures marker text -- see the
block comment above panel(), which has the numbers. The HUD draws the same
fields as an aligned table instead. The TCP reference frame is not drawn: it is a
parameter, so it is logged once at startup instead of costing nine characters
of every row.

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

TWO PANELS, ONE SET OF FIELDS
Both are published every tick from the same Field list, so a key added to the
hardware component appears in both:

  /gripper_status_hud      rviz_2d_overlay_msgs/OverlayText -- a screen-fixed
                           corner panel drawn by rviz_2d_overlay_plugins with
                           a real monospaced font, so the columns line up and
                           nothing occludes it. This is the one to read.

  /gripper_status_markers  the TEXT_VIEW_FACING marker above, stock
                           rviz_default_plugins, hanging over the rover. It
                           stays because it needs nothing installed, and
                           because it is right there in the 3D view.

The HUD needs two apt packages, both in the ROS repo and both pulled in by the
repo's usual `rosdep install --from-paths src -r -y`:

  ros-jazzy-rviz-2d-overlay-msgs      the message. On the ROVER, which
                                      publishes it.
  ros-jazzy-rviz-2d-overlay-plugins   the display. On whatever runs RVIZ,
                                      which in the field is the base station.

Without the msgs package the node still runs and publishes the marker alone,
and says so once at startup -- the import is guarded on purpose. Without the
plugins package RViz will complain about an unknown display class for the
GripperHUD entry in the .rviz configs and carry on.

The HUD exists because the marker has a floor it cannot get under: MovableText
letter-spaces at 100% and line-spaces at 200% and no message field changes
either. See the block comment above panel() for the measurements.

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

from collections import namedtuple

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

import tf2_ros

from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# The HUD half is optional AT RUNTIME, not just at build: a machine that has
# not installed the overlay packages still gets the marker, which is the whole
# point of keeping both. Import failure is expected, not an error.
try:
    from rviz_2d_overlay_msgs.msg import OverlayText
except ImportError:
    OverlayText = None

# One row of the panel, rendered differently by each of the two layouts.
# `extra` is whatever qualifies the value -- a limit, a command, a second unit.
Field = namedtuple("Field", "key label value extra")

# How the marker packs those rows onto lines. Each tuple is one line, and the
# grouping exists to even out line widths -- see the long comment above panel().
MARKER_GROUPS = (
    ("gap",),
    ("joint",),
    ("step", "load"),
    ("tcp",),
    ("current",),
    ("voltage",),
    ("temp",),
    ("status",),
)

# HUD column widths, in characters of a MONOSPACED font. Only valid because
# the HUD picks its own font; the marker cannot use these.
HUD_LABEL_W = 8
HUD_VALUE_W = 12

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

        # THE HUD. A screen-fixed, monospaced panel drawn by
        # rviz_2d_overlay_plugins -- the thing the marker cannot be. Sized and
        # placed in PIXELS, not metres, so it does not move when the camera
        # does and does not have to clear the robot.
        self.declare_parameter("hud_topic", "gripper_status_hud")
        self.declare_parameter("hud_font", "Liberation Mono")
        self.declare_parameter("hud_text_size", 12.0)
        # Region the plugin paints. Text is clipped to it, so if a longer
        # /diagnostics message ever runs off the bottom, raise the height.
        self.declare_parameter("hud_size", [320, 300])
        self.declare_parameter("hud_margin", [10, 10])
        self.declare_parameter("hud_anchor", "top_left")
        self.declare_parameter("hud_bg_alpha", 0.55)

        self.hud_font = str(self.get_parameter("hud_font").value)
        self.hud_text_size = float(self.get_parameter("hud_text_size").value)
        self.hud_size = [int(v) for v in self.get_parameter("hud_size").value]
        self.hud_margin = [int(v) for v in self.get_parameter("hud_margin").value]
        self.hud_bg_alpha = float(self.get_parameter("hud_bg_alpha").value)

        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 1)

        self.hud_pub = None
        if OverlayText is not None:
            anchor = str(self.get_parameter("hud_anchor").value).strip().lower()
            corners = {
                "top_left": (OverlayText.LEFT, OverlayText.TOP),
                "top_right": (OverlayText.RIGHT, OverlayText.TOP),
                "bottom_left": (OverlayText.LEFT, OverlayText.BOTTOM),
                "bottom_right": (OverlayText.RIGHT, OverlayText.BOTTOM),
            }
            if anchor not in corners:
                self.get_logger().warn(
                    f"Unknown hud_anchor={anchor!r}; using top_left.")
            self.hud_h_align, self.hud_v_align = corners.get(
                anchor, corners["top_left"])
            self.hud_pub = self.create_publisher(
                OverlayText, str(self.get_parameter("hud_topic").value), 1)
        # Default (reliable) QoS, matching the hardware component's publisher.
        # A best-effort subscription would not match a reliable publisher at
        # all, and the panel would stay blank with the topic visibly alive in
        # `ros2 topic hz` - so leave this alone.
        self.create_subscription(DiagnosticArray, "/diagnostics", self.on_diagnostics, 5)

        rate = max(float(self.get_parameter("publish_rate_hz").value), 0.5)
        self.create_timer(1.0 / rate, self.publish_markers)

        # The TCP reference frame is logged here because it is deliberately NOT
        # in the panel: it never changes during a run, so drawing it 5 times a
        # second only made every row wider.
        hud = (f"HUD on {self.get_parameter('hud_topic').value} "
               f"({self.hud_font} {self.hud_text_size:g})"
               if self.hud_pub is not None else
               "HUD off: rviz_2d_overlay_msgs is not installed, so only the "
               "world marker is published. `rosdep install --from-paths src -r "
               "-y` pulls it in, together with rviz_2d_overlay_plugins for "
               "whichever machine runs RViz")
        self.get_logger().info(
            f"Gripper status overlay up: frame={self.frame_id}, "
            f"tcp {self.tcp_frame} in {self.reference_frame}, "
            f"matching /diagnostics entries containing '{self.match}'. {hud}")

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
    #
    # ONE set of fields, TWO renderers, because the two panels have opposite
    # constraints. The HUD is monospaced, so it gets a real table with aligned
    # columns and one field per line. The marker cannot align anything (see
    # below), so it packs the same fields onto fewer, roughly equal-width
    # lines. Both read the same Field list, so a key added to the hardware
    # component shows up in both without a second edit.
    #
    # HOW WIDE A MARKER LINE ACTUALLY IS, which is why that layout looks the
    # way it does. From rviz_rendering/src/.../movable_text.cpp (jazzy):
    #
    #   effective_char_height = char_height * 2
    #   advance(c)            = glyphAspectRatio(c) * effective_char_height
    #   advance(' ')          = space_width_ = glyphAspectRatio('A') * "   "
    #   newline               : top -= effective_char_height + 0.01
    #
    # Three consequences, all of them counter-intuitive:
    #
    #   - A SPACE IS AS WIDE AS A CAPITAL A. Not the font's space (which is
    #     less than half that) -- MovableText substitutes its own. Measured on
    #     Liberation Sans at the marker's own text height h: space = 1.19 h,
    #     digit = 0.99 h, '.' = 0.49 h. So every doubled space was a full
    #     character of blank, and "{label:<8}{value:>7}" columns were pouring
    #     five to eight characters of it into the middle of every row. That is
    #     what "scattered" was. SINGLE SPACES ONLY in the marker, no padding.
    #
    #   - Each glyph is drawn one char_height wide but advanced TWO, so marker
    #     text is permanently letter-spaced at 100%. No message field changes
    #     it; characters are simply expensive and short labels win.
    #
    #   - Lines sit 2 h apart with 1 h of glyph, so the marker is double spaced
    #     vertically whatever it says. A ragged right edge on top of that is
    #     what stops it reading as a block, so MARKER_GROUPS below packs the
    #     fields into lines that land within about 19-28 h of each other --
    #     measured, not guessed -- with the shortest line LAST, where it reads
    #     as the end of a paragraph instead of a hole.
    #
    # None of that applies to the HUD: QPainter draws it with a real font at a
    # real size, so it is a tight monospaced block and the numbers line up.
    def panel(self):
        """(level label, colour, [Field], [note lines]).

        Fields carry whichever keys the hardware component published; notes are
        whole sentences that belong under the table rather than in it.
        """
        # The level label goes in the header line of both panels, so these
        # branches must NOT repeat it in their notes.
        if self.status is None or self.status_stamp is None:
            return STALE_STYLE[0], STALE_STYLE[1], [], [
                "nothing publishes gripper",
                "diagnostics. normal on mock",
                "and sim; on the rover the",
                "ros2_control component is",
                "not up.",
            ]

        age = (self.get_clock().now() - self.status_stamp).nanoseconds * 1e-9
        if age > self.stale_after:
            return f"STALE {age:.0f}s", STALE_STYLE[1], [], [
                "the gripper component",
                "stopped publishing. last:",
                *wrap(self.status.message[:120], 26),
            ]

        label, colour = LEVEL_STYLE.get(self.status.level, STALE_STYLE)
        fields, notes = [], []

        def field(key, label, value, extra=""):
            """Keep a row only if its value exists, so an unpublished key
            costs no line in either panel."""
            if value not in (None, ""):
                fields.append(Field(key, label, str(value), str(extra or "")))

        gap = self.value("gap_mm")
        cmd_gap = self.value("command_gap_mm")
        field("gap", "gap", gap and f"{gap} mm",
              f"cmd {cmd_gap}" if cmd_gap is not None else "")

        # Both units on one row: it is the same number, and the operator reads
        # deg while the servo talks rad.
        q = self.value("position_rad")
        deg = self.value("position_deg")
        field("joint", "joint", q and f"{q} rad", deg and f"{deg} deg")
        field("step", "step", self.value("position_steps"))

        load = self.value("load_percent")
        nm = self.value("effort_nm")
        field("load", "load", load and f"{load} %", nm and f"{nm} Nm")

        # No reference frame in the row: it is a launch parameter, constant for
        # the whole run, so it is logged once at startup instead of costing
        # nine characters on every redraw.
        if self.show_tcp:
            tcp = self.tcp_pose()
            if tcp is not None:
                field("tcp", "tcp",
                      f"{tcp[0]:.3f} {tcp[1]:.3f} {tcp[2]:.3f} m")

        # Each measurement carries the limit it will be cut off at, so the
        # number can be judged without knowing the servo's configuration.
        field("current", "current", f"{self.value('current_ma', '?')} mA",
              f"max {self.value('limit_current_ma', '?')}")
        field("voltage", "voltage", f"{self.value('voltage_v', '?')} V",
              f"{self.value('limit_voltage_min_v', '?')}-"
              f"{self.value('limit_voltage_max_v', '?')}")
        field("temp", "temp", f"{self.value('temperature_c', '?')} C",
              f"max {self.value('limit_temperature_c', '?')}")
        field("status", "status", self.value("status_byte"),
              self.value("status_flags"))

        if self.value("squeeze_relax_active") == "true":
            notes.append("holding, squeeze-relax")
        if self.value("comms") == "no reply":
            notes.append(f"NO REPLY {self.value('read_failures')} bad reads")

        # The reason last and on its own lines: it is a sentence, not a field,
        # and it is the only thing that matters when it is there.
        if self.status.level != DiagnosticStatus.OK and self.status.message:
            if notes:
                notes.append("")
            notes.extend(wrap(self.status.message, 26))

        return label, colour, fields, notes

    def marker_lines(self, level, fields, notes):
        """The fields packed into MARKER_GROUPS, single-spaced."""
        by_key = {f.key: f for f in fields}
        out = [f"ST3215 GRIPPER {level}"]
        for group in MARKER_GROUPS:
            parts = []
            for key in group:
                f = by_key.get(key)
                if f:
                    # Label, value and qualifier run together with one space
                    # each; a group of two fields becomes one line, which is
                    # how "step 2016 load 18 % 0.53 Nm" is built.
                    parts.append(" ".join(p for p in (f.label, f.value, f.extra) if p))
            if parts:
                out.append(" ".join(parts))
        if notes:
            out.append("")
            out.extend(notes)
        return out

    def hud_lines(self, level, fields, notes):
        """The same fields as a monospaced table: label, right-aligned value,
        then whatever qualifies it. Only legible because the HUD picks its own
        font -- do not copy this layout into the marker."""
        head = f"{'ST3215 GRIPPER':<{HUD_LABEL_W}} {level:>{HUD_VALUE_W}}"
        out = [head, "-" * len(head)]
        for f in fields:
            row = f"{f.label:<{HUD_LABEL_W}} {f.value:>{HUD_VALUE_W}}"
            out.append(f"{row}  {f.extra}".rstrip())
        if notes:
            out.append("")
            out.extend(notes)
        return out

    # -- output --------------------------------------------------------
    def publish_markers(self):
        level, colour, fields, notes = self.panel()
        self.publish_marker(level, colour, self.marker_lines(level, fields, notes))
        if self.hud_pub is not None:
            self.publish_hud(colour, self.hud_lines(level, fields, notes))

    def publish_marker(self, level, colour, lines):
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

    def publish_hud(self, colour, lines):
        """The same panel as a screen-fixed overlay.

        Positions are pixels from the anchored corner and the plugin paints a
        background rectangle behind the text, so unlike the marker this cannot
        be occluded by the robot, does not swing with the camera and does not
        need a clear patch of world to hang in.

        Colour still comes from the diagnostic level, so OK/DANGER/CUTOFF read
        the same in both panels.
        """
        msg = OverlayText()
        msg.action = OverlayText.ADD
        msg.width, msg.height = self.hud_size
        msg.horizontal_alignment = self.hud_h_align
        msg.vertical_alignment = self.hud_v_align
        msg.horizontal_distance, msg.vertical_distance = self.hud_margin
        msg.bg_color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=self.hud_bg_alpha)
        msg.fg_color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=1.0)
        msg.line_width = 1
        msg.text_size = self.hud_text_size
        msg.font = self.hud_font
        msg.text = "\n".join(lines)
        self.hud_pub.publish(msg)


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
