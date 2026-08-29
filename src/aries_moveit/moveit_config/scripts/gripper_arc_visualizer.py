#!/usr/bin/env python3
"""RViz visualizer for the four-bar gripper.

Publishes a MarkerArray in the arm_gripper_base_link frame showing:
  - the arc each bucket jaw sweeps between fully open and fully closed,
  - the path of the jaw midpoint (where a grasped object centre travels),
  - the point of closing (where the jaws meet at full close),
  - the current jaw positions + gap read live from /joint_states,
  - the gripper_tcp IK frame for comparison.

Add the topic (default: gripper_arc_markers) as a Marker/MarkerArray display
in RViz. Everything is expressed in the gripper base frame, so the overlay
follows the arm automatically.

Geometry comes from the same URDF/STL-derived four-bar tables used by
vision_grasp_node: joint q -> jaw gap, and joint q -> true jaw-midpoint z.
"""

import bisect
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# Four-bar geometry tables (q in rad, distances in m), derived from
# gripper_new.xacro + gripper_bucket.stl. q=-1.57 fully open, q=+0.07 closed.
Q_GAP = [-1.570, -1.365, -1.160, -0.955, -0.750,
         -0.545, -0.340, -0.2879, -0.2271, -0.1976,
         -0.1861, -0.1385, -0.0498, 0.0093, 0.070]
GAP = [0.1826, 0.1790, 0.1684, 0.1512, 0.1281,
       0.1002, 0.0685, 0.0600, 0.0500, 0.0451,
       0.0431, 0.0351, 0.0200, 0.0099, 0.0000]

Q_MID = [-1.570, -1.000, -0.500, -0.200, -0.140, -0.050, 0.000, 0.070]
Z_MID = [0.1342, 0.1680, 0.2092, 0.2180, 0.2189, 0.2196, 0.2197, 0.2195]

Q_OPEN = -1.57
Q_CLOSE = 0.07

# The SECONDARY gripper (gripper_type:=st3215) needs no tables: the pinion
# direct-drives two racks, so the jaws translate and both relationships are
# exact.  The "arc" it sweeps is a straight line at constant height, which is
# the whole visible difference between the two mechanisms and worth drawing
# honestly rather than approximating with the four-bar's curve.
#
# THIS IS THE THIRD COPY of the pitch radius (the others are
# aries/urdf/gripper_st3215.xacro and aries_vision_grasp/fourbar.py) and it is
# copied rather than imported because those two packages cannot be imported
# from here -- the URDF is XML and the grasp package is COLCON_IGNOREd.
# src/aries/test/test_gripper_st3215.py cross-checks all three.
ST3215_PITCH_R = 0.01002676
ST3215_Q_OPEN = -4.065
ST3215_Q_CLOSE = 0.07
ST3215_CONTACT_Z = 0.2078


def _interp(x, xs, ys):
    """Linear interpolation with clamped ends (xs ascending)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def _color(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


class GripperArcVisualizer(Node):

    def __init__(self):
        super().__init__("gripper_arc_visualizer")
        self.declare_parameter("frame_id", "arm_gripper_base_link")
        self.declare_parameter("marker_topic", "gripper_arc_markers")
        self.declare_parameter("joint_state_topic", "joint_states")
        self.declare_parameter("gripper_joint_name", "gripper_gear_left_joint")
        # Which gripper is fitted. MUST match the URDF's gripper_type: the two
        # share the joint name and the closed angle but nothing else, so a
        # mismatch draws a confident overlay in the wrong place.
        self.declare_parameter("gripper_type", "v2")
        self.declare_parameter("publish_rate_hz", 10.0)
        # Jaw midpoint sits ~25.9 mm off the base link centreline in +y.
        self.declare_parameter("contact_y_offset_m", 0.001)
        # Extra shift of the whole arc along +z (toward the jaw tips), to line
        # the overlay up with the bucket meshes. Live-tunable:
        #   ros2 param set /gripper_arc_visualizer arc_z_offset_m 0.03
        self.declare_parameter("arc_z_offset_m", 0.02)
        # Must match gripper_tcp_z in gripper_new.xacro.
        self.declare_parameter("tcp_z_m", 0.15)
        self.declare_parameter("arc_samples", 60)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.gripper_joint = str(self.get_parameter("gripper_joint_name").value)
        self.gripper_type = str(self.get_parameter("gripper_type").value).strip().lower()
        if self.gripper_type not in ("v2", "st3215"):
            self.gripper_type = "v2"
        self.samples = max(int(self.get_parameter("arc_samples").value), 8)
        self._refresh_tunables()

        self.current_q = self.q_range()[0]

        self.marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), 1)
        self.create_subscription(
            JointState, str(self.get_parameter("joint_state_topic").value),
            self.on_joint_state, 10)
        rate = max(float(self.get_parameter("publish_rate_hz").value), 0.5)
        self.create_timer(1.0 / rate, self.publish_markers)

        self.get_logger().info(
            f"Gripper arc visualizer up: frame={self.frame_id} "
            f"topic={self.marker_pub.topic_name}")

    def _refresh_tunables(self):
        """Re-read live-tunable parameters so `ros2 param set` takes effect."""
        self.y_off = float(self.get_parameter("contact_y_offset_m").value)
        self.z_off = float(self.get_parameter("arc_z_offset_m").value)
        self.tcp_z = float(self.get_parameter("tcp_z_m").value)

    def on_joint_state(self, msg: JointState):
        try:
            idx = msg.name.index(self.gripper_joint)
        except ValueError:
            return
        if idx < len(msg.position):
            self.current_q = float(msg.position[idx])

    def q_range(self):
        """(open, closed) joint angle for the fitted gripper."""
        if self.gripper_type == "st3215":
            return ST3215_Q_OPEN, ST3215_Q_CLOSE
        return Q_OPEN, Q_CLOSE

    def contact_z(self, q):
        """Height of the jaw contact midpoint at joint angle q.

        Constant on the ST3215: its jaws translate. The four-bar's climbs 86 mm
        between open and closed, which is what the grey midpoint path draws.
        """
        if self.gripper_type == "st3215":
            return ST3215_CONTACT_Z + self.z_off
        return _interp(q, Q_MID, Z_MID) + self.z_off

    def jaw_points(self, q):
        """Left and right jaw contact points at joint angle q."""
        if self.gripper_type == "st3215":
            q = min(max(q, ST3215_Q_OPEN), ST3215_Q_CLOSE)
            half_gap = ST3215_PITCH_R * (ST3215_Q_CLOSE - q)
            z = ST3215_CONTACT_Z + self.z_off
        else:
            half_gap = 0.5 * _interp(q, Q_GAP, GAP)
            z = _interp(q, Q_MID, Z_MID) + self.z_off
        left = Point(x=-half_gap, y=self.y_off, z=z)
        right = Point(x=half_gap, y=self.y_off, z=z)
        return left, right

    def _marker(self, mid, mtype, scale, color):
        m = Marker()
        m.header.frame_id = self.frame_id
        # Zero stamp + frame_locked keeps the overlay glued to the gripper:
        # RViz re-transforms with the latest TF every render pass. Stamping
        # "now" instead makes the markers jitter, because TF for that exact
        # moment lags behind joint_states while the arm moves.
        m.frame_locked = True
        m.ns = "gripper_arc"
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color = color
        m.pose.orientation.w = 1.0
        return m

    def publish_markers(self):
        self._refresh_tunables()
        arr = MarkerArray()
        q_open, q_close = self.q_range()
        qs = [q_open + (q_close - q_open) * i / (self.samples - 1)
              for i in range(self.samples)]

        # Jaw sweep arcs, open -> close (cyan).
        left_arc = self._marker(0, Marker.LINE_STRIP, (0.003, 0.0, 0.0),
                                _color(0.1, 0.8, 0.9))
        right_arc = self._marker(1, Marker.LINE_STRIP, (0.003, 0.0, 0.0),
                                 _color(0.1, 0.8, 0.9))
        # Midpoint (object-centre) path during closing (gray).
        mid_path = self._marker(2, Marker.LINE_STRIP, (0.002, 0.0, 0.0),
                                _color(0.7, 0.7, 0.7, 0.8))
        for q in qs:
            left, right = self.jaw_points(q)
            left_arc.points.append(left)
            right_arc.points.append(right)
            mid_path.points.append(
                Point(x=0.0, y=self.y_off, z=self.contact_z(q)))
        arr.markers += [left_arc, right_arc, mid_path]

        # Point of closing: where the jaws meet at full close (red).
        close_pt = self._marker(3, Marker.SPHERE, (0.014, 0.014, 0.014),
                                _color(0.95, 0.15, 0.15))
        close_pt.pose.position = Point(
            x=0.0, y=self.y_off, z=self.contact_z(q_close))
        arr.markers.append(close_pt)

        # Fully-open jaw endpoints (green).
        open_left, open_right = self.jaw_points(q_open)
        for mid, pt in ((4, open_left), (5, open_right)):
            m = self._marker(mid, Marker.SPHERE, (0.010, 0.010, 0.010),
                             _color(0.2, 0.85, 0.3))
            m.pose.position = pt
            arr.markers.append(m)

        # Current jaw positions + gap line from live joint state (yellow).
        cur_left, cur_right = self.jaw_points(self.current_q)
        for mid, pt in ((6, cur_left), (7, cur_right)):
            m = self._marker(mid, Marker.SPHERE, (0.012, 0.012, 0.012),
                             _color(1.0, 0.85, 0.1))
            m.pose.position = pt
            arr.markers.append(m)
        gap_line = self._marker(8, Marker.LINE_STRIP, (0.002, 0.0, 0.0),
                                _color(1.0, 0.85, 0.1, 0.9))
        gap_line.points = [cur_left, cur_right]
        arr.markers.append(gap_line)

        # Gap readout above the gripper.
        gap_mm = _interp(self.current_q, Q_GAP, GAP) * 1000.0
        text = self._marker(9, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.02),
                            _color(1.0, 1.0, 1.0))
        text.text = f"gap {gap_mm:.0f} mm  q {self.current_q:+.2f}"
        text.pose.position = Point(x=0.0, y=-0.06, z=0.10)
        arr.markers.append(text)

        # gripper_tcp IK frame for comparison (magenta).
        tcp = self._marker(10, Marker.SPHERE, (0.010, 0.010, 0.010),
                           _color(0.9, 0.2, 0.9))
        tcp.pose.position = Point(x=0.0, y=0.0, z=self.tcp_z)
        arr.markers.append(tcp)

        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = GripperArcVisualizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
