#!/usr/bin/env python3
"""ARIES Base Station Status Checker Node

The operator-side counterpart of aries_bringup/nodes/full_hardware_checker.py,
in the same style. That one runs on the ROVER and answers none of the questions
that matter at this end: it probes serial ports, the CAN link and USB
enumeration -- all of which are on the robot -- and it prints to the robot's
console, which nobody is looking at.

What can be wrong at the base station is a shorter list, and every item on it
is silent:

  * this machine is not on the field link, or is on it under the wrong address;
  * the DDS environment of THIS PROCESS is not the one the rover is using --
    the classic being a launch started from a terminal that predates the
    exports, which sits on domain 0 with the default RMW and sees an empty
    graph on a link that pings fine;
  * the pad is not enumerated, or -- worse, because it half works -- the rover
    is publishing /joy too and the consumers see two pads interleaved;
  * the downlink is arriving but the decompressors are not, or the other way
    around;
  * the rover is up and the antenna is fine but nothing is being received,
    which is a QoS or discovery fault rather than a link fault;
  * more than one RViz, which is how the two-viewer bug went unnoticed.

WHY THIS DOES NOT SUBSCRIBE TO /downlink/*

    Subscribing to a compressed stream is what pulls it across the antenna, and
    a second process doing so pulls a SECOND copy: separate DDS participants
    get separate copies, which is the whole reason camera_view.launch.py
    decompresses once and everything else reads its local output. A checker
    that doubled the link load whenever it ran would be measuring itself.

    So the downlink is judged two ways that cost nothing: the publisher count
    on /downlink/<cam>/... comes from the graph, and the frames are counted on
    the machine-local /<cam>/view/... instead. That is the better measurement
    anyway -- a frame there means the link AND the decompressor both worked.

    For actual bandwidth, run aries_bringup's downlink_report.py deliberately
    and briefly. It subscribes on purpose and says so.

WHAT OF THE ROVER IS CHECKED HERE

    full_hardware_checker probes the robot's serial ports, its CAN adapter and
    its USB tree, and prints the result to the robot's console. None of that
    can move to this end: they are facts about the far side of a radio link.
    What CAN move is everything it learns from TOPICS, and that half is
    repeated here -- arm and gripper controllers, MoveGroup and Servo, the six
    ODrive axes, the drive bridge, the IMU -- so the operator reads the same
    subsystem rows without opening an SSH session to the robot.

    Two things are deliberately unlike the rover's copy:

    ABSENT IS NOT BROKEN. This end cannot know what the rover was launched
    with, and a drive-only run has no arm by design. A subsystem with no
    publisher at all is a WARNING here, never a "not ready"; only one that is
    half up -- advertised and silent, an axis in a fault, axes that disagree
    with the drive bridge -- is a problem. A checker that shouted on every
    partial launch is a checker people stop reading.

    THE ROVER'S OWN VERDICT BEATS A GUESS FROM HERE. /aries_drive/status is
    2 Hz of JSON from the drive bridge carrying `armed`, `pending_axes` and the
    CAN link state IT observed. That is the CAN row, relayed by the only
    process in a position to see it, for a few hundred bytes a second. Where it
    and the per-axis rows disagree the disagreement is itself the finding: the
    bridge reporting armed while no axis arrives here is the LINK, not the
    drive.

WHAT THIS COSTS ON THE LINK

    Presence is answered by graph queries, which are free -- discovery already
    carries them. On top of that it subscribes only to topics that were sized
    for a 250 kbit/s CAN bus in the first place: six axes at 5 Hz from the
    poller, the drive bridge at 2 Hz, /cmd_vel at 20 Hz. Together under
    0.1 Mbit/s, against a downlink budget of ~28.

    The IMU is the one exception, 100 Hz of sensor_msgs/Imu at roughly
    0.3 Mbit/s. It is a percent of the link and it is the only way to tell a
    dead IMU from a live one, but check_imu:=false drops it on a link that is
    already marginal.

    Everything requests BEST_EFFORT, including the topics the rover publishes
    RELIABLE. A checker must never be the reason a topic looks dead, and it
    must never make the antenna retransmit on its own account.

Manual check:
  ros2 service call /check_base_station std_srvs/srv/Trigger
"""

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, Imu, JointState, Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from aries_common import comms
from aries_common.odrive_axes import (
    AXIS_HEADER,
    AXIS_LABELS,
    CLOSED_LOOP,
    NUM_AXES,
    axis_state_name,
)
from aries_common.odrive_errors import format_odrive_error

# The vendor messages, and the same guard full_hardware_checker uses. A base
# station that has not built src/vendor/ros_odrive still has to report the
# link, the pad and the downlink -- an ImportError here would take all of that
# down over a subsystem the operator may not even be using.
try:
    from odrive_can.msg import ControllerStatus, ODriveStatus
except Exception:
    ControllerStatus = None
    ODriveStatus = None

# Topics whose PRESENCE is answered from the graph rather than by subscribing.
# Discovery already carries the publisher counts, so these rows cost nothing on
# the link no matter how fast the topic underneath them is.
ARM_CONTROLLER = "rebel_arm_trajectory_controller"
GRIPPER_CONTROLLER = "rebel_gripper_controller"

# Requested BEST_EFFORT matches a BEST_EFFORT publisher and a RELIABLE one
# alike; requested RELIABLE matches only the latter. A checker must never be
# the reason a topic looks dead, so everything it samples is requested on the
# permissive side.
SAMPLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
    depth=1,
)

# /robot_description is latched and published once, when the rover starts. A
# VOLATILE subscriber joining afterwards -- which is always, here -- would
# never see it, so this one has to ask for the durability it was offered.
LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


def _as_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _rate_band(hz, expected):
    """Rates bucketed for change detection.

    The status line is only reprinted when something CHANGES, and a frame rate
    never repeats exactly, so a snapshot carrying raw floats would reprint on
    every tick and bury the thing you are watching for. Only the band goes in
    the snapshot; the exact figure is still displayed.
    """
    if hz <= 0.0:
        return "none"
    if hz < expected * 0.5:
        return "low"
    return "ok"


class _Pinger:
    """ICMP reachability, sampled off the executor thread.

    `ping` blocks for up to its timeout, and three of them inside a ROS timer
    callback would stall every subscription for as long as they took -- so the
    freshness checks in the same pass would report stale data that the checker
    itself caused.
    """

    def __init__(self, targets, interval):
        self._targets = dict(targets)
        self._interval = max(2.0, float(interval))
        self._results = {name: None for name in self._targets}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def results(self):
        with self._lock:
            return dict(self._results)

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            for name, address in self._targets.items():
                if self._stop.is_set():
                    return
                reachable = self._ping(address)
                with self._lock:
                    self._results[name] = reachable
            self._stop.wait(self._interval)

    @staticmethod
    def _ping(address):
        try:
            done = subprocess.run(
                ["ping", "-n", "-c", "1", "-W", "1", str(address)],
                capture_output=True,
                timeout=3,
            )
            return done.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return None


class BaseStationChecker(Node):
    def __init__(self):
        super().__init__("base_station_checker")

        # ANSI colours, same style as the rover-side checkers.
        self.GREEN = "\033[92m"
        self.RED = "\033[91m"
        self.YELLOW = "\033[93m"
        self.BLUE = "\033[94m"
        self.CYAN = "\033[96m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"
        self.GREY = "\033[90m"

        self.declare_parameter("check_interval", 4.0)
        self.declare_parameter("timeout", 5.0)
        self.declare_parameter("print_only_on_change", True)

        self.declare_parameter("check_link", True)
        self.declare_parameter("check_joystick", True)
        self.declare_parameter("check_downlink", True)
        self.declare_parameter("check_rover", True)
        self.declare_parameter("check_viewers", True)

        # The rover's own subsystems, judged from the topics that cross the
        # link. Each one can be dropped on its own: a marginal link, or a run
        # where that subsystem was deliberately not started.
        self.declare_parameter("check_arm", True)
        self.declare_parameter("check_gripper", True)
        self.declare_parameter("check_drive", True)
        # The only expensive one -- 100 Hz of sensor_msgs/Imu, ~0.3 Mbit/s.
        self.declare_parameter("check_imu", True)

        # Comma-separated, not a list: a launch file cannot build a
        # STRING_ARRAY out of a LaunchConfiguration, and handing a STRING to a
        # list-declared parameter kills the node at startup with
        # InvalidParameterTypeException -- so it reports nothing at all, which
        # is the one thing a checker must never do. _as_list splits either.
        self.declare_parameter("cameras", "gripper_camera,rover_camera,rear_camera")
        self.declare_parameter("color_only", "rear_camera")
        self.declare_parameter("expected_color_hz", 15.0)
        self.declare_parameter("expected_depth_hz", 5.0)

        # Comma-separated for the same reason `cameras` is: a launch file
        # cannot build a STRING_ARRAY out of a LaunchConfiguration.
        self.declare_parameter(
            "arm_joint_names", "joint1,joint2,joint3,joint4,joint5,joint6"
        )
        # The one joint rebel_gripper_controller drives; the rest of the
        # four-bar follows it through the URDF's mimics and is not published.
        self.declare_parameter("gripper_joint_names", "gripper_gear_left_joint")
        # 2 Hz of JSON from cmd_vel_odrive_bridge: armed, pending_axes, and the
        # CAN link state as the rover sees it. The nearest thing this end has
        # to full_hardware_checker's CAN probe, and far more honest than
        # inferring the link from axis silence.
        self.declare_parameter("drive_status_topic", "/aries_drive/status")
        self.declare_parameter("imu_topic", "/microstrain/imu/data")
        self.declare_parameter("expected_odrive_axes", NUM_AXES)

        self.declare_parameter("joystick_device", "/dev/input/js0")
        # True when this machine runs the joy driver, which is the field
        # default. False if the pad is plugged into the rover instead -- then a
        # local /joy publisher is the fault, not the absence of one.
        self.declare_parameter("expect_local_joy", True)
        self.declare_parameter("expected_joy_hz", 80.0)

        self.check_interval = float(self.get_parameter("check_interval").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.print_only_on_change = bool(self.get_parameter("print_only_on_change").value)

        self.check_link = bool(self.get_parameter("check_link").value)
        self.check_joystick = bool(self.get_parameter("check_joystick").value)
        self.check_downlink = bool(self.get_parameter("check_downlink").value)
        self.check_rover = bool(self.get_parameter("check_rover").value)
        self.check_viewers = bool(self.get_parameter("check_viewers").value)
        self.check_arm = bool(self.get_parameter("check_arm").value)
        self.check_gripper = bool(self.get_parameter("check_gripper").value)
        self.check_drive = bool(self.get_parameter("check_drive").value)
        self.check_imu = bool(self.get_parameter("check_imu").value)

        self.arm_joint_names = _as_list(self.get_parameter("arm_joint_names").value)
        self.gripper_joint_names = _as_list(
            self.get_parameter("gripper_joint_names").value
        )
        self.drive_status_topic = str(self.get_parameter("drive_status_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.expected_axes = max(
            0, min(int(self.get_parameter("expected_odrive_axes").value), NUM_AXES)
        )

        self.cameras = _as_list(self.get_parameter("cameras").value)
        self.color_only = set(_as_list(self.get_parameter("color_only").value))
        self.expected_color_hz = float(self.get_parameter("expected_color_hz").value)
        self.expected_depth_hz = float(self.get_parameter("expected_depth_hz").value)

        self.joystick_device = str(self.get_parameter("joystick_device").value)
        self.expect_local_joy = bool(self.get_parameter("expect_local_joy").value)
        self.expected_joy_hz = float(self.get_parameter("expected_joy_hz").value)

        # ── What the link is supposed to look like, from devices.yaml ─────────
        # Read once: it is a file on disk, and re-reading it every tick would
        # hide an edit behind a restart-free "it still says the old thing".
        self.hosts = comms.hosts()
        self.radios = comms.radios()
        self.domain_expected = comms.domain_id()
        self.local_address = comms.local_address() if self.check_link else None
        self.local_host_name = (
            comms.host_name_for(self.local_address) if self.local_address else None
        )

        targets = {}
        if self.check_link:
            for name, address in self.hosts.items():
                if address != self.local_address:
                    targets[f"host:{name}"] = address
            for name, address in self.radios.items():
                targets[f"radio:{name}"] = address
        self.pinger = _Pinger(targets, self.check_interval) if targets else None

        # ── Message arrival ───────────────────────────────────────────────────
        self.joy_times = deque(maxlen=200)
        self.view_times = {}
        self.tf_time = None
        self.joint_state_time = None
        self.joint_times = {}
        self.robot_description_seen = False

        self.arm_joystick_time = None
        self.arm_joystick_status = ""

        self.drive_status_time = None
        self.drive_status = {}
        self.mock_rover_time = None
        self.cmd_vel_time = None
        self.cmd_vel_moving_time = None

        # Free-running values, kept off the snapshot on purpose: a bus voltage
        # that wanders in the third decimal would reprint the whole report on
        # every tick. Only the booleans derived from them go in the key; these
        # are read straight off the node when the report is actually printed.
        self.ctrl_times = [None] * NUM_AXES
        self.ctrl_states = [None] * NUM_AXES
        self.ctrl_errors = [None] * NUM_AXES
        self.ctrl_vel = [None] * NUM_AXES
        self.odrv_times = [None] * NUM_AXES
        self.odrv_voltage = [None] * NUM_AXES
        self.odrv_errors = [None] * NUM_AXES

        self.imu_time = None
        self.imu_frame_id = ""

        self.create_subscription(Joy, "/joy", self._joy_cb, SAMPLE_QOS)

        if self.check_downlink:
            for camera in self.cameras:
                streams = ["color"] if camera in self.color_only else ["color", "depth"]
                for stream in streams:
                    topic = f"/{camera}/view/{stream}"
                    self.view_times[topic] = deque(maxlen=200)
                    self.create_subscription(
                        Image, topic, self._make_view_cb(topic), SAMPLE_QOS
                    )

        if self.check_rover:
            self.create_subscription(TFMessage, "/tf", self._tf_cb, SAMPLE_QOS)
            self.create_subscription(
                String, "/robot_description", self._description_cb, LATCHED_QOS
            )

        # One subscription, three readers: the rover section wants freshness,
        # the arm and gripper sections want per-joint freshness out of the same
        # messages. Subscribing three times would pull three copies over the
        # antenna for one 100 Hz stream.
        if self.check_rover or self.check_arm or self.check_gripper:
            self.create_subscription(
                JointState, "/joint_states", self._joint_state_cb, SAMPLE_QOS
            )

        if self.check_arm:
            self.create_subscription(
                String, "/arm_joystick/status", self._arm_joystick_cb, SAMPLE_QOS
            )

        if self.check_drive:
            self.create_subscription(
                String, self.drive_status_topic, self._drive_status_cb, SAMPLE_QOS
            )
            self.create_subscription(
                String, "/mock_rover/status", self._mock_rover_cb, SAMPLE_QOS
            )
            # The round trip: the pad is read HERE, and /cmd_vel is what the
            # rover made of it. Nothing else on this machine proves the whole
            # path -- a fresh /joy only proves the near half.
            self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, SAMPLE_QOS)

            # odrive_can_poller runs at 5 Hz per axis, so twelve subscriptions
            # here are ~60 messages a second of about a hundred bytes.
            if ControllerStatus is not None:
                for i in range(NUM_AXES):
                    self.create_subscription(
                        ControllerStatus,
                        f"/odrive_axis{i}/controller_status",
                        lambda msg, idx=i: self._ctrl_cb(msg, idx),
                        SAMPLE_QOS,
                    )
            if ODriveStatus is not None:
                for i in range(NUM_AXES):
                    self.create_subscription(
                        ODriveStatus,
                        f"/odrive_axis{i}/odrive_status",
                        lambda msg, idx=i: self._odrv_cb(msg, idx),
                        SAMPLE_QOS,
                    )

        if self.check_imu:
            self.create_subscription(Imu, self.imu_topic, self._imu_cb, SAMPLE_QOS)

        self.prev_snapshot = None
        self.initial_check_done = False
        self.manual_check_requested = False

        self.create_service(Trigger, "check_base_station", self._service_cb)
        self.create_timer(self.check_interval, self._check_status)

        self.get_logger().info(
            "base station checker up; manual: "
            "ros2 service call /check_base_station std_srvs/srv/Trigger"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _service_cb(self, request, response):
        self.manual_check_requested = True
        response.success = True
        response.message = "Base station check requested"
        return response

    def _joy_cb(self, msg):
        self.joy_times.append(time.monotonic())

    def _make_view_cb(self, topic):
        def cb(msg):
            self.view_times[topic].append(time.monotonic())
        return cb

    def _tf_cb(self, msg):
        self.tf_time = time.monotonic()

    def _joint_state_cb(self, msg):
        self.joint_state_time = time.monotonic()
        # Per joint, not just per message: the arm and the gripper are
        # published by different controllers into the same topic, so a fresh
        # /joint_states proves nothing about which of them is actually running.
        for name in msg.name:
            self.joint_times[name] = self.joint_state_time

    def _description_cb(self, msg):
        self.robot_description_seen = True

    def _arm_joystick_cb(self, msg):
        self.arm_joystick_time = time.monotonic()
        self.arm_joystick_status = msg.data

    def _drive_status_cb(self, msg):
        self.drive_status_time = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            # A malformed status is not a dead one: keep the timestamp so the
            # row reads "arriving" and let the fields fall back to unknown.
            return
        if isinstance(payload, dict):
            self.drive_status = payload

    def _mock_rover_cb(self, msg):
        self.mock_rover_time = time.monotonic()

    def _cmd_vel_cb(self, msg):
        # Two clocks. Any message at all proves the rover's teleop stack is
        # alive and reading the pad across the link; a NONZERO one proves the
        # operator is actually commanding motion, which is what
        # full_hardware_checker reports. Only tracking the second made a parked
        # rover look like a broken one.
        self.cmd_vel_time = time.monotonic()
        if abs(msg.linear.x) > 1e-5 or abs(msg.angular.z) > 1e-5:
            self.cmd_vel_moving_time = self.cmd_vel_time

    def _ctrl_cb(self, msg, idx):
        self.ctrl_times[idx] = time.monotonic()
        self.ctrl_states[idx] = getattr(msg, "axis_state", None)
        self.ctrl_errors[idx] = getattr(msg, "active_errors", None)
        self.ctrl_vel[idx] = getattr(msg, "vel_estimate", None)

    def _odrv_cb(self, msg, idx):
        self.odrv_times[idx] = time.monotonic()
        self.odrv_voltage[idx] = getattr(msg, "bus_voltage", None)
        self.odrv_errors[idx] = getattr(msg, "active_errors", None)

    def _imu_cb(self, msg):
        self.imu_time = time.monotonic()
        self.imu_frame_id = msg.header.frame_id

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_recent(self, stamp):
        return stamp is not None and (time.monotonic() - stamp) <= self.timeout

    def _hz(self, times):
        """Rate over the freshness window, not over all time.

        An average since startup keeps reporting a healthy figure for minutes
        after a stream stops, which is exactly the case this has to catch.
        """
        now = time.monotonic()
        window = max(self.timeout, 1.0)
        recent = [t for t in times if now - t <= window]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def _dds_environment(self):
        """The DDS settings of THIS process, and whether they are usable.

        Read from os.environ rather than recomputed, because the failure being
        looked for is precisely a process that inherited something different
        from what the config file now says. A launch action sets these for the
        nodes it starts, so this node's own environment IS the stack's.
        """
        domain = os.environ.get("ROS_DOMAIN_ID", "")
        rmw = os.environ.get("RMW_IMPLEMENTATION", "")
        uri = os.environ.get("CYCLONEDDS_URI", "")

        path = uri[len("file://"):] if uri.startswith("file://") else ""
        pinned = None
        readable = None
        if path:
            try:
                text = Path(path).read_text()
                readable = True
                match = re.search(r'<NetworkInterface\s+address="([^"]+)"', text)
                pinned = match.group(1) if match else None
            except OSError:
                readable = False

        return {
            "domain": domain,
            "domain_ok": domain == self.domain_expected,
            "rmw": rmw,
            "rmw_ok": rmw == comms.RMW,
            "uri": uri,
            "uri_path": path,
            "uri_readable": readable,
            "pinned": pinned,
            # A pinned address this machine does not hold is fatal to Cyclone,
            # not a warning: it refuses to create the domain and every node
            # dies at startup. If this node is alive to report it, the address
            # was right when it started -- but the config on disk can have been
            # rewritten since by another launch, so it is still worth saying.
            "pinned_ok": pinned is None or pinned == self.local_address,
        }

    def _any_publisher(self, *topics):
        """Presence from the graph, which discovery already paid for.

        Used for everything whose absence is the question -- a controller, a
        planner, a servo node. Subscribing to answer it would pull the topic
        across the antenna for no information the graph did not already have.
        """
        return any(self.count_publishers(topic) > 0 for topic in topics)

    def _any_subscriber(self, *topics):
        """The mirror image, for command topics the rover CONSUMES.

        A trajectory controller does not publish its command topic, it
        subscribes to it -- so on a rover that has not yet published a state
        message this is the only evidence it exists. Safe from here because
        nothing on the base station subscribes to these.
        """
        return any(self.count_subscribers(topic) > 0 for topic in topics)

    def _joints_fresh(self, names):
        """Which of `names` arrived within the freshness window, and which did not."""
        missing = [
            name
            for name in names
            if not self._is_recent(self.joint_times.get(name))
        ]
        return (not missing and bool(names)), missing

    def _viewer_count(self):
        try:
            return sum(
                1 for name, _ in self.get_node_names_and_namespaces()
                if name == "rviz2" or name.endswith("_rviz")
            )
        except Exception:
            return -1

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def _make_snapshot(self):
        s = {}

        if self.check_link:
            s["dds"] = self._dds_environment()
            s["local_address"] = self.local_address
            s["local_host_name"] = self.local_host_name
            s["ping"] = self.pinger.results() if self.pinger else {}

        if self.check_joystick:
            s["joy_device_present"] = os.path.exists(self.joystick_device)
            s["joy_publishers"] = self.count_publishers("/joy")
            s["joy_raw_publishers"] = self.count_publishers("/joy/raw")
            joy_hz = self._hz(self.joy_times)
            s["joy_hz"] = joy_hz
            s["joy_band"] = _rate_band(joy_hz, self.expected_joy_hz)

        if self.check_downlink:
            streams = []
            for camera in self.cameras:
                for stream in ("color", "depth"):
                    if stream == "depth" and camera in self.color_only:
                        continue
                    src = (
                        f"/downlink/{camera}/color/compressed"
                        if stream == "color"
                        else f"/downlink/{camera}/depth/compressedDepth"
                    )
                    view = f"/{camera}/view/{stream}"
                    hz = self._hz(self.view_times.get(view, ()))
                    expected = (
                        self.expected_color_hz if stream == "color"
                        else self.expected_depth_hz
                    )
                    streams.append({
                        "camera": camera,
                        "stream": stream,
                        "src": src,
                        "view": view,
                        # Graph queries only: subscribing to the compressed
                        # topic here would pull a second copy over the antenna.
                        "src_publishers": self.count_publishers(src),
                        "view_publishers": self.count_publishers(view),
                        "hz": hz,
                        "band": _rate_band(hz, expected),
                        "expected": expected,
                    })
            s["streams"] = streams

        if self.check_rover:
            s["tf_ok"] = self._is_recent(self.tf_time)
            s["joint_states_ok"] = self._is_recent(self.joint_state_time)
            s["robot_description_ok"] = self.robot_description_seen
            s["tf_publishers"] = self.count_publishers("/tf")

        if self.check_rover or self.check_arm or self.check_gripper:
            s["joint_stream_ok"] = self._is_recent(self.joint_state_time)

        if self.check_arm:
            joints_ok, missing = self._joints_fresh(self.arm_joint_names)
            s["arm"] = {
                "controller": self._any_publisher(
                    f"/{ARM_CONTROLLER}/state",
                    f"/{ARM_CONTROLLER}/controller_state",
                ) or self._any_subscriber(f"/{ARM_CONTROLLER}/joint_trajectory"),
                "joints_ok": joints_ok,
                "missing_joints": missing,
                "move_group": self._any_publisher(
                    "/move_action/_action/status", "/monitored_planning_scene"
                ),
                "servo": self._any_publisher("/servo_node/status")
                or self._any_subscriber("/delta_twist_cmds", "/delta_joint_cmds"),
                "joystick_ok": self._is_recent(self.arm_joystick_time),
                "joystick_status": self.arm_joystick_status,
            }

        if self.check_gripper:
            joints_ok, missing = self._joints_fresh(self.gripper_joint_names)
            s["gripper"] = {
                "controller": self._any_publisher(
                    f"/{GRIPPER_CONTROLLER}/state",
                    f"/{GRIPPER_CONTROLLER}/controller_state",
                ) or self._any_subscriber(f"/{GRIPPER_CONTROLLER}/joint_trajectory"),
                "joints_ok": joints_ok,
                "missing_joints": missing,
            }

        if self.check_drive:
            status = dict(self.drive_status)
            axes = []
            for i in range(NUM_AXES):
                ctrl_fresh = self._is_recent(self.ctrl_times[i])
                axes.append({
                    "index": i,
                    # Advertised is not the same as answering: the vendor node
                    # keeps publishing the topic when the ODrive behind it has
                    # stopped replying, so both are reported.
                    "advertised": self.count_publishers(
                        f"/odrive_axis{i}/controller_status"
                    ) > 0,
                    "ctrl_fresh": ctrl_fresh,
                    "odrv_fresh": self._is_recent(self.odrv_times[i]),
                    "closed_loop": (
                        ctrl_fresh and self.ctrl_states[i] == CLOSED_LOOP
                    ),
                    "faulted": (
                        (self.ctrl_errors[i] or 0) != 0
                        or (self.odrv_errors[i] or 0) != 0
                    ),
                })
            s["drive"] = {
                "status_ok": self._is_recent(self.drive_status_time),
                "status_publishers": self.count_publishers(self.drive_status_topic),
                "armed": status.get("armed"),
                "enable_requested": status.get("enable_requested"),
                "can_link": status.get("can_link"),
                "pending_axes": list(status.get("pending_axes") or []),
                "command_valid": status.get("command_valid"),
                "mock_ok": self._is_recent(self.mock_rover_time),
                "cmd_vel_ok": self._is_recent(self.cmd_vel_time),
                "cmd_vel_moving": self._is_recent(self.cmd_vel_moving_time),
                "cmd_vel_publishers": self.count_publishers("/cmd_vel"),
                # No odrive_can build here means the axis table cannot be read
                # at all, which must not be mistaken for six dead axes.
                "messages_available": ControllerStatus is not None,
                "axes": axes,
            }

        if self.check_imu:
            s["imu"] = {
                "ok": self._is_recent(self.imu_time),
                "publishers": self.count_publishers(self.imu_topic),
                "frame_id": self.imu_frame_id,
            }

        if self.check_viewers:
            s["viewers"] = self._viewer_count()

        return s

    def _snapshot_key(self, s):
        """The part of the snapshot that decides whether to reprint.

        Rates are replaced by their bands and reachability by its booleans; the
        exact numbers still reach the display, they just do not retrigger it.
        """
        key = {}
        if self.check_link:
            dds = s["dds"]
            key["dds"] = (dds["domain_ok"], dds["rmw_ok"], dds["uri_readable"],
                          dds["pinned_ok"])
            key["local"] = (s["local_address"], s["local_host_name"])
            key["ping"] = tuple(sorted(s["ping"].items()))
        if self.check_joystick:
            key["joy"] = (s["joy_device_present"], s["joy_publishers"],
                          s["joy_raw_publishers"], s["joy_band"])
        if self.check_downlink:
            key["streams"] = tuple(
                (st["view"], st["src_publishers"], st["view_publishers"], st["band"])
                for st in s["streams"]
            )
        if self.check_rover:
            key["rover"] = (s["tf_ok"], s["joint_states_ok"],
                            s["robot_description_ok"], s["tf_publishers"])
        if self.check_arm:
            arm = s["arm"]
            key["arm"] = (arm["controller"], arm["joints_ok"],
                          tuple(arm["missing_joints"]), arm["move_group"],
                          arm["servo"], arm["joystick_ok"],
                          arm["joystick_status"])
        if self.check_gripper:
            gripper = s["gripper"]
            key["gripper"] = (gripper["controller"], gripper["joints_ok"],
                              tuple(gripper["missing_joints"]))
        if self.check_drive:
            drive = s["drive"]
            # command_age_s and the bus voltages are deliberately absent: they
            # move on every tick and would reprint the whole report forever.
            key["drive"] = (
                drive["status_ok"], drive["status_publishers"], drive["armed"],
                drive["enable_requested"], drive["can_link"],
                tuple(drive["pending_axes"]), drive["command_valid"],
                drive["mock_ok"], drive["cmd_vel_ok"], drive["cmd_vel_moving"],
                drive["messages_available"],
                tuple(
                    (a["advertised"], a["ctrl_fresh"], a["odrv_fresh"],
                     a["closed_loop"], a["faulted"])
                    for a in drive["axes"]
                ),
            )
        if self.check_imu:
            key["imu"] = (s["imu"]["ok"], s["imu"]["publishers"],
                          s["imu"]["frame_id"])
        if self.check_viewers:
            key["viewers"] = s["viewers"]
        return key

    # ── Main status check ─────────────────────────────────────────────────────

    def _check_status(self):
        snapshot = self._make_snapshot()
        key = self._snapshot_key(snapshot)

        if (
            not self.initial_check_done
            or key != self.prev_snapshot
            or self.manual_check_requested
            or not self.print_only_on_change
        ):
            self.manual_check_requested = False
            self._print_status(snapshot)
            self.prev_snapshot = key
            self.initial_check_done = True

    def _print_status(self, s):
        W = 74
        G, R, Y, C = self.GREEN, self.RED, self.YELLOW, self.CYAN
        B, RST, GREY = self.BOLD, self.RESET, self.GREY

        problems = []
        warnings = []

        print(f"\n{'═'*W}", flush=True)
        print(f"{B}{self.BLUE}  ARIES BASE STATION STATUS CHECK{RST}", flush=True)
        print(f"{'═'*W}", flush=True)

        # ── Field link ────────────────────────────────────────────────────────
        if self.check_link:
            print(f"\n{B}  Field Link:{RST}", flush=True)
            dds = s["dds"]

            if s["local_address"] and s["local_host_name"]:
                print(
                    f"  {G}✓{RST} this machine — {G}{s['local_host_name']} "
                    f"at {s['local_address']}{RST}",
                    flush=True,
                )
            elif s["local_address"]:
                # Not in the hosts table: comms.py guessed, and a guess on a
                # shared field can land on a stranger's DHCP lease.
                print(
                    f"  {Y}~{RST} this machine — {Y}{s['local_address']}, which is "
                    f"not a configured host{RST}",
                    flush=True,
                )
                warnings.append(
                    "this machine's identity was GUESSED; run "
                    "./scripts/setup_field_link.sh base"
                )
            else:
                print(f"  {R}✗{RST} this machine — {R}no interface on the field link{RST}",
                      flush=True)
                problems.append(
                    "not on the field link — check the antenna cable and "
                    "`ip -4 -br addr`"
                )

            if dds["domain_ok"]:
                print(f"  {G}✓{RST} ROS_DOMAIN_ID — {G}{dds['domain']}{RST}", flush=True)
            else:
                shown = dds["domain"] or "unset (= 0)"
                print(
                    f"  {R}✗{RST} ROS_DOMAIN_ID — {R}{shown}, rover is on "
                    f"{self.domain_expected}{RST}",
                    flush=True,
                )
                problems.append(
                    f"domain {shown} != {self.domain_expected}: this process "
                    "started before the DDS environment was set"
                )

            if dds["rmw_ok"]:
                print(f"  {G}✓{RST} RMW — {G}{dds['rmw']}{RST}", flush=True)
            else:
                shown = dds["rmw"] or "unset (= rmw_fastrtps_cpp)"
                print(f"  {R}✗{RST} RMW — {R}{shown}, rover uses {comms.RMW}{RST}",
                      flush=True)
                problems.append(f"RMW {shown} != {comms.RMW}: the two never match")

            if dds["uri_readable"] is None:
                print(f"  {R}✗{RST} CYCLONEDDS_URI — {R}not set{RST}", flush=True)
                problems.append("no CYCLONEDDS_URI: discovery is multicast, which "
                                "the airMAX link sends at its lowest rate")
            elif dds["uri_readable"] is False:
                print(f"  {R}✗{RST} CYCLONEDDS_URI — {R}{dds['uri']} cannot be read{RST}",
                      flush=True)
                problems.append("the Cyclone config file is unreadable; Cyclone "
                                "treats that as FATAL, not a warning")
            elif not dds["pinned_ok"]:
                print(
                    f"  {R}✗{RST} Cyclone interface — {R}pinned to {dds['pinned']}, "
                    f"this machine has {s['local_address']}{RST}",
                    flush=True,
                )
                problems.append(
                    f"the config on disk now pins {dds['pinned']}: something "
                    "rewrote it after this stack started"
                )
            else:
                where = dds["pinned"] or "unpinned (local-only config)"
                print(f"  {G}✓{RST} Cyclone interface — {G}{where}{RST}", flush=True)

            for label, reachable in sorted(s["ping"].items()):
                kind, _, name = label.partition(":")
                address = (self.hosts if kind == "host" else self.radios).get(name, "?")
                if reachable is True:
                    print(f"  {G}✓{RST} {kind} {name} — {G}{address} reachable{RST}",
                          flush=True)
                elif reachable is False:
                    print(f"  {R}✗{RST} {kind} {name} — {R}{address} no reply{RST}",
                          flush=True)
                    if kind == "host":
                        problems.append(
                            f"{name} ({address}) does not answer: the link is down "
                            "below DDS. ./scripts/watch_field_link.sh names the layer"
                        )
                    else:
                        warnings.append(f"radio {name} ({address}) does not answer")
                else:
                    print(f"  {GREY}○{RST} {kind} {name} — {address} not probed yet",
                          flush=True)

        # ── Operator input ────────────────────────────────────────────────────
        if self.check_joystick:
            print(f"\n{B}  Operator Input:{RST}", flush=True)

            if not self.expect_local_joy:
                print(f"  {GREY}○{RST} pad — read on the rover (expect_local_joy false)",
                      flush=True)
            elif s["joy_device_present"]:
                print(f"  {G}✓{RST} {self.joystick_device} — {G}present{RST}", flush=True)
            else:
                print(f"  {R}✗{RST} {self.joystick_device} — {R}not enumerated{RST}",
                      flush=True)
                problems.append(
                    f"no pad at {self.joystick_device}; the rover has no other "
                    "source of /joy and stops after 0.35 s"
                )

            publishers = s["joy_publishers"]
            joy_hz = s["joy_hz"]
            if publishers == 0:
                print(f"  {R}✗{RST} /joy — {R}no publisher{RST}", flush=True)
                problems.append("/joy has no publisher: nothing can be driven")
            elif publishers > 1:
                # The failure this is here for: it looks like a working pad.
                print(
                    f"  {R}✗{RST} /joy — {R}{publishers} publishers, "
                    f"{joy_hz:.0f} Hz{RST}",
                    flush=True,
                )
                problems.append(
                    f"{publishers} publishers on /joy — the consumers see the pads "
                    "interleaved at double rate and buttons appear to chatter. "
                    "Exactly one machine may set use_joy_node:=true"
                )
            elif s["joy_band"] == "none":
                print(f"  {R}✗{RST} /joy — {R}1 publisher, no messages{RST}", flush=True)
                problems.append(
                    "/joy is advertised but silent: check the pad, or the QoS if "
                    "the publisher is on the rover"
                )
            elif s["joy_band"] == "low":
                print(
                    f"  {Y}~{RST} /joy — {Y}{joy_hz:.0f} Hz, expected "
                    f"~{self.expected_joy_hz:.0f}{RST}",
                    flush=True,
                )
                warnings.append(
                    f"/joy at {joy_hz:.0f} Hz: autorepeat_rate is 80, so a lower "
                    "rate here is the link dropping packets"
                )
            else:
                print(f"  {G}✓{RST} /joy — {G}1 publisher, {joy_hz:.0f} Hz{RST}",
                      flush=True)

            if self.expect_local_joy and s["joy_raw_publishers"] == 0:
                print(f"  {Y}~{RST} /joy/raw — {Y}no local joy driver{RST}", flush=True)
                warnings.append(
                    "no publisher on /joy/raw: the driver is not running here, so "
                    "whatever is on /joy came from somewhere else"
                )

        # ── Downlink ──────────────────────────────────────────────────────────
        if self.check_downlink:
            print(f"\n{B}  Camera Downlink  {GREY}(measured on the local "
                  f"/<cam>/view/*, never on /downlink/*){RST}{B}:{RST}", flush=True)
            for st in s["streams"]:
                label = f"{st['camera']} {st['stream']}"
                if st["src_publishers"] == 0 and st["view_publishers"] == 0:
                    print(
                        f"  {R}✗{RST} {label:<26} — {R}rover is not sending, and no "
                        f"decompressor here{RST}",
                        flush=True,
                    )
                    problems.append(
                        f"{label}: nothing publishes {st['src']} and nothing "
                        f"publishes {st['view']}"
                    )
                elif st["src_publishers"] == 0:
                    print(
                        f"  {R}✗{RST} {label:<26} — {R}decompressor up, rover is not "
                        f"sending{RST}",
                        flush=True,
                    )
                    problems.append(
                        f"{label}: no publisher on {st['src']} — the rover is not "
                        "running the downlink, or that camera did not start"
                    )
                elif st["view_publishers"] == 0:
                    print(
                        f"  {R}✗{RST} {label:<26} — {R}rover is sending, no "
                        f"decompressor here{RST}",
                        flush=True,
                    )
                    problems.append(
                        f"{label}: nothing publishes {st['view']} — start "
                        "camera_view, or add the camera to `cameras`"
                    )
                elif st["band"] == "none":
                    print(
                        f"  {R}✗{RST} {label:<26} — {R}both ends up, no frames "
                        f"arriving{RST}",
                        flush=True,
                    )
                    problems.append(
                        f"{label}: publisher and decompressor both present but no "
                        "frames. QoS mismatch, or the codec package is missing"
                    )
                elif st["band"] == "low":
                    print(
                        f"  {Y}~{RST} {label:<26} — {Y}{st['hz']:.1f} Hz, expected "
                        f"~{st['expected']:.0f}{RST}",
                        flush=True,
                    )
                    warnings.append(
                        f"{label} at {st['hz']:.1f} Hz of ~{st['expected']:.0f}: the "
                        "link is dropping frames — downlink_profile:=lean"
                    )
                else:
                    print(f"  {G}✓{RST} {label:<26} — {G}{st['hz']:.1f} Hz{RST}",
                          flush=True)

        # ── Rover ─────────────────────────────────────────────────────────────
        if self.check_rover:
            print(f"\n{B}  Rover State Stream  {GREY}(model, TF, joints){RST}{B}:{RST}",
                  flush=True)

            if s["tf_ok"]:
                print(f"  {G}✓{RST} /tf — {G}fresh, {s['tf_publishers']} publisher(s){RST}",
                      flush=True)
            elif s["tf_publishers"]:
                print(f"  {R}✗{RST} /tf — {R}advertised but stale/silent{RST}", flush=True)
                problems.append("/tf is advertised and not arriving: QoS or a "
                                "half-open link")
            else:
                print(f"  {R}✗{RST} /tf — {R}no publisher{RST}", flush=True)
                problems.append(
                    "no /tf at all: the rover stack is down, or this machine is "
                    "not on its domain"
                )

            if s["joint_states_ok"]:
                print(f"  {G}✓{RST} /joint_states — {G}fresh{RST}", flush=True)
            else:
                print(f"  {R}✗{RST} /joint_states — {R}stale or absent{RST}", flush=True)
                problems.append("/joint_states is not arriving: the model in RViz "
                                "will not move")

            if s["robot_description_ok"]:
                print(f"  {G}✓{RST} /robot_description — {G}received (latched){RST}",
                      flush=True)
            else:
                # Not fatal here: base_station.launch.py builds its own
                # description from this workspace, which is why RViz can render
                # the arm at all. Worth saying because it is the cheapest
                # evidence that reliable+transient_local traffic crosses.
                print(f"  {Y}~{RST} /robot_description — {Y}not received{RST}",
                      flush=True)
                warnings.append(
                    "/robot_description not received; RViz uses the local build, "
                    "so check finger_type matches the rover by eye"
                )

        # ── Rover arm ─────────────────────────────────────────────────────────
        # Everything below judges the ROVER from this end. A subsystem with no
        # publisher at all is a warning, not a problem: the operator may have
        # launched a drive-only rover on purpose, and this end has no way to
        # know. Half up -- advertised and silent -- is the problem.
        if self.check_arm:
            print(f"\n{B}  Rover Arm, as seen from here:{RST}", flush=True)
            arm = s["arm"]

            if arm["controller"]:
                print(f"  {G}✓{RST} {ARM_CONTROLLER} — {G}running on the rover{RST}",
                      flush=True)
            else:
                print(f"  {Y}○{RST} {ARM_CONTROLLER} — {Y}not running{RST}", flush=True)
                warnings.append(
                    "no arm trajectory controller on the rover: the arm was not "
                    "started, or ros2_control did not activate it"
                )

            if arm["joints_ok"]:
                print(
                    f"  {G}✓{RST} arm joints — {G}all {len(self.arm_joint_names)} "
                    f"fresh in /joint_states{RST}",
                    flush=True,
                )
            elif arm["controller"] or s.get("joint_stream_ok"):
                # The rover is publishing joints and these are not among them:
                # the broadcaster came up without the arm's state interfaces,
                # which is a real fault rather than a subsystem left off.
                missing = ", ".join(arm["missing_joints"][:3])
                suffix = "..." if len(arm["missing_joints"]) > 3 else ""
                print(f"  {R}✗{RST} arm joints — {R}missing {missing}{suffix}{RST}",
                      flush=True)
                problems.append(
                    f"{len(arm['missing_joints'])} arm joint(s) absent from a "
                    "/joint_states that IS arriving: the arm's hardware "
                    "interface did not come up, so RViz shows it parked"
                )
            else:
                print(f"  {Y}○{RST} arm joints — {Y}no /joint_states from the rover{RST}",
                      flush=True)

            if arm["move_group"]:
                print(f"  {G}✓{RST} move_group — {G}planning available{RST}", flush=True)
            else:
                print(f"  {Y}○{RST} move_group — {Y}not running{RST}", flush=True)
                warnings.append(
                    "no move_group across the link: RViz's MotionPlanning panel "
                    "here has nothing to plan with"
                )

            if arm["servo"]:
                print(f"  {G}✓{RST} MoveIt Servo — {G}running{RST}", flush=True)
            else:
                print(f"  {GREY}○{RST} MoveIt Servo — not running (servo teleop off)",
                      flush=True)

            if arm["joystick_ok"]:
                print(f"  {G}✓{RST} arm joystick — {G}{arm['joystick_status']}{RST}",
                      flush=True)
            elif s.get("joy_band") == "ok" and arm["controller"]:
                # The pad is being read HERE and the arm is up THERE, so the
                # teleop node in between is the missing piece -- and it is the
                # one that turns the operator's sticks into arm motion.
                print(f"  {R}✗{RST} arm joystick — {R}no status across the link{RST}",
                      flush=True)
                problems.append(
                    "/joy is healthy and the arm controller is up, but nothing "
                    "publishes /arm_joystick/status: the arm teleop node is not "
                    "running on the rover, so the pad moves nothing"
                )
            else:
                print(f"  {Y}○{RST} arm joystick — {Y}no status{RST}", flush=True)

        # ── Rover gripper ─────────────────────────────────────────────────────
        if self.check_gripper:
            print(f"\n{B}  Rover Gripper:{RST}", flush=True)
            gripper = s["gripper"]

            if gripper["controller"]:
                print(f"  {G}✓{RST} {GRIPPER_CONTROLLER} — {G}running on the rover{RST}",
                      flush=True)
            else:
                print(f"  {Y}○{RST} {GRIPPER_CONTROLLER} — {Y}not running{RST}",
                      flush=True)
                warnings.append("no gripper controller on the rover")

            if gripper["joints_ok"]:
                print(f"  {G}✓{RST} gripper joint — {G}fresh in /joint_states{RST}",
                      flush=True)
            elif gripper["controller"] or s.get("joint_stream_ok"):
                missing = ", ".join(gripper["missing_joints"])
                print(f"  {R}✗{RST} gripper joint — {R}missing {missing}{RST}",
                      flush=True)
                problems.append(
                    f"gripper joint(s) {missing} absent from a /joint_states that "
                    "IS arriving: the Teensy resolved to mock, or the controller "
                    "failed to activate"
                )
            else:
                print(f"  {Y}○{RST} gripper joint — {Y}no /joint_states from the rover{RST}",
                      flush=True)

        # ── Rover drive ───────────────────────────────────────────────────────
        if self.check_drive:
            print(f"\n{B}  Rover Drive  {GREY}(the bridge's own 2 Hz report){RST}{B}:{RST}",
                  flush=True)
            drive = s["drive"]

            if drive["status_ok"]:
                # The CAN row, relayed. full_hardware_checker reads the adapter
                # directly; from here the bridge's own view of it is the only
                # honest source, and it is the one that decides whether to arm.
                can_link = drive["can_link"] or "not reported"
                if str(can_link).startswith("up ("):
                    print(f"  {G}✓{RST} CAN link — {G}{can_link}{RST}", flush=True)
                else:
                    print(f"  {R}✗{RST} CAN link — {R}{can_link}{RST}", flush=True)
                    problems.append(
                        f"the rover reports its CAN link as '{can_link}' — the "
                        "wheels cannot be commanded until that is fixed ON the rover"
                    )

                pending = drive["pending_axes"]
                if drive["armed"]:
                    print(f"  {G}✓{RST} drive bridge — {G}ARMED{RST}", flush=True)
                elif pending:
                    # The failure this row exists for: the bridge arms only when
                    # ALL six axes reach CLOSED_LOOP, so one bad encoder stops
                    # the rover while every teleop topic still ticks normally.
                    listed = ", ".join(
                        f"{i} ({AXIS_LABELS.get(i, '?').strip()})" for i in pending
                    )
                    print(f"  {R}✗{RST} drive bridge — {R}disarmed, waiting on axes "
                          f"{listed}{RST}", flush=True)
                    problems.append(
                        f"the bridge is held disarmed by axis {listed}: every wheel "
                        "stays dead until that axis reaches CLOSED_LOOP"
                    )
                elif drive["enable_requested"]:
                    print(f"  {R}✗{RST} drive bridge — {R}arm requested, not armed{RST}",
                          flush=True)
                    problems.append(
                        "the bridge was asked to arm and has not: the ODrives are "
                        "not answering on CAN"
                    )
                else:
                    print(f"  {Y}○{RST} drive bridge — {Y}disarmed (press LB+Y){RST}",
                          flush=True)

                if drive["command_valid"]:
                    print(f"  {G}✓{RST} drive command — {G}valid and fresh{RST}",
                          flush=True)
                else:
                    print(f"  {GREY}○{RST} drive command — none within the bridge's "
                          f"timeout (sticks centred)", flush=True)
            elif drive["status_publishers"]:
                print(f"  {R}✗{RST} {self.drive_status_topic} — {R}advertised, not "
                      f"arriving{RST}", flush=True)
                problems.append(
                    f"{self.drive_status_topic} has a publisher and no messages "
                    "reach here: QoS, or the link is half open"
                )
            elif drive["mock_ok"]:
                print(f"  {Y}~{RST} drive bridge — {Y}absent; mock_rover_drive is "
                      f"running instead{RST}", flush=True)
                warnings.append(
                    "the rover is on mock_rover_drive: nothing it is told to do "
                    "reaches a wheel"
                )
            else:
                print(f"  {Y}○{RST} drive bridge — {Y}not running on the rover{RST}",
                      flush=True)
                warnings.append(
                    "no cmd_vel_odrive_bridge across the link: the rover cannot "
                    "be driven"
                )

            # The round trip, and the half of it this machine cannot see any
            # other way: /joy leaves here, /cmd_vel is what came back.
            if drive["cmd_vel_moving"]:
                print(f"  {G}✓{RST} /cmd_vel — {G}nonzero command, the pad is "
                      f"reaching the rover{RST}", flush=True)
            elif drive["cmd_vel_ok"]:
                print(f"  {G}✓{RST} /cmd_vel — {G}arriving, currently zero{RST}",
                      flush=True)
            elif drive["cmd_vel_publishers"]:
                print(f"  {Y}~{RST} /cmd_vel — {Y}advertised, silent{RST}", flush=True)
            elif s.get("joy_band") == "ok":
                print(f"  {R}✗{RST} /cmd_vel — {R}no publisher, though /joy is "
                      f"healthy{RST}", flush=True)
                problems.append(
                    "the pad is fine here and nothing publishes /cmd_vel on the "
                    "rover: its teleop node is down, or it never saw /joy — which "
                    "is a one-way link, not a dead pad"
                )
            else:
                print(f"  {Y}○{RST} /cmd_vel — {Y}no publisher{RST}", flush=True)

            # ── ODrive axes ───────────────────────────────────────────────────
            if not drive["messages_available"]:
                print(
                    f"\n{B}  ODrive Axes:{RST}\n"
                    f"  {Y}○{RST} {Y}odrive_can messages are not built on this "
                    f"machine — the axis table cannot be read{RST}",
                    flush=True,
                )
                warnings.append(
                    "odrive_can is not built here, so the per-axis table is blank. "
                    "That is a missing package at THIS end, not six dead axes"
                )
            else:
                print(f"\n{B}  ODrive Axes  {GREY}({AXIS_HEADER}){RST}{B}:{RST}",
                      flush=True)
                silent = []
                for axis in drive["axes"][:self.expected_axes]:
                    i = axis["index"]
                    label = AXIS_LABELS.get(i, f"axis {i}  ")

                    if not axis["advertised"]:
                        print(f"  {Y}~{RST} Axis {i}  ({label})  — {Y}CAN node not "
                              f"running{RST}", flush=True)
                        continue
                    if not axis["ctrl_fresh"]:
                        print(f"  {Y}~{RST} Axis {i}  ({label})  — {Y}advertised, no "
                              f"heartbeat here{RST}", flush=True)
                        silent.append(i)
                        continue

                    state = axis_state_name(self.ctrl_states[i])
                    vel = self.ctrl_vel[i]
                    volts = self.odrv_voltage[i]
                    detail = (
                        f"{state}  "
                        f"{'vel:%+.3f' % vel if vel is not None else 'vel:---'}  "
                        f"{'%.1fV' % volts if volts is not None else '---V'}"
                    )

                    if axis["faulted"]:
                        ctrl_e = self.ctrl_errors[i] or 0
                        drv_e = self.odrv_errors[i] or 0
                        codes = " ".join(
                            part for part in (
                                f"ctrl:{format_odrive_error(ctrl_e)}" if ctrl_e else "",
                                f"drv:{format_odrive_error(drv_e)}" if drv_e else "",
                            ) if part
                        )
                        print(f"  {R}✗{RST} Axis {i}  ({label})  — {R}{detail}  "
                              f"{codes}{RST}", flush=True)
                        problems.append(f"axis {i} ({label.strip()}) fault: {codes}")
                    elif not axis["closed_loop"]:
                        print(f"  {Y}~{RST} Axis {i}  ({label})  — {Y}{detail}{RST}",
                              flush=True)
                    else:
                        print(f"  {G}✓{RST} Axis {i}  ({label})  — {G}{detail}{RST}",
                              flush=True)

                # The cross-check the two sources exist for. The rover said it
                # is armed, which it only does once every axis answered THERE,
                # so axes missing HERE are the link losing them and nothing else.
                if silent and drive["status_ok"] and drive["armed"]:
                    problems.append(
                        f"axes {', '.join(str(i) for i in silent)} are silent here "
                        "while the rover reports itself ARMED — those messages are "
                        "being dropped on the link, the drive is fine"
                    )
                elif silent and drive["status_ok"] and drive["can_link"] and \
                        str(drive["can_link"]).startswith("up ("):
                    warnings.append(
                        f"axes {', '.join(str(i) for i in silent)} advertised and "
                        "silent on a CAN link the rover calls up: either the poller "
                        "is down or the link is dropping their 5 Hz status"
                    )

        # ── Rover IMU ─────────────────────────────────────────────────────────
        if self.check_imu:
            print(f"\n{B}  Rover IMU:{RST}", flush=True)
            imu = s["imu"]
            if imu["ok"]:
                frame = f" frame={imu['frame_id']}" if imu["frame_id"] else ""
                print(f"  {G}✓{RST} {self.imu_topic} — {G}fresh{frame}{RST}", flush=True)
            elif imu["publishers"]:
                print(f"  {R}✗{RST} {self.imu_topic} — {R}advertised, not arriving{RST}",
                      flush=True)
                problems.append(
                    f"{self.imu_topic} has a publisher and no data reaches here: "
                    "the driver is up and the device stopped, or the link is "
                    "dropping it"
                )
            else:
                # Not a fault by itself: use_imu:=false is a supported run and
                # the EKF falls back to wheel odometry.
                print(f"  {Y}○{RST} {self.imu_topic} — {Y}no publisher{RST}", flush=True)
                warnings.append(
                    "no IMU across the link: the rover is on the wheel-odometry "
                    "fallback, so heading will drift"
                )

        # ── Viewers ───────────────────────────────────────────────────────────
        if self.check_viewers:
            print(f"\n{B}  Viewers:{RST}", flush=True)
            viewers = s["viewers"]
            if viewers == 1:
                print(f"  {G}✓{RST} RViz — {G}1 running{RST}", flush=True)
            elif viewers == 0:
                print(f"  {Y}○{RST} RViz — none running", flush=True)
            elif viewers > 1:
                print(f"  {Y}~{RST} RViz — {Y}{viewers} running{RST}", flush=True)
                warnings.append(
                    f"{viewers} RViz instances: each one renders and each one "
                    "subscribes, so they compete for the GPU. Started by hand?"
                )

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'═'*W}", flush=True)
        if not problems:
            print(f"  {G}{B}✓  BASE STATION READY{RST}", flush=True)
        else:
            print(f"  {R}{B}✗  BASE STATION NOT READY{RST}", flush=True)
        for problem in problems:
            print(f"  {R}→  {problem}{RST}", flush=True)
        for warning in warnings:
            print(f"  {Y}⚠   {warning}{RST}", flush=True)
        if problems:
            print(
                f"  {C}→  bottom up: ./scripts/setup_field_link.sh --check, then "
                f"the DDS environment, then QoS{RST}",
                flush=True,
            )
        # What is still invisible from this end, said once so nobody reads a
        # clean report as a clean robot. Serial ports, the CAN adapter and the
        # USB tree are physical facts about the far end of the link.
        if self.check_arm or self.check_gripper or self.check_drive:
            print(
                f"  {GREY}·  topics only; the Teensy, the CAN adapter and the "
                f"RealSense USB tree are on the rover — full_hardware_checker "
                f"there reads those{RST}",
                flush=True,
            )
        print(f"{'═'*W}\n", flush=True)

    def destroy_node(self):
        if self.pinger:
            self.pinger.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BaseStationChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl-C during launch shutdown arrives a second time, inside
        # destroy_node's own teardown, and the traceback it prints looks like a
        # crash in the checker. It is not, and a status tool that appears to
        # die on every exit is a tool people stop trusting.
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
