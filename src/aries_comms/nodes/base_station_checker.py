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

Manual check:
  ros2 service call /check_base_station std_srvs/srv/Trigger
"""

import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, JointState, Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from aries_common import comms

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

        # Comma-separated, not a list: a launch file cannot build a
        # STRING_ARRAY out of a LaunchConfiguration, and handing a STRING to a
        # list-declared parameter kills the node at startup with
        # InvalidParameterTypeException -- so it reports nothing at all, which
        # is the one thing a checker must never do. _as_list splits either.
        self.declare_parameter("cameras", "gripper_camera,rover_camera,rear_camera")
        self.declare_parameter("color_only", "rear_camera")
        self.declare_parameter("expected_color_hz", 15.0)
        self.declare_parameter("expected_depth_hz", 5.0)

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
        self.robot_description_seen = False

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
                JointState, "/joint_states", self._joint_state_cb, SAMPLE_QOS
            )
            self.create_subscription(
                String, "/robot_description", self._description_cb, LATCHED_QOS
            )

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

    def _description_cb(self, msg):
        self.robot_description_seen = True

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
            print(f"\n{B}  Rover, as seen from here:{RST}", flush=True)

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
