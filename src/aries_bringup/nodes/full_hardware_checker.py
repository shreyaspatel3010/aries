#!/usr/bin/env python3
"""
ARIES Full Hardware Status Checker Node

Style intentionally matches aries_bringup/nodes/hardware_checker.py.

Checks:
  • Arm TCP hardware reachability
  • Arm controller/joint-state activity
  • Gripper serial or mock fallback
  • Rover CAN / ODrive axes
  • Rover BNO055 or picoScan IMU, when forced or auto-detected
  • SICK picoScan LaserScan freshness
  • Mock rover fallback heartbeat
  • Joystick /joy
  • Optional RealSense USB detection and image stream activity

Manual check:
  ros2 service call /check_full_hardware std_srvs/srv/Trigger
"""

import glob
import os
import socket
from pathlib import Path

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, Imu, Joy, JointState, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    from odrive_can.msg import ControllerStatus, ODriveStatus
except Exception:
    ControllerStatus = None
    ODriveStatus = None


# ── ODrive axis_state values ─────────────────────────────────────────────────
AXIS_STATE_NAMES = {
    0: "UNDEFINED",
    1: "IDLE",
    2: "STARTUP_SEQ",
    3: "FULL_CALIBRATION",
    4: "MOTOR_CALIBRATION",
    6: "ENCODER_OFFSET_CAL",
    7: "ENCODER_INDEX_SEARCH",
    8: "CLOSED_LOOP",
    9: "LOCKIN_SPIN",
    10: "ENCODER_DIR_FIND",
    11: "HOMING",
    12: "ENCODER_HALL_PHASE_CAL",
    13: "ENCODER_HALL_POLARITY_CAL",
}
CLOSED_LOOP = 8

AXIS_LABELS = {
    0: "Right-Front",
    1: "Right-Mid  ",
    2: "Right-Rear ",
    3: "Left-Front ",
    4: "Left-Mid   ",
    5: "Left-Rear  ",
}
NUM_AXES = 6


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


class FullHardwareChecker(Node):
    def __init__(self):
        super().__init__("full_hardware_checker")

        # ── ANSI colours, same style as rover hardware_checker.py ─────────────
        self.GREEN  = "\033[92m"
        self.RED    = "\033[91m"
        self.YELLOW = "\033[93m"
        self.BLUE   = "\033[94m"
        self.CYAN   = "\033[96m"
        self.RESET  = "\033[0m"
        self.BOLD   = "\033[1m"
        self.GREY   = "\033[90m"

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("check_interval", 3.0)
        self.declare_parameter("timeout", 5.0)
        self.declare_parameter("print_only_on_change", True)

        self.declare_parameter("check_arm", True)
        self.declare_parameter("check_gripper", True)
        self.declare_parameter("check_rover", True)
        self.declare_parameter("check_imu", True)
        self.declare_parameter("check_joystick", True)
        self.declare_parameter("check_realsense", True)
        self.declare_parameter("check_mock_fallbacks", True)

        self.declare_parameter("require_all_rover_axes", True)
        self.declare_parameter("require_closed_loop", True)
        self.declare_parameter("check_odrive_status", True)

        self.declare_parameter("arm_host", "192.168.3.11")
        self.declare_parameter("arm_port", 3920)
        self.declare_parameter("arm_socket_timeout", 0.25)
        self.declare_parameter("arm_joint_names", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])

        self.declare_parameter(
            "gripper_serial_port",
            "/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00",
        )
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("use_imu", "auto")
        self.declare_parameter("imu_port", "/dev/ttyUSB0")
        self.declare_parameter("ybimu_port", "/dev/imu_ybimu")
        self.declare_parameter("imu_topic", "/bno055/imu")
        self.declare_parameter("ybimu_topic", "/ybimu/imu")
        self.declare_parameter("picoscan_imu_topic", "/picoscan/imu")
        self.declare_parameter("imu_frame", "bno055")
        self.declare_parameter("ybimu_frame", "imu_frame")
        self.declare_parameter("lidar_topic", "/scan")
        self.declare_parameter("expected_odrive_axes", 6)
        self.declare_parameter("realsense_color_topic", "/gripper_camera/color/image_raw")

        check_interval = float(self.get_parameter("check_interval").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.print_only_on_change = _as_bool(self.get_parameter("print_only_on_change").value)

        self.check_arm = _as_bool(self.get_parameter("check_arm").value)
        self.check_gripper = _as_bool(self.get_parameter("check_gripper").value)
        self.check_rover = _as_bool(self.get_parameter("check_rover").value)
        self.check_imu = _as_bool(self.get_parameter("check_imu").value)
        self.check_joystick = _as_bool(self.get_parameter("check_joystick").value)
        self.check_realsense = _as_bool(self.get_parameter("check_realsense").value)
        self.check_mock_fallbacks = _as_bool(self.get_parameter("check_mock_fallbacks").value)

        self.require_all_rover_axes = _as_bool(self.get_parameter("require_all_rover_axes").value)
        self.require_closed_loop = _as_bool(self.get_parameter("require_closed_loop").value)
        self.check_odrive_status = _as_bool(self.get_parameter("check_odrive_status").value)

        self.arm_host = str(self.get_parameter("arm_host").value)
        self.arm_port = int(self.get_parameter("arm_port").value)
        self.arm_socket_timeout = float(self.get_parameter("arm_socket_timeout").value)
        self.arm_joint_names = _as_list(self.get_parameter("arm_joint_names").value)
        self.gripper_serial_port = str(self.get_parameter("gripper_serial_port").value)
        self.can_interface = str(self.get_parameter("can_interface").value)
        self.use_imu = str(self.get_parameter("use_imu").value).strip().lower()
        self.imu_port = str(self.get_parameter("imu_port").value)
        self.ybimu_port = str(self.get_parameter("ybimu_port").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.ybimu_topic = str(self.get_parameter("ybimu_topic").value)
        self.picoscan_imu_topic = str(self.get_parameter("picoscan_imu_topic").value)
        self.imu_frame = str(self.get_parameter("imu_frame").value)
        self.ybimu_frame = str(self.get_parameter("ybimu_frame").value)
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.expected_odrive_axes = int(self.get_parameter("expected_odrive_axes").value)
        self.realsense_color_topic = str(self.get_parameter("realsense_color_topic").value)

        # ── State memory ──────────────────────────────────────────────────────
        self.ctrl_times  = [None] * NUM_AXES
        self.ctrl_states = [None] * NUM_AXES
        self.ctrl_errors = [None] * NUM_AXES
        self.ctrl_vel    = [None] * NUM_AXES

        self.odrv_times   = [None] * NUM_AXES
        self.odrv_voltage = [None] * NUM_AXES
        self.odrv_errors  = [None] * NUM_AXES

        self.joy_time = None
        self.joint_state_time = None
        self.joint_names = set()
        self.joint_times = {}
        self.mock_rover_time = None
        self.cmd_vel_time = None
        self.arm_joystick_time = None
        self.arm_joystick_status = "waiting for arm joystick status"
        self.realsense_color_time = None
        self.imu_time = None
        self.imu_frame_id = ""
        self.ybimu_time = None
        self.ybimu_frame_id = ""
        self.picoscan_imu_time = None
        self.picoscan_imu_frame_id = ""
        self.lidar_time = None
        self.lidar_frame_id = ""

        self.prev_snapshot = None
        self.initial_check_done = False
        self.manual_check_requested = False

        # ── QoS, same idea as hardware_checker.py ─────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── ODrive subscriptions ──────────────────────────────────────────────
        if ControllerStatus is not None:
            for i in range(NUM_AXES):
                ns = f"odrive_axis{i}"
                self.create_subscription(
                    ControllerStatus,
                    f"/{ns}/controller_status",
                    lambda msg, idx=i: self._ctrl_cb(msg, idx),
                    reliable_qos,
                )

        if ODriveStatus is not None:
            for i in range(NUM_AXES):
                ns = f"odrive_axis{i}"
                self.create_subscription(
                    ODriveStatus,
                    f"/{ns}/odrive_status",
                    lambda msg, idx=i: self._odrv_cb(msg, idx),
                    reliable_qos,
                )

        # ── General subscriptions ─────────────────────────────────────────────
        self.create_subscription(Joy, "/joy", self._joy_cb, sensor_qos)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, sensor_qos)
        self.create_subscription(String, "/mock_rover/status", self._mock_rover_cb, sensor_qos)
        self.create_subscription(String, "/arm_joystick/status", self._arm_joystick_cb, sensor_qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, sensor_qos)
        self.create_subscription(Imu, self.imu_topic, self._imu_cb, sensor_qos)
        self.create_subscription(
            Imu, self.ybimu_topic, self._ybimu_cb, sensor_qos
        )
        self.create_subscription(
            Imu, self.picoscan_imu_topic, self._picoscan_imu_cb, sensor_qos
        )
        self.create_subscription(LaserScan, self.lidar_topic, self._lidar_cb, sensor_qos)
        self.create_subscription(Image, self.realsense_color_topic, self._realsense_color_cb, sensor_qos)

        # ── Manual service ────────────────────────────────────────────────────
        self.create_service(Trigger, "/check_full_hardware", self._service_cb)

        # Backward friendly service name, useful if you are used to rover checker.
        self.create_service(Trigger, "/check_all_hardware", self._service_cb)

        self.create_timer(check_interval, self._check_status)

        self.get_logger().info(
            f"{self.BOLD}{self.BLUE}ARIES Full Hardware Checker Started{self.RESET}\n"
            f"{self.YELLOW}Manual check: "
            f"ros2 service call /check_full_hardware std_srvs/srv/Trigger{self.RESET}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _service_cb(self, request, response):
        self.manual_check_requested = True
        self._check_status()
        response.success = True
        response.message = "Full hardware status check triggered"
        return response

    def _ctrl_cb(self, msg, idx: int):
        self.ctrl_times[idx]  = self.get_clock().now()
        self.ctrl_states[idx] = getattr(msg, "axis_state", None)
        self.ctrl_errors[idx] = getattr(msg, "active_errors", None)
        self.ctrl_vel[idx]    = getattr(msg, "vel_estimate", None)

    def _odrv_cb(self, msg, idx: int):
        self.odrv_times[idx]   = self.get_clock().now()
        self.odrv_voltage[idx] = getattr(msg, "bus_voltage", None)
        self.odrv_errors[idx]  = getattr(msg, "active_errors", None)

    def _joy_cb(self, msg: Joy):
        self.joy_time = self.get_clock().now()

    def _joint_state_cb(self, msg: JointState):
        self.joint_state_time = self.get_clock().now()
        for name in msg.name:
            self.joint_names.add(name)
            self.joint_times[name] = self.joint_state_time

    def _mock_rover_cb(self, msg: String):
        self.mock_rover_time = self.get_clock().now()

    def _arm_joystick_cb(self, msg: String):
        self.arm_joystick_time = self.get_clock().now()
        self.arm_joystick_status = msg.data

    def _cmd_vel_cb(self, msg: Twist):
        if abs(msg.linear.x) > 1e-5 or abs(msg.angular.z) > 1e-5:
            self.cmd_vel_time = self.get_clock().now()

    def _imu_cb(self, msg: Imu):
        self.imu_time = self.get_clock().now()
        self.imu_frame_id = msg.header.frame_id

    def _ybimu_cb(self, msg: Imu):
        self.ybimu_time = self.get_clock().now()
        self.ybimu_frame_id = msg.header.frame_id

    def _picoscan_imu_cb(self, msg: Imu):
        self.picoscan_imu_time = self.get_clock().now()
        self.picoscan_imu_frame_id = msg.header.frame_id

    def _lidar_cb(self, msg: LaserScan):
        self.lidar_time = self.get_clock().now()
        self.lidar_frame_id = msg.header.frame_id

    def _realsense_color_cb(self, msg: Image):
        self.realsense_color_time = self.get_clock().now()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_recent(self, ts) -> bool:
        if ts is None:
            return False
        age = (self.get_clock().now().nanoseconds - ts.nanoseconds) / 1e9
        return age < self.timeout

    def _arm_tcp_reachable(self) -> bool:
        try:
            with socket.create_connection(
                (self.arm_host, self.arm_port),
                timeout=self.arm_socket_timeout,
            ):
                return True
        except OSError:
            return False

    def _gripper_serial_present(self) -> bool:
        if Path(self.gripper_serial_port).exists():
            return True
        if glob.glob("/dev/serial/by-id/*Teensy*"):
            return True
        if glob.glob("/dev/ttyACM*"):
            return True
        if glob.glob("/dev/ttyUSB*"):
            return True
        return False

    def _can_present(self) -> bool:
        return Path(f"/sys/class/net/{self.can_interface}").exists()

    def _imu_port_present(self) -> bool:
        return Path(self.imu_port).exists()

    def _ybimu_port_present(self) -> bool:
        return Path(self.ybimu_port).exists()

    def _bno055_package_present(self) -> bool:
        try:
            get_package_share_directory("bno055")
            return True
        except PackageNotFoundError:
            return False

    def _ybimu_package_present(self) -> bool:
        try:
            get_package_share_directory("ybimu_ros2")
            return True
        except PackageNotFoundError:
            return False

    def _imu_expected(self, imu_port_present: bool, bno055_package_present: bool) -> bool:
        if self.use_imu in ("auto", "detect"):
            return imu_port_present and bno055_package_present
        return _as_bool(self.use_imu)

    def _realsense_present(self) -> bool:
        for dev_path in glob.glob("/sys/bus/usb/devices/*/"):
            vendor_file = os.path.join(dev_path, "idVendor")
            product_file = os.path.join(dev_path, "idProduct")
            try:
                if open(vendor_file).read().strip() != "8086":
                    continue
                product = int(open(product_file).read().strip(), 16)
                if 0x0AD1 <= product <= 0x0B64:
                    return True
            except Exception:
                continue
        return False

    def _image_topics_with_publishers(self):
        image_topics = []
        for topic_name, topic_types in self.get_topic_names_and_types():
            if topic_name == self.realsense_color_topic:
                continue
            if "sensor_msgs/msg/Image" not in topic_types:
                continue
            if "camera" not in topic_name and "image_raw" not in topic_name:
                continue
            if self.count_publishers(topic_name) > 0:
                image_topics.append(topic_name)
        return sorted(image_topics)[:6]

    def _make_snapshot(self):
        recent_joint_names = {
            joint_name
            for joint_name, joint_time in self.joint_times.items()
            if self._is_recent(joint_time)
        }
        arm_joint_states_ok = self._is_recent(self.joint_state_time) and all(
            joint_name in recent_joint_names for joint_name in self.arm_joint_names
        )
        missing_arm_joints = [
            joint_name
            for joint_name in self.arm_joint_names
            if joint_name not in recent_joint_names
        ]

        axis_node_up = [
            self.count_publishers(f"/odrive_axis{i}/controller_status") > 0
            for i in range(NUM_AXES)
        ]
        axis_has_data = [self._is_recent(self.ctrl_times[i]) for i in range(NUM_AXES)]
        axis_has_odrv = [self._is_recent(self.odrv_times[i]) for i in range(NUM_AXES)]

        axis_cl = [
            self.ctrl_states[i] == CLOSED_LOOP if axis_has_data[i] else False
            for i in range(NUM_AXES)
        ]
        axis_err = [
            ((self.ctrl_errors[i] or 0) != 0 or (self.odrv_errors[i] or 0) != 0)
            for i in range(NUM_AXES)
        ]
        imu_port_present = self._imu_port_present() if self.check_imu else False
        ybimu_port_present = (
            self._ybimu_port_present() if self.check_imu else False
        )
        bno055_package_present = self._bno055_package_present() if self.check_imu else False
        ybimu_package_present = (
            self._ybimu_package_present() if self.check_imu else False
        )
        ybimu_publishers = self.count_publishers(self.ybimu_topic)
        picoscan_publishers = self.count_publishers(self.picoscan_imu_topic)
        lidar_publishers = self.count_publishers(self.lidar_topic)

        selected_imu = "none"
        selected_imu_topic = ""
        selected_imu_time = None
        selected_imu_frame = ""
        if self.check_imu and self.use_imu not in ("false", "0", "no", "off", "none", "odom_only", "wheel_odom"):
            if self.use_imu in ("picoscan", "picoscan150", "sick", "lidar"):
                selected_imu = "picoscan"
            elif self.use_imu in ("ybimu", "yaboom"):
                selected_imu = "ybimu"
            elif self.use_imu == "bno055":
                selected_imu = "bno055"
            elif ybimu_port_present and ybimu_package_present:
                selected_imu = "ybimu"
            elif imu_port_present and bno055_package_present:
                selected_imu = "bno055"
            elif ybimu_publishers > 0:
                selected_imu = "ybimu"
            elif picoscan_publishers > 0:
                selected_imu = "picoscan"

        if selected_imu == "ybimu":
            selected_imu_topic = self.ybimu_topic
            selected_imu_time = self.ybimu_time
            selected_imu_frame = self.ybimu_frame_id
        elif selected_imu == "bno055":
            selected_imu_topic = self.imu_topic
            selected_imu_time = self.imu_time
            selected_imu_frame = self.imu_frame_id
        elif selected_imu == "picoscan":
            selected_imu_topic = self.picoscan_imu_topic
            selected_imu_time = self.picoscan_imu_time
            selected_imu_frame = self.picoscan_imu_frame_id

        return {
            "arm_tcp": self._arm_tcp_reachable() if self.check_arm else False,
            "gripper_serial": self._gripper_serial_present() if self.check_gripper else False,
            "can_present": self._can_present() if self.check_rover else False,
            "imu_expected": selected_imu != "none",
            "selected_imu": selected_imu,
            "selected_imu_topic": selected_imu_topic,
            "imu_port_present": imu_port_present,
            "ybimu_port_present": ybimu_port_present,
            "bno055_package_present": bno055_package_present,
            "ybimu_package_present": ybimu_package_present,
            "imu_ok": self._is_recent(selected_imu_time) if self.check_imu else False,
            "imu_publishers": self.count_publishers(selected_imu_topic) if selected_imu_topic else 0,
            "imu_frame_id": selected_imu_frame,
            "lidar_ok": self._is_recent(self.lidar_time),
            "lidar_publishers": lidar_publishers,
            "lidar_frame_id": self.lidar_frame_id,
            "realsense": self._realsense_present() if self.check_realsense else False,
            "realsense_color_ok": self._is_recent(self.realsense_color_time),
            "realsense_color_publishers": self.count_publishers(self.realsense_color_topic),
            "other_image_topics": self._image_topics_with_publishers() if self.check_realsense else [],
            "joy_ok": self._is_recent(self.joy_time) if self.check_joystick else False,
            "joint_states_ok": self._is_recent(self.joint_state_time),
            "arm_joint_states_ok": arm_joint_states_ok,
            "missing_arm_joints": missing_arm_joints,
            "mock_rover_ok": self._is_recent(self.mock_rover_time),
            "cmd_vel_recent": self._is_recent(self.cmd_vel_time),
            "arm_joystick_ok": self._is_recent(self.arm_joystick_time),
            "arm_joystick_status": self.arm_joystick_status,

            "arm_controller_up": (
                self.count_publishers("/rebel_arm_trajectory_controller/state") > 0 or
                self.count_subscribers("/rebel_arm_trajectory_controller/joint_trajectory") > 0
            ),
            "gripper_controller_up": (
                self.count_publishers("/rebel_gripper_controller/state") > 0 or
                self.count_subscribers("/rebel_gripper_controller/joint_trajectory") > 0
            ),
            "move_group_up": (
                self.count_publishers("/move_action/_action/status") > 0 or
                self.count_publishers("/monitored_planning_scene") > 0
            ),
            "servo_up": (
                self.count_publishers("/servo_node/status") > 0 or
                self.count_subscribers("/delta_twist_cmds") > 0 or
                self.count_subscribers("/delta_joint_cmds") > 0
            ),

            "axis_node_up": axis_node_up,
            "axis_has_data": axis_has_data,
            "axis_has_odrv": axis_has_odrv,
            "axis_cl": axis_cl,
            "axis_err": axis_err,
        }

    # ── Main status check ─────────────────────────────────────────────────────

    def _check_status(self):
        snapshot = self._make_snapshot()

        status_changed = snapshot != self.prev_snapshot

        if (
            not self.initial_check_done or
            status_changed or
            self.manual_check_requested or
            not self.print_only_on_change
        ):
            self.manual_check_requested = False
            self._print_status(snapshot)
            self.prev_snapshot = snapshot
            self.initial_check_done = True

    def _print_status(self, s):
        W = 74
        G = self.GREEN
        R = self.RED
        Y = self.YELLOW
        C = self.CYAN
        B = self.BOLD
        RST = self.RESET

        axis_node_up = s["axis_node_up"]
        axis_has_data = s["axis_has_data"]
        axis_has_odrv = s["axis_has_odrv"]
        axis_cl = s["axis_cl"]
        axis_err = s["axis_err"]

        expected_axis_count = max(0, min(self.expected_odrive_axes, NUM_AXES))
        expected_axes = list(range(expected_axis_count))
        responding_axes = [i for i in expected_axes if axis_has_data[i]]

        rover_any_real = any(axis_node_up) or any(axis_has_data) or s["can_present"]
        rover_all_node_up = all(axis_node_up[i] for i in expected_axes) if expected_axes else False
        rover_all_responding = all(axis_has_data[i] for i in expected_axes) if expected_axes else False
        rover_all_odrv = all(axis_has_odrv[i] for i in expected_axes) if expected_axes else False
        rover_responding_odrv = all(axis_has_odrv[i] for i in responding_axes) if responding_axes else False
        rover_all_cl = (
            all(axis_cl[i] for i in responding_axes)
            if responding_axes else False
        )
        rover_any_err = any(axis_err[i] for i in expected_axes)

        if s["mock_rover_ok"]:
            rover_ready = True
        elif self.require_all_rover_axes:
            rover_ready = (
                rover_all_node_up and
                rover_all_responding and
                not rover_any_err and
                (rover_all_cl if self.require_closed_loop else True) and
                (rover_all_odrv if self.check_odrive_status else True)
            )
        else:
            rover_ready = (
                s["mock_rover_ok"] or
                (
                    bool(responding_axes) and
                    not rover_any_err and
                    (rover_all_cl if self.require_closed_loop else True) and
                    (rover_responding_odrv if self.check_odrive_status else True)
                )
            )

        arm_ready = s["arm_tcp"] or s["arm_controller_up"] or s["arm_joint_states_ok"]
        gripper_ready = s["gripper_serial"] or s["gripper_controller_up"]
        imu_ready = (not s["imu_expected"]) or s["imu_ok"]

        overall_ok = (
            (arm_ready if self.check_arm else True) and
            (gripper_ready if self.check_gripper else True) and
            (rover_ready if self.check_rover else True) and
            (imu_ready if self.check_imu else True)
        )

        print(f"\n{'═'*W}", flush=True)
        print(f"{B}{self.BLUE}  ARIES FULL HARDWARE STATUS CHECK{RST}", flush=True)
        print(f"{'═'*W}", flush=True)

        # ── Manipulator rows ──────────────────────────────────────────────────
        print(f"\n{B}  Manipulator / Arm:{RST}", flush=True)

        if s["arm_tcp"]:
            print(f"  {G}✓{RST} Arm TCP {self.arm_host}:{self.arm_port} — {G}REAL reachable{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} Arm TCP {self.arm_host}:{self.arm_port} — {Y}not reachable, mock/auto fallback expected{RST}", flush=True)

        if s["arm_controller_up"]:
            print(f"  {G}✓{RST} rebel_arm_trajectory_controller — {G}active/detected{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} rebel_arm_trajectory_controller — {Y}not detected yet{RST}", flush=True)

        if s["joint_states_ok"]:
            if s["arm_joint_states_ok"]:
                print(f"  {G}✓{RST} /joint_states — {G}fresh arm joints present{RST}", flush=True)
            else:
                missing = ", ".join(s["missing_arm_joints"][:3])
                suffix = "..." if len(s["missing_arm_joints"]) > 3 else ""
                print(
                    f"  {Y}~{RST} /joint_states — {Y}fresh, missing arm joints: {missing}{suffix}{RST}",
                    flush=True,
                )
        else:
            print(f"  {Y}~{RST} /joint_states — {Y}stale/not received yet{RST}", flush=True)

        if s["move_group_up"]:
            print(f"  {G}✓{RST} MoveGroup/RViz planner — {G}detected{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} MoveGroup/RViz planner — {Y}not detected yet{RST}", flush=True)

        if s["servo_up"]:
            print(f"  {G}✓{RST} MoveIt Servo — {G}detected for servo joystick mode{RST}", flush=True)

        if s.get("arm_joystick_ok", False):
            print(f"  {G}✓{RST} Arm joystick — {G}{s.get('arm_joystick_status', '')}{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} Arm joystick — {Y}{s.get('arm_joystick_status', 'waiting')}{RST}", flush=True)

        # ── Gripper rows ──────────────────────────────────────────────────────
        print(f"\n{B}  Gripper:{RST}", flush=True)

        if s["gripper_serial"]:
            print(f"  {G}✓{RST} Serial/Teensy — {G}present{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} Serial/Teensy — {Y}not found, mock/auto fallback expected{RST}", flush=True)

        if s["gripper_controller_up"]:
            print(f"  {G}✓{RST} rebel_gripper_controller — {G}active/detected{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} rebel_gripper_controller — {Y}not detected yet{RST}", flush=True)

        # ── Rover rows ────────────────────────────────────────────────────────
        print(f"\n{B}  Rover Backend:{RST}", flush=True)

        if s["can_present"]:
            print(f"  {G}✓{RST} CAN interface {self.can_interface} — {G}present{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} CAN interface {self.can_interface} — {Y}not present, mock rover expected{RST}", flush=True)

        if s["mock_rover_ok"]:
            print(f"  {G}✓{RST} mock_rover_drive — {G}active heartbeat{RST}", flush=True)
        elif self.check_mock_fallbacks and not rover_any_real:
            print(f"  {Y}~{RST} mock_rover_drive — {Y}not active yet{RST}", flush=True)

        if s["cmd_vel_recent"]:
            print(f"  {G}✓{RST} /cmd_vel — {G}recent nonzero command seen{RST}", flush=True)
        else:
            print(f"  {Y}~{RST} /cmd_vel — {Y}no recent nonzero command{RST}", flush=True)

        print(f"\n{B}  Rover IMU:{RST}", flush=True)

        if s["imu_expected"]:
            if s["selected_imu"] == "ybimu":
                if s["ybimu_port_present"]:
                    print(
                        f"  {G}✓{RST} YaBoom IMU port {self.ybimu_port} — "
                        f"{G}present{RST}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {R}✗{RST} YaBoom IMU port {self.ybimu_port} — "
                        f"{R}missing but ybimu selected{RST}",
                        flush=True,
                    )
                if not s["ybimu_package_present"]:
                    print(
                        f"  {R}✗{RST} ybimu_ros2 package — {R}not found{RST}",
                        flush=True,
                    )
            elif s["selected_imu"] == "bno055":
                if s["imu_port_present"]:
                    print(f"  {G}✓{RST} IMU port {self.imu_port} — {G}present{RST}", flush=True)
                else:
                    print(f"  {R}✗{RST} IMU port {self.imu_port} — {R}missing but BNO055 selected{RST}", flush=True)
                if not s["bno055_package_present"]:
                    print(f"  {R}✗{RST} bno055 package — {R}not found{RST}", flush=True)
            elif s["selected_imu"] == "picoscan":
                print(f"  {G}✓{RST} IMU source — {G}picoScan fallback selected{RST}", flush=True)

            if s["imu_ok"]:
                frame_detail = f" frame={s['imu_frame_id']}" if s["imu_frame_id"] else ""
                print(f"  {G}✓{RST} {s['selected_imu_topic']} — {G}fresh IMU data{frame_detail}{RST}", flush=True)
            elif s["imu_publishers"] > 0:
                print(
                    f"  {Y}~{RST} {s['selected_imu_topic']} — "
                    f"{Y}publisher present, no recent IMU messages{RST}",
                    flush=True,
                )
            else:
                print(f"  {R}✗{RST} {s['selected_imu_topic']} — {R}no publisher/data{RST}", flush=True)
        elif self.check_imu:
            mode = f"use_imu:={self.use_imu}"
            if s["ybimu_port_present"] or s["imu_port_present"]:
                print(
                    f"  {Y}~{RST} serial IMU present but not selected "
                    f"({mode}){RST}",
                    flush=True,
                )
            else:
                print(
                    f"  {Y}○{RST} IMU — not detected; "
                    f"wheel-odom EKF fallback expected ({mode})",
                    flush=True,
                )

        print(f"\n{B}  ODrive Axes  (right: 0-2 | left: 3-5):{RST}", flush=True)

        for i in range(NUM_AXES):
            label = AXIS_LABELS[i]

            if not axis_node_up[i]:
                print(f"  {Y}~{RST} Axis {i}  ({label})  — {Y}CAN NODE NOT RUNNING / using mock possible{RST}", flush=True)
                continue

            if not axis_has_data[i]:
                print(f"  {Y}~{RST} Axis {i}  ({label})  — {Y}NO HEARTBEAT  (hardware not responding){RST}", flush=True)
                continue

            state_name = AXIS_STATE_NAMES.get(self.ctrl_states[i], f"STATE_{self.ctrl_states[i]}")
            vel_str = f"vel:{self.ctrl_vel[i]:+.3f}" if self.ctrl_vel[i] is not None else "vel:---"
            volt_str = f"{self.odrv_voltage[i]:.1f}V" if self.odrv_voltage[i] is not None else "---V"

            ctrl_e = self.ctrl_errors[i] or 0
            drv_e = self.odrv_errors[i] or 0
            odrv_stale = self.check_odrive_status and not axis_has_odrv[i]

            if axis_err[i]:
                err_str = f"{R}ERR ctrl:0x{ctrl_e:08X}  drv:0x{drv_e:08X}{RST}"
                print(
                    f"  {R}✗{RST} Axis {i}  ({label})  — "
                    f"{Y}{state_name}{RST}  {volt_str}  {vel_str}  {err_str}",
                    flush=True,
                )
            elif axis_cl[i]:
                if odrv_stale:
                    print(
                        f"  {Y}~{RST} Axis {i}  ({label})  — "
                        f"{G}CLOSED_LOOP{RST}  {Y}ODRIVE STATUS STALE{RST}  {vel_str}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {G}✓{RST} Axis {i}  ({label})  — "
                        f"{G}CLOSED_LOOP{RST}  {volt_str}  {vel_str}",
                        flush=True,
                    )
            else:
                print(
                    f"  {Y}~{RST} Axis {i}  ({label})  — "
                    f"{Y}{state_name}{RST}  {volt_str}  {vel_str}",
                    flush=True,
                )

        # ── Sensors / Joystick rows ───────────────────────────────────────────
        print(f"\n{B}  Sensors / Operator Input:{RST}", flush=True)

        if s["joy_ok"]:
            print(f"  {G}✓{RST} /joy — {G}Connected{RST}", flush=True)
        elif self.check_joystick:
            print(f"  {Y}○{RST} /joy — Not detected yet", flush=True)

        if s["lidar_ok"]:
            frame_detail = f" frame={s['lidar_frame_id']}" if s["lidar_frame_id"] else ""
            print(f"  {G}✓{RST} {self.lidar_topic} — {G}fresh picoScan data{frame_detail}{RST}", flush=True)
        elif s["lidar_publishers"] > 0:
            print(f"  {Y}~{RST} {self.lidar_topic} — {Y}publisher present, no recent scans{RST}", flush=True)
        else:
            print(f"  {Y}○{RST} {self.lidar_topic} — picoScan not detected / optional", flush=True)

        if s["realsense"]:
            print(f"  {G}✓{RST} RealSense USB — {G}present{RST}", flush=True)
        elif self.check_realsense:
            print(f"  {Y}○{RST} RealSense USB — not detected / optional", flush=True)

        if self.check_realsense and not s["realsense_color_ok"]:
            if s["realsense_color_publishers"] > 0:
                print(
                    f"  {Y}~{RST} {self.realsense_color_topic} — "
                    f"{Y}publisher present, no recent frames{RST}",
                    flush=True,
                )
            elif s["other_image_topics"]:
                print(
                    f"  {Y}~{RST} {self.realsense_color_topic} — "
                    f"{Y}no publisher; image topics found: {', '.join(s['other_image_topics'])}{RST}",
                    flush=True,
                )
            elif s["realsense"]:
                print(
                    f"  {Y}~{RST} {self.realsense_color_topic} — "
                    f"{Y}no publisher yet; check realsense2_camera logs{RST}",
                    flush=True,
                )

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'═'*W}", flush=True)

        physical_status = []
        if self.check_arm:
            if s["arm_tcp"]:
                physical_status.append(
                    (G, "✓", "Arm TCP", f"connected at {self.arm_host}:{self.arm_port}")
                )
            elif arm_ready:
                physical_status.append(
                    (Y, "~", "Arm TCP", f"not reachable at {self.arm_host}:{self.arm_port}; fallback/controller active")
                )
            else:
                physical_status.append(
                    (R, "✗", "Arm TCP", f"not reachable at {self.arm_host}:{self.arm_port}; no arm fallback/controller")
                )

        if self.check_gripper:
            if s["gripper_serial"]:
                physical_status.append((G, "✓", "Gripper serial", "connected"))
            elif gripper_ready:
                physical_status.append(
                    (Y, "~", "Gripper serial", "not found; fallback/controller active")
                )
            else:
                physical_status.append(
                    (R, "✗", "Gripper serial", "not found; no gripper fallback/controller")
                )

        if self.check_imu:
            if s["imu_expected"] and s["imu_ok"]:
                physical_status.append((G, "✓", "Rover IMU", f"fresh data on {s['selected_imu_topic']}"))
            elif s["imu_expected"]:
                physical_status.append(
                    (R, "✗", "Rover IMU", f"{s['selected_imu']} selected, no fresh data")
                )
            else:
                physical_status.append(
                    (Y, "○", "Rover IMU", "not selected; wheel odom fallback")
                )

        def print_physical_status():
            if not physical_status:
                return
            print(f"\n{B}  Physical Hardware:{RST}", flush=True)
            for color, marker, label, detail in physical_status:
                print(f"  {color}{marker}{RST} {label} — {color}{detail}{RST}", flush=True)

        if overall_ok:
            print(f"  {G}{B}✓  FULL HARDWARE BRINGUP READY{RST}", flush=True)
            print_physical_status()

            if not s["can_present"]:
                print(f"  {Y}⚠   CAN not present — mock rover fallback is being used or expected{RST}", flush=True)

            if s["mock_rover_ok"] and not responding_axes:
                print(f"  {Y}⚠   Rover is using mock fallback, not real ODrive hardware{RST}", flush=True)

            if responding_axes and len(responding_axes) < expected_axis_count:
                print(f"  {Y}⚠   Rover is degraded: active ODrive axes {responding_axes}/{expected_axes}{RST}", flush=True)
        else:
            print(f"  {R}{B}✗  FULL SYSTEM NOT READY{RST}", flush=True)
            print_physical_status()

            if self.check_arm and not arm_ready:
                print(
                    f"  {R}→  Arm stack not ready: no TCP, no controller, no fresh arm joint_states{RST}",
                    flush=True,
                )

            if self.check_gripper and not gripper_ready:
                print(f"  {R}→  Gripper stack not ready: no serial and no controller{RST}", flush=True)

            if self.check_imu and not imu_ready:
                print(
                    f"  {R}→  Rover IMU expected but not ready: "
                    f"source={s['selected_imu']} topic={s['selected_imu_topic']}{RST}",
                    flush=True,
                )

            if self.check_rover and not rover_ready:
                if responding_axes:
                    print(
                        f"  {R}→  Rover stack not ready: ODrive data is present, "
                        f"but the rover is not armed/ready{RST}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {R}→  Rover stack not ready: no mock heartbeat and no usable ODrive data{RST}",
                        flush=True,
                    )
                print(
                    f"  {Y}→  If testing without powered ODrive hardware, launch with "
                    f"rover_hardware_protocol:=mock_hardware{RST}",
                    flush=True,
                )

            if s["can_present"] and not any(axis_has_data):
                print(f"  {Y}→  CAN exists but no ODrive heartbeat received. Check ODrive power + CAN wiring.{RST}", flush=True)

            if rover_any_err:
                err_axes = [i for i in expected_axes if axis_err[i]]
                print(f"  {R}→  ODrive axes reporting errors: {err_axes}{RST}", flush=True)

            if self.require_closed_loop and responding_axes and not rover_all_cl:
                not_cl = [i for i in responding_axes if not axis_cl[i]]
                print(f"  {Y}→  ODrive axes not in CLOSED_LOOP: {not_cl}{RST}", flush=True)

        print(f"{'═'*W}\n", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = FullHardwareChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
