#!/usr/bin/env python3
"""Autonomous soil-sample collection from the ground with the bucket fingertip.

Sequence, one scoop per trigger:

    survey posture -> survey ground -> select site -> pre-screen IK
      -> approach above site -> open bucket -> descend into soil
      -> CLOSE bucket -> extract along the entry axis
      -> re-survey and measure the divot -> transport posture

Why there is no detector here: a probe is an object, so it has to be recognised
before it can be grasped. Soil is terrain -- any flat, level, reachable,
unobstructed patch is a valid place to put the bucket -- and that is a geometry
question the depth image answers on its own. See ``aries_soil_sample.terrain``.

Why the divot is the verification: the two signals a pick would normally use are
both unavailable on this robot. The gripper reports its command rather than a
measured position (the Teensy runs USE_SERVO_FEEDBACK=false), so "the jaws
stopped early" cannot be observed; and MoveIt's padded self-filter blanks the
wrist camera's entire near field, so nothing can see inside the jaws. The ground
at 0.4-0.5 m is well inside the camera's working range, so measuring the hole
the scoop left is the one piece of positive evidence available.

Deliberately NOT auto-started: this drives a gripper into the ground. Call the
``~/scoop`` service (or launch with auto_start:=true) to run one cycle.
"""

import math
import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    RobotState,
)
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionIK
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

# Reused from the grasp package rather than duplicated: the NumPy image bridge
# (cv_bridge segfaults under NumPy 2.x), depth back-projection, and the
# field-calibrated four-bar tables for the bucket fingertip.
from aries_vision_grasp import fourbar
from aries_vision_grasp.grasp_verification import backproject_depth
from aries_vision_grasp.image_bridge import NumpyImageBridge

from aries_soil_sample import scoop as scoop_lib
from aries_soil_sample import terrain


def quat_to_matrix(q: Quaternion) -> np.ndarray:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat(R: np.ndarray) -> Quaternion:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
            y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
            y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
            y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return Quaternion(x=x / n, y=y / n, z=z / n, w=w / n)


class SoilSampleNode(Node):

    def __init__(self) -> None:
        super().__init__('soil_sample_node')
        self._cb = ReentrantCallbackGroup()
        self.bridge = NumpyImageBridge()
        self._abort = threading.Event()
        self._busy = threading.Lock()

        p = self.declare_parameter
        p('planning_frame', 'base_link')
        p('planning_link', 'arm_gripper_base_link')
        p('planning_group', 'igus_rebel_arm')
        p('finger_type', 'bucket')
        p('use_aligned_depth', True)
        p('auto_start', False)

        p('work_region_x', [0.40, 0.65])
        p('work_region_y', [-0.18, 0.18])
        p('work_region_z', [-0.20, 0.10])
        p('prefer_scoop_xy', [0.52, 0.03])

        p('height_map_cell_m', 0.010)
        p('height_map_percentile', 50.0)
        p('height_map_min_points_per_cell', 2)
        p('height_map_min_valid_fraction', 0.35)
        p('depth_stride', 2)

        p('scoop_footprint_m', 0.060)
        p('scoop_max_roughness_m', 0.006)
        p('scoop_max_slope_deg', 12.0)
        p('scoop_min_coverage', 0.70)
        p('scoop_max_candidate_sites', 8)

        p('scoop_standoff_m', 0.060)
        p('scoop_depth_m', 0.030)
        # Non-zero by necessity: a vertical bucket cannot reach ground level with
        # a 214 mm gripper. See the config for the measurement.
        p('scoop_attack_deg', 30.0)
        p('scoop_attack_azimuth_ref', [1.0, 0.0, 0.0])
        p('scoop_max_depth_m', 0.060)
        p('scoop_depth_margin_m', 0.010)
        p('scoop_retrace_extraction', True)

        p('bucket_entry_q', -0.34)
        p('bucket_close_q', 0.07)
        p('bucket_open_q', -1.30)
        p('gripper_command_duration_sec', 3.0)
        p('gripper_settle_sec', 1.0)
        p('bucket_close_hold_sec', 1.5)

        p('verify_capture_enabled', True)
        p('min_sample_volume_m3', 2.0e-5)
        p('verify_radius_m', 0.050)
        p('verify_min_cells', 12)
        p('verify_min_drop_m', 0.002)
        p('verify_disturbed_drop_m', 0.004)
        p('verify_settle_sec', 1.5)

        p('velocity_scale', 0.12)
        p('acceleration_scale', 0.10)
        p('allowed_planning_time', 8.0)
        p('cartesian_eef_step', 0.005)
        p('cartesian_min_fraction', 0.90)
        p('absolute_min_contact_z', -0.16)
        p('prescreen_ik_all_waypoints', True)
        # Wrist rolls to try about the entry axis before giving up on a site.
        # Roll is free for the bucket but NOT for the arm: near the envelope edge
        # only some rolls solve, so trying one made the pre-screen a coin flip.
        p('scoop_wrist_roll_candidates', 12)
        # 300 ms single-shot IK is marginal at these near-limit poses.
        p('ik_prescreen_timeout_sec', 1.0)
        p('max_scoop_attempts', 3)
        # Required for a scoop to execute at all: see set_octomap_collisions.
        p('octomap_disable_during_scoop', True)

        p('arm_joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        p('survey_joint_positions', [0.0, 1.074, 0.639, 0.0349066, 1.394, 1.50098])
        p('transport_joint_positions', [0.0, 0.366519, 1.18682, 0.0349066, 1.55334, 1.50098])
        p('return_to_transport_after_scoop', True)
        p('joint_goal_tolerance', 0.03)

        p('gripper_joint_name', 'gripper_gear_left_joint')
        p('gripper_action_name', '/rebel_gripper_controller/follow_joint_trajectory')
        p('move_action_name', '/move_action')
        p('execute_action_name', '/execute_trajectory')
        p('markers_topic', '/soil_sample/markers')
        p('publish_debug_markers', True)

        g = self.get_parameter
        self.planning_frame = str(g('planning_frame').value)
        self.planning_link = str(g('planning_link').value)
        self.planning_group = str(g('planning_group').value)
        self.auto_start = bool(g('auto_start').value)

        xr = [float(v) for v in g('work_region_x').value]
        yr = [float(v) for v in g('work_region_y').value]
        zr = [float(v) for v in g('work_region_z').value]
        self.region = terrain.WorkRegion(min(xr), max(xr), min(yr), max(yr),
                                         min(zr), max(zr))
        self.prefer_xy = np.array([float(v) for v in g('prefer_scoop_xy').value],
                                  dtype=np.float64)

        self.cell_m = max(0.002, float(g('height_map_cell_m').value))
        self.percentile = float(g('height_map_percentile').value)
        self.min_pts_cell = max(1, int(g('height_map_min_points_per_cell').value))
        self.min_valid_fraction = float(g('height_map_min_valid_fraction').value)
        self.depth_stride = max(1, int(g('depth_stride').value))

        self.footprint_m = max(0.01, float(g('scoop_footprint_m').value))
        self.max_roughness_m = float(g('scoop_max_roughness_m').value)
        self.max_slope_deg = float(g('scoop_max_slope_deg').value)
        self.min_coverage = float(g('scoop_min_coverage').value)
        self.max_sites = max(1, int(g('scoop_max_candidate_sites').value))

        self.scoop_params = scoop_lib.ScoopParams(
            standoff_m=float(g('scoop_standoff_m').value),
            depth_m=float(g('scoop_depth_m').value),
            attack_deg=float(g('scoop_attack_deg').value),
            max_depth_m=float(g('scoop_max_depth_m').value),
            depth_margin_m=float(g('scoop_depth_margin_m').value),
        )
        self.attack_azimuth_ref = np.array(
            [float(v) for v in g('scoop_attack_azimuth_ref').value], dtype=np.float64)

        self.entry_q = float(g('bucket_entry_q').value)
        self.close_q = float(g('bucket_close_q').value)
        self.open_q = float(g('bucket_open_q').value)
        self.grip_duration = float(g('gripper_command_duration_sec').value)
        self.grip_settle = float(g('gripper_settle_sec').value)
        self.close_hold = float(g('bucket_close_hold_sec').value)

        self.verify_enabled = bool(g('verify_capture_enabled').value)
        self.min_volume = float(g('min_sample_volume_m3').value)
        self.verify_radius = float(g('verify_radius_m').value)
        self.verify_min_cells = int(g('verify_min_cells').value)
        self.verify_min_drop = float(g('verify_min_drop_m').value)
        self.verify_disturbed = float(g('verify_disturbed_drop_m').value)
        self.verify_settle = float(g('verify_settle_sec').value)

        self.velocity_scale = float(g('velocity_scale').value)
        self.acceleration_scale = float(g('acceleration_scale').value)
        self.planning_time = float(g('allowed_planning_time').value)
        self.eef_step = float(g('cartesian_eef_step').value)
        self.min_fraction = float(g('cartesian_min_fraction').value)
        self.abs_min_z = float(g('absolute_min_contact_z').value)
        self.prescreen = bool(g('prescreen_ik_all_waypoints').value)
        self.roll_candidates = max(1, int(g('scoop_wrist_roll_candidates').value))
        self.ik_timeout_sec = max(0.05, float(g('ik_prescreen_timeout_sec').value))
        self.max_attempts = max(1, int(g('max_scoop_attempts').value))
        self.octomap_off_during_scoop = bool(g('octomap_disable_during_scoop').value)

        self.arm_joints = [str(v) for v in g('arm_joint_names').value]
        self.survey_q = [float(v) for v in g('survey_joint_positions').value]
        self.transport_q = [float(v) for v in g('transport_joint_positions').value]
        self.return_transport = bool(g('return_to_transport_after_scoop').value)
        self.joint_tol = float(g('joint_goal_tolerance').value)
        self.gripper_joint = str(g('gripper_joint_name').value)
        self.publish_markers = bool(g('publish_debug_markers').value)

        applied = fourbar.set_finger(str(g('finger_type').value))
        if applied != 'bucket':
            self.get_logger().error(
                f'finger_type={g("finger_type").value!r} resolved to {applied!r}. Only the '
                'bucket fingertip can scoop; the contact point differs by up to 23 mm '
                'between jaws, so this WILL drive the bucket to the wrong depth.')
        self.get_logger().info(
            f'Fingertip tables: {fourbar.active_finger()} '
            f'(entry q={self.entry_q:+.3f} -> gap {fourbar.gap_from_q(self.entry_q)*1000:.0f}mm, '
            f'contact z={fourbar.contact_offset_z(self.entry_q)*1000:.0f}mm)')

        # Perception
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_stamp: Optional[rclpy.time.Time] = None
        self.latest_depth_frame: str = ''
        self.camera_info: Optional[CameraInfo] = None
        self.joint_positions = {}

        aligned = bool(g('use_aligned_depth').value)
        depth_topic = ('/gripper_camera/aligned_depth_to_color/image_raw' if aligned
                       else '/gripper_camera/depth/image_rect_raw')
        info_topic = ('/gripper_camera/color/camera_info' if aligned
                      else '/gripper_camera/depth/camera_info')
        self.get_logger().info(f'Depth: {depth_topic} | info: {info_topic}')
        self.create_subscription(Image, depth_topic, self._depth_cb,
                                 qos_profile_sensor_data, callback_group=self._cb)
        self.create_subscription(CameraInfo, info_topic, self._info_cb,
                                 qos_profile_sensor_data, callback_group=self._cb)
        self.create_subscription(JointState, '/joint_states', self._joints_cb, 10,
                                 callback_group=self._cb)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.move_client = ActionClient(self, MoveGroup, str(g('move_action_name').value),
                                        callback_group=self._cb)
        self.exec_client = ActionClient(self, ExecuteTrajectory,
                                        str(g('execute_action_name').value),
                                        callback_group=self._cb)
        self.grip_client = ActionClient(self, FollowJointTrajectory,
                                        str(g('gripper_action_name').value),
                                        callback_group=self._cb)
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path',
                                             callback_group=self._cb)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik',
                                           callback_group=self._cb)
        self.get_scene_client = self.create_client(
            GetPlanningScene, '/get_planning_scene', callback_group=self._cb)
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene', callback_group=self._cb)
        self._octomap_collisions_enabled = True

        self.marker_pub = self.create_publisher(MarkerArray, str(g('markers_topic').value), 1)

        self.create_service(Trigger, '~/scoop', self._srv_scoop, callback_group=self._cb)
        self.create_service(Trigger, '~/survey', self._srv_survey, callback_group=self._cb)
        self.create_service(Trigger, '~/dry_run', self._srv_dry_run, callback_group=self._cb)
        self.create_service(Trigger, '~/abort', self._srv_abort, callback_group=self._cb)

        self.get_logger().info(
            'soil_sample_node ready. Call ~/survey to inspect the ground without '
            'moving, or ~/scoop to run one full cycle. Work region x '
            f'[{self.region.x_min:.2f},{self.region.x_max:.2f}] y '
            f'[{self.region.y_min:.2f},{self.region.y_max:.2f}].')
        if self.auto_start:
            self.get_logger().warning('auto_start:=true -- running one scoop cycle now.')
            threading.Thread(target=self.run_scoop_cycle, daemon=True).start()

    # ---------------------------------------------------------------- sensors

    def _depth_cb(self, msg: Image) -> None:
        try:
            if msg.encoding == '32FC1':
                depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, '32FC1'), dtype=np.float32)
            elif msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32) / 1000.0
            else:
                depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, 'passthrough'),
                                   dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the node
            self.get_logger().warning(f'depth conversion failed: {exc}',
                                      throttle_duration_sec=10.0)
            return
        self.latest_depth = depth
        self.latest_depth_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        self.latest_depth_frame = str(msg.header.frame_id)

    def _info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _joints_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = float(pos)

    # ------------------------------------------------------------- utilities

    def _sleep(self, seconds: float) -> None:
        """Clock-aware sleep, so sim time slows the waits like everything else."""
        if seconds <= 0.0:
            return
        try:
            self.get_clock().sleep_for(RclpyDuration(seconds=float(seconds)))
        except Exception:  # noqa: BLE001 - older rclpy, or no /clock yet
            time.sleep(float(seconds))

    @staticmethod
    def _wait(future, timeout_sec: float = 15.0):
        """Block the worker thread until a future resolves (executor spins elsewhere)."""
        deadline = time.monotonic() + float(timeout_sec)
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(0.01)
        return future.result()

    def _current_tool_rotation(self) -> Optional[np.ndarray]:
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.planning_frame, self.planning_link, rclpy.time.Time(),
                timeout=RclpyDuration(seconds=1.0))
        except Exception:  # noqa: BLE001
            return None
        return quat_to_matrix(tfm.transform.rotation)

    # ------------------------------------------------------------- perception

    def survey(self) -> Optional[terrain.HeightMap]:
        """Build a height map of the work region from the current depth frame."""
        if self.latest_depth is None or self.camera_info is None:
            self.get_logger().error('no depth frame or camera_info yet; cannot survey.')
            return None
        depth = self.latest_depth
        info = self.camera_info
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error('camera_info has no usable intrinsics.')
            return None
        if info.width and depth.shape[1] != info.width:
            s = depth.shape[1] / float(info.width)
            fx, cx, fy, cy = fx * s, cx * s, fy * s, cy * s

        try:
            pts_cam = backproject_depth(depth, fx, fy, cx, cy, self.depth_stride)
        except ValueError as exc:
            self.get_logger().error(f'cannot back-project depth: {exc}')
            return None
        if len(pts_cam) == 0:
            self.get_logger().error('depth frame carries no valid pixels.')
            return None

        stamp = self.latest_depth_stamp or rclpy.time.Time()
        frame = self.latest_depth_frame
        tfm = None
        for when, note in ((stamp, 'image stamp'), (rclpy.time.Time(), 'latest')):
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self.planning_frame, frame, when,
                    timeout=RclpyDuration(seconds=0.5))
                if note != 'image stamp':
                    self.get_logger().warning(
                        f'TF at the image stamp unavailable for {frame} -> '
                        f'{self.planning_frame}; used the latest transform.')
                break
            except Exception:  # noqa: BLE001
                continue
        if tfm is None:
            self.get_logger().error(
                f'no TF {frame} -> {self.planning_frame}; cannot place the survey.')
            return None

        R = quat_to_matrix(tfm.transform.rotation)
        t = np.array([tfm.transform.translation.x, tfm.transform.translation.y,
                      tfm.transform.translation.z], dtype=np.float64)
        pts = pts_cam @ R.T + t

        hmap = terrain.height_map(pts, self.region, self.cell_m, self.percentile)
        self.get_logger().info(
            f'Survey: {len(pts)} pts -> grid {hmap.shape[0]}x{hmap.shape[1]} @ '
            f'{self.cell_m*1000:.0f}mm, {hmap.valid_fraction*100:.0f}% of cells observed.')
        if hmap.valid_fraction < self.min_valid_fraction:
            self.get_logger().error(
                f'survey too sparse ({hmap.valid_fraction*100:.0f}% < '
                f'{self.min_valid_fraction*100:.0f}%): the camera is not looking at the '
                'work region, or the region is wrong.')
            return None
        return hmap

    def select_site(self, hmap: terrain.HeightMap) -> List[terrain.ScoopSite]:
        sites = terrain.select_scoop_site(
            hmap, self.footprint_m,
            max_roughness_m=self.max_roughness_m,
            max_slope_deg=self.max_slope_deg,
            min_coverage=self.min_coverage,
            min_points_per_cell=self.min_pts_cell,
            prefer_xy=self.prefer_xy,
            max_sites=self.max_sites,
        )
        if not sites:
            self.get_logger().error(
                'no scoop site passed the terrain gates (roughness <= '
                f'{self.max_roughness_m*1000:.0f}mm, slope <= {self.max_slope_deg:.0f}deg, '
                f'coverage >= {self.min_coverage*100:.0f}%). The ground in the work region '
                'is too rough, too sloped, or too poorly observed to cut.')
            return []
        self.get_logger().info(f'{len(sites)} candidate site(s); best {sites[0].summary}')
        self._publish_site_markers(sites)
        return sites

    # ------------------------------------------------------------- geometry

    def scoop_link_poses(
        self, site: terrain.ScoopSite
    ) -> Optional[Tuple[List[Tuple[str, Pose]], float, np.ndarray]]:
        """Turn a site into gripper-link poses for each scoop waypoint."""
        waypoints, depth, note = scoop_lib.plan_scoop(
            site.centre, site.normal, self.scoop_params,
            azimuth_ref=self.attack_azimuth_ref)
        if note:
            self.get_logger().warning(f'[scoop] penetration depth {note}.')

        # The hard Z floor depends only on the contact positions, not on the wrist
        # roll, so it is checked once before any IK work.
        for wp in waypoints:
            if float(wp.position[2]) < self.abs_min_z:
                self.get_logger().error(
                    f'[safety] waypoint {wp.label} at z={wp.position[2]:.3f} is below '
                    f'absolute_min_contact_z={self.abs_min_z:.3f}. Refusing the scoop: a '
                    'bad depth frame must not drive the bucket into the ground.')
                return None

        # The bucket's contact point swings a long way with the linkage, so take
        # the offset at the angle the jaws will actually be holding during entry.
        offset = fourbar.contact_offset(self.entry_q, fourbar.CONTACT_Y_OFFSET_M)
        R_now = self._current_tool_rotation()
        prefer_pinch = R_now[:, 0] if R_now is not None else None
        axis = waypoints[0].tool_axis

        self.get_logger().info(
            f'[scoop] site z={site.centre[2]:.3f}, penetrating {depth*1000:.0f}mm along '
            f'the surface normal (slope {site.slope_deg:.1f}deg); entry axis '
            f'[{axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}].')

        frames = scoop_lib.roll_frames(axis, prefer_pinch, self.roll_candidates)
        if not self.prescreen:
            poses = self._poses_for_frame(waypoints, frames[0], offset)
            return poses, depth, axis

        # Search the free roll for one where EVERY waypoint has IK. One roll is
        # not enough: near the envelope edge only some rolls solve, so committing
        # to the roll nearest the current wrist made this a coin flip.
        failures = {}
        for i, R in enumerate(frames):
            poses = self._poses_for_frame(waypoints, R, offset)
            bad = next((label for label, pose in poses
                        if not self._ik_exists(pose)), None)
            if bad is None:
                if i:
                    self.get_logger().info(
                        f'[scoop] wrist roll candidate {i + 1}/{len(frames)} has IK for '
                        'all waypoints; the nearer rolls did not.')
                return poses, depth, axis
            failures[bad] = failures.get(bad, 0) + 1

        summary = ', '.join(f'{k}x{v}' for k, v in sorted(failures.items()))
        self.get_logger().error(
            f'[scoop] no wrist roll out of {len(frames)} gives IK for the whole scoop '
            f'(first failing waypoint per roll: {summary}). The site is out of the arm\'s '
            'reach at this depth and attack angle, not merely awkwardly oriented. Try a '
            'shallower scoop_depth_m, a different scoop_attack_deg, or a work region '
            'closer in.')
        return None

    def _poses_for_frame(
        self,
        waypoints: Sequence,
        R_tool: np.ndarray,
        offset: np.ndarray,
    ) -> List[Tuple[str, Pose]]:
        """Gripper-link poses for one wrist orientation, held across the stroke."""
        poses: List[Tuple[str, Pose]] = []
        orientation = matrix_to_quat(R_tool)
        for wp in waypoints:
            link_pos = scoop_lib.link_position_for_contact(wp.position, R_tool, offset)
            pose = Pose()
            pose.position.x = float(link_pos[0])
            pose.position.y = float(link_pos[1])
            pose.position.z = float(link_pos[2])
            pose.orientation = orientation
            poses.append((wp.label, pose))
        return poses

    def _ik_exists(self, pose: Pose) -> bool:
        """Quiet single-pose IK check used by the wrist-roll search."""
        if not self.ik_client.wait_for_service(timeout_sec=3.0):
            return True          # cannot screen; let the motion layer decide
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.planning_group
        req.ik_request.ik_link_name = self.planning_link
        req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions = True
        ns = int(self.ik_timeout_sec * 1e9)
        req.ik_request.timeout = Duration(sec=ns // 1_000_000_000,
                                          nanosec=ns % 1_000_000_000)
        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose = pose
        req.ik_request.pose_stamped = ps
        res = self._wait(self.ik_client.call_async(req), self.ik_timeout_sec + 5.0)
        return bool(res is not None and res.error_code.val == 1)

    # --------------------------------------------------------------- motion

    def move_to_posture(self, positions: Sequence[float], label: str) -> bool:
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] MoveGroup action unavailable.')
            return False
        req = MotionPlanRequest()
        req.group_name = self.planning_group
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scale
        req.max_acceleration_scaling_factor = self.acceleration_scale
        req.num_planning_attempts = 5
        req.start_state.is_diff = True
        con = Constraints()
        for name, value in zip(self.arm_joints, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = self.joint_tol
            jc.tolerance_below = self.joint_tol
            jc.weight = 1.0
            con.joint_constraints.append(jc)
        req.goal_constraints = [con]

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(f'[{label}] moving to posture.')
        gh = self._wait(self.move_client.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] MoveGroup rejected the goal.')
            return False
        result = self._wait(gh.get_result_async(), self.planning_time + 40.0)
        if result is None:
            self.get_logger().error(f'[{label}] MoveGroup did not return a result.')
            return False
        code = result.result.error_code.val
        if code != 1:
            self.get_logger().error(f'[{label}] MoveGroup failed, moveit_error_code={code}.')
            return False
        return True

    def move_cartesian(self, poses: Sequence[Pose], label: str) -> bool:
        """Straight-line Cartesian motion through the given link poses."""
        if not self.cart_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] compute_cartesian_path unavailable.')
            return False
        req = GetCartesianPath.Request()
        req.header.frame_id = self.planning_frame
        req.group_name = self.planning_group
        req.link_name = self.planning_link
        req.waypoints = list(poses)
        req.max_step = self.eef_step
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.start_state.is_diff = True
        res = self._wait(self.cart_client.call_async(req), 20.0)
        if res is None:
            self.get_logger().error(f'[{label}] Cartesian service did not answer.')
            return False
        fraction = float(res.fraction)
        if fraction < self.min_fraction:
            self.get_logger().error(
                f'[{label}] Cartesian path solved only {fraction:.2f} of the way '
                f'(need {self.min_fraction:.2f}); not executing a partial scoop stroke.')
            return False

        traj = res.solution
        # Cartesian paths come back untimed by velocity scaling; MoveIt's own
        # time parameterisation already applied the request's limits, so execute
        # as returned rather than re-timing it here.
        if not self.exec_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] execute_trajectory unavailable.')
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        self.get_logger().info(f'[{label}] Cartesian path {fraction:.2f}; executing.')
        gh = self._wait(self.exec_client.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] execute_trajectory rejected the goal.')
            return False
        result = self._wait(gh.get_result_async(), 60.0)
        if result is None:
            self.get_logger().error(f'[{label}] execute_trajectory returned no result.')
            return False
        code = result.result.error_code.val
        if code != 1:
            self.get_logger().error(f'[{label}] execution failed, error_code={code}.')
            return False
        return True

    def command_bucket(self, q: float, label: str) -> bool:
        """Command the bucket jaw angle and wait out the motion.

        NOTE: on hardware the gripper reports its command rather than a measured
        position, so reaching the commanded angle proves nothing about what is in
        the bucket. That is fine here -- unlike a rigid-probe grasp, a soil scoop
        has nothing hard to stop the jaws, so arriving at the target is expected.
        Capture is judged from the divot instead, never from this.
        """
        if not self.grip_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] gripper action unavailable.')
            return False
        traj = JointTrajectory()
        traj.joint_names = [self.gripper_joint]
        current = self.joint_positions.get(self.gripper_joint)
        if current is not None:
            start = JointTrajectoryPoint()
            start.positions = [float(current)]
            start.velocities = [0.0]
            start.time_from_start = Duration(sec=0, nanosec=0)
            traj.points.append(start)
        end = JointTrajectoryPoint()
        end.positions = [float(q)]
        end.velocities = [0.0]
        ns = int(self.grip_duration * 1e9)
        end.time_from_start = Duration(sec=ns // 1_000_000_000,
                                       nanosec=ns % 1_000_000_000)
        traj.points.append(end)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.get_logger().info(
            f'[{label}] bucket -> q={q:+.3f} (gap {fourbar.gap_from_q(q)*1000:.0f}mm).')
        gh = self._wait(self.grip_client.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] gripper controller rejected the goal.')
            return False
        self._wait(gh.get_result_async(), self.grip_duration + 15.0)
        self._sleep(self.grip_settle)
        return True

    # ---------------------------------------------------- octomap vs digging

    def set_octomap_collisions(self, enabled: bool, reason: str) -> bool:
        """Turn octomap collision checking on or off for the whole robot.

        A digging task cannot be planned against an octomap of the ground it is
        digging into. Measured on the live scene: at the surveyed site with the
        30 deg tilted entry, approach/entry/penetrate are all 4/4 reachable and
        all 0/4 once collision checking is on, because the terrain the scoop
        exists to cut is modelled as an obstacle. There is no collision-free path
        into the material, by definition.

        Implemented exactly as the grasp package does it: the ACM DEFAULT entry
        for ``<octomap>``, one flag that makes it allowed against every element
        rather than N pairwise entries that need re-applying whenever a link or
        object appears. It is a planning-scene switch, so the sensor pipeline
        keeps running and the octomap keeps building -- MoveIt just stops
        colliding against it -- and it is fully reversible.

        TRADE-OFF, stated plainly: while off the arm will NOT avoid obstacles
        that exist only in the octomap. What replaces it for the scoop strokes is
        narrower but real -- the site was surveyed for roughness and slope, the
        waypoints are a short straight line whose geometry follows from that
        survey, every one is bounded by absolute_min_contact_z, and all of them
        were IK pre-screened. It is re-enabled the moment the stroke is done.
        """
        if not (self.get_scene_client.wait_for_service(timeout_sec=3.0)
                and self.apply_scene_client.wait_for_service(timeout_sec=3.0)):
            self.get_logger().error(
                'get/apply_planning_scene unavailable; cannot '
                f'{"enable" if enabled else "disable"} octomap collision checking.')
            return False
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        res = self._wait(self.get_scene_client.call_async(req), 8.0)
        if res is None:
            self.get_logger().error('get_planning_scene did not answer.')
            return False

        acm = res.scene.allowed_collision_matrix
        names = list(acm.default_entry_names)
        values = list(acm.default_entry_values)
        # An ACM entry means "collision ALLOWED", i.e. checking is OFF.
        allowed = not bool(enabled)
        if '<octomap>' in names:
            values[names.index('<octomap>')] = allowed
        else:
            names.append('<octomap>')
            values.append(allowed)
        acm.default_entry_names = names
        acm.default_entry_values = values

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm
        apply_req = ApplyPlanningScene.Request()
        apply_req.scene = scene
        if self._wait(self.apply_scene_client.call_async(apply_req), 8.0) is None:
            self.get_logger().error('apply_planning_scene did not answer.')
            return False
        self._octomap_collisions_enabled = bool(enabled)
        if enabled:
            self.get_logger().info(f'[CollisionWorld] Octomap collision checking ON: {reason}')
        else:
            self.get_logger().warning(
                f'[CollisionWorld] Octomap collision checking OFF: {reason} The arm will '
                'not avoid octomap-only obstacles until it is restored.')
        # The scene needs a moment to propagate before the next plan is requested.
        self._sleep(0.4)
        return True

    # --------------------------------------------------------- verification

    def verify_capture(self, site: terrain.ScoopSite,
                       before: terrain.HeightMap) -> Tuple[str, str]:
        if not self.verify_enabled:
            return scoop_lib.UNKNOWN, 'verification disabled'
        self._sleep(self.verify_settle)
        after = self.survey()
        if after is None:
            return scoop_lib.UNKNOWN, 'post-scoop survey failed'
        try:
            volume, max_drop, cells = terrain.divot_volume(
                before, after, site.centre[:2], self.verify_radius,
                min_drop_m=self.verify_min_drop)
        except ValueError as exc:
            return scoop_lib.UNKNOWN, f'height maps not comparable: {exc}'
        verdict, reason = scoop_lib.capture_verdict(
            volume, max_drop, cells,
            min_volume_m3=self.min_volume,
            min_cells=self.verify_min_cells,
            disturbed_drop_m=self.verify_disturbed)
        self.get_logger().info(
            f'[verify] divot {volume*1e6:.0f} cm^3, max drop {max_drop*1000:.1f} mm '
            f'over {cells} shared cells -> {verdict.upper()}: {reason}')
        return verdict, reason

    # ------------------------------------------------------------- sequence

    def run_scoop_cycle(self) -> Tuple[bool, str]:
        if not self._busy.acquire(blocking=False):
            return False, 'a cycle is already running'
        self._abort.clear()
        try:
            return self._run_scoop_cycle_locked()
        except Exception as exc:  # noqa: BLE001 - never leave the node wedged
            self.get_logger().error(f'scoop cycle raised: {exc}')
            return False, f'exception: {exc}'
        finally:
            self._busy.release()

    def _aborted(self, stage: str) -> bool:
        if self._abort.is_set():
            self.get_logger().warning(f'aborted before {stage}; stopping cleanly.')
            return True
        return False

    def _run_scoop_cycle_locked(self) -> Tuple[bool, str]:
        if not self.move_to_posture(self.survey_q, 'survey-posture'):
            return False, 'could not reach the survey posture'
        self._sleep(1.0)

        for attempt in range(1, self.max_attempts + 1):
            if self._aborted(f'attempt {attempt}'):
                return False, 'aborted'
            self.get_logger().info(f'--- scoop attempt {attempt}/{self.max_attempts} ---')

            before = self.survey()
            if before is None:
                return False, 'survey failed'
            sites = self.select_site(before)
            if not sites:
                return False, 'no acceptable scoop site'

            # One attempt per candidate site, so a retry moves to genuinely
            # different ground rather than re-cutting the same failed patch.
            site = sites[min(attempt - 1, len(sites) - 1)]

            # The octomap models the ground this scoop exists to cut, so nothing
            # in the scoop can be collision-checked against it -- including the
            # wrist-roll IK search inside scoop_link_poses, which is why this is
            # disabled BEFORE that call and not after it. Restored in the finally
            # below however this attempt ends.
            if self.octomap_off_during_scoop and not self.set_octomap_collisions(
                    False, f'scoop attempt {attempt}: the octomap models the soil being cut.'):
                return False, 'could not suppress octomap collisions for the scoop'
            try:
                built = self.scoop_link_poses(site)
                if built is None:
                    continue
                poses, depth, _axis = built
                by_label = dict(poses)

                # scoop_link_poses already picked a wrist roll with IK for every
                # waypoint, so there is no separate pre-screen step here.
                if not self.command_bucket(self.entry_q, 'open-bucket'):
                    return False, 'bucket would not open'
                if self._aborted('approach'):
                    return False, 'aborted'
                if not self.move_to_posture_pose(by_label['approach'], 'approach'):
                    continue
                if not self.move_cartesian([by_label['entry'], by_label['penetrate']],
                                           'descend'):
                    # Nothing is buried yet: the stroke never started.
                    continue

                # Past this point the bucket is IN the soil. Extraction must run
                # even if the close fails, or the gripper is left buried.
                closed = self.command_bucket(self.close_q, 'close-bucket')
                self._sleep(self.close_hold)
                extracted = self.move_cartesian([by_label['extract']], 'extract')
                if not extracted:
                    self.get_logger().error(
                        'EXTRACTION FAILED with the bucket in the soil. Not retrying and '
                        'not moving to a posture: the arm is left where it is for '
                        'inspection.')
                    return False, 'extraction failed with the bucket buried'
                if not closed:
                    self.get_logger().warning('close command failed; extracted anyway.')
                    continue
            finally:
                # Restore on every path, including the buried-bucket return: the
                # next thing anyone plans must see the obstacles again.
                if self.octomap_off_during_scoop:
                    self.set_octomap_collisions(
                        True, 'scoop stroke finished; obstacle avoidance restored.')

            verdict, reason = self.verify_capture(site, before)
            if verdict == scoop_lib.CAPTURED:
                if self.return_transport and not self._aborted('transport'):
                    self.move_to_posture(self.transport_q, 'transport')
                self.get_logger().info(f'SOIL SAMPLE COLLECTED ({reason}). '
                                       'Bucket stays closed.')
                return True, f'captured: {reason}'
            if verdict == scoop_lib.EMPTY:
                self.get_logger().warning(
                    f'bucket appears empty ({reason}); trying a different site.')
            else:
                self.get_logger().warning(
                    f'capture unconfirmed ({reason}); treating as a miss and retrying.')

        return False, f'no sample after {self.max_attempts} attempts'

    def move_to_posture_pose(self, pose: Pose, label: str) -> bool:
        """MoveGroup to a Cartesian link pose via an IK'd joint goal.

        The approach hop is planned in joint space on purpose: only the strokes
        that must be straight lines through the soil are Cartesian.
        """
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] compute_ik unavailable.')
            return False
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.planning_group
        req.ik_request.ik_link_name = self.planning_link
        req.ik_request.robot_state.is_diff = True
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = Duration(sec=0, nanosec=500_000_000)
        ps = PoseStamped()
        ps.header.frame_id = self.planning_frame
        ps.pose = pose
        req.ik_request.pose_stamped = ps
        res = self._wait(self.ik_client.call_async(req), 8.0)
        if res is None or res.error_code.val != 1:
            self.get_logger().error(f'[{label}] no IK solution for the approach pose.')
            return False
        wanted = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        try:
            positions = [wanted[j] for j in self.arm_joints]
        except KeyError as exc:
            self.get_logger().error(f'[{label}] IK solution missing joint {exc}.')
            return False
        return self.move_to_posture(positions, label)

    # -------------------------------------------------------------- markers

    def _publish_site_markers(self, sites: Sequence[terrain.ScoopSite]) -> None:
        if not self.publish_markers:
            return
        arr = MarkerArray()
        for i, s in enumerate(sites):
            m = Marker()
            m.header.frame_id = self.planning_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'scoop_sites'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position = Point(x=float(s.centre[0]), y=float(s.centre[1]),
                                    z=float(s.centre[2]))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = float(self.footprint_m)
            m.scale.z = 0.004
            best = (i == 0)
            m.color.r = 0.0 if best else 0.8
            m.color.g = 0.9 if best else 0.8
            m.color.b = 0.2 if best else 0.8
            m.color.a = 0.9 if best else 0.35
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ------------------------------------------------------------- services

    def _srv_scoop(self, _req, res):
        ok, msg = self.run_scoop_cycle()
        res.success = bool(ok)
        res.message = msg
        return res

    def _srv_survey(self, _req, res):
        hmap = self.survey()
        if hmap is None:
            res.success = False
            res.message = 'survey failed; see the log'
            return res
        sites = self.select_site(hmap)
        res.success = bool(sites)
        res.message = (f'{len(sites)} site(s); best {sites[0].summary}' if sites
                       else 'no site passed the terrain gates')
        return res

    def _srv_dry_run(self, _req, res):
        """Everything a scoop does except moving: survey, select, build the
        waypoints, check them against the safety floor, pre-screen IK.

        Worth running after any change to the work region, the depth pipeline or
        the rover's pose. The safety floor in particular has to sit below the
        deepest legitimate scoop, and that depends on where the ground actually
        is -- a floor set above the surveyed ground silently rejects every
        attempt, which this surfaces without putting the bucket in the soil.
        """
        hmap = self.survey()
        if hmap is None:
            res.success, res.message = False, 'survey failed; see the log'
            return res
        sites = self.select_site(hmap)
        if not sites:
            res.success, res.message = False, 'no site passed the terrain gates'
            return res
        # Screen under the same collision settings a real scoop runs under,
        # otherwise this reports a failure the scoop would never hit (or misses
        # one it would).
        restore = False
        if self.octomap_off_during_scoop:
            restore = self.set_octomap_collisions(
                False, 'dry run: screening under scoop collision settings.')
        try:
            built = self.scoop_link_poses(sites[0])
        finally:
            if restore:
                self.set_octomap_collisions(True, 'dry run finished.')
        if built is None:
            res.success, res.message = False, (
                'rejected by the safety floor or no wrist roll has IK for the whole '
                f'scoop (absolute_min_contact_z={self.abs_min_z:.3f}); see the log')
            return res
        poses, depth, _ = built
        zs = ', '.join(f'{label}={pose.position.z:.3f}' for label, pose in poses)
        res.success = True
        res.message = (
            f'site {sites[0].summary}; depth {depth*1000:.0f}mm; link z: {zs}; '
            'a wrist roll with IK for every waypoint was found')
        self.get_logger().info(f'[dry-run] {res.message} (nothing was moved)')
        return res

    def _srv_abort(self, _req, res):
        self._abort.set()
        self.get_logger().warning('abort requested; the cycle will stop at the next stage '
                                  'boundary (a stroke already in the soil still extracts).')
        res.success = True
        res.message = 'abort flag set'
        return res


def main() -> None:
    rclpy.init()
    node = SoilSampleNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
