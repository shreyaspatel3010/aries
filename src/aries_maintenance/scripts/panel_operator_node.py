#!/usr/bin/env python3
"""Find the maintenance panel with rover/gripper cameras and operate controls.

    ros2 run aries_maintenance panel_operator_node.py

Watches both rover and gripper cameras for ArUco 11/13/14/15, solves the panel's
pose, publishes it as a TF frame, and drives the arm through any control in
`panel_task.json`:

    ros2 topic pub --once /panel/operate std_msgs/String "data: 'mcb_3'"
    ros2 topic pub --once /panel/operate std_msgs/String "data: 'all_breakers'"

Controls enabled in ``config/panel_tasks.yaml`` can be run as one sequence:

    ros2 topic pub --once /panel/operate_enabled std_msgs/msg/Bool '{data: true}'

Three shapes of action, taken from the table rather than decided here:

  flick  14 MCB toggles - jaws CLOSED, drive into the toggle and sweep along the
         control's own `on_direction`, which lifts the lever (lever up is ON).
         They sit on a 17.7 mm pitch (3.9 mm between modules) so they cannot be
         grasped, only pushed. Same for the 5 buttons, which are pressed along
         the console normal.
  turn   5 selectors and 2 red disconnects - grip, then roll the wrist about the
         approach axis. The jaw line MUST run up-slope: the disconnects are
         65 mm across on a 76.7 mm pitch, leaving 11.7 mm between them, so a
         grasp closing across the console cannot fit, while up-slope has 35.8 mm
         clear above and 36.2 mm below.
  press  the 5 push buttons, 4 mm of travel along the normal.

MOTION SAFETY
MoveGroup first plans through free space to the clear standoff pose. Every
motion from standoff inward is then a Cartesian path, and a fraction < 1.0 is
reported and refused rather than executed part-way: a joint-space plan directly
to contact can sweep the tool through the console on the way there.
"""

import math
import pathlib
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus, GoalStatusArray
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume, Constraints, JointConstraint, OrientationConstraint,
    PlanningOptions, PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_action_status_default, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory

from aries_vision_grasp.image_bridge import NumpyImageBridge
from aries_maintenance.action_utils import (
    load_enabled_controls, log_operation_result, make_joint_trajectory,
    run_action,
)
from aries_maintenance.panel_alignment import (
    average_transforms, control_waypoints, detect_markers,
    flick_endpoint_in_planning_frame, load_task_table,
    panel_pose_from_markers, quaternion_from_matrix,
    refine_panel_pose_from_depth, roll_about_tool_z,
    transform_distance, transform_inlier_consensus,
)


def _pose_msg(transform):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(v) for v in np.asarray(transform)[:3, 3])
    x, y, z, w = quaternion_from_matrix(transform)
    pose.orientation.x, pose.orientation.y = float(x), float(y)
    pose.orientation.z, pose.orientation.w = float(z), float(w)
    return pose


def _rotation_about(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) * c + s * cross + (1 - c) * np.outer(axis, axis)


class PanelOperator(Node):

    def __init__(self):
        super().__init__('panel_operator')
        group = ReentrantCallbackGroup()
        p = self.declare_parameter

        # Installed, this script lives in lib/, so a path relative to __file__
        # only resolves when running from the source tree. Ask ament first.
        try:
            from ament_index_python.packages import get_package_share_directory
            share = pathlib.Path(get_package_share_directory('aries'))
        except Exception:
            share = pathlib.Path(__file__).resolve().parents[2] / 'aries'
        default_table = share / 'models' / 'maintenance_panel' / 'panel_task.json'
        p('config_file', '')
        p('task_table', str(default_table))
        p('planning_frame', 'base_link')
        p('planning_group', 'igus_rebel_arm')
        p('tool_frame', 'gripper_tcp')
        # BOTH cameras feed the localiser. The gripper camera is the accurate
        # one - it gets close - but it is only aimed at the panel some of the
        # time and loses the tags entirely once the tool closes in. The rover's
        # body camera sees the whole console from further back. Agreeing recent
        # estimates are fused after both are transformed into base_link.
        p('image_topics', ['/gripper_camera/color/image_raw',
                           '/camera/color/image_raw'])
        p('camera_info_topics', ['/gripper_camera/color/camera_info',
                                 '/camera/color/camera_info'])
        p('camera_frames', ['gripper_camera_color_optical_frame',
                            'camera_color_optical_frame'])
        p('depth_topics', [
            '/gripper_camera/aligned_depth_to_color/image_raw',
            '/camera/aligned_depth_to_color/image_raw'])
        p('use_depth_refinement', True)
        p('depth_sync_tolerance_sec', 0.12)
        p('depth_expected_band_m', 0.15)
        p('depth_min_m', 0.10)
        p('depth_max_m', 5.0)
        p('depth_min_plane_pixels', 24)
        p('depth_max_normal_correction_deg', 30.0)
        # A registered marker depth cloud resolves single-marker planar PnP,
        # so either camera may localise the panel on its own. Both cameras are
        # still fused whenever they have agreeing observations, and they never
        # need to see the same marker ID.
        p('allow_single_depth_camera', True)
        p('min_depth_markers', 1)
        # The simulated joint/TF publisher can trail rendered camera frames.
        # During calibration the arm is stationary, so a bounded latest-TF
        # fallback is more accurate than dropping that camera completely.
        p('max_camera_tf_fallback_age_sec', 1.5)
        p('cartesian_service_name', '/compute_cartesian_path')
        p('ik_service_name', '/compute_ik')
        p('move_group_action_name', '/move_action')
        p('arm_controller_action_name',
          '/rebel_arm_trajectory_controller/follow_joint_trajectory')
        p('arm_controller_status_topic',
          '/rebel_arm_trajectory_controller/follow_joint_trajectory/_action/status')
        p('controller_idle_timeout_sec', 30.0)
        # Use the active ros2_control trajectory controller. The old Float64
        # Gazebo bridge bypassed this controller and was ignored/fought by it.
        p('gripper_command_topic',
          '/rebel_gripper_controller/joint_trajectory')
        p('gripper_joint_name', 'gripper_gear_left_joint')
        p('gripper_open_position', -1.20)
        # v2 maintenance fingers physically meet at q ~= -0.030. The generic
        # bucket value (+0.07) over-closes this fingertip by about 0.10 rad.
        p('gripper_close_position', -0.03)
        p('gripper_command_duration_sec', 0.75)
        p('gripper_settle_sec', 1.5)
        # gripper_tcp is 150 mm from the gripper base; the maintenance fingers
        # meet at 215 mm, so their physical contact is 65 mm beyond the IK link.
        p('tool_contact_offset_m', 0.065)
        # With the jaws closed for a flick/press, the leading maintenance-tip
        # collision surface is at 233.7 mm from the gripper base: 83.7 mm past
        # gripper_tcp. Rotary grasps still use the 65 mm jaw meeting point.
        p('tool_push_contact_offset_m', 0.084)
        # Cartesian steps this fine keep the tool on the straight line between
        # waypoints; a coarse step lets MoveIt cut the corner into the console.
        p('cartesian_step_m', 0.002)
        p('cartesian_jump_threshold', 0.0)
        p('min_cartesian_fraction', 0.99)
        # The short Cartesian phase intentionally touches panel controls. The
        # free-space trip to standoff remains collision checked by MoveGroup.
        # Enable this only if the planning scene has a panel ACM configured.
        p('contact_avoid_collisions', False)
        # RGB-only localization needs two tags to remove planar-PnP ambiguity.
        # A registered depth-plane fit may satisfy this with one tag instead.
        p('min_markers', 2)
        p('max_reprojection_px', 3.0)
        # Estimates from both cameras inside this time window are transformed
        # into base_link and fused. A disagreeing camera is excluded instead of
        # averaging two incompatible planar-PnP solutions.
        p('camera_fusion_window_sec', 0.75)
        p('max_fusion_translation_m', 0.08)
        p('max_fusion_rotation_deg', 12.0)
        p('required_camera_count', 2)
        # Do not latch the first valid camera estimate. Select the densest
        # time-spanning inlier cluster so an isolated depth edge/PnP outlier
        # cannot block an otherwise stable calibration.
        p('calibration_sample_count', 15)
        p('calibration_min_duration_sec', 0.50)
        p('calibration_max_translation_spread_m', 0.012)
        p('calibration_max_rotation_spread_deg', 1.5)
        # Every operate-enabled trigger starts a new acquisition. Once that
        # fresh stable pose is accepted it is latched for the whole motion,
        # because the arm will occlude the markers on approach.
        p('recalibrate_on_operate_enabled', True)
        # The panel and rover must remain stationary after acquisition. Once a
        # valid pose is found, freeze it: a later partial/ambiguous
        # view must not corrupt the pose while the arm occludes the markers.
        p('latch_panel_pose', True)
        # The panel is bolted to the world, and the gripper camera LOSES the
        # tags as soon as the tool closes on a control - they leave the frame
        # long before contact. So the pose is latched once acquired and reused,
        # rather than required fresh at command time. Requiring freshness meant
        # every command was refused with "pose is stale" and the arm never
        # moved. Set `require_fresh_pose` if the panel might have been nudged.
        p('require_fresh_pose', False)
        p('pose_timeout_sec', 2.0)
        p('planning_time_sec', 8.0)
        p('action_timeout_sec', 60.0)
        p('planning_attempts', 5)
        p('ik_timeout_sec', 0.75)
        p('joint_goal_tolerance_rad', 0.02)
        p('arm_joint_names', [f'joint{k}' for k in range(1, 7)])
        p('position_tolerance_m', 0.008)
        p('orientation_tolerance_rad', 0.08)
        p('velocity_scale', 0.20)
        p('acceleration_scale', 0.20)
        p('max_cartesian_speed_mps', 0.03)
        p('motion_verify_position_m', 0.025)
        p('motion_verify_orientation_rad', 0.15)

        def get(name):
            return self.get_parameter(name).value

        self.get = get
        self.table = load_task_table(get('task_table'))
        self.controls = {c['name']: c for c in self.table['controls']}
        # These declarations are intentionally generated from the task table:
        # the ROS parameter file uses nested ``controls: name: true`` entries,
        # which rclpy exposes as ``controls.name`` parameters. This also makes
        # a stale YAML name fail at launch instead of silently moving the wrong
        # control.
        for name in self.controls:
            p(f'controls.{name}', False)
        self.get_logger().info(
            f"loaded {len(self.controls)} controls from {get('task_table')}")
        enabled = self._enabled_control_names()
        config_file = pathlib.Path(get('config_file')).expanduser()
        if get('config_file'):
            try:
                resolved = config_file.resolve(strict=True)
                modified = config_file.stat().st_mtime
                self.get_logger().info(
                    f'fresh YAML loaded at node startup from {resolved} '
                    f'(mtime {modified:.3f})')
            except OSError as exc:
                self.get_logger().warn(
                    f'configured YAML path {config_file} cannot be inspected: {exc}')
        self.get_logger().info(
            'YAML-enabled controls: ' + (', '.join(enabled) if enabled else 'none'))

        self.bridge = NumpyImageBridge()
        self.cameras = {}
        for topic, info_topic, frame in zip(get('image_topics'),
                                            get('camera_info_topics'),
                                            get('camera_frames')):
            self.cameras[topic] = dict(
                info=None, frame=frame, frames=0, tags=[], pose=None,
                pose_stamp=None, pose_info=None, depth=None, depth_stamp=None)
        self.panel_pose = None          # 4x4, planning frame from panel
        self.panel_stamp = None
        self.localization_generation = 0
        self.calibration_samples = []
        self.lock = threading.Lock()
        # Only one command sequence may own the arm. The executor is
        # multi-threaded so a second topic publication can otherwise interleave
        # gripper and Cartesian commands with the first one.
        self.operation_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending_names = None
        self.controller_status_lock = threading.Lock()
        self.controller_status_seen = False
        self.controller_active = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        for topic, info_topic in zip(get('image_topics'), get('camera_info_topics')):
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, t=topic: self.cameras[t].__setitem__('info', msg),
                qos_profile_sensor_data)
            self.create_subscription(
                Image, topic, lambda msg, t=topic: self._image_cb(msg, t),
                qos_profile_sensor_data)
        for color_topic, depth_topic in zip(
                get('image_topics'), get('depth_topics')):
            self.create_subscription(
                Image, depth_topic,
                lambda msg, t=color_topic: self._depth_cb(msg, t),
                qos_profile_sensor_data)
        self.get_logger().info(
            f"localising from {len(self.cameras)} cameras: "
            f"{', '.join(self.cameras)}")
        self.create_subscription(String, '/panel/operate', self._operate_cb, 1,
                                 callback_group=group)
        self.create_subscription(
            Bool, '/panel/operate_enabled', self._operate_enabled_cb, 1,
            callback_group=group)
        self.status_pub = self.create_publisher(String, '/panel/status', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, get('gripper_command_topic'), 10)

        self.create_timer(5.0, self._heartbeat)
        self.cartesian = self.create_client(
            GetCartesianPath, get('cartesian_service_name'), callback_group=group)
        self.compute_ik = self.create_client(
            GetPositionIK, get('ik_service_name'), callback_group=group)
        self.move_group = ActionClient(
            self, MoveGroup, get('move_group_action_name'), callback_group=group)
        self.arm_controller = ActionClient(
            self, FollowJointTrajectory, get('arm_controller_action_name'),
            callback_group=group)
        self.create_subscription(
            GoalStatusArray, get('arm_controller_status_topic'),
            self._controller_status_cb, qos_profile_action_status_default,
            callback_group=group)

    # ------------------------------------------------------------------ vision

    def _heartbeat(self):
        """Say what is going on, every few seconds.

        Without this the node is silent whenever it cannot see the panel, which
        is indistinguishable from it being broken - and "no tags in view" is by
        far the most likely reason a command appears to do nothing.
        """
        seen = {t: c for t, c in self.cameras.items() if c['frames']}
        if not seen:
            self.get_logger().warn(
                f"no images yet on any of {', '.join(self.cameras)}")
            return
        view = '; '.join(f"{t.split('/')[1]}: {c['tags'] or 'none'}"
                         for t, c in seen.items())
        with self.lock:
            have = self.panel_pose is not None
            pose = None if not have else self.panel_pose.copy()
            age = (0.0 if not have else
                   (self.get_clock().now() - self.panel_stamp).nanoseconds / 1e9)
            source = getattr(self, 'panel_source', '?')
        if pose is not None:
            # Keep the dynamic TF usable even though image callbacks are no
            # longer allowed to modify a latched panel pose.
            self._broadcast(pose)
        if not have:
            self.get_logger().warn(
                f"panel NOT localised. Tags in view - {view}. Point a camera at "
                f"the panel. Different marker IDs across cameras are valid; "
                f"one stable marker with a registered depth plane is also "
                f"enough from either camera.")
        elif age > self.get('pose_timeout_sec'):
            self.get_logger().info(
                f"panel localised by {source} (latched {age:.0f} s ago); ready "
                f"for /panel/operate. Tags in view now - {view}")

    def _image_cb(self, msg, topic):
        camera = self.cameras[topic]
        if camera['info'] is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = image if image.ndim == 2 else image[..., :3].mean(2).astype('uint8')
        detections = detect_markers(gray)
        camera['frames'] += 1
        camera['tags'] = sorted(detections)
        if not detections:
            return
        with self.lock:
            if self.panel_pose is not None and self.get('latch_panel_pose'):
                return
            generation = self.localization_generation
        solve_detections = detections
        camera_from_panel, info = panel_pose_from_markers(
            solve_detections, self.table,
            np.asarray(camera['info'].k, float).reshape(3, 3),
            np.asarray(camera['info'].d, float).ravel(),
            min_markers=1)
        if camera_from_panel is None:
            return
        if info['reprojection_px'] > self.get('max_reprojection_px'):
            # One corrupted/mis-associated tag must not discard every valid
            # observation from this camera. Fall back to its largest visible
            # tag; registered depth and the other camera then constrain range
            # and resolve the single-marker ambiguity before latching.
            fallback = None
            if len(info['markers']) > 1:
                def marker_pixel_area(marker_id):
                    corners = detections[marker_id]
                    edge_a = corners[1] - corners[0]
                    edge_b = corners[3] - corners[0]
                    return abs(float(
                        edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))

                ranked = sorted(
                    info['markers'], reverse=True,
                    key=marker_pixel_area)
                for marker_id in ranked:
                    candidate_detections = {marker_id: detections[marker_id]}
                    candidate_pose, candidate_info = panel_pose_from_markers(
                        candidate_detections, self.table,
                        np.asarray(camera['info'].k, float).reshape(3, 3),
                        np.asarray(camera['info'].d, float).ravel(),
                        min_markers=1)
                    if (candidate_pose is not None and
                            candidate_info['reprojection_px'] <=
                            self.get('max_reprojection_px')):
                        fallback = (candidate_detections, candidate_pose,
                                    candidate_info)
                        break
            if fallback is None:
                self.get_logger().warn(
                    f"{topic}: panel fit rejected, {info['reprojection_px']:.1f} px "
                    f"over {len(info['markers'])} tags {info['markers']}",
                    throttle_duration_sec=5.0)
                return
            solve_detections, camera_from_panel, fallback_info = fallback
            self.get_logger().warn(
                f"{topic}: combined tag fit {info['reprojection_px']:.1f} px; "
                f"using largest valid tag {fallback_info['markers'][0]} with "
                'depth and two-camera consensus',
                throttle_duration_sec=5.0)
            info = fallback_info
        depth = camera['depth']
        depth_stamp = camera['depth_stamp']
        if (self.get('use_depth_refinement') and depth is not None and
                depth_stamp is not None):
            depth_age = abs((rclpy.time.Time.from_msg(msg.header.stamp) -
                             depth_stamp).nanoseconds) / 1e9
            if depth_age <= self.get('depth_sync_tolerance_sec'):
                camera_from_panel, depth_info = refine_panel_pose_from_depth(
                    camera_from_panel, solve_detections, self.table,
                    np.asarray(camera['info'].k, float).reshape(3, 3), depth,
                    np.asarray(camera['info'].d, float).ravel(),
                    min_depth=self.get('depth_min_m'),
                    max_depth=self.get('depth_max_m'),
                    expected_band=self.get('depth_expected_band_m'),
                    minimum_plane_pixels=self.get('depth_min_plane_pixels'),
                    max_normal_correction_deg=
                    self.get('depth_max_normal_correction_deg'))
                info.update(depth_info)
        image_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        tf_stamp = (image_stamp if image_stamp.nanoseconds
                    else rclpy.time.Time())
        try:
            tf = self.tf_buffer.lookup_transform(
                self.get('planning_frame'), camera['frame'], tf_stamp,
                timeout=rclpy.duration.Duration(seconds=0.10))
        except TransformException as exc:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.get('planning_frame'), camera['frame'],
                    rclpy.time.Time())
                tf_time = rclpy.time.Time.from_msg(tf.header.stamp)
                tf_age = (abs((image_stamp - tf_time).nanoseconds) / 1e9
                          if tf_time.nanoseconds else 0.0)
                if tf_age > self.get('max_camera_tf_fallback_age_sec'):
                    raise TransformException(
                        f'latest transform is {tf_age:.3f} s from image')
                self.get_logger().warn(
                    f'{camera["frame"]}: exact image-time TF unavailable; '
                    f'using latest transform {tf_age:.3f} s away while arm is '
                    'stationary for calibration',
                    throttle_duration_sec=5.0)
            except TransformException as fallback_exc:
                self.get_logger().warn(
                    f"no usable transform for {camera['frame']} at image time: "
                    f'{exc}; fallback: {fallback_exc}',
                    throttle_duration_sec=5.0)
                return

        base_from_camera = np.eye(4)
        t, q = tf.transform.translation, tf.transform.rotation
        base_from_camera[:3, :3] = self._matrix_from_quaternion(q)
        base_from_camera[:3, 3] = (t.x, t.y, t.z)
        observation = base_from_camera @ camera_from_panel
        now = self.get_clock().now()
        with self.lock:
            # A trigger may have reset calibration while solvePnP/TF lookup was
            # running. Never let an observation from the previous acquisition
            # cycle satisfy the new request.
            if generation != self.localization_generation:
                return
            camera['pose'] = observation
            camera['pose_stamp'] = image_stamp
            camera['pose_info'] = info

            # Continue recording what each camera sees for diagnostics, but do
            # not let later one-camera estimates replace the accepted fused
            # pose. Restarting the node intentionally resets this latch.
            if self.panel_pose is not None and self.get('latch_panel_pose'):
                return

            recent = []
            for candidate_topic, candidate in self.cameras.items():
                if candidate['pose'] is None:
                    continue
                # Camera acquisition times, not callback arrival times: image
                # processing and DDS latency differ between the two streams.
                age = abs((image_stamp - candidate['pose_stamp']).nanoseconds) / 1e9
                if age <= self.get('camera_fusion_window_sec'):
                    recent.append((candidate_topic, candidate))

            # Anchor on the observation with most tags, then lowest pixel error.
            anchor_topic, anchor = max(
                recent,
                key=lambda item: (len(item[1]['pose_info']['markers']),
                                  -item[1]['pose_info']['reprojection_px']))
            agreeing = [(anchor_topic, anchor)]
            rejected = []
            anchor_marker_count = len(anchor['pose_info']['markers'])
            for candidate_topic, candidate in recent:
                if candidate_topic == anchor_topic:
                    continue
                # A two/three-marker observation has a much wider geometric
                # baseline than a single-marker estimate. Adding and removing
                # the intermittent gripper estimate shifted the average every
                # time marker 14 entered the frame, so keep the stronger pose
                # and use the one-marker camera only when no multi-marker
                # camera is available.
                if (anchor_marker_count >= self.get('min_markers') and
                        len(candidate['pose_info']['markers']) <
                        self.get('min_markers')):
                    continue
                distance, angle = transform_distance(
                    anchor['pose'], candidate['pose'])
                if (distance <= self.get('max_fusion_translation_m') and
                        math.degrees(angle) <= self.get('max_fusion_rotation_deg')):
                    agreeing.append((candidate_topic, candidate))
                else:
                    rejected.append((candidate_topic, distance, math.degrees(angle)))

            # A one-tag PnP can fit its four corners almost perfectly even when
            # depth is wrong, so never let its near-zero reprojection error
            # dominate a multi-tag observation. The 1 px floor makes marker
            # count the primary confidence signal in this accuracy range.
            weights = [
                len(candidate['pose_info']['markers']) /
                max(1.0, candidate['pose_info']['reprojection_px']) ** 2
                * (2.0 if candidate['pose_info'].get('depth_markers') else 1.0)
                for _, candidate in agreeing]
            fused_pose = average_transforms(
                [candidate['pose'] for _, candidate in agreeing], weights)
            fused_markers = sorted({marker
                                    for _, candidate in agreeing
                                    for marker in candidate['pose_info']['markers']})
            depth_pose_markers = sorted({marker
                                         for _, candidate in agreeing
                                         for marker in candidate['pose_info'].get(
                                             'depth_pose_markers', [])})
            depth_multimarker_cameras = sorted(
                candidate_topic for candidate_topic, candidate in agreeing
                if (len(candidate['pose_info']['markers']) >=
                    self.get('min_markers') and
                    candidate['pose_info'].get('depth_markers')))
            depth_camera_fallback = (
                bool(self.get('allow_single_depth_camera')) and
                (len(depth_pose_markers) >= self.get('min_depth_markers') or
                 bool(depth_multimarker_cameras)))

            # RGB-only localization keeps the conservative two-camera/two-tag
            # contract. A registered depth surface supplies the missing 3-D
            # constraint, so one stable marker from either camera is complete.
            camera_quorum = len(agreeing) >= self.get('required_camera_count')
            marker_quorum = len(fused_markers) >= self.get('min_markers')
            if not ((camera_quorum and marker_quorum) or
                    depth_camera_fallback):
                return

            # Do not let a momentary one-tag dropout poison an otherwise
            # stable two/three-tag calibration window. Upgrade immediately
            # when more markers become visible; ignore lower-quality samples
            # until the stronger observations have had enough time to latch.
            if self.calibration_samples:
                best_marker_count = max(
                    len(sample['markers']) for sample in self.calibration_samples)
                if len(fused_markers) < best_marker_count:
                    return
                if len(fused_markers) > best_marker_count:
                    self.calibration_samples = []

            self.calibration_samples.append(dict(
                monotonic=time.monotonic(), pose=fused_pose,
                markers=fused_markers,
                cameras=sorted(candidate_topic
                               for candidate_topic, _ in agreeing),
                depth_pose_markers=depth_pose_markers,
                depth_multimarker_cameras=depth_multimarker_cameras,
                depth_backed=depth_camera_fallback,
                reprojection_px=float(np.average(
                    [candidate['pose_info']['reprojection_px']
                     for _, candidate in agreeing], weights=weights)),
                weight=float(sum(weights))))
            sample_count = max(2, int(self.get('calibration_sample_count')))
            # Two camera callbacks can yield ~30 candidates/s. Exactly 15 of
            # those span less than the required 0.5 s forever, so retain up to
            # three windows and let sample count grow until duration is met.
            self.calibration_samples = self.calibration_samples[-3 * sample_count:]
            if len(self.calibration_samples) < sample_count:
                return
            consensus, translation_spread, rotation_spread, inlier_indices = (
                transform_inlier_consensus(
                    [sample['pose'] for sample in self.calibration_samples],
                    self.get('calibration_max_translation_spread_m'),
                    math.radians(
                        self.get('calibration_max_rotation_spread_deg')),
                    [sample['weight'] for sample in self.calibration_samples]))
            if len(inlier_indices) < sample_count:
                self.get_logger().warn(
                    f'panel calibration not stable yet: '
                    f'{len(inlier_indices)}/{len(self.calibration_samples)} '
                    f'samples in best maintenance-panel cluster, '
                    f'spread {translation_spread * 1000:.1f} mm/'
                    f'{math.degrees(rotation_spread):.2f} deg',
                    throttle_duration_sec=2.0)
                return
            inlier_samples = [self.calibration_samples[index]
                              for index in inlier_indices]
            duration = (inlier_samples[-1]['monotonic'] -
                        inlier_samples[0]['monotonic'])
            if duration < self.get('calibration_min_duration_sec'):
                return

            first = self.panel_pose is None
            previous_source = getattr(self, 'panel_source', None)
            self.panel_pose = consensus
            self.panel_stamp = now
            self.panel_info = dict(
                markers=sorted({marker
                                for sample in inlier_samples
                                for marker in sample['markers']}),
                reprojection_px=float(np.average(
                    [sample['reprojection_px'] for sample in inlier_samples],
                    weights=[sample['weight'] for sample in inlier_samples])),
                cameras=sorted({camera_topic
                                for sample in inlier_samples
                                for camera_topic in sample['cameras']}),
                samples=len(inlier_samples),
                translation_spread_m=translation_spread,
                rotation_spread_rad=rotation_spread)
            self.panel_info['depth_pose_markers'] = sorted({
                marker for sample in inlier_samples
                for marker in sample['depth_pose_markers']})
            self.panel_info['depth_backed'] = any(
                sample['depth_backed'] for sample in inlier_samples)
            self.panel_info['depth_multimarker_cameras'] = sorted({
                camera for sample in inlier_samples
                for camera in sample['depth_multimarker_cameras']})
            depth_observations = sum(
                len(candidate['pose_info'].get('depth_markers', []))
                for _, candidate in agreeing)
            self.panel_source = '+'.join(self.panel_info['cameras'])
            source_changed = self.panel_source != previous_source
            pose_to_broadcast = self.panel_pose.copy()
        if first or source_changed:
            position = pose_to_broadcast[:3, 3]
            self.get_logger().info(
                f"panel pose {'localised' if first else 'fused'} by "
                f"{self.panel_source} from tags "
                f"{self.panel_info['markers']} at "
                f"[{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}], "
                f"{self.panel_info['reprojection_px']:.2f} px; "
                f"{self.panel_info['samples']} samples over {duration:.2f} s, "
                f"spread {translation_spread * 1000:.1f} mm/"
                f"{math.degrees(rotation_spread):.2f} deg; "
                f"depth tags in final pair {depth_observations}; "
                f"depth 6-DoF tags {self.panel_info['depth_pose_markers']}; "
                f"depth multi-marker cameras "
                f"{self.panel_info['depth_multimarker_cameras']}; "
                f"pose latched")
        for rejected_topic, distance, angle in rejected:
            self.get_logger().warn(
                f'{rejected_topic}: not fused; camera estimates disagree by '
                f'{distance:.3f} m/{angle:.1f} deg', throttle_duration_sec=5.0)
        self._broadcast(pose_to_broadcast)
        if first:
            self._start_pending_sequence()

    @staticmethod
    def _matrix_from_quaternion(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def _depth_cb(self, msg, topic):
        try:
            if msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                depth_mm = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                depth = depth_mm.astype(np.float32) / 1000.0
            else:
                depth = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough').astype(np.float32)
            self.cameras[topic]['depth'] = depth
            self.cameras[topic]['depth_stamp'] = rclpy.time.Time.from_msg(
                msg.header.stamp)
        except Exception as exc:
            self.get_logger().warn(
                f'{topic}: depth conversion failed: {exc}',
                throttle_duration_sec=5.0)

    def _broadcast(self, pose):
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get('planning_frame')
        msg.child_frame_id = 'maintenance_panel'
        msg.transform.translation.x = float(pose[0, 3])
        msg.transform.translation.y = float(pose[1, 3])
        msg.transform.translation.z = float(pose[2, 3])
        x, y, z, w = quaternion_from_matrix(pose)
        msg.transform.rotation.x, msg.transform.rotation.y = float(x), float(y)
        msg.transform.rotation.z, msg.transform.rotation.w = float(z), float(w)
        self.tf_broadcaster.sendTransform(msg)

    def _fresh_pose(self):
        with self.lock:
            if self.panel_pose is None:
                return None, ('panel has never been seen - aim the gripper '
                              'camera at it until "panel localised" is logged')
            age = (self.get_clock().now() - self.panel_stamp).nanoseconds / 1e9
            markers = len(self.panel_info['markers'])
            depth_markers = len(self.panel_info.get('depth_pose_markers', []))
            depth_backed = bool(self.panel_info.get('depth_backed', False))
            pose = self.panel_pose.copy()
        if (markers < self.get('min_markers') and
                not (depth_backed and
                     depth_markers >= self.get('min_depth_markers'))):
            return None, (f'the latched pose rests on {markers} tag(s); a '
                          f"single RGB tag needs a valid registered depth plane")
        if age > self.get('pose_timeout_sec'):
            if self.get('require_fresh_pose'):
                return None, f'panel pose is {age:.1f} s stale'
            # Expected, not a fault: the tool occludes the tags on approach.
            self.get_logger().info(
                f'using the latched panel pose, {age:.1f} s old')
        return pose, None

    # ------------------------------------------------------------------ motion

    def _controller_status_cb(self, msg):
        active_values = {
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
            GoalStatus.STATUS_CANCELING,
        }
        with self.controller_status_lock:
            self.controller_status_seen = True
            self.controller_active = any(
                status.status in active_values for status in msg.status_list)

    def _wait_for_controller_idle(self):
        """Avoid preempting a preset/joystick trajectory already in flight."""
        deadline = time.monotonic() + float(self.get('controller_idle_timeout_sec'))
        announced = False
        while time.monotonic() < deadline:
            with self.controller_status_lock:
                active = self.controller_active
            if not active:
                return True, ''
            if not announced:
                self.get_logger().info(
                    'arm controller is executing another goal; waiting for it '
                    'to become idle before panel motion')
                announced = True
            time.sleep(0.05)
        return False, 'arm controller remained busy'

    def _execute_arm_trajectory(self, trajectory, label):
        """Execute a MoveIt-planned trajectory without its wedged TEM.

        A concurrent pick-home and panel request left MoveIt's trajectory
        execution manager permanently saying "another is being executed" even
        after the ros2_control goal completed. Planning remains in MoveIt; only
        delivery of the already checked/timed joint trajectory goes directly
        to the same FollowJointTrajectory controller MoveIt normally uses.
        """
        if not trajectory.points:
            return False, f'{label}: planned trajectory has no points'
        if not self.arm_controller.wait_for_server(timeout_sec=5.0):
            return False, (f'no {self.get("arm_controller_action_name")} '
                           'action server')
        idle, why = self._wait_for_controller_idle()
        if not idle:
            return False, f'{label}: {why}'
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        # Zero means start immediately. A stale planning timestamp can make a
        # valid trajectory be rejected as OLD_HEADER_TIMESTAMP.
        goal.trajectory.header.stamp.sec = 0
        goal.trajectory.header.stamp.nanosec = 0
        self.get_logger().info(
            f'{label}: sending {len(goal.trajectory.points)} planned points '
            'directly to the arm controller')
        result, why = run_action(
            self.arm_controller, goal, label, self.get('action_timeout_sec'))
        if result is None:
            return False, why
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            detail = result.result.error_string or 'no controller detail'
            return False, (f'{label} controller error '
                           f'{result.result.error_code}: {detail}')
        self.get_logger().info(f'{label}: controller completed successfully')
        return True, ''

    def _verify_tool_pose(self, expected):
        """Confirm TF reports that the arm actually reached the commanded pose."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.get('planning_frame'), self.get('tool_frame'),
                rclpy.time.Time())
        except TransformException as exc:
            return False, f'cannot verify tool motion: {exc}'
        measured = np.eye(4)
        measured[:3, :3] = self._matrix_from_quaternion(tf.transform.rotation)
        t = tf.transform.translation
        measured[:3, 3] = (t.x, t.y, t.z)
        distance, angle = transform_distance(measured, expected)
        if (distance > self.get('motion_verify_position_m') or
                angle > self.get('motion_verify_orientation_rad')):
            return False, (f'controller returned success but measured gripper_tcp '
                           f'is {distance:.3f} m/{math.degrees(angle):.1f} deg '
                           f'from its target')
        return True, ''

    def _ik_for_pose(self, transform, seed_state=None, avoid_collisions=True):
        """Return a MoveIt RobotState for an exact pose, or its error code.

        OMPL's pose-goal sampler was the source of the repeated
        ``Unable to sample any valid states for goal tree`` failure.  Asking
        MoveIt's IK service explicitly gives a useful error and, on success, a
        concrete joint goal that OMPL does not need to sample again.
        """
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.get('planning_group')
        request.ik_request.ik_link_name = self.get('tool_frame')
        if seed_state is None:
            request.ik_request.robot_state.is_diff = True
        else:
            request.ik_request.robot_state = seed_state
        request.ik_request.avoid_collisions = bool(avoid_collisions)
        timeout = float(self.get('ik_timeout_sec'))
        request.ik_request.timeout = Duration(
            sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9))
        stamped = PoseStamped()
        stamped.header.frame_id = self.get('planning_frame')
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.pose = _pose_msg(transform)
        request.ik_request.pose_stamped = stamped
        try:
            response = self.compute_ik.call(request)
        except Exception as exc:
            return None, None, f'compute_ik call failed: {exc}'
        if response is None:
            return None, None, 'compute_ik returned no response'
        code = int(response.error_code.val)
        if code != 1:
            detail = response.error_code.message or 'no detail'
            return None, code, detail
        return response.solution, code, ''

    def _select_reachable_waypoints(self, approach, contact, operate, way):
        """Choose a reachable but mechanically equivalent wrist orientation.

        Reversing both jaw directions does not change the line along which the
        jaws close.  Trying the 180-degree equivalent is especially important
        for the Rebel's bounded sixth joint.  A turn is also checked at its
        final wrist angle before the arm is allowed to move.
        """
        if not self.compute_ik.wait_for_service(timeout_sec=5.0):
            return None, ('no /compute_ik service (is MoveIt move_group '
                          'running?)')

        failures = []
        for roll in (0.0, math.pi):
            candidate_approach = roll_about_tool_z(approach, roll)
            candidate_contact = roll_about_tool_z(contact, roll)
            candidate_operate = roll_about_tool_z(operate, roll)
            solution, code, detail = self._ik_for_pose(
                candidate_approach, avoid_collisions=True)
            label = f'{math.degrees(roll):.0f} deg'
            if solution is None:
                failures.append(f'{label}: {code} ({detail})')
                continue

            # Make sure the selected IK family can continue all the way through
            # a rotary operation. Contact itself is deliberately collision
            # unchecked later because touching the panel is the task.
            if way['action'] == 'turn':
                final = candidate_contact.copy()
                final[:3, :3] = (
                    _rotation_about(candidate_contact[:3, 2],
                                    way['turn_about_approach'])
                    @ candidate_contact[:3, :3])
                final_solution, final_code, final_detail = self._ik_for_pose(
                    final, seed_state=solution, avoid_collisions=False)
                if final_solution is None:
                    failures.append(
                        f'{label} final turn: {final_code} ({final_detail})')
                    continue

            self.get_logger().info(
                f'selected reachable tool roll {label} after IK pre-check')
            return (candidate_approach, candidate_contact, candidate_operate,
                    solution), ''

        target = approach[:3, 3]
        return None, (f'no collision-free IK for standoff '
                      f'[{target[0]:.3f} {target[1]:.3f} {target[2]:.3f}] '
                      f'at either equivalent jaw roll; ' + '; '.join(failures))

    def _move_to_standoff(self, transform, ik_solution=None):
        """Use collision-aware free-space planning to reach a clear pose.

        Cartesian interpolation is reserved for the short approach from this
        standoff to the panel. Asking a Cartesian service to solve all the way
        from an arbitrary current posture produced the observed 0.32 fraction.
        """
        if not self.move_group.wait_for_server(timeout_sec=5.0):
            return False, 'no /move_action (is MoveIt move_group running?)'

        target = transform[:3, 3]
        self.get_logger().info(
            f'planning collision-aware standoff to gripper_tcp '
            f'[{target[0]:.3f} {target[1]:.3f} {target[2]:.3f}]')

        constraint = Constraints()
        if ik_solution is not None:
            wanted = dict(zip(ik_solution.joint_state.name,
                              ik_solution.joint_state.position))
            missing = [name for name in self.get('arm_joint_names')
                       if name not in wanted]
            if missing:
                return False, f'IK solution omitted arm joints: {missing}'
            tolerance = float(self.get('joint_goal_tolerance_rad'))
            for name in self.get('arm_joint_names'):
                joint = JointConstraint()
                joint.joint_name = name
                joint.position = float(wanted[name])
                joint.tolerance_above = tolerance
                joint.tolerance_below = tolerance
                joint.weight = 1.0
                constraint.joint_constraints.append(joint)
        else:
            # Kept for callers/tests that do not pre-compute IK.
            pose = _pose_msg(transform)
            position = PositionConstraint()
            position.header.frame_id = self.get('planning_frame')
            position.link_name = self.get('tool_frame')
            region = BoundingVolume()
            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [float(self.get('position_tolerance_m'))]
            region.primitives.append(sphere)
            region.primitive_poses.append(pose)
            position.constraint_region = region
            position.weight = 1.0
            constraint.position_constraints.append(position)

            orientation = OrientationConstraint()
            orientation.header.frame_id = self.get('planning_frame')
            orientation.link_name = self.get('tool_frame')
            orientation.orientation = pose.orientation
            tolerance = float(self.get('orientation_tolerance_rad'))
            orientation.absolute_x_axis_tolerance = tolerance
            orientation.absolute_y_axis_tolerance = tolerance
            orientation.absolute_z_axis_tolerance = tolerance
            orientation.parameterization = getattr(
                OrientationConstraint, 'ROTATION_VECTOR', 1)
            orientation.weight = 1.0
            constraint.orientation_constraints.append(orientation)

        goal = MoveGroup.Goal()
        goal.request.workspace_parameters.header.frame_id = self.get('planning_frame')
        goal.request.start_state.is_diff = True
        goal.request.group_name = self.get('planning_group')
        goal.request.num_planning_attempts = int(self.get('planning_attempts'))
        goal.request.allowed_planning_time = float(self.get('planning_time_sec'))
        goal.request.max_velocity_scaling_factor = float(self.get('velocity_scale'))
        goal.request.max_acceleration_scaling_factor = float(
            self.get('acceleration_scale'))
        goal.request.goal_constraints = [constraint]
        goal.planning_options = PlanningOptions()
        # Plan only here. Execution through MoveIt's global trajectory manager
        # can be wedged by another client (for example pick_home). The resulting
        # collision-checked, time-parameterized joint trajectory is sent to the
        # controller directly below.
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        goal.planning_options.replan_attempts = 0
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        result, why = run_action(
            self.move_group, goal, 'MoveGroup standoff',
            self.get('action_timeout_sec'))
        if result is None:
            return False, why
        code = result.result.error_code.val
        if code != 1:
            return False, (f'MoveGroup planning error {code} for standoff '
                           f'[{target[0]:.3f} {target[1]:.3f} {target[2]:.3f}]; '
                           'if this is NO_IK (-31), move the rover closer')
        ok, why = self._execute_arm_trajectory(
            result.result.planned_trajectory.joint_trajectory,
            'standoff trajectory')
        if not ok:
            return False, why
        verified, why = self._verify_tool_pose(transform)
        if verified:
            self.get_logger().info('standoff reached and verified from TF')
        return verified, why

    def _move_through(self, poses):
        """Cartesian move through `poses`, refusing an incomplete path."""
        request = GetCartesianPath.Request()
        request.header.frame_id = self.get('planning_frame')
        # An empty non-diff RobotState may be interpreted as the model's default
        # state. The path must begin at the measured state reached by MoveGroup.
        request.start_state.is_diff = True
        request.group_name = self.get('planning_group')
        request.link_name = self.get('tool_frame')
        request.waypoints = [_pose_msg(p) for p in poses]
        request.max_step = float(self.get('cartesian_step_m'))
        request.jump_threshold = float(self.get('cartesian_jump_threshold'))
        request.avoid_collisions = bool(self.get('contact_avoid_collisions'))
        request.max_velocity_scaling_factor = float(self.get('velocity_scale'))
        request.max_acceleration_scaling_factor = float(
            self.get('acceleration_scale'))
        request.max_cartesian_speed = float(self.get('max_cartesian_speed_mps'))
        if not self.cartesian.wait_for_service(timeout_sec=5.0):
            return False, 'no /compute_cartesian_path'
        response = self.cartesian.call(request)
        if response is None:
            return False, 'cartesian service call failed'
        if response.fraction < self.get('min_cartesian_fraction'):
            # Not a planner hiccup: a short fraction here means the straight
            # line is geometrically blocked or out of reach, and executing the
            # prefix would park the tool somewhere arbitrary.
            return False, f'cartesian fraction {response.fraction:.2f}'
        self.get_logger().info(
            f'Cartesian path complete ({len(response.solution.joint_trajectory.points)} '
            f'points); executing at <= '
            f'{self.get("max_cartesian_speed_mps"):.3f} m/s')
        ok, why = self._execute_arm_trajectory(
            response.solution.joint_trajectory, 'Cartesian trajectory')
        if not ok:
            return False, why
        verified, why = self._verify_tool_pose(poses[-1])
        if verified:
            self.get_logger().info('Cartesian target reached and verified from TF')
        return verified, why

    def _gripper(self, position):
        topic = self.get('gripper_command_topic')
        if self.gripper_pub.get_subscription_count() == 0:
            return False, f'no gripper controller subscribes to {topic}'
        command = make_joint_trajectory(
            self.get('gripper_joint_name'), position,
            self.get('gripper_command_duration_sec'))
        # Keep the trajectory stamp at zero: JointTrajectoryController defines
        # that as "start immediately". This node may use wall time while a
        # Gazebo controller uses /clock; stamping with wall time schedules the
        # command billions of seconds into the controller's future.
        self.gripper_pub.publish(command)
        self.get_logger().info(
            f'gripper command {float(position):.3f} rad via {topic}')
        self.get_clock().sleep_for(
            rclpy.duration.Duration(seconds=float(self.get('gripper_settle_sec'))))
        return True, ''

    def operate(self, name):
        control = self.controls.get(name)
        if control is None:
            return False, f'unknown control {name!r}'
        base_from_panel, why = self._fresh_pose()
        if base_from_panel is None:
            return False, why

        contact_offset = (
            self.get('tool_contact_offset_m') if control['action'] == 'turn'
            else self.get('tool_push_contact_offset_m'))
        way = control_waypoints(
            control, self.table, tool_contact_offset=contact_offset)
        approach = base_from_panel @ way['approach']
        contact = base_from_panel @ way['contact']
        operate = base_from_panel @ way['operate']
        if way['action'] == 'flick':
            # The model states which way ON is; carry it into the planning
            # frame with the recovered panel pose rather than deriving it from
            # world up or from the face's up-slope, either of which reverses
            # depending on how the console happens to be mounted.
            on_dir = base_from_panel[:3, :3] @ np.asarray(
                control['on_direction'], float)
            operate = flick_endpoint_in_planning_frame(
                contact, control['travel'], on_dir)
            delta = operate[:3, 3] - contact[:3, 3]
            self.get_logger().info(
                f'[{name}] MCB ON stroke in {self.get("planning_frame")}: '
                f'[{delta[0]:.3f} {delta[1]:.3f} {delta[2]:.3f}] m, '
                f'{control["motion_direction"]} on the console face; '
                f'the handle rises {delta[2] * 1000:+.0f} mm '
                '(positive = lever up = ON)')

        selected, why = self._select_reachable_waypoints(
            approach, contact, operate, way)
        if selected is None:
            return False, f'IK pre-check: {why}'
        approach, contact, operate, ik_solution = selected

        # Jaws are set BEFORE going in: opening or closing them against the
        # console is how you shear a fingertip off a control.
        gripper_target = (self.get('gripper_open_position') if way['grip']
                          else self.get('gripper_close_position'))
        gripper_ok, gripper_why = self._gripper(gripper_target)
        if not gripper_ok:
            return False, f'gripper: {gripper_why}'

        ok, why = self._move_to_standoff(approach, ik_solution)
        if not ok:
            return False, f'standoff: {why}'

        ok, why = self._move_through([contact])
        if not ok:
            return False, f'approach: {why}'

        if way['action'] == 'turn':
            gripper_ok, gripper_why = self._gripper(
                self.get('gripper_close_position'))
            if not gripper_ok:
                return False, f'gripper: {gripper_why}'
            steps = []
            for k in range(1, 9):
                angle = way['turn_about_approach'] * k / 8.0
                spun = contact.copy()
                spun[:3, :3] = _rotation_about(contact[:3, 2], angle) @ contact[:3, :3]
                steps.append(spun)
            ok, why = self._move_through(steps)
            # Always release before reporting a failed turn; the sequence may
            # continue to another enabled control and must never carry the
            # previous selector in closed jaws.
            self._gripper(self.get('gripper_open_position'))
            final_pose = steps[-1]
        else:
            ok, why = self._move_through([operate])
            final_pose = operate
        if not ok:
            return False, f'{way["action"]}: {why}'

        # Leave along the console normal while retaining the final orientation.
        # Returning diagonally to the original approach after an upward MCB
        # flick can drag the toggle downward again while the finger is close to
        # the face. Clear the panel at the operated endpoint first.
        retreat = final_pose.copy()
        retreat[:3, 3] += approach[:3, 3] - contact[:3, 3]
        back, _ = self._move_through([retreat])
        return True, f'{name} operated' + ('' if back else ' (retract failed)')

    # ------------------------------------------------------------------ command

    def _enabled_control_names(self):
        """Return YAML-enabled controls in the deterministic task-table order."""
        return [c['name'] for c in self.table['controls']
                if self.get(f"controls.{c['name']}")]

    def _queue_or_run(self, names, recalibrate=False):
        """Run now, or queue until a newly requested fused pose is available."""
        names = list(names)
        if recalibrate and self.operation_lock.locked():
            detail = ('busy: cannot recalibrate panel pose while another panel '
                      'operation is running')
            self.get_logger().warn(detail)
            self.status_pub.publish(String(data=detail))
            return
        with self.pending_lock:
            if recalibrate:
                # A second true message while acquisition is pending should
                # reload/update the queued YAML controls, but must not throw
                # away the marker samples already collected. Repeated trigger
                # publications previously kept incrementing the calibration
                # cycle and could prevent the arm from ever starting.
                if self.pending_names is not None:
                    self.pending_names = names
                    detail = (
                        f'fresh marker calibration cycle '
                        f'{self.localization_generation} already in progress; '
                        'updated queued controls without resetting samples: ' +
                        ', '.join(names))
                    self.get_logger().info(detail)
                    self.status_pub.publish(String(data=detail))
                    return
                with self.lock:
                    self.localization_generation += 1
                    self.panel_pose = None
                    self.panel_stamp = None
                    self.calibration_samples = []
                    for camera in self.cameras.values():
                        camera['pose'] = None
                        camera['pose_stamp'] = None
                        camera['pose_info'] = None
                self.pending_names = names
                detail = (
                    f'fresh marker recalibration requested (cycle '
                    f'{self.localization_generation}); waiting for new '
                    'stable depth-backed observations before operating: ' +
                    ', '.join(names))
                self.get_logger().info(detail)
                self.status_pub.publish(String(data=detail))
                return
            with self.lock:
                ready = self.panel_pose is not None
            if not ready:
                self.pending_names = names
                detail = ('queued until stable panel localisation: ' +
                          ', '.join(names))
                self.get_logger().info(detail)
                self.status_pub.publish(String(data=detail))
                return
        self._run_sequence(names)

    def _start_pending_sequence(self):
        with self.pending_lock:
            names = self.pending_names
            self.pending_names = None
        if not names:
            return
        self.get_logger().info(
            'panel localised; starting queued controls: ' + ', '.join(names))
        # Do not block the image callback that established the fused pose.
        threading.Thread(
            target=self._run_sequence, args=(names,), daemon=True,
            name='panel-operation').start()

    def _run_sequence(self, names):
        if not names:
            detail = 'no controls are enabled in panel_tasks.yaml'
            self.get_logger().warn(detail)
            self.status_pub.publish(String(data=detail))
            return
        if not self.operation_lock.acquire(blocking=False):
            detail = 'busy: another panel operation is already running'
            self.get_logger().warn(detail)
            self.status_pub.publish(String(data=detail))
            return
        try:
            succeeded = []
            failed = []
            for name in names:
                self.get_logger().info(f'[{name}] starting')
                try:
                    ok, detail = self.operate(name)
                except Exception as exc:  # keep one failed command from killing ROS
                    ok = False
                    detail = f'unexpected operation error: {type(exc).__name__}: {exc}'
                log_operation_result(self.get_logger(), name, ok, detail)
                self.status_pub.publish(
                    String(data=f'{name}: {"ok" if ok else detail}'))
                if ok:
                    succeeded.append(name)
                else:
                    failed.append(name)
                if not ok and name != names[-1]:
                    self.get_logger().error(
                        f'[{name}] failed; continuing with the next '
                        'YAML-enabled control')

            summary = (f'sequence complete: {len(succeeded)}/{len(names)} '
                       'controls succeeded')
            if failed:
                summary += '; failed: ' + ', '.join(failed)
                self.get_logger().error(summary)
            else:
                self.get_logger().info(summary)
            self.status_pub.publish(String(data=summary))
        finally:
            self.operation_lock.release()

    def _operate_cb(self, msg):
        target = msg.data.strip()
        groups = {
            'all_breakers': [c['name'] for c in self.table['controls']
                             if c['action'] == 'flick'],
            'all_buttons': [c['name'] for c in self.table['controls']
                            if c['action'] == 'press'],
            'all_switches': [c['name'] for c in self.table['controls']
                             if c['action'] == 'turn'],
        }
        names = groups.get(target, [target])
        self._queue_or_run(names)

    def _operate_enabled_cb(self, msg):
        if not msg.data:
            self.get_logger().info(
                'ignoring /panel/operate_enabled false (true is the trigger)')
            return
        config_file = self.get('config_file')
        if not config_file:
            detail = ('cannot reload enabled controls: config_file parameter '
                      'is empty')
            self.get_logger().error(detail)
            self.status_pub.publish(String(data=detail))
            return
        try:
            names, resolved, modified = load_enabled_controls(
                config_file, self.get_name(),
                [c['name'] for c in self.table['controls']])
        except (OSError, ValueError, yaml.YAMLError) as exc:
            detail = f'YAML reload failed; no motion started: {exc}'
            self.get_logger().error(detail)
            self.status_pub.publish(String(data=detail))
            return
        detail = (
            f'YAML reloaded for this trigger from {resolved} '
            f'(mtime {modified:.3f}); enabled: ' +
            (', '.join(names) if names else 'none'))
        self.get_logger().info(detail)
        self.status_pub.publish(String(data=detail))
        self._queue_or_run(
            names,
            recalibrate=bool(self.get('recalibrate_on_operate_enabled')))


def main():
    rclpy.init()
    node = PanelOperator()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
