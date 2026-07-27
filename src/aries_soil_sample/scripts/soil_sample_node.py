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

from dataclasses import replace

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from action_msgs.srv import CancelGoal
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
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetPlanningScene,
    GetPositionFK,
    GetPositionIK,
)
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

from aries_soil_sample import deposit as deposit_lib
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
        # Goal handles we own, so Ctrl-C can cancel them instead of abandoning
        # them. An abandoned MoveIt goal leaves trajectory_execution_manager
        # believing a trajectory is still in flight, and every later plan then
        # dies instantly with CONTROL_FAILED (-4). Nothing can clear that from
        # outside -- cancel-all returns zero goals at both MoveIt action servers
        # AND both controllers, because no goal is live; the manager's own state
        # is stuck. There is no stop-execution service, so the only cure is
        # restarting move_group. Hence: never abandon a goal.
        # A list, not a set: rclpy's ClientGoalHandle defines __eq__ without
        # __hash__, so it is unhashable and set.add() raises.
        self._active_goals = []
        self._active_goals_lock = threading.Lock()

        p = self.declare_parameter
        p('planning_frame', 'base_link')
        p('planning_link', 'arm_gripper_base_link')
        p('planning_group', 'igus_rebel_arm')
        p('finger_type', 'bucket')
        p('use_aligned_depth', True)
        p('auto_start', False)
        # How long to wait for the rest of the stack before the first motion.
        # 5 s of ActionClient discovery is not enough right after a sim launch:
        # the symptom is `MoveGroup action unavailable` while move_group is in
        # fact running perfectly well.
        p('startup_wait_sec', 40.0)

        p('work_region_x', [0.40, 0.65])
        p('work_region_y', [-0.18, 0.18])
        p('work_region_z', [-0.20, 0.10])
        p('prefer_scoop_xy', [0.52, 0.03])

        p('height_map_cell_m', 0.010)
        p('height_map_percentile', 50.0)
        p('height_map_min_points_per_cell', 2)
        p('height_map_min_valid_fraction', 0.35)
        p('depth_stride', 2)
        # Pool several depth frames into one height map. One frame is a noisy
        # sample; the surveyed ground has moved tens of mm between runs, which is
        # enough to push the penetrate pose out of the envelope.
        p('ground_accumulate_frames', True)
        p('ground_frames_to_accumulate', 5)
        p('ground_frame_interval_sec', 0.15)

        p('scoop_footprint_m', 0.060)
        p('scoop_max_roughness_m', 0.006)
        p('scoop_max_slope_deg', 12.0)
        p('scoop_min_coverage', 0.70)
        p('scoop_max_candidate_sites', 8)

        # Height ABOVE the soil surface at which the scoop stroke begins. With a
        # straight-down entry this is literally vertical height; with a tilted
        # entry it is measured along the entry axis.
        p('scoop_start_above_ground_m', 0.030)
        p('scoop_depth_m', 0.030)
        # Non-zero by necessity: a vertical bucket cannot reach ground level with
        # a 214 mm gripper. See the config for the measurement.
        p('scoop_attack_deg', 30.0)
        p('scoop_attack_azimuth_ref', [1.0, 0.0, 0.0])
        p('scoop_max_depth_m', 0.060)
        # Floor for the automatic depth reduction below. A scoop shallower than
        # this is not worth taking.
        p('scoop_min_depth_m', 0.012)
        # Cartesian for the whole stroke: one straight-line path through
        # approach -> entry -> penetrate, then (after the close) extract. The
        # only split is at the close, which is a gripper command, not a motion.
        # The alternative (joint-space hop to the approach, then Cartesian from
        # there) leaves the arm at a pose MoveIt reached by its own route, and the
        # descend then had to start from wherever that put it -- which is how
        # `Cartesian path solved only 0.00` happened.
        p('scoop_all_cartesian', True)
        p('scoop_depth_margin_m', 0.010)
        p('scoop_retrace_extraction', True)

        # FULL OPEN to descend, FULL CLOSE to collect. No partial angles.
        p('bucket_full_open_q', -1.57)
        p('bucket_full_close_q', 0.07)
        # Reference point the scoop geometry is measured to. 0 = derive it from
        # the CLOSED jaw angle, which is the right choice and worth stating:
        # the four-bar "contact" is where the two jaw faces would MEET, and wide
        # open that is a virtual point 134 mm from the link, high between the
        # splayed shells. It is not what touches the soil. Referencing the scoop
        # to it drove the link to 134 mm above the ground -- unreachable, and it
        # would have buried the shells. The sample ends up where the CLOSED
        # bucket is, so the closed contact (219.5 mm) is the meaningful datum.
        p('scoop_reference_offset_m', 0.0)
        # Start closing DURING the descent, this far above the surveyed surface,
        # so the shells sweep material as they shut.
        p('close_during_descent', True)
        p('close_start_above_ground_m', 0.020)
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
        # Hard depth limit, expressed RELATIVE to the ground the camera detected.
        # An absolute Z floor is unusable in deployment: you do not know the
        # terrain height in advance, and a floor set from one site silently
        # rejects every scoop at a site 40 mm lower. This one travels with the
        # measured surface.
        p('max_depth_below_ground_m', 0.060)
        # Optional SECOND floor in absolute planning-frame Z, for a known fixed
        # workspace. Leave at the default to disable it -- in the field the
        # relative limit above is the one that means anything.
        p('absolute_min_contact_z', -9.0)
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
        # pick_home is the START posture, the posture the loaded bucket RETURNS
        # to, and the posture it leaves from to reach the box -- one posture, as
        # requested, matching aries_vision_grasp's pick_home.
        p('home_joint_positions', [0.0, 0.366519, 1.18682, 0.0349066, 1.55334, 1.50098])
        # Return to home along a straight Cartesian line with the sample aboard.
        # Falls back to a joint move if the straight line cannot be solved --
        # better a curved return than a bucket stranded in the hole.
        p('cartesian_return_home', True)
        p('return_to_transport_after_scoop', True)
        # Deposit the sample into the box carried ON THE ROVER, same target the
        # probe task uses (base_box_center_xyz in aries_vision_grasp), so the
        # sample ends up where the mission wants it rather than back on the
        # ground. Contact point, not link origin, in the planning frame.
        # Sampling point (config/sample_points.yaml, loaded last).
        p('use_fixed_sample_point', False)
        # XY ONLY. The height always comes from the depth camera: in deployment
        # the terrain height is unknown, so a configured Z would be a guess that
        # either scoops air or drives the bucket into the ground.
        p('sample_point_xy', [0.470, 0.110])
        # Sample DIRECTLY BELOW the pick_home posture. This is the deployment
        # default because it needs no map, no survey region and no coordinate at
        # all: the arm goes home, the camera looks at whatever ground is under
        # it, and the bucket goes straight down into it. The XY is computed from
        # pick_home's forward kinematics at run time, so it follows the posture
        # rather than being copied out of it.
        p('sample_below_home', True)
        # Half-width of the survey window centred on that column. Big enough to
        # fit the bucket footprint and judge the ground around it, small enough
        # that the site scorer cannot wander off to some other patch.
        p('sample_below_home_halfwidth_m', 0.070)
        p('sample_point_strict', False)

        p('deposit_enabled', True)
        # Describe the box ONCE, the way the probe task does, and derive every
        # pose from it. Move the box and the dump pose follows.
        p('deposit_box_center_xyz', [0.003, 0.215, 0.287])
        p('deposit_box_dimensions_xyz', [0.14, 0.20, 0.15])
        p('deposit_box_rpy', [0.0, 0.0, 0.0])
        p('deposit_box_wall_thickness_m', 0.006)
        # Height above the box RIM at which the jaws open. 38 mm is the measured
        # 4/4 collision-free sweet spot over the rover box.
        p('deposit_rim_clearance_m', 0.038)
        # Shift of the dump point within the opening, in the BOX frame.
        p('deposit_offset_xy', [0.0, 0.0])
        # Keep the dump point this far inside the opening, so a mis-set offset
        # cannot tip the sample onto the rim.
        p('deposit_edge_margin_m', 0.015)
        # A known-good dump POSTURE beats a computed pose: joint values verified
        # over the box are immune to the box moving out of the arm's comfortable
        # band. Defaults are the posture demonstrated in RViz, which FK puts at
        # bucket contact (0.137,0.205,0.412) with the mouth 2 deg off straight
        # down -- over the opening, 113 mm above the rim.
        p('deposit_use_joint_posture', True)
        p('deposit_joint_positions', [1.3090, -0.3665, 1.7977, -0.0175, 1.6755, 2.8099])
        # If the posture is off, the derived pose is searched upward from
        # deposit_rim_clearance_m: tipping from higher still lands in the box,
        # and height is what the arm's envelope actually cares about. The old
        # single fixed clearance failed the moment the box moved down.
        p('deposit_max_rim_clearance_m', 0.150)
        p('deposit_dump_open_q', -1.30)
        p('deposit_settle_sec', 2.0)
        p('publish_deposit_box_marker', True)
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
        self.startup_wait_sec = max(0.0, float(g('startup_wait_sec').value))

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
        self.ground_accumulate = bool(g('ground_accumulate_frames').value)
        self.ground_frames = max(1, int(g('ground_frames_to_accumulate').value))
        self.ground_frame_interval_sec = max(
            0.0, float(g('ground_frame_interval_sec').value))

        self.footprint_m = max(0.01, float(g('scoop_footprint_m').value))
        self.max_roughness_m = float(g('scoop_max_roughness_m').value)
        self.max_slope_deg = float(g('scoop_max_slope_deg').value)
        self.min_coverage = float(g('scoop_min_coverage').value)
        self.max_sites = max(1, int(g('scoop_max_candidate_sites').value))

        self.scoop_params = scoop_lib.ScoopParams(
            standoff_m=float(g('scoop_start_above_ground_m').value),
            depth_m=float(g('scoop_depth_m').value),
            attack_deg=float(g('scoop_attack_deg').value),
            max_depth_m=float(g('scoop_max_depth_m').value),
            depth_margin_m=float(g('scoop_depth_margin_m').value),
        )
        self.attack_azimuth_ref = np.array(
            [float(v) for v in g('scoop_attack_azimuth_ref').value], dtype=np.float64)

        self.entry_q = float(g('bucket_full_open_q').value)
        self.close_during_descent = bool(g('close_during_descent').value)
        self.close_start_above_ground_m = float(g('close_start_above_ground_m').value)
        self.close_q = float(g('bucket_full_close_q').value)
        ref = float(g('scoop_reference_offset_m').value)
        self.scoop_ref_offset = (
            fourbar.contact_offset(self.close_q, fourbar.CONTACT_Y_OFFSET_M)
            if ref <= 0.0 else
            np.array([0.0, fourbar.CONTACT_Y_OFFSET_M, ref], dtype=np.float64))
        self.get_logger().info(
            f'Scoop referenced to {self.scoop_ref_offset[2]*1000:.0f}mm from the link '
            f'(closed-jaw datum); descends fully open (q={self.entry_q:+.2f}, gap '
            f'{fourbar.gap_from_q(self.entry_q)*1000:.0f}mm) and collects fully closed '
            f'(q={self.close_q:+.2f}).')
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
        self.max_depth_below_ground_m = max(
            0.0, float(g('max_depth_below_ground_m').value))
        self.prescreen = bool(g('prescreen_ik_all_waypoints').value)
        self.roll_candidates = max(1, int(g('scoop_wrist_roll_candidates').value))
        self.ik_timeout_sec = max(0.05, float(g('ik_prescreen_timeout_sec').value))
        self.max_attempts = max(1, int(g('max_scoop_attempts').value))
        self.scoop_min_depth_m = max(0.002, float(g('scoop_min_depth_m').value))
        self.scoop_all_cartesian = bool(g('scoop_all_cartesian').value)
        self.octomap_off_during_scoop = bool(g('octomap_disable_during_scoop').value)

        self.arm_joints = [str(v) for v in g('arm_joint_names').value]
        self.home_q = [float(v) for v in g('home_joint_positions').value]
        self.survey_q = self.home_q
        self.transport_q = self.home_q
        self.cartesian_return_home = bool(g('cartesian_return_home').value)
        self.return_transport = bool(g('return_to_transport_after_scoop').value)
        self.use_fixed_sample_point = bool(g('use_fixed_sample_point').value)
        self.sample_point = np.array(
            [float(v) for v in g('sample_point_xy').value], dtype=np.float64).reshape(2,)
        self.sample_below_home = bool(g('sample_below_home').value)
        self.sample_below_home_halfwidth_m = max(
            0.03, float(g('sample_below_home_halfwidth_m').value))
        self.sample_point_strict = bool(g('sample_point_strict').value)
        if self.use_fixed_sample_point:
            self.get_logger().info(
                f'[sample] fixed sampling point XY '
                f'({self.sample_point[0]:.3f},{self.sample_point[1]:.3f}); the height '
                'always comes from the depth camera.')

        self.deposit_enabled = bool(g('deposit_enabled').value)
        self.deposit_box = deposit_lib.DepositBox(
            centre=[float(v) for v in g('deposit_box_center_xyz').value],
            dimensions=[float(v) for v in g('deposit_box_dimensions_xyz').value],
            rpy=[float(v) for v in g('deposit_box_rpy').value],
            wall_thickness_m=float(g('deposit_box_wall_thickness_m').value),
        )
        self.deposit_rim_clearance_m = float(g('deposit_rim_clearance_m').value)
        self.deposit_offset_xy = [float(v) for v in g('deposit_offset_xy').value]
        self.deposit_edge_margin_m = float(g('deposit_edge_margin_m').value)
        self.deposit_use_joint_posture = bool(g('deposit_use_joint_posture').value)
        self.deposit_joint_positions = [
            float(v) for v in g('deposit_joint_positions').value]
        self.deposit_max_rim_clearance_m = float(g('deposit_max_rim_clearance_m').value)
        self.deposit_dump_open_q = float(g('deposit_dump_open_q').value)
        self.deposit_settle_sec = float(g('deposit_settle_sec').value)
        self.publish_box_marker = bool(g('publish_deposit_box_marker').value)

        # Fail loudly at startup rather than mid-cycle with the sample in the
        # bucket: a box you cannot dump into is a configuration error.
        ok, why = self.deposit_box.validate(
            self.deposit_rim_clearance_m, self.deposit_offset_xy,
            self.deposit_edge_margin_m)
        if self.deposit_enabled and not ok:
            self.get_logger().error(
                f'[deposit] deposit box is misconfigured: {why}. Deposit is DISABLED '
                'for this run; the cycle will keep the sample in the bucket.')
            self.deposit_enabled = False
        elif self.deposit_enabled:
            self.get_logger().info(f'[deposit] box {self.deposit_box.summary}; {why}')
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

        self.move_action_name = str(g('move_action_name').value)
        self.execute_action_name = str(g('execute_action_name').value)
        self.move_client = ActionClient(self, MoveGroup, self.move_action_name,
                                        callback_group=self._cb)
        self.exec_client = ActionClient(self, ExecuteTrajectory,
                                        self.execute_action_name,
                                        callback_group=self._cb)
        self.grip_client = ActionClient(self, FollowJointTrajectory,
                                        str(g('gripper_action_name').value),
                                        callback_group=self._cb)
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path',
                                             callback_group=self._cb)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik',
                                           callback_group=self._cb)
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk',
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

    def _points_from_latest_depth(self) -> Optional[Tuple[np.ndarray, float]]:
        """Back-project the newest depth frame into the planning frame.

        Returns ``(points, stamp_sec)`` or None. Each frame carries its OWN TF,
        so accumulating frames stays correct even if the arm drifts slightly
        between them.
        """
        depth = self.latest_depth
        info = self.camera_info
        if depth is None or info is None:
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        if info.width and depth.shape[1] != info.width:
            sc = depth.shape[1] / float(info.width)
            fx, cx, fy, cy = fx * sc, cx * sc, fy * sc, cy * sc
        try:
            pts_cam = backproject_depth(depth, fx, fy, cx, cy, self.depth_stride)
        except ValueError as exc:
            self.get_logger().error(f'cannot back-project depth: {exc}')
            return None
        if len(pts_cam) == 0:
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
                        f'{self.planning_frame}; used the latest transform.',
                        throttle_duration_sec=10.0)
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
        return pts_cam @ R.T + t, float(stamp.nanoseconds) * 1e-9

    def survey(self) -> Optional[terrain.HeightMap]:
        """Height map of the work region, built from ACCUMULATED depth frames.

        One depth frame is a noisy sample of the ground: the surveyed height has
        moved by tens of millimetres between runs, which is enough to push the
        penetrate pose out of the arm's envelope. Several frames pooled into one
        height map fix that cheaply -- the per-cell percentile in
        ``terrain.height_map`` already does the averaging, so accumulating is
        just a matter of feeding it more points, each transformed with its own TF.

        The per-frame spread is logged: if it is large the drift is sensor noise,
        if it is small the drift is real geometry moving between runs. That
        distinction is worth having, so it is measured rather than assumed.
        """
        if self.latest_depth is None or self.camera_info is None:
            self.get_logger().error('no depth frame or camera_info yet; cannot survey.')
            return None

        wanted = self.ground_frames if self.ground_accumulate else 1
        clouds: List[np.ndarray] = []
        per_frame_median: List[float] = []
        last_stamp = -1.0
        deadline = time.monotonic() + max(2.0, wanted * self.ground_frame_interval_sec * 4)
        while len(clouds) < wanted and time.monotonic() < deadline:
            got = self._points_from_latest_depth()
            if got is None:
                time.sleep(0.05)
                continue
            pts, stamp_sec = got
            if stamp_sec <= last_stamp:      # wait for a genuinely new frame
                time.sleep(0.02)
                continue
            last_stamp = stamp_sec
            clouds.append(pts)
            inside = pts[self.region.contains(pts)]
            if len(inside):
                per_frame_median.append(float(np.median(inside[:, 2])))
            if len(clouds) < wanted:
                time.sleep(self.ground_frame_interval_sec)

        if not clouds:
            self.get_logger().error('no usable depth frame for the survey.')
            return None

        pts = np.vstack(clouds)
        hmap = terrain.height_map(pts, self.region, self.cell_m, self.percentile)
        spread = ''
        if len(per_frame_median) > 1:
            lo, hi = min(per_frame_median), max(per_frame_median)
            spread = (f', per-frame ground median {lo:.3f}..{hi:.3f} '
                      f'(spread {(hi-lo)*1000:.1f}mm)')
        self.get_logger().info(
            f'Survey: {len(clouds)} frame(s), {len(pts)} pts -> grid '
            f'{hmap.shape[0]}x{hmap.shape[1]} @ {self.cell_m*1000:.0f}mm, '
            f'{hmap.valid_fraction*100:.0f}% of cells observed{spread}.')
        if hmap.valid_fraction < self.min_valid_fraction:
            self.get_logger().error(
                f'survey too sparse ({hmap.valid_fraction*100:.0f}% < '
                f'{self.min_valid_fraction*100:.0f}%): the camera is not looking at the '
                'work region, or the region is wrong.')
            return None
        return hmap

    def select_site(self, hmap: terrain.HeightMap) -> List[terrain.ScoopSite]:
        if self.use_fixed_sample_point:
            site, why = terrain.site_at_xy(
                hmap, self.sample_point, self.footprint_m,
                max_roughness_m=self.max_roughness_m,
                max_slope_deg=self.max_slope_deg,
                min_coverage=self.min_coverage,
                min_points_per_cell=self.min_pts_cell,
            )
            if site is not None:
                self.get_logger().info(
                    f'[sample] using the configured point: {site.summary}')
                self._publish_site_markers([site])
                return [site]
            if self.sample_point_strict:
                self.get_logger().error(
                    f'[sample] configured point rejected: {why}. sample_point_strict '
                    'is set, so no scoop will be attempted.')
                return []
            self.get_logger().warning(
                f'[sample] configured point rejected: {why}. Falling back to '
                'automatic site selection.')

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
        # Depth ladder. The DEEPEST waypoint is the binding constraint, and how
        # deep the arm can go depends on where the ground actually is -- surveyed
        # ground has read -0.167, -0.175 and -0.185 across runs depending on the
        # survey posture, and 10 mm of that moves the penetrate pose past the
        # envelope. Rather than fail (the symptom was `penetratex12` with every
        # approach and entry solving), ask for the configured depth and settle for
        # a shallower real scoop if that is what the arm can reach.
        requested = float(self.scoop_params.depth_m)
        # depth 0 is a legitimate setting (skim the surface, do not penetrate),
        # and there is nothing to reduce -- do not build a ladder that reads as
        # "no depth from 0mm down to 12mm", which says nothing useful.
        ladder = [requested]
        if requested > self.scoop_min_depth_m:
            while ladder[-1] > self.scoop_min_depth_m * 1.5:
                ladder.append(max(self.scoop_min_depth_m, ladder[-1] * 0.66))
        for attempt_depth in ladder:
            params = replace(self.scoop_params, depth_m=attempt_depth)
            waypoints, depth, note = scoop_lib.plan_scoop(
                site.centre, site.normal, params,
                azimuth_ref=self.attack_azimuth_ref)
            if note:
                self.get_logger().warning(f'[scoop] penetration depth {note}.')
            built = self._poses_for_depth(site, waypoints, depth)
            if built is not None:
                if attempt_depth < requested - 1e-6:
                    self.get_logger().warning(
                        f'[scoop] reduced the scoop from {requested*1000:.0f}mm to '
                        f'{depth*1000:.0f}mm: the deeper penetrate pose had no IK '
                        'solution at any wrist roll. Still a real sample, just a '
                        'smaller one.')
                return built
        if len(ladder) > 1:
            self.get_logger().error(
                f'[scoop] no depth from {requested*1000:.0f}mm down to '
                f'{ladder[-1]*1000:.0f}mm gives a solvable scoop at this site.')
        else:
            self.get_logger().error(
                f'[scoop] the scoop is unsolvable at this site even at '
                f'{requested*1000:.0f}mm depth. With the bucket datum '
                f'{self.scoop_ref_offset[2]*1000:.0f}mm from the link, reaching ground '
                f'z={site.centre[2]:.3f} needs the wrist at z='
                f'{site.centre[2]+self.scoop_ref_offset[2]:.3f}; if that is below the '
                'arm\'s limit no wrist roll or depth can help. Either the datum is too '
                'small (measure the real bucket tip and set scoop_reference_offset_m) '
                'or the ground is out of reach from this posture.')
        return None

    def _poses_for_depth(self, site, waypoints, depth):
        """Roll search for one candidate depth; None if no roll solves."""

        # The hard Z floor depends only on the contact positions, not on the wrist
        # roll, so it is checked once before any IK work.
        # Floor derived from the DETECTED ground at this site, plus the optional
        # absolute one. Relative is what protects a real deployment: it moves
        # with the terrain instead of encoding one site's height.
        ground_z = float(site.centre[2])
        relative_floor = ground_z - self.max_depth_below_ground_m
        floor = max(relative_floor, self.abs_min_z)
        for wp in waypoints:
            if float(wp.position[2]) < floor:
                which = ('max_depth_below_ground_m='
                         f'{self.max_depth_below_ground_m*1000:.0f}mm below the detected '
                         f'ground at z={ground_z:.3f}'
                         if relative_floor >= self.abs_min_z
                         else f'absolute_min_contact_z={self.abs_min_z:.3f}')
                self.get_logger().error(
                    f'[safety] waypoint {wp.label} at z={wp.position[2]:.3f} is below the '
                    f'floor z={floor:.3f} ({which}). Refusing the scoop: a bad depth frame '
                    'must not drive the bucket into the ground.')
                return None

        # The bucket's contact point swings a long way with the linkage, so take
        # the offset at the angle the jaws will actually be holding during entry.
        offset = self.scoop_ref_offset
        R_now = self._current_tool_rotation()
        prefer_pinch = R_now[:, 0] if R_now is not None else None
        axis = waypoints[0].tool_axis

        self.get_logger().info(
            f'[scoop] site z={site.centre[2]:.3f}, trying {depth*1000:.0f}mm along '
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
        self.get_logger().warning(
            f'[scoop] no wrist roll out of {len(frames)} solves the whole scoop at '
            f'{depth*1000:.0f}mm (first failing waypoint per roll: {summary}).')
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

    def _pose_like(self, ref_pose: Pose, contact: np.ndarray,
                   _unused: Pose) -> Pose:
        """A link pose at ``contact``, holding ``ref_pose``'s orientation.

        The whole stroke keeps one wrist orientation, so an intermediate
        waypoint only needs its position recomputed from the contact point.
        """
        R = quat_to_matrix(ref_pose.orientation)
        link = scoop_lib.link_position_for_contact(contact, R, self.scoop_ref_offset)
        pose = Pose()
        pose.position.x = float(link[0])
        pose.position.y = float(link[1])
        pose.position.z = float(link[2])
        pose.orientation = ref_pose.orientation
        return pose

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

    def _track(self, goal_handle):
        if goal_handle is not None:
            with self._active_goals_lock:
                if not any(h is goal_handle for h in self._active_goals):
                    self._active_goals.append(goal_handle)
        return goal_handle

    def _untrack(self, goal_handle) -> None:
        if goal_handle is None:
            return
        with self._active_goals_lock:
            # Identity, not equality: ClientGoalHandle.__eq__ compares goal ids
            # and list.remove() would use it, but identity is what we mean here.
            self._active_goals = [h for h in self._active_goals if h is not goal_handle]

    def cancel_active_goals_on_shutdown(self) -> None:
        """Cancel every goal we still own, spinning by hand.

        Called from main() AFTER the executor has stopped, so it cannot rely on
        the background executor and spins the node itself. Best effort: on
        shutdown a failure here is not worth raising, but skipping it wedges
        MoveIt for the next run.
        """
        with self._active_goals_lock:
            handles = list(self._active_goals)
            self._active_goals.clear()
        if not handles:
            return
        self.get_logger().warning(
            f'shutting down with {len(handles)} goal(s) in flight; cancelling so '
            'MoveIt does not stay wedged for the next run.')
        for gh in handles:
            try:
                fut = gh.cancel_goal_async()
                rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
            except Exception:  # noqa: BLE001 - shutdown path
                pass

    def cancel_stale_execution(self) -> bool:
        """Clear a trajectory MoveIt still believes is executing.

        A Ctrl-C'd run abandons its action goals without cancelling them, so
        MoveIt's trajectory_execution_manager keeps the old trajectory in flight.
        Every later plan then dies instantly with CONTROL_FAILED (-4) and
        `Cannot push a new trajectory while another is being executed` -- planning
        succeeded, execution was refused. There is no stop service to call, and
        the stale goal belongs to a dead process so we hold no handle to it.

        The ROS 2 action spec covers exactly this: a CancelGoal request with a
        zeroed goal id AND a zero stamp means "cancel all goals", whoever owns
        them. That is what this sends, to both MoveIt action servers.
        """
        cleared = False
        for action in (self.move_action_name, self.execute_action_name):
            srv = f'{action.rstrip("/")}/_action/cancel_goal'
            cli = self.create_client(CancelGoal, srv, callback_group=self._cb)
            if not cli.wait_for_service(timeout_sec=3.0):
                self.get_logger().warning(f'[recover] {srv} unavailable.')
                continue
            req = CancelGoal.Request()          # zero uuid + zero stamp = cancel all
            res = self._wait(cli.call_async(req), 8.0)
            n = len(res.goals_canceling) if res is not None else 0
            if n:
                cleared = True
            self.get_logger().warning(
                f'[recover] cancel-all on {action}: {n} goal(s) cancelling.')
        if cleared:
            self._sleep(1.0)
        return cleared

    def move_to_posture(self, positions: Sequence[float], label: str,
                        allow_recovery: bool = True) -> bool:
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
        gh = self._track(self._wait(self.move_client.send_goal_async(goal), 10.0))
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] MoveGroup rejected the goal.')
            return False
        result = self._wait(gh.get_result_async(), self.planning_time + 40.0)
        self._untrack(gh)
        if result is None:
            self.get_logger().error(f'[{label}] MoveGroup did not return a result.')
            return False
        code = result.result.error_code.val
        if code == -4 and allow_recovery:
            # CONTROL_FAILED. Planning worked; execution was refused. Overwhelmingly
            # this is a trajectory left in flight by a previous run, so clear it
            # and try once more rather than reporting a dead end.
            self.get_logger().warning(
                f'[{label}] CONTROL_FAILED (-4): MoveIt refused to execute. This is '
                'almost always a trajectory still in flight from an earlier run '
                '("Cannot push a new trajectory while another is being executed"). '
                'Cancelling all MoveIt goals and retrying once.')
            self.cancel_stale_execution()
            return self.move_to_posture(positions, label, allow_recovery=False)
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
        gh = self._track(self._wait(self.exec_client.send_goal_async(goal), 10.0))
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] execute_trajectory rejected the goal.')
            return False
        result = self._wait(gh.get_result_async(), 60.0)
        self._untrack(gh)
        if result is None:
            self.get_logger().error(f'[{label}] execute_trajectory returned no result.')
            return False
        code = result.result.error_code.val
        if code != 1:
            self.get_logger().error(f'[{label}] execution failed, error_code={code}.')
            return False
        return True

    def command_bucket_async(self, q: float, label: str) -> bool:
        """Start closing and return immediately, so the arm keeps descending.

        Closing only after the bucket has stopped scrapes rather than scoops:
        the shells need to be shutting while they are still moving through the
        material. Fire-and-forget is deliberate -- the descent leg that follows
        is what gives the close time to happen.
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
            f'[{label}] bucket -> q={q:+.3f} (gap {fourbar.gap_from_q(q)*1000:.0f}mm), '
            'closing while the bucket keeps moving.')
        self._track(self._wait(self.grip_client.send_goal_async(goal), 10.0))
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
        gh = self._track(self._wait(self.grip_client.send_goal_async(goal), 10.0))
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] gripper controller rejected the goal.')
            return False
        self._wait(gh.get_result_async(), self.grip_duration + 15.0)
        self._untrack(gh)
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

    def deposit_sample(self) -> Tuple[bool, str]:
        """Tip the sample into the rover-mounted box.

        Measured over the base-box column (contact at the box centre, bucket
        mouth pointing straight DOWN, 4 wrist rolls):

            contact z   vs rim   link z   reach   collision-free
              0.380     +18mm    0.594     4/4        3/4
              0.400     +38mm    0.614     4/4        4/4
              0.420     +58mm    0.634     4/4        2/4
              0.450     +88mm    0.664     1/4        0/4

        So 0.400 is the sweet spot and the default. Two things this proves:

        * The bucket CAN use the rover box even though the probe could not. The
          probe needed the gripper link at z=0.787 (200 mm of probe hanging below
          a 270 mm contact); the bucket needs 0.614 -- 173 mm less -- because
          nothing protrudes past the jaws.
        * The dump must be VERTICAL. At a 30 deg tilt reach is still fine but
          collisions reject it (0/4 at every height): leaning over the box puts
          the arm back into the fold that blocked the probe. This is the opposite
          of the scoop, which cannot be vertical.

        Dumping only needs to be ABOVE the rim, not inside the box, so the bucket
        never enters it.
        """
        if not self.deposit_enabled:
            return False, 'deposit disabled'
        self._publish_box_marker()

        # 1) The configured posture first: it is a pose someone has actually seen
        #    the bucket hold over the box, so it cannot be argued out of reach.
        if self.deposit_use_joint_posture and self.deposit_joint_positions:
            if self._posture_is_over_box() and self.move_to_posture(
                    self.deposit_joint_positions, 'deposit-posture'):
                self._sleep(self.deposit_settle_sec)
                if not self.command_bucket(self.deposit_dump_open_q, 'dump'):
                    return False, 'reached the box but the bucket would not open'
                self._sleep(self.deposit_settle_sec)
                self.get_logger().info('[deposit] sample tipped in from the configured '
                                       'dump posture.')
                return True, 'deposited in the box (configured posture)'
            self.get_logger().warning(
                '[deposit] configured dump posture unusable; falling back to a pose '
                'derived from the box geometry.')

        # 2) Otherwise derive it, searching UPWARD: a higher tip still lands in
        #    the box, and height is what the envelope cares about.
        clearances = [self.deposit_rim_clearance_m]
        while clearances[-1] < self.deposit_max_rim_clearance_m - 1e-6:
            clearances.append(min(self.deposit_max_rim_clearance_m,
                                  clearances[-1] + 0.025))
        for clearance in clearances:
            if self._try_dump_at(clearance):
                return True, (f'deposited in the box '
                              f'({clearance*1000:.0f}mm above the rim)')
        self.get_logger().error(
            f'[deposit] no wrist roll reaches a dump pose from '
            f'{self.deposit_rim_clearance_m*1000:.0f}mm to '
            f'{self.deposit_max_rim_clearance_m*1000:.0f}mm above the rim. The bucket '
            'is still closed and still holding the sample; nothing was released.')
        return False, 'dump pose unreachable at any clearance'

    def _posture_is_over_box(self) -> bool:
        """Does the configured dump posture actually put the bucket over the box?"""
        pose = self.link_pose_for_joints(self.deposit_joint_positions)
        if pose is None:
            self.get_logger().warning('[deposit] could not FK the dump posture.')
            return False
        R = quat_to_matrix(pose.orientation)
        link = np.array([pose.position.x, pose.position.y, pose.position.z])
        contact = link + R @ fourbar.contact_offset(
            self.close_q, fourbar.CONTACT_Y_OFFSET_M)
        over = self.deposit_box.over_opening(contact, self.deposit_edge_margin_m)
        down_deg = math.degrees(math.acos(max(-1.0, min(1.0, -float(R[2, 2])))))
        self.get_logger().info(
            f'[deposit] dump posture puts the bucket at '
            f'({contact[0]:.3f},{contact[1]:.3f},{contact[2]:.3f}), '
            f'{(contact[2]-self.deposit_box.rim_z)*1000:+.0f}mm vs the rim, mouth '
            f'{down_deg:.0f} deg off straight down, over the opening={over}.')
        if not over:
            self.get_logger().warning(
                '[deposit] that posture is NOT over the box opening -- the sample '
                'would miss. Check deposit_joint_positions against the box.')
        if down_deg > 35.0:
            self.get_logger().warning(
                '[deposit] the bucket mouth is far from vertical in that posture; '
                'soil may not tip out cleanly.')
        return over

    def _try_dump_at(self, clearance: float) -> bool:
        """Move to a derived dump pose at this clearance and open the jaws."""
        contact = self.deposit_box.dump_contact(clearance, self.deposit_offset_xy)
        axis = np.array([0.0, 0.0, -1.0])          # mouth down; must not tilt
        offset = fourbar.contact_offset(self.close_q, fourbar.CONTACT_Y_OFFSET_M)
        R_now = self._current_tool_rotation()
        prefer = R_now[:, 0] if R_now is not None else None

        self.get_logger().info(
            f'[deposit] trying a dump pose at ({contact[0]:.3f},{contact[1]:.3f},'
            f'{contact[2]:.3f}) = {clearance*1000:.0f}mm above the rim, mouth down.')
        for i, R in enumerate(scoop_lib.roll_frames(axis, prefer, self.roll_candidates)):
            link_pos = scoop_lib.link_position_for_contact(contact, R, offset)
            pose = Pose()
            pose.position.x = float(link_pos[0])
            pose.position.y = float(link_pos[1])
            pose.position.z = float(link_pos[2])
            pose.orientation = matrix_to_quat(R)
            if not self._ik_exists(pose):
                continue
            if not self.move_to_posture_pose(pose, 'deposit'):
                continue
            self._sleep(self.deposit_settle_sec)
            # Opening the jaws over the box IS the dump; there is nothing to
            # release-and-regrasp, so a single open is the whole operation.
            if not self.command_bucket(self.deposit_dump_open_q, 'dump'):
                self.get_logger().error('[deposit] reached the box but the bucket '
                                        'would not open.')
                return False
            self._sleep(self.deposit_settle_sec)
            self.get_logger().info(
                f'[deposit] sample tipped into the box (wrist roll '
                f'{i + 1}/{self.roll_candidates}).')
            return True

        return False

    # ------------------------------------------------------------- sequence

    def _aim_below_home(self) -> bool:
        """Point the survey and the scoop straight down from pick_home.

        Deployment reality: there is no tray, no prepared bed and no map. The
        only place the robot can be sure about is the ground directly beneath the
        posture it is already holding. So the sampling column is pick_home's own
        XY, taken from forward kinematics rather than copied into a config file
        where it would silently rot the moment the posture changed.
        """
        pose = self.link_pose_for_joints(self.home_q)
        if pose is None:
            self.get_logger().error(
                '[sample] could not FK pick_home; cannot aim below it.')
            return False
        x, y = float(pose.position.x), float(pose.position.y)
        half = self.sample_below_home_halfwidth_m
        self.sample_point = np.array([x, y], dtype=np.float64)
        self.use_fixed_sample_point = True
        self.prefer_xy = self.sample_point.copy()
        self.region = terrain.WorkRegion(
            x - half, x + half, y - half, y + half,
            self.region.z_min, self.region.z_max)
        self.get_logger().info(
            f'[sample] aiming straight down from pick_home: column '
            f'({x:.3f}, {y:.3f}), survey window +/-{half*1000:.0f}mm. Ground height '
            'comes from the depth camera.')
        return True

    def wait_for_stack(self) -> Tuple[bool, str]:
        """Block until MoveIt, the gripper controller and the camera are live.

        Discovery is not instant after a sim launch, and a cycle that starts
        before the stack is up fails on the first motion for a reason that looks
        like a real fault (`MoveGroup action unavailable`) but is only impatience.
        Waits for what the FIRST stage actually needs, and names whatever is
        missing rather than timing out silently.
        """
        deadline = time.monotonic() + self.startup_wait_sec
        checks = (
            ('MoveGroup action', lambda: self.move_client.server_is_ready()),
            ('execute_trajectory action', lambda: self.exec_client.server_is_ready()),
            ('gripper action', lambda: self.grip_client.server_is_ready()),
            ('compute_ik service', lambda: self.ik_client.service_is_ready()),
            ('cartesian path service', lambda: self.cart_client.service_is_ready()),
            ('depth frame', lambda: self.latest_depth is not None),
            ('camera_info', lambda: self.camera_info is not None),
        )
        announced = False
        while time.monotonic() < deadline:
            missing = [name for name, ready in checks if not ready()]
            if not missing:
                return True, 'stack ready'
            if not announced:
                self.get_logger().info(
                    f'waiting up to {self.startup_wait_sec:.0f}s for: {", ".join(missing)}')
                announced = True
            time.sleep(0.5)
        missing = [name for name, ready in checks if not ready()]
        return False, f'still missing after {self.startup_wait_sec:.0f}s: {", ".join(missing)}'

    def run_scoop_cycle(self) -> Tuple[bool, str]:
        if not self._busy.acquire(blocking=False):
            return False, 'a cycle is already running'
        self._abort.clear()
        try:
            ready, why = self.wait_for_stack()
            if not ready:
                self.get_logger().error(f'not starting a cycle: {why}')
                return False, why
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
        # Start from pick_home, as the sequence specifies.
        if not self.move_to_posture(self.home_q, 'pick_home-start'):
            return False, 'could not reach the pick_home start posture'
        if self.sample_below_home and not self._aim_below_home():
            return False, 'could not work out where pick_home is pointing'
        # Open the bucket BEFORE surveying. The camera looks straight down its
        # own column, and a closed bucket sits in that view: with the jaws shut
        # the survey collapsed to 22% coverage, against 87% with them open.
        # Opening first is also free -- the descent needs them open anyway.
        if not self.command_bucket(self.entry_q, 'open-bucket-for-survey'):
            return False, 'bucket would not open before the survey'
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
                # Already open from before the survey; re-assert only if a
                # previous attempt closed it.
                if abs(self.joint_positions.get(self.gripper_joint, self.entry_q)
                       - self.entry_q) > 0.15:
                    if not self.command_bucket(self.entry_q, 'open-bucket'):
                        return False, 'bucket would not open'
                if self._aborted('approach'):
                    return False, 'aborted'

                # Where, along the descent, the jaws start shutting.
                trigger = None
                if self.close_during_descent:
                    axis = np.asarray(_axis, dtype=np.float64)
                    trig_contact = (np.asarray(site.centre, dtype=np.float64)
                                    - axis * float(self.close_start_above_ground_m))
                    trigger = self._pose_like(by_label['entry'], trig_contact,
                                              by_label['approach'])

                if self.scoop_all_cartesian:
                    # One Cartesian stroke: approach -> entry -> penetrate. The
                    # arm follows a straight line the whole way in, so the tool
                    # never takes a joint-space detour through the soil, and the
                    # descend no longer has to start from wherever a joint move
                    # happened to leave it.
                    if trigger is not None:
                        # Descend to the trigger height, START closing, then keep
                        # going: the shells sweep material as they shut.
                        if not self.move_cartesian(
                                [by_label['approach'], trigger], 'descend-to-close'):
                            continue
                        self.command_bucket_async(self.close_q, 'close-while-descending')
                        if not self.move_cartesian([by_label['penetrate']],
                                                   'descend-through-soil'):
                            continue
                    elif not self.move_cartesian(
                            [by_label['approach'], by_label['entry'],
                             by_label['penetrate']], 'approach+descend'):
                        # Nothing is buried yet: the stroke never started.
                        continue
                else:
                    if not self.move_to_posture_pose(by_label['approach'], 'approach'):
                        continue
                    if not self.move_cartesian(
                            [by_label['entry'], by_label['penetrate']], 'descend'):
                        continue

                # Past this point the bucket is IN the soil. Extraction must run
                # even if the close fails, or the gripper is left buried.
                if trigger is not None:
                    # Already commanded on the way down; just let it finish
                    # loading against the soil.
                    closed = True
                    self._sleep(self.grip_settle)
                else:
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
                self.get_logger().info(f'SOIL SAMPLE COLLECTED ({reason}).')
                # Back to pick_home with the sample aboard, bucket still closed,
                # BEFORE going anywhere near the box.
                if not self._aborted('return-home'):
                    self.move_home_cartesian('pick_home-return')
                if self.deposit_enabled and not self._aborted('deposit'):
                    ok, dmsg = self.deposit_sample()
                    if self.return_transport and not self._aborted('transport'):
                        self.move_home_cartesian('pick_home-final')
                    if ok:
                        return True, f'captured and {dmsg}'
                    # The sample is still in the bucket, so this is a partial
                    # success, not a failed scoop: say which half worked.
                    return False, f'captured ({reason}) but NOT deposited: {dmsg}'
                self.get_logger().info('Bucket stays closed (deposit disabled).')
                return True, f'captured: {reason}'
            if verdict == scoop_lib.EMPTY:
                self.get_logger().warning(
                    f'bucket appears empty ({reason}); trying a different site.')
            else:
                self.get_logger().warning(
                    f'capture unconfirmed ({reason}); treating as a miss and retrying.')

        return False, f'no sample after {self.max_attempts} attempts'

    def link_pose_for_joints(self, positions: Sequence[float]) -> Optional[Pose]:
        """Forward kinematics for a joint posture, so it can be a Cartesian goal."""
        if not self.fk_client.wait_for_service(timeout_sec=3.0):
            return None
        req = GetPositionFK.Request()
        req.header.frame_id = self.planning_frame
        req.fk_link_names = [self.planning_link]
        req.robot_state.joint_state.name = list(self.arm_joints)
        req.robot_state.joint_state.position = [float(v) for v in positions]
        res = self._wait(self.fk_client.call_async(req), 8.0)
        if res is None or res.error_code.val != 1 or not res.pose_stamped:
            return None
        return res.pose_stamped[0].pose

    def move_home_cartesian(self, label: str) -> bool:
        """Return to pick_home along a straight line, gripper untouched.

        The sample rides in a closed bucket, so the return is a transport move:
        keep it straight and predictable rather than letting the planner pick an
        arbitrary joint-space arc through the scene. If the straight line cannot
        be solved the joint move still runs -- a curved return beats leaving the
        bucket in the hole.
        """
        if self.cartesian_return_home:
            pose = self.link_pose_for_joints(self.home_q)
            if pose is not None and self.move_cartesian([pose], f'{label}-cartesian'):
                return True
            self.get_logger().warning(
                f'[{label}] straight-line return to pick_home not solvable; '
                'falling back to a joint-space move.')
        return self.move_to_posture(self.home_q, label)

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
            if best:
                # THE POINT WHERE THE JAWS START CLOSING. This is the number
                # close_start_above_ground_m sets, drawn where it actually lands
                # so it can be checked by eye instead of trusted.
                n_hat = np.asarray(s.normal, dtype=np.float64)
                closing = s.centre + n_hat * self.close_start_above_ground_m
                start = s.centre + n_hat * self.scoop_params.standoff_m
                bottom = s.centre - n_hat * self.scoop_params.depth_m

                dot = Marker()
                dot.header.frame_id = self.planning_frame
                dot.header.stamp = m.header.stamp
                dot.ns = 'scoop_sites'
                dot.id = 1001
                dot.type = Marker.SPHERE
                dot.action = Marker.ADD
                dot.pose.position = Point(x=float(closing[0]), y=float(closing[1]),
                                          z=float(closing[2]))
                dot.pose.orientation.w = 1.0
                dot.scale.x = dot.scale.y = dot.scale.z = 0.022
                dot.color.r, dot.color.g, dot.color.b, dot.color.a = (1.0, 0.2, 0.1, 0.95)
                arr.markers.append(dot)

                label = Marker()
                label.header.frame_id = self.planning_frame
                label.header.stamp = m.header.stamp
                label.ns = 'scoop_sites'
                label.id = 1002
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position = Point(x=float(closing[0]), y=float(closing[1]),
                                            z=float(closing[2]) + 0.045)
                label.pose.orientation.w = 1.0
                label.scale.z = 0.022
                label.color.r, label.color.g, label.color.b, label.color.a = (
                    1.0, 0.35, 0.1, 0.95)
                label.text = (f'close +{self.close_start_above_ground_m*1000:.0f}mm\n'
                              f'ground {s.centre[2]:.3f}')
                arr.markers.append(label)

                # The whole stroke: start height -> closing point -> scoop depth.
                stroke = Marker()
                stroke.header.frame_id = self.planning_frame
                stroke.header.stamp = m.header.stamp
                stroke.ns = 'scoop_sites'
                stroke.id = 1003
                stroke.type = Marker.LINE_STRIP
                stroke.action = Marker.ADD
                stroke.pose.orientation.w = 1.0
                stroke.scale.x = 0.005
                stroke.color.r, stroke.color.g, stroke.color.b, stroke.color.a = (
                    1.0, 0.6, 0.0, 0.85)
                for pt in (start, closing, bottom):
                    stroke.points.append(Point(x=float(pt[0]), y=float(pt[1]),
                                               z=float(pt[2])))
                arr.markers.append(stroke)

                # The surface normal at the chosen point, so the entry direction
                # is visible rather than inferred from the disc alone.
                arrow = Marker()
                arrow.header.frame_id = self.planning_frame
                arrow.header.stamp = m.header.stamp
                arrow.ns = 'scoop_sites'
                arrow.id = 1000
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.006, 0.012, 0.0
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (
                    0.0, 0.9, 0.2, 0.9)
                tip = s.centre + np.asarray(s.normal, dtype=np.float64) * 0.08
                arrow.points.append(Point(x=float(s.centre[0]), y=float(s.centre[1]),
                                          z=float(s.centre[2])))
                arrow.points.append(Point(x=float(tip[0]), y=float(tip[1]),
                                          z=float(tip[2])))
                arr.markers.append(arrow)
        self.marker_pub.publish(arr)
        self._publish_box_marker()

    def _publish_box_marker(self) -> None:
        """Draw the configured box and its dump point, for tuning it by eye.

        The box outline is a wireframe rather than a solid cube so the opening
        stays visible, and the dump point is drawn separately: if the sphere is
        not floating just above the opening, the configuration is wrong and this
        shows it before a scoop puts soil on the deck.
        """
        if not (self.publish_box_marker and self.publish_markers):
            return
        b = self.deposit_box
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        outline = Marker()
        outline.header.frame_id = self.planning_frame
        outline.header.stamp = stamp
        outline.ns = 'deposit_box'
        outline.id = 0
        outline.type = Marker.LINE_LIST
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.004
        outline.color.r, outline.color.g, outline.color.b, outline.color.a = (
            0.2, 0.6, 1.0, 0.9)
        corners = b.corners
        # Pairs of corner indices that share exactly two sign components.
        for i in range(8):
            for j in range(i + 1, 8):
                same = sum(1 for k in range(3)
                           if abs(corners[i][k] - corners[j][k]) < 1e-9)
                if same >= 1 and np.count_nonzero(
                        np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                    for c in (corners[i], corners[j]):
                        outline.points.append(
                            Point(x=float(c[0]), y=float(c[1]), z=float(c[2])))
        arr.markers.append(outline)

        contact = b.dump_contact(self.deposit_rim_clearance_m, self.deposit_offset_xy)
        dot = Marker()
        dot.header.frame_id = self.planning_frame
        dot.header.stamp = stamp
        dot.ns = 'deposit_box'
        dot.id = 1
        dot.type = Marker.SPHERE
        dot.action = Marker.ADD
        dot.pose.position = Point(x=float(contact[0]), y=float(contact[1]),
                                  z=float(contact[2]))
        dot.pose.orientation.w = 1.0
        dot.scale.x = dot.scale.y = dot.scale.z = 0.018
        dot.color.r, dot.color.g, dot.color.b, dot.color.a = (1.0, 0.7, 0.1, 0.95)
        arr.markers.append(dot)
        self.marker_pub.publish(arr)

    # ------------------------------------------------------------- services

    def _srv_scoop(self, _req, res):
        ok, msg = self.run_scoop_cycle()
        res.success = bool(ok)
        res.message = msg
        return res

    def _srv_survey(self, _req, res):
        self._publish_box_marker()
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
        self._publish_box_marker()
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
                'scoop; see the log')
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
        node.cancel_active_goals_on_shutdown()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
