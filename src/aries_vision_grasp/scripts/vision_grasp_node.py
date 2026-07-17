#!/usr/bin/env python3
"""
Vision grasp node with RViz markers and adaptive gripper sizing.

New gripper: gripper_gear_left_joint (revolute), open=-1.57 rad, closed=0.07 rad.
The jaw gap varies with joint angle. This node estimates the 3D object width
from the YOLO segmentation mask + depth image and computes the optimal close
angle per object using the calibrated four-bar gap table in
``aries_vision_grasp.fourbar`` (measured from gripper_new.xacro +
gripper_bucket.stl; e.g. a 45 mm probe needs q ≈ -0.20 rad).

YOLO inference runs in a background thread (``aries_vision_grasp.inference``)
so the rclpy executor — gripper ticks, action results, TF — is never blocked
by the model. Each detection is processed against the exact color/depth frame
pair that inference saw, with TF looked up at the depth frame's stamp.
"""

import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import ColorRGBA, Float64
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from aries_vision_grasp import fourbar, stages
from aries_vision_grasp.geometry import (
    CameraOffsetEstimate,
    estimate_stationary_target_camera_offset,
    matrix_to_quat,
    normalize,
    quat_to_matrix,
    quaternion_distance_rad,
    quaternion_rotation_vector_error,
    rpy_to_quat,
    wrap_to_pi,
)
from aries_vision_grasp.inference import YoloWorker, load_yolo_model

try:
    import ultralytics  # noqa: F401
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class FrameSnapshot:
    """A color/depth frame pair captured together for one inference pass.

    Detection results are always processed against the frames inference
    actually saw (not whatever arrived later), and TF is looked up at the
    depth stamp, so a moving wrist camera cannot skew the 3D target.
    """
    color: np.ndarray
    color_stamp_sec: float
    depth: np.ndarray
    depth_stamp_sec: float
    depth_frame: str
    stamp: rclpy.time.Time = field(default=None)


class VisionGraspNode(Node):
    def __init__(self) -> None:
        super().__init__('vision_grasp_node')
        self.bridge = CvBridge()

        # Vision/model. The default weights are installed with this package.
        _default_model = os.path.join(
            get_package_share_directory('aries_vision_grasp'), 'models', 'grasp.pt'
        )
        self.declare_parameter('model_path', _default_model)
        self.declare_parameter('target_class', 'probe')
        self.declare_parameter('confidence_threshold', 0.55)
        self.declare_parameter('detect_period_sec', 0.25)
        self.declare_parameter('roi_half_size_px', 4)
        self.declare_parameter('max_depth_m', 1.5)
        self.declare_parameter('min_depth_m', 0.08)
        # When True (real hardware), subscribe to the D435i hardware-aligned depth topic
        # (depth reprojected into the color camera frame) so YOLO detections in the color
        # image map 1-to-1 onto depth pixels.  Set False in simulation where the Gazebo
        # sensors already share the same optical frame.
        self.declare_parameter('use_aligned_depth', True)
        # A detection is processed only when its color and depth frames were
        # captured within this window of each other. On a moving wrist camera
        # a mismatched pair projects the mask onto the wrong depth pixels.
        self.declare_parameter('max_color_depth_stamp_gap_sec', 0.15)
        self.declare_parameter('sensor_sync_queue_size', 20)

        # Close-range tracking:
        # YOLO often fails when the probe is very close, partially cropped,
        # or hidden by the gripper. Use lower refine confidence and a depth
        # tracker around the projected locked target.
        self.declare_parameter('refine_confidence_threshold', 0.20)
        self.declare_parameter('refine_use_projection_fallback', True)
        self.declare_parameter('refine_projection_roi_half_size_px', 45)
        self.declare_parameter('refine_min_depth_m', 0.02)
        self.declare_parameter('refine_depth_band_m', 0.12)

        # YOLO segmentation support.
        # Your yolo26-seg model provides masks. Do not use only bbox center.
        self.declare_parameter('use_segmentation_mask', True)
        self.declare_parameter('mask_score_threshold', 0.50)
        self.declare_parameter('mask_min_pixels', 80)
        self.declare_parameter('mask_erode_px', 2)
        self.declare_parameter('mask_depth_percentile', 35.0)

        # Planning / frames
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('planning_group', 'igus_rebel_arm')
        self.declare_parameter('planning_link', 'arm_gripper_base_link')
        self.declare_parameter('move_action_name', '/move_action')
        self.declare_parameter('keep_current_orientation', False)
        self.declare_parameter('fixed_roll', math.pi)
        self.declare_parameter('fixed_pitch', 0.0)
        self.declare_parameter('fixed_yaw', 0.0)
        self.declare_parameter('approach_axis_in_tool', [0.0, 0.0, 1.0])

        # Grasp distances
        self.declare_parameter('pre_grasp_distance', 0.15)
        # Positive value means insert downward along the approach direction
        # below the detected probe surface.
        self.declare_parameter('grasp_depth_below_surface_m', 0.018)
        self.declare_parameter('retreat_distance', 0.15)
        # Physical pinch/contact point in arm_gripper_base_link.
        # 0.065 m is near the finger joint, not the fingertip/pinch point.
        # The gripper finger origin is around z=0.0645 and the finger mesh extends
        # another ~0.095 m, so the usable contact point is around 0.14-0.15 m.
        # Stable pinch center inside the fingers, not the fingertip end.
        # Finger z range is roughly 0.064–0.159 m from arm_gripper_base_link.
        # 0.145 puts the object near the tip and gives poor holding.
        # 0.115 places the probe deeper between the fingers.
        self.declare_parameter('target_point_offset_in_link', [0.0, 0.0, 0.235])
        self.declare_parameter('use_orientation_constraint', True)
        self.declare_parameter('min_pose_z', 0.05)
        # Pre-grasp uses a large position sphere so IK can satisfy position + orientation together.
        # This prevents joint 6 from arriving in a random orientation before the Cartesian stroke.
        self.declare_parameter('pre_grasp_position_tol', 0.05)
        # Raw detection calibration in depth-camera axes. Applying this before
        # TF keeps the correction camera-relative as the wrist camera moves.
        self.declare_parameter('grasp_target_offset_camera_xyz_m', [0.0, 0.0, 0.0])
        self.declare_parameter('auto_calibrate_camera_offset_enabled', False)
        self.declare_parameter('auto_calibrate_camera_offset_min_samples', 10)
        self.declare_parameter('auto_calibrate_camera_offset_min_rotation_deg', 12.0)
        self.declare_parameter('auto_calibrate_camera_offset_max_condition', 250.0)
        self.declare_parameter('auto_calibrate_camera_offset_max_m', 0.060)
        self.declare_parameter('auto_calibrate_camera_offset_max_rms_m', 0.012)
        self.declare_parameter('auto_calibrate_camera_offset_min_improvement_m', 0.003)
        self.declare_parameter('auto_calibrate_camera_offset_max_step_m', 0.015)
        self.declare_parameter('auto_calibrate_camera_offset_max_samples', 80)
        # Small calibrated target bias in planning-frame axes. Use this for
        # repeatable rover-front/left/up camera calibration errors.
        self.declare_parameter('grasp_target_bias_base_x_m', 0.0)
        self.declare_parameter('grasp_target_bias_base_y_m', 0.0)
        self.declare_parameter('grasp_target_bias_base_z_m', 0.0)
        # Small calibrated target bias in arm_gripper_base_link axes. Keep at
        # zero until a repeated same-direction miss is measured on hardware.
        self.declare_parameter('grasp_target_bias_tool_x_m', 0.0)
        self.declare_parameter('grasp_target_bias_tool_y_m', 0.0)
        self.declare_parameter('grasp_target_bias_tool_z_m', 0.0)

        # Joint 4 wrist lock (prevents spinning during free-space pre-grasp move)
        self.declare_parameter('lock_wrist_joint', False)
        self.declare_parameter('lock_wrist_joint_name', 'joint4')
        self.declare_parameter('lock_wrist_joint_tolerance', 0.30)  # ~17 deg

        # Cartesian motion (straight-line approach / retreat)
        self.declare_parameter('cartesian_service_name', '/compute_cartesian_path')
        self.declare_parameter('execute_action_name', '/execute_trajectory')
        self.declare_parameter('cartesian_lock_orientation', False)  # lock orientation during Cartesian stroke
        self.declare_parameter('diagnose_final_cartesian_failure', True)
        self.declare_parameter('cartesian_max_step', 0.005)
        self.declare_parameter('cartesian_jump_threshold', 0.0)
        self.declare_parameter('cartesian_fraction_min', 0.70)

        # Joint retreat/home
        self.declare_parameter('use_joint_retreat_home', True)
        self.declare_parameter('retreat_home_joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('retreat_home_joint_positions', [-0.0523599, 0.244346, 1.41372, 0.0, 1.53589, 1.01229])
        self.declare_parameter('joint_goal_tolerance', 0.03)

        # Gripper
        # Nominal commanded endpoints. Gazebo gets an extra 0.05 rad of model
        # travel outside these values so the four-bar does not wedge at a hard
        # physics stop; the commanded open/close positions remain unchanged.
        self.declare_parameter('gripper_open_width', -1.57)
        # Probe mesh is about 45 mm wide.
        # Fully closing to 0.085 can push the probe out.
        # Finger gap approx = 0.1786 - 2*q.
        # q=0.070 gives about 38.6 mm gap, enough to grip a 45 mm probe.
        self.declare_parameter('gripper_close_width', 0.07)
        self.declare_parameter('gripper_preclose_width', -0.75)
        self.declare_parameter('gripper_joint_lower_limit', -1.62)
        self.declare_parameter('gripper_joint_upper_limit', 0.12)
        self.declare_parameter('gripper_joint_limit_margin', 0.05)
        self.declare_parameter('final_grasp_arm_settle_sec', 0.0)
        self.declare_parameter('final_grasp_pose_check_enabled', True)
        self.declare_parameter('final_grasp_pose_position_tolerance_m', 0.025)
        self.declare_parameter('final_grasp_pose_orientation_tolerance_rad', 0.35)
        self.declare_parameter('final_grasp_pose_check_timeout_sec', 4.0)
        self.declare_parameter('final_grasp_pose_check_period_sec', 0.10)
        self.declare_parameter('gripper_topic', '/aries/gripper_gear_left_joint/cmd_pos')
        self.declare_parameter('gripper_settle_sec', 1.5)
        self.declare_parameter('gripper_command_duration_sec', 0.0)
        self.declare_parameter('gripper_no_feedback_close_complete_sec', 0.0)
        self.declare_parameter('gripper_command_mode', 'auto')  # auto|trajectory_action|topic
        self.declare_parameter('gripper_action_name', '/rebel_gripper_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_joint_name', 'gripper_gear_left_joint')
        self.declare_parameter('gripper_action_timeout_sec', 5.0)
        self.declare_parameter('gripper_require_action_success_for_completion', True)

        # Adaptive gripper sizing: estimate object 3D width from detection mask
        # and compute the optimal close angle for each detected object using
        # the calibrated four-bar gap table (aries_vision_grasp.fourbar).
        self.declare_parameter('adaptive_gripper_enabled', True)
        self.declare_parameter('object_width_safety_margin_m', 0.015)
        self.declare_parameter('adaptive_gripper_min_width_m', 0.008)
        self.declare_parameter('adaptive_gripper_max_width_m', 0.15)
        self.declare_parameter('adaptive_gripper_width_percentile', 30.0)
        # Final gap is intentionally only slightly larger than the object.
        # Pre-close gap is also close to final gap so the last ground-level
        # four-bar sweep is very small.
        self.declare_parameter('minimum_probe_width_m', 0.045)
        # Width estimator guard for a known probe. Segmentation sometimes returns
        # the probe length/visible mask diagonal as the width (e.g. 100+ mm),
        # which leaves the gripper too open. Clamp to a physical probe range.
        self.declare_parameter('nominal_probe_width_m', 0.045)
        self.declare_parameter('maximum_probe_width_m', 0.060)
        self.declare_parameter('clamp_probe_width_for_grasp', True)
        self.declare_parameter('object_width_final_clearance_m', -0.004)
        self.declare_parameter('object_width_preclose_clearance_m', 0.012)
        self.declare_parameter('preclose_min_q_margin_rad', 0.004)

        # Four-bar / ground-safety supervisor.
        self.declare_parameter('fourbar_preclose_before_grasp', True)
        self.declare_parameter('fourbar_final_close_steps', 6)
        self.declare_parameter('fourbar_final_close_step_wait_sec', 0.08)
        self.declare_parameter('freeze_arm_during_gripper_enabled', True)
        self.declare_parameter('hold_after_close_no_motion', True)

        # Post-grasp transport supervisor.
        # After the gripper has closed, do NOT open it: attach the probe mesh
        # to the planning scene and send only arm joints to the pick_home
        # posture through MoveGroup collision checking.  The gripper joint is
        # intentionally excluded from pick_home so the object stays grasped.
        # (The old segmented Cartesian vertical-lift subsystem and its
        # post_grasp_lift_* tuning parameters were removed as dead code; see
        # git history if it ever needs to be resurrected.)
        self.declare_parameter('post_grasp_lift_then_pick_home', True)
        self.declare_parameter('post_grasp_planning_time_sec', 10.0)
        self.declare_parameter('pick_home_joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('pick_home_joint_positions', [0.0, 0.366519, 1.18682, 0.0349066, 1.55334, 1.50098])

        # Base-box placement.  The arm first reaches pick_home with the probe
        # attached, then moves to the calibrated SRDF ``pick_drop`` posture,
        # opens the gripper, detaches the probe from MoveIt's planning scene,
        # and returns to pick_home.  Only arm joints are commanded for the two
        # transport poses so the gripper cannot open before the release stage.
        self.declare_parameter('place_in_base_box_after_grasp', False)
        self.declare_parameter('base_box_drop_use_pose', False)
        self.declare_parameter('base_box_drop_frame', 'base_link')
        self.declare_parameter('base_box_drop_xyz', [0.45078823, 0.07073892, 0.64813140])
        self.declare_parameter('base_box_drop_rpy', [2.05331746, 0.12332939, 1.83238021])
        self.declare_parameter('base_box_drop_target_point_offset_in_link', [0.0, 0.0259, 0.2180])
        self.declare_parameter('base_box_drop_position_tolerance_m', 0.015)
        self.declare_parameter('base_box_drop_orientation_tolerance_rad', 0.10)
        self.declare_parameter('base_box_drop_joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('base_box_drop_joint_positions', [0.15708, -0.837758, 1.93732, 0.15708, 0.959931, 1.27409])
        self.declare_parameter('base_box_planning_time_sec', 10.0)
        self.declare_parameter('base_box_release_wait_sec', 0.0)
        self.declare_parameter('return_pick_home_after_base_box_place', True)
        self.declare_parameter('base_box_drop_marker_enabled', True)
        self.declare_parameter('base_box_drop_marker_scale_m', 0.060)
        self.declare_parameter('base_box_drop_marker_axes_length_m', 0.120)

        # Floor-safe grasping: never insert the gripper contact point deeply
        # below the detected object surface when the object lies on the floor.
        self.declare_parameter('floor_safe_grasp_enabled', True)
        self.declare_parameter('max_grasp_descent_below_target_m', 0.006)
        self.declare_parameter('min_grasp_height_above_floor_m', 0.035)

        # Exact four-bar geometry model from gripper_new.xacro + gripper_bucket.stl
        # (tables live in aries_vision_grasp.fourbar).  This replaced the old
        # linear gap model, which incorrectly made q≈+0.068 rad for a 45 mm
        # probe; the real STL geometry gives almost zero jaw gap at that angle.
        # The correct q for a ~45 mm gap is ≈ -0.20 rad.
        self.declare_parameter('fourbar_contact_y_offset_m', 0.0259)
        self.declare_parameter('fourbar_contact_z_open_m', 0.1342)
        self.declare_parameter('fourbar_contact_z_closed_m', 0.2180)
        self.declare_parameter('fourbar_q_min_for_floor_grasp', -0.42)
        self.declare_parameter('fourbar_q_max_for_floor_grasp', -0.08)
        self.declare_parameter('fourbar_max_contact_lift_m', 0.014)
        self.declare_parameter('fourbar_min_arc_clearance_m', 0.006)

        # Conservative bucket/floor safety for the true four-bar gripper.
        # The bucket mesh extends much farther than the old 0.115 m contact offset.
        # For a top-down floor grasp, the lowest bucket point is roughly:
        #     world_z = contact_z + R[:,2].z * (bucket_tip_z - offset_z)
        # If offset_z is too small, the bucket tip is below the floor even when the
        # planned contact point looks safe.
        self.declare_parameter('fourbar_ground_guard_enabled', True)
        self.declare_parameter('fourbar_bucket_tip_z_max_m', 0.275)
        self.declare_parameter('fourbar_ground_clearance_m', 0.0)
        self.declare_parameter('floor_safe_contact_height_m', 0.060)

        # 6D object pose tracking.
        # PCA on the masked point cloud yields a stable centroid + orientation.
        # The full pose is published for downstream tracking/visualization, while
        # the grasp logic can still choose to use only yaw/top-down orientation.
        self.declare_parameter('publish_object_pose', True)
        self.declare_parameter('object_pose_topic', '/vision_grasp/object_pose')
        self.declare_parameter('object_pose_axis_length_m', 0.080)
        self.declare_parameter('object_yaw_align_enabled', True)
        self.declare_parameter('object_yaw_rotation_offset_deg', 90.0)
        self.declare_parameter('object_orientation_min_eigenratio', 3.0)
        self.declare_parameter('stl_yaw_correction_deg', 0.0)   # trim if STL/mask mismatch

        # Safety / filtering
        self.declare_parameter('position_tolerance_xyz', 0.015)
        self.declare_parameter('orientation_tolerance_rad', 0.20)
        self.declare_parameter('allowed_planning_time', 8.0)
        self.declare_parameter('num_planning_attempts', 20)
        self.declare_parameter('velocity_scale', 0.25)
        self.declare_parameter('acceleration_scale', 0.25)
        # Whole-process motion completion supervisor. MoveIt/ExecuteTrajectory
        # success is necessary but not sufficient: fresh measured robot state
        # must also reach and remain at every commanded target before the next
        # stage can begin.
        self.declare_parameter('arm_require_feedback_for_completion', True)
        self.declare_parameter('arm_feedback_joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('arm_feedback_max_age_sec', 0.50)
        self.declare_parameter('arm_feedback_settle_sec', 0.25)
        self.declare_parameter('arm_feedback_timeout_sec', 5.0)
        self.declare_parameter('arm_feedback_check_period_sec', 0.10)
        self.declare_parameter('arm_feedback_stable_samples', 3)
        self.declare_parameter('arm_joint_confirmation_tolerance_rad', 0.04)
        self.declare_parameter('arm_pose_confirmation_position_tolerance_m', 0.020)
        self.declare_parameter('arm_pose_confirmation_orientation_tolerance_rad', 0.15)
        # Initial target acquisition uses a robust temporal cluster instead of
        # locking the last raw YOLO/depth sample. This prevents mask-edge and
        # depth noise from moving the committed grasp point frame to frame.
        self.declare_parameter('target_stability_samples', 6)
        self.declare_parameter('target_stability_max_jump_m', 0.012)
        self.declare_parameter('target_stability_rms_m', 0.005)
        self.declare_parameter('target_filter_window_samples', 9)
        self.declare_parameter('target_filter_outlier_distance_m', 0.025)
        self.declare_parameter('target_stability_max_sample_gap_sec', 0.75)
        self.declare_parameter('target_lock_min_confidence', 0.70)

        # Rover-motion interlock.  The wrist/arm must not move while the rover
        # base is being driven because the camera target and collision geometry
        # are no longer stationary in the planning frame.
        self.declare_parameter('pause_arm_when_rover_moving', True)
        self.declare_parameter('rover_motion_cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('rover_motion_linear_threshold_mps', 0.02)
        self.declare_parameter('rover_motion_angular_threshold_radps', 0.03)
        self.declare_parameter('rover_motion_pause_hold_sec', 0.75)
        self.declare_parameter('rover_motion_cancel_active_arm_motion', True)

        # Multi-layer live tracking / task supervisor
        # For a floor grasp with a wrist camera, perception must be frozen once a
        # stable target is locked. Otherwise YOLO/depth starts seeing the moving
        # gripper, probe tip, or floor during closure and can trigger hidden target
        # updates/replans while the fingers are closing.
        self.declare_parameter('continuous_tracking_enabled', False)
        self.declare_parameter('hard_freeze_perception_after_lock', True)
        self.declare_parameter('disable_refinement_after_lock', True)
        self.declare_parameter('disable_live_replan_after_lock', True)
        self.declare_parameter('replan_target_move_threshold_m', 0.035)
        self.declare_parameter('max_replans_per_grasp', 2)
        self.declare_parameter('tracking_lost_timeout_sec', 1.2)
        self.declare_parameter('auto_restart_after_success', False)
        self.declare_parameter('success_lockout_sec', 999999.0)
        self.declare_parameter('hold_object_after_success', True)
        self.declare_parameter('clear_target_after_success', False)

        # Gripper confirmation.
        # The arm must not retreat until the gripper command is finished.
        self.declare_parameter('gripper_feedback_available', True)
        self.declare_parameter('gripper_require_feedback_for_completion', True)
        self.declare_parameter('gripper_feedback_max_age_sec', 0.50)
        self.declare_parameter('gripper_confirm_timeout_sec', 12.0)
        self.declare_parameter('gripper_goal_tolerance', 0.006)
        self.declare_parameter('gripper_contact_min_position', 0.018)
        self.declare_parameter('gripper_contact_stall_sec', 0.35)
        self.declare_parameter('gripper_contact_position_epsilon_rad', 0.003)
        self.declare_parameter('gripper_contact_min_closing_travel_rad', 0.20)
        self.declare_parameter('gripper_contact_gap_tolerance_m', 0.015)
        self.declare_parameter('trust_gripper_contact_for_success', True)
        self.declare_parameter('lift_check_floor_fail_samples', 3)
        self.declare_parameter('never_open_after_contact_during_retry', True)
        self.declare_parameter('keep_closed_on_lift_check_failure_without_feedback', True)
        self.declare_parameter('require_lift_check_success_for_transport', True)
        self.declare_parameter('close_gripper_extra_wait_sec', 0.4)

        # Active pre-grasp servo/refinement supervisor.
        # Detection is allowed only until pre-grasp refinement finishes.  After
        # the final grasp pose is committed, perception is frozen so the arm
        # cannot chase the finger/floor during closing.
        self.declare_parameter('pregrasp_active_correction_enabled', True)
        self.declare_parameter('pregrasp_active_correction_threshold_m', 0.012)
        self.declare_parameter('pregrasp_active_correction_max_cycles', 3)
        self.declare_parameter('close_in_one_go_after_pregrasp_refine', True)
        self.declare_parameter('lock_grasp_orientation_after_initial_plan', True)
        self.declare_parameter('preserve_orientation_across_pregrasp_retries', True)
        self.declare_parameter('pregrasp_retry_orientation_hold_sec', 120.0)
        self.declare_parameter('pregrasp_retry_target_radius_m', 0.080)
        self.declare_parameter('fourbar_arc_guard_enabled', True)
        self.declare_parameter('fourbar_arc_sample_count', 15)
        self.declare_parameter('fourbar_open_close_guard_extra_m', 0.015)

        # Pre-grasp supervision.
        # During long MoveIt pre-grasp motion the wrist camera moves, so YOLO/depth
        # can appear to jump even when the probe is static. Treat that as advisory.
        self.declare_parameter('ignore_live_replan_during_pregrasp', True)
        self.declare_parameter('use_recent_live_target_after_pregrasp', False)
        self.declare_parameter('pregrasp_recent_target_max_age_sec', 6.0)
        self.declare_parameter('pregrasp_live_update_accept_m', 0.055)
        self.declare_parameter('continue_if_live_target_stale_after_pregrasp', True)
        self.declare_parameter('probe_shape_aware_center_enabled', True)
        self.declare_parameter('probe_parallel_center_update_scale', 0.0)
        # Pre-grasp watchdog/finalizer: MoveIt can keep executing forever when
        # a loose position/orientation constraint is almost satisfied. If the
        # link is already near pre-grasp, cancel/finalize instead of staying in
        # an endless live-refinement loop.
        self.declare_parameter('pregrasp_watchdog_enabled', True)
        self.declare_parameter('pregrasp_watchdog_timeout_sec', 7.0)
        self.declare_parameter('pregrasp_watchdog_min_sec', 2.0)
        self.declare_parameter('pregrasp_link_arrival_tolerance_m', 0.065)
        self.declare_parameter('pregrasp_watchdog_force_after_timeout', True)
        self.declare_parameter('pregrasp_max_final_replans', 1)
        self.declare_parameter('pregrasp_finalize_even_if_moveit_silent', True)

        # Failure/retry supervision. Prevent instant restart loops after path failure.
        self.declare_parameter('failure_cooldown_sec', 3.0)
        self.declare_parameter('cartesian_retry_lift_m', 0.035)
        self.declare_parameter('cartesian_max_retries', 1)
        self.declare_parameter('stop_after_final_approach_failure', True)
        self.declare_parameter('final_approach_failure_lockout_sec', 999999.0)

        # Lift-check verification after closing.
        self.declare_parameter('verify_grasp_after_lift', True)
        self.declare_parameter('lift_check_distance_m', 0.055)
        self.declare_parameter('lift_check_detect_timeout_sec', 1.0)
        self.declare_parameter('max_grasp_attempts', 2)
        self.declare_parameter('retry_extra_grasp_depth_m', 0.012)
        self.declare_parameter('grasp_failure_same_place_radius_m', 0.090)
        self.declare_parameter('grasp_success_min_lift_m', 0.030)
        self.declare_parameter('lift_check_require_positive_z_success', True)

        # Visual refinement: after pre-grasp move, re-detect close-range to correct grasp pose.
        # Default OFF for floor probe grasping: close-range wrist-camera frames are
        # frequently contaminated by the gripper/floor and can move the target while closing.
        self.declare_parameter('refine_enabled', False)
        self.declare_parameter('refine_samples', 4)        # frames to average
        self.declare_parameter('refine_min_samples_to_accept', 1)
        self.declare_parameter('refine_commit_on_timeout', True)
        self.declare_parameter('refine_timeout_sec', 2.2)  # bounded; never stay in refine forever
        self.declare_parameter('refine_max_jump_m', 0.05)  # discard noisy frames

        # Reject refinement if the close-range detection jumps to another object edge/floor point.
        self.declare_parameter('refine_accept_radius_m', 0.045)
        self.declare_parameter('refine_lateral_max_m', 0.035)
        self.declare_parameter('refine_vertical_max_m', 0.060)

        # Do not use free-space MoveGroup fallback during final grasp.
        # This was causing the bad/wild movement near the probe.
        self.declare_parameter('allow_movegroup_fallback_for_grasp', False)
        self.declare_parameter('final_grasp_movegroup_fallback_position_tol', 0.012)

        # Markers
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('markers_topic', '/vision_grasp/markers')
        self.declare_parameter('show_camera_visibility', True)
        self.declare_parameter('camera_visibility_range_m', 0.60)
        self.declare_parameter('marker_scale', 0.025)
        self.declare_parameter('camera_frustum_line_width', 0.004)
        self.declare_parameter('marker_frame', '')
        self.declare_parameter('marker_use_zero_stamp', True)
        self.declare_parameter('floor_z_min', -0.08)
        self.declare_parameter('reject_targets_below_floor', False)

        # Read params
        p = self.get_parameter
        self.model_path = p('model_path').value
        self.target_class = p('target_class').value
        self.confidence_threshold = float(p('confidence_threshold').value)
        self.detect_period_sec = float(p('detect_period_sec').value)
        self.roi_half_size_px = int(p('roi_half_size_px').value)
        self.max_depth_m = float(p('max_depth_m').value)
        self.min_depth_m = float(p('min_depth_m').value)
        self.max_color_depth_stamp_gap_sec = max(
            0.0, float(p('max_color_depth_stamp_gap_sec').value)
        )
        self.sensor_sync_queue_size = max(2, int(p('sensor_sync_queue_size').value))

        self.refine_confidence_threshold = float(p('refine_confidence_threshold').value)
        self.refine_use_projection_fallback = bool(p('refine_use_projection_fallback').value)
        self.refine_projection_roi_half_size_px = int(p('refine_projection_roi_half_size_px').value)
        self.refine_min_depth_m = float(p('refine_min_depth_m').value)
        self.refine_depth_band_m = float(p('refine_depth_band_m').value)

        self.use_segmentation_mask = bool(p('use_segmentation_mask').value)
        self.mask_score_threshold = float(p('mask_score_threshold').value)
        self.mask_min_pixels = int(p('mask_min_pixels').value)
        self.mask_erode_px = int(p('mask_erode_px').value)
        self.mask_depth_percentile = float(p('mask_depth_percentile').value)

        self.planning_frame = p('planning_frame').value
        self.planning_group = p('planning_group').value
        self.planning_link = p('planning_link').value
        self.keep_current_orientation = bool(p('keep_current_orientation').value)
        self.fixed_roll = float(p('fixed_roll').value)
        self.fixed_pitch = float(p('fixed_pitch').value)
        self.fixed_yaw = float(p('fixed_yaw').value)
        self.approach_axis_in_tool = normalize(np.array(p('approach_axis_in_tool').value, dtype=np.float64))

        self.pre_grasp_distance = float(p('pre_grasp_distance').value)
        self.grasp_depth_below_surface_m = float(p('grasp_depth_below_surface_m').value)
        self.base_grasp_depth_below_surface_m = self.grasp_depth_below_surface_m
        self.retreat_distance = float(p('retreat_distance').value)
        self.target_point_offset_in_link = [float(v) for v in p('target_point_offset_in_link').value]
        self.use_orientation_constraint = bool(p('use_orientation_constraint').value)
        self.min_pose_z = float(p('min_pose_z').value)
        self.pre_grasp_position_tol = float(p('pre_grasp_position_tol').value)
        camera_offset = list(p('grasp_target_offset_camera_xyz_m').value)
        if len(camera_offset) != 3:
            raise ValueError(
                'grasp_target_offset_camera_xyz_m must contain exactly three values: [x, y, z].'
            )
        self.grasp_target_offset_in_camera = np.array(
            [float(v) for v in camera_offset], dtype=np.float64
        )
        self.auto_calibrate_camera_offset_enabled = bool(
            p('auto_calibrate_camera_offset_enabled').value
        )
        self.auto_calibrate_camera_offset_min_samples = max(
            4, int(p('auto_calibrate_camera_offset_min_samples').value)
        )
        self.auto_calibrate_camera_offset_min_rotation_rad = math.radians(max(
            1.0, float(p('auto_calibrate_camera_offset_min_rotation_deg').value)
        ))
        self.auto_calibrate_camera_offset_max_condition = max(
            1.0, float(p('auto_calibrate_camera_offset_max_condition').value)
        )
        self.auto_calibrate_camera_offset_max_m = max(
            0.001, float(p('auto_calibrate_camera_offset_max_m').value)
        )
        self.auto_calibrate_camera_offset_max_rms_m = max(
            0.0005, float(p('auto_calibrate_camera_offset_max_rms_m').value)
        )
        self.auto_calibrate_camera_offset_min_improvement_m = max(
            0.0, float(p('auto_calibrate_camera_offset_min_improvement_m').value)
        )
        self.auto_calibrate_camera_offset_max_step_m = max(
            0.001, float(p('auto_calibrate_camera_offset_max_step_m').value)
        )
        self.auto_calibrate_camera_offset_max_samples = max(
            self.auto_calibrate_camera_offset_min_samples,
            int(p('auto_calibrate_camera_offset_max_samples').value),
        )
        self.grasp_target_bias_in_base = np.array([
            float(p('grasp_target_bias_base_x_m').value),
            float(p('grasp_target_bias_base_y_m').value),
            float(p('grasp_target_bias_base_z_m').value),
        ], dtype=np.float64)
        self.grasp_target_bias_in_tool = np.array([
            float(p('grasp_target_bias_tool_x_m').value),
            float(p('grasp_target_bias_tool_y_m').value),
            float(p('grasp_target_bias_tool_z_m').value),
        ], dtype=np.float64)
        self.lock_wrist_joint = bool(p('lock_wrist_joint').value)
        self.lock_wrist_joint_name = p('lock_wrist_joint_name').value
        self.lock_wrist_joint_tolerance = float(p('lock_wrist_joint_tolerance').value)

        self.cartesian_lock_orientation = bool(p('cartesian_lock_orientation').value)
        self.diagnose_final_cartesian_failure = bool(p('diagnose_final_cartesian_failure').value)
        self.cartesian_max_step = float(p('cartesian_max_step').value)
        self.cartesian_jump_threshold = float(p('cartesian_jump_threshold').value)
        self.cartesian_fraction_min = float(p('cartesian_fraction_min').value)

        self.use_joint_retreat_home = bool(p('use_joint_retreat_home').value)
        self.retreat_home_joint_names = list(p('retreat_home_joint_names').value)
        self.retreat_home_joint_positions = [float(v) for v in p('retreat_home_joint_positions').value]
        self.joint_goal_tolerance = float(p('joint_goal_tolerance').value)

        self.gripper_joint_lower_limit = float(p('gripper_joint_lower_limit').value)
        self.gripper_joint_upper_limit = float(p('gripper_joint_upper_limit').value)
        if self.gripper_joint_upper_limit <= self.gripper_joint_lower_limit:
            raise ValueError(
                'gripper_joint_upper_limit must be greater than '
                'gripper_joint_lower_limit'
            )
        limit_span = self.gripper_joint_upper_limit - self.gripper_joint_lower_limit
        self.gripper_joint_limit_margin = float(np.clip(
            float(p('gripper_joint_limit_margin').value),
            0.0,
            0.49 * limit_span,
        ))
        self.gripper_safe_lower_limit = (
            self.gripper_joint_lower_limit + self.gripper_joint_limit_margin
        )
        self.gripper_safe_upper_limit = (
            self.gripper_joint_upper_limit - self.gripper_joint_limit_margin
        )

        requested_open = float(p('gripper_open_width').value)
        requested_close = float(p('gripper_close_width').value)
        requested_preclose = float(p('gripper_preclose_width').value)
        self.gripper_open = self._limit_gripper_target(requested_open, 'configured open')
        self.gripper_close = self._limit_gripper_target(requested_close, 'configured close')
        self.gripper_preclose = self._limit_gripper_target(
            requested_preclose, 'configured pre-close'
        )
        if self.gripper_open >= self.gripper_close:
            raise ValueError(
                'Safe gripper configuration requires gripper_open_width < '
                'gripper_close_width'
            )
        self.final_grasp_arm_settle_sec = float(p('final_grasp_arm_settle_sec').value)
        self.final_grasp_pose_check_enabled = bool(p('final_grasp_pose_check_enabled').value)
        self.final_grasp_pose_position_tolerance_m = max(
            0.001,
            float(p('final_grasp_pose_position_tolerance_m').value),
        )
        self.final_grasp_pose_orientation_tolerance_rad = max(
            0.01,
            float(p('final_grasp_pose_orientation_tolerance_rad').value),
        )
        self.final_grasp_pose_check_timeout_sec = max(
            0.0,
            float(p('final_grasp_pose_check_timeout_sec').value),
        )
        self.final_grasp_pose_check_period_sec = max(
            0.05,
            float(p('final_grasp_pose_check_period_sec').value),
        )
        self.gripper_settle_sec = float(p('gripper_settle_sec').value)
        self.gripper_command_duration_sec = float(p('gripper_command_duration_sec').value)
        self.gripper_no_feedback_close_complete_sec = float(p('gripper_no_feedback_close_complete_sec').value)
        self.gripper_command_mode = p('gripper_command_mode').value
        self.gripper_action_name = p('gripper_action_name').value
        self.gripper_joint_name = p('gripper_joint_name').value
        self.gripper_action_timeout_sec = float(p('gripper_action_timeout_sec').value)
        self.gripper_require_action_success_for_completion = bool(
            p('gripper_require_action_success_for_completion').value
        )

        self.adaptive_gripper_enabled = bool(p('adaptive_gripper_enabled').value)
        self.object_width_safety_margin_m = float(p('object_width_safety_margin_m').value)
        self.adaptive_gripper_min_width_m = float(p('adaptive_gripper_min_width_m').value)
        self.adaptive_gripper_max_width_m = float(p('adaptive_gripper_max_width_m').value)
        self.adaptive_gripper_width_percentile = float(p('adaptive_gripper_width_percentile').value)
        self.minimum_probe_width_m = float(p('minimum_probe_width_m').value)
        self.nominal_probe_width_m = float(p('nominal_probe_width_m').value)
        self.maximum_probe_width_m = float(p('maximum_probe_width_m').value)
        self.clamp_probe_width_for_grasp = bool(p('clamp_probe_width_for_grasp').value)
        self.object_width_final_clearance_m = float(p('object_width_final_clearance_m').value)
        self.object_width_preclose_clearance_m = float(p('object_width_preclose_clearance_m').value)
        self.preclose_min_q_margin_rad = float(p('preclose_min_q_margin_rad').value)
        self.fourbar_preclose_before_grasp = bool(p('fourbar_preclose_before_grasp').value)
        self.fourbar_final_close_steps = max(1, int(p('fourbar_final_close_steps').value))
        self.fourbar_final_close_step_wait_sec = float(p('fourbar_final_close_step_wait_sec').value)
        self.freeze_arm_during_gripper_enabled = bool(p('freeze_arm_during_gripper_enabled').value)
        self.hold_after_close_no_motion = bool(p('hold_after_close_no_motion').value)
        self.post_grasp_lift_then_pick_home = bool(p('post_grasp_lift_then_pick_home').value)
        self.post_grasp_planning_time_sec = max(1.0, float(p('post_grasp_planning_time_sec').value))
        self.pick_home_joint_names = list(p('pick_home_joint_names').value)
        self.pick_home_joint_positions = [float(v) for v in p('pick_home_joint_positions').value]
        self.place_in_base_box_after_grasp = bool(p('place_in_base_box_after_grasp').value)
        self.base_box_drop_use_pose = bool(p('base_box_drop_use_pose').value)
        self.base_box_drop_frame = str(p('base_box_drop_frame').value)
        self.base_box_drop_xyz = [float(v) for v in p('base_box_drop_xyz').value]
        self.base_box_drop_rpy = [float(v) for v in p('base_box_drop_rpy').value]
        self.base_box_drop_target_point_offset_in_link = [
            float(v) for v in p('base_box_drop_target_point_offset_in_link').value
        ]
        self.base_box_drop_position_tolerance_m = max(
            0.001, float(p('base_box_drop_position_tolerance_m').value)
        )
        self.base_box_drop_orientation_tolerance_rad = max(
            0.01, float(p('base_box_drop_orientation_tolerance_rad').value)
        )
        self.base_box_drop_joint_names = list(p('base_box_drop_joint_names').value)
        self.base_box_drop_joint_positions = [float(v) for v in p('base_box_drop_joint_positions').value]
        self.base_box_planning_time_sec = max(1.0, float(p('base_box_planning_time_sec').value))
        self.base_box_release_wait_sec = max(0.0, float(p('base_box_release_wait_sec').value))
        self.return_pick_home_after_base_box_place = bool(p('return_pick_home_after_base_box_place').value)
        self.base_box_drop_marker_enabled = bool(p('base_box_drop_marker_enabled').value)
        self.base_box_drop_marker_scale_m = max(0.01, float(p('base_box_drop_marker_scale_m').value))
        self.base_box_drop_marker_axes_length_m = max(
            0.02, float(p('base_box_drop_marker_axes_length_m').value)
        )
        if not self._base_box_drop_pose_config_valid():
            self.get_logger().error('Invalid base-box pose configuration: base_box_drop_frame must be non-empty, '
                'and base_box_drop_xyz/base_box_drop_rpy must each contain exactly three values. '
                'The drop marker and pose-based placement will remain disabled.')
        self.floor_safe_grasp_enabled = bool(p('floor_safe_grasp_enabled').value)
        self.max_grasp_descent_below_target_m = float(p('max_grasp_descent_below_target_m').value)
        self.min_grasp_height_above_floor_m = float(p('min_grasp_height_above_floor_m').value)

        self.fourbar_contact_y_offset_m = float(p('fourbar_contact_y_offset_m').value)
        self.fourbar_contact_z_open_m = float(p('fourbar_contact_z_open_m').value)
        self.fourbar_contact_z_closed_m = float(p('fourbar_contact_z_closed_m').value)
        self.fourbar_q_min_for_floor_grasp = float(p('fourbar_q_min_for_floor_grasp').value)
        self.fourbar_q_max_for_floor_grasp = float(p('fourbar_q_max_for_floor_grasp').value)
        self.fourbar_max_contact_lift_m = float(p('fourbar_max_contact_lift_m').value)
        self.fourbar_min_arc_clearance_m = float(p('fourbar_min_arc_clearance_m').value)
        self.fourbar_ground_guard_enabled = bool(p('fourbar_ground_guard_enabled').value)
        self.fourbar_bucket_tip_z_max_m = float(p('fourbar_bucket_tip_z_max_m').value)
        self.fourbar_ground_clearance_m = float(p('fourbar_ground_clearance_m').value)
        self.floor_safe_contact_height_m = float(p('floor_safe_contact_height_m').value)
        self.publish_object_pose_enabled = bool(p('publish_object_pose').value)
        self.object_pose_topic = p('object_pose_topic').value
        self.object_pose_axis_length_m = float(p('object_pose_axis_length_m').value)
        self.object_yaw_align_enabled = bool(p('object_yaw_align_enabled').value)
        self.object_yaw_rotation_offset_deg = float(p('object_yaw_rotation_offset_deg').value)
        self.object_orientation_min_eigenratio = float(p('object_orientation_min_eigenratio').value)
        self.stl_yaw_correction_deg = float(p('stl_yaw_correction_deg').value)

        self.position_tol = float(p('position_tolerance_xyz').value)
        self.orientation_tol = float(p('orientation_tolerance_rad').value)
        self.allowed_planning_time = float(p('allowed_planning_time').value)
        self.num_planning_attempts = int(p('num_planning_attempts').value)
        self.velocity_scale = float(p('velocity_scale').value)
        self.acceleration_scale = float(p('acceleration_scale').value)
        self.arm_require_feedback_for_completion = bool(
            p('arm_require_feedback_for_completion').value
        )
        self.arm_feedback_joint_names = [str(v) for v in p('arm_feedback_joint_names').value]
        self.arm_feedback_max_age_sec = max(0.05, float(p('arm_feedback_max_age_sec').value))
        self.arm_feedback_settle_sec = max(0.0, float(p('arm_feedback_settle_sec').value))
        self.arm_feedback_timeout_sec = max(0.5, float(p('arm_feedback_timeout_sec').value))
        self.arm_feedback_check_period_sec = max(
            0.02, float(p('arm_feedback_check_period_sec').value)
        )
        self.arm_feedback_stable_samples = max(1, int(p('arm_feedback_stable_samples').value))
        self.arm_joint_confirmation_tolerance_rad = max(
            0.005, float(p('arm_joint_confirmation_tolerance_rad').value)
        )
        self.arm_pose_confirmation_position_tolerance_m = max(
            0.002, float(p('arm_pose_confirmation_position_tolerance_m').value)
        )
        self.arm_pose_confirmation_orientation_tolerance_rad = max(
            0.01, float(p('arm_pose_confirmation_orientation_tolerance_rad').value)
        )
        self.target_filter_window_samples = max(
            3, int(p('target_filter_window_samples').value)
        )
        self.target_stability_samples = int(np.clip(
            int(p('target_stability_samples').value),
            3,
            self.target_filter_window_samples,
        ))
        self.target_stability_max_jump_m = max(
            0.001, float(p('target_stability_max_jump_m').value)
        )
        self.target_stability_rms_m = max(
            0.0005, float(p('target_stability_rms_m').value)
        )
        self.target_filter_outlier_distance_m = max(
            self.target_stability_max_jump_m,
            float(p('target_filter_outlier_distance_m').value),
        )
        self.target_stability_max_sample_gap_sec = max(
            self.detect_period_sec * 1.5,
            float(p('target_stability_max_sample_gap_sec').value),
        )
        self.target_lock_min_confidence = float(np.clip(
            float(p('target_lock_min_confidence').value), 0.0, 1.0
        ))

        self.pause_arm_when_rover_moving = bool(p('pause_arm_when_rover_moving').value)
        self.rover_motion_cmd_vel_topic = str(p('rover_motion_cmd_vel_topic').value)
        self.rover_motion_linear_threshold_mps = float(p('rover_motion_linear_threshold_mps').value)
        self.rover_motion_angular_threshold_radps = float(p('rover_motion_angular_threshold_radps').value)
        self.rover_motion_pause_hold_sec = float(p('rover_motion_pause_hold_sec').value)
        self.rover_motion_cancel_active_arm_motion = bool(p('rover_motion_cancel_active_arm_motion').value)

        self.continuous_tracking_enabled = bool(p('continuous_tracking_enabled').value)
        self.hard_freeze_perception_after_lock = bool(p('hard_freeze_perception_after_lock').value)
        self.disable_refinement_after_lock = bool(p('disable_refinement_after_lock').value)
        self.disable_live_replan_after_lock = bool(p('disable_live_replan_after_lock').value)
        self.replan_target_move_threshold_m = float(p('replan_target_move_threshold_m').value)
        self.max_replans_per_grasp = int(p('max_replans_per_grasp').value)
        self.tracking_lost_timeout_sec = float(p('tracking_lost_timeout_sec').value)
        self.auto_restart_after_success = bool(p('auto_restart_after_success').value)
        self.success_lockout_sec = float(p('success_lockout_sec').value)
        self.hold_object_after_success = bool(p('hold_object_after_success').value)
        self.clear_target_after_success = bool(p('clear_target_after_success').value)

        self.gripper_feedback_available = bool(p('gripper_feedback_available').value)
        self.gripper_require_feedback_for_completion = bool(
            p('gripper_require_feedback_for_completion').value
        )
        self.gripper_feedback_max_age_sec = max(
            0.05, float(p('gripper_feedback_max_age_sec').value)
        )
        self.gripper_confirm_timeout_sec = float(p('gripper_confirm_timeout_sec').value)
        self.gripper_goal_tolerance = float(p('gripper_goal_tolerance').value)
        self.gripper_contact_min_position = float(p('gripper_contact_min_position').value)
        self.gripper_contact_stall_sec = max(0.0, float(p('gripper_contact_stall_sec').value))
        self.gripper_contact_position_epsilon_rad = max(
            0.0, float(p('gripper_contact_position_epsilon_rad').value)
        )
        self.gripper_contact_min_closing_travel_rad = max(
            0.0, float(p('gripper_contact_min_closing_travel_rad').value)
        )
        self.gripper_contact_gap_tolerance_m = max(
            0.0, float(p('gripper_contact_gap_tolerance_m').value)
        )
        self.trust_gripper_contact_for_success = bool(p('trust_gripper_contact_for_success').value)
        self.lift_check_floor_fail_samples = int(p('lift_check_floor_fail_samples').value)
        self.never_open_after_contact_during_retry = bool(p('never_open_after_contact_during_retry').value)
        self.keep_closed_on_lift_check_failure_without_feedback = bool(
            p('keep_closed_on_lift_check_failure_without_feedback').value
        )
        self.require_lift_check_success_for_transport = bool(p('require_lift_check_success_for_transport').value)
        self.close_gripper_extra_wait_sec = float(p('close_gripper_extra_wait_sec').value)

        self.pregrasp_active_correction_enabled = bool(p('pregrasp_active_correction_enabled').value)
        self.pregrasp_active_correction_threshold_m = float(p('pregrasp_active_correction_threshold_m').value)
        self.pregrasp_active_correction_max_cycles = int(p('pregrasp_active_correction_max_cycles').value)
        self.close_in_one_go_after_pregrasp_refine = bool(p('close_in_one_go_after_pregrasp_refine').value)
        self.lock_grasp_orientation_after_initial_plan = bool(p('lock_grasp_orientation_after_initial_plan').value)
        self.preserve_orientation_across_pregrasp_retries = bool(
            p('preserve_orientation_across_pregrasp_retries').value
        )
        self.pregrasp_retry_orientation_hold_sec = max(
            0.0, float(p('pregrasp_retry_orientation_hold_sec').value)
        )
        self.pregrasp_retry_target_radius_m = max(
            0.001, float(p('pregrasp_retry_target_radius_m').value)
        )
        self.fourbar_arc_guard_enabled = bool(p('fourbar_arc_guard_enabled').value)
        self.fourbar_arc_sample_count = max(3, int(p('fourbar_arc_sample_count').value))
        self.fourbar_open_close_guard_extra_m = float(p('fourbar_open_close_guard_extra_m').value)

        self.ignore_live_replan_during_pregrasp = bool(p('ignore_live_replan_during_pregrasp').value)
        self.use_recent_live_target_after_pregrasp = bool(p('use_recent_live_target_after_pregrasp').value)
        self.pregrasp_recent_target_max_age_sec = float(p('pregrasp_recent_target_max_age_sec').value)
        self.pregrasp_live_update_accept_m = float(p('pregrasp_live_update_accept_m').value)
        self.continue_if_live_target_stale_after_pregrasp = bool(p('continue_if_live_target_stale_after_pregrasp').value)
        self.probe_shape_aware_center_enabled = bool(p('probe_shape_aware_center_enabled').value)
        self.probe_parallel_center_update_scale = float(p('probe_parallel_center_update_scale').value)
        self.pregrasp_watchdog_enabled = bool(p('pregrasp_watchdog_enabled').value)
        self.pregrasp_watchdog_timeout_sec = float(p('pregrasp_watchdog_timeout_sec').value)
        self.pregrasp_watchdog_min_sec = float(p('pregrasp_watchdog_min_sec').value)
        self.pregrasp_link_arrival_tolerance_m = float(p('pregrasp_link_arrival_tolerance_m').value)
        self.pregrasp_watchdog_force_after_timeout = bool(p('pregrasp_watchdog_force_after_timeout').value)
        self.pregrasp_max_final_replans = int(p('pregrasp_max_final_replans').value)
        self.pregrasp_finalize_even_if_moveit_silent = bool(p('pregrasp_finalize_even_if_moveit_silent').value)

        self.failure_cooldown_sec = float(p('failure_cooldown_sec').value)
        self.cartesian_retry_lift_m = float(p('cartesian_retry_lift_m').value)
        self.cartesian_max_retries = int(p('cartesian_max_retries').value)
        self.stop_after_final_approach_failure = bool(p('stop_after_final_approach_failure').value)
        self.final_approach_failure_lockout_sec = float(p('final_approach_failure_lockout_sec').value)

        self.verify_grasp_after_lift = bool(p('verify_grasp_after_lift').value)
        self.lift_check_distance_m = float(p('lift_check_distance_m').value)
        self.lift_check_detect_timeout_sec = float(p('lift_check_detect_timeout_sec').value)
        self.max_grasp_attempts = int(p('max_grasp_attempts').value)
        self.retry_extra_grasp_depth_m = float(p('retry_extra_grasp_depth_m').value)
        self.grasp_failure_same_place_radius_m = float(p('grasp_failure_same_place_radius_m').value)
        self.grasp_success_min_lift_m = float(p('grasp_success_min_lift_m').value)
        self.lift_check_require_positive_z_success = bool(p('lift_check_require_positive_z_success').value)

        self.refine_enabled = bool(p('refine_enabled').value)
        self.refine_samples = int(p('refine_samples').value)
        self.refine_min_samples_to_accept = max(0, int(p('refine_min_samples_to_accept').value))
        self.refine_commit_on_timeout = bool(p('refine_commit_on_timeout').value)
        self.refine_timeout_sec = float(p('refine_timeout_sec').value)
        self.refine_max_jump_m = float(p('refine_max_jump_m').value)

        self.refine_accept_radius_m = float(p('refine_accept_radius_m').value)
        self.refine_lateral_max_m = float(p('refine_lateral_max_m').value)
        self.refine_vertical_max_m = float(p('refine_vertical_max_m').value)
        self.allow_movegroup_fallback_for_grasp = bool(p('allow_movegroup_fallback_for_grasp').value)
        self.final_grasp_movegroup_fallback_position_tol = float(p('final_grasp_movegroup_fallback_position_tol').value)

        self.publish_markers_enabled = bool(p('publish_markers').value)
        self.markers_topic = p('markers_topic').value
        self.show_camera_visibility = bool(p('show_camera_visibility').value)
        self.camera_visibility_range_m = float(p('camera_visibility_range_m').value)
        self.marker_scale = float(p('marker_scale').value)
        self.camera_frustum_line_width = float(p('camera_frustum_line_width').value)
        self.marker_frame = str(p('marker_frame').value) if str(p('marker_frame').value) else self.planning_frame
        self.marker_use_zero_stamp = bool(p('marker_use_zero_stamp').value)
        self.floor_z_min = float(p('floor_z_min').value)
        self.reject_targets_below_floor = bool(p('reject_targets_below_floor').value)

        self.latest_color: Optional[np.ndarray] = None
        self.latest_color_stamp: Optional[rclpy.time.Time] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_stamp: Optional[rclpy.time.Time] = None
        self.latest_depth_frame: Optional[str] = None
        self._color_frame_queue = deque(maxlen=self.sensor_sync_queue_size)
        self._depth_frame_queue = deque(maxlen=self.sensor_sync_queue_size)
        self._last_inference_pair_key = None
        self._camera_calibration_raw_world = deque(
            maxlen=self.auto_calibrate_camera_offset_max_samples
        )
        self._camera_calibration_rotations = deque(
            maxlen=self.auto_calibrate_camera_offset_max_samples
        )
        self._camera_calibration_last_raw_world: Optional[np.ndarray] = None
        self._camera_calibration_last_rotation: Optional[np.ndarray] = None
        self._pending_camera_offset_estimate: Optional[CameraOffsetEstimate] = None
        self._auto_camera_calibration_applied_for_sequence = False
        self._post_grasp_floor_active = False
        self._post_grasp_probe_attached = False
        self.camera_info: Optional[CameraInfo] = None
        self._yolo_worker: Optional[YoloWorker] = None
        self._stamp_gap_warned_sec = 0.0
        self.busy = False
        self.sequence_stage = 'idle'
        self.current_target_point_base: Optional[np.ndarray] = None
        self.target_history: List[np.ndarray] = []
        self.target_history_stamps: List[float] = []
        self.target_confidence_history: List[float] = []
        self.filtered_target_point_base: Optional[np.ndarray] = None
        self.filtered_target_confidence: float = 0.0
        self.target_filter_max_residual_m: float = float('inf')
        self.target_filter_rms_m: float = float('inf')
        self.pending_timers = []
        self.pre_grasp_pose: Optional[PoseStamped] = None
        self.grasp_pose: Optional[PoseStamped] = None
        self.retreat_pose: Optional[PoseStamped] = None
        self.grasp_orientation: Optional[Quaternion] = None
        self._retry_grasp_orientation: Optional[Quaternion] = None
        self._retry_grasp_target: Optional[np.ndarray] = None
        self._retry_grasp_orientation_until_sec: float = 0.0
        self.sequence_wrist_value: Optional[float] = None
        self.current_joint_positions: dict = {}
        self.current_joint_update_sec: dict = {}
        self._refine_buffer: List[np.ndarray] = []
        self._refine_start_sec: float = 0.0

        # Supervisor memory
        self.task_complete = False
        self.holding_object = False
        self.success_until_sec = 0.0
        self.live_target_point_base: Optional[np.ndarray] = None
        self.live_target_stamp_sec: float = 0.0
        self.sequence_locked_target_point_base: Optional[np.ndarray] = None
        self.sequence_locked_object_long_axis_base: Optional[np.ndarray] = None
        self.perception_frozen_for_sequence = False
        self.last_detection_name = ''
        self.last_detection_conf = 0.0
        self.pending_replan_after_motion = False
        self.replan_count = 0
        self.blocked_until_sec = 0.0
        self.paused_after_failure = False
        self.failure_count = 0
        self.last_failure_reason = ''
        self._cartesian_grasp_retries = 0
        self.grasp_attempt_count = 0
        self.locked_target_before_lift: Optional[np.ndarray] = None
        self.retry_target_from_lift_check: Optional[np.ndarray] = None
        self._lift_check_last_nonlifted_target: Optional[np.ndarray] = None
        self.rover_motion_pause_until_sec = 0.0
        self.last_rover_linear_speed = 0.0
        self.last_rover_angular_speed = 0.0

        # Contact-aware gripper memory.
        self.gripper_contact_detected = False
        self.last_gripper_actual: Optional[float] = None
        self.last_gripper_target: Optional[float] = None
        self._lift_floor_fail_count = 0

        # Adaptive gripper sizing state (updated each detection cycle).
        self.computed_gripper_close: float = self.gripper_close
        self.computed_gripper_preclose: float = self.gripper_preclose
        self.last_estimated_object_width_m: Optional[float] = None
        self._last_detected_width_m: Optional[float] = None

        # 6D pose state (updated each detection cycle).
        # effective_target_point_offset_in_link has the Z corrected for the four-bar kinematics.
        self.effective_target_point_offset_in_link: List[float] = list(self.target_point_offset_in_link)
        self._last_detected_orientation_cam: Optional[np.ndarray] = None  # 3x3 rot matrix in camera frame
        self.detected_object_pose: Optional[PoseStamped] = None           # full pose in planning_frame
        self._last_detected_object_rotation_base: Optional[np.ndarray] = None
        self.detected_object_yaw_rad: Optional[float] = None              # yaw in planning_frame
        self._lift_check_timer = None
        self._lift_check_start_sec = 0.0

        # Gripper confirmation state
        self._gripper_wait_timer = None
        self._gripper_wait_start_sec = 0.0
        self._gripper_wait_target = 0.0
        self._gripper_wait_cb: Optional[Callable[[], None]] = None
        self._gripper_wait_seq: int = 0
        self._gripper_wait_stage: str = ''
        self._gripper_wait_start_position: Optional[float] = None
        self._gripper_wait_last_position: Optional[float] = None
        self._gripper_wait_last_motion_sec: float = 0.0
        self._gripper_command_used_action = False
        self._gripper_action_goal_handle = None
        self._gripper_action_accepted = False
        self._gripper_action_succeeded = False
        self._gripper_action_failed_reason: Optional[str] = None

        # Motion-token supervisor.  Every arm motion and delayed timer is tagged
        # with the current sequence id and expected stage.  Stale callbacks are
        # ignored instead of triggering a movement during gripper closure.
        self.sequence_id = 0
        self._close_step_targets: List[float] = []
        self._close_step_index = 0
        self.preclosed_in_air = False
        self.pregrasp_correction_count = 0
        self._pregrasp_motion_start_sec = 0.0
        self._auto_camera_calibration_applied_for_sequence = False
        self._pregrasp_watchdog_timer = None
        self._final_grasp_pose_check_timer = None
        self._final_grasp_pose_check_start_sec = 0.0
        self._pregrasp_force_finalize = False
        self._pregrasp_final_replan_count = 0
        self._active_move_goal_handle = None
        self._pending_arm_motion_confirmation: Optional[dict] = None
        self._arm_confirmation_timer = None
        self._cartesian_plan_in_flight: Optional[Tuple[str, int]] = None
        self._refine_width_buffer: List[float] = []
        self._refine_orientation_cam_last: Optional[np.ndarray] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.move_group_client = ActionClient(self, MoveGroup, p('move_action_name').value)
        self.cartesian_client = self.create_client(GetCartesianPath, p('cartesian_service_name').value)
        self.execute_client = ActionClient(self, ExecuteTrajectory, p('execute_action_name').value)
        self.gripper_action_client = ActionClient(self, FollowJointTrajectory, self.gripper_action_name)

        use_aligned = bool(p('use_aligned_depth').value)
        depth_topic = ('/gripper_camera/aligned_depth_to_color/image_raw'
                       if use_aligned else '/gripper_camera/depth/image_rect_raw')
        info_topic  = ('/gripper_camera/color/camera_info'
                       if use_aligned else '/gripper_camera/depth/camera_info')
        self.get_logger().info(f'Depth source: {depth_topic}  |  Camera info: {info_topic}')
        # Sensor-data QoS (best-effort, shallow queue): only the newest frame
        # matters, and buffering ten stale images just adds latency and memory.
        self.color_sub = self.create_subscription(
            Image, '/gripper_camera/color/image_raw', self.color_cb, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_cb, qos_profile_sensor_data)
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_cb, qos_profile_sensor_data)
        self.joint_states_sub = self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10)
        self.rover_motion_sub = None
        if self.pause_arm_when_rover_moving and self.rover_motion_cmd_vel_topic:
            self.rover_motion_sub = self.create_subscription(
                Twist,
                self.rover_motion_cmd_vel_topic,
                self._rover_cmd_vel_cb,
                10,
            )
            self.get_logger().info(f'Rover-motion arm safety enabled: topic={self.rover_motion_cmd_vel_topic}, '
                f'linear>{self.rover_motion_linear_threshold_mps:.3f} m/s or '
                f'angular>{self.rover_motion_angular_threshold_radps:.3f} rad/s pauses arm motion.')
        self.det_vis_pub = self.create_publisher(Image, '/vision_grasp/detection_image', 10)
        self.gripper_pub = self.create_publisher(Float64, p('gripper_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, self.object_pose_topic, 10)
        self._collision_object_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        self._attached_object_pub = self.create_publisher(AttachedCollisionObject, '/attached_collision_object', 10)

        self.detect_timer = self.create_timer(self.detect_period_sec, self.detect_and_maybe_grasp)
        self.marker_timer = self.create_timer(0.25, self.publish_markers)

        if YOLO_AVAILABLE:
            try:
                self.model, _device = load_yolo_model(
                    self.model_path, logger=self.get_logger()
                )
                # Inference runs in a background thread so it never blocks the
                # executor (gripper ticks, action results, TF). The main
                # thread submits the newest frame pair and consumes the newest
                # completed result on the next detect tick.
                self._yolo_worker = YoloWorker(
                    self.model, device=_device, logger=self.get_logger()
                )
            except Exception as exc:
                self.model = None
                self.get_logger().error(f'Failed to load YOLO model {self.model_path}: {exc}')
        else:
            self.model = None
            self.get_logger().error('ultralytics is not installed in this environment.')

        self.get_logger().info(f'vision_grasp_node ready | target_class={self.target_class} | planning_group={self.planning_group} | '
            f'planning_link={self.planning_link} | planning_frame={self.planning_frame} | gripper_mode={self.gripper_command_mode}')

    def _joint_states_cb(self, msg: JointState) -> None:
        update_sec = self._now_sec()
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_positions[name] = float(pos)
            self.current_joint_update_sec[name] = update_sec

    def color_cb(self, msg: Image) -> None:
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_color_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            self._color_frame_queue.append((self.latest_color_stamp, self.latest_color))
        except Exception as exc:
            self.get_logger().error(f'Color conversion failed: {exc}')

    def depth_cb(self, msg: Image) -> None:
        try:
            self.latest_depth_frame = msg.header.frame_id
            if msg.encoding == '32FC1':
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                depth_mm = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.latest_depth = depth_mm.astype(np.float32) / 1000.0
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
            self.latest_depth_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            self._depth_frame_queue.append((
                self.latest_depth_stamp,
                self.latest_depth_frame,
                self.latest_depth,
            ))
        except Exception as exc:
            self.get_logger().error(f'Depth conversion failed: {exc}')

    def info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        if not self.latest_depth_frame:
            self.latest_depth_frame = msg.header.frame_id

    def call_later(self, seconds: float, cb: Callable[[], None]) -> None:
        """Create a one-shot timer guarded by the current grasp sequence id.

        Without this guard, an old lift/retreat timer can fire after the system
        has moved into a gripper-close stage, which looks exactly like "the arm
        moved when the gripper was about to close".
        """
        holder = {'timer': None}
        seq = self.sequence_id

        def wrapped() -> None:
            timer = holder['timer']
            if timer is not None:
                timer.cancel()
            if seq != self.sequence_id:
                self.get_logger().warning('Ignoring stale delayed callback from an old grasp sequence.')
                return
            cb()

        holder['timer'] = self.create_timer(seconds, wrapped)
        self.pending_timers.append(holder['timer'])

    def _cancel_pending_timers(self) -> None:
        for timer in list(self.pending_timers):
            try:
                timer.cancel()
            except Exception:
                pass
        self.pending_timers.clear()

    def _cancel_final_grasp_pose_check_timer(self) -> None:
        if getattr(self, '_final_grasp_pose_check_timer', None) is not None:
            try:
                self._final_grasp_pose_check_timer.cancel()
            except Exception:
                pass
            self._final_grasp_pose_check_timer = None

    def _new_sequence(self) -> None:
        self.sequence_id += 1
        self._cancel_pending_timers()
        self._cancel_final_grasp_pose_check_timer()
        self._close_step_targets = []
        self._close_step_index = 0
        self.sequence_locked_target_point_base = None
        self.sequence_locked_object_long_axis_base = None
        self.preclosed_in_air = False
        self.pregrasp_correction_count = 0
        self._auto_camera_calibration_applied_for_sequence = False
        self._pregrasp_force_finalize = False
        self._pregrasp_final_replan_count = 0
        self._pregrasp_motion_start_sec = 0.0
        if getattr(self, '_pregrasp_watchdog_timer', None) is not None:
            try:
                self._pregrasp_watchdog_timer.cancel()
            except Exception:
                pass
            self._pregrasp_watchdog_timer = None
        self._active_move_goal_handle = None
        self._refine_width_buffer = []
        self._refine_orientation_cam_last = None
        # Perception remains active during move_pre_grasp/refine only.  It is
        # frozen automatically for grasp, transport, release, and completion.
        self.perception_frozen_for_sequence = True

    def _perception_updates_forbidden_now(self) -> bool:
        """Return True when YOLO/depth must not update target state.

        Required behavior for the probe task:
          1. Before and during move_pre_grasp: live feedback is allowed.
          2. At pre-grasp: explicit refinement is allowed.
          3. After refined grasp geometry is committed: perception is frozen so
             the arm cannot chase gripper fingers, the probe tip, or the floor.
        """
        if not self.hard_freeze_perception_after_lock:
            return False
        if not self.busy:
            return False
        if not self.perception_frozen_for_sequence:
            return False

        if self.sequence_stage in stages.LIVE_FEEDBACK_STAGES:
            return False

        return self.sequence_stage not in stages.TERMINAL_STAGES

    def _rover_motion_active(self) -> bool:
        if not self.pause_arm_when_rover_moving:
            return False
        return self._now_sec() < self.rover_motion_pause_until_sec

    def _rover_cmd_vel_cb(self, msg: Twist) -> None:
        if not self.pause_arm_when_rover_moving:
            return

        linear_speed = math.sqrt(
            float(msg.linear.x) ** 2
            + float(msg.linear.y) ** 2
            + float(msg.linear.z) ** 2
        )
        angular_speed = math.sqrt(
            float(msg.angular.x) ** 2
            + float(msg.angular.y) ** 2
            + float(msg.angular.z) ** 2
        )
        self.last_rover_linear_speed = linear_speed
        self.last_rover_angular_speed = angular_speed

        moving = (
            linear_speed > self.rover_motion_linear_threshold_mps
            or angular_speed > self.rover_motion_angular_threshold_radps
        )
        if not moving:
            return

        was_active = self._rover_motion_active()
        self.rover_motion_pause_until_sec = max(
            self.rover_motion_pause_until_sec,
            self._now_sec() + max(0.0, self.rover_motion_pause_hold_sec),
        )

        if not was_active:
            self.get_logger().warning(f'Rover motion detected on {self.rover_motion_cmd_vel_topic}: '
                f'linear={linear_speed:.3f} m/s angular={angular_speed:.3f} rad/s. '
                'Pausing/canceling arm motion until the rover stops.')

        if self.busy and self.rover_motion_cancel_active_arm_motion:
            self._pause_sequence_for_rover_motion('Rover motion detected while vision grasp was active.')

    def _pause_sequence_for_rover_motion(self, reason: str) -> None:
        if not self.pause_arm_when_rover_moving:
            return
        if not self.busy and self.sequence_stage == 'idle':
            return

        self.get_logger().warning(f'{reason} Arm auto-grasp is paused; no arm trajectory will be sent while '
            f'rover cmd_vel remains above threshold. Last rover speed: '
            f'linear={self.last_rover_linear_speed:.3f} m/s, '
            f'angular={self.last_rover_angular_speed:.3f} rad/s.')
        self._cancel_active_moveit_goal()
        self.reset_sequence(reason)
        self.blocked_until_sec = max(self.blocked_until_sec, self.rover_motion_pause_until_sec)

    def _arm_motion_forbidden_now(self, requested_stage: str) -> bool:
        if self._rover_motion_active():
            self.get_logger().warning(f'Blocked arm motion during rover movement: requested_stage={requested_stage}. '
                f'linear={self.last_rover_linear_speed:.3f} m/s, '
                f'angular={self.last_rover_angular_speed:.3f} rad/s.', throttle_duration_sec=1.0)
            if self.busy and self.rover_motion_cancel_active_arm_motion:
                self._pause_sequence_for_rover_motion(
                    f'Blocked {requested_stage} because rover is moving.'
                )
            return True

        if not self.freeze_arm_during_gripper_enabled:
            return False
        # No MoveIt/Cartesian arm command may be created while a gripper stage is
        # active.  Only pure gripper commands are allowed in these stages.
        if self.sequence_stage in stages.GRIPPER_STAGES:
            return True
        if self.hold_after_close_no_motion and self.sequence_stage == 'verify_gripper':
            return True
        return False

    def _limit_gripper_target(self, width: float, description: str) -> float:
        """Keep gripper commands clear of hard stops that can lock the four-bar."""
        requested = float(width)
        limited = float(np.clip(
            requested,
            self.gripper_safe_lower_limit,
            self.gripper_safe_upper_limit,
        ))
        if not math.isclose(requested, limited, rel_tol=0.0, abs_tol=1e-9):
            self.get_logger().warning(
                f'Clamped {description} gripper target from {requested:.5f} to '
                f'{limited:.5f}; safe range is '
                f'[{self.gripper_safe_lower_limit:.5f}, '
                f'{self.gripper_safe_upper_limit:.5f}].',
            )
        return limited

    def publish_gripper(self, width: float) -> None:
        mode = self.gripper_command_mode
        sent = False
        self._gripper_command_used_action = False
        if mode in ('auto', 'trajectory_action'):
            sent = self.send_gripper_action(width)
            if sent:
                self._gripper_command_used_action = True
                self.get_logger().info(f'Gripper action submitted; waiting for controller acceptance: target={width:.5f}')
        if not sent and mode in ('auto', 'topic'):
            msg = Float64()
            msg.data = width
            self.gripper_pub.publish(msg)
            self.get_logger().info(f'Gripper topic command -> {width:.5f}')

    def send_gripper_action(self, width: float) -> bool:
        if not self.gripper_action_client.wait_for_server(timeout_sec=self.gripper_action_timeout_sec):
            self.get_logger().warning('Gripper action server not available; falling back to topic command.')
            return False
        expected_seq = self.sequence_id
        expected_stage = self.sequence_stage
        expected_target = float(width)
        self._gripper_action_goal_handle = None
        self._gripper_action_accepted = False
        self._gripper_action_succeeded = False
        self._gripper_action_failed_reason = None
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = [self.gripper_joint_name]

        # This controller needs an explicit measured start waypoint followed by
        # the target. A single future waypoint is accepted by Jazzy's JTC but,
        # with this Gazebo/open-loop configuration, remains active indefinitely
        # and never reports success.
        current_pos = self.current_joint_positions.get(self.gripper_joint_name)
        points = []
        if current_pos is not None:
            start_pt = JointTrajectoryPoint()
            start_pt.positions = [float(current_pos)]
            start_pt.velocities = [0.0]
            start_pt.time_from_start = Duration(sec=0, nanosec=0)
            points.append(start_pt)

        end_pt = JointTrajectoryPoint()
        end_pt.positions = [float(width)]
        end_pt.velocities = [0.0]  # explicit zero-velocity at target → smooth stop
        # Keep the action trajectory duration separate from post-close timing.
        # On physical hardware without gripper feedback, the user may want a
        # long hold after closure without slowing the close command itself.
        traj_sec = self.gripper_command_duration_sec
        if traj_sec <= 0.0:
            # Legacy fallback: keep trajectory slightly longer than settle_sec.
            traj_sec = self.gripper_settle_sec + 0.5
        traj_ns = int(traj_sec * 1e9)
        end_pt.time_from_start = Duration(
            sec=traj_ns // 1_000_000_000,
            nanosec=traj_ns % 1_000_000_000,
        )
        points.append(end_pt)
        traj.points = points
        goal.trajectory = traj
        future = self.gripper_action_client.send_goal_async(goal)
        future.add_done_callback(
            lambda fut, seq=expected_seq, stage=expected_stage, target=expected_target:
                self._on_gripper_goal_response(fut, seq, stage, target)
        )
        return True

    def _on_gripper_goal_response(
        self,
        future,
        expected_seq: int,
        expected_stage: str,
        expected_target: float,
    ) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            if expected_seq == self.sequence_id and expected_stage == self.sequence_stage:
                self._gripper_action_failed_reason = f'goal request failed: {exc}'
            return

        if expected_seq != self.sequence_id or expected_stage != self.sequence_stage:
            if goal_handle.accepted:
                try:
                    goal_handle.cancel_goal_async()
                except Exception as exc:
                    self.get_logger().error(f'Could not cancel stale gripper goal: {exc}')
            return

        if not goal_handle.accepted:
            self._gripper_action_failed_reason = (
                f'controller rejected target {expected_target:.5f}'
            )
            self.get_logger().error(f'Gripper action rejected by controller: '
                f'stage={expected_stage}, target={expected_target:.5f}.')
            return

        self._gripper_action_goal_handle = goal_handle
        self._gripper_action_accepted = True
        self.get_logger().info(f'Gripper action accepted by controller: '
            f'stage={expected_stage}, target={expected_target:.5f}.')
        goal_handle.get_result_async().add_done_callback(
            lambda fut, seq=expected_seq, stage=expected_stage, target=expected_target:
                self._on_gripper_goal_result(fut, seq, stage, target)
        )

    def _on_gripper_goal_result(
        self,
        future,
        expected_seq: int,
        expected_stage: str,
        expected_target: float,
    ) -> None:
        if expected_seq != self.sequence_id or expected_stage != self.sequence_stage:
            return
        self._gripper_action_goal_handle = None
        try:
            result_wrap = future.result()
        except Exception as exc:
            self._gripper_action_failed_reason = f'action result unavailable: {exc}'
            return

        result = result_wrap.result
        error_code = int(getattr(result, 'error_code', 0))
        successful_code = int(getattr(FollowJointTrajectory.Result, 'SUCCESSFUL', 0))
        if (
            result_wrap.status == GoalStatus.STATUS_SUCCEEDED
            and error_code == successful_code
        ):
            self._gripper_action_succeeded = True
            self.get_logger().info(f'Gripper controller reported action success: '
                f'stage={expected_stage}, target={expected_target:.5f}.')
            return

        error_string = str(getattr(result, 'error_string', '')).strip()
        self._gripper_action_failed_reason = (
            f'action finished with status={result_wrap.status}, '
            f'error_code={error_code}, error="{error_string}"'
        )
        self.get_logger().error(f'Gripper controller action failed: '
            f'stage={expected_stage}, target={expected_target:.5f}, '
            f'{self._gripper_action_failed_reason}.')

    def command_gripper_and_then(
        self,
        width: float,
        cb: Callable[[], None],
        stage_name: str,
        description: str
    ) -> None:
        """
        Gripper Agent:
        Send a gripper command and continue only when both the minimum command
        time has elapsed and fresh /joint_states feedback confirms the target.
        """
        if (
            self._pending_arm_motion_confirmation is not None
            or self._active_move_goal_handle is not None
            or self._cartesian_plan_in_flight is not None
        ):
            reason = (
                f'Blocked gripper command "{description}" because the arm stage '
                'has not completed action + measured-state confirmation.'
            )
            if self.holding_object:
                self._cancel_active_moveit_goal()
                self._hold_closed_after_transport_failure(reason)
            else:
                self.reset_sequence(reason)
            return
        width = self._limit_gripper_target(width, description)
        self.sequence_stage = stage_name
        self._gripper_wait_target = float(width)
        self._gripper_wait_cb = cb
        self._gripper_wait_start_sec = self._now_sec()
        start_position = self.current_joint_positions.get(self.gripper_joint_name)
        self._gripper_wait_start_position = (
            float(start_position) if start_position is not None else None
        )
        self._gripper_wait_last_position = self._gripper_wait_start_position
        self._gripper_wait_last_motion_sec = self._gripper_wait_start_sec
        self._gripper_wait_seq = self.sequence_id
        self._gripper_wait_stage = stage_name

        self.get_logger().info(f'Gripper command requested ({description}): {width:.5f}')

        self.publish_gripper(width)

        if self._gripper_wait_timer is not None:
            self._gripper_wait_timer.cancel()

        self._gripper_wait_timer = self.create_timer(
            0.05,
            self._gripper_wait_tick
        )

    def _gripper_wait_tick(self) -> None:
        if self._gripper_wait_seq != self.sequence_id:
            if self._gripper_wait_timer is not None:
                self._gripper_wait_timer.cancel()
                self._gripper_wait_timer = None
            self.get_logger().warning('Ignoring stale gripper wait from an old grasp sequence.')
            return

        if self.sequence_stage != self._gripper_wait_stage:
            if self._gripper_wait_timer is not None:
                self._gripper_wait_timer.cancel()
                self._gripper_wait_timer = None
            self.get_logger().warning(f'Ignoring stale gripper wait for stage={self._gripper_wait_stage}; '
                f'current_stage={self.sequence_stage}.')
            return

        target = self._gripper_wait_target
        current = self.current_joint_positions.get(self.gripper_joint_name)
        now_sec = self._now_sec()
        elapsed = now_sec - self._gripper_wait_start_sec
        command_duration_sec = self.gripper_command_duration_sec
        if command_duration_sec <= 0.0:
            command_duration_sec = self.gripper_settle_sec + 0.5

        minimum_completion_sec = max(self.gripper_settle_sec, command_duration_sec)
        if self.sequence_stage == 'release_in_base_box' and self.base_box_release_wait_sec > 0.0:
            minimum_completion_sec = max(minimum_completion_sec, self.base_box_release_wait_sec)

        feedback_stamp = self.current_joint_update_sec.get(self.gripper_joint_name)
        feedback_fresh = (
            current is not None
            and feedback_stamp is not None
            and float(feedback_stamp) >= self._gripper_wait_start_sec
            and (now_sec - float(feedback_stamp)) <= self.gripper_feedback_max_age_sec
        )
        position_reached = (
            feedback_fresh
            and abs(float(current) - target) <= self.gripper_goal_tolerance
        )
        minimum_time_elapsed = elapsed >= minimum_completion_sec

        if current is not None:
            if (
                self._gripper_wait_last_position is None
                or abs(float(current) - self._gripper_wait_last_position)
                > self.gripper_contact_position_epsilon_rad
            ):
                self._gripper_wait_last_motion_sec = now_sec
                self._gripper_wait_last_position = float(current)

        contact_stalled = (
            now_sec - self._gripper_wait_last_motion_sec
        ) >= self.gripper_contact_stall_sec
        contact_confirmed = (
            self.sequence_stage == 'close_gripper'
            and minimum_time_elapsed
            and feedback_fresh
            and current is not None
            and self._gripper_wait_start_position is not None
            and contact_stalled
            and fourbar.plausible_probe_contact(
                self._gripper_wait_start_position,
                float(current),
                float(target),
                self.minimum_probe_width_m,
                self.maximum_probe_width_m,
                target_tolerance_rad=self.gripper_goal_tolerance,
                minimum_closing_travel_rad=self.gripper_contact_min_closing_travel_rad,
                gap_tolerance_m=self.gripper_contact_gap_tolerance_m,
            )
        )

        # A rigid probe is expected to stop an intentionally over-closed final
        # command. Do not wait for the trajectory controller to abort that valid
        # contact: fresh, stationary feedback plus calibrated jaw geometry is a
        # stronger completion signal for this close stage.
        if contact_confirmed:
            actual_gap = fourbar.gap_from_q(float(current))
            self.gripper_contact_detected = True
            if self._gripper_wait_timer is not None:
                self._gripper_wait_timer.cancel()
                self._gripper_wait_timer = None
            cb = self._gripper_wait_cb
            self._gripper_wait_cb = None
            self.last_gripper_actual = float(current)
            self.last_gripper_target = float(target)
            self._cancel_active_gripper_goal()
            self.get_logger().warning(
                'Final close confirmed by fresh stalled-contact feedback: '
                f'target={target:.5f}, actual={current:.5f}, '
                f'jaw_gap={actual_gap*1000.0:.1f}mm, elapsed={elapsed:.2f}s. '
                'Continuing with lift verification while keeping the gripper closed.'
            )
            if cb is not None:
                cb()
            return

        if self._gripper_command_used_action and self._gripper_action_failed_reason is not None:
            reason = (
                'Gripper controller did not execute the command and measured '
                'feedback was not consistent with probe contact: '
                f'{self._gripper_action_failed_reason}. target={target:.5f}'
            )
            self._finish_failed_gripper_wait(reason, current)
            return

        action_complete = (
            not self._gripper_command_used_action
            or not self.gripper_require_action_success_for_completion
            or self._gripper_action_succeeded
        )
        feedback_complete = (
            self.gripper_feedback_available
            and minimum_time_elapsed
            and position_reached
            and action_complete
        )

        # Optional legacy fallback for systems that genuinely have no state
        # feedback. Hardware launch keeps require_feedback=true, so this path is
        # disabled on the rover and time alone can never complete a command.
        no_feedback_close_complete_sec = self.gripper_no_feedback_close_complete_sec
        if self.sequence_stage == 'release_in_base_box' and self.base_box_release_wait_sec > 0.0:
            no_feedback_close_complete_sec = self.base_box_release_wait_sec
        elif no_feedback_close_complete_sec <= 0.0:
            no_feedback_close_complete_sec = command_duration_sec
        legacy_time_complete = (
            not self.gripper_require_feedback_for_completion
            and not self.gripper_feedback_available
            and elapsed >= no_feedback_close_complete_sec
            and action_complete
        )

        if feedback_complete or legacy_time_complete:
            if self._gripper_wait_timer is not None:
                self._gripper_wait_timer.cancel()
                self._gripper_wait_timer = None

            cb = self._gripper_wait_cb
            self._gripper_wait_cb = None
            self.last_gripper_actual = float(current) if current is not None else None
            self.last_gripper_target = float(target)

            if feedback_complete:
                self.get_logger().info(f'Gripper command confirmed by time + fresh joint feedback: '
                    f'target={target:.5f}, actual={current:.5f}, elapsed={elapsed:.2f}s, '
                    f'minimum_time={minimum_completion_sec:.2f}s, '
                    f'action_success={self._gripper_action_succeeded}.')
            else:
                self.get_logger().warning(f'Legacy open-loop gripper completion: target={target:.5f}, '
                    f'elapsed={elapsed:.2f}s. This mode is disabled by default.')

            if cb is not None:
                cb()
            return

        effective_timeout_sec = max(
            self.gripper_confirm_timeout_sec,
            minimum_completion_sec + 0.5,
        )
        if elapsed < effective_timeout_sec:
            return

        if self._gripper_wait_timer is not None:
            self._gripper_wait_timer.cancel()
            self._gripper_wait_timer = None
        cb = self._gripper_wait_cb
        self._gripper_wait_cb = None
        self.last_gripper_actual = float(current) if current is not None else None
        self.last_gripper_target = float(target)

        feedback_detail = (
            f'actual={current}, fresh={feedback_fresh}, position_reached={position_reached}, '
            f'action_accepted={self._gripper_action_accepted}, '
            f'action_succeeded={self._gripper_action_succeeded}, '
            f'elapsed={elapsed:.2f}s, required_time={minimum_completion_sec:.2f}s'
        )

        if self.sequence_stage == 'release_in_base_box':
            self._stop_after_uncertain_base_box_release(
                f'Gripper failed to confirm release in the base box: target={target:.5f}, {feedback_detail}.'
            )
            return

        if self.sequence_stage in ('open_gripper', 'retry_open_gripper'):
            self.reset_sequence(
                f'Gripper failed to confirm open: target={target:.5f}, {feedback_detail}.'
            )
            return

        if self.sequence_stage == 'close_gripper':
            self._hold_closed_after_failed_grasp_check(
                f'Final gripper close was not confirmed: target={target:.5f}, {feedback_detail}.'
            )
            return

        self.reset_sequence(
            f'Gripper command was not confirmed at stage={self.sequence_stage}: '
            f'target={target:.5f}, {feedback_detail}.'
        )

    def _finish_failed_gripper_wait(
        self,
        reason: str,
        current: Optional[float],
    ) -> None:
        """Stop immediately when the gripper controller rejects/aborts a command."""
        if self._gripper_wait_timer is not None:
            self._gripper_wait_timer.cancel()
            self._gripper_wait_timer = None
        self._gripper_wait_cb = None
        self.last_gripper_actual = float(current) if current is not None else None
        self.last_gripper_target = float(self._gripper_wait_target)

        if self.sequence_stage == 'release_in_base_box':
            self._stop_after_uncertain_base_box_release(reason)
        elif self.sequence_stage in ('open_gripper', 'retry_open_gripper'):
            self.reset_sequence(reason)
        elif self.sequence_stage == 'close_gripper':
            self._hold_closed_after_failed_grasp_check(reason)
        else:
            self.reset_sequence(reason)

    def publish_debug_image(self, img: np.ndarray) -> None:
        try:
            self.det_vis_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        except Exception as exc:
            self.get_logger().warning(f'Failed to publish detection image: {exc}')

    def _annotate_yolo_results(self, img: np.ndarray, results) -> np.ndarray:
        annotated = img
        for result in results:
            try:
                annotated = result.plot()
                break
            except Exception:
                pass
        return annotated

    def _stamp_debug_status(
        self,
        annotated: np.ndarray,
        status_text: str,
        color: Tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        cv2.putText(
            annotated,
            status_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    def get_depth_roi_median(
        self,
        u: int,
        v: int,
        half_size_px: Optional[int] = None,
        min_depth_m: Optional[float] = None,
        max_depth_m: Optional[float] = None,
        expected_depth_m: Optional[float] = None,
        depth_band_m: Optional[float] = None,
        prefer_nearest: bool = False,
        depth_image: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Depth Agent:
        Read a robust depth value around a pixel.

        At close range the bbox center can land on an invalid pixel,
        gripper finger, or floor/background. This version supports
        a larger ROI and optional filtering around the predicted depth
        of the locked target.

        ``depth_image`` selects the frame to sample (a snapshot paired with a
        detection); it defaults to the latest received depth image.
        """
        depth = depth_image if depth_image is not None else self.latest_depth
        if depth is None:
            return None

        h, w = depth.shape[:2]

        if u < 0 or v < 0 or u >= w or v >= h:
            return None

        hs = int(self.roi_half_size_px if half_size_px is None else half_size_px)
        min_d = float(self.min_depth_m if min_depth_m is None else min_depth_m)
        max_d = float(self.max_depth_m if max_depth_m is None else max_depth_m)

        x0, x1 = max(0, u - hs), min(w, u + hs + 1)
        y0, y1 = max(0, v - hs), min(h, v + hs + 1)

        roi = depth[y0:y1, x0:x1]

        valid = roi[np.isfinite(roi) & (roi > min_d) & (roi < max_d)]

        if valid.size == 0:
            return None

        # When we know approximately where the locked target should be in depth,
        # ignore gripper/finger/background points too far from that expected plane.
        if expected_depth_m is not None and depth_band_m is not None:
            lo = max(min_d, float(expected_depth_m) - float(depth_band_m))
            hi = min(max_d, float(expected_depth_m) + float(depth_band_m))
            band_valid = valid[(valid >= lo) & (valid <= hi)]

            if band_valid.size > 0:
                valid = band_valid

        if prefer_nearest:
            # For a thin probe, median can select the floor/background.
            # 25th percentile tracks the closer object surface without using noisy min().
            return float(np.percentile(valid, 25.0))

        return float(np.median(valid))

    def pixel_to_point_camera(self, u: int, v: int, depth: float) -> Optional[np.ndarray]:
        if self.camera_info is None:
            return None
        fx, fy = float(self.camera_info.k[0]), float(self.camera_info.k[4])
        cx, cy = float(self.camera_info.k[2]), float(self.camera_info.k[5])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None
        return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=np.float64)

    # ------------------------------------------------------------------
    # Layer 0 — Object Dimension Estimator Agent
    # ------------------------------------------------------------------

    def _estimate_object_width_3d(
        self,
        mask_bool: np.ndarray,
        depth_image: np.ndarray,
    ) -> Optional[float]:
        """
        Object Width Estimator Agent.

        Estimates the smallest 3D bounding dimension of the detected object
        from its segmentation mask and depth image.

        The gripper must span the shortest cross-section (diameter, not length),
        so we return min(horizontal_3d_extent, vertical_3d_extent).  A lower
        percentile depth (35th) selects the near object surface and avoids
        background / floor contamination.
        """
        if self.camera_info is None:
            return None

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None

        h_img, w_img = depth_image.shape[:2]
        if mask_bool.shape[0] != h_img or mask_bool.shape[1] != w_img:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (w_img, h_img),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        ys, xs = np.where(mask_bool)
        if xs.size < 10:
            return None

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        if (x_max - x_min) < 2 or (y_max - y_min) < 2:
            return None

        # Depth: near-surface percentile avoids background / floor.
        depths = depth_image[ys, xs]
        valid_d = depths[
            np.isfinite(depths)
            & (depths > self.min_depth_m)
            & (depths < self.max_depth_m)
        ]
        if valid_d.size < 5:
            return None
        depth_val = float(np.percentile(valid_d, 35.0))

        # Convert 2D pixel extents to 3D metric extents via perspective projection.
        width_3d = (x_max - x_min) * depth_val / fx
        height_3d = (y_max - y_min) * depth_val / fy

        # Return the smaller dimension: for a probe viewed top-down, horizontal
        # extent is the probe length (large) and vertical extent is the diameter (small).
        return float(min(width_3d, height_3d))

    # ------------------------------------------------------------------
    # Layer 1 — Gripper Sizing Agent
    # ------------------------------------------------------------------

    def _sanitize_probe_width_for_grasp(self, object_width_m: float) -> float:
        """Return a physically plausible probe width for gripper sizing.

        The mask width estimator can accidentally measure the probe length, the
        diagonal of the mask, or a floor-contaminated blob when the wrist camera
        is close. For this task the probe width is known, so impossible values
        must not control the final q_close.
        """
        raw = float(object_width_m)
        if not self.clamp_probe_width_for_grasp:
            return max(raw, self.minimum_probe_width_m)

        if raw < self.minimum_probe_width_m:
            return self.minimum_probe_width_m

        if raw > self.maximum_probe_width_m:
            self.get_logger().warning(f'Detected width {raw*1000:.1f} mm is above physical probe max '
                f'{self.maximum_probe_width_m*1000:.1f} mm; using nominal '
                f'{self.nominal_probe_width_m*1000:.1f} mm for q_close.', throttle_duration_sec=1.0)
            return float(np.clip(self.nominal_probe_width_m, self.minimum_probe_width_m, self.maximum_probe_width_m))

        return raw

    def _fourbar_q_from_actual_gap(self, gap_m: float) -> float:
        """Desired jaw gap -> joint q, clamped to this task's safe q window."""
        q = fourbar.q_from_gap(gap_m)
        q = float(np.clip(q, self.gripper_open, self.gripper_close))
        q = float(np.clip(q, self.fourbar_q_min_for_floor_grasp, self.fourbar_q_max_for_floor_grasp))
        return q

    def _fourbar_actual_contact_offset(self, q: float) -> np.ndarray:
        """arm_gripper_base_link -> object-centre offset from true bucket midpoint."""
        return fourbar.contact_offset(q, self.fourbar_contact_y_offset_m)

    def _compute_adaptive_gripper_close(self, object_width_m: float) -> Tuple[float, float]:
        """
        Layer — Actual Four-Bar Jaw-Gap Agent.

        Uses the true URDF/STL four-bar gap curve (aries_vision_grasp.fourbar).
        This is essential: for the real gripper, q≈+0.07 rad is almost fully
        closed, not a 45 mm gap.  A 45 mm probe needs q≈-0.20 rad.
        """
        object_width_eff = self._sanitize_probe_width_for_grasp(float(object_width_m))
        final_gap = max(object_width_eff + self.object_width_final_clearance_m, 0.006)
        preclose_gap = max(object_width_eff + self.object_width_preclose_clearance_m, final_gap + 0.002)

        q_close = self._fourbar_q_from_actual_gap(final_gap)
        q_preclose = self._fourbar_q_from_actual_gap(preclose_gap)
        # Ensure preclose is more open than close.
        if q_preclose > q_close - self.preclose_min_q_margin_rad:
            q_preclose = max(self.gripper_open, q_close - self.preclose_min_q_margin_rad)
        actual_final_gap = fourbar.gap_from_q(q_close)
        actual_pre_gap = fourbar.gap_from_q(q_preclose)
        self.get_logger().info(f'Actual four-bar sizing: object_width={object_width_m*1000:.1f} mm  '
            f'used_width={object_width_eff*1000:.1f} mm  '
            f'target_final_gap={final_gap*1000:.1f} mm  '
            f'q_close={q_close:.4f} rad -> actual_gap={actual_final_gap*1000:.1f} mm  '
            f'q_preclose={q_preclose:.4f} rad -> actual_pre_gap={actual_pre_gap*1000:.1f} mm')
        return float(q_close), float(q_preclose)

    def _apply_fourbar_contact_offset(self, q_close: float) -> None:
        """Set the effective link->contact offset for the selected close angle."""
        off = self._fourbar_actual_contact_offset(float(q_close))
        self.effective_target_point_offset_in_link = [float(off[0]), float(off[1]), float(off[2])]
        self.get_logger().info(f'Actual 4-bar contact offset at q={q_close:.4f}: '
            f'x={off[0]*1000:.1f} y={off[1]*1000:.1f} z={off[2]*1000:.1f} mm  '
            f'actual_gap={fourbar.gap_from_q(q_close)*1000:.1f} mm')

    def _estimate_object_orientation_3d(
        self,
        mask_bool: np.ndarray,
        depth_image: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Layer 0.6 — 6D Pose Estimator Agent (orientation component).

        Back-projects every masked pixel to a 3D point in the camera frame using
        depth + camera intrinsics, then runs PCA (eigendecomposition of the
        covariance matrix) to find the principal axes of the point cloud.

        Returns a tuple `(centroid_cam, R_obj_cam)` where `centroid_cam` is the
        3D centroid of the masked cloud in camera frame and `R_obj_cam` is a 3×3
        rotation matrix whose columns, in camera frame, are:
            col 0: long axis  — direction of maximum variance (e.g. probe length)
            col 1: short axis — direction of medium variance  (e.g. probe diameter)
            col 2: normal     — direction of minimum variance (approx depth axis)

        Returns None when:
          - camera_info is unavailable
          - fewer than 20 valid 3D points in the mask
          - the object appears too round (eigenvalue ratio < object_orientation_min_eigenratio)
            meaning orientation cannot be reliably determined from shape alone
        """
        if self.camera_info is None:
            return None
        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None

        h_img, w_img = depth_image.shape[:2]
        if mask_bool.shape[0] != h_img or mask_bool.shape[1] != w_img:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (w_img, h_img),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        ys, xs = np.where(mask_bool)
        if xs.size < 20:
            return None

        depths = depth_image[ys, xs]
        valid = (
            np.isfinite(depths)
            & (depths > self.min_depth_m)
            & (depths < self.max_depth_m)
        )
        if int(valid.sum()) < 20:
            return None

        d_v = depths[valid].astype(np.float64)
        u_v = xs[valid].astype(np.float64)
        v_v = ys[valid].astype(np.float64)

        # Back-project to 3D camera frame.
        X = (u_v - cx) * d_v / fx
        Y = (v_v - cy) * d_v / fy
        Z = d_v
        pts_all = np.column_stack([X, Y, Z])      # N × 3

        centroid = pts_all.mean(axis=0)

        # Deterministic down-sampling keeps the pose estimate stable across
        # frames; random sampling adds visible jitter to 6D tracking.
        pts = pts_all
        if len(pts_all) > 1000:
            idx = np.linspace(0, len(pts_all) - 1, 1000, dtype=np.int32)
            pts = pts_all[idx]

        centered = pts - centroid
        cov = (centered.T @ centered) / len(centered)

        # eigh returns eigenvalues in ascending order; flip to descending
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]    # columns = eigenvectors

        # Reject if the shape is too symmetric to determine orientation reliably
        if eigenvalues[1] < 1e-9:
            return None
        ratio = float(eigenvalues[0] / eigenvalues[1])
        if ratio < self.object_orientation_min_eigenratio:
            self.get_logger().info(f'Object orientation skipped: eigenratio={ratio:.1f} '
                f'< min={self.object_orientation_min_eigenratio:.1f} '
                f'(object too round to determine orientation reliably)', throttle_duration_sec=1.0)
            return None

        # Ensure right-handed rotation matrix
        R = eigenvectors.copy()
        if np.linalg.det(R) < 0:
            R[:, 2] = -R[:, 2]

        self.get_logger().info(f'Object 3D orientation: eigenratio={ratio:.1f}  '
            f'long_axis_cam=[{R[0,0]:.2f},{R[1,0]:.2f},{R[2,0]:.2f}]', throttle_duration_sec=1.0)
        return centroid, R

    def _clear_detected_object_pose(self) -> None:
        self.detected_object_pose = None
        self._last_detected_object_rotation_base = None

    def _compute_object_pose_in_planning_frame(
        self,
        centroid_cam: np.ndarray,
        R_obj_cam: np.ndarray,
        depth_frame: Optional[str] = None,
        stamp: Optional[rclpy.time.Time] = None,
    ) -> Optional[Tuple[PoseStamped, np.ndarray]]:
        depth_frame = depth_frame or self.latest_depth_frame
        if depth_frame is None:
            return None

        centroid_base = self.transform_point(
            np.array(centroid_cam, dtype=np.float64),
            depth_frame,
            self.planning_frame,
            stamp=stamp,
        )
        if centroid_base is None:
            return None

        try:
            tfm = self.tf_buffer.lookup_transform(
                self.planning_frame,
                depth_frame,
                stamp if stamp is not None else rclpy.time.Time(),
            )
        except TransformException:
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self.planning_frame,
                    depth_frame,
                    rclpy.time.Time(),
                )
            except TransformException as exc:
                self.get_logger().warning(f'TF lookup for object pose failed: {exc}')
                return None

        R_tf = quat_to_matrix(tfm.transform.rotation)
        long_axis = normalize(R_tf @ R_obj_cam[:, 0].reshape(3,))
        normal_axis = normalize(R_tf @ R_obj_cam[:, 2].reshape(3,))

        prev_R = self._last_detected_object_rotation_base
        if prev_R is not None:
            if float(np.dot(long_axis, prev_R[:, 0])) < 0.0:
                long_axis = -long_axis
            if float(np.dot(normal_axis, prev_R[:, 2])) < 0.0:
                normal_axis = -normal_axis
        elif float(normal_axis[2]) < 0.0:
            normal_axis = -normal_axis

        short_axis = np.cross(normal_axis, long_axis)
        if float(np.linalg.norm(short_axis)) < 1e-9:
            short_axis = R_tf @ R_obj_cam[:, 1].reshape(3,)
        short_axis = normalize(short_axis)
        normal_axis = normalize(np.cross(long_axis, short_axis))

        R_obj_base = np.column_stack([long_axis, short_axis, normal_axis])
        if np.linalg.det(R_obj_base) < 0.0:
            R_obj_base[:, 1] = -R_obj_base[:, 1]

        pose = self.make_pose(centroid_base, matrix_to_quat(R_obj_base))
        return pose, R_obj_base

    def _update_detected_object_pose_from_camera(
        self,
        centroid_cam: np.ndarray,
        R_obj_cam: np.ndarray,
        depth_frame: Optional[str] = None,
        stamp: Optional[rclpy.time.Time] = None,
    ) -> None:
        pose_result = self._compute_object_pose_in_planning_frame(
            centroid_cam, R_obj_cam, depth_frame=depth_frame, stamp=stamp
        )
        if pose_result is None:
            self._clear_detected_object_pose()
            return

        pose_msg, R_obj_base = pose_result
        self.detected_object_pose = pose_msg
        self._last_detected_object_rotation_base = R_obj_base

        if self.publish_object_pose_enabled:
            self.object_pose_pub.publish(pose_msg)

        p = pose_msg.pose.position
        self.get_logger().info(f'6D object pose tracked: '
            f'x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}', throttle_duration_sec=1.0)

    def _compute_grasp_yaw_from_object(
        self,
        R_obj_cam: np.ndarray,
    ) -> Optional[float]:
        """
        Object Yaw Extractor Agent.

        Transforms the object's long axis (column 0 of R_obj_cam) from camera
        frame to planning_frame using TF2 (rotation only — not translation,
        because this is a direction vector, not a position).

        Returns: gripper yaw = atan2(long_y, long_x) + object_yaw_rotation_offset_deg,
        but resolved through the gripper's 180° symmetry so the wrist keeps the
        object alignment while choosing the smaller yaw rotation.
        """
        long_axis_base = None
        if self._last_detected_object_rotation_base is not None:
            long_axis_base = self._last_detected_object_rotation_base[:, 0].reshape(3,)

        if long_axis_base is None:
            if self.latest_depth_frame is None:
                return None

            long_axis_cam = R_obj_cam[:, 0].reshape(3,)  # direction vector

            # Apply rotation only (direction vector — no translation component)
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self.planning_frame,
                    self.latest_depth_frame,
                    rclpy.time.Time(),
                )
            except TransformException as exc:
                self.get_logger().warning(f'TF lookup for object yaw failed: {exc}')
                return None

            R_tf = quat_to_matrix(tfm.transform.rotation)
            long_axis_base = R_tf @ long_axis_cam

        yaw_obj = math.atan2(float(long_axis_base[1]), float(long_axis_base[0]))
        offset_rad = math.radians(self.object_yaw_rotation_offset_deg)
        yaw_gripper_raw = wrap_to_pi(yaw_obj + offset_rad)

        reference_yaw = self.fixed_yaw
        cur = self.get_current_tool_orientation_in_planning_frame()
        if cur is not None:
            R_cur = quat_to_matrix(cur)
            reference_yaw = math.atan2(R_cur[1, 0], R_cur[0, 0])

        yaw_candidates = [
            wrap_to_pi(yaw_gripper_raw - math.pi),
            yaw_gripper_raw,
            wrap_to_pi(yaw_gripper_raw + math.pi),
        ]
        yaw_gripper = min(
            yaw_candidates,
            key=lambda yaw: abs(wrap_to_pi(yaw - reference_yaw)),
        )

        self.get_logger().info(f'Object yaw in {self.planning_frame}: {math.degrees(yaw_obj):.1f}°  '
            f'Gripper yaw (+ {self.object_yaw_rotation_offset_deg:.0f}° offset, nearest symmetric): '
            f'{math.degrees(yaw_gripper):.1f}°')
        return float(yaw_gripper)

    def _get_detected_object_long_axis_base(self) -> Optional[np.ndarray]:
        if self._last_detected_object_rotation_base is None:
            return None

        axis = self._last_detected_object_rotation_base[:, 0].reshape(3,)
        if float(np.linalg.norm(axis)) < 1e-9:
            return None
        return normalize(axis)

    def _get_probe_reference_long_axis_base(self) -> Optional[np.ndarray]:
        locked_axis = self.sequence_locked_object_long_axis_base
        live_axis = self._get_detected_object_long_axis_base()

        if locked_axis is not None and float(np.linalg.norm(locked_axis)) >= 1e-9:
            locked_axis = normalize(np.array(locked_axis, dtype=np.float64))
        else:
            locked_axis = None

        if live_axis is not None and locked_axis is not None:
            if float(np.dot(live_axis, locked_axis)) < 0.0:
                live_axis = -live_axis

        return live_axis if live_axis is not None else locked_axis

    def _apply_probe_shape_aware_target_correction(
        self,
        point_base: np.ndarray,
        reference_target: Optional[np.ndarray],
        reason: str,
    ) -> np.ndarray:
        if not self.probe_shape_aware_center_enabled or self.target_class != 'probe':
            return point_base

        ref = reference_target
        if ref is None:
            ref = self.sequence_locked_target_point_base
        if ref is None:
            ref = self.current_target_point_base
        if ref is None:
            return point_base

        axis = self._get_probe_reference_long_axis_base()
        if axis is None:
            return point_base

        ref = np.array(ref, dtype=np.float64)
        candidate = np.array(point_base, dtype=np.float64)
        delta = candidate - ref
        parallel_mag = float(np.dot(delta, axis))
        parallel = axis * parallel_mag
        perpendicular = delta - parallel
        parallel_scale = float(np.clip(self.probe_parallel_center_update_scale, 0.0, 1.0))
        corrected = ref + perpendicular + parallel_scale * parallel

        if abs(parallel_mag) > 0.010 and abs(parallel_mag) > 1.5 * float(np.linalg.norm(perpendicular)):
            self.get_logger().info(f'Probe shape-aware center hold during {reason}: '
                f'parallel_drift={parallel_mag:.3f}m, '
                f'perpendicular={float(np.linalg.norm(perpendicular)):.3f}m, '
                f'parallel_scale={parallel_scale:.2f}.', throttle_duration_sec=0.7)

        return corrected

    def camera_point_to_pixel(self, point_cam: np.ndarray) -> Optional[Tuple[int, int]]:
        if self.camera_info is None:
            return None

        z = float(point_cam[2])
        if z <= 1e-6:
            return None

        fx, fy = float(self.camera_info.k[0]), float(self.camera_info.k[4])
        cx, cy = float(self.camera_info.k[2]), float(self.camera_info.k[5])

        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None

        u = int(round((float(point_cam[0]) * fx / z) + cx))
        v = int(round((float(point_cam[1]) * fy / z) + cy))

        if self.latest_depth is not None:
            h, w = self.latest_depth.shape[:2]
        elif self.camera_info is not None:
            w, h = int(self.camera_info.width), int(self.camera_info.height)
        else:
            return None

        if u < 0 or v < 0 or u >= w or v >= h:
            return None

        return u, v

    def projected_locked_target_refinement(self) -> Optional[np.ndarray]:
        """
        Close-Range Tracking Agent:
        If YOLO loses the probe close to the gripper, project the already locked
        3D target into the current camera image and refine it with local depth.

        This keeps tracking when the probe is too close, too large, or partially
        hidden for YOLO.
        """
        if (
            self.current_target_point_base is None
            or self.latest_depth is None
            or self.latest_depth_frame is None
            or self.camera_info is None
        ):
            return None

        pred_cam = self.transform_point(
            self.current_target_point_base,
            self.planning_frame,
            self.latest_depth_frame,
        )

        if pred_cam is None or float(pred_cam[2]) <= 0.0:
            return None

        pix = self.camera_point_to_pixel(pred_cam)

        if pix is None:
            self.get_logger().warning('Projection fallback: locked target is outside the close camera image.', throttle_duration_sec=0.7)
            return None

        u, v = pix

        depth = self.get_depth_roi_median(
            u,
            v,
            half_size_px=self.refine_projection_roi_half_size_px,
            min_depth_m=self.refine_min_depth_m,
            max_depth_m=self.max_depth_m,
            expected_depth_m=float(pred_cam[2]),
            depth_band_m=self.refine_depth_band_m,
            prefer_nearest=True,
        )

        if depth is None:
            self.get_logger().warning(f'Projection fallback: no valid local depth around u={u}, v={v}. '
                f'Predicted depth={pred_cam[2]:.3f}m. Using locked target if refinement times out.', throttle_duration_sec=0.7)
            return None

        point_cam = self.pixel_to_point_camera(u, v, depth)

        if point_cam is None:
            return None

        point_base = self._camera_grasp_target_to_planning_frame(
            point_cam,
            self.latest_depth_frame,
        )

        if point_base is None:
            return None

        self.get_logger().info(f'Projection fallback sample: u={u} v={v} depth={depth:.3f} '
            f'base=({point_base[0]:.3f},{point_base[1]:.3f},{point_base[2]:.3f})', throttle_duration_sec=0.5)

        return point_base

    def transform_point(
        self,
        point_xyz: np.ndarray,
        source_frame: str,
        target_frame: str,
        stamp: Optional[rclpy.time.Time] = None,
    ) -> Optional[np.ndarray]:
        """Transform a point, preferring TF at the sensor stamp.

        On a moving wrist camera, using the latest TF for an older image skews
        the 3D target by however far the camera moved since the frame was
        captured. When the exact stamp is not yet available in the buffer,
        fall back to the latest transform rather than dropping the detection.
        """
        tfm = self._lookup_transform(source_frame, target_frame, stamp)
        if tfm is None:
            return None
        q = tfm.transform.rotation
        t = tfm.transform.translation
        return quat_to_matrix(q) @ point_xyz.reshape(3,) + np.array([t.x, t.y, t.z], dtype=np.float64)

    def _lookup_transform(
        self,
        source_frame: str,
        target_frame: str,
        stamp: Optional[rclpy.time.Time] = None,
    ):
        """Look up target<-source TF with the sensor-stamp fallback policy."""
        tfm = None
        if stamp is not None:
            try:
                tfm = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
            except TransformException:
                self.get_logger().warning(
                    f'TF at image stamp unavailable for {source_frame} -> {target_frame}; '
                    'using latest transform for this detection.',
                    throttle_duration_sec=5.0,
                )
        if tfm is None:
            try:
                tfm = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except TransformException as exc:
                self.get_logger().warning(f'TF lookup failed {source_frame} -> {target_frame}: {exc}')
                return None
        return tfm

    def _camera_grasp_target_to_planning_frame(
        self,
        point_camera: np.ndarray,
        camera_frame: str,
        stamp: Optional[rclpy.time.Time] = None,
    ) -> Optional[np.ndarray]:
        """Apply camera-axis grasp calibration, then transform to the planning frame."""
        tfm = self._lookup_transform(camera_frame, self.planning_frame, stamp)
        if tfm is None:
            return None
        rotation = quat_to_matrix(tfm.transform.rotation)
        translation = np.array([
            float(tfm.transform.translation.x),
            float(tfm.transform.translation.y),
            float(tfm.transform.translation.z),
        ], dtype=np.float64)
        point_camera = np.array(point_camera, dtype=np.float64).reshape(3,)
        raw_world = rotation @ point_camera + translation
        self._record_camera_offset_calibration_sample(raw_world, rotation)
        return raw_world + rotation @ self.grasp_target_offset_in_camera

    def _record_camera_offset_calibration_sample(
        self,
        raw_world: np.ndarray,
        rotation_world_camera: np.ndarray,
    ) -> None:
        """Build a guarded multi-view estimate while the probe is stationary."""
        if (
            not self.auto_calibrate_camera_offset_enabled
            or self.holding_object
            or self.task_complete
            or self.sequence_stage not in ('idle', 'open_gripper', 'move_pre_grasp')
            or abs(self.last_rover_linear_speed) > self.rover_motion_linear_threshold_mps
            or abs(self.last_rover_angular_speed) > self.rover_motion_angular_threshold_radps
        ):
            return

        raw_world = np.asarray(raw_world, dtype=np.float64).reshape(3,)
        rotation_world_camera = np.asarray(
            rotation_world_camera, dtype=np.float64
        ).reshape(3, 3)
        self._camera_calibration_raw_world.append(raw_world.copy())
        self._camera_calibration_rotations.append(rotation_world_camera.copy())
        self._camera_calibration_last_raw_world = raw_world.copy()
        self._camera_calibration_last_rotation = rotation_world_camera.copy()

        if len(self._camera_calibration_raw_world) < self.auto_calibrate_camera_offset_min_samples:
            return
        estimate = estimate_stationary_target_camera_offset(
            np.stack(self._camera_calibration_raw_world),
            np.stack(self._camera_calibration_rotations),
        )
        if estimate is None:
            return
        improvement = estimate.raw_rms_m - estimate.corrected_rms_m
        offset_norm = float(np.linalg.norm(estimate.offset_camera))
        accepted = (
            estimate.rotation_span_rad >= self.auto_calibrate_camera_offset_min_rotation_rad
            and estimate.condition_number <= self.auto_calibrate_camera_offset_max_condition
            and offset_norm <= self.auto_calibrate_camera_offset_max_m
            and estimate.corrected_rms_m <= self.auto_calibrate_camera_offset_max_rms_m
            and improvement >= self.auto_calibrate_camera_offset_min_improvement_m
        )
        if not accepted:
            self.get_logger().info(
                'Camera-offset auto-calibration not yet trustworthy: '
                f'samples={len(self._camera_calibration_raw_world)}, '
                f'rotation_span={math.degrees(estimate.rotation_span_rad):.1f}deg, '
                f'condition={estimate.condition_number:.1f}, '
                f'offset_norm={offset_norm*1000.0:.1f}mm, '
                f'raw_rms={estimate.raw_rms_m*1000.0:.1f}mm, '
                f'corrected_rms={estimate.corrected_rms_m*1000.0:.1f}mm.',
                throttle_duration_sec=2.0,
            )
            return
        self._pending_camera_offset_estimate = estimate

    def _commit_pending_camera_offset_calibration(self) -> bool:
        """Apply one bounded calibration step immediately before finalization."""
        estimate = self._pending_camera_offset_estimate
        if (
            estimate is None
            or self._auto_camera_calibration_applied_for_sequence
            or self._camera_calibration_last_raw_world is None
            or self._camera_calibration_last_rotation is None
        ):
            return False

        desired = np.asarray(estimate.offset_camera, dtype=np.float64)
        delta = desired - self.grasp_target_offset_in_camera
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.auto_calibrate_camera_offset_max_step_m:
            delta *= self.auto_calibrate_camera_offset_max_step_m / delta_norm
        applied = self.grasp_target_offset_in_camera + delta
        corrected_target = (
            self._camera_calibration_last_raw_world
            + self._camera_calibration_last_rotation @ applied
        )

        self.grasp_target_offset_in_camera = applied
        self.current_target_point_base = corrected_target.copy()
        self.live_target_point_base = corrected_target.copy()
        self.sequence_locked_target_point_base = corrected_target.copy()
        self.live_target_stamp_sec = self._now_sec()
        self._auto_camera_calibration_applied_for_sequence = True
        self._pending_camera_offset_estimate = None
        self.get_logger().warning(
            'Applied bounded automatic camera grasp calibration: '
            f'xyz=({applied[0]:.4f}, {applied[1]:.4f}, {applied[2]:.4f}) m, '
            f'estimated=({desired[0]:.4f}, {desired[1]:.4f}, {desired[2]:.4f}) m, '
            f'corrected_rms={estimate.corrected_rms_m*1000.0:.1f} mm. '
            'To persist after restart, copy the applied xyz into '
            'grasp_target_offset_camera_xyz_m in pick_place.yaml.'
        )
        return True

    def get_current_tool_orientation_in_planning_frame(self) -> Optional[Quaternion]:
        try:
            tfm = self.tf_buffer.lookup_transform(self.planning_frame, self.planning_link, rclpy.time.Time())
            return tfm.transform.rotation
        except TransformException as exc:
            self.get_logger().warning(f'Could not read current tool transform: {exc}')
            return None

    def get_current_link_pose_in_planning_frame(self) -> Optional[Pose]:
        """Return the actual current arm_gripper_base_link pose from TF.

        This is used after final gripper close.  At that moment we must lift
        from the real current link pose, not from an old planned contact pose,
        otherwise the post-grasp lift can create a lateral correction that looks
        like the arm is pulling the probe away.
        """
        try:
            tfm = self.tf_buffer.lookup_transform(self.planning_frame, self.planning_link, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warning(f'Could not read current link pose for post-grasp lift: {exc}')
            return None

        pose = Pose()
        pose.position = Point(
            x=float(tfm.transform.translation.x),
            y=float(tfm.transform.translation.y),
            z=float(tfm.transform.translation.z),
        )
        pose.orientation = tfm.transform.rotation
        return pose

    def _make_current_robot_state(self, arm_joints_only: bool = False) -> Optional[RobotState]:
        if not self.current_joint_positions:
            return None

        seed_names: List[str] = []
        if arm_joints_only:
            for joint_names in (self.pick_home_joint_names, self.retreat_home_joint_names):
                for name in joint_names:
                    if name in self.current_joint_positions and name not in seed_names:
                        seed_names.append(name)

        if not seed_names:
            seed_names = list(self.current_joint_positions.keys())

        if not seed_names:
            return None

        rs = RobotState()
        rs.is_diff = True
        seed_js = JointState()
        seed_js.name = seed_names
        seed_js.position = [self.current_joint_positions[name] for name in seed_names]
        rs.joint_state = seed_js
        return rs

    # ------------------------------------------------------------------ #
    #  Collision-world management for post-grasp transport                #
    # ------------------------------------------------------------------ #

    def _add_collision_floor(self, floor_z: float) -> None:
        """Publish a wide floor plane into the MoveIt collision world.

        The plane sits 10 mm *below* the given floor_z so that the current
        gripper position (at floor level) is NOT inside the object.  This
        stops MoveGroup from planning paths that drag the arm back down into
        the floor.
        """
        obj = CollisionObject()
        obj.header.frame_id = self.planning_frame
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = 'post_grasp_floor'
        obj.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [4.0, 4.0, 0.02]   # 4 m × 4 m × 20 mm slab

        box_pose = Pose()
        box_pose.position.x = 0.0
        box_pose.position.y = 0.0
        # Centre the slab so its top face is at floor_z - 0.010
        box_pose.position.z = floor_z - 0.020
        box_pose.orientation.w = 1.0

        obj.primitives = [box]
        obj.primitive_poses = [box_pose]
        self._collision_object_pub.publish(obj)
        self._post_grasp_floor_active = True
        self.get_logger().info(f'[CollisionWorld] Added floor plane at z={floor_z:.4f} m '
            f'(slab top at z={(floor_z - 0.010):.4f} m)')

    # ------------------------------------------------------------------ #
    #  Probe-STL helpers                                                  #
    # ------------------------------------------------------------------ #

    def _find_probe_stl(self) -> Optional[str]:
        """Return the path to probe.stl, searching known locations."""
        candidates = []
        try:
            candidates.append(
                os.path.join(get_package_share_directory('aries'), 'models', 'probe.stl')
            )
        except PackageNotFoundError:
            pass

        candidates = [
            *candidates,
            str(Path(__file__).resolve().parent / '../../../aries/models/probe.stl'),
            str(Path(__file__).resolve().parent / '../../aries/models/probe.stl'),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_stl_mesh(self, path: str) -> Optional[Mesh]:
        """Read a binary STL file and return a shape_msgs/Mesh (vertices deduplicated)."""
        import struct
        try:
            with open(path, 'rb') as f:
                f.read(80)                          # 80-byte header
                n_tris = struct.unpack('<I', f.read(4))[0]
                verts: List[Tuple[float, float, float]] = []
                tris: List[Tuple[int, int, int]] = []
                vi: dict = {}
                for _ in range(n_tris):
                    f.read(12)                      # face normal (ignored)
                    t_idx = []
                    for _ in range(3):
                        xyz = struct.unpack('<fff', f.read(12))
                        key = (round(xyz[0], 7), round(xyz[1], 7), round(xyz[2], 7))
                        if key not in vi:
                            vi[key] = len(verts)
                            verts.append(key)
                        t_idx.append(vi[key])
                    tris.append((t_idx[0], t_idx[1], t_idx[2]))
                    f.read(2)                       # attribute byte count
            mesh = Mesh()
            for v in verts:
                pt = Point()
                pt.x, pt.y, pt.z = float(v[0]), float(v[1]), float(v[2])
                mesh.vertices.append(pt)
            for t in tris:
                mt = MeshTriangle()
                mt.vertex_indices = [t[0], t[1], t[2]]
                mesh.triangles.append(mt)
            return mesh
        except Exception as exc:
            self.get_logger().warning(f'[CollisionWorld] STL load failed ({path}): {exc}')
            return None

    def _attach_probe_object(self) -> None:
        """Attach the probe mesh (probe.stl) to arm_gripper_base_link.

        Orientation is taken directly from the PCA long axis stored in
        _last_detected_object_rotation_base (bypasses gripper-symmetry
        resolution which previously caused a consistent 180° flip).

        Contact point is taken from the camera-detected probe centre
        (current_target_point_base) transformed into the link frame via live
        TF — more accurate than using the theoretical fourbar offsets.

        STL dimensions (metres):
          X: 0 → 0.045  (cross-section width)
          Y: 0 → 0.045  (cross-section height)
          Z: 0 → 0.300  (long axis / probe length)

        Flat-on-floor coordinate mapping:
          STL Z (long, 300 mm) → world [cos(probe_yaw), sin(probe_yaw), 0]
          STL Y (height, 45 mm)→ world [0, 0, 1]  (pointing up)
          STL X (width, 45 mm) → world [-sin(yaw), cos(yaw), 0]
        """
        STL_LEN = 0.300
        STL_W   = 0.045
        STL_H   = 0.045
        stl_cx  = STL_W   / 2.0   # 0.0225 m
        stl_cy  = STL_H   / 2.0   # 0.0225 m
        stl_cz  = STL_LEN / 2.0   # 0.150  m

        contact_y = float(getattr(self, 'fourbar_contact_y_offset_m', 0.026))
        contact_z = float(getattr(self, 'fourbar_contact_z_closed_m', 0.218))

        # ── Probe yaw: use raw PCA long axis to avoid gripper-symmetry 180° flip ──
        # _last_detected_object_rotation_base[:, 0] is stabilised across frames by
        # _update_detected_object_pose_from_camera, so its sign is consistent within
        # a run.  We use it directly instead of backing out gripper_yaw - offset_rad,
        # which was the source of the consistent 180° misalignment.
        correction_rad = math.radians(float(getattr(self, 'stl_yaw_correction_deg', 0.0)))
        if self._last_detected_object_rotation_base is not None:
            long_ax = self._last_detected_object_rotation_base[:, 0]
            probe_yaw = math.atan2(float(long_ax[1]), float(long_ax[0])) + correction_rad
        elif self.detected_object_yaw_rad is not None:
            offset_rad = math.radians(float(getattr(self, 'object_yaw_rotation_offset_deg', 90.0)))
            probe_yaw = self.detected_object_yaw_rad - offset_rad + correction_rad
        else:
            probe_yaw = correction_rad   # fallback

        # ── R_world_in_link: use stored GRASP orientation, NOT current TF ─────────
        # The mesh_pose is expressed in the link frame (arm_gripper_base_link).
        # R_stl_in_link = R_world_in_link @ R_stl_in_world is CONSTANT once the
        # probe is grasped (rigid body).  We must compute it from the link
        # orientation AT GRASP TIME.  If we used the current TF (at pick_home),
        # the link would have a completely different orientation and R_stl_in_link
        # would be wrong — producing the "mirrored" STL seen in RViz.
        if self.grasp_orientation is not None:
            # grasp_orientation is arm_gripper_base_link in base_link = R_link_in_world
            R_link_in_world = quat_to_matrix(self.grasp_orientation)
            R_world_in_link = R_link_in_world.T
        else:
            R_world_in_link = np.eye(3)
            self.get_logger().warning('[CollisionWorld] grasp_orientation not set; using identity rotation for STL.')

        # ── Contact point: actual fourbar offset used during the grasp ───────────
        # effective_target_point_offset_in_link is the probe centre in
        # arm_gripper_base_link frame as computed by the fourbar model at close
        # time.  This is more accurate than the static fourbar_contact_* params
        # because it accounts for the specific computed_gripper_close angle.
        eff = getattr(self, 'effective_target_point_offset_in_link', None)
        if eff is not None and len(eff) >= 3:
            contact_in_link = np.array([float(eff[0]), float(eff[1]), float(eff[2])])
        else:
            contact_in_link = np.array([0.0, contact_y, contact_z])
        self.get_logger().info(f'[CollisionWorld] Probe centre in link (grasp-time offset): '
            f'[{contact_in_link[0]:.3f}, {contact_in_link[1]:.3f}, {contact_in_link[2]:.3f}]')

        # ── R_stl_in_world: STL basis vectors in world frame ────────────────────
        cy, sy = math.cos(probe_yaw), math.sin(probe_yaw)
        R_stl_in_world = np.array([
            [-sy, 0.0,  cy],
            [ cy, 0.0,  sy],
            [0.0, 1.0, 0.0],
        ], dtype=float)

        # ── R_stl_in_link = R_world_in_link @ R_stl_in_world ──────────────────
        R_stl_in_link = R_world_in_link @ R_stl_in_world

        # ── Mesh origin in link frame ───────────────────────────────────────────
        # Place STL so geometric centre [stl_cx, stl_cy, stl_cz] lands at contact_in_link.
        stl_center_in_stl = np.array([stl_cx, stl_cy, stl_cz])
        origin_in_link    = contact_in_link - R_stl_in_link @ stl_center_in_stl

        mesh_pose = Pose()
        mesh_pose.position.x = float(origin_in_link[0])
        mesh_pose.position.y = float(origin_in_link[1])
        mesh_pose.position.z = float(origin_in_link[2])
        mesh_pose.orientation = matrix_to_quat(R_stl_in_link)

        # ── Load STL mesh ───────────────────────────────────────────────────────
        stl_path = self._find_probe_stl()
        mesh = self._load_stl_mesh(stl_path) if stl_path else None

        inner = CollisionObject()
        inner.header.frame_id = self.planning_link
        inner.header.stamp = self.get_clock().now().to_msg()
        inner.id = 'post_grasp_probe'
        inner.operation = CollisionObject.ADD

        if mesh is not None:
            inner.meshes = [mesh]
            inner.mesh_poses = [mesh_pose]
            shape_info = (
                f'STL mesh {int(STL_LEN*1000)}×{int(STL_W*1000)}×{int(STL_H*1000)} mm, '
                f'probe_yaw={math.degrees(probe_yaw):.1f}°, '
                f'origin_in_link=[{origin_in_link[0]:.3f}, '
                f'{origin_in_link[1]:.3f}, {origin_in_link[2]:.3f}]'
            )
        else:
            cyl = SolidPrimitive()
            cyl.type = SolidPrimitive.CYLINDER
            cyl.dimensions = [STL_LEN, STL_W / 2.0]
            fb_pose = Pose()
            fb_pose.position.x = 0.0
            fb_pose.position.y = contact_y
            fb_pose.position.z = contact_z / 2.0
            fb_pose.orientation.w = 1.0
            inner.primitives = [cyl]
            inner.primitive_poses = [fb_pose]
            shape_info = f'fallback cylinder h={STL_LEN:.3f} m r={STL_W/2:.3f} m'

        aco = AttachedCollisionObject()
        aco.link_name = self.planning_link
        aco.object = inner
        aco.touch_links = [
            self.planning_link,
            'gripper_gear_left_link',
            'gripper_gear_right_link',
            'gripper_left_link',
            'gripper_right_link',
            'gripper_bucket_left_link',
            'gripper_bucket_right_link',
            'gripper_gear_tip_left_link',
            'gripper_gear_tip_right_link',
            'gripper_link_tip_left_link',
            'gripper_link_tip_right_link',
            'gripper_camera_link',
        ]
        self._attached_object_pub.publish(aco)
        self._post_grasp_probe_attached = True
        self.get_logger().info(f'[CollisionWorld] Attached probe: {shape_info}')

    def _remove_post_grasp_collision_objects(self) -> None:
        """Remove the floor plane and detach the probe from the collision world."""
        removed = []
        if self._post_grasp_floor_active:
            floor_obj = CollisionObject()
            floor_obj.header.frame_id = self.planning_frame
            floor_obj.header.stamp = self.get_clock().now().to_msg()
            floor_obj.id = 'post_grasp_floor'
            floor_obj.operation = CollisionObject.REMOVE
            self._collision_object_pub.publish(floor_obj)
            self._post_grasp_floor_active = False
            removed.append('floor')

        if self._post_grasp_probe_attached:
            # Detach only when this node actually attached the object. MoveIt
            # otherwise emits a misleading ERROR on every pre-grasp reset.
            det_inner = CollisionObject()
            det_inner.header.frame_id = self.planning_link
            det_inner.header.stamp = self.get_clock().now().to_msg()
            det_inner.id = 'post_grasp_probe'
            det_inner.operation = CollisionObject.REMOVE

            det_aco = AttachedCollisionObject()
            det_aco.link_name = self.planning_link
            det_aco.object = det_inner
            self._attached_object_pub.publish(det_aco)

            world_probe = CollisionObject()
            world_probe.header.frame_id = self.planning_frame
            world_probe.header.stamp = self.get_clock().now().to_msg()
            world_probe.id = 'post_grasp_probe'
            world_probe.operation = CollisionObject.REMOVE
            self._collision_object_pub.publish(world_probe)
            self._post_grasp_probe_attached = False
            removed.append('probe')

        if removed:
            self.get_logger().info(
                f'[CollisionWorld] Removed post-grasp objects: {", ".join(removed)}.'
            )

    def compute_approach_axis_in_planning_frame(self, orientation: Quaternion) -> np.ndarray:
        return normalize(quat_to_matrix(orientation) @ self.approach_axis_in_tool.reshape(3,))

    def _compute_downward_orientation(self) -> Optional[Quaternion]:
        """Return orientation with gripper pointing straight down (roll=pi, pitch=0).
        Uses object 6D pose yaw when available; falls back to current arm yaw."""
        # 6D Pose Agent: align gripper yaw with detected object orientation.
        if self.object_yaw_align_enabled and self.detected_object_yaw_rad is not None:
            yaw = self.detected_object_yaw_rad
            self.get_logger().info(f'Grasp orientation from object 6D pose: '
                f'roll=180° pitch=0° yaw={math.degrees(yaw):.1f}° '
                f'(object long axis + {self.object_yaw_rotation_offset_deg:.0f}° offset)')
            return rpy_to_quat(math.pi, 0.0, yaw)
        # Fallback: use current arm yaw to minimise joint motion.
        # Simplest reliable approach: read the current end-effector yaw from TF,
        # then build roll=pi, pitch=0, yaw=<current_yaw> quaternion.
        cur = self.get_current_tool_orientation_in_planning_frame()
        if cur is None:
            self.get_logger().warning('TF unavailable; using fixed RPY for grasp orientation.')
            return rpy_to_quat(self.fixed_roll, self.fixed_pitch, self.fixed_yaw)
        # Extract yaw of current end-effector about Z-axis in planning frame
        R = quat_to_matrix(cur)
        yaw = math.atan2(R[1, 0], R[0, 0])  # yaw = atan2(R10, R00)
        result = rpy_to_quat(math.pi, 0.0, yaw)
        self.get_logger().info(f'Downward orientation from arm: roll=180° pitch=0° yaw={math.degrees(yaw):.1f}°')
        return result

    def choose_target_orientation(self) -> Optional[Quaternion]:
        if self.keep_current_orientation:
            return self.get_current_tool_orientation_in_planning_frame()
        # Compute minimum-rotation correction so approach_axis points straight down
        return self._compute_downward_orientation()

    def make_pose(self, point_base: np.ndarray, orientation: Quaternion) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.planning_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position = Point(x=float(point_base[0]), y=float(point_base[1]), z=float(point_base[2]))
        msg.pose.orientation = orientation
        return msg

    def _pose_xyz(self, pose: PoseStamped) -> np.ndarray:
        return np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ], dtype=np.float64)

    def contact_pose_to_link_pose(self, contact_pose: PoseStamped) -> Pose:
        """
        Convert desired grasp/contact point pose to real arm_gripper_base_link pose.

        Important:
        send_pose_goal() uses PositionConstraint.target_point_offset.
        But GetCartesianPath does NOT support target_point_offset.

        Without this conversion, Cartesian grasp is wrong by target_point_offset_in_link,
        by the full true four-bar XYZ contact offset.
        """
        q = contact_pose.pose.orientation
        contact_xyz = self._pose_xyz(contact_pose)
        offset = np.array(self.effective_target_point_offset_in_link, dtype=np.float64)

        link_xyz = contact_xyz - quat_to_matrix(q) @ offset

        out = Pose()
        out.position = Point(
            x=float(link_xyz[0]),
            y=float(link_xyz[1]),
            z=float(link_xyz[2]),
        )
        out.orientation = q
        return out

    def _measure_final_grasp_pose_error(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """Return actual-vs-committed final grasp pose error for arm_gripper_base_link."""
        if self.grasp_pose is None:
            return None

        desired = self.contact_pose_to_link_pose(self.grasp_pose)
        current = self.get_current_link_pose_in_planning_frame()
        if current is None:
            return None

        desired_xyz = np.array([
            float(desired.position.x),
            float(desired.position.y),
            float(desired.position.z),
        ], dtype=np.float64)
        current_xyz = np.array([
            float(current.position.x),
            float(current.position.y),
            float(current.position.z),
        ], dtype=np.float64)
        delta = current_xyz - desired_xyz
        pos_err = float(np.linalg.norm(delta))

        R_des = quat_to_matrix(desired.orientation)
        R_cur = quat_to_matrix(current.orientation)
        trace_val = float(np.trace(R_des.T @ R_cur))
        ori_err = math.acos(float(np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)))

        return delta, pos_err, ori_err

    def _log_final_grasp_pose_error(
        self,
        throttle_duration_sec: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        """Compare actual arm_gripper_base_link TF against the committed grasp pose."""
        measured = self._measure_final_grasp_pose_error()
        if measured is None:
            return None

        delta, pos_err, ori_err = measured
        log = self.get_logger().info if pos_err <= 0.012 else self.get_logger().warn
        msg = (
            f'Final grasp link pose error before close: '
            f'dxyz=({delta[0]*1000:.1f},{delta[1]*1000:.1f},{delta[2]*1000:.1f})mm '
            f'pos={pos_err*1000:.1f}mm ori={math.degrees(ori_err):.1f}deg. '
            'Cartesian final approach used MoveIt collision checking.'
        )
        if throttle_duration_sec is None:
            log(msg)
        else:
            log(msg, throttle_duration_sec=throttle_duration_sec)
        return measured

    def apply_fourbar_ground_guard_to_offset(self, contact_point: np.ndarray, orientation: Quaternion) -> None:
        """
        Ground guard for the actual four-bar model.

        Important change: do NOT fake safety by increasing the local contact
        offset.  That was the reason the gripper stayed above/away from the
        probe.  With the real model, the local offset must remain the true bucket
        contact midpoint.  Floor safety is handled by lifting the selected
        contact point, not by lying about the gripper geometry.
        """
        if not self.fourbar_ground_guard_enabled:
            return
        off = self._fourbar_actual_contact_offset(float(self.computed_gripper_close))
        self.effective_target_point_offset_in_link = [float(off[0]), float(off[1]), float(off[2])]
        self.get_logger().info(f'Actual four-bar offset locked: '
            f'({off[0]*1000:.1f}, {off[1]*1000:.1f}, {off[2]*1000:.1f}) mm. '
            'Ground guard will lift contact point if required; offset will not be inflated.')

    def _predict_fourbar_arc_min_z(self, contact_point: np.ndarray, orientation: Quaternion) -> Tuple[float, float]:
        """Predict lowest bucket z while closing from open to q_close."""
        R = quat_to_matrix(orientation)
        contact = np.array(contact_point, dtype=np.float64)
        closed_offset = np.array(self.effective_target_point_offset_in_link, dtype=np.float64)
        link_origin = contact - R @ closed_offset
        n = max(3, int(self.fourbar_arc_sample_count))
        samples = np.linspace(float(self.gripper_open), float(self.computed_gripper_close), n)
        min_z = float('inf')
        for q_sample in samples:
            local_contact = self._fourbar_actual_contact_offset(float(q_sample))
            # Bucket tip is below the local contact point by approximately
            # max_z-contact_z.  Use actual STL max z, but do not inflate the
            # contact offset.  This protects the sweep without making the
            # gripper miss the object.
            local_bucket_z = max(
                float(local_contact[2]),
                float(self.fourbar_bucket_tip_z_max_m),
            ) + float(self.fourbar_open_close_guard_extra_m)
            p_bucket = link_origin + R @ np.array([0.0, float(self.fourbar_contact_y_offset_m), local_bucket_z], dtype=np.float64)
            min_z = min(min_z, float(p_bucket[2]))
        required = float(self.floor_z_min + max(0.0, self.fourbar_ground_clearance_m))
        return min_z, float(min_z - required)

    def apply_fourbar_arc_guard_to_grasp_point(self, grasp_point: np.ndarray, orientation: Quaternion) -> np.ndarray:
        """Lift the contact only enough for the configured physical arc clearance."""
        if not self.fourbar_arc_guard_enabled:
            return grasp_point
        min_z, clearance = self._predict_fourbar_arc_min_z(grasp_point, orientation)
        required = float(
            self.floor_z_min
            + max(0.0, self.fourbar_ground_clearance_m)
            + max(0.0, self.fourbar_min_arc_clearance_m)
        )
        if min_z < required:
            lift = required - min_z
            configured_cap = max(0.0, float(self.fourbar_max_contact_lift_m))
            if configured_cap > 0.0 and lift > configured_cap:
                # The old implementation capped this correction and then
                # executed a close whose own log still showed negative safety
                # clearance. A safety guard must never knowingly return an
                # unsafe pose. Apply the full required lift; a missed grasp is
                # recoverable, a floor collision is not.
                self.get_logger().warning(f'Required four-bar floor lift {lift*1000:.1f} mm exceeds '
                    f'configured advisory cap {configured_cap*1000:.1f} mm; applying the full safety correction.')
            grasp_point = np.array(grasp_point, dtype=np.float64).copy()
            grasp_point[2] += float(lift)
            min_z_after, clearance_after = self._predict_fourbar_arc_min_z(grasp_point, orientation)
            self.get_logger().warning(f'Four-bar actual closing-arc guard lifted contact point by {lift*1000:.1f} mm: '
                f'predicted_min_bucket_z {min_z:.3f} < required {required:.3f}. '
                f'After lift min_z={min_z_after:.3f}, clearance={clearance_after*1000:.1f}mm.')
        else:
            self.get_logger().info(f'Four-bar actual closing-arc clearance OK: min_bucket_z={min_z:.3f}, '
                f'clearance={clearance*1000:.1f}mm.')
        return grasp_point

    def _log_committed_grasp_geometry(self, label: str) -> None:
        if self.current_target_point_base is None or self.grasp_orientation is None or self.grasp_pose is None:
            return
        grasp_point = self._pose_xyz(self.grasp_pose)
        min_z, clearance = self._predict_fourbar_arc_min_z(grasp_point, self.grasp_orientation)
        used_width = self.last_estimated_object_width_m
        if used_width is None and self._last_detected_width_m is not None:
            used_width = self._last_detected_width_m
        self.get_logger().info(f'[{label}] committed grasp geometry: '
            f'target=({self.current_target_point_base[0]:.3f},{self.current_target_point_base[1]:.3f},{self.current_target_point_base[2]:.3f}) '
            f'grasp=({grasp_point[0]:.3f},{grasp_point[1]:.3f},{grasp_point[2]:.3f}) '
            f'width={(used_width*1000.0 if used_width is not None else -1):.1f}mm '
            f'q_open={self.gripper_open:.3f} q_close={self.computed_gripper_close:.3f} '
            f'offset=({self.effective_target_point_offset_in_link[0]:.3f},'
            f'{self.effective_target_point_offset_in_link[1]:.3f},'
            f'{self.effective_target_point_offset_in_link[2]:.3f}) '
            f'arc_min_z={min_z:.3f} clearance={clearance*1000:.1f}mm')

    def _refresh_grasp_geometry_from_latest_estimates(self, label: str) -> bool:
        """Recalculate width, q_close, contact offset, orientation, and poses."""
        if self.current_target_point_base is None:
            return False

        if self.adaptive_gripper_enabled:
            width_for_grasp = (
                self._last_detected_width_m
                if self._last_detected_width_m is not None
                else self.nominal_probe_width_m
            )
            self.computed_gripper_close, self.computed_gripper_preclose = self._compute_adaptive_gripper_close(
                width_for_grasp
            )
            self.last_estimated_object_width_m = width_for_grasp
            if self._last_detected_width_m is None:
                self.get_logger().warning(
                    'No reliable 3D width estimate; using nominal_probe_width_m '
                    f'({width_for_grasp*1000.0:.1f} mm) instead of commanding the gripper fully closed.'
                )
        else:
            self.computed_gripper_close = self.gripper_close
            self.computed_gripper_preclose = self.gripper_preclose

        # One-go close: final geometry must be based on q_close, not preclose.
        self._apply_fourbar_contact_offset(self.computed_gripper_close)

        orientation_locked = (
            self.lock_grasp_orientation_after_initial_plan
            and self.grasp_orientation is not None
        )

        if not orientation_locked and self.object_yaw_align_enabled and self._last_detected_orientation_cam is not None:
            yaw = self._compute_grasp_yaw_from_object(self._last_detected_orientation_cam)
            if yaw is not None:
                self.detected_object_yaw_rad = yaw

        if orientation_locked:
            orientation = self.grasp_orientation
            if self._last_detected_orientation_cam is not None:
                self.get_logger().info(f'Keeping initial grasp orientation during {label}; '
                    'close-range object yaw update ignored so the final Cartesian descent stays straight.')
        else:
            orientation = self.choose_target_orientation()
        if orientation is None:
            return False
        self.grasp_orientation = orientation

        self.update_contact_poses_from_target(self.current_target_point_base, orientation)
        self.publish_markers()
        self._log_committed_grasp_geometry(label)
        return True

    def _apply_grasp_target_bias(self, target: np.ndarray, orientation: Quaternion) -> np.ndarray:
        """Apply operator-calibrated target bias in base and tool axes."""
        corrected = np.array(target, dtype=np.float64)

        bias_base = getattr(self, 'grasp_target_bias_in_base', None)
        if bias_base is not None:
            bias_base = np.array(bias_base, dtype=np.float64).reshape(3,)
            if float(np.linalg.norm(bias_base)) >= 1e-9:
                corrected = corrected + bias_base
                self.get_logger().info(f'Applying calibrated grasp target base bias: '
                    f'({bias_base[0]*1000:.1f},{bias_base[1]*1000:.1f},{bias_base[2]*1000:.1f})mm.')

        bias_tool = getattr(self, 'grasp_target_bias_in_tool', None)
        if bias_tool is None:
            return corrected

        bias_tool = np.array(bias_tool, dtype=np.float64).reshape(3,)
        if float(np.linalg.norm(bias_tool)) < 1e-9:
            return corrected

        bias_world = quat_to_matrix(orientation) @ bias_tool
        corrected = corrected + bias_world
        self.get_logger().info(f'Applying calibrated grasp target bias: '
            f'tool=({bias_tool[0]*1000:.1f},{bias_tool[1]*1000:.1f},{bias_tool[2]*1000:.1f})mm '
            f'world=({bias_world[0]*1000:.1f},{bias_world[1]*1000:.1f},{bias_world[2]*1000:.1f})mm.')
        return corrected

    def update_contact_poses_from_target(self, target: np.ndarray, orientation: Quaternion) -> None:
        """
        Recompute pre-grasp, grasp and retreat consistently from one target point.
        """
        approach_axis = self.compute_approach_axis_in_planning_frame(orientation)
        motion_target = self._apply_grasp_target_bias(target, orientation)

        pre_grasp_point = motion_target - approach_axis * self.pre_grasp_distance
        # Positive grasp_depth_below_surface_m pushes the finger contacts
        # further along the approach axis, past the detected surface, so the
        # fingers actually wrap around the probe body instead of just touching
        # its top.
        grasp_point = motion_target + approach_axis * self.grasp_depth_below_surface_m

        if self.floor_safe_grasp_enabled:
            # Do not let the computed contact point dig downward.  The detected
            # mask point is already on/near the visible object surface; for a
            # floor probe, extra insertion mostly becomes ground collision.
            downward_descent = float(target[2] - grasp_point[2])
            if downward_descent > self.max_grasp_descent_below_target_m:
                grasp_point[2] = float(target[2] - self.max_grasp_descent_below_target_m)
                self.get_logger().warning(f'Floor-safe grasp clamp: descent {downward_descent*1000:.1f} mm '
                    f'limited to {self.max_grasp_descent_below_target_m*1000:.1f} mm.')

            # Stronger floor clamp: the contact target must be high enough for a
            # top/down bucket grasp.  This is separate from the bucket offset guard.
            min_contact_z = max(
                self.floor_z_min + self.min_grasp_height_above_floor_m,
                self.floor_z_min + self.floor_safe_contact_height_m,
            )
            if grasp_point[2] < min_contact_z:
                lift = min_contact_z - float(grasp_point[2])
                grasp_point[2] = min_contact_z
                self.get_logger().warning(f'Floor-safe grasp lifted contact point by {lift*1000:.1f} mm '
                    f'to keep contact target above floor-safe height.')

            # After the contact point is decided, update the local offset so the
            # bucket swept volume also stays above the floor.
            self.apply_fourbar_ground_guard_to_offset(grasp_point, orientation)
            # Then verify the actual open→closed four-bar/bucket arc with the
            # selected q_close.  This protects the final one-go close.
            grasp_point = self.apply_fourbar_arc_guard_to_grasp_point(grasp_point, orientation)

        retreat_point = motion_target - approach_axis * self.retreat_distance

        # Clamp only pre-grasp and retreat.
        # Grasp point is protected by the floor-safe clamp above.
        pre_grasp_point[2] = max(float(pre_grasp_point[2]), self.min_pose_z)
        retreat_point[2] = max(float(retreat_point[2]), self.min_pose_z)

        self.pre_grasp_pose = self.make_pose(pre_grasp_point, orientation)
        self.grasp_pose = self.make_pose(grasp_point, orientation)
        self.retreat_pose = self.make_pose(retreat_point, orientation)

    def is_refined_target_acceptable(self, candidate: np.ndarray) -> bool:
        """
        Reject close-range refinement if it jumps away from the original locked target.
        In your video, refinement jumped around 13 cm sideways. That must be rejected.
        """
        if self.current_target_point_base is None:
            return False

        delta = candidate - self.current_target_point_base
        lateral = float(np.linalg.norm(delta[:2]))
        vertical = abs(float(delta[2]))
        total = float(np.linalg.norm(delta))

        if (
            total > self.refine_accept_radius_m
            or lateral > self.refine_lateral_max_m
            or vertical > self.refine_vertical_max_m
        ):
            self.get_logger().warning(f'Refinement rejected: jump total={total:.3f}m '
                f'lateral={lateral:.3f}m vertical={vertical:.3f}m. '
                f'Keeping original target.', throttle_duration_sec=0.5)
            return False

        return True

    def _clear_target_stability_history(self) -> None:
        self.target_history.clear()
        self.target_history_stamps.clear()
        self.target_confidence_history.clear()
        self.filtered_target_point_base = None
        self.filtered_target_confidence = 0.0
        self.target_filter_max_residual_m = float('inf')
        self.target_filter_rms_m = float('inf')

    def _expire_target_stability_history(self, now_sec: float) -> None:
        if (
            self.target_history_stamps
            and now_sec - self.target_history_stamps[-1]
            > self.target_stability_max_sample_gap_sec
        ):
            self._clear_target_stability_history()

    def is_target_stable(self, p: np.ndarray, confidence: float) -> bool:
        """Build a consecutive, spatially coherent 3D cluster and median-filter it."""
        now_sec = self._now_sec()
        self._expire_target_stability_history(now_sec)

        if confidence < self.target_lock_min_confidence:
            self.get_logger().info(
                f'Probe confidence {confidence:.2f} is below lock threshold '
                f'{self.target_lock_min_confidence:.2f}; not adding it to the '
                'target stability window.',
                throttle_duration_sec=1.0,
            )
            return False

        candidate = np.asarray(p, dtype=np.float64)
        if self.target_history:
            existing_center = np.median(
                np.asarray(self.target_history, dtype=np.float64), axis=0
            )
            outlier_distance = float(np.linalg.norm(candidate - existing_center))
            if outlier_distance > self.target_filter_outlier_distance_m:
                self.get_logger().warning(
                    f'Resetting probe stability window after a '
                    f'{outlier_distance*1000.0:.1f} mm 3D detection jump.',
                    throttle_duration_sec=0.5,
                )
                self._clear_target_stability_history()

        self.target_history.append(candidate.copy())
        self.target_history_stamps.append(now_sec)
        self.target_confidence_history.append(float(confidence))

        while len(self.target_history) > self.target_filter_window_samples:
            self.target_history.pop(0)
            self.target_history_stamps.pop(0)
            self.target_confidence_history.pop(0)

        points = np.asarray(self.target_history, dtype=np.float64)
        filtered = np.median(points, axis=0)
        residuals = np.linalg.norm(points - filtered, axis=1)
        self.filtered_target_point_base = filtered
        self.filtered_target_confidence = float(np.median(self.target_confidence_history))
        self.target_filter_max_residual_m = float(np.max(residuals))
        self.target_filter_rms_m = float(np.sqrt(np.mean(np.square(residuals))))

        if len(self.target_history) < self.target_stability_samples:
            return False

        return (
            self.target_filter_max_residual_m <= self.target_stability_max_jump_m
            and self.target_filter_rms_m <= self.target_stability_rms_m
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _get_result_mask_for_box(self, result, box_index: int, image_shape) -> Optional[np.ndarray]:
        """
        Return a boolean mask for one YOLO segmentation result.
        Works with ultralytics segmentation models such as yolo26-seg.
        """
        if not self.use_segmentation_mask:
            return None

        if not hasattr(result, 'masks') or result.masks is None:
            return None

        try:
            masks = result.masks.data
            if masks is None or box_index >= len(masks):
                return None

            mask = masks[box_index].detach().cpu().numpy()
            h_img, w_img = image_shape[:2]

            if mask.shape[0] != h_img or mask.shape[1] != w_img:
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

            mask_bool = mask > self.mask_score_threshold

            if self.mask_erode_px > 0:
                kernel = np.ones(
                    (self.mask_erode_px, self.mask_erode_px),
                    dtype=np.uint8
                )
                mask_bool = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)

            if int(mask_bool.sum()) < self.mask_min_pixels:
                return None

            return mask_bool

        except Exception as exc:
            self.get_logger().warning(f'Failed to read YOLO segmentation mask: {exc}', throttle_duration_sec=1.0)
            return None

    def _mask_depth_target(
        self,
        mask_bool: np.ndarray,
        expected_depth_m: Optional[float] = None,
        depth_band_m: Optional[float] = None,
        prefer_nearest: bool = True,
        depth_image: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Convert segmentation mask into a robust pixel+depth target.

        This is much better than bbox center because bbox center often lands on:
          - floor,
          - gripper finger,
          - empty background,
          - cropped part of object.
        """
        depth_img = depth_image if depth_image is not None else self.latest_depth
        if depth_img is None:
            return None

        h, w = depth_img.shape[:2]

        if mask_bool.shape[0] != h or mask_bool.shape[1] != w:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        ys, xs = np.where(mask_bool)
        if xs.size < self.mask_min_pixels:
            return None

        depths = depth_img[ys, xs]
        valid = np.isfinite(depths) & (depths > self.min_depth_m) & (depths < self.max_depth_m)

        if expected_depth_m is not None and depth_band_m is not None:
            lo = max(self.min_depth_m, float(expected_depth_m) - float(depth_band_m))
            hi = min(self.max_depth_m, float(expected_depth_m) + float(depth_band_m))
            valid = valid & (depths >= lo) & (depths <= hi)

        if int(valid.sum()) < max(20, self.mask_min_pixels // 4):
            return None

        xs_v = xs[valid]
        ys_v = ys[valid]
        d_v = depths[valid]

        def axis_midpoint_pixel() -> Tuple[int, int]:
            coords = np.column_stack((xs_v.astype(np.float64), ys_v.astype(np.float64)))
            if coords.shape[0] < 3:
                return int(np.median(xs_v)), int(np.median(ys_v))

            center = coords.mean(axis=0)
            centered = coords - center
            cov = centered.T @ centered / max(1, coords.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, int(np.argmax(eigvals))]
            ortho_axis = np.array([-axis[1], axis[0]], dtype=np.float64)
            proj = centered @ axis
            ortho = np.abs(centered @ ortho_axis)

            midpoint_proj = 0.5 * (float(proj.min()) + float(proj.max()))
            score = np.abs(proj - midpoint_proj) + 0.25 * ortho
            idx = int(np.argmin(score))
            return int(xs_v[idx]), int(ys_v[idx])

        if prefer_nearest:
            depth = float(np.percentile(d_v, self.mask_depth_percentile))
            near = np.abs(d_v - depth) < 0.02
            if int(near.sum()) >= 10:
                u = int(np.median(xs_v[near]))
                v = int(np.median(ys_v[near]))
            else:
                u = int(np.median(xs_v))
                v = int(np.median(ys_v))
        else:
            depth = float(np.median(d_v))
            # For elongated probes, the area-weighted mask median can still sit
            # closer to the visually larger upper side. Pick the midpoint of the
            # mask's principal axis instead.
            u, v = axis_midpoint_pixel()

        return u, v, depth

    def _select_best_detection(self, results, confidence_threshold: float,
                               image_shape: Optional[Tuple[int, ...]] = None):
        """
        Select best YOLO detection and keep its segmentation mask if available.
        """
        best = None
        best_conf = -1.0
        if image_shape is None and self.latest_color is not None:
            image_shape = self.latest_color.shape

        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls]

                if self.target_class != 'any' and name != self.target_class:
                    continue

                if conf < confidence_threshold:
                    continue

                if conf > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    mask_bool = self._get_result_mask_for_box(result, i, image_shape)

                    best = {
                        'name': name,
                        'conf': conf,
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'u_bbox': int((x1 + x2) * 0.5),
                        'v_bbox': int((y1 + y2) * 0.5),
                        'mask': mask_bool,
                    }
                    best_conf = conf

        return best

    def _poll_inference(self) -> Optional[Tuple[FrameSnapshot, list]]:
        """Submit the newest consistent frame pair and return the newest
        completed inference, or None while one is still in flight.

        The model runs in a background thread, so this never blocks the
        executor. Results always come with the exact FrameSnapshot they were
        computed from (typically one detect tick old); downstream depth
        sampling and TF use that snapshot, never newer sensor data.
        """
        if self._yolo_worker is None:
            return None

        best_pair = None
        best_key = None
        for color_stamp, color in self._color_frame_queue:
            for depth_stamp, depth_frame, depth in self._depth_frame_queue:
                gap_sec = abs((color_stamp - depth_stamp).nanoseconds) * 1e-9
                newest_sec = max(color_stamp.nanoseconds, depth_stamp.nanoseconds) * 1e-9
                key = (gap_sec, -newest_sec)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (
                        color_stamp, color, depth_stamp, depth_frame, depth, gap_sec
                    )

        if best_pair is not None:
            color_stamp, color, depth_stamp, depth_frame, depth, gap_sec = best_pair
            if gap_sec <= self.max_color_depth_stamp_gap_sec:
                pair_key = (color_stamp.nanoseconds, depth_stamp.nanoseconds)
                if pair_key != self._last_inference_pair_key:
                    self._last_inference_pair_key = pair_key
                    # Drop submitted/older frames so an old exact pair cannot
                    # permanently beat newer near-synchronized pairs.
                    self._color_frame_queue = deque(
                        (item for item in self._color_frame_queue
                         if item[0].nanoseconds > color_stamp.nanoseconds),
                        maxlen=self.sensor_sync_queue_size,
                    )
                    self._depth_frame_queue = deque(
                        (item for item in self._depth_frame_queue
                         if item[0].nanoseconds > depth_stamp.nanoseconds),
                        maxlen=self.sensor_sync_queue_size,
                    )
                    snap = FrameSnapshot(
                        color=color.copy(),
                        color_stamp_sec=color_stamp.nanoseconds * 1e-9,
                        # depth_cb replaces the array, never mutates it in place,
                        # so holding a reference is safe.
                        depth=depth,
                        depth_stamp_sec=depth_stamp.nanoseconds * 1e-9,
                        depth_frame=depth_frame,
                        stamp=depth_stamp,
                    )
                    self._yolo_worker.submit(snap, snap.color)
            else:
                now_sec = self._now_sec()
                if now_sec - self._stamp_gap_warned_sec > 5.0:
                    self._stamp_gap_warned_sec = now_sec
                    self.get_logger().warning(
                        f'Waiting for synchronized camera frames: closest color/depth pair differs by '
                        f'{gap_sec:.3f}s (> {self.max_color_depth_stamp_gap_sec:.3f}s '
                        'max_color_depth_stamp_gap_sec). A mismatched pair on a '
                        'moving camera yields a wrong 3D target; check camera rates '
                        'if this persists.'
                    )

        return self._yolo_worker.take_result()

    def detect_target_once(
        self,
        publish_debug: bool = True,
        allow_state_updates: bool = True,
    ) -> Optional[Tuple[np.ndarray, str, float]]:
        """
        Segmentation Perception Agent:
        Use yolo26-seg mask first. Fall back to bbox center only if no mask exists.

        Consumes the newest completed background inference (submitted on a
        previous call) and submits the current frames for the next one.
        """
        completed = self._poll_inference()

        if completed is None or self.camera_info is None:
            if publish_debug and self.latest_color is not None:
                annotated = self.latest_color.copy()
                if not allow_state_updates and self.busy:
                    self._stamp_debug_status(
                        annotated,
                        f'Live detect view: {self.sequence_stage}',
                    )
                self.publish_debug_image(annotated)
            return None

        snap, results = completed

        annotated = self._annotate_yolo_results(snap.color, results)
        if not allow_state_updates and self.busy:
            self._stamp_debug_status(
                annotated,
                f'Live detect view: {self.sequence_stage}',
            )

        best = self._select_best_detection(
            results, self.confidence_threshold, image_shape=snap.color.shape
        )

        if best is None:
            if allow_state_updates:
                self._clear_detected_object_pose()
            if publish_debug:
                self.publish_debug_image(annotated)
            return None

        source = 'bbox'
        mask_target = None

        if best['mask'] is not None:
            mask_target = self._mask_depth_target(
                best['mask'],
                # For the floor probe task, use the geometric mask centre.
                # Nearest-surface bias pulls the pick point toward the visible
                # upper edge of the probe instead of its middle.
                prefer_nearest=False,
                depth_image=snap.depth,
            )

        # Object Width Estimator Agent: estimate 3D size for adaptive gripper sizing.
        if allow_state_updates:
            self._last_detected_width_m = None
            if self.adaptive_gripper_enabled and best['mask'] is not None:
                w3d = self._estimate_object_width_3d(best['mask'], snap.depth)
                if (
                    w3d is not None
                    and self.adaptive_gripper_min_width_m <= w3d <= self.adaptive_gripper_max_width_m
                ):
                    self._last_detected_width_m = w3d

        # 6D Pose Agent — Orientation: estimate object long axis from point-cloud PCA.
        if allow_state_updates:
            self._last_detected_orientation_cam = None
        if (
            allow_state_updates
            and self.object_yaw_align_enabled
            and best['mask'] is not None
        ):
            pose_estimate = self._estimate_object_orientation_3d(best['mask'], snap.depth)
            if pose_estimate is not None:
                centroid_cam, self._last_detected_orientation_cam = pose_estimate
                self._update_detected_object_pose_from_camera(
                    centroid_cam,
                    self._last_detected_orientation_cam,
                    depth_frame=snap.depth_frame,
                    stamp=snap.stamp,
                )
            else:
                self._clear_detected_object_pose()
        elif allow_state_updates:
            self._clear_detected_object_pose()

        if mask_target is not None:
            u, v, depth = mask_target
            source = 'mask'
        else:
            u = best['u_bbox']
            v = best['v_bbox']
            depth = self.get_depth_roi_median(u, v, depth_image=snap.depth)

        if depth is None:
            if publish_debug:
                cv2.putText(
                    annotated,
                    'No valid depth',
                    (best['x1'], max(0, best['y1'] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                self.publish_debug_image(annotated)
            return None

        point_cam = self.pixel_to_point_camera(u, v, depth)

        if point_cam is None:
            if publish_debug:
                self.publish_debug_image(annotated)
            return None

        point_base = self._camera_grasp_target_to_planning_frame(
            point_cam,
            snap.depth_frame,
            stamp=snap.stamp,
        )

        if point_base is None:
            if publish_debug:
                self.publish_debug_image(annotated)
            return None

        if self.reject_targets_below_floor and float(point_base[2]) < self.floor_z_min:
            if publish_debug:
                cv2.putText(
                    annotated,
                    f'Target below floor z={point_base[2]:.3f}',
                    (best['x1'], max(20, best['y1'] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                self.publish_debug_image(annotated)

            if allow_state_updates:
                self.get_logger().warning(f'Rejecting target below floor threshold: '
                    f'z={point_base[2]:.3f} < floor_z_min={self.floor_z_min:.3f}', throttle_duration_sec=1.0)
            return None

        if publish_debug:
            cv2.circle(annotated, (u, v), 6, (0, 255, 255), -1)
            cv2.putText(
                annotated,
                f'{best["name"]} {best["conf"]:.2f} {source} depth={depth:.3f}m',
                (best['x1'], max(20, best['y1'] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            # Orientation overlay: draw detected probe long axis (orange) +
            # gripper approach direction (green arrow) on the debug image.
            if self._last_detected_orientation_cam is not None:
                long_cam = self._last_detected_orientation_cam[:, 0]  # long axis in cam
                dx2d = float(long_cam[0])
                dy2d = float(long_cam[1])
                mag2d = math.sqrt(dx2d * dx2d + dy2d * dy2d)
                if mag2d > 0.01:
                    dx2d /= mag2d
                    dy2d /= mag2d
                    arrow_len = 70
                    # Orange double-headed line = probe long axis
                    p1 = (int(u - dx2d * arrow_len), int(v - dy2d * arrow_len))
                    p2 = (int(u + dx2d * arrow_len), int(v + dy2d * arrow_len))
                    cv2.line(annotated, p1, p2, (0, 165, 255), 2)
                    # Green arrow = gripper approach direction (perpendicular)
                    gx, gy = -dy2d, dx2d
                    g1 = (int(u - gx * arrow_len), int(v - gy * arrow_len))
                    g2 = (int(u + gx * arrow_len), int(v + gy * arrow_len))
                    cv2.arrowedLine(annotated, g1, g2, (0, 255, 0), 2, tipLength=0.3)
            self.publish_debug_image(annotated)

        if allow_state_updates:
            self.get_logger().info(f'Detection [{source}]: x={point_base[0]:.3f} '
                f'y={point_base[1]:.3f} z={point_base[2]:.3f} '
                f'conf={best["conf"]:.2f}', throttle_duration_sec=1.0)

        return point_base, best['name'], best['conf']

    def _update_live_track(
        self,
        detection: Optional[Tuple[np.ndarray, str, float]]
    ) -> None:
        """
        Live Tracking Agent:
        Keep tracking the probe even while the arm is busy.
        """
        if detection is None:
            return

        if self._perception_updates_forbidden_now():
            self.get_logger().info(f'Perception freeze active during {self.sequence_stage}: '
                'ignoring live YOLO/depth target update.', throttle_duration_sec=1.0)
            return

        point_base, name, conf = detection

        point_base = self._apply_probe_shape_aware_target_correction(
            point_base,
            self.sequence_locked_target_point_base,
            self.sequence_stage,
        )

        self.live_target_point_base = point_base
        self.live_target_stamp_sec = self._now_sec()
        self.last_detection_name = name
        self.last_detection_conf = conf

        if not self.busy or self.current_target_point_base is None:
            return

        if self.sequence_stage in stages.LIVE_TRACK_LOCKED_STAGES:
            return

        moved = float(np.linalg.norm(point_base - self.current_target_point_base))

        if moved > self.replan_target_move_threshold_m:
            if self.sequence_stage == 'move_pre_grasp' and self.ignore_live_replan_during_pregrasp:
                # The wrist camera is moving during pre-grasp. Apparent target motion
                # is usually projection/depth drift, not real probe motion. Verify after arrival.
                self.get_logger().warning(f'Apparent live target shift {moved:.3f}m during move_pre_grasp; '
                    f'not aborting. Will verify/refine at pre-grasp.', throttle_duration_sec=0.7)
                return

            self.pending_replan_after_motion = True
            self.get_logger().warning(f'Live target moved {moved:.3f}m during {self.sequence_stage}; '
                f'will replan before final grasp.', throttle_duration_sec=0.7)

    def _target_recent_enough(self) -> bool:
        if self.live_target_point_base is None:
            return False

        return (
            self._now_sec() - self.live_target_stamp_sec
        ) <= self.tracking_lost_timeout_sec

    def detect_and_maybe_grasp(self) -> None:
        now_sec = self._now_sec()
        if self._rover_motion_active():
            self.get_logger().warning('Vision grasp paused because rover is moving.', throttle_duration_sec=1.0)
            return

        if self.paused_after_failure:
            if now_sec < self.blocked_until_sec:
                return
            self.paused_after_failure = False
            self.get_logger().info('Failure lockout expired; auto-grasp may acquire a new stable target.')

        if now_sec < self.blocked_until_sec:
            return

        # Prevent loop after successful grasp.
        if self.task_complete and not self.auto_restart_after_success:
            return

        if self.task_complete and now_sec < self.success_until_sec:
            return

        if self._perception_updates_forbidden_now():
            self.detect_target_once(publish_debug=True, allow_state_updates=False)
            return

        # During explicit pre-grasp refinement, the refinement timer owns YOLO.
        # Running the normal detector simultaneously slows the node and can keep
        # the supervisor in a busy loop.
        if self.busy and self.sequence_stage == 'refine':
            return

        detection = self.detect_target_once(publish_debug=True)

        if self.continuous_tracking_enabled:
            self._update_live_track(detection)

        if self.busy:
            return

        if detection is None:
            self._expire_target_stability_history(now_sec)
            return

        point_base, name, conf = detection

        if not self.is_target_stable(point_base, conf):
            self.get_logger().info(
                f'Target seen but waiting for stable filtered 3D position '
                f'({len(self.target_history)}/{self.target_stability_samples} samples, '
                f'max residual={self.target_filter_max_residual_m*1000.0:.1f} mm, '
                f'RMS={self.target_filter_rms_m*1000.0:.1f} mm)...',
                throttle_duration_sec=1.0,
            )
            return

        if self.filtered_target_point_base is None:
            return
        point_base = self.filtered_target_point_base.copy()
        conf = self.filtered_target_confidence
        locked_target_point = point_base.copy()
        locked_object_axis = self._get_detected_object_long_axis_base()

        self.current_target_point_base = point_base
        self.live_target_point_base = point_base
        self.live_target_stamp_sec = self._now_sec()
        self.last_detection_name = name
        self.last_detection_conf = conf
        self.pending_replan_after_motion = False
        self.replan_count = 0
        self._cartesian_grasp_retries = 0

        # New grasp task starts here, but no physical grasp attempt has happened yet.
        # Attempt count must increase only when send_grasp() is actually called.
        self.grasp_attempt_count = 0
        self.grasp_depth_below_surface_m = self.base_grasp_depth_below_surface_m
        self.retry_target_from_lift_check = None
        self.gripper_contact_detected = False
        self.last_gripper_actual = None
        self.last_gripper_target = None
        self._lift_floor_fail_count = 0

        self.task_complete = False
        self.holding_object = False

        self.busy = True
        self._new_sequence()
        self.sequence_locked_target_point_base = locked_target_point
        self.sequence_locked_object_long_axis_base = locked_object_axis
        self.sequence_stage = 'open_gripper'

        # Gripper Sizing Agent: compute optimal close angle from detected object width.
        if self.adaptive_gripper_enabled:
            width_for_grasp = (
                self._last_detected_width_m
                if self._last_detected_width_m is not None
                else self.nominal_probe_width_m
            )
            self.computed_gripper_close, self.computed_gripper_preclose = \
                self._compute_adaptive_gripper_close(width_for_grasp)
            self.last_estimated_object_width_m = width_for_grasp
            if self._last_detected_width_m is None:
                self.get_logger().warning(
                    'No reliable 3D width estimate at target lock; using '
                    f'nominal_probe_width_m={width_for_grasp*1000.0:.1f} mm.'
                )
        else:
            self.computed_gripper_close = self.gripper_close
            self.computed_gripper_preclose = self.gripper_preclose
            self.last_estimated_object_width_m = None

        # Four-Bar Contact Point Compensation Agent:
        # Update the effective offset so the arm positions the object correctly
        # at the bucket contact surface for whatever closing angle is needed.
        self._apply_fourbar_contact_offset(self.computed_gripper_close)

        # 6D Pose Agent — Yaw: compute gripper approach yaw from object orientation.
        self.detected_object_yaw_rad = None
        if self.object_yaw_align_enabled and self._last_detected_orientation_cam is not None:
            self.detected_object_yaw_rad = self._compute_grasp_yaw_from_object(
                self._last_detected_orientation_cam
            )

        self.get_logger().info(f'Stable target acquired in {self.planning_frame}: '
            f'x={point_base[0]:.3f}, y={point_base[1]:.3f}, '
            f'z={point_base[2]:.3f}, median_conf={conf:.2f}, '
            f'max_residual={self.target_filter_max_residual_m*1000.0:.1f}mm, '
            f'RMS={self.target_filter_rms_m*1000.0:.1f}mm')

        self.start_grasp_sequence()

    def start_grasp_sequence(self) -> None:
        if self.current_target_point_base is None:
            self.reset_sequence('No target point available.')
            return
        orientation = None
        retry_orientation_valid = (
            self.preserve_orientation_across_pregrasp_retries
            and self._retry_grasp_orientation is not None
            and self._retry_grasp_target is not None
            and self._now_sec() <= self._retry_grasp_orientation_until_sec
            and float(np.linalg.norm(
                self.current_target_point_base - self._retry_grasp_target
            )) <= self.pregrasp_retry_target_radius_m
        )
        if retry_orientation_valid:
            orientation = self._retry_grasp_orientation
            self.get_logger().info(
                'Reusing the original locked grasp orientation after a pre-grasp retry; '
                'the moving wrist-camera PCA yaw will not replace it.'
            )
        else:
            self._retry_grasp_orientation = None
            self._retry_grasp_target = None
            self._retry_grasp_orientation_until_sec = 0.0
            orientation = self.choose_target_orientation()
        if orientation is None:
            self.reset_sequence('Could not determine tool orientation.')
            return
        self.grasp_orientation = orientation
        # Capture wrist joint position now; lock it during the free-space pre-grasp move
        if self.lock_wrist_joint:
            self.sequence_wrist_value = self.current_joint_positions.get(self.lock_wrist_joint_name)
            if self.sequence_wrist_value is None:
                self.get_logger().warning(f'Joint "{self.lock_wrist_joint_name}" not found in /joint_states; wrist lock disabled for this sequence.')
        target = self.current_target_point_base
        self.update_contact_poses_from_target(target, orientation)

        pre_grasp_point = self._pose_xyz(self.pre_grasp_pose)
        grasp_point = self._pose_xyz(self.grasp_pose)
        self.publish_markers()
        self.get_logger().info('Grasp plan | target=(%.3f, %.3f, %.3f) pre=(%.3f, %.3f, %.3f) grasp=(%.3f, %.3f, %.3f) '
            'offset_in_link=(%.3f, %.3f, %.3f) use_ori=%s' % (
                target[0], target[1], target[2],
                pre_grasp_point[0], pre_grasp_point[1], pre_grasp_point[2],
                grasp_point[0], grasp_point[1], grasp_point[2],
                self.target_point_offset_in_link[0], self.target_point_offset_in_link[1], self.target_point_offset_in_link[2],
                str(self.use_orientation_constraint),
            ))
        self.get_logger().info('Effective offset (four-bar compensated): (%.3f, %.3f, %.3f)' % (
                self.effective_target_point_offset_in_link[0],
                self.effective_target_point_offset_in_link[1],
                self.effective_target_point_offset_in_link[2],
            ))
        self.command_gripper_and_then(
            self.gripper_open,
            self.send_pre_grasp,
            stage_name='open_gripper',
            description='open before approach'
        )

    def handle_pregrasp_arrival(self) -> None:
        """Finalize pre-grasp exactly once, then refine/descend.

        Live tracking is useful while travelling to pre-grasp, but it must not
        become an endless loop. When this method is called, the arm is either at
        pre-grasp according to MoveIt or the watchdog decided it is close enough
        / timed out. We may use the latest live target to refresh geometry, but
        we do not keep sending new pre-grasp goals indefinitely.
        """
        if self.current_target_point_base is None or self.grasp_orientation is None:
            self.reset_sequence('Pre-grasp arrived but target/orientation is missing.')
            return

        # Stop the pre-grasp watchdog; we are committing to finalization now.
        if getattr(self, '_pregrasp_watchdog_timer', None) is not None:
            try:
                self._pregrasp_watchdog_timer.cancel()
            except Exception:
                pass
            self._pregrasp_watchdog_timer = None

        auto_calibration_applied = self._commit_pending_camera_offset_calibration()

        used_live_update = auto_calibration_applied
        moved = 0.0
        recent = False
        age = 999.0
        live_target_point = self.live_target_point_base

        if live_target_point is not None:
            live_target_point = self._apply_probe_shape_aware_target_correction(
                live_target_point,
                self.sequence_locked_target_point_base,
                'pregrasp_finalization',
            )
            age = self._now_sec() - self.live_target_stamp_sec
            moved = float(np.linalg.norm(live_target_point - self.current_target_point_base))
            recent = age <= self.pregrasp_recent_target_max_age_sec

        if recent and not self.use_recent_live_target_after_pregrasp:
            self.get_logger().info('Pre-grasp live correction disabled; using the originally locked probe center.')
        elif recent:
            # Use live feedback at pre-grasp as an advisory final correction,
            # but do not create another endless pre-grasp orbit. Small drift is
            # accepted directly; impossible large jumps are ignored.
            if moved <= self.pregrasp_live_update_accept_m:
                if moved > 0.003:
                    self.current_target_point_base = live_target_point.copy()
                    used_live_update = True
                    self.get_logger().info(f'Pre-grasp final live correction committed once: moved={moved:.3f}m age={age:.2f}s. '
                        'No more pre-grasp replans will be sent; next step is bounded refinement/final descent.')
            else:
                self.get_logger().warning(f'Pre-grasp live jump {moved:.3f}m is larger than accept limit '
                    f'{self.pregrasp_live_update_accept_m:.3f}m; ignoring it and using locked target.')
        elif not self.continue_if_live_target_stale_after_pregrasp:
            self.reset_sequence(f'Pre-grasp live track is stale: age={age:.2f}s.')
            return

        self.pending_replan_after_motion = False
        self._pregrasp_force_finalize = True

        if not self._refresh_grasp_geometry_from_latest_estimates(
            'pregrasp-final-live' if used_live_update else 'pregrasp-arrival'
        ):
            self.reset_sequence('Failed to compute final grasp geometry at pre-grasp.')
            return

        if self.refine_enabled:
            self._start_refine()
        else:
            self.preclose_before_grasp_then_send_grasp()

    def try_replan_from_live_target(self, reason: str) -> bool:
        """
        Motion Supervisor Agent:
        If the object moved after the plan was created, update target and replan.
        """
        if self.disable_live_replan_after_lock:
            if self.pending_replan_after_motion:
                self.get_logger().warning(f'Live replan suppressed during {reason}: target is locked until sequence completion.')
            self.pending_replan_after_motion = False
            return False

        if not self.pending_replan_after_motion:
            return False

        if self.live_target_point_base is None or not self._target_recent_enough():
            self.reset_sequence(
                f'Target moved during {reason}, but live track is stale/lost.'
            )
            return True

        if self.replan_count >= self.max_replans_per_grasp:
            self.reset_sequence(
                f'Target kept moving during {reason}; '
                f'exceeded max_replans_per_grasp={self.max_replans_per_grasp}.'
            )
            return True

        moved = float(
            np.linalg.norm(
                self.live_target_point_base - self.current_target_point_base
            )
        )

        if moved < self.replan_target_move_threshold_m:
            self.pending_replan_after_motion = False
            return False

        self.replan_count += 1
        self.pending_replan_after_motion = False

        self.current_target_point_base = self.live_target_point_base.copy()
        self.update_contact_poses_from_target(
            self.current_target_point_base,
            self.grasp_orientation,
        )

        self.publish_markers()

        self.get_logger().warning(f'Replanning pre-grasp from live target because {reason}: '
            f'moved={moved:.3f}m, '
            f'replan={self.replan_count}/{self.max_replans_per_grasp}')

        self.send_pre_grasp()
        return True

    def _start_pregrasp_watchdog(self) -> None:
        """Monitor pre-grasp motion and force finalization if MoveIt stays silent."""
        if not self.pregrasp_watchdog_enabled:
            return
        if getattr(self, '_pregrasp_watchdog_timer', None) is not None:
            try:
                self._pregrasp_watchdog_timer.cancel()
            except Exception:
                pass
        self._pregrasp_motion_start_sec = self._now_sec()
        seq = self.sequence_id
        self._pregrasp_watchdog_timer = self.create_timer(
            0.25,
            lambda seq=seq: self._pregrasp_watchdog_tick(seq)
        )

    def _cancel_active_moveit_goal(self) -> None:
        """Best-effort cancellation of a MoveIt goal before we proceed."""
        gh = getattr(self, '_active_move_goal_handle', None)
        if gh is None:
            self._clear_arm_motion_confirmation()
            return
        try:
            gh.cancel_goal_async()
            self.get_logger().warning('Requested cancellation of active arm motion goal.')
        except Exception as exc:
            self.get_logger().warning(f'Could not cancel active MoveIt goal cleanly: {exc}')
        self._active_move_goal_handle = None
        self._clear_arm_motion_confirmation()

    def _cancel_active_gripper_goal(self) -> None:
        """Best-effort cancellation and cleanup of the current gripper action."""
        goal_handle = getattr(self, '_gripper_action_goal_handle', None)
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
                self.get_logger().warning('Requested cancellation of active gripper action.')
            except Exception as exc:
                self.get_logger().warning(f'Could not cancel active gripper action cleanly: {exc}')
        self._gripper_action_goal_handle = None
        self._gripper_command_used_action = False
        self._gripper_action_accepted = False
        self._gripper_action_succeeded = False
        self._gripper_action_failed_reason = None

    def _pregrasp_watchdog_tick(self, expected_seq: int) -> None:
        if expected_seq != self.sequence_id:
            if getattr(self, '_pregrasp_watchdog_timer', None) is not None:
                self._pregrasp_watchdog_timer.cancel()
                self._pregrasp_watchdog_timer = None
            return
        if self.sequence_stage != 'move_pre_grasp':
            if getattr(self, '_pregrasp_watchdog_timer', None) is not None:
                self._pregrasp_watchdog_timer.cancel()
                self._pregrasp_watchdog_timer = None
            return

        elapsed = self._now_sec() - self._pregrasp_motion_start_sec
        if elapsed < self.pregrasp_watchdog_min_sec:
            return

        desired = self.contact_pose_to_link_pose(self.pre_grasp_pose) if self.pre_grasp_pose is not None else None
        current = self.get_current_link_pose_in_planning_frame()
        dist = None
        if desired is not None and current is not None:
            dist = math.sqrt(
                (float(current.position.x) - float(desired.position.x)) ** 2 +
                (float(current.position.y) - float(desired.position.y)) ** 2 +
                (float(current.position.z) - float(desired.position.z)) ** 2
            )

        near = dist is not None and dist <= self.pregrasp_link_arrival_tolerance_m
        timed_out = elapsed >= self.pregrasp_watchdog_timeout_sec

        if not near and not timed_out:
            return
        if timed_out and not self.pregrasp_watchdog_force_after_timeout and not near:
            return
        if self.arm_require_feedback_for_completion:
            self.get_logger().info('Pre-grasp watchdog sees the arm near its target, but strict whole-process '
                'confirmation is enabled; waiting for MoveIt success plus stable measured feedback '
                'before advancing.', throttle_duration_sec=1.0)
            return

        if getattr(self, '_pregrasp_watchdog_timer', None) is not None:
            self._pregrasp_watchdog_timer.cancel()
            self._pregrasp_watchdog_timer = None

        self._pregrasp_force_finalize = True
        self.pending_replan_after_motion = False
        # Stop treating late MoveIt result as authoritative. The watchdog is now
        # the owner of this transition and will call handle_pregrasp_arrival().
        self.sequence_stage = 'pregrasp_finalizing'
        self.get_logger().warning('Pre-grasp watchdog is finalizing the sequence: '
            f'elapsed={elapsed:.2f}s, link_dist={(dist if dist is not None else -1):.3f}m, '
            f'near={near}, timed_out={timed_out}. This prevents endless pre-grasp refinement.')
        self._cancel_active_moveit_goal()
        # Give cancel a short moment, then commit final pre-grasp geometry.
        self.call_later(0.20, self.handle_pregrasp_arrival)

    def send_pre_grasp(self) -> None:
        self.sequence_stage = 'move_pre_grasp'
        self._pregrasp_force_finalize = False
        # Use a large position sphere + orientation: IK sampler has plenty of freedom to find a
        # solution satisfying both, so OMPL won't time out (status 6). The arm arrives at pre-grasp
        # already aligned — joint 6 won't spin before the Cartesian approach stroke.
        self.send_pose_goal(self.pre_grasp_pose,
                            pos_tol=self.pre_grasp_position_tol,
                            with_orientation=True)
        self._start_pregrasp_watchdog()

    def preclose_before_grasp_then_send_grasp(self) -> None:
        """Commit final refined geometry and go to grasp with gripper open.

        User-requested behavior: no half-close/preclose.  The gripper remains
        open for the final approach.  The pre-grasp refinement calculates q_close,
        the final contact offset, and the predicted closing arc.  Then the arm
        descends once and the gripper closes in one go.
        """
        self.preclosed_in_air = False

        # Always use closed contact geometry for the final link pose because the
        # object should be centred at the end of the one-go close.
        self._apply_fourbar_contact_offset(self.computed_gripper_close)

        if self.current_target_point_base is not None and self.grasp_orientation is not None:
            self.update_contact_poses_from_target(self.current_target_point_base, self.grasp_orientation)
            self.publish_markers()
            self._log_committed_grasp_geometry('final-before-descent')

        self.get_logger().info('No preclose will be used. Gripper stays open during final approach; '
            'after reaching grasp pose it closes once using the refined four-bar geometry.')
        self.send_grasp()

    def send_grasp(self) -> None:
        """
        Cartesian straight-line approach.

        A grasp attempt is counted only here, because this is the moment when
        the arm physically tries to insert the gripper around the probe.
        """
        if self.grasp_pose is None:
            self.reset_sequence('No grasp pose available.')
            return

        self.grasp_attempt_count += 1

        self.get_logger().info(f'Starting physical grasp attempt '
            f'{self.grasp_attempt_count}/{self.max_grasp_attempts}')

        self.sequence_stage = 'move_grasp'
        self._send_cartesian_path([self.contact_pose_to_link_pose(self.grasp_pose)])

    # ------------------------------------------------------------------
    # Visual refinement: collect close-range detections after pre-grasp
    # ------------------------------------------------------------------
    def _start_refine(self) -> None:
        """Begin collecting close-range frames to refine grasp_pose."""
        # Important: refinement is explicitly allowed at pre-grasp.  The older
        # code checked the perception firewall while the stage was still
        # 'pregrasp_finalizing', so refinement was skipped every time and the
        # node went directly to final descent with unrefined geometry.
        self.sequence_stage = 'refine'

        if self._perception_updates_forbidden_now():
            self.get_logger().info('Close-range refinement requested but perception is still frozen; using current committed target.')
            self.preclose_before_grasp_then_send_grasp()
            return

        self._refine_buffer = []
        self._refine_width_buffer = []
        self._refine_orientation_cam_last = None
        self._refine_start_sec = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info('Pre-grasp refinement active: collecting close-range YOLO/depth samples, '
            'then calculating one-go close contact point and four-bar closing arc.')
        # Poll at the same rate as detect_period_sec
        self._refine_timer = self.create_timer(self.detect_period_sec, self._refine_tick)

    def _refine_tick(self) -> None:
        """Called each detection period during refinement."""
        if self.sequence_stage != 'refine':
            self._refine_timer.cancel()
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._refine_start_sec > self.refine_timeout_sec:
            self._refine_timer.cancel()
            if self.refine_commit_on_timeout and len(self._refine_buffer) >= self.refine_min_samples_to_accept and self._refine_buffer:
                refined_target = np.mean(self._refine_buffer, axis=0)
                if self.is_refined_target_acceptable(refined_target):
                    self.current_target_point_base = refined_target
                    if self._refine_width_buffer:
                        self._last_detected_width_m = float(np.median(self._refine_width_buffer))
                    if self._refine_orientation_cam_last is not None:
                        self._last_detected_orientation_cam = self._refine_orientation_cam_last
                    self._refresh_grasp_geometry_from_latest_estimates('pregrasp-refine-timeout-commit')
                    self.get_logger().warning(f'Refinement timed out after {self.refine_timeout_sec}s with '
                        f'{len(self._refine_buffer)} sample(s); committing bounded average and proceeding.')
                else:
                    self.get_logger().warning(f'Refinement timed out after {self.refine_timeout_sec}s; average sample rejected. '
                        'Using locked target and proceeding.')
            else:
                self.get_logger().warning(f'Refinement timed out after {self.refine_timeout_sec}s '
                    f'({len(self._refine_buffer)} samples). Using original grasp pose and proceeding.')
            self.preclose_before_grasp_then_send_grasp()
            return
        if self.camera_info is None:
            return
        completed = self._poll_inference()
        if completed is None:
            return
        snap, results = completed
        annotated = self._annotate_yolo_results(snap.color, results)
        self._stamp_debug_status(annotated, 'Refine live view', color=(255, 255, 0))
        self.publish_debug_image(annotated)
        best = self._select_best_detection(
            results, self.refine_confidence_threshold, image_shape=snap.color.shape
        )

        point_base = None
        source = 'none'

        if best is not None:
            mask_target = None

            if best['mask'] is not None:
                w3d = self._estimate_object_width_3d(best['mask'], snap.depth)
                if w3d is not None and self.adaptive_gripper_min_width_m <= w3d <= self.adaptive_gripper_max_width_m:
                    self._refine_width_buffer.append(float(w3d))
                pose_estimate = self._estimate_object_orientation_3d(best['mask'], snap.depth)
                if pose_estimate is not None:
                    centroid_ref_cam, R_ref = pose_estimate
                    self._refine_orientation_cam_last = R_ref
                    self._update_detected_object_pose_from_camera(
                        centroid_ref_cam, R_ref,
                        depth_frame=snap.depth_frame, stamp=snap.stamp,
                    )

            if best['mask'] is not None:
                mask_target = self._mask_depth_target(
                    best['mask'],
                    # Keep refinement aligned with the initial center pick.
                    prefer_nearest=False,
                    depth_image=snap.depth,
                )

            if mask_target is not None:
                u, v, depth = mask_target
                source = 'mask'
            else:
                u = best['u_bbox']
                v = best['v_bbox']
                source = 'bbox'
                depth = self.get_depth_roi_median(
                    u,
                    v,
                    half_size_px=max(self.roi_half_size_px, 8),
                    min_depth_m=self.refine_min_depth_m,
                    max_depth_m=self.max_depth_m,
                    prefer_nearest=True,
                    depth_image=snap.depth,
                )

            if depth is not None:
                point_cam = self.pixel_to_point_camera(u, v, depth)

                if point_cam is not None:
                    point_base = self._camera_grasp_target_to_planning_frame(
                        point_cam,
                        snap.depth_frame,
                        stamp=snap.stamp,
                    )

        if point_base is None and self.refine_use_projection_fallback:
            source = 'projection_depth'
            point_base = self.projected_locked_target_refinement()

        if point_base is None:
            self.get_logger().info('Refinement: YOLO/depth target unavailable this frame; keeping previous lock.', throttle_duration_sec=0.5)
            return

        point_base = self._apply_probe_shape_aware_target_correction(
            point_base,
            self.sequence_locked_target_point_base,
            'refine',
        )

        if not self.is_refined_target_acceptable(point_base):
            moved = (
                float(np.linalg.norm(point_base - self.current_target_point_base))
                if self.current_target_point_base is not None
                else 0.0
            )

            if moved > self.replan_target_move_threshold_m and not self.disable_live_replan_after_lock:
                self.live_target_point_base = point_base
                self.live_target_stamp_sec = self._now_sec()
                self.pending_replan_after_motion = True
                self._refine_timer.cancel()
                self.try_replan_from_live_target('visual refinement')
            else:
                self.get_logger().warning('Rejected refinement was ignored completely; locked target remains unchanged.')

            return

        if self._refine_buffer:
            if np.linalg.norm(point_base - self._refine_buffer[-1]) > self.refine_max_jump_m:
                self.get_logger().info('Refinement: noisy frame discarded.', throttle_duration_sec=0.5)
                return
        self._refine_buffer.append(point_base)
        self.get_logger().info(f'Refinement sample {len(self._refine_buffer)}/{self.refine_samples} '
            f'[{source}]: x={point_base[0]:.3f} '
            f'y={point_base[1]:.3f} z={point_base[2]:.3f}')
        if len(self._refine_buffer) >= self.refine_samples:
            self._refine_timer.cancel()
            refined_target = np.mean(self._refine_buffer, axis=0)

            if not self.is_refined_target_acceptable(refined_target):
                self.get_logger().warning('Average refinement rejected. Using original locked target.')
                self.preclose_before_grasp_then_send_grasp()
                return

            self.current_target_point_base = refined_target
            if self._refine_width_buffer:
                self._last_detected_width_m = float(np.median(self._refine_width_buffer))
            if self._refine_orientation_cam_last is not None:
                self._last_detected_orientation_cam = self._refine_orientation_cam_last

            if not self._refresh_grasp_geometry_from_latest_estimates('pregrasp-refined'):
                self.reset_sequence('Failed to compute final grasp geometry after pre-grasp refinement.')
                return

            self.get_logger().info(f'Refined grasp target accepted: x={refined_target[0]:.3f} '
                f'y={refined_target[1]:.3f} z={refined_target[2]:.3f}. '
                'Final close will be one gripper command; no preclose will be used.')
            self.preclose_before_grasp_then_send_grasp()

    def close_gripper_and_retreat(self) -> None:
        """Close directly once after final grasp approach; no preclose stage."""
        self.gripper_contact_detected = False
        self._lift_floor_fail_count = 0
        self.locked_target_before_lift = (
            self.current_target_point_base.copy()
            if self.current_target_point_base is not None
            else None
        )
        self._lift_check_last_nonlifted_target = None
        settle_sec = max(0.0, float(self.final_grasp_arm_settle_sec))
        if settle_sec > 0.0:
            self.sequence_stage = 'preclose_in_air'
            self.get_logger().info(f'Final grasp motion result received. Waiting {settle_sec:.2f}s for the arm to settle, '
                'then verifying the TCP is actually at the committed grasp pose before closing. '
                'All perception refinement and arm replanning remain frozen.')
            self.call_later(settle_sec, self._begin_final_grasp_pose_check)
            return

        self._begin_final_grasp_pose_check()

    def _begin_final_grasp_pose_check(self) -> None:
        if not self.final_grasp_pose_check_enabled:
            self.get_logger().warning('Final grasp pose check is disabled; closing gripper without TCP verification.')
            self.final_close_gripper()
            return

        self.sequence_stage = 'verify_final_grasp_pose'
        self._final_grasp_pose_check_start_sec = self._now_sec()
        self._cancel_final_grasp_pose_check_timer()
        self.get_logger().info(f'Verifying final grasp TCP before close: '
            f'pos_tol={self.final_grasp_pose_position_tolerance_m*1000:.1f}mm, '
            f'ori_tol={math.degrees(self.final_grasp_pose_orientation_tolerance_rad):.1f}deg, '
            f'timeout={self.final_grasp_pose_check_timeout_sec:.1f}s.')
        self._final_grasp_pose_check_timer = self.create_timer(
            self.final_grasp_pose_check_period_sec,
            self._final_grasp_pose_check_tick,
        )
        self._final_grasp_pose_check_tick()

    def _final_grasp_pose_check_tick(self) -> None:
        if self.sequence_stage != 'verify_final_grasp_pose':
            self._cancel_final_grasp_pose_check_timer()
            return

        measured = self._log_final_grasp_pose_error(throttle_duration_sec=0.5)
        elapsed = self._now_sec() - self._final_grasp_pose_check_start_sec

        if measured is not None:
            _delta, pos_err, ori_err = measured
            if (
                pos_err <= self.final_grasp_pose_position_tolerance_m
                and ori_err <= self.final_grasp_pose_orientation_tolerance_rad
            ):
                self._cancel_final_grasp_pose_check_timer()
                self.get_logger().info(f'Final grasp pose check PASSED: '
                    f'pos={pos_err*1000:.1f}mm, ori={math.degrees(ori_err):.1f}deg. '
                    'Closing gripper in one go; arm replanning remains frozen.')
                self.final_close_gripper()
                return

        if elapsed < self.final_grasp_pose_check_timeout_sec:
            return

        self._cancel_final_grasp_pose_check_timer()
        if measured is None:
            reason = (
                'Final grasp pose check failed: could not read current TCP TF before close. '
                'Refusing to close the gripper without a verified arm pose.'
            )
        else:
            delta, pos_err, ori_err = measured
            reason = (
                f'Final grasp pose check failed: TCP is still '
                f'dxyz=({delta[0]*1000:.1f},{delta[1]*1000:.1f},{delta[2]*1000:.1f})mm '
                f'from the committed grasp pose '
                f'(pos={pos_err*1000:.1f}mm, ori={math.degrees(ori_err):.1f}deg). '
                'Refusing to close the gripper above/beside the probe.'
            )
        self._halt_after_final_pose_check_failure(reason)

    def final_close_gripper(self) -> None:
        """Final close: one gripper command, arm frozen."""
        end_q = float(self.computed_gripper_close)

        if self.close_in_one_go_after_pregrasp_refine or self.fourbar_final_close_steps <= 1:
            self.get_logger().info(f'Starting one-go final gripper close to q={end_q:.5f}. '
                'No arm motion, no refinement, no live replan is allowed during this command.')
            self.command_gripper_and_then(
                end_q,
                self.after_gripper_closed,
                stage_name='close_gripper',
                description='one-go final close after pre-grasp refinement'
            )
            return

        # Optional legacy slow close, normally disabled.
        start_q = self.current_joint_positions.get(self.gripper_joint_name, self.gripper_open)
        start_q = float(start_q)
        self._close_step_targets = [
            float(v) for v in np.linspace(start_q, end_q, self.fourbar_final_close_steps + 1)[1:]
        ]
        self._close_step_index = 0
        self.get_logger().info(f'Starting fallback stepped close: {len(self._close_step_targets)} steps.')
        self._command_next_close_step()

    def _command_next_close_step(self) -> None:
        if self._close_step_index >= len(self._close_step_targets):
            self.after_gripper_closed()
            return

        q = self._close_step_targets[self._close_step_index]
        i = self._close_step_index + 1
        n = len(self._close_step_targets)
        self._close_step_index += 1

        def next_step() -> None:
            if self.fourbar_final_close_step_wait_sec > 0.0:
                self.call_later(self.fourbar_final_close_step_wait_sec, self._command_next_close_step)
            else:
                self._command_next_close_step()

        self.command_gripper_and_then(
            q,
            next_step,
            stage_name='close_gripper',
            description=f'stationary four-bar close step {i}/{n}'
        )

    def after_gripper_closed(self) -> None:
        self.sequence_stage = 'verify_gripper'

        self.get_logger().info('Gripper close phase finished; holding briefly before lift-check.')

        if self.hold_after_close_no_motion:
            self.holding_object = True
            self.task_complete = True
            self.sequence_stage = 'done_holding'
            self.success_until_sec = self._now_sec() + self.success_lockout_sec
            self.get_logger().info('Hold-after-close enabled: no lift, retreat, or home motion will be sent. '
                'This verifies that the arm stays fixed during and after final close.')
            return

        self.holding_object = True

        if self.verify_grasp_after_lift:
            self.call_later(
                self.close_gripper_extra_wait_sec,
                self.start_lift_verification
            )
            return

        if self.post_grasp_lift_then_pick_home:
            self.get_logger().info('Post-grasp transport enabled without lift-check: gripper will stay closed; '
                'arm will move to pick_home through MoveGroup collision checking.')
            self.call_later(
                self.close_gripper_extra_wait_sec,
                self.send_post_grasp_vertical_lift
            )
            return

        self.call_later(
            self.close_gripper_extra_wait_sec,
            self.send_retreat
        )

    def _after_lift_verification_success(self, reason: str) -> None:
        self.holding_object = True
        if self.post_grasp_lift_then_pick_home:
            self.get_logger().info(f'{reason} Proceeding to collision-aware pick_home transport with the gripper closed.')
            self.send_post_grasp_vertical_lift()
            return

        self.get_logger().info(f'{reason} Retreating through collision-aware Cartesian/MoveGroup motions.')
        self.send_retreat()

    def _hold_closed_after_failed_grasp_check(self, reason: str) -> None:
        """Stop arm transport after a failed grasp check without opening the gripper."""
        self.holding_object = True
        self.task_complete = True
        self.busy = True
        self.sequence_stage = 'grasp_check_failed_holding'
        self.success_until_sec = self._now_sec() + self.success_lockout_sec
        self.get_logger().error(f'{reason} Grasp check failed, so the gripper will remain closed and '
            'no pick_home/retreat/open command will be sent automatically.')

    # ------------------------------------------------------------------ #
    #   Post-grasp lift-to-pick-home transport                             #
    # ------------------------------------------------------------------ #

    def send_post_grasp_vertical_lift(self) -> None:
        """Post-grasp transport: attach probe collision mesh then plan direct to pick_home.

        No floor collision plane is added — the arm starts AT floor level
        during a floor-probe grasp, so a floor slab would put the gripper
        inside a collision object, causing MoveGroup to reject the start state
        and immediately fail (which manifested as an instant gripper-open / retry
        loop).  The OMPL planner is trusted to find a safe upward path using
        the real robot collision model.
        """
        self.get_logger().info('[PostGrasp] Attaching probe mesh and planning to pick_home.')

        # Attach STL mesh so MoveGroup knows the gripper is holding an object.
        self._attach_probe_object()

        # Wait 500 ms for the planning scene monitor to register the attached
        # object before sending the joint goal.
        self.call_later(0.5, self._post_grasp_collision_scene_ready)

    def _post_grasp_collision_scene_ready(self) -> None:
        """Called 500 ms after collision objects were published; triggers pick_home.

        call_later() already guards against stale sequence_ids, so by the time
        this fires we know it belongs to the current grasp cycle.  We only
        bail out if the arm is already in 'move_pick_home' (double-call guard)
        or in 'idle'/'done_holding' (sequence was reset externally).
        """
        done_stages = ('idle', 'done_holding', 'move_pick_home')
        if self.sequence_stage in done_stages:
            return
        self.get_logger().info('[PostGrasp] Collision scene ready. Sending pick_home joint goal via MoveGroup.')
        self.send_pick_home_closed()

    def send_pick_home_closed(self) -> None:
        """Move arm joints to pick_home while leaving the gripper untouched/closed.

        Uses a longer planning time (post_grasp_planning_time_sec) and more
        planning attempts than the default so that OMPL can find a path out of
        the near-floor configuration without self-collisions.
        """
        if len(self.pick_home_joint_names) != len(self.pick_home_joint_positions):
            reason = 'pick_home_joint_names and pick_home_joint_positions length mismatch.'
            if self.holding_object:
                self._hold_closed_after_transport_failure(reason)
            else:
                self.reset_sequence(reason)
            return
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self._hold_closed_after_transport_failure(
                'MoveIt action server unavailable for held-probe transport to pick_home.'
            )
            return
        self.sequence_stage = 'move_pick_home'
        self.get_logger().info(f'Moving to pick_home (gripper closed). '
            f'MoveGroup planning time={self.post_grasp_planning_time_sec:.1f} s, '
            f'live-joint seeded start state.')
        self.send_joint_goal(
            self.pick_home_joint_names,
            self.pick_home_joint_positions,
            planning_time_override=self.post_grasp_planning_time_sec,
            num_attempts_override=max(self.num_planning_attempts, 15),
        )

    def send_base_box_drop_closed(self) -> None:
        """Move to the configured base-box release pose with the gripper closed."""
        if self.base_box_drop_use_pose:
            if not self._base_box_drop_pose_config_valid():
                self._hold_closed_after_transport_failure(
                    'Invalid base-box pose: frame must be non-empty and XYZ/RPY must each contain three values.'
                )
                return
            if len(self.base_box_drop_target_point_offset_in_link) != 3:
                self._hold_closed_after_transport_failure(
                    'base_box_drop_target_point_offset_in_link must contain exactly three values.'
                )
                return
        elif len(self.base_box_drop_joint_names) != len(self.base_box_drop_joint_positions):
            self._hold_closed_after_transport_failure(
                'base_box_drop_joint_names and base_box_drop_joint_positions length mismatch.'
            )
            return
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self._hold_closed_after_transport_failure(
                'MoveIt action server unavailable for held-probe transport to the base box.'
            )
            return

        self.sequence_stage = 'move_base_box_drop'
        if self.base_box_drop_use_pose:
            pose = self.get_base_box_drop_pose()
            self.get_logger().info(f'Moving held probe to base-box release pose: '
                f'frame={self.base_box_drop_frame}, xyz={self.base_box_drop_xyz}, '
                f'rpy={self.base_box_drop_rpy}.')
            self.send_pose_goal(
                pose,
                pos_tol=self.base_box_drop_position_tolerance_m,
                with_orientation=True,
                orientation_override=pose.pose.orientation,
                orientation_tol=self.base_box_drop_orientation_tolerance_rad,
                target_point_offset=self.base_box_drop_target_point_offset_in_link,
                arm_joints_only_start_state=True,
                planning_time_override=self.base_box_planning_time_sec,
                num_attempts_override=max(self.num_planning_attempts, 15),
            )
        else:
            self.get_logger().info(f'Moving held probe from pick_home to the base-box joint posture. '
                f'MoveGroup planning time={self.base_box_planning_time_sec:.1f} s.')
            self.send_joint_goal(
                self.base_box_drop_joint_names,
                self.base_box_drop_joint_positions,
                planning_time_override=self.base_box_planning_time_sec,
                num_attempts_override=max(self.num_planning_attempts, 15),
            )

    def _base_box_drop_pose_config_valid(self) -> bool:
        return bool(self.base_box_drop_frame) and len(self.base_box_drop_xyz) == 3 and len(self.base_box_drop_rpy) == 3

    def get_base_box_drop_pose(self) -> PoseStamped:
        """Return the configured physical release-point pose for planning/markers."""
        pose = PoseStamped()
        pose.header.frame_id = self.base_box_drop_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(
            x=float(self.base_box_drop_xyz[0]),
            y=float(self.base_box_drop_xyz[1]),
            z=float(self.base_box_drop_xyz[2]),
        )
        pose.pose.orientation = rpy_to_quat(
            float(self.base_box_drop_rpy[0]),
            float(self.base_box_drop_rpy[1]),
            float(self.base_box_drop_rpy[2]),
        )
        return pose

    def release_probe_in_base_box(self) -> None:
        """Open only after MoveIt confirms that the drop posture was reached."""
        self.get_logger().info('Base-box drop posture reached; releasing the probe.')
        self.command_gripper_and_then(
            self.gripper_open,
            self._after_probe_released_in_base_box,
            stage_name='release_in_base_box',
            description='release probe in rover base box',
        )

    def _after_probe_released_in_base_box(self) -> None:
        self.holding_object = False

        # The physical probe is now supported by the box rather than the tool.
        # Remove the attached planning-scene object before planning the empty-arm
        # return motion, otherwise MoveIt would continue carrying a ghost probe.
        self._remove_post_grasp_collision_objects()
        self.get_logger().info('Probe released in the rover base box.')

        if not self.return_pick_home_after_base_box_place:
            self.finish_placement_successfully(returned_home=False)
            return

        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.finish_placement_successfully(
                returned_home=False,
                warning='Probe was placed, but MoveIt is unavailable for the empty-arm return to pick_home.',
            )
            return

        self.sequence_stage = 'move_pick_home_after_place'
        self.send_joint_goal(
            self.pick_home_joint_names,
            self.pick_home_joint_positions,
            planning_time_override=self.post_grasp_planning_time_sec,
            num_attempts_override=max(self.num_planning_attempts, 15),
        )

    def _hold_closed_after_transport_failure(self, reason: str) -> None:
        """Stop safely if transport fails while the probe is still grasped."""
        self.holding_object = True
        self.task_complete = True
        self.busy = True
        self.sequence_stage = 'transport_failed_holding'
        self.success_until_sec = self._now_sec() + self.success_lockout_sec
        self.get_logger().error(f'{reason} The gripper remains closed; no release or automatic restart will occur.')

    def _stop_after_uncertain_base_box_release(self, reason: str) -> None:
        """Lock the task when gripper feedback cannot confirm box release."""
        self.task_complete = True
        self.busy = True
        self.sequence_stage = 'base_box_release_unconfirmed'
        self.success_until_sec = self._now_sec() + self.success_lockout_sec
        self.get_logger().error(f'{reason} The arm will remain at the box and automatic restart is disabled; '
            'inspect whether the probe was released before sending another command.')

    # ------------------------------------------------------------------ #
    #   Lift-check verification                                            #
    # ------------------------------------------------------------------ #

    def send_lift_check(self) -> None:
        """Move straight up by lift_check_distance_m to see if the probe came with us."""
        if self.grasp_pose is None:
            self.send_retreat()
            return

        grasp_xyz = self._pose_xyz(self.grasp_pose)
        lift_xyz = grasp_xyz.copy()
        lift_xyz[2] += self.lift_check_distance_m
        lift_pose = self.make_pose(lift_xyz, self.grasp_orientation)

        if self.locked_target_before_lift is None:
            self.locked_target_before_lift = grasp_xyz.copy()

        self.sequence_stage = 'move_lift_check'
        self._lift_floor_fail_count = 0
        self._lift_check_last_nonlifted_target = None
        self._send_cartesian_path([self.contact_pose_to_link_pose(lift_pose)])

    def start_lift_verification(self) -> None:
        """After lift motion succeeds, wait briefly for a detection that matches lifted position."""
        self.send_lift_check()

    def _lift_verification_tick(self) -> None:
        """
        Lift Verification Agent.

        Uses fresh YOLO26-seg mask detection after lift.
        Does not release the probe after one uncertain failure.
        """
        if self.sequence_stage != 'verify_lift':
            if getattr(self, '_lift_check_timer', None) is not None:
                try:
                    self._lift_check_timer.cancel()
                except Exception:
                    pass
                self._lift_check_timer = None
            return

        elapsed = self._now_sec() - self._lift_check_start_sec

        detection = self.detect_target_once(publish_debug=True)

        if detection is not None and self.locked_target_before_lift is not None:
            point_base, name, conf = detection

            dist_from_old_xy = float(
                np.linalg.norm(point_base[:2] - self.locked_target_before_lift[:2])
            )
            z_lift = float(point_base[2] - self.locked_target_before_lift[2])

            self.get_logger().info(f'Lift verification fresh detection: '
                f'dist_xy={dist_from_old_xy:.3f}m '
                f'z_lift={z_lift:.3f}m conf={conf:.2f} '
                f'contact={self.gripper_contact_detected}', throttle_duration_sec=0.5)

            if self.lift_check_require_positive_z_success:
                lifted_like = z_lift >= self.grasp_success_min_lift_m
                floor_like = not lifted_like
            else:
                floor_like = (
                    dist_from_old_xy < self.grasp_failure_same_place_radius_m
                    and z_lift < self.grasp_success_min_lift_m
                )
                lifted_like = (
                    dist_from_old_xy >= self.grasp_failure_same_place_radius_m
                    or z_lift >= self.grasp_success_min_lift_m
                )

            if lifted_like:
                self._lift_floor_fail_count = 0

                if getattr(self, '_lift_check_timer', None) is not None:
                    try:
                        self._lift_check_timer.cancel()
                    except Exception:
                        pass
                    self._lift_check_timer = None

                self.get_logger().info(f'Lift-check PASSED: probe moved/lifted '
                    f'(dist_xy={dist_from_old_xy:.3f}m, z_lift={z_lift:.3f}m).')

                self._after_lift_verification_success('Lift-check PASSED.')
                return

            if floor_like:
                self._lift_floor_fail_count += 1
                self._lift_check_last_nonlifted_target = point_base.copy()

                self.get_logger().warning(f'Lift-check floor-like detection '
                    f'{self._lift_floor_fail_count}/{self.lift_check_floor_fail_samples}: '
                    f'dist_xy={dist_from_old_xy:.3f}m z_lift={z_lift:.3f}m '
                    f'contact={self.gripper_contact_detected}. '
                    f'require_z_lift={self.lift_check_require_positive_z_success}')

                # If gripper contact was detected, do not open the gripper immediately.
                # This prevents the exact problem in your video: grasp then release.
                if (
                    self.gripper_contact_detected
                    and self.trust_gripper_contact_for_success
                    and self._lift_floor_fail_count < self.lift_check_floor_fail_samples
                ):
                    return

                if self._lift_floor_fail_count >= self.lift_check_floor_fail_samples:
                    if getattr(self, '_lift_check_timer', None) is not None:
                        try:
                            self._lift_check_timer.cancel()
                        except Exception:
                            pass
                        self._lift_check_timer = None

                    self.retry_target_from_lift_check = point_base.copy()

                    if self.gripper_contact_detected and self.never_open_after_contact_during_retry:
                        if self.require_lift_check_success_for_transport and not self.trust_gripper_contact_for_success:
                            self._hold_closed_after_failed_grasp_check(
                                'Lift-check was uncertain and contact is not trusted as success.'
                            )
                            return
                        self.get_logger().warning('Lift-check is uncertain but gripper contact was detected. '
                            'Keeping gripper closed and continuing instead of opening/releasing.')
                        self._after_lift_verification_success('Lift-check uncertain but contact is present.')
                        return

                    if (
                        not self.gripper_feedback_available
                        and self.keep_closed_on_lift_check_failure_without_feedback
                    ):
                        self.get_logger().warning('Lift-check visual verification failed repeatedly, but gripper feedback '
                            'is unavailable; this is inconclusive, so the gripper will stay closed.')
                    else:
                        self.get_logger().warning('Lift-check FAILED with repeated fresh floor detections. '
                            'Retry is allowed because no reliable gripper contact was detected.')
                    self.handle_failed_grasp_after_lift()
                    return

        if elapsed >= self.lift_check_detect_timeout_sec:
            if getattr(self, '_lift_check_timer', None) is not None:
                try:
                    self._lift_check_timer.cancel()
                except Exception:
                    pass
                self._lift_check_timer = None

            if self.lift_check_require_positive_z_success and self._lift_floor_fail_count > 0:
                if self._lift_check_last_nonlifted_target is not None:
                    self.retry_target_from_lift_check = self._lift_check_last_nonlifted_target.copy()
                self.get_logger().warning(f'Lift-check FAILED by timeout: {self._lift_floor_fail_count} fresh detection(s) '
                    f'were seen, but none rose by the required '
                    f'{self.grasp_success_min_lift_m:.3f}m. Retrying instead of treating '
                    'sideways/no-lift detections as success.')
                self.handle_failed_grasp_after_lift()
                return

            if self.gripper_contact_detected and self.trust_gripper_contact_for_success:
                self.get_logger().info('Lift-check PASSED by timeout + gripper contact: '
                    'probe is likely occluded/held between fingers.')
            else:
                self.get_logger().info('Lift-check PASSED by timeout: probe not detected at original floor pose.')

            self._after_lift_verification_success('Lift-check PASSED by timeout.')

    def handle_failed_grasp_after_lift(self) -> None:
        """
        Recovery Agent.

        Only open/retry if failure is strong and there was no gripper contact.
        If contact was detected, opening the gripper may release a successful grasp.
        """
        if (
            not self.gripper_feedback_available
            and self.keep_closed_on_lift_check_failure_without_feedback
        ):
            if self.require_lift_check_success_for_transport:
                self._hold_closed_after_failed_grasp_check(
                    'Lift-check failed visually and gripper feedback is disabled.'
                )
                return

            self.get_logger().warning('Lift-check failed visually, but gripper feedback is disabled. '
                'Keeping the gripper closed and continuing instead of opening/releasing.')
            self._after_lift_verification_success(
                'Lift-check inconclusive with open-loop gripper control.'
            )
            return

        if self.gripper_contact_detected and self.never_open_after_contact_during_retry:
            self.get_logger().warning('Retry blocked: gripper contact was detected. '
                'Keeping gripper closed and retreating instead of releasing the probe.')
            self.holding_object = True
            self.send_retreat()
            return

        if self.grasp_attempt_count >= self.max_grasp_attempts:
            self._hold_closed_after_failed_grasp_check(
                f'All {self.max_grasp_attempts} physical grasp attempts failed at lift-check.'
            )
            return

        next_attempt = self.grasp_attempt_count + 1

        self.get_logger().warning(f'Grasp attempt {self.grasp_attempt_count}/{self.max_grasp_attempts} failed; '
            f'preparing in-place retry {next_attempt}/{self.max_grasp_attempts}.')

        if self.retry_target_from_lift_check is not None:
            self.current_target_point_base = self.retry_target_from_lift_check.copy()

        self.grasp_depth_below_surface_m += self.retry_extra_grasp_depth_m
        self.grasp_depth_below_surface_m = min(self.grasp_depth_below_surface_m, 0.055)

        if self.current_target_point_base is not None and self.grasp_orientation is not None:
            self.update_contact_poses_from_target(
                self.current_target_point_base,
                self.grasp_orientation
            )
            self.publish_markers()

        self.get_logger().warning(f'Retrying without full reset: new grasp_depth_below_surface_m='
            f'{self.grasp_depth_below_surface_m:.3f}m')

        self.retry_target_from_lift_check = None
        self.pending_replan_after_motion = False
        self._lift_floor_fail_count = 0

        self._return_to_grasp_pose_before_retry()

    def _return_to_grasp_pose_before_retry(self) -> None:
        """Lower back to the retry grasp pose while still closed, then open."""
        if self.grasp_pose is None:
            self.reset_sequence('Cannot retry failed grasp: no grasp pose is available.')
            return

        self.sequence_stage = 'move_retry_return'
        self.get_logger().warning('Returning to the retry grasp pose with the gripper still closed before opening. '
            'This keeps a possible false-negative grasp close to the ground; MoveIt collision '
            'checking remains enabled for the return path.')
        self._send_cartesian_path([self.contact_pose_to_link_pose(self.grasp_pose)])

    def _open_gripper_for_in_place_retry(self) -> None:
        self.holding_object = False
        self.command_gripper_and_then(
            self.gripper_open,
            self.send_grasp,
            stage_name='retry_open_gripper',
            description='open before in-place retry grasp'
        )

    def send_retreat(self) -> None:
        """Cartesian straight-line retreat: grasp -> pre_grasp, then joint home."""
        if self.pre_grasp_pose is None:
            self.reset_sequence('No pre-grasp pose available for retreat.')
            return

        self.sequence_stage = 'move_cartesian_retreat'
        self._send_cartesian_path([self.contact_pose_to_link_pose(self.pre_grasp_pose)])

    def _send_movegroup_grasp_fallback(self, fraction: float) -> bool:
        """Constrained final-approach fallback when Cartesian IK fraction is low.

        GetCartesianPath is very strict near the floor and can stop at 5-60%
        even when a valid constrained MoveGroup solution exists.  Resetting the
        whole sequence causes the endless pre-grasp loop seen in the logs.
        This fallback still uses the same final grasp pose, the same four-bar
        ground guard, the same target_point_offset, and the same orientation
        constraint, but asks OMPL to find a safe constrained path to that pose.
        """
        if self.sequence_stage != 'move_grasp' or self.grasp_pose is None:
            return False

        self.get_logger().warning(f'Cartesian grasp fraction={fraction:.2f}; using constrained MoveGroup '
            f'fallback to final grasp pose instead of resetting. '
            f'pos_tol={self.final_grasp_movegroup_fallback_position_tol:.3f}m, '
            'orientation locked, four-bar floor guard already applied.')
        self.send_pose_goal(
            self.grasp_pose,
            pos_tol=self.final_grasp_movegroup_fallback_position_tol,
            with_orientation=True,
        )
        return True

    def try_lifted_cartesian_retry(self, fraction: float) -> bool:
        """
        Cartesian Grasp Recovery Agent:
        If the final straight-line grasp is too deep for collision/IK, try one
        slightly higher contact point instead of restarting the whole pipeline.
        """
        if self.sequence_stage != 'move_grasp':
            return False

        if self.grasp_pose is None or self.grasp_orientation is None:
            return False

        if self._cartesian_grasp_retries >= self.cartesian_max_retries:
            return False

        self._cartesian_grasp_retries += 1

        approach_axis = self.compute_approach_axis_in_planning_frame(self.grasp_orientation)
        grasp_xyz = self._pose_xyz(self.grasp_pose)

        # approach_axis points downward for the grasp.
        # Subtracting it lifts the contact point slightly upward.
        lifted_xyz = grasp_xyz - approach_axis * self.cartesian_retry_lift_m

        self.grasp_pose = self.make_pose(lifted_xyz, self.grasp_orientation)
        self.publish_markers()

        self.get_logger().warning(f'Cartesian grasp fraction={fraction:.2f}; retrying with grasp point lifted '
            f'{self.cartesian_retry_lift_m:.3f}m '
            f'(retry {self._cartesian_grasp_retries}/{self.cartesian_max_retries}).')

        self._send_cartesian_path([self.contact_pose_to_link_pose(self.grasp_pose)])
        return True

    def _build_cartesian_path_request(
        self,
        waypoints: List[Pose],
        expected_stage: str,
        *,
        avoid_collisions: bool,
        lock_orientation: Optional[bool] = None,
    ) -> GetCartesianPath.Request:
        req = GetCartesianPath.Request()
        req.header.frame_id = self.planning_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = self.planning_group
        req.link_name = self.planning_link
        req.waypoints = waypoints

        req.max_step = self.cartesian_max_step
        req.jump_threshold = self.cartesian_jump_threshold
        req.avoid_collisions = bool(avoid_collisions)

        seed_state = self._make_current_robot_state()
        if seed_state is not None:
            req.start_state = seed_state
        else:
            req.start_state.is_diff = True

        # Lock orientation during Cartesian stroke only when explicitly enabled.
        # The waypoint itself still contains the desired orientation; this extra
        # path constraint is stricter and can make KDL return fraction=0.00 near
        # the floor even when the waypoint orientation is reachable.
        if lock_orientation is None:
            lock_orientation = self.cartesian_lock_orientation

        constraint_orientation = self.grasp_orientation

        if constraint_orientation is not None and lock_orientation:
            cart_ori = OrientationConstraint()
            cart_ori.header.frame_id = self.planning_frame
            cart_ori.header.stamp = req.header.stamp
            cart_ori.link_name = self.planning_link
            cart_ori.orientation = constraint_orientation
            cart_ori.absolute_x_axis_tolerance = self.orientation_tol
            cart_ori.absolute_y_axis_tolerance = self.orientation_tol
            cart_ori.absolute_z_axis_tolerance = self.orientation_tol
            cart_ori.weight = 1.0
            cart_path_c = Constraints()
            cart_path_c.orientation_constraints.append(cart_ori)
            req.path_constraints = cart_path_c

        return req

    def _send_cartesian_path(self, waypoints: List[Pose]) -> None:
        """Compute and execute a Cartesian straight-line path through waypoints."""
        expected_stage = self.sequence_stage
        expected_seq = self.sequence_id
        if self._arm_motion_forbidden_now(expected_stage):
            self.get_logger().error(f'Blocked unsafe Cartesian arm command during gripper stage: requested_stage={expected_stage}')
            return
        if self._cartesian_plan_in_flight is not None or self._pending_arm_motion_confirmation is not None:
            self._cancel_active_moveit_goal()
            self._arm_motion_confirmation_failed(
                expected_stage,
                'Blocked overlapping Cartesian planning/execution request.',
            )
            return
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            if self.sequence_stage == 'move_grasp':
                if self.allow_movegroup_fallback_for_grasp and self._send_movegroup_grasp_fallback(0.0):
                    return
                self._halt_after_final_approach_failure(
                    'GetCartesianPath service unavailable during grasp; no safe fallback available.'
                )
            else:
                self.get_logger().warning('GetCartesianPath service unavailable; going to joint home.')
                self._do_joint_home()
            return
        req = self._build_cartesian_path_request(
            waypoints,
            expected_stage,
            avoid_collisions=True,
            lock_orientation=self.cartesian_lock_orientation,
        )
        if expected_stage == 'move_grasp' and waypoints:
            current_pose = self.get_current_link_pose_in_planning_frame()
            goal_pose = waypoints[-1]
            if current_pose is not None:
                dx = float(goal_pose.position.x - current_pose.position.x)
                dy = float(goal_pose.position.y - current_pose.position.y)
                dz = float(goal_pose.position.z - current_pose.position.z)
                self.get_logger().info(f'Final Cartesian request: '
                    f'current_link=({current_pose.position.x:.3f},{current_pose.position.y:.3f},{current_pose.position.z:.3f}) '
                    f'goal_link=({goal_pose.position.x:.3f},{goal_pose.position.y:.3f},{goal_pose.position.z:.3f}) '
                    f'delta=({dx*1000:.1f},{dy*1000:.1f},{dz*1000:.1f})mm '
                    f'collision_check=on orientation_path_constraint={self.cartesian_lock_orientation}.')
        final_waypoint = waypoints[-1] if waypoints else None
        self._cartesian_plan_in_flight = (expected_stage, expected_seq)
        future = self.cartesian_client.call_async(req)
        future.add_done_callback(
            lambda fut, st=expected_stage, seq=expected_seq, wp=final_waypoint: self._on_cartesian_path(
                fut, st, seq, wp
            )
        )

    def _on_cartesian_path(
        self,
        future,
        expected_stage: str,
        expected_seq: int,
        final_waypoint: Optional[Pose],
    ) -> None:
        try:
            resp = future.result()
        except Exception as exc:
            if self._cartesian_plan_in_flight == (expected_stage, expected_seq):
                self._cartesian_plan_in_flight = None
            self.reset_sequence(f'GetCartesianPath call failed: {exc}')
            return
        if expected_seq != self.sequence_id or self.sequence_stage != expected_stage:
            if self._cartesian_plan_in_flight == (expected_stage, expected_seq):
                self._cartesian_plan_in_flight = None
            self.get_logger().warning(f'Ignoring stale Cartesian response for stage={expected_stage}; '
                f'current_stage={self.sequence_stage}.')
            return
        self._cartesian_plan_in_flight = None

        stage = expected_stage
        min_fraction = self.cartesian_fraction_min
        if resp.fraction < min_fraction:
            if stage == 'move_grasp':
                if self.allow_movegroup_fallback_for_grasp and self._send_movegroup_grasp_fallback(resp.fraction):
                    return
                if self.try_lifted_cartesian_retry(resp.fraction):
                    return
                reason = (
                    f'Cartesian path only {resp.fraction:.2f} complete at {stage}; '
                    f'no safe final-approach fallback succeeded.'
                )
                if self._start_final_cartesian_failure_diagnostic(reason, expected_seq):
                    return
                self._halt_after_final_approach_failure(reason)
                return

            if stage == 'move_retry_return':
                self.reset_sequence(
                    f'Collision-aware retry return path only {resp.fraction:.2f} complete; '
                    'keeping the gripper closed and stopping retry.'
                )
                return

            self.get_logger().warning(f'Cartesian path only {resp.fraction:.2f} complete at {stage}; going to joint home.')
            self._do_joint_home()
            return

        # ── Trajectory Safety Validator (Layer 5) ─────────────────────────────
        # Verify the first trajectory waypoint is close to the actual robot joint
        # positions read from /joint_states.  A large deviation means MoveIt used
        # a stale or incorrect start state; executing such a trajectory would move
        # the arm to the wrong configuration before the first waypoint, driving it
        # in the wrong direction and risking self-collision.
        if resp.solution.joint_trajectory.points:
            first_pt = resp.solution.joint_trajectory.points[0]
            cur_joints = self.current_joint_positions
            if cur_joints and resp.solution.joint_trajectory.joint_names:
                worst_dev = 0.0
                worst_name = ''
                for jname, jpos in zip(resp.solution.joint_trajectory.joint_names,
                                       first_pt.positions):
                    if jname in cur_joints:
                        dev = abs(float(jpos) - float(cur_joints[jname]))
                        if dev > worst_dev:
                            worst_dev = dev
                            worst_name = jname
                if worst_dev > 0.35:
                    # 0.35 rad ≈ 20°: clearly a wrong-configuration plan.
                    self.get_logger().error(f'[Safety Validator] Trajectory REJECTED – '
                        f'start-state deviation {worst_dev:.3f} rad on joint {worst_name}. '
                        f'MoveIt planned from a wrong configuration; executing would drive '
                        f'the arm in the wrong direction.')
                    reason = (
                        f'Cartesian trajectory for {stage} rejected: start-state deviation '
                        f'{worst_dev:.3f} rad on {worst_name}.'
                    )
                    if stage == 'move_grasp':
                        self._halt_after_final_approach_failure(reason)
                    elif self.holding_object:
                        self._hold_closed_after_transport_failure(reason)
                    else:
                        self.reset_sequence(reason)
                    return
                else:
                    self.get_logger().info(f'[Safety Validator] Cartesian trajectory start-state OK: '
                        f'stage={stage}, max joint deviation={worst_dev:.3f} rad on {worst_name}.')

        self.get_logger().info(f'Cartesian path {resp.fraction:.2f} at {stage}; executing.')
        if not self.execute_client.wait_for_server(timeout_sec=2.0):
            self.reset_sequence('ExecuteTrajectory action server unavailable.')
            return
        if final_waypoint is None:
            self.reset_sequence(f'Cartesian stage {stage} has no final waypoint to confirm.')
            return
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.planning_frame
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose = final_waypoint
        if not self._register_pose_motion_confirmation(
            stage,
            expected_seq,
            target_pose,
            [0.0, 0.0, 0.0],
            self.arm_pose_confirmation_position_tolerance_m,
            final_waypoint.orientation,
            self.arm_pose_confirmation_orientation_tolerance_rad,
        ):
            self._arm_motion_confirmation_failed(
                stage,
                'Blocked Cartesian execution because another arm stage is still active.',
            )
            return
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = resp.solution
        f = self.execute_client.send_goal_async(goal)
        f.add_done_callback(
            lambda fut, st=stage, seq=expected_seq: self.on_goal_response(fut, st, seq)
        )

    def _start_final_cartesian_failure_diagnostic(self, reason: str, expected_seq: int) -> bool:
        """Run one non-executed diagnostic request to classify a final descent failure."""
        if not self.diagnose_final_cartesian_failure:
            return False
        if self.grasp_pose is None:
            return False
        if not self.cartesian_client.wait_for_service(timeout_sec=0.0):
            return False

        req = self._build_cartesian_path_request(
            [self.contact_pose_to_link_pose(self.grasp_pose)],
            'move_grasp',
            avoid_collisions=False,
            lock_orientation=self.cartesian_lock_orientation,
        )
        self.get_logger().warning('Final descent failed with collision-aware Cartesian planning. '
            'Running diagnostic-only Cartesian request with collisions disabled; '
            'this trajectory will NOT execute.')
        future = self.cartesian_client.call_async(req)
        future.add_done_callback(
            lambda fut, seq=expected_seq, why=reason: self._on_final_cartesian_failure_diagnostic(fut, seq, why)
        )
        return True

    def _on_final_cartesian_failure_diagnostic(self, future, expected_seq: int, reason: str) -> None:
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warning(f'Final Cartesian diagnostic call failed: {exc}')
            self._halt_after_final_approach_failure(reason)
            return

        if expected_seq != self.sequence_id or self.sequence_stage != 'move_grasp':
            self.get_logger().warning(f'Ignoring stale final Cartesian diagnostic; current_stage={self.sequence_stage}.')
            return

        diag_fraction = float(resp.fraction)
        error_code = getattr(getattr(resp, 'error_code', None), 'val', 'unknown')
        if diag_fraction >= self.cartesian_fraction_min:
            self.get_logger().error(f'Final Cartesian diagnostic: fraction={diag_fraction:.2f} with collisions disabled '
                f'(error_code={error_code}). The waypoint is IK-reachable, so MoveIt collision '
                'checking is blocking the descent. Keep collision checking on; inspect RViz '
                'Planning Scene contacts/body self-collisions near the final pose.')
        else:
            self.get_logger().error(f'Final Cartesian diagnostic: fraction={diag_fraction:.2f} even with collisions disabled '
                f'(error_code={error_code}). This points to IK, joint limits, start-state mismatch, '
                'or the waypoint orientation itself, not body collision checking.')

        self._halt_after_final_approach_failure(reason)

    def _do_joint_home(self) -> None:
        self.sequence_stage = 'move_retreat_home'
        self.send_joint_goal(self.retreat_home_joint_names, self.retreat_home_joint_positions)

    def _halt_after_final_approach_failure(self, reason: str) -> None:
        """Stop auto-grasp after a repeatable collision-aware final approach failure."""
        self.reset_sequence(reason)
        if not self.stop_after_final_approach_failure:
            return

        lockout = max(0.0, float(self.final_approach_failure_lockout_sec))
        self.paused_after_failure = True
        self.blocked_until_sec = self._now_sec() + lockout
        self.sequence_stage = 'failed_final_approach'
        self.get_logger().error(f'Auto-grasp paused after final approach failure: {reason} '
            f'Lockout={lockout:.1f}s. MoveIt collision-aware Cartesian planning refused '
            'the final descent, so the node will not reacquire and loop. '
            'Relaunch the node or lower final_approach_failure_lockout_sec to retry automatically.')

    def _halt_after_final_pose_check_failure(self, reason: str) -> None:
        """Stop auto-grasp when execution did not reach the committed grasp pose."""
        self.reset_sequence(reason)
        if not self.stop_after_final_approach_failure:
            return

        lockout = max(0.0, float(self.final_approach_failure_lockout_sec))
        self.paused_after_failure = True
        self.blocked_until_sec = self._now_sec() + lockout
        self.sequence_stage = 'failed_final_pose_check'
        self.get_logger().error(f'Auto-grasp paused before gripper close: {reason} '
            f'Lockout={lockout:.1f}s. The final Cartesian trajectory was planned with '
            'MoveIt collision checking, but the measured TCP did not reach the committed '
            'grasp pose, so the node will not close on empty space or reacquire in a loop. '
            'Relaunch the node or lower final_approach_failure_lockout_sec to retry automatically.')

    def reset_sequence(self, reason: str) -> None:
        failed_stage = self.sequence_stage
        if (
            self.preserve_orientation_across_pregrasp_retries
            and failed_stage in ('move_pre_grasp', 'pregrasp_finalizing')
            and self.grasp_orientation is not None
            and self.current_target_point_base is not None
        ):
            self._retry_grasp_orientation = Quaternion(
                x=float(self.grasp_orientation.x),
                y=float(self.grasp_orientation.y),
                z=float(self.grasp_orientation.z),
                w=float(self.grasp_orientation.w),
            )
            self._retry_grasp_target = self.current_target_point_base.copy()
            self._retry_grasp_orientation_until_sec = (
                self._now_sec() + self.pregrasp_retry_orientation_hold_sec
            )
            self.get_logger().warning(
                'Preserving the initially locked grasp orientation for the next '
                'nearby pre-grasp retry.'
            )
        self.get_logger().warning(f'Resetting grasp sequence: {reason}')
        self._cancel_active_moveit_goal()
        self._cancel_active_gripper_goal()
        self.sequence_id += 1
        self._cancel_pending_timers()
        self._cancel_final_grasp_pose_check_timer()
        self._clear_arm_motion_confirmation()

        # Clean up any post-grasp collision objects (floor plane + probe) from
        # the planning scene so the next grasp attempt starts clean.
        self._remove_post_grasp_collision_objects()

        if getattr(self, '_gripper_wait_timer', None) is not None:
            self._gripper_wait_timer.cancel()
            self._gripper_wait_timer = None

        if getattr(self, '_refine_timer', None) is not None:
            try:
                self._refine_timer.cancel()
            except Exception:
                pass

        if getattr(self, '_pregrasp_watchdog_timer', None) is not None:
            try:
                self._pregrasp_watchdog_timer.cancel()
            except Exception:
                pass
            self._pregrasp_watchdog_timer = None

        if getattr(self, '_lift_check_timer', None) is not None:
            try:
                self._lift_check_timer.cancel()
            except Exception:
                pass
            self._lift_check_timer = None

        self.last_failure_reason = reason
        if reason:
            self.failure_count += 1
            self.blocked_until_sec = self._now_sec() + self.failure_cooldown_sec

        self.busy = False
        self.sequence_stage = 'idle'
        self.perception_frozen_for_sequence = False
        self.paused_after_failure = False
        self.current_target_point_base = None
        self.grasp_orientation = None
        self.pre_grasp_pose = None
        self.grasp_pose = None
        self.retreat_pose = None
        self.pending_replan_after_motion = False
        self.replan_count = 0
        self._cartesian_grasp_retries = 0
        self.grasp_attempt_count = 0
        self.grasp_depth_below_surface_m = self.base_grasp_depth_below_surface_m
        self.computed_gripper_close = self.gripper_close
        self.computed_gripper_preclose = self.gripper_preclose
        self.last_estimated_object_width_m = None
        self.effective_target_point_offset_in_link = list(self.target_point_offset_in_link)
        self.detected_object_yaw_rad = None
        self._last_detected_orientation_cam = None
        self._clear_detected_object_pose()
        self.locked_target_before_lift = None
        self.retry_target_from_lift_check = None
        self._lift_check_last_nonlifted_target = None
        self.sequence_locked_target_point_base = None
        self.sequence_locked_object_long_axis_base = None
        self.gripper_contact_detected = False
        self.last_gripper_actual = None
        self.last_gripper_target = None
        self._lift_floor_fail_count = 0
        self._close_step_targets = []
        self._close_step_index = 0
        self.preclosed_in_air = False
        self.pregrasp_correction_count = 0
        self._pregrasp_force_finalize = False
        self._pregrasp_final_replan_count = 0
        self._pregrasp_motion_start_sec = 0.0
        self._active_move_goal_handle = None
        self._refine_width_buffer = []
        self._refine_orientation_cam_last = None
        self._clear_target_stability_history()

        self.publish_markers()

    # ---------- whole-process arm motion confirmation ----------
    def _clear_arm_motion_confirmation(self) -> None:
        if getattr(self, '_arm_confirmation_timer', None) is not None:
            try:
                self._arm_confirmation_timer.cancel()
            except Exception:
                pass
            self._arm_confirmation_timer = None
        self._pending_arm_motion_confirmation = None
        self._cartesian_plan_in_flight = None

    def _register_joint_motion_confirmation(
        self,
        stage: str,
        sequence_id: int,
        joint_names,
        joint_positions,
    ) -> bool:
        if self._pending_arm_motion_confirmation is not None:
            self.get_logger().error(f'Blocked overlapping arm command at stage={stage}; '
                f'previous_stage={self._pending_arm_motion_confirmation.get("stage")}.')
            return False
        self._pending_arm_motion_confirmation = {
            'kind': 'joint',
            'stage': stage,
            'sequence_id': sequence_id,
            'command_start_sec': self._now_sec(),
            'joint_names': [str(v) for v in joint_names],
            'joint_positions': [float(v) for v in joint_positions],
            'stable_samples': 0,
        }
        return True

    def _register_pose_motion_confirmation(
        self,
        stage: str,
        sequence_id: int,
        target_pose: PoseStamped,
        target_point_offset: List[float],
        position_tolerance: float,
        orientation: Optional[Quaternion],
        orientation_tolerance: float,
    ) -> bool:
        if self._pending_arm_motion_confirmation is not None:
            self.get_logger().error(f'Blocked overlapping arm command at stage={stage}; '
                f'previous_stage={self._pending_arm_motion_confirmation.get("stage")}.')
            return False
        self._pending_arm_motion_confirmation = {
            'kind': 'pose',
            'stage': stage,
            'sequence_id': sequence_id,
            'command_start_sec': self._now_sec(),
            'target_pose': target_pose,
            'target_point_offset': [float(v) for v in target_point_offset],
            'position_tolerance': max(
                float(position_tolerance),
                self.arm_pose_confirmation_position_tolerance_m,
            ),
            'orientation': orientation,
            'orientation_tolerance': max(
                float(orientation_tolerance),
                self.arm_pose_confirmation_orientation_tolerance_rad,
            ),
            'stable_samples': 0,
        }
        return True

    def _arm_feedback_is_fresh(self, joint_names: List[str], command_start_sec: float) -> bool:
        now_sec = self._now_sec()
        for name in joint_names:
            stamp = self.current_joint_update_sec.get(name)
            if (
                stamp is None
                or float(stamp) < command_start_sec
                or now_sec - float(stamp) > self.arm_feedback_max_age_sec
            ):
                return False
        return True

    def _pose_target_in_planning_frame(
        self,
        target_pose: PoseStamped,
        orientation: Optional[Quaternion],
    ) -> Optional[Tuple[np.ndarray, Optional[Quaternion]]]:
        target_xyz = np.array([
            float(target_pose.pose.position.x),
            float(target_pose.pose.position.y),
            float(target_pose.pose.position.z),
        ], dtype=np.float64)
        source_frame = target_pose.header.frame_id or self.planning_frame
        if source_frame == self.planning_frame:
            return target_xyz, orientation
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.planning_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warning(f'Arm completion check cannot transform '
                f'{source_frame} -> {self.planning_frame}: {exc}', throttle_duration_sec=1.0)
            return None

        R_tf = quat_to_matrix(tfm.transform.rotation)
        t_tf = np.array([
            float(tfm.transform.translation.x),
            float(tfm.transform.translation.y),
            float(tfm.transform.translation.z),
        ], dtype=np.float64)
        target_xyz = t_tf + R_tf @ target_xyz
        target_orientation = orientation
        if orientation is not None:
            target_orientation = matrix_to_quat(R_tf @ quat_to_matrix(orientation))
        return target_xyz, target_orientation

    def _arm_motion_feedback_reached(self, pending: dict) -> Tuple[bool, str]:
        command_start_sec = float(pending['command_start_sec'])
        if pending['kind'] == 'joint':
            names = pending['joint_names']
            if not self._arm_feedback_is_fresh(names, command_start_sec):
                return False, 'joint feedback is missing, stale, or predates the command'
            worst_error = 0.0
            worst_name = ''
            for name, target in zip(names, pending['joint_positions']):
                actual = self.current_joint_positions.get(name)
                if actual is None:
                    return False, f'joint {name} has no measured position'
                error = abs(wrap_to_pi(float(actual) - float(target)))
                if error > worst_error:
                    worst_error = error
                    worst_name = name
            tolerance = max(self.joint_goal_tolerance, self.arm_joint_confirmation_tolerance_rad)
            return (
                worst_error <= tolerance,
                f'worst_joint={worst_name}, error={worst_error:.4f}rad, tolerance={tolerance:.4f}rad',
            )

        if not self._arm_feedback_is_fresh(self.arm_feedback_joint_names, command_start_sec):
            return False, 'arm feedback is missing, stale, or predates the pose command'
        current = self.get_current_link_pose_in_planning_frame()
        if current is None:
            return False, 'current planning-link TF is unavailable'
        transformed = self._pose_target_in_planning_frame(
            pending['target_pose'],
            pending['orientation'],
        )
        if transformed is None:
            return False, 'target pose transform is unavailable'
        target_xyz, target_orientation = transformed
        R_current = quat_to_matrix(current.orientation)
        offset = np.array(pending['target_point_offset'], dtype=np.float64)
        actual_xyz = np.array([
            float(current.position.x),
            float(current.position.y),
            float(current.position.z),
        ], dtype=np.float64) + R_current @ offset
        position_error = float(np.linalg.norm(actual_xyz - target_xyz))
        orientation_error = 0.0
        orientation_axis_error = 0.0
        orientation_ok = True
        if target_orientation is not None:
            orientation_error = quaternion_distance_rad(current.orientation, target_orientation)
            rotation_vector_error = quaternion_rotation_vector_error(
                target_orientation, current.orientation
            )
            orientation_axis_error = float(np.max(np.abs(rotation_vector_error)))
            orientation_ok = orientation_axis_error <= float(pending['orientation_tolerance'])
        reached = (
            position_error <= float(pending['position_tolerance'])
            and orientation_ok
        )
        return reached, (
            f'position_error={position_error:.4f}m/{float(pending["position_tolerance"]):.4f}m, '
            f'orientation_error={orientation_error:.4f}rad total, '
            f'max_axis_error={orientation_axis_error:.4f}rad/'
            f'{float(pending["orientation_tolerance"]):.4f}rad'
        )

    def _start_arm_motion_confirmation(self, stage: str, sequence_id: int) -> None:
        pending = self._pending_arm_motion_confirmation
        if (
            pending is None
            or pending.get('stage') != stage
            or pending.get('sequence_id') != sequence_id
        ):
            self._arm_motion_confirmation_failed(
                stage,
                'MoveIt succeeded but no matching measured-target confirmation was registered.',
            )
            return
        if not self.arm_require_feedback_for_completion:
            self._clear_arm_motion_confirmation()
            self._handle_confirmed_arm_motion(stage)
            return
        pending['action_success_sec'] = self._now_sec()
        pending['stable_samples'] = 0
        self._arm_confirmation_timer = self.create_timer(
            self.arm_feedback_check_period_sec,
            self._arm_motion_confirmation_tick,
        )

    def _arm_motion_confirmation_tick(self) -> None:
        pending = self._pending_arm_motion_confirmation
        if pending is None:
            self._clear_arm_motion_confirmation()
            return
        stage = str(pending['stage'])
        if pending['sequence_id'] != self.sequence_id or self.sequence_stage != stage:
            self._clear_arm_motion_confirmation()
            return
        elapsed = self._now_sec() - float(pending['action_success_sec'])
        if elapsed < self.arm_feedback_settle_sec:
            return
        reached, detail = self._arm_motion_feedback_reached(pending)
        pending['last_detail'] = detail
        pending['stable_samples'] = int(pending['stable_samples']) + 1 if reached else 0
        if int(pending['stable_samples']) >= self.arm_feedback_stable_samples:
            self.get_logger().info(f'Arm stage confirmed by action result + fresh measured state: '
                f'stage={stage}, {detail}, stable_samples={pending["stable_samples"]}.')
            self._clear_arm_motion_confirmation()
            self._handle_confirmed_arm_motion(stage)
            return
        if elapsed >= self.arm_feedback_timeout_sec:
            self._clear_arm_motion_confirmation()
            self._arm_motion_confirmation_failed(
                stage,
                f'Measured arm state did not confirm the target after MoveIt success: {detail}.',
            )

    def _arm_motion_confirmation_failed(self, stage: str, reason: str) -> None:
        if stage == 'move_pick_home_after_place':
            self.finish_placement_successfully(returned_home=False, warning=reason)
            return
        if stage == 'move_grasp':
            self._halt_after_final_pose_check_failure(reason)
            return
        if self.holding_object:
            self._hold_closed_after_transport_failure(reason)
            return
        self.reset_sequence(reason)

    def send_pose_goal(self, pose: PoseStamped,
                        pos_tol: Optional[float] = None,
                        with_orientation: bool = False,
                        orientation_override: Optional[Quaternion] = None,
                        orientation_tol: Optional[float] = None,
                        target_point_offset: Optional[List[float]] = None,
                        path_constraints: Optional[Constraints] = None,
                        velocity_scale: Optional[float] = None,
                        acceleration_scale: Optional[float] = None,
                        arm_joints_only_start_state: bool = False,
                        planning_time_override: Optional[float] = None,
                        num_attempts_override: Optional[int] = None) -> None:
        expected_stage = self.sequence_stage
        expected_seq = self.sequence_id
        if self._arm_motion_forbidden_now(expected_stage):
            self.get_logger().error(f'Blocked unsafe MoveGroup command during gripper stage: requested_stage={expected_stage}')
            return
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.reset_sequence('MoveIt action server not available.')
            return
        goal = MoveGroup.Goal()
        goal.request.workspace_parameters.header.frame_id = self.planning_frame
        goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        seed_state = self._make_current_robot_state(arm_joints_only=arm_joints_only_start_state)
        if seed_state is not None:
            goal.request.start_state = seed_state
        else:
            goal.request.start_state.is_diff = True
        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = (
            num_attempts_override if num_attempts_override is not None
            else self.num_planning_attempts
        )
        goal.request.allowed_planning_time = (
            planning_time_override if planning_time_override is not None
            else self.allowed_planning_time
        )
        goal.request.max_velocity_scaling_factor = float(
            self.velocity_scale if velocity_scale is None else velocity_scale
        )
        goal.request.max_acceleration_scaling_factor = float(
            self.acceleration_scale if acceleration_scale is None else acceleration_scale
        )
        tol = pos_tol if pos_tol is not None else self.position_tol
        c = Constraints()
        pos = PositionConstraint()
        pos.header.frame_id = pose.header.frame_id
        pos.header.stamp = pose.header.stamp
        pos.link_name = self.planning_link
        offset = target_point_offset
        if offset is None:
            offset = list(self.effective_target_point_offset_in_link)
        pos.target_point_offset = Vector3(
            x=float(offset[0]),
            y=float(offset[1]),
            z=float(offset[2]))
        region = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [tol]
        region.primitives.append(sphere)
        region.primitive_poses.append(pose.pose)
        pos.constraint_region = region
        pos.weight = 1.0
        c.position_constraints.append(pos)
        orientation_constraint = orientation_override if orientation_override is not None else self.grasp_orientation
        if with_orientation and orientation_constraint is not None:
            ori_tol = self.orientation_tol if orientation_tol is None else float(orientation_tol)
            ori = OrientationConstraint()
            ori.header.frame_id = pose.header.frame_id
            ori.header.stamp = pose.header.stamp
            ori.link_name = self.planning_link
            ori.orientation = orientation_constraint
            ori.absolute_x_axis_tolerance = ori_tol
            ori.absolute_y_axis_tolerance = ori_tol
            ori.absolute_z_axis_tolerance = ori_tol
            ori.parameterization = getattr(OrientationConstraint, 'ROTATION_VECTOR', 1)
            ori.weight = 1.0
            c.orientation_constraints.append(ori)
        goal.request.goal_constraints = [c]
        if path_constraints is not None:
            goal.request.path_constraints = path_constraints
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        confirmation_orientation = orientation_constraint if with_orientation else None
        if not self._register_pose_motion_confirmation(
            expected_stage,
            expected_seq,
            pose,
            offset,
            float(tol),
            confirmation_orientation,
            float(self.orientation_tol if orientation_tol is None else orientation_tol),
        ):
            self._arm_motion_confirmation_failed(
                expected_stage,
                'Blocked a pose command because another arm stage is still active.',
            )
            return
        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(
            lambda fut, st=expected_stage, seq=expected_seq: self.on_goal_response(fut, st, seq)
        )

    def send_joint_goal(
        self,
        joint_names,
        joint_positions,
        planning_time_override: Optional[float] = None,
        num_attempts_override: Optional[int] = None,
    ) -> None:
        expected_stage = self.sequence_stage
        expected_seq = self.sequence_id
        if self._arm_motion_forbidden_now(expected_stage):
            self.get_logger().error(f'Blocked unsafe joint/home command during gripper stage: requested_stage={expected_stage}')
            return
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.reset_sequence('MoveIt action server not available.')
            return
        if len(joint_names) != len(joint_positions):
            self.reset_sequence('retreat_home_joint_names and retreat_home_joint_positions length mismatch.')
            return
        goal = MoveGroup.Goal()
        goal.request.workspace_parameters.header.frame_id = self.planning_frame
        goal.request.workspace_parameters.header.stamp = self.get_clock().now().to_msg()

        # Seed start state from live /joint_states so MoveGroup plans from the
        # ACTUAL current configuration, not the (potentially stale) planning scene.
        seed_state = self._make_current_robot_state(arm_joints_only=False)
        if seed_state is not None:
            goal.request.start_state = seed_state
        else:
            goal.request.start_state.is_diff = True

        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = (
            num_attempts_override if num_attempts_override is not None
            else self.num_planning_attempts
        )
        goal.request.allowed_planning_time = (
            planning_time_override if planning_time_override is not None
            else self.allowed_planning_time
        )
        goal.request.max_velocity_scaling_factor = self.velocity_scale
        goal.request.max_acceleration_scaling_factor = self.acceleration_scale
        c = Constraints()
        for name, pos in zip(joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = str(name)
            jc.position = float(pos)
            jc.tolerance_above = self.joint_goal_tolerance
            jc.tolerance_below = self.joint_goal_tolerance
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        goal.request.goal_constraints = [c]
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        if not self._register_joint_motion_confirmation(
            expected_stage,
            expected_seq,
            joint_names,
            joint_positions,
        ):
            self._arm_motion_confirmation_failed(
                expected_stage,
                'Blocked a joint command because another arm stage is still active.',
            )
            return
        future = self.move_group_client.send_goal_async(goal)
        future.add_done_callback(
            lambda fut, st=expected_stage, seq=expected_seq: self.on_goal_response(fut, st, seq)
        )

    def on_goal_response(self, future, expected_stage: str, expected_seq: int) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            if expected_seq != self.sequence_id or self.sequence_stage != expected_stage:
                self.get_logger().warning(f'Ignoring stale failed motion goal response for stage={expected_stage}; '
                    f'current_stage={self.sequence_stage}: {exc}')
                return
            self._clear_arm_motion_confirmation()
            if expected_stage == 'move_pick_home_after_place':
                self.finish_placement_successfully(
                    returned_home=False,
                    warning=f'Probe was placed, but the return-home goal could not be sent: {exc}',
                )
                return
            if expected_stage == 'move_base_box_drop' or (
                expected_stage == 'move_pick_home' and self.place_in_base_box_after_grasp
            ):
                self._hold_closed_after_transport_failure(
                    f'MoveIt held-probe transport goal could not be sent: {exc}'
                )
                return
            self.reset_sequence(f'MoveIt goal send failed: {exc}')
            return
        if expected_seq != self.sequence_id or self.sequence_stage != expected_stage:
            # A reset can happen after send_goal_async() but before MoveIt
            # returns the goal handle. Never leave that late-accepted goal
            # executing underneath a newer process stage.
            if goal_handle.accepted:
                try:
                    goal_handle.cancel_goal_async()
                    self.get_logger().warning(f'Cancelled stale accepted arm goal for stage={expected_stage}; '
                        f'current_stage={self.sequence_stage}.')
                except Exception as exc:
                    self.get_logger().error(f'Could not cancel stale accepted arm goal for stage={expected_stage}: {exc}')
            else:
                self.get_logger().warning(f'Ignoring stale rejected arm goal for stage={expected_stage}; '
                    f'current_stage={self.sequence_stage}.')
            return
        if not goal_handle.accepted:
            self._clear_arm_motion_confirmation()
            if expected_stage == 'move_pick_home_after_place':
                self.finish_placement_successfully(
                    returned_home=False,
                    warning='Probe was placed, but MoveIt rejected the return-home goal.',
                )
                return
            if expected_stage == 'move_base_box_drop':
                self._hold_closed_after_transport_failure(
                    'MoveIt rejected the motion to the base-box drop posture.'
                )
                return
            if expected_stage == 'move_pick_home' and self.place_in_base_box_after_grasp:
                self._hold_closed_after_transport_failure(
                    'MoveIt rejected the held-probe motion to pick_home.'
                )
                return
            self.reset_sequence(f'MoveIt rejected goal during stage {expected_stage}.')
            return
        self._active_move_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda fut, st=expected_stage, seq=expected_seq: self.on_goal_result(fut, st, seq)
        )

    def on_goal_result(self, future, expected_stage: str, expected_seq: int) -> None:
        if expected_seq != self.sequence_id or self.sequence_stage != expected_stage:
            self.get_logger().warning(f'Ignoring stale motion result for stage={expected_stage}; '
                f'current_stage={self.sequence_stage}.')
            return
        try:
            result_wrap = future.result()
        except Exception as exc:
            self._clear_arm_motion_confirmation()
            if expected_stage == 'move_pick_home_after_place':
                self.finish_placement_successfully(
                    returned_home=False,
                    warning=f'Probe was placed, but the return-home result was unavailable: {exc}',
                )
                return
            if expected_stage == 'move_base_box_drop' or (
                expected_stage == 'move_pick_home' and self.place_in_base_box_after_grasp
            ):
                self._hold_closed_after_transport_failure(
                    f'MoveIt held-probe transport result was unavailable: {exc}'
                )
                return
            self.reset_sequence(f'MoveIt result failed: {exc}')
            return
        self._active_move_goal_handle = None
        if result_wrap.status != GoalStatus.STATUS_SUCCEEDED:
            self._clear_arm_motion_confirmation()
            if expected_stage == 'move_pick_home_after_place':
                self.finish_placement_successfully(
                    returned_home=False,
                    warning=(
                        'Probe was placed, but the empty-arm return to pick_home failed '
                        f'with status {result_wrap.status}.'
                    ),
                )
                return
            if expected_stage == 'move_base_box_drop':
                self._hold_closed_after_transport_failure(
                    f'MoveIt motion to the base-box drop posture failed with status {result_wrap.status}.'
                )
                return
            if expected_stage == 'move_pick_home' and self.place_in_base_box_after_grasp:
                self._hold_closed_after_transport_failure(
                    f'MoveIt held-probe motion to pick_home failed with status {result_wrap.status}.'
                )
                return
            if (
                expected_stage == 'move_pre_grasp'
                and self.pregrasp_finalize_even_if_moveit_silent
                and not self.arm_require_feedback_for_completion
            ):
                self.get_logger().warning(f'MoveIt pre-grasp returned status {result_wrap.status}; finalizing from current/locked target instead of restarting.')
                self.handle_pregrasp_arrival()
                return
            self.reset_sequence(f'MoveIt motion failed with status {result_wrap.status} at {expected_stage}.')
            return
        self._start_arm_motion_confirmation(expected_stage, expected_seq)
        return

    def _handle_confirmed_arm_motion(self, expected_stage: str) -> None:
        """Advance the state machine only after action and measured-state success."""
        if expected_stage == 'move_pre_grasp':
            self.handle_pregrasp_arrival()

        elif expected_stage == 'move_grasp':
            self.close_gripper_and_retreat()

        elif expected_stage == 'move_lift_check':
            # Arm has risen; now run fresh YOLO+depth checks to confirm whether
            # the probe is still at the original floor pose.
            self.sequence_stage = 'verify_lift'
            self._lift_check_start_sec = self._now_sec()

            if getattr(self, '_lift_check_timer', None) is not None:
                try:
                    self._lift_check_timer.cancel()
                except Exception:
                    pass

            tick_period = min(0.1, self.lift_check_detect_timeout_sec / 5.0)
            self._lift_check_timer = self.create_timer(
                tick_period,
                self._lift_verification_tick
            )

        elif expected_stage == 'move_retry_return':
            self._open_gripper_for_in_place_retry()

        elif expected_stage == 'move_cartesian_retreat':
            self._do_joint_home()

        elif expected_stage == 'move_pick_home':
            if self.place_in_base_box_after_grasp:
                self.send_base_box_drop_closed()
            else:
                self.finish_successfully()

        elif expected_stage == 'move_base_box_drop':
            self.release_probe_in_base_box()

        elif expected_stage == 'move_pick_home_after_place':
            self.finish_placement_successfully(returned_home=True)

        elif expected_stage in ('move_retreat', 'move_retreat_home'):
            self.finish_successfully()

    def finish_placement_successfully(
        self,
        returned_home: bool,
        warning: Optional[str] = None,
    ) -> None:
        """Mark the pick-and-place task complete after the probe was released."""
        self._remove_post_grasp_collision_objects()
        if warning:
            self.get_logger().warning(warning)
        self.get_logger().info('Probe placement finished successfully. '
            f'Probe is in the rover base box; returned_home={returned_home}.')

        self.task_complete = True
        self.holding_object = False
        self.success_until_sec = self._now_sec() + self.success_lockout_sec
        self.pending_replan_after_motion = False
        self.replan_count = 0
        self._clear_target_stability_history()

        if self.clear_target_after_success:
            self.current_target_point_base = None
            self.pre_grasp_pose = None
            self.grasp_pose = None
            self.retreat_pose = None

        if not self.auto_restart_after_success:
            self.busy = True
            self.sequence_stage = 'done_placed'
        else:
            self.busy = False
            self.sequence_stage = 'idle'

        self.publish_markers()

    def finish_successfully(self) -> None:
        """
        Completion Supervisor Agent:
        Mark task complete and stop automatic restart loop.
        """
        # Remove any post-grasp collision objects (floor plane + probe attachment)
        # so they don't pollute the planning scene for the next cycle.
        self._remove_post_grasp_collision_objects()

        self.get_logger().info('Grasp sequence finished successfully. '
            'Object is held with gripper closed; automatic restart is locked.')

        self.task_complete = True
        self.holding_object = True
        self.success_until_sec = self._now_sec() + self.success_lockout_sec
        self.pending_replan_after_motion = False
        self.replan_count = 0
        self._clear_target_stability_history()

        if self.clear_target_after_success:
            self.current_target_point_base = None
            self.pre_grasp_pose = None
            self.grasp_pose = None
            self.retreat_pose = None

        if self.hold_object_after_success and not self.auto_restart_after_success:
            # Keep busy=True so no new grasp sequence starts.
            # This prevents opening the gripper again and dropping the probe.
            self.busy = True
            self.sequence_stage = 'done_holding'
        else:
            self.busy = False
            self.sequence_stage = 'idle'

        self.publish_markers()

    # ---------- markers ----------
    def publish_markers(self) -> None:
        if not self.publish_markers_enabled:
            return
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        if self.marker_use_zero_stamp:
            now.sec = 0
            now.nanosec = 0
        arr.markers.append(self.make_deleteall_marker())

        marker_id = 1
        if self.base_box_drop_marker_enabled and self._base_box_drop_pose_config_valid():
            drop_markers = self.make_base_box_drop_markers(marker_id, now)
            arr.markers.extend(drop_markers)
            marker_id += len(drop_markers)
        if self.current_target_point_base is not None:
            arr.markers.append(self.make_sphere_marker(marker_id, self.marker_frame, now, self.current_target_point_base, 1.0, 0.9, 0.1, 0.95, 'vision_target'))
            marker_id += 1
        if self.detected_object_pose is not None:
            arr.markers.extend(self.make_pose_axes_markers(marker_id, self.planning_frame, now, self.detected_object_pose))
            marker_id += 3
        if self.pre_grasp_pose is not None:
            p = self.pre_grasp_pose.pose.position
            arr.markers.append(self.make_sphere_marker(marker_id, self.marker_frame, now, np.array([p.x, p.y, p.z]), 0.1, 0.4, 1.0, 0.95, 'pre_grasp'))
            marker_id += 1
        if self.grasp_pose is not None:
            p = self.grasp_pose.pose.position
            arr.markers.append(self.make_sphere_marker(marker_id, self.marker_frame, now, np.array([p.x, p.y, p.z]), 0.1, 0.9, 0.2, 0.95, 'grasp'))
            marker_id += 1
        if self.retreat_pose is not None and not self.use_joint_retreat_home:
            p = self.retreat_pose.pose.position
            arr.markers.append(self.make_sphere_marker(marker_id, self.marker_frame, now, np.array([p.x, p.y, p.z]), 1.0, 0.4, 1.0, 0.85, 'retreat'))
            marker_id += 1

        # ── Gripper contact / pinch-point markers ──────────────────────────────
        # Always shown in arm_gripper_base_link frame so they track the live arm.
        # • Centre sphere: green (open) → red (closed) based on current joint q.
        # • Two orange spheres: left (+X) and right (−X) finger tips, spread by
        #   the actual four-bar jaw gap at the current joint angle.
        q_g = float(self.current_joint_positions.get(
            self.gripper_joint_name, float(self.gripper_open)))
        contact_off = self._fourbar_actual_contact_offset(q_g)
        gap_m = fourbar.gap_from_q(q_g)
        t_g = float(np.clip(
            (q_g - float(self.gripper_open)) /
            max(float(self.gripper_close) - float(self.gripper_open), 1e-9),
            0.0, 1.0))
        # Centre pinch sphere
        mc = Marker()
        mc.header.frame_id = self.planning_link
        mc.header.stamp = now
        mc.ns = 'gripper_contact'
        mc.id = marker_id
        mc.type = Marker.SPHERE
        mc.action = Marker.ADD
        mc.pose.position.x = float(contact_off[0])
        mc.pose.position.y = float(contact_off[1])
        mc.pose.position.z = float(contact_off[2])
        mc.pose.orientation.w = 1.0
        mc.scale.x = mc.scale.y = mc.scale.z = 0.012
        mc.color = ColorRGBA(r=t_g, g=1.0 - t_g, b=0.0, a=0.92)
        mc.lifetime = Duration(sec=0, nanosec=0)
        arr.markers.append(mc)
        marker_id += 1
        # Left finger tip (+X)
        ml = Marker()
        ml.header.frame_id = self.planning_link
        ml.header.stamp = now
        ml.ns = 'gripper_finger_L'
        ml.id = marker_id
        ml.type = Marker.SPHERE
        ml.action = Marker.ADD
        ml.pose.position.x = float(gap_m / 2.0)
        ml.pose.position.y = float(contact_off[1])
        ml.pose.position.z = float(contact_off[2])
        ml.pose.orientation.w = 1.0
        ml.scale.x = ml.scale.y = ml.scale.z = 0.008
        ml.color = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.85)
        ml.lifetime = Duration(sec=0, nanosec=0)
        arr.markers.append(ml)
        marker_id += 1
        # Right finger tip (−X)
        mr = Marker()
        mr.header.frame_id = self.planning_link
        mr.header.stamp = now
        mr.ns = 'gripper_finger_R'
        mr.id = marker_id
        mr.type = Marker.SPHERE
        mr.action = Marker.ADD
        mr.pose.position.x = float(-gap_m / 2.0)
        mr.pose.position.y = float(contact_off[1])
        mr.pose.position.z = float(contact_off[2])
        mr.pose.orientation.w = 1.0
        mr.scale.x = mr.scale.y = mr.scale.z = 0.008
        mr.color = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.85)
        mr.lifetime = Duration(sec=0, nanosec=0)
        arr.markers.append(mr)
        marker_id += 1

        if self.show_camera_visibility and self.camera_info is not None and self.latest_depth_frame:
            frustum = self.make_camera_frustum_marker(marker_id, self.latest_depth_frame, now)
            if frustum is not None:
                arr.markers.append(frustum)
        self.marker_pub.publish(arr)

    def make_deleteall_marker(self) -> Marker:
        m = Marker()
        m.action = Marker.DELETEALL
        return m

    def make_sphere_marker(self, marker_id: int, frame: str, stamp, xyz: np.ndarray, r: float, g: float, b: float, a: float, ns: str) -> Marker:
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.pose.orientation.w = 1.0
        m.scale.x = self.marker_scale
        m.scale.y = self.marker_scale
        m.scale.z = self.marker_scale
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a
        m.lifetime = Duration(sec=0, nanosec=0)
        return m

    def make_pose_axes_markers(
        self,
        marker_id: int,
        frame: str,
        stamp,
        pose: PoseStamped,
        namespace_prefix: str = 'object_pose',
        axis_length: Optional[float] = None,
    ) -> List[Marker]:
        origin = np.array([
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        ], dtype=np.float64)
        R = quat_to_matrix(pose.pose.orientation)
        axis_len = max(
            float(self.object_pose_axis_length_m) if axis_length is None else float(axis_length),
            float(self.marker_scale) * 2.0,
        )
        specs = [
            (f'{namespace_prefix}_x', 1.0, 0.2, 0.2, R[:, 0]),
            (f'{namespace_prefix}_y', 0.2, 1.0, 0.2, R[:, 1]),
            (f'{namespace_prefix}_z', 0.2, 0.6, 1.0, R[:, 2]),
        ]
        markers: List[Marker] = []
        for idx, (ns, r, g, b, axis) in enumerate(specs):
            end = origin + normalize(axis) * axis_len
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = marker_id + idx
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = max(self.camera_frustum_line_width * 1.5, 0.004)
            m.scale.y = m.scale.x * 1.8
            m.scale.z = m.scale.x * 2.2
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            m.points = [
                Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
                Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
            ]
            m.lifetime = Duration(sec=0, nanosec=0)
            markers.append(m)
        return markers

    def make_base_box_drop_markers(self, marker_id: int, stamp) -> List[Marker]:
        """Show the configured release point and its gripper-link XYZ axes."""
        pose = self.get_base_box_drop_pose()
        point = Marker()
        point.header.frame_id = pose.header.frame_id
        point.header.stamp = stamp
        point.ns = 'base_box_drop_point'
        point.id = marker_id
        point.type = Marker.SPHERE
        point.action = Marker.ADD
        point.pose = pose.pose
        point.scale.x = self.base_box_drop_marker_scale_m
        point.scale.y = self.base_box_drop_marker_scale_m
        point.scale.z = self.base_box_drop_marker_scale_m
        point.color = ColorRGBA(r=0.95, g=0.25, b=0.95, a=0.85)
        point.lifetime = Duration(sec=0, nanosec=0)

        axes = self.make_pose_axes_markers(
            marker_id + 1,
            pose.header.frame_id,
            stamp,
            pose,
            namespace_prefix='base_box_drop',
            axis_length=self.base_box_drop_marker_axes_length_m,
        )

        label = Marker()
        label.header.frame_id = pose.header.frame_id
        label.header.stamp = stamp
        label.ns = 'base_box_drop_label'
        label.id = marker_id + 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(
            x=pose.pose.position.x,
            y=pose.pose.position.y,
            z=pose.pose.position.z + self.base_box_drop_marker_scale_m,
        )
        label.pose.orientation.w = 1.0
        label.scale.z = max(0.025, self.base_box_drop_marker_scale_m * 0.65)
        label.color = ColorRGBA(r=1.0, g=0.75, b=1.0, a=0.95)
        label.text = 'BASE BOX DROP'
        label.lifetime = Duration(sec=0, nanosec=0)
        return [point, *axes, label]

    def make_camera_frustum_marker(self, marker_id: int, frame: str, stamp) -> Optional[Marker]:
        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        width = float(self.camera_info.width)
        height = float(self.camera_info.height)
        z = self.camera_visibility_range_m
        x_l = (0.0 - cx) * z / fx
        x_r = (width - cx) * z / fx
        y_t = (0.0 - cy) * z / fy
        y_b = (height - cy) * z / fy
        local_pts = [
            np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([x_l, y_t, z], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([x_r, y_t, z], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([x_r, y_b, z], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64), np.array([x_l, y_b, z], dtype=np.float64),
            np.array([x_l, y_t, z], dtype=np.float64), np.array([x_r, y_t, z], dtype=np.float64),
            np.array([x_r, y_t, z], dtype=np.float64), np.array([x_r, y_b, z], dtype=np.float64),
            np.array([x_r, y_b, z], dtype=np.float64), np.array([x_l, y_b, z], dtype=np.float64),
            np.array([x_l, y_b, z], dtype=np.float64), np.array([x_l, y_t, z], dtype=np.float64),
        ]
        if frame != self.marker_frame:
            try:
                tfm = self.tf_buffer.lookup_transform(self.marker_frame, frame, rclpy.time.Time())
                R = quat_to_matrix(tfm.transform.rotation)
                t = np.array([tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z], dtype=np.float64)
                world_pts = [R @ p + t for p in local_pts]
                frame_id = self.marker_frame
            except TransformException as exc:
                self.get_logger().warning(f'Could not transform camera frustum {frame} -> {self.marker_frame}: {exc}', throttle_duration_sec=2.0)
                return None
        else:
            world_pts = local_pts
            frame_id = frame
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = 'camera_frustum'
        m.id = marker_id
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = self.camera_frustum_line_width
        m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in world_pts]
        m.lifetime = Duration(sec=0, nanosec=0)
        return m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionGraspNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node._yolo_worker is not None:
            node._yolo_worker.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
