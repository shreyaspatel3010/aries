#!/usr/bin/env python3
"""
Vision grasp node with RViz markers and adaptive gripper sizing.

New gripper: gripper_gear_left_joint (revolute), open=-1.57 rad, closed=0.07 rad.
The jaw gap varies with joint angle. This node estimates the 3D object width
from the YOLO segmentation mask + depth image and computes the optimal close
angle per object using the calibrated four-bar gap table in
``aries_vision_grasp.fourbar`` (measured from gripper_new.xacro +
gripper_bucket.stl; e.g. the 30 mm probe needs q ≈ -0.085 rad).

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
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetPlanningScene,
    GetPositionIK,
    GetStateValidity,
)
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PlanningSceneComponents,
    PositionConstraint,
    RobotState,
)
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import ColorRGBA, Float64, Int32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from aries_vision_grasp import (
    box_drop,
    fourbar,
    grasp_verification,
    probe_alignment,
    stages,
    wrist_planning,
)
from aries_vision_grasp.image_bridge import NumpyImageBridge
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
    wrist_extension_shortfall_m,
)
from aries_vision_grasp.inference import (
    YoloWorker,
    classify_frame_age,
    load_yolo_model,
)

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


# Gripper links that legitimately touch the probe: the four-bar linkage, both
# fingertips and the wrist camera. Used both as the attached object's
# touch_links and as the ACM allowance for the detected-probe mesh in the
# world, so intentional jaw contact is not rejected as a collision.
GRIPPER_PROBE_CONTACT_LINKS = (
    'gripper_link',
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
)


@dataclass
class ProbeTrack:
    """A persistent identity for one probe across the whole grasp process.

    The per-frame shape fit produces a fresh full-model 6D pose every cycle;
    this holds the smoothed, ID-stamped pose that survives partial occlusion,
    mask flicker and other detections. New fits are associated to it by
    position + long-axis agreement (see _update_probe_track); a fit that does
    not match is rejected so the pose stays locked onto the same physical probe
    rather than jumping to a neighbour. The track is dropped only when it has
    gone unseen past probe_track_timeout_sec, letting a genuinely new object
    claim a fresh id.
    """
    track_id: int
    centre_base: np.ndarray          # full-model centre in the planning frame
    R_base: np.ndarray               # 3x3; col0 long axis (fat->tip), col2 normal
    dims: np.ndarray                 # probe extents (X, Y, Z long) in metres
    created_sec: float
    last_update_sec: float           # last associated visual observation
    last_pose_update_sec: float      # last observation that changed the 6D pose
    hit_count: int = 0
    miss_count: int = 0
    confidence: float = 0.0

    def long_axis(self) -> np.ndarray:
        return self.R_base[:, 0].reshape(3,)


# Probe STL extents in metres (X width, Y height, Z long axis). Shared by the
# initial attach and the held-probe re-alignment fit.
# Fallback probe extents (X width, Y height, Z long axis) used only until
# probe.stl has been read. The real values are measured from the mesh itself by
# _probe_dims(): hard-coding them here and in box_drop let the constants drift
# from the model when the STL was swapped, which sizes the attached collision
# body wrongly with no warning.
PROBE_STL_DIMS_FALLBACK = np.array([0.030, 0.030, 0.200], dtype=np.float64)


class VisionGraspNode(Node):
    def __init__(self) -> None:
        super().__init__('vision_grasp_node')
        # Probe mesh state first: parameter validation below reaches
        # _probe_dims() (via the base-box layout) before the main state block
        # runs, so these must already exist.
        self._probe_mesh_msg: Optional[Mesh] = None
        self._probe_mesh_load_attempted: bool = False
        self._probe_stl_dims_measured: Optional[np.ndarray] = None
        self._probe_stl_fat_end_sign: Optional[int] = None
        self._probe_stl_fat_span: Optional[Tuple[float, float]] = None
        self._probe_stl_fat_span_computed: bool = False
        self.bridge = NumpyImageBridge()

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
        self.declare_parameter('inference_result_max_age_sec', 2.0)

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
        # Fraction of the smaller box that must overlap a larger same-class box
        # before it is treated as a duplicate detection of the same object.
        self.declare_parameter('detection_nested_overlap_threshold', 0.70)
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
        # Final approach must be a pure gripper-frame insertion. One waypoint lets
        # MoveIt interpolate a straight line in the PLANNING frame, so whatever
        # lateral error is left at pre-grasp (refinement moves the target after the
        # standoff is reached) gets swept out diagonally while the jaws are already
        # down at probe height -- the fingers cut ACROSS the probe instead of
        # sliding down around it, and collision-aware Cartesian planning refuses
        # the descent even though every waypoint is IK-reachable. Splitting the
        # move fixes the geometry: take out the lateral error first at the current
        # standoff, then translate along tool +Z only.
        self.declare_parameter('final_approach_tool_frame_only', True)
        self.declare_parameter('final_approach_lateral_align_tol_m', 0.003)

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
        # Y is ~0: the fingers mount on the inboard face of the four-bar bars, so
        # the jaw line runs down the arm_gripper_base_link Z axis (the residual
        # millimetre is the finger mesh's own asymmetry). Keep this in step with
        # fourbar_contact_y_offset_m -- it seeds effective_target_point_offset_in_link
        # and is only corrected once apply_fourbar_ground_guard_to_offset runs, so a
        # stale value here skews the first target built in a sequence.
        self.declare_parameter('target_point_offset_in_link', [0.0, 0.001, 0.235])
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
        # Hard close limit only; the grasp itself uses the q_close computed from
        # the measured probe width via the calibrated fourbar table. Per that
        # table q=0.070 is gap≈0 (fully closed), so this is a travel limit, not
        # a grasp width. Do not size it from the old linear gap model
        # (0.1786 - 2q), which is wrong — see the fourbar note below.
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
        # Sized for the 30 mm probe in models/probe.stl. This is a FLOOR on the
        # measured width, so it must never exceed the real probe: a 45 mm floor
        # on a 30 mm probe computes q_close for a 41 mm jaw gap and the fingers
        # stop 11 mm short of ever touching it. _warn_on_probe_width_mismatch
        # checks these against the mesh at startup.
        self.declare_parameter('minimum_probe_width_m', 0.030)
        # Width estimator guard for a known probe. Segmentation sometimes returns
        # the probe length/visible mask diagonal as the width (e.g. 100+ mm),
        # which leaves the gripper too open. Clamp to a physical probe range.
        self.declare_parameter('nominal_probe_width_m', 0.030)
        self.declare_parameter('maximum_probe_width_m', 0.040)
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
        # Attached-probe pose sync. Without it the collision mesh assumes the
        # probe's geometric centre sits exactly at the bucket contact point
        # and uses the COMMANDED grasp orientation — an off-centre grasp
        # shifts the mesh from reality by the full grasp offset. With sync,
        # the mesh is placed from the measured link TF at close time and the
        # mask-estimated probe centre projected along the probe's long axis;
        # laterally/vertically the probe is snapped to the bucket contact,
        # which the four-bar close physically enforces. The along-axis offset
        # is clamped below half the probe length.
        self.declare_parameter('attach_probe_pose_sync_enabled', True)
        self.declare_parameter('attach_probe_max_centre_offset_m', 0.140)
        # Held-probe mesh re-alignment. While the probe mesh is attached, every
        # detection tick back-projects the YOLO26-seg mask through the depth
        # image into the gripper link frame (where a rigidly held probe is
        # stationary even while the arm moves), gates the points against the
        # currently attached box model, refines the pose with a trimmed
        # point-to-box ICP, and republishes the AttachedCollisionObject the
        # moment the fit disagrees with the published mesh. This corrects an
        # inaccurate grasp right after close and follows in-hand slips
        # immediately; the base-box drop facts (world yaw, centre and axis in
        # the link frame) are updated with it.
        # Place the attached probe mesh from the repeatable grip geometry rather
        # than from perception. The coaxial grasp always closes on the Ø30 body
        # with the tool axis down the shaft, so the held pose is a known
        # consequence of the grip. Perceiving it instead was the wrong trade: the
        # along-axis offset is what a cylinder fit resolves worst, so the mesh
        # started tens of mm off, and letting live fits correct it allowed fits
        # that had locked onto a different probe-shaped cloud to swing the mesh
        # by 27-61°, which painted the arm into START_STATE_IN_COLLISION.
        # Normally paired with attached_probe_realign_enabled: false.
        self.declare_parameter('attached_probe_fixed_grip_enabled', True)
        # Extra distance along the tool axis, ON TOP of the grip offset, from the
        # modelled four-bar contact point to the probe centre. It exists because
        # the modelled contact point is short of where the jaws physically pinch:
        # measured against Gazebo with the probe gripped, the attached mesh sat
        # 50.2 mm up the shaft from the real probe (1.9 mm lateral, 1.3° of axis
        # error, so the placement is wrong ONLY along the axis). The modelled
        # contact at z=0.220 in the link even falls 25 mm past the probe's butt
        # end at 0.245, which nothing gripping the Ø30 body (z 0.245-0.295) can
        # do -- the physical pinch is at ~0.270.
        # This corrects the MESH ONLY, deliberately. The same contact offset also
        # aims the grasp, and that aim currently lands the jaws on the fat body,
        # so moving it would break a working grasp to fix a collision model.
        self.declare_parameter('attached_probe_axial_offset_m', 0.050)
        self.declare_parameter('attached_probe_realign_enabled', True)
        self.declare_parameter('attached_probe_realign_confidence_threshold', 0.30)
        self.declare_parameter('attached_probe_realign_gate_m', 0.10)
        self.declare_parameter('attached_probe_realign_min_gate_fraction', 0.35)
        self.declare_parameter('attached_probe_realign_min_points', 80)
        self.declare_parameter('attached_probe_realign_icp_iterations', 50)
        self.declare_parameter('attached_probe_realign_max_rms_m', 0.020)
        self.declare_parameter('attached_probe_realign_position_deadband_m', 0.008)
        self.declare_parameter('attached_probe_realign_angle_deadband_deg', 4.0)
        self.declare_parameter('attached_probe_realign_confirm_samples', 2)
        self.declare_parameter('attached_probe_realign_agreement_position_m', 0.015)
        self.declare_parameter('attached_probe_realign_agreement_angle_deg', 6.0)
        self.declare_parameter('attached_probe_realign_fast_position_m', 0.030)
        self.declare_parameter('attached_probe_realign_fast_angle_deg', 12.0)
        # Ceilings on how far one correction may move the attached mesh. Every
        # other threshold here is a floor (how much change is worth acting on)
        # or a mutual-agreement window; none bounds disagreement with the
        # ATTACHED pose, so a run of fits locked onto the wrong probe-shaped
        # cloud committed 27° and 61° axis swings. Chosen above
        # attached_probe_realign_fast_angle_deg (12°) so a genuine fast slip
        # still commits, but well under the tens of degrees that only a
        # mis-association can produce.
        self.declare_parameter('attached_probe_realign_max_correction_deg', 20.0)
        # The position ceiling must be anisotropic, because the grip is. Closed
        # jaws pin the probe LATERALLY (that is what clamping means), so a fit
        # that moves the centre sideways is describing a different object. Along
        # its OWN axis the probe is free — it sits whereever it was when the jaws
        # shut, and that offset is exactly what attach time estimates worst,
        # since a cylinder sliding along its axis barely changes how it looks.
        # An isotropic ceiling here would reject the one correction most needed:
        # a measured attach was 78 mm short along the axis and laterally right.
        self.declare_parameter('attached_probe_realign_max_lateral_correction_m', 0.020)
        self.declare_parameter('attached_probe_realign_max_axial_correction_m', 0.100)
        self.declare_parameter('attached_probe_realign_min_republish_sec', 0.2)
        self.declare_parameter('attached_probe_realign_stale_warn_sec', 4.0)
        # Re-acquisition: tracking gates measurements against the currently
        # attached box pose, so a grossly wrong initial attach (flipped or
        # far-off mesh) rejects every observation and can never self-correct.
        # After reacquire_after_sec without a gated measurement, a confident
        # probe mask whose points sit within reacquire_max_dist_m of the
        # gripper link is fitted from scratch (PCA-initialised box ICP) and,
        # once consecutive fits agree, replaces the attached pose outright.
        self.declare_parameter('attached_probe_realign_reacquire_after_sec', 2.0)
        self.declare_parameter('attached_probe_realign_reacquire_max_dist_m', 0.40)
        self.declare_parameter('attached_probe_realign_reacquire_confirm_samples', 2)

        # Held-probe verification. An empty close is otherwise silent: the
        # controller reaches the deliberately over-closed target because
        # nothing stopped it, the probe is no longer visible at its old floor
        # pose, and every later stage reports a held object. A ProbeRealign box
        # fit on the jaw axis supplies positive held-object evidence.
        self.declare_parameter('held_probe_verification_enabled', True)
        self.declare_parameter('held_probe_region_radius_m', 0.055)
        self.declare_parameter('held_probe_region_along_min_m', -0.030)
        self.declare_parameter('held_probe_region_along_max_m', 0.170)
        self.declare_parameter('held_probe_evidence_window_sec', 8.0)
        self.declare_parameter('held_probe_evidence_min_votes', 3)
        self.declare_parameter('held_probe_evidence_min_held_votes', 2)
        self.declare_parameter('held_probe_evidence_empty_fraction', 0.75)
        # An EMPTY verdict during transport means the arm is carrying nothing
        # and the attached mesh is a lie. Stop instead of finishing the task.
        self.declare_parameter('held_probe_abort_transport_on_empty', True)
        # A lift check that saw no detection at all used to PASS: absence of
        # evidence was read as success. With this on, a timeout needs positive
        # evidence -- a held-probe verdict or trusted jaw contact -- and an
        # empty final close vetoes both.
        self.declare_parameter('lift_check_require_positive_evidence', True)
        # Back out of the grasp along the reverse of the final descent instead
        # of translating straight up in the world frame. The probe goes in
        # coaxially along tool +Z; a world-vertical lift on a pitched tool has
        # a large component ACROSS the shaft, which levers the clamped probe
        # against its socket and ends at a pose the transport planner has never
        # visited. Retracing the descent ends at the standoff the arm reached
        # seconds earlier.
        self.declare_parameter('post_grasp_retrace_descent', True)
        # Margin above grasp_success_min_lift_m that the retrace must clear
        # before the vertical top-up is skipped entirely.
        self.declare_parameter('post_grasp_retrace_extra_rise_m', 0.015)
        self.declare_parameter('movegroup_replan_enabled', True)
        self.declare_parameter('movegroup_replan_attempts', 5)
        self.declare_parameter('movegroup_replan_delay_sec', 0.5)

        # Detected probe published as its own STL mesh in the planning scene.
        self.declare_parameter('world_probe_object_enabled', True)
        self.declare_parameter('world_probe_object_id', 'detected_probe')
        self.declare_parameter('world_probe_object_min_republish_sec', 0.5)
        self.declare_parameter('world_probe_object_move_threshold_m', 0.010)
        # Held-probe transport goal validation. At the calibrated pick_home
        # posture the probe hangs directly over the chassis front, so with a
        # long attached probe the goal state itself is in collision and
        # MoveGroup aborts within milliseconds (status 6). Each transport
        # candidate (pick_home first, then the flattened 6-joint alternatives
        # below) is checked with /check_state_validity against the live
        # planning scene INCLUDING the attached probe mesh; the first
        # collision-free posture is commanded and OMPL then plans a
        # collision-aware path to it.
        self.declare_parameter('transport_goal_validity_check_enabled', True)
        self.declare_parameter('state_validity_service_name', '/check_state_validity')
        # Defaults: arm extended forward so the probe hangs ahead of the
        # chassis (tool ~(0.56, -0.05, 0.43) in base_link), then the same
        # posture yawed right by 0.9 and 1.5 rad over open ground.
        self.declare_parameter('pick_home_alternative_joint_positions_flat', [
            0.0, 1.074, 0.639, 0.0349066, 1.394, 1.50098,
            -0.9, 1.074, 0.639, 0.0349066, 1.394, 1.50098,
            -1.5, 1.074, 0.639, 0.0349066, 1.394, 1.50098,
        ])
        self.declare_parameter('pick_home_joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('pick_home_joint_positions', [0.0, 0.366519, 1.18682, 0.0349066, 1.55334, 1.50098])

        # Base-box placement.  The arm first reaches pick_home with the probe
        # attached, then moves to the calibrated SRDF ``pick_drop`` posture,
        # opens the gripper, detaches the probe from MoveIt's planning scene,
        # and returns to pick_home.  Only arm joints are commanded for the two
        # transport poses so the gripper cannot open before the release stage.
        self.declare_parameter('place_in_base_box_after_grasp', False)
        # Automatic box-derived placement. The box pose describes its centre;
        # dimensions are its outside local XYZ size. A continuous release
        # volume is calculated above the top rim; MoveIt may choose any point
        # and any tool orientation inside it.
        self.declare_parameter('base_box_auto_drop_enabled', True)
        self.declare_parameter('base_box_center_xyz', [0.15, 0.20, 0.18])
        self.declare_parameter('base_box_dimensions_xyz', [0.18, 0.36, 0.12])
        self.declare_parameter('base_box_rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('base_box_drop_use_pose', False)
        self.declare_parameter('base_box_drop_frame', 'base_link')
        self.declare_parameter('base_box_drop_xyz', [0.45078823, 0.07073892, 0.64813140])
        self.declare_parameter('base_box_drop_rpy', [2.05331746, 0.12332939, 1.83238021])
        self.declare_parameter('base_box_drop_target_point_offset_in_link', [0.0, 0.001, 0.2180])
        # Legacy fixed-pose probe alignment. Automatic placement intentionally
        # does not use this constraint: arbitrary probe/tool orientation is
        # allowed over the box so IK is not blocked by a requested wrist yaw.
        self.declare_parameter('base_box_drop_align_attached_probe', True)
        self.declare_parameter('base_box_drop_probe_axis_world_yaw_rad', 1.5708)
        self.declare_parameter('base_box_drop_position_tolerance_m', 0.015)
        self.declare_parameter('base_box_drop_orientation_tolerance_rad', 0.10)
        # Escalating base-box release ladder. When every automatic wrist
        # candidate fails, the drop is retried in rounds instead of locking:
        # round 2 rebuilds the candidates from the CURRENT attached-probe
        # geometry (a held-probe re-alignment may have corrected the mesh
        # since round 1) with the relaxed orientation tolerance below; the
        # final round drops the orientation constraint entirely and plans the
        # probe centre into the release volume position-only. Release
        # verification uses the normal axis tolerance for oriented rounds and
        # the final tolerance for the position-only round.
        self.declare_parameter('base_box_drop_relaxed_orientation_tolerance_rad', 0.35)
        self.declare_parameter('base_box_drop_final_round_position_only', True)
        self.declare_parameter('base_box_release_axis_tolerance_deg', 12.0)
        self.declare_parameter('base_box_release_axis_tolerance_final_deg', 60.0)
        self.declare_parameter('base_box_drop_joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('base_box_drop_joint_positions', [0.15708, -0.837758, 1.93732, 0.15708, 0.959931, 1.27409])
        # Tilted insertion release. The probe is longer than the box's usable
        # opening, so a horizontal centred pose above the rim can only drop it
        # across the mouth. Instead lean it in: the leading end goes
        # base_box_insert_depth_m below the rim at base_box_insert_tilt_deg
        # above the top plane, displaced along the long axis so the probe leaves
        # through the mouth rather than clipping a wall. Shallow tilts come
        # first — they need the least vertical drop, so for a probe short enough
        # to fit the interior diagonal the release ends up fully inside the box
        # rather than leaning out. Each tilt is checked for opening clearance
        # before it costs any planning time.
        self.declare_parameter('base_box_insert_enabled', True)
        self.declare_parameter('base_box_insert_tilt_options_deg', [30.0, 45.0, 60.0])
        self.declare_parameter('base_box_insert_depth_m', 0.050)
        self.declare_parameter('base_box_insert_entry_offset_m', 0.035)
        # Reject unreachable release candidates with a cheap IK query instead of
        # a full planning timeout. Without this every unreachable wrist option
        # costs base_box_planning_time_sec (measured: 10.2 s each, 8 candidates
        # deep) before the ladder moves on.
        self.declare_parameter('base_box_ik_prescreen_enabled', True)
        self.declare_parameter('base_box_ik_prescreen_timeout_sec', 0.5)
        # Hard upper bound on candidates handed to the planner in each
        # insertion strategy. The full candidate set is cheap to IK-screen but
        # expensive to plan; this cap also covers mixed IK-success/OMPL-failure
        # runs so they cannot consume tens of full planning timeouts.
        self.declare_parameter('base_box_max_planned_candidates', 2)
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
        # linear gap model, which incorrectly made q≈+0.068 rad for a wide
        # probe; the real STL geometry gives almost zero jaw gap at that angle.
        # These contact offsets are gripper geometry indexed by q, so they hold
        # across probe sizes: contact z moves under 1 mm between the q for a
        # 30 mm probe (-0.085) and a 45 mm one (-0.20).
        # Which physically swappable fingertip is mounted. The bucket,
        # maintenance and probe fingers share the four-bar pivots but have
        # very different jaw geometry, so this selects the matching gap and
        # contact tables in fourbar. It MUST match the finger_type the URDF
        # was launched with, or the attached probe mesh lands ~30 mm off.
        self.declare_parameter('finger_type', 'bucket')
        self.declare_parameter('fourbar_contact_y_offset_m', 0.001)
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
        # Shape-aware 6D pose. Instead of reporting the visible mask centroid --
        # which slides toward the exposed body when the tip is buried -- fit the
        # KNOWN probe box (probe.stl extents) to the masked cloud and report the
        # full-model centre + axis. Predicts the occluded part, gives correct
        # dimensions, and yields a pose that stays put as occlusion changes. Falls
        # back to the raw PCA pose when the fit is unreliable.
        self.declare_parameter('shape_aware_pose_enabled', True)
        self.declare_parameter('shape_fit_icp_iterations', 6)
        self.declare_parameter('shape_fit_trim_fraction', 0.10)
        self.declare_parameter('shape_fit_max_rms_m', 0.012)
        self.declare_parameter('shape_fit_min_inliers', 50)
        # Persistent probe identity. The first stable shape fit is stamped with a
        # track id; later fits are associated to it only if the centre is within
        # probe_track_max_jump_m and the long axis within probe_track_max_axis_deg,
        # so the pose locks onto one physical probe for the whole process instead
        # of hopping between detections. A non-matching fit is rejected (the held
        # pose is kept) unless the track has gone unseen past the timeout.
        self.declare_parameter('probe_track_enabled', True)
        self.declare_parameter('probe_track_max_jump_m', 0.08)
        self.declare_parameter('probe_track_max_axis_deg', 25.0)
        self.declare_parameter('probe_track_axis_min_eigenratio', 6.0)
        self.declare_parameter('probe_track_raw_validation_max_jump_m', 0.035)
        self.declare_parameter('probe_track_pose_hold_sec', 3.0)
        # How long a decisive fat-end (taper) resolution stays valid for frames
        # that cannot decide. Without this the centre convention and the axis
        # direction both flip frame to frame on marginal clouds.
        self.declare_parameter('probe_fat_dir_latch_sec', 3.0)
        self.declare_parameter('probe_track_timeout_sec', 2.0)
        self.declare_parameter('probe_track_position_smoothing', 0.5)
        self.declare_parameter('probe_track_axis_smoothing', 0.5)
        self.declare_parameter('object_track_id_topic', '/vision_grasp/object_track_id')
        # A probe standing upright (e.g. planted in sand) cannot be picked with the
        # fixed top-down (roll=pi, pitch=0) approach aimed at the object centroid:
        # that drives the jaws onto the pointy tip. This gripper is built to hold the
        # probe COAXIALLY -- probe long axis along the gripper's own long axis (tool
        # +Z), the wide cylindrical body clamped in the jaws and the tapered tip
        # protruding past the fingers. When the detected long axis is within this many
        # degrees of vertical, align tool +Z with the shaft and slide the grip up onto
        # the fat body. See _compute_axial_grasp_orientation.
        self.declare_parameter('vertical_grasp_enabled', True)
        self.declare_parameter('vertical_grasp_max_tilt_from_vertical_deg', 45.0)
        # Distance to slide the grasp contact up the shaft (away from the tip, toward
        # the exposed fat body) for a near-vertical coaxial grasp. Grabbing the body
        # instead of the centroid both clamps the wide cylindrical section the jaws
        # are sized for and lifts the wrist closer to the shoulder so the pose stays
        # inside the reach envelope. Clamped so the grip never runs off the fat end.
        # With _auto (default) the distance is measured from probe.stl -- the middle
        # of its widest section -- instead of using the constant below, which is only
        # the fallback for an unreadable mesh. The constant cannot be kept correct by
        # hand: probe.stl's body centre is 75 mm from the reported centroid, so the
        # old 60 mm default gripped the neck 10 mm below the shoulder.
        # Grasping a round probe leaves the rotation about its axis free, and the
        # jaws are 180°-symmetric on top of that. Spend both on keeping the wrist
        # near where it already is instead of on the probe's (arbitrary) lean
        # direction. Pure symmetry exploitation: the grasp itself is identical.
        self.declare_parameter('grasp_azimuth_follow_wrist', True)
        # Before moving, evaluate equivalent gripper azimuths with MoveIt IK at
        # BOTH pre-grasp and final grasp. Choose the branch with the least live
        # joint motion, heavily weighting bounded joint6 and avoiding its
        # limits. Tool-quaternion proximity alone cannot distinguish IK
        # families and was observed to send joint6 the long way around.
        self.declare_parameter('grasp_joint_aware_orientation_search_enabled', True)
        self.declare_parameter('grasp_joint_aware_azimuth_candidates', 12)
        self.declare_parameter('grasp_joint_aware_ik_timeout_sec', 0.20)
        self.declare_parameter('grasp_joint_aware_cartesian_timeout_sec', 0.75)
        self.declare_parameter('grasp_wrist_joint_name', 'joint6')
        self.declare_parameter('grasp_wrist_joint_weight', 6.0)
        self.declare_parameter('grasp_wrist_joint_lower_rad', -3.12414)
        self.declare_parameter('grasp_wrist_joint_upper_rad', 3.12414)
        self.declare_parameter('grasp_wrist_joint_limit_margin_rad', 0.30)
        # A final descent that stops short is only a failure if the jaws miss the
        # probe. The grip aims at the MIDDLE of the wide body, so a shortfall under
        # that body's half-length still closes on the same section, and does it
        # further from the floor. Closing there beats resetting a committed grasp.
        self.declare_parameter('final_approach_accept_partial_enabled', True)
        self.declare_parameter('final_approach_accept_shortfall_margin_m', 0.0)
        self.declare_parameter('final_approach_accept_shortfall_fallback_m', 0.020)
        self.declare_parameter('vertical_grasp_body_offset_auto', True)
        self.declare_parameter('vertical_grasp_body_offset_m', 0.075)
        self.declare_parameter('vertical_grasp_body_offset_end_margin_m', 0.030)

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
        self.declare_parameter('target_lock_continue_min_confidence', 0.58)
        self.declare_parameter('target_stability_min_pose_fit_samples', 3)

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
        # Gazebo's bucket/probe collision mesh repeatedly settles near
        # q=-0.105 (calibrated gap ~=29.4 mm), and hardware runs have stalled
        # as tight as q=-0.089 (26.75 mm): physics penetration and off-square
        # contact make the calibrated gap read tighter than the true probe
        # width. A 24 mm geometry allowance keeps those observed rigid-probe
        # stops inside the bounded contact window while still rejecting a
        # nearly/full-closed miss (full close reads 0 mm).
        self.declare_parameter('gripper_contact_gap_tolerance_m', 0.024)
        # Uncertain-stall completion: a final close that stalls with fresh,
        # stationary feedback, substantial closing travel, and a jaw gap that
        # is clearly not empty-closed but falls OUTSIDE the plausible probe
        # window must not hard-lock the sequence — something is between the
        # jaws, we just cannot prove it is the probe from geometry alone.
        # Complete the close without contact trust and let the vision lift
        # check decide (it requires a positive z lift to pass, and the retry
        # machinery handles a miss).
        self.declare_parameter('gripper_contact_uncertain_stall_enabled', True)
        self.declare_parameter('gripper_contact_uncertain_min_gap_m', 0.010)
        # Final close target. The computed four-bar contact angle positions the
        # jaws exactly at the ESTIMATED probe width — a few mm of width error
        # leaves the probe loose. With full close enabled the final command
        # deliberately over-closes to gripper_close_width; the rigid probe
        # stops the jaws at the true contact angle and the stalled-contact
        # check above (travel + calibrated jaw gap) completes the close. With
        # no probe in the jaws the gripper simply reaches full close and the
        # lift verification catches the miss.
        self.declare_parameter('final_close_full_close', True)
        # Keep the full-close setpoint commanded after contact completes the
        # close. Contact completion cancels the gripper action, and JTC answers
        # a cancel by holding the MEASURED position — which silently abandons
        # the deliberate over-close and drops the clamping load to whatever
        # penetration happens to remain. The probe is then retained by friction
        # alone and can be lost during transport while the close still reports
        # success. Re-asserting the target keeps the jaws loaded against it.
        self.declare_parameter('reassert_close_after_contact', True)
        # ...but re-asserting the FULL-close target presses without limit. On a
        # rigid flat-sided object that just holds; on the probe's tapered shaft
        # it acts as a wedge and walks the jaws shut, squeezing the probe out —
        # measured creeping from q=-0.0516 at contact to -0.0383 while "holding".
        # Re-assert a bounded overshoot past the angle contact actually stopped
        # at instead: enough travel to keep the jaws loaded, not enough to walk
        # them closed. 0.02 rad is ~3 mm of jaw gap on the four-bar curve.
        self.declare_parameter('reassert_close_overshoot_rad', 0.02)
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

        # Pre-flight reachability guard. The final Cartesian descent fails deep
        # into the sequence (and triggers the lockout above) when the grasp
        # point is outside the arm workspace, so the committed grasp link pose
        # is checked against a shoulder-sphere model before any motion starts.
        # Defaults match igus_rebel2.urdf + gripper_new.xacro:
        #  - shoulder xyz: joint1 axis x/y and joint2 pitch-axis height in base_link
        #  - max wrist extension: upper arm + forearm (joint2->joint5)
        #  - wrist backoff: joint5 -> arm_gripper_base_link along tool +Z
        self.declare_parameter('reach_guard_enabled', True)
        self.declare_parameter('reach_guard_shoulder_xyz_in_base', [0.05121, 0.0, 0.4864])
        self.declare_parameter('reach_guard_max_wrist_extension_m', 0.5410)
        self.declare_parameter('reach_guard_wrist_backoff_in_link_m', 0.12903)
        self.declare_parameter('reach_guard_margin_m', 0.010)

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
        self.inference_result_max_age_sec = max(
            0.1, float(p('inference_result_max_age_sec').value)
        )

        self.refine_confidence_threshold = float(p('refine_confidence_threshold').value)
        self.refine_use_projection_fallback = bool(p('refine_use_projection_fallback').value)
        self.refine_projection_roi_half_size_px = int(p('refine_projection_roi_half_size_px').value)
        self.refine_min_depth_m = float(p('refine_min_depth_m').value)
        self.refine_depth_band_m = float(p('refine_depth_band_m').value)

        self.use_segmentation_mask = bool(p('use_segmentation_mask').value)
        self.mask_score_threshold = float(p('mask_score_threshold').value)
        self.mask_min_pixels = int(p('mask_min_pixels').value)
        self.detection_nested_overlap_threshold = float(
            np.clip(float(p('detection_nested_overlap_threshold').value), 0.05, 1.0)
        )
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
        self.final_approach_tool_frame_only = bool(p('final_approach_tool_frame_only').value)
        self.final_approach_lateral_align_tol_m = float(
            p('final_approach_lateral_align_tol_m').value
        )

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
        self.attach_probe_pose_sync_enabled = bool(p('attach_probe_pose_sync_enabled').value)
        self.attach_probe_max_centre_offset_m = max(0.0, float(p('attach_probe_max_centre_offset_m').value))
        self.attached_probe_fixed_grip_enabled = bool(
            p('attached_probe_fixed_grip_enabled').value)
        self.attached_probe_axial_offset_m = float(
            p('attached_probe_axial_offset_m').value)
        self.attached_probe_realign_enabled = bool(p('attached_probe_realign_enabled').value)
        self.attached_probe_realign_confidence_threshold = float(
            p('attached_probe_realign_confidence_threshold').value)
        self.attached_probe_realign_gate_m = max(
            0.01, float(p('attached_probe_realign_gate_m').value))
        self.attached_probe_realign_min_gate_fraction = float(np.clip(
            float(p('attached_probe_realign_min_gate_fraction').value), 0.0, 1.0))
        self.attached_probe_realign_min_points = max(
            12, int(p('attached_probe_realign_min_points').value))
        self.attached_probe_realign_icp_iterations = max(
            1, int(p('attached_probe_realign_icp_iterations').value))
        self.attached_probe_realign_max_rms_m = max(
            0.001, float(p('attached_probe_realign_max_rms_m').value))
        self.attached_probe_realign_position_deadband_m = max(
            0.0, float(p('attached_probe_realign_position_deadband_m').value))
        self.attached_probe_realign_angle_deadband_deg = max(
            0.0, float(p('attached_probe_realign_angle_deadband_deg').value))
        self.attached_probe_realign_confirm_samples = max(
            1, int(p('attached_probe_realign_confirm_samples').value))
        self.attached_probe_realign_agreement_position_m = max(
            0.001, float(p('attached_probe_realign_agreement_position_m').value))
        self.attached_probe_realign_agreement_angle_deg = max(
            0.5, float(p('attached_probe_realign_agreement_angle_deg').value))
        self.attached_probe_realign_fast_position_m = max(
            0.0, float(p('attached_probe_realign_fast_position_m').value))
        self.attached_probe_realign_fast_angle_deg = max(
            0.0, float(p('attached_probe_realign_fast_angle_deg').value))
        # Never below the fast-slip thresholds: a ceiling under the floor that
        # decides "act immediately" would reject every correction it let through.
        self.attached_probe_realign_max_correction_deg = max(
            self.attached_probe_realign_fast_angle_deg,
            float(p('attached_probe_realign_max_correction_deg').value))
        self.attached_probe_realign_max_lateral_correction_m = max(
            0.0, float(p('attached_probe_realign_max_lateral_correction_m').value))
        # Never below half the probe: sliding further than that would put the
        # grip off the end of the mesh, so it is a real bound rather than a
        # tuning knob.
        self.attached_probe_realign_max_axial_correction_m = min(
            max(0.0, float(p('attached_probe_realign_max_axial_correction_m').value)),
            0.5 * float(self._probe_dims()[2]),
        )
        self.attached_probe_realign_min_republish_sec = max(
            0.0, float(p('attached_probe_realign_min_republish_sec').value))
        self.attached_probe_realign_stale_warn_sec = max(
            0.5, float(p('attached_probe_realign_stale_warn_sec').value))
        self.attached_probe_realign_reacquire_after_sec = max(
            0.5, float(p('attached_probe_realign_reacquire_after_sec').value))
        self.attached_probe_realign_reacquire_max_dist_m = max(
            0.05, float(p('attached_probe_realign_reacquire_max_dist_m').value))
        self.attached_probe_realign_reacquire_confirm_samples = max(
            1, int(p('attached_probe_realign_reacquire_confirm_samples').value))
        self.held_probe_verification_enabled = bool(p('held_probe_verification_enabled').value)
        self.held_probe_region_radius_m = max(0.005, float(p('held_probe_region_radius_m').value))
        self.held_probe_region_along_min_m = float(p('held_probe_region_along_min_m').value)
        self.held_probe_region_along_max_m = float(p('held_probe_region_along_max_m').value)
        if self.held_probe_region_along_max_m <= self.held_probe_region_along_min_m:
            self.held_probe_region_along_max_m = self.held_probe_region_along_min_m + 0.05
        self.held_probe_abort_transport_on_empty = bool(
            p('held_probe_abort_transport_on_empty').value)
        self.lift_check_require_positive_evidence = bool(
            p('lift_check_require_positive_evidence').value)
        self.post_grasp_retrace_descent = bool(p('post_grasp_retrace_descent').value)
        self.post_grasp_retrace_extra_rise_m = max(
            0.0, float(p('post_grasp_retrace_extra_rise_m').value))
        self.movegroup_replan_enabled = bool(p('movegroup_replan_enabled').value)
        self.movegroup_replan_attempts = max(1, int(p('movegroup_replan_attempts').value))
        self.movegroup_replan_delay_sec = max(0.0, float(p('movegroup_replan_delay_sec').value))
        self.world_probe_object_enabled = bool(p('world_probe_object_enabled').value)
        self.world_probe_object_id = str(p('world_probe_object_id').value)
        self.world_probe_object_min_republish_sec = max(
            0.0, float(p('world_probe_object_min_republish_sec').value))
        self.world_probe_object_move_threshold_m = max(
            0.0, float(p('world_probe_object_move_threshold_m').value))
        self.transport_goal_validity_check_enabled = bool(p('transport_goal_validity_check_enabled').value)
        self.state_validity_service_name = str(p('state_validity_service_name').value)
        self.pick_home_alternative_joint_positions_flat = [
            float(v) for v in p('pick_home_alternative_joint_positions_flat').value
        ]
        self.pick_home_joint_names = list(p('pick_home_joint_names').value)
        self.pick_home_joint_positions = [float(v) for v in p('pick_home_joint_positions').value]
        self.place_in_base_box_after_grasp = bool(p('place_in_base_box_after_grasp').value)
        self.base_box_auto_drop_enabled = bool(p('base_box_auto_drop_enabled').value)
        self.base_box_center_xyz = [float(v) for v in p('base_box_center_xyz').value]
        self.base_box_dimensions_xyz = [float(v) for v in p('base_box_dimensions_xyz').value]
        self.base_box_rpy = [float(v) for v in p('base_box_rpy').value]
        self.base_box_drop_use_pose = bool(p('base_box_drop_use_pose').value)
        self.base_box_drop_frame = str(p('base_box_drop_frame').value)
        self.base_box_drop_xyz = [float(v) for v in p('base_box_drop_xyz').value]
        self.base_box_drop_rpy = [float(v) for v in p('base_box_drop_rpy').value]
        self.base_box_drop_target_point_offset_in_link = [
            float(v) for v in p('base_box_drop_target_point_offset_in_link').value
        ]
        self.base_box_drop_align_attached_probe = bool(p('base_box_drop_align_attached_probe').value)
        self.base_box_drop_probe_axis_world_yaw_rad = float(p('base_box_drop_probe_axis_world_yaw_rad').value)
        self.base_box_drop_position_tolerance_m = max(
            0.001, float(p('base_box_drop_position_tolerance_m').value)
        )
        self.base_box_drop_orientation_tolerance_rad = max(
            0.01, float(p('base_box_drop_orientation_tolerance_rad').value)
        )
        self.base_box_drop_relaxed_orientation_tolerance_rad = max(
            0.01, float(p('base_box_drop_relaxed_orientation_tolerance_rad').value)
        )
        self.base_box_drop_final_round_position_only = bool(
            p('base_box_drop_final_round_position_only').value
        )
        self.base_box_release_axis_tolerance_deg = max(
            1.0, float(p('base_box_release_axis_tolerance_deg').value)
        )
        self.base_box_release_axis_tolerance_final_deg = max(
            self.base_box_release_axis_tolerance_deg,
            float(p('base_box_release_axis_tolerance_final_deg').value),
        )
        self.base_box_drop_joint_names = list(p('base_box_drop_joint_names').value)
        self.base_box_drop_joint_positions = [float(v) for v in p('base_box_drop_joint_positions').value]
        self.base_box_insert_enabled = bool(p('base_box_insert_enabled').value)
        self.base_box_insert_tilt_options_deg = [
            float(v) for v in p('base_box_insert_tilt_options_deg').value
        ] or [45.0]
        self.base_box_insert_depth_m = max(0.0, float(p('base_box_insert_depth_m').value))
        self.base_box_insert_entry_offset_m = float(p('base_box_insert_entry_offset_m').value)
        self.base_box_ik_prescreen_enabled = bool(p('base_box_ik_prescreen_enabled').value)
        self.base_box_ik_prescreen_timeout_sec = max(
            0.05, float(p('base_box_ik_prescreen_timeout_sec').value)
        )
        self.base_box_max_planned_candidates = max(
            1, int(p('base_box_max_planned_candidates').value)
        )
        self.base_box_planning_time_sec = max(1.0, float(p('base_box_planning_time_sec').value))
        self.base_box_release_wait_sec = max(0.0, float(p('base_box_release_wait_sec').value))
        self.return_pick_home_after_base_box_place = bool(p('return_pick_home_after_base_box_place').value)
        self.base_box_drop_marker_enabled = bool(p('base_box_drop_marker_enabled').value)
        self.base_box_drop_marker_scale_m = max(0.01, float(p('base_box_drop_marker_scale_m').value))
        self.base_box_drop_marker_axes_length_m = max(
            0.02, float(p('base_box_drop_marker_axes_length_m').value)
        )
        if not self._base_box_drop_pose_config_valid():
            self.get_logger().error('Invalid base-box configuration. Automatic mode requires a non-empty frame, '
                'three-value centre/RPY/dimensions, positive dimensions, and at least one non-negative release '
                'height. Legacy pose mode requires three-value drop XYZ/RPY. Placement and its marker are disabled.')
        self.floor_safe_grasp_enabled = bool(p('floor_safe_grasp_enabled').value)
        self.max_grasp_descent_below_target_m = float(p('max_grasp_descent_below_target_m').value)
        self.min_grasp_height_above_floor_m = float(p('min_grasp_height_above_floor_m').value)

        requested_finger = str(p('finger_type').value)
        self.finger_type = fourbar.set_finger(requested_finger)
        if self.finger_type != requested_finger.strip().lower():
            self.get_logger().error(
                f'finger_type="{requested_finger}" is not one of '
                f'{list(fourbar.FINGER_TYPES)}; falling back to '
                f'"{self.finger_type}" geometry. A wrong finger here places the '
                f'attached probe mesh about 30 mm off the real probe.')

        self.fourbar_contact_y_offset_m = float(p('fourbar_contact_y_offset_m').value)
        self.fourbar_contact_z_open_m = float(p('fourbar_contact_z_open_m').value)
        self.fourbar_contact_z_closed_m = float(p('fourbar_contact_z_closed_m').value)
        # The configured contact heights are the bucket's calibrated values.
        # For any other finger they describe the wrong jaw, so take the
        # mounted finger's own contact geometry from its table. For the bucket
        # the table reproduces the configured values exactly, so this only
        # changes behaviour when a different finger is actually selected.
        if self.finger_type != fourbar.DEFAULT_FINGER:
            derived_open = fourbar.contact_offset_z(fourbar.Q_MIN)
            derived_closed = fourbar.contact_offset_z(-0.200)
            self.get_logger().info(
                f'[FourBar] {self.finger_type} finger mounted: contact z '
                f'{self.fourbar_contact_z_open_m*1000:.0f}->{derived_open*1000:.0f} mm open, '
                f'{self.fourbar_contact_z_closed_m*1000:.0f}->{derived_closed*1000:.0f} mm closed '
                f'(configured values describe the bucket).')
            self.fourbar_contact_z_open_m = derived_open
            self.fourbar_contact_z_closed_m = derived_closed
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
        self.vertical_grasp_enabled = bool(p('vertical_grasp_enabled').value)
        self.vertical_grasp_max_tilt_from_vertical_deg = float(
            p('vertical_grasp_max_tilt_from_vertical_deg').value
        )
        self.grasp_azimuth_follow_wrist = bool(p('grasp_azimuth_follow_wrist').value)
        self.grasp_joint_aware_orientation_search_enabled = bool(
            p('grasp_joint_aware_orientation_search_enabled').value
        )
        self.grasp_joint_aware_azimuth_candidates = max(
            2, int(p('grasp_joint_aware_azimuth_candidates').value)
        )
        self.grasp_joint_aware_ik_timeout_sec = max(
            0.05, float(p('grasp_joint_aware_ik_timeout_sec').value)
        )
        self.grasp_joint_aware_cartesian_timeout_sec = max(
            0.10, float(p('grasp_joint_aware_cartesian_timeout_sec').value)
        )
        self.grasp_wrist_joint_name = str(p('grasp_wrist_joint_name').value)
        self.grasp_wrist_joint_weight = max(
            1.0, float(p('grasp_wrist_joint_weight').value)
        )
        self.grasp_wrist_joint_lower_rad = float(
            p('grasp_wrist_joint_lower_rad').value
        )
        self.grasp_wrist_joint_upper_rad = float(
            p('grasp_wrist_joint_upper_rad').value
        )
        self.grasp_wrist_joint_limit_margin_rad = max(
            0.0, float(p('grasp_wrist_joint_limit_margin_rad').value)
        )
        self.final_approach_accept_partial_enabled = bool(
            p('final_approach_accept_partial_enabled').value
        )
        self.final_approach_accept_shortfall_margin_m = float(
            p('final_approach_accept_shortfall_margin_m').value
        )
        self.final_approach_accept_shortfall_fallback_m = float(
            p('final_approach_accept_shortfall_fallback_m').value
        )
        self.vertical_grasp_body_offset_auto = bool(
            p('vertical_grasp_body_offset_auto').value
        )
        self.vertical_grasp_body_offset_m = float(
            p('vertical_grasp_body_offset_m').value
        )
        self.vertical_grasp_body_offset_end_margin_m = float(
            p('vertical_grasp_body_offset_end_margin_m').value
        )
        self.shape_aware_pose_enabled = bool(p('shape_aware_pose_enabled').value)
        self.shape_fit_icp_iterations = int(p('shape_fit_icp_iterations').value)
        self.shape_fit_trim_fraction = float(p('shape_fit_trim_fraction').value)
        self.shape_fit_max_rms_m = float(p('shape_fit_max_rms_m').value)
        self.shape_fit_min_inliers = int(p('shape_fit_min_inliers').value)
        self.probe_track_enabled = bool(p('probe_track_enabled').value)
        self.probe_track_max_jump_m = float(p('probe_track_max_jump_m').value)
        self.probe_track_max_axis_deg = float(p('probe_track_max_axis_deg').value)
        self.probe_track_axis_min_eigenratio = max(
            self.object_orientation_min_eigenratio,
            float(p('probe_track_axis_min_eigenratio').value),
        )
        self.probe_track_raw_validation_max_jump_m = max(
            0.001, min(
                self.probe_track_max_jump_m,
                float(p('probe_track_raw_validation_max_jump_m').value),
            )
        )
        self.probe_track_pose_hold_sec = max(
            0.0, float(p('probe_track_pose_hold_sec').value)
        )
        self.probe_fat_dir_latch_sec = float(p('probe_fat_dir_latch_sec').value)
        self.probe_track_timeout_sec = float(p('probe_track_timeout_sec').value)
        self.probe_track_position_smoothing = float(
            np.clip(p('probe_track_position_smoothing').value, 0.0, 1.0)
        )
        self.probe_track_axis_smoothing = float(
            np.clip(p('probe_track_axis_smoothing').value, 0.0, 1.0)
        )
        self.object_track_id_topic = str(p('object_track_id_topic').value)

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
        self.target_lock_continue_min_confidence = float(np.clip(
            float(p('target_lock_continue_min_confidence').value),
            0.0,
            self.target_lock_min_confidence,
        ))
        self.target_stability_min_pose_fit_samples = int(np.clip(
            int(p('target_stability_min_pose_fit_samples').value),
            1,
            self.target_stability_samples,
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
        self.final_close_full_close = bool(p('final_close_full_close').value)
        self.reassert_close_after_contact = bool(
            p('reassert_close_after_contact').value
        )
        self.reassert_close_overshoot_rad = max(
            0.0, float(p('reassert_close_overshoot_rad').value)
        )
        self.gripper_contact_gap_tolerance_m = max(
            0.0, float(p('gripper_contact_gap_tolerance_m').value)
        )
        self.gripper_contact_uncertain_stall_enabled = bool(
            p('gripper_contact_uncertain_stall_enabled').value
        )
        self.gripper_contact_uncertain_min_gap_m = max(
            0.0, float(p('gripper_contact_uncertain_min_gap_m').value)
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

        self.reach_guard_enabled = bool(p('reach_guard_enabled').value)
        self.reach_guard_shoulder_xyz_in_base = np.array(
            p('reach_guard_shoulder_xyz_in_base').value, dtype=np.float64
        )
        self.reach_guard_max_wrist_extension_m = float(p('reach_guard_max_wrist_extension_m').value)
        self.reach_guard_wrist_backoff_in_link_m = float(p('reach_guard_wrist_backoff_in_link_m').value)
        self.reach_guard_margin_m = float(p('reach_guard_margin_m').value)

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
        self.latest_color_frame: str = ''
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
        self._attached_probe_world_yaw: Optional[float] = None
        self._attached_probe_grasp_orientation: Optional[Quaternion] = None
        self._attached_probe_centre_in_link: Optional[np.ndarray] = None
        self._attached_probe_axis_in_link: Optional[np.ndarray] = None
        self._attached_probe_tip_resolved: bool = False
        # Held-probe re-alignment: full attached box pose, the STL mesh cache
        # (loaded once, republished on every correction), and the recent
        # link-frame measurements awaiting confirmation.
        self._attached_probe_R_stl_in_link: Optional[np.ndarray] = None
        # Probe mesh state (_probe_mesh_msg, _probe_mesh_load_attempted,
        # _probe_stl_dims_measured, _probe_stl_fat_end_sign) is initialised at
        # the top of __init__ because parameter validation needs it earlier.
        self._probe_realign_measurements: deque = deque(maxlen=6)
        self._probe_reacquire_measurements: deque = deque(maxlen=4)
        self._probe_realign_last_commit_sec = 0.0
        self._probe_realign_last_measurement_sec = 0.0
        # Held-probe verification: pooled ProbeRealign evidence and the
        # empty-close latch set by the final close. The latch is a hard veto --
        # it means the jaws measurably shut
        # past the probe's width, so no later "PASSED by timeout" may override
        # it without positive evidence.
        self._held_probe_evidence = grasp_verification.HeldProbeEvidence(
            window_sec=float(p('held_probe_evidence_window_sec').value),
            min_votes=int(p('held_probe_evidence_min_votes').value),
            min_held_votes=int(p('held_probe_evidence_min_held_votes').value),
            empty_fraction=float(p('held_probe_evidence_empty_fraction').value),
        )
        self._empty_close_detected: bool = False
        self._empty_close_gap_m: Optional[float] = None
        self._held_probe_empty_reported: bool = False
        # planning_link first: it carries the attached body, and the
        # AttachedCollisionObject's own link must always be a touch link.
        self.gripper_probe_contact_links = [self.planning_link] + [
            link for link in GRIPPER_PROBE_CONTACT_LINKS if link != self.planning_link
        ]
        # Detected (not yet grasped) probe published as an STL collision object.
        self._world_probe_published: bool = False
        self._world_probe_acm_applied: bool = False
        self._world_probe_acm_pending: bool = False
        self._world_probe_last_centre: Optional[np.ndarray] = None
        self._world_probe_last_publish_sec: float = 0.0
        self._base_box_drop_candidates: List[dict] = []
        self._base_box_drop_candidate_index: int = -1
        self._base_box_drop_round: int = 0
        self._base_box_drop_position_only_active: bool = False
        self._base_box_tip_fallback_active: bool = False
        self._base_box_drop_start_collision_retry_index: int = -1
        self._base_box_drop_ik_skipped: int = 0
        self._base_box_center_plans_attempted: int = 0
        self._base_box_tip_plans_attempted: int = 0
        self._base_box_ik_request_token: int = 0
        # Set once the screen has rejected a whole round: from then on the
        # candidates are planned directly instead of screened away.
        self._base_box_ik_screen_exhausted: bool = False
        self._base_box_ik_reject_codes: dict = {}
        self._active_base_box_drop_pose: Optional[PoseStamped] = None
        self._computed_base_box_probe_axis_yaw_rad: Optional[float] = None
        # Mask pose frozen at final grasp commit: the camera is buried during
        # the descent and close, so the live detection goes stale before the
        # probe mesh is attached.
        self._grasp_time_object_pose: Optional[PoseStamped] = None
        self._grasp_time_object_R: Optional[np.ndarray] = None
        self._grasp_time_tip_resolved: bool = False
        self._grasp_close_link_pose: Optional[Pose] = None
        self.camera_info: Optional[CameraInfo] = None
        self._yolo_worker: Optional[YoloWorker] = None
        self._stamp_gap_warned_sec = 0.0
        self.busy = False
        self.sequence_stage = 'idle'
        self.current_target_point_base: Optional[np.ndarray] = None
        self.target_history: List[np.ndarray] = []
        self.target_history_stamps: List[float] = []
        self.target_confidence_history: List[float] = []
        # Parallel to target_history: whether each sample had a fresh 6D pose
        # fit. Kept in lockstep by _clear_target_stability_history() and the
        # window trim, and gates the stability verdict.
        self.target_pose_fit_history: List[bool] = []
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
        self._grasp_orientation_search: Optional[dict] = None
        self._grasp_orientation_search_token: int = 0
        self._grasp_pregrasp_joint_targets: dict = {}
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
        # Set when the active grasp is a near-vertical coaxial (axial) grasp:
        # the base-frame vector that shifts the contact point up the shaft onto
        # the fat body. None for ordinary top-down grasps. See
        # _compute_axial_grasp_orientation / update_contact_poses_from_target.
        self._vertical_grasp_body_shift_base: Optional[np.ndarray] = None
        self._last_final_descent_waypoints: List[Pose] = []
        self._last_final_descent_start: Optional[Pose] = None
        self._accepted_descent_shortfall_m: float = 0.0
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
        # True when the latest PCA long axis was pointed at the probe tip from
        # the cloud's own taper rather than left on eigh's arbitrary sign.
        self._object_axis_tip_resolved: bool = False
        self._last_object_orientation_eigenratio: float = 0.0
        self._probe_fat_dir_cam: Optional[np.ndarray] = None
        self._probe_fat_dir_sec: float = -1e9
        self._probe_track_association_rejected: bool = False
        self._probe_track_pose_held: bool = False
        self._last_detection_uses_held_pose: bool = False
        self._last_object_shape_fit_ok: bool = False
        self._last_shape_raw_world: Optional[np.ndarray] = None
        self._last_shape_rotation_world_camera: Optional[np.ndarray] = None
        self._camera_calibration_track_id: Optional[int] = None
        self.detected_object_yaw_rad: Optional[float] = None              # yaw in planning_frame
        # Persistent probe identity (see ProbeTrack / _update_probe_track).
        self._probe_track: Optional[ProbeTrack] = None
        self._next_probe_track_id: int = 1
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
        self.get_planning_scene_client = self.create_client(GetPlanningScene, 'get_planning_scene')
        self.apply_planning_scene_client = self.create_client(ApplyPlanningScene, 'apply_planning_scene')
        self.state_validity_client = self.create_client(GetStateValidity, self.state_validity_service_name)
        self.compute_ik_client = self.create_client(GetPositionIK, 'compute_ik')
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
        # Offer TRANSIENT_LOCAL so RViz/rqt subscribers that request a latched
        # image are compatible, while VOLATILE live viewers remain compatible
        # too. Keeping one frame also makes the panel populate immediately
        # instead of waiting for the next YOLO cycle.
        detection_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.det_vis_pub = self.create_publisher(
            Image,
            '/vision_grasp/detection_image',
            detection_image_qos,
        )
        self.gripper_pub = self.create_publisher(Float64, p('gripper_topic').value, 10)
        # JTC's topic interface, used only to re-assert the close setpoint after
        # contact completes the close. It has no goal lifecycle, so the standing
        # command cannot later abort on a goal-tolerance violation the way a
        # re-sent action goal would. Derived from the action name so both follow
        # the same controller.
        self._gripper_traj_pub = self.create_publisher(
            JointTrajectory,
            self.gripper_action_name.replace(
                '/follow_joint_trajectory', '/joint_trajectory'
            ),
            10,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, self.object_pose_topic, 10)
        self.object_track_id_pub = self.create_publisher(Int32, self.object_track_id_topic, 10)
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

        self._warn_on_probe_width_mismatch()
        self.get_logger().info(f'vision_grasp_node ready | target_class={self.target_class} | planning_group={self.planning_group} | '
            f'planning_link={self.planning_link} | planning_frame={self.planning_frame} | gripper_mode={self.gripper_command_mode}')
        if self.base_box_auto_drop_enabled and self._base_box_drop_pose_config_valid():
            layout = self._compute_base_box_layout()
            self.get_logger().info(
                f'Automatic base box ready: centre={self.base_box_center_xyz}, '
                f'dimensions={self.base_box_dimensions_xyz}m, release-volume-centre='
                f'({layout.release_volume_center[0]:.3f},{layout.release_volume_center[1]:.3f},'
                f'{layout.release_volume_center[2]:.3f}), release-volume-size='
                f'({layout.release_volume_dimensions[0]:.3f},{layout.release_volume_dimensions[1]:.3f},'
                f'{layout.release_volume_dimensions[2]:.3f})m, wrist orientation is searched automatically.')

    def _joint_states_cb(self, msg: JointState) -> None:
        update_sec = self._now_sec()
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_positions[name] = float(pos)
            self.current_joint_update_sec[name] = update_sec

    def color_cb(self, msg: Image) -> None:
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_color_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            self.latest_color_frame = str(msg.header.frame_id)
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
        self._grasp_orientation_search = None
        self._grasp_orientation_search_token += 1
        self._grasp_pregrasp_joint_targets = {}
        self._base_box_drop_candidates = []
        self._base_box_drop_candidate_index = -1
        self._base_box_drop_round = 0
        self._base_box_drop_position_only_active = False
        self._base_box_tip_fallback_active = False
        self._base_box_drop_start_collision_retry_index = -1
        self._base_box_drop_ik_skipped = 0
        self._base_box_center_plans_attempted = 0
        self._base_box_tip_plans_attempted = 0
        self._base_box_ik_request_token += 1
        self._base_box_ik_screen_exhausted = False
        self._base_box_ik_reject_codes = {}
        self._active_base_box_drop_pose = None
        self._computed_base_box_probe_axis_yaw_rad = None
        self._close_step_targets = []
        self._close_step_index = 0
        self.sequence_locked_target_point_base = None
        self.sequence_locked_object_long_axis_base = None
        self._vertical_grasp_body_shift_base = None
        self._last_final_descent_waypoints = []
        self._last_final_descent_start = None
        self._accepted_descent_shortfall_m = 0.0
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
        # Do not let the controller's short default goal_time abort an
        # intentionally over-closed grasp before our stalled-contact/feedback
        # logic classifies it. The node's bounded confirmation watchdog remains
        # authoritative and explicitly cancels the action on completion or
        # timeout, so this does not create an unbounded command.
        controller_grace_sec = max(30.0, self.gripper_confirm_timeout_sec + 5.0)
        controller_grace_ns = int(controller_grace_sec * 1e9)
        goal.goal_time_tolerance = Duration(
            sec=controller_grace_ns // 1_000_000_000,
            nanosec=controller_grace_ns % 1_000_000_000,
        )
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

        goal_tolerance_code = int(getattr(
            FollowJointTrajectory.Result,
            'GOAL_TOLERANCE_VIOLATED',
            -5,
        ))
        if expected_stage == 'close_gripper' and error_code == goal_tolerance_code:
            # Compatibility with a controller that was already running with
            # the old 5 s deadline when this node was restarted. A deliberately
            # over-closed grasp is expected to stop short, so controller timing
            # is not a failure signal here. Keep evaluating measured stall/gap
            # evidence until the node-owned watchdog completes or cancels it.
            self._gripper_action_failed_reason = None
            self.get_logger().warning(
                'Ignoring the gripper controller goal deadline during final close; '
                'continuing with measured stalled-contact validation.')
            return

        error_string = str(getattr(result, 'error_string', '')).strip()
        self._gripper_action_failed_reason = (
            f'action finished with status={result_wrap.status}, '
            f'error_code={error_code}, error="{error_string}"'
        )
        if expected_stage == 'close_gripper':
            # A goal-tolerance abort is an expected result when the rigid probe
            # blocks the deliberately over-closed command.  The gripper wait
            # tick classifies it using fresh position/travel/gap evidence and
            # emits an error only if that bounded contact check also fails.
            self.get_logger().warning(
                f'Final-close action stopped short; validating rigid-probe contact: '
                f'target={expected_target:.5f}, {self._gripper_action_failed_reason}.')
        else:
            self.get_logger().error(f'Gripper controller action failed: '
                f'stage={expected_stage}, target={expected_target:.5f}, '
                f'{self._gripper_action_failed_reason}.')

    def _complete_final_close_contact(
        self,
        target: float,
        current: float,
        elapsed: float,
        evidence: str,
        contact_trusted: bool = True,
    ) -> None:
        """Finish the final close as probe contact.

        With ``contact_trusted=False`` the close is completed (jaws stay
        closed) but gripper_contact_detected remains False, so the vision
        lift check must positively confirm the pickup and the retry
        machinery stays fully armed on a miss.
        """
        actual_gap = fourbar.gap_from_q(float(current))
        self.gripper_contact_detected = bool(contact_trusted)
        if self._gripper_wait_timer is not None:
            self._gripper_wait_timer.cancel()
            self._gripper_wait_timer = None
        cb = self._gripper_wait_cb
        self._gripper_wait_cb = None
        self.last_gripper_actual = float(current)
        self.last_gripper_target = float(target)
        self._cancel_active_gripper_goal()
        self._reassert_close_setpoint(float(target), float(current))
        self.get_logger().warning(
            f'Final close {"confirmed" if contact_trusted else "completed WITHOUT contact trust"} '
            f'by {evidence}: '
            f'target={target:.5f}, actual={current:.5f}, '
            f'jaw_gap={actual_gap*1000.0:.1f}mm, elapsed={elapsed:.2f}s. '
            'Continuing with lift verification while keeping the gripper closed.'
        )
        if cb is not None:
            cb()

    def _reassert_close_setpoint(self, target: float, contact_q: float) -> None:
        """Re-command a bounded close setpoint so the jaws stay loaded.

        Cancelling the gripper action leaves JTC holding the measured contact
        position, so the over-close the grasp relies on is released the moment
        contact is confirmed. Publishing on the controller's topic interface
        restores a standing closing setpoint instead.

        The setpoint is ``contact_q`` plus a bounded overshoot, NOT the full
        close: commanding full close presses without limit, which on the probe's
        tapered shaft wedges the jaws progressively shut and ejects it. Bounded
        overshoot keeps preload against a rigid stop while capping how far the
        jaws can travel if the grip is not in fact rigid. Fire-and-forget on
        purpose — no wait timer and no callback, so this cannot disturb the state
        machine, and the release open command later supersedes it.
        """
        if not self.reassert_close_after_contact:
            return
        # Closing is +q, so the bounded setpoint is the tighter of "a little
        # past contact" and the commanded target; it never opens the jaws.
        bounded = min(float(target), float(contact_q) + self.reassert_close_overshoot_rad)
        if bounded <= float(contact_q):
            return
        target = bounded
        traj = JointTrajectory()
        traj.joint_names = [self.gripper_joint_name]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        point.velocities = [0.0]
        point.time_from_start = Duration(sec=0, nanosec=200_000_000)
        traj.points = [point]
        self._gripper_traj_pub.publish(traj)
        self.get_logger().info(
            f'Re-asserted a bounded close setpoint at q={target:.5f} '
            f'(contact stopped at {contact_q:.5f}, overshoot capped at '
            f'{self.reassert_close_overshoot_rad:.3f} rad) so the jaws stay '
            'loaded without walking shut; cancelling the action alone would '
            'have left the controller holding the measured contact position.'
        )

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
            self._complete_final_close_contact(
                float(target), float(current), elapsed, 'fresh stalled-contact feedback'
            )
            return

        # Uncertain stall: the jaws stopped on SOMETHING (fresh stationary
        # feedback, large closing travel, well short of the over-closed
        # target, gap clearly above empty-closed) but the calibrated gap is
        # outside the plausible probe window — physics penetration or an
        # off-square grasp reads tighter than the true probe width. Hard-
        # locking here previously stranded a physically held probe over a
        # sub-millimetre gap-window miss. Complete the close without contact
        # trust instead and let the vision lift check settle it.
        if (
            self.gripper_contact_uncertain_stall_enabled
            and self.sequence_stage == 'close_gripper'
            and minimum_time_elapsed
            and feedback_fresh
            and current is not None
            and self._gripper_wait_start_position is not None
            and contact_stalled
            and (float(current) - self._gripper_wait_start_position)
                >= self.gripper_contact_min_closing_travel_rad
            and float(current) < target - self.gripper_goal_tolerance
            and fourbar.gap_from_q(float(current)) >= self.gripper_contact_uncertain_min_gap_m
        ):
            self._complete_final_close_contact(
                float(target), float(current), elapsed,
                'stalled jaws with a gap outside the plausible probe window',
                contact_trusted=False,
            )
            return

        if self._gripper_command_used_action and self._gripper_action_failed_reason is not None:
            # The controller aborting the deliberately over-closed final command
            # with a goal-tolerance error is itself the designed contact signal
            # (see fourbar.plausible_probe_contact): physics jitter against the
            # rigid probe can keep the joint moving more than the stall epsilon,
            # so re-check the measured jaw geometry before declaring a miss.
            if (
                self.sequence_stage == 'close_gripper'
                and feedback_fresh
                and current is not None
                and self._gripper_wait_start_position is not None
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
            ):
                self._complete_final_close_contact(
                    float(target), float(current), elapsed,
                    'controller goal-tolerance abort with plausible jaw geometry',
                )
                return
            if current is not None and self._gripper_wait_start_position is not None:
                measured_txt = (
                    f'measured q={float(current):.5f}, '
                    f'jaw_gap={fourbar.gap_from_q(float(current))*1000.0:.1f}mm, '
                    f'closing_travel={float(current) - self._gripper_wait_start_position:.2f}rad, '
                    f'feedback_fresh={feedback_fresh}'
                )
            else:
                measured_txt = f'no usable feedback (current={current}, fresh={feedback_fresh})'
            reason = (
                'Gripper controller did not execute the command and measured '
                'feedback was not consistent with probe contact: '
                f'{self._gripper_action_failed_reason}. target={target:.5f}, {measured_txt}'
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
                # An EMPTY close reaches this path looking like a success: the
                # controller happily drives the deliberately over-closed command
                # home because nothing stopped it. The whole premise of
                # over-closing is that a real probe PREVENTS reaching the
                # target, so arriving at a gap far below the probe's width is
                # positive evidence the jaws shut on air. Say so here -- the
                # alternative is the sequence going on to announce "Object is
                # held with gripper closed" and locking itself while empty.
                self._flag_empty_final_close(float(current))
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

        # Capture the action state BEFORE cancelling: cancellation resets the
        # accepted/succeeded flags, which previously made this failure log
        # claim action_accepted=False for a goal the controller had accepted.
        feedback_detail = (
            f'actual={current}, fresh={feedback_fresh}, position_reached={position_reached}, '
            f'action_accepted={self._gripper_action_accepted}, '
            f'action_succeeded={self._gripper_action_succeeded}, '
            f'elapsed={elapsed:.2f}s, required_time={minimum_completion_sec:.2f}s'
        )

        # The per-goal controller deadline deliberately exceeds this watchdog.
        # Cancel here so a stopped gripper is handled by our measured-feedback
        # safety decision and never later reports GOAL_TOLERANCE_VIOLATED.
        self._cancel_active_gripper_goal()
        if self._gripper_wait_timer is not None:
            self._gripper_wait_timer.cancel()
            self._gripper_wait_timer = None
        cb = self._gripper_wait_cb
        self._gripper_wait_cb = None
        self.last_gripper_actual = float(current) if current is not None else None
        self.last_gripper_target = float(target)

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
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            # RViz DepthCloud uses approximate-time synchronization between the
            # depth and color images. Preserve the camera timestamp/frame rather
            # than publishing the YOLO image with the default zero header.
            if self.latest_color_stamp is not None:
                msg.header.stamp = self.latest_color_stamp.to_msg()
            msg.header.frame_id = self.latest_color_frame
            self.det_vis_pub.publish(msg)
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
        closed, not a wide gap.  The 30 mm probe needs q≈-0.085 rad.
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
        self._last_object_orientation_eigenratio = 0.0
        pts_all = self._mask_cloud_camera(mask_bool, depth_image)
        if pts_all is None:
            return None

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
        self._last_object_orientation_eigenratio = ratio
        if ratio < self.object_orientation_min_eigenratio:
            self.get_logger().info(f'Object orientation skipped: eigenratio={ratio:.1f} '
                f'< min={self.object_orientation_min_eigenratio:.1f} '
                f'(object too round to determine orientation reliably)', throttle_duration_sec=1.0)
            return None

        # Ensure right-handed rotation matrix
        R = eigenvectors.copy()
        if np.linalg.det(R) < 0:
            R[:, 2] = -R[:, 2]

        # Shape-aware refinement: fit the KNOWN probe box to the cloud so the
        # reported centre is the FULL-model centre (occluded tip included) and
        # the long axis is the box axis, not just the visible-blob PCA. Falls
        # back to the raw PCA pose when the fit is unreliable.
        shape_fit_ok = False
        if self.shape_aware_pose_enabled:
            refined = self._refine_pose_with_box_fit(pts, centroid, R)
            if refined is not None:
                centroid, R, shape_fit_ok = refined

        # Point the long axis from the probe's fat body toward its tapered tip,
        # matching the STL's own +Z. eigh's eigenvector signs are arbitrary, so
        # without this the long axis is a coin flip that the attach step bakes
        # into the collision mesh as a 180° end-for-end flip.
        self._object_axis_tip_resolved = False
        fat_sign = probe_alignment.long_axis_fat_end_sign(
            pts, centroid, R[:, 0],
            min_reach_m=0.40 * (0.5 * float(self._probe_dims()[2])),
        )
        if fat_sign == 0 and not self.busy and self._probe_fat_dir_cam is not None and (
            self._now_sec() - self._probe_fat_dir_sec <= self.probe_fat_dir_latch_sec
        ):
            # Same latch as the box-fit anchoring: an undecided frame must not
            # hand the axis direction back to LAPACK's arbitrary eigenvector
            # sign, or the reported pose flips 180 deg frame to frame.
            fat_dot = float(np.dot(R[:, 0], self._probe_fat_dir_cam))
            if abs(fat_dot) > 1e-9:
                fat_sign = -1 if fat_dot > 0.0 else 1
        if fat_sign != 0:
            self._object_axis_tip_resolved = True
            if fat_sign != self._probe_stl_end_sign_or_default():
                R[:, 0] = -R[:, 0]
                R[:, 1] = -R[:, 1]   # keep the frame right-handed

        self.get_logger().info(f'Object 3D orientation: eigenratio={ratio:.1f}  '
            f'long_axis_cam=[{R[0,0]:.2f},{R[1,0]:.2f},{R[2,0]:.2f}]  '
            f'tip_resolved={self._object_axis_tip_resolved}  '
            f'shape_fit={"on" if shape_fit_ok else "off"}', throttle_duration_sec=1.0)
        self._last_object_shape_fit_ok = bool(shape_fit_ok)
        return centroid, R

    def _mask_cloud_camera(
        self,
        mask_bool: np.ndarray,
        depth_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Back-project a segmentation mask to an (N, 3) camera-frame cloud.

        Returns None if intrinsics are missing or fewer than 20 valid points
        survive the depth range filter.
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
        return np.column_stack([X, Y, Z])      # N × 3

    def _refine_pose_with_box_fit(
        self,
        pts: np.ndarray,
        centroid_visible: np.ndarray,
        R_pca: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
        """Fit the known probe box to the masked cloud (camera frame).

        Returns ``(centre_cam, R_cam, True)`` with the FULL-model box centre and
        a right-handed frame whose col 0 is the fitted long axis, col 2 the PCA
        normal re-orthogonalised against it. Returns None when the trimmed ICP
        cannot lock on (too few inliers or RMS above shape_fit_max_rms_m), so the
        caller keeps the raw PCA pose.

        The visible points sit on the near surface, so the true box centre is
        about half a cross-section deeper along the viewing ray -- the same
        correction _fit_probe_box_from_cloud uses for re-acquisition.
        """
        view_dir = normalize(np.asarray(centroid_visible, dtype=np.float64))
        if float(np.linalg.norm(view_dir)) < 1e-9:
            view_dir = np.array([0.0, 0.0, 1.0])
        dims = self._probe_dims()
        centre_init = centroid_visible + view_dir * (0.5 * float(dims[0]))
        # fit_box_to_points wants rotation columns in the half-extents order
        # (X, Y, Z=long); R_pca is (long, short, normal), so reorder to
        # (short, normal, long).
        R_init = np.column_stack([R_pca[:, 1], R_pca[:, 2], R_pca[:, 0]])
        half = 0.5 * dims
        fit = probe_alignment.fit_box_to_points(
            pts, half, R_init, centre_init,
            iterations=self.shape_fit_icp_iterations,
            trim_fraction=self.shape_fit_trim_fraction,
            outlier_residual_m=self.shape_fit_max_rms_m * 2.0,
        )
        if fit is None:
            return None
        if fit.rms_m > self.shape_fit_max_rms_m or fit.inlier_count < self.shape_fit_min_inliers:
            self.get_logger().info(
                f'Shape fit skipped: rms={fit.rms_m*1000:.1f}mm '
                f'(max {self.shape_fit_max_rms_m*1000:.0f}), inliers={fit.inlier_count} '
                f'(min {self.shape_fit_min_inliers}); using raw PCA pose.',
                throttle_duration_sec=2.0)
            return None

        long_fit = normalize(fit.rotation[:, 2])
        # Keep the long axis on the same side as PCA; the taper step below fixes
        # the true fat->tip direction afterwards.
        if float(np.dot(long_fit, R_pca[:, 0])) < 0.0:
            long_fit = -long_fit
        normal_pca = R_pca[:, 2]
        short = np.cross(normal_pca, long_fit)
        if float(np.linalg.norm(short)) < 1e-6:
            return None
        short = normalize(short)
        normal = normalize(np.cross(long_fit, short))
        R_cam = np.column_stack([long_fit, short, normal])
        if np.linalg.det(R_cam) < 0.0:
            R_cam[:, 1] = -R_cam[:, 1]

        centre_fit = np.asarray(fit.centre, dtype=np.float64)
        # Predict the occluded length. A flat box fit is under-constrained ALONG
        # the shaft when one end is buried -- it slides the centre toward the
        # visible body. When the taper decisively resolves the fat end and the
        # cloud is clearly shorter than the probe, re-anchor the along-axis centre
        # to (visible fat end + half the known length), so the reported centre is
        # the true full-model centre with the buried tip predicted. The lateral
        # (cross-section) centre from the fit is kept -- that part is well
        # constrained by the visible surface. Skipped when the taper cannot pick a
        # side, so a heavily occluded cloud is never anchored on a guess.
        long_len = float(self._probe_dims()[2])
        fat_sign = probe_alignment.long_axis_fat_end_sign(
            pts, centre_fit, long_fit, min_reach_m=0.40 * (0.5 * long_len))
        # Anchoring and not-anchoring are two DIFFERENT centre conventions,
        # separated by up to half the probe. The taper test flickers between
        # decisive and undecided on marginal clouds, so gating purely on this
        # frame's answer makes the reported centre alternate between the two --
        # which blows probe_track_max_jump_m, gets every other fit rejected, and
        # starves the track until it re-acquires, over and over. Latch the last
        # decisive answer and keep using it while it is fresh, so the convention
        # stays put even when a single frame cannot decide.
        axis_ft = None
        if fat_sign != 0:
            axis_ft = -long_fit if fat_sign > 0 else long_fit   # oriented fat->tip
            if not self.busy:
                self._probe_fat_dir_cam = axis_ft.copy()
                self._probe_fat_dir_sec = self._now_sec()
        elif (
            not self.busy
            and
            self._probe_fat_dir_cam is not None
            and self._now_sec() - self._probe_fat_dir_sec <= self.probe_fat_dir_latch_sec
        ):
            # A camera-frame latch is valid only during idle acquisition.
            # Once a sequence is busy the wrist may move between frames, so a
            # remembered camera-frame vector must never orient a later cloud.
            axis_ft = (
                long_fit
                if float(np.dot(long_fit, self._probe_fat_dir_cam)) > 0.0
                else -long_fit
            )
            self.get_logger().info(
                'Taper undecided this frame; reusing the last resolved fat-end '
                'direction so the centre convention does not flip.',
                throttle_duration_sec=2.0,
            )
        if axis_ft is not None:
            anchored = probe_alignment.full_model_centre_along_axis(
                pts, centre_fit, axis_ft, long_len)
            if anchored is not None:
                self.get_logger().info(
                    f'Shape fit predicted the occluded tip: anchored the centre '
                    f'{float(np.dot(anchored - centre_fit, axis_ft))*1000:+.0f}mm '
                    f'along the shaft from the visible fat end.',
                    throttle_duration_sec=2.0)
                centre_fit = anchored
        return centre_fit, R_cam, True

    def _clear_detected_object_pose(self, clear_fat_dir: bool = False) -> None:
        self.detected_object_pose = None
        self._last_detected_object_rotation_base = None
        self._object_axis_tip_resolved = False
        self._last_object_shape_fit_ok = False
        if clear_fat_dir:
            self._probe_fat_dir_cam = None
            self._probe_fat_dir_sec = -1e9

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

        tfm = self._lookup_transform(
            depth_frame, self.planning_frame, stamp
        )
        if tfm is None:
            return None
        R_tf = quat_to_matrix(tfm.transform.rotation)
        translation = np.array([
            float(tfm.transform.translation.x),
            float(tfm.transform.translation.y),
            float(tfm.transform.translation.z),
        ], dtype=np.float64)
        # Apply the same calibrated camera-axis correction used by the raw
        # mask target. Previously the shape-aware pose silently bypassed this
        # offset and then replaced the calibrated target downstream.
        raw_centroid_base = (
            R_tf @ np.asarray(centroid_cam, dtype=np.float64).reshape(3,)
            + translation
        )
        self._last_shape_raw_world = raw_centroid_base.copy()
        self._last_shape_rotation_world_camera = R_tf.copy()
        centroid_base = (
            raw_centroid_base + R_tf @ self.grasp_target_offset_in_camera
        )
        long_axis = normalize(R_tf @ R_obj_cam[:, 0].reshape(3,))
        normal_axis = normalize(R_tf @ R_obj_cam[:, 2].reshape(3,))

        prev_R = self._last_detected_object_rotation_base
        if prev_R is not None:
            # Only inherit the previous frame's long-axis direction when this
            # frame could not resolve the tip itself; a decisive taper reading
            # must win, otherwise the first frame's arbitrary PCA sign is
            # latched for the whole run and never self-corrects.
            if (
                not self._object_axis_tip_resolved
                and float(np.dot(long_axis, prev_R[:, 0])) < 0.0
            ):
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
        confidence: Optional[float] = None,
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
        centre_base = self._pose_xyz(pose_msg)

        # Persistent identity: fuse this fit into the tracked probe (or start a
        # new track). The published/consumed pose is the SMOOTHED track pose, so
        # it stays locked on one physical probe for the whole process instead of
        # hopping between per-frame detections.
        if self.probe_track_enabled:
            self._probe_track_pose_held = False
            track = self._update_probe_track(
                centre_base,
                R_obj_base,
                self.last_detection_conf if confidence is None else confidence,
                self._now_sec(),
                pose_reliable=self._last_object_shape_fit_ok,
                axis_reliable=(
                    self._last_object_orientation_eigenratio
                    >= self.probe_track_axis_min_eigenratio
                ),
            )
            if track is None:
                self._clear_detected_object_pose()
                return
            centre_base = track.centre_base.copy()
            R_obj_base = track.R_base.copy()
            pose_msg = self.make_pose(centre_base, matrix_to_quat(R_obj_base))

        self.detected_object_pose = pose_msg
        self._last_detected_object_rotation_base = R_obj_base

        # Auto-calibration must observe one rigid geometric point. Raw
        # mask/bbox centres slide with occlusion and viewpoint; an accepted
        # full-model shape centre does not. Reset the regression window when
        # track identity changes so two physical probes are never mixed.
        if (
            self._last_object_shape_fit_ok
            and self._last_shape_raw_world is not None
            and self._last_shape_rotation_world_camera is not None
        ):
            track_id = (
                int(self._probe_track.track_id)
                if self._probe_track is not None else None
            )
            if track_id != self._camera_calibration_track_id:
                self._camera_calibration_raw_world.clear()
                self._camera_calibration_rotations.clear()
                self._pending_camera_offset_estimate = None
                self._camera_calibration_track_id = track_id
            self._record_camera_offset_calibration_sample(
                self._last_shape_raw_world,
                self._last_shape_rotation_world_camera,
            )

        if self.publish_object_pose_enabled:
            self.object_pose_pub.publish(pose_msg)
        if self.probe_track_enabled and self._probe_track is not None:
            self.object_track_id_pub.publish(Int32(data=int(self._probe_track.track_id)))

        # Give MoveIt the probe's actual shape at this pose.
        self._publish_world_probe_object(centre_base, R_obj_base)

        p = pose_msg.pose.position
        track_tag = (f'track#{self._probe_track.track_id} '
                     if self.probe_track_enabled and self._probe_track is not None else '')
        self.get_logger().info(f'6D object pose tracked: {track_tag}'
            f'x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}', throttle_duration_sec=1.0)

    def _update_probe_track(
        self,
        centre_base: np.ndarray,
        R_obj_base: np.ndarray,
        conf: float,
        now_sec: float,
        pose_reliable: bool = True,
        axis_reliable: bool = True,
    ) -> Optional['ProbeTrack']:
        """Associate a fresh shape-fit pose with the persistent probe track.

        Returns the track whose (smoothed) pose should be used this frame:
          * no track yet, or the current one has gone unseen past
            probe_track_timeout_sec -> start a fresh id at this pose;
          * fit matches the track (centre within probe_track_max_jump_m AND long
            axis within probe_track_max_axis_deg) -> EMA-blend it in, keep id;
          * fit does NOT match but the track is still fresh -> reject the fit and
            hold the existing track pose, so a neighbouring object or a mask
            flicker cannot steal the identity.
        """
        centre_base = np.asarray(centre_base, dtype=np.float64).reshape(3,)
        long_new = normalize(R_obj_base[:, 0].reshape(3,))
        track = self._probe_track

        if track is None or (now_sec - track.last_update_sec) > self.probe_track_timeout_sec:
            if track is not None:
                self.get_logger().info(
                    f'Probe track#{track.track_id} dropped (unseen '
                    f'{now_sec - track.last_update_sec:.1f}s); acquiring a new identity.')
            track = ProbeTrack(
                track_id=self._next_probe_track_id,
                centre_base=centre_base.copy(),
                R_base=R_obj_base.copy(),
                dims=self._probe_dims().copy(),
                created_sec=now_sec,
                last_update_sec=now_sec,
                last_pose_update_sec=now_sec,
                hit_count=1,
                miss_count=0,
                confidence=float(conf),
            )
            self._next_probe_track_id += 1
            self._probe_track = track
            self.get_logger().info(
                f'Probe track#{track.track_id} acquired at '
                f'({centre_base[0]:.3f},{centre_base[1]:.3f},{centre_base[2]:.3f}).')
            return track

        jump = float(np.linalg.norm(centre_base - track.centre_base))
        axis_deg = probe_alignment.axis_angle_deg(long_new, track.long_axis())
        axis_mismatch = axis_reliable and axis_deg > self.probe_track_max_axis_deg
        if jump > self.probe_track_max_jump_m or axis_mismatch:
            track.miss_count += 1
            self._probe_track_association_rejected = True
            self.get_logger().warning(
                f'Probe track#{track.track_id} rejected a non-matching fit '
                f'(jump={jump*1000:.0f}mm/{self.probe_track_max_jump_m*1000:.0f} max, '
                f'axis={axis_deg:.0f}°/{self.probe_track_max_axis_deg:.0f} max'
                f'{"; axis trusted" if axis_reliable else "; axis marginal/ignored"}); '
                f'holding the locked pose.', throttle_duration_sec=1.0)
            return None

        # A marginal PCA frame still proves that the same probe is visible, but
        # its raw visible centroid and arbitrary long axis must not pull a good
        # full-model fit around. Refresh identity while holding the last reliable
        # 6D pose; a later reliable shape fit will resume the EMA.
        if not pose_reliable:
            track.last_update_sec = now_sec
            track.hit_count += 1
            track.miss_count = 0
            track.confidence = float(conf)
            self._probe_track_pose_held = True
            return track

        # Match: EMA-blend centre and axis, keep the id. Flip the incoming long
        # axis onto the track's directed side first so a per-frame sign flip
        # (arbitrary when the taper is not decisive) never rotates the estimate.
        if float(np.dot(long_new, track.long_axis())) < 0.0:
            R_obj_base = R_obj_base.copy()
            R_obj_base[:, 0] = -R_obj_base[:, 0]
            R_obj_base[:, 1] = -R_obj_base[:, 1]
        a_p = self.probe_track_position_smoothing
        a_a = self.probe_track_axis_smoothing
        track.centre_base = (1.0 - a_p) * track.centre_base + a_p * centre_base
        track.R_base = self._blend_rotation(track.R_base, R_obj_base, a_a)
        track.last_update_sec = now_sec
        track.last_pose_update_sec = now_sec
        track.hit_count += 1
        track.miss_count = 0
        track.confidence = float(conf)
        return track

    @staticmethod
    def _blend_rotation(R_old: np.ndarray, R_new: np.ndarray, alpha: float) -> np.ndarray:
        """Cheap orthonormal blend of two rotations (no SciPy dependency).

        Linearly interpolates the columns toward R_new by ``alpha`` and
        re-orthonormalises with Gram-Schmidt. Good enough for the small
        frame-to-frame deltas of a tracked probe and avoids a slerp import.
        """
        if alpha <= 0.0:
            return R_old
        if alpha >= 1.0:
            return R_new.copy()
        blended = (1.0 - alpha) * R_old + alpha * R_new
        x = normalize(blended[:, 0])
        y = blended[:, 1] - x * float(np.dot(x, blended[:, 1]))
        if float(np.linalg.norm(y)) < 1e-9:
            return R_new.copy()
        y = normalize(y)
        z = np.cross(x, y)
        R = np.column_stack([x, y, z])
        if np.linalg.det(R) < 0.0:
            R[:, 1] = -R[:, 1]
        return R

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
                if self._arm_motion_in_flight():
                    self.get_logger().warning(
                        f'TF at image stamp unavailable for {source_frame} -> '
                        f'{target_frame} while the arm is moving; dropping this '
                        f'detection instead of projecting it with the latest TF.',
                        throttle_duration_sec=2.0,
                    )
                    return None
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
        """Read an STL file and return a shape_msgs/Mesh (vertices deduplicated).

        Handles both binary and ASCII STL. Detecting the format matters:
        parsing an ASCII file with the binary reader takes the character data
        as a triangle count (892 million for probe.stl) and fails, which
        silently demotes the attached probe to the fallback cylinder at
        whatever hard-coded size the node assumed.
        """
        import struct
        try:
            with open(path, 'rb') as f:
                raw = f.read()

            verts: List[Tuple[float, float, float]] = []
            tris: List[Tuple[int, int, int]] = []
            vi: dict = {}

            def add_triangle(points: List[Tuple[float, float, float]]) -> None:
                t_idx = []
                for xyz in points:
                    key = (round(xyz[0], 7), round(xyz[1], 7), round(xyz[2], 7))
                    if key not in vi:
                        vi[key] = len(verts)
                        verts.append(key)
                    t_idx.append(vi[key])
                tris.append((t_idx[0], t_idx[1], t_idx[2]))

            # A binary STL may also begin with "solid", so confirm the body
            # really is ASCII facet text before choosing the parser.
            head = raw[:512].lstrip().lower()
            is_ascii = head.startswith(b'solid') and b'facet' in raw[:2048].lower()

            if is_ascii:
                pending: List[Tuple[float, float, float]] = []
                for line in raw.decode('ascii', 'ignore').splitlines():
                    parts = line.split()
                    if not parts or parts[0] != 'vertex' or len(parts) < 4:
                        continue
                    pending.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    if len(pending) == 3:
                        add_triangle(pending)
                        pending = []
            else:
                n_tris = struct.unpack('<I', raw[80:84])[0]
                expected = 84 + n_tris * 50
                if len(raw) < expected:
                    raise ValueError(
                        f'binary STL claims {n_tris} triangles but the file holds '
                        f'{len(raw)} bytes (needs {expected})'
                    )
                for i in range(n_tris):
                    offset = 84 + i * 50
                    values = struct.unpack('<12f', raw[offset:offset + 48])
                    add_triangle([values[3:6], values[6:9], values[9:12]])

            if not tris:
                raise ValueError('no triangles found')
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

        STL axes, with extents measured from the mesh by _probe_dims() rather
        than hard-coded — probe.stl has been re-modelled before (45x45x300 mm,
        now 30x30x200 mm) and any constant quoted here goes stale silently:
          X: cross-section width
          Y: cross-section height
          Z: long axis / probe length
        The mesh origin sits at the min corner, not the centroid, which is why
        _publish_probe_attachment subtracts half the extents to place it.

        Flat-on-floor coordinate mapping:
          STL Z (long)   → world [cos(probe_yaw), sin(probe_yaw), 0]
          STL Y (height) → world [0, 0, 1]  (pointing up)
          STL X (width)  → world [-sin(yaw), cos(yaw), 0]
        """
        contact_y = float(getattr(self, 'fourbar_contact_y_offset_m', 0.001))
        contact_z = float(getattr(self, 'fourbar_contact_z_closed_m', 0.218))

        # ── Probe yaw: use raw PCA long axis to avoid gripper-symmetry 180° flip ──
        # _last_detected_object_rotation_base[:, 0] is stabilised across frames by
        # _update_detected_object_pose_from_camera, so its sign is consistent within
        # a run.  We use it directly instead of backing out gripper_yaw - offset_rad,
        # which was the source of the consistent 180° misalignment.
        # The probe is about to exist as an attached body. Leaving the world
        # copy behind would put the attached mesh permanently in collision with
        # a stationary ghost of itself, which reads as an invalid start state.
        self._remove_world_probe_object('the probe is now attached to the gripper')
        # That removal drops the ACM entries with the object, and the pre-grasp
        # publisher (the only other caller) stops running once attached -- so
        # re-assert here so intentional gripper contact remains allowed.
        self._ensure_world_probe_collisions_allowed()

        correction_rad = math.radians(float(getattr(self, 'stl_yaw_correction_deg', 0.0)))
        # Prefer the mask pose frozen at final grasp commit — the live one is
        # stale or cleared by the time the gripper has closed.
        mask_R = None
        mask_pose = None
        mask_source = 'none'
        if self.attach_probe_pose_sync_enabled and self._grasp_time_object_R is not None:
            mask_R = self._grasp_time_object_R
            mask_pose = self._grasp_time_object_pose
            mask_source = 'grasp-commit snapshot'
        elif self._last_detected_object_rotation_base is not None:
            mask_R = self._last_detected_object_rotation_base
            mask_pose = self.detected_object_pose
            mask_source = 'live detection'
        # long_world keeps the measured out-of-plane tilt; probe_yaw is its
        # horizontal heading, still reported and used by the yaw-only fallbacks.
        long_world = None
        if mask_R is not None:
            long_ax = normalize(mask_R[:, 0])
            if correction_rad:
                cc, sc = math.cos(correction_rad), math.sin(correction_rad)
                long_ax = np.array([
                    [cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0],
                ]) @ long_ax
            long_world = normalize(long_ax)
            probe_yaw = math.atan2(float(long_world[1]), float(long_world[0]))
        elif self.detected_object_yaw_rad is not None:
            offset_rad = math.radians(float(getattr(self, 'object_yaw_rotation_offset_deg', 90.0)))
            probe_yaw = self.detected_object_yaw_rad - offset_rad + correction_rad
        else:
            probe_yaw = correction_rad   # fallback

        axis_world = (
            long_world if long_world is not None
            else np.array([math.cos(probe_yaw), math.sin(probe_yaw), 0.0])
        )

        # ── R_world_in_link: use stored GRASP orientation, NOT current TF ─────────
        # The mesh_pose is expressed in the link frame (arm_gripper_base_link).
        # R_stl_in_link = R_world_in_link @ R_stl_in_world is CONSTANT once the
        # probe is grasped (rigid body).  We must compute it from the link
        # orientation AT GRASP TIME.  If we used the current TF (at pick_home),
        # the link would have a completely different orientation and R_stl_in_link
        # would be wrong — producing the "mirrored" STL seen in RViz.
        actual_link_pose = None
        if self.attach_probe_pose_sync_enabled:
            actual_link_pose = self._grasp_close_link_pose
            if actual_link_pose is None:
                actual_link_pose = self.get_current_link_pose_in_planning_frame()
                self.get_logger().warning(
                    '[CollisionWorld] Grasp-close link snapshot is unavailable; '
                    'falling back to the current TF for attachment geometry.'
                )
        link_orientation: Optional[Quaternion] = None
        if actual_link_pose is not None:
            # The arm is stationary at the grasp pose, so the measured TF is
            # the true link orientation — the commanded grasp_orientation can
            # differ by the pose/joint confirmation tolerances.
            link_orientation = actual_link_pose.orientation
            R_link_in_world = quat_to_matrix(link_orientation)
            R_world_in_link = R_link_in_world.T
            if self.grasp_orientation is not None:
                ori_delta_deg = math.degrees(
                    quaternion_distance_rad(link_orientation, self.grasp_orientation)
                )
                if ori_delta_deg > 1.0:
                    self.get_logger().info(f'[CollisionWorld] Measured link orientation differs from the '
                        f'commanded grasp orientation by {ori_delta_deg:.1f} deg; using the measured TF.')
        elif self.grasp_orientation is not None:
            # grasp_orientation is arm_gripper_base_link in base_link = R_link_in_world
            link_orientation = self.grasp_orientation
            R_link_in_world = quat_to_matrix(self.grasp_orientation)
            R_world_in_link = R_link_in_world.T
        else:
            R_link_in_world = np.eye(3)
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

        # A partly buried probe often exposes too little of its tapered end for
        # the mask-width test to resolve the PCA sign. That sign is otherwise
        # arbitrary, and treating it as STL +Z reverses the asymmetric mesh
        # end-for-end in RViz. The grasp itself supplies a stronger physical
        # constraint: this node deliberately closes on the probe's wide body,
        # so the STL fat end must be the end nearest the four-bar contact.
        #
        # Apply this even when the earlier taper test claimed a direction: at
        # close time the measured contact is direct task geometry, while the
        # taper estimate can be contaminated by the socket, fingers or a
        # point-blank partial mask. Abstain when centre/contact separation along
        # the shaft is too small to distinguish the ends.
        contact_resolved_tip_direction = False
        if (
            long_world is not None
            and actual_link_pose is not None
            and mask_pose is not None
        ):
            link_pos = np.array([
                float(actual_link_pose.position.x),
                float(actual_link_pose.position.y),
                float(actual_link_pose.position.z),
            ])
            contact_world = link_pos + R_link_in_world @ contact_in_link
            centre_est_world = np.array([
                float(mask_pose.pose.position.x),
                float(mask_pose.pose.position.y),
                float(mask_pose.pose.position.z),
            ])
            min_end_separation = min(0.01, 0.10 * float(self._probe_dims()[2]))
            contact_resolved_tip_direction = bool(
                abs(float(np.dot(centre_est_world - contact_world, long_world)))
                >= min_end_separation
            )
            long_world, end_flipped = probe_alignment.orient_long_axis_from_fat_contact(
                long_world,
                centre_est_world,
                contact_world,
                self._probe_stl_end_sign_or_default(),
                min_axial_separation_m=min_end_separation,
            )
            if end_flipped:
                self.get_logger().warning(
                    '[CollisionWorld] Reversed the detected probe axis at attach time: '
                    'the STL fat end must be nearest the four-bar grasp contact. '
                    'This corrects the 180° end-for-end mesh ambiguity.'
                )
            probe_yaw = math.atan2(float(long_world[1]), float(long_world[0]))
            axis_world = long_world

        # ── Probe centre: where along the probe was it actually grasped? ────────
        # The four-bar close physically snaps the probe to the bucket contact
        # laterally and vertically, but ALONG its long axis the probe stays
        # wherever the gripper came down.  Project the mask-estimated centre
        # onto the probe axis to recover that offset; without it the mesh
        # assumes a perfectly centred grasp.
        probe_centre_in_link = contact_in_link
        along_axis_offset_m = 0.0
        if (
            self.attach_probe_pose_sync_enabled
            and actual_link_pose is not None
            and mask_pose is not None
        ):
            link_pos = np.array([
                float(actual_link_pose.position.x),
                float(actual_link_pose.position.y),
                float(actual_link_pose.position.z),
            ])
            contact_world = link_pos + R_link_in_world @ contact_in_link
            centre_est_world = np.array([
                float(mask_pose.pose.position.x),
                float(mask_pose.pose.position.y),
                float(mask_pose.pose.position.z),
            ])
            along = float(np.dot(centre_est_world - contact_world, axis_world))
            # Never allow a grasp offset past the probe's own half length: the
            # configured clamp is an absolute limit tuned for the longest probe,
            # so it must also shrink with a shorter mesh.
            clamp = min(
                self.attach_probe_max_centre_offset_m,
                0.5 * float(self._probe_dims()[2]),
            )
            along_axis_offset_m = float(np.clip(along, -clamp, clamp))
            if abs(along) > clamp:
                self.get_logger().warning(f'[CollisionWorld] Mask centre is {along*1000:.0f} mm from the '
                    f'contact point along the probe axis; clamping to {along_axis_offset_m*1000:.0f} mm.')
            probe_centre_in_link = contact_in_link + along_axis_offset_m * (R_world_in_link @ axis_world)
        self.get_logger().info(f'[CollisionWorld] Probe centre in link: '
            f'[{probe_centre_in_link[0]:.3f}, {probe_centre_in_link[1]:.3f}, {probe_centre_in_link[2]:.3f}] '
            f'(contact offset [{contact_in_link[0]:.3f}, {contact_in_link[1]:.3f}, {contact_in_link[2]:.3f}], '
            f'probe centre {along_axis_offset_m*1000:+.0f} mm from the grasp contact along its axis, '
            f'mask={mask_source}).')

        # ── R_stl_in_world: STL basis vectors in world frame ────────────────────
        # Prefer the measured 3D axes. Rebuilding the frame from probe_yaw alone
        # projects the long axis onto the world XY plane and pins STL Y to world
        # up, which silently models every probe as perfectly horizontal — a probe
        # tilted in the gripper or lying on uneven ground then gets a collision
        # mesh that disagrees with reality by the whole pitch angle.
        R_stl_in_world = None
        if long_world is not None:
            # STL Z is the long axis; put STL Y as close to world up as the
            # long axis allows, and complete a right-handed frame.
            up = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(long_world, up))) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            stl_x = normalize(np.cross(up, long_world))
            stl_y = normalize(np.cross(long_world, stl_x))
            R_stl_in_world = np.column_stack([stl_x, stl_y, long_world])
            pitch_deg = abs(math.degrees(math.asin(
                float(np.clip(long_world[2], -1.0, 1.0))
            )))
            if pitch_deg > 2.0:
                self.get_logger().info(
                    f'[CollisionWorld] Probe is pitched {pitch_deg:.1f}° out of horizontal; '
                    f'using the measured 3D axes for the mesh instead of a flat model.')
        if R_stl_in_world is None:
            # No 3D axes available (yaw-only fallback): assume flat on the floor.
            cy, sy = math.cos(probe_yaw), math.sin(probe_yaw)
            R_stl_in_world = np.array([
                [-sy, 0.0,  cy],
                [ cy, 0.0,  sy],
                [0.0, 1.0, 0.0],
            ], dtype=float)

        # ── R_stl_in_link = R_world_in_link @ R_stl_in_world ──────────────────
        R_stl_in_link = R_world_in_link @ R_stl_in_world

        # Fixed-grip attach: the grasp is repeatable, so take the held pose from
        # that geometry instead of from perception. Everything above still runs
        # because its logging is how a mis-detection stays visible, but the
        # published pose comes from the grip, not the mask.
        if self.attached_probe_fixed_grip_enabled:
            R_fixed, centre_fixed = self._fixed_held_probe_pose_in_link(contact_in_link)
            axial_delta_m = float(np.dot(
                centre_fixed - probe_centre_in_link, normalize(R_fixed[:, 2])
            ))
            self.get_logger().info(
                '[CollisionWorld] Fixed-grip attach: placing the probe from the '
                f'repeatable coaxial grip ({self._attached_probe_centre_offset_m()*1000:.0f} mm '
                f'from its centre to the modelled jaw contact = '
                f'{self._fat_body_grip_offset_m()*1000:.0f} mm grip offset + '
                f'{self.attached_probe_axial_offset_m*1000:.0f} mm axial correction, '
                'down the tool axis, fat body in '
                f'the jaws) instead of the mask fit. Centre '
                f'[{centre_fixed[0]:.3f}, {centre_fixed[1]:.3f}, {centre_fixed[2]:.3f}] '
                f'vs mask-derived '
                f'[{probe_centre_in_link[0]:.3f}, {probe_centre_in_link[1]:.3f}, '
                f'{probe_centre_in_link[2]:.3f}] '
                f'({axial_delta_m*1000:+.0f} mm along the shaft).'
            )
            R_stl_in_link = R_fixed
            probe_centre_in_link = centre_fixed
            # Keep the recorded world yaw consistent with the pose published, or
            # the base-box drop orients the probe from a different fact than the
            # collision mesh describes.
            axis_world_fixed = R_link_in_world @ normalize(R_fixed[:, 2])
            probe_yaw = math.atan2(
                float(axis_world_fixed[1]), float(axis_world_fixed[0])
            )

        shape_info = self._publish_probe_attachment(R_stl_in_link, probe_centre_in_link)
        # Rigid-body facts recorded at attach time, used by the base-box drop
        # to orient the held probe: its world yaw at grasp, the link
        # orientation (measured TF when available), and its synced geometric
        # centre in the link frame. The held-probe re-alignment keeps these
        # facts updated if the probe shifts in the gripper afterwards.
        self._attached_probe_world_yaw = float(probe_yaw)
        self._attached_probe_grasp_orientation = (
            Quaternion(
                x=float(link_orientation.x),
                y=float(link_orientation.y),
                z=float(link_orientation.z),
                w=float(link_orientation.w),
            )
            if link_orientation is not None else None
        )
        self._attached_probe_axis_in_link = normalize(R_world_in_link @ axis_world)
        self._attached_probe_tip_resolved = bool(
            self._grasp_time_tip_resolved or contact_resolved_tip_direction
        )
        if not self._attached_probe_tip_resolved:
            self.get_logger().warning(
                '[CollisionWorld] Attached probe end direction remains ambiguous; '
                'tip-leading base-box insertion will stay disabled until live '
                'width-profile evidence resolves the tapered end.'
            )
        self._reset_probe_realign_state()
        self.get_logger().info(
            f'[CollisionWorld] Attached probe: {shape_info}, '
            f'probe_yaw={math.degrees(probe_yaw):.1f}°')

    def _publish_probe_attachment(
        self,
        R_stl_in_link: np.ndarray,
        probe_centre_in_link: np.ndarray,
    ) -> str:
        """Publish the probe AttachedCollisionObject at the given box pose.

        ``R_stl_in_link`` columns are the STL axes (X width, Y height, Z long
        axis) in the planning-link frame; ``probe_centre_in_link`` is the box
        geometric centre. Re-publishing with the same object id replaces the
        attached body in the MoveIt planning scene, so this is used both for
        the initial attach and for every held-probe re-alignment update.
        """
        self._ensure_probe_mesh()
        stl_centre = 0.5 * self._probe_dims()
        origin_in_link = probe_centre_in_link - R_stl_in_link @ stl_centre

        inner = CollisionObject()
        inner.header.frame_id = self.planning_link
        inner.header.stamp = self.get_clock().now().to_msg()
        inner.id = 'post_grasp_probe'
        inner.operation = CollisionObject.ADD

        if self._probe_mesh_msg is not None:
            mesh_pose = Pose()
            mesh_pose.position.x = float(origin_in_link[0])
            mesh_pose.position.y = float(origin_in_link[1])
            mesh_pose.position.z = float(origin_in_link[2])
            mesh_pose.orientation = matrix_to_quat(R_stl_in_link)
            inner.meshes = [self._probe_mesh_msg]
            inner.mesh_poses = [mesh_pose]
            shape_info = (
                f'STL mesh {int(self._probe_dims()[2]*1000)}×{int(self._probe_dims()[0]*1000)}'
                f'×{int(self._probe_dims()[1]*1000)} mm, '
                f'origin_in_link=[{origin_in_link[0]:.3f}, '
                f'{origin_in_link[1]:.3f}, {origin_in_link[2]:.3f}]'
            )
        else:
            # Fallback: cylinder along the same long axis at the same centre.
            cyl = SolidPrimitive()
            cyl.type = SolidPrimitive.CYLINDER
            cyl.dimensions = [float(self._probe_dims()[2]), float(self._probe_dims()[0]) / 2.0]
            fb_pose = Pose()
            fb_pose.position.x = float(probe_centre_in_link[0])
            fb_pose.position.y = float(probe_centre_in_link[1])
            fb_pose.position.z = float(probe_centre_in_link[2])
            fb_pose.orientation = matrix_to_quat(R_stl_in_link)
            inner.primitives = [cyl]
            inner.primitive_poses = [fb_pose]
            shape_info = (
                f'fallback cylinder h={self._probe_dims()[2]:.3f} m '
                f'r={self._probe_dims()[0]/2:.3f} m'
            )

        aco = AttachedCollisionObject()
        aco.link_name = self.planning_link
        aco.object = inner
        aco.touch_links = list(self.gripper_probe_contact_links)
        self._attached_object_pub.publish(aco)
        self._post_grasp_probe_attached = True
        self._attached_probe_R_stl_in_link = np.array(R_stl_in_link, dtype=np.float64)
        self._attached_probe_centre_in_link = np.array(probe_centre_in_link, dtype=np.float64)
        return shape_info

    def _probe_stl_end_sign(self) -> int:
        """Which STL long-axis end is the fat one (+1 = +Z, -1 = -Z, 0 = n/a).

        Computed once from the loaded mesh vertices: mean radial distance
        from the central axis for each Z half. probe.stl has the wide body at
        low Z and the tapered tip at high Z, so this returns -1 there.
        """
        if self._probe_stl_fat_end_sign is not None:
            return self._probe_stl_fat_end_sign
        sign = 0
        if self._probe_mesh_msg is not None and len(self._probe_mesh_msg.vertices) >= 10:
            v = np.array([[p.x, p.y, p.z] for p in self._probe_mesh_msg.vertices])
            centre = 0.5 * self._probe_dims()
            radial = np.hypot(v[:, 0] - centre[0], v[:, 1] - centre[1])
            low = v[:, 2] < centre[2]
            if int(low.sum()) >= 5 and int((~low).sum()) >= 5:
                w_low = float(radial[low].mean())
                w_high = float(radial[~low].mean())
                if w_low > 1.15 * w_high:
                    sign = -1
                elif w_high > 1.15 * w_low:
                    sign = 1
        self._probe_stl_fat_end_sign = sign
        return sign

    def _flag_empty_final_close(self, actual_q: float) -> None:
        """Latch the fact that the final close arrived at an impossible gap.

        A physics penetration or an off-square grip can read a little tighter
        than the true width, so this is not by itself a verdict — but the jaws
        travelling all the way past the probe's width is strong evidence they
        shut on air, and it must not be silently outvoted downstream. The latch
        is therefore a VETO rather than a failure: the lift check may still
        pass on positive evidence that the probe is in the jaws, but it may no
        longer pass merely because it stopped seeing the probe on the floor.
        """
        if self.sequence_stage != 'close_gripper':
            return
        gap = float(fourbar.gap_from_q(float(actual_q)))
        if not grasp_verification.empty_close_gap(
            gap, self.minimum_probe_width_m, self.gripper_contact_gap_tolerance_m
        ):
            return
        self._empty_close_detected = True
        self._empty_close_gap_m = gap
        self.get_logger().error(
            f'Final close reached a {gap*1000:.1f} mm jaw gap at q={actual_q:.5f}. The probe is '
            f'{self.minimum_probe_width_m*1000:.0f} mm wide, so nothing was between the jaws -- '
            'the over-closed command was supposed to be stopped by the probe and was not. '
            'The lift check now requires positive evidence that the probe is in the jaws; it '
            'can no longer pass on "the probe is not where it used to be".'
        )

    def _probe_fat_section_span_m(self) -> Optional[Tuple[float, float]]:
        """Extent of the wide cylindrical body, measured from the mesh.

        Returns ``(lo, hi)`` in metres as distances from the probe's full-model
        centre along the long axis TOWARD the fat end, so ``hi`` is nearer the
        flat butt and ``lo`` nearer the tip. probe.stl is a 50 mm Ø30 body, a
        Ø20 shaft and a taper to a point, which gives ``(0.050, 0.100)`` -- the
        body's centre therefore sits 75 mm from the reported centroid, not the
        60 mm the old hand-tuned constant assumed.

        The mesh only carries vertices where the profile changes (a stepped
        solid has none along the straight walls), so this walks the distinct
        long-axis levels rather than fixed-width bins, which would come out
        mostly empty. Per level we take the LARGEST radius: at the shoulder the
        step contributes both the body and the shaft radius, and the body is
        what defines the section. Returns None when the mesh is unavailable or
        shows no clear radius step to grab.
        """
        if self._probe_stl_fat_span_computed:
            return self._probe_stl_fat_span
        self._probe_stl_fat_span_computed = True

        self._ensure_probe_mesh()
        mesh = self._probe_mesh_msg
        sign = self._probe_stl_end_sign()
        if mesh is None or len(mesh.vertices) < 10 or sign == 0:
            return None

        v = np.array([[p.x, p.y, p.z] for p in mesh.vertices], dtype=np.float64)
        centre = 0.5 * (v.max(axis=0) + v.min(axis=0))
        radial = np.hypot(v[:, 0] - centre[0], v[:, 1] - centre[1])
        # Measure outward from the centre toward the fat end, so the walk below
        # runs butt-to-tip whichever way the STL is modelled. Quantise to a
        # micron so co-planar vertices land on exactly one level: STL floats
        # carry rounding noise, and matching against un-quantised values leaves
        # levels with no vertices at all.
        axial = np.round((v[:, 2] - centre[2]) * float(sign), 6)

        levels = np.unique(axial)
        if levels.size < 3:
            return None
        r_max = float(radial.max())
        if r_max <= 1e-4:
            return None
        threshold = 0.90 * r_max

        # Walk down from the fat end while the profile stays at full width; the
        # last qualifying level is the shoulder where the body necks down.
        hi = float(levels[-1])
        lo = hi
        for level in levels[::-1]:
            if float(radial[np.abs(axial - level) < 1e-9].max()) < threshold:
                break
            lo = float(level)
        if (hi - lo) < 1e-3:
            return None

        self._probe_stl_fat_span = (lo, hi)
        return self._probe_stl_fat_span

    def _ensure_probe_mesh(self) -> Optional[Mesh]:
        """Load probe.stl once and measure the probe from it.

        Every consumer goes through here so the collision body, the ICP box
        model and the box-drop layout all describe the same probe as the mesh
        actually on disk.
        """
        if self._probe_mesh_msg is not None or self._probe_mesh_load_attempted:
            return self._probe_mesh_msg
        self._probe_mesh_load_attempted = True
        stl_path = self._find_probe_stl()
        if stl_path is None:
            self.get_logger().warning(
                f'[CollisionWorld] probe.stl not found; falling back to '
                f'{PROBE_STL_DIMS_FALLBACK[2]*1000:.0f} mm probe extents.')
            return None
        mesh = self._load_stl_mesh(stl_path)
        if mesh is None or not mesh.vertices:
            return None
        self._probe_mesh_msg = mesh

        v = np.array([[p.x, p.y, p.z] for p in mesh.vertices], dtype=np.float64)
        measured = v.max(axis=0) - v.min(axis=0)
        if np.all(measured > 1e-4):
            self._probe_stl_dims_measured = measured
        # Always state the size actually in use. A silent mismatch between the
        # mesh and the configured probe is exactly the failure this logging
        # exists to make visible, so "no message" must not mean "loaded fine".
        differs = not np.allclose(measured, PROBE_STL_DIMS_FALLBACK, atol=2e-3)
        self.get_logger().info(
            f'[CollisionWorld] Probe mesh loaded from {os.path.basename(stl_path)}: '
            f'{len(mesh.triangles)} triangles, '
            f'{measured[0]*1000:.0f}x{measured[1]*1000:.0f}x{measured[2]*1000:.0f} mm'
            + (f' (built-in default is '
               f'{PROBE_STL_DIMS_FALLBACK[0]*1000:.0f}x'
               f'{PROBE_STL_DIMS_FALLBACK[1]*1000:.0f}x'
               f'{PROBE_STL_DIMS_FALLBACK[2]*1000:.0f} mm; using the measured size)'
               if differs else '.'))
        return self._probe_mesh_msg

    def _warn_on_probe_width_mismatch(self) -> None:
        """Check the grasp width window against the probe mesh at startup.

        minimum_probe_width_m is a floor applied to the measured width before
        it sizes q_close, so a floor above the real probe silently commands a
        jaw gap wider than the probe and the gripper closes on nothing. That
        failure looks like a bad grasp, not a bad parameter, so it is worth
        catching the moment the mesh disagrees with the configuration.
        """
        dims = self._probe_dims()
        measured = float(max(dims[0], dims[1]))
        if self.minimum_probe_width_m > measured + 1e-4:
            gap = fourbar.gap_from_q(
                fourbar.q_from_gap(
                    self.minimum_probe_width_m + self.object_width_final_clearance_m
                )
            )
            self.get_logger().error(
                f'minimum_probe_width_m={self.minimum_probe_width_m*1000:.0f} mm exceeds the '
                f'{measured*1000:.0f} mm probe measured from the mesh. The width floor will size '
                f'q_close for a {gap*1000:.0f} mm jaw gap, so the fingers stop about '
                f'{(gap - measured)*1000:.0f} mm short and never grip. Set '
                f'minimum_probe_width_m/nominal_probe_width_m to about '
                f'{measured*1000:.0f} mm.')
        elif self.maximum_probe_width_m < measured - 1e-4:
            self.get_logger().warning(
                f'maximum_probe_width_m={self.maximum_probe_width_m*1000:.0f} mm is below the '
                f'{measured*1000:.0f} mm probe measured from the mesh; a correct width reading '
                f'will be rejected as implausible and replaced by the nominal width.')

    def _probe_dims(self) -> np.ndarray:
        """Probe extents (X width, Y height, Z long axis), measured if possible."""
        self._ensure_probe_mesh()
        if self._probe_stl_dims_measured is not None:
            return self._probe_stl_dims_measured
        return PROBE_STL_DIMS_FALLBACK

    def _probe_stl_end_sign_or_default(self) -> int:
        """``_probe_stl_end_sign`` with the mesh loaded on demand.

        Detection runs long before the attach step loads probe.stl, so the
        cached sign is not yet available there. If the mesh cannot be read,
        fall back to probe.stl's shipped profile (wide body at low Z, tapered
        tip at high Z).
        """
        self._ensure_probe_mesh()
        sign = self._probe_stl_end_sign()
        return sign if sign != 0 else -1

    def _orient_probe_fit_end(
        self,
        R_fit: np.ndarray,
        centre: np.ndarray,
        pts: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Resolve the 180° end-for-end ambiguity from the cloud's width profile.

        A symmetric box fit cannot see which way the probe's tapered tip
        points, and inheriting the attached pose's convention preserves a
        flipped attach forever. When the measured cloud covers both ends and
        its per-half radial widths are decisively asymmetric, orient the STL
        so its fat end matches the cloud's fat end. Returns the (possibly
        flipped) rotation, or None when the cloud is not decisive.
        """
        stl_sign = self._probe_stl_end_sign()
        if stl_sign == 0:
            return None
        w_neg, w_pos, n_neg, n_pos = probe_alignment.axis_half_widths(pts, R_fit, centre)
        if n_neg < 40 or n_pos < 40:
            return None
        q_along = (pts - centre) @ R_fit[:, 2]
        # Require the cloud to reach 40% of the way to each end. A fixed
        # distance would be unreachable on a short probe and lax on a long one.
        min_reach = 0.40 * (0.5 * float(self._probe_dims()[2]))
        if (
            float(np.percentile(q_along, 95)) < min_reach
            or float(np.percentile(q_along, 5)) > -min_reach
        ):
            return None  # cloud does not reach far enough into both halves
        if w_neg > 1.15 * w_pos:
            measured_sign = -1
        elif w_pos > 1.15 * w_neg:
            measured_sign = 1
        else:
            return None
        self._attached_probe_tip_resolved = True
        if measured_sign == stl_sign:
            return R_fit
        flipped = R_fit.copy()
        flipped[:, 0] = -flipped[:, 0]
        flipped[:, 2] = -flipped[:, 2]
        return flipped

    def _reset_probe_realign_state(self) -> None:
        self._probe_realign_measurements.clear()
        self._probe_reacquire_measurements.clear()
        now_sec = self._now_sec()
        self._probe_realign_last_commit_sec = now_sec
        self._probe_realign_last_measurement_sec = now_sec

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
            # MoveIt drops post_grasp_probe's ACM entries along with the object,
            # so the next attach must re-assert them.
            self._world_probe_acm_applied = False
            self._attached_probe_world_yaw = None
            self._attached_probe_grasp_orientation = None
            self._attached_probe_centre_in_link = None
            self._attached_probe_axis_in_link = None
            self._attached_probe_R_stl_in_link = None
            self._attached_probe_tip_resolved = False
            self._probe_realign_measurements.clear()
            self._probe_reacquire_measurements.clear()
            removed.append('probe')

        if removed:
            self.get_logger().info(
                f'[CollisionWorld] Removed post-grasp objects: {", ".join(removed)}.'
            )

    # ------------------------------------------------------------------ #
    #   Detected probe as an explicit planning-scene mesh                  #
    # ------------------------------------------------------------------ #

    def _ensure_world_probe_collisions_allowed(self) -> None:
        """Let the gripper links touch the detected-probe mesh.

        The gripper is the end-effector that intentionally closes onto the
        probe. Arm links keep full checking, so the probe remains a real
        obstacle to everything that is not meant to touch it. MoveIt deletes
        an object's ACM entry when the object is REMOVEd, so this allowance is
        re-asserted afterwards. A diff PlanningScene REPLACES the ACM wholesale
        instead of merging into it, hence the read-modify-write.
        """
        if self._world_probe_acm_applied or self._world_probe_acm_pending:
            return
        if not (self.get_planning_scene_client.service_is_ready()
                and self.apply_planning_scene_client.service_is_ready()):
            self.get_logger().warning('[CollisionWorld] get/apply_planning_scene are not available; '
                'cannot allow the gripper links to touch the detected-probe mesh. The final '
                'descent may abort on the probe\'s own collision object.',
                throttle_duration_sec=30.0)
            return
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        self._world_probe_acm_pending = True
        self.get_planning_scene_client.call_async(req).add_done_callback(
            self._apply_world_probe_acm_allowance
        )

    def _apply_world_probe_acm_allowance(self, future) -> None:
        """Merge the gripper allowance into the fetched ACM and apply it back."""
        self._world_probe_acm_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f'[CollisionWorld] get_planning_scene failed while allowing the '
                f'detected-probe mesh: {exc}')
            return
        if response is None:
            return

        acm = response.scene.allowed_collision_matrix
        names = list(acm.entry_names)
        rows = [list(entry.enabled) for entry in acm.entry_values]

        def index_of(name: str) -> int:
            if name not in names:
                for row in rows:
                    row.append(False)
                names.append(name)
                rows.append([False] * len(names))
            return names.index(name)

        for obj in (self.world_probe_object_id, 'post_grasp_probe'):
            probe_i = index_of(obj)
            for link in self.gripper_probe_contact_links:
                i = index_of(link)
                rows[probe_i][i] = True
                rows[i][probe_i] = True

        acm.entry_names = names
        acm.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm
        apply_req = ApplyPlanningScene.Request()
        apply_req.scene = scene
        self._world_probe_acm_applied = True
        self.apply_planning_scene_client.call_async(apply_req)
        self.get_logger().info(f'[CollisionWorld] Gripper links may now overlap '
            f'{self.world_probe_object_id} and post_grasp_probe; arm links still avoid them.')

    def _publish_world_probe_object(
        self,
        centre_base: np.ndarray,
        R_obj_base: np.ndarray,
    ) -> None:
        """Publish the detected probe as its own STL mesh in the world.

        Publishing the real mesh at the tracked 6D pose gives MoveIt the
        probe's actual shape as an explicit obstacle.

        ``R_obj_base`` columns are (long axis, short axis, normal) in the
        planning frame; the probe STL is (X width, Y height, Z long axis), so
        the STL rotation is that cyclic column permutation.
        """
        if not self.world_probe_object_enabled or self._post_grasp_probe_attached:
            return
        # Same race as attached-mesh realignment: changing a collision object
        # under a live plan invalidates it. The probe is stationary during one
        # motion, so defer the update.
        if self._arm_motion_in_flight():
            return
        self._ensure_probe_mesh()
        self._ensure_world_probe_collisions_allowed()

        centre = np.asarray(centre_base, dtype=np.float64).reshape(3,)
        now_sec = self._now_sec()
        moved = (
            self._world_probe_last_centre is None
            or float(np.linalg.norm(centre - self._world_probe_last_centre))
                >= self.world_probe_object_move_threshold_m
        )
        if (
            self._world_probe_published
            and not moved
            and now_sec - self._world_probe_last_publish_sec
                < self.world_probe_object_min_republish_sec
        ):
            return

        R_stl = np.column_stack([R_obj_base[:, 1], R_obj_base[:, 2], R_obj_base[:, 0]])
        obj = CollisionObject()
        obj.header.frame_id = self.planning_frame
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = self.world_probe_object_id
        obj.operation = CollisionObject.ADD

        if self._probe_mesh_msg is not None:
            # Mesh vertices are referenced from an STL corner, not the centre.
            origin = centre - R_stl @ (0.5 * self._probe_dims())
            mesh_pose = Pose()
            mesh_pose.position.x = float(origin[0])
            mesh_pose.position.y = float(origin[1])
            mesh_pose.position.z = float(origin[2])
            mesh_pose.orientation = matrix_to_quat(R_stl)
            obj.meshes = [self._probe_mesh_msg]
            obj.mesh_poses = [mesh_pose]
        else:
            cyl = SolidPrimitive()
            cyl.type = SolidPrimitive.CYLINDER
            cyl.dimensions = [float(self._probe_dims()[2]), float(self._probe_dims()[0]) / 2.0]
            fb_pose = Pose()
            fb_pose.position.x = float(centre[0])
            fb_pose.position.y = float(centre[1])
            fb_pose.position.z = float(centre[2])
            fb_pose.orientation = matrix_to_quat(R_stl)
            obj.primitives = [cyl]
            obj.primitive_poses = [fb_pose]

        self._collision_object_pub.publish(obj)
        first = not self._world_probe_published
        self._world_probe_published = True
        self._world_probe_last_centre = centre.copy()
        self._world_probe_last_publish_sec = now_sec
        if first:
            shape = 'STL mesh' if self._probe_mesh_msg is not None else 'fallback cylinder'
            self.get_logger().info(f'[CollisionWorld] Detected probe added to the planning scene as a '
                f'{shape} (id={self.world_probe_object_id}) at '
                f'({centre[0]:.3f},{centre[1]:.3f},{centre[2]:.3f}). MoveIt now plans against the '
                'probe\'s measured shape.')

    def _remove_world_probe_object(self, reason: str) -> None:
        """Drop the world probe mesh (grasped, or no longer tracked)."""
        if not self._world_probe_published:
            return
        obj = CollisionObject()
        obj.header.frame_id = self.planning_frame
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = self.world_probe_object_id
        obj.operation = CollisionObject.REMOVE
        self._collision_object_pub.publish(obj)
        self._world_probe_published = False
        self._world_probe_last_centre = None
        # MoveIt drops an object's ACM entry along with the object, so the
        # gripper allowance has to be re-applied before the next publish.
        self._world_probe_acm_applied = False
        self.get_logger().info(f'[CollisionWorld] Removed the detected-probe mesh '
            f'(id={self.world_probe_object_id}): {reason}')

    # ------------------------------------------------------------------ #
    #   Held-probe mesh re-alignment                                       #
    # ------------------------------------------------------------------ #

    def _attached_probe_realign_should_run(self) -> bool:
        """True while the attached probe mesh should track live perception.

        Skipped during the base-box release: the probe sliding out of the
        opening gripper must not yank the mesh around mid-release.

        Also skipped while an arm motion is in flight, because a commit
        REPUBLISHES the attached collision object. Doing that to the scene
        MoveGroup is planning against invalidates the solve:
        the pick_home transport was seen aborting (status 6) after burning its
        full 10 s planning window, having had the probe mesh moved 84 ms after
        the goal was sent -- a 180 deg end-for-end flip.
        The grip is rigid over those few seconds, so deferring the correction
        until the motion finishes costs nothing: the realignment rides the
        detection stream, so it re-measures and commits on the next tick after
        the motion completes.
        """
        return (
            self.attached_probe_realign_enabled
            and self._post_grasp_probe_attached
            and self.holding_object
            and self._attached_probe_R_stl_in_link is not None
            and self._attached_probe_centre_in_link is not None
            and self.sequence_stage not in (
                stages.RELEASE_IN_BASE_BOX,
                'move_base_box_drop',
            )
            and not self._arm_motion_in_flight()
        )

    def _held_probe_monitor_should_run(self) -> bool:
        """True when the grip should be watched without touching the mesh.

        Same scope as ``_attached_probe_realign_should_run`` minus the
        realignment switch, and minus the in-flight-motion guard: that guard
        exists because a realign COMMIT republishes the collision object under
        a live plan. Sampling evidence does not, so
        it is safe while the arm moves — which is exactly when a probe slips.
        """
        return (
            self.held_probe_verification_enabled
            and self._post_grasp_probe_attached
            and self.holding_object
            and self.sequence_stage not in (
                stages.RELEASE_IN_BASE_BOX,
                'move_base_box_drop',
            )
        )

    def _arm_motion_in_flight(self) -> bool:
        """True while MoveIt is planning or executing an arm motion.

        Covers both routes: a MoveGroup goal (which plans AND executes inside
        one action, so the scene must hold still for the whole thing) and a
        Cartesian path request awaiting its plan.
        """
        if getattr(self, '_active_move_goal_handle', None) is not None:
            return True
        if getattr(self, '_cartesian_plan_in_flight', None) is not None:
            return True
        return getattr(self, '_pending_arm_motion_confirmation', None) is not None

    def _mask_points_in_link(
        self,
        mask_bool: np.ndarray,
        depth_image: np.ndarray,
        R_cam_in_link: np.ndarray,
        t_cam_in_link: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Back-project masked depth pixels into the planning-link frame."""
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
        pts_cam = np.column_stack([
            (u_v - cx) * d_v / fx,
            (v_v - cy) * d_v / fy,
            d_v,
        ])
        return pts_cam @ R_cam_in_link.T + t_cam_in_link

    def _measure_attached_probe_in_link(
        self,
        snap: FrameSnapshot,
        results: list,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fit the attached box model to the held probe seen in this frame.

        Considers every probe detection in the frame (another probe lying on
        the floor is also class 'probe'), keeps the one whose depth points
        overlap the currently attached box model, and refines the box pose
        with the trimmed point-to-box ICP. Returns (R_stl_in_link, centre) or
        None when no candidate passes the gates.
        """
        R_att = self._attached_probe_R_stl_in_link
        c_att = self._attached_probe_centre_in_link
        if R_att is None or c_att is None:
            return None

        tfm = self._lookup_transform(snap.depth_frame, self.planning_link, snap.stamp)
        if tfm is None:
            return None
        R_cam_in_link = quat_to_matrix(tfm.transform.rotation)
        t_cam_in_link = np.array([
            float(tfm.transform.translation.x),
            float(tfm.transform.translation.y),
            float(tfm.transform.translation.z),
        ], dtype=np.float64)

        half = 0.5 * self._probe_dims()
        gate = self.attached_probe_realign_gate_m
        best_pts: Optional[np.ndarray] = None
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                name = self.model.names[int(box.cls[0])]
                if self.target_class != 'any' and name != self.target_class:
                    continue
                if float(box.conf[0]) < self.attached_probe_realign_confidence_threshold:
                    continue
                mask_bool = self._get_result_mask_for_box(result, i, snap.color.shape)
                if mask_bool is None:
                    continue
                pts_link = self._mask_points_in_link(
                    mask_bool, snap.depth, R_cam_in_link, t_cam_in_link
                )
                if pts_link is None:
                    continue
                dist = probe_alignment.box_surface_distances(pts_link, half, R_att, c_att)
                inliers = pts_link[dist <= gate]
                if len(inliers) < self.attached_probe_realign_min_points:
                    continue
                if len(inliers) < self.attached_probe_realign_min_gate_fraction * len(pts_link):
                    continue
                if best_pts is None or len(inliers) > len(best_pts):
                    best_pts = inliers

        if best_pts is None:
            return None
        if len(best_pts) > 800:
            idx = np.linspace(0, len(best_pts) - 1, 800, dtype=np.int32)
            best_pts = best_pts[idx]

        fit = probe_alignment.fit_box_to_points(
            best_pts, half, R_att, c_att,
            iterations=self.attached_probe_realign_icp_iterations,
        )
        if fit is None:
            return None
        if fit.rms_m > self.attached_probe_realign_max_rms_m:
            self.get_logger().info(f'[ProbeRealign] Box fit rejected: rms={fit.rms_m*1000:.1f} mm '
                f'> {self.attached_probe_realign_max_rms_m*1000:.1f} mm '
                f'({fit.inlier_count} pts).', throttle_duration_sec=2.0)
            return None

        R_fit = fit.rotation
        centre_fit = np.array(fit.centre, dtype=np.float64)
        # Resolve which way the tapered tip points from the cloud's own width
        # profile; only when the cloud is not decisive fall back to keeping
        # the attached pose's end convention (negating columns X and Z is a
        # 180° flip that maps the collision box onto itself).
        oriented = self._orient_probe_fit_end(R_fit, centre_fit, best_pts)
        if oriented is not None:
            R_fit = oriented
        elif float(np.dot(R_fit[:, 2], R_att[:, 2])) < 0.0:
            R_fit = R_fit.copy()
            R_fit[:, 0] = -R_fit[:, 0]
            R_fit[:, 2] = -R_fit[:, 2]
        return R_fit, centre_fit

    def _reacquire_attached_probe_fit(
        self,
        snap: FrameSnapshot,
        results: list,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fit the held-probe box from scratch, ignoring the attached pose.

        Tracking gates points against the attached box, so a grossly wrong
        attach (flipped/far-off mesh) rejects every observation and can never
        self-correct. This path instead takes the most complete probe mask
        whose points sit near the gripper link (the floor at transport height
        is much further away), initialises the box from the cloud's PCA long
        axis with the centre pushed half a box-height behind the visible
        surface, and runs the same trimmed ICP.
        """
        tfm = self._lookup_transform(snap.depth_frame, self.planning_link, snap.stamp)
        if tfm is None:
            return None
        R_cam_in_link = quat_to_matrix(tfm.transform.rotation)
        t_cam_in_link = np.array([
            float(tfm.transform.translation.x),
            float(tfm.transform.translation.y),
            float(tfm.transform.translation.z),
        ], dtype=np.float64)

        max_dist = self.attached_probe_realign_reacquire_max_dist_m
        best_pts: Optional[np.ndarray] = None
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                name = self.model.names[int(box.cls[0])]
                if self.target_class != 'any' and name != self.target_class:
                    continue
                if float(box.conf[0]) < self.attached_probe_realign_confidence_threshold:
                    continue
                mask_bool = self._get_result_mask_for_box(result, i, snap.color.shape)
                if mask_bool is None:
                    continue
                pts_link = self._mask_points_in_link(
                    mask_bool, snap.depth, R_cam_in_link, t_cam_in_link
                )
                if pts_link is None or len(pts_link) < self.attached_probe_realign_min_points:
                    continue
                if float(np.linalg.norm(pts_link.mean(axis=0))) > max_dist:
                    continue  # too far from the gripper — a probe on the floor
                if best_pts is None or len(pts_link) > len(best_pts):
                    best_pts = pts_link

        source = 'mask'
        if best_pts is None:
            # At point-blank range YOLO often cannot detect the held probe at
            # all (it fills the frame from an unusual viewpoint). Fall back to
            # raw depth anchored at the grasp contact point.
            best_pts = self._depth_prior_probe_points(snap, R_cam_in_link, t_cam_in_link)
            source = 'depth-prior'
        if best_pts is None:
            self.get_logger().warning('[ProbeRealign] Re-acquisition active but no probe mask was detected '
                'and the depth-prior cloud near the grasp contact was unusable; '
                'attached mesh cannot be corrected from this frame.',
                throttle_duration_sec=5.0)
            return None
        if len(best_pts) > 800:
            idx = np.linspace(0, len(best_pts) - 1, 800, dtype=np.int32)
            best_pts = best_pts[idx]
        # Moderate fractional trim; contamination (gripper surfaces in the
        # depth-prior cloud) is rejected by the absolute residual gate inside
        # the fit — a heavy trim would discard legitimate far-face points
        # while the pose is still converging and keep the contamination.
        return self._fit_probe_box_from_cloud(best_pts, t_cam_in_link, 0.15, source)

    def _depth_prior_probe_points(
        self,
        snap: FrameSnapshot,
        R_cam_in_link: np.ndarray,
        t_cam_in_link: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Mask-free held-probe cloud from raw depth near the grasp contact.

        The probe passes through the four-bar contact point recorded at close
        time, so depth points within half a probe length of that contact —
        refined to a cylinder around the cloud's own long axis and required to
        be strongly elongated — isolate the probe from the gripper body and
        background without any segmentation.
        """
        if self.camera_info is None or snap.depth is None:
            return None
        eff = getattr(self, 'effective_target_point_offset_in_link', None)
        if eff is None or len(eff) < 3:
            return None
        contact = np.array([float(eff[0]), float(eff[1]), float(eff[2])])

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None

        stride = 3
        depth = snap.depth[::stride, ::stride]
        h_s, w_s = depth.shape[:2]
        vs, us = np.mgrid[0:h_s, 0:w_s]
        d = depth.reshape(-1).astype(np.float64)
        u = (us.reshape(-1) * stride).astype(np.float64)
        v = (vs.reshape(-1) * stride).astype(np.float64)
        near_limit = min(self.max_depth_m, self.attached_probe_realign_reacquire_max_dist_m + 0.2)
        valid = np.isfinite(d) & (d > self.min_depth_m) & (d < near_limit)
        if int(valid.sum()) < 150:
            return None
        d = d[valid]
        u = u[valid]
        v = v[valid]
        pts_cam = np.column_stack([(u - cx) * d / fx, (v - cy) * d / fy, d])
        pts = pts_cam @ R_cam_in_link.T + t_cam_in_link

        # Sphere gate around the grasp contact: the probe cannot be further
        # than half its length (plus margin) from where it was pinched.
        reach = 0.5 * float(self._probe_dims()[2]) + 0.05
        pts = pts[np.linalg.norm(pts - contact, axis=1) <= reach]
        if len(pts) < 150:
            return None

        # Two cylinder-refinement passes: PCA axis of the current subset, then
        # keep only points radially close to that axis. This peels off gripper
        # bucket/finger surfaces that share the contact region.
        for _ in range(2):
            centroid = pts.mean(axis=0)
            centered = pts - centroid
            cov = (centered.T @ centered) / max(1, len(pts) - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = normalize(eigvecs[:, int(np.argmax(eigvals))])
            along = centered @ axis
            radial = np.linalg.norm(centered - np.outer(along, axis), axis=1)
            pts = pts[radial <= 0.08]
            if len(pts) < 150:
                return None

        # The surviving cloud must actually look like the probe: strongly
        # elongated with a plausible physical extent along its axis.
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = (centered.T @ centered) / max(1, len(pts) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[1] < 1e-12 or float(eigvals[2] / eigvals[1]) < 6.0:
            return None
        axis = normalize(eigvecs[:, 2])
        along = centered @ axis
        extent = float(np.percentile(along, 98) - np.percentile(along, 2))
        if extent < 0.15 or extent > 0.45:
            return None
        return pts

    def _fit_probe_box_from_cloud(
        self,
        pts: np.ndarray,
        t_cam_in_link: np.ndarray,
        trim: float,
        source: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """PCA-initialised box ICP for re-acquisition (no attached-pose prior)."""
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = (centered.T @ centered) / max(1, len(pts) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        long_axis = normalize(eigvecs[:, int(np.argmax(eigvals))])
        view_dir = centroid - t_cam_in_link
        view_norm = float(np.linalg.norm(view_dir))
        view_dir = view_dir / view_norm if view_norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        # Visible points lie on the near surface; the true box centre is about
        # half a cross-section further along the viewing ray.
        centre_init = centroid + view_dir * (0.5 * float(self._probe_dims()[0]))
        y_axis = np.cross(long_axis, view_dir)
        if float(np.linalg.norm(y_axis)) < 1e-6:
            fallback = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(long_axis, fallback))) > 0.9:
                fallback = np.array([1.0, 0.0, 0.0])
            y_axis = np.cross(long_axis, fallback)
        y_axis = normalize(y_axis)
        x_axis = normalize(np.cross(y_axis, long_axis))
        R_init = np.column_stack([x_axis, y_axis, long_axis])

        half = 0.5 * self._probe_dims()
        fit = probe_alignment.fit_box_to_points(
            pts, half, R_init, centre_init,
            iterations=2 * self.attached_probe_realign_icp_iterations,
            trim_fraction=trim,
            outlier_residual_m=self.attached_probe_realign_max_rms_m,
        )
        if fit is None:
            return None
        max_dist = self.attached_probe_realign_reacquire_max_dist_m
        if (
            fit.rms_m > self.attached_probe_realign_max_rms_m
            or fit.inlier_count < self.attached_probe_realign_min_points
            or float(np.linalg.norm(fit.centre)) > max_dist
        ):
            self.get_logger().warning(f'[ProbeRealign] Re-acquisition fit rejected ({source}): '
                f'rms={fit.rms_m*1000:.1f}mm, inliers={fit.inlier_count}, '
                f'centre_dist={float(np.linalg.norm(fit.centre)):.3f}m.',
                throttle_duration_sec=5.0)
            return None

        R_fit = fit.rotation
        centre_fit = np.array(fit.centre, dtype=np.float64)
        # Prefer the cloud's own width profile to point the tapered tip the
        # right way; otherwise keep the axis on a consistent side across the
        # confirmation window so consecutive re-acquisitions are comparable.
        oriented = self._orient_probe_fit_end(R_fit, centre_fit, pts)
        if oriented is not None:
            R_fit = oriented
        else:
            prev = self._probe_reacquire_measurements[-1] if self._probe_reacquire_measurements else None
            ref_axis = prev[1][:, 2] if prev is not None else long_axis
            if float(np.dot(R_fit[:, 2], ref_axis)) < 0.0:
                R_fit = R_fit.copy()
                R_fit[:, 0] = -R_fit[:, 0]
                R_fit[:, 2] = -R_fit[:, 2]
        self.get_logger().info(f'[ProbeRealign] Re-acquisition fit accepted ({source}): '
            f'rms={fit.rms_m*1000:.1f}mm, inliers={fit.inlier_count}.',
            throttle_duration_sec=2.0)
        return R_fit, centre_fit

    def _try_reacquire_attached_probe(
        self, snap: FrameSnapshot, results: list, now_sec: float
    ) -> bool:
        """Replace a lost/wrong attached pose from consecutive scratch fits."""
        silent_sec = now_sec - max(
            self._probe_realign_last_measurement_sec,
            self._probe_realign_last_commit_sec,
        )
        if silent_sec < self.attached_probe_realign_reacquire_after_sec:
            return False
        fit = self._reacquire_attached_probe_fit(snap, results)
        if fit is None:
            return False
        R_meas, c_meas = fit
        self._probe_reacquire_measurements.append((now_sec, R_meas, c_meas))
        n = self.attached_probe_realign_reacquire_confirm_samples
        recent = list(self._probe_reacquire_measurements)[-n:]
        if len(recent) < n:
            return True  # recorded; waiting for confirmation
        window_sec = max(1.0, 4.0 * self.detect_period_sec * n)
        for stamp_sec, R_i, c_i in recent[:-1]:
            if now_sec - stamp_sec > window_sec:
                return True
            if float(np.linalg.norm(c_i - c_meas)) > self.attached_probe_realign_agreement_position_m:
                return True
            if probe_alignment.axis_angle_deg(R_i[:, 2], R_meas[:, 2]) \
                    > self.attached_probe_realign_agreement_angle_deg:
                return True
            if float(np.dot(R_i[:, 2], R_meas[:, 2])) < 0.0:
                return True  # consecutive fits disagree on the tip direction

        c_att = self._attached_probe_centre_in_link
        R_att = self._attached_probe_R_stl_in_link
        d_centre_m = float(np.linalg.norm(c_meas - c_att)) if c_att is not None else 0.0
        d_axis_deg = (
            probe_alignment.axis_angle_deg(R_meas[:, 2], R_att[:, 2])
            if R_att is not None else 0.0
        )
        self._probe_reacquire_measurements.clear()
        self._commit_attached_probe_realignment(
            R_meas, c_meas, d_centre_m, d_axis_deg, fast=True, reacquired=True
        )
        return True

    def _realign_attached_probe_from_results(self, snap: FrameSnapshot, results: list) -> None:
        """Keep the attached probe mesh aligned with the live mask+depth fit.

        Small deviations are committed after the last confirm_samples
        measurements agree with each other (rejects a single bad mask or a
        TF-lag frame during arm motion); a large deviation — the probe
        actually shifting inside the gripper — bypasses the republish rate
        limit so the collision world updates immediately.
        """
        now_sec = self._now_sec()
        measurement = self._measure_attached_probe_in_link(snap, results)
        if measurement is None:
            reacquired = self._try_reacquire_attached_probe(snap, results, now_sec)
            # Whether or not a from-scratch fit was recorded, this frame
            # produced no measurement GATED against the attached pose, so it
            # is not evidence that the probe is in the jaws. The re-acquisition
            # path fits any probe-shaped cloud within 400 mm of the link, which
            # includes probes still on the floor. A missing fit records no vote;
            # absence of a detection is not evidence of an empty gripper.
            if self._check_held_probe_during_transport() or reacquired:
                return
            silent_sec = now_sec - max(
                self._probe_realign_last_measurement_sec,
                self._probe_realign_last_commit_sec,
            )
            if silent_sec > self.attached_probe_realign_stale_warn_sec:
                self.get_logger().warning(f'[ProbeRealign] No usable held-probe observation for '
                    f'{silent_sec:.1f}s; keeping the last attached mesh pose '
                    f'(rigid-grip assumption). Held-probe evidence: '
                    f'{self._held_probe_evidence.summary(now_sec)}.',
                    throttle_duration_sec=5.0)
            self._check_held_probe_during_transport()
            return

        self._probe_reacquire_measurements.clear()
        R_meas, c_meas = measurement

        R_att = self._attached_probe_R_stl_in_link
        c_att = self._attached_probe_centre_in_link
        d_centre_m = float(np.linalg.norm(c_meas - c_att))
        d_axis_deg = probe_alignment.axis_angle_deg(R_meas[:, 2], R_att[:, 2])
        # An end-for-end flip is invisible to the symmetric axis metric
        # (axis_angle_deg reads ~0°): detect it from the signed dot product so
        # the deadband cannot swallow a tip-direction correction.
        end_flip = float(np.dot(R_meas[:, 2], R_att[:, 2])) < 0.0

        # Plausibility ceiling. Everything below only checks that consecutive
        # fits agree with EACH OTHER, which two fits of the same WRONG object
        # satisfy perfectly — and the fit accepts any probe-shaped cloud within
        # reach of the link, including a probe lying on the floor. The physical
        # invariant nothing else encodes: a probe clamped in closed jaws cannot
        # swing its axis by tens of degrees. Commits of 27° and 61° were slamming
        # the attached mesh around and painting the arm into
        # START_STATE_IN_COLLISION. Past the ceiling this is a mis-association,
        # not a slip, so it must not count as a held-probe vote either, and the
        # measurement timestamp must keep ageing so the stale/re-acquire path
        # takes over instead of a wrong pose being committed. An end flip is
        # exempt: it reads ~0° here and is a genuine correction.
        # Split the centre correction along/across the attached probe axis: the
        # jaws pin the lateral component, the axial one is free to slide.
        axis_att = normalize(R_att[:, 2])
        delta_centre = c_meas - c_att
        d_axial_m = abs(float(np.dot(delta_centre, axis_att)))
        d_lateral_m = float(np.linalg.norm(delta_centre - axis_att * np.dot(delta_centre, axis_att)))
        if not end_flip and (
            d_axis_deg > self.attached_probe_realign_max_correction_deg
            or d_lateral_m > self.attached_probe_realign_max_lateral_correction_m
            or d_axial_m > self.attached_probe_realign_max_axial_correction_m
        ):
            self.get_logger().warning(
                f'[ProbeRealign] Rejected an implausible fit: it implies the held '
                f'probe rotated {d_axis_deg:.1f} deg and moved {d_lateral_m*1000:.0f} mm '
                f'across / {d_axial_m*1000:.0f} mm along its own axis (ceilings '
                f'{self.attached_probe_realign_max_correction_deg:.0f} deg, '
                f'{self.attached_probe_realign_max_lateral_correction_m*1000:.0f} mm '
                f'across, {self.attached_probe_realign_max_axial_correction_m*1000:.0f} mm '
                'along). Closed jaws cannot rotate or sideways-shift the probe that '
                'far, so this is another probe-shaped cloud rather than a slip; not '
                'committing it and not counting it as held.',
                throttle_duration_sec=5.0,
            )
            self._check_held_probe_during_transport()
            return

        self._sample_held_probe_evidence(fit_centre_in_link=c_meas)
        if self._check_held_probe_during_transport():
            return  # the attached mesh has just been removed; do not republish it
        self._probe_realign_last_measurement_sec = now_sec
        self._probe_realign_measurements.append((now_sec, R_meas, c_meas))
        if (
            not end_flip
            and d_centre_m < self.attached_probe_realign_position_deadband_m
            and d_axis_deg < self.attached_probe_realign_angle_deadband_deg
        ):
            return  # attached mesh already matches reality

        n = self.attached_probe_realign_confirm_samples
        recent = list(self._probe_realign_measurements)[-n:]
        if len(recent) < n:
            return
        window_sec = max(1.0, 4.0 * self.detect_period_sec * n)
        for stamp_sec, R_i, c_i in recent[:-1]:
            if now_sec - stamp_sec > window_sec:
                return  # confirmation samples too old — wait for fresh ones
            if float(np.linalg.norm(c_i - c_meas)) > self.attached_probe_realign_agreement_position_m:
                return
            if probe_alignment.axis_angle_deg(R_i[:, 2], R_meas[:, 2]) \
                    > self.attached_probe_realign_agreement_angle_deg:
                return
            if float(np.dot(R_i[:, 2], R_meas[:, 2])) < 0.0:
                return  # consecutive fits disagree on the tip direction

        fast = (
            end_flip
            or d_centre_m >= self.attached_probe_realign_fast_position_m
            or d_axis_deg >= self.attached_probe_realign_fast_angle_deg
        )
        if (
            not fast
            and now_sec - self._probe_realign_last_commit_sec
                < self.attached_probe_realign_min_republish_sec
        ):
            return
        self._commit_attached_probe_realignment(
            R_meas, c_meas, d_centre_m, d_axis_deg, fast, end_flip=end_flip
        )

    def _commit_attached_probe_realignment(
        self,
        R_new: np.ndarray,
        c_new: np.ndarray,
        d_centre_m: float,
        d_axis_deg: float,
        fast: bool,
        reacquired: bool = False,
        end_flip: bool = False,
    ) -> None:
        prev_axis = self._attached_probe_axis_in_link
        shape_info = self._publish_probe_attachment(R_new, c_new)

        axis_in_link = normalize(R_new[:, 2].copy())
        if prev_axis is not None and float(np.dot(axis_in_link, prev_axis)) < 0.0:
            axis_in_link = -axis_in_link
        self._attached_probe_axis_in_link = axis_in_link

        # Refresh the base-box drop fact pair (probe world yaw + link
        # orientation) together so drop alignment and release verification use
        # the corrected geometry. If TF is briefly unavailable the old pair
        # stays consistent with itself and refreshes on the next commit.
        link_pose = self.get_current_link_pose_in_planning_frame()
        if link_pose is not None:
            R_link_in_world = quat_to_matrix(link_pose.orientation)
            axis_world = R_link_in_world @ axis_in_link
            self._attached_probe_world_yaw = math.atan2(
                float(axis_world[1]), float(axis_world[0])
            )
            self._attached_probe_grasp_orientation = Quaternion(
                x=float(link_pose.orientation.x),
                y=float(link_pose.orientation.y),
                z=float(link_pose.orientation.z),
                w=float(link_pose.orientation.w),
            )

        now_sec = self._now_sec()
        self._probe_realign_last_commit_sec = now_sec
        self._probe_realign_measurements.clear()
        if reacquired:
            evidence = 're-acquired from scratch after tracking loss'
        elif end_flip:
            evidence = '180° end-for-end flip corrected from the probe width profile'
        elif fast:
            evidence = 'fast slip path'
        else:
            evidence = 'confirmed drift'
        self.get_logger().info(f'[ProbeRealign] Attached probe mesh updated from live mask+depth box fit: '
            f'centre moved {d_centre_m*1000:.0f} mm, axis rotated {d_axis_deg:.1f} deg '
            f'({evidence}); {shape_info}')

    # ------------------------------------------------------------------ #
    #   Held-probe verification (is anything actually in the jaws?)        #
    # ------------------------------------------------------------------ #

    def _reset_held_probe_evidence(self) -> None:
        self._held_probe_evidence.reset()
        self._empty_close_detected = False
        self._empty_close_gap_m = None
        self._held_probe_empty_reported = False

    def _jaw_volume_in_link(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(contact_point, axis)`` of the jaw volume in the planning link frame."""
        eff = getattr(self, 'effective_target_point_offset_in_link', None)
        if eff is None or len(eff) < 3:
            return None
        contact = np.array([float(eff[0]), float(eff[1]), float(eff[2])], dtype=np.float64)
        axis = normalize(np.asarray(self.approach_axis_in_tool, dtype=np.float64).reshape(3,))
        return contact, axis

    def _vote_held_probe(self, verdict: str, detail: str) -> None:
        """Record one frame's verdict and log the pooled result when it flips."""
        if not self.held_probe_verification_enabled:
            return
        now_sec = self._now_sec()
        before = self._held_probe_evidence.verdict(now_sec)
        self._held_probe_evidence.add(verdict, now_sec, detail)
        after = self._held_probe_evidence.verdict(now_sec)
        if after != before:
            level = (self.get_logger().warning if after == grasp_verification.EMPTY
                     else self.get_logger().info)
            level(f'[HeldProbe] Verdict is now {after.upper()}: '
                  f'{self._held_probe_evidence.summary(now_sec)}. Last frame: {detail}')

    def _held_probe_verdict(self) -> str:
        if not self.held_probe_verification_enabled:
            return grasp_verification.UNKNOWN
        return self._held_probe_evidence.verdict(self._now_sec())

    def _sample_held_probe_evidence(
        self,
        fit_centre_in_link: Optional[np.ndarray] = None,
    ) -> None:
        """Vote once on whether the probe is in the jaws.

        ``fit_centre_in_link`` is this frame's ProbeRealign box-fit centre, if
        one was produced. A fit is only evidence of a HELD probe when it lands
        in the jaw volume: the re-acquisition path fits any probe-shaped cloud
        within 400 mm of the gripper link, which in the logged failure
        accepted a fit (rms 1.1 mm, 680 inliers) while the jaws were empty —
        a probe still lying on the floor is exactly that. Both evidence
        sources therefore answer the same geometric question. Without a fresh
        fit this function records no vote; absence of a detection is not proof
        that the gripper is empty.
        """
        if not self.held_probe_verification_enabled:
            return

        if fit_centre_in_link is not None:
            volume = self._jaw_volume_in_link()
            if volume is not None:
                contact, axis = volume
                inside = bool(grasp_verification.jaw_region_mask(
                    np.asarray(fit_centre_in_link, dtype=np.float64).reshape(1, 3),
                    contact,
                    axis,
                    self.held_probe_region_radius_m,
                    self.held_probe_region_along_min_m,
                    self.held_probe_region_along_max_m,
                )[0])
                offset = np.asarray(fit_centre_in_link, dtype=np.float64).reshape(3,) - contact
                if inside:
                    self._vote_held_probe(
                        grasp_verification.HELD,
                        f'ProbeRealign fit centred in the jaw volume, '
                        f'{float(np.linalg.norm(offset))*1000:.0f}mm from the contact point',
                    )
                    return
                # A fit somewhere else is a different probe, not this grasp.
                self._vote_held_probe(
                    grasp_verification.UNKNOWN,
                    f'ProbeRealign fit lies {float(np.linalg.norm(offset))*1000:.0f}mm off the '
                    'contact point, outside the jaw volume',
                )
                return

        return

    def _check_held_probe_during_transport(self) -> bool:
        """Stop transport when the pooled evidence says the gripper is empty.

        Returns True when it stopped the task, so the caller abandons whatever
        it was doing to the attached mesh this frame.

        The attached probe mesh is a claim about reality; ProbeRealign is what
        checks it. Once the pooled verdict is EMPTY the arm is carrying
        nothing, so finishing the transport would file an empty gripper as a
        completed pick.
        """
        if not (self.held_probe_verification_enabled and self.held_probe_abort_transport_on_empty):
            return False
        if self._held_probe_empty_reported or not self.holding_object:
            return False
        if self.sequence_stage in ('idle', 'done_holding', 'grasp_check_failed_holding'):
            return False
        if self._held_probe_verdict() != grasp_verification.EMPTY:
            return False
        self._held_probe_empty_reported = True
        now_sec = self._now_sec()
        failed_stage = self.sequence_stage
        self.get_logger().error(f'[HeldProbe] The gripper is EMPTY during {failed_stage}: '
            f'{self._held_probe_evidence.summary(now_sec)}. ProbeRealign finds no probe in the '
            'jaw volume, so the attached probe mesh and the "object held" state are both wrong. Stopping transport '
            'instead of completing an empty pick.')
        self._cancel_active_moveit_goal()
        self._cancel_pending_timers()
        # The mesh claimed a probe that is not there; carrying it would make
        # every later plan avoid a phantom.
        self._remove_post_grasp_collision_objects()
        self.holding_object = False
        self.task_complete = True
        self.busy = True
        self.sequence_stage = 'grasp_failed_empty_gripper'
        self.success_until_sec = now_sec + self.success_lockout_sec
        self.get_logger().error('Task stopped: the arm is carrying nothing. No pick_home, base-box '
            'drop or open command will be sent automatically. Re-run the grasp once the cause is '
            'understood (start with the final-close jaw gap in this log).')
        return True

    def compute_approach_axis_in_planning_frame(self, orientation: Quaternion) -> np.ndarray:
        return normalize(quat_to_matrix(orientation) @ self.approach_axis_in_tool.reshape(3,))

    def _azimuth_aligned_pinch_axis(self, approach: np.ndarray) -> Optional[np.ndarray]:
        """Jaw azimuth about the shaft that costs the arm the least motion.

        The probe body is a circular cylinder, so rotating the gripper about the
        shaft grips it identically -- the azimuth is a genuinely free DOF. Fixing
        it to ``cross(world_up, approach)`` spends that freedom on the probe's
        LEAN direction, which for a near-plumb probe is arbitrary: a probe
        leaning north and one leaning east give jaw azimuths 90 deg apart, and
        the wrist has to travel there for no gain at all (up to 180 deg in the
        worst case, since the jaws are 180-symmetric).

        Spend it on the arm instead: project the current pinch axis onto the
        plane normal to the shaft. That projection IS the nearest reachable
        azimuth, and because the jaws are symmetric it is automatically the
        nearer of the two equivalent solutions. Returns None when there is no
        preference to express -- no TF, or the current pinch axis lies along the
        shaft so every azimuth is equidistant -- and the caller falls back.
        """
        cur = self.get_current_tool_orientation_in_planning_frame()
        if cur is None:
            return None
        cur_pinch = quat_to_matrix(cur)[:, 0]
        perp = cur_pinch - approach * float(np.dot(cur_pinch, approach))
        if float(np.linalg.norm(perp)) < 1e-3:
            return None
        return normalize(perp)

    def _compute_axial_grasp_orientation(
        self,
        long_axis_base: np.ndarray,
    ) -> Optional[Tuple[Quaternion, np.ndarray]]:
        """Coaxial grasp orientation for a near-vertical probe.

        This gripper holds the probe like a pencil: the shaft lies ALONG the
        gripper's own long axis (tool +Z), the wide cylindrical body is clamped
        between the jaws and the tapered tip protrudes past the fingertips (see
        the CAD render -- the four-bar contact sits ~219 mm out along tool +Z).
        So the right orientation for an upright probe is not a horizontal side
        poke at the shaft, it is to line tool +Z up with the shaft and come down
        it. That also keeps the wrist high up the shaft (near the shoulder)
        instead of thrown out low and horizontal, which is what made the
        previous perpendicular side approach fall outside the reach envelope.

        Returns (orientation, approach_unit_base) where approach_unit_base is
        tool +Z in the planning frame, always pointing DOWN the shaft (toward
        the tip) so the gripper descends from the exposed upper end regardless
        of the detected axis sign. Returns None if the axis is degenerate.

            tool Z (approach)   = the shaft axis, pointing down toward the tip
            tool X (pinch axis) = horizontal, where the jaws close
            tool Y              = completes a right-handed frame
        """
        approach = normalize(np.asarray(long_axis_base, dtype=np.float64).reshape(3,))
        if float(np.linalg.norm(approach)) < 1e-9:
            return None
        # Descend from the exposed (upper) end: tool +Z points down the shaft so
        # the long gripper reaches up the shaft to the wrist, not down past it.
        if approach[2] > 0.0:
            approach = -approach

        # Rotation about the shaft is free (round body), so spend it on keeping
        # the wrist near where it already is rather than on the probe's lean.
        pinch = None
        azimuth_source = 'wrist-aligned (free rotation about the round body)'
        if self.grasp_azimuth_follow_wrist:
            pinch = self._azimuth_aligned_pinch_axis(approach)
        if pinch is None:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            pinch = np.cross(world_up, approach)
            if float(np.linalg.norm(pinch)) < 1e-6:
                # Shaft is exactly vertical: any horizontal jaw azimuth works.
                pinch = np.cross(np.array([1.0, 0.0, 0.0]), approach)
            pinch = normalize(pinch)
            azimuth_source = 'from the probe lean (no wrist preference available)'
        rod = normalize(np.cross(approach, pinch))

        R = np.column_stack([pinch, rod, approach])
        if np.linalg.det(R) < 0.0:
            R[:, 0] = -R[:, 0]

        travel = self._azimuth_travel_deg(R)
        self.get_logger().info(
            f'Vertical-probe coaxial grasp: shaft_axis_base='
            f'[{approach[0]:.2f},{approach[1]:.2f},{approach[2]:.2f}] '
            f'(tool +Z down the shaft; jaws close on the fat body). '
            f'Jaw azimuth {azimuth_source}'
            + (f'; wrist rotates {travel:.0f}° to reach it.' if travel is not None else '.')
        )
        return matrix_to_quat(R), approach

    def _azimuth_travel_deg(self, R_target: np.ndarray) -> Optional[float]:
        """Wrist rotation about the shaft implied by a target orientation.

        Reported so a needlessly spinning wrist is visible in the log rather
        than something you only notice watching the arm.
        """
        cur = self.get_current_tool_orientation_in_planning_frame()
        if cur is None:
            return None
        R_cur = quat_to_matrix(cur)
        cos_a = float(np.clip(np.dot(R_cur[:, 0], R_target[:, 0]), -1.0, 1.0))
        return abs(math.degrees(math.acos(cos_a)))

    def _fat_body_grip_offset_m(self) -> float:
        """Distance from the probe's centre to the grip station, toward the fat end.

        This is the ONE definition of "where along the probe do the jaws close".
        The approach shift and the attached-mesh pose both read it, so the
        collision mesh cannot describe a different grip than the one the arm
        actually aimed for — the two drifting apart is what put the attached mesh
        tens of mm off along the probe axis.
        """
        span = self._probe_fat_section_span_m() if self.vertical_grasp_body_offset_auto else None
        if span is not None:
            lo, hi = span
            offset = 0.5 * (lo + hi)
            # Keep the whole contact band on the body: never closer to either
            # end of the section than the end margin.
            margin = min(
                float(self.vertical_grasp_body_offset_end_margin_m),
                0.5 * (hi - lo),
            )
            offset = float(np.clip(offset, lo + margin, hi - margin))
            self.get_logger().info(
                f'Coaxial grip target from mesh: wide body spans '
                f'{lo*1000:.0f}-{hi*1000:.0f} mm from the probe centre, '
                f'gripping its middle at {offset*1000:.0f} mm '
                f'(configured fallback was '
                f'{self.vertical_grasp_body_offset_m*1000:.0f} mm).',
                once=True,
            )
            return offset

        offset = float(self.vertical_grasp_body_offset_m)
        half_len = 0.5 * float(self._probe_dims()[2])
        clamp = max(0.0, half_len - float(self.vertical_grasp_body_offset_end_margin_m))
        if offset > clamp:
            # A silently clamped offset grips a different section than the
            # one configured, which reads as a bad grasp rather than a bad
            # parameter -- exactly the confusion this branch must not add.
            self.get_logger().warning(
                f'vertical_grasp_body_offset_m={offset*1000:.0f} mm exceeds the '
                f'{clamp*1000:.0f} mm limit (half the {half_len*2*1000:.0f} mm probe less '
                f'the {self.vertical_grasp_body_offset_end_margin_m*1000:.0f} mm end margin) '
                f'and will grip {(offset - clamp)*1000:.0f} mm lower down the shaft. '
                'Enable vertical_grasp_body_offset_auto to take this from the mesh instead.',
                once=True,
            )
        return min(offset, clamp)

    def _fixed_held_probe_pose_in_link(
        self, contact_in_link: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic held-probe pose: the grip the robot actually repeats.

        The coaxial grasp is repeatable — the jaws always close on the Ø30
        cylindrical body with the tool axis down the shaft — so the probe's pose
        in the gripper frame is a fixed consequence of that geometry and does not
        need to be perceived. Estimating it from the mask instead put the mesh
        tens of mm off along the axis (the direction a cylinder fit resolves
        worst), and correcting that from live fits let mis-associated fits slam
        the mesh around.

        STL +Z runs fat-end → tip, and the fat end is the end nearest the
        gripper, so STL +Z is the tool approach axis. The centre then sits one
        grip offset -- plus ``attached_probe_axial_offset_m``, which absorbs the
        modelled contact point being short of the physical pinch -- further along
        that axis than the contact point. Rotation about the shaft is a free DOF
        for a round probe, so any right-handed frame on that axis is equally
        valid; it is built off the link frame to stay deterministic run to run.
        """
        axis = normalize(np.asarray(self.approach_axis_in_tool, dtype=np.float64).reshape(3,))
        # Pick whichever link axis is least parallel to the shaft as the seed, so
        # the cross products stay well conditioned for any tool orientation.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, axis))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        stl_x = normalize(np.cross(seed, axis))
        stl_y = normalize(np.cross(axis, stl_x))
        R_stl_in_link = np.column_stack([stl_x, stl_y, axis])
        centre_in_link = (
            np.asarray(contact_in_link, dtype=np.float64)
            + axis * self._attached_probe_centre_offset_m()
        )
        return R_stl_in_link, centre_in_link

    def _attached_probe_centre_offset_m(self) -> float:
        """Distance from the modelled contact point to the attached probe centre.

        The grip offset says where along the probe the jaws close; the axial
        offset corrects the modelled contact point itself, which sits short of
        the physical pinch. Only the attached mesh uses this sum -- the approach
        still aims with ``_fat_body_grip_offset_m`` alone, so correcting the
        collision model cannot move the grasp.
        """
        return self._fat_body_grip_offset_m() + float(self.attached_probe_axial_offset_m)

    def _vertical_grasp_body_shift(self, approach_unit_base: np.ndarray) -> Optional[np.ndarray]:
        """Base-frame shift that slides the coaxial grasp up onto the fat body.

        ``approach_unit_base`` is tool +Z pointing down the shaft (toward the
        tip), so ``-approach`` points up toward the exposed body. The detected
        pose reports the centre of the WHOLE probe, which for probe.stl sits on
        the thin Ø20 shaft; the jaws are sized for the Ø30 body, so aiming there
        pinches the wrong section. We therefore move the contact up the shaft to
        the middle of the wide cylindrical base measured off the mesh
        (``_probe_fat_section_span_m``), rather than to a hand-tuned distance.

        Clamped to keep the contact band inside that measured body section, so
        the grip can neither slide back down onto the shaft nor run off the flat
        butt. Falls back to ``vertical_grasp_body_offset_m`` when the mesh cannot
        be profiled, or when the offset is pinned by configuration. Returns None
        when the offset is disabled or clamps to nothing.
        """
        offset = self._fat_body_grip_offset_m()
        if offset <= 1e-4:
            return None
        return normalize(np.asarray(approach_unit_base, dtype=np.float64)) * (-offset)

    def _compute_downward_orientation(self) -> Optional[Quaternion]:
        """Return orientation with gripper pointing straight down (roll=pi, pitch=0),
        unless the detected object is near-vertical, in which case align the tool
        with the shaft for a coaxial grasp of the fat body instead (see
        _compute_axial_grasp_orientation). Uses object 6D pose yaw when available;
        falls back to current arm yaw."""
        # Default: not a near-vertical coaxial grasp, so no along-shaft body shift.
        self._vertical_grasp_body_shift_base = None
        if self.vertical_grasp_enabled and self.current_target_point_base is not None:
            long_axis_base = self._get_probe_reference_long_axis_base()
            if long_axis_base is not None:
                tilt_from_vertical_deg = math.degrees(
                    math.acos(float(np.clip(abs(long_axis_base[2]), 0.0, 1.0)))
                )
                if tilt_from_vertical_deg <= self.vertical_grasp_max_tilt_from_vertical_deg:
                    axial = self._compute_axial_grasp_orientation(long_axis_base)
                    if axial is not None:
                        axial_orientation, approach = axial
                        self._vertical_grasp_body_shift_base = \
                            self._vertical_grasp_body_shift(approach)
                        self.get_logger().info(
                            f'Grasp orientation: near-vertical object '
                            f'({tilt_from_vertical_deg:.1f}° from vertical) -- '
                            'coaxial grasp of the fat body (tool +Z along the shaft) '
                            'instead of top-down onto the tip.'
                        )
                        return axial_orientation
        # 6D Pose Agent: align gripper yaw with detected object orientation.
        if self.object_yaw_align_enabled and self.detected_object_yaw_rad is not None:
            yaw = self.detected_object_yaw_rad
            # The jaws are symmetric, so yaw and yaw+180° are the SAME grasp.
            # Taking whichever branch is nearer the current wrist saves up to a
            # 180° spin that buys nothing.
            flipped = False
            if self.grasp_azimuth_follow_wrist:
                cur = self.get_current_tool_orientation_in_planning_frame()
                if cur is not None:
                    R_cur = quat_to_matrix(cur)
                    cur_yaw = math.atan2(R_cur[1, 0], R_cur[0, 0])
                    def _delta(a: float) -> float:
                        return abs(math.atan2(math.sin(a - cur_yaw), math.cos(a - cur_yaw)))
                    if _delta(yaw + math.pi) < _delta(yaw):
                        yaw = math.atan2(math.sin(yaw + math.pi), math.cos(yaw + math.pi))
                        flipped = True
            self.get_logger().info(f'Grasp orientation from object 6D pose: '
                f'roll=180° pitch=0° yaw={math.degrees(yaw):.1f}° '
                f'(object long axis + {self.object_yaw_rotation_offset_deg:.0f}° offset)'
                + ('; took the 180°-equivalent branch nearer the current wrist.'
                   if flipped else '.'))
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

    def _final_descent_waypoints(self, goal_link_pose: Pose) -> List[Pose]:
        """Waypoints for a descent to the grasp contact, kept axial in the
        GRIPPER's frame. Used by every approach to that point: the initial
        descent, the lifted retry and the closed-gripper retry return.

        MoveIt interpolates Cartesian waypoints as straight lines in the
        planning frame, so a single goal waypoint turns leftover lateral error
        into a diagonal sweep taken at probe height -- the jaws travel sideways
        through the probe rather than sliding down around it. That is a
        collision, not a reach problem, which is why the descent fails with
        ``fraction<1`` while the collisions-off diagnostic returns 1.00.

        So split it: one waypoint that zeroes the lateral (perpendicular to
        tool +Z) error while still at the current standoff, then the goal, whose
        segment is now a pure translation along the tool axis. Returns the goal
        alone when the motion is already axial or the feature is disabled.
        """
        if not self.final_approach_tool_frame_only or self.grasp_orientation is None:
            return [goal_link_pose]

        current = self.get_current_link_pose_in_planning_frame()
        if current is None:
            # Without TF we cannot tell axial from lateral; a plain goal
            # waypoint is the existing, already-validated behaviour.
            return [goal_link_pose]

        approach = self.compute_approach_axis_in_planning_frame(self.grasp_orientation)
        cur_xyz = np.array(
            [current.position.x, current.position.y, current.position.z], dtype=np.float64
        )
        goal_xyz = np.array(
            [goal_link_pose.position.x, goal_link_pose.position.y, goal_link_pose.position.z],
            dtype=np.float64,
        )
        delta = goal_xyz - cur_xyz
        lateral = delta - approach * float(np.dot(delta, approach))
        lateral_mm = float(np.linalg.norm(lateral)) * 1000.0
        if float(np.linalg.norm(lateral)) <= self.final_approach_lateral_align_tol_m:
            self.get_logger().info(
                f'Final approach already axial in the gripper frame '
                f'(lateral error {lateral_mm:.1f} mm); descending along tool +Z.'
            )
            return [goal_link_pose]

        align_xyz = cur_xyz + lateral
        align = Pose()
        align.position = Point(
            x=float(align_xyz[0]), y=float(align_xyz[1]), z=float(align_xyz[2])
        )
        # Both waypoints hold the committed grasp orientation: the align step is
        # a translation onto the approach line, not a re-orientation.
        align.orientation = goal_link_pose.orientation
        self.get_logger().info(
            f'Final approach split into gripper-frame segments: align {lateral_mm:.1f} mm '
            f'laterally at the current standoff, then descend '
            f'{float(np.dot(delta, approach))*1000:.1f} mm along tool +Z only '
            '(no diagonal sweep through the probe at jaw height).'
        )
        return [align, goal_link_pose]

    def _final_ascent_waypoints(self) -> Optional[List[Pose]]:
        """The final descent, retraced backwards in the gripper's own frame.

        The probe was entered coaxially: pure translation along tool +Z down
        the shaft, after a lateral align taken at the standoff. The only motion
        that backs out of that without levering the probe sideways is the same
        path reversed -- pure tool -Z first, then the lateral step undone at
        the standoff.

        A world-vertical lift is not that motion. With the tool pitched (57 deg
        out of horizontal on this probe) a straight-up translation has a large
        component ACROSS the shaft, so it pries the probe in its socket while
        the jaws are clamped on it, and it leaves the arm at a pose the
        transport planner has never visited. Retracing the descent instead ends
        at the standoff the arm demonstrably reached seconds earlier, which is
        a far better start state for the pick_home plan.

        Returns None when no descent was recorded (nothing to reverse).
        """
        wps = self._last_final_descent_waypoints
        start = self._last_final_descent_start
        if not wps or start is None:
            return None
        # Drop the descent's own goal: that IS where the arm is standing now.
        # What remains, reversed, is every intermediate waypoint followed by the
        # standoff the descent began from.
        back: List[Pose] = list(wps[:-1])[::-1] + [start]
        goal_orientation = wps[-1].orientation
        out: List[Pose] = []
        for w in back:
            pose = Pose()
            pose.position = Point(
                x=float(w.position.x), y=float(w.position.y), z=float(w.position.z)
            )
            # Hold the committed grasp orientation throughout, exactly as the
            # descent did: this is a translation back out, not a re-orientation.
            pose.orientation = goal_orientation
            out.append(pose)
        return out

    def _final_descent_shortfall_m(self, fraction: float) -> Optional[float]:
        """How far short of the commanded contact a partial descent stops.

        Measured along the path actually sent (which may be the two-segment
        align-then-insert split), not along the goal delta, so the number stays
        honest when the descent was split.
        """
        wps = self._last_final_descent_waypoints
        if not wps:
            return None
        current = self.get_current_link_pose_in_planning_frame()
        if current is None:
            return None
        pts = [np.array([current.position.x, current.position.y, current.position.z])]
        pts += [np.array([w.position.x, w.position.y, w.position.z]) for w in wps]
        total = float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))
        return max(0.0, total * (1.0 - float(fraction)))

    def _accept_partial_final_descent(self, fraction: float) -> bool:
        """Close where the arm got to, when that still lands on the probe body.

        A descent that stops short is only a failure if the jaws end up off the
        probe. The grip is aimed at the MIDDLE of the wide cylindrical body, so
        any shortfall smaller than that body's half-length still closes on the
        Ø30 section the jaws are sized for -- and it does so further from the
        floor, not nearer, so the four-bar ground and arc guards stay satisfied.
        Resetting a sequence that has already committed and frozen perception is
        strictly worse than closing 20 mm high on the same body.

        Records the accepted shortfall so the pre-close TCP check widens by the
        same amount instead of rejecting a pose we deliberately chose.
        """
        if not self.final_approach_accept_partial_enabled:
            return False
        shortfall = self._final_descent_shortfall_m(fraction)
        if shortfall is None:
            return False

        span = self._probe_fat_section_span_m()
        if span is not None:
            half_body = 0.5 * (span[1] - span[0])
            source = f'half the {(span[1]-span[0])*1000:.0f} mm wide body measured from probe.stl'
        else:
            half_body = float(self.final_approach_accept_shortfall_fallback_m)
            source = 'the configured fallback (probe mesh unavailable)'
        limit = max(0.0, half_body - float(self.final_approach_accept_shortfall_margin_m))
        if shortfall > limit:
            self.get_logger().warning(
                f'Final descent stopped {shortfall*1000:.1f} mm short, beyond the '
                f'{limit*1000:.1f} mm that still lands on the probe body '
                f'({source}); not closing here.'
            )
            return False

        self._accepted_descent_shortfall_m = float(shortfall)
        self.get_logger().warning(
            f'Final descent {fraction:.2f} complete, {shortfall*1000:.1f} mm short of the '
            f'commanded contact. That still grips the wide body '
            f'{shortfall*1000:.1f} mm above its centre, {(half_body - shortfall)*1000:.1f} mm '
            f'below the shoulder, and further from the floor than commanded -- '
            'executing and closing here rather than resetting the grasp.'
        )
        return True

    def _grasp_reach_shortfall_mm(self, link_pose: Pose) -> Optional[float]:
        """How far (mm) a final-descent link pose lies beyond the arm's reach.

        Shoulder-sphere model: the wrist (joint5) sits
        reach_guard_wrist_backoff_in_link_m behind arm_gripper_base_link along
        tool +Z and can extend at most reach_guard_max_wrist_extension_m from
        the shoulder point. Positive result = out of reach by that much
        (including reach_guard_margin_m); <= 0 = reachable. None = guard off.
        """
        if not self.reach_guard_enabled:
            return None
        link_xyz = np.array([
            float(link_pose.position.x),
            float(link_pose.position.y),
            float(link_pose.position.z),
        ], dtype=np.float64)
        return wrist_extension_shortfall_m(
            link_xyz,
            link_pose.orientation,
            self.reach_guard_shoulder_xyz_in_base,
            self.reach_guard_max_wrist_extension_m,
            self.reach_guard_wrist_backoff_in_link_m,
            self.reach_guard_margin_m,
        ) * 1000.0

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
        # With full-close mode the jaws sweep past the computed contact angle
        # whenever the probe is missed, so the ground guard must cover the
        # whole commanded arc, not just the expected contact point.
        arc_end_q = float(self.computed_gripper_close)
        if self.final_close_full_close:
            arc_end_q = max(arc_end_q, float(self.gripper_close))
        samples = np.linspace(float(self.gripper_open), arc_end_q, n)
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
        target = np.asarray(target, dtype=np.float64)
        # Near-vertical coaxial grasp: slide the aim point up the shaft onto the
        # fat body (set in _compute_downward_orientation). This clamps the wide
        # cylindrical section the jaws are sized for and lifts the wrist toward
        # the shoulder so the pose stays inside the reach envelope.
        if self._vertical_grasp_body_shift_base is not None:
            shift = self._vertical_grasp_body_shift_base
            target = target + shift
            self.get_logger().info(
                'Coaxial fat-body grip: aim point shifted '
                f'{float(np.linalg.norm(shift))*1000:.0f} mm up the shaft '
                f'(base=({shift[0]*1000:.0f},{shift[1]*1000:.0f},{shift[2]*1000:.0f})mm) '
                'off the tip onto the body.',
                throttle_duration_sec=1.0,
            )
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
        self.target_pose_fit_history.clear()
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

    def is_target_stable(
        self,
        p: np.ndarray,
        confidence: float,
        pose_fit_fresh: bool = True,
    ) -> bool:
        """Build a consecutive, spatially coherent 3D cluster and median-filter it."""
        now_sec = self._now_sec()
        self._expire_target_stability_history(now_sec)

        confidence_floor = (
            self.target_lock_continue_min_confidence
            if self.target_history
            else self.target_lock_min_confidence
        )
        if confidence < confidence_floor:
            self.get_logger().info(
                f'Probe confidence {confidence:.2f} is below '
                f'{"continuation" if self.target_history else "initial lock"} threshold '
                f'{confidence_floor:.2f}; not adding it to the '
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
        self.target_pose_fit_history.append(bool(pose_fit_fresh))

        while len(self.target_history) > self.target_filter_window_samples:
            self.target_history.pop(0)
            self.target_history_stamps.pop(0)
            self.target_confidence_history.pop(0)
            self.target_pose_fit_history.pop(0)

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
            and sum(self.target_pose_fit_history)
            >= self.target_stability_min_pose_fit_samples
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

        The model regularly returns two overlapping boxes for one probe: the
        whole object plus a stub covering one end. Plain confidence ranking can
        pick the stub, and a stub mask makes the PCA long axis meaningless
        (short axis, low eigenratio), so nested duplicates are suppressed
        before ranking. Standard IoU barely notices a small box inside a big
        one; intersection-over-smaller-area does.
        """
        if image_shape is None and self.latest_color is not None:
            image_shape = self.latest_color.shape

        detections = []
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

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'name': name,
                    'conf': conf,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'u_bbox': int((x1 + x2) * 0.5),
                    'v_bbox': int((y1 + y2) * 0.5),
                    'result': result,
                    'box_index': i,
                })

        if not detections:
            return None

        kept = self._suppress_nested_detections(detections)
        best = max(kept, key=lambda det: det['conf'])
        best['mask'] = self._get_result_mask_for_box(
            best.pop('result'), best.pop('box_index'), image_shape
        )
        return best

    def _suppress_nested_detections(self, detections: List[dict]) -> List[dict]:
        """Drop boxes largely contained inside a larger same-class box."""
        threshold = self.detection_nested_overlap_threshold
        areas = [
            max(1, (d['x2'] - d['x1'])) * max(1, (d['y2'] - d['y1']))
            for d in detections
        ]
        # Largest first, so a survivor is always the most complete box seen.
        order = sorted(range(len(detections)), key=lambda i: -areas[i])
        kept_idx: List[int] = []
        for i in order:
            a = detections[i]
            nested_in = None
            for j in kept_idx:
                b = detections[j]
                if a['name'] != b['name']:
                    continue
                overlap_w = max(0, min(a['x2'], b['x2']) - max(a['x1'], b['x1']))
                overlap_h = max(0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
                inter = overlap_w * overlap_h
                if inter and inter / float(min(areas[i], areas[j])) >= threshold:
                    nested_in = j
                    break
            if nested_in is None:
                kept_idx.append(i)
            else:
                self.get_logger().info(
                    f'Suppressed duplicate {a["name"]} detection (conf={a["conf"]:.2f}, '
                    f'{a["x2"]-a["x1"]}x{a["y2"]-a["y1"]} px) nested inside a larger one '
                    f'(conf={detections[nested_in]["conf"]:.2f}); a partial mask would '
                    f'corrupt the 6D pose.', throttle_duration_sec=2.0)
        return [detections[i] for i in kept_idx]

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
        closest_gap_sec = float('inf')
        for color_stamp, color in self._color_frame_queue:
            for depth_stamp, depth_frame, depth in self._depth_frame_queue:
                gap_sec = abs((color_stamp - depth_stamp).nanoseconds) * 1e-9
                closest_gap_sec = min(closest_gap_sec, gap_sec)
                if gap_sec > self.max_color_depth_stamp_gap_sec:
                    continue
                newest_sec = max(color_stamp.nanoseconds, depth_stamp.nanoseconds) * 1e-9
                # Freshness wins once a pair is inside the synchronization
                # bound. Otherwise an old exact pair can make the inference
                # worker drain a backlog while newer usable frames wait.
                key = (-newest_sec, gap_sec)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (
                        color_stamp, color, depth_stamp, depth_frame, depth, gap_sec
                    )

        if best_pair is not None:
            color_stamp, color, depth_stamp, depth_frame, depth, gap_sec = best_pair
            pair_key = (color_stamp.nanoseconds, depth_stamp.nanoseconds)
            if pair_key != self._last_inference_pair_key:
                self._last_inference_pair_key = pair_key
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
                    depth=depth,
                    depth_stamp_sec=depth_stamp.nanoseconds * 1e-9,
                    depth_frame=depth_frame,
                    stamp=depth_stamp,
                )
                self._yolo_worker.submit(snap, snap.color)
        elif closest_gap_sec < float('inf'):
            now_sec = self._now_sec()
            if now_sec - self._stamp_gap_warned_sec > 5.0:
                self._stamp_gap_warned_sec = now_sec
                self.get_logger().warning(
                    f'Waiting for synchronized camera frames: closest color/depth pair differs by '
                    f'{closest_gap_sec:.3f}s (> {self.max_color_depth_stamp_gap_sec:.3f}s '
                    'max_color_depth_stamp_gap_sec). A mismatched pair on a '
                    'moving camera yields a wrong 3D target; check camera rates '
                    'if this persists.'
                )

        completed = self._yolo_worker.take_result()
        if completed is None:
            return None
        completed_snap, _ = completed
        result_age = self._now_sec() - completed_snap.depth_stamp_sec
        verdict, reason = classify_frame_age(
            result_age,
            using_sim_time=bool(self.get_parameter('use_sim_time').value),
            max_age_sec=self.inference_result_max_age_sec,
        )
        if verdict == 'clock_mismatch':
            # Not a tuning problem: inference is fine and every result is being
            # thrown away, so this is an error rather than a warning.
            self.get_logger().error(
                f'Camera stamps are on a different clock than this node: frame '
                f'stamp={completed_snap.depth_stamp_sec:.2f}s vs node '
                f'now={self._now_sec():.2f}s. No inference can pass the '
                f'freshness gate until this is fixed: {reason}.',
                throttle_duration_sec=5.0,
            )
            return None
        if verdict == 'stale':
            self.get_logger().warning(
                f'Dropping stale vision inference result: {reason}.',
                throttle_duration_sec=2.0,
            )
            return None
        return completed

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
        self._probe_track_association_rejected = False
        self._probe_track_pose_held = False
        self._last_detection_uses_held_pose = False

        annotated = self._annotate_yolo_results(snap.color, results)
        if not allow_state_updates and self.busy:
            self._stamp_debug_status(
                annotated,
                f'Live detect view: {self.sequence_stage}',
            )

        # Held-probe mesh tracking rides on the same inference stream. It only
        # touches the attached collision object, never the grasp target state,
        # so it runs regardless of allow_state_updates/perception freeze.
        if self._attached_probe_realign_should_run():
            self._realign_attached_probe_from_results(snap, results)
        elif self._held_probe_monitor_should_run():
            # Realignment is off, but "is the probe still in the jaws" is a
            # separate question from "where exactly is it". Keep sampling the
            # empty-grip evidence during transport — it is read-only and never
            # moves the attached mesh, so it cannot reintroduce the mesh being
            # yanked around. Without this, disabling realignment would also
            # silently retire the only check that catches a dropped probe
            # between the lift check and the release.
            self._sample_held_probe_evidence()
            self._check_held_probe_during_transport()

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
                    confidence=best['conf'],
                    depth_frame=snap.depth_frame,
                    stamp=snap.stamp,
                )
                self._last_detection_uses_held_pose = self._probe_track_pose_held
                if self._probe_track_association_rejected:
                    if publish_debug:
                        self._stamp_debug_status(
                            annotated, 'Rejected non-matching probe fit',
                            color=(0, 0, 255),
                        )
                        self.publish_debug_image(annotated)
                    return None
            else:
                self._clear_detected_object_pose()
        elif allow_state_updates:
            self._clear_detected_object_pose()

        if (
            allow_state_updates
            and self.busy
            and self.probe_track_enabled
            and self._probe_track is not None
            and (
                self._now_sec() - self._probe_track.last_update_sec
                <= self.probe_track_timeout_sec
            )
            and self.detected_object_pose is None
        ):
            # Do not let an unassociated mask/bbox centre replace a still-fresh
            # shape track during arm motion. Partial occlusion made those raw
            # points jump by 60–90 mm in the supplied trace.
            if publish_debug:
                self._stamp_debug_status(
                    annotated, 'Shape track held; raw target ignored',
                    color=(0, 165, 255),
                )
                self.publish_debug_image(annotated)
            return None

        # Shape-aware width: once the known probe box is fitted, the width is a
        # model fact, not a mask guess. This replaces the noisy mask width (which
        # over-reads to ~70 mm on partial masks) with the true probe cross-section.
        if (
            allow_state_updates
            and self.adaptive_gripper_enabled
            and self.shape_aware_pose_enabled
            and self.detected_object_pose is not None
        ):
            dims = self._probe_track.dims if self._probe_track is not None else self._probe_dims()
            model_w = float(max(dims[0], dims[1]))
            if self.adaptive_gripper_min_width_m <= model_w <= self.adaptive_gripper_max_width_m:
                self._last_detected_width_m = model_w

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

        # If PCA/shape fitting is marginal for one frame, validate the fresh
        # track against the independent raw mask point and keep the last good
        # full-model pose. This prevents alternating between the fitted centre
        # and a visible-mask centroid. The pose hold is deliberately bounded
        # and held samples alone cannot satisfy the final stability lock.
        if (
            allow_state_updates
            and self.shape_aware_pose_enabled
            and self.detected_object_pose is None
            and self.probe_track_enabled
            and self._probe_track is not None
        ):
            track = self._probe_track
            now_sec = self._now_sec()
            pose_age = now_sec - track.last_pose_update_sec
            raw_jump = float(np.linalg.norm(point_base - track.centre_base))
            if (
                pose_age <= self.probe_track_pose_hold_sec
                and raw_jump <= self.probe_track_raw_validation_max_jump_m
            ):
                track.last_update_sec = now_sec
                track.hit_count += 1
                track.miss_count = 0
                track.confidence = float(best['conf'])
                self.detected_object_pose = self.make_pose(
                    track.centre_base, matrix_to_quat(track.R_base)
                )
                self._last_detected_object_rotation_base = track.R_base.copy()
                self._last_detection_uses_held_pose = True
                self.get_logger().info(
                    f'Probe track#{track.track_id} held through a marginal 6D '
                    f'frame (raw-mask agreement={raw_jump*1000:.0f}mm, '
                    f'pose age={pose_age:.1f}s).',
                    throttle_duration_sec=1.0,
                )

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

        # Shape-aware grasp target: aim at the fitted FULL-model centre held by
        # the track, not the raw mask pick point. The centre already predicts the
        # occluded part, so the grasp (and the coaxial fat-body offset that runs
        # from it) references the whole probe, not just its visible surface.
        if self.shape_aware_pose_enabled and self.detected_object_pose is not None:
            point_base = self._pose_xyz(self.detected_object_pose)
            source = (f'track#{self._probe_track.track_id}'
                      if self._probe_track is not None else 'shape')

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
        if self.latest_color is None:
            self.get_logger().warning(
                'Waiting for gripper-camera color frames on '
                '/gripper_camera/color/image_raw; detection image cannot be '
                'published yet. Check enable_depth_sensor and camera topics.',
                throttle_duration_sec=5.0,
            )
            return
        if self.latest_depth is None or self.camera_info is None:
            self.get_logger().warning(
                'Color frames are available, but aligned depth/camera_info is '
                'still missing; publishing the live color view while waiting.',
                throttle_duration_sec=5.0,
            )
            self.publish_debug_image(self.latest_color)
            return
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

        # Prevent loop after successful grasp — but while a probe is still
        # attached (done_holding / failed-transport holds), keep running
        # read-only detection so the held-probe mesh keeps tracking reality.
        if self.task_complete and not self.auto_restart_after_success:
            if self._attached_probe_realign_should_run():
                self.detect_target_once(publish_debug=True, allow_state_updates=False)
            return

        if self.task_complete and now_sec < self.success_until_sec:
            if self._attached_probe_realign_should_run():
                self.detect_target_once(publish_debug=True, allow_state_updates=False)
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

        if not self.is_target_stable(
            point_base,
            conf,
            pose_fit_fresh=not self._last_detection_uses_held_pose,
        ):
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
        self._reset_held_probe_evidence()
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

    @staticmethod
    def _joint_map_from_robot_state(robot_state: RobotState) -> dict:
        """Return finite named joint positions from a MoveIt robot state."""
        result = {}
        if robot_state is None:
            return result
        joint_state = getattr(robot_state, 'joint_state', None)
        if joint_state is None:
            return result
        for name, position in zip(joint_state.name, joint_state.position):
            value = float(position)
            if name and math.isfinite(value):
                result[str(name)] = value
        return result

    def _grasp_ik_request(
        self,
        contact_pose: PoseStamped,
        orientation: Quaternion,
        seed_state: Optional[RobotState],
    ) -> GetPositionIK.Request:
        """Build an exact, collision-aware IK query for one grasp azimuth."""
        candidate_contact = PoseStamped()
        candidate_contact.header.frame_id = contact_pose.header.frame_id
        candidate_contact.header.stamp = self.get_clock().now().to_msg()
        candidate_contact.pose.position = Point(
            x=float(contact_pose.pose.position.x),
            y=float(contact_pose.pose.position.y),
            z=float(contact_pose.pose.position.z),
        )
        candidate_contact.pose.orientation = orientation

        candidate_link = PoseStamped()
        candidate_link.header = candidate_contact.header
        candidate_link.pose = self.contact_pose_to_link_pose(candidate_contact)

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.planning_group
        req.ik_request.ik_link_name = self.planning_link
        req.ik_request.pose_stamped = candidate_link
        req.ik_request.avoid_collisions = True
        timeout = self.grasp_joint_aware_ik_timeout_sec
        req.ik_request.timeout = Duration(
            sec=int(timeout),
            nanosec=int((timeout % 1.0) * 1e9),
        )
        if seed_state is not None:
            req.ik_request.robot_state = seed_state
        return req

    def _start_joint_aware_grasp_orientation_search(
        self,
        base_orientation: Quaternion,
    ) -> bool:
        """Select an equivalent grasp that minimizes the complete joint path.

        The old heuristic compared tool-frame azimuth only. It could therefore
        choose an inverse-kinematics family that put bounded joint6 on the
        opposite side of its range, causing a nearly full revolution and
        leaving no IK continuation for the final descent. This search checks
        both pre-grasp and grasp before any motion is commanded.
        """
        if not self.grasp_joint_aware_orientation_search_enabled:
            return False
        if self.pre_grasp_pose is None or self.grasp_pose is None:
            return False
        if not self.compute_ik_client.service_is_ready():
            self.get_logger().warning(
                'Joint-aware grasp orientation search skipped because '
                'the MoveIt compute_ik service is unavailable.'
            )
            return False
        if not self.cartesian_client.service_is_ready():
            self.get_logger().warning(
                'Joint-aware grasp orientation search skipped because '
                'the MoveIt Cartesian-path service is unavailable.'
            )
            return False

        required_names = tuple(self.arm_feedback_joint_names)
        wrist_name = self.grasp_wrist_joint_name
        if wrist_name not in required_names:
            self.get_logger().warning(
                f'Joint-aware grasp orientation search skipped: wrist joint '
                f'"{wrist_name}" is not in arm_feedback_joint_names.'
            )
            return False
        missing = [
            name for name in required_names
            if name not in self.current_joint_positions
        ]
        if missing:
            self.get_logger().warning(
                'Joint-aware grasp orientation search skipped because live '
                f'joint feedback is incomplete: missing={missing}.'
            )
            return False

        # A coaxial grasp of the round probe is invariant to arbitrary tool
        # rotation about local +Z. For other grasps only use the jaw's 180°
        # symmetry; intermediate rotations could change the grasp itself.
        count = (
            self.grasp_joint_aware_azimuth_candidates
            if self._vertical_grasp_body_shift_base is not None
            else 2
        )
        variants = [
            (angle, matrix_to_quat(rotation))
            for angle, rotation in wrist_planning.azimuth_variants(
                quat_to_matrix(base_orientation),
                count,
            )
        ]
        self._grasp_orientation_search = {
            'sequence_id': self.sequence_id,
            'candidates': variants,
            'index': 0,
            'phase': 'pregrasp',
            'best': None,
            'current': {
                name: float(self.current_joint_positions[name])
                for name in required_names
            },
            'required_names': required_names,
            'pre_solution': None,
            'pre_joints': None,
            'grasp_joints': None,
            'indeterminate': 0,
            'definitive_rejections': 0,
        }
        self.sequence_stage = 'optimize_grasp_orientation'
        self.get_logger().info(
            f'Joint-aware grasp search: evaluating {len(variants)} equivalent '
            f'azimuths at both pre-grasp and final grasp before moving; '
            f'{wrist_name} motion has {self.grasp_wrist_joint_weight:.1f}x weight.'
        )
        self._request_grasp_orientation_ik()
        return True

    def _request_grasp_orientation_ik(self) -> None:
        search = self._grasp_orientation_search
        if search is None or search['sequence_id'] != self.sequence_id:
            return
        index = int(search['index'])
        candidates = search['candidates']
        if index >= len(candidates):
            self._finish_grasp_orientation_search()
            return

        phase = str(search['phase'])
        orientation = candidates[index][1]
        if phase == 'pregrasp':
            contact_pose = self.pre_grasp_pose
            seed_state = self._make_current_robot_state()
        else:
            contact_pose = self.grasp_pose
            seed_state = search['pre_solution']
        if contact_pose is None:
            self._finish_grasp_orientation_search()
            return

        req = self._grasp_ik_request(contact_pose, orientation, seed_state)
        try:
            future = self.compute_ik_client.call_async(req)
        except Exception as exc:
            search['indeterminate'] += 1
            self.get_logger().warning(
                f'Joint-aware grasp IK request failed for candidate '
                f'{index + 1}/{len(candidates)} ({phase}): {exc}'
            )
            self._advance_grasp_orientation_candidate()
            return

        self._grasp_orientation_search_token += 1
        token = self._grasp_orientation_search_token
        future.add_done_callback(
            lambda fut, seq=self.sequence_id, i=index, p=phase, t=token:
                self._on_grasp_orientation_ik(fut, seq, i, p, t)
        )
        self.call_later(
            self.grasp_joint_aware_ik_timeout_sec + 0.25,
            lambda seq=self.sequence_id, i=index, p=phase, t=token:
                self._on_grasp_orientation_search_timeout(seq, i, p, t),
        )

    def _on_grasp_orientation_search_timeout(
        self,
        expected_seq: int,
        index: int,
        phase: str,
        request_token: int,
    ) -> None:
        search = self._grasp_orientation_search
        if (
            search is None
            or expected_seq != self.sequence_id
            or search['sequence_id'] != expected_seq
            or self.sequence_stage != 'optimize_grasp_orientation'
            or request_token != self._grasp_orientation_search_token
            or int(search['index']) != index
            or str(search['phase']) != phase
        ):
            return
        self._grasp_orientation_search_token += 1
        search['indeterminate'] += 1
        self.get_logger().warning(
            f'Joint-aware grasp {phase} check timed out for candidate '
            f'{index + 1}/{len(search["candidates"])} ({phase}); trying the next azimuth.'
        )
        self._advance_grasp_orientation_candidate()

    def _on_grasp_orientation_ik(
        self,
        future,
        expected_seq: int,
        index: int,
        phase: str,
        request_token: int,
    ) -> None:
        search = self._grasp_orientation_search
        if (
            search is None
            or expected_seq != self.sequence_id
            or search['sequence_id'] != expected_seq
            or self.sequence_stage != 'optimize_grasp_orientation'
            or request_token != self._grasp_orientation_search_token
            or int(search['index']) != index
            or str(search['phase']) != phase
        ):
            return
        self._grasp_orientation_search_token += 1
        try:
            result = future.result() if future.done() else None
        except Exception as exc:
            search['indeterminate'] += 1
            self.get_logger().warning(
                f'Joint-aware grasp IK response failed for candidate '
                f'{index + 1}/{len(search["candidates"])} ({phase}): {exc}'
            )
            self._advance_grasp_orientation_candidate()
            return

        if (
            result is None
            or int(result.error_code.val) != MoveItErrorCodes.SUCCESS
        ):
            if result is None:
                search['indeterminate'] += 1
            else:
                search['definitive_rejections'] += 1
            self._advance_grasp_orientation_candidate()
            return

        joints = self._joint_map_from_robot_state(result.solution)
        required_names = search['required_names']
        if any(name not in joints for name in required_names):
            search['indeterminate'] += 1
            self._advance_grasp_orientation_candidate()
            return

        if phase == 'pregrasp':
            search['pre_solution'] = result.solution
            search['pre_joints'] = joints
            search['phase'] = 'grasp'
            self._request_grasp_orientation_ik()
            return

        search['grasp_joints'] = joints
        search['phase'] = 'cartesian'
        self._request_grasp_orientation_cartesian()

    def _request_grasp_orientation_cartesian(self) -> None:
        """Verify that IK remains continuous through the final insertion."""
        search = self._grasp_orientation_search
        if search is None or search['sequence_id'] != self.sequence_id:
            return
        index = int(search['index'])
        orientation = search['candidates'][index][1]
        if self.grasp_pose is None or search['pre_solution'] is None:
            self._advance_grasp_orientation_candidate()
            return

        candidate_contact = PoseStamped()
        candidate_contact.header.frame_id = self.grasp_pose.header.frame_id
        candidate_contact.header.stamp = self.get_clock().now().to_msg()
        candidate_contact.pose.position = Point(
            x=float(self.grasp_pose.pose.position.x),
            y=float(self.grasp_pose.pose.position.y),
            z=float(self.grasp_pose.pose.position.z),
        )
        candidate_contact.pose.orientation = orientation

        req = GetCartesianPath.Request()
        req.header.frame_id = self.planning_frame
        req.header.stamp = candidate_contact.header.stamp
        req.group_name = self.planning_group
        req.link_name = self.planning_link
        req.waypoints = [self.contact_pose_to_link_pose(candidate_contact)]
        req.max_step = self.cartesian_max_step
        req.jump_threshold = self.cartesian_jump_threshold
        req.avoid_collisions = True
        req.start_state = search['pre_solution']

        try:
            future = self.cartesian_client.call_async(req)
        except Exception as exc:
            search['indeterminate'] += 1
            self.get_logger().warning(
                f'Joint-aware Cartesian request failed for candidate '
                f'{index + 1}/{len(search["candidates"])}: {exc}'
            )
            self._advance_grasp_orientation_candidate()
            return

        self._grasp_orientation_search_token += 1
        token = self._grasp_orientation_search_token
        future.add_done_callback(
            lambda fut, seq=self.sequence_id, i=index, t=token:
                self._on_grasp_orientation_cartesian(fut, seq, i, t)
        )
        self.call_later(
            self.grasp_joint_aware_cartesian_timeout_sec,
            lambda seq=self.sequence_id, i=index, t=token:
                self._on_grasp_orientation_search_timeout(
                    seq, i, 'cartesian', t
                ),
        )

    def _on_grasp_orientation_cartesian(
        self,
        future,
        expected_seq: int,
        index: int,
        request_token: int,
    ) -> None:
        search = self._grasp_orientation_search
        if (
            search is None
            or expected_seq != self.sequence_id
            or search['sequence_id'] != expected_seq
            or self.sequence_stage != 'optimize_grasp_orientation'
            or request_token != self._grasp_orientation_search_token
            or int(search['index']) != index
            or str(search['phase']) != 'cartesian'
        ):
            return
        self._grasp_orientation_search_token += 1
        try:
            result = future.result() if future.done() else None
        except Exception as exc:
            search['indeterminate'] += 1
            self.get_logger().warning(
                f'Joint-aware Cartesian response failed for candidate '
                f'{index + 1}/{len(search["candidates"])}: {exc}'
            )
            self._advance_grasp_orientation_candidate()
            return

        if result is None:
            search['indeterminate'] += 1
            self._advance_grasp_orientation_candidate()
            return
        fraction = float(result.fraction)
        if fraction < self.cartesian_fraction_min:
            search['definitive_rejections'] += 1
            self.get_logger().info(
                f'Joint-aware grasp rejected azimuth '
                f'{math.degrees(search["candidates"][index][0]):+.1f}°: '
                f'pre-grasp IK exists but final Cartesian continuity is only '
                f'{fraction:.2f}.',
                throttle_duration_sec=0.5,
            )
            self._advance_grasp_orientation_candidate()
            return

        # Prefer the path's actual final joint state over the separately
        # queried endpoint IK when MoveIt returned a complete trajectory.
        trajectory = getattr(result.solution, 'joint_trajectory', None)
        if trajectory is not None and trajectory.points:
            end_positions = trajectory.points[-1].positions
            if len(trajectory.joint_names) == len(end_positions):
                endpoint = dict(search['grasp_joints'])
                endpoint.update({
                    str(name): float(position)
                    for name, position in zip(
                        trajectory.joint_names, end_positions
                    )
                })
                search['grasp_joints'] = endpoint
        self._record_grasp_orientation_candidate()

    def _record_grasp_orientation_candidate(self) -> None:
        search = self._grasp_orientation_search
        if search is None:
            return
        index = int(search['index'])
        angle, orientation = search['candidates'][index]
        try:
            score, wrist_travel, clearance = wrist_planning.score_joint_path(
                search['current'],
                search['pre_joints'],
                search['grasp_joints'],
                search['required_names'],
                wrist_joint_name=self.grasp_wrist_joint_name,
                wrist_weight=self.grasp_wrist_joint_weight,
                wrist_limits=(
                    self.grasp_wrist_joint_lower_rad,
                    self.grasp_wrist_joint_upper_rad,
                ),
                wrist_limit_margin=self.grasp_wrist_joint_limit_margin_rad,
            )
        except ValueError:
            search['indeterminate'] += 1
            self._advance_grasp_orientation_candidate()
            return

        candidate = {
            'score': score,
            'angle': angle,
            'orientation': orientation,
            'pre_joints': search['pre_joints'],
            'grasp_joints': search['grasp_joints'],
            'wrist_travel': wrist_travel,
            'clearance': clearance,
        }
        best = search['best']
        if best is None or candidate['score'] < best['score']:
            search['best'] = candidate
        self._advance_grasp_orientation_candidate()

    def _advance_grasp_orientation_candidate(self) -> None:
        search = self._grasp_orientation_search
        if search is None:
            return
        search['index'] += 1
        search['phase'] = 'pregrasp'
        search['pre_solution'] = None
        search['pre_joints'] = None
        search['grasp_joints'] = None
        self._request_grasp_orientation_ik()

    def _finish_grasp_orientation_search(self) -> None:
        search = self._grasp_orientation_search
        self._grasp_orientation_search = None
        if search is None or search['sequence_id'] != self.sequence_id:
            return
        best = search['best']
        if best is None:
            if (
                search['indeterminate'] > 0
                and search['definitive_rejections'] == 0
            ):
                self.get_logger().warning(
                    'Joint-aware grasp search had no complete result because '
                    f'{search["indeterminate"]} IK queries were unavailable or incomplete; '
                    'falling back to the original orientation and bounded planner.'
                )
                self._commit_grasp_plan_and_open(self.grasp_orientation)
                return
            self.reset_sequence(
                'No collision-aware IK branch reaches both pre-grasp and final '
                f'grasp with a continuous Cartesian insertion across '
                f'{len(search["candidates"])} equivalent wrist azimuths '
                f'({search["definitive_rejections"]} definitive rejections, '
                f'{search["indeterminate"]} unavailable checks). Rejecting '
                'this target before spending time on a doomed pre-grasp motion.'
            )
            return

        wrist = self.grasp_wrist_joint_name
        current_q = search['current'][wrist]
        pre_q = best['pre_joints'][wrist]
        grasp_q = best['grasp_joints'][wrist]
        self._grasp_pregrasp_joint_targets = {wrist: pre_q}
        self.get_logger().info(
            f'Joint-aware grasp selected azimuth '
            f'{math.degrees(best["angle"]):+.1f}°: {wrist} '
            f'{math.degrees(current_q):+.1f}° -> '
            f'{math.degrees(pre_q):+.1f}° -> '
            f'{math.degrees(grasp_q):+.1f}°, total wrist travel '
            f'{math.degrees(best["wrist_travel"]):.1f}°, '
            f'limit clearance {math.degrees(best["clearance"]):.1f}°.'
        )
        self._commit_grasp_plan_and_open(best['orientation'])

    def _commit_grasp_plan_and_open(self, orientation: Quaternion) -> None:
        """Commit one checked orientation, then begin physical grasp motion."""
        if self.current_target_point_base is None:
            self.reset_sequence('Target disappeared before grasp plan commit.')
            return
        self.grasp_orientation = orientation
        target = self.current_target_point_base
        self.update_contact_poses_from_target(target, orientation)

        # Retain the legacy optional lock for deployments that explicitly use
        # it. The joint-aware search below independently constrains joint6 to
        # the selected IK family.
        if self.lock_wrist_joint:
            self.sequence_wrist_value = self.current_joint_positions.get(
                self.lock_wrist_joint_name
            )
            if self.sequence_wrist_value is None:
                self.get_logger().warning(
                    f'Joint "{self.lock_wrist_joint_name}" not found in '
                    '/joint_states; legacy wrist lock disabled for this sequence.'
                )

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
        shortfall_mm = self._grasp_reach_shortfall_mm(
            self.contact_pose_to_link_pose(self.grasp_pose)
        )
        if shortfall_mm is not None and shortfall_mm > 0.0:
            self.reset_sequence(
                f'Grasp target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) is outside the arm '
                f'workspace: final-descent wrist point is {shortfall_mm:.0f}mm beyond max extension '
                f'(incl. {self.reach_guard_margin_m * 1000.0:.0f}mm margin). Not starting the sequence. '
                f'Move the rover closer to the target; acquisition will retry after the cooldown.'
            )
            return
        self.command_gripper_and_then(
            self.gripper_open,
            self.send_pre_grasp,
            stage_name='open_gripper',
            description='open before approach'
        )

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
        target = self.current_target_point_base
        self.update_contact_poses_from_target(target, orientation)
        if self._start_joint_aware_grasp_orientation_search(orientation):
            return
        self._commit_grasp_plan_and_open(orientation)

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
                            with_orientation=True,
                            preferred_joint_positions=self._grasp_pregrasp_joint_targets,
                            preferred_joint_tolerance=self.lock_wrist_joint_tolerance)
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

        # Freeze the freshest mask pose now: the camera is buried during the
        # descent and close, so this is the last reliable estimate of the
        # probe's centre and axes for the attach-time pose sync.
        self._grasp_time_object_pose = self.detected_object_pose
        self._grasp_time_object_R = (
            self._last_detected_object_rotation_base.copy()
            if self._last_detected_object_rotation_base is not None else None
        )
        self._grasp_time_tip_resolved = bool(self._object_axis_tip_resolved)

        self.get_logger().info('No preclose will be used. Gripper stays open during final approach; '
            'after reaching grasp pose it closes once using the refined four-bar geometry.')
        if self.grasp_pose is not None:
            shortfall_mm = self._grasp_reach_shortfall_mm(self.contact_pose_to_link_pose(self.grasp_pose))
            if shortfall_mm is not None and shortfall_mm > 0.0:
                self._halt_after_final_approach_failure(
                    f'Refined grasp point is {shortfall_mm:.0f}mm beyond the arm reach envelope '
                    f'(incl. {self.reach_guard_margin_m * 1000.0:.0f}mm margin); the final descent '
                    f'cannot succeed. Reposition the rover closer to the target.'
                )
                return
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
        self._send_cartesian_path(
            self._final_descent_waypoints(self.contact_pose_to_link_pose(self.grasp_pose))
        )

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
        self._probe_track_association_rejected = False
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
                    if self._probe_track_association_rejected:
                        self.get_logger().warning(
                            'Refinement ignored a non-matching probe fit; '
                            'the locked target remains unchanged.',
                            throttle_duration_sec=0.5,
                        )
                        return
                elif (
                    self.probe_track_enabled
                    and self._probe_track is not None
                    and (
                        self._now_sec() - self._probe_track.last_update_sec
                        <= self.probe_track_timeout_sec
                    )
                ):
                    self.get_logger().info(
                        'Refinement kept the locked shape track because this '
                        'frame could not resolve a reliable probe pose.',
                        throttle_duration_sec=0.5,
                    )
                    return

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
        # Each physical attempt is judged on its own evidence: votes and the
        # empty-close latch from a previous attempt must not carry over.
        self._reset_held_probe_evidence()
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
            # A deliberately accepted partial descent is a known, bounded offset
            # from the commanded pose -- allow exactly that much extra, or this
            # gate rejects the pose the node just chose on purpose.
            pos_tol = (
                self.final_grasp_pose_position_tolerance_m
                + float(self._accepted_descent_shortfall_m)
            )
            if (
                pos_err <= pos_tol
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
        if self.final_close_full_close:
            end_q = max(end_q, float(self.gripper_close))
            self.get_logger().info(
                f'Final close deliberately over-closes to full close q={end_q:.5f} '
                f'(computed contact q={float(self.computed_gripper_close):.5f}): the probe is expected '
                'to stop the jaws at its true width and stalled-contact feedback completes the close.')

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
        # Freeze the measured link pose while the probe is still at the grasp
        # location.  Attachment happens after the visual lift check, but its
        # object-to-link transform must combine the pre-lift object snapshot
        # with this pre-lift link pose—not with the link after an 80+ mm lift.
        self._grasp_close_link_pose = (
            self.get_current_link_pose_in_planning_frame()
        )

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
        """Move arm joints to a held-probe transport posture, gripper closed.

        The calibrated pick_home posture holds the probe directly over the
        chassis front, so a long attached probe can put the GOAL state itself
        in collision — MoveGroup then aborts within milliseconds (status 6)
        before OMPL ever runs.  Each transport candidate (pick_home first,
        then pick_home_alternative_joint_positions_flat) is therefore checked
        with /check_state_validity against the live planning scene including
        the attached probe mesh, and the first collision-free posture is
        commanded.  OMPL still plans the collision-aware path to it.
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

        candidates = [('pick_home', list(self.pick_home_joint_positions))]
        n = len(self.pick_home_joint_names)
        flat = self.pick_home_alternative_joint_positions_flat
        if len(flat) % n != 0:
            self.get_logger().warning(
                f'pick_home_alternative_joint_positions_flat length {len(flat)} is not a multiple '
                f'of {n}; ignoring the trailing values.')
        for i in range(0, len(flat) - n + 1, n):
            candidates.append((f'transport-alt-{i // n + 1}', [float(v) for v in flat[i:i + n]]))

        if not self.transport_goal_validity_check_enabled:
            self._send_transport_goal(candidates[0])
            return
        if not self.state_validity_client.service_is_ready():
            self.get_logger().warning(
                f'{self.state_validity_service_name} service unavailable; sending the pick_home '
                'transport goal without the held-probe goal-state pre-check.')
            self._send_transport_goal(candidates[0])
            return
        self._check_transport_goal_candidate(candidates, 0, self.sequence_id)

    def _send_transport_goal(self, candidate: Tuple[str, List[float]]) -> None:
        label, positions = candidate
        self.get_logger().info(f'Moving to transport posture "{label}" (gripper closed). '
            f'MoveGroup planning time={self.post_grasp_planning_time_sec:.1f} s, '
            f'live-joint seeded start state.')
        self.send_joint_goal(
            self.pick_home_joint_names,
            positions,
            planning_time_override=self.post_grasp_planning_time_sec,
            num_attempts_override=max(self.num_planning_attempts, 15),
        )

    def _check_transport_goal_candidate(
        self,
        candidates: List[Tuple[str, List[float]]],
        index: int,
        expected_seq: int,
    ) -> None:
        if expected_seq != self.sequence_id or self.sequence_stage != 'move_pick_home':
            return
        if index >= len(candidates):
            tried = ', '.join(label for label, _ in candidates)
            self._hold_closed_after_transport_failure(
                f'Every held-probe transport posture ({tried}) leaves the attached probe in collision '
                'with the rover. Extend pick_home_alternative_joint_positions_flat with a clear posture.'
            )
            return
        label, positions = candidates[index]
        req = GetStateValidity.Request()
        goal_state = RobotState()
        # Diff onto the live scene state: unlisted joints keep their current
        # values and the attached probe mesh stays attached for the check.
        goal_state.is_diff = True
        goal_js = JointState()
        goal_js.name = list(self.pick_home_joint_names)
        goal_js.position = [float(v) for v in positions]
        goal_state.joint_state = goal_js
        req.robot_state = goal_state
        req.group_name = self.planning_group
        future = self.state_validity_client.call_async(req)
        future.add_done_callback(
            lambda fut, c=candidates, i=index, seq=expected_seq: self._on_transport_goal_validity(fut, c, i, seq)
        )

    def _on_transport_goal_validity(
        self,
        future,
        candidates: List[Tuple[str, List[float]]],
        index: int,
        expected_seq: int,
    ) -> None:
        label, _ = candidates[index]
        try:
            resp = future.result()
        except Exception as exc:
            if expected_seq != self.sequence_id or self.sequence_stage != 'move_pick_home':
                return
            self.get_logger().warning(f'{self.state_validity_service_name} call failed for "{label}" ({exc}); '
                'sending the posture unchecked and letting MoveGroup decide.')
            self._send_transport_goal(candidates[index])
            return
        if expected_seq != self.sequence_id or self.sequence_stage != 'move_pick_home':
            return
        if resp.valid:
            if index > 0:
                self.get_logger().warning(
                    f'pick_home would leave the held probe in collision with the rover; using '
                    f'collision-free transport posture "{label}" instead.')
            self._send_transport_goal(candidates[index])
            return
        pairs = sorted({f'{c.contact_body_1}<->{c.contact_body_2}' for c in resp.contacts})
        detail = '; '.join(pairs[:4]) if pairs else 'no contact detail reported'
        self.get_logger().warning(f'Transport posture "{label}" is in collision ({detail}); trying next candidate.')
        self._check_transport_goal_candidate(candidates, index + 1, expected_seq)

    def send_base_box_drop_closed(self) -> None:
        """Calculate and execute a base-box release plan with the gripper closed."""
        use_pose = self.base_box_auto_drop_enabled or self.base_box_drop_use_pose
        if use_pose:
            if not self._base_box_drop_pose_config_valid():
                self._hold_closed_after_transport_failure(
                    'Invalid automatic base-box geometry or legacy drop pose.'
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
        if use_pose:
            try:
                self._base_box_drop_candidates = self._build_base_box_drop_candidates()
            except ValueError as exc:
                self._hold_closed_after_transport_failure(f'Cannot calculate base-box drop: {exc}')
                return
            self._base_box_drop_round = 0
            self._base_box_drop_position_only_active = False
            self._base_box_tip_fallback_active = False
            self._base_box_drop_start_collision_retry_index = -1
            self._base_box_drop_candidate_index = -1
            self.call_later(0.1, lambda: self._send_base_box_drop_candidate(0))
        else:
            self.get_logger().info(f'Moving held probe from pick_home to the base-box joint posture. '
                f'MoveGroup planning time={self.base_box_planning_time_sec:.1f} s.')
            self.send_joint_goal(
                self.base_box_drop_joint_names,
                self.base_box_drop_joint_positions,
                planning_time_override=self.base_box_planning_time_sec,
                num_attempts_override=max(self.num_planning_attempts, 15),
            )

    def _compute_base_box_layout(self) -> box_drop.BoxDropLayout:
        if len(self.base_box_rpy) != 3:
            raise ValueError('base_box_rpy must contain exactly three values')
        rotation = quat_to_matrix(rpy_to_quat(*self.base_box_rpy))
        probe_dims = self._probe_dims()
        settings = box_drop.derive_automatic_box_settings(
            self.base_box_dimensions_xyz,
            probe_length_m=float(probe_dims[2]),
            probe_width_m=float(max(probe_dims[0], probe_dims[1])),
        )
        return box_drop.compute_box_drop_layout(
            self.base_box_center_xyz,
            self.base_box_dimensions_xyz,
            rotation,
            settings,
        )

    def _make_base_box_drop_pose(self, point: np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.base_box_drop_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        pose.pose.orientation = rpy_to_quat(*self.base_box_drop_rpy)
        return pose

    def _align_drop_pose_to_attached_probe(
        self, pose: PoseStamped, probe_axis_yaw_rad: float
    ) -> Tuple[List[float], float]:
        target_offset = list(self.base_box_drop_target_point_offset_in_link)
        applied_yaw = 0.0
        if (
            self.base_box_drop_align_attached_probe
            and self._attached_probe_grasp_orientation is not None
            and self._attached_probe_world_yaw is not None
        ):
            # Yaw the proven grasp-time tool-down orientation so the probe lies
            # along the longest usable box axis. Probe ends are symmetric.
            applied_yaw = wrap_to_pi(probe_axis_yaw_rad - self._attached_probe_world_yaw)
            if applied_yaw > math.pi / 2.0:
                applied_yaw -= math.pi
            elif applied_yaw < -math.pi / 2.0:
                applied_yaw += math.pi
            cz, sz = math.cos(applied_yaw), math.sin(applied_yaw)
            world_z_spin = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            pose.pose.orientation = matrix_to_quat(
                world_z_spin @ quat_to_matrix(self._attached_probe_grasp_orientation)
            )
            if self._attached_probe_centre_in_link is not None:
                target_offset = [float(v) for v in self._attached_probe_centre_in_link]
        return target_offset, applied_yaw

    def _automatic_base_box_orientation_options(
        self,
        layout: box_drop.BoxDropLayout,
        desired_axis: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, Quaternion, float]]:
        """Search wrist rolls that put the probe along ``desired_axis``.

        Defaults to the box's long horizontal axis (the flat overhead drop).
        The tilted-insertion path passes the leaning axis instead.
        """
        if (
            self._attached_probe_axis_in_link is None
            or self._attached_probe_grasp_orientation is None
        ):
            raise ValueError('attached probe axis/orientation is unavailable')

        if desired_axis is None:
            axis_index = 0 if layout.probe_axis_name == 'X' else 1
            desired_axis = layout.rotation[:, axis_index]
        desired_axis = normalize(np.asarray(desired_axis, dtype=np.float64))
        grasp_R = quat_to_matrix(self._attached_probe_grasp_orientation)
        initial_axis = normalize(grasp_R @ self._attached_probe_axis_in_link)
        # Both axis signs are generated: they point opposite probe ends into the
        # box and land in different wrist IK families. The caller orders them by
        # wrist travel, so whichever end is already closer to leading goes first.
        options: List[Tuple[str, Quaternion, float]] = []
        orientation_tol = max(self.base_box_drop_orientation_tolerance_rad, 0.12)
        for axis_sign in (1.0, -1.0):
            signed_axis = axis_sign * desired_axis
            aligned_R = box_drop.rotation_aligning_vectors(initial_axis, signed_axis) @ grasp_R
            for roll_deg in (0.0, 90.0, -90.0, 180.0):
                roll = math.radians(roll_deg)
                x, y, z = signed_axis
                c, s = math.cos(roll), math.sin(roll)
                one_minus_c = 1.0 - c
                roll_about_probe = np.array([
                    [c + x*x*one_minus_c, x*y*one_minus_c - z*s, x*z*one_minus_c + y*s],
                    [y*x*one_minus_c + z*s, c + y*y*one_minus_c, y*z*one_minus_c - x*s],
                    [z*x*one_minus_c - y*s, z*y*one_minus_c + x*s, c + z*z*one_minus_c],
                ], dtype=np.float64)
                orientation = matrix_to_quat(roll_about_probe @ aligned_R)
                options.append((
                    f'box-{layout.probe_axis_name} axis {axis_sign:+.0f}, '
                    f'wrist roll {roll_deg:+.0f}deg',
                    orientation,
                    orientation_tol,
                ))
        return options

    def _wrist_travel_rad(self, orientation: Quaternion) -> float:
        """Rotation from the tool's current orientation to ``orientation``."""
        current = self.get_current_tool_orientation_in_planning_frame()
        if current is None:
            return 0.0
        return float(quaternion_distance_rad(current, orientation))

    def _attached_probe_tip_geometry(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return tapered-tip point and centre->tip axis in link coordinates."""
        if (
            self._attached_probe_centre_in_link is None
            or self._attached_probe_R_stl_in_link is None
        ):
            raise ValueError(
                'attached probe centre and signed STL orientation are unavailable'
            )
        if not self._attached_probe_tip_resolved:
            raise ValueError(
                'the tapered probe end has not been resolved confidently'
            )
        tip = box_drop.probe_tip_from_attached_geometry(
            self._attached_probe_centre_in_link,
            self._attached_probe_R_stl_in_link[:, 2],
            float(self._probe_dims()[2]),
            self._probe_stl_end_sign_or_default(),
        )
        return tip, normalize(tip - self._attached_probe_centre_in_link)

    def _tip_leading_orientation_options(
        self, desired_tip_axis: np.ndarray
    ) -> List[Tuple[str, Quaternion, float]]:
        """Orient the asymmetric probe with its tapered tip leading."""
        if self._attached_probe_grasp_orientation is None:
            raise ValueError('attached probe link orientation is unavailable')
        _, tip_axis_in_link = self._attached_probe_tip_geometry()
        desired_tip_axis = normalize(
            np.asarray(desired_tip_axis, dtype=np.float64)
        )
        reference_R = quat_to_matrix(self._attached_probe_grasp_orientation)
        reference_tip_axis = normalize(reference_R @ tip_axis_in_link)
        aligned_R = (
            box_drop.rotation_aligning_vectors(
                reference_tip_axis, desired_tip_axis
            )
            @ reference_R
        )
        x, y, z = desired_tip_axis
        options: List[Tuple[str, Quaternion, float]] = []
        orientation_tol = max(
            self.base_box_drop_orientation_tolerance_rad, 0.16
        )
        for roll_deg in (0.0, 90.0, -90.0, 180.0):
            roll = math.radians(roll_deg)
            c, s = math.cos(roll), math.sin(roll)
            one_minus_c = 1.0 - c
            roll_about_tip = np.array([
                [c + x*x*one_minus_c, x*y*one_minus_c - z*s, x*z*one_minus_c + y*s],
                [y*x*one_minus_c + z*s, c + y*y*one_minus_c, y*z*one_minus_c - x*s],
                [z*x*one_minus_c - y*s, z*y*one_minus_c + x*s, c + z*z*one_minus_c],
            ], dtype=np.float64)
            options.append((
                f'tapered tip leading, wrist roll {roll_deg:+.0f}deg',
                matrix_to_quat(roll_about_tip @ aligned_R),
                orientation_tol,
            ))
        return options

    def _build_base_box_insertion_candidates(self, layout: box_drop.BoxDropLayout) -> List[dict]:
        """Lean the probe into the box with one end below the rim.

        The probe does not fit flat through the usable opening, so this drives
        its tapered end inside at several diagonal tilts. Candidates cover
        both lean directions but never reverse the asymmetric probe: the fat,
        grasped end always stays outside. Geometrically impossible poses are
        dropped before planning.
        """
        target_offset = list(self.base_box_drop_target_point_offset_in_link)
        if self._attached_probe_centre_in_link is not None:
            target_offset = [float(v) for v in self._attached_probe_centre_in_link]

        scored: List[Tuple[float, dict]] = []
        skipped_no_clearance = 0
        for tilt_deg in self.base_box_insert_tilt_options_deg:
            for axis_sign in (1.0, -1.0):
                insertion = box_drop.compute_probe_insertion(
                    layout,
                    math.radians(float(tilt_deg)),
                    self.base_box_insert_depth_m,
                    axis_sign=axis_sign,
                    entry_offset_m=self.base_box_insert_entry_offset_m,
                )
                if not insertion.clears_opening:
                    skipped_no_clearance += 1
                    continue
                options = self._tip_leading_orientation_options(
                    -insertion.axis
                )
                for orientation_name, orientation, orientation_tol in options:
                    pose = self._make_base_box_drop_pose(insertion.probe_centre)
                    pose.pose.orientation = orientation
                    scored.append((
                        self._wrist_travel_rad(orientation),
                        {
                            'pose': pose,
                            'target_offset': target_offset,
                            'applied_yaw': 0.0,
                            'with_orientation': True,
                            'orientation_tol': orientation_tol,
                            'position_region_dimensions': None,
                            'position_region_orientation': None,
                            'display_volume_dimensions': [
                                float(v) for v in layout.release_volume_dimensions
                            ],
                            'insertion': insertion,
                            'tip_only': False,
                            'label': (
                                f'insertion tilt {tilt_deg:.0f}deg lean {axis_sign:+.0f} '
                                f'({orientation_name})'
                            ),
                        },
                    ))

        if not scored:
            raise ValueError(
                'no probe insertion pose clears the box opening; check '
                'base_box_insert_tilt_options_deg / base_box_insert_entry_offset_m'
            )

        # Least wrist travel first: that is the "closest end leads" choice.
        scored.sort(key=lambda item: item[0])
        candidates = [item[1] for item in scored]
        for index, candidate in enumerate(candidates):
            candidate['label'] = (
                f'insertion solution {index + 1}/{len(candidates)} — ' + candidate['label']
            )

        first = candidates[0]['insertion']
        self._computed_base_box_probe_axis_yaw_rad = layout.probe_axis_yaw_rad
        self.get_logger().info(
            f'[base-box-insert] Probe {layout.settings.probe_length_m*1000:.0f} mm does not fit the '
            f'{layout.usable_xy[0]*1000:.0f}x{layout.usable_xy[1]*1000:.0f} mm opening; leaning it in '
            f'instead. Leading end {first.depth_m*1000:.0f} mm below the rim, '
            f'tilt options={[f"{t:.0f}" for t in self.base_box_insert_tilt_options_deg]}deg, '
            f'{len(candidates)} candidates ordered by wrist travel'
            + (f', {skipped_no_clearance} rejected for wall clearance.' if skipped_no_clearance else '.')
        )
        return candidates

    def _build_base_box_tip_insertion_candidates(
        self, layout: box_drop.BoxDropLayout
    ) -> List[dict]:
        """Build the safe last resort: put only the tapered tip in the box.

        The normal insertion goal constrains the probe centre.  When that pose
        is unreachable, relaxing it can move the wrist and fingers toward the
        box interior.  This fallback instead constrains the tapered endpoint
        just below the rim.  Since the probe extends from that point upward and
        outward toward its fat, grasped end, the gripper remains outside.
        """
        if (
            self._attached_probe_centre_in_link is None
            or self._attached_probe_R_stl_in_link is None
            or self._attached_probe_grasp_orientation is None
        ):
            raise ValueError(
                'attached probe centre, signed STL axis, and link orientation '
                'are required for a tip-only insertion'
            )

        tip_in_link, _ = self._attached_probe_tip_geometry()

        # Twenty millimetres is enough to put the endpoint decisively below
        # the rim while preserving substantially more gripper clearance than
        # the normal 50 mm centre-insertion target.
        tip_depth_m = min(0.020, max(0.008, self.base_box_insert_depth_m))
        scored: List[Tuple[float, dict]] = []
        skipped_no_clearance = 0
        for tilt_deg in self.base_box_insert_tilt_options_deg:
            for axis_sign in (1.0, -1.0):
                insertion = box_drop.compute_probe_insertion(
                    layout,
                    math.radians(float(tilt_deg)),
                    tip_depth_m,
                    axis_sign=axis_sign,
                    entry_offset_m=self.base_box_insert_entry_offset_m,
                )
                if not insertion.clears_opening:
                    skipped_no_clearance += 1
                    continue

                # ProbeInsertion.axis points from the endpoint in the box
                # toward the endpoint outside it.  The centre->tip direction
                # therefore has to point the other way.
                for orientation_name, orientation, orientation_tol in (
                    self._tip_leading_orientation_options(-insertion.axis)
                ):
                    pose = self._make_base_box_drop_pose(insertion.leading_end)
                    pose.pose.orientation = orientation
                    scored.append((
                        self._wrist_travel_rad(orientation),
                        {
                            'pose': pose,
                            'target_offset': [float(v) for v in tip_in_link],
                            'applied_yaw': 0.0,
                            'with_orientation': True,
                            'orientation_tol': orientation_tol,
                            'position_region_dimensions': None,
                            'position_region_orientation': None,
                            'display_volume_dimensions': None,
                            'insertion': insertion,
                            'tip_only': True,
                            'planning_time_sec': min(
                                3.0, self.base_box_planning_time_sec
                            ),
                            'label': (
                                f'tip-only insertion tilt {tilt_deg:.0f}deg '
                                f'lean {axis_sign:+.0f}, {orientation_name}'
                            ),
                        },
                    ))

        if not scored:
            raise ValueError(
                'no tip-only insertion pose clears the box opening; check '
                'base_box_insert_tilt_options_deg / base_box_insert_entry_offset_m'
            )
        scored.sort(key=lambda item: item[0])
        candidates = [item[1] for item in scored]
        for index, candidate in enumerate(candidates):
            candidate['label'] = (
                f'tip fallback {index + 1}/{len(candidates)} — '
                + candidate['label']
            )
        self.get_logger().warning(
            f'[base-box-tip] Switching to {len(candidates)} tip-only candidates: '
            f'the tapered endpoint will be {tip_depth_m*1000:.0f} mm below the '
            f'rim while the probe centre and gripper stay outside'
            + (
                f'; {skipped_no_clearance} candidates rejected for wall clearance.'
                if skipped_no_clearance else '.'
            )
        )
        return candidates

    def _start_base_box_tip_fallback(self, reason: str) -> bool:
        """Replace failed centre-insertion goals with bounded tip-only goals."""
        if (
            not self.base_box_auto_drop_enabled
            or not self.base_box_insert_enabled
            or self._base_box_tip_fallback_active
        ):
            return False
        try:
            candidates = self._build_base_box_tip_insertion_candidates(
                self._compute_base_box_layout()
            )
        except ValueError as exc:
            self.get_logger().error(
                f'Cannot build the safe tip-only base-box fallback: {exc}'
            )
            return False

        self._base_box_tip_fallback_active = True
        self._base_box_drop_position_only_active = False
        self._base_box_drop_candidates = candidates
        self._base_box_drop_candidate_index = -1
        self._base_box_drop_start_collision_retry_index = -1
        self._base_box_ik_screen_exhausted = False
        self._base_box_drop_ik_skipped = 0
        self._base_box_ik_reject_codes = {}
        self.get_logger().warning(
            f'{reason} Trying the safe tip-only fallback instead of planning '
            f'known-unreachable probe-centre poses or moving the gripper into the box.'
        )
        self.call_later(0.2, lambda: self._send_base_box_drop_candidate(0))
        return True

    def _build_base_box_drop_candidates(self) -> List[dict]:
        if self.base_box_auto_drop_enabled:
            layout = self._compute_base_box_layout()
            if not layout.probe_length_fits:
                self.get_logger().warning(
                    f'Probe length {layout.settings.probe_length_m*1000:.0f} mm exceeds the '
                    f'{max(layout.usable_xy)*1000:.0f} mm usable box axis. This is allowed: the robot '
                    'will centre it lengthwise over that axis with symmetric overhang.')
            if not layout.probe_width_fits:
                self.get_logger().warning(
                    f'Probe width {layout.settings.probe_width_m*1000:.0f} mm exceeds the narrow usable '
                    'opening. This is allowed for the overhead drop; no insertion pose is required.')
            if self.base_box_insert_enabled and not layout.probe_length_fits:
                return self._build_base_box_insertion_candidates(layout)

            orientation_options = self._automatic_base_box_orientation_options(layout)
            if not orientation_options:
                raise ValueError('no automatic wrist orientations could be generated')
            self._computed_base_box_probe_axis_yaw_rad = layout.probe_axis_yaw_rad
            self.get_logger().info(
                f'Automatic base-box plan: frame={self.base_box_drop_frame}, '
                f'centre=({layout.center[0]:.3f},{layout.center[1]:.3f},{layout.center[2]:.3f}), '
                f'size=({layout.dimensions[0]:.3f},{layout.dimensions[1]:.3f},{layout.dimensions[2]:.3f})m, '
                f'top_z={layout.top_center[2]:.3f}, release_volume='
                f'({layout.release_volume_dimensions[0]:.3f}x{layout.release_volume_dimensions[1]:.3f}x'
                f'{layout.release_volume_dimensions[2]:.3f})m, inferred_wall='
                f'{layout.settings.wall_thickness_m*1000:.0f}mm, automatic_wrist_options='
                f'{len(orientation_options)}, required_probe_axis=box-{layout.probe_axis_name} '
                f'({math.degrees(layout.probe_axis_yaw_rad):.1f}deg).')
            target_offset = list(self.base_box_drop_target_point_offset_in_link)
            if self._attached_probe_centre_in_link is not None:
                target_offset = [float(v) for v in self._attached_probe_centre_in_link]
            candidates = []
            for index, (orientation_name, orientation, orientation_tol) in enumerate(orientation_options):
                pose = self._make_base_box_drop_pose(layout.release_volume_center)
                pose.pose.orientation = orientation
                candidates.append({
                    'pose': pose,
                    'target_offset': target_offset,
                    'applied_yaw': 0.0,
                    'with_orientation': True,
                    'orientation_tol': orientation_tol,
                    # Use the standard spherical point constraint proven by
                    # grasp planning. The box-shaped, position-only endpoint
                    # was rejected by MoveIt as INVALID_MOTION_PLAN (-2).
                    'position_region_dimensions': None,
                    'position_region_orientation': None,
                    'display_volume_dimensions': [float(v) for v in layout.release_volume_dimensions],
                    'label': (
                        f'central release solution {index + 1}/{len(orientation_options)} '
                        f'({orientation_name})'
                    ),
                    'tip_only': False,
                })
            return candidates
        else:
            points = (np.asarray(self.base_box_drop_xyz, dtype=np.float64),)
            axis_yaw = self.base_box_drop_probe_axis_world_yaw_rad
            self._computed_base_box_probe_axis_yaw_rad = axis_yaw

        candidates: List[dict] = []
        for index, point in enumerate(points):
            pose = self._make_base_box_drop_pose(point)
            target_offset, applied_yaw = self._align_drop_pose_to_attached_probe(pose, axis_yaw)
            candidates.append({
                'pose': pose,
                'target_offset': target_offset,
                'applied_yaw': applied_yaw,
                'with_orientation': True,
                'orientation_tol': self.base_box_drop_orientation_tolerance_rad,
                'position_region_dimensions': None,
                'position_region_orientation': None,
                'display_volume_dimensions': None,
                'label': f'candidate {index + 1}/{len(points)}',
                'tip_only': False,
            })
        return candidates

    def _candidate_link_pose(self, candidate: dict) -> PoseStamped:
        """The planning-link pose implied by a candidate's probe-centre goal.

        Candidate poses target the probe centre via the constraint's
        ``target_point_offset``; IK needs the link itself, so undo the offset.
        """
        pose = candidate['pose']
        offset = np.asarray(candidate['target_offset'], dtype=np.float64).reshape(3,)
        R = quat_to_matrix(pose.pose.orientation)
        link_pose = PoseStamped()
        link_pose.header.frame_id = pose.header.frame_id
        link_pose.header.stamp = self.get_clock().now().to_msg()
        position = np.array([
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        ]) - R @ offset
        link_pose.pose.position = Point(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        )
        link_pose.pose.orientation = pose.pose.orientation
        return link_pose

    def _send_base_box_drop_candidate(self, index: int) -> None:
        """Screen the candidate with IK, then plan it.

        A collision-aware IK query costs milliseconds; letting MoveIt discover
        the same unreachable goal costs a full planning timeout (measured at
        10.2 s per candidate). The query is async because this node runs on a
        single-threaded executor that must never block.
        """
        if index < 0 or index >= len(self._base_box_drop_candidates):
            self._hold_closed_after_transport_failure('No valid base-box release solution remains.')
            return
        if (
            not self.base_box_ik_prescreen_enabled
            or self._base_box_ik_screen_exhausted
            or not self.compute_ik_client.service_is_ready()
        ):
            self._dispatch_base_box_drop_candidate(index)
            return

        candidate = self._base_box_drop_candidates[index]
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.planning_group
        req.ik_request.ik_link_name = self.planning_link
        req.ik_request.pose_stamped = self._candidate_link_pose(candidate)
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = Duration(
            sec=int(self.base_box_ik_prescreen_timeout_sec),
            nanosec=int((self.base_box_ik_prescreen_timeout_sec % 1.0) * 1e9),
        )
        seed_state = self._make_current_robot_state()
        if seed_state is not None:
            req.ik_request.robot_state = seed_state

        future = self.compute_ik_client.call_async(req)
        self._base_box_ik_request_token += 1
        request_token = self._base_box_ik_request_token
        future.add_done_callback(
            lambda fut, i=index, seq=self.sequence_id, stage=self.sequence_stage,
                token=request_token:
                self._on_base_box_drop_ik(fut, i, seq, stage, token)
        )
        self.call_later(
            self.base_box_ik_prescreen_timeout_sec + 0.35,
            lambda i=index, seq=self.sequence_id, stage=self.sequence_stage,
                token=request_token:
                self._on_base_box_drop_ik_timeout(i, seq, stage, token),
        )

    def _on_base_box_drop_ik_timeout(
        self,
        index: int,
        expected_seq: int,
        expected_stage: str,
        request_token: int,
    ) -> None:
        if (
            request_token != self._base_box_ik_request_token
            or expected_seq != self.sequence_id
            or self.sequence_stage != expected_stage
        ):
            return
        self._base_box_ik_request_token += 1
        self.get_logger().warning(
            f'Base-box IK pre-screen callback timed out for candidate '
            f'{index + 1}; handing it to the bounded planner instead of '
            f'leaving the task stuck.'
        )
        self._dispatch_base_box_drop_candidate(index)

    def _on_base_box_drop_ik(
        self,
        future,
        index: int,
        expected_seq: int,
        expected_stage: str,
        request_token: int,
    ) -> None:
        if (
            request_token != self._base_box_ik_request_token
            or expected_seq != self.sequence_id
            or self.sequence_stage != expected_stage
        ):
            return
        self._base_box_ik_request_token += 1
        try:
            result = future.result() if future.done() else None
        except Exception as exc:
            # A service transport failure is not evidence that the pose is
            # unreachable.  Preserve the documented optimisation-only
            # semantics and hand this candidate to the planner.
            self.get_logger().warning(
                f'Base-box IK pre-screen failed to return a result ({exc}); '
                f'planning candidate {index + 1} directly.'
            )
            self._dispatch_base_box_drop_candidate(index)
            return
        # Only a definitive "no IK solution" rejects a candidate. A missing or
        # errored response must fall through to planning, never skip.
        if (
            result is not None
            and int(result.error_code.val) == MoveItErrorCodes.NO_IK_SOLUTION
        ):
            code = int(result.error_code.val)
            self._base_box_drop_ik_skipped += 1
            self._base_box_ik_reject_codes[code] = (
                self._base_box_ik_reject_codes.get(code, 0) + 1
            )
            next_index = index + 1
            if next_index >= len(self._base_box_drop_candidates):
                insertion_round = any(
                    cand.get('insertion') is not None
                    for cand in self._base_box_drop_candidates
                )
                rejection = (
                    f'IK pre-screen rejected all '
                    f'{len(self._base_box_drop_candidates)} release candidates '
                    f'({self._describe_ik_rejections()}).'
                )
                if insertion_round and self._start_base_box_tip_fallback(rejection):
                    return
                if self._base_box_tip_fallback_active:
                    self._hold_closed_after_transport_failure(
                        f'{rejection} No collision-free tip-only box insertion '
                        f'exists; stopping without moving the gripper into the box.'
                    )
                    return
                # The screen is an optimisation, never a veto. One IK seed
                # failing does not prove the goal unplannable — MoveIt searches
                # differently and can reach configurations a single query
                # misses. Rather than give up without commanding any motion,
                # disable screening and actually plan the ordered candidates.
                total = len(self._base_box_drop_candidates)
                keep = max(1, int(self.base_box_max_planned_candidates))
                # Candidates are ordered by wrist travel, so the head of the
                # list is the most promising. Planning all 48 at the full
                # timeout would take minutes; cap it to a bounded sweep.
                self._base_box_drop_candidates = self._base_box_drop_candidates[:keep]
                self.get_logger().warning(
                    f'IK pre-screen rejected all {total} release candidates '
                    f'({self._describe_ik_rejections()}). The screen only skips planning, it '
                    f'does not decide reachability, so the {keep} closest by wrist travel will '
                    f'be planned anyway (up to '
                    f'{keep * self.base_box_planning_time_sec:.0f} s).')
                self._base_box_ik_screen_exhausted = True
                self._base_box_drop_candidate_index = -1
                self._send_base_box_drop_candidate(0)
                return
            self.get_logger().info(
                f'[base-box-insert] IK pre-screen rejected release solution '
                f'{index + 1}/{len(self._base_box_drop_candidates)} '
                f'({self._moveit_error_name(code)}) in place of a '
                f'{self.base_box_planning_time_sec:.0f} s planning timeout; trying '
                f'{next_index + 1}.')
            self._send_base_box_drop_candidate(next_index)
            return
        self._dispatch_base_box_drop_candidate(index)

    @staticmethod
    def _moveit_error_name(code: int) -> str:
        """Readable MoveIt error code; the number alone hides the cause."""
        names = {
            MoveItErrorCodes.SUCCESS: 'SUCCESS',
            MoveItErrorCodes.PLANNING_FAILED: 'PLANNING_FAILED',
            MoveItErrorCodes.INVALID_MOTION_PLAN: 'INVALID_MOTION_PLAN',
            MoveItErrorCodes.TIMED_OUT: 'TIMED_OUT',
            MoveItErrorCodes.START_STATE_IN_COLLISION: 'START_STATE_IN_COLLISION',
            MoveItErrorCodes.GOAL_IN_COLLISION: 'GOAL_IN_COLLISION',
            MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
            MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: 'GOAL_CONSTRAINTS_VIOLATED',
            MoveItErrorCodes.NO_IK_SOLUTION: 'NO_IK_SOLUTION',
            MoveItErrorCodes.FAILURE: 'FAILURE',
        }
        return names.get(int(code), f'code={int(code)}')

    def _describe_ik_rejections(self) -> str:
        if not self._base_box_ik_reject_codes:
            return 'no reasons recorded'
        return ', '.join(
            f'{self._moveit_error_name(code)} x{count}'
            for code, count in sorted(
                self._base_box_ik_reject_codes.items(), key=lambda kv: -kv[1]
            )
        )

    def _dispatch_base_box_drop_candidate(self, index: int) -> None:
        if index < 0 or index >= len(self._base_box_drop_candidates):
            self._hold_closed_after_transport_failure('No valid base-box release solution remains.')
            return
        self._base_box_drop_candidate_index = index
        candidate = self._base_box_drop_candidates[index]
        if candidate.get('insertion') is not None:
            if candidate.get('tip_only', False):
                if (
                    self._base_box_tip_plans_attempted
                    >= self.base_box_max_planned_candidates
                ):
                    self._hold_closed_after_transport_failure(
                        f'The bounded tip-only box fallback used '
                        f'{self._base_box_tip_plans_attempted} planner attempts '
                        f'without a solution. Stopping with the gripper closed.'
                    )
                    return
                self._base_box_tip_plans_attempted += 1
            else:
                if (
                    self._base_box_center_plans_attempted
                    >= self.base_box_max_planned_candidates
                ):
                    reason = (
                        f'The bounded probe-centre insertion search used '
                        f'{self._base_box_center_plans_attempted} planner attempts '
                        f'without a solution.'
                    )
                    if self._start_base_box_tip_fallback(reason):
                        return
                    self._hold_closed_after_transport_failure(reason)
                    return
                self._base_box_center_plans_attempted += 1
        pose = candidate['pose']
        pose.header.stamp = self.get_clock().now().to_msg()
        self._active_base_box_drop_pose = pose
        self.publish_markers()
        if candidate.get('display_volume_dimensions') is not None:
            size = candidate['display_volume_dimensions']
            detail = (
                f'central-target=({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},'
                f'{pose.pose.position.z:.3f}), marker-zone=({size[0]:.3f},{size[1]:.3f},'
                f'{size[2]:.3f})m, target-radius={self.base_box_drop_position_tolerance_m:.3f}m, '
                f'probe-axis={math.degrees(self._computed_base_box_probe_axis_yaw_rad or 0.0):.1f}deg, '
                f'automatic wrist tolerance='
                f'{math.degrees(candidate["orientation_tol"]):.1f}deg'
            )
        elif candidate['with_orientation']:
            detail = (
                f'point=({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},{pose.pose.position.z:.3f}), '
                f'probe_axis_yaw={math.degrees(self._computed_base_box_probe_axis_yaw_rad or 0.0):.1f}deg, '
                f'applied_tool_yaw={math.degrees(candidate["applied_yaw"]):.1f}deg'
            )
        else:
            detail = 'orientation unconstrained'
        self.get_logger().info(
            f'Moving held probe to base-box {candidate["label"]}: {detail}.')
        planning_time = float(
            candidate.get('planning_time_sec', self.base_box_planning_time_sec)
        )
        self.send_pose_goal(
            pose,
            pos_tol=self.base_box_drop_position_tolerance_m,
            with_orientation=candidate['with_orientation'],
            orientation_override=pose.pose.orientation if candidate['with_orientation'] else None,
            orientation_tol=candidate['orientation_tol'],
            target_point_offset=candidate['target_offset'],
            position_region_dimensions=candidate['position_region_dimensions'],
            position_region_orientation=candidate['position_region_orientation'],
            arm_joints_only_start_state=True,
            planning_time_override=planning_time,
            num_attempts_override=max(self.num_planning_attempts, 15),
        )

    def _try_next_base_box_drop_candidate(
        self, reason: str, moveit_code: Optional[int] = None
    ) -> bool:
        if not self.base_box_auto_drop_enabled:
            return False
        next_index = self._base_box_drop_candidate_index + 1
        if next_index >= len(self._base_box_drop_candidates):
            return self._start_next_base_box_drop_round(reason)
        self.get_logger().warning(
            f'{reason} Automatically trying the next base-box release solution '
            f'({next_index + 1}/{len(self._base_box_drop_candidates)}).')
        self._send_base_box_drop_candidate(next_index)
        return True

    def _start_next_base_box_drop_round(self, reason: str) -> bool:
        """Escalate the release strategy instead of locking after one sweep.

        Round 2 rebuilds every wrist candidate from the CURRENT attached-probe
        geometry — the held-probe re-alignment may have corrected the mesh
        while round 1 was failing — with the relaxed orientation tolerance.
        The final round plans the probe centre into the release volume with
        the wrist orientation unconstrained; release verification then applies
        the relaxed axis tolerance before opening.
        """
        # In insertion mode, a failed probe-centre sweep must never be relaxed
        # into a centre/position-only goal: that can pull the gripper into the
        # box.  Switch once to the endpoint-constrained fallback, then stop
        # closed if even that bounded strategy cannot be planned.
        insertion_round = any(
            cand.get('insertion') is not None
            for cand in self._base_box_drop_candidates
        )
        if insertion_round or self._base_box_tip_fallback_active:
            if not self._base_box_tip_fallback_active:
                return self._start_base_box_tip_fallback(reason)
            return False

        next_round = self._base_box_drop_round + 1
        if next_round == 1:
            try:
                candidates = self._build_base_box_drop_candidates()
            except ValueError as exc:
                self.get_logger().error(f'Cannot rebuild base-box candidates for the relaxed round: {exc}')
                return False
            relaxed_tol = max(
                self.base_box_drop_relaxed_orientation_tolerance_rad,
                self.base_box_drop_orientation_tolerance_rad,
            )
            for cand in candidates:
                cand['orientation_tol'] = relaxed_tol
                cand['label'] += ' [relaxed orientation]'
        elif next_round == 2 and self.base_box_drop_final_round_position_only:
            try:
                layout = self._compute_base_box_layout()
            except ValueError as exc:
                self.get_logger().error(f'Cannot compute base-box layout for the position-only round: {exc}')
                return False
            target_offset = list(self.base_box_drop_target_point_offset_in_link)
            if self._attached_probe_centre_in_link is not None:
                target_offset = [float(v) for v in self._attached_probe_centre_in_link]
            pose = self._make_base_box_drop_pose(layout.release_volume_center)
            candidates = [{
                'pose': pose,
                'target_offset': target_offset,
                'applied_yaw': 0.0,
                'with_orientation': False,
                'orientation_tol': math.pi,
                'position_region_dimensions': None,
                'position_region_orientation': None,
                'display_volume_dimensions': [float(v) for v in layout.release_volume_dimensions],
                'label': 'position-only last resort',
            }]
            self._base_box_drop_position_only_active = True
        else:
            return False

        self._base_box_drop_round = next_round
        self._base_box_drop_candidates = candidates
        self._base_box_drop_candidate_index = -1
        self._base_box_drop_start_collision_retry_index = -1
        self.get_logger().warning(
            f'{reason} All release solutions in the previous round failed; escalating to '
            f'round {next_round + 1} with {len(candidates)} candidate(s) '
            f'({"position-only" if self._base_box_drop_position_only_active else "relaxed orientation"}).')
        self.call_later(0.1, lambda: self._send_base_box_drop_candidate(0))
        return True

    def _base_box_drop_pose_config_valid(self) -> bool:
        if not self.base_box_drop_frame:
            return False
        if self.base_box_auto_drop_enabled:
            try:
                self._compute_base_box_layout()
                return True
            except ValueError:
                return False
        return len(self.base_box_drop_xyz) == 3 and len(self.base_box_drop_rpy) == 3

    def get_base_box_drop_pose(self) -> PoseStamped:
        """Return the active or nominal calculated release pose for markers."""
        if self._active_base_box_drop_pose is not None:
            return self._active_base_box_drop_pose
        if self.base_box_auto_drop_enabled:
            layout = self._compute_base_box_layout()
            return self._make_base_box_drop_pose(layout.release_volume_center)
        return self._make_base_box_drop_pose(np.asarray(self.base_box_drop_xyz, dtype=np.float64))

    def release_probe_in_base_box(self) -> None:
        """Open only after MoveIt confirms that the drop posture was reached."""
        self.get_logger().info('Base-box drop posture reached; releasing the probe.')
        self.command_gripper_and_then(
            self.gripper_open,
            self._after_probe_released_in_base_box,
            stage_name='release_in_base_box',
            description='release probe in rover base box',
        )

    def _verify_base_box_release_geometry(self) -> Tuple[bool, str]:
        """Verify the held probe centre and rigid long axis before opening."""
        if not self.base_box_auto_drop_enabled:
            return True, 'legacy placement mode'
        if self.base_box_drop_frame != self.planning_frame:
            return False, (
                f'automatic release verification requires base_box_drop_frame '
                f'({self.base_box_drop_frame}) to equal planning_frame ({self.planning_frame})'
            )
        if self._attached_probe_centre_in_link is None or self._attached_probe_axis_in_link is None:
            return False, 'attached probe centre/axis geometry is unavailable'
        current = self.get_current_link_pose_in_planning_frame()
        if current is None:
            return False, 'current planning-link TF is unavailable'

        layout = self._compute_base_box_layout()
        link_R = quat_to_matrix(current.orientation)
        link_xyz = np.array([
            float(current.position.x),
            float(current.position.y),
            float(current.position.z),
        ])
        probe_center = link_xyz + link_R @ self._attached_probe_centre_in_link
        probe_axis = normalize(link_R @ self._attached_probe_axis_in_link)
        active_candidate = (
            self._base_box_drop_candidates[self._base_box_drop_candidate_index]
            if 0 <= self._base_box_drop_candidate_index
                < len(self._base_box_drop_candidates)
            else {}
        )
        if active_candidate.get('insertion') is not None:
            insertion = active_candidate['insertion']
            tip_offset, tip_axis_in_link = self._attached_probe_tip_geometry()
            tip_world = link_xyz + link_R @ tip_offset
            tip_axis_world = normalize(link_R @ tip_axis_in_link)
            desired_tip_axis = -normalize(insertion.axis)
            tip_axis_error = math.acos(float(np.clip(
                np.dot(tip_axis_world, desired_tip_axis), -1.0, 1.0
            )))
            tip_local = layout.rotation.T @ (tip_world - layout.top_center)
            centre_local = layout.rotation.T @ (
                probe_center - layout.top_center
            )
            link_local = layout.rotation.T @ (link_xyz - layout.top_center)
            contact_offset = np.asarray(
                self.effective_target_point_offset_in_link, dtype=np.float64
            )
            contact_world = link_xyz + link_R @ contact_offset
            contact_local = layout.rotation.T @ (
                contact_world - layout.top_center
            )
            bucket_tip_offset = np.array(
                [0.0, 0.0, self.fourbar_bucket_tip_z_max_m],
                dtype=np.float64,
            )
            bucket_tip_world = link_xyz + link_R @ bucket_tip_offset
            bucket_tip_local = layout.rotation.T @ (
                bucket_tip_world - layout.top_center
            )
            usable_half = 0.5 * np.maximum(layout.usable_xy, 0.0)
            xy_margin = 0.003
            tip_inside_xy = bool(
                np.all(np.abs(tip_local[:2]) <= usable_half + xy_margin)
            )
            interior_depth = max(
                0.0,
                float(layout.dimensions[2])
                - float(layout.settings.wall_thickness_m),
            )
            tip_below_rim = bool(
                -interior_depth <= tip_local[2] <= -0.002
            )
            tip_aligned = bool(
                tip_axis_error
                <= max(
                    float(active_candidate['orientation_tol']),
                    math.radians(3.0),
                )
            )
            # These checks are the key safety property of insertion: only the
            # tapered endpoint enters.  The probe centre, grasp contact,
            # representative furthest-forward bucket point, and planning-link
            # origin all stay above the rim.  The base-link collision mesh
            # supplies the full bucket/wall collision check during planning.
            centre_min_height = (
                0.005 if active_candidate.get('tip_only', False) else -0.010
            )
            centre_outside = bool(centre_local[2] >= centre_min_height)
            link_outside = bool(link_local[2] >= 0.0)
            gripper_clearance = 0.005
            contact_outside = bool(contact_local[2] >= gripper_clearance)
            bucket_outside = bool(bucket_tip_local[2] >= gripper_clearance)
            detail = (
                f'tip_local=({tip_local[0]*1000:.1f},'
                f'{tip_local[1]*1000:.1f},{tip_local[2]*1000:.1f})mm, '
                f'allowed_tip_xy=({(usable_half[0]+xy_margin)*1000:.1f},'
                f'{(usable_half[1]+xy_margin)*1000:.1f})mm, '
                f'tip_axis_error={math.degrees(tip_axis_error):.1f}deg, '
                f'probe_centre_above_rim={centre_local[2]*1000:.1f}mm, '
                f'grasp_contact_above_rim={contact_local[2]*1000:.1f}mm, '
                f'bucket_tip_above_rim={bucket_tip_local[2]*1000:.1f}mm, '
                f'planning_link_above_rim={link_local[2]*1000:.1f}mm'
            )
            return (
                tip_inside_xy
                and tip_below_rim
                and tip_aligned
                and centre_outside
                and contact_outside
                and bucket_outside
                and link_outside
            ), detail

        axis_index = 0 if layout.probe_axis_name == 'X' else 1
        desired_axis = normalize(layout.rotation[:, axis_index])
        axis_error = math.acos(float(np.clip(abs(np.dot(probe_axis, desired_axis)), 0.0, 1.0)))

        local_delta = layout.rotation.T @ (probe_center - layout.release_volume_center)
        half_xy = 0.5 * layout.release_volume_dimensions[:2]
        xy_margin = 0.005
        centered = bool(np.all(np.abs(local_delta[:2]) <= half_xy + xy_margin))
        # The position-only last-resort round leaves the wrist unconstrained,
        # so the achieved probe axis can sit far from the box's long axis; a
        # centred overhead drop still lands in/over the box with symmetric
        # overhang, so only the relaxed axis bound applies there.
        axis_tol_deg = (
            self.base_box_release_axis_tolerance_final_deg
            if self._base_box_drop_position_only_active
            else self.base_box_release_axis_tolerance_deg
        )
        aligned = axis_error <= math.radians(axis_tol_deg)
        detail = (
            f'probe_center=({probe_center[0]:.3f},{probe_center[1]:.3f},{probe_center[2]:.3f}), '
            f'local_xy_error=({local_delta[0]*1000:.1f},{local_delta[1]*1000:.1f})mm, '
            f'allowed_xy=({(half_xy[0]+xy_margin)*1000:.1f},'
            f'{(half_xy[1]+xy_margin)*1000:.1f})mm, probe_axis_error='
            f'{math.degrees(axis_error):.1f}deg/{axis_tol_deg:.1f}deg'
        )
        return centered and aligned, detail

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
        """Back out of the grasp and rise far enough to see if the probe came too.

        The extraction retraces the final descent backwards in the gripper
        frame (see _final_ascent_waypoints) rather than translating straight up
        in the world: pulling a coaxially gripped probe out of its socket
        sideways levers it against the hole. Only once the arm is back at the
        standoff does a world-vertical segment top the motion up to
        lift_check_distance_m, which the visual lift check needs to tell a
        lifted probe from one still on the floor.
        """
        if self.grasp_pose is None:
            self.send_retreat()
            return

        grasp_xyz = self._pose_xyz(self.grasp_pose)
        lift_xyz = grasp_xyz.copy()
        lift_xyz[2] += self.lift_check_distance_m
        lift_pose = self.make_pose(lift_xyz, self.grasp_orientation)

        if self.locked_target_before_lift is None:
            self.locked_target_before_lift = grasp_xyz.copy()

        waypoints: List[Pose] = []
        if self.post_grasp_retrace_descent:
            back = self._final_ascent_waypoints()
            if back:
                waypoints = list(back)
                start = self.get_current_link_pose_in_planning_frame()
                risen_mm = (
                    (waypoints[-1].position.z - start.position.z) * 1000.0
                    if start is not None else 0.0
                )
                self.get_logger().info(f'Post-grasp extraction retraces the final descent in reverse: '
                    f'{len(waypoints)} gripper-frame waypoint(s) back to the standoff '
                    f'({risen_mm:.1f} mm of vertical rise), instead of prying the probe '
                    'sideways with a world-vertical lift.')

        # Top up only to the rise the visual check actually needs, not to the
        # full lift_check_distance_m. The check calls it lifted at
        # grasp_success_min_lift_m, so asking for 200 mm when 55 mm settles it
        # appends a long world-vertical segment that the arm may not be able to
        # follow: a 0.86 Cartesian fraction was seen aborting the whole task
        # while the probe was already gripped. Any residual is still requested
        # straight up, but only for what is missing.
        goal_link = self.contact_pose_to_link_pose(lift_pose)
        if not waypoints:
            waypoints.append(goal_link)
        else:
            start = self.get_current_link_pose_in_planning_frame()
            risen = (
                float(waypoints[-1].position.z - start.position.z)
                if start is not None else 0.0
            )
            needed = min(
                float(self.lift_check_distance_m),
                float(self.grasp_success_min_lift_m) + float(self.post_grasp_retrace_extra_rise_m),
            )
            if risen < needed:
                top = Pose()
                top.position = Point(
                    x=float(waypoints[-1].position.x),
                    y=float(waypoints[-1].position.y),
                    z=float(waypoints[-1].position.z + (needed - risen)),
                )
                top.orientation = waypoints[-1].orientation
                waypoints.append(top)
                self.get_logger().info(f'Retrace rose {risen*1000:.1f} mm, short of the '
                    f'{needed*1000:.1f} mm the lift check needs; adding a {(needed-risen)*1000:.1f} mm '
                    'vertical top-up (the probe is clear of its socket by then).')
            else:
                self.get_logger().info(f'Retrace rose {risen*1000:.1f} mm, past the '
                    f'{needed*1000:.1f} mm the lift check needs; no vertical top-up required.')

        self.sequence_stage = 'move_lift_check'
        self._lift_floor_fail_count = 0
        self._lift_check_last_nonlifted_target = None
        self._send_cartesian_path(waypoints)

    def start_lift_verification(self) -> None:
        """After lift motion succeeds, wait briefly for a detection that matches lifted position."""
        self.send_lift_check()

    def _lift_verification_tick(self) -> None:
        """
        Lift Verification Agent.

        Uses fresh YOLO26-seg mask detection after lift, plus the direct
        held-probe check on the jaw volume. The floor detection answers "did
        the probe leave its old pose", which a rover parked in front of the
        probe answers by accident; the jaw-volume check answers "is the probe
        in the gripper", which is the question.
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

        # The probe mesh is not attached yet, so the lift check uses fresh
        # detections, the close-gap latch, and trusted gripper contact.

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

            self._finish_lift_check_by_timeout()

    def _finish_lift_check_by_timeout(self) -> None:
        """Decide the lift check when no detection settled it.

        Timing out means the probe was not seen at its old floor pose. That
        used to PASS outright, which is exactly how an empty gripper reached
        "object held": after the grasp the rover is parked over the probe, so
        NOT seeing it there is the expected outcome whether or not the jaws
        picked it up. A pass now needs positive evidence that something is in
        the jaws — a held-probe verdict, or trusted jaw contact — and an empty
        final close vetoes the contact route.
        """
        verdict = self._held_probe_verdict()
        evidence_txt = self._held_probe_evidence.summary(self._now_sec())
        contact_ok = self.gripper_contact_detected and self.trust_gripper_contact_for_success

        if verdict == grasp_verification.HELD:
            self.get_logger().info(f'Lift-check PASSED by timeout + held-probe verification: '
                f'the probe is in the jaw volume ({evidence_txt}).')
            self._after_lift_verification_success('Lift-check PASSED by timeout.')
            return

        if verdict == grasp_verification.EMPTY:
            self.get_logger().error(f'Lift-check FAILED by timeout: held-probe verification says the jaws '
                f'are EMPTY ({evidence_txt}). The probe was not seen at its old floor pose '
                'either, but with the rover parked over it that proves nothing.')
            self.handle_failed_grasp_after_lift()
            return

        if not self.lift_check_require_positive_evidence:
            if contact_ok:
                self.get_logger().info('Lift-check PASSED by timeout + gripper contact: '
                    'probe is likely occluded/held between fingers.')
            else:
                self.get_logger().info('Lift-check PASSED by timeout: probe not detected at original floor pose.')
            self._after_lift_verification_success('Lift-check PASSED by timeout.')
            return

        if self._empty_close_detected:
            gap_txt = (f'{self._empty_close_gap_m*1000:.1f} mm'
                       if self._empty_close_gap_m is not None else 'sub-probe')
            self.get_logger().error(f'Lift-check FAILED by timeout: the final close reached a {gap_txt} jaw gap '
                f'on a {self.minimum_probe_width_m*1000:.0f} mm probe, and nothing since has shown '
                f'the probe in the jaws ({evidence_txt}). Not seeing the probe at its old floor '
                'pose is not evidence of a grasp.')
            self.handle_failed_grasp_after_lift()
            return

        if contact_ok:
            self.get_logger().info(f'Lift-check PASSED by timeout + gripper contact: the jaws stalled at a '
                f'probe-width gap, so something is between them ({evidence_txt}).')
            self._after_lift_verification_success('Lift-check PASSED by timeout.')
            return

        self.get_logger().error(f'Lift-check FAILED by timeout: no fresh detection, no trusted jaw contact, and '
            f'held-probe verification is inconclusive ({evidence_txt}). Passing here would be '
            'passing on absence of evidence.')
        self.handle_failed_grasp_after_lift()

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
        """Lower back to the retry grasp pose while still closed, then open.

        This descends to the same contact point as the original grasp, so it
        carries the same diagonal-sweep exposure and goes through
        _final_descent_waypoints too: closed fingers cutting sideways at probe
        height are worse than open ones, and if a false-negative grasp is in
        fact holding the probe, that sweep drags it along.
        """
        if self.grasp_pose is None:
            self.reset_sequence('Cannot retry failed grasp: no grasp pose is available.')
            return

        self.sequence_stage = 'move_retry_return'
        self.get_logger().warning('Returning to the retry grasp pose with the gripper still closed before opening. '
            'This keeps a possible false-negative grasp close to the ground; MoveIt collision '
            'checking remains enabled for the return path.')
        self._send_cartesian_path(
            self._final_descent_waypoints(self.contact_pose_to_link_pose(self.grasp_pose))
        )

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

        self._send_cartesian_path(
            self._final_descent_waypoints(self.contact_pose_to_link_pose(self.grasp_pose))
        )
        return True

    def _apply_replan_options(self, goal: MoveGroup.Goal) -> None:
        """Let MoveGroup replan when the planning scene invalidates a running plan.

        MoveGroup's PlanExecution re-validates the executing trajectory against
        every planning-scene update, and with ``replan`` false (the default) the
        first update that invalidates it aborts the motion outright:
        ``moveit_error_code=-3 MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE``.

        Explicit probe, attachment, and floor objects can legitimately update
        during transport. Replanning from the latest explicit scene is safer
        than abandoning the motion on its first invalidation.
        """
        if not self.movegroup_replan_enabled:
            return
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = int(self.movegroup_replan_attempts)
        goal.planning_options.replan_delay = float(self.movegroup_replan_delay_sec)

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
            # Keep what was actually sent: the failure diagnostic must re-test
            # this exact path with collisions off, or its verdict describes a
            # trajectory the arm never attempted.
            self._last_final_descent_waypoints = list(waypoints)
            current_pose = self.get_current_link_pose_in_planning_frame()
            goal_pose = waypoints[-1]
            # Where the descent started. The post-grasp lift retraces this path
            # backwards, so it needs the standoff pose the arm actually left
            # from, not a recomputed one.
            if current_pose is not None:
                self._last_final_descent_start = Pose(
                    position=Point(
                        x=float(current_pose.position.x),
                        y=float(current_pose.position.y),
                        z=float(current_pose.position.z),
                    ),
                    orientation=goal_pose.orientation,
                )
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
        # Length of the path being asked for, so a partial plan can be confirmed
        # against how far it actually goes rather than against the full goal.
        path_len = 0.0
        start_pose = self.get_current_link_pose_in_planning_frame()
        if start_pose is not None and waypoints:
            pts = [np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z])]
            pts += [np.array([w.position.x, w.position.y, w.position.z]) for w in waypoints]
            path_len = float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))
        self._cartesian_plan_in_flight = (expected_stage, expected_seq)
        future = self.cartesian_client.call_async(req)
        future.add_done_callback(
            lambda fut, st=expected_stage, seq=expected_seq, wp=final_waypoint, pl=path_len:
                self._on_cartesian_path(fut, st, seq, wp, pl)
        )

    def _on_cartesian_path(
        self,
        future,
        expected_stage: str,
        expected_seq: int,
        final_waypoint: Optional[Pose],
        path_length_m: float = 0.0,
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
        # A short descent that still lands the jaws on the probe body is not a
        # failure -- close there. Checked before the fraction gate because the
        # alternatives (MoveGroup replan, lifted retry, reset) all move the arm
        # again for a grasp that is already good enough.
        accepted_partial = (
            stage == 'move_grasp'
            and resp.fraction < min_fraction
            and self._accept_partial_final_descent(resp.fraction)
        )
        if resp.fraction < min_fraction and not accepted_partial:
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
        # A partial plan stops short BY DESIGN -- fractions down to
        # cartesian_fraction_min are accepted and executed. Confirming such a
        # trajectory against the full goal pose is then guaranteed to fail: a
        # 0.86 fraction here reported position_error=0.0637m/0.0200m and halted
        # the task with the probe already gripped. Widen the gate by the
        # shortfall the plan itself declared, so the check still catches an arm
        # that did not follow the trajectory it was given.
        position_tol = self.arm_pose_confirmation_position_tolerance_m
        shortfall = max(0.0, path_length_m * (1.0 - float(resp.fraction)))
        if shortfall > 0.0:
            position_tol += shortfall
            self.get_logger().info(f'Cartesian plan is {resp.fraction:.2f} of a '
                f'{path_length_m*1000:.0f} mm path, so it stops about {shortfall*1000:.1f} mm '
                f'short; confirming against {position_tol*1000:.1f} mm instead of '
                f'{self.arm_pose_confirmation_position_tolerance_m*1000:.1f} mm.')
        if not self._register_pose_motion_confirmation(
            stage,
            expected_seq,
            target_pose,
            [0.0, 0.0, 0.0],
            position_tol,
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
            self._last_final_descent_waypoints
            or [self.contact_pose_to_link_pose(self.grasp_pose)],
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
        self._grasp_orientation_search = None
        self._grasp_orientation_search_token += 1
        self._grasp_pregrasp_joint_targets = {}
        self._clear_arm_motion_confirmation()

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

        # Never turn a post-close failure into a new idle acquisition. The
        # next sequence starts by opening the gripper, which would drop a
        # possibly held probe at an arbitrary failure location. Preserve the
        # attachment and close state and require explicit operator recovery.
        if self.holding_object:
            self.last_failure_reason = reason
            if reason:
                self.failure_count += 1
            self.task_complete = True
            self.busy = True
            self.sequence_stage = 'transport_failed_holding'
            self.success_until_sec = self._now_sec() + self.success_lockout_sec
            self.get_logger().error(
                f'{reason} Reset was converted to a safe closed hold because '
                f'the node may still be carrying the probe. The attached '
                f'collision object is retained and automatic reacquisition is disabled.'
            )
            self.publish_markers()
            return

        # Clean up any post-grasp collision objects (floor plane + probe) from
        # the planning scene only when the gripper is known not to be holding.
        self._remove_post_grasp_collision_objects()
        self._remove_world_probe_object('grasp sequence reset')
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
        self._grasp_time_object_pose = None
        self._grasp_time_object_R = None
        self._grasp_time_tip_resolved = False
        self._grasp_close_link_pose = None
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
        self._clear_detected_object_pose(clear_fat_dir=True)
        self.locked_target_before_lift = None
        self.retry_target_from_lift_check = None
        self._lift_check_last_nonlifted_target = None
        self.sequence_locked_target_point_base = None
        self.sequence_locked_object_long_axis_base = None
        self.gripper_contact_detected = False
        self._reset_held_probe_evidence()
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
        position_region_dimensions: Optional[List[float]] = None,
        position_region_orientation: Optional[Quaternion] = None,
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
            'position_region_dimensions': (
                [float(v) for v in position_region_dimensions]
                if position_region_dimensions is not None else None
            ),
            'position_region_orientation': position_region_orientation,
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
        region_dimensions = pending.get('position_region_dimensions')
        position_ok = False
        if region_dimensions is not None:
            transformed_region = self._pose_target_in_planning_frame(
                pending['target_pose'],
                pending.get('position_region_orientation'),
            )
            if transformed_region is None or transformed_region[1] is None:
                return False, 'position-region transform is unavailable'
            region_xyz, region_orientation = transformed_region
            delta_local = quat_to_matrix(region_orientation).T @ (actual_xyz - region_xyz)
            half_extents = 0.5 * np.asarray(region_dimensions, dtype=np.float64)
            overflow = np.maximum(np.abs(delta_local) - half_extents, 0.0)
            position_error = float(np.linalg.norm(overflow))
            position_ok = bool(
                np.all(np.abs(delta_local) <= half_extents + float(pending['position_tolerance']))
            )
            position_detail = (
                f'region_local=({delta_local[0]:.4f},{delta_local[1]:.4f},{delta_local[2]:.4f})m, '
                f'region_half=({half_extents[0]:.4f},{half_extents[1]:.4f},{half_extents[2]:.4f})m, '
                f'outside_error={position_error:.4f}m/{float(pending["position_tolerance"]):.4f}m'
            )
        else:
            position_error = float(np.linalg.norm(actual_xyz - target_xyz))
            position_ok = position_error <= float(pending['position_tolerance'])
            position_detail = (
                f'position_error={position_error:.4f}m/'
                f'{float(pending["position_tolerance"]):.4f}m'
            )
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
            position_ok
            and orientation_ok
        )
        return reached, (
            f'{position_detail}, '
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
        if stage == 'move_base_box_drop':
            if self._try_next_base_box_drop_candidate(
                f'Base-box motion failed measured-state confirmation: {reason}'
            ):
                return
            self._hold_closed_after_transport_failure(reason)
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
                        position_region_dimensions: Optional[List[float]] = None,
                        position_region_orientation: Optional[Quaternion] = None,
                        path_constraints: Optional[Constraints] = None,
                        velocity_scale: Optional[float] = None,
                        acceleration_scale: Optional[float] = None,
                        arm_joints_only_start_state: bool = False,
                        planning_time_override: Optional[float] = None,
                        num_attempts_override: Optional[int] = None,
                        preferred_joint_positions: Optional[dict] = None,
                        preferred_joint_tolerance: Optional[float] = None) -> None:
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
        if position_region_dimensions is not None:
            if (
                len(position_region_dimensions) != 3
                or any(float(v) <= 0.0 for v in position_region_dimensions)
            ):
                self._arm_motion_confirmation_failed(
                    expected_stage,
                    'Pose position region must contain three positive dimensions.',
                )
                return
            box_region = SolidPrimitive()
            box_region.type = SolidPrimitive.BOX
            box_region.dimensions = [float(v) for v in position_region_dimensions]
            region_pose = Pose()
            region_pose.position = pose.pose.position
            region_pose.orientation = (
                position_region_orientation
                if position_region_orientation is not None else pose.pose.orientation
            )
            region.primitives.append(box_region)
            region.primitive_poses.append(region_pose)
        else:
            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [tol]
            region.primitives.append(sphere)
            region.primitive_poses.append(pose.pose)
        pos.constraint_region = region
        pos.weight = 1.0
        c.position_constraints.append(pos)
        if preferred_joint_positions:
            joint_tol = max(
                0.01,
                float(
                    self.lock_wrist_joint_tolerance
                    if preferred_joint_tolerance is None
                    else preferred_joint_tolerance
                ),
            )
            for joint_name, joint_position in preferred_joint_positions.items():
                jc = JointConstraint()
                jc.joint_name = str(joint_name)
                jc.position = float(joint_position)
                jc.tolerance_above = joint_tol
                jc.tolerance_below = joint_tol
                jc.weight = 1.0
                c.joint_constraints.append(jc)
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
        self._apply_replan_options(goal)
        confirmation_orientation = orientation_constraint if with_orientation else None
        if not self._register_pose_motion_confirmation(
            expected_stage,
            expected_seq,
            pose,
            offset,
            float(tol),
            confirmation_orientation,
            float(self.orientation_tol if orientation_tol is None else orientation_tol),
            position_region_dimensions=position_region_dimensions,
            position_region_orientation=position_region_orientation,
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
        self._apply_replan_options(goal)
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
            if expected_stage == 'move_base_box_drop':
                reason = f'MoveIt base-box goal could not be sent: {exc}'
                if self._try_next_base_box_drop_candidate(reason):
                    return
                self._hold_closed_after_transport_failure(reason)
                return
            if expected_stage == 'move_pick_home' and self.place_in_base_box_after_grasp:
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
                reason = 'MoveIt rejected the motion into the base-box release volume.'
                if self._try_next_base_box_drop_candidate(reason):
                    return
                self._hold_closed_after_transport_failure(
                    reason
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
            if expected_stage == 'move_base_box_drop':
                reason = f'MoveIt base-box result was unavailable: {exc}'
                if self._try_next_base_box_drop_candidate(reason):
                    return
                self._hold_closed_after_transport_failure(reason)
                return
            if expected_stage == 'move_pick_home' and self.place_in_base_box_after_grasp:
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
                moveit_code = getattr(
                    getattr(getattr(result_wrap, 'result', None), 'error_code', None),
                    'val',
                    'unknown',
                )
                reason = (
                    'MoveIt motion into the base-box release volume failed with '
                    f'action_status={result_wrap.status}, moveit_error_code={moveit_code}.'
                )
                code_int = moveit_code if isinstance(moveit_code, int) else None
                if self._try_next_base_box_drop_candidate(reason, moveit_code=code_int):
                    return
                self._hold_closed_after_transport_failure(
                    reason
                )
                return
            if expected_stage == 'move_pick_home' and self.place_in_base_box_after_grasp:
                # action_status alone is just ABORTED(6) and hides WHY. The
                # MoveItErrorCode separates a genuine planning failure (-1)
                # from a start state already in collision (-10) or a goal in
                # collision (-12) -- three completely different fixes.
                moveit_code = getattr(
                    getattr(getattr(result_wrap, 'result', None), 'error_code', None),
                    'val', 'unknown',
                )
                hint = {
                    -1: 'PLANNING_FAILED: OMPL searched and found nothing.',
                    -10: 'START_STATE_IN_COLLISION: the arm or attached probe is already '
                         'in collision before moving.',
                    -12: 'GOAL_IN_COLLISION: the transport posture itself is blocked.',
                }.get(moveit_code if isinstance(moveit_code, int) else None, '')
                self._hold_closed_after_transport_failure(
                    f'MoveIt held-probe motion to pick_home failed with '
                    f'action_status={result_wrap.status}, moveit_error_code={moveit_code}. {hint}'
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
            geometry_ok, geometry_detail = self._verify_base_box_release_geometry()
            if geometry_ok:
                self.get_logger().info(
                    f'Base-box release geometry PASSED: {geometry_detail}.')
                self.release_probe_in_base_box()
            else:
                reason = f'Base-box release geometry rejected after measured arm confirmation: {geometry_detail}.'
                if self._try_next_base_box_drop_candidate(reason):
                    return
                self._hold_closed_after_transport_failure(reason)

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
        # Retire this probe's identity so the next pick starts a fresh track id.
        if self._probe_track is not None:
            self.get_logger().info(f'Probe track#{self._probe_track.track_id} retired '
                'after successful placement.')
            self._probe_track = None

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
        """Show the box, central release zone, and required probe long axis."""
        pose = self.get_base_box_drop_pose()
        markers: List[Marker] = []
        next_id = marker_id

        layout = None
        if self.base_box_auto_drop_enabled:
            try:
                layout = self._compute_base_box_layout()
            except ValueError:
                layout = None

        if layout is not None:
            box = Marker()
            box.header.frame_id = self.base_box_drop_frame
            box.header.stamp = stamp
            box.ns = 'base_box_volume'
            box.id = next_id
            next_id += 1
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position = Point(
                x=float(layout.center[0]), y=float(layout.center[1]), z=float(layout.center[2])
            )
            box.pose.orientation = rpy_to_quat(*self.base_box_rpy)
            box.scale.x = float(layout.dimensions[0])
            box.scale.y = float(layout.dimensions[1])
            box.scale.z = float(layout.dimensions[2])
            box.color = ColorRGBA(r=0.10, g=0.75, b=1.0, a=0.20)
            box.lifetime = Duration(sec=0, nanosec=0)
            markers.append(box)

            release_volume = Marker()
            release_volume.header.frame_id = self.base_box_drop_frame
            release_volume.header.stamp = stamp
            release_volume.ns = 'base_box_drop_release_volume'
            release_volume.id = next_id
            next_id += 1
            release_volume.type = Marker.CUBE
            release_volume.action = Marker.ADD
            release_volume.pose.position = Point(
                x=float(layout.release_volume_center[0]),
                y=float(layout.release_volume_center[1]),
                z=float(layout.release_volume_center[2]),
            )
            release_volume.pose.orientation = rpy_to_quat(*self.base_box_rpy)
            release_volume.scale.x = float(layout.release_volume_dimensions[0])
            release_volume.scale.y = float(layout.release_volume_dimensions[1])
            release_volume.scale.z = float(layout.release_volume_dimensions[2])
            release_volume.color = ColorRGBA(r=0.95, g=0.25, b=0.95, a=0.28)
            release_volume.lifetime = Duration(sec=0, nanosec=0)
            markers.append(release_volume)

            axis_index = 0 if layout.probe_axis_name == 'X' else 1
            probe_axis = normalize(layout.rotation[:, axis_index])
            half_probe = 0.5 * box_drop.PROBE_LENGTH_M
            axis_start = layout.release_volume_center - half_probe * probe_axis
            axis_end = layout.release_volume_center + half_probe * probe_axis
            probe_axis_marker = Marker()
            probe_axis_marker.header.frame_id = self.base_box_drop_frame
            probe_axis_marker.header.stamp = stamp
            probe_axis_marker.ns = 'base_box_drop_probe_axis'
            probe_axis_marker.id = next_id
            next_id += 1
            probe_axis_marker.type = Marker.ARROW
            probe_axis_marker.action = Marker.ADD
            probe_axis_marker.pose.orientation.w = 1.0
            probe_axis_marker.points = [
                Point(x=float(axis_start[0]), y=float(axis_start[1]), z=float(axis_start[2])),
                Point(x=float(axis_end[0]), y=float(axis_end[1]), z=float(axis_end[2])),
            ]
            probe_axis_marker.scale.x = 0.010
            probe_axis_marker.scale.y = 0.022
            probe_axis_marker.scale.z = 0.030
            probe_axis_marker.color = ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.95)
            probe_axis_marker.lifetime = Duration(sec=0, nanosec=0)
            markers.append(probe_axis_marker)

        point = Marker()
        point.header.frame_id = pose.header.frame_id
        point.header.stamp = stamp
        point.ns = 'base_box_drop_point'
        point.id = next_id
        next_id += 1
        point.type = Marker.SPHERE
        point.action = Marker.ADD
        point.pose = pose.pose
        point.scale.x = self.base_box_drop_marker_scale_m
        point.scale.y = self.base_box_drop_marker_scale_m
        point.scale.z = self.base_box_drop_marker_scale_m
        point.color = ColorRGBA(r=0.95, g=0.25, b=0.95, a=0.85)
        point.lifetime = Duration(sec=0, nanosec=0)
        markers.append(point)

        if not self.base_box_auto_drop_enabled:
            axes = self.make_pose_axes_markers(
                next_id,
                pose.header.frame_id,
                stamp,
                pose,
                namespace_prefix='base_box_drop',
                axis_length=self.base_box_drop_marker_axes_length_m,
            )
            markers.extend(axes)
            next_id += len(axes)

        label = Marker()
        label.header.frame_id = pose.header.frame_id
        label.header.stamp = stamp
        label.ns = 'base_box_drop_label'
        label.id = next_id
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
        if layout is not None:
            label.text = (
                f'CENTRAL AXIS-ALIGNED DROP  {layout.dimensions[0]:.2f} x '
                f'{layout.dimensions[1]:.2f} x {layout.dimensions[2]:.2f} m'
            )
        else:
            label.text = 'BASE BOX DROP'
        label.lifetime = Duration(sec=0, nanosec=0)
        markers.append(label)
        return markers

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
