from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_model() -> str:
    """Return the model installed with the vision package."""
    return str(
        get_package_share_directory('aries_vision_grasp') + '/models/grasp.pt'
    )


def _find_pick_place_config() -> str:
    """Return the installed pick/place posture configuration."""
    return str(
        get_package_share_directory('aries_vision_grasp') + '/config/pick_place.yaml'
    )


def generate_launch_description():
    args = [
        # aries_bringup/my_robot.launch.py defaults to Gazebo/use_sim_time=true.
        # Keep action durations and watchdogs on that same clock. Override with
        # use_sim_time:=false when running the physical rover.
        ('use_sim_time', 'true'),
        ('model_path', _find_model()),
        ('target_class', 'probe'),
        ('confidence_threshold', '0.55'),
        ('detect_period_sec', '0.25'),

        # Depth/ROI settings.
        ('min_depth_m', '0.02'),
        ('max_depth_m', '2.0'),
        ('roi_half_size_px', '8'),

        # YOLO26-seg mask handling.
        ('use_segmentation_mask', 'true'),
        ('mask_score_threshold', '0.50'),
        ('mask_min_pixels', '80'),
        ('mask_erode_px', '2'),
        ('mask_depth_percentile', '35.0'),

        # Close-range refinement is enabled with a bounded correction window.
        # Hardware logs now show real close-range lateral corrections around
        # 45-58 mm; allow those while keeping vertical correction tight.
        ('refine_enabled', 'true'),
        ('refine_confidence_threshold', '0.25'),
        ('refine_use_projection_fallback', 'true'),
        ('refine_projection_roi_half_size_px', '45'),
        ('refine_min_depth_m', '0.02'),
        ('refine_depth_band_m', '0.08'),
        ('refine_samples', '4'),
        ('refine_min_samples_to_accept', '1'),
        ('refine_commit_on_timeout', 'true'),
        ('refine_timeout_sec', '2.2'),
        ('refine_accept_radius_m', '0.045'),
        ('refine_lateral_max_m', '0.045'),
        ('refine_vertical_max_m', '0.025'),

        # Planning frames/groups.
        ('planning_frame', 'base_link'),
        ('planning_group', 'igus_rebel_arm'),
        ('planning_link', 'arm_gripper_base_link'),
        ('keep_current_orientation', 'false'),
        ('use_orientation_constraint', 'true'),
        ('orientation_tolerance_rad', '0.12'),
        ('pre_grasp_position_tol', '0.020'),
        # Hardware calibration: the probe was observed about 20 mm in front of
        # the grasp point.  Base +X is rover/front, so bias the grasp target
        # forward without changing MoveIt collision checking.
        ('grasp_target_bias_base_x_m', '0.020'),
        ('grasp_target_bias_base_y_m', '0.000'),
        ('grasp_target_bias_base_z_m', '0.000'),
        ('grasp_target_bias_tool_x_m', '0.010'),
        ('grasp_target_bias_tool_y_m', '0.000'),
        ('grasp_target_bias_tool_z_m', '0.000'),

        # Floor-safe grasp distances.
        ('pre_grasp_distance', '0.12'),
        ('surface_offset', '0.000'),
        ('grasp_depth_below_surface_m', '0.002'),
        ('retreat_distance', '0.08'),
        ('min_pose_z', '-0.18'),
        ('floor_z_min', '-0.22'),
        ('reject_targets_below_floor', 'true'),
        ('floor_safe_grasp_enabled', 'true'),
        ('max_grasp_descent_below_target_m', '0.006'),
        ('min_grasp_height_above_floor_m', '0.035'),
        ('floor_safe_contact_height_m', '0.060'),

        # Gripper URDF: -1.57 open, 0.07 closed.
        ('gripper_open_width', '-1.57'),
        ('gripper_close_width', '0.07'),
        ('gripper_preclose_width', '-0.26'),
        ('final_grasp_arm_settle_sec', '2.00'),
        # Do not close unless the measured TCP actually reached the committed
        # grasp pose. This catches controller lag/stale execution success before
        # the gripper closes above the probe.
        ('final_grasp_pose_check_enabled', 'true'),
        ('final_grasp_pose_position_tolerance_m', '0.015'),
        ('final_grasp_pose_orientation_tolerance_rad', '0.35'),
        ('final_grasp_pose_check_timeout_sec', '6.0'),
        ('final_grasp_pose_check_period_sec', '0.10'),
        ('gripper_command_mode', 'auto'),
        ('gripper_contact_min_position', '0.000'),

        # Always request the full mechanical close and let the object stop the
        # fingers. Completion is still gated by time + fresh joint feedback;
        # those settings live in config/pick_place.yaml.
        ('adaptive_gripper_enabled', 'false'),
        ('gripper_gap_at_zero_rad', '0.1786'),
        ('gripper_gap_slope', '2.0'),
        ('object_width_safety_margin_m', '0.010'),
        ('minimum_probe_width_m', '0.045'),
        ('nominal_probe_width_m', '0.045'),
        ('maximum_probe_width_m', '0.060'),
        ('clamp_probe_width_for_grasp', 'true'),
        ('object_width_final_clearance_m', '-0.008'),
        ('object_width_preclose_clearance_m', '0.010'),
        ('preclose_min_q_margin_rad', '0.004'),
        ('adaptive_gripper_min_width_m', '0.008'),
        ('adaptive_gripper_max_width_m', '0.070'),
        ('adaptive_gripper_width_percentile', '30.0'),

        # Four-bar contact/supervisor.
        ('gripper_contact_z_gear_arm_m', '0.089'),
        ('gripper_contact_z_ref_q', '0.07'),
        ('fourbar_use_urdf_geometry_model', 'true'),
        ('fourbar_contact_y_offset_m', '0.0259'),
        ('fourbar_contact_z_open_m', '0.1342'),
        ('fourbar_contact_z_closed_m', '0.2180'),
        ('fourbar_q_min_for_floor_grasp', '-0.42'),
        ('fourbar_q_max_for_floor_grasp', '-0.08'),
        ('fourbar_max_contact_lift_m', '0.014'),
        ('fourbar_min_arc_clearance_m', '0.006'),
        ('fourbar_ground_guard_enabled', 'true'),
        ('fourbar_bucket_tip_z_max_m', '0.275'),
        ('fourbar_safe_contact_offset_z_min_m', '0.225'),
        ('fourbar_ground_clearance_m', '0.035'),
        ('fourbar_preclose_before_grasp', 'false'),
        ('fourbar_final_close_steps', '1'),
        ('fourbar_final_close_step_wait_sec', '0.00'),
        ('freeze_arm_during_gripper_enabled', 'true'),
        # After final close: keep the gripper closed through pickup verification
        # and transport to pick_home. Release is allowed only later at the
        # calibrated base-box posture.
        ('hold_after_close_no_motion', 'false'),
        ('post_grasp_lift_then_pick_home', 'true'),
        ('post_grasp_lift_distance_m', '0.160'),
        ('post_grasp_lift_speed_scale', '1.00'),
        ('post_grasp_lift_segment_m', '0.015'),
        ('post_grasp_lift_min_fraction', '0.05'),
        ('post_grasp_lift_avoid_collisions', 'true'),
        ('post_grasp_lift_escape_distance_m', '0.025'),
        ('post_grasp_lift_waypoint_step_m', '0.003'),
        ('post_grasp_lift_min_progress_m', '0.002'),
        ('post_grasp_lift_max_retries', '6'),
        ('post_grasp_lift_goal_position_tol_m', '0.004'),
        ('post_grasp_lift_path_xy_tolerance_m', '0.012'),
        ('post_grasp_lift_orientation_tol_rad', '0.12'),
        # Dense IK step (2 mm) + tight jump guard (0.5 rad):
        #   0.5 rad prevents the IK solver from leaping between arm
        #   configurations on consecutive waypoints.  For a 2 mm Cartesian
        #   step the joint change should be < 0.1 rad; 0.5 is conservative.
        #   5.0 was the previous value — too loose, allowed IK branch-flips
        #   which caused the arm to swing in the wrong direction.
        #   NOTE: The Cartesian segmented lift is no longer the primary path;
        #   post-grasp transport now goes directly to pick_home via MoveGroup
        #   joint goal.  This parameter is kept for legacy code paths.
        ('post_grasp_lift_jump_threshold', '0.5'),
        ('post_grasp_lift_max_step_m', '0.002'),
        ('post_grasp_min_link_z_m', '0.140'),
        # Home and base-box drop postures are loaded from config/pick_place.yaml.

        # Object yaw only; keep floor probe grasp top-down.
        ('publish_object_pose', 'true'),
        ('object_pose_topic', '/vision_grasp/object_pose'),
        ('object_pose_axis_length_m', '0.080'),
        ('object_yaw_align_enabled', 'true'),
        ('object_yaw_rotation_offset_deg', '90.0'),
        ('object_orientation_min_eigenratio', '3.0'),
        ('stl_yaw_correction_deg', '0.0'),   # +/- degrees to fix STL/mask misalignment
        # Waypoint orientations are still supplied.  This disables the extra
        # path constraint that can make KDL return fraction=0.00 near the floor;
        # MoveIt collision checking remains enabled for the Cartesian descent.
        ('cartesian_lock_orientation', 'false'),

        # Slow and clean arm movement.
        ('allowed_planning_time', '8.0'),
        ('num_planning_attempts', '20'),
        ('velocity_scale', '0.12'),
        ('acceleration_scale', '0.12'),
        ('cartesian_max_step', '0.004'),
        ('cartesian_fraction_min', '0.85'),
        ('cartesian_retry_lift_m', '0.030'),
        ('cartesian_max_retries', '1'),
        ('allow_movegroup_fallback_for_grasp', 'false'),
        ('final_grasp_movegroup_fallback_position_tol', '0.012'),
        ('diagnose_final_cartesian_failure', 'true'),
        ('stop_after_final_approach_failure', 'true'),
        ('final_approach_failure_lockout_sec', '999999.0'),

        # Rover-motion interlock: pause/cancel arm auto-grasp whenever the
        # rover is commanded to move. This keeps the wrist camera target and
        # MoveIt planning frame stationary during arm motion.
        ('pause_arm_when_rover_moving', 'true'),
        ('rover_motion_cmd_vel_topic', '/cmd_vel'),
        ('rover_motion_linear_threshold_mps', '0.020'),
        ('rover_motion_angular_threshold_radps', '0.030'),
        ('rover_motion_pause_hold_sec', '0.75'),
        ('rover_motion_cancel_active_arm_motion', 'true'),

        # Target lock: no live target update after pregrasp by default.
        ('target_stability_samples', '6'),
        ('target_stability_max_jump_m', '0.012'),
        ('continuous_tracking_enabled', 'true'),
        ('hard_freeze_perception_after_lock', 'true'),
        ('disable_refinement_after_lock', 'true'),
        ('disable_live_replan_after_lock', 'true'),
        ('replan_target_move_threshold_m', '0.030'),
        ('max_replans_per_grasp', '1'),
        ('tracking_lost_timeout_sec', '1.0'),
        ('ignore_live_replan_during_pregrasp', 'true'),
        ('use_recent_live_target_after_pregrasp', 'false'),
        ('pregrasp_recent_target_max_age_sec', '3.0'),
        ('pregrasp_live_update_accept_m', '0.055'),
        ('probe_shape_aware_center_enabled', 'true'),
        ('probe_parallel_center_update_scale', '0.00'),
        ('continue_if_live_target_stale_after_pregrasp', 'true'),
        ('pregrasp_watchdog_enabled', 'true'),
        ('pregrasp_watchdog_timeout_sec', '12.0'),
        ('pregrasp_watchdog_min_sec', '5.0'),
        ('pregrasp_link_arrival_tolerance_m', '0.025'),
        ('pregrasp_watchdog_force_after_timeout', 'false'),
        ('pregrasp_max_final_replans', '1'),
        ('pregrasp_finalize_even_if_moveit_silent', 'false'),
        ('lock_grasp_orientation_after_initial_plan', 'true'),

        # Verify pickup before the collision-aware pick_home transport.
        ('verify_grasp_after_lift', 'true'),
        ('lift_check_distance_m', '0.200'),
        ('lift_check_detect_timeout_sec', '1.5'),
        ('max_grasp_attempts', '2'),
        ('retry_extra_grasp_depth_m', '0.0001'),
        ('grasp_failure_same_place_radius_m', '0.060'),
        ('grasp_success_min_lift_m', '0.055'),
        ('lift_check_require_positive_z_success', 'true'),
        ('trust_gripper_contact_for_success', 'false'),
        ('lift_check_floor_fail_samples', '5'),
        ('never_open_after_contact_during_retry', 'true'),
        ('keep_closed_on_lift_check_failure_without_feedback', 'false'),
        ('require_lift_check_success_for_transport', 'true'),
        ('close_gripper_extra_wait_sec', '4.0'),

        # Active live feedback until pre-grasp, then one final refinement pass.
        ('pregrasp_active_correction_enabled', 'true'),
        ('pregrasp_active_correction_threshold_m', '0.045'),
        ('pregrasp_active_correction_max_cycles', '1'),
        ('close_in_one_go_after_pregrasp_refine', 'true'),
        ('fourbar_arc_guard_enabled', 'true'),
        ('fourbar_arc_sample_count', '17'),
        ('fourbar_open_close_guard_extra_m', '0.015'),
        ('failure_cooldown_sec', '3.0'),

        # Completion behavior.
        ('auto_restart_after_success', 'false'),
        ('success_lockout_sec', '999999.0'),
        ('hold_object_after_success', 'true'),
        ('clear_target_after_success', 'false'),
        # Markers.
        ('publish_markers', 'true'),
        ('markers_topic', '/vision_grasp/markers'),
        ('marker_frame', ''),
        ('marker_use_zero_stamp', 'true'),
    ]

    declared = [DeclareLaunchArgument(name, default_value=value) for name, value in args]
    params = {name: LaunchConfiguration(name) for name, _ in args}
    pick_place_config_arg = DeclareLaunchArgument(
        'pick_place_config',
        default_value=_find_pick_place_config(),
        description='YAML file containing home and base-box placement postures.',
    )

    return LaunchDescription(declared + [pick_place_config_arg,
        Node(
            package='aries_vision_grasp',
            executable='vision_grasp_node.py',
            name='vision_grasp_node',
            output='screen',
            # The YAML is last so posture values live in one authoritative file.
            parameters=[params, LaunchConfiguration('pick_place_config')],
        ),
    ])
