"""Grasp state-machine stage names and the shared stage sets.

Stage names are plain strings (they are logged, compared, and stored in
per-callback captures throughout vision_grasp_node). Defining them — and the
behavioural groupings the supervisor relies on — in one place prevents the
sets in different methods from silently drifting apart.
"""

IDLE = 'idle'
OPEN_GRIPPER = 'open_gripper'
MOVE_PRE_GRASP = 'move_pre_grasp'
PREGRASP_FINALIZING = 'pregrasp_finalizing'
REFINE = 'refine'
MOVE_GRASP = 'move_grasp'
PRECLOSE_IN_AIR = 'preclose_in_air'
VERIFY_FINAL_GRASP_POSE = 'verify_final_grasp_pose'
PRECLOSE_GRIPPER = 'preclose_gripper'
CLOSE_GRIPPER = 'close_gripper'
VERIFY_GRIPPER = 'verify_gripper'
MOVE_LIFT_CHECK = 'move_lift_check'
VERIFY_LIFT = 'verify_lift'
MOVE_RETRY_RETURN = 'move_retry_return'
RETRY_OPEN_GRIPPER = 'retry_open_gripper'
MOVE_PICK_HOME = 'move_pick_home'
MOVE_BASE_BOX_DROP = 'move_base_box_drop'
RELEASE_IN_BASE_BOX = 'release_in_base_box'
MOVE_PICK_HOME_AFTER_PLACE = 'move_pick_home_after_place'
MOVE_CARTESIAN_RETREAT = 'move_cartesian_retreat'
MOVE_RETREAT_HOME = 'move_retreat_home'
DONE_HOLDING = 'done_holding'
DONE_PLACED = 'done_placed'

# Terminal/error holding stages (set directly by failure handlers).
GRASP_CHECK_FAILED_HOLDING = 'grasp_check_failed_holding'
TRANSPORT_FAILED_HOLDING = 'transport_failed_holding'
BASE_BOX_RELEASE_UNCONFIRMED = 'base_box_release_unconfirmed'
FAILED_FINAL_APPROACH = 'failed_final_approach'
FAILED_FINAL_POSE_CHECK = 'failed_final_pose_check'

# Stages during which no MoveIt/Cartesian arm command may be created; only
# pure gripper commands are allowed.
GRIPPER_STAGES = frozenset({
    PRECLOSE_IN_AIR,
    VERIFY_FINAL_GRASP_POSE,
    PRECLOSE_GRIPPER,
    CLOSE_GRIPPER,
    VERIFY_GRIPPER,
    RELEASE_IN_BASE_BOX,
})

# Stages where live YOLO/depth feedback may still update target state even
# when the hard perception freeze is armed.
LIVE_FEEDBACK_STAGES = frozenset({
    OPEN_GRIPPER,
    MOVE_PRE_GRASP,
    PREGRASP_FINALIZING,
    REFINE,
})

# Stages during which a live-track detection must never modify the committed
# target or trigger a replan (the grasp geometry is already locked).
LIVE_TRACK_LOCKED_STAGES = frozenset({
    PRECLOSE_IN_AIR,
    MOVE_GRASP,
    REFINE,
    PRECLOSE_GRIPPER,
    CLOSE_GRIPPER,
    VERIFY_GRIPPER,
    MOVE_PICK_HOME,
    MOVE_BASE_BOX_DROP,
    RELEASE_IN_BASE_BOX,
    MOVE_PICK_HOME_AFTER_PLACE,
    MOVE_LIFT_CHECK,
    VERIFY_LIFT,
    RETRY_OPEN_GRIPPER,
    MOVE_CARTESIAN_RETREAT,
    MOVE_RETREAT_HOME,
    DONE_HOLDING,
    DONE_PLACED,
})

# Stages that mean "the sequence is finished or not running".
TERMINAL_STAGES = frozenset({IDLE, DONE_HOLDING, DONE_PLACED})
