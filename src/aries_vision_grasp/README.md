# aries_vision_grasp

Standalone ROS 2 package for Aries camera visualization, YOLO inference, and
autonomous MoveIt grasp execution.

```bash
colcon build --symlink-install --packages-up-to aries_vision_grasp
source install/setup.bash
ros2 launch aries_vision_grasp vision_grasp.launch.py
```

The vision launch defaults to simulation time to match
`aries_bringup my_robot.launch.py`. On the physical rover, launch with
`use_sim_time:=false`.

## Layout

- `scripts/vision_grasp_node.py` — perception + grasp state machine.
- `scripts/yolo_detection_node.py` — standalone annotated-detection stream
  (debug only; `vision_grasp_node` runs the same model itself and publishes
  `/vision_grasp/detection_image`). On hardware it is gated behind
  `enable_yolo_debug:=true` in `aries_bringup aries_hardware.launch.py`.
- `scripts/camera_viewer.py` — local OpenCV viewer for the gripper camera.
- `aries_vision_grasp/` — shared python library installed with the package:
  - `geometry.py` — quaternion/rotation/duration helpers.
  - `fourbar.py` — calibrated four-bar jaw-gap and contact-offset tables
    (measured from `gripper_new.xacro` + `gripper_bucket.stl`; a 45 mm probe
    needs q ≈ −0.20 rad, covered by unit tests in `test/`).
  - `stages.py` — state-machine stage names and the shared stage sets.
  - `inference.py` — YOLO loading (device fallback) and the background
    inference worker, so the model never blocks the rclpy executor.

## Dependencies

The YOLO runtime has **no rosdep key** and must be installed via pip in the
python environment the nodes run under:

```bash
pip install ultralytics   # pulls in torch
```

## Configuration

- `config/vision_grasp_params.yaml` — the full vision/grasp tuning set
  (detection thresholds, refinement window, floor guards, planning scales,
  rover-motion interlock, lift-check verification, …). Override with
  `params_file:=/path/to/file.yaml`.
- `config/pick_place.yaml` — postures and gripper completion gating. Loaded
  **last**, so it stays the authoritative file for those values. Override
  with `pick_place_config:=/path/to/file.yaml`.
- Launch arguments: `use_sim_time`, `model_path`, `target_class`,
  `params_file`, `pick_place_config`.

Gripper open, close, pre-close, joint limits, and the hard-stop safety margin
are configured in `config/pick_place.yaml`. The nominal `-1.57` open and `0.07`
closed commands are preserved; the Gazebo-only URDF limits include additional
travel so those commands do not coincide with a physics hard stop.

Probe acquisition filtering is configured in the same YAML under
`target_stability_*`, `target_filter_*`, and `target_lock_min_confidence`. The
committed grasp point is the median of a consecutive high-confidence 3D cluster;
large jumps reset acquisition instead of moving the robot target.

Camera-relative grasp calibration is configured in `pick_place.yaml` with
`grasp_target_offset_camera_xyz_m`. The `[x, y, z]` offset is in metres and is
added in the depth-camera frame before TF transforms the target to `base_link`.
For a standard ROS optical frame, +X is image-right, +Y is image-down, and +Z
points forward/away from the camera.

`auto_calibrate_camera_offset_enabled` enables a guarded multi-view estimate:
the probe must remain stationary while the wrist camera changes orientation on
the way to pre-grasp. The node solves for the camera-axis offset that makes the
transformed observations agree, rejects rank-deficient/noisy/oversized results,
and applies at most `auto_calibrate_camera_offset_max_step_m` per sequence. The
accepted XYZ value is printed in the log; copy it into
`grasp_target_offset_camera_xyz_m` to persist it across launches. This corrects
camera-relative systematic error, but it cannot infer an unobserved mechanical
finger-contact error without a known target or manual alignment measurement.

The default model is installed as `models/grasp.pt`: a YOLO26l segmentation
model trained for the `probe` class. Override `model_path` when testing other
weights.

## Perception timing

Inference runs in a background thread; each detect tick consumes the newest
completed result together with the exact color/depth frame pair it was
computed from. A detection is only processed when the color and depth stamps
agree within `max_color_depth_stamp_gap_sec`, and TF is looked up at the
depth frame's stamp — on a moving wrist camera this keeps the 3D target from
being skewed by camera motion between capture and processing.

## Grasp / placement flow

After a verified grasp, the hardware launch carries the probe through
`pick_home`, moves above the rover's base-mounted box, releases it, and returns
the empty arm to `pick_home`. The behavior is controlled by
`place_in_base_box_after_grasp`.

Configure only `base_box_center_xyz`, `base_box_dimensions_xyz`, and
`base_box_rpy` in `pick_place.yaml`. The node derives the known probe size,
wall allowance, edge clearance, top rim, and a continuous overhead release
volume automatically. The volume is restricted to the inner centre of the box
and kept close to the top rim so the probe cannot be released at an edge or
bounce out after a long fall. MoveIt targets the centre of that zone with a
small spherical tolerance. The probe's measured rigid long axis is aligned
lengthwise with the box's longest opening. The node searches rotations around
that probe axis until one produces a valid collision-checked wrist trajectory;
this changes robot IK without changing the probe's required drop alignment.
Probe length/width do not reject an overhead drop. Automatic placement
overrides the legacy manual XYZ/RPY and joint drop modes.

RViz `/vision_grasp/markers` shows a translucent cyan box at the configured
pose, a translucent magenta central release zone, and an orange 300 mm arrow
showing the required probe long axis. This makes position, dimensions, and
orientation visible immediately after relaunching the node.

With `gripper_require_feedback_for_completion: true`, opening and closing
advance only after the configured command time has elapsed, the trajectory
controller reports action success, **and** a fresh `/joint_states` gripper
position is within `gripper_goal_tolerance`. Controller rejection/abort and
timeout do not count as success.

Arm stages use the same rule across the full process. A MoveIt or Cartesian
action result must succeed, then fresh measured arm joints/TF must remain at
the commanded target for `arm_feedback_stable_samples` before the next arm or
gripper stage can start. Executed Cartesian paths always request collision
checking, attached-probe geometry remains active during transport, and
overlapping arm/gripper commands are rejected.

## Tests

```bash
colcon test --packages-select aries_vision_grasp
# or directly:
python3 -m pytest src/aries_vision_grasp/test/
```

The tests pin the four-bar calibration table (gap↔q roundtrips, the 45 mm
probe anchor point, contact offsets) and the quaternion/rotation helpers.
