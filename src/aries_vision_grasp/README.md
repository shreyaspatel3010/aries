# aries_vision_grasp

Standalone ROS 2 package for Aries camera visualization, YOLO inference, and
autonomous MoveIt grasp execution.

```bash
colcon build --symlink-install --packages-up-to aries_vision_grasp
source install/setup.bash
ros2 launch aries_vision_grasp vision_grasp.launch.py
```

The vision launch defaults to wall-clock time (`use_sim_time:=false`) because
standalone use means the physical rover, and a `true` default with no `/clock`
publisher freezes every detector timer. Against Gazebo — including
`aries_bringup my_robot.launch.py` — pass `use_sim_time:=true` explicitly:

```bash
ros2 launch aries_vision_grasp vision_grasp.launch.py use_sim_time:=true
```

Omitting it there leaves the node comparing wall-clock `now()` against
sim-stamped camera frames, so every inference is dropped with
`age=1784972829.58s > inference_result_max_age_sec=2.00s` — an age near the Unix
epoch is always this clock-domain mismatch, never a real delay.

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

### Depth visualisation

MoveIt no longer builds an occupancy map. RViz shows separate colored
`rover_DepthCloud` and `gripper_DepthCloud` displays using each camera's own
aligned depth, RGB image, CameraInfo, optical frame, and TF. No `PointCloud2`
topic is produced. Planning safety comes from robot self-collision, explicit
collision objects such as the detected-probe mesh, and the task's height/reach
guards.

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

## Held-probe mesh re-alignment

The attached probe collision mesh is not frozen at grasp time. While the probe
is held (`attached_probe_realign_enabled`), every detection tick back-projects
the YOLO26-seg mask through the depth image into the gripper link frame —
where a rigidly held probe is stationary even while the arm moves — gates the
points against the currently attached box model (so a second probe on the
floor is ignored), and refines the pose with a trimmed point-to-box ICP
(`aries_vision_grasp/probe_alignment.py`; dimensions are read from the active
probe STL, so closest-surface correspondences are analytic). When the fit disagrees with
the published mesh, the `AttachedCollisionObject` is republished in place:
small drifts commit after `attached_probe_realign_confirm_samples` agreeing
frames, and deviations beyond the `attached_probe_realign_fast_*` thresholds
bypass the republish rate limit so a slip inside the gripper updates the
collision world immediately. Each commit also refreshes the base-box drop
facts (probe world yaw, centre and long axis in the link frame) used for drop
alignment and release verification. This corrects
an off-centre or rotated grasp within about half a second of closing and keeps
tracking through transport and the terminal holding states.

Tracking gates observations against the current attached pose, so a grossly
wrong attach (a flipped or far-off mesh) would reject every frame and could
never self-correct. After `attached_probe_realign_reacquire_after_sec` without
a gated measurement, a probe mask (or, when YOLO cannot detect the point-blank
probe at all, a raw-depth cloud anchored at the four-bar grasp contact) is
fitted from scratch (PCA-initialised box ICP) and, once consecutive fits
agree, replaces the attached pose outright.

The probe's 180° end-for-end orientation is invisible to a symmetric box fit,
so it is resolved separately from shape: probe.stl has a fat body at low STL-Z
and a tapered tip at high STL-Z, and the measured cloud's per-half radial
width profile (`axis_half_widths`) is matched against that. When the cloud
covers both ends decisively, the mesh is flipped to point the tip the right
way — this correction bypasses the deadband (an end flip reads as 0° in the
symmetric axis metric) and takes the fast commit path.

For an over-length probe, the base-box release first screens the ordinary
leaning, probe-centre insertion poses with collision-aware IK. If all are
unreachable it switches immediately to a tip-only fallback instead of spending
up to a minute planning those rejected poses: the signed STL geometry locates
the tapered endpoint, targets only that point 20 mm below the rim, and gives
each reachable candidate a three-second planning budget. Measured release
verification requires the tip to be inside the usable opening while both the
probe centre and planning link remain above the rim. In insertion mode the
state machine never escalates to the old position-only probe-centre goal, so a
failed fallback stops safely with the gripper closed.

The relaxed-orientation and position-only rounds remain available for
non-insertion overhead-drop mode. Their release verification applies
`base_box_release_axis_tolerance_deg` to oriented rounds and
`base_box_release_axis_tolerance_final_deg` to the position-only round.

## Tests

```bash
colcon test --packages-select aries_vision_grasp
# or directly:
python3 -m pytest src/aries_vision_grasp/test/
```

The tests pin the four-bar calibration table (gap↔q roundtrips, the 45 mm
probe anchor point, contact offsets), the quaternion/rotation helpers, and the
point-to-box probe fit (pose recovery from a perturbed prior, outlier
trimming, closest-surface projection).
