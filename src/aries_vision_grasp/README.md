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

Gripper open, close, pre-close, joint limits, and the hard-stop safety margin
are configured in `config/pick_place.yaml`. The nominal `-1.57` open and `0.07`
closed commands are preserved; the Gazebo-only URDF limits include additional
travel so those commands do not coincide with a physics hard stop.

Probe acquisition filtering is configured in the same YAML under
`target_stability_*`, `target_filter_*`, and `target_lock_min_confidence`. The
committed grasp point is the median of a consecutive high-confidence 3D cluster;
large jumps reset acquisition instead of moving the robot target.

The default model is installed as `models/grasp.pt`. It is the model previously
named `best(1).pt`: a YOLO26l segmentation model trained for the `probe` class.
The old `best(2).pt` is a different, smaller detection-only model and is not the
default. Override `model_path` when testing other weights.

After a verified grasp, the hardware launch carries the probe through
`pick_home`, moves to the calibrated `pick_drop` posture above the rover's
base-mounted box, releases it, and returns the empty arm to `pick_home`. The
behavior is controlled by `place_in_base_box_after_grasp`; tune
`base_box_drop_joint_positions` if the physical box position changes.

Home and box-placement postures are configured in
`config/pick_place.yaml`. All joint positions are radians. The file contains:

- `pick_home_joint_positions`: transport home used before and after placement.
- `base_box_drop_xyz`: physical release point in metres, relative to `base_link`.
- `base_box_drop_rpy`: gripper orientation in radians (`roll, pitch, yaw`).
- `base_box_drop_joint_positions`: optional joint fallback when pose mode is off.
- `retreat_home_joint_positions`: fallback/recovery home posture.

RViz shows a persistent magenta `BASE BOX DROP` marker and red/green/blue XYZ
axes on `/vision_grasp/markers`. Set `base_box_drop_use_pose: true` to plan to
the configured XYZ/RPY pose, or `false` to use the joint fallback.

The same YAML configures gripper completion. With
`gripper_require_feedback_for_completion: true`, opening and closing advance
only after the configured command time has elapsed, the trajectory controller
reports action success, **and** a fresh `/joint_states` gripper position is
within `gripper_goal_tolerance`. Controller rejection/abort and timeout do not
count as success.

Arm stages use the same rule across the full process. A MoveIt or Cartesian
action result must succeed, then fresh measured arm joints/TF must remain at
the commanded target for `arm_feedback_stable_samples` before the next arm or
gripper stage can start. Executed Cartesian paths always request collision
checking, attached-probe geometry remains active during transport, and
overlapping arm/gripper commands are rejected.

The launch file loads this configuration automatically. To use another file:

```bash
ros2 launch aries_vision_grasp vision_grasp.launch.py \
  pick_place_config:=/absolute/path/to/pick_place.yaml
```
