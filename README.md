# Aries Mars Rover Workspace

ROS 2 workspace for the Aries Mars rover: a 6-wheel differential-drive rover
base with an igus ReBeL 6-DOF arm, gripper, MoveIt 2 integration, Gazebo
simulation, ODrive/CAN rover hardware support, joystick bringup, and two
autonomous manipulation tasks — vision-guided probe grasping and terrain-guided
soil sampling.

## Repository Layout

```text
aries/
├── assets/                Source art, terrain sources, and mesh alternatives
├── docs/                  Operator and subsystem guides, demo captures
├── firmware/              Microcontroller sketches (not colcon packages)
├── scripts/vision/        Vision environment and model utilities
├── src/                   First-party ROS 2 packages
│   ├── aries/              Robot description, worlds, Gazebo launch
│   ├── aries_bringup/      Recommended launch wrappers and hardware checker
│   ├── aries_common/       Shared hardware auto-detection
│   ├── aries_drive/        Fail-safe cmd_vel-to-ODrive backend
│   ├── aries_imu/          YaBoom / BNO055 / picoScan selection
│   ├── aries_lidar/        SICK picoScan150 driver wrapper
│   ├── aries_localization/ Wheel odometry and EKF orchestration
│   ├── aries_moveit/       Arm, gripper, and MoveIt packages
│   ├── aries_soil_sample/  Autonomous soil scooping and deposit
│   ├── aries_teleop/       Rover joystick normalization and Twist output
│   ├── aries_vision_grasp/ Vision grasp nodes, launch, and model
│   ├── rover_nav/          Rover odometry and localization
│   ├── ybimu_ros2/         YaBoom 10-axis serial IMU driver
│   └── vendor/             Vendored upstream ROS dependencies
└── workspace.repos        Optional vcstool dependency manifest
```

Generated `build/`, `install/`, and `log/` trees, Python environments, editor
state, training runs, and datasets are machine-local and intentionally ignored.
Local archives and downloaded weights belong in `artifacts/`; local training
datasets and runs belong in `data/`.

## Packages

- `aries`: main robot description, Gazebo launch files, sensors, and base model.
- `aries_bringup`: recommended launch wrappers for simulation, hardware, rover drive, joystick, and hardware checking.
- `aries_drive`: fail-safe `/cmd_vel` conversion, ODrive arming, command timeout, acceleration limiting, and mock fallback.
- `aries_lidar`: real SICK picoScan150 driver wrapper and `/scan` relay.
- `aries_imu`: YaBoom, BNO055, and picoScan IMU selection.
- `aries_localization`: physical wheel-odometry/EKF and simulation ground-truth/EKF orchestration.
- `aries_teleop`: rover joystick normalization and `/cmd_vel/teleop` output.
- `ybimu_ros2`: YaBoom 10-axis serial driver with planar gyro/magnetometer filtering.
- `aries_common`: shared hardware auto-detection used by rover launch files.
- `aries_moveit`: MoveIt 2 configuration, arm/gripper controllers, Servo teleop, and gripper hardware plugins.
- `aries_vision_grasp`: camera tools, YOLO inference, and autonomous MoveIt grasping of the probe.
- `aries_soil_sample`: autonomous soil scooping from the ground with the bucket fingertip, and deposit into the rover box. Terrain geometry only, no trained model.
- `rover_nav`: rover odometry, localization configs, and legacy rover navigation/control scripts.

Vendored packages live under `src/vendor/`; colcon discovers packages
recursively, so this grouping does not change package or launch names.

## Requirements

- ROS 2 Jazzy or Humble
- Gazebo Harmonic or newer for simulation
- MoveIt 2
- `ros2_control`
- `joy` package for gamepad input
- `odrive_can` for real rover ODrive/CAN hardware
- `sick_scan_xd` for the real SICK picoScan150 LiDAR
- `realsense2_camera` only when using the RealSense camera
- `ultralytics` (pulls in `torch`) for `aries_vision_grasp`. There is no rosdep
  key for it, so install it with pip into the same Python environment the nodes
  run under. `aries_soil_sample` needs neither, and no GPU.

The vision Python environment runs NumPy 2.x, where `cv_bridge` (built against
the 1.x ABI) segfaults. Use `aries_vision_grasp.image_bridge.NumpyImageBridge`
for image conversion instead.

## Build

Run these commands from a new terminal:

```bash
cd ~/aries
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src -r -y
colcon build --symlink-install
source install/setup.bash
```

Use your installed ROS distro in place of `jazzy` if needed.

## Autonomous Manipulation

Two autonomous tasks share the arm, the gripper, and the wrist camera. They
solve different perception problems: the probe is an *object* and needs a
trained detector, while soil is *terrain* and is located by depth geometry
alone.

Both launches default to `use_sim_time:=false`, because standalone use means
the physical rover and a `true` default with no `/clock` publisher freezes
every timer. **Against Gazebo you must pass `use_sim_time:=true` explicitly.**
Omitting it leaves the node comparing wall-clock `now()` against sim-stamped
camera frames, so every frame is dropped as too old — an age near the Unix
epoch in the logs is always this clock-domain mismatch, never a real delay.

### Vision Grasp (Probe)

Vision setup and operation are documented under [`docs/vision`](docs/vision),
with package detail in
[`src/aries_vision_grasp/README.md`](src/aries_vision_grasp/README.md). The
maintained utilities are in `scripts/vision/`, and the default model is
installed from `src/aries_vision_grasp/models/grasp.pt` with the
`aries_vision_grasp` package.

```bash
./scripts/vision/install_dependencies.sh
source scripts/vision/setup_environment.bash
```

Run the pipeline against the simulator. The default `sandbox_world.sdf` is the
one with a probe planted in it:

```bash
ros2 launch aries_bringup my_robot.launch.py

# In a second terminal:
ros2 launch aries_vision_grasp vision_grasp.launch.py use_sim_time:=true
```

The node detects the probe, plans and executes the grasp, verifies it, then
carries the probe to the rover's base-mounted box and releases it. While the
probe is held, its attached collision mesh is continuously re-fitted to live
perception rather than frozen at grasp time. Tuning lives in
`config/vision_grasp_params.yaml` and `config/pick_place.yaml`; override either
with `params_file:=` or `pick_place_config:=`.

On hardware, bring the arm, gripper, MoveIt, and the gripper RealSense up with
`aries_bringup aries_hardware.launch.py` first (`enable_depth_sensor:=auto` by
default), then start `vision_grasp.launch.py` alongside it. The standalone
annotated-detection node is debug-only — it duplicates inference the grasp node
already performs and publishes `/vision_grasp/detection_image` itself — so it is
gated behind `enable_yolo_debug:=true` on the hardware launch.

### Soil Sampling

Package detail is in
[`src/aries_soil_sample/README.md`](src/aries_soil_sample/README.md).

```bash
# Simulation: loose grains and a deposit box, no probe.
ros2 launch aries my_robot.launch.py world:=soil_world.sdf finger_type:=bucket

# In a second terminal:
ros2 launch aries_soil_sample soil_sample.launch.py use_sim_time:=true
```

The node does **not** scoop on launch — it drives a gripper into the ground, so
a cycle runs only when triggered (or with `auto_start:=true`):

```bash
ros2 service call /soil_sample_node/survey  std_srvs/srv/Trigger  # look, don't move
ros2 service call /soil_sample_node/dry_run std_srvs/srv/Trigger  # plan, don't execute
ros2 service call /soil_sample_node/scoop   std_srvs/srv/Trigger  # one full cycle
ros2 service call /soil_sample_node/abort   std_srvs/srv/Trigger
```

Sites are picked from a 2.5-D height map scored on roughness, slope, and
coverage, with roughness as the primary safety gate. Capture is verified by
re-surveying and differencing the height map — a scoop that collected soil
leaves a hole — because the gripper reports its command rather than a measured
position and MoveIt's self-filter blanks the near field inside the jaws.

Note that `octomap_disable_during_scoop` defaults to **true**. The octomap
models the ground the scoop exists to dig into, so a collision-free path into
it cannot exist; while suppressed, the arm will not avoid obstacles that exist
only in the octomap. See the package README for the safeguards that replace it.

## Firmware

Microcontroller sources are kept outside `src/` so colcon scans only ROS
packages:

- `firmware/teensy_gripper/teensy_gripper.ino`: current gripper controller.
- `firmware/legacy_controller/legacy_controller.ino`: legacy controller sketch.

## Quick Start

### Full Simulation

Gazebo rover simulation with MoveIt/RViz and joystick support:

```bash
ros2 launch aries_bringup my_robot.launch.py use_joystick:=true
```

If another node already publishes `/joy`, keep the default. If you need this
launch file to start a separate rover `joy_node`, add:

```bash
use_rover_joy_node:=true
```

#### Worlds And Fingertips

The bringup wrapper does not forward the world or fingertip arguments, so
choose them on the `aries` launch directly:

```bash
ros2 launch aries my_robot.launch.py \
  world:=sandbox_world.sdf \
  finger_type:=bucket
```

- `world:=` accepts `sandbox_world.sdf` (default; probe planted in a sand box,
  for the grasp task) or `soil_world.sdf` (loose grains and a deposit box, for
  the soil task).
- `finger_type:=` accepts `bucket` (default), `maintenance`, or `probe`. It must
  match the jaw the task expects: the four-bar contact point differs by up to
  23 mm between the three fingertips, which is a 23 mm depth error on every
  approach.

### Full Hardware

Arm, gripper, MoveIt/RViz, rover drive, auto mock fallback, and full hardware
checker:

```bash
ros2 launch aries_bringup full_hardware.launch.py
```

Useful overrides:

```bash
ros2 launch aries_bringup full_hardware.launch.py \
  use_joystick:=true \
  rover_hardware_protocol:=auto \
  can_interface:=can0 \
  use_rover_lidar:=auto \
  serial_port:=/dev/ttyACM0
```

`rover_hardware_protocol:=auto` uses real ODrive/CAN when `can0` exists and
falls back to `mock_rover_drive` when rover hardware is unavailable.
This top-level launch runs rover CAN setup automatically by default.

The physical ODrive bridge starts disarmed. After the hardware check passes and
the rover is clear, enable it explicitly:

```bash
ros2 service call /aries_drive/enable std_srvs/srv/SetBool "{data: true}"
```

## SICK picoScan150 LiDAR

Real rover launches now auto-detect the picoScan on its SOPAS TCP port. When
available, `sick_scan_xd` publishes raw data under `/picoscan`, the Aries relay
publishes the cleaned planar scan on `/scan`, and the built-in IMU is available
as `/picoscan/imu`. Auto selection prefers a YaBoom IMU at
`/dev/imu_ybimu`, then the existing BNO055, and finally the picoScan IMU.

The default rover network values are:

```text
sensor: 169.254.136.69
host:   169.254.180.121
```

Override or force them with:

```bash
ros2 launch aries_bringup full_hardware.launch.py \
  use_rover_lidar:=true \
  rover_lidar_sensor_ip:=169.254.136.69 \
  rover_lidar_host_ip:=169.254.180.121
```

Run only the sensor wrapper with:

```bash
ros2 launch aries_lidar lidar.launch.py use_lidar:=true
```

Use `use_rover_lidar:=false` to disable it. Only the EKF publishes
`odom -> base_footprint`; the picoScan IMU integration does not publish a
competing odometry transform.

### Arm And Gripper Hardware Only

Arm/gripper hardware with MoveIt/RViz:

```bash
ros2 launch aries_bringup aries_hardware.launch.py use_joystick:=true
```

Arm only:

```bash
ros2 launch aries_bringup igus_rebel_hardware.launch.py use_joystick:=true
```

Gripper only:

```bash
ros2 launch aries_bringup gripper_hardware.launch.py
```

### Arm Simulation Only

```bash
ros2 launch aries_bringup igus_rebel_simulated.launch.py use_joystick:=true
```

### Rover Drive Only

Real rover ODrive/CAN drive:

```bash
ros2 launch aries_bringup rover_drive.launch.py
```

Auto real-or-mock rover drive:

```bash
ros2 launch aries_bringup rover_drive_auto.launch.py use_joy_node:=true
```

Force mock rover drive for RViz/testing:

```bash
ros2 launch aries_bringup rover_drive_auto.launch.py \
  rover_hardware_protocol:=mock_hardware \
  use_joy_node:=true
```

### RViz Display Only

```bash
ros2 launch aries_bringup display.launch.xml
```

## Joystick Controls

The shared rover joystick config is:

```text
src/aries_bringup/config/joystick.yaml
```

The arm/gripper joystick config is:

```text
src/aries_moveit/moveit_config/config/gamepad.yaml
```

Xbox-style `/joy` mapping:

- A: button 0
- B: button 1
- X: button 2
- Y: button 3
- LB: button 4
- RB: button 5
- BACK: button 6
- START: button 7

Current operating mapping:

- Hold LB to drive the rover base.
- Left stick vertical drives rover forward/back.
- Left stick horizontal turns the rover.
- Hold RB to enable arm/gripper joystick output.
- Default `servo` mode uses the old smooth Cartesian/Twist joystick movement.
- Press RB to toggle arm mode between Cartesian/Twist and Chain/JointJog.
- Servo output is checked by MoveIt Servo and `servo_collision_guard` before it
  reaches the arm controller.
- Release RB to stop arm joystick commands.
- Hold X to manually open the gripper; release to hold that angle.
- Hold B to manually close the gripper; release to hold that angle.
- Press A to toggle full open/close.
- Release LB to stop rover commands.

LB is reserved for the rover. When LB rover drive is active, arm and gripper
joystick output is blocked so the same controller does not command both systems.

There is no timed 180-degree turn trigger. BACK, START, and the D-pad are not
used for an automatic 180-degree rover command.

## MoveIt And RViz

When using a MoveIt-enabled launch with RViz:

1. Open the MotionPlanning panel.
2. Select the `arm` or `gripper` planning group.
3. Set a goal with the interactive marker or joint targets.
4. Click `Plan`, then `Execute`.

The default joystick arm mode uses the old continuous MoveIt Servo Cartesian
teleop path for smooth XYZ/rotation movement. Servo commands pass through
MoveIt Servo collision checking and `servo_collision_guard` before reaching
`rebel_arm_trajectory_controller`.

For RViz planning/execution, release LB and RB so the joystick is not actively
sending arm or rover commands.

To use the newer planned MoveGroup joystick backend instead:

```bash
ros2 launch aries_bringup igus_rebel_hardware.launch.py \
  use_joystick:=true \
  joystick_control_mode:=move_group
```

In default Servo mode, joystick trajectories pass through a collision guard
before reaching the arm controller:

```text
MoveIt Servo -> /servo_guard/input_joint_trajectory -> servo_collision_guard -> rebel_arm_trajectory_controller
```

The guard checks each commanded arm trajectory against the MoveIt self-collision
model for `arm_with_gripper`. Commands that enter self-collision or move deeper
inside the safety margin are blocked and replaced with a hold command.

## Hardware Checker

The full hardware launch starts a checker by default. You can also trigger it
manually:

```bash
ros2 service call /check_full_hardware std_srvs/srv/Trigger
```

It checks:

- Arm TCP reachability and joint states
- Gripper serial/mock status
- Rover ODrive/CAN or mock rover fallback
- Joystick `/joy`
- Optional RealSense detection

## Rover CAN Setup

Bringup launches do not configure CAN with sudo by default. You can set CAN up
manually before launching the real rover drive:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 250000
```

Then launch rover drive. CAN setup is automatic by default:

```bash
ros2 launch aries_bringup rover_drive.launch.py can_interface:=can0
```

All CAN-related Aries bringup launches use automatic CAN setup by default:

```bash
ros2 launch aries_bringup rover_drive.launch.py
ros2 launch aries_bringup rover_drive_auto.launch.py
ros2 launch aries_bringup full_hardware.launch.py
```

Disable automatic CAN setup only when you already configured CAN yourself:

```bash
ros2 launch aries_bringup rover_drive.launch.py setup_can:=false
ros2 launch aries_bringup full_hardware.launch.py setup_rover_can:=false
```

To use automatic CAN setup without a password prompt, install the limited
sudoers rule provided by this workspace:

```bash
sudo visudo -cf src/aries_drive/setup/rover_can
sudo install -m 440 src/aries_drive/setup/rover_can /etc/sudoers.d/rover_can
```

That rule allows only these commands without a password:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 250000
```

Quick checks:

```bash
ip link show can0
ros2 topic list | grep odrive
ros2 topic echo /odrive_axis0/controller_status
```

## Teensy Gripper Serial

Check that the Teensy is detected:

```bash
ls -l /dev/serial/by-id
ls -l /dev/ttyACM*
```

If the user is not in the `dialout` group:

```bash
sudo usermod -aG dialout $USER
```

Then log out and log back in.

Prefer a stable `/dev/serial/by-id/...` path when possible:

```bash
ros2 launch aries_bringup gripper_hardware.launch.py \
  serial_port:=/dev/serial/by-id/<your_teensy_id>
```

## Troubleshooting

If a package or executable is not found:

```bash
cd ~/aries
colcon build --symlink-install
source install/setup.bash
```

If the joystick does not work:

```bash
ros2 topic echo /joy
```

If no `/joy` messages appear, start a launch with `use_joy_node:=true` or run:

```bash
ros2 run joy joy_node
```

If real rover hardware is not connected, use mock rover mode:

```bash
ros2 launch aries_bringup rover_drive_auto.launch.py \
  rover_hardware_protocol:=mock_hardware \
  use_joy_node:=true
```

If the default Servo joystick stops before an obstacle or self-collision, check:

```bash
ros2 topic echo /arm_joystick/status
ros2 topic echo /servo_node/status
```

`/arm_joystick/status` reports the reason a joystick command was blocked.

## Development Notes

Common package builds:

```bash
colcon build --packages-select aries_bringup rover_nav
colcon build --packages-select aries_moveit
```

Tests:

```bash
colcon test --packages-select aries_vision_grasp aries_soil_sample
# or directly, without a colcon overlay:
python3 -m pytest src/aries_vision_grasp/test/ src/aries_soil_sample/test/
```

The suites cover the pure-NumPy libraries that carry the manipulation geometry:
the four-bar jaw calibration, quaternion/rotation helpers, depth
back-projection, the point-to-box probe fit, and the terrain height map, scoop
waypoints, and deposit poses.

Direct MoveIt launch files still live under:

```text
src/aries_moveit/moveit_config/launch
```

Use `aries_bringup` launch files for normal operation because they wrap the
MoveIt, rover, joystick, mock fallback, and checker pieces together.

## Credits

Developed by Shreyas Patel.

## Demo


https://github.com/user-attachments/assets/8119749f-b088-4221-86b2-9a9341b05857


[simulation.webm](https://github.com/user-attachments/assets/fc0d2934-1fe8-4681-bb4a-53667f0bb681)

The same captures are tracked in the repository under
[`docs/`](docs): `simulation.webm`, and `automatic_pick_probe.mp4` showing a
full autonomous probe pick.
