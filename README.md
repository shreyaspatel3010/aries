# Aries Mars Rover Workspace

ROS 2 workspace for the Aries Mars rover: a 6-wheel differential-drive rover
base with an igus ReBeL 6-DOF arm, gripper, MoveIt 2 integration, Gazebo
simulation, ODrive/CAN rover hardware support, joystick bringup, and two
autonomous manipulation tasks — vision-guided probe grasping and terrain-guided
soil sampling.

## Operating it remotely

For field operation the rover and the base station each run one command, over a
Ubiquiti Rocket 5AC link. Setup — static addresses, radio settings, verification
— is **[FIELD_SETUP.md](FIELD_SETUP.md)**.

```bash
# ROVER
ros2 launch aries_comms rover_field.launch.py

# BASE STATION  — the joystick plugs in HERE
ros2 launch aries_comms base_station.launch.py
```

## Repository Layout

```text
aries/
├── assets/                Source art, terrain sources, and mesh alternatives
├── docs/                  Operator and subsystem guides, demo captures
├── firmware/              Microcontroller sketches (not colcon packages)
├── scripts/               Vision environment/model utilities, teleop measurement
├── src/                   First-party ROS 2 packages
│   ├── aries/              Robot description, worlds, Gazebo launch
│   ├── aries_bringup/      Recommended launch wrappers and hardware checker
│   ├── aries_common/       Shared hardware auto-detection and DDS transport
│   ├── aries_comms/        The field link, both ends: rover_field and base_station
│   ├── aries_drive/        Fail-safe cmd_vel-to-ODrive backend
│   ├── aries_imu/          MicroStrain 3DM-GX5-AHRS driver integration
│   ├── aries_localization/ Wheel odometry and EKF orchestration
│   ├── aries_moveit/       Arm, gripper, and MoveIt packages
│   ├── aries_soil_sample/  Autonomous soil scooping and deposit
│   ├── aries_teleop/       Rover joystick normalization and Twist output
│   ├── aries_vision_grasp/ Vision grasp nodes, launch, and model
│   ├── rover_nav/          Rover odometry and localization
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
- `aries_imu`: MicroStrain by HBK 3DM-GX5-AHRS driver integration, params, and udev rule.
- `aries_localization`: physical wheel-odometry/EKF and simulation ground-truth/EKF orchestration.
- `aries_teleop`: rover joystick normalization and `/cmd_vel/teleop` output.
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
- `microstrain_inertial_driver` for the 3DM-GX5-AHRS IMU
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

## Camera DepthClouds

RViz displays two independent colored `DepthCloud` views. Each camera uses its
own depth aligned to its own RGB image, CameraInfo, optical frame, and TF:

- `rover_DepthCloud`: `/camera/aligned_depth_to_color/image_raw`
- `gripper_DepthCloud`: `/gripper_camera/aligned_depth_to_color/image_raw`

The camera paths do not publish a `PointCloud2` or build an occupancy map.

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

MoveIt does not build a live occupancy map from the camera data. Soil safety is
provided by the task's height-map, roughness, slope, reach, and motion guards.

## Firmware

Microcontroller sources are kept outside `src/` so colcon scans only ROS
packages:

- `firmware/teensy_drill_sys/`: **the current firmware.** A PlatformIO project,
  not an Arduino sketch. One Teensy 4.1 runs the drill, the gripper servo and
  the mast stack light, over one micro-ROS serial link. Pin map and wiring
  notes in `firmware/teensy_drill_sys/PINOUT.md`; build and flash instructions
  in its README.
- `firmware/legacy/`: the retired `.ino` sketches, kept for reference and not
  flashed. `teensy_gripper.ino` was the gripper controller this replaced;
  `legacy_controller.ino` came before that.
- `firmware/teensy_science_sys/`: **the science board**, a SECOND Teensy on its
  own USB port with its own micro-ROS agent. Sensors only -- pH, soil moisture,
  TDS/EC, ORP, soil temperature, BME688 and SCD41 CO2 -- published as one
  Float32MultiArray that `aries_science` splits into named topics. The two
  boards cannot be merged: four of the science pins are the drill's motors and
  load cells. See `firmware/teensy_science_sys/PINOUT.md`.

Building the firmware needs the PlatformIO IDE extension for VS Code — no
manual library installs, no Arduino IDE, and no separate uploader. Everything
the project needs is pinned in its `platformio.ini`.

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
  serial_port:=/dev/ttyACM0
```

`rover_hardware_protocol:=auto` uses real ODrive/CAN when `can0` exists and
falls back to `mock_rover_drive` when rover hardware is unavailable.
This top-level launch runs rover CAN setup automatically by default.

Only one publisher may own the wheel joint states. `full_hardware.launch.py`
defaults `use_static_wheel_joint_publisher` to `false` because the rover's
encoder-backed publisher is running; the arm-only
`aries_hardware.launch.py` defaults its `use_wheel_joint_publisher` to `true`
so the wheels still have a pose when no rover stack is up. Set them the other
way only if you swap which side owns the joints.

The physical ODrive bridge starts disarmed. After the hardware check passes and
the rover is clear, enable it explicitly:

```bash
ros2 service call /aries_drive/enable std_srvs/srv/SetBool "{data: true}"
```

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
src/aries_teleop/config/joystick.yaml
```

Simulation overrides it with `src/aries_teleop/config/rover_cmd_vel_joystick.yaml`,
which exists only because Gazebo has no ODrive bridge for LB+Y to re-arm.

The arm/gripper joystick config is:

```text
src/aries_moveit/moveit_config/config/gamepad.yaml
```

Canonical Xbox-style `/joy` mapping:

- A: button 0
- B: button 1
- X: button 2
- Y: button 3
- LB: button 4
- RB: button 5
- BACK: button 6
- START: button 7

Those numbers hold regardless of which driver is running. `joy_driver` defaults
to `game_controller_node` (SDL ordering, where LB/RB are buttons 9/10 and the
D-pad is buttons 11-14), and `joy_layout_normalizer` republishes `/joy/raw` on
`/joy` in the canonical layout above. It also normalizes the analog triggers to
`0.0` released and `1.0` fully pressed, which the drivers otherwise disagree
about. Pick the conversion with `joy_layout:=`, which accepts `auto` (default),
`dongle`, `bluetooth`, `game_controller`, or `passthrough`.

Current operating mapping:

- Hold LB to drive the rover base.
- Left stick vertical drives rover forward/back.
- Left stick horizontal turns the rover.
- Hold RB for arm Cartesian movement. RB is also the arm/gripper enable.
- Hold RT for direct joint jog. RT is also the arm/gripper enable.
- RB and RT are independent holds, **not** a toggle. Holding both gives
  Cartesian — RB wins, so brushing the trigger mid-move cannot silently change
  what the sticks do. Release both to stop the arm.
- Hold X to manually open the gripper; release to hold that angle.
- Hold B to manually close the gripper; release to hold that angle.
- Press A to toggle full open/close.
- Hold RB + Y for physical hand guiding (ZeroTorque) on the real arm.
- Press LB + Y to re-initialise the ODrives: `clear_errors` on all six axes,
  then re-arm `CLOSED_LOOP_CONTROL`. Edge-triggered, so holding the pair fires
  once.
- Press Y on its own for sound.
- Release LB to stop rover commands.

Y is shared three ways and disambiguated by which shoulder button is held:
bare Y is sound, LB+Y is the rover's ODrive recovery, RB+Y is arm hand guiding.

LB is reserved for the rover. When LB rover drive is active, arm and gripper
joystick output is blocked so the same controller does not command both systems.

There is no timed 180-degree turn trigger. BACK, START, and the D-pad are not
used for an automatic 180-degree rover command.

`arm_toggle_mode: false` in `gamepad.yaml` is the legacy escape hatch: set it
true to restore the old single-button behaviour where RB alone gates the arm
and toggles Cartesian/joint, with RT ignored.

## MoveIt And RViz

When using a MoveIt-enabled launch with RViz:

1. Open the MotionPlanning panel.
2. Select the `arm` or `gripper` planning group.
3. Set a goal with the interactive marker or joint targets.
4. Click `Plan`, then `Execute`.

The default joystick arm mode (`joystick_control_mode:=servo`) is
`rebel_servo_teleop_gamepad`, which is the primary way the arm is driven. It
solves the Cartesian command itself with a damped-least-squares Jacobian and
publishes joint trajectories straight to the controller:

```text
/joy -> rebel_servo_teleop_gamepad (DLS IK + collision guard) -> rebel_arm_trajectory_controller
```

It does not go through MoveIt Servo. `servo_node` and `servo_collision_guard`
are still launched and still serve the keyboard teleop
(`rebel_servo_teleop_keyboard`), whose path is:

```text
MoveIt Servo -> /servo_guard/input_joint_trajectory -> servo_collision_guard -> rebel_arm_trajectory_controller
```

Both guards check each commanded arm trajectory against the MoveIt
self-collision model for `arm_with_gripper` — the arm chain **and** the gripper,
because MoveIt only reports a contact when at least one of the two links is
active in the requested group, and the fingers are not in `igus_rebel_arm`.
Commands that enter self-collision or move deeper inside the safety margin are
blocked and replaced with a hold command. When the arm is already touching in
the model, a slow command is allowed only if its preview measurably reduces
penetration, so a genuine escape stays available but tangential motion cannot
scrape along the rover.

Damping is ramped in only as the arm approaches a singularity
(`dls_sigma_threshold`), not applied at every pose. A constant lambda attenuates
each singular direction unevenly, which made a pure "straight up" push also
drift sideways; above the threshold the solve is undamped and tracks the
command exactly.

For RViz planning/execution, release LB, RB, and RT so the joystick is not
actively sending arm or rover commands. The teleop node deliberately goes
silent shortly after release so RViz and MoveIt can drive the same controller.

To use the planned MoveGroup joystick backend instead:

```bash
ros2 launch aries_bringup igus_rebel_hardware.launch.py \
  use_joystick:=true \
  joystick_control_mode:=move_group
```

### Teleop Tuning

Every speed and motion value is owned by
`src/aries_moveit/moveit_config/config/teleop_speeds.yaml`, which is loaded
after `gamepad.yaml` and overrides it. Retune there — editing the numbers in
`gamepad.yaml` has no effect. What `gamepad.yaml` still owns is the button/axis
mapping, topic names, joint limits, and frame names.

`linear_scale` and `angular_scale` set the Cartesian speed and therefore the
release overshoot; `max_joint_velocity` is only a ceiling and is rarely reached.
When the stick is released the arm keeps travelling for roughly 0.25 s no matter
what — transport delay plus the ReBeL's own firmware deceleration, neither
reachable from ROS — so overshoot is release speed times that time, and the only
lever is releasing at a lower speed. Do **not** raise the trajectory controller
`p` gain against it: measured on hardware, `p=3.0` was worse than `p=1.0`
(5.78 deg vs 3.15 deg overshoot), because more gain against ~75 ms of dead time
simply drives the loop under-damped. Several `teleop_speeds.yaml` keys are
marked dead in the file — they are read by no node and are kept only so the two
config files stay aligned.

Measure rather than guess with:

```bash
python3 scripts/measure_teleop_tracking.py
```

It records `/joy`, the commanded joint trajectory, and `/joint_states`, then
reports tracking lag and per-release overshoot on Ctrl-C. It is a workspace
script rather than an installed entry point, so run it by path. It publishes
nothing, so it cannot move the arm.

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

## New computer setup

Everything the stack needs from root is in one script. Run it once per machine,
as the account that will launch the robot (not with `sudo` — it needs to know
which account to grant the rules to):

```bash
./scripts/setup_system.sh
```

It installs:

- `/etc/sudoers.d/rover_can` — passwordless `ip link set <can> up/down` for the
  interface and bitrate in `src/aries_common/config/devices.yaml`, and nothing
  else. The drive stack needs it at launch and again whenever the CAN adapter is
  unplugged and plugged back in.
- `/etc/udev/rules.d/99-aries-realsense.rules` — keeps USB autosuspend from
  powering a D435i down mid-session.
- `/etc/udev/rules.d/99-aries-teensy.rules` — serial and flashing access to the
  Teensy gripper board.
- Membership of `dialout`, `plugdev` and `input`. **These only take effect after
  logging out and back in.**

It does not install apt packages or bring the CAN interface up; anything missing
is printed at the end with the exact command. Re-running is safe — every step is
idempotent and reports whether it changed anything.

```bash
./scripts/setup_system.sh --check      # verify only, change nothing
./scripts/setup_system.sh --dry-run    # print what would change
```

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

To use automatic CAN setup without a password prompt, run the one-time system
setup (see [New computer setup](#new-computer-setup)):

```bash
./scripts/setup_system.sh
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

## IMU (MicroStrain 3DM-GX5-AHRS)

The rover carries one IMU: a MicroStrain by HBK 3DM-GX5-AHRS. It is an attitude
and heading reference, so its onboard estimation filter supplies absolute roll,
pitch and heading rather than rates alone, and the EKF fuses yaw directly.

No udev rule ships in this repo. `ros-jazzy-microstrain-inertial-driver` already
installs `60-ros-jazzy-microstrain-inertial-driver.rules`, which creates
`/dev/microstrain_main` — that is the path the whole stack uses. Note it is
*not* `/dev/microstrain`: the rule that would create that plain symlink is
commented out upstream, because it gets overridden when several devices are
attached.

```bash
sudo apt install ros-jazzy-microstrain-inertial-driver
ls -l /dev/microstrain_main    # -> ../../ttyACM0
```

The driver starts automatically when that device node exists and the driver
package is installed; otherwise the stack falls back to a wheel-odometry-only
EKF. Topics land under the `microstrain` namespace, so the physical IMU never
collides with Gazebo's simulated `/imu`:

```bash
ros2 topic echo /microstrain/imu/data     # orientation + rates, 100 Hz
ros2 topic echo /microstrain/ekf/imu/data # estimation filter, tighter covariance
```

Run only the IMU, or force/disable it:

```bash
ros2 launch aries_imu imu.launch.py
ros2 launch aries_bringup rover_drive.launch.py use_imu:=false
```

Two settings in `src/aries_imu/config/microstrain.yaml` are load-bearing and
should not be flipped without reading why:

- `use_enu_frame: true` — the device talks NED natively and the driver default
  is NED, which would silently invert yaw for `robot_localization` and TF.
- `publish_mount_to_frame_id_transform: false` — `base_link -> imu_frame` is a
  fixed joint in `aries_base.xacro`, so `robot_state_publisher` owns it. The
  driver default is `true` and would publish a competing transform.

To fuse the estimation-filter output instead of the raw IMU topic, repoint the
whole stack with one argument:

```bash
ros2 launch aries_bringup rover_drive.launch.py \
  imu_topic:=/microstrain/ekf/imu/data
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

If the joystick arm stops before an obstacle or self-collision, check:

```bash
ros2 topic echo /arm_joystick/status
```

`/arm_joystick/status` reports the reason a joystick command was blocked, and it
is the right topic for the default gamepad path. `/servo_node/status` only
describes the MoveIt Servo chain, which the gamepad does not use — check it when
debugging the keyboard teleop.

Camera depth is visualized through the two independent DepthCloud displays; it
is not inserted into MoveIt's collision world.

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

### Robot Model Notes

- Arm joint limits are the REBEL-6DOF-03 factory software limits (179, 80/140,
  80/140, 179, 95, 179 degrees). They are stated in three places — the URDF,
  `moveit_config/config/joint_limits.yaml`, and `gamepad.yaml` — and must be
  kept in sync.
- `base_link_height` in `common_properties.xacro` is derived from
  `wheel_radius` rather than hard-coded, so `base_footprint` sits exactly on
  the wheel contact plane. The old hard-coded 0.165 left the wheels 41 mm below
  it, and everything treating that frame as ground (EKF,
  `odom -> base_footprint`, nav ground filtering) inherited the
  error.
- Arm visuals are `.glb`, and the gripper carries the near plates and pivot
  pins of the double-plate four-bar as visual-only geometry. Superseded meshes
  are archived under `src/aries/meshes/unused/` rather than deleted.
- `finger_type` must match the jaw the task expects; the four-bar contact point
  differs by up to 23 mm between the three fingertips.

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
