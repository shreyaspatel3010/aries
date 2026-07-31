# Aries Bringup Launch Guide

This package contains wrapper launch files for common Aries bringup workflows.
Use this guide as a quick operational reference for simulation, hardware, and
gripper bringup.

## Prerequisites

Run this in every new terminal before launching:

```bash
cd ~/aries
source install/setup.bash
```

## Quickstart

Recommended launch order based on workflow.

### Full Rover + Gazebo + MoveIt (most common)

1. Terminal 1:
   ```bash
   ros2 launch aries_bringup my_robot.launch.py use_joystick:=true
   ```

### Arm Hardware (igus Rebel over Ethernet)

1. Terminal 1:
   ```bash
   ros2 launch aries_bringup igus_rebel_hardware.launch.py use_joystick:=true
   ```

#### Move the arm physically by hand

The real-arm launch exposes a coordinated hand-guiding service. Before enabling
it, remove the payload if possible, clear the workspace, keep the emergency stop
within reach, and support the arm so it cannot fall when motor torque is removed.

Enable hand guiding:

```bash
ros2 service call /arm/set_hand_guiding std_srvs/srv/SetBool "{data: true}"
```

With the normalized joystick running, you can instead hold Y to enter hand
guiding and release Y to restore normal control. If joystick messages stop while
Y is held, a watchdog requests torque restoration after two seconds.

Move the arm slowly by hand, staying away from joint limits. When it is in the
desired pose, keep holding it and restore normal control:

```bash
ros2 service call /arm/set_hand_guiding std_srvs/srv/SetBool "{data: false}"
```

The service stops the arm trajectory controller before entering the Rebel CRI
`ZeroTorque` mode. On exit, it disables `ZeroTorque`, re-enables the motors, and
starts a fresh trajectory controller so an old trajectory cannot resume.

If the request says that `ZeroTorque` is unavailable, enable zero-torque mode in
the Rebel/iRC robot configuration and verify that the installed motor modules
support torque mode. A raw all-motors-disabled fallback is intentionally not used
because the ReBeL's spring-applied electromagnetic brakes engage without motor
voltage, so disabling the drives locks the joints instead of making them
backdrivable.

The iRC documentation does not define a separate physical teach/freedrive mode.
It calls powered movement with the software controls or gamepad followed by
recording a pose "teach-in programming"; `ZeroTorque` is the documented manual
guidance mode.

Use iRC **File -> Configuration Files**, load the robot configuration from the
robot, save a backup, and inspect its `ZeroTorque` entry. Each intended axis must
be enabled, for example `A1="true"` through `A6="true"`. Write the configuration
back to the robot and restart the controller. On an embedded controller the same
file is under
`~/RobotControl/Data/Robots/<category>/<robot type>/<robot type>.xml`, but the
iRC workflow is preferred. Only enable axes that igus confirms can be safely used
without torque. Joint-module firmware must report product `0x42`, firmware
`0x05 0x03` or newer.

### Gripper Hardware Only (Teensy over USB serial)

1. Terminal 1:
   ```bash
   ros2 launch aries_bringup gripper_hardware.launch.py
   ```

### Full Hardware (Arm + Gripper + Rover + MoveIt)

1. Terminal 1:
   ```bash
   ros2 launch aries_bringup full_hardware.launch.py
   ```

The full hardware launch also auto-detects the SICK picoScan150. It publishes
the cleaned scan on `/scan`, raw driver data under `/picoscan`, and uses the
picoScan yaw-rate IMU as a localization fallback when neither the YaBoom IMU
nor the serial BNO055 is available. Override the link-local addresses when
needed:

```bash
ros2 launch aries_bringup full_hardware.launch.py \
  use_rover_lidar:=true \
  rover_lidar_sensor_ip:=169.254.136.69 \
  rover_lidar_host_ip:=169.254.180.121
```

LiDAR-only bringup:

```bash
ros2 launch aries_lidar lidar.launch.py use_lidar:=true
```

### Arm Simulation Only

1. Terminal 1:
   ```bash
   ros2 launch aries_bringup igus_rebel_simulated.launch.py use_joystick:=true
   ```

Joystick controls:
- Hold RB for Cartesian/Twist movement, or RT for Chain/JointJog. Either one
  also enables arm/gripper joystick output; holding both gives Cartesian.
- On the real arm, hold Y for physical hand guiding; release Y to restore torque.
- Servo output is filtered by `servo_collision_guard` before it reaches the arm controller.
- Release RB and RT to stop arm joystick commands.
- Hold X to manually open the gripper; hold B to manually close it.
- Release X/B to hold the current gripper angle. Press A to toggle full open/close.
- Use `joystick_control_mode:=move_group` only if you want the planned MoveGroup
  joystick backend.
- Joystick launch uses `joy_driver:=game_controller_node` by default and then
  normalizes the result back to the existing Xbox-style `gamepad.yaml` mapping.
  If you need the older raw Linux driver for testing, launch with
  `joy_driver:=joy_node joy_layout:=bluetooth`.

### RViz Display Only

1. Terminal 1:
   ```bash
   ros2 launch aries_bringup display.launch.xml
   ```

## Command Reference

- Rover + Gazebo + MoveIt: `ros2 launch aries_bringup my_robot.launch.py use_joystick:=true`
- Display-only RViz: `ros2 launch aries_bringup display.launch.xml`
- Simulated arm: `ros2 launch aries_bringup igus_rebel_simulated.launch.py use_joystick:=true`
- Arm hardware: `ros2 launch aries_bringup igus_rebel_hardware.launch.py use_joystick:=true`
- Gripper hardware: `ros2 launch aries_bringup gripper_hardware.launch.py`
- Full hardware: `ros2 launch aries_bringup full_hardware.launch.py`

## Rover CAN Setup

Bringup launch files do not run sudo CAN setup by default. Configure CAN
manually before launching real rover drive:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 250000
```

Then launch normally. CAN setup is automatic by default:

```bash
ros2 launch aries_bringup rover_drive.launch.py can_interface:=can0
```

All CAN-related Aries bringup launches use automatic CAN setup by default:

```bash
ros2 launch aries_bringup rover_drive.launch.py
ros2 launch aries_bringup rover_drive_auto.launch.py
ros2 launch aries_bringup full_hardware.launch.py
```

The fail-safe ODrive bridge starts disarmed by default. After the hardware
check passes and the rover is clear:

```bash
ros2 service call /aries_drive/enable std_srvs/srv/SetBool "{data: true}"
```

It stops on stale `/cmd_vel`, clamps chassis and wheel speeds, ramps commands,
and confirms all six axes reached closed-loop control before allowing motion.

Disable automatic CAN setup only when you already configured CAN yourself:

```bash
ros2 launch aries_bringup rover_drive.launch.py setup_can:=false
ros2 launch aries_bringup full_hardware.launch.py setup_rover_can:=false
```

To use automatic CAN setup without a password prompt, install the limited
sudoers rule:

```bash
sudo visudo -cf src/aries_drive/setup/rover_can
sudo install -m 440 src/aries_drive/setup/rover_can /etc/sudoers.d/rover_can
```

It only permits:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 250000
```

## Teensy Serial Troubleshooting

Use this section when `gripper_hardware.launch.py` or `aries_hardware.launch.py`
fails to connect to the gripper controller.

### 1) Check that Teensy is detected

```bash
ls -l /dev/serial/by-id
ls -l /dev/ttyACM*
```

Expected: a Teensy entry such as
`/dev/serial/by-id/usb-Teensyduino_USB_Serial_... -> ../../ttyACM0`.

### 2) Check permissions

```bash
groups
```

If `dialout` is missing:

```bash
sudo usermod -aG dialout $USER
```

Then log out and log back in.

### 3) Override serial port explicitly

For gripper-only launch:

```bash
ros2 launch aries_bringup gripper_hardware.launch.py \
  serial_port:=/dev/serial/by-id/<your_teensy_id>
```

For full hardware launch:

```bash
ros2 launch aries_bringup aries_hardware.launch.py serial_port:=/dev/ttyACM0
```

Note:
- `gripper_hardware.launch.py` default is a `/dev/serial/by-id/...` path.
- `aries_hardware.launch.py` default is `/dev/ttyACM0`.

### 4) If the port keeps changing

- Prefer `/dev/serial/by-id/...` instead of `/dev/ttyACM*`.
- Replug Teensy and re-run `ls -l /dev/serial/by-id`.
- Close other serial tools (for example Arduino Serial Monitor) that may lock
  the port.

### 5) Quick ROS checks

```bash
ros2 control list_controllers
ros2 topic list | grep -E "gripper|controller_manager"
```
