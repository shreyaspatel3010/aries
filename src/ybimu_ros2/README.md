# YaBoom 10-axis IMU

Standalone ROS 2 Python package for YaBoom's 10-axis serial IMU. The driver
implements the protocol from YaBoom's `YbImuLib`, so installing that separate
vendor library is not required.

## Reuse in another workspace

Copy this complete `ybimu_ros2` directory into the other workspace's `src`
directory, then install dependencies and build it:

```bash
cd /path/to/other_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select ybimu_ros2
source install/setup.bash
```

Run with the included default configuration:

```bash
ros2 launch ybimu_ros2 ybimu.launch.py
```

The commonly changed settings are launch arguments, so another robot can use
the package without editing it:

```bash
ros2 launch ybimu_ros2 ybimu.launch.py \
  port:=/dev/imu_ybimu frame_id:=imu_link imu_topic:=/imu \
  report_rate_hz:=100 orientation_mode:=planar_gyro_mag \
  publish_linear_acceleration:=false
```

For additional parameters, copy `config/ybimu.yaml`, edit the copy, and pass
`config_file:=/absolute/path/to/ybimu.yaml`.

## Interface

- UART/USB serial at 115200 baud
- Persistent device path: `/dev/imu_ybimu`
- `/imu` (`sensor_msgs/Imu`): orientation and bias-corrected gyro in rad/s;
  acceleration is unavailable by default
- `/imu/mag` (`sensor_msgs/MagneticField`): magnetic field in tesla
- `/imu/pressure` (`sensor_msgs/FluidPressure`): pressure in pascal
- `/imu/temperature` (`sensor_msgs/Temperature`): temperature in °C
- `/imu/altitude` (`std_msgs/Float64`): vendor barometric altitude in metres

YaBoom's example ROS node forwards acceleration in g and magnetometer data in
µT. When enabled, this driver converts both to the SI units required by ROS.

## Datasheet compatibility

The supplied 10-axis datasheet matches the conversions used by this package:

| Item | Datasheet | Driver |
| --- | --- | --- |
| UART | 115200 bit/s | 115200 baud |
| Output rate | 25 Hz default, adjustable 10--100 Hz | 100 Hz configured; parameter accepts 10--100 Hz |
| Accelerometer | +/-16 g, about 0.0005 g/LSB | +/-16 g converted to m/s^2 |
| Gyroscope | +/-2000 deg/s, 0.061 deg/s/LSB | converted to rad/s |
| Magnetometer | +/-8 gauss, 0.244 mG/LSB | converted to tesla |
| Barometer | 300--2000 hPa | hPa or Pa firmware values normalized to ROS pascals |
| Cold startup | 5000 ms | 6 s startup grace before the 1 s stream timeout |

The 10-axis model consists of the 9-axis accelerometer/gyro/magnetometer set
plus a barometer. Its fused pitch/roll/yaw and quaternion outputs are available
in `vendor_fused` mode; planar mode deliberately replaces that fused attitude
to avoid acceleration-induced tilt.

The board is approximately 24.5 x 31 mm, with 25.2 x 17 mm mounting-hole
spacing and four 2.2 mm holes. The serial header is labelled RX, TX, GND, 5V;
the I2C header is SDA, SCL, GND, 3V3 and the datasheet specifies 100 kHz I2C.
Do not power the board from more than one connector at the same time.

The configured 10--120 microtesla magnetic-validity window is intentionally
narrower than the sensor's +/-800 microtesla measurement range. It represents
a plausible Earth field for heading correction and rejects nearby motors,
steel, magnets, and high-current wiring; it is not a limitation of the sensor.

## Orientation modes

The default `planar_gyro_mag` mode is intended for ground robots. It publishes
a yaw-only quaternion derived from Z angular velocity with slow magnetometer
correction. Linear acceleration is never an input, so translating the IMU does
not appear as roll or pitch. The driver:

- estimates gyro bias during the first three stationary seconds;
- rejects magnetic magnitudes outside the configured valid range;
- rejects implausibly large magnetic heading corrections;
- publishes all three bias-corrected angular velocities;
- continues publishing magnetometer, pressure, temperature, and altitude; and
- marks `linear_acceleration_covariance[0] = -1` when acceleration is disabled.

Keep the IMU still for the startup bias-calibration interval. A warning is
printed while calibration is active.

Use `orientation_mode:=vendor_fused` to restore YaBoom's full 3D quaternion.
That quaternion uses acceleration internally and can therefore tilt during
linear acceleration. Set `publish_linear_acceleration:=true` only when a
consumer actually needs the acceleration measurement.

## Persistent device name

Prefer a persistent symlink over a bare `/dev/ttyUSB0`, especially on robots
with multiple serial devices. With the IMU connected, identify its attributes:

```bash
udevadm info --attribute-walk --name=/dev/ttyUSB0
```

Create `/etc/udev/rules.d/99-ybimu.rules`, replacing the placeholder values
with the IMU adapter's actual values. Include `serial` when the adapter exposes
one:

```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="XXXX", ATTRS{idProduct}=="YYYY", ATTRS{serial}=="SERIAL", SYMLINK+="imu_ybimu", GROUP="dialout", MODE="0660"
```

Then reload the rule and reconnect the IMU:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/imu_ybimu
```

The user running ROS must belong to `dialout`.

## Calibration

Keep the robot stationary, then calibrate accelerometer and gyroscope:

```bash
ros2 service call /ybimu/calibrate_imu std_srvs/srv/Trigger
```

For the magnetometer, move the robot/sensor through figure-eight motions and
orientations after calling:

```bash
ros2 service call /ybimu/calibrate_magnetometer std_srvs/srv/Trigger
```

Calibration is stored by the IMU firmware. Keep it away from VESC power wiring,
motors, and steel during magnetometer calibration.

## Verification

```bash
ros2 topic hz /imu
ros2 topic echo /imu --once
ros2 topic echo /imu/pressure --once
```

After startup calibration, angular velocity should be close to zero and the
quaternion norm close to one. In planar mode quaternion X and Y remain zero.
Mount the marked sensor axes as ROS FLU: X forward, Y left, Z up. If the board
is physically rotated, describe that fixed rotation in the `imu_link` URDF
joint; the TF broadcaster offsets are intended only for visualization.

Parameters are in `config/ybimu.yaml`. The driver retries missing or
disconnected devices automatically, so reconnecting USB does not require
restarting the launch.
