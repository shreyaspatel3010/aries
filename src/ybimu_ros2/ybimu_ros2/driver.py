"""ROS 2 driver for the YaBoom 10-axis serial IMU."""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure, Imu, MagneticField, Temperature
import serial
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from ybimu_ros2.planar_filter import PlanarGyroMagFilter
from ybimu_ros2.protocol import YbImuProtocol


class YbImuDriver(Node):
    """Publish SI-correct ROS messages and recover automatically after hot-plug."""

    def __init__(self) -> None:
        super().__init__('ybimu')

        self.declare_parameter('port', '/dev/imu_ybimu')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('magnetic_field_topic', '/imu/mag')
        self.declare_parameter('pressure_topic', '/imu/pressure')
        self.declare_parameter('temperature_topic', '/imu/temperature')
        self.declare_parameter('altitude_topic', '/imu/altitude')
        # Datasheet maximum; the config file is authoritative for the robot.
        self.declare_parameter('report_rate_hz', 100)
        self.declare_parameter('fusion_axes', 9)
        self.declare_parameter('orientation_mode', 'planar_gyro_mag')
        self.declare_parameter('publish_linear_acceleration', False)
        self.declare_parameter('estimate_gyro_bias', True)
        self.declare_parameter('gyro_bias_stillness_rad_s', 0.08)
        self.declare_parameter('gyro_noise_rad_s', 0.003)
        self.declare_parameter('gyro_bias_walk_rad_s', 0.0002)
        self.declare_parameter('gyro_bias_initial_uncertainty_rad_s', 0.5)
        self.declare_parameter('gyro_bias_confident_rad_s', 0.001)
        self.declare_parameter(
            'gyro_bias_temperature_sensitivity_rad_s_per_c', 0.002)
        self.declare_parameter('gyro_bias_max_corrected_rate_rad_s', 0.02)
        self.declare_parameter('gyro_bias_relock_s', 60.0)
        self.declare_parameter('magnetic_correction_time_constant_s', 5.0)
        self.declare_parameter('magnetic_field_min_t', 10.0e-6)
        self.declare_parameter('magnetic_field_max_t', 120.0e-6)
        self.declare_parameter('magnetic_norm_tolerance', 0.25)
        self.declare_parameter('magnetic_heading_rejection_deg', 45.0)
        self.declare_parameter('magnetic_max_rate_deg_s', 30.0)
        self.declare_parameter('estimate_hard_iron', True)
        self.declare_parameter('hard_iron_offset_t', [0.0, 0.0, 0.0])
        self.declare_parameter('hard_iron_field_norm_t', 0.0)
        self.declare_parameter('poll_rate_hz', 200.0)
        self.declare_parameter('reconnect_interval_s', 2.0)
        self.declare_parameter('startup_grace_s', 6.0)
        self.declare_parameter('data_timeout_s', 1.0)
        self.declare_parameter('orientation_covariance', [0.01, 0.01, 0.01])
        self.declare_parameter('angular_velocity_covariance', [0.001, 0.001, 0.001])
        self.declare_parameter('linear_acceleration_covariance', [0.1, 0.1, 0.1])
        self.declare_parameter('magnetic_field_covariance', [4.0e-10, 4.0e-10, 4.0e-10])
        self.declare_parameter('pressure_variance', 4.0)
        self.declare_parameter('temperature_variance', 0.25)

        self.port = str(self.get_parameter('port').value)
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.report_rate_hz = int(self.get_parameter('report_rate_hz').value)
        self.fusion_axes = int(self.get_parameter('fusion_axes').value)
        if self.fusion_axes not in (6, 9):
            raise ValueError('fusion_axes must be 6 or 9')
        self.orientation_mode = str(self.get_parameter('orientation_mode').value)
        if self.orientation_mode not in ('planar_gyro_mag', 'vendor_fused'):
            raise ValueError(
                'orientation_mode must be planar_gyro_mag or vendor_fused')
        self.publish_linear_acceleration = bool(
            self.get_parameter('publish_linear_acceleration').value)
        self.reconnect_interval_s = max(
            0.1, float(self.get_parameter('reconnect_interval_s').value))
        # The YaBoom datasheet specifies a 5000 ms startup time. Do not apply
        # the normal streaming timeout until that cold-start window has passed.
        self.startup_grace_s = max(
            5.0, float(self.get_parameter('startup_grace_s').value))
        self.data_timeout_s = max(0.1, float(self.get_parameter('data_timeout_s').value))

        self.orientation_covariance = self._diagonal_covariance(
            'orientation_covariance')
        self.angular_velocity_covariance = self._diagonal_covariance(
            'angular_velocity_covariance')
        self.linear_acceleration_covariance = self._diagonal_covariance(
            'linear_acceleration_covariance')
        self.magnetic_field_covariance = self._diagonal_covariance(
            'magnetic_field_covariance')
        seed_norm = float(self.get_parameter('hard_iron_field_norm_t').value)
        self.planar_filter = PlanarGyroMagFilter(
            estimate_gyro_bias=bool(
                self.get_parameter('estimate_gyro_bias').value),
            gyro_bias_stillness_rad_s=float(
                self.get_parameter('gyro_bias_stillness_rad_s').value),
            gyro_noise_rad_s=float(
                self.get_parameter('gyro_noise_rad_s').value),
            gyro_bias_walk_rad_s=float(
                self.get_parameter('gyro_bias_walk_rad_s').value),
            gyro_bias_initial_uncertainty_rad_s=float(
                self.get_parameter('gyro_bias_initial_uncertainty_rad_s').value),
            gyro_bias_confident_rad_s=float(
                self.get_parameter('gyro_bias_confident_rad_s').value),
            gyro_bias_temperature_sensitivity_rad_s_per_c=float(
                self.get_parameter(
                    'gyro_bias_temperature_sensitivity_rad_s_per_c').value),
            gyro_bias_max_corrected_rate_rad_s=float(
                self.get_parameter('gyro_bias_max_corrected_rate_rad_s').value),
            gyro_bias_relock_s=float(
                self.get_parameter('gyro_bias_relock_s').value),
            magnetic_correction_time_constant_s=float(
                self.get_parameter('magnetic_correction_time_constant_s').value),
            magnetic_field_min_t=float(
                self.get_parameter('magnetic_field_min_t').value),
            magnetic_field_max_t=float(
                self.get_parameter('magnetic_field_max_t').value),
            magnetic_norm_tolerance=float(
                self.get_parameter('magnetic_norm_tolerance').value),
            magnetic_heading_rejection_rad=math.radians(float(
                self.get_parameter('magnetic_heading_rejection_deg').value)),
            magnetic_max_rate_rad_s=math.radians(float(
                self.get_parameter('magnetic_max_rate_deg_s').value)),
            estimate_hard_iron=bool(
                self.get_parameter('estimate_hard_iron').value),
            hard_iron_offset_t=tuple(
                float(v) for v in self.get_parameter('hard_iron_offset_t').value),
            hard_iron_field_norm_t=seed_norm if seed_norm > 0.0 else None,
        )
        self._hard_iron_logged = False
        self._ever_connected = False

        self.imu_publisher = self.create_publisher(
            Imu, str(self.get_parameter('imu_topic').value), qos_profile_sensor_data)
        self.mag_publisher = self.create_publisher(
            MagneticField, str(self.get_parameter('magnetic_field_topic').value),
            qos_profile_sensor_data)
        self.pressure_publisher = self.create_publisher(
            FluidPressure, str(self.get_parameter('pressure_topic').value),
            qos_profile_sensor_data)
        self.temperature_publisher = self.create_publisher(
            Temperature, str(self.get_parameter('temperature_topic').value),
            qos_profile_sensor_data)
        self.altitude_publisher = self.create_publisher(
            Float64, str(self.get_parameter('altitude_topic').value), qos_profile_sensor_data)
        self.create_service(Trigger, '~/calibrate_imu', self._calibrate_imu)
        self.create_service(
            Trigger, '~/calibrate_magnetometer', self._calibrate_magnetometer)

        self.protocol = YbImuProtocol()
        self.serial_device: Optional[serial.Serial] = None
        self.last_connect_attempt = 0.0
        self.connected_at = 0.0
        self.last_valid_frame = 0.0
        self.received_valid_frame = False
        self.last_imu_sequence = 0
        self.last_quaternion_sequence = 0
        self.last_barometer_sequence = 0
        self.pending_fusion_configuration_at: Optional[float] = None
        self._missing_logged = False
        self._planar_calibration_logged = False
        # Report rate and fusion axes are PERSISTENT device writes. A reconnect
        # loop must not resend them freely: at a 6 s startup grace a silent IMU
        # would take ~20 flash writes per minute for as long as the launch runs.
        # Bound the attempts rather than allowing exactly one, so a write that
        # lands during the 5 s datasheet cold start still gets retried, while a
        # device that never answers costs a handful of writes instead of
        # hundreds. The settings are persistent, so surviving reconnects inherit
        # them without another write.
        self.max_configuration_attempts = 3
        self._configuration_attempts = 0
        self._bytes_since_connect = 0

        poll_rate = max(20.0, float(self.get_parameter('poll_rate_hz').value))
        self.timer = self.create_timer(1.0 / poll_rate, self._poll)
        self.get_logger().info(
            f'YaBoom 10-axis IMU driver ready: {self.port} at '
            f'{self.baudrate} baud, orientation_mode={self.orientation_mode}, '
            f'linear_acceleration={self.publish_linear_acceleration}')

    def _diagonal_covariance(self, parameter_name: str) -> List[float]:
        diagonal = list(self.get_parameter(parameter_name).value)
        if len(diagonal) != 3 or any(float(value) < 0.0 for value in diagonal):
            raise ValueError(f'{parameter_name} must contain three non-negative values')
        covariance = [0.0] * 9
        covariance[0], covariance[4], covariance[8] = map(float, diagonal)
        return covariance

    def _connect(self, now: float) -> None:
        if now - self.last_connect_attempt < self.reconnect_interval_s:
            return
        self.last_connect_attempt = now

        try:
            self.serial_device = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0,
                write_timeout=0.5,
            )
            self.serial_device.reset_input_buffer()
            self.protocol.reset()
            # A reconnect does not move the rover. Keep the heading and the
            # gyro/hard-iron calibration so a mid-mission cable glitch does not
            # zero yaw and rerun the bias average while the wheels are turning.
            self.planar_filter.reset(preserve_calibration=self._ever_connected)
            self._planar_calibration_logged = False
            self._ever_connected = True
            if self._configuration_attempts < self.max_configuration_attempts:
                self._configuration_attempts += 1
                self.serial_device.write(
                    self.protocol.build_report_rate_command(self.report_rate_hz))
                # The reference library gives each persistent configuration write
                # one second to settle. Send the fusion selection later without
                # blocking the ROS executor or serial receive path.
                self.pending_fusion_configuration_at = now + 1.0
            self._bytes_since_connect = 0
            self.connected_at = now
            self.last_valid_frame = now
            self.received_valid_frame = False
            self.last_imu_sequence = 0
            self.last_quaternion_sequence = 0
            self.last_barometer_sequence = 0
            self._missing_logged = False
            self.get_logger().info(f'Connected to YaBoom IMU on {self.port}')
        except (OSError, serial.SerialException) as error:
            self.serial_device = None
            if not self._missing_logged:
                suffix = '' if os.path.exists(self.port) else ' (device path does not exist)'
                self.get_logger().warning(
                    f'Waiting for YaBoom IMU at {self.port}{suffix}: {error}')
                self._missing_logged = True

    def _link_diagnosis(self) -> str:
        """Say WHY no frames arrived, so silence and framing errors differ.

        Zero bytes means nothing is driving the receive line - unpowered IMU,
        disconnected TX wire, or wrong port. Bytes with checksum errors mean the
        link is live but mis-framed, which points at the baud rate instead.
        """
        if self._bytes_since_connect == 0:
            return (' (0 bytes received: nothing is driving RX - check IMU power,'
                    ' the TX wire, and that the port is the IMU)')
        if self.protocol.checksum_errors:
            # A checksum error means a 7e23 header with a plausible length was
            # found, so the device speaks the protocol and only framing is off.
            return (f' ({self._bytes_since_connect} bytes received,'
                    f' {self.protocol.checksum_errors} bad checksums: device is'
                    ' streaming but mis-framed - check the baud rate)')
        # Bytes with no header at all are not a stream. A floating RX pin picks
        # up crosstalk from our own config write, which is how a disconnected TX
        # wire imitates a live link - so compare against the expected volume
        # instead of trusting that bytes arrived.
        expected = int(self.report_rate_hz * 20 * self.startup_grace_s)
        return (f' ({self._bytes_since_connect} bytes received but no valid'
                f' header, versus ~{expected} expected: not a data stream -'
                ' likely line noise or echo on a floating RX, or the wrong port)')

    def _disconnect(self, reason: str) -> None:
        if self.serial_device is not None:
            try:
                self.serial_device.close()
            except (OSError, serial.SerialException):
                pass
        self.serial_device = None
        self.pending_fusion_configuration_at = None
        self.received_valid_frame = False
        self.last_connect_attempt = 0.0
        self.get_logger().warning(f'YaBoom IMU disconnected: {reason}; retrying')

    def _poll(self) -> None:
        now = time.monotonic()
        if self.serial_device is None:
            self._connect(now)
            return

        try:
            if (self.pending_fusion_configuration_at is not None
                    and now >= self.pending_fusion_configuration_at):
                self.serial_device.write(self.protocol.build_config_command(
                    self.protocol.FUNC_ALGORITHM, self.fusion_axes))
                self.pending_fusion_configuration_at = None
            available = self.serial_device.in_waiting
            if available:
                chunk = self.serial_device.read(available)
                self._bytes_since_connect += len(chunk)
                functions = self.protocol.feed(chunk)
                if functions:
                    self.last_valid_frame = now
                    self.received_valid_frame = True
                    self._publish_new_samples()
            if (self.received_valid_frame
                    and now - self.last_valid_frame > self.data_timeout_s):
                self._disconnect(
                    f'no valid packets for {self.data_timeout_s:.1f} s'
                    f'{self._link_diagnosis()}')
            elif (not self.received_valid_frame
                  and now - self.connected_at > self.startup_grace_s):
                self._disconnect(
                    f'no valid packets during {self.startup_grace_s:.1f} s startup'
                    f'{self._link_diagnosis()}')
        except (OSError, serial.SerialException) as error:
            self._disconnect(str(error))

    def _publish_new_samples(self) -> None:
        planar_orientation = self.orientation_mode == 'planar_gyro_mag'
        vendor_orientation_ready = (
            self.protocol.quaternion is not None
            and self.protocol.quaternion_sequence != self.last_quaternion_sequence
        )
        if (self.protocol.imu is not None
                and self.protocol.imu_sequence != self.last_imu_sequence
                and (planar_orientation or vendor_orientation_ready)):
            self._publish_imu_and_magnetometer()
            self.last_imu_sequence = self.protocol.imu_sequence
            if not planar_orientation:
                self.last_quaternion_sequence = self.protocol.quaternion_sequence

        if (self.protocol.barometer is not None
                and self.protocol.barometer_sequence != self.last_barometer_sequence):
            self._publish_barometer()
            self.last_barometer_sequence = self.protocol.barometer_sequence

    def _publish_imu_and_magnetometer(self) -> None:
        sample = self.protocol.imu
        if sample is None:
            return

        if self.orientation_mode == 'planar_gyro_mag':
            # The barometer's temperature drives the bias estimator's thermal
            # uncertainty, which is the dominant cause of MEMS bias drift.
            temperature_c = (None if self.protocol.barometer is None
                             else self.protocol.barometer.temperature)
            filter_output = self.planar_filter.update(
                time.monotonic(), sample.angular_velocity, sample.magnetic_field,
                temperature_c)
            quaternion = filter_output.quaternion
            angular_velocity = filter_output.angular_velocity
            if filter_output.calibrating and not self._planar_calibration_logged:
                self.get_logger().info(
                    'Planar gyro bias converging; hold the IMU still briefly')
                self._planar_calibration_logged = True
            if filter_output.calibration_completed:
                bias = self.planar_filter.gyro_bias
                sigma = self.planar_filter.gyro_bias_uncertainty
                self.get_logger().info(
                    'Planar gyro bias calibrated: '
                    f'x={bias[0]:+.6f}, y={bias[1]:+.6f}, z={bias[2]:+.6f} rad/s '
                    f'(yaw-axis 1-sigma {sigma[2]:.6f} rad/s = '
                    f'{sigma[2] * 60.0 * 180.0 / math.pi:.2f} deg/min drift)')
            if filter_output.hard_iron_calibrated and not self._hard_iron_logged:
                offset = filter_output.hard_iron_offset
                norm = self.planar_filter.hard_iron.field_norm or 0.0
                self.get_logger().info(
                    'Hard-iron offset estimated: '
                    f'x={offset[0] * 1e6:+.2f}, y={offset[1] * 1e6:+.2f} uT '
                    f'against a {norm * 1e6:.2f} uT horizontal field; '
                    'set hard_iron_offset_t/hard_iron_field_norm_t to reuse it')
                self._hard_iron_logged = True
        else:
            quaternion = self.protocol.quaternion
            angular_velocity = sample.angular_velocity
            if quaternion is None:
                return

        stamp = self.get_clock().now().to_msg()
        message = Imu()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.orientation.w, message.orientation.x, message.orientation.y, \
            message.orientation.z = quaternion
        message.angular_velocity.x, message.angular_velocity.y, \
            message.angular_velocity.z = angular_velocity
        if self.publish_linear_acceleration:
            message.linear_acceleration.x, message.linear_acceleration.y, \
                message.linear_acceleration.z = sample.acceleration
            message.linear_acceleration_covariance = \
                self.linear_acceleration_covariance
        else:
            # sensor_msgs/Imu uses covariance[0] = -1 to mark an estimate as
            # unavailable. Values remain zero and downstream filters must not
            # treat them as measurements.
            message.linear_acceleration_covariance[0] = -1.0
        message.orientation_covariance = self.orientation_covariance
        message.angular_velocity_covariance = self.angular_velocity_covariance
        self.imu_publisher.publish(message)

        magnetic = MagneticField()
        magnetic.header.stamp = stamp
        magnetic.header.frame_id = self.frame_id
        magnetic.magnetic_field.x, magnetic.magnetic_field.y, \
            magnetic.magnetic_field.z = sample.magnetic_field
        magnetic.magnetic_field_covariance = self.magnetic_field_covariance
        self.mag_publisher.publish(magnetic)

    def _publish_barometer(self) -> None:
        sample = self.protocol.barometer
        if sample is None:
            return

        stamp = self.get_clock().now().to_msg()
        pressure = FluidPressure()
        pressure.header.stamp = stamp
        pressure.header.frame_id = self.frame_id
        pressure.fluid_pressure = sample.pressure
        pressure.variance = float(self.get_parameter('pressure_variance').value)
        self.pressure_publisher.publish(pressure)

        temperature = Temperature()
        temperature.header.stamp = stamp
        temperature.header.frame_id = self.frame_id
        temperature.temperature = sample.temperature
        temperature.variance = float(self.get_parameter('temperature_variance').value)
        self.temperature_publisher.publish(temperature)

        altitude = Float64()
        altitude.data = sample.altitude
        self.altitude_publisher.publish(altitude)

    def _send_action(self, function: int, instruction: str, response) -> object:
        if self.serial_device is None:
            response.success = False
            response.message = f'YaBoom IMU is not connected on {self.port}'
            return response
        try:
            self.serial_device.write(
                self.protocol.build_config_command(function, 1))
            response.success = True
            response.message = instruction
        except (OSError, serial.SerialException) as error:
            self._disconnect(str(error))
            response.success = False
            response.message = f'Failed to send calibration command: {error}'
        return response

    def _calibrate_imu(self, request, response):
        del request
        return self._send_action(
            self.protocol.FUNC_CALIBRATE_IMU,
            'IMU calibration started; keep the sensor completely stationary',
            response)

    def _calibrate_magnetometer(self, request, response):
        del request
        return self._send_action(
            self.protocol.FUNC_CALIBRATE_MAGNETOMETER,
            'Magnetometer calibration started; rotate the sensor through all axes',
            response)

    def destroy_node(self) -> bool:
        if self.serial_device is not None:
            try:
                self.serial_device.close()
            except (OSError, serial.SerialException):
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YbImuDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
