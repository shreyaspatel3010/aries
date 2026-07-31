import math
import struct

import pytest
from ybimu_ros2.protocol import STANDARD_GRAVITY, YbImuProtocol


def frame(function, payload):
    data = bytearray((0x7E, 0x23, len(payload) + 5, function))
    data.extend(payload)
    data.append(sum(data) & 0xFF)
    return bytes(data)


def test_parses_fragmented_raw_frame_and_converts_to_si():
    protocol = YbImuProtocol()
    payload = struct.pack('<9h', 2048, -4096, 32767, 100, -200, 300,
                          1000, -2000, 3000)
    packet = frame(YbImuProtocol.FUNC_IMU_RAW, payload)

    assert protocol.feed(b'noise' + packet[:7]) == []
    assert protocol.feed(packet[7:]) == [YbImuProtocol.FUNC_IMU_RAW]
    assert protocol.imu is not None
    assert protocol.imu.acceleration[2] == pytest.approx(16.0 * STANDARD_GRAVITY)
    assert protocol.imu.angular_velocity[0] == pytest.approx(
        100 * (2000.0 / 32767.0) * math.pi / 180.0)
    assert protocol.imu.magnetic_field[0] == pytest.approx(
        1000 * (800.0 / 32767.0) * 1.0e-6)
    # Magnetometer axes pass through unmodified: negating a single axis would
    # mirror the magnetometer frame relative to the gyro frame and reverse the
    # sense of the field angle used for heading.
    assert protocol.imu.magnetic_field[1] == pytest.approx(
        -2000 * (800.0 / 32767.0) * 1.0e-6)


def test_raw_scale_factors_match_datasheet_resolutions():
    protocol = YbImuProtocol()
    payload = struct.pack('<9h', 1, 0, 0, 1, 0, 0, 1, 0, 0)

    protocol.feed(frame(YbImuProtocol.FUNC_IMU_RAW, payload))

    assert protocol.imu is not None
    assert protocol.imu.acceleration[0] / STANDARD_GRAVITY == pytest.approx(
        0.0005, abs=0.00002)
    assert math.degrees(protocol.imu.angular_velocity[0]) == pytest.approx(
        0.061, abs=0.0001)
    magnetic_milligauss = protocol.imu.magnetic_field[0] * 1.0e7
    assert magnetic_milligauss == pytest.approx(0.244, abs=0.001)


def test_normalizes_quaternion_and_rejects_bad_checksum():
    protocol = YbImuProtocol()
    bad = bytearray(frame(YbImuProtocol.FUNC_QUATERNION,
                          struct.pack('<4f', 2.0, 0.0, 0.0, 0.0)))
    bad[-1] ^= 0xFF

    assert protocol.feed(bytes(bad)) == []
    assert protocol.checksum_errors == 1
    good = frame(YbImuProtocol.FUNC_QUATERNION,
                 struct.pack('<4f', 2.0, 0.0, 0.0, 0.0))
    assert protocol.feed(good) == [YbImuProtocol.FUNC_QUATERNION]
    assert protocol.quaternion == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_parses_barometer_and_builds_report_rate_command():
    protocol = YbImuProtocol()
    packet = frame(YbImuProtocol.FUNC_BAROMETER,
                   struct.pack('<4f', 12.5, 24.0, 101325.0, -2.0))
    assert protocol.feed(packet) == [YbImuProtocol.FUNC_BAROMETER]
    assert protocol.barometer is not None
    assert protocol.barometer.pressure == pytest.approx(101325.0)

    command = protocol.build_report_rate_command(50)
    assert command == bytes((0x7E, 0x23, 0x07, 0x60, 50, 0x5F,
                             (0x7E + 0x23 + 0x07 + 0x60 + 50 + 0x5F) & 0xFF))


def test_normalizes_datasheet_hectopascal_pressure_to_ros_pascal():
    protocol = YbImuProtocol()
    packet = frame(YbImuProtocol.FUNC_BAROMETER,
                   struct.pack('<4f', 12.5, 24.0, 1013.25, -0.02))

    protocol.feed(packet)

    assert protocol.barometer is not None
    assert protocol.barometer.pressure == pytest.approx(101325.0)
    assert protocol.barometer.pressure_difference == pytest.approx(-2.0)
