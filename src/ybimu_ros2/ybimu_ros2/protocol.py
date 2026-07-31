"""
Support the YaBoom 10-axis IMU UART protocol.

The packet layout and scale factors are adapted from YaBoom's YbImuLib source.
Values exposed by :class:`YbImuProtocol` use ROS SI
units: m/s^2, rad/s, tesla, pascal, metre, and degree Celsius.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import List, Optional, Tuple


STANDARD_GRAVITY = 9.80665


@dataclass
class ImuSample:
    acceleration: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    magnetic_field: Tuple[float, float, float]


@dataclass
class BarometerSample:
    altitude: float
    temperature: float
    pressure: float
    pressure_difference: float


class YbImuProtocol:
    """Incrementally parse the byte stream produced by a YaBoom 10-axis IMU."""

    HEADER = b'\x7e\x23'
    MAX_FRAME_LENGTH = 40

    FUNC_VERSION = 0x01
    FUNC_IMU_RAW = 0x04
    FUNC_QUATERNION = 0x16
    FUNC_EULER = 0x26
    FUNC_BAROMETER = 0x32
    FUNC_REPORT_RATE = 0x60
    FUNC_ALGORITHM = 0x61
    FUNC_CALIBRATE_IMU = 0x70
    FUNC_CALIBRATE_MAGNETOMETER = 0x71

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.imu: Optional[ImuSample] = None
        self.quaternion: Optional[Tuple[float, float, float, float]] = None
        self.euler: Optional[Tuple[float, float, float]] = None
        self.barometer: Optional[BarometerSample] = None
        self.version: Optional[str] = None

        self.imu_sequence = 0
        self.quaternion_sequence = 0
        self.barometer_sequence = 0
        self.valid_frames = 0
        self.checksum_errors = 0

    def reset(self) -> None:
        """Discard buffered and sampled data after a serial reconnect."""
        self.__init__()

    @staticmethod
    def build_report_rate_command(rate_hz: int) -> bytes:
        """Build the vendor command that selects a 10--100 Hz report rate."""
        rate_hz = max(10, min(100, int(rate_hz)))
        return YbImuProtocol.build_config_command(
            YbImuProtocol.FUNC_REPORT_RATE, rate_hz)

    @staticmethod
    def build_config_command(function: int, value: int) -> bytes:
        """Build a protected YaBoom configuration/calibration command."""
        frame = bytearray((0x7E, 0x23, 0x00, int(function) & 0xFF,
                           int(value) & 0xFF, 0x5F))
        frame[2] = len(frame) + 1
        frame.append(sum(frame) & 0xFF)
        return bytes(frame)

    def feed(self, data: bytes) -> List[int]:
        """Consume bytes and return the function IDs of valid parsed frames."""
        if data:
            self._buffer.extend(data)

        parsed_functions: List[int] = []
        while True:
            start = self._buffer.find(self.HEADER)
            if start < 0:
                # Keep a possible first header byte split across serial reads.
                self._buffer[:] = self._buffer[-1:] if self._buffer[-1:] == b'\x7e' else b''
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 3:
                break

            frame_length = self._buffer[2]
            if frame_length < 5 or frame_length > self.MAX_FRAME_LENGTH:
                del self._buffer[0]
                continue
            if len(self._buffer) < frame_length:
                break

            frame = bytes(self._buffer[:frame_length])
            if (sum(frame[:-1]) & 0xFF) != frame[-1]:
                self.checksum_errors += 1
                # Advance one byte, then search for the next full header.
                del self._buffer[0]
                continue

            del self._buffer[:frame_length]
            function = frame[3]
            payload = frame[4:-1]
            if self._parse(function, payload):
                self.valid_frames += 1
                parsed_functions.append(function)

        return parsed_functions

    def _parse(self, function: int, payload: bytes) -> bool:
        if function == self.FUNC_IMU_RAW and len(payload) >= 18:
            ax, ay, az, gx, gy, gz, mx, my, mz = struct.unpack_from('<9h', payload)
            # Datasheet full scales: accel +/-16 g, gyro +/-2000 deg/s,
            # magnetometer +/-8 gauss (+/-800 microtesla). These produce the
            # stated approximate resolutions of 0.0005 g, 0.061 deg/s, and
            # 0.244 milligauss per signed 16-bit count.
            accel_scale = (16.0 / 32767.0) * STANDARD_GRAVITY
            gyro_scale = (2000.0 / 32767.0) * (math.pi / 180.0)
            magnetic_scale = (800.0 / 32767.0) * 1.0e-6
            self.imu = ImuSample(
                tuple(value * accel_scale for value in (ax, ay, az)),
                tuple(value * gyro_scale for value in (gx, gy, gz)),
                # Do NOT negate a single magnetometer axis here. Reversing only
                # Y makes the magnetometer frame a reflection of the gyro
                # frame, so the field angle advances with +yaw instead of
                # against it and any heading correction fights the gyro on
                # every turn. Measured on this rover over a 1600 deg turn:
                # d(field angle)/d(gyro yaw) = +0.85 with the negation and
                # -0.85 without it, where a right-handed frame requires a
                # negative slope.
                tuple(value * magnetic_scale for value in (mx, my, mz)),
            )
            self.imu_sequence += 1
            return True

        if function == self.FUNC_QUATERNION and len(payload) >= 16:
            qw, qx, qy, qz = struct.unpack_from('<4f', payload)
            norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if not math.isfinite(norm) or norm < 1.0e-6:
                return False
            self.quaternion = (qw / norm, qx / norm, qy / norm, qz / norm)
            self.quaternion_sequence += 1
            return True

        if function == self.FUNC_EULER and len(payload) >= 12:
            # Firmware sends radians. Keep degrees for parity with YbImuLib.
            self.euler = tuple(math.degrees(value) for value in struct.unpack_from('<3f', payload))
            return True

        if function == self.FUNC_BAROMETER and len(payload) >= 16:
            altitude, temperature, pressure, pressure_difference = struct.unpack_from(
                '<4f', payload)
            # The datasheet expresses the 300--2000 pressure range in hPa,
            # while the vendor library leaves the transmitted float unit
            # unspecified. Accept either hPa (for example 1013.25) or Pa
            # (101325) and expose only ROS-standard pascals.
            pressure_scale = 100.0 if 300.0 <= pressure <= 2000.0 else 1.0
            self.barometer = BarometerSample(
                float(altitude), float(temperature),
                float(pressure) * pressure_scale,
                float(pressure_difference) * pressure_scale)
            self.barometer_sequence += 1
            return True

        if function == self.FUNC_VERSION and len(payload) >= 3:
            self.version = f'V{payload[0]}.{payload[1]}.{payload[2]}'
            return True

        return False
