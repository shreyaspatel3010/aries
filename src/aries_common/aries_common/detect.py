"""Consistent hardware auto-detection for Aries launch files."""

import socket
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


TRUE_VALUES = ("1", "true", "yes", "on")
FALSE_VALUES = ("0", "false", "no", "off", "none")
AUTO_VALUES = ("auto", "detect")
LIDAR_PROBE_PORT = 2111
LIDAR_PROBE_TIMEOUT = 0.4


def as_bool(value):
    return str(value).strip().lower() in TRUE_VALUES


def as_int(value, default):
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def package_exists(package_name):
    try:
        get_package_share_directory(package_name)
        return True
    except PackageNotFoundError:
        return False


def can_interface_exists(interface):
    return Path(f"/sys/class/net/{interface}").exists()


def lidar_reachable(sensor_ip, port=LIDAR_PROBE_PORT, timeout=LIDAR_PROBE_TIMEOUT):
    try:
        with socket.create_connection((sensor_ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def bno055_available(imu_port):
    return Path(imu_port).exists() and package_exists("bno055")


def ybimu_available(imu_port):
    return Path(imu_port).exists() and package_exists("ybimu_ros2")


def resolve_rover_backend(protocol, can_interface):
    """Resolve the requested drive backend to ``odrive`` or ``mock_hardware``."""
    mode = str(protocol).strip().lower()
    if mode in AUTO_VALUES:
        return "odrive" if can_interface_exists(can_interface) else "mock_hardware"
    return mode


def resolve_lidar_enabled(mode, sensor_ip):
    mode = str(mode).strip().lower()
    if mode in AUTO_VALUES:
        return package_exists("sick_scan_xd") and lidar_reachable(sensor_ip)
    return mode in TRUE_VALUES


def resolve_imu_source(
    use_imu,
    imu_port,
    lidar_available,
    ybimu_port="/dev/imu_ybimu",
):
    """Select ybimu, BNO055, picoScan, or no IMU.

    ``imu_port`` remains the BNO055 port for compatibility with existing
    launches.  The YaBoom driver uses its own persistent ``ybimu_port``.
    Returns ``(source, ybimu_present, bno055_present)``.
    """
    mode = str(use_imu).strip().lower()
    bno_available = bno055_available(imu_port)
    yb_available = ybimu_available(ybimu_port)

    if mode in FALSE_VALUES or mode in ("odom_only", "wheel_odom"):
        return "none", yb_available, bno_available
    if mode in ("ybimu", "yaboom"):
        return ("ybimu" if yb_available else "none"), yb_available, bno_available
    if mode == "bno055":
        return ("bno055" if bno_available else "none"), yb_available, bno_available
    if mode == "serial":
        if yb_available:
            return "ybimu", yb_available, bno_available
        return ("bno055" if bno_available else "none"), yb_available, bno_available
    if mode in ("picoscan", "picoscan150", "sick", "lidar"):
        return (
            "picoscan" if lidar_available else "none",
            yb_available,
            bno_available,
        )

    if yb_available:
        return "ybimu", yb_available, bno_available
    if bno_available:
        return "bno055", yb_available, bno_available
    if lidar_available:
        return "picoscan", yb_available, bno_available
    return "none", yb_available, bno_available
