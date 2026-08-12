"""Consistent hardware auto-detection for Aries launch files."""

from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


TRUE_VALUES = ("1", "true", "yes", "on")
FALSE_VALUES = ("0", "false", "no", "off", "none")
AUTO_VALUES = ("auto", "detect")
# Persistent symlink from aries_imu/setup/99-microstrain.rules.
MICROSTRAIN_PORT = "/dev/microstrain_main"


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


def microstrain_available(imu_port):
    """True when the 3DM-GX5-AHRS is plugged in and its driver is installed."""
    return Path(imu_port).exists() and package_exists("microstrain_inertial_driver")


def resolve_rover_backend(protocol, can_interface):
    """Resolve the requested drive backend to ``odrive`` or ``mock_hardware``."""
    mode = str(protocol).strip().lower()
    if mode in AUTO_VALUES:
        return "odrive" if can_interface_exists(can_interface) else "mock_hardware"
    return mode


def resolve_imu_source(use_imu, imu_port=MICROSTRAIN_PORT):
    """Select the MicroStrain 3DM-GX5-AHRS or no IMU.

    The rover carries one IMU, so this is a presence check rather than a
    preference order.  Naming the device explicitly still fails closed when it
    is unplugged: a forced ``use_imu:=microstrain`` must not start a driver
    against a port that does not exist, because robot_localization would then
    wait on a topic that never publishes instead of falling back to wheel
    odometry.  Returns ``(source, microstrain_present)``.
    """
    mode = str(use_imu).strip().lower()
    present = microstrain_available(imu_port)

    if mode in FALSE_VALUES or mode in ("odom_only", "wheel_odom"):
        return "none", present
    return ("microstrain" if present else "none"), present
