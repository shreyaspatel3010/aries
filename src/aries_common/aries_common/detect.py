"""Consistent hardware auto-detection for Aries launch files."""

from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from aries_common.devices import device


TRUE_VALUES = ("1", "true", "yes", "on")
FALSE_VALUES = ("0", "false", "no", "off", "none")
AUTO_VALUES = ("auto", "detect")
# Device identity lives in aries_common/config/devices.yaml; see devices.py.
MICROSTRAIN_PORT = device("imu.port")


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


# IFF_UP, bit 0 of the interface flags word in sysfs.
_IFF_UP = 0x1


def can_link_state(interface):
    """Report whether a SocketCAN interface can actually carry traffic.

    ``can_interface_exists`` only answers whether the netdev node is there. A
    USB CAN adapter that was unplugged and plugged back in reappears
    immediately, so that probe keeps saying "present" while every send fails:
    the interface comes back administratively DOWN with no bitrate, and with a
    new interface index that already-bound sockets can never reach again.

    ``ifindex`` is what tells a re-plug apart from a link that was merely
    brought down: the kernel hands out a fresh index for the new netdev, and a
    socket bound to the old one fails with ENXIO forever after.

    ``carrier`` is 0 while the controller is bus-off — can_bus_off() in the CAN
    core drops the carrier — so it separates "up and able to transmit" from
    "up but wedged", which look identical in the flags word.
    """
    root = Path(f"/sys/class/net/{interface}")
    state = {
        "interface": interface,
        "present": root.exists(),
        "ifindex": None,
        "up": False,
        "carrier": False,
        "usable": False,
    }
    if not state["present"]:
        return state

    state["ifindex"] = _read_int(root / "ifindex")
    flags = _read_int(root / "flags", base=16)
    state["up"] = bool(flags is not None and flags & _IFF_UP)
    # Only readable while the interface is up; the kernel returns EINVAL below
    # that, which _read_int reports as None.
    state["carrier"] = _read_int(root / "carrier") == 1
    state["usable"] = state["up"] and state["carrier"]
    return state


def describe_can_link(state):
    """One-line human summary of a can_link_state() result."""
    if not state["present"]:
        return "not present"
    if not state["up"]:
        return "present but DOWN (needs 'ip link set ... up type can bitrate ...')"
    if not state["carrier"]:
        return "up but no carrier (bus-off or no bus power)"
    return f"up (ifindex {state['ifindex']})"


def _read_int(path, base=10):
    try:
        return int(path.read_text().strip(), base)
    except (OSError, ValueError):
        return None


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
