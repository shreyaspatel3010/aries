"""Identity of every physical device on Aries, read from one YAML file.

Ports, serials and addresses were duplicated as ``default_value`` literals
across a dozen launch files, so changing a camera or reflashing a Teensy meant
finding every copy. They live in ``aries_common/config/devices.yaml`` now, and
launch files take their defaults from here:

    from aries_common.devices import device_str

    DeclareLaunchArgument("can_interface", default_value=device_str("rover.can_interface"))

A command-line argument still overrides the file, because this only supplies
the default. Point ``ARIES_DEVICES_FILE`` at another YAML to swap the whole set
for a second robot or a bench rig.

The built-in defaults below are a fallback for a missing or unreadable file, so
a broken edit degrades to the shipped values instead of taking the stack down
at launch. They are deliberately a copy of what the YAML ships with.
"""

import os
from pathlib import Path

DEFAULTS = {
    "arm": {"host": "192.168.3.11", "port": 3920},
    "gripper": {
        # Keep in step with devices.yaml -- these two blocks have been SWAPPED
        # relative to it before, which is the worst possible drift: a fallback
        # to these defaults then points the gripper, the drill and the stack
        # light at the board answering as the science module. Re-pinned to
        # 16739090 on 2026-09-06, verified against the connected board and
        # flashed from firmware/teensy_drill_sys.
        #
        # "Dual_Serial", not "USB_Serial": that half of the name comes from the
        # -D USB_DUAL_SERIAL BUILD flag, not from the board, and a by-id path
        # that does not exist resolves the gripper to mock_hardware without
        # failing. See the note in devices.yaml.
        "serial_port": "/dev/serial/by-id/usb-Teensyduino_Dual_Serial_16739090-if00",
    },
    "science": {
        # NOT verified against hardware -- that board was not connected on
        # 2026-09-06. Mirrors devices.yaml, which carries the same caveat.
        "serial_port": "/dev/serial/by-id/usb-Teensyduino_Dual_Serial_20385500-if00",
    },
    "rover": {"can_interface": "can0", "can_bitrate": 250000},
    "imu": {"port": "/dev/microstrain_main"},
    # The SECONDARY gripper's wire, and the port scripts/st3215_test.py opens.
    # Back to the USB servo driver's udev symlink on 2026-09-06, after the
    # 2026-09-01..09-06 spell on the drill Teensy's second CDC (lib/servobus).
    # The bridge remains a FALLBACK in resolve_servo_bus(), not the config.
    # See devices.yaml.
    "servo_bus": {
        "port": "/dev/aries_servo_bus",
        "baud": 1000000,
        "gripper_servo_id": 1,
        # The fitted CH343's USB serial, read from the hardware 2026-09-06.
        # Only setup_system.sh uses it, to generate 99-aries-servo-bus.rules.
        "serial": "5B61034961",
    },
    "cameras": {
        "gripper_serial": "216322070216",
        "front_serial": "207522077539",
        # A device path, not a serial: the rear camera is a UVC webcam driven by
        # usb_cam, not a RealSense. See the note in devices.yaml.
        "rear_device": "/dev/v4l/by-id/usb-046d_Brio_100_2446ZBZ4XXN8-video-index0",
    },
    "joystick": {"device": "/dev/input/js0"},
    "network": {
        "domain_id": 30,
        "subnet_prefix": 24,
        "hosts": {"rover": "192.168.1.10", "base": "192.168.1.11"},
        "interface": {"rover": "enp130s0", "base": "enp130s0"},
        "radios": {"rover": "192.168.1.20", "base": "192.168.1.21"},
    },
}

_ENV_OVERRIDE = "ARIES_DEVICES_FILE"
_cache = None


def devices_file():
    """Path the device table is read from, or None when only defaults apply."""
    override = os.environ.get(_ENV_OVERRIDE, "").strip()
    if override:
        return Path(override)
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("aries_common")) / "config" / "devices.yaml"
    except Exception:
        return None


def load_devices(refresh=False):
    """The device table: the YAML merged over DEFAULTS, section by section."""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    merged = {section: dict(values) for section, values in DEFAULTS.items()}
    path = devices_file()
    if path is not None and path.is_file():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001 - a bad edit must not stop a launch
            print(f"[aries_common] Could not read {path} ({exc}); using built-in device defaults")
            loaded = {}
        for section, values in loaded.items():
            if isinstance(values, dict):
                merged.setdefault(section, {}).update(values)

    _cache = merged
    return merged


def device(path, default=None):
    """One entry, addressed as "section.key" (e.g. "cameras.gripper_serial")."""
    section, _, key = str(path).partition(".")
    if not key:
        raise ValueError(f'device() takes "section.key", got "{path}"')
    value = load_devices().get(section, {}).get(key, default)
    if value is None and default is None:
        raise KeyError(f'No device entry "{path}" in {devices_file()}')
    return value


def device_str(path, default=None):
    """As device(), rendered for a launch argument's default_value.

    Launch arguments are strings; an int in the YAML (a port, a bitrate) has to
    be handed over as text or the substitution machinery rejects it.
    """
    return str(device(path, default))
