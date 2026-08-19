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
        "serial_port": "/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00",
    },
    "rover": {"can_interface": "can0", "can_bitrate": 250000},
    "imu": {"port": "/dev/microstrain_main"},
    "cameras": {
        "gripper_serial": "216322070216",
        "front_serial": "207522077539",
    },
    "joystick": {"device": "/dev/input/js0"},
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
