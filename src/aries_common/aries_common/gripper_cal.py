"""Bench calibration for the ST3215 secondary gripper, from one YAML file.

``aries_common/config/gripper_st3215.yaml`` holds the four numbers that come off
the hardware rather than out of the CAD. This module reads them and DERIVES
everything the rest of the stack needs, so those derived values exist in exactly
one place instead of being copied into the URDF, the SRDF, the teleop overlay
and the hardware component by hand:

    from aries_common.gripper_cal import gripper_cal

    cal = gripper_cal()
    cal["closed_steps"]        3307   -> the hardware component
    cal["invert"]              False  -> the hardware component
    cal["open_travel_m"]       ...    -> the URDF joint limit
    cal["command_open_rad"]    ...    -> SRDF `open`, teleop, min_pos

WHY THE DERIVATION LIVES HERE
Changing the closed point changes the stroke, which changes both the joint limit
and the commanded open angle. When those were literals in five files, moving one
of them meant finding the other four - and a missed copy is silent: the gripper
still works, it just stops somewhere wrong or stalls against a stop.

Kept SEPARATE from devices.py deliberately. That file is device identity - which
port, which serial, which address - and says so in its own header. This is
tuning.
"""

import math
import os
from pathlib import Path

# Steps per radian of pinion, from the servo's 4096-step single turn.
STEPS_PER_RAD = 4096.0 / (2.0 * math.pi)

# Jaw travel per radian, m. Measured in scripts/build_gripper_st3215_meshes.py:
# 21 teeth on a 3.0000 mm rack pitch, so 21 * 3 / (2*pi) mm per radian. This one
# IS derivable from the CAD, which is why it is not in the YAML.
PITCH_RADIUS_M = 0.01002676

# The joint angle this whole stack calls "closed", on both grippers.
Q_CLOSED = 0.07

# Fallback for a missing or unreadable file, so a broken edit degrades to the
# shipped values instead of taking the stack down at launch. Deliberately a copy
# of what the YAML ships with.
DEFAULTS = {
    "closed_steps": 3307,
    "open_stop_steps": 489,
    "invert": False,
    "margin_steps": 50,
    "limit_slack_steps": 25,
    "position_correction_reg31": 1232,
}

_ENV_OVERRIDE = "ARIES_GRIPPER_CAL_FILE"
_cache = None


def cal_file():
    """Path the calibration is read from, or None when only defaults apply."""
    override = os.environ.get(_ENV_OVERRIDE, "").strip()
    if override:
        return Path(override)
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("aries_common")) / "config" / "gripper_st3215.yaml"
    except Exception:
        return None


def gripper_cal(refresh=False):
    """The calibration, with every derived value computed from it."""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    values = dict(DEFAULTS)
    path = cal_file()
    if path is not None and path.is_file():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001 - a bad edit must not stop a launch
            print(f"[aries_common] Could not read {path} ({exc}); using built-in gripper defaults")
            loaded = {}
        for key in DEFAULTS:
            if key in loaded:
                values[key] = loaded[key]

    closed = int(values["closed_steps"])
    open_stop = int(values["open_stop_steps"])
    margin = int(values["margin_steps"])
    slack = int(values["limit_slack_steps"])
    stroke = abs(closed - open_stop)

    # Opening is whichever way the open stop lies from closed. The sign of the
    # step difference carries it, so this stays right if the pair is ever
    # swapped - unlike a hard-coded direction.
    values["stroke_steps"] = stroke
    values["command_open_rad"] = Q_CLOSED - (stroke - margin) / STEPS_PER_RAD
    values["limit_open_rad"] = Q_CLOSED - (stroke + slack) / STEPS_PER_RAD
    # The URDF's gripper_open_travel is measured from q = 0, not from closed.
    values["open_travel_m"] = -values["limit_open_rad"] * PITCH_RADIUS_M
    values["full_gap_m"] = 2.0 * PITCH_RADIUS_M * stroke / STEPS_PER_RAD
    values["command_gap_m"] = 2.0 * PITCH_RADIUS_M * (Q_CLOSED - values["command_open_rad"])

    _cache = values
    return values


def cal_str(key):
    """One entry, rendered for a launch argument's default_value.

    Launch arguments are strings, and a bool has to go over as the lowercase
    "true"/"false" that xacro and the choices= lists expect - str(False) gives
    "False", which fails a choices check and reads as a string to xacro.
    """
    value = gripper_cal()[key]
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
