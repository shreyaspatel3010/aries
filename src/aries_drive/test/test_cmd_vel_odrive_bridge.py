"""Unit tests for the hardware-facing skid-steer conversion."""

import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "nodes"
    / "cmd_vel_odrive_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("cmd_vel_odrive_bridge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def convert(linear, angular, limit=10.0):
    return MODULE.twist_to_wheel_rps(
        linear,
        angular,
        track_width_m=0.566,
        wheel_circumference_m=0.697,
        max_wheel_rps=limit,
    )


def test_forward_respects_opposite_left_motor_mounting():
    right, left = convert(0.697, 0.0)
    assert right == pytest.approx(1.0)
    assert left == pytest.approx(-1.0)


def test_reverse_respects_opposite_left_motor_mounting():
    right, left = convert(-0.697, 0.0)
    assert right == pytest.approx(-1.0)
    assert left == pytest.approx(1.0)


def test_positive_yaw_turns_left_in_place():
    angular = 2.0 * 0.697 / 0.566
    right, left = convert(0.0, angular)
    assert right == pytest.approx(1.0)
    assert left == pytest.approx(1.0)


def test_wheel_limit_preserves_curvature():
    unlimited = convert(0.45, 2.1)
    limited = convert(0.45, 2.1, limit=0.5)
    assert max(abs(limited[0]), abs(limited[1])) == pytest.approx(0.5)
    assert limited[0] / limited[1] == pytest.approx(
        unlimited[0] / unlimited[1]
    )


@pytest.mark.parametrize(
    "values",
    [
        (math.nan, 0.0),
        (0.0, math.inf),
        (0.0, 0.0, -1.0),
    ],
)
def test_invalid_inputs_fail_closed(values):
    if len(values) == 2:
        with pytest.raises(ValueError):
            convert(*values)
    else:
        with pytest.raises(ValueError):
            convert(values[0], values[1], limit=values[2])


def test_ramp_limits_step_without_overshoot():
    assert MODULE.ramp(0.0, 1.0, 0.2) == pytest.approx(0.2)
    assert MODULE.ramp(0.9, 1.0, 0.2) == pytest.approx(1.0)
    assert MODULE.ramp(0.0, -1.0, 0.2) == pytest.approx(-0.2)


@pytest.mark.parametrize(
    ("enable_requested", "armed"),
    [(False, False), (False, True), (True, False)],
)
def test_disarmed_periodic_output_stays_silent(enable_requested, armed):
    assert (
        MODULE.select_drive_output_mode(enable_requested, armed, True)
        == "silent"
    )


def test_armed_stale_command_continuously_commands_stop():
    assert MODULE.select_drive_output_mode(True, True, False) == "stop"


def test_armed_fresh_command_is_forwarded():
    assert MODULE.select_drive_output_mode(True, True, True) == "drive"
