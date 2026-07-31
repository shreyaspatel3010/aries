"""Tests for physical ODrive encoder to wheel-joint conversion."""

import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "nodes"
    / "publish_wheel_joints.py"
)
SPEC = importlib.util.spec_from_file_location(
    "publish_wheel_joints", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_axis_to_physical_wheel_mapping():
    assert MODULE.AXIS_JOINTS == (
        "R_1_Wheel_Joint",
        "R_2_Wheel_Joint",
        "R_3_Wheel_Joint",
        "L_1_Wheel_Joint",
        "L_2_Wheel_Joint",
        "L_3_Wheel_Joint",
    )


def test_left_wheel_visualization_sign_matches_opposite_motor_mounting():
    assert MODULE.DEFAULT_AXIS_SIGNS == (
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
    )


def test_encoder_turns_are_relative_radians():
    position, velocity = MODULE.encoder_to_joint(12.25, 0.5, 12.0)
    assert position == pytest.approx(math.pi / 2.0)
    assert velocity == pytest.approx(math.pi)


def test_encoder_sign_applies_to_position_and_velocity():
    position, velocity = MODULE.encoder_to_joint(2.5, 0.25, 2.0, -1.0)
    assert position == pytest.approx(-math.pi)
    assert velocity == pytest.approx(-math.pi / 2.0)


@pytest.mark.parametrize(
    "values",
    [
        (math.nan, 0.0, 0.0, 1.0),
        (0.0, math.inf, 0.0, 1.0),
        (0.0, 0.0, math.nan, 1.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_invalid_encoder_values_fail_closed(values):
    with pytest.raises(ValueError):
        MODULE.encoder_to_joint(*values)
