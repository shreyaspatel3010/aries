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
    """Physically identified after the 2026-08-12 reassembly, axis by axis.

    _3 is the FRONT wheel and _1 the REAR, measured from base_link through TF.
    The joint origins in the xacro suggest the opposite order because _1/_2 are
    parented to the boggie and _3 to the rocker.
    """
    assert MODULE.AXIS_JOINTS == (
        "R_3_Wheel_Joint",  # axis 0 Right-Front
        "L_2_Wheel_Joint",  # axis 1 Left-Mid
        "L_1_Wheel_Joint",  # axis 2 Left-Rear
        "R_1_Wheel_Joint",  # axis 3 Right-Rear
        "R_2_Wheel_Joint",  # axis 4 Right-Mid
        "L_3_Wheel_Joint",  # axis 5 Left-Front
    )


def test_left_wheel_visualization_sign_matches_opposite_motor_mounting():
    """Signs follow the side each axis is on, not a 0..2 / 3..5 split."""
    assert MODULE.DEFAULT_AXIS_SIGNS == (
        1.0,   # axis 0 right
        -1.0,  # axis 1 left
        -1.0,  # axis 2 left
        1.0,   # axis 3 right
        1.0,   # axis 4 right
        -1.0,  # axis 5 left
    )


def test_joint_mapping_agrees_with_the_drive_side_lists():
    """AXIS_JOINTS and DEFAULT_AXIS_SIGNS must not drift apart.

    The R_/L_ prefix of each joint is the side that axis is on, so it has to
    match the sign applied to that same axis. Catches an edit to one tuple that
    forgets the other.
    """
    right_axes = [0, 4, 3]
    left_axes = [5, 1, 2]
    for axis in right_axes:
        assert MODULE.AXIS_JOINTS[axis].startswith("R_")
        assert MODULE.DEFAULT_AXIS_SIGNS[axis] > 0
    for axis in left_axes:
        assert MODULE.AXIS_JOINTS[axis].startswith("L_")
        assert MODULE.DEFAULT_AXIS_SIGNS[axis] < 0
    assert sorted(MODULE.AXIS_JOINTS) == sorted(set(MODULE.AXIS_JOINTS))


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
