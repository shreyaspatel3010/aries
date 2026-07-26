"""Unit tests for the configurable deposit box and its derived dump pose."""

import math

import numpy as np
import pytest

from aries_soil_sample.deposit import DepositBox, rpy_matrix

# The rover box the probe task already describes: centre, outer size, level.
CENTRE = [0.003, 0.215, 0.287]
DIMS = [0.14, 0.20, 0.15]


def box(**kw):
    kw.setdefault('centre', CENTRE)
    kw.setdefault('dimensions', DIMS)
    return DepositBox(**kw)


def test_rim_is_the_top_of_a_level_box():
    b = box()
    assert b.rim_z == pytest.approx(0.287 + 0.075)


def test_rim_of_a_tilted_box_is_its_highest_corner():
    """A rolled box presents a higher edge than centre_z + dz/2."""
    level = box().rim_z
    tilted = box(rpy=[math.radians(20.0), 0.0, 0.0]).rim_z
    assert tilted > level
    # Highest corner of a 0.20 x 0.15 cross-section rolled by 20 deg.
    expected = 0.287 + 0.5 * (0.20 * math.sin(math.radians(20.0))
                              + 0.15 * math.cos(math.radians(20.0)))
    assert tilted == pytest.approx(expected, abs=1e-9)


def test_dump_contact_sits_above_the_rim_over_the_centre():
    b = box()
    p = b.dump_contact(0.038)
    assert p[0] == pytest.approx(CENTRE[0])
    assert p[1] == pytest.approx(CENTRE[1])
    assert p[2] == pytest.approx(b.rim_z + 0.038)
    # This is the pose measured as 4/4 collision-free over the rover box.
    assert p[2] == pytest.approx(0.400, abs=1e-9)


def test_dump_pose_follows_the_box_when_it_moves():
    """Move the box and the dump pose moves with it -- nothing to keep in sync."""
    a = box().dump_contact(0.038)
    moved = box(centre=[0.20, -0.10, 0.30]).dump_contact(0.038)
    assert moved[0] == pytest.approx(0.20)
    assert moved[1] == pytest.approx(-0.10)
    assert moved[2] == pytest.approx(0.30 + 0.075 + 0.038)
    assert not np.allclose(a, moved)


def test_dump_offset_rotates_with_the_box():
    b = box(rpy=[0.0, 0.0, math.radians(90.0)])
    p = b.dump_contact(0.04, offset_xy=[0.05, 0.0])
    # +X in the box frame is +Y in the planning frame after a 90 deg yaw.
    assert p[0] == pytest.approx(CENTRE[0], abs=1e-9)
    assert p[1] == pytest.approx(CENTRE[1] + 0.05, abs=1e-9)


def test_over_opening_accepts_the_centre_and_rejects_outside():
    b = box()
    assert b.over_opening(b.dump_contact(0.04))
    assert not b.over_opening([CENTRE[0] + 0.5, CENTRE[1], 0.4])
    # Just inside vs just outside the 70 mm half-width.
    assert b.over_opening([CENTRE[0] + 0.069, CENTRE[1], 0.4])
    assert not b.over_opening([CENTRE[0] + 0.071, CENTRE[1], 0.4])


def test_walls_narrow_the_usable_opening():
    b = box(wall_thickness_m=0.010)
    assert b.opening_half_extents == pytest.approx([0.060, 0.090])
    assert not b.over_opening([CENTRE[0] + 0.065, CENTRE[1], 0.4])
    assert b.over_opening([CENTRE[0] + 0.055, CENTRE[1], 0.4])


def test_validate_accepts_the_measured_configuration():
    ok, reason = box().validate(0.038)
    assert ok
    assert 'rim at z=0.362' in reason


def test_validate_rejects_a_negative_clearance():
    ok, reason = box().validate(-0.01)
    assert not ok
    assert 'inside the box' in reason


def test_validate_rejects_a_degenerate_box():
    ok, reason = box(dimensions=[0.14, 0.0, 0.15]).validate(0.04)
    assert not ok
    assert 'zero dimension' in reason


def test_validate_rejects_walls_thicker_than_the_box():
    ok, reason = box(wall_thickness_m=0.20).validate(0.04)
    assert not ok
    assert 'no opening' in reason


def test_validate_rejects_a_dump_point_over_a_wall():
    """The mistake that puts soil on the deck instead of in the box."""
    ok, reason = box().validate(0.04, offset_xy=[0.090, 0.0])
    assert not ok
    assert 'miss the box' in reason


def test_validate_margin_keeps_the_dump_away_from_the_rim():
    b = box()
    assert b.validate(0.04, offset_xy=[0.055, 0.0])[0]
    ok, _ = b.validate(0.04, offset_xy=[0.055, 0.0], margin_m=0.030)
    assert not ok


def test_rpy_matrix_is_a_rotation():
    R = rpy_matrix(0.3, -0.2, 1.1)
    assert R.T @ R == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_corners_count_and_centroid():
    b = box(rpy=[0.1, 0.2, 0.3])
    assert b.corners.shape == (8, 3)
    assert b.corners.mean(axis=0) == pytest.approx(b.centre)
