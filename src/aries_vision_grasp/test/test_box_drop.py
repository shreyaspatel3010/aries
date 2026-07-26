import math

import numpy as np
import pytest

from aries_vision_grasp.box_drop import (
    AutomaticBoxSettings,
    compute_box_drop_layout,
    compute_probe_insertion,
    derive_automatic_box_settings,
    probe_tip_from_attached_geometry,
    rotation_aligning_vectors,
)

# The rover's actual base box, as logged by the automatic planner.
ROVER_BOX_DIMS = [0.140, 0.200, 0.150]
ROVER_BOX_CENTRE = [0.003, 0.215, 0.287]


def rover_layout(rotation=None):
    settings = derive_automatic_box_settings(ROVER_BOX_DIMS)
    return compute_box_drop_layout(
        ROVER_BOX_CENTRE,
        ROVER_BOX_DIMS,
        np.eye(3) if rotation is None else rotation,
        settings,
    )


def yaw_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_box_layout_uses_long_axis_and_top_rim():
    settings = AutomaticBoxSettings(
        wall_thickness_m=0.01,
        edge_clearance_m=0.015,
        release_heights_m=(0.06, 0.12, 0.06),
    )
    layout = compute_box_drop_layout(
        [0.15, 0.20, 0.18],
        [0.18, 0.36, 0.12],
        np.eye(3),
        settings,
    )
    assert layout.probe_axis_name == 'Y'
    assert layout.probe_axis_yaw_rad == pytest.approx(math.pi / 2.0)
    assert layout.top_center == pytest.approx([0.15, 0.20, 0.24])
    assert len(layout.candidate_points) == 2
    assert layout.candidate_points[0] == pytest.approx([0.15, 0.20, 0.30])
    assert layout.candidate_points[1] == pytest.approx([0.15, 0.20, 0.36])
    assert layout.release_volume_center == pytest.approx([0.15, 0.20, 0.33])
    assert layout.release_volume_dimensions == pytest.approx([0.039, 0.045, 0.06])
    assert layout.probe_fits


def test_box_yaw_rotates_automatic_probe_axis():
    settings = derive_automatic_box_settings([0.20, 0.40, 0.10])
    layout = compute_box_drop_layout(
        [0.0, 0.0, 0.0], [0.20, 0.40, 0.10], yaw_rotation(math.radians(30.0)), settings,
    )
    assert layout.probe_axis_yaw_rad == pytest.approx(math.radians(120.0))


def test_box_layout_reports_probe_that_does_not_fit():
    settings = derive_automatic_box_settings([0.10, 0.20, 0.10])
    layout = compute_box_drop_layout(
        [0.0, 0.0, 0.0], [0.10, 0.20, 0.10], np.eye(3), settings,
    )
    assert not layout.probe_fits
    assert layout.probe_width_fits
    assert not layout.probe_length_fits


def test_box_layout_rejects_probe_width_that_cannot_enter():
    settings = derive_automatic_box_settings([0.05, 0.20, 0.10])
    layout = compute_box_drop_layout(
        [0.0, 0.0, 0.0], [0.05, 0.20, 0.10], np.eye(3), settings,
    )
    assert not layout.probe_width_fits


@pytest.mark.parametrize('dimensions', ([0.0, 0.2, 0.1], [0.1, -0.2, 0.1], [0.1, 0.2]))
def test_box_layout_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        derive_automatic_box_settings(dimensions)


def test_automatic_settings_need_only_box_dimensions():
    settings = derive_automatic_box_settings([0.18, 0.36, 0.12])
    assert settings.wall_thickness_m == pytest.approx(0.009)
    assert settings.edge_clearance_m == pytest.approx(0.012)
    assert settings.release_heights_m == pytest.approx((0.035, 0.059))


def test_current_box_targets_only_central_low_release_zone():
    dimensions = [0.14, 0.20, 0.15]
    settings = derive_automatic_box_settings(dimensions)
    layout = compute_box_drop_layout(
        [0.003, 0.215, 0.287], dimensions, np.eye(3), settings,
    )
    assert layout.top_center == pytest.approx([0.003, 0.215, 0.362])
    assert layout.release_volume_center == pytest.approx([0.003, 0.215, 0.412])
    assert layout.release_volume_dimensions == pytest.approx([0.0306, 0.045, 0.030])
    assert layout.release_volume_dimensions[0] < 0.5 * layout.usable_xy[0]
    assert layout.release_volume_dimensions[1] < 0.5 * layout.usable_xy[1]


def test_release_volume_stays_available_when_probe_does_not_fit():
    settings = derive_automatic_box_settings([0.05, 0.08, 0.10])
    layout = compute_box_drop_layout(
        [0.0, 0.0, 0.0], [0.05, 0.08, 0.10], np.eye(3), settings,
    )
    assert not layout.probe_fits
    assert np.all(layout.release_volume_dimensions > 0.0)
    assert layout.release_volume_center[2] > layout.top_center[2]


def test_release_volume_does_not_require_a_horizontal_probe_axis():
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    settings = derive_automatic_box_settings([0.10, 0.30, 0.10])
    layout = compute_box_drop_layout(
        [0.0, 0.0, 0.0], [0.10, 0.30, 0.10], rotation, settings,
    )
    assert np.all(layout.release_volume_dimensions > 0.0)
    assert layout.probe_axis_yaw_rad == 0.0


@pytest.mark.parametrize(
    ('source', 'target'),
    (
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.2, -0.4, 0.8], [-0.3, 0.9, 0.1]),
    ),
)
def test_rotation_aligns_probe_axis(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    rotation = rotation_aligning_vectors(source, target)
    actual = rotation @ (source / np.linalg.norm(source))
    expected = target / np.linalg.norm(target)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_rotation_rejects_zero_probe_axis():
    with pytest.raises(ValueError):
        rotation_aligning_vectors([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ('fat_sign', 'expected_tip'),
    (
        (-1, [0.1, 0.2, 0.4]),
        (+1, [0.1, 0.2, 0.2]),
    ),
)
def test_attached_probe_tip_is_opposite_the_stl_fat_end(fat_sign, expected_tip):
    tip = probe_tip_from_attached_geometry(
        [0.1, 0.2, 0.3], [0.0, 0.0, 4.0], 0.2, fat_sign
    )
    assert tip == pytest.approx(expected_tip)


@pytest.mark.parametrize(
    ('centre', 'axis', 'length', 'fat_sign'),
    (
        ([0.0, 0.0], [0.0, 0.0, 1.0], 0.2, -1),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.2, -1),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, -1),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.2, 0),
    ),
)
def test_attached_probe_tip_rejects_invalid_geometry(centre, axis, length, fat_sign):
    with pytest.raises(ValueError):
        probe_tip_from_attached_geometry(centre, axis, length, fat_sign)


def test_probe_needs_tilting_but_does_fit_the_rover_box_diagonally():
    """Why the release is tilted rather than flat: the probe is too long to lie
    along the opening, yet short enough to fit the interior diagonal — so
    leaning it in can actually land it inside instead of across the mouth."""
    layout = rover_layout()
    interior = np.array([
        layout.usable_xy[0],
        layout.usable_xy[1],
        layout.dimensions[2] - layout.settings.wall_thickness_m,
    ])
    probe_length = layout.settings.probe_length_m
    assert not layout.probe_length_fits            # cannot lie flat
    assert probe_length > layout.usable_xy.max()
    assert float(np.linalg.norm(interior)) > probe_length   # but fits diagonally


def test_shallow_tilt_keeps_the_probe_inside_the_box():
    """A shallow lean needs less vertical drop than the box is deep, so the
    released probe settles inside rather than sticking out of the mouth."""
    layout = rover_layout()
    interior_depth = layout.dimensions[2] - layout.settings.wall_thickness_m
    shallow = compute_probe_insertion(layout, math.radians(30.0), 0.050, +1.0, 0.035)
    steep = compute_probe_insertion(layout, math.radians(75.0), 0.050, +1.0, 0.035)
    drop_shallow = float(shallow.trailing_end[2] - shallow.leading_end[2])
    drop_steep = float(steep.trailing_end[2] - steep.leading_end[2])
    assert drop_shallow < interior_depth      # fully containable
    assert drop_steep > interior_depth        # would stand out of the box
    assert shallow.clears_opening


def test_insertion_puts_the_leading_end_below_the_rim():
    layout = rover_layout()
    depth = 0.050
    insertion = compute_probe_insertion(
        layout, math.radians(45.0), depth, axis_sign=+1.0, entry_offset_m=0.035
    )
    rim_z = float(layout.top_center[2])
    assert insertion.leading_end[2] == pytest.approx(rim_z - depth, abs=1e-9)
    # The rest of the probe leans out above the rim.
    assert insertion.trailing_end[2] > rim_z
    assert insertion.probe_centre[2] > rim_z
    # Centre stays the midpoint of the two ends.
    midpoint = 0.5 * (insertion.leading_end + insertion.trailing_end)
    assert insertion.probe_centre == pytest.approx(midpoint)
    # Ends are a full probe length apart.
    span = float(np.linalg.norm(insertion.trailing_end - insertion.leading_end))
    assert span == pytest.approx(layout.settings.probe_length_m)


@pytest.mark.parametrize('tilt_deg', (30.0, 45.0, 60.0))
def test_tip_only_depth_leaves_probe_centre_outside_box(tilt_deg):
    """The fallback inserts the endpoint, never the probe centre/gripper."""
    layout = rover_layout()
    insertion = compute_probe_insertion(
        layout,
        math.radians(tilt_deg),
        0.020,
        axis_sign=+1.0,
        entry_offset_m=0.035,
    )
    local_tip = layout.rotation.T @ (
        insertion.leading_end - layout.top_center
    )
    local_centre = layout.rotation.T @ (
        insertion.probe_centre - layout.top_center
    )
    assert insertion.clears_opening
    assert local_tip[2] == pytest.approx(-0.020)
    assert local_centre[2] >= 0.005


def test_insertion_axis_signs_lean_opposite_ways():
    """The two signs point opposite probe ends into the box; the caller picks
    whichever needs less wrist travel."""
    layout = rover_layout()
    a = compute_probe_insertion(layout, math.radians(45.0), 0.050, +1.0, 0.035)
    b = compute_probe_insertion(layout, math.radians(45.0), 0.050, -1.0, 0.035)
    assert a.leading_end[2] == pytest.approx(b.leading_end[2])
    # Same elevation, mirrored horizontal heading.
    assert a.axis[2] == pytest.approx(b.axis[2])
    assert a.axis[1] == pytest.approx(-b.axis[1], abs=1e-9)


def test_insertion_flags_a_pose_that_would_clip_the_wall():
    """A shallow tilt entering at the box centre leaves through a side wall,
    not the opening; that candidate must be rejected before planning."""
    layout = rover_layout()
    clipping = compute_probe_insertion(layout, math.radians(30.0), 0.050, +1.0, 0.0)
    assert not clipping.clears_opening
    # Displacing the entry along the long axis fixes it.
    clearing = compute_probe_insertion(layout, math.radians(30.0), 0.050, +1.0, 0.040)
    assert clearing.clears_opening


def test_insertion_depth_cannot_exceed_the_box_interior():
    layout = rover_layout()
    insertion = compute_probe_insertion(layout, math.radians(60.0), 10.0, +1.0, 0.0)
    interior_depth = layout.dimensions[2] - layout.settings.wall_thickness_m
    assert insertion.depth_m == pytest.approx(interior_depth)
    assert insertion.leading_end[2] >= float(layout.center[2]) - 0.5 * layout.dimensions[2]


def test_insertion_follows_a_rotated_box():
    """A yawed box must be leaned into along its own long axis."""
    layout = rover_layout(yaw_rotation(math.radians(35.0)))
    insertion = compute_probe_insertion(layout, math.radians(45.0), 0.050, +1.0, 0.035)
    long_axis = layout.rotation[:, 1]   # 200 mm side is local Y
    horizontal = np.array([insertion.axis[0], insertion.axis[1], 0.0])
    horizontal /= np.linalg.norm(horizontal)
    assert abs(float(np.dot(horizontal, long_axis))) == pytest.approx(1.0, abs=1e-6)
