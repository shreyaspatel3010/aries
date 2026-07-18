import math

import numpy as np
import pytest

from aries_vision_grasp.box_drop import (
    AutomaticBoxSettings,
    compute_box_drop_layout,
    derive_automatic_box_settings,
    rotation_aligning_vectors,
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
