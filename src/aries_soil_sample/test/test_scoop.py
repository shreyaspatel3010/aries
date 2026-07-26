"""Unit tests for the scoop trajectory geometry and capture verdict."""

import math

import numpy as np
import pytest

from aries_soil_sample.scoop import (
    BUCKET_LENGTH_M,
    CAPTURED,
    EMPTY,
    UNKNOWN,
    ScoopParams,
    capture_verdict,
    clamp_penetration_depth,
    entry_axis,
    link_position_for_contact,
    nominal_bucket_capacity_m3,
    plan_scoop,
    roll_frames,
    tool_frame,
)

SURFACE = np.array([0.52, 0.02, -0.010])
UP = np.array([0.0, 0.0, 1.0])


# --- penetration depth ------------------------------------------------------

def test_depth_within_limits_is_untouched():
    depth, note = clamp_penetration_depth(ScoopParams(depth_m=0.030))
    assert depth == pytest.approx(0.030)
    assert note is None


def test_depth_is_clamped_by_the_operator_limit():
    depth, note = clamp_penetration_depth(
        ScoopParams(depth_m=0.090, max_depth_m=0.040))
    assert depth == pytest.approx(0.040)
    assert 'max_depth_m' in note


def test_depth_is_clamped_by_the_bucket_length():
    """Past bucket length the four-bar itself would enter the soil."""
    depth, note = clamp_penetration_depth(
        ScoopParams(depth_m=0.200, max_depth_m=0.500, depth_margin_m=0.010))
    assert depth == pytest.approx(BUCKET_LENGTH_M - 0.010)
    assert 'bucket length' in note


# --- entry axis -------------------------------------------------------------

def test_zero_attack_enters_straight_down_the_normal():
    axis = entry_axis(UP, 0.0)
    assert axis == pytest.approx([0.0, 0.0, -1.0])


def test_entry_axis_follows_a_tilted_surface():
    normal = np.array([0.0, math.sin(math.radians(20)), math.cos(math.radians(20))])
    axis = entry_axis(normal, 0.0)
    assert axis == pytest.approx(-normal)
    assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_attack_angle_tilts_by_the_requested_amount():
    axis = entry_axis(UP, 30.0, azimuth_ref=[1.0, 0.0, 0.0])
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    # 30 deg off the inward normal, leaning toward +X.
    assert math.degrees(math.acos(np.clip(np.dot(axis, -UP), -1, 1))) == pytest.approx(30.0)
    assert axis[0] > 0.0


def test_attack_azimuth_parallel_to_the_normal_falls_back_gracefully():
    axis = entry_axis(UP, 30.0, azimuth_ref=[0.0, 0.0, 1.0])
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert axis[2] < 0.0


# --- waypoints --------------------------------------------------------------

def test_scoop_waypoints_are_ordered_approach_entry_penetrate_extract():
    wps, depth, note = plan_scoop(SURFACE, UP, ScoopParams(standoff_m=0.06, depth_m=0.03))
    assert [w.label for w in wps] == ['approach', 'entry', 'penetrate', 'extract']
    assert note is None
    assert depth == pytest.approx(0.030)
    # Approach above the surface, penetrate below it.
    assert wps[0].position[2] == pytest.approx(SURFACE[2] + 0.060)
    assert wps[1].position[2] == pytest.approx(SURFACE[2])
    assert wps[2].position[2] == pytest.approx(SURFACE[2] - 0.030)


def test_extraction_retraces_the_entry_exactly():
    """A world-vertical lift out of an angled channel levers the tool against
    the material; leaving along the entry axis does not."""
    normal = np.array([0.10, -0.05, 1.0])
    normal = normal / np.linalg.norm(normal)
    wps, _, _ = plan_scoop(SURFACE, normal, ScoopParams())
    assert wps[3].position == pytest.approx(wps[0].position)
    for w in wps:
        assert w.tool_axis == pytest.approx(wps[0].tool_axis)
    # The stroke runs along the axis, not along world Z.
    stroke = wps[2].position - wps[1].position
    assert stroke / np.linalg.norm(stroke) == pytest.approx(wps[0].tool_axis)


def test_waypoints_follow_a_tilted_surface_rather_than_world_vertical():
    normal = np.array([0.0, math.sin(math.radians(25)), math.cos(math.radians(25))])
    wps, _, _ = plan_scoop(SURFACE, normal, ScoopParams(standoff_m=0.05, depth_m=0.02))
    # Approach is offset along the normal, so it moves in Y as well as Z.
    assert wps[0].position[1] != pytest.approx(SURFACE[1])
    assert np.linalg.norm(wps[0].position - SURFACE) == pytest.approx(0.05)


# --- tool frame -------------------------------------------------------------

def test_tool_frame_z_is_the_entry_axis_and_is_orthonormal():
    axis = entry_axis([0.1, 0.2, 1.0], 0.0)
    R = tool_frame(axis)
    assert R[:, 2] == pytest.approx(axis)
    assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_tool_frame_spends_the_free_roll_on_the_current_wrist():
    axis = np.array([0.0, 0.0, -1.0])
    prefer = np.array([0.0, 1.0, 0.0])
    R = tool_frame(axis, prefer_pinch=prefer)
    assert R[:, 0] == pytest.approx(prefer)
    assert R[:, 2] == pytest.approx(axis)


def test_tool_frame_ignores_a_preference_along_the_axis():
    axis = np.array([0.0, 0.0, -1.0])
    R = tool_frame(axis, prefer_pinch=[0.0, 0.0, 1.0])
    assert R[:, 2] == pytest.approx(axis)
    assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9)


# --- contact -> link --------------------------------------------------------

def test_link_position_puts_the_contact_where_asked():
    axis = np.array([0.0, 0.0, -1.0])
    R = tool_frame(axis, prefer_pinch=[1.0, 0.0, 0.0])
    offset = np.array([0.0, 0.001, 0.214])       # bucket contact at the entry angle
    contact = np.array([0.52, 0.02, -0.04])
    link = link_position_for_contact(contact, R, offset)
    # Round-trip: link origin plus the rotated offset must land on the contact.
    assert link + R @ offset == pytest.approx(contact)
    # Tool +Z points down, so the link sits ABOVE the contact.
    assert link[2] > contact[2]


def test_link_position_follows_the_tool_rotation():
    offset = np.array([0.0, 0.0, 0.220])
    contact = np.zeros(3)
    down = link_position_for_contact(contact, tool_frame([0, 0, -1]), offset)
    forward = link_position_for_contact(contact, tool_frame([1, 0, 0]), offset)
    assert down[2] == pytest.approx(0.220)
    assert forward[0] == pytest.approx(-0.220)


# --- capture verdict --------------------------------------------------------

def test_a_full_divot_is_a_captured_sample():
    verdict, reason = capture_verdict(3.0e-5, 0.025, 40, min_volume_m3=2.0e-5)
    assert verdict == CAPTURED
    assert 'cm^3' in reason


def test_untouched_ground_is_an_empty_bucket():
    verdict, reason = capture_verdict(1.0e-7, 0.001, 40, min_volume_m3=2.0e-5)
    assert verdict == EMPTY
    assert 'unchanged' in reason


def test_disturbed_but_insufficient_is_unknown_not_empty():
    """Absence of enough material is not evidence the bucket is empty."""
    verdict, _ = capture_verdict(5.0e-6, 0.010, 40, min_volume_m3=2.0e-5)
    assert verdict == UNKNOWN


def test_no_post_survey_is_unknown():
    assert capture_verdict(None, None, 0, min_volume_m3=2.0e-5)[0] == UNKNOWN


def test_too_few_shared_cells_is_unknown():
    verdict, reason = capture_verdict(9.9e-5, 0.03, 3,
                                      min_volume_m3=2.0e-5, min_cells=12)
    assert verdict == UNKNOWN
    assert 'not re-observed' in reason


def test_nominal_capacity_is_a_sane_fraction_of_the_bucket_envelope():
    cap = nominal_bucket_capacity_m3(0.35)
    assert 1e-5 < cap < 2.2e-4
    assert nominal_bucket_capacity_m3(0.0) == 0.0
    assert nominal_bucket_capacity_m3(2.0) == pytest.approx(nominal_bucket_capacity_m3(1.0))


# --- wrist-roll candidates --------------------------------------------------

def test_roll_frames_all_share_the_entry_axis_and_are_orthonormal():
    axis = entry_axis([0.1, 0.0, 1.0], 30.0)
    frames = roll_frames(axis, count=8)
    assert len(frames) == 8
    for R in frames:
        assert R[:, 2] == pytest.approx(axis)
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_roll_frames_tries_the_preferred_wrist_first():
    """The cheapest wrist motion is still tried before walking outward."""
    axis = np.array([0.0, 0.0, -1.0])
    prefer = np.array([0.0, 1.0, 0.0])
    frames = roll_frames(axis, prefer_pinch=prefer, count=6)
    assert frames[0][:, 0] == pytest.approx(prefer)
    assert frames[0] == pytest.approx(tool_frame(axis, prefer))


def test_roll_frames_walk_outward_in_both_directions():
    """Ordering is 0, +step, -step, +2step... so a near miss is tried early."""
    axis = np.array([0.0, 0.0, -1.0])
    frames = roll_frames(axis, prefer_pinch=[1.0, 0.0, 0.0], count=12)
    x0 = frames[0][:, 0]
    angles = [math.degrees(math.acos(np.clip(np.dot(x0, R[:, 0]), -1, 1)))
              for R in frames]
    assert angles[0] == pytest.approx(0.0, abs=1e-6)
    # frames 1 and 2 are the same magnitude either side of the preference
    assert angles[1] == pytest.approx(angles[2], abs=1e-6)
    assert angles[1] == pytest.approx(30.0, abs=1e-6)
    # and they are genuinely distinct orientations
    assert not np.allclose(frames[1], frames[2])


def test_roll_frames_covers_distinct_rolls():
    axis = np.array([0.0, 0.0, -1.0])
    frames = roll_frames(axis, count=12)
    xs = [R[:, 0] for R in frames]
    for i, a in enumerate(xs):
        for b in xs[i + 1:]:
            assert not np.allclose(a, b, atol=1e-6)


def test_roll_frames_count_one_is_just_the_preference():
    axis = np.array([0.0, 0.0, -1.0])
    frames = roll_frames(axis, prefer_pinch=[1.0, 0.0, 0.0], count=1)
    assert len(frames) == 1
    assert frames[0] == pytest.approx(tool_frame(axis, [1.0, 0.0, 0.0]))
