"""Unit tests for the held-probe evidence pooling and jaw-volume geometry."""

import numpy as np
import pytest

from aries_vision_grasp.grasp_verification import (
    EMPTY,
    HELD,
    UNKNOWN,
    HeldProbeEvidence,
    cloud_is_probe_like,
    empty_close_gap,
    jaw_region_mask,
    jaw_volume_near_range,
    self_filter_blinds_jaw_volume,
)

# Four-bar contact point in the planning link frame, and the tool approach
# axis, as the node reports them for the coaxial probe grasp.
CONTACT = np.array([0.0, 0.001, 0.219])
AXIS = np.array([0.0, 0.0, 1.0])
RADIUS = 0.055
ALONG_LO = -0.030
ALONG_HI = 0.170


def held_probe_cloud(n=200, offset=(0.0, 0.0)):
    """Points along a rod protruding from the jaws down the tool axis."""
    rng = np.random.default_rng(7)
    along = rng.uniform(0.01, 0.15, n)
    pts = np.column_stack([
        CONTACT[0] + offset[0] + rng.normal(0.0, 0.004, n),
        CONTACT[1] + offset[1] + rng.normal(0.0, 0.004, n),
        CONTACT[2] + along,
    ])
    return pts


def test_jaw_region_keeps_the_held_rod():
    mask = jaw_region_mask(held_probe_cloud(), CONTACT, AXIS, RADIUS, ALONG_LO, ALONG_HI)
    assert mask.all()


def test_jaw_region_rejects_points_beside_and_beyond_the_jaws():
    beside = held_probe_cloud() + np.array([0.20, 0.0, 0.0])
    beyond = held_probe_cloud() + np.array([0.0, 0.0, 0.40])
    behind = held_probe_cloud() - np.array([0.0, 0.0, 0.40])
    for cloud in (beside, beyond, behind):
        assert not jaw_region_mask(cloud, CONTACT, AXIS, RADIUS, ALONG_LO, ALONG_HI).any()


def test_jaw_region_follows_the_axis_not_the_world():
    """A tilted tool carries the volume with it."""
    axis = np.array([0.0, np.sin(np.radians(30.0)), np.cos(np.radians(30.0))])
    along = np.linspace(0.01, 0.15, 60)
    tilted = CONTACT + np.outer(along, axis)
    assert jaw_region_mask(tilted, CONTACT, axis, RADIUS, ALONG_LO, ALONG_HI).all()
    # The same points judged against the untilted axis fall out of the cylinder.
    assert not jaw_region_mask(tilted, CONTACT, AXIS, RADIUS, ALONG_LO, ALONG_HI).all()


def test_empty_region_has_no_points():
    """An empty gripper leaves the volume empty even with a busy scene."""
    rng = np.random.default_rng(3)
    floor = np.column_stack([
        rng.uniform(0.3, 0.9, 500),
        rng.uniform(-0.4, 0.4, 500),
        rng.uniform(-0.3, -0.1, 500),
    ])
    assert int(jaw_region_mask(floor, CONTACT, AXIS, RADIUS, ALONG_LO, ALONG_HI).sum()) == 0


def test_rod_is_probe_like_and_a_blob_is_not():
    ok, elongation, extent = cloud_is_probe_like(held_probe_cloud())
    assert ok
    assert elongation > 3.0
    assert extent > 0.10

    rng = np.random.default_rng(11)
    blob = CONTACT + rng.normal(0.0, 0.012, (200, 3))
    ok, _, _ = cloud_is_probe_like(blob)
    assert not ok


def test_too_few_points_is_not_probe_like():
    ok, _, _ = cloud_is_probe_like(held_probe_cloud(n=8))
    assert not ok


def test_empty_close_gap_flags_a_shut_gripper():
    # 0.1 mm gap on a 30 mm probe with a 24 mm tolerance: nothing was there.
    assert empty_close_gap(0.0001, 0.030, 0.024)
    # A probe stopping the jaws at its true width is not an empty close.
    assert not empty_close_gap(0.030, 0.030, 0.024)
    # Nor is a slightly over-tight grip within tolerance.
    assert not empty_close_gap(0.010, 0.030, 0.024)


def test_evidence_needs_a_minimum_number_of_decisive_votes():
    ev = HeldProbeEvidence(window_sec=10.0, min_votes=3)
    ev.add(EMPTY, 0.0)
    ev.add(EMPTY, 1.0)
    assert ev.verdict(1.0) == UNKNOWN
    ev.add(EMPTY, 2.0)
    assert ev.verdict(2.0) == EMPTY


def test_unknown_votes_never_decide():
    """Frames where the sensors could not look must not move the verdict."""
    ev = HeldProbeEvidence(window_sec=10.0, min_votes=3)
    for t in range(20):
        ev.add(UNKNOWN, float(t))
    assert ev.verdict(19.0) == UNKNOWN


def test_solid_fits_outvote_the_occluded_frames():
    """The fingers occlude the held probe most of the time.

    Those EMPTY frames are the expected noise around a good grasp, so
    min_held_votes solid fits settle it however many sit alongside.
    """
    ev = HeldProbeEvidence(window_sec=10.0, min_votes=3, min_held_votes=2)
    for t in range(8):
        ev.add(EMPTY, float(t))
    assert ev.verdict(7.0) == EMPTY
    ev.add(HELD, 8.0)
    ev.add(HELD, 9.0)
    assert ev.verdict(9.0) == HELD


def test_one_held_look_withdraws_an_empty_verdict():
    """A single fit is not proof of a grasp, but it does break the unanimity."""
    ev = HeldProbeEvidence(window_sec=10.0, min_votes=3, min_held_votes=2,
                           empty_fraction=0.75)
    for t in range(4):
        ev.add(EMPTY, float(t))
    assert ev.verdict(3.0) == EMPTY
    ev.add(HELD, 4.0)
    assert ev.verdict(4.0) == UNKNOWN


def test_votes_expire_out_of_the_window():
    ev = HeldProbeEvidence(window_sec=5.0, min_votes=3)
    for t in range(4):
        ev.add(EMPTY, float(t))
    assert ev.verdict(3.0) == EMPTY
    assert ev.verdict(30.0) == UNKNOWN


def test_reset_clears_the_pool():
    ev = HeldProbeEvidence(window_sec=10.0, min_votes=3)
    for t in range(4):
        ev.add(EMPTY, float(t))
    assert ev.verdict(3.0) == EMPTY
    ev.reset()
    assert ev.verdict(3.0) == UNKNOWN


def test_last_held_sec_reports_the_newest_held_vote():
    ev = HeldProbeEvidence()
    assert ev.last_held_sec() is None
    ev.add(HELD, 1.0)
    ev.add(EMPTY, 2.0)
    ev.add(HELD, 3.0)
    ev.add(EMPTY, 4.0)
    assert ev.last_held_sec() == pytest.approx(3.0)


def test_summary_names_the_verdict():
    ev = HeldProbeEvidence(window_sec=10.0, min_votes=3)
    for t in range(3):
        ev.add(EMPTY, float(t))
    text = ev.summary(2.0)
    assert 'empty=3' in text
    assert text.endswith(EMPTY)


def test_the_logged_failure_reproduces_as_empty():
    """The regression this was written for.

    Close reaches a 0.1 mm gap on a 30 mm probe, ProbeRealign then finds no
    held probe for several seconds while the filtered cloud shows the jaw
    volume clear. The old code passed the lift check by timeout; the pooled
    verdict must be EMPTY.
    """
    assert empty_close_gap(0.0001, 0.030, 0.024)

    ev = HeldProbeEvidence(window_sec=8.0, min_votes=3,
                           min_held_votes=2, empty_fraction=0.75)
    rng = np.random.default_rng(5)
    scene = np.column_stack([
        rng.uniform(0.2, 1.0, 2000),
        rng.uniform(-0.5, 0.5, 2000),
        rng.uniform(-0.3, 0.05, 2000),
    ])
    for t in range(5):
        in_volume = int(jaw_region_mask(
            scene, CONTACT, AXIS, RADIUS, ALONG_LO, ALONG_HI).sum())
        ev.add(EMPTY if in_volume < 25 else HELD, float(t))
    assert ev.verdict(4.0) == EMPTY


# --- self-filter blindness -------------------------------------------------
#
# Numbers below are measured off the running sim: the gripper camera sits at
# (0, 0.045, 0.035) in the planning link looking down the tool axis, so the jaw
# volume spans 0.155-0.355 m in front of it, and MoveIt's padded self-filter
# leaves nothing nearer than 0.349 m alive.
CAM_TO_LINK_Z = 0.035
CONTACT_IN_SENSOR = np.array([0.0, -0.044, CONTACT[2] - CAM_TO_LINK_Z])


def test_jaw_volume_near_range_is_the_near_end_face_for_a_coaxial_camera():
    near = jaw_volume_near_range(
        CONTACT_IN_SENSOR, AXIS, RADIUS, ALONG_LO, ALONG_HI)
    # Camera looks along the axis, so the radius contributes nothing.
    assert near == pytest.approx(CONTACT[2] - CAM_TO_LINK_Z + ALONG_LO, abs=1e-9)


def test_jaw_volume_near_range_subtracts_the_radius_when_viewed_side_on():
    side_axis = np.array([1.0, 0.0, 0.0])
    near = jaw_volume_near_range(
        CONTACT_IN_SENSOR, side_axis, RADIUS, ALONG_LO, ALONG_HI)
    assert near == pytest.approx(CONTACT_IN_SENSOR[2] - RADIUS, abs=1e-9)


def test_padded_self_filter_is_reported_blind_at_jaw_range():
    """The logged false EMPTY: filter keeps nothing nearer than 0.349 m."""
    near = jaw_volume_near_range(
        CONTACT_IN_SENSOR, AXIS, RADIUS, ALONG_LO, ALONG_HI)
    ranges = np.random.default_rng(3).uniform(0.349, 1.166, 5000)
    blinded, nearest = self_filter_blinds_jaw_volume(ranges, near, 0.010)
    assert blinded
    assert nearest == pytest.approx(0.349, abs=0.01)


def test_a_filter_that_keeps_the_near_field_is_trusted():
    """Reduce the padding until returns survive at jaw range and the cloud
    becomes usable evidence again -- the gate must re-open by itself."""
    near = jaw_volume_near_range(
        CONTACT_IN_SENSOR, AXIS, RADIUS, ALONG_LO, ALONG_HI)
    ranges = np.concatenate([
        np.random.default_rng(4).uniform(0.14, 0.30, 500),   # near field kept
        np.random.default_rng(5).uniform(0.35, 1.0, 4500),
    ])
    blinded, nearest = self_filter_blinds_jaw_volume(ranges, near, 0.010)
    assert not blinded
    assert nearest < near


def test_an_entirely_empty_cloud_is_blind_not_empty():
    blinded, nearest = self_filter_blinds_jaw_volume(np.empty(0), 0.155, 0.010)
    assert blinded
    assert nearest == float('inf')


def test_blind_frames_never_reach_an_empty_verdict():
    """The regression: three blind frames used to pool into EMPTY and abort a
    good grasp. They must pool into UNKNOWN, which keeps transport waiting."""
    near = jaw_volume_near_range(
        CONTACT_IN_SENSOR, AXIS, RADIUS, ALONG_LO, ALONG_HI)
    ranges = np.random.default_rng(6).uniform(0.349, 1.166, 5000)
    ev = HeldProbeEvidence(window_sec=8.0, min_votes=3,
                           min_held_votes=2, empty_fraction=0.75)
    for t in range(3):
        blinded, _ = self_filter_blinds_jaw_volume(ranges, near, 0.010)
        ev.add(UNKNOWN if blinded else EMPTY, float(t))
    assert ev.verdict(2.0) == UNKNOWN
