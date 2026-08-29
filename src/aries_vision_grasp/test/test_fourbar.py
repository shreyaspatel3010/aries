"""Unit tests for the calibrated FOUR-BAR gripper model (gripper_v2).

That gripper is retired - the ST3215 rack-and-pinion replaced it and is now the
default in fourbar.set_gripper() - but its tables are the only record of its
measured geometry and the fingertip work behind it, so they are kept and still
tested. Every case here selects 'v2' explicitly through the fixture below; the
module default is no longer it.

These tables drive the physical grasp geometry; a regression here means the
gripper closes to the wrong gap or the arm targets the wrong contact point.
"""

import numpy as np
import pytest

from aries_vision_grasp import fourbar


# Calibration anchor points quoted in the node/docstrings.
KNOWN_Q_GAP = [
    (0.070, 0.0000),    # fully closed
    (-0.1976, 0.0451),  # ~45 mm probe
    (-1.570, 0.1826),   # fully open
]


@pytest.mark.parametrize('q, gap', KNOWN_Q_GAP)
def test_gap_from_q_matches_calibration(q, gap):
    assert fourbar.gap_from_q(q) == pytest.approx(gap, abs=1e-4)


def test_gap_from_q_clips_outside_table():
    assert fourbar.gap_from_q(-5.0) == pytest.approx(0.1826, abs=1e-4)
    assert fourbar.gap_from_q(1.0) == pytest.approx(0.0, abs=1e-4)


def test_gap_is_monotonically_decreasing_in_q():
    qs = np.linspace(fourbar.Q_MIN, fourbar.Q_MAX, 200)
    gaps = [fourbar.gap_from_q(q) for q in qs]
    assert all(g1 >= g2 - 1e-12 for g1, g2 in zip(gaps, gaps[1:]))


def test_q_from_gap_roundtrip():
    for q in np.linspace(fourbar.Q_MIN, fourbar.Q_MAX, 50):
        gap = fourbar.gap_from_q(float(q))
        assert fourbar.q_from_gap(gap) == pytest.approx(float(q), abs=5e-3)


def test_q_for_45mm_probe_is_near_minus_0_2_rad():
    # The core calibration fact: a 45 mm gap needs q ≈ -0.20 rad,
    # NOT the old linear model's q ≈ +0.07.
    assert fourbar.q_from_gap(0.045) == pytest.approx(-0.1976, abs=0.01)


def test_contact_offset_near_closed():
    # Near a 45 mm grasp the true contact midpoint is ~(0, 1, 218) mm: the jaw
    # line runs down the gripper base axis, so Y is ~0 rather than the 25.9 mm
    # that compensated for the old bucket-joint step sign in gripper_new.xacro.
    off = fourbar.contact_offset(-0.20, y_offset_m=fourbar.CONTACT_Y_OFFSET_M)
    assert off[0] == pytest.approx(0.0)
    assert off[1] == pytest.approx(0.001)
    assert off[2] == pytest.approx(0.2180, abs=1e-3)


def test_contact_offset_z_monotonic_and_bounded():
    zs = [fourbar.contact_offset_z(q) for q in np.linspace(-1.57, 0.07, 100)]
    assert min(zs) >= 0.134 - 1e-6
    assert max(zs) <= 0.2197 + 1e-6


def test_plausible_probe_contact_accepts_rigid_probe_stop():
    assert fourbar.plausible_probe_contact(
        start_q=-1.57,
        actual_q=-0.20,
        target_q=-0.15,
        minimum_probe_width_m=0.045,
        maximum_probe_width_m=0.060,
    )


def test_plausible_probe_contact_accepts_observed_gazebo_stop():
    """Regression: the rigid probe repeatedly settles here in Gazebo."""
    assert fourbar.plausible_probe_contact(
        start_q=-1.565,
        actual_q=-0.10510,
        target_q=0.070,
        minimum_probe_width_m=0.045,
        maximum_probe_width_m=0.060,
    )


def test_plausible_probe_contact_accepts_tight_hardware_stop():
    """Regression: a run stalled at q=-0.08945 (26.75 mm gap) and missed the
    old 18 mm window by 0.25 mm, hard-locking a physically held probe. The
    deployed 24 mm tolerance must accept it and still reject a near-closed
    miss (q=-0.02 reads 14.9 mm)."""
    common = dict(
        minimum_probe_width_m=0.045,
        maximum_probe_width_m=0.060,
        target_tolerance_rad=0.012,
        minimum_closing_travel_rad=0.20,
        gap_tolerance_m=0.024,
    )
    assert fourbar.plausible_probe_contact(-1.57, -0.08945499534309975, 0.07, **common)
    assert not fourbar.plausible_probe_contact(-1.57, -0.02, 0.07, **common)


def test_plausible_probe_contact_rejects_wrong_direction_or_gap():
    common = dict(
        minimum_probe_width_m=0.045,
        maximum_probe_width_m=0.060,
    )
    assert not fourbar.plausible_probe_contact(-1.57, -1.50, -0.15, **common)
    assert not fourbar.plausible_probe_contact(-1.57, -0.02, 0.07, **common)
    assert not fourbar.plausible_probe_contact(-0.15, -0.20, -1.57, **common)
    assert not fourbar.plausible_probe_contact(-1.57, -0.155, -0.15, **common)


def test_close_angle_reaches_the_thirty_millimetre_probe():
    """The width floor sizes q_close. A floor above the real probe commands a
    jaw gap wider than the probe, so the fingers never make contact."""
    probe_width = 0.030
    clearance = -0.004

    # Configured floor matching the probe: jaws close past it and grip.
    effective = max(probe_width, 0.030)
    gap = fourbar.gap_from_q(fourbar.q_from_gap(effective + clearance))
    assert gap < probe_width

    # A stale 45 mm floor leaves the jaws wide open around a 30 mm probe.
    effective_stale = max(probe_width, 0.045)
    gap_stale = fourbar.gap_from_q(fourbar.q_from_gap(effective_stale + clearance))
    assert gap_stale > probe_width + 0.005


def test_contact_offset_is_stable_across_probe_widths():
    """Contact offsets are gripper geometry indexed by q, so they carry over
    when the probe size changes and must not be retuned with it."""
    q_30mm = fourbar.q_from_gap(0.026)
    q_45mm = fourbar.q_from_gap(0.041)
    assert abs(fourbar.contact_offset_z(q_30mm) - fourbar.contact_offset_z(q_45mm)) < 0.001


@pytest.fixture(autouse=True)
def _reset_finger():
    """Select the four-bar and its default fingertip for every case here.

    Both selections are module state, so leaving either set would leak into
    whichever test ran next and silently change the geometry under it. The
    gripper has to be selected explicitly now: the default is 'st3215', whose
    geometry is a closed form with no fingertip tables at all, so without this
    every table assertion below compares against the wrong mechanism.
    """
    fourbar.set_gripper('v2')
    fourbar.set_finger(fourbar.DEFAULT_FINGER)
    yield
    fourbar.set_finger(fourbar.DEFAULT_FINGER)
    fourbar.set_gripper(fourbar.DEFAULT_GRIPPER)


def test_default_finger_is_the_calibrated_bucket():
    """The bucket rows are the field-calibrated originals; the default must
    keep reproducing them exactly or every existing grasp shifts."""
    assert fourbar.active_finger() == 'bucket'
    assert fourbar.gap_from_q(-1.570) == pytest.approx(0.1826, abs=1e-4)
    assert fourbar.contact_offset_z(fourbar.Q_MIN) == pytest.approx(0.1342, abs=1e-4)
    assert fourbar.contact_offset_z(-0.200) == pytest.approx(0.2180, abs=1e-4)


def test_each_finger_has_its_own_geometry():
    """The three fingers are physically swappable but not geometrically
    interchangeable — the whole point of selecting one."""
    seen_gap, seen_z = [], []
    for name in fourbar.FINGER_TYPES:
        fourbar.set_finger(name)
        seen_gap.append(fourbar.gap_from_q(-0.200))
        seen_z.append(fourbar.contact_offset_z(-0.200))
    assert len(set(round(g, 4) for g in seen_gap)) == len(fourbar.FINGER_TYPES)
    assert len(set(round(z, 4) for z in seen_z)) == len(fourbar.FINGER_TYPES)


def test_longer_fingers_contact_further_out_than_the_bucket():
    """Reach is what still separates the fingers after the re-model, and it is
    what offsets the attached probe mesh when the wrong one is selected:
    contact sits ~9 mm beyond the bucket for maintenance, ~23 mm for probe."""
    reach = {}
    for name in fourbar.FINGER_TYPES:
        fourbar.set_finger(name)
        reach[name] = fourbar.contact_offset_z(-0.200)
    assert reach['bucket'] < reach['maintenance'] < reach['probe']
    assert reach['maintenance'] - reach['bucket'] > 0.005
    assert reach['probe'] - reach['bucket'] > 0.020


def test_redesigned_fingers_all_share_the_bucket_gap_curve():
    """Pins the 2026-07-20 re-model: the reshaped maintenance and probe jaws
    grip like the bucket, so a probe-width close angle carries over between
    all three. If a future re-export breaks that, the commanded close angle
    changes silently and the grasp crushes or misses."""
    sample_q = (-1.570, -0.955, -0.340, -0.0498)
    fourbar.set_finger('bucket')
    bucket = [fourbar.gap_from_q(q) for q in sample_q]
    for name in ('maintenance', 'probe'):
        fourbar.set_finger(name)
        assert [fourbar.gap_from_q(q) for q in sample_q] == pytest.approx(bucket, abs=1e-3)


def test_close_angle_for_the_probe_carries_across_fingers():
    """The practical consequence of the shared gap curve: one close angle
    serves every finger, so only the contact height still needs selecting."""
    angles = []
    for name in fourbar.FINGER_TYPES:
        fourbar.set_finger(name)
        angles.append(fourbar.q_from_gap(0.030))
    assert max(angles) - min(angles) < 0.01


def test_q_from_gap_handles_the_flat_closed_tail():
    """The narrow fingers bottom out at gap 0 for several trailing angles;
    np.interp needs a strictly increasing table or it returns garbage."""
    for name in fourbar.FINGER_TYPES:
        fourbar.set_finger(name)
        q = fourbar.q_from_gap(0.0)
        assert fourbar.Q_MIN <= q <= fourbar.Q_MAX
        assert np.isfinite(q)
        # A wider request must never demand a more-closed angle.
        assert fourbar.q_from_gap(0.030) <= fourbar.q_from_gap(0.010) + 1e-9


def test_unknown_finger_falls_back_to_bucket():
    """A launch-argument typo should degrade to the previous behaviour rather
    than take the grasp stack down mid-run."""
    assert fourbar.set_finger('buckett') == 'bucket'
    assert fourbar.set_finger('') == 'bucket'
    assert fourbar.active_finger() == 'bucket'


def test_finger_selection_is_case_and_whitespace_tolerant():
    assert fourbar.set_finger('  PROBE ') == 'probe'
    assert fourbar.active_finger() == 'probe'
