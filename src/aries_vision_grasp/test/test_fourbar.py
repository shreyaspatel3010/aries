"""Unit tests for the calibrated four-bar gripper model.

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
    # Near a 45 mm grasp the true contact midpoint is ~(0, 25.9, 218) mm.
    off = fourbar.contact_offset(-0.20, y_offset_m=0.0259)
    assert off[0] == pytest.approx(0.0)
    assert off[1] == pytest.approx(0.0259)
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


def test_plausible_probe_contact_rejects_wrong_direction_or_gap():
    common = dict(
        minimum_probe_width_m=0.045,
        maximum_probe_width_m=0.060,
    )
    assert not fourbar.plausible_probe_contact(-1.57, -1.50, -0.15, **common)
    assert not fourbar.plausible_probe_contact(-1.57, -0.02, 0.07, **common)
    assert not fourbar.plausible_probe_contact(-0.15, -0.20, -1.57, **common)
    assert not fourbar.plausible_probe_contact(-1.57, -0.155, -0.15, **common)
