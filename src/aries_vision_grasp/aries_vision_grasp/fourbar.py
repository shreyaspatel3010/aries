"""Calibrated four-bar gripper geometry.

The tables below were measured from gripper_new.xacro joint chain plus the
gripper_bucket.stl mesh. They map the commanded gear joint angle q (rad) to
the physical jaw gap and to the bucket contact midpoint expressed in
arm_gripper_base_link.

Key calibration facts encoded here:
  - q = +0.07 rad is almost fully closed (gap ≈ 0), NOT a 45 mm gap.
  - a 45 mm probe needs q ≈ -0.20 rad.
  - near a 45 mm grasp the true contact midpoint is approximately
    (x=0, y=25.9 mm, z=218 mm), not (0, 0, 249 mm).
"""

import numpy as np

# Joint angle → jaw gap (metres). Ascending in q.
_Q_GAP = np.array([-1.570, -1.365, -1.160, -0.955, -0.750,
                   -0.545, -0.340, -0.2879, -0.2271, -0.1976,
                   -0.1861, -0.1385, -0.0498, 0.0093, 0.070], dtype=np.float64)
_GAP = np.array([0.1826, 0.1790, 0.1684, 0.1512, 0.1281,
                 0.1002, 0.0685, 0.0600, 0.0500, 0.0451,
                 0.0431, 0.0351, 0.0200, 0.0099, 0.0000], dtype=np.float64)

# Joint angle → contact-midpoint local Z (metres). Ascending in q.
_Q_CONTACT = np.array([-1.570, -1.000, -0.500, -0.200, -0.140,
                       -0.050, 0.000, 0.070], dtype=np.float64)
_Z_CONTACT = np.array([0.1342, 0.1680, 0.2092, 0.2180, 0.2189,
                       0.2196, 0.2197, 0.2195], dtype=np.float64)

Q_MIN = float(_Q_GAP[0])
Q_MAX = float(_Q_GAP[-1])


def gap_from_q(q: float) -> float:
    """Actual bucket inner gap (m) for a gear joint angle (rad)."""
    q_clip = float(np.clip(q, _Q_GAP[0], _Q_GAP[-1]))
    return float(np.interp(q_clip, _Q_GAP, _GAP))


def q_from_gap(gap_m: float) -> float:
    """Invert the gap table: desired jaw gap (m) -> gear joint angle (rad).

    The result is clipped to the table's physical range only; task-specific
    clamps (open/close command limits, floor-grasp q window) are applied by
    the caller.
    """
    return float(np.interp(float(gap_m), _GAP[::-1], _Q_GAP[::-1]))


def contact_offset_z(q: float) -> float:
    """Local Z (m) of the bucket contact midpoint for a gear joint angle."""
    q_clip = float(np.clip(q, _Q_CONTACT[0], _Q_CONTACT[-1]))
    return float(np.interp(q_clip, _Q_CONTACT, _Z_CONTACT))


def contact_offset(q: float, y_offset_m: float) -> np.ndarray:
    """arm_gripper_base_link -> bucket-contact-midpoint offset for angle q."""
    return np.array([0.0, float(y_offset_m), contact_offset_z(q)], dtype=np.float64)


def plausible_probe_contact(
    start_q: float,
    actual_q: float,
    target_q: float,
    minimum_probe_width_m: float,
    maximum_probe_width_m: float,
    *,
    target_tolerance_rad: float = 0.012,
    minimum_closing_travel_rad: float = 0.20,
    gap_tolerance_m: float = 0.015,
) -> bool:
    """Return whether stopped-short feedback is consistent with a probe.

    A position-controlled gripper normally reports a goal-tolerance failure
    when a rigid object prevents it from reaching a deliberately over-closed
    target. Such a failure is useful contact evidence only when the measured
    joint moved substantially in the closing direction, stopped short of the
    target, and the calibrated jaw gap is within the physical probe range.
    """
    start = float(start_q)
    actual = float(actual_q)
    target = float(target_q)
    if not np.all(np.isfinite([start, actual, target])):
        return False
    if target <= start:
        return False
    if (actual - start) < max(0.0, float(minimum_closing_travel_rad)):
        return False
    if actual >= target - max(0.0, float(target_tolerance_rad)):
        return False

    actual_gap = gap_from_q(actual)
    gap_tol = max(0.0, float(gap_tolerance_m))
    min_gap = max(0.0, float(minimum_probe_width_m) - gap_tol)
    max_gap = max(min_gap, float(maximum_probe_width_m) + gap_tol)
    return min_gap <= actual_gap <= max_gap
