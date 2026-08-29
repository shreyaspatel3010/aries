"""Calibrated gripper geometry: jaw gap and contact height against joint angle.

TWO GRIPPERS SHARE THIS MODULE, selected with set_gripper():

  'v2'      the primary four-bar, three swappable fingertips, geometry
            interpolated from the measured tables below. This is what the rest
            of this docstring is about and the default.

  'st3215'  the secondary rack and pinion. No tables and no fingertip choice:
            the pinion direct-drives two racks, so the jaws TRANSLATE and both
            relationships are exact closed forms rather than curves --

                gap(q)      = 2 * 0.01002676 * (0.07 - q)      [linear]
                contact z   = 0.2078 m                          [constant]

            The constant contact height is the real structural difference and
            not a simplification: a four-bar's fingertip swings through an arc,
            so its contact point climbs 86 mm between open and closed, which is
            why v2 needs a contact table at all. A rack does not move in Z.
            Anything that compensates for contact drift can simply skip it on
            this gripper.

            Its usable range is q in [-3.76, +0.07] and it opens to 76.8 mm
            against v2's 83 - WIDER, and over twice the joint travel. A q
            value is not portable between the two even though both call
            +0.07 closed.

The rest of this file, and everything below about fingertips, applies to 'v2'.

---

Calibrated four-bar gripper geometry, per swappable fingertip.

The tables map the commanded gear joint angle q (rad) to the physical jaw gap
and to the finger contact midpoint expressed in arm_gripper_base_link. They
were measured from the gripper_new.xacro joint chain plus the fingertip mesh.

The three fingertips are physically interchangeable but are NOT
interchangeable geometrically. After the 2026-07-20 re-model of the
maintenance and probe fingers all three share the same jaw-gap curve to within
0.5 mm — so a probe-width close angle now carries over between them — but they
still reach different distances, contact sitting at z ≈ 218 / 227 / 241 mm for
bucket / maintenance / probe. Using the bucket tables with another finger
mounted therefore places the attached probe collision mesh up to 23 mm off the
real probe, which reads as a mis-aligned mesh in RViz and shows up as
START_STATE_IN_COLLISION on every plan. Select the mounted finger with
set_finger() before any geometry call.

These tables describe the GRIPPER only and are independent of probe size —
never retune them when the probe model changes; pass the new width through
q_from_gap instead.

Key calibration facts encoded here:
  - q = +0.07 rad is almost fully closed (gap ≈ 0), NOT a wide gap.
  - with the bucket, the current 30 mm probe needs q ≈ -0.085 rad; the earlier
    45 mm probe needed q ≈ -0.20 rad.
  - across that whole range the bucket contact midpoint barely moves
    (z ≈ 218-219 mm at y = 25.9 mm), so the contact offset defaults hold for
    both probes. It is NOT (0, 0, 249 mm). The maintenance and probe fingers
    sit at z ≈ 227 mm and ≈ 241 mm instead.

Provenance: the bucket rows are the field-calibrated originals and are left
untouched. The maintenance and probe rows are derived from the same URDF joint
chain applied to their own meshes; that derivation reproduces the bucket gap
row to within 0.5 mm across all 15 points, and its contact-Z is corrected by
the constant offset that maps it onto the calibrated bucket row. They are
geometric predictions, not bench measurements, so re-check them against
hardware before trusting a tight grasp.
"""

import numpy as np

# Joint-angle grids. Shared by every finger so only the value rows differ.
_Q_GAP = np.array([-1.570, -1.365, -1.160, -0.955, -0.750,
                   -0.545, -0.340, -0.2879, -0.2271, -0.1976,
                   -0.1861, -0.1385, -0.0498, 0.0093, 0.070], dtype=np.float64)
_Q_CONTACT = np.array([-1.570, -1.000, -0.500, -0.200, -0.140,
                       -0.050, 0.000, 0.070], dtype=np.float64)

# finger name -> (jaw gap over _Q_GAP, contact-midpoint local Z over _Q_CONTACT)
_FINGER_TABLES = {
    'bucket': (
        np.array([0.1826, 0.1790, 0.1684, 0.1512, 0.1281,
                  0.1002, 0.0685, 0.0600, 0.0500, 0.0451,
                  0.0431, 0.0351, 0.0200, 0.0099, 0.0000], dtype=np.float64),
        np.array([0.1342, 0.1680, 0.2092, 0.2180, 0.2189,
                  0.2196, 0.2197, 0.2195], dtype=np.float64),
    ),
    # Re-derived 2026-07-20 after the maintenance finger was re-modelled (now
    # 15x23x125 mm — same profile as the re-modelled probe finger, half its
    # thickness). Like the probe it now tracks the bucket's gap curve; it is
    # reach that separates them, contact sitting 9 mm beyond the bucket.
    'maintenance': (
        np.array([0.1821, 0.1785, 0.1679, 0.1507, 0.1276,
                  0.0997, 0.0680, 0.0595, 0.0495, 0.0446,
                  0.0426, 0.0346, 0.0195, 0.0094, 0.0000], dtype=np.float64),
        np.array([0.1431, 0.1893, 0.2182, 0.2269, 0.2278,
                  0.2285, 0.2286, 0.2284], dtype=np.float64),
    ),
    # Re-derived 2026-07-20 after the probe finger was re-modelled (now
    # 30x23x125 mm, 30 mm longer and shallower than the first design). The
    # reshaped jaw tracks the bucket's gap curve to within 0.2 mm, so the
    # close angle for a given probe width is effectively the bucket's; what
    # still differs is reach — contact sits 23 mm further out.
    'probe': (
        np.array([0.1824, 0.1788, 0.1682, 0.1510, 0.1279,
                  0.1000, 0.0683, 0.0598, 0.0498, 0.0449,
                  0.0429, 0.0349, 0.0198, 0.0097, 0.0000], dtype=np.float64),
        np.array([0.1576, 0.2037, 0.2326, 0.2414, 0.2422,
                  0.2430, 0.2431, 0.2429], dtype=np.float64),
    ),
}

# Contact-midpoint lateral offset (m) in arm_gripper_base_link. The four-bar
# carries every finger on the same pivots, so this is the same for all three.
# Near zero because the fingers are mounted on the inboard face of the bars and
# the jaw line runs down the gripper base axis; the residual millimetre is the
# finger mesh's own asymmetry. This was 0.0259 while gripper_new.xacro stepped
# the bucket joints the wrong way and hung the fingers off the base's +Y face.
CONTACT_Y_OFFSET_M = 0.001

FINGER_TYPES = tuple(_FINGER_TABLES)
DEFAULT_FINGER = 'bucket'

Q_MIN = float(_Q_GAP[0])
Q_MAX = float(_Q_GAP[-1])

_active_finger = DEFAULT_FINGER


# --------------------------------------------------------------------------
# Which gripper is bolted on. See the module docstring.
# --------------------------------------------------------------------------
DEFAULT_GRIPPER = 'v2'
GRIPPERS = ('v2', 'st3215')

# Jaw travel per radian of pinion, and the joint angle at which the jaws touch.
# Both are properties of the mechanism, measured in
# scripts/build_gripper_st3215_meshes.py: 21 teeth on a 3.0000 mm rack pitch
# gives 21 * 3 / (2*pi) mm per radian, and the URDF's zero pose is calibrated
# so contact lands on the +0.07 rad this whole stack calls closed.
ST3215_PITCH_R = 0.01002676
ST3215_Q_CLOSED = 0.07
ST3215_Q_OPEN = -3.76
# The lips meet 207.8 mm above arm_gripper_base_link, and stay there: the jaws
# translate, so this does not vary with q.
ST3215_CONTACT_Z = 0.2078

_active_gripper = DEFAULT_GRIPPER


def set_gripper(name: str) -> str:
    """Select the fitted gripper. Returns the name actually applied.

    An unknown name falls back to 'v2' rather than raising, for the same reason
    set_finger() does: a typo in a launch argument should degrade to the
    previous behaviour, not take the grasp stack down mid-run. Log the return
    value so a silent fallback is still visible.
    """
    global _active_gripper
    key = str(name).strip().lower()
    _active_gripper = key if key in GRIPPERS else DEFAULT_GRIPPER
    return _active_gripper


def active_gripper() -> str:
    """Name of the gripper currently driving the geometry."""
    return _active_gripper


def q_limits() -> tuple:
    """(most open, most closed) joint angle the fitted gripper can reach."""
    if _active_gripper == 'st3215':
        return (ST3215_Q_OPEN, ST3215_Q_CLOSED)
    return (float(_Q_GAP[0]), float(_Q_GAP[-1]))


def set_finger(name: str) -> str:
    """Select the mounted fingertip. Returns the name actually applied.

    An unknown name falls back to the bucket rather than raising: a typo in a
    launch argument should degrade to the previous behaviour, not take the
    grasp stack down mid-run. The caller is expected to log the return value
    so a silent fallback is still visible.
    """
    global _active_finger
    key = str(name).strip().lower()
    _active_finger = key if key in _FINGER_TABLES else DEFAULT_FINGER
    return _active_finger


def active_finger() -> str:
    """Name of the fingertip currently driving the tables."""
    return _active_finger


def _tables():
    return _FINGER_TABLES[_active_finger]


def gap_from_q(q: float) -> float:
    """Actual finger inner gap (m) for a gear joint angle (rad)."""
    if _active_gripper == 'st3215':
        q_clip = float(np.clip(q, ST3215_Q_OPEN, ST3215_Q_CLOSED))
        return 2.0 * ST3215_PITCH_R * (ST3215_Q_CLOSED - q_clip)
    gap, _ = _tables()
    q_clip = float(np.clip(q, _Q_GAP[0], _Q_GAP[-1]))
    return float(np.interp(q_clip, _Q_GAP, gap))


def q_from_gap(gap_m: float) -> float:
    """Invert the gap table: desired jaw gap (m) -> gear joint angle (rad).

    The result is clipped to the table's physical range only; task-specific
    clamps (open/close command limits, floor-grasp q window) are applied by
    the caller.
    """
    if _active_gripper == 'st3215':
        q = ST3215_Q_CLOSED - float(gap_m) / (2.0 * ST3215_PITCH_R)
        return float(np.clip(q, ST3215_Q_OPEN, ST3215_Q_CLOSED))
    gap, _ = _tables()
    # The narrow fingers bottom out at gap 0 for several trailing angles.
    # np.interp needs a strictly increasing xp, so collapse that flat tail to
    # its first (most open) angle -- the smallest q that actually shuts the
    # jaws -- instead of interpolating across the duplicates.
    asc_gap = gap[::-1]
    asc_q = _Q_GAP[::-1]
    keep = np.concatenate(([True], np.diff(asc_gap) > 1e-9))
    return float(np.interp(float(gap_m), asc_gap[keep], asc_q[keep]))


def contact_offset_z(q: float) -> float:
    """Local Z (m) of the finger contact midpoint for a gear joint angle."""
    if _active_gripper == 'st3215':
        # Constant by construction: the jaws translate along X only.
        return ST3215_CONTACT_Z
    _, z_contact = _tables()
    q_clip = float(np.clip(q, _Q_CONTACT[0], _Q_CONTACT[-1]))
    return float(np.interp(q_clip, _Q_CONTACT, z_contact))


def contact_offset(q: float, y_offset_m: float) -> np.ndarray:
    """arm_gripper_base_link -> finger-contact-midpoint offset for angle q."""
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
    gap_tolerance_m: float = 0.018,
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
