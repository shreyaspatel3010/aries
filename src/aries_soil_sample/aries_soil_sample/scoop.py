"""Scoop trajectory geometry and sample-capture verdict for the bucket finger.

The bucket fingertips are two 50 x 44 x 100 mm shells carried on the same
four-bar as the other jaws (measured from ``gripper_bucket.stl``). Closing them
under the soil traps a sample, so a scoop is a straight-line penetration
followed by a close, not a grasp:

    approach -> entry (at the surface) -> penetrate (below it) -> CLOSE -> extract

Everything runs along the surface normal, and extraction retraces the
penetration exactly in reverse. That mirrors what the probe grasp had to learn
the hard way: a world-vertical lift out of an angled hole levers the tool
against the material instead of backing out of the channel it cut.

Frames: ``*_point`` arguments and results are in the planning frame
(``base_link``); ``tool_axis`` is tool +Z, pointing from the gripper into the
soil. Positions here describe the four-bar CONTACT point, not the gripper link
origin -- use ``link_position_for_contact`` for the pose to send MoveIt.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

CAPTURED = 'captured'
EMPTY = 'empty'
UNKNOWN = 'unknown'

# Bucket fingertip bounding box from gripper_bucket.stl, in metres.
BUCKET_WIDTH_M = 0.050
BUCKET_DEPTH_M = 0.0436
BUCKET_LENGTH_M = 0.0998


def normalize(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3,)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError('cannot normalize a zero-length vector')
    return a / n


@dataclass(frozen=True)
class ScoopParams:
    """Tuning for one scoop, all metres/degrees."""

    standoff_m: float = 0.060
    depth_m: float = 0.030
    attack_deg: float = 0.0           # tilt of the entry axis from the surface normal
    max_depth_m: float = 0.060
    bucket_length_m: float = BUCKET_LENGTH_M
    depth_margin_m: float = 0.010     # keep the fingertip shells from fully burying


@dataclass(frozen=True)
class ScoopWaypoint:
    """One contact-point target along the scoop, with the tool axis to hold."""

    position: np.ndarray
    tool_axis: np.ndarray
    label: str

    @property
    def summary(self) -> str:
        return (f'{self.label}: ({self.position[0]:.3f},{self.position[1]:.3f},'
                f'{self.position[2]:.3f})')


def clamp_penetration_depth(params: ScoopParams) -> Tuple[float, Optional[str]]:
    """Usable penetration depth, and why it was reduced if it was.

    Two independent ceilings: the operator's ``max_depth_m``, and the bucket's
    own length less a margin -- past that the shells stop being a scoop and the
    four-bar linkage itself starts entering the soil.
    """
    requested = float(params.depth_m)
    geometric = max(0.0, float(params.bucket_length_m) - float(params.depth_margin_m))
    limit = min(float(params.max_depth_m), geometric)
    if requested <= limit:
        return requested, None
    which = ('max_depth_m' if float(params.max_depth_m) <= geometric
             else f'bucket length {params.bucket_length_m*1000:.0f}mm '
                  f'less {params.depth_margin_m*1000:.0f}mm margin')
    return limit, (f'requested {requested*1000:.0f}mm exceeds {limit*1000:.0f}mm '
                   f'({which}); clamped')


def entry_axis(surface_normal: Sequence[float], attack_deg: float,
               azimuth_ref: Optional[Sequence[float]] = None) -> np.ndarray:
    """Tool +Z for the entry, i.e. into the soil.

    Straight down the inward normal when ``attack_deg`` is 0. A non-zero attack
    tilts the axis toward ``azimuth_ref`` (projected onto the surface plane), for
    slicing into the material at an angle rather than punching straight in.
    """
    n = normalize(surface_normal)
    inward = -n
    tilt = math.radians(float(attack_deg))
    if abs(tilt) < 1e-9:
        return inward
    ref = (np.asarray(azimuth_ref, dtype=np.float64).reshape(3,)
           if azimuth_ref is not None else np.array([1.0, 0.0, 0.0]))
    tangent = ref - n * float(np.dot(ref, n))
    if float(np.linalg.norm(tangent)) < 1e-6:
        alt = np.array([0.0, 1.0, 0.0])
        tangent = alt - n * float(np.dot(alt, n))
        if float(np.linalg.norm(tangent)) < 1e-6:
            return inward
    tangent = normalize(tangent)
    return normalize(inward * math.cos(tilt) + tangent * math.sin(tilt))


def plan_scoop(
    surface_point: Sequence[float],
    surface_normal: Sequence[float],
    params: ScoopParams,
    azimuth_ref: Optional[Sequence[float]] = None,
) -> Tuple[List[ScoopWaypoint], float, Optional[str]]:
    """Contact-point waypoints for one scoop.

    Returns ``(waypoints, depth_used_m, clamp_note)``. The waypoints are
    ``approach`` / ``entry`` / ``penetrate`` / ``extract``; the gripper closes
    between ``penetrate`` and ``extract``, which is the caller's job because it
    is a gripper command rather than an arm motion.

    ``extract`` returns to the approach standoff along the SAME axis, so the
    bucket leaves through the channel it cut.
    """
    p = np.asarray(surface_point, dtype=np.float64).reshape(3,)
    n = normalize(surface_normal)
    axis = entry_axis(n, params.attack_deg, azimuth_ref)
    depth, note = clamp_penetration_depth(params)
    standoff = max(0.0, float(params.standoff_m))

    approach = p - axis * standoff
    entry = p.copy()
    penetrate = p + axis * depth
    return (
        [
            ScoopWaypoint(approach, axis, 'approach'),
            ScoopWaypoint(entry, axis, 'entry'),
            ScoopWaypoint(penetrate, axis, 'penetrate'),
            ScoopWaypoint(approach, axis, 'extract'),
        ],
        depth,
        note,
    )


def tool_frame(tool_axis: Sequence[float],
               prefer_pinch: Optional[Sequence[float]] = None) -> np.ndarray:
    """Right-handed tool frame whose +Z is ``tool_axis``.

    Rotation about the axis is free -- the bucket is symmetric about its own
    closing plane for the purpose of entering soil -- so spend it on the wrist:
    ``prefer_pinch`` (typically the CURRENT jaw-line direction) is projected onto
    the plane normal to the axis and used as tool +X. That makes the scoop cost
    the wrist as little travel as possible instead of pinning the jaw line to an
    arbitrary world direction.
    """
    z = normalize(tool_axis)
    x = None
    if prefer_pinch is not None:
        cand = np.asarray(prefer_pinch, dtype=np.float64).reshape(3,)
        proj = cand - z * float(np.dot(cand, z))
        if float(np.linalg.norm(proj)) > 1e-3:
            x = normalize(proj)
    if x is None:
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, z))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        x = normalize(np.cross(seed, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def roll_frames(
    tool_axis: Sequence[float],
    prefer_pinch: Optional[Sequence[float]] = None,
    count: int = 12,
) -> List[np.ndarray]:
    """Candidate tool frames about the entry axis, closest to the preference first.

    Rotation about the shaft is a free DOF for a bucket entering soil, and it is
    not a free choice for the ARM: near the edge of the envelope some rolls have
    an IK solution and some do not. Measured at a real scoop site 190 mm below
    the link, only about three quarters of rolls solved -- so committing to the
    single roll nearest the current wrist makes the pre-screen a coin flip, which
    is exactly how it behaved (rejecting the approach on one attempt and the
    entry on the next, at sites that were otherwise fine).

    Returned in order of increasing |roll| from ``prefer_pinch`` so the caller
    still gets the cheapest wrist motion when it is available, and only walks
    outward when it is not.
    """
    z = normalize(tool_axis)
    base = tool_frame(z, prefer_pinch)
    x0, y0 = base[:, 0], base[:, 1]
    n = max(1, int(count))
    step = 2.0 * math.pi / n
    # 0, +1, -1, +2, -2, ... so the preference is tried first.
    order = [0]
    for k in range(1, n // 2 + 1):
        order.append(k)
        if -k not in order and len(order) < n:
            order.append(-k)
    frames: List[np.ndarray] = []
    for k in order[:n]:
        roll = k * step
        x = math.cos(roll) * x0 + math.sin(roll) * y0
        x = normalize(x)
        frames.append(np.column_stack([x, np.cross(z, x), z]))
    return frames


def link_position_for_contact(
    contact_point: Sequence[float],
    tool_rotation: np.ndarray,
    contact_offset_in_tool: Sequence[float],
) -> np.ndarray:
    """Gripper link origin that places the four-bar contact at a target point.

    ``contact_offset_in_tool`` is the link-to-contact offset for the mounted
    finger at the CURRENT jaw angle -- take it from ``fourbar.contact_offset``
    with the angle the arm will actually be holding, because the bucket's
    contact point swings a long way with the linkage (z 134 mm wide open to
    220 mm closed).
    """
    c = np.asarray(contact_point, dtype=np.float64).reshape(3,)
    R = np.asarray(tool_rotation, dtype=np.float64).reshape(3, 3)
    off = np.asarray(contact_offset_in_tool, dtype=np.float64).reshape(3,)
    return c - R @ off


def nominal_bucket_capacity_m3(fill_factor: float = 0.35) -> float:
    """Rough closed-bucket sample volume from the fingertip bounding box.

    Two shells of 50 x 44 x 100 mm close into a cavity far smaller than their
    bounding boxes, so this is deliberately a fraction of it. It exists to give
    the volume thresholds a defensible starting point, NOT as a measurement --
    weigh or water-fill a real scoop before trusting the number.
    """
    envelope = BUCKET_WIDTH_M * BUCKET_DEPTH_M * BUCKET_LENGTH_M
    return float(max(0.0, min(1.0, fill_factor)) * envelope)


def capture_verdict(
    divot_volume_m3: Optional[float],
    max_drop_m: Optional[float],
    observed_cells: int,
    *,
    min_volume_m3: float,
    min_cells: int = 12,
    disturbed_drop_m: float = 0.004,
) -> Tuple[str, str]:
    """Did the scoop actually collect soil? Returns ``(verdict, reason)``.

    The evidence is the divot: material that left the ground is material the
    bucket took. Deliberately asymmetric, for the same reason the probe grasp's
    verdict is -- with no gripper feedback and no view inside the jaws, the only
    honest EMPTY is "the ground is measurably untouched". Anything in between
    (ground disturbed, not enough moved) is UNKNOWN, which the caller should
    treat as "re-survey or retry", never as success.
    """
    if divot_volume_m3 is None or max_drop_m is None:
        return UNKNOWN, 'no post-scoop survey available'
    if int(observed_cells) < int(min_cells):
        return UNKNOWN, (f'only {int(observed_cells)} cells seen in both surveys '
                         f'(need {int(min_cells)}); the site was not re-observed')
    vol = float(divot_volume_m3)
    drop = float(max_drop_m)
    if vol >= float(min_volume_m3):
        return CAPTURED, (f'divot {vol*1e6:.0f} cm^3 >= {float(min_volume_m3)*1e6:.0f} cm^3, '
                          f'max drop {drop*1000:.1f} mm')
    if drop < float(disturbed_drop_m):
        return EMPTY, (f'ground unchanged: max drop {drop*1000:.1f} mm < '
                       f'{float(disturbed_drop_m)*1000:.1f} mm, divot {vol*1e6:.0f} cm^3')
    return UNKNOWN, (f'ground disturbed (max drop {drop*1000:.1f} mm) but only '
                     f'{vol*1e6:.0f} cm^3 removed, under the '
                     f'{float(min_volume_m3)*1e6:.0f} cm^3 threshold')
