"""Pure helpers for choosing a low-motion wrist IK branch."""

import math
from typing import Mapping, Sequence, Tuple

import numpy as np


def azimuth_variants(
    base_rotation: np.ndarray,
    count: int,
) -> Tuple[Tuple[float, np.ndarray], ...]:
    """Rotate a tool frame about its local +Z without changing its approach.

    A round probe is invariant to this rotation.  Returning angles in
    shortest-first order makes useful candidates available early while the
    caller still evaluates every branch against actual robot IK.
    """
    R = np.asarray(base_rotation, dtype=np.float64)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError('base_rotation must be a finite 3x3 matrix')
    n = max(1, int(count))
    raw_angles = [2.0 * math.pi * i / n for i in range(n)]
    angles = sorted(
        (math.atan2(math.sin(a), math.cos(a)) for a in raw_angles),
        key=lambda a: (abs(a), a),
    )
    result = []
    for angle in angles:
        c, s = math.cos(angle), math.sin(angle)
        local_z_rotation = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        result.append((angle, R @ local_z_rotation))
    return tuple(result)


def score_joint_path(
    current: Mapping[str, float],
    pregrasp: Mapping[str, float],
    grasp: Mapping[str, float],
    joint_names: Sequence[str],
    wrist_joint_name: str = 'joint6',
    wrist_weight: float = 6.0,
    wrist_limits: Tuple[float, float] = (-3.12414, 3.12414),
    wrist_limit_margin: float = 0.30,
) -> Tuple[float, float, float]:
    """Score current→pregrasp→grasp joint motion.

    Returns ``(score, wrist_travel, minimum_wrist_limit_clearance)``. Revolute
    deltas are deliberately *not* wrapped across ±pi: joint6 is bounded just
    inside ±pi, so crossing from +3.0 to -3.0 is a six-radian physical motion,
    not a short continuous-joint wrap.
    """
    names = tuple(str(name) for name in joint_names)
    if not names or wrist_joint_name not in names:
        raise ValueError('joint_names must contain the wrist joint')
    if any(
        name not in current or name not in pregrasp or name not in grasp
        for name in names
    ):
        raise ValueError('current, pregrasp, and grasp must contain every joint')
    weight = max(1.0, float(wrist_weight))
    lower, upper = (float(wrist_limits[0]), float(wrist_limits[1]))
    if not lower < upper:
        raise ValueError('wrist limits must be ordered')
    margin = max(0.0, float(wrist_limit_margin))

    score = 0.0
    wrist_travel = 0.0
    for name in names:
        d_pre = abs(float(pregrasp[name]) - float(current[name]))
        d_grasp = abs(float(grasp[name]) - float(pregrasp[name]))
        joint_weight = weight if name == wrist_joint_name else 1.0
        score += joint_weight * (d_pre + d_grasp)
        if name == wrist_joint_name:
            wrist_travel = d_pre + d_grasp

    wrist_positions = (
        float(pregrasp[wrist_joint_name]),
        float(grasp[wrist_joint_name]),
    )
    clearance = min(
        min(q - lower, upper - q) for q in wrist_positions
    )
    if clearance < 0.0:
        score += 1e6
    elif clearance < margin:
        # Strongly prefer an equivalent IK family that leaves room for the
        # final Cartesian insertion instead of arriving against the stop.
        score += 100.0 * (margin - clearance)
    return float(score), float(wrist_travel), float(clearance)
