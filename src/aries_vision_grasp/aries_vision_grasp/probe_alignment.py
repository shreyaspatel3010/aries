"""Point-to-box pose fitting for the held probe.

The probe is a rectangular box (45 x 45 x 300 mm STL), so aligning its
collision mesh to masked depth points does not need a mesh ICP library: the
closest point on an origin-centred axis-aligned box is analytic, which turns
the alignment into a small trimmed ICP loop over numpy arrays.

Frames: ``points`` are expressed in whatever frame the caller works in
(vision_grasp_node uses the gripper link frame, where the held probe is
stationary). ``rotation`` columns are the box axes in that frame and
``centre`` is the box geometric centre; ``half_extents`` is the per-axis half
size of the box in its own frame.
"""

import math
from typing import NamedTuple, Optional, Tuple

import numpy as np


class BoxFitResult(NamedTuple):
    rotation: np.ndarray   # 3x3; columns = box axes in the point frame
    centre: np.ndarray     # box geometric centre in the point frame
    rms_m: float           # RMS point-to-surface distance over the kept inliers
    inlier_count: int      # points kept by the trimming step
    iterations: int        # ICP iterations actually run


def closest_points_on_box(points_box: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Closest surface points of an origin-centred axis-aligned box (vectorised).

    Points outside the box clamp onto it; points inside project onto the
    nearest face so interior sensor noise still produces a valid
    correspondence instead of a zero residual.
    """
    h = np.asarray(half_extents, dtype=np.float64).reshape(3,)
    q = np.asarray(points_box, dtype=np.float64).reshape(-1, 3)
    s = np.clip(q, -h, h)
    inside = np.all(np.abs(q) < h, axis=1)
    if np.any(inside):
        qi = q[inside]
        gap = h - np.abs(qi)
        axis = np.argmin(gap, axis=1)
        si = qi.copy()
        rows = np.arange(len(qi))
        signs = np.sign(si[rows, axis])
        signs[signs == 0.0] = 1.0
        si[rows, axis] = signs * h[axis]
        s[inside] = si
    return s


def box_surface_distances(
    points: np.ndarray,
    half_extents: np.ndarray,
    rotation: np.ndarray,
    centre: np.ndarray,
) -> np.ndarray:
    """Distance from each point to the surface of the posed box."""
    R = np.asarray(rotation, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64).reshape(3,)
    q = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - c) @ R
    s = closest_points_on_box(q, half_extents)
    return np.linalg.norm(q - s, axis=1)


def axis_half_widths(
    points: np.ndarray,
    rotation: np.ndarray,
    centre: np.ndarray,
) -> Tuple[float, float, int, int]:
    """Mean radial width of the cloud on each side of the box centre.

    Returns ``(width_neg, width_pos, n_neg, n_pos)`` where *neg*/*pos* refer
    to the sign of the box-frame long-axis (Z) coordinate. The probe STL is
    end-asymmetric (fat body on one end, tapered tip on the other), so
    comparing these widths against the STL's own profile disambiguates the
    180° end-for-end flip that a symmetric box fit cannot observe.
    """
    R = np.asarray(rotation, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64).reshape(3,)
    q = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - c) @ R
    radial = np.hypot(q[:, 0], q[:, 1])
    neg = q[:, 2] < 0.0
    n_neg = int(neg.sum())
    n_pos = int(len(q) - n_neg)
    width_neg = float(radial[neg].mean()) if n_neg else 0.0
    width_pos = float(radial[~neg].mean()) if n_pos else 0.0
    return width_neg, width_pos, n_neg, n_pos


def axis_angle_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    """Angle between two undirected axes (probe ends are symmetric)."""
    a = np.asarray(axis_a, dtype=np.float64).reshape(3,)
    b = np.asarray(axis_b, dtype=np.float64).reshape(3,)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos_val = abs(float(np.dot(a, b)) / (na * nb))
    return math.degrees(math.acos(min(1.0, cos_val)))


def fit_box_to_points(
    points: np.ndarray,
    half_extents: np.ndarray,
    rotation_init: np.ndarray,
    centre_init: np.ndarray,
    iterations: int = 50,
    trim_fraction: float = 0.10,
    convergence_m: float = 1e-4,
    outlier_residual_m: Optional[float] = None,
) -> Optional[BoxFitResult]:
    """Trimmed point-to-box ICP starting from a pose prior.

    Each iteration transforms the points into the current box frame, takes the
    analytic closest surface point as the correspondence, drops the worst
    ``trim_fraction`` residuals, and solves the rigid update with Kabsch/SVD.
    The initial pose must be a reasonable prior (here: the currently attached
    mesh pose) — with only one side of the box visible the trim step plus the
    prior keep the unobservable directions anchored.

    ``outlier_residual_m`` additionally drops any point further than that
    absolute distance from the box surface once the pose has settled (from the
    third iteration on). Fractional trimming alone cannot reject a compact
    off-box blob — e.g. gripper surfaces in a mask-free cloud — without also
    sacrificing legitimate far-face points; the absolute gate can.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 12:
        return None
    h = np.asarray(half_extents, dtype=np.float64).reshape(3,)
    if np.any(h <= 0.0):
        return None

    R = np.array(rotation_init, dtype=np.float64)
    c = np.array(centre_init, dtype=np.float64).reshape(3,)
    keep = np.ones(len(pts), dtype=bool)
    last_rms = None
    used_iterations = 0

    for it in range(max(1, int(iterations))):
        used_iterations += 1
        q = (pts - c) @ R
        s = closest_points_on_box(q, h)
        resid = np.linalg.norm(q - s, axis=1)

        if trim_fraction > 0.0 and len(pts) >= 30:
            cutoff = float(np.percentile(resid, 100.0 * (1.0 - trim_fraction)))
        else:
            cutoff = float('inf')
        if outlier_residual_m is not None and it >= 2:
            cutoff = min(cutoff, float(outlier_residual_m))
        keep = resid <= max(cutoff, 1e-6)
        if int(keep.sum()) < 12:
            return None

        src = s[keep]
        dst = pts[keep]
        src_c = src.mean(axis=0)
        dst_c = dst.mean(axis=0)
        H = (src - src_c).T @ (dst - dst_c)
        U, _, Vt = np.linalg.svd(H)
        d = float(np.sign(np.linalg.det(Vt.T @ U.T))) or 1.0
        R_new = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        c_new = dst_c - R_new @ src_c

        rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
        shift = float(np.linalg.norm(c_new - c))
        R, c = R_new, c_new
        if last_rms is not None and shift < convergence_m and abs(last_rms - rms) < convergence_m:
            break
        last_rms = rms

    q = (pts - c) @ R
    s = closest_points_on_box(q, h)
    resid = np.linalg.norm(q - s, axis=1)
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
    return BoxFitResult(
        rotation=R,
        centre=c,
        rms_m=rms,
        inlier_count=int(keep.sum()),
        iterations=used_iterations,
    )
