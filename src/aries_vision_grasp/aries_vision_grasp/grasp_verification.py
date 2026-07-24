"""Positive evidence that the probe is actually between the jaws.

An empty close is silent. The gripper controller happily drives the
deliberately over-closed final command home because nothing stopped it, the
probe is no longer detectable at its old floor pose (the arm is now standing
in front of it), and every downstream stage then reports a held object. The
lift check's "I no longer see the probe on the floor" is *absence* of
evidence, which is not evidence of a grasp.

This module turns the two sensors that CAN see the jaw volume into a verdict:

* the held-probe box fit (``probe_alignment``) run against the wrist camera —
  a fit that lands on the jaw axis is direct evidence of a held probe;
* the self-filtered depth cloud that feeds MoveIt's octomap — the robot's own
  links are already removed from it, so any point left inside the jaw volume
  is a non-robot object, and an otherwise healthy cloud with *nothing* there
  is positive evidence the jaws are empty.

Verdicts are pooled over a short window rather than trusted per frame: a
single occluded or TF-lagged frame must not condemn a good grasp, and one
lucky reflection must not rescue a bad one.

Frames: ``points``/``contact``/``axis`` are all expressed in the planning link
frame, where a rigidly held probe is stationary.
"""

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

HELD = 'held'
EMPTY = 'empty'
UNKNOWN = 'unknown'


def jaw_region_mask(
    points: np.ndarray,
    contact: np.ndarray,
    axis: np.ndarray,
    radius_m: float,
    along_lo_m: float,
    along_hi_m: float,
) -> np.ndarray:
    """Boolean mask of the points inside the volume a held probe must occupy.

    The volume is a cylinder segment around the line through the four-bar
    contact point along the tool approach axis: ``along_lo_m``/``along_hi_m``
    bound the along-axis coordinate measured from the contact point, positive
    toward the fingertips.

    A coaxially gripped probe runs down this axis and protrudes past the
    fingertips, so it fills the segment lengthwise; a side-gripped probe
    crosses it near the contact point and still falls inside the radius.
    Either way an empty gripper leaves the segment empty.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    c = np.asarray(contact, dtype=np.float64).reshape(3,)
    a = np.asarray(axis, dtype=np.float64).reshape(3,)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return np.zeros(len(pts), dtype=bool)
    a = a / norm

    rel = pts - c
    along = rel @ a
    radial = np.linalg.norm(rel - np.outer(along, a), axis=1)
    return (
        (along >= float(along_lo_m))
        & (along <= float(along_hi_m))
        & (radial <= float(radius_m))
    )


def cloud_is_probe_like(
    points: np.ndarray,
    min_points: int = 25,
    min_elongation: float = 3.0,
    min_extent_m: float = 0.04,
) -> Tuple[bool, float, float]:
    """Does this cloud look like a rod rather than a blob of clutter?

    Returns ``(is_probe_like, elongation, extent_m)``. The jaw volume is small
    and the self-filter already removed the robot, so the bar is deliberately
    low: enough points, a dominant long axis, and a physical extent along it.
    A stray corner of the sand box clipping the volume is round and short and
    fails both shape gates.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < max(4, int(min_points)):
        return False, 0.0, 0.0
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / max(1, len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[2] <= 1e-15:
        return False, 0.0, 0.0
    mid = max(float(eigvals[1]), 1e-15)
    elongation = float(math.sqrt(eigvals[2] / mid))
    axis = eigvecs[:, 2]
    along = centered @ axis
    extent = float(np.percentile(along, 98.0) - np.percentile(along, 2.0))
    ok = elongation >= float(min_elongation) and extent >= float(min_extent_m)
    return ok, elongation, extent


def empty_close_gap(
    gap_m: float,
    probe_width_m: float,
    tolerance_m: float,
) -> bool:
    """True when the final close arrived at a gap no probe could allow.

    The final close is deliberately over-closed so that the probe itself stops
    the jaws. Reaching a gap below ``probe_width_m - tolerance_m`` means
    nothing did.
    """
    return float(gap_m) < float(probe_width_m) - float(tolerance_m)


class HeldProbeEvidence:
    """Rolling vote over per-frame held/empty verdicts.

    ``UNKNOWN`` frames (no cloud, no TF, jaw volume out of view) are recorded
    but never counted toward the decision: they mean the sensors could not
    look, which must not drift the verdict either way. A decision needs
    ``min_votes`` decisive frames inside ``window_sec``; below that the pool
    reports ``UNKNOWN`` and the caller keeps waiting.
    """

    def __init__(
        self,
        window_sec: float = 6.0,
        min_votes: int = 3,
        min_held_votes: int = 2,
        empty_fraction: float = 0.75,
        capacity: int = 64,
    ) -> None:
        self.window_sec = max(0.5, float(window_sec))
        self.min_votes = max(1, int(min_votes))
        self.min_held_votes = max(1, int(min_held_votes))
        self.empty_fraction = float(np.clip(empty_fraction, 0.0, 1.0))
        self._votes: deque = deque(maxlen=max(4, int(capacity)))

    def reset(self) -> None:
        self._votes.clear()

    def add(self, verdict: str, now_sec: float, detail: str = '') -> None:
        self._votes.append((float(now_sec), str(verdict), str(detail)))

    def counts(self, now_sec: float) -> Tuple[int, int, int]:
        """``(held, empty, unknown)`` inside the window."""
        held = empty = unknown = 0
        for stamp, verdict, _ in self._votes:
            if now_sec - stamp > self.window_sec:
                continue
            if verdict == HELD:
                held += 1
            elif verdict == EMPTY:
                empty += 1
            else:
                unknown += 1
        return held, empty, unknown

    def last_held_sec(self) -> Optional[float]:
        for stamp, verdict, _ in reversed(self._votes):
            if verdict == HELD:
                return float(stamp)
        return None

    def verdict(self, now_sec: float) -> str:
        """Pooled verdict, deliberately asymmetric.

        The two evidence sources are not equally strong. A HELD vote is a
        probe-shaped cloud found on the jaw axis, past gates strict enough
        that it is hard to produce by accident — so ``min_held_votes`` of them
        settle it outright, however many EMPTY frames sit alongside (the
        fingers occlude the held probe most of the time, and those frames are
        the expected noise). An EMPTY vote is only ever an *absence*, so
        condemning the grasp needs a near-unanimous ``empty_fraction`` of
        clear looks AND not one held look in the window: a single solid fit is
        enough to withdraw the verdict back to UNKNOWN.
        """
        held, empty, _ = self.counts(now_sec)
        if held >= self.min_held_votes:
            return HELD
        decisive = held + empty
        if decisive < self.min_votes:
            return UNKNOWN
        if held == 0 and empty >= math.ceil(self.empty_fraction * decisive):
            return EMPTY
        return UNKNOWN

    def summary(self, now_sec: float) -> str:
        held, empty, unknown = self.counts(now_sec)
        return (f'held={held} empty={empty} unknown={unknown} '
                f'over {self.window_sec:.1f}s -> {self.verdict(now_sec)}')
