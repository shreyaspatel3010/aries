"""Terrain perception for soil sampling -- geometry only, no learned model.

There is no trained detector for soil, and for this task there does not need to
be one. A probe is an *object*: you must recognise which pixels are the thing
before you can grasp it. Soil is *terrain*: any sufficiently flat, level,
reachable, unobstructed patch of ground is a valid place to put the bucket. That
question is answered by geometry, so this module answers it from the depth image
alone and nothing here imports a model.

The pipeline is deliberately small:

1. back-project the depth image and keep the points inside a work region
   expressed in the planning frame (``height_map``);
2. reduce them to a 2.5-D height map, one representative surface height per
   grid cell -- soil has no overhangs, so a height map loses nothing;
3. slide the bucket footprint over the map and score each placement on
   roughness, slope and coverage (``select_scoop_site``).

The same height map does double duty as scoop *verification*: a successful scoop
removes material, so re-surveying the site and differencing the two maps
measures the divot (``divot_volume``). That matters on this robot because the
two signals a normal pick would use are both unavailable -- the gripper has no
position feedback on hardware, and the wrist camera cannot see inside the jaws
(the padded self-filter blanks its whole near field). The ground at 0.4-0.5 m is
comfortably inside the camera's working range, so the divot is the one piece of
positive evidence that can actually be measured.

Frames: every public function takes and returns points in the planning frame
(``base_link``), with +Z up. Heights are metres.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WorkRegion:
    """Axis-aligned volume of the planning frame that may be sampled.

    The Z bounds are a sanity gate, not a target: they reject the rover's own
    deck above and stray long-range returns below, leaving the ground itself.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (
            (p[:, 0] >= self.x_min) & (p[:, 0] <= self.x_max)
            & (p[:, 1] >= self.y_min) & (p[:, 1] <= self.y_max)
            & (p[:, 2] >= self.z_min) & (p[:, 2] <= self.z_max)
        )

    @property
    def centre_xy(self) -> np.ndarray:
        return np.array([0.5 * (self.x_min + self.x_max),
                         0.5 * (self.y_min + self.y_max)], dtype=np.float64)


@dataclass(frozen=True)
class HeightMap:
    """2.5-D surface height grid over a work region.

    ``grid[ix, iy]`` is the representative surface height of that cell, or NaN
    where no point landed. ``counts`` is how many points backed each cell, which
    the site scorer uses to refuse thinly-observed ground.
    """

    grid: np.ndarray
    counts: np.ndarray
    x_min: float
    y_min: float
    cell_m: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def cell_centre(self, ix: int, iy: int) -> np.ndarray:
        return np.array([
            self.x_min + (float(ix) + 0.5) * self.cell_m,
            self.y_min + (float(iy) + 0.5) * self.cell_m,
        ], dtype=np.float64)

    @property
    def valid_fraction(self) -> float:
        if self.grid.size == 0:
            return 0.0
        return float(np.count_nonzero(np.isfinite(self.grid))) / float(self.grid.size)


def height_map(
    points: np.ndarray,
    region: WorkRegion,
    cell_m: float = 0.01,
    percentile: float = 50.0,
) -> HeightMap:
    """Reduce a point cloud to one surface height per grid cell.

    ``percentile`` selects the representative height within a cell. The default
    is the median rather than the maximum: every return in a cell is a sample of
    the same thin surface, so the median rejects the one bright speckle that a
    max would promote into a phantom mound the bucket then tries to cut.
    """
    cell = max(1e-4, float(cell_m))
    nx = max(1, int(math.ceil((region.x_max - region.x_min) / cell)))
    ny = max(1, int(math.ceil((region.y_max - region.y_min) / cell)))
    grid = np.full((nx, ny), np.nan, dtype=np.float64)
    counts = np.zeros((nx, ny), dtype=np.int32)

    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) == 0:
        return HeightMap(grid, counts, region.x_min, region.y_min, cell)
    p = p[region.contains(p)]
    if len(p) == 0:
        return HeightMap(grid, counts, region.x_min, region.y_min, cell)

    ix = np.clip(((p[:, 0] - region.x_min) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((p[:, 1] - region.y_min) / cell).astype(np.int64), 0, ny - 1)
    flat = ix * ny + iy
    order = np.argsort(flat, kind='stable')
    flat_sorted = flat[order]
    z_sorted = p[order, 2]
    # One pass over runs of equal cell index; np.percentile per cell would be
    # thousands of tiny calls on a 25x40 grid.
    bounds = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(flat_sorted)]))
    for s, e in zip(starts, ends):
        cell_idx = int(flat_sorted[s])
        gx, gy = divmod(cell_idx, ny)
        grid[gx, gy] = float(np.percentile(z_sorted[s:e], float(percentile)))
        counts[gx, gy] = int(e - s)
    return HeightMap(grid, counts, region.x_min, region.y_min, cell)


def fit_plane(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Least-squares plane through 3+ points.

    Returns ``(normal, centroid, rms_residual)`` with the normal oriented +Z up,
    or None when the points are too few or degenerate. The residual is the
    roughness measure the site scorer thresholds on: a rock or a probe sticking
    out of the ground shows up here long before it shows up in the slope.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        return None
    centroid = p.mean(axis=0)
    centred = p - centroid
    try:
        _, sing, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(sing) < 3:
        return None
    normal = vt[2, :]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return None
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    rms = float(math.sqrt(max(0.0, float(np.mean((centred @ normal) ** 2)))))
    return normal, centroid, rms


def slope_deg(normal: Sequence[float]) -> float:
    """Angle of a surface normal from vertical, in degrees."""
    n = np.asarray(normal, dtype=np.float64).reshape(3,)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return 90.0
    nz = abs(float(n[2]) / norm)
    return float(math.degrees(math.acos(min(1.0, nz))))


@dataclass(frozen=True)
class ScoopSite:
    """A candidate place to put the bucket."""

    centre: np.ndarray          # (3,) surface point in the planning frame
    normal: np.ndarray          # (3,) surface normal, +Z up
    roughness_m: float          # rms residual to the fitted plane
    slope_deg: float
    coverage: float             # fraction of footprint cells that were observed
    score: float                # higher is better

    @property
    def summary(self) -> str:
        return (f'({self.centre[0]:.3f},{self.centre[1]:.3f},{self.centre[2]:.3f}) '
                f'roughness={self.roughness_m*1000:.1f}mm slope={self.slope_deg:.1f}deg '
                f'coverage={self.coverage*100:.0f}% score={self.score:.3f}')


def select_scoop_site(
    hmap: HeightMap,
    footprint_m: float,
    *,
    max_roughness_m: float = 0.006,
    max_slope_deg: float = 12.0,
    min_coverage: float = 0.70,
    min_points_per_cell: int = 1,
    prefer_xy: Optional[Sequence[float]] = None,
    max_sites: int = 8,
) -> List[ScoopSite]:
    """Rank bucket placements over a height map, best first.

    A site is accepted only when the bucket's whole footprint is observed
    (``min_coverage``), locally flat (``max_roughness_m``) and level
    (``max_slope_deg``). Flatness is what keeps the bucket out of trouble: a
    sloped patch deflects it sideways as it penetrates, and a rough patch means
    something is buried there.

    Among accepted sites, closeness to ``prefer_xy`` breaks ties -- pass the
    posture the arm is happiest in, so an equally good scoop is taken where the
    wrist is comfortable rather than at the edge of the envelope.
    """
    nx, ny = hmap.shape
    win = max(1, int(math.ceil(float(footprint_m) / hmap.cell_m)))
    if nx < win or ny < win:
        return []
    prefer = (np.asarray(prefer_xy, dtype=np.float64).reshape(2,)
              if prefer_xy is not None else None)
    cell_area = hmap.cell_m ** 2
    sites: List[ScoopSite] = []

    for ix in range(0, nx - win + 1):
        for iy in range(0, ny - win + 1):
            sub = hmap.grid[ix:ix + win, iy:iy + win]
            sub_counts = hmap.counts[ix:ix + win, iy:iy + win]
            observed = np.isfinite(sub) & (sub_counts >= int(min_points_per_cell))
            coverage = float(np.count_nonzero(observed)) / float(win * win)
            if coverage < float(min_coverage):
                continue
            gx, gy = np.nonzero(observed)
            pts = np.column_stack([
                hmap.x_min + (ix + gx + 0.5) * hmap.cell_m,
                hmap.y_min + (iy + gy + 0.5) * hmap.cell_m,
                sub[gx, gy],
            ])
            fit = fit_plane(pts)
            if fit is None:
                continue
            normal, centroid, rms = fit
            tilt = slope_deg(normal)
            if rms > float(max_roughness_m) or tilt > float(max_slope_deg):
                continue

            # Normalised badness terms, so the weights are readable.
            rough_term = rms / max(1e-9, float(max_roughness_m))
            slope_term = tilt / max(1e-9, float(max_slope_deg))
            score = 2.0 * coverage - 1.0 * rough_term - 0.5 * slope_term
            if prefer is not None:
                reach = float(np.linalg.norm(centroid[:2] - prefer))
                score -= 0.5 * reach          # metres; ~0.1 m costs 0.05
            sites.append(ScoopSite(
                centre=centroid,
                normal=normal,
                roughness_m=rms,
                slope_deg=tilt,
                coverage=coverage,
                score=float(score),
            ))

    sites.sort(key=lambda s: -s.score)
    return _spread_out(sites, min_separation_m=float(footprint_m), limit=int(max_sites))


def _spread_out(sites: List[ScoopSite], min_separation_m: float,
                limit: int) -> List[ScoopSite]:
    """Thin a ranked list so retries move to genuinely different ground.

    The sliding window produces dozens of near-identical overlapping placements.
    Handing those to the retry logic would re-attempt the same failed scoop a
    centimetre over; keeping them separated by a footprint makes attempt 2 a
    real second try.
    """
    kept: List[ScoopSite] = []
    for s in sites:
        if all(float(np.linalg.norm(s.centre[:2] - k.centre[:2])) >= min_separation_m
               for k in kept):
            kept.append(s)
        if len(kept) >= limit:
            break
    return kept


def site_at_xy(
    hmap: HeightMap,
    xy: Sequence[float],
    footprint_m: float,
    *,
    max_roughness_m: float = 0.006,
    max_slope_deg: float = 12.0,
    min_coverage: float = 0.70,
    min_points_per_cell: int = 1,
) -> Tuple[Optional[ScoopSite], str]:
    """Evaluate ONE operator-chosen XY instead of searching for the best patch.

    Returns ``(site, reason)``. The site carries the MEASURED surface height and
    normal at that location, so a configured point says *where* to sample while
    perception still decides how deep and at what angle -- a hand-typed Z would
    otherwise either scoop air or drive the bucket into the ground.

    The same roughness/slope/coverage gates apply as for an automatic site: being
    chosen by hand does not make a patch safe to cut, and ``reason`` says which
    gate rejected it so a bad coordinate is obvious rather than mysterious.
    """
    target = np.asarray(xy, dtype=np.float64).reshape(2,)
    win = max(1, int(math.ceil(float(footprint_m) / hmap.cell_m)))
    nx, ny = hmap.shape
    # Centre the footprint window on the requested XY.
    cx = int(round((target[0] - hmap.x_min) / hmap.cell_m - 0.5))
    cy = int(round((target[1] - hmap.y_min) / hmap.cell_m - 0.5))
    ix = cx - win // 2
    iy = cy - win // 2
    if ix < 0 or iy < 0 or ix + win > nx or iy + win > ny:
        return None, (f'requested point ({target[0]:.3f},{target[1]:.3f}) is outside the '
                      'surveyed work region, or too close to its edge to fit the '
                      f'{footprint_m*1000:.0f}mm bucket footprint')

    sub = hmap.grid[ix:ix + win, iy:iy + win]
    sub_counts = hmap.counts[ix:ix + win, iy:iy + win]
    observed = np.isfinite(sub) & (sub_counts >= int(min_points_per_cell))
    coverage = float(np.count_nonzero(observed)) / float(win * win)
    if coverage < float(min_coverage):
        return None, (f'only {coverage*100:.0f}% of the footprint at that point was '
                      f'observed (need {float(min_coverage)*100:.0f}%)')
    gx, gy = np.nonzero(observed)
    pts = np.column_stack([
        hmap.x_min + (ix + gx + 0.5) * hmap.cell_m,
        hmap.y_min + (iy + gy + 0.5) * hmap.cell_m,
        sub[gx, gy],
    ])
    fit = fit_plane(pts)
    if fit is None:
        return None, 'could not fit a surface plane at that point'
    normal, centroid, rms = fit
    tilt = slope_deg(normal)
    if rms > float(max_roughness_m):
        return None, (f'ground there is too rough: {rms*1000:.1f}mm rms vs '
                      f'{float(max_roughness_m)*1000:.1f}mm limit (something is buried '
                      'or protruding)')
    if tilt > float(max_slope_deg):
        return None, (f'ground there is too sloped: {tilt:.1f}deg vs '
                      f'{float(max_slope_deg):.1f}deg limit')
    # Keep the operator's XY; take Z and the normal from the measured surface.
    centre = np.array([target[0], target[1], float(centroid[2])], dtype=np.float64)
    return ScoopSite(centre=centre, normal=normal, roughness_m=rms, slope_deg=tilt,
                     coverage=coverage, score=0.0), 'configured point accepted'


def divot_volume(
    before: HeightMap,
    after: HeightMap,
    centre_xy: Sequence[float],
    radius_m: float,
    *,
    min_drop_m: float = 0.002,
) -> Tuple[float, float, int]:
    """Volume of material removed between two surveys of the same ground.

    Returns ``(volume_m3, max_drop_m, cells)`` over cells within ``radius_m`` of
    ``centre_xy`` that were observed in BOTH surveys. Only downward changes
    count: soil pushed up at the rim of the hole is displaced, not collected,
    and letting it cancel the hole would hide a successful scoop.

    ``min_drop_m`` ignores per-cell changes at or below depth noise, so an
    unchanged surface integrates to zero instead of accumulating sensor jitter
    into a phantom sample.
    """
    if before.cell_m != after.cell_m or before.shape != after.shape:
        raise ValueError('height maps must share a grid to be differenced')
    c = np.asarray(centre_xy, dtype=np.float64).reshape(2,)
    nx, ny = before.shape
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    xs = before.x_min + (gx + 0.5) * before.cell_m
    ys = before.y_min + (gy + 0.5) * before.cell_m
    within = ((xs - c[0]) ** 2 + (ys - c[1]) ** 2) <= float(radius_m) ** 2
    both = within & np.isfinite(before.grid) & np.isfinite(after.grid)
    if not both.any():
        return 0.0, 0.0, 0
    drop = before.grid[both] - after.grid[both]
    drop = np.where(drop >= float(min_drop_m), drop, 0.0)
    volume = float(np.sum(drop) * (before.cell_m ** 2))
    return volume, float(drop.max(initial=0.0)), int(np.count_nonzero(both))
