"""Unit tests for the terrain height map and scoop-site selection."""

import numpy as np
import pytest

from aries_soil_sample.terrain import (
    HeightMap,
    WorkRegion,
    divot_volume,
    fit_plane,
    height_map,
    select_scoop_site,
    slope_deg,
)

REGION = WorkRegion(0.40, 0.60, -0.10, 0.10, -0.20, 0.10)
CELL = 0.010


def flat_cloud(z=-0.010, n=8000, seed=1, region=REGION):
    """Dense points on level ground filling the region."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(region.x_min, region.x_max, n),
        rng.uniform(region.y_min, region.y_max, n),
        np.full(n, z) + rng.normal(0.0, 0.0005, n),
    ])


def test_height_map_recovers_a_level_surface():
    hmap = height_map(flat_cloud(z=-0.010), REGION, CELL)
    assert hmap.shape == (20, 20)
    assert hmap.valid_fraction > 0.95
    observed = hmap.grid[np.isfinite(hmap.grid)]
    assert observed.mean() == pytest.approx(-0.010, abs=0.001)


def test_height_map_ignores_points_outside_the_region():
    inside = flat_cloud(n=2000)
    outside = np.column_stack([
        np.full(500, 1.5),                     # far beyond x_max
        np.zeros(500),
        np.full(500, -0.01),
    ])
    above = np.column_stack([                  # the rover's own deck
        np.full(500, 0.5), np.zeros(500), np.full(500, 0.6),
    ])
    hmap = height_map(np.vstack([inside, outside, above]), REGION, CELL)
    assert int(hmap.counts.sum()) == len(inside)


def test_height_map_median_rejects_a_speckle():
    """One bright return must not become a mound the bucket tries to cut."""
    cloud = flat_cloud(z=-0.010, n=4000)
    speckle = np.tile([0.50, 0.0, 0.20], (3, 1))     # 210 mm above the ground
    hmap = height_map(np.vstack([cloud, speckle]), REGION, CELL)
    ix = int((0.50 - REGION.x_min) / CELL)
    iy = int((0.0 - REGION.y_min) / CELL)
    assert hmap.grid[ix, iy] < 0.0            # median stayed on the surface


def test_height_map_of_nothing_is_empty_not_an_error():
    hmap = height_map(np.empty((0, 3)), REGION, CELL)
    assert hmap.valid_fraction == 0.0
    assert not np.isfinite(hmap.grid).any()


def test_fit_plane_on_level_ground():
    fit = fit_plane(flat_cloud(z=-0.02, n=400))
    assert fit is not None
    normal, centroid, rms = fit
    assert normal[2] > 0.99
    assert centroid[2] == pytest.approx(-0.02, abs=0.002)
    assert rms < 0.002
    assert slope_deg(normal) < 3.0


def test_fit_plane_measures_slope_but_not_roughness():
    """A tilted-but-smooth patch is sloped, not rough -- they are separate gates."""
    rng = np.random.default_rng(4)
    x = rng.uniform(0.40, 0.50, 400)
    y = rng.uniform(-0.05, 0.05, 400)
    z = 0.5 * (x - 0.45)                     # ~26.6 deg ramp, perfectly smooth
    fit = fit_plane(np.column_stack([x, y, z]))
    assert fit is not None
    normal, _, rms = fit
    assert rms < 1e-9
    assert slope_deg(normal) == pytest.approx(26.57, abs=0.5)


def test_fit_plane_needs_three_points():
    assert fit_plane(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])) is None


def test_level_ground_yields_scoop_sites():
    hmap = height_map(flat_cloud(), REGION, CELL)
    sites = select_scoop_site(hmap, 0.060, prefer_xy=[0.50, 0.0])
    assert sites
    best = sites[0]
    assert best.roughness_m < 0.006
    assert best.slope_deg < 12.0
    assert best.coverage > 0.9
    assert REGION.x_min <= best.centre[0] <= REGION.x_max


def test_a_buried_rock_is_rejected_as_rough_ground():
    """Roughness is the primary "safe to cut" gate: it catches an obstacle long
    before the obstacle tilts the fitted plane."""
    cloud = flat_cloud(z=-0.010, n=12000)
    rock = np.column_stack([
        np.random.default_rng(9).uniform(0.498, 0.512, 400),
        np.random.default_rng(10).uniform(-0.006, 0.006, 400),
        np.full(400, 0.030),                  # 40 mm proud of the ground
    ])
    hmap = height_map(np.vstack([cloud, rock]), REGION, CELL)
    sites = select_scoop_site(hmap, 0.060, prefer_xy=[0.505, 0.0])
    # Sites still exist elsewhere, but none may sit on the rock.
    assert sites
    for s in sites:
        assert np.linalg.norm(s.centre[:2] - np.array([0.505, 0.0])) > 0.03


def test_steep_ground_is_rejected():
    rng = np.random.default_rng(6)
    x = rng.uniform(REGION.x_min, REGION.x_max, 12000)
    y = rng.uniform(REGION.y_min, REGION.y_max, 12000)
    z = -0.05 + 1.0 * (x - REGION.x_min)      # 45 deg ramp
    hmap = height_map(np.column_stack([x, y, z]), REGION, CELL)
    assert select_scoop_site(hmap, 0.060, max_slope_deg=12.0) == []


def test_sparse_ground_is_rejected_for_coverage():
    hmap = height_map(flat_cloud(n=60), REGION, CELL)     # a handful of cells
    assert select_scoop_site(hmap, 0.060, min_coverage=0.70) == []


def test_candidate_sites_are_spread_out_so_a_retry_moves():
    hmap = height_map(flat_cloud(), REGION, CELL)
    sites = select_scoop_site(hmap, 0.060, max_sites=5)
    assert len(sites) >= 2
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            assert np.linalg.norm(a.centre[:2] - b.centre[:2]) >= 0.060 - 1e-9


def test_preference_pulls_the_best_site_toward_the_comfortable_posture():
    hmap = height_map(flat_cloud(), REGION, CELL)
    near = select_scoop_site(hmap, 0.060, prefer_xy=[0.43, -0.07])[0]
    far = select_scoop_site(hmap, 0.060, prefer_xy=[0.57, 0.07])[0]
    assert near.centre[0] < far.centre[0]
    assert near.centre[1] < far.centre[1]


# --- capture verification ---------------------------------------------------

def _map_from_grid(grid, counts=None):
    g = np.asarray(grid, dtype=np.float64)
    c = np.full(g.shape, 5, dtype=np.int32) if counts is None else counts
    return HeightMap(g, c, 0.40, -0.10, 0.010)


def test_divot_volume_measures_a_known_hole():
    before = _map_from_grid(np.zeros((20, 20)))
    after_grid = np.zeros((20, 20))
    after_grid[9:12, 9:12] = -0.010          # 3x3 cells, 10 mm deep
    after = _map_from_grid(after_grid)
    centre = before.cell_centre(10, 10)
    volume, max_drop, cells = divot_volume(before, after, centre, 0.05)
    assert volume == pytest.approx(9 * (0.010 ** 2) * 0.010, rel=1e-6)
    assert max_drop == pytest.approx(0.010)
    assert cells > 9


def test_untouched_ground_integrates_to_zero():
    rng = np.random.default_rng(2)
    before = _map_from_grid(rng.normal(0.0, 0.0005, (20, 20)))
    after = _map_from_grid(rng.normal(0.0, 0.0005, (20, 20)))
    volume, _, _ = divot_volume(before, after, before.cell_centre(10, 10), 0.05,
                               min_drop_m=0.002)
    assert volume == 0.0


def test_displaced_soil_piled_up_does_not_cancel_the_hole():
    before = _map_from_grid(np.zeros((20, 20)))
    after_grid = np.zeros((20, 20))
    after_grid[10, 10] = -0.020              # hole
    after_grid[10, 11] = +0.020              # spoil pile at the rim
    after = _map_from_grid(after_grid)
    volume, _, _ = divot_volume(before, after, before.cell_centre(10, 10), 0.05)
    assert volume == pytest.approx((0.010 ** 2) * 0.020, rel=1e-6)


def test_cells_unobserved_in_either_survey_are_skipped():
    before = _map_from_grid(np.zeros((20, 20)))
    after_grid = np.full((20, 20), np.nan)
    after_grid[10, 10] = -0.010
    after = _map_from_grid(after_grid)
    volume, _, cells = divot_volume(before, after, before.cell_centre(10, 10), 0.05)
    assert cells == 1
    assert volume == pytest.approx((0.010 ** 2) * 0.010, rel=1e-6)


def test_mismatched_grids_are_refused():
    a = _map_from_grid(np.zeros((20, 20)))
    b = _map_from_grid(np.zeros((10, 10)))
    with pytest.raises(ValueError):
        divot_volume(a, b, [0.5, 0.0], 0.05)


# --- operator-configured sampling point -------------------------------------

def test_configured_point_takes_its_z_from_the_measured_surface():
    """The point says WHERE; perception still says how deep."""
    from aries_soil_sample.terrain import site_at_xy
    hmap = height_map(flat_cloud(z=-0.175), REGION, CELL)
    site, why = site_at_xy(hmap, [0.50, 0.0], 0.060)
    assert site is not None, why
    assert site.centre[0] == pytest.approx(0.50)
    assert site.centre[1] == pytest.approx(0.0)
    assert site.centre[2] == pytest.approx(-0.175, abs=0.002)
    assert site.slope_deg < 3.0


def test_configured_point_outside_the_region_is_rejected_with_a_reason():
    from aries_soil_sample.terrain import site_at_xy
    hmap = height_map(flat_cloud(), REGION, CELL)
    site, why = site_at_xy(hmap, [1.20, 0.0], 0.060)
    assert site is None
    assert 'outside the surveyed work region' in why


def test_configured_point_too_close_to_the_edge_is_rejected():
    from aries_soil_sample.terrain import site_at_xy
    hmap = height_map(flat_cloud(), REGION, CELL)
    site, why = site_at_xy(hmap, [REGION.x_min + 0.005, 0.0], 0.060)
    assert site is None
    assert 'edge' in why or 'outside' in why


def test_configured_point_on_a_rock_is_rejected_for_roughness():
    """Choosing a patch by hand does not make it safe to cut."""
    from aries_soil_sample.terrain import site_at_xy
    cloud = flat_cloud(z=-0.175, n=12000)
    rock = np.column_stack([
        np.random.default_rng(21).uniform(0.495, 0.515, 500),
        np.random.default_rng(22).uniform(-0.010, 0.010, 500),
        np.full(500, -0.130),
    ])
    hmap = height_map(np.vstack([cloud, rock]), REGION, CELL)
    site, why = site_at_xy(hmap, [0.505, 0.0], 0.060)
    assert site is None
    assert 'too rough' in why


def test_configured_point_on_unobserved_ground_is_rejected_for_coverage():
    from aries_soil_sample.terrain import site_at_xy
    hmap = height_map(flat_cloud(n=40), REGION, CELL)
    site, why = site_at_xy(hmap, [0.50, 0.0], 0.060)
    assert site is None
    assert 'observed' in why
