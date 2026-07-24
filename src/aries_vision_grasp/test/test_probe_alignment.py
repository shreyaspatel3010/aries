"""Unit tests for the point-to-box probe pose fitting."""

import math

import numpy as np
import pytest

from aries_vision_grasp.probe_alignment import (
    axis_angle_deg,
    axis_half_widths,
    box_surface_distances,
    closest_points_on_box,
    fit_box_to_points,
    full_model_centre_along_axis,
    long_axis_fat_end_sign,
)

# Probe half extents (metres): X width, Y height, Z long axis.
HALF = np.array([0.0225, 0.0225, 0.150])


def rodrigues(axis, angle_rad):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


def sample_box_surface(rotation, centre, n_per_face=160, noise_m=0.0015, seed=7):
    """Sample the faces a wrist camera actually sees: top, both ends, one side."""
    rng = np.random.default_rng(seed)
    faces = []
    # Top face (+Y).
    x = rng.uniform(-HALF[0], HALF[0], n_per_face)
    z = rng.uniform(-HALF[2], HALF[2], n_per_face)
    faces.append(np.column_stack([x, np.full(n_per_face, HALF[1]), z]))
    # Both end caps (±Z).
    for sign in (1.0, -1.0):
        x = rng.uniform(-HALF[0], HALF[0], n_per_face // 4)
        y = rng.uniform(-HALF[1], HALF[1], n_per_face // 4)
        faces.append(np.column_stack([x, y, np.full(n_per_face // 4, sign * HALF[2])]))
    # One side face (+X).
    y = rng.uniform(-HALF[1], HALF[1], n_per_face // 2)
    z = rng.uniform(-HALF[2], HALF[2], n_per_face // 2)
    faces.append(np.column_stack([np.full(n_per_face // 2, HALF[0]), y, z]))

    q = np.vstack(faces)
    q += rng.normal(scale=noise_m, size=q.shape)
    return q @ np.asarray(rotation).T + np.asarray(centre)


def test_closest_points_outside_clamp():
    pts = np.array([[0.1, 0.0, 0.0], [0.0, -0.5, 0.2]])
    s = closest_points_on_box(pts, HALF)
    assert np.allclose(s[0], [HALF[0], 0.0, 0.0])
    assert np.allclose(s[1], [0.0, -HALF[1], HALF[2]])


def test_closest_points_inside_projects_to_nearest_face():
    pts = np.array([[0.020, 0.0, 0.0]])   # 2.5 mm from the +X face
    s = closest_points_on_box(pts, HALF)
    assert np.allclose(s[0], [HALF[0], 0.0, 0.0])


def test_surface_distances_zero_on_surface():
    R = rodrigues([0.3, -0.5, 0.8], 0.7)
    c = np.array([0.02, -0.03, 0.21])
    pts = sample_box_surface(R, c, noise_m=0.0)
    dist = box_surface_distances(pts, HALF, R, c)
    assert float(dist.max()) < 1e-9


def test_axis_angle_symmetric_under_flip():
    a = np.array([1.0, 0.0, 0.0])
    assert axis_angle_deg(a, -a) == pytest.approx(0.0, abs=1e-9)
    b = rodrigues([0.0, 0.0, 1.0], math.radians(10.0)) @ a
    assert axis_angle_deg(a, b) == pytest.approx(10.0, abs=1e-6)
    assert axis_angle_deg(a, -b) == pytest.approx(10.0, abs=1e-6)


def test_fit_recovers_perturbed_pose():
    R_gt = rodrigues([0.2, 0.9, -0.4], 0.9)
    c_gt = np.array([0.005, 0.028, 0.205])
    pts = sample_box_surface(R_gt, c_gt)

    R_init = R_gt @ rodrigues([1.0, 1.0, 0.3], math.radians(8.0))
    c_init = c_gt + np.array([0.010, -0.008, 0.012])

    fit = fit_box_to_points(pts, HALF, R_init, c_init)
    assert fit is not None
    assert float(np.linalg.norm(fit.centre - c_gt)) < 0.004
    assert axis_angle_deg(fit.rotation[:, 2], R_gt[:, 2]) < 1.5
    assert fit.rms_m < 0.004


def test_fit_survives_outliers():
    rng = np.random.default_rng(11)
    R_gt = rodrigues([0.0, 0.0, 1.0], 0.4)
    c_gt = np.array([0.0, 0.026, 0.218])
    pts = sample_box_surface(R_gt, c_gt)
    # 8% far outliers (floor / gripper finger pixels leaking into the mask).
    n_out = int(0.08 * len(pts))
    outliers = c_gt + rng.uniform(-0.15, 0.15, size=(n_out, 3))
    pts_all = np.vstack([pts, outliers])

    fit = fit_box_to_points(pts_all, HALF, R_gt, c_gt + np.array([0.006, -0.004, 0.008]))
    assert fit is not None
    assert float(np.linalg.norm(fit.centre - c_gt)) < 0.006
    assert axis_angle_deg(fit.rotation[:, 2], R_gt[:, 2]) < 2.5


def test_fit_rejects_too_few_points():
    assert fit_box_to_points(np.zeros((5, 3)), HALF, np.eye(3), np.zeros(3)) is None


def sample_tapered_probe(rotation, centre, tip_sign, n=600, seed=13):
    """Cone-like probe cloud: full-width body on one half, tapering to a
    point toward ``tip_sign`` * Z — mimics the real probe.stl profile."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-HALF[2], HALF[2], n)
    taper = np.clip(1.0 - np.maximum(0.0, tip_sign * z / HALF[2]), 0.05, 1.0)
    angle = rng.uniform(0.0, 2.0 * math.pi, n)
    r = HALF[0] * taper
    q = np.column_stack([r * np.cos(angle), r * np.sin(angle), z])
    q += rng.normal(scale=0.001, size=q.shape)
    return q @ np.asarray(rotation).T + np.asarray(centre)


def test_axis_half_widths_detects_tapered_end():
    """Regression for the 180° end-for-end flip: the width profile of a
    tapered cloud must reveal which Z half holds the fat body."""
    R = rodrigues([0.2, 0.8, 0.5], 0.6)
    c = np.array([0.01, -0.02, 0.19])
    # Tip toward +Z → fat end toward -Z → width_neg > width_pos.
    pts = sample_tapered_probe(R, c, tip_sign=+1)
    w_neg, w_pos, n_neg, n_pos = axis_half_widths(pts, R, c)
    assert n_neg > 100 and n_pos > 100
    assert w_neg > 1.3 * w_pos
    # Flipped probe: fat end toward +Z.
    pts = sample_tapered_probe(R, c, tip_sign=-1)
    w_neg, w_pos, _, _ = axis_half_widths(pts, R, c)
    assert w_pos > 1.3 * w_neg


def test_long_axis_fat_end_sign_is_direction_aware():
    """The PCA long axis is sign-arbitrary; the taper must resolve which end
    is the fat body regardless of which way the caller's axis points."""
    R = rodrigues([0.2, 0.8, 0.5], 0.6)
    c = np.array([0.01, -0.02, 0.19])
    axis = R[:, 2]
    # Tip toward +axis → fat body sits on the -axis side.
    pts = sample_tapered_probe(R, c, tip_sign=+1)
    assert long_axis_fat_end_sign(pts, c, axis) == -1
    # Same cloud, reversed axis → the answer must flip with it.
    assert long_axis_fat_end_sign(pts, c, -axis) == 1
    # Physically flipped probe.
    pts = sample_tapered_probe(R, c, tip_sign=-1)
    assert long_axis_fat_end_sign(pts, c, axis) == 1


def test_long_axis_fat_end_sign_abstains_when_not_decisive():
    """Better to keep the previous convention than to guess from a cloud that
    cannot see the taper."""
    R = rodrigues([0.1, 0.3, 0.9], 0.4)
    c = np.array([0.0, 0.0, 0.2])
    axis = R[:, 2]

    # Untapered cloud: both halves are equally wide.
    rng = np.random.default_rng(7)
    z = rng.uniform(-HALF[2], HALF[2], 1200)
    ang = rng.uniform(0.0, 2.0 * math.pi, 1200)
    q = np.column_stack([HALF[0] * np.cos(ang), HALF[0] * np.sin(ang), z])
    assert long_axis_fat_end_sign(q @ R.T + c, c, axis) == 0

    # Only the body half visible — the tip is occluded, so nothing to compare.
    pts = sample_tapered_probe(R, c, tip_sign=+1)
    along = (pts - c) @ axis
    assert long_axis_fat_end_sign(pts[along < 0.0], c, axis) == 0

    # Too few points to trust either half.
    assert long_axis_fat_end_sign(pts[:20], c, axis) == 0


LENGTH = 2.0 * HALF[2]


def test_full_model_centre_predicts_occluded_tip():
    """With the tip buried, the centre must be pinned to the visible fat end
    plus half the known length -- not left at the biased visible centroid."""
    R = rodrigues([0.2, 0.8, 0.5], 0.6)
    c_true = np.array([0.03, -0.02, 0.20])
    axis = R[:, 2]                       # tip_sign=+1 -> fat->tip is +axis
    pts = sample_tapered_probe(R, c_true, tip_sign=+1, n=2000)
    # Bury the tip: keep only the body + a stub of the taper.
    along = (pts - c_true) @ axis
    visible = pts[along < 0.03]
    c_visible = visible.mean(axis=0)
    # The visible centroid is dragged toward the fat end by the occlusion.
    assert np.linalg.norm(c_visible - c_true) > 0.03

    anchored = full_model_centre_along_axis(visible, c_visible, axis, LENGTH)
    assert anchored is not None
    # The predicted full-model centre is far closer to the truth...
    assert np.linalg.norm(anchored - c_true) < np.linalg.norm(c_visible - c_true)
    assert np.linalg.norm(anchored - c_true) < 0.02
    # ...and only the along-axis coordinate moved (lateral position preserved).
    assert np.linalg.norm(np.cross(anchored - c_visible, axis)) < 1e-9


def test_full_model_centre_abstains_when_fully_visible():
    """A cloud that already spans the whole length has nothing occluded to
    predict, so the direct fit centre is kept (returns None)."""
    R = rodrigues([0.1, 0.3, 0.9], 0.4)
    c = np.array([0.0, 0.0, 0.2])
    pts = sample_tapered_probe(R, c, tip_sign=+1, n=2000)
    assert full_model_centre_along_axis(pts, c, R[:, 2], LENGTH) is None


def sample_visible_box_surface(rotation, centre, cam_pos, n_per_face=220,
                               noise_m=0.0015, seed=3):
    """Sample only the faces whose outward normal faces the camera — what a
    depth camera's segmentation mask actually observes."""
    rng = np.random.default_rng(seed)
    R = np.asarray(rotation)
    c = np.asarray(centre)
    pts = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal_world = sign * R[:, axis]
            face_centre = c + normal_world * HALF[axis]
            if float(np.dot(np.asarray(cam_pos) - face_centre, normal_world)) <= 0.0:
                continue  # back face — invisible
            other = [a for a in range(3) if a != axis]
            q = np.zeros((n_per_face, 3))
            q[:, axis] = sign * HALF[axis]
            for a in other:
                q[:, a] = rng.uniform(-HALF[a], HALF[a], n_per_face)
            pts.append(q)
    q_all = np.vstack(pts)
    q_all += rng.normal(scale=noise_m, size=q_all.shape)
    return q_all @ R.T + c


def test_reacquisition_recovers_from_pca_init():
    """Regression for flipped-mesh recovery: the re-acquisition path ignores
    the (wrong) attached pose and initialises from the cloud's own PCA long
    axis plus the centroid pushed half a cross-section along the viewing ray.
    That init must converge to the true pose even when the previous attached
    pose was arbitrarily wrong."""
    R_gt = rodrigues([0.7, 0.1, 0.7], 1.3)   # arbitrary held pose (tilted)
    c_gt = np.array([-0.03, 0.05, 0.24])
    pts = sample_visible_box_surface(R_gt, c_gt, cam_pos=np.zeros(3))

    # Node-style init: PCA long axis + centroid pushed along the view ray.
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered / (len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, int(np.argmax(eigvals))]
    cam = np.zeros(3)                        # camera at the link origin-ish
    view = centroid - cam
    view /= np.linalg.norm(view)
    centre_init = centroid + view * HALF[0]
    y_axis = np.cross(long_axis, view)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, long_axis)
    R_init = np.column_stack([x_axis, y_axis / np.linalg.norm(y_axis), long_axis])

    fit = fit_box_to_points(pts, HALF, R_init, centre_init, iterations=100, trim_fraction=0.15)
    assert fit is not None
    # With only the camera-facing faces observed, the centre is constrained to
    # roughly the sensor-noise level along the seen directions and to the box
    # model along the hidden one — ~1 cm total is the observable limit, and a
    # residual flip would show up as >100 mm / ~90 deg.
    assert float(np.linalg.norm(fit.centre - c_gt)) < 0.012
    assert axis_angle_deg(fit.rotation[:, 2], R_gt[:, 2]) < 2.5


def test_fit_survives_gripper_blob_contamination():
    """The depth-prior re-acquisition cloud includes gripper bucket/finger
    surfaces near the grasp contact. The absolute outlier-residual gate must
    recover the box pose when ~20% of the points are a compact off-box blob
    (fractional trimming alone converges onto the blob instead)."""
    rng = np.random.default_rng(9)
    R_gt = rodrigues([0.1, 0.9, 0.2], 0.8)
    c_gt = np.array([0.0, 0.03, 0.22])
    pts = sample_visible_box_surface(R_gt, c_gt, cam_pos=np.zeros(3), seed=5)
    # Compact "gripper" blob 4 cm off the box surface near the grasp point.
    blob_centre = c_gt + R_gt[:, 1] * (HALF[1] + 0.04)
    n_blob = int(0.25 * len(pts))
    blob = blob_centre + rng.normal(scale=0.015, size=(n_blob, 3))
    pts_all = np.vstack([pts, blob])
    rng.shuffle(pts_all)

    centroid = pts_all.mean(axis=0)
    centered = pts_all - centroid
    cov = centered.T @ centered / (len(pts_all) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, int(np.argmax(eigvals))]
    view = centroid / np.linalg.norm(centroid)
    centre_init = centroid + view * HALF[0]
    y_axis = np.cross(long_axis, view)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, long_axis)
    R_init = np.column_stack([x_axis, y_axis, long_axis])

    fit = fit_box_to_points(pts_all, HALF, R_init, centre_init,
                            iterations=100, trim_fraction=0.15,
                            outlier_residual_m=0.020)
    assert fit is not None
    assert float(np.linalg.norm(fit.centre - c_gt)) < 0.010
    assert axis_angle_deg(fit.rotation[:, 2], R_gt[:, 2]) < 3.0
