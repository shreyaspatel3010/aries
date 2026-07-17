"""Unit tests for the quaternion/rotation helpers."""

import math

import numpy as np
import pytest

from aries_vision_grasp.geometry import (
    estimate_stationary_target_camera_offset,
    matrix_to_quat,
    normalize,
    quat_to_matrix,
    quaternion_distance_rad,
    quaternion_rotation_vector_error,
    rpy_to_quat,
    sec_to_duration,
    duration_to_sec,
    wrap_to_pi,
)


def random_quaternions(n=100, seed=42):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, 4))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.mark.parametrize('xyzw', random_quaternions())
def test_quat_matrix_roundtrip(xyzw):
    from geometry_msgs.msg import Quaternion
    q = Quaternion(x=xyzw[0], y=xyzw[1], z=xyzw[2], w=xyzw[3])
    R = quat_to_matrix(q)
    # R must be a proper rotation.
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    q2 = matrix_to_quat(R)
    # q and -q are the same rotation.
    assert quaternion_distance_rad(q, q2) == pytest.approx(0.0, abs=1e-6)


def test_rpy_to_quat_identity_and_axes():
    q = rpy_to_quat(0.0, 0.0, 0.0)
    assert (q.x, q.y, q.z, q.w) == pytest.approx((0.0, 0.0, 0.0, 1.0))

    # 90° yaw maps +X to +Y.
    R = quat_to_matrix(rpy_to_quat(0.0, 0.0, math.pi / 2))
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)

    # roll=pi flips Z (the top-down grasp orientation).
    R = quat_to_matrix(rpy_to_quat(math.pi, 0.0, 0.0))
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-9)


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(3.5 * math.pi) == pytest.approx(-0.5 * math.pi, abs=1e-12)
    assert wrap_to_pi(-3.5 * math.pi) == pytest.approx(0.5 * math.pi, abs=1e-12)


def test_duration_roundtrip():
    for seconds in [0.0, 0.4999999, 1.0, 2.25, 12.000001]:
        assert duration_to_sec(sec_to_duration(seconds)) == pytest.approx(seconds, abs=1e-8)
    # Negative durations clamp to zero.
    assert duration_to_sec(sec_to_duration(-3.0)) == 0.0


def test_normalize():
    assert np.allclose(normalize(np.array([0.0, 0.0, 2.0])), [0.0, 0.0, 1.0])
    tiny = np.array([1e-12, 0.0, 0.0])
    assert np.allclose(normalize(tiny), tiny)  # near-zero vectors returned unchanged


def test_rotation_vector_error_matches_quaternion_angle():
    target = rpy_to_quat(0.0, 0.0, 0.0)
    actual = rpy_to_quat(0.10, -0.08, 0.06)
    error = quaternion_rotation_vector_error(target, actual)
    assert np.linalg.norm(error) == pytest.approx(
        quaternion_distance_rad(target, actual), abs=1e-9
    )
    assert np.max(np.abs(error)) < np.linalg.norm(error)


def test_stationary_target_camera_offset_multiview_recovery():
    target = np.array([0.35, -0.08, -0.15])
    expected_offset = np.array([0.012, -0.007, 0.018])
    rotations = np.stack([
        quat_to_matrix(rpy_to_quat(0.0, 0.0, 0.0)),
        quat_to_matrix(rpy_to_quat(0.20, 0.0, 0.0)),
        quat_to_matrix(rpy_to_quat(0.0, -0.25, 0.0)),
        quat_to_matrix(rpy_to_quat(0.0, 0.0, 0.30)),
        quat_to_matrix(rpy_to_quat(-0.15, 0.18, -0.20)),
    ])
    raw = target - np.einsum('nij,j->ni', rotations, expected_offset)
    estimate = estimate_stationary_target_camera_offset(raw, rotations)
    assert estimate is not None
    assert np.allclose(estimate.offset_camera, expected_offset, atol=1e-10)
    assert np.allclose(estimate.target_world, target, atol=1e-10)
    assert estimate.corrected_rms_m < 1e-10
    assert estimate.rotation_span_rad > 0.2


def test_stationary_target_camera_offset_rejects_unobservable_views():
    points = np.array([[0.3, -0.1, -0.15]] * 6)
    rotations = np.repeat(np.eye(3)[None, :, :], 6, axis=0)
    assert estimate_stationary_target_camera_offset(points, rotations) is None
