"""Pure geometry/time helpers shared by the vision-grasp nodes."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion


@dataclass(frozen=True)
class CameraOffsetEstimate:
    """Observable stationary-target camera-offset least-squares result."""

    offset_camera: np.ndarray
    target_world: np.ndarray
    raw_rms_m: float
    corrected_rms_m: float
    condition_number: float
    rotation_span_rad: float


def quat_to_matrix(q: Quaternion) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def matrix_to_quat(R: np.ndarray) -> Quaternion:
    """Convert a 3×3 rotation matrix to a ROS Quaternion (Shepperd method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    q = Quaternion()
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q.w = 0.25 / s
        q.x = (R[2, 1] - R[1, 2]) * s
        q.y = (R[0, 2] - R[2, 0]) * s
        q.z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q.w = (R[2, 1] - R[1, 2]) / s
        q.x = 0.25 * s
        q.y = (R[0, 1] + R[1, 0]) / s
        q.z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q.w = (R[0, 2] - R[2, 0]) / s
        q.x = (R[0, 1] + R[1, 0]) / s
        q.y = 0.25 * s
        q.z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q.w = (R[1, 0] - R[0, 1]) / s
        q.x = (R[0, 2] + R[2, 0]) / s
        q.y = (R[1, 2] + R[2, 1]) / s
        q.z = 0.25 * s
    return q


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-9 else v / n


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def wrist_extension_shortfall_m(
    link_xyz: np.ndarray,
    link_orientation: Quaternion,
    shoulder_xyz: np.ndarray,
    max_wrist_extension_m: float,
    wrist_backoff_in_link_m: float,
    margin_m: float,
) -> float:
    """How far (m) a tool-link pose lies beyond the arm's reach envelope.

    Shoulder-sphere model: the wrist centre sits wrist_backoff_in_link_m
    behind the tool link along its +Z axis and can extend at most
    max_wrist_extension_m from shoulder_xyz. Positive result = out of reach
    by that much (including margin_m); <= 0 = reachable.
    """
    backoff = np.array([0.0, 0.0, wrist_backoff_in_link_m], dtype=np.float64)
    wrist_xyz = np.asarray(link_xyz, dtype=np.float64) - quat_to_matrix(link_orientation) @ backoff
    d = float(np.linalg.norm(wrist_xyz - np.asarray(shoulder_xyz, dtype=np.float64)))
    return d + margin_m - max_wrist_extension_m


def duration_to_sec(duration: Duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def sec_to_duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1000000000:
        sec += 1
        nanosec -= 1000000000
    return Duration(sec=sec, nanosec=nanosec)


def robot_trajectory_duration_sec(robot_trajectory) -> float:
    joint_traj = robot_trajectory.joint_trajectory
    if not joint_traj.points:
        return 0.0
    return duration_to_sec(joint_traj.points[-1].time_from_start)


def quaternion_distance_rad(a: Quaternion, b: Quaternion) -> float:
    qa = np.array([float(a.x), float(a.y), float(a.z), float(a.w)], dtype=np.float64)
    qb = np.array([float(b.x), float(b.y), float(b.z), float(b.w)], dtype=np.float64)
    na = float(np.linalg.norm(qa))
    nb = float(np.linalg.norm(qb))
    if na < 1e-9 or nb < 1e-9:
        return math.inf
    dot = abs(float(np.dot(qa / na, qb / nb)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def quaternion_rotation_vector_error(
    target: Quaternion,
    actual: Quaternion,
) -> np.ndarray:
    """Return target->actual error as a shortest-path rotation vector.

    MoveIt's ``OrientationConstraint.ROTATION_VECTOR`` checks the absolute
    value of these three components independently. Using this helper for arm
    feedback therefore avoids accepting a goal in MoveIt and subsequently
    rejecting the exact same orientation with a different metric.
    """
    qt = np.array(
        [float(target.x), float(target.y), float(target.z), float(target.w)],
        dtype=np.float64,
    )
    qa = np.array(
        [float(actual.x), float(actual.y), float(actual.z), float(actual.w)],
        dtype=np.float64,
    )
    nt = float(np.linalg.norm(qt))
    na = float(np.linalg.norm(qa))
    if nt < 1e-12 or na < 1e-12:
        return np.full(3, math.inf, dtype=np.float64)
    qt /= nt
    qa /= na

    # q_error = conjugate(q_target) * q_actual, ROS xyzw ordering.
    tx, ty, tz, tw = qt
    ax, ay, az, aw = qa
    vec = np.array([
        tw * ax - tx * aw - ty * az + tz * ay,
        tw * ay + tx * az - ty * aw - tz * ax,
        tw * az - tx * ay + ty * ax - tz * aw,
    ], dtype=np.float64)
    scalar = tw * aw + tx * ax + ty * ay + tz * az
    if scalar < 0.0:
        vec = -vec
        scalar = -scalar
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, float(np.clip(scalar, 0.0, 1.0)))
    return vec * (angle / sin_half)


def estimate_stationary_target_camera_offset(
    raw_points_world: np.ndarray,
    rotations_world_camera: np.ndarray,
) -> Optional[CameraOffsetEstimate]:
    """Estimate a constant camera-axis offset from multi-view observations.

    For a stationary target, each raw observation satisfies
    ``target_world = raw_world_i + R_world_camera_i @ offset_camera``.
    Camera rotation is required: translation-only views cannot distinguish the
    unknown world target from a constant camera-frame offset, so rank-deficient
    data returns ``None`` instead of producing a dangerous calibration.
    """
    points = np.asarray(raw_points_world, dtype=np.float64)
    rotations = np.asarray(rotations_world_camera, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or rotations.shape != (points.shape[0], 3, 3)
        or points.shape[0] < 4
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(rotations))
    ):
        return None

    mean_point = points.mean(axis=0)
    mean_rotation = rotations.mean(axis=0)
    design = (rotations - mean_rotation).reshape(-1, 3)
    rhs = -(points - mean_point).reshape(-1)
    offset, _, rank, singular = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3 or singular.size < 3 or float(singular[-1]) < 1e-6:
        return None
    condition = float(singular[0] / singular[-1])

    corrected = points + np.einsum('nij,j->ni', rotations, offset)
    target = corrected.mean(axis=0)
    raw_rms = float(np.sqrt(np.mean(np.sum((points - mean_point) ** 2, axis=1))))
    corrected_rms = float(np.sqrt(np.mean(np.sum((corrected - target) ** 2, axis=1))))

    rotation_span = 0.0
    for i in range(rotations.shape[0]):
        for j in range(i + 1, rotations.shape[0]):
            relative = rotations[i].T @ rotations[j]
            cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
            rotation_span = max(rotation_span, math.acos(cosine))

    return CameraOffsetEstimate(
        offset_camera=np.asarray(offset, dtype=np.float64),
        target_world=np.asarray(target, dtype=np.float64),
        raw_rms_m=raw_rms,
        corrected_rms_m=corrected_rms,
        condition_number=condition,
        rotation_span_rad=float(rotation_span),
    )
