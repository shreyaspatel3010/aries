"""Locate the ERC maintenance panel from its ArUco tags and aim the gripper.

Two jobs, both pure geometry so they can be tested without a camera or a sim:

1. `panel_pose_from_markers` turns whatever subset of tags 11/13/14/15 the
   gripper camera can see into the panel's pose.
2. `control_waypoints` turns a control out of `panel_task.json` into the poses
   the arm actually has to reach.

WHY ONE PnP OVER ALL TAGS, NOT ONE PER TAG AND AVERAGE
Each 50 mm tag on its own is a tiny, nearly-planar target: its out-of-plane
rotation is poorly conditioned and it flips between two near-equal solutions
(the classic planar-PnP ambiguity), so per-tag poses disagree by tens of
degrees and averaging them averages the ambiguity in. Feeding every visible
corner into a single solvePnP instead makes the tags one rigid 260 x 380 mm
target, which is well conditioned as soon as two of them are in frame. With
only one tag visible the ambiguity is real and unavoidable, so that case is
reported with `single_marker=True` rather than pretended away.

THE TAGS ARE NOT COPLANAR
The console is a stack of sub-plates and the four tags sit at depths spanning
31 mm. That is a feature here - it breaks the planar degeneracy - but it means
the model points must come from `panel_task.json`, which carries each tag's own
measured depth, and not from a flat 260 x 380 rectangle.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

# ORIGINAL library, as the ERC update report specifies for the panel.
ARUCO_DICT_NAME = "DICT_ARUCO_ORIGINAL"


def load_task_table(path):
    """Read `panel_task.json` as emitted by scripts/build_erc2026_props.py."""
    table = json.loads(pathlib.Path(path).read_text())
    for key in ("markers", "controls", "console_normal", "standoff"):
        if key not in table:
            raise ValueError(f"{path}: not a panel task table, missing {key!r}")
    return table


def _basis(pitch):
    """Console frame: normal out of the face, up-slope, across."""
    normal = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
    across = np.array([0.0, 1.0, 0.0])
    up_slope = np.cross(normal, across)
    return normal, up_slope / np.linalg.norm(up_slope), across


def marker_corners(marker):
    """The tag's four corners in panel coordinates, in OpenCV's order.

    cv2.aruco returns corners top-left, top-right, bottom-right, bottom-left as
    seen on the printed tag. The question is which way that tag is printed on
    the console, and it is NOT the intuitive one: the texture lands rotated
    180 deg, so the tag's "up" runs DOWN-slope and its "right" runs -across.

    Measured, not assumed. Solving the same live frame against all four
    rotations of the corner ring:

        rot0    49.36 px      rot180    1.72 px
        rot90   36.07 px      rot270   97.17 px

    Get this wrong and nothing downstream complains: solvePnP happily returns a
    pose that is rotated by a multiple of 90 deg, and only the reprojection
    error gives it away. That is why `panel_pose_from_markers` reports the error
    and the node gates on it.

    Handedness is settled separately: reversing the ring (looking at the tag
    from behind) fits the same 1.72 px, because two coplanar tags carry the
    planar mirror ambiguity. It cannot be the right answer though - a mirrored
    ArUco pattern is not a valid code and would not have decoded at all.
    """
    normal, up_slope, across = _basis(marker["pitch"])
    centre = np.asarray(marker["position"], float)
    half = marker["size"] / 2.0
    return np.array([
        centre + half * across - half * up_slope,
        centre - half * across - half * up_slope,
        centre - half * across + half * up_slope,
        centre + half * across + half * up_slope,
    ])


def detect_markers(gray, dictionary=None, parameters=None):
    """Detect panel tags. Returns {id: 4x2 image corners}."""
    import cv2

    if dictionary is None:
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, ARUCO_DICT_NAME))
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary, parameters or cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64)
            for i, c in zip(ids.flatten(), corners)}


def panel_pose_from_markers(detections, table, camera_matrix, dist_coeffs=None,
                            min_markers=1):
    """Panel pose in the camera frame as a 4x4, or None.

    Returns `(transform, info)` where info carries the reprojection error and
    whether the fit rested on a single tag.
    """
    import cv2

    known = {m["id"]: m for m in table["markers"]}
    object_points, image_points, used = [], [], []
    for marker_id, image_corners in sorted(detections.items()):
        if marker_id not in known:
            continue        # a landmark or drone tag wandering through frame
        object_points.append(marker_corners(known[marker_id]))
        image_points.append(image_corners)
        used.append(marker_id)
    if len(used) < max(1, min_markers):
        return None, dict(markers=used, reason="too few panel tags")

    object_points = np.concatenate(object_points).astype(np.float64)
    image_points = np.concatenate(image_points).astype(np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, np.asarray(camera_matrix, float).reshape(3, 3),
        np.asarray(dist_coeffs, float).ravel(),
        flags=cv2.SOLVEPNP_ITERATIVE if len(used) > 1 else cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, dict(markers=used, reason="solvePnP failed")

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, np.asarray(camera_matrix, float).reshape(3, 3),
        np.asarray(dist_coeffs, float).ravel())
    error = float(np.linalg.norm(
        projected.reshape(-1, 2) - image_points, axis=1).mean())

    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    return transform, dict(markers=used, reprojection_px=error,
                           single_marker=len(used) == 1)


def refine_panel_translation_from_depth(
        camera_from_panel, detections, table, camera_matrix, depth_image,
        dist_coeffs=None, min_depth=0.10, max_depth=5.0,
        expected_band=0.15, minimum_pixels=6):
    """Constrain PnP translation with registered depth inside ArUco tags.

    ArUco supplies the full orientation. For each tag with valid aligned depth,
    its known panel-frame centre and measured camera-frame 3D centre provide an
    independent translation estimate ``observed - R * known``. Their median is
    robust to a few invalid/background pixels and removes most single-tag range
    jitter without requiring both cameras to see the same marker.
    """
    import cv2

    pose = np.asarray(camera_from_panel, float)
    depth = np.asarray(depth_image, float)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be 2D, got {depth.shape}")
    k = np.asarray(camera_matrix, float).reshape(3, 3)
    distortion = (np.zeros(5) if dist_coeffs is None
                  else np.asarray(dist_coeffs, float).ravel())
    known = {marker["id"]: marker for marker in table["markers"]}
    estimates = []
    residuals = []
    used = []
    height, width = depth.shape

    for marker_id, corners in detections.items():
        marker = known.get(marker_id)
        if marker is None:
            continue
        corners = np.asarray(corners, float).reshape(4, 2)
        centre_uv = corners.mean(axis=0)
        # Keep away from tag edges, where registration rounding mixes the
        # panel/background with the marker plane.
        inner = centre_uv + 0.55 * (corners - centre_uv)
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.rint(inner).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 1)
        values = depth[(mask != 0) & np.isfinite(depth) &
                       (depth > float(min_depth)) &
                       (depth < float(max_depth))]
        marker_position = np.asarray(marker["position"], float)
        predicted = pose[:3, :3] @ marker_position + pose[:3, 3]
        if values.size:
            in_band = values[np.abs(values - predicted[2]) <= expected_band]
            if in_band.size >= minimum_pixels:
                values = in_band
        if values.size < minimum_pixels:
            continue
        measured_depth = float(np.median(values))
        undistorted = cv2.undistortPoints(
            centre_uv.reshape(1, 1, 2), k, distortion, P=k).reshape(2)
        observed = np.array([
            (undistorted[0] - k[0, 2]) * measured_depth / k[0, 0],
            (undistorted[1] - k[1, 2]) * measured_depth / k[1, 1],
            measured_depth,
        ])
        estimates.append(observed - pose[:3, :3] @ marker_position)
        residuals.append(float(np.linalg.norm(observed - predicted)))
        used.append(marker_id)

    if not estimates:
        return pose.copy(), dict(depth_markers=[], depth_correction_m=0.0,
                                 depth_residual_m=None)
    refined = pose.copy()
    translation = np.median(np.asarray(estimates), axis=0)
    refined[:3, 3] = translation
    return refined, dict(
        depth_markers=sorted(used),
        depth_correction_m=float(np.linalg.norm(translation - pose[:3, 3])),
        depth_residual_m=float(np.median(residuals)))


def tool_orientation(approach, jaw_axis):
    """Rotation whose +Z is the approach and whose +X is the jaw line.

    Tool +Z is the approach direction because that is the axis the rest of this
    stack drives the final descent along; see the tool-frame approach note in
    the grasp node. `jaw_axis` is orthogonalised against it rather than trusted,
    so a table entry only has to be roughly right.
    """
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(jaw_axis, float)
    x = x - z * float(x @ z)
    if np.linalg.norm(x) < 1e-6:
        # Degenerate: the requested jaw line is along the approach. Any
        # perpendicular will do, so take the most stable one.
        x = np.cross(z, [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def control_waypoints(control, table, standoff=None, tool_contact_offset=0.0):
    """Approach, contact and operate poses for one control, in panel frame.

    - `approach` sits `standoff` out along the console normal, clear of the
      tallest thing on the console.
    - `contact` is on the control itself.
    - `operate` is where the tool ends up: for a press or a flick that is a
      translation, for a turn it is the same point and the wrist rolls about
      the approach axis by `turn` radians.

    The tool always points INTO the console, so its +Z is the negated normal.
    """
    approach_dir = np.asarray(control["approach"], float)
    approach_dir = approach_dir / np.linalg.norm(approach_dir)
    position = np.asarray(control["position"], float)
    standoff = table["standoff"] if standoff is None else standoff

    rotation = tool_orientation(-approach_dir, control["jaw_axis"])

    def pose(point):
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = point
        return transform

    # ``position`` is the physical switch surface. The IK link is gripper_tcp,
    # but the maintenance fingertips project beyond that virtual frame (65 mm
    # for the v2 tool). Pull the TCP outward so TCP + tool-Z*offset lands the
    # actual fingertip on the switch rather than driving it through the panel.
    contact = position + approach_dir * float(tool_contact_offset)
    approach = contact + approach_dir * standoff
    turn = 0.0
    if control["action"] == "turn":
        operate = contact
        turn = float(control["travel"])
    elif control["action"] == "press":
        operate = contact - approach_dir * float(control["travel"])
    else:                                   # flick an enabled MCB upward
        # A YAML ``true`` means operate/set the breaker upward. Do not inherit
        # the sign of jaw_axis here: that axis describes an undirected jaw line
        # and reversing it is mechanically equivalent for a grasp, whereas an
        # MCB's switching direction is not equivalent.
        upward = np.asarray(table["console_up_slope"], float)
        upward = upward / np.linalg.norm(upward)
        operate = contact + upward * float(control["travel"])
    return dict(approach=pose(approach), contact=pose(contact),
                operate=pose(operate), turn_about_approach=turn,
                grip=bool(control["grip"]), action=control["action"],
                name=control["name"], joint=control["joint"])


def upward_flick_in_planning_frame(contact_pose, travel,
                                   planning_up=(0.0, 0.0, 1.0)):
    """Return an MCB endpoint that is unambiguously upward in robot space.

    Marker texture orientation and a panel-frame convention must never decide
    whether an ON command moves up or down. Project planning-frame +Z onto the
    panel face recovered from the tool's contact orientation. The resulting
    motion remains tangent to the console and always has a positive +Z part.
    """
    contact = np.asarray(contact_pose, float)
    result = contact.copy()
    # Tool +Z points into the panel at contact; negate it for the outward normal.
    normal = -contact[:3, 2]
    normal /= np.linalg.norm(normal)
    up = np.asarray(planning_up, float)
    up = up - normal * float(up @ normal)
    length = np.linalg.norm(up)
    if length < 1e-6:
        raise ValueError("planning-frame up is parallel to the panel normal")
    up /= length
    if float(up @ np.asarray(planning_up, float)) <= 0.0:
        raise ValueError("projected MCB direction is not upward")
    result[:3, 3] += up * float(travel)
    return result


def transform_pose(parent_from_panel, pose):
    """Move a panel-frame 4x4 into the parent frame."""
    return np.asarray(parent_from_panel, float) @ np.asarray(pose, float)


def roll_about_tool_z(transform, angle):
    """Return the same tool pose rolled about its own approach axis.

    The maintenance jaws and fingers are symmetric about their centre line.
    A 180 degree local-Z roll therefore presents the same jaw line to a switch,
    but places a bounded robot wrist joint in the other IK family.  Position
    and tool +Z are deliberately invariant.
    """
    transform = np.asarray(transform, float)
    result = transform.copy()
    c, s = math.cos(float(angle)), math.sin(float(angle))
    local_roll = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    result[:3, :3] = transform[:3, :3] @ local_roll
    return result


def transform_distance(a, b):
    """Return translation metres and rotation radians between two transforms."""
    delta = np.linalg.inv(np.asarray(a, float)) @ np.asarray(b, float)
    cosine = max(-1.0, min(1.0, (float(np.trace(delta[:3, :3])) - 1.0) / 2.0))
    return float(np.linalg.norm(delta[:3, 3])), math.acos(cosine)


def average_transforms(transforms, weights=None):
    """Weighted rigid-transform mean, including a sign-safe quaternion mean."""
    transforms = [np.asarray(t, float) for t in transforms]
    if not transforms:
        raise ValueError("at least one transform is required")
    weights = (np.ones(len(transforms), float) if weights is None
               else np.asarray(weights, float))
    if weights.shape != (len(transforms),) or np.any(weights < 0) or not weights.sum():
        raise ValueError("weights must be non-negative and match transforms")
    weights = weights / weights.sum()

    result = np.eye(4)
    result[:3, 3] = sum(w * t[:3, 3] for w, t in zip(weights, transforms))

    # Markley's quaternion average uses q*q^T, so q and -q contribute exactly
    # the same rotation and cannot cancel each other.
    accumulator = np.zeros((4, 4))
    for weight, transform in zip(weights, transforms):
        q = np.asarray(quaternion_from_matrix(transform), float)
        accumulator += weight * np.outer(q, q)
    q = np.linalg.eigh(accumulator)[1][:, -1]
    x, y, z, w = q
    result[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)]])
    return result


def transform_consensus(transforms, weights=None):
    """Return mean pose and maximum translation/rotation deviation from it."""
    mean = average_transforms(transforms, weights)
    deviations = [transform_distance(mean, pose) for pose in transforms]
    return (mean, max(distance for distance, _ in deviations),
            max(angle for _, angle in deviations))


def quaternion_from_matrix(transform):
    """(x, y, z, w) from a 4x4, without pulling in a transforms library."""
    m = np.asarray(transform, float)[:3, :3]
    trace = m.trace()
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, m[i, i] - m[j, j] - m[k, k] + 1.0)) * 2
    q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
    q[i], q[j], q[k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return tuple(q)
