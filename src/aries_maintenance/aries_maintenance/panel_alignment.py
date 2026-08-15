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


def control_waypoints(control, table, standoff=None):
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

    contact = position
    approach = position + approach_dir * standoff
    turn = 0.0
    if control["action"] == "turn":
        operate = contact
        turn = float(control["travel"])
    elif control["action"] == "press":
        operate = contact - approach_dir * float(control["travel"])
    else:                                   # flick
        jaw = np.asarray(control["jaw_axis"], float)
        operate = contact + jaw / np.linalg.norm(jaw) * float(control["travel"])
    return dict(approach=pose(approach), contact=pose(contact),
                operate=pose(operate), turn_about_approach=turn,
                grip=bool(control["grip"]), action=control["action"],
                name=control["name"], joint=control["joint"])


def transform_pose(parent_from_panel, pose):
    """Move a panel-frame 4x4 into the parent frame."""
    return np.asarray(parent_from_panel, float) @ np.asarray(pose, float)


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
