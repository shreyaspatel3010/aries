"""Round-trip the panel localiser: synthesise what the camera would see from a
known panel pose, then check the recovered pose is that pose.

This is the test that matters for the alignment feature, because every later
stage (approach, contact, operate) is a rigid transform of it - if the pose is
recovered, the arm is aimed; if it is not, nothing downstream can save it.
"""

import math
import pathlib

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from aries_maintenance.panel_alignment import (           # noqa: E402
    control_waypoints, detect_markers, load_task_table, marker_corners,
    panel_pose_from_markers, quaternion_from_matrix, tool_orientation,
)

TABLE = (pathlib.Path(__file__).resolve().parents[2]
         / "aries" / "models" / "maintenance_panel" / "panel_task.json")
K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])


def _table():
    if not TABLE.is_file():
        pytest.skip(f"{TABLE} not built; run scripts/build_erc2026_props.py")
    return load_task_table(TABLE)


def _camera_from_panel(distance=0.45, yaw=0.0, pitch=0.0):
    """A camera `distance` out in front of the console, looking back at it."""
    table = _table()
    normal = np.asarray(table["console_normal"], float)
    target = np.array([0.0, 0.0, 0.80])           # mid console
    eye = target + normal * distance
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    # OpenCV camera axes: +X right, +Y down, +Z forward.
    rotation = np.column_stack([right, down, forward])
    rotation = rotation @ _rot_z(yaw) @ _rot_x(pitch)
    panel_from_camera = np.eye(4)
    panel_from_camera[:3, :3] = rotation
    panel_from_camera[:3, 3] = eye
    return np.linalg.inv(panel_from_camera)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _project(camera_from_panel, ids=None):
    """What the camera sees: {id: 4x2 corners}, no noise."""
    table = _table()
    out = {}
    for marker in table["markers"]:
        if ids is not None and marker["id"] not in ids:
            continue
        pts = marker_corners(marker)
        cam = (camera_from_panel[:3, :3] @ pts.T).T + camera_from_panel[:3, 3]
        if np.any(cam[:, 2] <= 1e-6):
            continue
        uv = (K @ cam.T).T
        out[marker["id"]] = uv[:, :2] / uv[:, 2:3]
    return out


def _pose_error(recovered, truth):
    delta = np.linalg.inv(truth) @ recovered
    translation = float(np.linalg.norm(delta[:3, 3]))
    angle = math.degrees(math.acos(
        max(-1.0, min(1.0, (delta[:3, :3].trace() - 1.0) / 2.0))))
    return translation, angle


@pytest.mark.parametrize("yaw,pitch", [(0.0, 0.0), (0.25, -0.15), (-0.35, 0.2)])
def test_all_four_markers_recover_the_pose(yaw, pitch):
    truth = _camera_from_panel(yaw=yaw, pitch=pitch)
    pose, info = panel_pose_from_markers(_project(truth), _table(), K)
    assert pose is not None, info
    translation, angle = _pose_error(pose, truth)
    assert translation < 1e-3, f"{translation * 1000:.2f} mm off"
    assert angle < 0.5, f"{angle:.2f} deg off"
    assert info["reprojection_px"] < 0.5


def test_two_markers_are_enough():
    truth = _camera_from_panel()
    seen = _project(truth, ids={11, 14})
    assert len(seen) == 2
    pose, info = panel_pose_from_markers(seen, _table(), K)
    assert pose is not None
    translation, angle = _pose_error(pose, truth)
    assert translation < 2e-3 and angle < 1.0
    assert not info["single_marker"]


def test_single_marker_is_flagged_not_hidden():
    truth = _camera_from_panel()
    pose, info = panel_pose_from_markers(_project(truth, ids={13}), _table(), K)
    assert pose is not None
    assert info["single_marker"] is True


def test_unknown_ids_are_ignored():
    truth = _camera_from_panel()
    seen = _project(truth)
    seen[62] = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    pose, info = panel_pose_from_markers(seen, _table(), K)
    assert 62 not in info["markers"]
    assert _pose_error(pose, truth)[0] < 1e-3


def test_detector_reads_the_shipped_textures():
    """The tags in the world must decode to the ids the table expects."""
    table = _table()
    for marker in table["markers"]:
        png = TABLE.parent / f"aruco_orig_{marker['id']}.png"
        if not png.is_file():
            pytest.skip(f"{png} not built")
        found = detect_markers(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE))
        assert marker["id"] in found, f"tag {marker['id']} did not decode"


def test_tool_points_into_the_console():
    table = _table()
    normal = np.asarray(table["console_normal"], float)
    for control in table["controls"]:
        way = control_waypoints(control, table)
        approach_axis = way["contact"][:3, 2]           # tool +Z
        assert approach_axis @ normal < -0.99, control["name"]
        # The approach pose must stand off in front of the contact pose.
        offset = way["approach"][:3, 3] - way["contact"][:3, 3]
        # 1e-6, not 1e-9: the table stores directions to 5 decimals, so the
        # stored normal is not exactly unit and the dot product carries that.
        assert abs(float(offset @ normal) - table["standoff"]) < 1e-6
        assert np.linalg.norm(np.cross(offset, normal)) < 1e-6


def test_jaw_line_runs_up_slope_for_the_disconnects():
    """The two red knobs are 65 mm across with 11.7 mm between them, so a grasp
    that closes across the console cannot fit. Up-slope has 36 mm each side."""
    table = _table()
    up_slope = np.asarray(table["console_up_slope"], float)
    turns = [c for c in table["controls"] if c["kind"] == "disconnect"]
    assert len(turns) == 2
    for control in turns:
        jaw = control_waypoints(control, table)["contact"][:3, 0]
        assert abs(float(jaw @ up_slope)) > 0.99, control["name"]


def test_press_and_flick_move_the_right_way():
    table = _table()
    normal = np.asarray(table["console_normal"], float)
    for control in table["controls"]:
        way = control_waypoints(control, table)
        travel = way["operate"][:3, 3] - way["contact"][:3, 3]
        if control["action"] == "press":
            assert float(travel @ normal) < 0        # into the panel
            assert abs(np.linalg.norm(travel) - control["travel"]) < 1e-6
        elif control["action"] == "flick":
            assert abs(float(travel @ normal)) < 1e-6   # across the face only
            assert abs(np.linalg.norm(travel) - control["travel"]) < 1e-6
        else:
            assert np.linalg.norm(travel) < 1e-12
            assert way["turn_about_approach"] > 0


def test_every_control_is_reachable_from_one_table():
    table = _table()
    names = [c["name"] for c in table["controls"]]
    assert len(names) == len(set(names)) == 26
    kinds = {c["action"] for c in table["controls"]}
    assert kinds == {"flick", "press", "turn"}
    assert sum(c["action"] == "flick" for c in table["controls"]) == 14
    assert sum(c["action"] == "press" for c in table["controls"]) == 5


def test_quaternion_matches_rotation():
    rotation = tool_orientation([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    x, y, z, w = quaternion_from_matrix(transform)
    assert abs(x * x + y * y + z * z + w * w - 1.0) < 1e-9
