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
import yaml

cv2 = pytest.importorskip("cv2")

from aries_maintenance.panel_alignment import (           # noqa: E402
    average_transforms, control_waypoints, detect_markers,
    flick_endpoint_in_planning_frame, load_task_table,
    marker_corners, panel_pose_from_markers, quaternion_from_matrix,
    refine_panel_pose_from_depth, refine_panel_translation_from_depth,
    roll_about_tool_z, tool_orientation,
    transform_consensus, transform_distance, transform_inlier_consensus,
)

TABLE = (pathlib.Path(__file__).resolve().parents[2]
         / "aries" / "models" / "maintenance_panel" / "panel_task.json")
TASK_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "config" / "panel_tasks.yaml"
K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])


def _table():
    if not TABLE.is_file():
        pytest.skip(f"{TABLE} not built; run scripts/build_maintenance_panel.py")
    return load_task_table(TABLE)


def _camera_from_panel(distance=0.45, yaw=0.0, pitch=0.0, target=None):
    """A camera `distance` out in front of the console, looking back at it."""
    table = _table()
    normal = np.asarray(table["console_normal"], float)
    # Aim at the centroid of the marker triangle rather than a hard-coded
    # height: that point is on the control face for any panel build, where
    # z = 0.80 was only mid-face on the taller CAD-derived console.
    target = (np.mean([np.asarray(m["position"], float)
                       for m in table["markers"]], axis=0)
              if target is None else np.asarray(target, float))
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
def test_all_installed_markers_recover_the_pose(yaw, pitch):
    truth = _camera_from_panel(yaw=yaw, pitch=pitch)
    pose, info = panel_pose_from_markers(_project(truth), _table(), K)
    assert pose is not None, info
    translation, angle = _pose_error(pose, truth)
    assert translation < 1e-3, f"{translation * 1000:.2f} mm off"
    assert angle < 0.5, f"{angle:.2f} deg off"
    assert info["reprojection_px"] < 0.5


def test_two_markers_are_enough():
    truth = _camera_from_panel()
    ids = {marker["id"] for marker in _table()["markers"][:2]}
    seen = _project(truth, ids=ids)
    assert len(seen) == 2
    pose, info = panel_pose_from_markers(seen, _table(), K)
    assert pose is not None
    translation, angle = _pose_error(pose, truth)
    assert translation < 2e-3 and angle < 1.0
    assert not info["single_marker"]


def test_single_marker_is_flagged_not_hidden():
    truth = _camera_from_panel()
    marker_id = _table()["markers"][0]["id"]
    pose, info = panel_pose_from_markers(
        _project(truth, ids={marker_id}), _table(), K)
    assert pose is not None
    assert info["single_marker"] is True


def test_registered_depth_corrects_aruco_translation():
    truth = _camera_from_panel()
    shift = np.array([200.0, 300.0])
    detections = {marker_id: corners + shift
                  for marker_id, corners in
                  _project(truth).items()}
    shifted_k = K.copy()
    shifted_k[0, 2] += shift[0]
    shifted_k[1, 2] += shift[1]
    initial = truth.copy()
    initial[:3, 3] += [0.025, -0.010, 0.040]
    depth = np.zeros((1000, 1000), dtype=np.float32)
    known = {m["id"]: m for m in _table()["markers"]}
    for marker_id, corners in detections.items():
        centre = np.rint(corners.mean(axis=0)).astype(int)
        point = (truth[:3, :3] @ np.asarray(known[marker_id]["position"]) +
                 truth[:3, 3])
        depth[centre[1] - 8:centre[1] + 9,
              centre[0] - 8:centre[0] + 9] = point[2]
    refined, info = refine_panel_translation_from_depth(
        initial, detections, _table(), shifted_k, depth)
    assert len(info["depth_markers"]) == 3
    assert np.linalg.norm(refined[:3, 3] - truth[:3, 3]) < 0.003
    assert info["depth_correction_m"] > 0.03


@pytest.mark.parametrize("marker_index", [0, 1, 2])
def test_single_marker_registered_depth_cloud_recovers_full_pose(marker_index):
    marker = _table()["markers"][marker_index]
    truth = _camera_from_panel(
        yaw=0.12, pitch=-0.08, target=marker["position"])
    detections = _project(truth, ids={marker["id"]})
    corners = detections[marker["id"]]
    depth = np.zeros((480, 640), dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1)
    rows, cols = np.nonzero(mask)
    panel_normal = np.asarray(_table()["console_normal"], float)
    camera_normal = truth[:3, :3] @ panel_normal
    camera_point = (truth[:3, :3] @ np.asarray(marker["position"], float) +
                    truth[:3, 3])
    rays = np.column_stack([
        (cols - K[0, 2]) / K[0, 0],
        (rows - K[1, 2]) / K[1, 1],
        np.ones(rows.size)])
    ranges = (camera_normal @ camera_point) / (rays @ camera_normal)
    depth[rows, cols] = ranges.astype(np.float32)

    initial = truth.copy()
    initial[:3, :3] = _rot_x(math.radians(7.0)) @ initial[:3, :3]
    initial[:3, 3] += [0.020, -0.012, 0.035]
    refined, info = refine_panel_pose_from_depth(
        initial, detections, _table(), K, depth)
    translation, angle = _pose_error(refined, truth)
    assert info["depth_pose_markers"] == [marker["id"]]
    assert info["depth_points"] >= 24
    assert translation < 0.004
    assert angle < 0.75


def test_multi_marker_depth_keeps_wide_baseline_rgb_rotation():
    truth = _camera_from_panel(yaw=0.12, pitch=-0.08)
    markers = _table()["markers"][:2]
    detections = _project(truth, ids={marker["id"] for marker in markers})
    depth = np.zeros((480, 640), dtype=np.float32)
    # Deliberately use a fronto-parallel local depth patch for each tag. This
    # mimics quantised/noisy depth normals at range; the two-marker RGB
    # rotation must win while the measured centres still correct translation.
    for marker in markers:
        corners = detections[marker["id"]]
        mask = np.zeros(depth.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1)
        centre = (truth[:3, :3] @ np.asarray(marker["position"], float) +
                  truth[:3, 3])
        depth[mask != 0] = centre[2]

    initial = truth.copy()
    initial[:3, 3] += [0.018, -0.009, 0.030]
    refined, info = refine_panel_pose_from_depth(
        initial, detections, _table(), K, depth)
    translation, angle = _pose_error(refined, truth)
    assert info["depth_pose_markers"] == []
    assert info["depth_markers"] == sorted(marker["id"] for marker in markers)
    assert translation < 0.008
    assert angle < 0.01


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


def test_maintenance_fingertip_offset_keeps_tcp_outside_panel():
    table = _table()
    offset = 0.065
    for control in table["controls"]:
        way = control_waypoints(control, table, tool_contact_offset=offset)
        tcp = way["contact"][:3, 3]
        tool_z = way["contact"][:3, 2]
        physical_contact = tcp + tool_z * offset
        assert np.allclose(physical_contact, control["position"], atol=1e-9)
        separation = way["approach"][:3, 3] - tcp
        assert abs(np.linalg.norm(separation) - table["standoff"]) < 1e-9


def test_closed_finger_push_uses_leading_surface_not_jaw_meeting_point():
    table = _table()
    closed_tip_offset = 0.084
    for control in table["controls"]:
        if control["action"] not in ("flick", "press"):
            continue
        way = control_waypoints(
            control, table, tool_contact_offset=closed_tip_offset)
        physical_leading_surface = (
            way["contact"][:3, 3] +
            way["contact"][:3, 2] * closed_tip_offset)
        assert np.allclose(
            physical_leading_surface, control["position"], atol=1e-9)


def test_mcb_endpoint_follows_the_on_direction_and_is_tangent_to_panel():
    normal = np.array([0.55, 0.10, 0.83])
    normal /= np.linalg.norm(normal)
    on = np.cross(np.cross(normal, [0.0, 0.0, 1.0]), normal)
    on /= np.linalg.norm(on)
    contact = np.eye(4)
    contact[:3, :3] = tool_orientation(-normal, [0.0, 1.0, 0.0])
    operate = flick_endpoint_in_planning_frame(contact, 0.012, on)
    travel = operate[:3, 3] - contact[:3, 3]
    assert float(travel @ normal) == pytest.approx(0.0, abs=1e-9)
    assert float(travel @ on) == pytest.approx(0.012)
    assert np.linalg.norm(travel) == pytest.approx(0.012)


def _panel_as_the_worlds_mount_it(pitch=-1.57, position=(1.35, 0.0, 0.3)):
    """The pose both worlds spawn the console with: face out, but rolled, so
    the drawing's up-slope points below horizontal."""
    transform = np.eye(4)
    transform[:3, :3] = np.array([[math.cos(pitch), 0.0, math.sin(pitch)],
                                  [0.0, 1.0, 0.0],
                                  [-math.sin(pitch), 0.0, math.cos(pitch)]])
    transform[:3, 3] = position
    return transform


def test_every_mcb_lever_travels_upward_in_the_world_to_reach_on():
    """The one property the user cares about: lever up is ON, lever down is OFF.

    Regression for two ways of getting it wrong on this mount, where the face is
    rolled so the drawing's up-slope points 57 deg BELOW horizontal: deriving
    the stroke from planning-frame +Z (what the code first did) and deriving it
    from ``console_up_slope`` both reverse here. The model states the direction
    and this checks the result in the world, which is where the user sees it."""
    table = _table()
    base_from_panel = _panel_as_the_worlds_mount_it()
    up_slope = base_from_panel[:3, :3] @ np.asarray(
        table["console_up_slope"], float)
    assert up_slope[2] < 0.0, "this fixture is meant to be the rolled mount"

    breakers = [c for c in table["controls"] if c["action"] == "flick"]
    assert len(breakers) == 14
    for control in breakers:
        on = base_from_panel[:3, :3] @ np.asarray(control["on_direction"], float)
        contact = base_from_panel @ control_waypoints(control, table)["contact"]
        operate = flick_endpoint_in_planning_frame(
            contact, control["travel"], on)
        stroke = operate[:3, 3] - contact[:3, 3]
        assert np.linalg.norm(stroke) == pytest.approx(control["travel"])
        assert stroke[2] > 0.0, f"{control['name']} lever travels downward"
        stroke /= np.linalg.norm(stroke)
        # 1e-5 rather than exact: the table rounds its direction vectors to
        # five decimals, which is ~0.1 mdeg of tilt against the face.
        assert float(stroke @ on) == pytest.approx(1.0, abs=1e-5), control["name"]
        assert float(stroke @ up_slope) < 0.0, "ON is down-slope on this mount"


def test_mcb_flick_rejects_an_on_direction_left_in_the_panel_frame():
    """Forgetting the panel rotation leaves a direction that is not tangent to
    the face; silently projecting it would aim the stroke somewhere else."""
    table = _table()
    control = next(c for c in table["controls"] if c["action"] == "flick")
    base_from_panel = _panel_as_the_worlds_mount_it()
    contact = base_from_panel @ control_waypoints(control, table)["contact"]
    with pytest.raises(ValueError, match="tangent"):
        flick_endpoint_in_planning_frame(
            contact, control["travel"], control["on_direction"])


def test_contact_targets_are_on_modeled_surfaces_not_joint_pivots():
    """Aiming at the buried joint pivots drove the gripper into the panel."""
    table = _table()
    normal = np.asarray(table["console_normal"], float)
    normal /= np.linalg.norm(normal)
    offsets_by_action = {"flick": [], "turn": [], "press": []}
    for control in table["controls"]:
        position = np.asarray(control["position"], float)
        pivot = np.asarray(control["pivot_position"], float)
        offset = float(control["surface_offset_m"])
        displacement = position - pivot
        assert np.linalg.norm(np.cross(displacement, normal)) < 2e-5
        assert float(displacement @ normal) == pytest.approx(offset, abs=2e-5)
        assert offset > 0.0
        offsets_by_action[control["action"]].append(offset)

    # Values are derived from the shipped control meshes. These broad bounds
    # catch a regression to pivots without coupling the test to mesh rounding.
    assert min(offsets_by_action["flick"]) > 0.008
    assert max(offsets_by_action["flick"]) < 0.010
    # Cam switches stand 25 mm proud of their sub-plate and the disconnect
    # knobs 30 mm, both measured off the organisers' control meshes.
    assert min(offsets_by_action["turn"]) > 0.020
    assert max(offsets_by_action["turn"]) < 0.045
    assert offsets_by_action["press"] == pytest.approx([0.008] * 5)


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


def test_half_turn_tool_roll_is_the_same_mechanical_jaw_pose():
    """The IK fallback changes wrist family, not the grasp geometry."""
    way = control_waypoints(
        next(c for c in _table()["controls"]
             if c["name"] == "rotary_control_switch_0"),
        _table())
    original = way["approach"]
    rolled = roll_about_tool_z(original, math.pi)
    assert np.allclose(rolled[:3, 3], original[:3, 3])
    assert np.allclose(rolled[:3, 2], original[:3, 2])
    assert np.allclose(rolled[:3, 0], -original[:3, 0])
    assert abs(float(rolled[:3, 0] @ original[:3, 0])) > 0.999


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
            # YAML true sets the breaker ON, along the direction the model
            # declares - not up-slope, which on this mount is the OFF end.
            on = np.asarray(control["on_direction"], float)
            on /= np.linalg.norm(on)
            assert float(travel @ on) == pytest.approx(control["travel"])
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


def test_mcbs_are_numbered_left_to_right_and_command_upward_on():
    mcbs = [c for c in _table()["controls"] if c["action"] == "flick"]
    assert [c["name"] for c in mcbs] == [f"mcb_{index}" for index in range(14)]
    assert [c["position"][1] for c in mcbs] == sorted(
        (c["position"][1] for c in mcbs), reverse=True)
    # Every breaker is now its own single-module device, so the links are a
    # plain mcb_0..mcb_13; the old 1mcb_0/1 names came from the CAD split where
    # twelve of the fourteen were poles of ganged four-module blocks.
    assert [c["model_name"] for c in mcbs] == [f"mcb_{i}" for i in range(14)]
    assert [c["joint"] for c in mcbs] == [f"mcb_{i}_joint" for i in range(14)]
    assert len(set(c["model_name"] for c in mcbs)) == 14
    assert all(c["target_state"] == "on" for c in mcbs)
    # The model states the ON direction rather than leaving it to be derived
    # from a frame convention, and on this mount ON is down-slope on the face -
    # which is what puts the lever UP in the world. See the world-frame test.
    assert all(c["motion_direction"] == "down-slope" for c in mcbs)
    down_slope = -np.asarray(_table()["console_up_slope"], float)
    for control in mcbs:
        assert np.allclose(control["on_direction"], down_slope, atol=1e-5)
        assert np.dot(control["on_direction"], control["approach"]) == \
            pytest.approx(0.0, abs=1e-5)


def test_yaml_flags_match_every_control_in_task_order():
    """A stale or misspelled YAML flag must not silently skip a control."""
    configured = yaml.safe_load(TASK_CONFIG.read_text())
    flags = configured["panel_operator"]["ros__parameters"]["controls"]
    names = [c["name"] for c in _table()["controls"]]
    assert list(flags) == names
    assert all(type(enabled) is bool for enabled in flags.values())
    params = configured["panel_operator"]["ros__parameters"]
    assert params["required_camera_count"] == 2
    assert params["allow_single_depth_camera"] is True
    assert params["min_depth_markers"] == 1
    assert params["latch_panel_pose"] is True
    assert params["recalibrate_on_operate_enabled"] is True
    assert params["tool_contact_offset_m"] == pytest.approx(0.065)
    assert params["tool_push_contact_offset_m"] == pytest.approx(0.084)


def test_quaternion_matches_rotation():
    rotation = tool_orientation([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    x, y, z, w = quaternion_from_matrix(transform)
    assert abs(x * x + y * y + z * z + w * w - 1.0) < 1e-9


def test_two_camera_transform_fusion_averages_pose():
    rover = np.eye(4)
    gripper = np.eye(4)
    rover[:3, :3] = _rot_z(-0.1)
    gripper[:3, :3] = _rot_z(0.1)
    rover[:3, 3] = [1.0, -0.02, 0.2]
    gripper[:3, 3] = [1.04, 0.02, 0.2]
    fused = average_transforms([rover, gripper])
    assert np.allclose(fused[:3, 3], [1.02, 0.0, 0.2])
    identity_at_mean = np.eye(4)
    identity_at_mean[:3, 3] = [1.02, 0.0, 0.2]
    translation, rotation = transform_distance(fused, identity_at_mean)
    assert translation < 1e-9
    assert rotation < 1e-6


def test_pose_consensus_reports_sample_spread():
    samples = []
    for dx in (-0.003, 0.0, 0.002):
        pose = np.eye(4)
        pose[:3, 3] = [1.0 + dx, 0.2, 0.4]
        samples.append(pose)
    consensus, translation_spread, rotation_spread = transform_consensus(samples)
    assert consensus[0, 3] == pytest.approx(1.0 - 0.001 / 3.0)
    assert translation_spread < 0.004
    assert rotation_spread < 1e-9


def test_panel_calibration_consensus_rejects_depth_pose_outliers():
    truth = np.eye(4)
    truth[:3, 3] = [1.1, -0.2, 0.65]
    samples = []
    for index in range(18):
        sample = truth.copy()
        sample[:3, 3] += [0.0015 * math.sin(index),
                          0.0010 * math.cos(index),
                          0.0005 * math.sin(2 * index)]
        sample[:3, :3] = _rot_z(math.radians(0.2 * math.sin(index)))
        samples.append(sample)
    for shift, angle in (([0.14, 0.00, 0.00], 8.0),
                         ([-0.09, 0.05, 0.00], -6.0),
                         ([0.04, 0.00, 0.08], 5.0)):
        outlier = truth.copy()
        outlier[:3, 3] += shift
        outlier[:3, :3] = _rot_z(math.radians(angle))
        samples.insert(4, outlier)

    consensus, spread_m, spread_rad, inliers = transform_inlier_consensus(
        samples, max_translation=0.012,
        max_rotation=math.radians(1.5))
    assert len(inliers) == 18
    assert spread_m < 0.004
    assert math.degrees(spread_rad) < 0.5
    assert transform_distance(consensus, truth)[0] < 0.001
