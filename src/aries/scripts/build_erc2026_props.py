#!/usr/bin/env python3
"""Build the ERC 2026 drone-cage and maintenance-panel props for Gazebo.

    python3 build_erc2026_props.py \
        --step "/path/to/Maintenance Task Panel .STEP"

Three stages:

1. Convert the organisers' SolidWorks STEP to a mesh (OpenCASCADE via cascadio).
2. Generate the ArUco textures the two tasks need, straight from OpenCV so the
   bit patterns are the real ones and a detector recovers the right ids.
3. Hand both to Blender, which conditions the panel and models the cage.

Then it prints the SDF the world file needs.  Sources, all from the ERC 2026 MY
Update Report Rev.1:

  Drone cage   10 x 10 x 4 m; effective area a disc of radius 3 m at the centre;
               lift-off spot a 1 x 1 m square at the centre carrying ArUco 101;
               landing target a disc of radius 0.5 m carrying ArUco 102; both
               markers 15 x 15 cm, ORIGINAL ArUco library.
  Panel        three marker locations, 50 x 50 mm, ORIGINAL ArUco library;
               allowed ids 11/13/14/15. The default practice arrangement uses
               11/13/14 at top-left/top-right/bottom-left, spanning 260 +/-1 mm
               across and 380 +/-1 mm down the console.

One caveat worth keeping in view: the report says the cage is "divided into
1 x 1 m sectors", but its own worked examples are half that - "A2 covers
(0 m; 0.5 m) to (0.5 m; 1 m)" and "D1 covers (-1.5 m; 0 m) to (-1 m; 0.5 m)"
are both 0.5 m cells, and the diagram shows the 1 x 1 m lift-off square
covering a 2 x 2 block of them.  The 0.5 m reading is the one implemented here
because two independent examples and the diagram agree on it against one
sentence.
"""

import argparse
import math
import pathlib
import shutil
import subprocess
import sys

import numpy as np

CAGE_SIDE = 10.0
CAGE_HEIGHT = 4.0
EFFECTIVE_RADIUS = 3.0
LIFTOFF_SIDE = 1.0
LANDING_RADIUS = 0.5
DRONE_MARKER_M = 0.15
SECTOR_M = 0.5

PANEL_MARKER_M = 0.050
PANEL_MARKER_SPAN_X = 0.260
PANEL_MARKER_SPAN_Y = 0.380
PANEL_MARKER_IDS = (11, 13, 14, 15)
PANEL_DEFAULT_MARKER_IDS = (11, 13, 14)
ARUCO_TEXTURE_PX = 512
ARUCO_QUIET_PX = int(ARUCO_TEXTURE_PX * 0.14)
ARUCO_TEXTURE_SCALE = ARUCO_TEXTURE_PX / (ARUCO_TEXTURE_PX - 2 * ARUCO_QUIET_PX)
# Keep the panel this far from every surveyed point: the markers are physical
# objects on the yard and the panel must not be planted on one.
PANEL_SURVEY_CLEARANCE = 1.5
# Half-diagonal of the panel's 0.49 x 0.39 m base, plus a little air so it
# stands on the rocks rather than in them.
PANEL_FOOTPRINT = 0.35
PANEL_CLEARANCE = 0.01


def log(msg):
    print(f"[erc2026-props] {msg}", flush=True)


def convert_step(step_path, out_glb):
    import cascadio

    log(f"converting {step_path.name}")
    cascadio.step_to_glb(str(step_path), str(out_glb), tol_linear=0.5, tol_angular=0.4)
    import trimesh

    mesh = trimesh.load(out_glb).to_mesh()
    log(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
        f"extent {np.round(mesh.extents, 3)} m")
    return out_glb


def aruco(dictionary, marker_id, px):
    import cv2

    return cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(dictionary), marker_id, px)


ALL_MARKER_IDS = PANEL_MARKER_IDS + (101, 102)


def write_textures(out_dir, panel_out_dir):
    """Generate the cage floor and one texture per ArUco tag.

    Each tag is its own square texture with a white quiet zone, applied to a
    thin box in the world.  The alternative - one composite texture per feature
    with the tag baked in and transparency around it - needs alpha blending,
    which ogre2 handles inconsistently for SDF primitives and would put the
    detector's quiet zone at the mercy of a blend mode.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_out_dir.mkdir(parents=True, exist_ok=True)
    orig = cv2.aruco.DICT_ARUCO_ORIGINAL

    # Cage floor: the effective area is a disc of radius 3 m, which the report
    # draws as a distinct region against the rest of the pad.
    px_per_m = 200
    side_px = int(CAGE_SIDE * px_per_m)
    floor = np.full((side_px, side_px, 3), (96, 104, 112), np.uint8)
    centre = side_px // 2
    cv2.circle(floor, (centre, centre), int(EFFECTIVE_RADIUS * px_per_m), (225, 228, 232), -1)
    cv2.circle(floor, (centre, centre), int(EFFECTIVE_RADIUS * px_per_m), (70, 70, 70), 3)
    cv2.imwrite(str(out_dir / "drone_cage_floor.png"), floor)

    for mid in ALL_MARKER_IDS:
        size = ARUCO_TEXTURE_PX
        quiet = ARUCO_QUIET_PX
        board = np.full((size, size), 255, np.uint8)
        board[quiet:size - quiet, quiet:size - quiet] = aruco(orig, mid, size - 2 * quiet)
        marker_dir = panel_out_dir if mid in PANEL_MARKER_IDS else out_dir
        cv2.imwrite(str(marker_dir / f"aruco_orig_{mid}.png"), board)

    log(f"  wrote cage floor and ORIGINAL-library tags "
        f"{'/'.join(str(i) for i in ALL_MARKER_IDS)}")


def verify_textures(out_dir, panel_out_dir):
    """Round-trip every generated tag through a detector.

    A texture that looks like an ArUco marker but decodes to nothing, or to the
    wrong id, is the failure mode worth catching here rather than in the sim.
    """
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())
        detect = detector.detectMarkers
    else:
        # OpenCV releases before 4.7 expose the equivalent module function.
        detect = lambda image: cv2.aruco.detectMarkers(image, dictionary)
    bad = []
    for mid in ALL_MARKER_IDS:
        marker_dir = panel_out_dir if mid in PANEL_MARKER_IDS else out_dir
        img = cv2.imread(str(marker_dir / f"aruco_orig_{mid}.png"), cv2.IMREAD_GRAYSCALE)
        _, ids, _ = detect(img)
        if ids is None or mid not in ids.flatten():
            bad.append((mid, None if ids is None else ids.flatten().tolist()))
    if bad:
        for want, got in bad:
            log(f"  FAIL aruco_orig_{want}.png: detector saw {got}")
        sys.exit("generated ArUco textures do not decode")
    log(f"  all {len(ALL_MARKER_IDS)} ArUco textures decode to their expected ids")


def panel_marker_poses(panel_glb):
    """Work out the three marker locations shown on the panel drawing.

    The console is a plane tilted 33 deg off vertical carrying several recessed
    sub-plates, so the outermost plane alone is too small to hold the drawing's
    260 x 380 mm marker span.  Taking every face parallel to the console normal
    recovers the full 390 x 549 mm working area, and the tags are then placed
    about its centre and floated 3 mm proud of the frontmost plate so nothing
    z-fights.
    """
    import trimesh

    mesh = trimesh.load(panel_glb)
    mesh = mesh.to_mesh() if hasattr(mesh, "to_mesh") else mesh
    normals = mesh.face_normals
    # Console face: tilted up and toward the model's +X front.
    cand = (normals[:, 0] > 0.2) & (normals[:, 2] > 0.5)
    n = (normals[cand] * mesh.area_faces[cand, None]).sum(0)
    n /= np.linalg.norm(n)

    coplanar = (normals @ n) > 0.99
    v = mesh.vertices[mesh.faces[coplanar]].reshape(-1, 3)
    u = np.array([0.0, 1.0, 0.0])              # across the console
    w = np.cross(n, u)
    w /= np.linalg.norm(w)                     # up the slope
    au, aw, ad = v @ u, v @ w, v @ n
    cu, cw = (au.min() + au.max()) / 2, (aw.min() + aw.max()) / 2
    front = ad.max()
    pitch = float(np.arctan2(n[0], n[2]))

    # Page 20 marks top-left, top-right and bottom-left with black squares.
    # The report lists four possible IDs but does not show a fourth location.
    layout = ((-1, +1, PANEL_DEFAULT_MARKER_IDS[0]),
              (+1, +1, PANEL_DEFAULT_MARKER_IDS[1]),
              (-1, -1, PANEL_DEFAULT_MARKER_IDS[2]))
    out = []
    for sx, sy, mid in layout:
        p = ((cu + sx * PANEL_MARKER_SPAN_X / 2) * u
             + (cw + sy * PANEL_MARKER_SPAN_Y / 2) * w
             + (front + 0.003) * n)
        out.append((mid, p, pitch))
    log(f"  console face {au.max() - au.min():.3f} x {aw.max() - aw.min():.3f} m, "
        f"tilt {np.degrees(pitch):.1f} deg; 3 marker locations placed")
    return out


def sector_table():
    """Reproduce the report's sector lettering over the effective disc.

    Quadrant letter then a ring index, with the 1 x 1 m lift-off square (a 2 x 2
    block of 0.5 m cells) unlabelled at the centre.  Anchored on the two worked
    examples: A2 = (0, 0.5)-(0.5, 1) and D1 = (-1.5, 0)-(-1, 0.5).
    """
    rows = []
    n = int(EFFECTIVE_RADIUS / SECTOR_M)
    for iy in range(-n, n):
        for ix in range(-n, n):
            x0, y0 = ix * SECTOR_M, iy * SECTOR_M
            cx, cy = x0 + SECTOR_M / 2, y0 + SECTOR_M / 2
            if np.hypot(cx, cy) > EFFECTIVE_RADIUS:
                continue
            if abs(cx) < LIFTOFF_SIDE / 2 and abs(cy) < LIFTOFF_SIDE / 2:
                continue
            quad = ("A" if cy > 0 else "B") if cx > 0 else ("D" if cy > 0 else "C")
            rows.append((quad, x0, y0))
    return rows


def read_survey_for_blender(coords_txt):
    """Survey points as [name, x, y, h], ready to hand to Blender as JSON.

    Same columns trap as everywhere else in this package: the file is
    Point, Y, X, H with a decimal comma.
    """
    import re

    pts = []
    for line in pathlib.Path(coords_txt).read_text(encoding="utf8").splitlines()[1:]:
        parts = [t for t in re.split(r"\s+", line.strip()) if t]
        if len(parts) < 4:
            continue
        y, x, h = (float(v.replace(",", ".")) for v in parts[1:4])
        pts.append([parts[0], x, y, h])
    log(f"  {len(pts)} survey points for the Blender scene")
    return pts


_DEM_CACHE = {}


def terrain_z(x, y, dem_png, zmin=-0.822577, span=2.254232,
              side=44.0, cx=0.190492, cy=14.019499, radius=0.0):
    """Highest marsyard2026 terrain within `radius` of (x, y), or None if outside.

    Defaults mirror the constants in marsyard2026.sdf; if the terrain is
    regenerated at a different resolution these must follow it.

    `radius` matters: sampling the single node under a prop puts it in whatever
    gap between rocks that node happens to be, and the prop then reads as
    sunk into the rocks beside it.  Pass the prop's own footprint.

    Both the collision grid and its downsampled visual twin are consulted -
    the smoothed visual can sit above the full-resolution collision in a
    hollow, and a prop needs to clear whichever the viewer or the wheels meet.
    """
    from PIL import Image

    visual_png = dem_png.parent / (dem_png.stem + "_visual" + dem_png.suffix)
    best = None
    for png in (dem_png, visual_png):
        if not png.is_file():
            continue
        if png not in _DEM_CACHE:
            _DEM_CACHE[png] = np.flipud(
                np.asarray(Image.open(png)).astype(np.float64)) / 65535.0
        grid = _DEM_CACHE[png]
        res = grid.shape[0]
        step = side / (res - 1)
        ix = int(round((x - (cx - side / 2)) / step))
        iy = int(round((y - (cy - side / 2)) / step))
        if not (0 <= ix < res and 0 <= iy < res):
            continue
        k = max(0, int(round(radius / step)))
        window = grid[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1]
        v = float(window.max()) * span + zmin
        best = v if best is None else max(best, v)
    return best


def tag_visual(name, mid, size, pose, texture_dir):
    # `size` is the outer black ArUco square specified by the report. Expand
    # the textured backing so its white quiet zone does not shrink that code.
    backing = size * ARUCO_TEXTURE_SCALE
    return f"""        <visual name='{name}'>
          <pose>{pose}</pose>
          <geometry><box><size>{backing:.4f} {backing:.4f} 0.002</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
            <pbr><metal>
              <albedo_map>{texture_dir}/aruco_orig_{mid}.png</albedo_map>
              <roughness>0.9</roughness><metalness>0</metalness>
            </metal></pbr>
          </material>
        </visual>"""


def emit_sdf(out_path, panel_glb, panel_texture_dir, cage_texture_dir,
             panel_xyz, panel_yaw,
             cage_xy, cage_z, landing_xy, include_cage=False):
    """Write the drone cage and maintenance panel as SDF models."""
    px, py, pz = panel_xyz
    ccx, ccy = cage_xy
    lx, ly = landing_xy
    tags = panel_marker_poses(panel_glb)

    marker_visuals = "\n".join(
        tag_visual(f"aruco_{mid}", mid, PANEL_MARKER_M,
                   f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 0 {pitch:.4f} 0", panel_texture_dir)
        for mid, p, pitch in tags)

    controls_json = panel_glb.parent / "panel_controls.json"
    control_links, control_joints, control_plugins, fixed_parts = "", "", "", ""
    n_controls = 0
    if controls_json.is_file():
        import json

        # Rotary shafts use the console normal; breaker handles use their
        # transverse hinge axis. Both are recorded by the Blender split stage.
        _, _, pitch = tags[0]
        controls = json.loads(controls_json.read_text())
        n_controls = len(controls)
        links, joints, fixed = [], [], []
        for c in controls:
            x, y, z = c["pivot"]
            fx, fy, fz = c.get("fixed_pivot", c["pivot"])
            axis = c.get("axis", [math.sin(pitch), 0, math.cos(pitch)])
            # Passive resistance keeps a released control in place without a
            # motor fighting the robot's gripper.
            default_dynamics = {
                "rotary": (0.04, 0.08),
                "disconnect": (0.08, 0.18),
                "breaker": (0.08, 0.12),
            }
            damping, friction = default_dynamics[c["kind"]]
            damping = c.get("damping", damping)
            friction = c.get("friction", friction)
            fixed.append(f"""        <visual name='{c['name']}_fixed'>
          <pose>{fx:.4f} {fy:.4f} {fz:.4f} 0 0 0</pose>
          <geometry><mesh><uri>{panel_texture_dir}/panel_{c['name']}_fixed.glb</uri></mesh></geometry>
        </visual>
        <collision name='{c['name']}_fixed_collision'>
          <pose>{fx:.4f} {fy:.4f} {fz:.4f} 0 0 0</pose>
          <geometry><mesh><uri>{panel_texture_dir}/panel_{c['name']}_fixed.glb</uri></mesh></geometry>
          <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface>
        </collision>""")
            links.append(f"""        <link name='{c['name']}'>
          <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
          <inertial>
            <mass>0.05</mass>
            <inertia><ixx>2e-5</ixx><iyy>2e-5</iyy><izz>2e-5</izz>
                     <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
          </inertial>
          <visual name='v'>
            <geometry><mesh><uri>{panel_texture_dir}/panel_{c['name']}.glb</uri></mesh></geometry>
          </visual>
          <collision name='collision'>
            <geometry><mesh><uri>{panel_texture_dir}/panel_{c['name']}.glb</uri></mesh></geometry>
            <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
          </collision>
        </link>""")
            # Axis is expressed directly in the model frame. The mesh was
            # already transformed into that frame before export, so adding a
            # link rotation here would double-tilt the control.
            joints.append(f"""        <joint name='{c['name']}_joint' type='revolute'>
          <parent>body</parent>
          <child>{c['name']}</child>
          <axis>
            <xyz expressed_in='__model__'>{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f}</xyz>
            <limit><lower>{c['lower']:.4f}</lower><upper>{c['upper']:.4f}</upper>
                   <effort>{c['effort']}</effort><velocity>{c['velocity']}</velocity></limit>
            <dynamics><damping>{damping:.2f}</damping><friction>{friction:.2f}</friction></dynamics>
          </axis>
        </joint>""")
        control_links = "\n" + "\n".join(links)
        control_joints = "\n" + "\n".join(joints)
        fixed_parts = "\n" + "\n".join(fixed)
        # No position controllers: contact from the arm is the only actuation.
        control_plugins = f"""
      <plugin filename='gz-sim-joint-state-publisher-system'
              name='gz::sim::systems::JointStatePublisher'>
        <topic>/maintenance_panel/joint_states</topic>
      </plugin>"""
        log(f"  articulating {n_controls} panel controls "
            f"({', '.join(sorted({c['kind'] for c in controls}))})")

    panel = f"""    <!-- Maintenance task panel, from the organisers' SolidWorks STEP
         ("Panel for Maintenance Tasks/Maintenance Task Panel .STEP") converted
         through OpenCASCADE and conditioned in Blender by
         scripts/build_erc2026_props.py. 0.49 x 0.39 x 1.00 m, base on the
         ground, console face 33 deg off vertical, model front is its own +X.

         Collision is a 2 322-face decimation of the 116 172-face visual: DART
         only needs the sloped-box silhouette, and the real mesh would put every
         switch and socket into the broad phase.

         The panel drawing shows three 50 x 50 mm ArUco locations: top-left,
         top-right and bottom-left. This deterministic practice setup uses
         ORIGINAL-library ids {'/'.join(str(m) for m, _, _ in tags)}, with
         {PANEL_MARKER_SPAN_X * 1000:.0f} mm horizontal and
         {PANEL_MARKER_SPAN_Y * 1000:.0f} mm vertical centre spacing. ID 15
         remains an organiser-permitted replacement texture, not a fourth slot.

         POSITION IS A PLACEHOLDER. The update report gives the panel's geometry
         and its starting point (S8) but never says where on the yard it stands,
         so this sits {2.5:.1f} m from S8 facing it. Move it when the organisers say. -->
    <model name='maintenance_panel'>
      <pose>{px:.3f} {py:.3f} {pz:.3f} 0 0 {panel_yaw:.4f}</pose>
      <!-- Not <static>: a static model cannot carry joints, so the body is
           welded to the world instead. Same immobility, but the switches move. -->
      <joint name='anchor' type='fixed'><parent>world</parent><child>body</child></joint>
      <link name='body'>
        <inertial>
          <mass>40</mass>
          <inertia><ixx>4.0</ixx><iyy>4.0</iyy><izz>2.0</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name='body_collision'>
          <geometry><mesh><uri>{panel_texture_dir}/maintenance_panel_collision.glb</uri></mesh></geometry>
          <surface>
            <friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction>
          </surface>
        </collision>
        <visual name='body_visual'>
          <geometry><mesh><uri>{panel_texture_dir}/maintenance_panel.glb</uri></mesh></geometry>
        </visual>
{marker_visuals}{fixed_parts}
      </link>{control_links}{control_joints}
{control_plugins}
    </model>"""

    half = CAGE_SIDE / 2
    cage = f"""    <!-- Droning sub-task cage: {CAGE_SIDE:.0f} x {CAGE_SIDE:.0f} x {CAGE_HEIGHT:.0f} m, per the update report,
         which places it OUTSIDE the Mars Yard without giving coordinates - it
         sits here clear of the terrain patch, on its own pad, so the yard and
         the cage can be worked on in one world.

         The frame and netting are a single Blender-built mesh ({CAGE_SIDE:.0f} m of posts,
         rails, wall net and roof net): one draw call instead of ~180 <visual>
         tags. Collision is four thin wall slabs plus a roof, which is all a
         drone can actually hit.

         Effective area is the disc of radius {EFFECTIVE_RADIUS:.0f} m marked on the pad. The
         lift-off spot is the {LIFTOFF_SIDE:.0f} x {LIFTOFF_SIDE:.0f} m square at the centre carrying
         ORIGINAL-library ArUco 101; the landing target is the disc of radius
         {LANDING_RADIUS:.1f} m carrying ArUco 102. Both tags are {DRONE_MARKER_M * 100:.0f} x {DRONE_MARKER_M * 100:.0f} cm.

         The report's 0.5 m reporting sectors are deliberately NOT drawn: they
         are a virtual division for reporting probe locations, and painting them
         on the pad would put lines in the drone camera's view that do not exist
         in the cage. Sector origin is the cage centre, letters A/B/C/D by
         quadrant (A = +x+y, B = +x-y, C = -x-y, D = -x+y). -->
    <model name='drone_cage'>
      <static>true</static>
      <pose>{ccx:.3f} {ccy:.3f} {cage_z:.3f} 0 0 0</pose>
      <link name='link'>
        <visual name='pad_visual'>
          <pose>0 0 0.005 0 0 0</pose>
          <geometry><box><size>{CAGE_SIDE:.2f} {CAGE_SIDE:.2f} 0.01</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
            <pbr><metal>
              <albedo_map>{cage_texture_dir}/drone_cage_floor.png</albedo_map>
              <roughness>0.95</roughness><metalness>0</metalness>
            </metal></pbr>
          </material>
        </visual>
        <collision name='pad_collision'>
          <pose>0 0 -0.05 0 0 0</pose>
          <geometry><box><size>{CAGE_SIDE:.2f} {CAGE_SIDE:.2f} 0.10</size></box></geometry>
          <surface>
            <friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction>
          </surface>
        </collision>
        <visual name='frame'>
          <geometry><mesh><uri>{cage_texture_dir}/drone_cage.glb</uri></mesh></geometry>
        </visual>
        <collision name='wall_north'>
          <pose>0 {half:.2f} {CAGE_HEIGHT / 2:.2f} 0 0 0</pose>
          <geometry><box><size>{CAGE_SIDE:.2f} 0.05 {CAGE_HEIGHT:.2f}</size></box></geometry>
        </collision>
        <collision name='wall_south'>
          <pose>0 -{half:.2f} {CAGE_HEIGHT / 2:.2f} 0 0 0</pose>
          <geometry><box><size>{CAGE_SIDE:.2f} 0.05 {CAGE_HEIGHT:.2f}</size></box></geometry>
        </collision>
        <collision name='wall_east'>
          <pose>{half:.2f} 0 {CAGE_HEIGHT / 2:.2f} 0 0 0</pose>
          <geometry><box><size>0.05 {CAGE_SIDE:.2f} {CAGE_HEIGHT:.2f}</size></box></geometry>
        </collision>
        <collision name='wall_west'>
          <pose>-{half:.2f} 0 {CAGE_HEIGHT / 2:.2f} 0 0 0</pose>
          <geometry><box><size>0.05 {CAGE_SIDE:.2f} {CAGE_HEIGHT:.2f}</size></box></geometry>
        </collision>
        <collision name='roof'>
          <pose>0 0 {CAGE_HEIGHT:.2f} 0 0 0</pose>
          <geometry><box><size>{CAGE_SIDE:.2f} {CAGE_SIDE:.2f} 0.05</size></box></geometry>
        </collision>
        <visual name='liftoff_pad'>
          <pose>0 0 0.012 0 0 0</pose>
          <geometry><box><size>{LIFTOFF_SIDE:.2f} {LIFTOFF_SIDE:.2f} 0.004</size></box></geometry>
          <material>
            <ambient>0.15 0.35 0.80 1</ambient><diffuse>0.15 0.35 0.80 1</diffuse>
          </material>
        </visual>
{tag_visual('liftoff_aruco', 101, DRONE_MARKER_M, '0 0 0.016 0 0 0', cage_texture_dir)}
        <visual name='landing_pad'>
          <pose>{lx:.2f} {ly:.2f} 0.012 0 0 0</pose>
          <geometry><cylinder><radius>{LANDING_RADIUS:.2f}</radius><length>0.004</length></cylinder></geometry>
          <material>
            <ambient>0.95 0.75 0.10 1</ambient><diffuse>0.95 0.75 0.10 1</diffuse>
          </material>
        </visual>
{tag_visual('landing_aruco', 102, DRONE_MARKER_M, f'{lx:.2f} {ly:.2f} 0.016 0 0 0', cage_texture_dir)}
      </link>
    </model>"""

    if include_cage:
        out_path.write_text(panel + "\n" + cage + "\n", encoding="utf8")
        log(f"  wrote {out_path.name}: maintenance_panel + drone_cage")
    else:
        out_path.write_text(panel + "\n", encoding="utf8")
        log(f"  wrote {out_path.name}: maintenance_panel (drone cage omitted; "
            f"pass --with-cage to include it)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = pathlib.Path(__file__).resolve().parents[1]
    ap.add_argument("--step", required=True, type=pathlib.Path,
                    help="Maintenance Task Panel .STEP from the organisers")
    ap.add_argument("--out", type=pathlib.Path, default=here / "models" / "marsyard2026")
    ap.add_argument("--panel-out", type=pathlib.Path,
                    default=here / "models" / "maintenance_panel",
                    help="standalone maintenance-panel model asset directory")
    ap.add_argument("--blender", default="blender")
    ap.add_argument("--render-dir", type=pathlib.Path,
                    help="also render Blender previews here")
    ap.add_argument("--skip-blender", action="store_true")
    ap.add_argument("--sdf-out", type=pathlib.Path,
                    help="write the two SDF models here for pasting into the world")
    ap.add_argument("--dem", type=pathlib.Path,
                    default=here / "models" / "dem" / "marsyard2026_terrain_hm.png",
                    help="heightmap used to stand the panel on the ground")
    ap.add_argument("--with-cage", action="store_true",
                    help="also emit the droning-task cage model (off by default)")
    ap.add_argument("--cage-xy", type=float, nargs=2, default=(-32.0, 8.0),
                    help="cage centre; must clear the 44 m terrain patch")
    ap.add_argument("--cage-z", type=float, default=0.0)
    ap.add_argument("--landing-xy", type=float, nargs=2, default=(-1.6, -1.1),
                    help="landing target centre within the cage")
    ap.add_argument("--blend-out", type=pathlib.Path,
                    help="also assemble the whole yard in Blender and save it here")
    ap.add_argument("--survey-txt", type=pathlib.Path,
                    help="Coordinates_MarsYard2026.txt, needed for --blend-out")
    ap.add_argument("--terrain-texture", type=pathlib.Path,
                    default=here / "models" / "dem" / "marsyard2026_terrain_texture.png")
    args = ap.parse_args()
    if args.blend_out and not args.survey_txt:
        ap.error("--blend-out needs --survey-txt")

    if not args.step.is_file():
        sys.exit(f"no such STEP file: {args.step}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.panel_out.mkdir(parents=True, exist_ok=True)

    write_textures(args.out, args.panel_out)
    verify_textures(args.out, args.panel_out)

    raw = args.panel_out / "_panel_raw.glb"
    convert_step(args.step, raw)

    # Panel placement, computed once and shared by the SDF and the .blend so the
    # two copies of the yard cannot drift apart.  Maintenance starts at S8
    # (X -15.676, Y 4.991), so the panel stands a few metres out from it facing
    # back at the start point.
    #
    # Choosing that spot by orthophoto brightness does NOT work: the brightest
    # thing near S8 is a landmark board, so the first attempt planted the panel
    # 0.34 m on top of L14.  Score on terrain instead - flat ground, clear of
    # every surveyed point - which is what the panel actually needs.
    s8 = np.array([-15.676, 4.991])
    survey_xy = np.array([[q[1], q[2]] for q in read_survey_for_blender(args.survey_txt)]) \
        if args.survey_txt else np.empty((0, 2))
    best = None
    for ox in np.arange(1.5, 7.01, 0.25):
        for oy in np.arange(-3.0, 3.01, 0.25):
            cand = s8 + np.array([ox, oy])
            z = terrain_z(cand[0], cand[1], args.dem, radius=PANEL_FOOTPRINT)
            if z is None:
                continue
            if survey_xy.size:
                clear = float(np.hypot(*(survey_xy - cand).T).min())
                if clear < PANEL_SURVEY_CLEARANCE:
                    continue
            else:
                clear = 99.0
            # Flatness over the panel's own footprint: it is a 0.5 x 0.4 m box
            # and wants to stand on something level, not straddle a rock.
            patch = [terrain_z(cand[0] + dx, cand[1] + dy, args.dem)
                     for dx in (-0.3, 0.0, 0.3) for dy in (-0.3, 0.0, 0.3)]
            patch = [v for v in patch if v is not None]
            rough = float(np.std(patch))
            score = -rough + 0.02 * clear
            if best is None or score > best[0]:
                best = (score, cand, z, rough, clear)
    if best is None:
        sys.exit("no viable panel position found near S8")
    _, pxy, pz, rough, clear = best
    pz += PANEL_CLEARANCE
    bearing = float(np.arctan2(s8[1] - pxy[1], s8[0] - pxy[0]))
    panel_pose = (float(pxy[0]), float(pxy[1]), float(pz), bearing)
    log(f"  panel at ({pxy[0]:.3f}, {pxy[1]:.3f}), terrain z {pz:.3f}, "
        f"yaw {np.degrees(bearing):.1f} deg toward S8")
    log(f"    ground roughness {rough * 100:.1f} cm std over its footprint, "
        f"nearest survey point {clear:.2f} m away")

    if not args.skip_blender:
        if shutil.which(args.blender) is None:
            sys.exit(f"blender not found on PATH as {args.blender!r}")
        script = pathlib.Path(__file__).resolve().parent / "erc2026_props_blender.py"
        cmd = [args.blender, "--background", "--python", str(script), "--",
               "--panel-glb", str(raw), "--out", str(args.out),
               "--panel-out", str(args.panel_out)]
        panel_3mf = args.step.parent / "3mf file Maintenance Task Panel.3MF"
        if panel_3mf.is_file():
            cmd += ["--panel-3mf", str(panel_3mf)]
        if args.render_dir:
            args.render_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--render-dir", str(args.render_dir)]
        if args.blend_out:
            # Hand Blender the survey points as JSON: it has no route to the
            # coordinates file's decimal commas and Y-before-X columns, and
            # parsing that twice is how the two copies drift apart.
            import json
            import tempfile

            pts = read_survey_for_blender(args.survey_txt)
            tmp = pathlib.Path(tempfile.gettempdir()) / "erc2026_survey.json"
            tmp.write_text(json.dumps(pts))
            args.blend_out.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--blend-out", str(args.blend_out), "--dem", str(args.dem),
                    "--terrain-texture", str(args.terrain_texture),
                    "--survey", str(tmp),
                    "--panel-pose", json.dumps(list(panel_pose)),
                    "--cage-pose", json.dumps([args.cage_xy[0], args.cage_xy[1], args.cage_z]),
                    "--landing-xy", json.dumps(list(args.landing_xy))]
        log(f"running blender: {' '.join(cmd[:4])} ...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        combined = res.stdout + res.stderr
        for line in combined.splitlines():
            if "erc2026-blender" in line:
                print(line, flush=True)
        # Blender exits 0 even when the driven script raises, so the return code
        # alone will happily report success on a traceback.  Match on our own
        # script name: Blender also spews unrelated tracebacks from addons while
        # shutting down, and failing on those cries wolf on a good build.
        ours = [i for i, line in enumerate(combined.splitlines())
                if script.name in line and "File " in line]
        if res.returncode != 0 or ours:
            lines = combined.splitlines()
            start = max(0, ours[0] - 3) if ours else max(0, len(lines) - 40)
            print("\n".join(lines[start:start + 40]), file=sys.stderr)
            sys.exit("blender stage failed")
        raw.unlink(missing_ok=True)

    rows = sector_table()
    log(f"  effective-area sector grid: {len(rows)} cells of {SECTOR_M} m "
        f"inside r={EFFECTIVE_RADIUS} m, lift-off square excluded")

    if args.sdf_out:
        emit_sdf(args.sdf_out, args.panel_out / "maintenance_panel.glb",
                 f"model://{args.panel_out.name}", f"model://{args.out.name}",
                 panel_pose[:3], panel_pose[3],
                 args.cage_xy, args.cage_z, args.landing_xy,
                 include_cage=args.with_cage)

    log("")
    log("Cage geometry for the world file:")
    log(f"  cage {CAGE_SIDE} x {CAGE_SIDE} x {CAGE_HEIGHT} m, effective disc r={EFFECTIVE_RADIUS} m")
    log(f"  lift-off {LIFTOFF_SIDE} m square (ArUco 101), landing disc "
        f"r={LANDING_RADIUS} m (ArUco 102), both tags {DRONE_MARKER_M} m")
    log(f"  panel markers {PANEL_MARKER_M} m at {PANEL_MARKER_SPAN_X} x "
        f"{PANEL_MARKER_SPAN_Y} m spacing, default ids "
        f"{PANEL_DEFAULT_MARKER_IDS}; allowed ids {PANEL_MARKER_IDS}")


if __name__ == "__main__":
    main()
