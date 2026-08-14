#!/usr/bin/env python3
"""Generate the Gazebo terrain assets for the 2026 MarsYard.

Input is the official survey package (``2026_MarsYard_3D.zip``), which ships a
photogrammetry mesh and a georeferenced orthophoto.  Output is the pair of
images ``marsyard2026.sdf`` needs: a 16-bit heightmap and a colour texture,
both covering the same square patch of the yard so they line up pixel for
pixel.

Why a heightmap rather than the mesh itself
-------------------------------------------
The survey mesh is 840 753 vertices / 1 677 535 triangles.  Handing that to
DART as a collision shape costs a broad-phase pass over 1.7 M triangles every
step, and mesh-vs-mesh contact against the wheels is where DART is least
robust.  A heightmap is a regular grid, so contact is an O(1) cell lookup, and
DART's bullet collision detector maps it onto ``btHeightfieldTerrainShape``.
The 2022 world already used this pattern; this script just automates it.

The tradeoff is that a heightmap is a function z = f(x, y), so it cannot
represent overhangs or the undercut of a boulder.  For a yard that is rocks
lying on graded sand this loses nothing a rover would ever drive under.

Frame
-----
The mesh vertices are already in the yard's local survey frame: origin at the
start area, +X right, +Y down the long axis, Z up, metres.  That is the frame
``Coordinates_MarsYard2026.txt`` uses (note its columns are Y, X, H) and the
frame the world file publishes, so no transform is applied anywhere here.

Usage
-----
    python3 build_marsyard2026_terrain.py --source /path/to/2026_MarsYard_3D_Model

Re-run with ``--res 2049`` for 2.1 cm/px if you need finer rocks and can pay
for the collision grid.
"""

import argparse
import pathlib
import sys

import numpy as np

# Gazebo image heightmaps must be square with a side of 2^n + 1.
DEFAULT_RES = 1025
DEFAULT_TEX = 4096
# The <visual> heightmap is downsampled: gz-sim's Ogre terrain renders a dense
# field of spikes from grids larger than this, while <collision> is fine at full
# resolution. Measured in-sim at 1025/513/257.
DEFAULT_VISUAL_RES = 257
# Square patch side.  The mesh is 37.4 x 43.8 m, so 44 m covers the long axis
# and leaves a apron on X that keeps the rover from driving off a cliff edge.
DEFAULT_SIDE = 44.0
# Lift markers this far clear of the terrain peak under them, so they do not
# z-fight with the surface they are standing on.
GROUND_CLEARANCE = 0.01


def log(msg):
    print(f"[marsyard2026] {msg}", flush=True)


def load_mesh(source):
    import trimesh

    ply = source / "Model3D_mesh1.ply"
    log(f"loading {ply.name}")
    mesh = trimesh.load(ply, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    log(f"  {len(verts)} vertices, {len(faces)} faces")
    log(f"  bounds min {verts.min(0)}")
    log(f"  bounds max {verts.max(0)}")
    return verts, faces


def rasterise_height(verts, faces, cx, cy, side, res):
    """Sample the top surface of the mesh onto a regular grid.

    A DEM is the *highest* surface over each cell, which is also what a wheel
    rides on, so this keeps the max rather than averaging.  Averaging would
    sink the rover into every rock.

    Interpolating the mesh's own triangulation would be the exact answer, but
    photogrammetry meshes are not planar in plan view - vertical rock faces
    produce overlapping triangles and matplotlib's trapezoid-map trifinder
    rejects the whole mesh as invalid.  Scattering dense samples over the
    triangles and keeping the max sidesteps that entirely: vertex spacing is
    ~4.4 cm against a ~4.3 cm cell, and adding centroids plus edge midpoints
    puts roughly seven samples in every cell.
    """
    half = side / 2.0
    step = side / (res - 1)
    x0, y0 = cx - half, cy - half
    log(f"  sampling {res}x{res} grid ({step * 100:.2f} cm/px)")

    tri = verts[faces]
    samples = [
        verts,
        tri.mean(axis=1),
        (tri[:, 0] + tri[:, 1]) * 0.5,
        (tri[:, 1] + tri[:, 2]) * 0.5,
        (tri[:, 2] + tri[:, 0]) * 0.5,
    ]
    pts = np.concatenate(samples, axis=0)
    log(f"  {len(pts)} surface samples from vertices, centroids and edge midpoints")

    ix = np.rint((pts[:, 0] - x0) / step).astype(np.int64)
    iy = np.rint((pts[:, 1] - y0) / step).astype(np.int64)
    inside = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
    ix, iy, pz = ix[inside], iy[inside], pts[inside, 2]

    height = np.full((res, res), -np.inf, dtype=np.float64)
    np.maximum.at(height, (iy, ix), pz)

    holes = ~np.isfinite(height)
    log(f"  {holes.sum()} empty cells ({100 * holes.mean():.1f}%)")
    return height, holes


def despike(height, valid, tol=0.05):
    """Knock out isolated photogrammetry spikes.

    Keeping the max over each cell also keeps any single bad vertex the
    reconstruction left floating above the surface.  A real rock at this
    resolution spans several cells, so anything that stands more than `tol`
    above its own 3x3 median is reconstruction noise, not terrain.

    Must run after the gaps are filled: on a grid that still holds -inf, a
    boundary cell's median is itself -inf, `height - med` is +inf, and every
    good cell along the survey edge gets "corrected" into a hole.
    """
    from scipy import ndimage

    med = ndimage.median_filter(height, size=3, mode="nearest")
    spikes = valid & (height - med > tol)
    if spikes.any():
        log(f"  flattening {spikes.sum()} spike cells >{tol * 100:.0f} cm above local median")
    return np.where(spikes, med, height)


def fill_interior(height, holes):
    """Linear-interpolate cells that fell inside the surveyed surface.

    These are cells that no sample happened to land in, almost always inside a
    triangle larger than one cell - i.e. flat ground, where linear
    interpolation across the gap is exactly right.
    """
    from scipy import ndimage
    from scipy.interpolate import griddata

    valid = ~holes
    # binary_fill_holes closes gaps enclosed by surveyed cells and leaves the
    # region outside the yard open, which separates the two fill strategies.
    enclosed = ndimage.binary_fill_holes(valid)
    interior = enclosed & holes
    exterior = ~enclosed
    log(f"  {interior.sum()} interior gaps, {exterior.sum()} cells outside the survey")

    if interior.any():
        vy, vx = np.nonzero(valid)
        hy, hx = np.nonzero(interior)
        height = height.copy()
        height[interior] = griddata(
            np.column_stack([vy, vx]), height[valid],
            np.column_stack([hy, hx]), method="linear",
        )
        still = interior & ~np.isfinite(height)
        if still.any():
            idx = ndimage.distance_transform_edt(
                ~valid, return_distances=False, return_indices=True
            )
            height = np.where(still, height[tuple(idx)], height)
    return height, exterior


def fill_apron(height, exterior):
    """Fill cells outside the surveyed area with a smooth skirt.

    Nearest-neighbour fill alone leaves radial streaks off every boundary
    vertex, which read as terrain the rover tries to climb.  Blurring only the
    filled cells flattens the skirt without touching a single surveyed height.
    """
    from scipy import ndimage

    if not exterior.any():
        return height

    idx = ndimage.distance_transform_edt(
        exterior, return_distances=False, return_indices=True
    )
    filled = height[tuple(idx)]
    smooth = ndimage.gaussian_filter(filled, sigma=8.0)
    out = np.where(exterior, smooth, height)
    # One narrow blur across the seam so the skirt meets the survey without a
    # step the wheels would catch on.
    seam = ndimage.binary_dilation(exterior, iterations=3) & ~exterior
    out = np.where(seam, ndimage.gaussian_filter(out, sigma=1.5), out)
    return out


def read_survey(source):
    """Parse Coordinates_MarsYard2026.txt.

    Columns are Point, Y, X, H in that order - Y before X - and the numbers use
    a decimal comma.  Both are easy to get backwards and both are silent.
    """
    import re

    path = source / "Coordinates_MarsYard2026.txt"
    pts = []
    for line in path.read_text(encoding="utf8").splitlines()[1:]:
        parts = [t for t in re.split(r"\s+", line.strip()) if t]
        if len(parts) < 4:
            continue
        name = parts[0]
        y, x, h = (float(v.replace(",", ".")) for v in parts[1:4])
        pts.append((name, x, y, h))
    log(f"  {len(pts)} survey points from {path.name}")
    return pts


def level_to_survey(height, pts, cx, cy, side, res):
    """De-tilt the grid onto the yard's leveled height datum.

    The photogrammetry mesh and the surveyed H column disagree by a plane: the
    reconstruction's vertical is off by about 1.3 deg across the long axis.
    Left alone it puts the far end of the yard more than half a metre below its
    surveyed height, so a waypoint reached in sim is not the waypoint on the
    day.  Fitting and removing that plane is the difference between a
    good-looking terrain and a correct one.

    The fit uses only S/W/P - starting locations, traverse waypoints and the
    deep sampling point, all of which are marks on open ground.  Landmarks (L)
    sit on objects, so the DEM there reads the object rather than the ground
    and they would bias the plane.
    """
    step = side / (res - 1)
    x0, y0 = cx - side / 2.0, cy - side / 2.0

    def sample(x, y):
        ix, iy = int(round((x - x0) / step)), int(round((y - y0) / step))
        if not (0 <= ix < res and 0 <= iy < res):
            return np.nan
        return height[iy, ix]

    names = np.array([p[0] for p in pts])
    px = np.array([p[1] for p in pts])
    py = np.array([p[2] for p in pts])
    ph = np.array([p[3] for p in pts])
    dem = np.array([sample(x, y) for x, y in zip(px, py)])

    ground = np.array([n[0] in "SWP" for n in names]) & np.isfinite(dem)
    resid = dem - ph
    design = np.column_stack([np.ones_like(px), px, py])
    coef, *_ = np.linalg.lstsq(design[ground], resid[ground], rcond=None)
    log(f"  datum plane dz = {coef[0]:+.4f} {coef[1]:+.5f}*X {coef[2]:+.5f}*Y")
    log(f"    tilt {np.degrees(np.arctan(coef[2])):+.3f} deg along Y, "
        f"{np.degrees(np.arctan(coef[1])):+.3f} deg along X")

    gx = x0 + np.arange(res) * step
    gy = y0 + np.arange(res) * step
    mx, my = np.meshgrid(gx, gy)
    height = height - (coef[0] + coef[1] * mx + coef[2] * my)

    after = np.array([sample(x, y) for x, y in zip(px, py)]) - ph
    for key, label in (("S", "starts"), ("W", "waypoints"), ("P", "deep sample"),
                       ("L", "landmarks")):
        sel = np.array([n[0] == key for n in names]) & np.isfinite(after)
        if sel.any():
            log(f"    {label:12s} n={sel.sum():2d} "
                f"mean {after[sel].mean():+.3f} m  std {after[sel].std():.3f} m")
    ok = np.isfinite(after)
    log(f"    all points   n={ok.sum():2d} std {after[ok].std():.3f} m "
        f"(was {resid[ok].std():.3f} m)")
    return height, coef


def write_visual_mesh(verts, faces, datum_coef, texture_png, out_glb,
                      face_count, texture_size):
    """Write a lightweight, texture-baked copy of the official survey mesh.

    Collision stays on the regular heightfield, but a heightfield visual turns
    vertical boulder faces into slopes. A decimated visual-only mesh preserves
    those silhouettes at a fraction of the 1.68-million-triangle source cost.
    The orthophoto is embedded in the GLB because Ogre2 does not display the PLY
    vertex colours exported by trimesh. The same fitted datum plane used by the
    heightmap is removed from every mesh vertex.
    """
    import trimesh

    aligned = verts.copy()
    aligned[:, 2] -= (datum_coef[0] + datum_coef[1] * aligned[:, 0]
                      + datum_coef[2] * aligned[:, 1])
    mesh = trimesh.Trimesh(vertices=aligned, faces=faces, process=False)

    target = min(int(face_count), len(faces))
    if target < len(faces):
        log(f"  decimating visual mesh {len(faces)} -> {target} faces")
        visual = mesh.simplify_quadric_decimation(face_count=target, aggression=7)
    else:
        visual = mesh

    # The generated terrain texture covers the 44 m square centered on the
    # mesh bounds. glTF's V axis is opposite the PNG row direction.
    lo, hi = verts.min(0), verts.max(0)
    cx, cy = (lo[:2] + hi[:2]) / 2
    side = DEFAULT_SIDE
    x0, y0 = cx - side / 2, cy - side / 2
    uv = np.column_stack(((visual.vertices[:, 0] - x0) / side,
                          1.0 - (visual.vertices[:, 1] - y0) / side))
    from PIL import Image
    texture = Image.open(texture_png).convert("RGB").resize(
        (texture_size, texture_size), Image.Resampling.LANCZOS)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture, roughnessFactor=0.95, metallicFactor=0.0)
    visual.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    visual.export(out_glb)
    log(f"  wrote {out_glb.name} ({len(visual.faces)} textured faces, "
        f"{out_glb.stat().st_size / (1024 * 1024):.1f} MiB)")


# Landmark board geometry.  The ERC update report gives the ArUco library and
# IDs but points at the Rules for the physical size, which is not in the
# package - these are the assumption, and they are the only lines to change
# when the Rules number is to hand.
LANDMARK_MARKER_M = 0.30      # side of the ArUco square itself
LANDMARK_BOARD_W = 0.40       # white board width
LANDMARK_BOARD_H = 0.55       # white board height, number panel included
LANDMARK_BOARD_BOTTOM = 0.25  # board underside above the surveyed ground point
LANDMARK_ARUCO_BASE = 50      # landmark L{n} carries 5x5 ArUco id 50+n
LANDMARK_LOOKOUT = 0.60       # radius searched for terrain the board must clear
LANDMARK_CLEARANCE = 0.05
LANDMARK_BOARD_CLEAR = LANDMARK_CLEARANCE


def write_aruco_boards(pts, out_dir, px=768):
    """Render one landmark board texture per L point.

    Reproduces the layout in the update report: a white portrait board with the
    landmark number boxed at the top and the 5x5 ArUco below, black envelope
    included.  Detection needs the envelope, so it is drawn explicitly rather
    than relying on the board's white background.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    written = 0
    for name, *_ in pts:
        if name[0] != "L":
            continue
        marker_id = LANDMARK_ARUCO_BASE + int(name[1:])
        aspect = LANDMARK_BOARD_H / LANDMARK_BOARD_W
        w, h = px, int(round(px * aspect))
        board = np.full((h, w, 3), 255, np.uint8)

        side = int(round(px * LANDMARK_MARKER_M / LANDMARK_BOARD_W))
        tag = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        x0 = (w - side) // 2
        y0 = h - side - int(0.06 * h)
        board[y0:y0 + side, x0:x0 + side] = tag[:, :, None]

        # Numbered panel above the tag, as printed on the real boards.
        bx0, bx1 = int(0.18 * w), int(0.82 * w)
        by0, by1 = int(0.05 * h), int(0.05 * h) + int(0.14 * h)
        cv2.rectangle(board, (bx0, by0), (bx1, by1), (0, 0, 0), max(2, px // 256))
        label = name[1:]
        scale = (by1 - by0) / 30.0
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        cv2.putText(board, label, ((w - tw) // 2, (by0 + by1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"landmark_{name}.png"), board)
        written += 1
    log(f"  wrote {written} landmark board textures to {out_dir.name}/ "
        f"(5x5 ArUco ids {LANDMARK_ARUCO_BASE + 1}-{LANDMARK_ARUCO_BASE + written})")


def make_ground_sampler(height, quant, visual_res, cx, cy, side, res, zmin, span):
    """Return ground(x, y): the highest terrain surface at a point.

    Markers are surveyed at the *ground mark*, but the DEM keeps the maximum
    over each cell (what a wheel rides on) and at a landmark that maximum is the
    landmark's own base or the rock it stands on, not the mark.  Placing a
    marker at its surveyed H therefore buries it - measured at up to 16.5 cm.

    Both grids are consulted because collision and visual are different
    resolutions and the smoothed visual can sit above the full-resolution
    collision in a hollow.  Taking the max of the two means a marker is never
    swallowed by either.
    """
    import numpy as np
    from PIL import Image

    step = side / (res - 1)
    x0, y0 = cx - side / 2, cy - side / 2
    vis = np.asarray(
        Image.fromarray(quant).resize((visual_res, visual_res), Image.BOX)
    ).astype(np.float64) / 65535.0
    vis = np.flipud(vis) * span + zmin
    vstep = side / (visual_res - 1)

    def peak(grid, gres, gstep, x, y, radius):
        ix = int(round((x - x0) / gstep))
        iy = int(round((y - y0) / gstep))
        k = max(0, int(round(radius / gstep)))
        window = grid[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1]
        if window.size == 0:
            return grid[min(gres - 1, max(0, iy)), min(gres - 1, max(0, ix))]
        return window.max()

    def ground(x, y, radius=0.0):
        """Highest terrain within `radius` of (x, y).

        Sampling the single node under a marker is not enough: the yard is rocks
        on sand at 4.3 cm/px, and a marker often lands in a gap between rocks
        whose neighbours stand 10-28 cm higher.  The marker then reads as buried
        even though the node beneath it is clear.  Taking the peak over the
        marker's own footprint is what actually keeps it above ground.
        """
        a = peak(height, res, step, x, y, radius)
        b = peak(vis, visual_res, vstep, x, y, radius)
        return max(float(a), float(b))

    return ground


def write_markers(pts, out_path, texture_uri="model://marsyard2026", ground=None):
    """Emit physical landmarks plus named frames for non-physical points.

    Landmarks become real ArUco boards - textured, collidable, and facing the
    start area - because they are physical objects the rover has to see and
    avoid, and a vision stack cannot be tested against a coloured stick.

    Starting locations, waypoints and the deep-sampling location are coordinate
    references in the organiser material, not coloured posts or painted pads.
    They therefore become world frames with no visual or collision geometry.
    This keeps the exact survey coordinates available to navigation software
    without inventing objects that hide the official orthophoto or overlap a
    robot spawned at S1.

    This block is already pasted into marsyard2026.sdf; regenerate it only if
    the coordinates file or the board geometry changes, then replace the
    survey-frame / landmark_* block in the world with the new output.
    """
    blocks = []
    sunk = 0
    for name, x, y, h_survey in pts:
        kind = name[0]
        # Sit on whichever is higher, the surveyed mark or the terrain the sim
        # actually has. X/Y stay exactly on the survey coordinate, so a nav test
        # still targets the official point.
        # Radius of the thing being planted, so the peak search covers what the
        # viewer actually sees intersecting the ground.
        footprint = {"L": 0.25, "P": 0.55, "S": 0.10, "W": 0.10}[kind]
        if ground is None:
            h = h_survey
        else:
            h = max(h_survey, ground(x, y, footprint) + GROUND_CLEARANCE)
        if kind == "L" and h - h_survey > 0.005:
            sunk += 1
        if kind == "L":
            # Face the board at the start area: the traverse begins at S1/S2 on
            # the origin, so a board's normal pointing there is the view the
            # rover most often gets. Yaw is +X-normal rotated to look at (0, 0).
            # The board box is W x 0.02 x H, so its printed face looks along the
            # model's +Y, not +X. To aim that face at the start area, the model
            # yaw is the bearing to the origin MINUS 90 deg. Without the offset
            # every board is turned 90 deg and the rover sees a 2 cm edge - which
            # is exactly what it looked like in-sim, and why detection kept
            # failing from any angle that should have worked.
            yaw = np.arctan2(-y, -x) - np.pi / 2
            # Stand the board clear of whatever is around it. The real
            # landmarks were in the yard when it was scanned, so the DEM already
            # has a mound where each one stood, and a fixed-height post plants
            # the board straight into it. Reach out far enough to see the mound.
            near = h if ground is None else ground(x, y, LANDMARK_LOOKOUT)
            post_len = max(LANDMARK_BOARD_BOTTOM, near - h + LANDMARK_BOARD_CLEAR)
            zc = h + post_len + LANDMARK_BOARD_H / 2
            blocks.append(f"""    <model name='landmark_{name}'>
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {h:.3f} 0 0 {yaw:.4f}</pose>
      <link name='link'>
        <collision name='post_collision'>
          <pose>0 0 {post_len / 2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.025</radius><length>{post_len:.3f}</length></cylinder></geometry>
        </collision>
        <visual name='post_visual'>
          <pose>0 0 {post_len / 2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>0.025</radius><length>{post_len:.3f}</length></cylinder></geometry>
          <material><ambient>0.25 0.25 0.25 1</ambient><diffuse>0.25 0.25 0.25 1</diffuse></material>
        </visual>
        <collision name='board_collision'>
          <pose>0 0 {post_len + LANDMARK_BOARD_H / 2:.3f} 0 0 0</pose>
          <geometry><box><size>{LANDMARK_BOARD_W:.3f} 0.02 {LANDMARK_BOARD_H:.3f}</size></box></geometry>
        </collision>
        <visual name='board_visual'>
          <pose>0 0 {post_len + LANDMARK_BOARD_H / 2:.3f} 0 0 0</pose>
          <geometry><box><size>{LANDMARK_BOARD_W:.3f} 0.02 {LANDMARK_BOARD_H:.3f}</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
            <pbr><metal>
              <albedo_map>{texture_uri}/landmark_{name}.png</albedo_map>
              <roughness>0.9</roughness><metalness>0</metalness>
            </metal></pbr>
          </material>
        </visual>
      </link>
    </model>""")
        else:
            blocks.append(f"""    <frame name='{name}'>
      <pose>{x:.3f} {y:.3f} {h_survey:.3f} 0 0 0</pose>
    </frame>""")
    header = (
        f"    <!-- {len(pts)} official survey points from Coordinates_MarsYard2026.txt.\n"
        f"         L1-L15 are ArUco landmark boards (5x5 library, ids "
        f"{LANDMARK_ARUCO_BASE + 1}-{LANDMARK_ARUCO_BASE + 15}, landmark n -> id "
        f"{LANDMARK_ARUCO_BASE}+n) and are collidable.\n"
        f"         S starts, W traverse waypoints and P deep-sampling location are\n"
        f"         named frames only: the report defines coordinates, not visible props. -->")
    out_path.write_text(header + "\n" + "\n".join(blocks) + "\n", encoding="utf8")
    log(f"  wrote {out_path.name} (15 physical landmark models, 19 survey frames; "
        f"{sunk} landmarks raised onto the terrain surface)")


def write_visual_heightmap(quant, out_png, res):
    """Write the downsampled heightmap the terrain's <visual> uses.

    gz-sim's Ogre terrain renders spike artifacts from a heightmap this large:
    at 1025 the yard grows a dense field of sharp cones, at 513 they are still
    visible, and at 257 the surface is clean.  Neither the bit depth nor
    <sampling> changes it - all three were tried against a camera in-sim - so
    the visual gets its own downsampled copy while <collision> keeps the full
    grid.  Physics stays accurate to the survey and the cameras see a surface
    that is smoothed, not spiked.

    The two agree in the mean: same extent, same <size>, area-averaged.
    """
    from PIL import Image

    Image.fromarray(quant).resize((res, res), Image.BOX).save(out_png, optimize=True)
    log(f"  wrote {out_png.name} ({res}x{res}) for the terrain <visual>")


def write_heightmap(height, out_png, res):
    from PIL import Image

    zmin = float(height.min())
    zmax = float(height.max())
    span = zmax - zmin
    log(f"  height range {zmin:.4f} .. {zmax:.4f} m (span {span:.4f} m)")

    # Gazebo maps pixel 0 -> 0 and pixel max -> size_z, measured up from the
    # collision's own pose, so the world file carries zmin as its pose Z.
    norm = (height - zmin) / span
    # Image row 0 is +Y (north).  The grid was built with y ascending, so flip.
    norm = np.flipud(norm)
    quant = np.clip(np.rint(norm * 65535.0), 0, 65535).astype(np.uint16)

    Image.fromarray(quant).save(out_png, optimize=True)
    log(f"  wrote {out_png.name} ({res}x{res}, 16-bit, {span / 65535 * 1000:.4f} mm/level)")
    return zmin, span, quant


def write_texture(source, out_png, cx, cy, side, tex, surveyed):
    """Resample the orthophoto onto the same square patch as the heightmap.

    `surveyed` is the heightmap-resolution mask of cells the survey actually
    covered.  It decides which pixels count as yard when picking the apron
    colour - the orthophoto also captures the treeline around the site, and a
    median over the whole frame comes out forest green.
    """
    import cv2
    from PIL import Image
    from PIL.TiffTags import TAGS

    Image.MAX_IMAGE_PIXELS = None
    ortho_path = source / "orthophoto.tif"
    log(f"loading {ortho_path.name}")
    im = Image.open(ortho_path)

    tags = {TAGS.get(k, k): v for k, v in im.tag_v2.items()}
    tie = tags["ModelTiepointTag"]
    scale = tags["ModelPixelScaleTag"]
    # Tiepoint maps raster (i, j) -> model (x, y).  The photogrammetry wrote it
    # in the yard's local frame despite the CRS key, which is what lets the
    # orthophoto drop straight onto the mesh with no reprojection.
    ox, oy = float(tie[3]), float(tie[4])
    px, py = float(scale[0]), float(scale[1])
    log(f"  tiepoint ({ox:.3f}, {oy:.3f}) m, {px * 1000:.4f} mm/px")

    rgba = np.asarray(im.convert("RGBA"))
    h, w = rgba.shape[:2]
    log(f"  {w}x{h} covering X [{ox:.2f}, {ox + w * px:.2f}] Y [{oy - h * py:.2f}, {oy:.2f}]")

    half = side / 2.0
    # Output row 0 is +Y, matching the flipped heightmap.
    wx = np.linspace(cx - half, cx + half, tex, dtype=np.float32)
    wy = np.linspace(cy + half, cy - half, tex, dtype=np.float32)
    mwx, mwy = np.meshgrid(wx, wy)
    map_x = ((mwx - ox) / px).astype(np.float32)
    map_y = ((oy - mwy) / py).astype(np.float32)

    sampled = cv2.remap(
        rgba, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    rgb = sampled[:, :, :3].astype(np.float32)
    alpha = (sampled[:, :, 3:4].astype(np.float32)) / 255.0
    # The orthophoto is transparent outside the surveyed polygon, and so is
    # anything past its edge.  Composite over the yard's own median sand so the
    # apron reads as more terrain rather than a black void.
    solid = alpha[:, :, 0] > 0.5
    # `surveyed` arrives in grid orientation (row 0 = -Y); this output, like the
    # heightmap PNG, has row 0 = +Y.
    yard = np.asarray(
        Image.fromarray(np.flipud(surveyed).astype(np.uint8) * 255)
        .resize((tex, tex), Image.NEAREST)
    ) > 127
    pick = solid & yard
    sand = np.median(rgb[pick], axis=0) if pick.any() else np.array([180.0, 150.0, 120.0])
    log(f"  apron fill RGB {sand.round(1)} from {pick.sum()} yard px, "
        f"{100 * (1 - solid.mean()):.1f}% of patch needs filling")
    out = rgb * alpha + sand[None, None, :] * (1.0 - alpha)

    Image.fromarray(out.astype(np.uint8), mode="RGB").save(out_png, optimize=True)
    log(f"  wrote {out_png.name} ({tex}x{tex}, {side / tex * 100:.2f} cm/px)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=pathlib.Path,
                    help="extracted 2026_MarsYard_3D_Model directory")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1] / "models" / "dem",
                    help="output directory (default: the aries dem model dir)")
    ap.add_argument("--res", type=int, default=DEFAULT_RES,
                    help=f"heightmap side, must be 2^n+1 (default {DEFAULT_RES})")
    ap.add_argument("--tex", type=int, default=DEFAULT_TEX, help="texture side")
    ap.add_argument("--visual-res", type=int, default=DEFAULT_VISUAL_RES,
                    help=f"heightmap side for the terrain <visual> (default "
                         f"{DEFAULT_VISUAL_RES}; above this gz renders spikes)")
    ap.add_argument("--side", type=float, default=DEFAULT_SIDE,
                    help="square patch side in metres")
    ap.add_argument("--no-level", action="store_true",
                    help="keep the mesh's own vertical datum instead of fitting "
                         "it to the surveyed heights")
    ap.add_argument("--markers-out", type=pathlib.Path,
                    help="also write the survey-point SDF models here, for "
                         "pasting into marsyard2026.sdf")
    ap.add_argument("--visual-faces", type=int, default=0,
                    help="optionally export a textured comparison GLB with "
                         "this triangle count (default 0: the SDF uses the "
                         "shared colour heightfield instead)")
    ap.add_argument("--visual-mesh-out", type=pathlib.Path,
                    help="visual GLB output (default: models/marsyard2026/"
                         "marsyard2026_visual.glb beside --out)")
    ap.add_argument("--visual-texture-size", type=int, default=2048,
                    help="embedded visual-mesh texture side (default 2048)")
    args = ap.parse_args()

    if (args.res - 1) & (args.res - 2) != 0 and bin(args.res - 1).count("1") != 1:
        sys.exit(f"--res must be 2^n + 1, got {args.res}")
    args.out.mkdir(parents=True, exist_ok=True)

    verts, faces = load_mesh(args.source)
    lo, hi = verts.min(0), verts.max(0)
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    log(f"patch centre ({cx:.4f}, {cy:.4f}) m, side {args.side} m")

    height, holes = rasterise_height(verts, faces, cx, cy, args.side, args.res)
    height, exterior = fill_interior(height, holes)
    height = fill_apron(height, exterior)
    height = despike(height, ~holes)
    if not np.isfinite(height).all():
        sys.exit("internal error: heightmap still has non-finite cells")

    pts = read_survey(args.source)
    datum_coef = np.zeros(3, dtype=np.float64)
    if not args.no_level:
        height, datum_coef = level_to_survey(height, pts, cx, cy, args.side, args.res)

    zmin, span, quant = write_heightmap(
        height, args.out / "marsyard2026_terrain_hm.png", args.res)
    write_visual_heightmap(quant, args.out / "marsyard2026_terrain_hm_visual.png",
                           args.visual_res)
    write_texture(args.source, args.out / "marsyard2026_terrain_texture.png",
                  cx, cy, args.side, args.tex, ~exterior)
    if args.visual_faces > 0:
        visual_out = (args.visual_mesh_out or
                      args.out.parent / "marsyard2026" / "marsyard2026_visual.glb")
        write_visual_mesh(verts, faces, datum_coef,
                          args.out / "marsyard2026_terrain_texture.png",
                          visual_out, args.visual_faces,
                          args.visual_texture_size)
    if args.markers_out:
        write_aruco_boards(pts, args.out.parent / "marsyard2026")
        ground = make_ground_sampler(height, quant, args.visual_res, cx, cy,
                                     args.side, args.res, zmin, span)
        write_markers(pts, args.markers_out, ground=ground)

    log("")
    log("SDF values for marsyard2026.sdf:")
    log(f"  <size>{args.side} {args.side} {span:.6f}</size>")
    log(f"  terrain model pose: {cx:.6f} {cy:.6f} {zmin:.6f} 0 0 0")
    log("  <pos> stays 0 0 0: gz-sim ignores a heightmap's <pos> Z and puts")
    log("  pixel 0 at the link origin, so the vertical datum rides on the model")
    log("  pose. Measured on this stack, not assumed.")


if __name__ == "__main__":
    main()
