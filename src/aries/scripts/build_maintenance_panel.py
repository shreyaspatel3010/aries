#!/usr/bin/env python3
"""Model the ERC 2026 maintenance panel from the MY Update Report Rev.1.

    blender --background --python build_maintenance_panel.py -- --out-dir DIR

Writes ``erc2026_maintenance_panel.blend`` and ``erc2026_maintenance_panel.3mf``.

Everything on the sloped control face is measured off the dimensioned front
view on slide 20 of "[ERC 2026] MY Update Report Rev.1.pdf".  That view is
normal to the control face -- the two callouts it carries, 260 +/-1 across the
top marker pair and 380 +/-1 down the left marker pair, come back at 3.5519 and
3.5789 px/mm from the render, a 0.76% spread, so nothing in it is foreshortened
and every distance below is a true one.  Calibrated on those two callouts the
layout falls out on a round grid:

    control face   330 wide x 450 down-slope
    marker pair    50 x 50, inset 35 from each side, centred in rows 1 and 5
                   -> 35 + 260 + 35 = 330 across, 35 + 310 + 35 = 380 down
    plate rows     70 / 110 / 110 / 90 / 70 (= 450) down-slope
    push buttons   40 pitch, centred on the face
    cam switches   60 pitch, centred on the face
    disconnects    70 pitch, centred on their own sub-plate
    DIN devices    18.0 pitch

The console body underneath is NOT dimensioned anywhere in the report, so it is
recovered photogrammetrically from the shaded isometric on the same slide.
Three markers plus five push buttons give the face-plane -> image affine map;
requiring the projection to stay orthographic with one scale fixes the
out-of-plane axis up to sign, and the sign follows from the buttons standing
proud.  Reading the silhouette back through that map puts the ridge 285 behind
and 54 above the top-left marker, and the front-bottom edge 306 ahead and 378
below it.  Two caveats on the body, neither of which touches the face:

  * The face angle rests on the corner feet reading dead vertical in the
    isometric -- 120 px of constant x at full resolution -- which pins world-up
    to image-up and gives 48 deg from horizontal.  Fitting the long lower-right
    silhouette edge as a horizontal instead would give 37 deg.  The feet win:
    that measurement is unambiguous, where the silhouette edge is only a
    horizontal if the console's underside is flat.
  * Under the 48 deg reading that underside comes out tilted 11 deg, which no
    piece of ground support equipment would be.  The base here is flat and the
    rear face takes up the difference, so the depth is right at the bottom and
    the rear face is ~11 deg steeper than the drawing's.

So: the control face is good to well under a millimetre, the body envelope to a
few per cent.  The face is the half the rover touches.
"""

import argparse
import json
import math
import pathlib
import sys

import addon_utils
import bpy
import bmesh
from mathutils import Matrix, Vector

THREEMF_ADDON = "bl_ext.blender_org.ThreeMF_io"

MM = 0.001

# --------------------------------------------------------------------------
# control face -- all mm, u across (0 = left edge seen from the front),
# v down-slope (0 = top edge), n out of the face (0 = the body's sloped skin).
# --------------------------------------------------------------------------
FACE_W = 330.0
FACE_H = 450.0
FACE_ANGLE_DEG = 48.0          # control face above horizontal

PLATE_T = 3.0                  # sub-plate thickness; components sit at n = 3
SEAM = 2.0                     # gap milled between adjacent sub-plates
FRAME_W = 34.0                 # corner post / border bar width
FRAME_PROUD = 12.0

ROW_V = (0.0, 70.0, 180.0, 290.0, 380.0, 450.0)
# (row index, u0, u1) for every sub-plate, in the order they are bolted on
SUBPLATES = (
    (0,   0.0,  70.0), (0,  70.0, 260.0), (0, 260.0, 330.0),
    (1,   0.0, 330.0),
    (2,   0.0, 330.0),
    (3,   0.0, 165.0), (3, 165.0, 330.0),
    (4,   0.0,  70.0), (4,  70.0, 330.0),
)

MARKER_MM = 50.0               # report: "actual size of the markers ... 50x50mm"
MARKER_POS = ((35.0, 35.0), (295.0, 35.0), (35.0, 415.0))
MARKER_IDS_DEFAULT = (11, 13, 14)      # report allows 11 / 13 / 14 / 15
MARKER_PROUD = 0.6

# ORIGINAL ArUco library, 5 data bits inside a 1-cell black envelope.  Taken
# from cv2.aruco.generateImageMarker(DICT_ARUCO_ORIGINAL, id, 7) and round
# tripped through the detector; 1 = white.
ARUCO_BITS = {
    11: ("0000000", "0100000", "0100000", "0100000", "0010010", "0011100", "0000000"),
    13: ("0000000", "0100000", "0100000", "0100000", "0011100", "0101110", "0000000"),
    14: ("0000000", "0100000", "0100000", "0100000", "0011100", "0010010", "0000000"),
    15: ("0000000", "0100000", "0100000", "0100000", "0011100", "0011100", "0000000"),
}

# row 1 -- square illuminated push buttons
BUTTON_U = (85.0, 125.0, 165.0, 205.0, 245.0)
BUTTON_V = 45.0
BUTTON_BEZEL = 19.2            # measured 19.1 outer
BUTTON_COLLAR = 15.2
BUTTON_LENS = 9.6
BUTTON_PROUD = 9.0
BUTTON_TRAVEL = 2.5

# row 2 -- DIN rail.  One 4-module device carrying a single handle at its left
# (an RCCB) followed by 13 single-module MCBs: 14 handles over 17 modules,
# 306 mm of rail against 308.6 measured.
DIN_MODULE = 18.0
DIN_RCCB_MODULES = 4
DIN_MCB_COUNT = 13
DIN_V0, DIN_V1 = 114.0, 160.0  # device outline down-slope
DIN_TOGGLE_V0, DIN_TOGGLE_V1 = 137.9, 148.9
DIN_BASE_N = 2.0               # backplate the rail is screwed to
DIN_BODY_N = 23.0              # device front face above that backplate
DIN_TOGGLE_N = 5.0
DIN_TOGGLE_W = 9.0

# row 3 -- rotary cam switches, lever handle hanging down-slope
CAM_U = (45.0, 105.0, 165.0, 225.0, 285.0)
CAM_V = 244.5
CAM_BEZEL_D = 29.5
CAM_KNOB_D = 26.0
CAM_LEVER_L = 38.0
CAM_LEVER_W = 12.0
CAM_LEVER_OFFSET = 5.0         # lever centre below the knob axis

# row 4 -- IEC C14 inlets and the two load-break disconnects
IEC_U = 110.0
IEC_V = (325.0, 355.0)
IEC_FLANGE = (51.0, 24.0)      # measured 50.96 x 24.03
DISC_U = (212.5, 282.5)
DISC_V = 335.0
DISC_BOX = 64.0                # measured 63.35 x 63.7 yellow enclosure
DISC_KNOB_D = 51.5

# row 5 -- pull handle
HANDLE_U0, HANDLE_U1 = 114.8, 324.8
HANDLE_V = 420.0
HANDLE_BRACKET_U = 17.0
HANDLE_BRACKET_V = 29.6
HANDLE_BAR_V = 8.0
HANDLE_PROUD = 22.0

# left-side rail, standing off the console's left flank parallel to the face
RAIL_V0, RAIL_V1 = 35.0, 320.0
RAIL_SECTION = 25.0
RAIL_STANDOFF = 45.0

# --------------------------------------------------------------------------
# console body -- side profile (x forward from the console centre, z up from
# the ground), extruded across the full width.
# --------------------------------------------------------------------------
BODY_W = FACE_W + 2 * FRAME_W          # 398, matching the drawing's posts
BODY_DEPTH = 796.0
BODY_HEIGHT = 432.0
FACE_TOP_X, FACE_TOP_Z = 69.0, 404.0   # face top edge, from the isometric fit
RIDGE_X, RIDGE_Z = -193.0, 432.0

MATERIALS = {
    "panel_steel":   (0.72, 0.73, 0.75, 1.0),
    "panel_plate":   (0.82, 0.83, 0.84, 1.0),
    "panel_frame":   (0.66, 0.67, 0.69, 1.0),
    "marker_white":  (0.92, 0.92, 0.92, 1.0),
    "marker_black":  (0.02, 0.02, 0.02, 1.0),
    "button_bezel":  (0.30, 0.31, 0.33, 1.0),
    "button_green":  (0.05, 0.85, 0.12, 1.0),
    "din_body":      (0.84, 0.86, 0.94, 1.0),
    "din_toggle":    (0.36, 0.37, 0.40, 1.0),
    "cam_grey":      (0.44, 0.45, 0.47, 1.0),
    "socket_black":  (0.16, 0.16, 0.17, 1.0),
    "disc_yellow":   (0.95, 0.90, 0.05, 1.0),
    "disc_red":      (0.86, 0.06, 0.06, 1.0),
    "handle_alu":    (0.78, 0.80, 0.88, 1.0),
}


# --------------------------------------------------------------------------
# scene helpers
# --------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"


def make_materials():
    out = {}
    for name, rgba in MATERIALS.items():
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.45
        mat.diffuse_color = rgba
        out[name] = mat
    return out


def collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def add_object(name, mesh, col, mat):
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def box_mesh(name, sx, sy, sz, bevel=0.0, segments=2):
    """Axis-aligned box centred on its own origin, sizes in mm."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector((sx * MM, sy * MM, sz * MM)), verts=bm.verts)
    if bevel > 0.0:
        bmesh.ops.bevel(
            bm, geom=list(bm.verts) + list(bm.edges), offset=bevel * MM,
            segments=segments, profile=0.5, affect="EDGES", clamp_overlap=True,
        )
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def cyl_mesh(name, diameter, height, segments=48):
    """Cylinder about local +Z, centred on its own origin."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segments,
        radius1=diameter * MM / 2.0, radius2=diameter * MM / 2.0,
        depth=height * MM,
    )
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def prism_mesh(name, profile_xz, width):
    """Extrude a closed 2-D profile (list of (x, z) in mm) along Y."""
    half = width * MM / 2.0
    n = len(profile_xz)
    verts = [(x * MM, -half, z * MM) for x, z in profile_xz]
    verts += [(x * MM, half, z * MM) for x, z in profile_xz]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    return mesh


# --------------------------------------------------------------------------
# face frame:  world = O + u*e_u + v*e_v + n*e_n
# Local axes of every placed part are (x -> e_u, y -> -e_v, z -> e_n) so that
# the frame stays right handed; a part at face (u, v) is placed at local y = -v.
# --------------------------------------------------------------------------
_theta = math.radians(FACE_ANGLE_DEG)
E_U = Vector((0.0, 1.0, 0.0))
E_V = Vector((math.cos(_theta), 0.0, -math.sin(_theta)))
E_N = E_V.cross(E_U)
O_FACE = Vector((FACE_TOP_X * MM, -FACE_W * MM / 2.0, FACE_TOP_Z * MM))
FACE_ROT = Matrix((
    (E_U.x, -E_V.x, E_N.x),
    (E_U.y, -E_V.y, E_N.y),
    (E_U.z, -E_V.z, E_N.z),
)).to_4x4()


def face_world(u, v, n):
    return O_FACE + E_U * (u * MM) + E_V * (v * MM) + E_N * (n * MM)


def place(obj, u, v, n, spin_deg=0.0):
    """Put a part centred at face coordinates (u, v, n)."""
    rot = FACE_ROT
    if spin_deg:
        rot = rot @ Matrix.Rotation(math.radians(spin_deg), 4, "Z")
    obj.matrix_world = Matrix.Translation(face_world(u, v, n)) @ rot
    return obj


# --------------------------------------------------------------------------
# body
# --------------------------------------------------------------------------
def build_body(col, mats):
    face_bottom_x = FACE_TOP_X + FACE_H * math.cos(_theta)
    face_bottom_z = FACE_TOP_Z - FACE_H * math.sin(_theta)
    front_x = BODY_DEPTH / 2.0
    rear_x = -BODY_DEPTH / 2.0
    profile = [
        (front_x, 0.0),                       # front-bottom, on the ground
        (front_x - 28.0, face_bottom_z),      # plinth top / face bottom edge
        (FACE_TOP_X, FACE_TOP_Z),             # face top edge
        (RIDGE_X, RIDGE_Z),                   # ridge, back of the top face
        (rear_x, 0.0),                        # rear-bottom, on the ground
    ]
    body = add_object("panel_body", prism_mesh("panel_body", profile, BODY_W),
                      col, mats["panel_steel"])
    # sanity: the modelled face must still be 450 long at the stated angle
    run = math.hypot(face_bottom_x - FACE_TOP_X, FACE_TOP_Z - face_bottom_z)
    assert abs(run - FACE_H) < 0.5, run
    return body


def build_frame(col, mats):
    """Border bars and rounded corner posts around the 330 x 450 plate area."""
    parts = []
    half = FRAME_PROUD / 2.0
    for name, u, v, du, dv in (
        ("frame_left",   -FRAME_W / 2.0, FACE_H / 2.0, FRAME_W, FACE_H + 2 * FRAME_W),
        ("frame_right",  FACE_W + FRAME_W / 2.0, FACE_H / 2.0, FRAME_W, FACE_H + 2 * FRAME_W),
        ("frame_top",    FACE_W / 2.0, -FRAME_W / 2.0, FACE_W, FRAME_W),
        ("frame_bottom", FACE_W / 2.0, FACE_H + FRAME_W / 2.0, FACE_W, FRAME_W),
    ):
        mesh = box_mesh(name, du, dv, FRAME_PROUD, bevel=4.0, segments=3)
        parts.append(place(add_object(name, mesh, col, mats["panel_frame"]),
                           u, v, half))
    return parts


def build_subplates(col, mats):
    plates = []
    for idx, (row, u0, u1) in enumerate(SUBPLATES):
        v0, v1 = ROW_V[row], ROW_V[row + 1]
        du = (u1 - u0) - SEAM
        dv = (v1 - v0) - SEAM
        name = f"subplate_{idx}_r{row + 1}"
        mesh = box_mesh(name, du, dv, PLATE_T, bevel=0.4)
        plates.append(place(add_object(name, mesh, col, mats["panel_plate"]),
                            (u0 + u1) / 2.0, (v0 + v1) / 2.0, PLATE_T / 2.0))
        # M5 washer-head fixings, 7 mm in from each corner of the plate
        for su in (u0 + 7.0, u1 - 7.0):
            for sv in (v0 + 7.0, v1 - 7.0):
                sname = f"{name}_screw_{su:.0f}_{sv:.0f}"
                smesh = cyl_mesh(sname, 8.0, 2.0, segments=16)
                place(add_object(sname, smesh, col, mats["panel_frame"]),
                      su, sv, PLATE_T + 1.0)
    return plates


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
def build_markers(col, mats, ids):
    """50 x 50 tiles built cell by cell, so the pattern survives into 3MF."""
    cell = MARKER_MM / 7.0
    for (cu, cv), mid in zip(MARKER_POS, ids):
        bits = ARUCO_BITS[mid]
        base = box_mesh(f"aruco_{mid}_base", MARKER_MM, MARKER_MM, MARKER_PROUD)
        place(add_object(f"aruco_{mid}_base", base, col, mats["marker_white"]),
              cu, cv, PLATE_T + MARKER_PROUD / 2.0)
        for r, row in enumerate(bits):
            run_start = None
            for c in range(8):
                black = c < 7 and row[c] == "0"
                if black and run_start is None:
                    run_start = c
                elif not black and run_start is not None:
                    width = (c - run_start) * cell
                    u = cu - MARKER_MM / 2.0 + (run_start + (c - run_start) / 2.0) * cell
                    v = cv - MARKER_MM / 2.0 + (r + 0.5) * cell
                    nm = f"aruco_{mid}_r{r}c{run_start}"
                    mesh = box_mesh(nm, width, cell, 0.2)
                    place(add_object(nm, mesh, col, mats["marker_black"]),
                          u, v, PLATE_T + MARKER_PROUD + 0.1)
                    run_start = None


def build_buttons(col, mats):
    out = []
    for i, u in enumerate(BUTTON_U):
        base = PLATE_T
        bez = box_mesh(f"button_{i}_bezel", BUTTON_BEZEL, BUTTON_BEZEL, 4.0, bevel=0.6)
        place(add_object(f"button_{i}_bezel", bez, col, mats["button_bezel"]),
              u, BUTTON_V, base + 2.0)
        col_ = box_mesh(f"button_{i}_collar", BUTTON_COLLAR, BUTTON_COLLAR, 3.0, bevel=0.4)
        place(add_object(f"button_{i}_collar", col_, col, mats["button_bezel"]),
              u, BUTTON_V, base + 5.5)
        lens = box_mesh(f"button_{i}_lens", BUTTON_LENS, BUTTON_LENS, 2.0, bevel=0.3)
        obj = place(add_object(f"button_{i}_lens", lens, col, mats["button_green"]),
                    u, BUTTON_V, base + BUTTON_PROUD - 1.0)
        obj["travel_mm"] = BUTTON_TRAVEL
        out.append(obj)
    return out


def build_din_row(col, mats):
    total_modules = DIN_RCCB_MODULES + DIN_MCB_COUNT
    block_w = total_modules * DIN_MODULE
    u_start = FACE_W / 2.0 - block_w / 2.0
    v_mid = (DIN_V0 + DIN_V1) / 2.0
    dev_h = DIN_V1 - DIN_V0

    back = box_mesh("din_backplate", block_w + 8.0, dev_h + 6.0, DIN_BASE_N)
    place(add_object("din_backplate", back, col, mats["panel_frame"]),
          FACE_W / 2.0, v_mid, PLATE_T + DIN_BASE_N / 2.0)

    toggles = []
    widths = [DIN_RCCB_MODULES * DIN_MODULE] + [DIN_MODULE] * DIN_MCB_COUNT
    u = u_start
    for i, w in enumerate(widths):
        kind = "rccb" if i == 0 else f"mcb_{i - 1}"
        body = box_mesh(f"din_{kind}", w - 0.4, dev_h, DIN_BODY_N, bevel=0.5)
        place(add_object(f"din_{kind}", body, col, mats["din_body"]),
              u + w / 2.0, v_mid, PLATE_T + DIN_BASE_N + DIN_BODY_N / 2.0)
        # the handle sits at the left edge of the 4-module device, centred on
        # each single one -- exactly how the front view draws them
        tu = u + (DIN_MODULE / 2.0 if i == 0 else w / 2.0)
        tv = (DIN_TOGGLE_V0 + DIN_TOGGLE_V1) / 2.0
        tog = box_mesh(f"din_{kind}_toggle", DIN_TOGGLE_W,
                       DIN_TOGGLE_V1 - DIN_TOGGLE_V0, DIN_TOGGLE_N, bevel=0.8)
        obj = place(add_object(f"din_{kind}_toggle", tog, col, mats["din_toggle"]),
                    tu, tv, PLATE_T + DIN_BASE_N + DIN_BODY_N + DIN_TOGGLE_N / 2.0 - 1.0)
        obj["throw_mm"] = 6.0
        toggles.append(obj)
        u += w
    return toggles


def build_cam_switches(col, mats):
    levers = []
    for i, u in enumerate(CAM_U):
        base = PLATE_T
        bez = cyl_mesh(f"cam_{i}_bezel", CAM_BEZEL_D, 3.0)
        place(add_object(f"cam_{i}_bezel", bez, col, mats["cam_grey"]),
              u, CAM_V, base + 1.5)
        knob = cyl_mesh(f"cam_{i}_knob", CAM_KNOB_D, 10.0)
        place(add_object(f"cam_{i}_knob", knob, col, mats["cam_grey"]),
              u, CAM_V, base + 8.0)
        lev = box_mesh(f"cam_{i}_lever", CAM_LEVER_W, CAM_LEVER_L, 12.0, bevel=3.0, segments=3)
        # built about the knob axis so a spin_deg on the object rotates it
        lev.transform(Matrix.Translation(Vector((0.0, CAM_LEVER_OFFSET * MM, 0.0))))
        obj = place(add_object(f"cam_{i}_lever", lev, col, mats["cam_grey"]),
                    u, CAM_V, base + 15.0)
        obj["rotates_about"] = "face_normal"
        levers.append(obj)
    return levers


def build_sockets(col, mats):
    for i, v in enumerate(IEC_V):
        fw, fh = IEC_FLANGE
        flange = box_mesh(f"iec_{i}_flange", fw, fh, 2.0, bevel=1.5, segments=3)
        place(add_object(f"iec_{i}_flange", flange, col, mats["socket_black"]),
              IEC_U, v, PLATE_T + 1.0)
        boss = box_mesh(f"iec_{i}_boss", 28.0, 20.0, 5.0, bevel=2.0, segments=3)
        place(add_object(f"iec_{i}_boss", boss, col, mats["socket_black"]),
              IEC_U, v, PLATE_T + 4.5)
        for k, (du, dv) in enumerate(((-7.0, 3.0), (0.0, -4.0), (7.0, 3.0))):
            pin = box_mesh(f"iec_{i}_pin{k}", 2.0, 4.5, 3.0)
            place(add_object(f"iec_{i}_pin{k}", pin, col, mats["cam_grey"]),
                  IEC_U + du, v + dv, PLATE_T + 5.5)
        for du in (-19.5, 19.5):
            scr = cyl_mesh(f"iec_{i}_screw{du:.0f}", 4.0, 1.5, segments=12)
            place(add_object(f"iec_{i}_screw{du:.0f}", scr, col, mats["cam_grey"]),
                  IEC_U + du, v, PLATE_T + 2.5)


def build_disconnects(col, mats):
    knobs = []
    for i, u in enumerate(DISC_U):
        base = PLATE_T
        box = box_mesh(f"disc_{i}_enclosure", DISC_BOX, DISC_BOX, 6.0, bevel=3.0, segments=3)
        place(add_object(f"disc_{i}_enclosure", box, col, mats["disc_yellow"]),
              u, DISC_V, base + 3.0)
        knob = cyl_mesh(f"disc_{i}_knob", DISC_KNOB_D, 14.0)
        obj = place(add_object(f"disc_{i}_knob", knob, col, mats["disc_red"]),
                    u, DISC_V, base + 12.0)
        obj["rotates_about"] = "face_normal"
        knobs.append(obj)
        grip = box_mesh(f"disc_{i}_grip", 12.0, 40.0, 6.0, bevel=2.5, segments=3)
        place(add_object(f"disc_{i}_grip", grip, col, mats["disc_red"]),
              u, DISC_V + 5.0, base + 21.0)
    return knobs


def build_handle(col, mats):
    for u in (HANDLE_U0 + HANDLE_BRACKET_U / 2.0, HANDLE_U1 - HANDLE_BRACKET_U / 2.0):
        br = box_mesh(f"handle_bracket_{u:.0f}", HANDLE_BRACKET_U,
                      HANDLE_BRACKET_V, HANDLE_PROUD, bevel=2.0, segments=3)
        place(add_object(f"handle_bracket_{u:.0f}", br, col, mats["cam_grey"]),
              u, HANDLE_V, PLATE_T + HANDLE_PROUD / 2.0)
    bar_len = (HANDLE_U1 - HANDLE_BRACKET_U) - (HANDLE_U0 + HANDLE_BRACKET_U)
    bar = box_mesh("handle_bar", bar_len, HANDLE_BAR_V, 12.0, bevel=3.0, segments=4)
    place(add_object("handle_bar", bar, col, mats["handle_alu"]),
          (HANDLE_U0 + HANDLE_U1) / 2.0, HANDLE_V, PLATE_T + HANDLE_PROUD - 4.0)


def build_side_rail(col, mats):
    """Square tube stood off the console's left flank, parallel to the face."""
    length = RAIL_V1 - RAIL_V0
    v_mid = (RAIL_V0 + RAIL_V1) / 2.0
    u = -FRAME_W - RAIL_SECTION / 2.0 - 6.0
    tube = box_mesh("side_rail", RAIL_SECTION, length, RAIL_SECTION, bevel=2.0, segments=3)
    place(add_object("side_rail", tube, col, mats["handle_alu"]),
          u, v_mid, RAIL_STANDOFF)
    for v in (RAIL_V0 - 12.0, RAIL_V1 + 12.0):
        pad = box_mesh(f"side_rail_pad_{v:.0f}", RAIL_SECTION + 8.0, 24.0, 6.0, bevel=1.5)
        place(add_object(f"side_rail_pad_{v:.0f}", pad, col, mats["handle_alu"]),
              u, v, RAIL_STANDOFF)
    for v in (RAIL_V0 - 4.0, RAIL_V1 + 4.0):
        strut = box_mesh(f"side_rail_strut_{v:.0f}", 8.0, 16.0, RAIL_STANDOFF, bevel=1.0)
        place(add_object(f"side_rail_strut_{v:.0f}", strut, col, mats["panel_frame"]),
              -FRAME_W / 2.0 - 2.0, v, RAIL_STANDOFF / 2.0)


# --------------------------------------------------------------------------
def build(ids):
    reset_scene()
    mats = make_materials()
    cols = {n: collection(n) for n in
            ("body", "frame", "subplates", "markers", "buttons", "breakers",
             "cam_switches", "sockets", "disconnects", "handle", "side_rail")}

    build_body(cols["body"], mats)
    build_frame(cols["frame"], mats)
    build_subplates(cols["subplates"], mats)
    build_markers(cols["markers"], mats, ids)
    build_buttons(cols["buttons"], mats)
    build_din_row(cols["breakers"], mats)
    build_cam_switches(cols["cam_switches"], mats)
    build_sockets(cols["sockets"], mats)
    build_disconnects(cols["disconnects"], mats)
    build_handle(cols["handle"], mats)
    build_side_rail(cols["side_rail"], mats)

    for obj in bpy.data.objects:
        obj.select_set(False)
    return {
        "objects": len(bpy.data.objects),
        "marker_ids": list(ids),
        "face_mm": [FACE_W, FACE_H],
        "face_angle_deg": FACE_ANGLE_DEG,
        "body_mm": [BODY_W, BODY_DEPTH, BODY_HEIGHT],
    }


def marker_world_report():
    """Marker centres and the two report callouts, straight off the model."""
    pts = [face_world(u, v, PLATE_T + MARKER_PROUD) for u, v in MARKER_POS]
    return {
        "top_span_mm": (pts[1] - pts[0]).length / MM,
        "left_span_mm": (pts[2] - pts[0]).length / MM,
        "centres_m": [[round(c, 5) for c in p] for p in pts],
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--name", default="erc2026_maintenance_panel")
    ap.add_argument("--marker-ids", default=",".join(str(i) for i in MARKER_IDS_DEFAULT),
                    help="three ids from 11,13,14,15 for top-left/top-right/bottom-left")
    args = ap.parse_args(argv)

    ids = tuple(int(x) for x in args.marker_ids.split(","))
    if len(ids) != 3 or any(i not in ARUCO_BITS for i in ids):
        raise SystemExit(f"--marker-ids must be three of {sorted(ARUCO_BITS)}")

    out = pathlib.Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stats = build(ids)
    stats.update(marker_world_report())

    blend = out / f"{args.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    threemf = out / f"{args.name}.3mf"
    addon_utils.enable(THREEMF_ADDON, default_set=False, persistent=True)
    # 3MF's native unit here is the millimetre and the scene is metres, so the
    # exporter has to scale by 1000 for the file to come out life size.
    bpy.ops.export_mesh.threemf(
        filepath=str(threemf), use_selection=False, global_scale=1000.0,
    )

    stats["blend"] = str(blend)
    stats["3mf"] = str(threemf)
    print("PANEL_BUILD " + json.dumps(stats))


if __name__ == "__main__":
    main()
