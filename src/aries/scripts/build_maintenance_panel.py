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
    DIN devices    17.7 pitch (from the organisers' CAD)

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

The control modules themselves are not modelled here -- they are the organisers'
own CAD, pulled out of "Panel for Maintenance Tasks.zip" by
extract_panel_parts.py and imported from parts/*.stl.  That is also what settles
the rail: three 654747 four-pole blocks, each with ONE handle bar across all
four poles, plus two single 1mcb modules.  14 modules, 5 operating handles.
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
FACE_ANGLE_DEG = 33.11         # control face above horizontal, from the CAD

PLATE_T = 3.0                  # sub-plate thickness; components sit at n = 3
SEAM = 2.0                     # gap milled between adjacent sub-plates
FRAME_W = 30.0                 # side border: 330 face + 2 x 30 = the CAD's 390
FRAME_V = 43.62                # up/down border, whatever the CAD face has spare
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

# The markers are the ORIGINAL-library PNGs already in the model directory;
# each is 512 px with a 13.9% white quiet border, so the black envelope the
# report dimensions at 50 mm runs px 71..440.
ARUCO_IDS = (11, 13, 14, 15)
ARUCO_PNG_PREFIX = "aruco_orig_"
ARUCO_PNG_PX = 512.0
ARUCO_ENVELOPE_PX = (71, 440)

# row 1 -- square illuminated push buttons
BUTTON_U = (85.0, 125.0, 165.0, 205.0, 245.0)
BUTTON_V = 45.0
BUTTON_BEZEL = 19.2            # measured 19.1 outer
BUTTON_COLLAR = 15.2
BUTTON_LENS = 9.6
BUTTON_PROUD = 9.0
BUTTON_TRAVEL = 2.5

# row 2 -- DIN rail.  Fourteen individual single-pole MCBs, each its own device
# with its own toggle, laid out exactly as the front view draws them.  Reading
# the toggle band (v 139..148) off that view gives 14 runs 13.2 mm wide at:
#
#   20.92 | 92.72 111.02 129.03 146.91 164.93 182.81 200.97 218.98 236.86
#         | 254.88 272.90 290.92 308.94
#
# -- one breaker hard against the left end of the rail, a 71.8 mm empty stretch,
# then a run of 13 on an 18.02 mm pitch out to the right end.  The gaps between
# the 13 are 17.88..18.30, so the pitch is a clean 18.0-ish, not the CAD's 17.7.
# (The zip also ships a ganged four-pole block, `654747`; not used here.)
DIN_PITCH = 18.02
DIN_LEFT_U = 20.9                      # the lone breaker at the left end
DIN_GROUP_FIRST_U = 92.7               # first of the run of 13
DIN_GROUP_COUNT = 13
DIN_RAIL_U0, DIN_RAIL_U1 = 12.2, 317.6  # bluish rail measured 12.62 .. 317.24
DIN_V0, DIN_V1 = 114.0, 160.0          # the 45 mm nose that shows through
DIN_PROUD = 15.0                       # toggle tip above the sub-plate
DIN_BLANK_N = 11.8                     # blanking strip, flush with the module noses
# The front view's toggle band centres on v = 143.4, but the organisers' 1mcb
# part carries its handle ~11 mm up-slope of that once the module is placed by
# its housing. Report placement wins for the module; the handle's own hinge and
# contact point are measured off the part (`outer_face_centre`), because an axis
# that misses the handle swings it through the console instead of rocking it.
DIN_TOGGLE_PIVOT_DEPTH = 9.0           # hinge below the toggle's outer face

# row 3 -- rotary cam switches, lever handle hanging down-slope
CAM_U = (45.0, 105.0, 165.0, 225.0, 285.0)
CAM_V = 244.5
CAM_PROUD = 25.0
# The CAD part is modelled with its lever at 48 deg off the across axis; the
# front view has every lever hanging straight down-slope, so spin it back.
CAM_SPIN_DEG = -42.0
# The CAD puts these five on a 55 mm pitch, not the 60 measured off the front
# view; the drawing's 60 is used because that is what the report specifies and
# it reads clean (60.03 / 59.83 / 60.11 / 59.97, no drift).

# row 4 -- IEC C14 inlets and the two load-break disconnects
IEC_U = 110.0
IEC_V = (325.0, 355.0)                 # 30 mm pitch per the front view
IEC_PROUD = 5.0
DISC_U = (212.5, 282.5)
DISC_V = 335.0
DISC_PROUD = 30.0

# row 5 -- pull handle
HANDLE_U0, HANDLE_U1 = 114.8, 324.8
HANDLE_V = 420.0
HANDLE_BRACKET_U = 17.0
HANDLE_BRACKET_V = 29.6
HANDLE_BAR_V = 8.0
HANDLE_PROUD = 22.0

# left-side rail, standing off the console's left flank parallel to the face
RAIL_V0, RAIL_V1 = 35.0, 283.0     # measured 35.2 .. 283.4 down-slope
RAIL_U = -65.4                     # measured -76.6 .. -54.1 across
RAIL_SECTION = 22.5
RAIL_STANDOFF = 45.0

# --------------------------------------------------------------------------
# console body -- side profile (x forward from the console centre, z up from
# the ground), extruded across the full width.
# --------------------------------------------------------------------------
BODY_W = FACE_W + 2 * FRAME_W          # 390
BODY_DEPTH = 490.0
BODY_HEIGHT = 1000.0
BODY_TOP_FLAT = 40.0                    # flat behind the face's high edge
# Densely sampling the organisers' console gives a wedge: a vertical back face,
# a short flat, then the control face falling forward at 33.11 deg, a vertical
# front, and an underside rising back at the same 33.11 deg.  1000 mm tall at
# the back, 413 at the front.  Placing the 330 x 450 plate centred on that face
# puts the marker row at z = 957 and the lower marker at z = 749, against 960
# and 752 in the CAD -- which is the check that this is the right console.
FACE_TOP_X, FACE_TOP_Z = -168.47, 976.17

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


def taper_mesh(name, w_base, w_tip, length, thickness):
    """Flat tapered paddle in local XY, extruded along Z; base at -Y."""
    hb, ht, hl, hz = w_base * MM / 2, w_tip * MM / 2, length * MM / 2, thickness * MM / 2
    face = [(-hb, -hl), (hb, -hl), (ht, hl), (-ht, hl)]
    verts = [(x, y, -hz) for x, y in face] + [(x, y, hz) for x, y in face]
    faces = [(3, 2, 1, 0), (4, 5, 6, 7)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append((i, j, j + 4, i + 4))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    return mesh


def import_part(stem, parts_dir, tag="body"):
    """Load one of the organisers' extracted modules.

    `extract_panel_parts.py` writes them in the face frame (x = u across,
    y = v down-slope, z = n out of the face). This builder's local frame has
    +y up-slope, so the mesh is mirrored in y and its winding reversed to keep
    the normals outward.
    """
    name = f"{stem}.stl" if tag == "body" else f"{stem}_{tag}.stl"
    path = pathlib.Path(parts_dir) / name
    if not path.exists():
        raise SystemExit(f"missing CAD part {path} -- run extract_panel_parts.py first")
    before = set(bpy.data.objects)
    # global_scale lands on the *object*, and only the mesh survives this
    # function, so import at 1:1 and fold the mm -> m conversion into the
    # same bmesh pass as the mirror.
    bpy.ops.wm.stl_import(filepath=str(path), global_scale=1.0,
                          forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.data.objects if o not in before]
    if len(new) != 1:
        raise SystemExit(f"{path} imported as {len(new)} objects")
    obj = new[0]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.scale(bm, vec=Vector((MM, -MM, MM)), verts=bm.verts)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


def part_copy(name, mesh, col, mat):
    obj = bpy.data.objects.new(name, mesh.copy())
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


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
# Row 1 of the report's front view -- the marker pair and the push buttons --
# sits at the LOW, front edge of this console, not the high one.  The CAD is
# unambiguous about it: the marker pair is at z = 0.7523 and the lone marker at
# 0.9599, and by height the rows run MCB < cam < sockets < handle going back and
# up.  So the drawing's "down the page" runs UP the slope, away from the rover.
# Getting this backwards mirrors the whole face and is what the first cut did.
E_U = Vector((0.0, -1.0, 0.0))                              # across, drawing +u
E_V = Vector((-math.cos(_theta), 0.0, math.sin(_theta)))    # drawing +v, up-slope
E_N = E_V.cross(E_U)                                        # out of the face
UP_SLOPE = Vector(E_V)          # physically back and up; the flick direction
# origin = the plate's front-low corner on the +Y side
O_FACE = Vector((
    (FACE_TOP_X + FACE_H * math.cos(_theta)) * MM,
    FACE_W * MM / 2.0,
    (FACE_TOP_Z - FACE_H * math.sin(_theta)) * MM,
))
FACE_ROT = Matrix((
    (E_U.x, -E_V.x, E_N.x),
    (E_U.y, -E_V.y, E_N.y),
    (E_U.z, -E_V.z, E_N.z),
)).to_4x4()


# Every operable control, filled in as the rows are built.  Index order mirrors
# the model this replaces so config/panel_tasks.yaml keeps meaning: mcb_*,
# rotary_switch_* and rotary_control_switch_* run in decreasing world Y,
# push_button_* in increasing world Y.
CONTROLS = []


def register_control(name, kind, action, objects, u, v,
                     pivot_n, surface_n, axis, travel, limits, grip):
    CONTROLS.append(dict(name=name, kind=kind, action=action, objects=objects,
                         u=u, v=v, pivot_n=pivot_n, surface_n=surface_n,
                         axis=axis, travel=travel, limits=limits, grip=grip))


def face_world(u, v, n):
    return O_FACE + E_U * (u * MM) + E_V * (v * MM) + E_N * (n * MM)


def face_of(point):
    """Inverse of `face_world`: a world point as face coordinates (u, v, n) mm."""
    d = Vector(point) - O_FACE
    return Vector((d.dot(E_U), d.dot(E_V), d.dot(E_N))) / MM


def outer_face_centre(obj, skin=3.0):
    """Face coordinates (v, n) of the outermost face of a placed part.

    Where a moving part is hinged has to be read off the part, not assumed from
    where the part was placed: a CAD module is placed by its housing and its
    handle can sit well off that centre.
    """
    points = [face_of(obj.matrix_world @ vert.co) for vert in obj.data.vertices]
    outer_n = max(p.z for p in points)
    outer = [p.y for p in points if p.z > outer_n - skin]
    return sum(outer) / len(outer), outer_n


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
    hinge_x = rear_x + BODY_TOP_FLAT
    drop = math.tan(_theta) * (front_x - hinge_x)
    profile = [
        (rear_x, 0.0),                        # back-bottom
        (rear_x, BODY_HEIGHT),                # back-top
        (hinge_x, BODY_HEIGHT),               # high edge of the control face
        (front_x, BODY_HEIGHT - drop),        # low edge, operator side
        (front_x, drop),                      # front-bottom
        (hinge_x, 0.0),                       # underside, back at the same angle
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
        ("frame_left",   -FRAME_W / 2.0, FACE_H / 2.0, FRAME_W, FACE_H + 2 * FRAME_V),
        ("frame_right",  FACE_W + FRAME_W / 2.0, FACE_H / 2.0, FRAME_W, FACE_H + 2 * FRAME_V),
        ("frame_top",    FACE_W / 2.0, -FRAME_V / 2.0, FACE_W, FRAME_V),
        ("frame_bottom", FACE_W / 2.0, FACE_H + FRAME_V / 2.0, FACE_W, FRAME_V),
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
        # M5 washer-head fixings 7 mm in from each corner, plus a mid-span pair
        # on any plate wide enough for the drawing to show one
        cols_u = [u0 + 7.0, u1 - 7.0]
        if (u1 - u0) > 150.0:
            cols_u.insert(1, (u0 + u1) / 2.0)
        for su in cols_u:
            for sv in (v0 + 7.0, v1 - 7.0):
                sname = f"{name}_screw_{su:.0f}_{sv:.0f}"
                smesh = cyl_mesh(sname, 8.0, 2.0, segments=16)
                place(add_object(sname, smesh, col, mats["panel_frame"]),
                      su, sv, PLATE_T + 1.0)
    return plates


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
def marker_material(mid, aruco_dir):
    """Material carrying the real ArUco PNG, packed into the .blend."""
    name = f"aruco_{mid}"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    path = pathlib.Path(aruco_dir) / f"{ARUCO_PNG_PREFIX}{mid}.png"
    if not path.exists():
        raise SystemExit(f"missing marker image {path}")
    img = bpy.data.images.load(str(path))
    img.pack()                              # keep the .blend self-contained
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.55
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"           # keep the cell edges hard
    tex.extension = "EXTEND"
    tex.location = (-400, 0)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)
    return mat


def marker_mesh(name, uv0, uv1):
    """Thin tile whose +n face samples the crop [uv0, uv1] of the marker PNG."""
    mesh = box_mesh(name, MARKER_MM, MARKER_MM, MARKER_PROUD)
    uv = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        # the face pointing out of the panel is the one to map; everything else
        # is an edge nobody sees, so it gets pinned to a corner of the crop
        front = poly.normal.z > 0.5
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            if front:
                fu = (co.x / (MARKER_MM * MM)) + 0.5
                fv = (co.y / (MARKER_MM * MM)) + 0.5
                uv.data[li].uv = (uv0 + (uv1 - uv0) * fu, uv0 + (uv1 - uv0) * fv)
            else:
                uv.data[li].uv = (uv0, uv0)
    return mesh


def build_markers(col, mats, ids, aruco_dir):
    """The three 50 x 50 markers, as the organisers' own ArUco images.

    The PNGs carry a 13.9% white quiet border (the black envelope runs px
    71..440 of 512), and the report's 50 mm is the *black square*, so the tile
    samples only the envelope and lets the light sub-plate around it act as the
    quiet zone -- which is exactly how the front view draws them.
    """
    uv0 = ARUCO_ENVELOPE_PX[0] / ARUCO_PNG_PX
    uv1 = (ARUCO_ENVELOPE_PX[1] + 1) / ARUCO_PNG_PX
    for (cu, cv), mid in zip(MARKER_POS, ids):
        mesh = marker_mesh(f"aruco_{mid}", uv0, uv1)
        obj = bpy.data.objects.new(f"aruco_{mid}", mesh)
        obj.data.materials.append(marker_material(mid, aruco_dir))
        col.objects.link(obj)
        place(obj, cu, cv, PLATE_T + MARKER_PROUD / 2.0)
        obj["aruco_id"] = mid
        obj["dictionary"] = "DICT_ARUCO_ORIGINAL"
        obj["size_mm"] = MARKER_MM


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
        register_control(f"push_button_{len(BUTTON_U) - 1 - i}", "button", "press", [obj],
                         u, BUTTON_V,
                         pivot_n=base + BUTTON_PROUD - 8.0,
                         surface_n=base + BUTTON_PROUD,
                         axis="push", travel=BUTTON_TRAVEL / 1000.0,
                         limits=(0.0, BUTTON_TRAVEL / 1000.0), grip=False)
    return out


def build_din_row(col, mats, parts_dir):
    """Fourteen individual single-pole MCBs, each its own device with its own
    toggle, placed where the front view puts them: one at the left end of the
    rail, then a run of 13 out to the right end, with an empty stretch between.

    Geometry is the organisers' `1mcb` part (17.7 wide, 84.8 tall, stepping down
    to a 45 mm nose in its front 10 mm -- the band that shows through the
    escutcheon).  The empty stretch is filled with blanking strip, which is what
    the drawing shows there and what a real rail would carry.
    """
    body = import_part("mcb_single", parts_dir)
    handle = import_part("mcb_single", parts_dir, "handle")

    centres = [DIN_LEFT_U] + [DIN_GROUP_FIRST_U + i * DIN_PITCH
                              for i in range(DIN_GROUP_COUNT)]
    n_mount = PLATE_T + DIN_PROUD
    v = (DIN_V0 + DIN_V1) / 2.0
    handles = []
    for i, u in enumerate(centres):
        place(part_copy(f"mcb_{i:02d}", body, col, mats["din_body"]), u, v, n_mount)
        obj = place(part_copy(f"mcb_{i:02d}_toggle", handle, col, mats["din_toggle"]),
                    u, v, n_mount)
        obj["poles"] = 1
        obj["throw_mm"] = 6.0
        handles.append(obj)

    # index right to left, so mcb_13 stays the lone breaker at the left end
    for j, (u, obj) in enumerate(sorted(zip(centres, handles),
                                        key=lambda p: p[0])):
        # Hinge the toggle under ITSELF. The 1mcb handle is not centred on its
        # housing - placed by the housing it lands ~11 mm up-slope of the
        # drawing's toggle band - so taking the hinge from DIN_TOGGLE_V put the
        # axis 14 mm off the handle. The handle then ORBITED that axis instead
        # of rocking: at the +0.4 ON end it swung 5 mm INTO the console face
        # (n 16.9 -> 11.9) rather than tipping up-slope, i.e. an operated
        # breaker sank into the panel. The same constant also aimed the
        # fingertip at bare housing 11 mm below the handle.
        toggle_v, toggle_n = outer_face_centre(obj)
        register_control(f"mcb_{j}", "breaker", "flick", [obj], u,
                         toggle_v,
                         pivot_n=toggle_n - DIN_TOGGLE_PIVOT_DEPTH,
                         surface_n=toggle_n,
                         axis=BREAKER_AXIS, travel=0.012,
                         limits=(-0.4, 0.4), grip=False)

    # blanking strip over the empty rail between the lone breaker and the run
    gap0 = DIN_LEFT_U + DIN_PITCH / 2.0
    gap1 = DIN_GROUP_FIRST_U - DIN_PITCH / 2.0
    blank = box_mesh("din_blank", gap1 - gap0, DIN_V1 - DIN_V0, DIN_BLANK_N, bevel=0.5)
    place(add_object("din_blank", blank, col, mats["din_body"]),
          (gap0 + gap1) / 2.0, v, PLATE_T + DIN_BLANK_N / 2.0)

    # escutcheon: only the 45 mm nose shows, so the plate is slotted rather than
    # open -- two bars, above and below, spanning the rail
    span = DIN_RAIL_U1 - DIN_RAIL_U0
    for tag, vv in (("top", DIN_V0 - 4.0), ("bottom", DIN_V1 + 4.0)):
        bar = box_mesh(f"din_escutcheon_{tag}", span + 8.0, 8.0, 6.0, bevel=1.0)
        place(add_object(f"din_escutcheon_{tag}", bar, col, mats["panel_frame"]),
              (DIN_RAIL_U0 + DIN_RAIL_U1) / 2.0, vv, PLATE_T + 3.0)
    return handles


def build_cam_switches(col, mats, parts_dir):
    """Five Rotary Switch modules on the drawing's 60 mm pitch."""
    mesh = import_part("cam_switch", parts_dir)
    out = []
    for i, u in enumerate(CAM_U):
        obj = place(part_copy(f"cam_switch_{i}", mesh, col, mats["cam_grey"]),
                    u, CAM_V, PLATE_T + CAM_PROUD, spin_deg=CAM_SPIN_DEG)
        obj["rotates_about"] = "face_normal"
        out.append(obj)
    for j, (u, obj) in enumerate(sorted(zip(CAM_U, out), key=lambda p: p[0])):
        register_control(f"rotary_switch_{j}", "rotary", "turn", [obj],
                         u, CAM_V, pivot_n=PLATE_T,
                         surface_n=PLATE_T + CAM_PROUD,
                         axis="normal", travel=math.radians(60.0),
                         limits=(-math.pi / 2, math.pi / 2), grip=True)
    return out


def build_sockets(col, mats, parts_dir):
    """Two 801954 IEC C14 appliance inlets, 30 mm apart as the drawing has them."""
    mesh = import_part("iec_inlet", parts_dir)
    for i, v in enumerate(IEC_V):
        place(part_copy(f"iec_inlet_{i}", mesh, col, mats["socket_black"]),
              IEC_U, v, PLATE_T + IEC_PROUD)


def build_disconnects(col, mats, parts_dir):
    """Two rotary control switches -- the red knob on the yellow enclosure."""
    body = import_part("disconnect", parts_dir)
    bezel = import_part("disconnect", parts_dir, "bezel")
    knob = import_part("disconnect", parts_dir, "handle")
    out = []
    for i, u in enumerate(DISC_U):
        place(part_copy(f"disconnect_{i}", body, col, mats["cam_grey"]),
              u, DISC_V, PLATE_T + DISC_PROUD)
        place(part_copy(f"disconnect_{i}_bezel", bezel, col, mats["disc_yellow"]),
              u, DISC_V, PLATE_T + DISC_PROUD)
        obj = place(part_copy(f"disconnect_{i}_knob", knob, col, mats["disc_red"]),
                    u, DISC_V, PLATE_T + DISC_PROUD)
        obj["rotates_about"] = "face_normal"
        out.append(obj)
    for j, (u, obj) in enumerate(sorted(zip(DISC_U, out), key=lambda p: p[0])):
        register_control(f"rotary_control_switch_{j}", "disconnect", "turn",
                         [obj], u, DISC_V, pivot_n=PLATE_T,
                         surface_n=PLATE_T + DISC_PROUD,
                         axis="normal", travel=math.pi / 2,
                         limits=(0.0, math.pi / 2), grip=True)
    return out


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
    u = RAIL_U
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

# --------------------------------------------------------------------------
# Gazebo model: meshes, model.sdf and the runtime task table
# --------------------------------------------------------------------------
SDF_MODEL_NAME = "maintenance_panel"
SDF_MESH_DIR = "meshes"
SDF_STANDOFF = 0.06
CONTROL_MASS = 0.05
BODY_MASS = 40.0

AXIS_LOCAL = {"across": (1.0, 0.0, 0.0),      # e_u; +ve throws a part up-slope
              "across_reversed": (-1.0, 0.0, 0.0),   # +ve throws it down-slope
              "normal": (0.0, 0.0, 1.0),      # e_n, out of the face
              "push":   (0.0, 0.0, -1.0)}     # into the face

# Which way a breaker's handle travels to reach ON, in face coordinates.
#
# On the drawing that is up-slope, and if the console ever stands on its base
# that is also up in the world. The worlds mount it face-out but rolled: rows
# run buttons(high) -> MCB -> cam -> sockets(low), i.e. the face is upside down
# and up-slope points 57 deg BELOW horizontal. The user's call (2026-08-18) is
# that the mount stays and the breakers follow the world: lever UP is ON, lever
# DOWN is OFF, as on any real MCB you look at. So ON is DOWN-slope on the face,
# which is upward in the world on this mount, and the breaker joints are given
# the reversed axis so a positive joint angle is also the ON direction.
#
# If the mount is ever turned right way up, flip these two back together.
BREAKER_ON_IS_UP_SLOPE = False
BREAKER_AXIS = "across" if BREAKER_ON_IS_UP_SLOPE else "across_reversed"


def _export_glb(objects, path, basis=None):
    """Write `objects` to a GLB, optionally re-expressed in `basis`."""
    for obj in bpy.data.objects:
        obj.select_set(False)
    dups = []
    inv = basis.inverted() if basis is not None else None
    for obj in objects:
        dup = obj.copy()
        dup.data = obj.data.copy()
        bpy.context.scene.collection.objects.link(dup)
        dup.matrix_world = (inv @ obj.matrix_world) if inv is not None \
            else obj.matrix_world
        dup.select_set(True)
        dups.append(dup)
    bpy.context.view_layer.objects.active = dups[0]
    # export_yup=False to match the other Gazebo meshes in this repo, which are
    # authored Z-up; flipping here would lay every control on its side.
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB",
                              use_selection=True, export_apply=True,
                              export_yup=False)
    for dup in dups:
        bpy.data.objects.remove(dup, do_unlink=True)


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def export_gazebo_model(out_dir, ids):
    """Write model.sdf + meshes + panel_task.json for the model just built."""
    mesh_dir = pathlib.Path(out_dir) / SDF_MESH_DIR
    mesh_dir.mkdir(parents=True, exist_ok=True)
    uri = f"model://{SDF_MODEL_NAME}/{SDF_MESH_DIR}"

    moving = {o for c in CONTROLS for o in c["objects"]}
    static = [o for o in bpy.data.objects
              if o.type == "MESH" and o not in moving]
    _export_glb(static, mesh_dir / "panel_body.glb")

    # collision is the bare console wedge: DART only needs the silhouette, and
    # a convex-per-switch collision would cost far more than it buys
    body = bpy.data.objects["panel_body"]
    _export_glb([body], mesh_dir / "panel_body_collision.glb")

    # emit in the order config/panel_tasks.yaml lists them: the operator runs
    # enabled controls top to bottom through this table
    order = {"mcb": 0, "rotary_control_switch": 1, "rotary_switch": 2,
             "push_button": 3}
    CONTROLS.sort(key=lambda c: (order[c["name"].rsplit("_", 1)[0]],
                                 int(c["name"].rsplit("_", 1)[1])))

    normal = [round(x, 5) for x in E_N]
    up_slope = [round(x, 5) for x in UP_SLOPE]
    links, joints, table = [], [], []
    for ctl in CONTROLS:
        name = ctl["name"]
        pivot = face_world(ctl["u"], ctl["v"], ctl["pivot_n"])
        basis = Matrix.Translation(pivot) @ FACE_ROT
        _export_glb(ctl["objects"], mesh_dir / f"{name}.glb", basis)

        rpy = basis.to_euler("XYZ")
        axis = AXIS_LOCAL[ctl["axis"]]
        jtype = "prismatic" if ctl["axis"] == "push" else "revolute"
        links.append(f"""  <link name='{name}'>
    <pose>{_fmt(pivot)} {_fmt(rpy)}</pose>
    <inertial><mass>{CONTROL_MASS}</mass>
      <inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>
               <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    <visual name='{name}_visual'>
      <geometry><mesh><uri>{uri}/{name}.glb</uri></mesh></geometry>
    </visual>
    <collision name='{name}_collision'>
      <geometry><mesh><uri>{uri}/{name}.glb</uri></mesh></geometry>
    </collision>
  </link>""")
        joints.append(f"""  <joint name='{name}_joint' type='{jtype}'>
    <parent>body</parent><child>{name}</child>
    <axis>
      <xyz>{_fmt(axis)}</xyz>
      <limit><lower>{ctl['limits'][0]:.6g}</lower><upper>{ctl['limits'][1]:.6g}</upper>
             <effort>1.0</effort><velocity>8.0</velocity></limit>
      <dynamics><damping>0.08</damping><friction>0.12</friction></dynamics>
    </axis>
  </joint>""")

        surface = face_world(ctl["u"], ctl["v"], ctl["surface_n"])
        jaw = up_slope if ctl["kind"] != "button" else [round(x, 5) for x in E_U]
        entry = {
            "name": name, "kind": ctl["kind"], "action": ctl["action"],
            "position": [round(x, 5) for x in surface],
            "pivot_position": [round(x, 5) for x in pivot],
            "surface_offset_m": round((ctl["surface_n"] - ctl["pivot_n"]) * MM, 5),
            "approach": normal,
            "joint_axis": [round(x, 5) for x in (FACE_ROT.to_3x3() @ Vector(axis))],
            "jaw_axis": jaw,
            "travel": round(ctl["travel"], 5),
            "grip": ctl["grip"],
            "joint": f"{name}_joint",
            "limits": [round(x, 5) for x in ctl["limits"]],
            "model_name": name,
        }
        if ctl["kind"] == "breaker":
            # The model states which way ON is; nothing downstream may guess it
            # from a frame convention. See BREAKER_ON_IS_UP_SLOPE for why it is
            # down-slope on this mount.
            entry["target_state"] = "on"
            entry["on_direction"] = (up_slope if BREAKER_ON_IS_UP_SLOPE else
                                     [round(-x, 5) for x in UP_SLOPE])
            entry["motion_direction"] = ("up-slope" if BREAKER_ON_IS_UP_SLOPE
                                         else "down-slope")
        table.append(entry)

    markers = []
    for (mu_, mv_), mid in zip(MARKER_POS, ids):
        centre = face_world(mu_, mv_, PLATE_T + MARKER_PROUD)
        markers.append({"id": mid, "position": [round(x, 5) for x in centre],
                        "pitch": round(math.radians(FACE_ANGLE_DEG), 5),
                        "size": MARKER_MM * MM})

    sdf = f"""<?xml version='1.0'?>
<!-- GENERATED by scripts/build_maintenance_panel.py. Do not hand-edit.

     Layout from "[ERC 2026] MY Update Report Rev.1.pdf"; control modules are
     the organisers' own CAD out of "Panel for Maintenance Tasks.zip", see
     ERC2026_PANEL_MODEL.md next to this file.

     {BODY_W / 1000:.3f} x {BODY_DEPTH / 1000:.3f} x {BODY_HEIGHT / 1000:.3f} m, base on the ground,
     console face {FACE_ANGLE_DEG:.0f} deg above horizontal, model front is its own +X.
     Collision is the bare console wedge, not the switch detail.

     Markers top-left/top-right/bottom-left: {'/'.join(str(i) for i in ids)}.
     There is no bottom-right marker slot.

     {len(CONTROLS)} operable controls, all free joints the gripper moves directly -
     no position controller fights it and joint friction holds a control where
     the arm leaves it. Read them on /{SDF_MODEL_NAME}/joint_states. -->
<sdf version='1.10'>
<model name='{SDF_MODEL_NAME}'>
  <!-- Not <static>: a static model cannot carry joints, so the body is welded
       to the world instead. Same immobility, but the switches still move. -->
  <joint name='anchor' type='fixed'><parent>world</parent><child>body</child></joint>
  <link name='body'>
    <inertial><mass>{BODY_MASS}</mass>
      <inertia><ixx>4.0</ixx><iyy>4.0</iyy><izz>2.0</izz>
               <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    <collision name='body_collision'>
      <geometry><mesh><uri>{uri}/panel_body_collision.glb</uri></mesh></geometry>
      <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface>
    </collision>
    <visual name='body_visual'>
      <geometry><mesh><uri>{uri}/panel_body.glb</uri></mesh></geometry>
    </visual>
  </link>
{chr(10).join(links)}
{chr(10).join(joints)}
</model>
</sdf>
"""
    (pathlib.Path(out_dir) / "model.sdf").write_text(sdf)

    task = {
        "frame": SDF_MODEL_NAME,
        # panel_alignment._basis() reads this as normal = (sin p, 0, cos p),
        # i.e. the angle of the face normal off vertical, which is the same
        # number as the face's angle off horizontal -- not its complement.
        "console_pitch": round(math.radians(FACE_ANGLE_DEG), 5),
        "console_normal": normal,
        "console_up_slope": up_slope,
        "standoff": SDF_STANDOFF,
        "markers": markers,
        "controls": table,
    }
    (pathlib.Path(out_dir) / "panel_task.json").write_text(
        json.dumps(task, indent=1) + "\n")
    return {"sdf_controls": len(table), "meshes": len(list(mesh_dir.glob("*.glb")))}


def build(ids, parts_dir, aruco_dir):
    CONTROLS.clear()
    reset_scene()
    mats = make_materials()
    cols = {n: collection(n) for n in
            ("body", "frame", "subplates", "markers", "buttons", "breakers",
             "cam_switches", "sockets", "disconnects", "handle", "side_rail")}

    build_body(cols["body"], mats)
    build_frame(cols["frame"], mats)
    build_subplates(cols["subplates"], mats)
    build_markers(cols["markers"], mats, ids, aruco_dir)
    build_buttons(cols["buttons"], mats)
    build_din_row(cols["breakers"], mats, parts_dir)
    build_cam_switches(cols["cam_switches"], mats, parts_dir)
    build_sockets(cols["sockets"], mats, parts_dir)
    build_disconnects(cols["disconnects"], mats, parts_dir)
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
    ap.add_argument("--no-sdf", action="store_true",
                    help="skip model.sdf / meshes / panel_task.json")
    ap.add_argument("--aruco-dir", default=None,
                    help="folder holding aruco_orig_<id>.png "
                         "(default: --out-dir)")
    ap.add_argument("--parts-dir", default=None,
                    help="CAD modules from extract_panel_parts.py "
                         "(default: a 'parts' folder beside --out-dir)")
    args = ap.parse_args(argv)

    ids = tuple(int(x) for x in args.marker_ids.split(","))
    if len(ids) != 3 or any(i not in ARUCO_IDS for i in ids):
        raise SystemExit(f"--marker-ids must be three of {sorted(ARUCO_IDS)}")

    out = pathlib.Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    parts_dir = pathlib.Path(args.parts_dir).expanduser().resolve() if args.parts_dir \
        else out / "parts"
    aruco_dir = pathlib.Path(args.aruco_dir).expanduser().resolve() \
        if args.aruco_dir else out
    stats = build(ids, parts_dir, aruco_dir)
    stats.update(marker_world_report())
    stats["parts_dir"] = str(parts_dir)

    if not args.no_sdf:
        stats.update(export_gazebo_model(out, ids))

    blend = out / f"{args.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    threemf = out / f"{args.name}.3mf"
    addon_utils.enable(THREEMF_ADDON, default_set=False, persistent=True)
    # The exporter already converts the metre-based scene to 3MF's millimetres
    # off scene.unit_settings, so global_scale stays 1.0 -- passing 1000 here
    # applies the conversion twice and ships a 798 metre console.  The plain
    # single-file layout (not Orca's per-object one) is what generic 3MF
    # readers expect.
    bpy.ops.export_mesh.threemf(
        filepath=str(threemf), use_selection=False,
        global_scale=1.0, use_orca_format="STANDARD",
    )

    stats["blend"] = str(blend)
    stats["3mf"] = str(threemf)
    print("PANEL_BUILD " + json.dumps(stats))


if __name__ == "__main__":
    main()
