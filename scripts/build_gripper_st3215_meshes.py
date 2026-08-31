#!/usr/bin/env python3
"""Split the ST3215 rack-and-pinion gripper CAD export into per-link URDF meshes.

Source assets (repo root):
    new_gripper.glb                     whole assembly, one flat list of solids
                                        in one frame, wearing the bucket scoops
    Maintenance Gripper Finger L.stl    the alternative fingertip pair, exported
    Maintenance Gripper Finger R.stl    per part with no assembly context

The fingertips are SWAPPABLE and the xacro branches on finger_type, so this
script emits a full set of numbers per pair rather than assuming the one the
assembly happens to be wearing.  The loose pairs carry no assembly frame at all;
fit_fingertip_pair() lands them by matching their bolt face against the bucket's,
which also derives which side each part belongs on.  See its docstring - getting
that backwards is close to invisible.

The 2026-08-28 19:28 re-export merged the original 20 solids down to 8 BY PRINT
COLOUR, not by link - the same trap igus.glb set.  Nothing moved, but the
pinion is now inside the ``body3756496`` node together with the servo output
boss, and the ``IGUS Mount 2`` end cap is now part of ``IGUS Mount``.  The
grouping below re-cuts them at the link boundaries; do not assume one node is
one link.

This is the SECONDARY gripper (``gripper_type:=st3215``).  It bolts to the same
igus ReBeL tool flange as the four-bar ``gripper_v2``, but nothing inside is
shared: a single ST3215 bus servo direct-drives a 21-tooth pinion that meshes
with two opposed racks, so the jaws TRANSLATE along +/-X instead of swinging on
bars.

Run from the repo root:  python3 scripts/build_gripper_st3215_meshes.py
Add --report to skip writing and only print the measurements.

Everything ``aries/urdf/gripper_st3215.xacro`` quotes is printed here, so the
xacro can be regenerated rather than trusted.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "src", "aries", "meshes", "gripper_st3215")
SRC = os.path.join(REPO, "new_gripper.glb")

# --------------------------------------------------------------------------
# Swappable fingertips
# --------------------------------------------------------------------------
# The bucket scoops come out of the assembly GLB above, but every OTHER
# fingertip arrives as its own pair of STLs exported from the part file, each
# on its own CAD datum with no assembly context.  Those are listed here.
#
# The exported names are kept EXACTLY as CAD writes them, spaces and all, so a
# re-export drops in without a rename step that someone will forget.
#
# NOTHING ABOUT THEIR PLACEMENT IS HAND-ENTERED.  fit_fingertip() below lands
# each one by matching its bolt-face outline against the bucket's, which is the
# same rack interface, so the offsets and - more importantly - which side each
# part belongs on are re-derived on every run.  See its docstring for why a
# bounding box or an ICP is the wrong tool for that job.
FINGER_SRC = {
    "maintenance": (os.path.join(REPO, "Maintenance Gripper Finger L.stl"),
                    os.path.join(REPO, "Maintenance Gripper Finger R.stl")),
}

# The two rigid placements a fingertip can have on the rack.  A physical part
# cannot be mirrored, so the only freedom is whether it bolts down as drawn or
# turned end for end; the maintenance pair uses one of each.
FINGER_PLACEMENTS = (
    ("as drawn", np.eye(4)),
    ("turned 180 deg about Z", trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])),
)

# --------------------------------------------------------------------------
# CAD frame -> arm_gripper_base_link
# --------------------------------------------------------------------------
# Once the Y-up container is undone (see YUP_TO_ZUP below) the CAD needs no
# further rotation: the tool axis is already +Z (flange face at the bottom,
# jaws at the top), the jaws separate along +/-X and the plate depth runs on
# +/-Y, which is the convention gripper_new.xacro established and
# gripper_v2.xacro kept.  So this map is a pure translation.
#
# The origin is the ReBeL flange: the six M3.5 clearance holes through the
# mount's bottom face fit a circle of radius 21.505 mm centred here, at the
# same 0/60/120... deg clocking as gripper_base.stl's own flange holes, so the
# two grippers hang off link6 in the same orientation and
# arm_gripper_base_joint is unchanged between them.
FLANGE = np.array([0.7151250, 0.9273100, 1.2648800])

# --------------------------------------------------------------------------
# The gear pair
# --------------------------------------------------------------------------
# 21 teeth (an order-parameter fit over the tooth-tip angles scores 0.911 at
# N=21 against 0.005 at every other count from 18 to 27; the tip lands repeat
# every 360/21 = 17.1429 deg exactly).  The racks' tooth-tip lands are 0.600 mm
# wide on a 3.0000 mm linear pitch - read straight off the exported vertex X
# coordinates, which land on -30.723, -27.723, -24.723 ... with no rounding.
#
# PITCH_R IS NOT MEASURED OFF THE TOOTH RADII, and deliberately so.  The tip
# radius is 11.985 mm and the root 9.986, which fit no standard tooth form: the
# rack flanks work out at a ~7 deg pressure angle, so these are simplified
# printed teeth, not involutes, and any addendum/dedendum reasoning about where
# the pitch circle sits is guesswork.  What IS exact for any rack-and-pinion
# that meshes without skipping is that the rack advances one linear pitch per
# pinion angular pitch.  That fixes the ratio outright:
#
#     travel per radian = 3.0000 mm / (2*pi/21 rad) = 21 * 3.0000 / (2*pi)
#
# One full servo turn is therefore exactly 21 teeth x 3 mm = 63.0 mm of rack.
TEETH = 21
RACK_PITCH = 0.0030
PITCH_R = TEETH * RACK_PITCH / (2.0 * np.pi)          # 0.01002676 m

# Pinion rotation axis height above the flange face, from the Pinion solid's
# own CAD origin.  Any point on the axis serves for a revolute joint; this one
# keeps the exported mesh a pure translation of the CAD solid.
PINION_Z = 1.3480754 - FLANGE[2]

# --------------------------------------------------------------------------
# Zero pose
# --------------------------------------------------------------------------
# The CAD snapshot is a pose, not a datum, and its two jaws are not symmetric
# with each other: the left jaw's flat clamping face sits at x = +0.620 mm and
# the right jaw's at x = +2.210 mm, so the whole pair is 1.415 mm off the
# pinion axis.  Mirroring one onto the other (both jaws measure 32.000 mm
# across the shank and 50.000 mm across the scoop, so the mirror is exact
# except at the nesting lip) puts the symmetry plane at x = +1.415 mm.
CAD_SYMMETRY_X = 0.001415

# From that symmetrised pose the jaws still have to close 0.500 mm per side
# before they touch.  Measured by voxel contact search at 0.4 mm on 600k
# surface samples per jaw, closing in 0.25 mm steps.  Note that it is the
# NESTING LIPS at z = 203-206 mm that meet, not the flat shank faces - those
# are still 0.59 mm apart at contact, because this is a clamshell scoop whose
# halves are handed (the right lip runs 3.000 mm wider than the left so the two
# nest).  Do not re-derive this from the flat faces.
CAD_TOUCH_TRAVEL = 0.000500

# "Closed" is +0.07 rad everywhere in this stack - the SRDF `closed` group
# state, gamepad.yaml's gripper_closed_position, the mock-hardware command
# limits.  The zero pose is calibrated so the jaws touch exactly there, which
# is what gripper_v2.xacro does too, so a controller or a cached pose does not
# care which gripper is fitted.
Q_CLOSED = 0.07

# --------------------------------------------------------------------------
# Node grouping
# --------------------------------------------------------------------------
# The GLB exports one node per solid with generated names, and a solid that
# carries two materials is split into several nodes with a random hex suffix
# per load - so match on prefixes, never on full names.
#
# WHICH RACK CARRIES WHICH JAW IS NOT WHAT THE NAMES SUGGEST.  The solid named
# "Rack" has its teeth on the +Y side and its carrier block under the -X
# (left) finger; "Rack 2" is the mirror.  Grouping them by name lands both
# fingers on the wrong rack and flips the whole sign convention, so the
# assignment below is by carrier position and is asserted at the bottom.
GROUPS = {
    # IGUS Mount now carries the +X end cap that used to be a separate "IGUS
    # Mount 2" solid; body3737688 is the ST3215 case.
    "base": ("IGUS Mount", "body3737688"),
    # Everything on the servo output shaft: the pinion, the horn washer and the
    # rear bearing disc, which the colour merge scattered across two node
    # names.  All three are coaxial with the flange, so carrying the boss that
    # sits down inside the servo case on this link rather than on the base is
    # invisible both to the renderer and to the inertia about Z.
    "pinion": ("body3756496", "body3759603"),
    "rack_left": ("Rack(",),
    "rack_right": ("Rack 2",),
    "bucket_left": ("Bucket Gripper Finger 1",),
    "bucket_right": ("Bucket Gripper Finger 2",),
}


# The re-export writes a Y-UP glTF container around Z-up CAD data, so trimesh
# hands back (x, -z_cad, y_cad).  Undoing it is a -90 deg turn about X.  Get
# the sign wrong and everything still looks plausible - the gripper is just
# lying on its face - so this is asserted against the flange in load_parts().
YUP_TO_ZUP = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0])


def load_parts():
    """Every solid, in metres, expressed in arm_gripper_base_link at the CAD pose."""
    scene = trimesh.load(SRC, process=False)
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(transform)
        mesh.apply_transform(YUP_TO_ZUP)
        mesh.apply_translation(-FLANGE)
        parts[node] = mesh
    mount = [m for n, m in parts.items() if n.startswith("IGUS Mount")][0]
    lo, hi = mount.bounds
    assert abs(lo[2]) < 1e-4, f"flange face is not at z=0 ({lo[2]:.5f}) - axis fix wrong?"
    assert hi[2] > 0.12, "mount does not extend up +Z - axis fix wrong?"
    return parts


def group(parts, key):
    sel = [m for n, m in parts.items() if n.startswith(GROUPS[key])]
    if not sel:
        raise SystemExit(f"no solids matched group {key!r}")
    return trimesh.util.concatenate(sel)


def jaw_shift(inner_face_x, sign):
    """X translation from the CAD pose to this jaw's q = 0 position.

    ``sign`` is +1 for the jaw that lives on +X.  At q = 0 each jaw stands
    Q_CLOSED * PITCH_R off the touching pose, because touching is defined to
    happen at q = +0.07.
    """
    del inner_face_x
    open_at_zero = Q_CLOSED * PITCH_R - CAD_TOUCH_TRAVEL
    return -CAD_SYMMETRY_X + sign * open_at_zero


def inertial(mesh, mass):
    """Mass, COM and the inertia tensor about the COM, at a uniform density."""
    m = mesh.copy()
    m.density = mass / m.volume if m.volume > 1e-12 else 1000.0
    com = m.center_mass
    return com, m.moment_inertia


def collision_bands(mesh, n_bands, samples=300000):
    """Axis-aligned boxes tiling the mesh in Z, from a surface sampling.

    In this gz-sim / DART-bullet stack a <mesh> collision generates no contacts
    against another <mesh> (see gripper_v2.xacro's long note), so the jaws have
    to carry primitive collision or they pass through everything they grip.
    """
    pts, _ = trimesh.sample.sample_surface(mesh, samples)
    z0, z1 = pts[:, 2].min(), pts[:, 2].max()
    edges = np.linspace(z0, z1, n_bands + 1)
    out = []
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        sel = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
        c = np.array([sel[:, 0].mean() if False else (sel[:, 0].min() + sel[:, 0].max()) / 2,
                      (sel[:, 1].min() + sel[:, 1].max()) / 2,
                      (lo + hi) / 2])
        size = np.array([sel[:, 0].max() - sel[:, 0].min(),
                         sel[:, 1].max() - sel[:, 1].min(),
                         hi - lo])
        out.append((c, size))
    return out


def mount_face(mesh):
    """The bolt face outline: the unique (x, y) vertices on the mesh's lowest Z.

    Every fingertip bolts to the flat top of its rack, so this face is the one
    feature that is identical across the whole fingertip family - same outline,
    same two counterbored bosses, same pair of locating notches.  It is
    therefore the datum, and it is returned sorted so two faces can be compared
    row for row.
    """
    v = mesh.vertices
    face = v[v[:, 2] < v[:, 2].min() + 1e-6]
    u = np.unique(np.round(face[:, :2], 6), axis=0)
    return u[np.lexsort((u[:, 1], u[:, 0]))]


def bolt_fit(raw, rot, reference):
    """Translation and residual landing ``raw`` rotated by ``rot`` on ``reference``.

    Both bolt faces are the same point set, so the centroid offset IS the
    translation and the residual afterwards either vanishes or it does not.
    """
    m = raw.copy()
    m.apply_transform(rot)
    face, ref = mount_face(m), mount_face(reference)
    if face.shape != ref.shape:
        return None, float("inf")
    dxy = ref.mean(axis=0) - face.mean(axis=0)
    residual = np.abs(face + dxy - ref).max()
    shift = np.array([dxy[0], dxy[1],
                      reference.vertices[:, 2].min() - m.vertices[:, 2].min()])
    return trimesh.transformations.translation_matrix(shift) @ rot, residual


def clamping_plane(mesh):
    """Which shank plane carries the flat clamping face: 'lo' (-X) or 'hi' (+X).

    The shank planes are the two X extremes of the bolt face, i.e. the flat
    sides of the 32 mm block every fingertip in this family starts as.  One of
    them is the jaw's working face and is very nearly solid; the other is the
    back, which is always relieved - scooped out on the bucket, swept away into
    the hook on the maintenance finger.  So the larger planar area is the
    working face, by a factor of 1.5 or better on every part measured so far.
    """
    v = mesh.vertices
    bolt = v[v[:, 2] < v[:, 2].min() + 1e-6][:, 0]
    tris = mesh.vertices[mesh.faces]
    def planar_area(x):
        return mesh.area_faces[np.all(np.abs(tris[:, :, 0] - x) < 1e-5, axis=1)].sum()
    lo, hi = planar_area(bolt.min()), planar_area(bolt.max())
    if min(lo, hi) > 0.7 * max(lo, hi):
        raise SystemExit("the two shank planes carry the same area - this part has "
                         "no working face to tell its handedness from")
    return "lo" if lo > hi else "hi"


def fit_fingertip_pair(raws, references, tol=5e-6):
    """Land a fingertip pair, deriving which side each part goes on.

    ``raws`` is {name: mesh} straight off disk, ``references`` is
    {"left": tip, "right": tip} already fitted.  Returns
    {side: (name, placed mesh, placement label, transform)}.

    THE FILENAME IS NOT EVIDENCE.  A part exported as "... L.stl" is labelled by
    whoever exported it, and a swapped pair is close to invisible: both halves
    have the same silhouette from the front, so it renders as a working gripper
    whose jaws present their backs to each other.  Two independent features
    decide it instead, and they answer different halves of the question:

      the BOLT FACE fixes which way round the part bolts down.  Its outline is
      mirror-symmetric in X, so it says nothing at all about the side - a pure
      X translation lands it on either rack - but its two locating notches sit
      0.250 mm off the outline's own midline, so of the two ways to turn a part
      about Z exactly one matches to the micron and the other misses by 0.500
      mm.  That also pins the X, Y and Z offsets outright.

      the CLAMPING FACE then fixes the side, by having to point at the other
      jaw.  See clamping_plane().

    Neither alone is enough and neither is a tolerance to be relaxed: loosen the
    bolt-face fit past a few microns and both turns pass, and the part goes on
    upside down.
    """
    fits = {}
    for name, raw in sorted(raws.items()):
        turns = []
        for label, rot in FINGER_PLACEMENTS:
            _, residual = bolt_fit(raw, rot, references["left"])
            if residual <= tol:
                turns.append((label, rot, residual))
        if not turns:
            raise SystemExit(
                f"{name}: neither turn lands its bolt face on the rack. Either it is "
                f"not a fingertip for this gripper, or the mount interface changed - "
                f"in which case the bucket is wrong too and the whole family needs "
                f"re-deriving.")
        if len(turns) > 1:
            raise SystemExit(f"{name}: the bolt face fits both turns; the 0.250 mm "
                             f"notch asymmetry is gone from the export")
        label, rot, residual = turns[0]

        turned = raw.copy()
        turned.apply_transform(rot)
        side = "left" if clamping_plane(turned) == "hi" else "right"
        if side in fits:
            raise SystemExit(f"{name} and {fits[side][0]} both land on the {side} rack "
                             f"- this is two copies of one hand, not a pair")

        transform, residual = bolt_fit(raw, rot, references[side])
        mesh = raw.copy()
        mesh.apply_transform(transform)
        fits[side] = (name, mesh, label, transform, residual)

    missing = set(references) - set(fits)
    if missing:
        raise SystemExit(f"no part landed on the {', '.join(sorted(missing))} rack")
    return fits


def travel_limit(jaw, mount, direction, cell=0.001):
    """Largest opening travel of ``jaw`` before it fouls the fixed structure.

    A plain swept-overlap test reports 0 mm here: at the CAD pose each rack's
    toothed tail already runs into the far end wall (the assembly is drawn
    interfered there, or the slot that clears it did not survive the re-export),
    so the two solids touch at every travel and the sweep never gets started.

    That interference is in the CLOSING direction and the CAD pose is closed,
    so it says nothing about the stroke.  What matters is the OPENING
    direction, and there the tail retreats out of the wall it points at while
    the carrier and fingertip advance on the wall behind them.  So the test is:
    ray-cast the wall along X to find which (y, z) cells are solid all the way
    through - a ray that crosses the slot registers no hits - and measure how
    far the frontmost jaw point standing in a solid cell is from that cell's
    inner face.  Nothing is hand-tuned and it re-derives itself if the slot
    moves.
    """
    band = mount.vertices[(mount.vertices[:, 2] > 0.092) & (mount.vertices[:, 2] < 0.130)]
    wall = band[band[:, 0] < -0.060] if direction < 0 else band[band[:, 0] > 0.060]
    if len(wall) == 0:
        return float("inf")
    x_lo, x_hi = wall[:, 0].min() - 1e-4, wall[:, 0].max() + 1e-4

    ys = np.arange(-0.032, 0.0325, cell)
    zs = np.arange(0.092, 0.1305, cell)
    grid = np.array([(y, z) for y in ys for z in zs])
    origins = np.column_stack([np.full(len(grid), -0.150), grid])
    caster = trimesh.ray.ray_triangle.RayMeshIntersector(mount)
    hits, ray_id, _ = caster.intersects_location(
        origins, np.tile([1.0, 0.0, 0.0], (len(grid), 1)), multiple_hits=True)

    through = {}
    for point, rid in zip(hits, ray_id):
        if x_lo <= point[0] <= x_hi:
            through.setdefault(rid, []).append(point[0])
    # Two crossings means the ray went in one face and out the other, i.e. the
    # cell is solid.  One crossing is a graze along a slot edge, and none is
    # clear air.
    solid = {}
    for rid, xs in through.items():
        if len(xs) >= 2:
            key = (round(grid[rid][0], 4), round(grid[rid][1], 4))
            solid[key] = max(xs) if direction < 0 else min(xs)
    if not solid:
        return float("inf")

    pts, _ = trimesh.sample.sample_surface(jaw, 900000)
    pts = pts[(pts[:, 2] > 0.092) & (pts[:, 2] < 0.1305)]
    keys = np.round(np.round(pts[:, 1:] / cell) * cell, 4)
    best = float("inf")
    for (y, z), x in zip(map(tuple, keys), pts[:, 0]):
        face = solid.get((y, z))
        if face is None:
            continue
        best = min(best, (x - face) if direction < 0 else (face - x))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="measure only, write nothing")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}")
    parts = load_parts()

    base = group(parts, "base")
    pinion = group(parts, "pinion")
    rack_l = group(parts, "rack_left")
    rack_r = group(parts, "rack_right")
    tip_l = group(parts, "bucket_left")
    tip_r = group(parts, "bucket_right")

    # --- assert the rack/jaw assignment rather than trusting the CAD names ---
    # The left jaw lives on -X, so its carrier block - the full-width part of
    # the rack, everything outside the 2 mm toothed band in Y - must sit under
    # the left fingertip.
    def carrier_x(rack):
        v = rack.vertices
        wide = v[np.abs(v[:, 1]) > 0.020]
        return wide[:, 0].mean()
    assert carrier_x(rack_l) < carrier_x(rack_r), "rack_left/rack_right are swapped"
    assert tip_l.bounds[1][0] < tip_r.bounds[1][0], "bucket_left/bucket_right are swapped"

    shift_l = jaw_shift(None, -1)
    shift_r = jaw_shift(None, +1)
    for m, dx in ((rack_l, shift_l), (tip_l, shift_l),
                  (rack_r, shift_r), (tip_r, shift_r)):
        m.apply_translation([dx, 0.0, 0.0])
    pinion.apply_translation([0.0, 0.0, -PINION_Z])

    print("=" * 72)
    print("GEAR PAIR")
    print("=" * 72)
    print(f"  teeth               {TEETH}")
    print(f"  rack linear pitch   {RACK_PITCH * 1e3:.4f} mm")
    print(f"  travel per radian   {PITCH_R * 1e3:.5f} mm      <- xacro rack_pitch_radius")
    print(f"  one servo turn      {2 * np.pi * PITCH_R * 1e3:.3f} mm of rack")
    print(f"  pinion joint origin 0 0 {PINION_Z:.6f}")
    print()
    print("ZERO POSE")
    print(f"  CAD symmetry plane  x = {CAD_SYMMETRY_X * 1e3:+.3f} mm")
    print(f"  touch travel        {CAD_TOUCH_TRAVEL * 1e3:.3f} mm/side from symmetrised CAD")
    print(f"  jaws touch at       q = {Q_CLOSED:+.3f} rad")
    print(f"  mesh shift  left    {shift_l * 1e3:+.4f} mm     right {shift_r * 1e3:+.4f} mm")

    # --- stroke ---------------------------------------------------------
    mount = group(parts, "base")
    jaw_l = trimesh.util.concatenate([rack_l, tip_l])
    jaw_r = trimesh.util.concatenate([rack_r, tip_r])
    d_l = travel_limit(jaw_l, mount, -1)
    d_r = travel_limit(jaw_r, mount, +1)
    d_max = min(d_l, d_r)
    print()
    print("STROKE  (opening travel per jaw before fouling the fixed structure)")
    print(f"  left  jaw clear to  {d_l * 1e3:.1f} mm")
    print(f"  right jaw clear to  {d_r * 1e3:.1f} mm")
    print(f"  binding             {d_max * 1e3:.1f} mm -> q = {-(d_max / PITCH_R):.4f} rad,"
          f" gap {2 * d_max * 1e3 + 2 * Q_CLOSED * PITCH_R * 1e3:.1f} mm")
    print(f"  with a 2 mm margin  q_lower = {-((d_max - 0.002) / PITCH_R):.4f} rad")

    print()
    print("JAW GAP TABLE   gap [mm] = 2 * PITCH_R * (0.07 - q)")
    qs = [0.07, 0.0, -0.2, -0.5, -1.0, -1.57, -2.0, -3.0, -4.0, -4.07]
    print("    q [rad] " + " ".join(f"{q:7.2f}" for q in qs))
    print("    gap[mm] " + " ".join(f"{2e3 * PITCH_R * (Q_CLOSED - q):7.2f}" for q in qs))

    # --- masses ---------------------------------------------------------
    # 1050 kg/m^3 for near-solid printed plastic, the density gripper_v2.xacro
    # uses, so the two grippers stay comparable.  The servo is the exception:
    # its solids are carried at the ST3215's datasheet 60 g rather than the
    # plastic density, and combined with the mount by parallel axis, so the
    # base tensor below is not a uniform-density one.
    PLASTIC = 1050.0
    SERVO_MASS = 0.060
    servo = trimesh.util.concatenate(
        [m for n, m in parts.items() if n.startswith("body3737688")])
    mount_only = trimesh.util.concatenate(
        [m for n, m in parts.items() if n.startswith("IGUS Mount")])

    print()
    print("=" * 72)
    print("INERTIALS   (mass, COM, tensor about the COM, in the link frame)")
    print("=" * 72)

    def report(name, mesh, mass):
        com, I = inertial(mesh, mass)
        print(f"  {name}")
        print(f"    mass {mass:.4f}  com {com[0]:+.5f} {com[1]:+.5f} {com[2]:+.5f}")
        print(f"    ixx {I[0][0]:.4e} iyy {I[1][1]:.4e} izz {I[2][2]:.4e}")
        print(f"    ixy {I[0][1]:+.4e} ixz {I[0][2]:+.4e} iyz {I[1][2]:+.4e}")
        return mass

    # base = mount at plastic density + servo at datasheet mass, as one solid
    m_mount = mount_only.volume * PLASTIC
    base_mass = m_mount + SERVO_MASS
    # Build the combined tensor by parallel axis rather than by faking a density.
    com_m, I_m = inertial(mount_only, m_mount)
    com_s, I_s = inertial(servo, SERVO_MASS)
    com_b = (m_mount * com_m + SERVO_MASS * com_s) / base_mass

    def shift_tensor(I, m, com, target):
        d = com - target
        return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    I_b = shift_tensor(I_m, m_mount, com_m, com_b) + shift_tensor(I_s, SERVO_MASS, com_s, com_b)
    print("  arm_gripper_base_link   (mount at 1050 kg/m^3 + ST3215 at 60 g)")
    print(f"    mass {base_mass:.4f}  com {com_b[0]:+.5f} {com_b[1]:+.5f} {com_b[2]:+.5f}")
    print(f"    ixx {I_b[0][0]:.4e} iyy {I_b[1][1]:.4e} izz {I_b[2][2]:.4e}")
    print(f"    ixy {I_b[0][1]:+.4e} ixz {I_b[0][2]:+.4e} iyz {I_b[1][2]:+.4e}")

    total = base_mass
    for name, mesh in (("gripper_pinion_link", pinion),
                       ("gripper_rack_left_link", rack_l),
                       ("gripper_rack_right_link", rack_r),
                       ("gripper_bucket_left_link", tip_l),
                       ("gripper_bucket_right_link", tip_r)):
        total += report(name, mesh, mesh.volume * PLASTIC)
    print(f"\n  TOTAL GRIPPER MASS {total * 1e3:.0f} g")

    # --- fingertip collision boxes --------------------------------------
    print()
    print("=" * 72)
    print("FINGERTIP COLLISION BOXES   (mesh collision generates no contacts here)")
    print("=" * 72)
    for name, mesh in (("bucket_left", tip_l), ("bucket_right", tip_r)):
        print(f"  {name}")
        for i, (c, s) in enumerate(collision_bands(mesh, 4)):
            print(f"    band{i}  xyz {c[0]:+.5f} {c[1]:+.5f} {c[2]:+.5f}"
                  f"   size {s[0]:.4f} {s[1]:.4f} {s[2]:.4f}")

    # --- jaw contact height ---------------------------------------------
    lip_z = tip_l.vertices[:, 2].max()
    shank = tip_l.vertices[(tip_l.vertices[:, 2] > 0.115) & (tip_l.vertices[:, 2] < 0.150)]
    print()
    print(f"  jaw span            z = {tip_l.vertices[:, 2].min() * 1e3:.1f} to {lip_z * 1e3:.1f} mm")
    print(f"  flat shank face at  x = {shank[:, 0].max() * 1e3:+.3f} mm at q = 0")

    # --- swappable fingertips -------------------------------------------
    # Everything above measures the bucket, which is the fingertip the assembly
    # was exported with.  Each alternative pair is landed on the same rack here
    # and measured the same way, because the xacro branches on finger_type and
    # needs a full set of numbers per branch - a fingertip that reuses the
    # bucket's inertia or its collision boxes is worse than no fingertip at
    # all, since it looks right in every view.
    exports = [("st3215_base.stl", base),
               ("st3215_pinion.stl", pinion),
               ("st3215_rack_left.stl", rack_l),
               ("st3215_rack_right.stl", rack_r),
               ("st3215_bucket_left.stl", tip_l),
               ("st3215_bucket_right.stl", tip_r)]
    references = {"left": tip_l, "right": tip_r}

    for family, paths in sorted(FINGER_SRC.items()):
        print()
        print("=" * 72)
        print(f"FINGERTIP  {family.upper()}   (finger_type:={family})")
        print("=" * 72)
        raws = {}
        for path in paths:
            if not os.path.exists(path):
                sys.exit(f"missing {path}")
            # process=True merges the duplicated STL vertices, without which
            # nothing is watertight and every volume and inertia is garbage.
            raws[os.path.basename(path)] = trimesh.load(path, process=True)
        fits = fit_fingertip_pair(raws, references)

        for side in ("left", "right"):
            name, mesh, label, transform, residual = fits[side]
            t = transform[:3, 3]
            print(f"  {side:5s} <- {name}")
            print(f"          {label}, then {t[0] * 1e3:+.3f} {t[1] * 1e3:+.3f} "
                  f"{t[2] * 1e3:+.3f} mm   (bolt face residual {residual * 1e6:.2f} um)")
            print(f"          watertight {mesh.is_watertight}   volume "
                  f"{mesh.volume * 1e6:.2f} cm^3   {len(mesh.faces)} faces")

        f_l, f_r = fits["left"][1], fits["right"][1]

        # THE CONTACT ANGLE IS A PROPERTY OF THE FINGER, NOT OF THE GRIPPER.
        # The bucket's scoops nest, so their lips meet at the +0.07 rad this
        # stack calls closed.  A flat-faced finger has nothing that reaches
        # past the shank plane, so it meets later - possibly past the joint
        # limit, in which case "closed" is a gap and everything that converts a
        # width into an angle has to know that.
        opening = f_r.bounds[0][0] - f_l.bounds[1][0]
        q_touch = opening / (2.0 * PITCH_R)
        gap_at_closed = opening - 2.0 * Q_CLOSED * PITCH_R
        print()
        print(f"  facing surfaces     {opening * 1e3:+.4f} mm apart at q = 0")
        print(f"  jaws touch at       q = {q_touch:+.5f} rad"
              f"   (bucket: {Q_CLOSED:+.3f})")
        print(f"  gap at q = {Q_CLOSED:+.2f}      {gap_at_closed * 1e3:+.4f} mm"
              + ("   <- NEVER TOUCHES within the joint limit"
                 if q_touch > Q_CLOSED + 1e-9 else ""))
        print(f"  jaw span            z = {f_l.bounds[0][2] * 1e3:.1f} to "
              f"{f_l.bounds[1][2] * 1e3:.1f} mm")
        print(f"  xacro:  gap [m] = {2.0 * PITCH_R:.8f} * ({q_touch:.6f} - q)")

        print()
        print("  INERTIALS  (1050 kg/m^3)")
        for side in ("left", "right"):
            mesh = fits[side][1]
            report(f"    gripper_bucket_{side}_link  [{family}]",
                   mesh, mesh.volume * PLASTIC)

        print()
        print("  COLLISION BOXES  (mesh collision generates no contacts here)")
        for side in ("left", "right"):
            print(f"    {side}")
            for i, (c, sz) in enumerate(collision_bands(fits[side][1], 4)):
                print(f"      band{i}  xyz {c[0]:+.5f} {c[1]:+.5f} {c[2]:+.5f}"
                      f"   size {sz[0]:.4f} {sz[1]:.4f} {sz[2]:.4f}")

        # Does this fingertip run out of stroke before the bucket does?  It
        # bolts to the same rack, so the answer is only ever "no" while its
        # footprint through the mount's end walls stays inside the bucket's.
        d_f = min(travel_limit(trimesh.util.concatenate([rack_l, f_l]), mount, -1),
                  travel_limit(trimesh.util.concatenate([rack_r, f_r]), mount, +1))
        print()
        print(f"  opening travel      {d_f * 1e3:.1f} mm per jaw before fouling"
              f"   (bucket {d_max * 1e3:.1f} mm)")
        if d_f < d_max - 5e-4:
            print("    ^ SHORTER THAN THE BUCKET - gripper_st3215.yaml's measured "
                  "stroke does not apply to this finger")

        for side in ("left", "right"):
            exports.append((f"st3215_{family}_{side}.stl", fits[side][1]))

    if args.report:
        print("\n--report: nothing written")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, mesh in exports:
        path = os.path.join(OUT_DIR, fname)
        mesh.export(path)
        print(f"  wrote {os.path.relpath(path, REPO)}  ({len(mesh.faces)} faces)")


if __name__ == "__main__":
    main()
