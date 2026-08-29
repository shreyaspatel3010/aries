#!/usr/bin/env python3
"""Split the ST3215 rack-and-pinion gripper CAD export into per-link URDF meshes.

Source asset (repo root):
    new_gripper.glb    whole assembly, one flat list of solids in one frame

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

    if args.report:
        print("\n--report: nothing written")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, mesh in (("st3215_base.stl", base),
                        ("st3215_pinion.stl", pinion),
                        ("st3215_rack_left.stl", rack_l),
                        ("st3215_rack_right.stl", rack_r),
                        ("st3215_bucket_left.stl", tip_l),
                        ("st3215_bucket_right.stl", tip_r)):
        path = os.path.join(OUT_DIR, fname)
        mesh.export(path)
        print(f"  wrote {os.path.relpath(path, REPO)}  ({len(mesh.faces)} faces)")


if __name__ == "__main__":
    main()
