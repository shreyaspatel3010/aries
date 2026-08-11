#!/usr/bin/env python3
"""Split the REBEL-6DOF-04 gripper CAD export into per-link URDF meshes.

Source assets (repo root):
    full gripper with bucket finger.glb   whole assembly, bucket fingers fitted
    maintanance_finger.glb                maintenance fingertip, ONE side only
    probe_finger.glb                      probe fingertip, ONE side only

The CAD assembly is one flat list of 32 unnamed solids in a single world frame.
This script groups them into the six moving links plus the base, re-expresses
each group in its own URDF link frame, and writes the meshes that
``aries/urdf/gripper_v2.xacro`` loads.  It also prints the joint origins,
inertials and jaw-gap table quoted in that file, so every number there can be
regenerated rather than trusted.

Run from the repo root:  python3 scripts/build_gripper_v2_meshes.py
Add --report to skip writing and only print the measurements.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "src", "aries", "meshes", "gripper_v2")

SRC_FULL = os.path.join(REPO, "full gripper with bucket finger.glb")
SRC_MAINT = os.path.join(REPO, "maintanance_finger.glb")
SRC_PROBE = os.path.join(REPO, "probe_finger.glb")

# --------------------------------------------------------------------------
# CAD frame -> arm_gripper_base_link
# --------------------------------------------------------------------------
# In the CAD export the gripper axis is +X, the jaws separate along +/-Z and
# the plate-stack depth runs along +Y.  The URDF convention inherited from
# gripper_new.xacro is: axis +Z, jaws +/-X, depth +/-Y.  The map
#     X_u = -(z_g - Z0)      Y_u = y_g - Y0      Z_u = x_g - X0
# is R_y(-90 deg) about the origin below - a proper rotation, no mirroring.
X0 = -0.134290   # -X face of the mount bracket, i.e. the link6 mating face
Y0 = 0.027700    # depth mid-plane of the plate stack (the bars sit at +/-12.5)
Z0 = 0.052420    # jaw symmetry plane

R_GU = np.array([[0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0]])
O_G = np.array([X0, Y0, Z0])

# --------------------------------------------------------------------------
# Pivot holes, recovered by RANSAC circle fits on the 4.5 mm bores
# (max residual 1 um).  All eight lie in the CAD's XZ plane.
# --------------------------------------------------------------------------
PIVOT_GLB = {
    "gear_axis_L": (-0.043820, 0.076670),   # driven bar, ground pivot
    "gear_axis_R": (-0.043820, 0.028170),
    "coup_axis_L": (-0.013090, 0.059920),   # coupler bar, ground pivot
    "coup_axis_R": (-0.013090, 0.044920),
    "gear_tip_L": (0.004680, 0.088790),     # finger pivot A
    "gear_tip_R": (0.004970, 0.017240),
    "coup_tip_L": (0.035420, 0.072050),     # finger pivot B
    "coup_tip_R": (0.035700, 0.033990),
}

BAR = 0.050      # both bars, pivot to pivot (measured 49.991 - 50.004)
GROUND = 0.035   # gear axis -> coupler axis, and pivot A -> pivot B

# Solid -> link.  Every CAD body is accounted for.
GROUPS = {
    "base": [
        "body95773",    # mount bracket (link6 flange)
        "body96587",    # main side plate, both material groups
        "body96386",    # lower stand-off bracket
        "body96185",    # upper stand-off bracket
        "body101476",   # servo body
        "body86949",    # servo pinion
        "body97347", "body114304", "body110541", "body86574",  # hub, shaft, bolt
        "body113966", "body113628", "body112952",              # pins
        "body113290", "body112614", "body112276",
        "body100988", "body101110", "body101232", "body101354",  # screw heads
    ],
    "gear_left": ["body88927", "body109521"],    # toothed plate + plain plate
    "gear_right": ["body92350", "body110031"],
    "coupler_left": ["body109266", "body109011"],
    "coupler_right": ["body109776", "body110286"],
    "bucket_left": ["body108180"],
    "bucket_right": ["body107401"],
}

# The two single-side fingertip exports.  Each carries the same 4.5 mm pivot
# pair, 35 mm apart, as the bucket bracket does; `pivot_a` is the hole that
# lands on the gear-bar tip (the one further from the blade) and `pivot_b` the
# one on the coupler tip.  `axis` is the pivot-axis direction in that file's
# own coordinates.
#
# Those two bores pin the fingertip down to a single remaining freedom: a half
# turn about the A-B line.  The blade is not parallel to that line - it leaves
# it at about 30 deg, which is what cancels the four-bar's 28.6 deg ground-link
# angle and leaves the jaw faces parallel to the gripper axis.  Get the half
# turn wrong and that 30 deg adds instead of cancelling: the jaws toe in by
# ~58 deg and foul each other after a few millimetres of travel.  `flip` is
# therefore chosen by measurement, not by eye - see resolve_flip().
SIDE_FINGERS = {
    "maintenance": dict(
        path=SRC_MAINT,
        pivot_a=(0.0, -0.013810, 0.086290),
        pivot_b=(0.0, -0.013810, 0.051290),
        axis=(1.0, 0.0, 0.0),
    ),
    "probe": dict(
        path=SRC_PROBE,
        pivot_a=(0.0, 0.005000, -0.004930),
        pivot_b=(0.0, 0.005000, -0.039930),
        axis=(1.0, 0.0, 0.0),
    ),
}

# 3D-printed structure.  Same value gripper_new.xacro used, so the two gripper
# models stay comparable.  The servo is overridden below with its datasheet
# mass; leaving it at plastic density would under-report it by ~4x.
DENSITY = 1050.0
SERVO_BODY = "body101476"
SERVO_MASS = 0.090

# Joint zero.  q = +0.07 rad is "closed" everywhere in this stack (the SRDF
# `closed` group state, the aries_vision_grasp gap tables).  ZERO_PSI is the
# bar angle off the gripper axis at q = 0, solved below so that the bucket
# jaws just touch at q = +0.07.  Recomputed by --report; kept here so the
# exported meshes are reproducible without re-solving.
ZERO_PSI_DEG = 11.3428


# --------------------------------------------------------------------------


def g2u_matrix():
    T = np.eye(4)
    T[:3, :3] = R_GU
    T[:3, 3] = -R_GU @ O_G
    return T


def pivot_u(name):
    """Pivot hole in arm_gripper_base_link coords (Y is 0 by construction)."""
    x, z = PIVOT_GLB[name]
    return np.array([-(z - Z0), 0.0, x - X0])


def load_bodies(path):
    """GLB -> {node name without the dedup suffix: mesh in CAD world coords}."""
    scene = trimesh.load(path, process=False)
    out = {}
    for node in scene.graph.nodes:
        T, geom = scene.graph[node]
        if geom is None:
            continue
        m = scene.geometry[geom].copy()
        m.apply_transform(T)
        out.setdefault(node.split("_")[0], []).append(m)
    merged = {}
    for k, v in out.items():
        m = trimesh.util.concatenate(v) if len(v) > 1 else v[0]
        m.merge_vertices()
        merged[k] = m
    return merged


def rot_y(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def about_y(deg, pivot):
    """4x4: rotate `deg` about the +Y axis through `pivot` (an X,Y,Z point)."""
    p = np.array([pivot[0], 0.0, pivot[2]])
    T = np.eye(4)
    T[:3, :3] = rot_y(deg)
    T[:3, 3] = p - rot_y(deg) @ p
    return T


def psi_of(vec):
    """Signed angle of an in-plane vector, from +Z_u toward +X_u, degrees."""
    return np.degrees(np.arctan2(vec[0], vec[2]))


def bar_dir(psi_deg):
    r = np.radians(psi_deg)
    return np.array([np.sin(r), 0.0, np.cos(r)])


def frame(origin, x_hat, y_hat):
    """4x4 world->local for a frame with the given origin and axes."""
    x_hat = x_hat / np.linalg.norm(x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)
    z_hat = np.cross(x_hat, y_hat)
    R = np.column_stack([x_hat, y_hat, z_hat])       # local -> world
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.asarray(origin, dtype=float)
    return T


class Model:
    """The CAD assembly, re-expressed link by link."""

    def __init__(self, zero_psi_deg=ZERO_PSI_DEG):
        self.zero_psi = zero_psi_deg
        self._flip = {}
        raw = load_bodies(SRC_FULL)
        Tgu = g2u_matrix()
        self.body = {}
        for k, m in raw.items():
            m = m.copy()
            m.apply_transform(Tgu)
            self.body[k] = m

        self.piv = {k: pivot_u(k) for k in PIVOT_GLB}
        # Bar angle in the CAD snapshot, per side.
        self.psi_cad = {
            "L": psi_of(self.piv["gear_tip_L"] - self.piv["gear_axis_L"]),
            "R": psi_of(self.piv["gear_tip_R"] - self.piv["gear_axis_R"]),
        }

    # -- kinematics --------------------------------------------------------
    def psi(self, side, q):
        """Bar angle at joint value q (rad).  q < 0 opens.

        The left bar sits at -ZERO_PSI and turns with +q about +Y; the right
        one mirrors it, which is what the meshing sector gears enforce.
        """
        s = +1.0 if side == "L" else -1.0
        return s * (-self.zero_psi + np.degrees(q))

    def tip(self, side, q):
        """Finger pivot A at joint value q."""
        return self.piv[f"gear_axis_{side}"] + BAR * bar_dir(self.psi(side, q))

    def link_tf(self, side, q):
        """Transforms that carry the CAD-pose solids to joint value q."""
        dpsi = self.psi(side, q) - self.psi_cad[side]
        return dict(
            gear=about_y(dpsi, self.piv[f"gear_axis_{side}"]),
            coupler=about_y(dpsi, self.piv[f"coup_axis_{side}"]),
            # A parallelogram holds the finger's orientation fixed, so the
            # fingertip only translates, by whatever pivot A does.
            finger=trimesh.transformations.translation_matrix(
                self.tip(side, q)
                - (self.piv[f"gear_axis_{side}"]
                   + BAR * bar_dir(self.psi_cad[side]))),
        )

    # -- link-local meshes -------------------------------------------------
    def joined(self, group):
        return trimesh.util.concatenate([self.body[b] for b in GROUPS[group]])

    def gear_mesh(self, side):
        """Gear bar in its link frame: origin on the ground pivot, axes
        parallel to arm_gripper_base_link at q = 0."""
        m = self.joined(f"gear_{'left' if side == 'L' else 'right'}")
        m = m.copy()
        m.apply_transform(self.link_tf(side, 0.0)["gear"])
        m.apply_translation(-self.piv[f"gear_axis_{side}"])
        return m

    def coupler_mesh(self, side):
        m = self.joined(f"coupler_{'left' if side == 'L' else 'right'}")
        m = m.copy()
        m.apply_transform(self.link_tf(side, 0.0)["coupler"])
        m.apply_translation(-self.piv[f"coup_axis_{side}"])
        return m

    def finger_frame(self, side):
        """Fingertip link frame: origin on pivot A, axes parallel to
        arm_gripper_base_link at q = 0."""
        return self.tip(side, 0.0)

    def bucket_mesh(self, side):
        m = self.body[GROUPS[f"bucket_{'left' if side == 'L' else 'right'}"][0]].copy()
        m.apply_transform(self.link_tf(side, 0.0)["finger"])
        m.apply_translation(-self.finger_frame(side))
        return m

    def side_finger_mesh(self, name, side, flip):
        """Place a single-side fingertip export on the given jaw.

        `flip` is the half turn about the A-B line left free by the two bores.
        The result is in the same base-aligned, pivot-A-origin link frame the
        bucket fingertip uses.
        """
        spec = SIDE_FINGERS[name]
        m = trimesh.util.concatenate(list(load_bodies(spec["path"]).values()))

        a = np.array(spec["pivot_a"], dtype=float)
        b = np.array(spec["pivot_b"], dtype=float)
        x_src = (b - a) / np.linalg.norm(b - a)
        y_src = np.array(spec["axis"], dtype=float) * (-1.0 if flip else 1.0)
        T_src = frame(a, x_src, y_src)

        # Target frame: pivot A at the origin, +X toward pivot B, +Y along the
        # pivot axis.  Expressed in link coordinates, which are parallel to
        # arm_gripper_base_link, so the mesh comes out ready to use.
        #
        # The two jaws are mirror images about X_u = 0, and a mirror is not a
        # rotation, so the pivot axis is taken the opposite way round on the
        # right.  That turns the same solid into its own mirror image, which
        # is only legitimate because these fingertips are symmetric about
        # their own mid-plane - asserted below.
        x_t = self.piv[f"coup_tip_{side}"] - self.piv[f"gear_tip_{side}"]
        x_t = x_t / np.linalg.norm(x_t)
        y_t = np.array([0.0, 1.0, 0.0]) * (1.0 if side == "L" else -1.0)
        span = m.bounds[:, np.argmax(np.abs(spec["axis"]))]
        if abs(span[0] + span[1]) > 1e-4:
            raise RuntimeError(
                f"{name}: not symmetric about its mid-plane ({span[0]:.4f}, "
                f"{span[1]:.4f}); it cannot serve both jaws")
        T_tgt = frame(np.zeros(3), x_t, y_t)

        m = m.copy()
        m.apply_transform(T_src)                 # -> canonical finger frame
        m.apply_transform(np.linalg.inv(T_tgt))  # -> link frame
        return m

    def resolve_flip(self, name):
        """Pick the half turn about the A-B line by opening the gripper.

        With the blade's 30 deg dog-leg cancelling the ground link the jaws
        run near-parallel and the stroke matches the bucket's; with it added
        they toe in and jam within a few millimetres.  The two cases differ by
        more than 3x, so this needs no tolerance.
        """
        if name in self._flip:
            return self._flip[name]
        q_open = -np.radians(90.0 - self.zero_psi)
        scores = []
        for flip in (False, True):
            self._flip[name] = flip
            scores.append(self.gap(name, q_open, n=4000))
        best = bool(np.argmax(scores))
        if max(scores) < 2.0 * min(scores):
            raise RuntimeError(
                f"{name}: the two mountings open to {scores[0] * 1000:.1f} and "
                f"{scores[1] * 1000:.1f} mm - too close to call, check the bores")
        self._flip[name] = best
        return best

    def finger_mesh(self, kind, side):
        if kind == "bucket":
            return self.bucket_mesh(side)
        return self.side_finger_mesh(kind, side, self.resolve_flip(kind))

    # -- measurements ------------------------------------------------------
    def posed_finger(self, kind, side, q):
        m = self.finger_mesh(kind, side).copy()
        m.apply_translation(self.tip(side, q))
        return m

    def _closest(self, kind, q, n=12000):
        """(distance, midpoint) of the closest approach between the jaws.

        Points are sampled on one jaw and measured against the OTHER jaw's
        triangles exactly, so the answer does not depend on two clouds
        happening to sample the same spot.
        """
        left = self.posed_finger(kind, "L", q)
        right = self.posed_finger(kind, "R", q)
        best = (np.inf, None)
        for src, dst in ((left, right), (right, left)):
            pts = np.vstack([trimesh.sample.sample_surface(src, n, seed=7)[0],
                             src.vertices])
            near, dist, _ = trimesh.proximity.closest_point(dst, pts)
            i = int(np.argmin(dist))
            if dist[i] < best[0]:
                best = (float(dist[i]), 0.5 * (pts[i] + near[i]))
        return best

    def gap(self, kind, q, **kw):
        """Closest approach between the two fingertips at joint value q."""
        return self._closest(kind, q, **kw)[0]

    def contact_z(self, kind, q):
        """Z of the midpoint between the closest points on the two jaws."""
        return float(self._closest(kind, q)[1][2])


def solve_zero_psi(target_q=0.07, kind="bucket", tol=2e-5):
    """Bar angle at q = 0 that makes the jaws touch exactly at target_q.

    gap() is a closest approach, so it bottoms out at 0 once the meshes
    interpenetrate; the bisection therefore brackets the angle at which the
    gap first lifts off zero.
    """
    lo, hi = 4.0, 20.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if Model(mid).gap(kind, target_q, n=4000) > tol:
            hi = mid          # still open at the target -> start closer in
        else:
            lo = mid
        if hi - lo < 1e-3:
            break
    return 0.5 * (lo + hi)


def inertial(parts):
    """Combine (mesh, mass) parts: total mass, COM and inertia about the COM.

    Each part is treated as uniform, so a link whose solids are different
    materials (the base carries a servo) still gets a sane tensor.
    """
    tot_m, com = 0.0, np.zeros(3)
    props = []
    for mesh, mass in parts:
        m = mesh.copy()
        m.merge_vertices()
        m.density = mass / m.volume
        props.append((mass, np.asarray(m.center_mass, dtype=float),
                      np.asarray(m.moment_inertia, dtype=float)))
        tot_m += mass
        com = com + mass * m.center_mass
    com = com / tot_m
    it = np.zeros((3, 3))
    for mass, c, i_c in props:
        d = c - com
        it += i_c + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return tot_m, com, it


def report(model):
    np.set_printoptions(suppress=True)
    print(f"zero-pose bar angle  ZERO_PSI = {model.zero_psi:.4f} deg")
    print(f"CAD snapshot bar angle  L={model.psi_cad['L']:+.3f} deg  "
          f"R={model.psi_cad['R']:+.3f} deg")
    print()
    print("joint origins in arm_gripper_base_link (xyz, m):")
    for name in ("gear_axis_L", "gear_axis_R", "coup_axis_L", "coup_axis_R"):
        p = model.piv[name]
        print(f"   {name:<12} {p[0]:+.6f} {p[1]:+.6f} {p[2]:+.6f}")
    d = BAR * bar_dir(-model.zero_psi)
    print(f"   bar tip offset L  {d[0]:+.6f} {d[1]:+.6f} {d[2]:+.6f}")
    d = BAR * bar_dir(+model.zero_psi)
    print(f"   bar tip offset R  {d[0]:+.6f} {d[1]:+.6f} {d[2]:+.6f}")
    print()
    q_open = -np.radians(90.0 - model.zero_psi)
    print(f"over-centre (bar perpendicular to the axis) at q = {q_open:.4f} rad")
    for name in SIDE_FINGERS:
        print(f"   {name} mounting flip about the A-B line: {model.resolve_flip(name)}")
    print()
    print("jaw gap [mm] and contact Z [mm]:")
    qs = [q_open, -1.2, -1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, -0.05,
          0.0, 0.02, 0.04, 0.06, 0.07]
    for kind in ("bucket", "maintenance", "probe"):
        row = "  ".join(f"{model.gap(kind, q) * 1000:6.1f}" for q in qs)
        print(f"   {kind:<12} {row}")
    print("   q            " + "  ".join(f"{q:+6.3f}" for q in qs))
    for kind in ("bucket", "maintenance", "probe"):
        print(f"   contact Z {kind:<12} {model.contact_z(kind, 0.07) * 1000:7.1f} mm "
              f"(at q=+0.07)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="measure only")
    ap.add_argument("--solve", action="store_true", help="re-solve ZERO_PSI")
    args = ap.parse_args()

    zero = solve_zero_psi() if args.solve else ZERO_PSI_DEG
    if args.solve:
        print(f"solved ZERO_PSI_DEG = {zero:.4f}")
    model = Model(zero)

    if not args.report:
        os.makedirs(OUT_DIR, exist_ok=True)
        # The base is the one link with a non-plastic solid in it.
        servo = model.body[SERVO_BODY]
        structure = trimesh.util.concatenate(
            [model.body[b] for b in GROUPS["base"] if b != SERVO_BODY])
        exports = {"base": (model.joined("base"),
                            [(structure, structure.volume * DENSITY),
                             (servo, SERVO_MASS)])}
        for side, tag in (("L", "left"), ("R", "right")):
            for name, mesh in ((f"gear_{tag}", model.gear_mesh(side)),
                               (f"coupler_{tag}", model.coupler_mesh(side))):
                exports[name] = (mesh, [(mesh, mesh.volume * DENSITY)])
            for kind in ("bucket", "maintenance", "probe"):
                mesh = model.finger_mesh(kind, side)
                exports[f"{kind}_{tag}"] = (mesh, [(mesh, mesh.volume * DENSITY)])
        for name, (mesh, parts) in exports.items():
            path = os.path.join(OUT_DIR, f"gripper_{name}.stl")
            mesh.export(path)
            mass, com, it = inertial(parts)
            print(f"{name:<20} -> {os.path.relpath(path, REPO)}")
            print(f"    faces={len(mesh.faces):5d} vol={mesh.volume * 1e6:8.2f} cm3 "
                  f"mass={mass * 1000:7.1f} g")
            print(f'    <xacro:gripper_inertial mass="{mass:.4f}" '
                  f'cx="{com[0]:.5f}" cy="{com[1]:.5f}" cz="{com[2]:.5f}"')
            print(f'         ixx="{it[0, 0]:.3e}" iyy="{it[1, 1]:.3e}" izz="{it[2, 2]:.3e}"')
            print(f'         ixy="{it[0, 1]:.3e}" ixz="{it[0, 2]:.3e}" iyz="{it[1, 2]:.3e}"/>')
    report(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
