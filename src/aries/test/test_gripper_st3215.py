"""Regression guards for the two grippers on the ReBeL flange.

WHAT THIS FILE IS FOR
gripper_st3215.xacro is sold as a DROP-IN swap for gripper_v2.xacro: the SRDF,
the controllers, gamepad.yaml and every cached pose keep working because both
grippers publish the same driver joint under the same name with the same sign
convention.  That promise is invisible - nothing fails loudly when it breaks,
the gripper just opens the wrong way or by the wrong amount - so it is asserted
here rather than trusted.

The numbers come from scripts/build_gripper_st3215_meshes.py.  If one of these
fails after a CAD re-export, re-run that script and move the value here to
match; do not relax the assertion.

XACRO SPANS PACKAGES, so these tests build their own AMENT_PREFIX_PATH out of
the source tree.  Without it `$(find aries_moveit)` fails and every case errors
out identically whether or not the URDF is actually broken.
"""

import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]          # <repo>/src
REPO = SRC.parent
URDF = SRC / "aries" / "urdf" / "my_robot.urdf.xacro"

# The mechanism, from the build script.
PITCH_R = 0.01002676          # m of jaw travel per radian of pinion
Q_CLOSED = 0.07               # rad at which the jaws touch
OPEN_TRAVEL = 0.0408          # m per jaw, 2 mm inside the CAD's 42.8
STEPS_PER_REV = 4096          # ST3215 encoder, one turn
SERVO_USABLE_STEPS = 4045 - 50


def _ament_env():
    """Make `$(find <pkg>)` resolve against the source tree."""
    env = dict(os.environ)
    prefixes = [str(SRC / name) for name in ("aries", "aries_moveit")]
    existing = env.get("AMENT_PREFIX_PATH", "")
    env["AMENT_PREFIX_PATH"] = os.pathsep.join(prefixes + ([existing] if existing else []))
    return env


def build(gripper_type="st3215", **args):
    argv = ["xacro", str(URDF), f"gripper_type:={gripper_type}"]
    argv += [f"{k}:={v}" for k, v in args.items()]
    out = subprocess.run(argv, capture_output=True, text=True, env=_ament_env())
    assert out.returncode == 0, f"xacro failed for {argv}:\n{out.stderr[-2000:]}"
    return ET.fromstring(out.stdout)


@pytest.fixture(scope="module")
def st3215():
    return build("st3215", hardware_protocol="mock_hardware")


@pytest.fixture(scope="module")
def v2():
    return build("v2", hardware_protocol="mock_hardware")


def joint(root, name):
    for j in root.findall("joint"):
        if j.get("name") == name:
            return j
    raise AssertionError(f"joint {name!r} not in the URDF")


def links(root):
    return {ln.get("name") for ln in root.findall("link")}


# ---------------------------------------------------------------------------
# The drop-in promise
# ---------------------------------------------------------------------------

def test_an_unknown_gripper_type_is_a_hard_error():
    """It must not silently produce an arm with no end effector.

    Before the guard in my_robot.urdf.xacro, a typo simply took neither include
    branch: xacro succeeded, the arm came up with nothing past link6, and it
    read in RViz as a mesh-loading problem.
    """
    out = subprocess.run(["xacro", str(URDF), "gripper_type:=bogus"],
                         capture_output=True, text=True, env=_ament_env())
    assert out.returncode != 0
    assert "gripper_type_must_be_v2_or_st3215" in out.stderr


@pytest.mark.parametrize("name", ["arm_gripper_base_link",
                                  "gripper_gear_left_link",
                                  "gripper_bucket_left_link",
                                  "gripper_bucket_right_link",
                                  "gripper_tcp"])
def test_both_grippers_publish_the_shared_link_names(st3215, v2, name):
    """These five names are what the SRDF, the ACM and the grasp stack address.

    Rename one on either gripper and the swap stops being a drop-in: the SRDF
    rows for it turn into startup warnings and the pair stops being collision
    checked at all.
    """
    assert name in links(st3215)
    assert name in links(v2)


def test_the_driver_joint_has_the_same_name_and_closed_angle(st3215, v2):
    for root in (st3215, v2):
        j = joint(root, "gripper_gear_left_joint")
        assert j.get("type") == "revolute"
        assert float(j.find("limit").get("upper")) == pytest.approx(Q_CLOSED, abs=1e-9)


def test_the_flange_joint_is_identical(st3215, v2):
    """Both bolt to the same six-hole circle at the same clocking, so anything
    that cached a link6 -> gripper transform must not care which is fitted."""
    a, b = joint(st3215, "arm_gripper_base_joint"), joint(v2, "arm_gripper_base_joint")
    assert a.get("type") == b.get("type") == "fixed"
    for attr in ("xyz", "rpy"):
        assert a.find("origin").get(attr) == b.find("origin").get(attr)


def test_the_tcp_sits_at_the_same_height(st3215, v2):
    for root in (st3215, v2):
        o = joint(root, "gripper_tcp_joint").find("origin")
        assert [float(v) for v in o.get("xyz").split()] == [0.0, 0.0, 0.15]


# ---------------------------------------------------------------------------
# The rack-and-pinion mechanism
# ---------------------------------------------------------------------------

def test_positive_q_closes_the_jaws(st3215):
    """The sign convention the whole stack assumes: +0.07 closed, negative open.

    It is carried by the pinion joint's -Z axis together with the racks' +X and
    -X axes.  Flip either and the gripper still looks like a working gripper -
    it just opens when told to close, which on the rover means driving the jaws
    into whatever they were meant to grip.
    """
    assert joint(st3215, "gripper_gear_left_joint").find("axis").get("xyz") == "0 0 -1"
    # The left jaw lives on -X, so closing means travelling +X.
    left = joint(st3215, "gripper_rack_left_joint")
    right = joint(st3215, "gripper_rack_right_joint")
    assert left.find("axis").get("xyz") == "1 0 0"
    assert right.find("axis").get("xyz") == "-1 0 0"
    for j in (left, right):
        assert float(j.find("mimic").get("multiplier")) > 0.0


def test_the_gear_ratio_is_the_measured_one(st3215):
    """21 teeth on a 3.0000 mm rack pitch. This single constant sets every jaw
    gap, so it is also duplicated in aries_vision_grasp/fourbar.py; the two are
    cross-checked in test_the_grasp_stack_agrees_about_the_gap below."""
    expected = 21 * 0.0030 / (2.0 * math.pi)
    assert expected == pytest.approx(PITCH_R, abs=1e-8)
    for side in ("left", "right"):
        m = joint(st3215, f"gripper_rack_{side}_joint").find("mimic")
        assert float(m.get("multiplier")) == pytest.approx(PITCH_R, abs=1e-8)
        assert m.get("joint") == "gripper_gear_left_joint"


def test_rack_limits_are_the_driver_limits_scaled(st3215):
    """A mimic joint's limits are in ITS units. Reusing the driver's radians on
    a prismatic joint puts every pose three orders of magnitude out of range and
    MoveIt rejects the lot."""
    drv = joint(st3215, "gripper_gear_left_joint").find("limit")
    for side in ("left", "right"):
        lim = joint(st3215, f"gripper_rack_{side}_joint").find("limit")
        assert float(lim.get("lower")) == pytest.approx(
            float(drv.get("lower")) * PITCH_R, abs=1e-9)
        assert float(lim.get("upper")) == pytest.approx(
            float(drv.get("upper")) * PITCH_R, abs=1e-9)


def test_the_stroke_is_the_measured_one(st3215):
    lower = float(joint(st3215, "gripper_gear_left_joint").find("limit").get("lower"))
    assert lower == pytest.approx(-OPEN_TRAVEL / PITCH_R, abs=1e-4)
    full_gap = 2.0 * PITCH_R * (Q_CLOSED - lower)
    assert full_gap == pytest.approx(0.0829, abs=5e-4)


def test_the_stroke_fits_inside_one_servo_turn(st3215):
    """The ST3215 is single-turn in position mode. If the stroke ever exceeds
    the usable step range there is no zero that works, and the failure in the
    field is the gripper stopping partway through an open with nothing logged.
    """
    lower = float(joint(st3215, "gripper_gear_left_joint").find("limit").get("lower"))
    steps = (Q_CLOSED - lower) * STEPS_PER_REV / (2.0 * math.pi)
    assert steps < SERVO_USABLE_STEPS, (
        f"the {Q_CLOSED - lower:.3f} rad stroke needs {steps:.0f} steps of the "
        f"{SERVO_USABLE_STEPS} usable")


def test_gazebo_widens_the_limits_off_the_hard_stop():
    """A commanded endpoint sitting exactly on a DART hard stop lets the
    mimic-constrained joints wedge there and refuse the next reverse command."""
    sim = build("st3215", hardware_protocol="gazebo")
    real = build("st3215", hardware_protocol="mock_hardware")
    s = joint(sim, "gripper_gear_left_joint").find("limit")
    r = joint(real, "gripper_gear_left_joint").find("limit")
    assert float(s.get("lower")) < float(r.get("lower"))
    assert float(s.get("upper")) > float(r.get("upper"))


# ---------------------------------------------------------------------------
# Things that fail silently in this stack
# ---------------------------------------------------------------------------

def test_the_fingertips_collide_as_boxes(st3215):
    """In this gz-sim / DART-bullet build a <mesh> collision generates NO
    contacts against another <mesh>, so a fingertip with mesh collision passes
    straight through everything it is asked to grip. The visual stays a mesh."""
    for side in ("left", "right"):
        link = next(ln for ln in st3215.findall("link")
                    if ln.get("name") == f"gripper_bucket_{side}_link")
        cols = link.findall("collision")
        assert cols, "the fingertip has no collision geometry at all"
        for c in cols:
            assert c.find("geometry/box") is not None, (
                f"gripper_bucket_{side}_link collision is not a box")
        assert link.find("visual/geometry/mesh") is not None


def test_the_jaws_nest_and_that_pair_is_acm_disabled(st3215):
    """The two scoops are HANDED: the right lip runs 3.000 mm wider than the
    left so the pair closes into one bowl, and the collision boxes reproduce
    that - at the closed pose they overlap by about 2.4 mm.

    That overlap is correct geometry, not a modelling error, and it is why the
    SRDF must disable this pair.  Both halves are asserted together because
    either one alone is misleading: trim the boxes and the closed pose stops
    matching the real part, drop the ACM row and MoveIt refuses to plan to the
    `closed` group state it is asked for on every grasp.

    (Gazebo needs no equivalent: SDF self_collide defaults to false and nothing
    in aries_gazebo.xacro turns it on, so intra-model contacts are off.)
    """
    inner = {}
    for side, sign in (("left", -1), ("right", +1)):
        link = next(ln for ln in st3215.findall("link")
                    if ln.get("name") == f"gripper_bucket_{side}_link")
        edges = []
        for c in link.findall("collision"):
            cx = float(c.find("origin").get("xyz").split()[0])
            sx = float(c.find("geometry/box").get("size").split()[0])
            edges.append(cx + sign * -sx / 2.0)     # the edge facing the axis
        inner[side] = max(edges) if sign < 0 else min(edges)

    travel = Q_CLOSED * PITCH_R                      # each jaw's closing travel
    overlap = (inner["left"] + travel) - (inner["right"] - travel)
    assert overlap == pytest.approx(0.0024, abs=5e-4), (
        f"jaw nesting overlap is {overlap * 1e3:.2f} mm, expected ~2.4. A value "
        f"near zero means the handedness was lost in a re-export; a much larger "
        f"one means the lips are modelled through each other.")

    srdf = (SRC / "aries_moveit" / "moveit_config" / "config" / "aries.srdf").read_text()
    assert ('link1="gripper_bucket_left_link" link2="gripper_bucket_right_link"'
            in srdf), "the nesting jaw pair is not disabled in the ACM"


def test_the_camera_is_not_left_on_the_other_mount(st3215, v2):
    """The bracket is modelled in neither base mesh, so the offset comes from
    the camera holes in the mount casting, and the two mounts disagree by
    18.2 mm. A wrong extrinsic is the one error the grasp stack cannot see."""
    y_st = float(joint(st3215, "gripper_camera_joint").find("origin").get("xyz").split()[1])
    y_v2 = float(joint(v2, "gripper_camera_joint").find("origin").get("xyz").split()[1])
    assert y_st == pytest.approx(0.065274, abs=1e-6)
    assert y_v2 == pytest.approx(0.047439, abs=1e-6)


@pytest.mark.parametrize("gripper_type", ["v2", "st3215"])
def test_every_referenced_mesh_exists(gripper_type):
    root = build(gripper_type, hardware_protocol="mock_hardware")
    missing = []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn.startswith("package://aries/"):
            continue
        path = SRC / "aries" / fn[len("package://aries/"):]
        if not path.exists():
            missing.append(fn)
    assert not missing, f"missing meshes: {missing}"


def _const_from(path, name):
    """Read a module-level float constant out of a file we cannot import."""
    import re
    src = Path(path).read_text()
    m = re.search(rf"^{name} = ([-\d.]+)$", src, re.M)
    assert m, f"{name} is gone from {Path(path).name}"
    return float(m.group(1))


def test_every_copy_of_the_mechanism_constants_agrees(st3215):
    """The pitch radius and the stroke are written down in FOUR places.

    The URDF is the source of truth, but three consumers cannot read it: the
    grasp package is COLCON_IGNOREd, the RViz overlay is a standalone script,
    and the teleop overlay is YAML.  So each carries its own copy, and the only
    thing keeping them in step is this test.  Drift is silent in the worst way -
    the gripper closes on a confident angle for the wrong width, or the overlay
    marks a point of closing the jaws never reach.

    The open angle is checked as a BOUND, not for equality: the copies use a
    round -4.065 where the URDF limit is -4.0691, which is 0.04 mm of jaw travel
    inside it.  Being inside is the invariant that matters - a command past the
    joint limit is rejected outright - so the test allows a small shortfall and
    no overrun at all.
    """
    import sys
    sys.path.insert(0, str(SRC / "aries_vision_grasp"))
    from aries_vision_grasp import fourbar

    urdf_lower = float(joint(st3215, "gripper_gear_left_joint").find("limit").get("lower"))
    viz = SRC / "aries_moveit" / "moveit_config" / "scripts" / "gripper_arc_visualizer.py"
    teleop = (SRC / "aries_moveit" / "moveit_config" / "config"
              / "teleop_speeds_st3215.yaml").read_text()

    pitch_copies = {
        "fourbar.py": fourbar.ST3215_PITCH_R,
        "gripper_arc_visualizer.py": _const_from(viz, "ST3215_PITCH_R"),
    }
    for where, value in pitch_copies.items():
        assert value == pytest.approx(PITCH_R, abs=1e-8), f"pitch radius drifted in {where}"

    closed_copies = {
        "fourbar.py": fourbar.ST3215_Q_CLOSED,
        "gripper_arc_visualizer.py": _const_from(viz, "ST3215_Q_CLOSE"),
    }
    for where, value in closed_copies.items():
        assert value == pytest.approx(Q_CLOSED, abs=1e-9), f"closed angle drifted in {where}"

    open_copies = {
        "fourbar.py": fourbar.ST3215_Q_OPEN,
        "gripper_arc_visualizer.py": _const_from(viz, "ST3215_Q_OPEN"),
    }
    for where, value in open_copies.items():
        assert urdf_lower <= value <= urdf_lower + 0.01, (
            f"{where} opens to {value:.4f} against the URDF limit {urdf_lower:.4f}; "
            f"{'past the limit, so the command is rejected' if value < urdf_lower else 'leaving stroke unused'}")

    # The joystick's open position. Parsed as YAML rather than grepped: the
    # file's own comments quote v2's -1.57 to explain why this overlay exists,
    # and a regex happily matches the explanation instead of the setting.
    import yaml
    cfg = yaml.safe_load(teleop)
    joy_open = [section["ros__parameters"]["gripper_open_position"]
                for section in cfg.values()]
    assert joy_open, "teleop_speeds_st3215.yaml no longer sets gripper_open_position"
    for value in joy_open:
        assert urdf_lower <= value <= urdf_lower + 0.01, (
            f"the joystick opens to {value} against the URDF limit {urdf_lower:.4f}")

    # And the hardware component's own clamp, in the control xacro.
    import re
    control = (SRC / "aries" / "urdf" / "igus_rebel2.control.xacro").read_text()
    m = re.search(r'<param name="min_pos">([-\d.]+)</param>',
                  control[control.index("ST3215GripperSystem"):])
    assert m, "the st3215 hardware block no longer sets min_pos"
    assert urdf_lower <= float(m.group(1)) <= urdf_lower + 0.01


def test_the_grasp_stack_gap_model_is_the_mechanism(st3215):
    """fourbar.py answers gap questions for whichever gripper is fitted, and on
    this one the answer is a closed form rather than an interpolated table."""
    import sys
    sys.path.insert(0, str(SRC / "aries_vision_grasp"))
    from aries_vision_grasp import fourbar

    assert fourbar.set_gripper("st3215") == "st3215"
    try:
        assert fourbar.gap_from_q(Q_CLOSED) == pytest.approx(0.0, abs=1e-9)
        # Round trip through the inverse, which is what sizes a real grasp.
        for gap in (0.005, 0.030, 0.060, 0.0829):
            assert fourbar.gap_from_q(fourbar.q_from_gap(gap)) == pytest.approx(gap, abs=1e-6)
        # The jaws translate, so contact height must not vary with q.
        assert fourbar.contact_offset_z(0.0) == fourbar.contact_offset_z(-4.0)
        # An unknown name must degrade to v2, not take the grasp stack down.
        assert fourbar.set_gripper("nonsense") == "v2"
    finally:
        fourbar.set_gripper("v2")
