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
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]          # <repo>/src
REPO = SRC.parent
URDF = SRC / "aries" / "urdf" / "my_robot.urdf.xacro"

# The mechanism, from the build script.
PITCH_R = 0.01002676          # m of jaw travel per radian of pinion
Q_CLOSED = 0.07               # rad at which the jaws touch
STEPS_PER_REV = 4096          # ST3215 encoder, one turn
SERVO_USABLE_STEPS = 4045 - 50
# THE CALIBRATION NOW LIVES IN ONE YAML, and these tests read it rather than
# repeating it. aries_common/config/gripper_st3215.yaml holds the four numbers
# that come off the hardware; aries_common.gripper_cal derives the rest.
#
# TWO DIFFERENT LIMITS come out of that, and conflating them broke arm planning
# once. OPEN_TRAVEL is what the joint may BE: the full measured stroke plus
# slack, so a gripper parked at its stop is never "out of bounds" (MoveIt then
# refuses to plan for the whole arm). COMMAND_OPEN is what we may COMMAND: held
# short of the stop so the servo never stalls. Observation generous, command
# tight.
sys.path.insert(0, str(SRC / "aries_common"))
from aries_common.gripper_cal import gripper_cal  # noqa: E402

_CAL = gripper_cal()
OPEN_TRAVEL = _CAL["open_travel_m"]       # m per jaw -> URDF joint limit
COMMAND_OPEN = _CAL["command_open_rad"]   # rad -> SRDF `open`, teleop, min_pos
CLOSED_STEPS = int(_CAL["closed_steps"])
OPEN_STOP_STEPS = int(_CAL["open_stop_steps"])
INVERT = bool(_CAL["invert"])


def _ament_env():
    """Make `$(find <pkg>)` resolve against the source tree."""
    env = dict(os.environ)
    prefixes = [str(SRC / name) for name in ("aries", "aries_moveit")]
    existing = env.get("AMENT_PREFIX_PATH", "")
    env["AMENT_PREFIX_PATH"] = os.pathsep.join(prefixes + ([existing] if existing else []))
    return env


def build(gripper_type="st3215", **args):
    # The launch files derive these from the YAML and pass them in; do the same
    # here or the test measures the bare-xacro defaults instead of what ships.
    args.setdefault("gripper_command_open", f"{COMMAND_OPEN:.4f}")
    args.setdefault("gripper_open_travel", f"{OPEN_TRAVEL:.6f}")
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
                         capture_output=True, text=True, env=_ament_env())  # noqa: E501
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
    # The commanded open pose is the usable stroke; the limit sits a little past
    # it so the physical stop is still inside bounds.
    assert 2.0 * PITCH_R * (Q_CLOSED - COMMAND_OPEN) == pytest.approx(_CAL["command_gap_m"], abs=1e-6)
    assert lower < COMMAND_OPEN, "the joint limit must not cut off the commanded open pose"


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

    pairs = {(d.get("link1"), d.get("link2")) for d in srdf_for("st3215").findall("disable_collisions")}
    assert ("gripper_bucket_left_link", "gripper_bucket_right_link") in pairs, \
        "the nesting jaw pair is not disabled in the ACM"


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


def srdf_for(gripper_type):
    """The SRDF is xacro, gated on gripper_type. See its header for why."""
    path = SRC / "aries_moveit" / "moveit_config" / "config" / "aries.srdf"
    out = subprocess.run(["xacro", str(path), f"gripper_type:={gripper_type}",
                          f"gripper_command_open:={COMMAND_OPEN:.4f}"],
                         capture_output=True, text=True, env=_ament_env())
    assert out.returncode == 0, f"xacro failed on the SRDF:\n{out.stderr[-1500:]}"
    return ET.fromstring(out.stdout)


@pytest.mark.parametrize("gripper_type", ["v2", "st3215"])
def test_the_srdf_names_no_link_the_urdf_lacks(gripper_type):
    """srdfdom does NOT quietly skip an unknown link in a GROUP.

    It logs `Error: Link 'x' declared as part of group 'gripper' is not known to
    the URDF`, once per node that loads the model - four of them here, so a
    plain union of both grippers' link sets printed twelve red lines at every
    startup on a stack that was working fine. The disable_collisions rows only
    warn, but both are checked here: a clean start is what lets a real fault be
    seen.
    """
    urdf = build(gripper_type, hardware_protocol="mock_hardware")
    srdf = srdf_for(gripper_type)
    known = links(urdf)

    unknown_group = [ln.get("name") for g in srdf.findall("group")
                     for ln in g.findall("link") if ln.get("name") not in known]
    assert not unknown_group, (
        f"{gripper_type}: SRDF groups name links the URDF does not have "
        f"({unknown_group}) - srdfdom logs these as Errors")

    unknown_acm = [(d.get("link1"), d.get("link2")) for d in srdf.findall("disable_collisions")
                   if d.get("link1") not in known or d.get("link2") not in known]
    assert not unknown_acm, f"{gripper_type}: ACM rows for absent links {unknown_acm[:5]}"


def test_the_gripper_group_still_covers_the_four_bar_on_v2():
    """Splitting the SRDF must not quietly shrink v2's guard.

    rebel_servo_teleop_gamepad sets req.group_name to `arm_with_gripper` and
    calls checkSelfCollision, so a link missing from the group is a link the
    joystick guard does not check. Dropping the bars would have been the easy
    way to silence the errors above and it would have cost real safety.
    """
    grp = {ln.get("name") for g in srdf_for("v2").findall("group")
           if g.get("name") == "gripper" for ln in g.findall("link")}
    for bar in ("gripper_gear_left_link", "gripper_gear_right_link",
                "gripper_left_link", "gripper_right_link",
                "gripper_bucket_left_link", "gripper_bucket_right_link"):
        assert bar in grp, f"v2's {bar} fell out of the gripper group"


def test_the_gripper_group_covers_the_racks_on_st3215():
    grp = {ln.get("name") for g in srdf_for("st3215").findall("group")
           if g.get("name") == "gripper" for ln in g.findall("link")}
    for ln in ("gripper_gear_left_link", "gripper_rack_left_link",
               "gripper_rack_right_link",
               "gripper_bucket_left_link", "gripper_bucket_right_link"):
        assert ln in grp, f"st3215's {ln} is not in the gripper group"


@pytest.mark.parametrize("gripper_type,expected", [("v2", -1.57), ("st3215", COMMAND_OPEN)])
def test_the_srdf_open_state_matches_the_fitted_mechanism(gripper_type, expected):
    """`open` is a property of the mechanism, not a shared constant.

    Before aries.srdf was xacro'd it was one static -1.57 for both, which on the
    ST3215 is a 32.9 mm jaw gap out of 82.9 - so a MoveIt "open" left more than
    half the stroke unreachable while the joystick, which has its own overlay,
    opened fully. The two must agree.
    """
    srdf = srdf_for(gripper_type)
    values = [float(j.get("value"))
              for gs in srdf.findall("group_state") if gs.get("name") == "open"
              for j in gs.findall("joint") if j.get("name") == "gripper_gear_left_joint"]
    assert values, "the SRDF has no `open` group state for the gripper"
    for value in values:
        assert value == pytest.approx(expected, abs=1e-3)

    # And it has to be reachable, or MoveIt rejects every plan to it.
    urdf = build(gripper_type, hardware_protocol="mock_hardware")
    lim = joint(urdf, "gripper_gear_left_joint").find("limit")
    assert float(lim.get("lower")) <= expected <= float(lim.get("upper"))


@pytest.mark.parametrize("gripper_type", ["v2", "st3215"])
def test_every_srdf_group_state_is_inside_the_urdf_limits(gripper_type):
    """A group state outside the joint limits is silently unplannable."""
    urdf = build(gripper_type, hardware_protocol="mock_hardware")
    limits = {}
    for j in urdf.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    bad = []
    for gs in srdf_for(gripper_type).findall("group_state"):
        for j in gs.findall("joint"):
            name, value = j.get("name"), float(j.get("value"))
            if not name.startswith("gripper"):
                continue        # joint5 has a pre-existing, unrelated overrun
            lo, hi = limits.get(name, (-1e9, 1e9))
            if not lo - 1e-9 <= value <= hi + 1e-9:
                bad.append((gs.get("name"), name, value, (lo, hi)))
    assert not bad, f"{gripper_type}: unreachable gripper group states {bad}"


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

    # Every copy carries the COMMAND limit, not the joint limit. It must equal
    # COMMAND_OPEN and sit inside the URDF limit; a copy at the joint limit
    # would command the servo into its stop.
    open_copies = {
        "fourbar.py": fourbar.ST3215_Q_OPEN,
        "gripper_arc_visualizer.py": _const_from(viz, "ST3215_Q_OPEN"),
    }
    for where, value in open_copies.items():
        # These are rounded copies, so allow a little slack - but they must be
        # INSIDE both the command limit's intent and the joint limit.
        assert abs(value - COMMAND_OPEN) <= 0.02, (
            f"{where} opens to {value:.4f}; the calibration derives {COMMAND_OPEN:.4f}")
        assert value > urdf_lower, f"{where} is past the joint limit {urdf_lower:.4f}"

    # The joystick's open position. Parsed as YAML rather than grepped: the
    # file's own comments quote v2's -1.57 to explain why this overlay exists,
    # and a regex happily matches the explanation instead of the setting.
    import yaml
    cfg = yaml.safe_load(teleop)
    joy_open = [section["ros__parameters"]["gripper_open_position"]
                for section in cfg.values()]
    assert joy_open, "teleop_speeds_st3215.yaml no longer sets gripper_open_position"
    for value in joy_open:
        assert abs(value - COMMAND_OPEN) <= 0.02, (
            f"the joystick opens to {value}; the calibration derives {COMMAND_OPEN:.4f}")

    # And the hardware component's own clamp. It is a xacro substitution now, so
    # assert on the BUILT description rather than on the template text.
    import re
    built = subprocess.run(
        ["xacro", str(URDF), "gripper_type:=st3215", "hardware_protocol:=rebel",
         "gripper_hardware_protocol:=st3215",
         f"gripper_command_open:={COMMAND_OPEN:.4f}"],
        capture_output=True, text=True, env=_ament_env()).stdout
    m = re.search(r'<param name="min_pos">([-\d.]+)</param>', built)
    assert m, "the st3215 hardware block no longer sets min_pos"
    assert abs(float(m.group(1)) - COMMAND_OPEN) <= 0.02, (
        "min_pos is the component's own clamp and is the last thing standing "
        "between a slider dragged to the joint limit and a stalled servo")


def test_the_measured_stroke_stays_inside_the_servo_stops():
    """The whole stroke has to sit between the servo's own hard stops.

    Measured: closed 572, open 1844 - opening RAISES the count, so invert is
    true. The component holds 50 steps back from the open stop, commanding 1791.
    Both ends must stay inside the 50..4045 working band or the servo crosses
    its 4095 -> 0 encoder seam mid-stroke.

    The direction is asserted here because the step numbers cannot carry it:
    swapping the two stop labels is equally valid arithmetic and yields a
    gripper that opens on close and closes on open, silently.
    """
    invert = INVERT
    direction = -1 if invert else 1
    per_rad = STEPS_PER_REV / (2 * math.pi)

    # What we COMMAND must stop short of the mechanical stop.
    cmd_steps = round(CLOSED_STEPS + direction * (COMMAND_OPEN - Q_CLOSED) * per_rad)
    toward_open = 1 if OPEN_STOP_STEPS > CLOSED_STEPS else -1
    assert (OPEN_STOP_STEPS - cmd_steps) * toward_open > 0, (
        f"the commanded open end (step {cmd_steps}) is past the measured stop at "
        f"{OPEN_STOP_STEPS} - the servo would stall against it")
    assert abs(OPEN_STOP_STEPS - cmd_steps) >= 25, "margin off the open stop is too thin"

    # What the joint may BE must cover the stop, or a parked gripper reports out
    # of bounds and MoveIt aborts planning for the whole arm.
    limit_steps = round(CLOSED_STEPS + direction * (-OPEN_TRAVEL / PITCH_R - Q_CLOSED) * per_rad)
    assert (limit_steps - OPEN_STOP_STEPS) * toward_open > 0, (
        f"the joint limit (step {limit_steps}) does not reach the mechanical stop "
        f"at {OPEN_STOP_STEPS}; a gripper parked there is out of bounds")

    for name, st in (("closed", CLOSED_STEPS), ("command open", cmd_steps),
                     ("joint limit", limit_steps)):
        assert 50 <= st <= 4045, f"{name} at step {st} is outside the servo band"

    # And the launch defaults must be these, or the calibration is only in a
    # comment. This is the pair that was wrong on the first bench run.
    # The launch defaults come from the YAML now, so assert that path rather
    # than a literal in the xacro.
    from aries_common.gripper_cal import cal_str
    assert cal_str("closed_steps") == str(CLOSED_STEPS)
    assert cal_str("invert") == str(invert).lower()


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
        for gap in (0.005, 0.030, 0.060, _CAL["command_gap_m"] - 0.001):  # up to the commanded open
            assert fourbar.gap_from_q(fourbar.q_from_gap(gap)) == pytest.approx(gap, abs=1e-6)
        # The jaws translate, so contact height must not vary with q.
        assert fourbar.contact_offset_z(0.0) == fourbar.contact_offset_z(-4.0)
        # An unknown name must degrade to v2, not take the grasp stack down.
        assert fourbar.set_gripper("nonsense") == "v2"
    finally:
        fourbar.set_gripper("v2")
