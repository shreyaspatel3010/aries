"""The rover and the base station have to agree, and nothing checks that at runtime.

Both launch files are in this package precisely so this test can compare them.
While they were split across aries_bringup and aries_base_station, each half of
a two-machine decision could be edited without the other.

Three mistakes here are silent and expensive in the field:

  * both machines running a joy driver -- two publishers on /joy, consumers
    seeing them interleaved at double rate, buttons that appear to chatter;
  * either machine starting nodes before the DDS environment is set -- the
    stack lands on domain 0 with the default RMW, and the far end sees nothing
    at all on a link that pings fine;
  * a second RViz, from an included launch file that declared `use_rviz` of its
    own and inherited this one's. It came up blank beside the real one, and
    both rendered and both subscribed.

None of them fails loudly at runtime, so they are pinned here instead.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
PKG = SRC / "aries_comms"
BASE = PKG / "launch" / "base_station.launch.py"
ROVER = PKG / "launch" / "rover_field.launch.py"
CHECKER = PKG / "launch" / "base_station_checker.launch.py"
FULL = SRC / "aries_bringup" / "launch" / "full_hardware.launch.py"
CAMERA_VIEW = SRC / "aries_bringup" / "launch" / "camera_view.launch.py"
MOVEIT = (
    SRC / "aries_moveit" / "moveit_config" / "launch" / "aries_hardware.launch.py"
)


def _default_of(source, name):
    """The default_value of a DeclareLaunchArgument, as written."""
    match = re.search(
        rf'DeclareLaunchArgument\(\s*["\']{re.escape(name)}["\']\s*,'
        rf'\s*default_value=["\']([^"\']*)["\']',
        source,
    )
    assert match, f'no DeclareLaunchArgument("{name}", default_value=...) found'
    return match.group(1)


@pytest.fixture(scope="module")
def base():
    return BASE.read_text()


@pytest.fixture(scope="module")
def rover():
    return ROVER.read_text()


def test_exactly_one_side_reads_the_pad_by_default(base, rover):
    """The whole point: the operator holds the pad, the rover does not."""
    assert _default_of(base, "use_joy_node") == "true"
    assert _default_of(rover, "use_joy_node") == "false"


def test_rover_keeps_the_teleop_consumers(rover):
    """use_joy_node moves only the driver; use_joystick still runs the arm,
    the presets and the rover drive on the robot."""
    assert _default_of(rover, "use_joystick") == "true"


def test_rover_does_not_run_rviz(rover):
    assert _default_of(rover, "use_gui") == "false"


def test_rover_publishes_the_downlink(rover):
    """With no GUI on the rover this is the only way an image leaves it."""
    assert _default_of(rover, "enable_camera_downlink") == "true"


def test_base_decompresses_locally(base):
    assert _default_of(base, "use_camera_view") == "true"


def test_both_sides_set_dds_before_any_node(base, rover):
    """A node started above the environment keeps the calling shell's domain."""
    for name, source in (("base_station", base), ("rover_field", rover)):
        assert "dds_launch_actions()" in source, f"{name} does not set the DDS environment"
        env_at = source.index("*dds_launch_actions()")
        for action in ("Node(", "IncludeLaunchDescription("):
            first = source.find(action, source.index("def generate_launch_description"))
            if first != -1:
                assert env_at < first, (
                    f"{name}: {action} appears before the DDS environment is set"
                )


def test_full_hardware_sets_its_own_dds_environment():
    """Launched directly on the rover, it is nobody's child.

    rover_field sets the environment before including this file, so the field
    path was always correct -- which is exactly why the gap here went unnoticed.
    Run on its own (`ros2 launch aries_bringup full_hardware.launch.py`) from a
    terminal that never sourced aries_dds_env.sh, the whole stack landed on
    domain 0 with rmw_fastrtps_cpp. Every driver came up, the cameras included,
    and not one topic was visible to anything else on the robot.
    """
    source = FULL.read_text()
    assert "*dds_launch_actions(" in source, (
        "full_hardware no longer sets the DDS environment; launching it "
        "directly is back to inheriting the calling terminal's domain"
    )
    env_at = source.index("*dds_launch_actions(")
    body = source.index("def generate_launch_description")
    for action in ("Node(", "IncludeLaunchDescription("):
        first = source.find(action, body)
        if first != -1:
            assert env_at < first, (
                f"full_hardware: {action} appears before the DDS environment is set"
            )


def test_full_hardware_does_not_require_the_field_link():
    """It is the bench entry point too, and a laptop has no antenna.

    require_link=True raises when the machine holds none of the addresses in
    the hosts table, which is correct for rover_field and would stop every
    single-machine run of this file dead.
    """
    source = FULL.read_text()
    assert "dds_launch_actions(require_link=False)" in source, (
        "full_hardware must fall back to a loopback-only DDS config off the "
        "field link, not refuse to launch"
    )


def test_rover_field_is_the_strict_one():
    """Out there a missing cable is the bug, not a configuration to degrade to."""
    source = ROVER.read_text()
    body = source.split('"""', 2)[-1]
    assert "*dds_launch_actions()" in body, (
        "rover_field must keep the default require_link=True, so a rover with "
        "the antenna cable out fails the launch instead of coming up on "
        "loopback where the base station will never see it"
    )


def test_moveit_launch_can_separate_driver_from_consumers():
    """The split this all depends on. Gating both on one flag is what made the
    pad un-movable in the first place."""
    source = MOVEIT.read_text()
    assert 'DeclareLaunchArgument(\n                "use_joy_node"' in source
    # The driver is gated on the combined condition; the consumers are not.
    assert source.count("condition=joy_driver_condition") == 2, (
        "expected exactly the joy driver and the layout normalizer to be gated"
    )


def test_use_joy_node_reaches_the_moveit_launch():
    """full_hardware must forward it, or the rover's flag does nothing."""
    source = FULL.read_text()
    assert '"use_joy_node": LaunchConfiguration("use_joy_node")' in source


def test_base_never_subscribes_to_a_rover_raw_topic(base):
    """A raw subscription is ~740 Mbit/s and collapses the link."""
    assert "image_raw" not in base
    assert "/view/" in base or "camera_view.launch.py" in base


def test_base_does_not_publish_tf(base):
    """/tf and /joint_states come from the rover; a second publisher fights it."""
    assert "robot_state_publisher" not in base.split('"""', 2)[-1]


def test_both_halves_live_in_one_package():
    """The reason this file can compare them at all."""
    assert BASE.parent == ROVER.parent, (
        "the two ends of the field link have been split across packages again"
    )


def test_exactly_one_rviz_at_the_base_station(base):
    """Two windows both render and both subscribe, and the extra one is blank."""
    assert base.count('executable="rviz2"') == 1


def test_camera_view_starts_no_viewer():
    """The include that caused the second window.

    An included launch description inherits the parent's launch configurations,
    and DeclareLaunchArgument leaves an already-set one alone -- so this file
    declaring `use_rviz` meant base_station's own `use_rviz` switched on a
    viewer here too. Image plumbing does not own a viewer.
    """
    body = CAMERA_VIEW.read_text().split('"""', 2)[-1]
    # Comments are allowed to explain the absence; code is not allowed to undo it.
    code = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )
    assert "rviz2" not in code, "camera_view.launch.py starts a viewer again"
    assert "use_rviz" not in code, (
        "camera_view.launch.py declares use_rviz again -- the parent's value "
        "leaks straight into it"
    )


def test_base_station_scopes_its_includes(base):
    """forwarding=False: the child sees the listed values and nothing else.

    Without it every argument declared in base_station.launch.py is visible to
    camera_view.launch.py under the same name, which is a collision waiting for
    the next argument either file gains.
    """
    assert "forwarding=False" in base


def test_base_station_runs_a_checker_by_default(base):
    """full_hardware_checker is on the rover and prints to the rover's console;
    it cannot see the link, the pad or the downlink from the operator's end."""
    assert _default_of(base, "start_checker") == "true"
    assert "base_station_checker.launch.py" in base


def test_the_checker_never_subscribes_to_the_downlink():
    """Subscribing to a compressed stream is what pulls it over the antenna, and
    a second participant pulls a second copy. The checker measures the local
    /<cam>/view/* output instead, which costs nothing and proves more."""
    node = (PKG / "nodes" / "base_station_checker.py").read_text()
    body = node.split('"""', 2)[-1]
    subscriptions = re.findall(r"create_subscription\(\s*\w+,\s*([^,]+),", body)
    for topic in subscriptions:
        assert "/downlink/" not in topic, (
            f"the checker subscribes to {topic.strip()}, doubling the link load"
        )


def test_checker_launch_matches_the_camera_lists(base):
    """A camera the checker knows about and the decompressors do not is a
    permanently dead stream in the report, which reads as a link fault."""
    checker = CHECKER.read_text()
    assert _default_of(checker, "cameras") == _default_of(base, "cameras")
    assert _default_of(checker, "color_only") == _default_of(base, "color_only")


def test_checker_toggles_reach_the_checker_launch(base):
    """base_station's `checker_*` and the checker's own `check_*` are one flag.

    They are spelled differently because both files declare arguments into the
    same launch context and `check_rover` already means something here. A
    default that disagrees across the two is invisible: the operator sets the
    one they can see and the node reads the other.
    """
    checker = CHECKER.read_text()
    for outer, inner in (
        ("checker_arm", "check_arm"),
        ("checker_gripper", "check_gripper"),
        ("checker_drive", "check_drive"),
        ("checker_imu", "check_imu"),
    ):
        assert f'"{inner}": LaunchConfiguration("{outer}")' in base, (
            f"base_station.launch.py does not forward {outer} as {inner}"
        )
        assert _default_of(base, outer) == _default_of(checker, inner), (
            f"{outer} and {inner} have different defaults, so the flag the "
            f"operator sets is not the one the node reads"
        )


def test_the_checker_reads_the_rover_without_a_second_state_publisher(base):
    """It samples the rover's topics; it must not become a source of them.

    The rover-subsystem rows subscribe to /joint_states, /cmd_vel and the
    ODrive status topics. Publishing any of those from this end would be a
    checker that measures itself, which is the same mistake the /downlink rule
    exists to prevent.
    """
    node = (PKG / "nodes" / "base_station_checker.py").read_text()
    body = node.split('"""', 2)[-1]
    assert "create_publisher" not in body, (
        "the base station checker publishes something; it is a listener"
    )


# --- the checker must read the profile the stack actually uses ---------------
#
# base_station_checker reads the DDS config of its OWN process to report what
# the stack actually has. It used to read CYCLONEDDS_URI and parse Cyclone's
# XML; when the stack moved to Fast DDS that reported "not set" and an unpinned
# interface on a completely healthy link. A checker that cries wolf on a working
# system is worse than no checker -- people stop reading it, and then it is
# silent for the failure it exists to catch.

import sys                                                        # noqa: E402

sys.path.insert(0, str(PKG.parents[0] / "aries_common"))
from aries_common import comms                                    # noqa: E402

CHECKER_NODE = (PKG / "nodes" / "base_station_checker.py").read_text()


def test_checker_finds_the_pinned_interface(tmp_path):
    """The checker's own parsing, applied to a real generated profile."""
    path, _ = comms.write_dds_config(tmp_path / "dds.xml", require_link=False)
    text = Path(path).read_text()
    match = re.search(r"<interfaceWhiteList>\s*<address>([^<]+)</address>", text)
    assert match, "write_dds_config produced a profile the checker cannot parse"
    assert match.group(1).strip() == (comms.local_address() or "127.0.0.1")


def test_checker_reads_both_spellings_of_the_profile_variable():
    """comms sets both; Fast DDS renamed it at 2.12 and honours the old one."""
    assert '"FASTDDS_DEFAULT_PROFILES_FILE"' in CHECKER_NODE
    assert '"FASTRTPS_DEFAULT_PROFILES_FILE"' in CHECKER_NODE


def test_checker_does_not_strip_a_file_prefix():
    """Fast DDS silently ignores a file:// value, so one appearing IS the bug.

    Stripping it would hide exactly the misconfiguration worth reporting.
    """
    body = CHECKER_NODE[CHECKER_NODE.index("def _dds_environment"):]
    body = body[:body.index("def _any_publisher")]
    assert 'len("file://")' not in body, "the profile path must be taken as-is"


def test_nothing_in_comms_still_speaks_cyclone():
    """Cyclone was removed, not deprecated. A half-removed vendor is worse than
    either: it leaves code paths nobody exercises and docs that mislead."""
    for path in sorted((PKG / "nodes").glob("*.py")) + sorted((PKG / "launch").glob("*.py")):
        text = path.read_text()
        assert "CYCLONEDDS_URI" not in text, f"{path.name} still reads CYCLONEDDS_URI"
        assert "rmw_cyclonedds_cpp" not in text, f"{path.name} still names Cyclone"
