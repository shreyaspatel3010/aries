"""The rover and the base station have to agree, and nothing checks that at runtime.

Two mistakes here are silent and expensive in the field:

  * both machines running a joy driver -- two publishers on /joy, consumers
    seeing them interleaved at double rate, buttons that appear to chatter;
  * either machine starting nodes before the DDS environment is set -- the
    stack lands on domain 0 with the default RMW, and the far end sees nothing
    at all on a link that pings fine.

Neither fails loudly at runtime, so they are pinned here instead.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
BASE = SRC / "aries_base_station" / "launch" / "base_station.launch.py"
ROVER = SRC / "aries_bringup" / "launch" / "rover_field.launch.py"
FULL = SRC / "aries_bringup" / "launch" / "full_hardware.launch.py"
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
