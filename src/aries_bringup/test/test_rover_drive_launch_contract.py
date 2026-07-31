"""Contract tests for the modular physical rover bringup."""

import ast
from pathlib import Path


LAUNCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "rover_drive.launch.py"
)
FULL_HARDWARE_PATH = LAUNCH_PATH.with_name("full_hardware.launch.py")
ARM_WRAPPER_PATH = LAUNCH_PATH.with_name("aries_hardware.launch.py")
MOVEIT_HARDWARE_PATH = (
    LAUNCH_PATH.parents[2]
    / "aries_moveit"
    / "moveit_config"
    / "launch"
    / "aries_hardware.launch.py"
)


def _declared_default(argument_name, launch_path=LAUNCH_PATH):
    tree = ast.parse(launch_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "DeclareLaunchArgument"
        ):
            continue
        if not node.args or ast.literal_eval(node.args[0]) != argument_name:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default_value":
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"launch argument not found: {argument_name}")


def test_physical_drive_starts_disarmed():
    assert _declared_default("drive_auto_arm") == "false"


def test_preflight_checker_accepts_disarmed_axes():
    assert _declared_default("checker_require_closed_loop") == "false"


def test_standalone_manual_drive_routes_teleop_twist():
    assert _declared_default("use_cmd_vel_relay") == "true"


def test_fail_safe_bridge_is_the_physical_command_owner():
    source = LAUNCH_PATH.read_text(encoding="utf-8")
    assert '"aries_drive", "drive.launch.py"' in source
    assert "custom_joystick_controller.py" not in source


def test_waypoint_stack_is_not_an_implicit_dependency():
    source = LAUNCH_PATH.read_text(encoding="utf-8")
    assert "grasshopper_waypoint_follower" not in source


def test_full_hardware_has_one_wheel_joint_state_owner():
    full_source = FULL_HARDWARE_PATH.read_text(encoding="utf-8")
    wrapper_source = ARM_WRAPPER_PATH.read_text(encoding="utf-8")
    moveit_source = MOVEIT_HARDWARE_PATH.read_text(encoding="utf-8")

    assert (
        _declared_default(
            "use_static_wheel_joint_publisher", FULL_HARDWARE_PATH
        )
        == "false"
    )
    assert (
        '"use_wheel_joint_publisher": LaunchConfiguration('
        in full_source
    )
    assert (
        '"use_wheel_joint_publisher": LaunchConfiguration('
        in wrapper_source
    )
    assert 'LaunchConfiguration("use_wheel_joint_publisher")' in moveit_source
