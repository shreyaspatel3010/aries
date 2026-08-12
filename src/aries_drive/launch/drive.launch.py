#!/usr/bin/env python3
"""
Rover drive backend: six ODrive axes over CAN, or the mock drive when CAN is absent.

`rover_hardware_protocol:=auto` picks odrive when the CAN interface exists and
mock_hardware otherwise. On the real backend the CAN interface is brought up
first (setup_can, which needs the sudoers rule in setup/rover_can), and the
poller, fail-safe cmd_vel bridge, and joystick-to-Twist path start two seconds
behind the ODrive nodes so they do not race the CAN bus coming up.

The real backend has one motor command owner:

  /joy -> /cmd_vel/teleop -> waypoint arbiter -> /cmd_vel
       -> cmd_vel_odrive_bridge -> six ODrive ControlMessage topics

The legacy joystick-to-ODrive controller is deliberately not started because
it would bypass the waypoint arbiter and collision supervisor.

In mock mode the joystick still runs, but the rover controller does not.
mock_rover_drive owns odom -> base_footprint there.

CAN setup is automatic by default. Install the limited sudoers rule to avoid a
password prompt:

  sudo visudo -cf src/aries_drive/setup/rover_can
  sudo install -m 440 src/aries_drive/setup/rover_can /etc/sudoers.d/rover_can

Or configure CAN yourself and launch with `setup_can:=false`.
"""

import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from aries_common.detect import as_bool, resolve_rover_backend

ODRIVE_AXES = 6
ODRIVE_STARTUP_DELAY = 2.0


def _joystick_launch(use_joystick_controller):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("aries_teleop"), "launch", "joystick.launch.py"])
        ),
        launch_arguments={
            "use_joy_node": LaunchConfiguration("use_joy_node"),
            "use_joystick_controller": use_joystick_controller,
            "joy_driver": LaunchConfiguration("joy_driver"),
            "joy_layout": LaunchConfiguration("joy_layout"),
            "joy_dev": LaunchConfiguration("joy_dev"),
        }.items(),
    )


def _mock_actions():
    joystick_config = os.path.join(
        get_package_share_directory("aries_teleop"), "config", "joystick.yaml"
    )
    return [
        LogInfo(msg="[rover drive] using mock_rover_drive because rover hardware is unavailable or disabled"),
        _joystick_launch("false"),
        Node(
            package="aries_drive",
            executable="mock_rover_drive.py",
            name="mock_rover_drive",
            output="screen",
            parameters=[joystick_config],
        ),
    ]


def _odrive_actions(can_interface):
    drive_config = os.path.join(
        get_package_share_directory("aries_drive"),
        "config",
        "cmd_vel_odrive_bridge.yaml",
    )
    joystick_config = os.path.join(
        get_package_share_directory("aries_teleop"),
        "config",
        "joystick.yaml",
    )
    odrive_nodes = [
        Node(
            package="odrive_can",
            executable="odrive_can_node",
            name="can_node",
            namespace=f"odrive_axis{i}",
            parameters=[{
                "node_id": i,
                "interface": can_interface,
            }],
            output="screen",
        )
        for i in range(ODRIVE_AXES)
    ]

    odrive_poller = Node(
        package="aries_drive",
        executable="odrive_can_poller.py",
        name="odrive_can_poller",
        parameters=[{
            "interface": can_interface,
            "num_axes": ODRIVE_AXES,
            # The Aries ODrives already broadcast encoder estimates at 100 Hz.
            # Two Hz is sufficient for the remaining diagnostic GET traffic
            # without crowding the 250 kbit/s control bus.
            "poll_rate_hz": 2.0,
        }],
        output="screen",
    )

    cmd_vel_bridge = Node(
        condition=IfCondition(LaunchConfiguration("use_cmd_vel_bridge")),
        package="aries_drive",
        executable="cmd_vel_odrive_bridge.py",
        name="cmd_vel_odrive_bridge",
        parameters=[
            drive_config,
            {
                "cmd_vel_topic": LaunchConfiguration("drive_cmd_vel_topic"),
                "auto_arm": LaunchConfiguration("drive_auto_arm"),
                "command_timeout_s": LaunchConfiguration("drive_command_timeout_s"),
                "max_linear_mps": LaunchConfiguration("drive_max_linear_mps"),
                "max_angular_rps": LaunchConfiguration("drive_max_angular_rps"),
                "max_wheel_rps": LaunchConfiguration("drive_max_wheel_rps"),
                "wheel_accel_rps2": LaunchConfiguration("drive_wheel_accel_rps2"),
            },
        ],
        output="screen",
    )

    cmd_vel_joystick = Node(
        condition=IfCondition(LaunchConfiguration("use_joystick")),
        package="aries_teleop",
        executable="rover_cmd_vel_joystick.py",
        name="rover_cmd_vel_joystick",
        parameters=[joystick_config],
        output="screen",
    )

    cmd_vel_relay = Node(
        condition=IfCondition(LaunchConfiguration("use_cmd_vel_relay")),
        package="aries_teleop",
        executable="cmd_vel_teleop_relay.py",
        name="cmd_vel_teleop_relay",
        parameters=[{
            "input_topic": "/cmd_vel/teleop",
            "output_topic": LaunchConfiguration("drive_cmd_vel_topic"),
        }],
        output="screen",
    )

    return [
        *odrive_nodes,
        TimerAction(
            period=ODRIVE_STARTUP_DELAY,
            actions=[
                odrive_poller,
                cmd_vel_bridge,
                _joystick_launch("false"),
                cmd_vel_joystick,
                cmd_vel_relay,
            ],
        ),
    ]


def _can_setup_process(can_interface, can_bitrate):
    quoted_interface = shlex.quote(can_interface)
    quoted_bitrate = shlex.quote(can_bitrate)
    return ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            f"sudo -n ip link set {quoted_interface} down 2>/dev/null; "
            f"sudo -n ip link set {quoted_interface} up type can bitrate {quoted_bitrate}",
        ],
        output="screen",
    )


def _start_rover_hardware(context, *args, **kwargs):
    protocol = LaunchConfiguration("rover_hardware_protocol").perform(context)
    can_interface = LaunchConfiguration("can_interface").perform(context)
    can_bitrate = LaunchConfiguration("can_bitrate").perform(context)
    setup_can = as_bool(LaunchConfiguration("setup_can").perform(context))

    resolved = resolve_rover_backend(protocol, can_interface)
    actions = [LogInfo(msg=f"[rover drive] rover_hardware_protocol={protocol} resolved={resolved}")]

    if resolved != "odrive":
        actions.extend(_mock_actions())
        return actions

    start_nodes = _odrive_actions(can_interface)

    if not setup_can:
        actions.extend(start_nodes)
        return actions

    can_setup = _can_setup_process(can_interface, can_bitrate)
    actions.extend([
        can_setup,
        RegisterEventHandler(OnProcessExit(target_action=can_setup, on_exit=start_nodes)),
    ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "rover_hardware_protocol",
            default_value="auto",
            choices=["auto", "odrive", "mock_hardware"],
            description="auto picks odrive when the CAN interface exists, otherwise mock.",
        ),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("can_bitrate", default_value="250000"),
        DeclareLaunchArgument(
            "setup_can",
            default_value="true",
            description="Bring the CAN interface up with sudo -n before starting ODrive nodes.",
        ),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("use_joy_node", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node",
                              choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto",
                              choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument(
            "use_cmd_vel_bridge",
            default_value="true",
            description="Convert the single hardware-facing Twist into ODrive commands.",
        ),
        DeclareLaunchArgument("drive_cmd_vel_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument(
            "drive_auto_arm",
            default_value="false",
            description="Arm every ODrive automatically. Keep false for physical testing.",
        ),
        DeclareLaunchArgument(
            "drive_command_timeout_s",
            default_value="0.25",
            description="Stop immediately when the hardware-facing Twist becomes stale.",
        ),
        DeclareLaunchArgument(
            "drive_max_linear_mps",
            default_value="0.45",
            description="Final hardware-bridge linear speed cap in m/s.",
        ),
        DeclareLaunchArgument(
            "drive_max_angular_rps",
            default_value="2.10",
            description="Final hardware-bridge yaw-rate cap in rad/s.",
        ),
        DeclareLaunchArgument(
            "drive_max_wheel_rps",
            default_value="1.50",
            description="Final per-wheel ODrive velocity cap in revolutions/s.",
        ),
        DeclareLaunchArgument(
            "drive_wheel_accel_rps2",
            default_value="3.0",
            description="ODrive wheel-command ramp limit in revolutions/s^2.",
        ),
        DeclareLaunchArgument(
            "use_cmd_vel_relay",
            default_value="false",
            description=(
                "Relay manual /cmd_vel/teleop when no arbiter is running. "
                "Keep false when launching the waypoint stack."
            ),
        ),
        OpaqueFunction(function=_start_rover_hardware),
    ])
