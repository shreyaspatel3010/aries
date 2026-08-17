"""Compatibility wrapper for the maintained full Aries simulation launch.

The core ``aries`` launch owns every runtime node. Keeping rover teleop, the
cmd_vel relay, and localization in one launch prevents two publishers while
also avoiding launch-configuration leakage between an include and this
wrapper. In particular, the old wrapper passed ``false`` for the core copies
and then accidentally evaluated its own conditions with those same values, so
only arm joystick control started.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_joystick = LaunchConfiguration("use_joystick")
    use_rover_joystick = LaunchConfiguration("use_rover_joystick")
    use_cmd_vel_relay = LaunchConfiguration("use_cmd_vel_relay")
    use_sim_ekf = LaunchConfiguration("use_sim_ekf")

    return LaunchDescription([
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument(
            "joy_driver",
            default_value="game_controller_node",
            choices=["game_controller_node", "joy_node"],
        ),
        DeclareLaunchArgument(
            "joy_layout",
            default_value="auto",
            choices=[
                "auto", "dongle", "bluetooth", "game_controller",
                "passthrough",
            ],
        ),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument(
            "joystick_control_mode",
            default_value="servo",
            choices=["move_group", "servo"],
        ),
        DeclareLaunchArgument("use_rover_joystick", default_value="true"),
        DeclareLaunchArgument("use_cmd_vel_relay", default_value="true"),
        DeclareLaunchArgument("use_sim_ekf", default_value="true"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries"),
                    "launch",
                    "my_robot.launch.py",
                ])
            ),
            launch_arguments={
                "use_joystick": use_joystick,
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "joystick_control_mode": LaunchConfiguration(
                    "joystick_control_mode"
                ),
                "use_rover_joystick": use_rover_joystick,
                "use_cmd_vel_relay": use_cmd_vel_relay,
                "use_sim_ekf": use_sim_ekf,
            }.items(),
        ),
    ])
