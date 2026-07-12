import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_joystick = LaunchConfiguration("use_joystick")
    joy_driver = LaunchConfiguration("joy_driver")
    joy_layout = LaunchConfiguration("joy_layout")
    joy_dev = LaunchConfiguration("joy_dev")
    use_rover_joystick = LaunchConfiguration("use_rover_joystick")
    use_rover_joy_node = LaunchConfiguration("use_rover_joy_node")

    joystick_config = os.path.join(
        get_package_share_directory("aries_bringup"),
        "config",
        "joystick.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"]),
        DeclareLaunchArgument("use_rover_joystick", default_value="true"),
        DeclareLaunchArgument("use_rover_joy_node", default_value="false"),

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
                "joy_driver": joy_driver,
                "joy_layout": joy_layout,
                "joy_dev": joy_dev,
                "joystick_control_mode": LaunchConfiguration("joystick_control_mode"),
            }.items(),
        ),

        Node(
            condition=IfCondition(use_rover_joystick),
            package="aries_bringup",
            executable="rover_cmd_vel_joystick.py",
            name="rover_cmd_vel_joystick",
            output="screen",
            parameters=[joystick_config],
        ),

        Node(
            condition=IfCondition(use_rover_joy_node),
            package="joy",
            executable=joy_driver,
            name="rover_joy_node",
            parameters=[{"dev": joy_dev}],
            remappings=[("joy", "joy/raw")],
            output="screen",
        ),

        Node(
            condition=IfCondition(use_rover_joy_node),
            package="aries_moveit",
            executable="joy_layout_normalizer.py",
            name="rover_joy_layout_normalizer",
            parameters=[{
                "input_topic": "joy/raw",
                "output_topic": "joy",
                "layout": joy_layout,
                "device": joy_dev,
            }],
            output="screen",
        ),
    ])
