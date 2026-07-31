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
    use_sim_ekf = LaunchConfiguration("use_sim_ekf")
    use_cmd_vel_relay = LaunchConfiguration("use_cmd_vel_relay")

    joystick_config = os.path.join(
        get_package_share_directory("aries_teleop"),
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
        DeclareLaunchArgument("use_cmd_vel_relay", default_value="true"),
        DeclareLaunchArgument(
            "use_sim_ekf",
            default_value="true",
            description=(
                "Fuse Gazebo ground-truth odometry and IMU into "
                "/odometry/filtered."
            ),
        ),

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

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_localization"),
                    "launch",
                    "localization.launch.py",
                ])
            ),
            condition=IfCondition(use_sim_ekf),
            launch_arguments={
                "use_sim_ekf": "true",
                "use_sim_time": "true",
                "sim_odom_topic": "/ground_truth/odom",
                "sim_imu_topic": "/imu",
                "filtered_odom_topic": "/odometry/filtered",
            }.items(),
        ),

        Node(
            condition=IfCondition(use_rover_joystick),
            package="aries_teleop",
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
            package="aries_teleop",
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

        Node(
            condition=IfCondition(use_cmd_vel_relay),
            package="aries_teleop",
            executable="cmd_vel_teleop_relay.py",
            name="cmd_vel_teleop_relay",
            output="screen",
            parameters=[
                {
                    "input_topic": "/cmd_vel/teleop",
                    "output_topic": "/cmd_vel",
                }
            ],
        ),
    ])
