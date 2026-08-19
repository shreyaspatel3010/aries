#!/usr/bin/env python3
"""Compatibility entry point that forces the modular real ODrive backend."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from aries_common.devices import device_str


def generate_launch_description():
    core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "rover_drive_auto.launch.py",
                ]
            )
        ),
        launch_arguments={
            "rover_hardware_protocol": "odrive",
            "can_interface": LaunchConfiguration("can_interface"),
            "can_bitrate": LaunchConfiguration("can_bitrate"),
            "setup_can": LaunchConfiguration("setup_can"),
            "drive_auto_arm": LaunchConfiguration("drive_auto_arm"),
            "use_joystick": LaunchConfiguration("use_joystick"),
            "use_joy_node": LaunchConfiguration("use_joy_node"),
            "joy_driver": LaunchConfiguration("joy_driver"),
            "joy_layout": LaunchConfiguration("joy_layout"),
            "joy_dev": LaunchConfiguration("joy_dev"),
            "start_imu_driver": LaunchConfiguration("start_imu_driver"),
            "use_imu": LaunchConfiguration("use_imu"),
            "imu_port": LaunchConfiguration("imu_port"),
            "imu_baudrate": LaunchConfiguration("imu_baudrate"),
            "imu_frame": LaunchConfiguration("imu_frame"),
            "imu_topic": LaunchConfiguration("imu_topic"),
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("can_interface", default_value=device_str("rover.can_interface")),
            DeclareLaunchArgument("can_bitrate", default_value=device_str("rover.can_bitrate")),
            DeclareLaunchArgument("setup_can", default_value="true"),
            DeclareLaunchArgument("drive_auto_arm", default_value="false"),
            DeclareLaunchArgument("use_joystick", default_value="true"),
            DeclareLaunchArgument("use_joy_node", default_value="false"),
            DeclareLaunchArgument(
                "joy_driver",
                default_value="game_controller_node",
                choices=["game_controller_node", "joy_node"],
            ),
            DeclareLaunchArgument(
                "joy_layout",
                default_value="auto",
                choices=[
                    "auto",
                    "dongle",
                    "bluetooth",
                    "game_controller",
                    "passthrough",
                ],
            ),
            DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
            DeclareLaunchArgument(
                "start_imu_driver",
                default_value="auto",
                choices=["auto", "true", "false"],
            ),
            DeclareLaunchArgument(
                "use_imu",
                default_value="auto",
                choices=[
                    "auto",
                    "true",
                    "false",
                    "microstrain",
                ],
            ),
            DeclareLaunchArgument("imu_port", default_value=device_str("imu.port")),
            DeclareLaunchArgument("imu_baudrate", default_value="115200"),
            DeclareLaunchArgument("imu_frame", default_value="imu_frame"),
            DeclareLaunchArgument(
                "imu_topic", default_value="/microstrain/imu/data"
            ),
            core,
        ]
    )
