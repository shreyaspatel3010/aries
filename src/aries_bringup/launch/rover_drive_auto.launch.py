#!/usr/bin/env python3
"""Compatibility wrapper around the modular rover bringup.

The full arm/gripper launch already owns robot_state_publisher and RViz, so this
wrapper starts only rover hardware, localization, teleop, and encoder-derived
wheel joint states.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rover = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "rover_drive.launch.py",
                ]
            )
        ),
        launch_arguments={
            "rover_hardware_protocol": LaunchConfiguration(
                "rover_hardware_protocol"
            ),
            "can_interface": LaunchConfiguration("can_interface"),
            "can_bitrate": LaunchConfiguration("can_bitrate"),
            "setup_can": LaunchConfiguration("setup_can"),
            "use_joystick": LaunchConfiguration("use_joystick"),
            "use_joy_node": LaunchConfiguration("use_joy_node"),
            "joy_driver": LaunchConfiguration("joy_driver"),
            "joy_layout": LaunchConfiguration("joy_layout"),
            "joy_dev": LaunchConfiguration("joy_dev"),
            "use_cmd_vel_bridge": "true",
            "drive_auto_arm": LaunchConfiguration("drive_auto_arm"),
            "use_cmd_vel_relay": "true",
            "use_robot_description": "false",
            "use_wheel_joint_publisher": "true",
            "use_rviz": "false",
            "start_checker": "false",
            "start_imu_driver": LaunchConfiguration("start_imu_driver"),
            "use_imu": LaunchConfiguration("use_imu"),
            "imu_port": LaunchConfiguration("imu_port"),
            "ybimu_port": LaunchConfiguration("ybimu_port"),
            "imu_baudrate": LaunchConfiguration("imu_baudrate"),
            "imu_frame": LaunchConfiguration("imu_frame"),
            "ybimu_frame": LaunchConfiguration("ybimu_frame"),
            "bno055_topic": LaunchConfiguration("imu_topic"),
            "ybimu_topic": LaunchConfiguration("ybimu_topic"),
            "use_lidar": LaunchConfiguration("use_lidar"),
            "lidar_sensor_ip": LaunchConfiguration("lidar_sensor_ip"),
            "lidar_host_ip": LaunchConfiguration("lidar_host_ip"),
            "lidar_frame": LaunchConfiguration("lidar_frame"),
            "lidar_scan_topic": LaunchConfiguration("lidar_scan_topic"),
            "lidar_raw_scan_topic": LaunchConfiguration(
                "lidar_raw_scan_topic"
            ),
            "lidar_restamp": LaunchConfiguration("lidar_restamp"),
            "picoscan_localization_imu_raw_topic": LaunchConfiguration(
                "picoscan_raw_imu_topic"
            ),
            "picoscan_localization_imu_topic": LaunchConfiguration(
                "picoscan_imu_topic"
            ),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rover_hardware_protocol",
                default_value="auto",
                choices=["auto", "odrive", "mock_hardware"],
            ),
            DeclareLaunchArgument("can_interface", default_value="can0"),
            DeclareLaunchArgument("can_bitrate", default_value="250000"),
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
                    "ybimu",
                    "bno055",
                    "picoscan",
                ],
            ),
            DeclareLaunchArgument("imu_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument(
                "ybimu_port", default_value="/dev/imu_ybimu"
            ),
            DeclareLaunchArgument("imu_baudrate", default_value="115200"),
            DeclareLaunchArgument("imu_frame", default_value="bno055"),
            DeclareLaunchArgument("ybimu_frame", default_value="imu_frame"),
            DeclareLaunchArgument("imu_topic", default_value="/bno055/imu"),
            DeclareLaunchArgument(
                "ybimu_topic", default_value="/ybimu/imu"
            ),
            DeclareLaunchArgument(
                "use_lidar",
                default_value="auto",
                choices=["auto", "true", "false"],
            ),
            DeclareLaunchArgument(
                "lidar_sensor_ip", default_value="169.254.136.69"
            ),
            DeclareLaunchArgument(
                "lidar_host_ip", default_value="169.254.180.121"
            ),
            DeclareLaunchArgument(
                "lidar_frame", default_value="Lidar_Scan_Link"
            ),
            DeclareLaunchArgument("lidar_scan_topic", default_value="/scan"),
            DeclareLaunchArgument(
                "lidar_raw_scan_topic",
                default_value="/picoscan/scan_raw",
            ),
            DeclareLaunchArgument(
                "picoscan_raw_imu_topic",
                default_value="/picoscan/imu_raw",
            ),
            DeclareLaunchArgument(
                "picoscan_imu_topic", default_value="/picoscan/imu"
            ),
            DeclareLaunchArgument("lidar_restamp", default_value="true"),
            rover,
        ]
    )
