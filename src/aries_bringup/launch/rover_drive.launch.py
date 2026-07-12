#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("setup_can", default_value="true"),
        DeclareLaunchArgument("use_imu", default_value="auto"),
        DeclareLaunchArgument("imu_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("imu_frame", default_value="bno055"),
        DeclareLaunchArgument("imu_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("use_lidar", default_value="auto", choices=["auto", "true", "false"]),
        DeclareLaunchArgument("lidar_sensor_ip", default_value="169.254.136.69"),
        DeclareLaunchArgument("lidar_host_ip", default_value="169.254.180.121"),
        DeclareLaunchArgument("lidar_frame", default_value="Lidar_Scan_Link"),
        DeclareLaunchArgument("lidar_scan_topic", default_value="/scan"),
        DeclareLaunchArgument("lidar_raw_scan_topic", default_value="/picoscan/scan_raw"),
        DeclareLaunchArgument("picoscan_raw_imu_topic", default_value="/picoscan/imu_raw"),
        DeclareLaunchArgument("picoscan_imu_topic", default_value="/picoscan/imu"),
        DeclareLaunchArgument("lidar_restamp", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_lidar"),
                    "launch",
                    "lidar.launch.py",
                ])
            ),
            launch_arguments={
                "use_lidar": LaunchConfiguration("use_lidar"),
                "lidar_sensor_ip": LaunchConfiguration("lidar_sensor_ip"),
                "lidar_host_ip": LaunchConfiguration("lidar_host_ip"),
                "lidar_frame": LaunchConfiguration("lidar_frame"),
                "lidar_scan_topic": LaunchConfiguration("lidar_scan_topic"),
                "lidar_raw_scan_topic": LaunchConfiguration("lidar_raw_scan_topic"),
                "lidar_raw_imu_topic": LaunchConfiguration("picoscan_raw_imu_topic"),
                "lidar_restamp": LaunchConfiguration("lidar_restamp"),
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "rover_drive_core.launch.py",
                ])
            ),
            launch_arguments={
                "use_imu": LaunchConfiguration("use_imu"),
                "use_joystick": "true",
                "use_joy_node": "true",
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "can_interface": LaunchConfiguration("can_interface"),
                "setup_can": LaunchConfiguration("setup_can"),
                "imu_port": LaunchConfiguration("imu_port"),
                "imu_baudrate": LaunchConfiguration("imu_baudrate"),
                "imu_frame": LaunchConfiguration("imu_frame"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "use_lidar": LaunchConfiguration("use_lidar"),
                "lidar_sensor_ip": LaunchConfiguration("lidar_sensor_ip"),
                "picoscan_raw_imu_topic": LaunchConfiguration("picoscan_raw_imu_topic"),
                "picoscan_imu_topic": LaunchConfiguration("picoscan_imu_topic"),
            }.items(),
        )
    ])
