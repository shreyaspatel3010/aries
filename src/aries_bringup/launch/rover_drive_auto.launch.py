#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _can_exists(interface):
    return Path(f"/sys/class/net/{interface}").exists()


def launch_setup(context, *args, **kwargs):
    protocol = LaunchConfiguration("rover_hardware_protocol").perform(context)
    can_interface = LaunchConfiguration("can_interface").perform(context)

    use_joystick = LaunchConfiguration("use_joystick")
    use_joy_node = LaunchConfiguration("use_joy_node")
    joy_driver = LaunchConfiguration("joy_driver")
    joy_layout = LaunchConfiguration("joy_layout")
    joy_dev = LaunchConfiguration("joy_dev")
    use_imu = LaunchConfiguration("use_imu")
    imu_port = LaunchConfiguration("imu_port")
    imu_baudrate = LaunchConfiguration("imu_baudrate")
    imu_frame = LaunchConfiguration("imu_frame")
    imu_topic = LaunchConfiguration("imu_topic")
    use_lidar = LaunchConfiguration("use_lidar")
    lidar_sensor_ip = LaunchConfiguration("lidar_sensor_ip")
    picoscan_raw_imu_topic = LaunchConfiguration("picoscan_raw_imu_topic")
    picoscan_imu_topic = LaunchConfiguration("picoscan_imu_topic")
    setup_can = LaunchConfiguration("setup_can")

    if protocol == "auto":
        resolved = "odrive" if _can_exists(can_interface) else "mock_hardware"
    else:
        resolved = protocol

    actions = [
        LogInfo(msg=f"[rover auto] rover_hardware_protocol={protocol} resolved={resolved}")
    ]

    if resolved == "odrive":
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("aries_bringup"),
                        "launch",
                        "rover_drive_core.launch.py",
                    ])
                ),
                launch_arguments={
                    "use_imu": use_imu,
                    "use_joystick": use_joystick,
                    "use_joy_node": use_joy_node,
                    "joy_driver": joy_driver,
                    "can_interface": can_interface,
                    "setup_can": setup_can,
                    "imu_port": imu_port,
                    "imu_baudrate": imu_baudrate,
                    "imu_frame": imu_frame,
                    "imu_topic": imu_topic,
                    "use_lidar": use_lidar,
                    "lidar_sensor_ip": lidar_sensor_ip,
                    "picoscan_raw_imu_topic": picoscan_raw_imu_topic,
                    "picoscan_imu_topic": picoscan_imu_topic,
                    "joy_layout": joy_layout,
                    "joy_dev": joy_dev,
                }.items(),
            )
        )
    else:
        joystick_config = os.path.join(
            get_package_share_directory("aries_bringup"),
            "config",
            "joystick.yaml",
        )
        actions.append(
            LogInfo(msg="[rover auto] using mock_rover_drive because rover hardware is unavailable or disabled")
        )
        actions.append(
            Node(
                condition=IfCondition(use_joy_node),
                package="joy",
                executable=joy_driver,
                name="joy_node",
                parameters=[{"dev": joy_dev}],
                remappings=[("joy", "joy/raw")],
                output="screen",
            )
        )
        actions.append(
            Node(
                condition=IfCondition(use_joy_node),
                package="aries_moveit",
                executable="joy_layout_normalizer.py",
                name="joy_layout_normalizer",
                parameters=[{
                    "input_topic": "joy/raw",
                    "output_topic": "joy",
                    "layout": joy_layout,
                    "device": joy_dev,
                }],
                output="screen",
            )
        )
        actions.append(
            Node(
                package="aries_bringup",
                executable="mock_rover_drive.py",
                name="mock_rover_drive",
                output="screen",
                parameters=[joystick_config],
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "rover_hardware_protocol",
            default_value="auto",
            choices=["auto", "odrive", "mock_hardware"],
        ),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("setup_can", default_value="true"),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("use_joy_node", default_value="false"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
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
        OpaqueFunction(function=launch_setup),
    ])
