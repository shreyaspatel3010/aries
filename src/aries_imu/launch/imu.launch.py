#!/usr/bin/env python3
"""Select the BNO055 or picoScan IMU without publishing odometry/TF."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aries_common.detect import as_int, resolve_imu_source, resolve_lidar_enabled


def _start_imu(context, *args, **kwargs):
    use_imu = LaunchConfiguration("use_imu").perform(context)
    imu_port = LaunchConfiguration("imu_port").perform(context)
    lidar_available = resolve_lidar_enabled(
        LaunchConfiguration("use_lidar").perform(context),
        LaunchConfiguration("lidar_sensor_ip").perform(context),
    )
    source, bno_available = resolve_imu_source(use_imu, imu_port, lidar_available)
    actions = [
        LogInfo(msg=(
            f"[rover imu] use_imu={use_imu} selected={source}; "
            f"bno_available={bno_available}, lidar_available={lidar_available}"
        ))
    ]

    if source == "bno055":
        imu_frame = LaunchConfiguration("imu_frame").perform(context)
        actions.extend([
            Node(
                package="bno055",
                executable="bno055",
                name="bno055",
                output="screen",
                parameters=[{
                    "connection_type": "uart",
                    "uart_port": imu_port,
                    "uart_baudrate": as_int(
                        LaunchConfiguration("imu_baudrate").perform(context), 115200
                    ),
                    "frame_id": imu_frame,
                }],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_bno055_tf",
                arguments=["0", "0", "0.08", "0", "0", "0", "base_link", imu_frame],
                output="screen",
            ),
        ])
    elif source == "picoscan":
        actions.append(
            Node(
                package="aries_imu",
                executable="picoscan_imu_relay.py",
                name="picoscan_imu_relay",
                output="screen",
                parameters=[{
                    "input_topic": LaunchConfiguration("picoscan_raw_imu_topic"),
                    "output_topic": LaunchConfiguration("picoscan_imu_topic"),
                    # Preserve Lidar_Scan_Link so robot_localization can apply
                    # the existing URDF transform instead of merely relabeling axes.
                    "target_frame": "",
                }],
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_imu",
            default_value="auto",
            choices=["auto", "true", "false", "bno055", "picoscan"],
        ),
        DeclareLaunchArgument("imu_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("imu_frame", default_value="bno055"),
        DeclareLaunchArgument("use_lidar", default_value="auto", choices=["auto", "true", "false"]),
        DeclareLaunchArgument("lidar_sensor_ip", default_value="169.254.136.69"),
        DeclareLaunchArgument("picoscan_raw_imu_topic", default_value="/picoscan/imu_raw"),
        DeclareLaunchArgument("picoscan_imu_topic", default_value="/picoscan/imu"),
        OpaqueFunction(function=_start_imu),
    ])
