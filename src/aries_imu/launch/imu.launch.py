#!/usr/bin/env python3
"""Select one rover IMU while preserving BNO055 and picoScan compatibility."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aries_common.detect import (
    AUTO_VALUES,
    as_bool,
    as_int,
    resolve_imu_source,
    resolve_lidar_enabled,
)


def _start_imu(context, *args, **kwargs):
    start_mode = LaunchConfiguration("start_imu_driver").perform(context)
    driver_enabled = (
        str(start_mode).strip().lower() in AUTO_VALUES or as_bool(start_mode)
    )
    use_imu = LaunchConfiguration("use_imu").perform(context)
    bno_port = LaunchConfiguration("imu_port").perform(context)
    ybimu_port = LaunchConfiguration("ybimu_port").perform(context)
    ybimu_topic = LaunchConfiguration("ybimu_topic").perform(context)
    lidar_available = resolve_lidar_enabled(
        LaunchConfiguration("use_lidar").perform(context),
        LaunchConfiguration("lidar_sensor_ip").perform(context),
    )
    source, yb_available, bno_available = resolve_imu_source(
        use_imu,
        bno_port,
        lidar_available,
        ybimu_port,
    )
    if not driver_enabled:
        source = "none"

    actions = [
        LogInfo(
            msg=(
                f"[rover imu] use_imu={use_imu} selected={source}; "
                f"ybimu_available={yb_available}, "
                f"bno_available={bno_available}, "
                f"lidar_available={lidar_available}, "
                f"driver_enabled={driver_enabled}"
            )
        )
    ]

    if source == "ybimu":
        ybimu_config = os.path.join(
            get_package_share_directory("ybimu_ros2"),
            "config",
            "ybimu.yaml",
        )
        actions.append(
            Node(
                package="ybimu_ros2",
                executable="ybimu_driver",
                name="ybimu",
                output="screen",
                parameters=[
                    ybimu_config,
                    {
                        "port": ybimu_port,
                        "baudrate": as_int(
                            LaunchConfiguration("imu_baudrate").perform(context),
                            115200,
                        ),
                        "frame_id": LaunchConfiguration("ybimu_frame"),
                        "imu_topic": ybimu_topic,
                        "magnetic_field_topic": f"{ybimu_topic}/mag",
                        "pressure_topic": f"{ybimu_topic}/pressure",
                        "temperature_topic": f"{ybimu_topic}/temperature",
                        "altitude_topic": f"{ybimu_topic}/altitude",
                    },
                ],
                respawn=True,
                respawn_delay=5.0,
            )
        )
    elif source == "bno055":
        imu_frame = LaunchConfiguration("imu_frame").perform(context)
        actions.extend(
            [
                Node(
                    package="bno055",
                    executable="bno055",
                    name="bno055",
                    output="screen",
                    parameters=[
                        {
                            "connection_type": "uart",
                            "uart_port": bno_port,
                            "uart_baudrate": as_int(
                                LaunchConfiguration("imu_baudrate").perform(
                                    context
                                ),
                                115200,
                            ),
                            "frame_id": imu_frame,
                        }
                    ],
                ),
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name="base_to_bno055_tf",
                    arguments=[
                        "0",
                        "0",
                        "0.08",
                        "0",
                        "0",
                        "0",
                        "base_link",
                        imu_frame,
                    ],
                    output="screen",
                ),
            ]
        )
    elif source == "picoscan":
        actions.append(
            Node(
                package="aries_imu",
                executable="picoscan_imu_relay.py",
                name="picoscan_imu_relay",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration(
                            "picoscan_raw_imu_topic"
                        ),
                        "output_topic": LaunchConfiguration(
                            "picoscan_imu_topic"
                        ),
                        # Preserve the sensor frame. robot_localization applies
                        # the existing URDF transform to vector measurements.
                        "target_frame": "",
                    }
                ],
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
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
            # Existing BNO055 interface is retained for compatibility.
            DeclareLaunchArgument("imu_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("imu_baudrate", default_value="115200"),
            DeclareLaunchArgument("imu_frame", default_value="bno055"),
            DeclareLaunchArgument(
                "bno055_topic", default_value="/bno055/imu"
            ),
            # YaBoom uses a separate persistent device path and ROS interface.
            DeclareLaunchArgument(
                "ybimu_port", default_value="/dev/imu_ybimu"
            ),
            DeclareLaunchArgument("ybimu_frame", default_value="imu_frame"),
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
                "picoscan_raw_imu_topic",
                default_value="/picoscan/imu_raw",
            ),
            DeclareLaunchArgument(
                "picoscan_imu_topic", default_value="/picoscan/imu"
            ),
            OpaqueFunction(function=_start_imu),
        ]
    )
