#!/usr/bin/env python3
"""Start the real SICK picoScan150 and publish a cleaned ``/scan`` topic."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aries_common.detect import package_exists, resolve_lidar_enabled


def _start_lidar(context, *args, **kwargs):
    mode = LaunchConfiguration("use_lidar").perform(context)
    sensor_ip = LaunchConfiguration("lidar_sensor_ip").perform(context)
    if not resolve_lidar_enabled(mode, sensor_ip):
        return [LogInfo(msg=f"[lidar] use_lidar={mode}; picoScan at {sensor_ip} not started")]

    if not package_exists("sick_scan_xd"):
        return [LogInfo(msg="[lidar] sick_scan_xd is not installed; picoScan not started")]

    driver_launch = os.path.join(
        get_package_share_directory("sick_scan_xd"), "launch", "sick_picoscan.launch"
    )
    driver = Node(
        package="sick_scan_xd",
        executable="sick_generic_caller",
        name="sick_picoscan",
        output="screen",
        # sick_generic_caller parses these name:=value arguments after loading
        # its ROS1-style XML configuration. ROS parameter files are ignored by
        # this driver version for the same fields.
        arguments=[
            driver_launch,
            ["hostname:=", LaunchConfiguration("lidar_sensor_ip")],
            ["udp_receiver_ip:=", LaunchConfiguration("lidar_host_ip")],
            ["publish_frame_id:=", LaunchConfiguration("lidar_frame")],
            ["publish_laserscan_fullframe_topic:=", LaunchConfiguration("lidar_raw_scan_topic")],
            ["publish_imu_frame_id:=", LaunchConfiguration("lidar_frame")],
            ["imu_topic:=", LaunchConfiguration("lidar_raw_imu_topic")],
            "tf_publish_rate:=0.0",
        ],
    )
    relay = Node(
        package="aries_lidar",
        executable="lidar_scan_frame_relay.py",
        name="lidar_scan_frame_relay",
        output="screen",
        parameters=[{
            "input_topic": LaunchConfiguration("lidar_raw_scan_topic"),
            "output_topic": LaunchConfiguration("lidar_scan_topic"),
            "target_frame": LaunchConfiguration("lidar_frame"),
            "restamp_to_ros_time": LaunchConfiguration("lidar_restamp"),
        }],
    )
    return [
        LogInfo(msg=f"[lidar] starting SICK picoScan150 at {sensor_ip}"),
        driver,
        relay,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_lidar", default_value="auto", choices=["auto", "true", "false"]
        ),
        DeclareLaunchArgument("lidar_sensor_ip", default_value="169.254.136.69"),
        DeclareLaunchArgument("lidar_host_ip", default_value="169.254.180.121"),
        DeclareLaunchArgument("lidar_frame", default_value="Lidar_Scan_Link"),
        DeclareLaunchArgument("lidar_raw_scan_topic", default_value="/picoscan/scan_raw"),
        DeclareLaunchArgument("lidar_scan_topic", default_value="/scan"),
        DeclareLaunchArgument("lidar_raw_imu_topic", default_value="/picoscan/imu_raw"),
        DeclareLaunchArgument(
            "lidar_restamp",
            default_value="true",
            description="Restamp scans with ROS time for RViz/TF compatibility.",
        ),
        OpaqueFunction(function=_start_lidar),
    ])
