#!/usr/bin/env python3
"""
Wheel odometry + EKF.

This package fuses; it does not drive sensors. The IMU node itself is started by
aries_imu, and this file only needs to know which source won, because that
decides which EKF config is loaded:

  ybimu     YaBoom ybimu on its serial port -> ekf_config.yaml
  bno055    BNO055 on its serial port        -> ekf_config.yaml
  none      no IMU, wheel odometry only     -> ekf_odom_only.yaml

It resolves the source with the same aries_common probes aries_imu uses, so both
packages reach the same answer from the same arguments without having to agree
out of band.

Skipped entirely on the mock drive backend, where mock_rover_drive already owns
/odom and the odom -> base_footprint transform.

Odom.py and the ekf_*.yaml configs live in rover_nav.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aries_common.detect import resolve_imu_source, resolve_rover_backend

ODOM_ONLY_CONFIG = [
    False, False, False,  # x, y, z
    False, False, False,  # roll, pitch, yaw
    True, False, False,   # vx, vy, vz
    False, False, True,   # vroll, vpitch, vyaw
    False, False, False,  # accelerations
]


def _ekf_config(name):
    return os.path.join(get_package_share_directory("rover_nav"), "config", name)


def _sim_ekf_config():
    return os.path.join(
        get_package_share_directory("aries_localization"),
        "config",
        "ekf_sim.yaml",
    )


def _enabled(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _start_localization(context, *args, **kwargs):
    use_sim_ekf = _enabled(LaunchConfiguration("use_sim_ekf").perform(context))
    if use_sim_ekf:
        odom_topic = LaunchConfiguration("sim_odom_topic")
        imu_topic = LaunchConfiguration("sim_imu_topic")
        filtered_topic = LaunchConfiguration("filtered_odom_topic")
        return [
            LogInfo(msg=(
                "[rover localization] simulation EKF enabled: "
                "/ground_truth/odom + /imu -> /odometry/filtered; "
                "EKF owns corrected odom->base_footprint TF"
            )),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[
                    _sim_ekf_config(),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "odom0": odom_topic,
                        "imu0": imu_topic,
                        "publish_tf": True,
                    },
                ],
                remappings=[("odometry/filtered", filtered_topic)],
            ),
        ]

    backend = resolve_rover_backend(
        LaunchConfiguration("rover_hardware_protocol").perform(context),
        LaunchConfiguration("can_interface").perform(context),
    )
    if backend != "odrive":
        return [LogInfo(msg=(
            "[rover localization] mock backend — skipping Odom.py/EKF; "
            "mock_rover_drive owns /odom and the odom->base_footprint TF."
        ))]

    use_imu = LaunchConfiguration("use_imu").perform(context)
    imu_port = LaunchConfiguration("imu_port").perform(context)
    ybimu_port = LaunchConfiguration("ybimu_port").perform(context)
    imu_frame = LaunchConfiguration("imu_frame").perform(context)
    bno055_topic = LaunchConfiguration("bno055_topic").perform(context)
    ybimu_topic = LaunchConfiguration("ybimu_topic").perform(context)

    imu_source, ybimu_available, bno_available = resolve_imu_source(
        use_imu,
        imu_port,
        ybimu_port,
    )

    ekf_overrides = {"base_link_frame": "base_footprint"}

    if imu_source == "ybimu":
        ekf_overrides["imu0"] = ybimu_topic
        ekf_config = _ekf_config("ekf_config.yaml")
    elif imu_source == "bno055":
        ekf_overrides["imu0"] = bno055_topic
        ekf_config = _ekf_config("ekf_config.yaml")
    else:
        ekf_overrides["odom0_config"] = ODOM_ONLY_CONFIG
        ekf_config = _ekf_config("ekf_odom_only.yaml")

    return [
        LogInfo(msg=(
            "[rover localization] "
            f"use_imu={use_imu} selected={imu_source} "
            f"ybimu_available={ybimu_available} "
            f"bno_available={bno_available} imu_frame={imu_frame}"
        )),
        Node(
            package="rover_nav",
            executable="Odom.py",
            name="odom_node",
            output="screen",
        ),
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            parameters=[ekf_config, ekf_overrides],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_ekf", default_value="false",
                              description="Fuse simulation /odom and /imu into filtered odometry."),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("sim_odom_topic", default_value="/ground_truth/odom"),
        DeclareLaunchArgument("sim_imu_topic", default_value="/imu"),
        DeclareLaunchArgument("filtered_odom_topic", default_value="/odometry/filtered"),

        DeclareLaunchArgument("rover_hardware_protocol", default_value="auto",
                              choices=["auto", "odrive", "mock_hardware"]),
        DeclareLaunchArgument("can_interface", default_value="can0"),

        DeclareLaunchArgument("use_imu", default_value="auto",
                              choices=["auto", "true", "false", "ybimu", "bno055"]),
        DeclareLaunchArgument("imu_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("ybimu_port", default_value="/dev/imu_ybimu"),
        DeclareLaunchArgument("imu_frame", default_value="bno055"),
        DeclareLaunchArgument("bno055_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("ybimu_topic", default_value="/ybimu/imu"),

        OpaqueFunction(function=_start_localization),
    ])
