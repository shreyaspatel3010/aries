#!/usr/bin/env python3

import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from aries_common.detect import as_bool, resolve_imu_source, resolve_lidar_enabled

ODOM_ONLY_CONFIG = [
    False, False, False,  # x, y, z
    False, False, False,  # roll, pitch, yaw
    True, False, False,   # vx, vy, vz
    False, False, True,   # vroll, vpitch, vyaw
    False, False, False,  # accelerations
]


def _start_rover_hardware(context, *args, **kwargs):
    can_interface = LaunchConfiguration("can_interface").perform(context)
    can_bitrate = LaunchConfiguration("can_bitrate").perform(context)
    setup_can = as_bool(LaunchConfiguration("setup_can").perform(context))

    use_joystick = LaunchConfiguration("use_joystick")
    use_joy_node = LaunchConfiguration("use_joy_node")
    joy_driver = LaunchConfiguration("joy_driver")
    joy_layout = LaunchConfiguration("joy_layout")
    joy_dev = LaunchConfiguration("joy_dev")

    joystick_config = os.path.join(
        get_package_share_directory("aries_bringup"),
        "config",
        "joystick.yaml",
    )

    odrive_nodes = [
        Node(
            package="odrive_can",
            executable="odrive_can_node",
            name="can_node",
            namespace=f"odrive_axis{i}",
            parameters=[{
                "node_id": i,
                "interface": can_interface,
            }],
            output="screen",
        )
        for i in range(6)
    ]

    odrive_poller = Node(
        package="aries_bringup",
        executable="odrive_can_poller.py",
        name="odrive_can_poller",
        parameters=[{
            "interface": can_interface,
            "num_axes": 6,
            "poll_rate_hz": 5.0,
        }],
        output="screen",
    )

    joy_node = Node(
        condition=IfCondition(use_joy_node),
        package="joy",
        executable=joy_driver,
        name="joy_node",
        parameters=[{"dev": joy_dev}],
        remappings=[("joy", "joy/raw")],
        output="screen",
    )

    joy_layout_normalizer_node = Node(
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

    rover_controller = Node(
        condition=IfCondition(use_joystick),
        package="aries_bringup",
        executable="custom_joystick_controller.py",
        name="rover_joystick_controller",
        parameters=[joystick_config],
        output="screen",
    )

    start_nodes = [
        *odrive_nodes,
        TimerAction(
            period=2.0,
            actions=[
                odrive_poller,
                joy_node,
                joy_layout_normalizer_node,
                rover_controller,
            ],
        ),
    ]

    if not setup_can:
        return start_nodes

    quoted_interface = shlex.quote(can_interface)
    quoted_bitrate = shlex.quote(can_bitrate)
    can_setup = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            f"sudo -n ip link set {quoted_interface} down 2>/dev/null; "
            f"sudo -n ip link set {quoted_interface} up type can bitrate {quoted_bitrate}",
        ],
        output="screen",
    )

    return [
        can_setup,
        RegisterEventHandler(
            OnProcessExit(
                target_action=can_setup,
                on_exit=start_nodes,
            )
        ),
    ]


def _start_localization(context, *args, **kwargs):
    use_imu_value = LaunchConfiguration("use_imu").perform(context)
    imu_port = LaunchConfiguration("imu_port").perform(context)
    imu_topic = LaunchConfiguration("imu_topic").perform(context)
    picoscan_imu_topic = LaunchConfiguration("picoscan_imu_topic").perform(context)
    lidar_available = resolve_lidar_enabled(
        LaunchConfiguration("use_lidar").perform(context),
        LaunchConfiguration("lidar_sensor_ip").perform(context),
    )

    imu_source, bno_available = resolve_imu_source(
        use_imu_value, imu_port, lidar_available
    )
    ekf_config_imu = os.path.join(get_package_share_directory("rover_nav"), "config", "ekf_config.yaml")
    ekf_config_picoscan = os.path.join(
        get_package_share_directory("rover_nav"), "config", "ekf_picoscan_imu.yaml"
    )
    ekf_config_no_imu = os.path.join(get_package_share_directory("rover_nav"), "config", "ekf_odom_only.yaml")

    # Keep EKF YAMLs generic; these runtime overrides match the Aries URDF/SRDF tree.
    ekf_overrides = {
        "base_link_frame": "base_footprint",
    }

    actions = [
        LogInfo(
            msg=(
                "[rover localization] "
                f"use_imu={use_imu_value} selected={imu_source} "
                f"bno_available={bno_available} lidar_available={lidar_available}"
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_imu"), "launch", "imu.launch.py"
                ])
            ),
            launch_arguments={
                "use_imu": LaunchConfiguration("use_imu"),
                "imu_port": LaunchConfiguration("imu_port"),
                "imu_baudrate": LaunchConfiguration("imu_baudrate"),
                "imu_frame": LaunchConfiguration("imu_frame"),
                "use_lidar": LaunchConfiguration("use_lidar"),
                "lidar_sensor_ip": LaunchConfiguration("lidar_sensor_ip"),
                "picoscan_raw_imu_topic": LaunchConfiguration("picoscan_raw_imu_topic"),
                "picoscan_imu_topic": LaunchConfiguration("picoscan_imu_topic"),
            }.items(),
        ),
        Node(
            package="rover_nav",
            executable="Odom.py",
            name="odom_node",
            output="screen",
        ),
    ]

    if imu_source == "bno055":
        ekf_overrides["imu0"] = imu_topic
        actions.append(
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[ekf_config_imu, ekf_overrides],
            )
        )
        return actions

    if imu_source == "picoscan":
        ekf_overrides["imu0"] = picoscan_imu_topic
        actions.append(
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[ekf_config_picoscan, ekf_overrides],
            )
        )
        return actions

    ekf_overrides["odom0_config"] = ODOM_ONLY_CONFIG
    actions.append(
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            parameters=[ekf_config_no_imu, ekf_overrides],
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
        DeclareLaunchArgument("imu_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("use_lidar", default_value="auto", choices=["auto", "true", "false"]),
        DeclareLaunchArgument("lidar_sensor_ip", default_value="169.254.136.69"),
        DeclareLaunchArgument("picoscan_raw_imu_topic", default_value="/picoscan/imu_raw"),
        DeclareLaunchArgument("picoscan_imu_topic", default_value="/picoscan/imu"),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("use_joy_node", default_value="false"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("can_bitrate", default_value="250000"),
        DeclareLaunchArgument(
            "setup_can",
            default_value="true",
            description="Bring the CAN interface up with sudo -n before starting ODrive nodes.",
        ),

        OpaqueFunction(function=_start_localization),
        OpaqueFunction(function=_start_rover_hardware),
    ])
