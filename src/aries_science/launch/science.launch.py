#!/usr/bin/env python3
"""The science module's host half.

The BOARD is a second Teensy with its own micro-ROS agent, started by
aries_hardware.launch.py alongside the drill's -- see `science:` in
aries_common/config/devices.yaml. This launch starts only the node that turns
the board's telemetry array into named topics.

So starting this without the agent is not an error and not useless: the node
comes up, publishes `no telemetry yet` on /science/status once a second, and
starts republishing the moment the board appears.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("aries_science"), "config", "science.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "science_config", default_value=config,
            description="Field table: the telemetry array's order, and the "
                        "name and unit each index carries."),

        Node(
            package="aries_science",
            executable="science_telemetry.py",
            name="science_telemetry",
            output="screen",
            parameters=[LaunchConfiguration("science_config")],
        ),
    ])
