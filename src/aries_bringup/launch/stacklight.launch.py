#!/usr/bin/env python3
"""The mast's three-colour stack light.

    RED     emergency stop pressed (either switch), or halted
    YELLOW  operating
    GREEN   ready, doing nothing

The light hangs off the gripper Teensy, whose firmware already subscribes to
std_msgs/UInt8 on /stacklight_subscription; this node is the publisher that was
missing, so the light stayed dark whatever the rover did. The Teensy is reached
through the micro-ROS agent aries_hardware.launch.py starts -- this node only
publishes the topic, so it does not care which machine the agent runs on.

The state -> colour table, and the topics each state is measured from, live in
config/stacklight.yaml. `emergency` and `halt` have no source yet: both e-stops
physically disconnect ODrive power, so there is nothing for software to read
until a sense line exists. See the notes in that file.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "stacklight_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("aries_bringup"), "config", "stacklight.yaml",
            ]),
            description="State -> colour table and the sources behind it.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false",
            description="Sim clock. The refresh and staleness windows are in "
                        "seconds, so this has to match the rest of the stack.",
        ),
        Node(
            package="aries_bringup",
            executable="stacklight.py",
            name="stacklight",
            output="screen",
            parameters=[
                LaunchConfiguration("stacklight_config"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
