#!/usr/bin/env python3
"""
Joystick input: the joy driver, the layout normalizer, and the rover controller.

The normalizer maps whichever driver/pad combination is connected back onto the
Xbox-style layout the controller expects, so joy/raw is always the driver's
output and /joy is always the normalized one.

`use_joystick_controller` is separate from `use_joy_node` because the mock drive
backend consumes /joy itself and must not also run the real rover controller.

Hold LB to enable rover drive output. The default driver is
`game_controller_node`; for the older raw Linux driver, launch with
`joy_driver:=joy_node joy_layout:=bluetooth`.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    joystick_config = os.path.join(
        get_package_share_directory("aries_teleop"),
        "config",
        "joystick.yaml",
    )

    use_joy_node = LaunchConfiguration("use_joy_node")
    joy_dev = LaunchConfiguration("joy_dev")

    return LaunchDescription([
        DeclareLaunchArgument("use_joy_node", default_value="true",
                              description="Start the joy driver and layout normalizer."),
        DeclareLaunchArgument("use_joystick_controller", default_value="true",
                              description="Start the rover joystick controller that drives the ODrives."),
        DeclareLaunchArgument("use_drill_teleop", default_value="true",
                              description="Start the LT-gated drill teleop (feed, sample bin, auger, sand box lid, pump)."),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node",
                              choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto",
                              choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),

        Node(
            condition=IfCondition(use_joy_node),
            package="joy",
            executable=LaunchConfiguration("joy_driver"),
            name="joy_node",
            parameters=[{"dev": joy_dev}],
            remappings=[("joy", "joy/raw")],
            output="screen",
        ),
        Node(
            condition=IfCondition(use_joy_node),
            package="aries_teleop",
            executable="joy_layout_normalizer.py",
            name="joy_layout_normalizer",
            parameters=[{
                "input_topic": "joy/raw",
                "output_topic": "joy",
                "layout": LaunchConfiguration("joy_layout"),
                "device": joy_dev,
            }],
            output="screen",
        ),
        Node(
            condition=IfCondition(LaunchConfiguration("use_joystick_controller")),
            package="aries_teleop",
            executable="custom_joystick_controller.py",
            name="rover_joystick_controller",
            parameters=[joystick_config],
            output="screen",
        ),
        # Gated behind LT, so it shares the pad with rover drive (LB) and the
        # arm (RB/RT) without any of them contending. It only publishes while
        # LT is held, so leaving it running costs nothing.
        Node(
            condition=IfCondition(LaunchConfiguration("use_drill_teleop")),
            package="aries_teleop",
            executable="drill_joystick.py",
            name="drill_joystick",
            parameters=[joystick_config],
            output="screen",
        ),
    ])
