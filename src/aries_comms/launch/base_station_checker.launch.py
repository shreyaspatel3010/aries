#!/usr/bin/env python3
"""The operator-side status check, on its own.

    ros2 launch aries_comms base_station_checker.launch.py

base_station.launch.py starts this already (start_checker:=true). Run it alone
against a base station that is already up -- to watch the link during a run
without restarting anything, or from a third machine that is only spectating.

It is the counterpart of aries_bringup/launch/full_hardware_checker.launch.py,
which stays on the rover: that one probes serial ports, CAN and USB, none of
which exist at this end, and prints to the robot's console.

STANDALONE RUNS NEED THE ENVIRONMENT
    Launched on its own this file sets the DDS environment like every other
    entry point, so it lands on the rover's domain even from a stale terminal.
    That is also what makes its own domain/RMW report meaningful: the node
    reads the environment it was STARTED with, which is the same thing every
    other node in the launch got.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from aries_common.comms import dds_launch_actions
from aries_common.devices import device_str


def _text(argument):
    """A launch argument forced to STRING.

    Same guard as full_hardware_checker.launch.py. Launch infers a parameter's
    type from the text, so a camera list that happened to be one name would
    still be a string but "true" would arrive as a BOOL -- and the node dies at
    startup with InvalidParameterTypeException instead of reporting anything.
    """
    return ParameterValue(LaunchConfiguration(argument), value_type=str)


def generate_launch_description():
    return LaunchDescription([
        # Only has an effect when this file is the top-level launch: included
        # from base_station.launch.py the variables are already set, and
        # SetEnvironmentVariable to the same value is a no-op.
        *dds_launch_actions(),

        DeclareLaunchArgument("checker_interval", default_value="4.0"),
        DeclareLaunchArgument(
            "timeout", default_value="5.0",
            description="How old a message may be and still count as arriving. "
                        "Also the window the rates are averaged over.",
        ),
        DeclareLaunchArgument(
            "cameras", default_value="gripper_camera,rover_camera,rear_camera",
            description="Must match what base_station.launch.py decompresses; a "
                        "camera listed here and not decompressed reads as a fault.",
        ),
        DeclareLaunchArgument(
            "color_only", default_value="rear_camera",
            description="Of those, the ones with no depth stream.",
        ),
        DeclareLaunchArgument(
            "joy_dev", default_value=device_str("joystick.device")),
        DeclareLaunchArgument(
            "expect_local_joy", default_value="true",
            description="True when the pad is plugged in HERE, which is the "
                        "field default. False makes the absence of a local joy "
                        "driver expected instead of a fault.",
        ),
        DeclareLaunchArgument(
            "check_link", default_value="true",
            description="Address, domain, RMW, Cyclone config and ICMP to the "
                        "other host and the radios.",
        ),
        DeclareLaunchArgument("check_downlink", default_value="true"),
        DeclareLaunchArgument("check_rover", default_value="true"),
        DeclareLaunchArgument(
            "print_only_on_change", default_value="true",
            description="Rates are bucketed before the comparison, so a healthy "
                        "link prints once and then stays quiet.",
        ),

        Node(
            package="aries_comms",
            executable="base_station_checker.py",
            name="base_station_checker",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "check_interval": LaunchConfiguration("checker_interval"),
                "timeout": LaunchConfiguration("timeout"),
                # Comma-separated; the node splits them. Forced to STRING so a
                # single-camera value cannot be inferred as something else.
                "cameras": _text("cameras"),
                "color_only": _text("color_only"),
                "joystick_device": _text("joy_dev"),
                "expect_local_joy": LaunchConfiguration("expect_local_joy"),
                "check_link": LaunchConfiguration("check_link"),
                "check_downlink": LaunchConfiguration("check_downlink"),
                "check_rover": LaunchConfiguration("check_rover"),
                "print_only_on_change": LaunchConfiguration("print_only_on_change"),
            }],
        ),
    ])
