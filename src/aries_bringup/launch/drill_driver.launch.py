#!/usr/bin/env python3
"""The drill driver: rate commands in, Teensy duty cycle out.

drill_joystick.py has published the drill's three axes in physical units since
it was written, and until now nothing consumed them -- its own docstring said
so: "Nothing consumes these topics on the real rover yet - there is no drill
driver." This is that driver.

    /aries/drill_bit_joint/cmd_vel        rad/s  ->  motor1/cmd_speed
    /aries/drill_motor_joint/cmd_vel      m/s    ->  motor2/cmd_speed
    /aries/drill_container_joint/cmd_vel  m/s    ->  linact/cext

The outputs are the drill Teensy's contract (firmware/teensy_drill_sys), reached
through the micro-ROS agent that aries_hardware.launch.py starts. This node only
publishes topics, so it does not care which machine the agent runs on.

The calibration -- what rate each axis reaches at full duty cycle -- lives in
config/drill_driver.yaml, and NONE OF IT IS MEASURED YET. The drill moves; the
numbers are not yet true. See the calibration recipe in that file.

IN SIMULATION THIS NODE IS NOT WANTED. gz consumes the rate topics directly
through aries/config/*_gazebo_bridge.yaml, in the physical units they are
already in. Running this as well would additionally publish PWM to a board that
is not there -- harmless, but it means the sim and the rover disagree about
which topic is the real command. Left out of the sim launch on purpose.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "drill_driver_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("aries_bringup"), "config", "drill_driver.yaml",
            ]),
            description="Rate -> duty cycle calibration and the topic map.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false",
            description="Sim clock. The staleness windows are in seconds, so "
                        "this has to match the rest of the stack.",
        ),
        Node(
            package="aries_bringup",
            executable="drill_driver.py",
            name="drill_driver",
            output="screen",
            parameters=[
                LaunchConfiguration("drill_driver_config"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
