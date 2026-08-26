#!/usr/bin/env python3
"""The one command to run on the ROVER for field operation.

    ros2 launch aries_comms rover_field.launch.py

Everything aries_bringup's full_hardware.launch.py brings up, plus the
communication layer, in a configuration that assumes the operator is at the
base station and not next to the robot. Its partner is base_station.launch.py,
beside it in this package:

WHY THIS LIVES HERE AND NOT IN aries_bringup

    It is half of a two-machine contract, and the other half is
    base_station.launch.py. Which end reads the pad, which end runs RViz, which
    end decompresses -- each of those is one decision spread across two files,
    and while they sat in different packages nothing compared them. They are
    now next to each other with test/test_field_link_contract.py pinning the
    pairs that must not drift.

    aries_bringup still owns everything that is about the ROBOT rather than
    about the link: full_hardware.launch.py, the camera pipeline at both ends,
    the rover-side checker. The dependency runs one way, aries_comms ->
    aries_bringup, and must stay that way or colcon has a cycle.

    rover                                    base station
    -----                                    ------------
    rover_field.launch.py                    base_station.launch.py
      DDS on the field link                    DDS on the field link
      arm + gripper + MoveIt                   joy driver -> /joy
      rover drive + IMU                        decompressors -> /<cam>/view/*
      cameras -> /downlink/<cam>/*             RViz  (one, see that file)
      teleop consumers <- /joy                 base_station_checker
      full_hardware_checker

WHAT THIS CHANGES AGAINST full_hardware.launch.py

  DDS environment set here with require_link=True, not merely inherited.
      full_hardware sets the environment too, as of 2026-08-25 -- it used to
      set none, so launching it directly landed on whatever the calling
      terminal had, and a terminal opened before the exports existed keeps
      domain 0 with rmw_fastrtps_cpp for as long as it lives. The base station
      is on domain 30 with Cyclone, so the two never see each other: ping
      works, `ros2 topic list` is empty, and nothing logs an error.

      What is still different here is require_link. full_hardware passes False
      because it is also the bench entry point and a laptop with no antenna
      must still come up; it falls back to a loopback-only config. This file
      passes True, so a rover with the cable out fails the launch instead of
      quietly bringing up a stack the base station will never see. Setting it
      here as well is not redundant: it runs FIRST, and it is the stricter of
      the two. See aries_common/comms.py.

  use_joy_node:=false
      The pad is plugged into the base station. Every teleop consumer still
      runs here and reads /joy across the link; only the driver moves. Set it
      true if you are standing next to the rover with the pad in the rover's
      USB port, and false at the base station -- exactly one machine may
      publish /joy.

  use_gui:=false
      RViz belongs at the base station. Running it here costs the rover CPU
      and GPU it needs for two RealSense pipelines, fails outright over SSH
      with no display, and starts a second set of decompressors that nothing
      on the rover looks at.

      full_hardware_checker still runs here, and still should: it reads serial
      ports, the CAN link and USB enumeration, none of which exist at the other
      end. Its console output stays on the rover, so the operator gets
      base_station_checker instead -- link, pad, downlink, and what of the
      rover is actually arriving.

  enable_camera_downlink:=true
      Explicit rather than inherited, because with no GUI here the downlink is
      the ONLY way any image leaves the robot.

Every other argument full_hardware.launch.py takes still works and reaches it
unchanged -- finger_type, hardware protocols, camera serials, IMU, CAN:

    ros2 launch aries_comms rover_field.launch.py finger_type:=probe
    ros2 launch aries_comms rover_field.launch.py downlink_profile:=lean

Not because they are re-listed here, but because an included launch file shares
the launch configuration context: a value set on the command line is already
present when full_hardware's DeclareLaunchArgument runs, and a declaration does
not overwrite a configuration that is already set. The same mechanism is how
the arguments declared below override full_hardware's own defaults. Copying its
argument list into this file would have doubled every default and guaranteed
the two drift apart.

SAFETY

  The pad is now on the far side of a radio link, so a dropout looks exactly
  like a held stick. Every teleop node stops on 0.35 s of /joy silence; the two
  rover drive nodes gained that in the same change that added this file. Do not
  raise joy_timeout_sec to paper over a marginal link -- fix the link.

VERIFY, from the base station

    ros2 topic list | grep ^/downlink/
    ros2 run aries_bringup downlink_report.py
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from aries_common.comms import dds_launch_actions


def generate_launch_description():
    return LaunchDescription([
        # These three must precede every other action. Launch executes in
        # order, and anything above them would inherit the calling shell's
        # environment -- which is the whole failure being prevented.
        *dds_launch_actions(),

        DeclareLaunchArgument(
            "use_joy_node", default_value="false",
            description="Read the pad on the ROVER. false is the field default: "
                        "the operator holds it at the base station and the "
                        "consumers here take /joy over the link.",
        ),
        DeclareLaunchArgument(
            "use_gui", default_value="false",
            description="RViz on the rover. false is the field default; RViz "
                        "belongs at the base station.",
        ),
        DeclareLaunchArgument(
            "use_joystick", default_value="true",
            description="Run the teleop consumers here. Almost never false: "
                        "this is what actually moves the robot.",
        ),
        DeclareLaunchArgument(
            "enable_camera_downlink", default_value="true",
            choices=["true", "false"],
            description="With use_gui false this is the only way an image "
                        "leaves the rover.",
        ),
        DeclareLaunchArgument(
            "downlink_profile", default_value="balanced",
            choices=["quality", "balanced", "lean"],
            description="quality 42.3 Mbit/s, balanced 28.3, lean 10.9 -- both "
                        "cameras, 15 Hz colour / 5 Hz depth.",
        ),

        LogInfo(msg="[rover_field] rover side up: pad and RViz expected at the "
                    "base station (ros2 launch aries_comms "
                    "base_station.launch.py)"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "full_hardware.launch.py",
                ])
            ),
            # Only the two the shared context alone would make easy to miss.
            # Everything else -- including the arguments declared above --
            # reaches full_hardware through the shared configuration context;
            # see the module docstring. Forwarding these explicitly is
            # belt-and-braces and, more usefully, documentation: they are the
            # two settings that define what "field" means here.
            launch_arguments={
                "use_gui": LaunchConfiguration("use_gui"),
                "use_joy_node": LaunchConfiguration("use_joy_node"),
            }.items(),
        ),
    ])
