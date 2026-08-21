#!/usr/bin/env python3
"""ARIES communication layer, on a domain that does not depend on your terminal.

    ros2 launch ~/aries/communication/communication.launch.py side:=operator

The DDS settings are applied here, as launch actions, rather than being left to
whatever the calling shell happens to export. That matters because a terminal
opened before ~/.bashrc gained the exports keeps the old domain for as long as
it lives, and every node launched from it inherits that -- which looks exactly
like a broken link: ping fine, `ros2 topic list` empty, no error anywhere.

Setting them here makes the domain a property of the launch instead. The nodes
below are correct no matter which terminal starts them.

WHAT THIS DOES NOT FIX
    Your interactive shell. SetEnvironmentVariable only reaches processes this
    launch spawns; `ros2 topic list`, rqt_image_view and RViz started by hand
    are not among them. Source comms_env.sh in those terminals.

    Nodes launched elsewhere. full_hardware.launch.py publishes the camera
    topics this layer carries, and it must be on the same domain or there is
    nothing here to compress. Start it from a comms_env.sh shell too.

    side:=rover      compress on the robot   (camera_downlink.launch.py)
    side:=operator   decompress here         (camera_view.launch.py)
    side:=both       one machine, end to end (mostly for testing)
    stack:=true      also start full_hardware.launch.py on this domain.
                     full_hardware then provides the camera chain itself, so
                     side is ignored: this layer adds nothing on top.
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

# SUPERSEDED -- see communication/README.txt. Kept for reference only.
#
# Resolved from this file rather than hardcoded, so moving the folder or
# cloning the repo to another path does not silently leave CYCLONEDDS_URI
# pointing at a file that is not there. NOTE: the claim that used to be here --
# that Cyclone merely warns on a missing URI and falls back to defaults -- is
# WRONG. Measured 2026-08-21: it refuses to create the domain and every node
# dies at startup. aries_common/comms.py has the correct account.
_HERE = os.path.dirname(os.path.realpath(__file__))
CYCLONEDDS_XML = os.path.join(_HERE, 'cyclonedds.xml')

ROS_DOMAIN_ID = '30'
RMW = 'rmw_cyclonedds_cpp'


def _bringup(context, *args, **kwargs):
    side = LaunchConfiguration('side').perform(context).lower()
    cameras = LaunchConfiguration('cameras')
    use_rviz = LaunchConfiguration('use_rviz')

    if side not in ('rover', 'operator', 'both'):
        raise RuntimeError(
            f"side must be rover, operator or both -- got '{side}'")

    bringup = FindPackageShare('aries_bringup')
    actions = []

    stack = LaunchConfiguration('stack').perform(context).lower() == 'true'
    want_rover = side in ('rover', 'both')
    want_operator = side in ('operator', 'both')

    # The robot stack publishes the camera topics this layer carries. Bringing
    # it up from here is the only way to guarantee it lands on the same domain:
    # launched by hand from an old terminal it silently keeps that terminal's
    # domain, and this layer then has nothing to compress.
    if stack:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                bringup, '/launch/full_hardware.launch.py']),
            launch_arguments={'enable_camera_downlink': 'true'}.items(),
        ))
        # full_hardware owns the WHOLE camera chain, not just half of it: it
        # includes camera_downlink.launch.py, and camera_view.launch.py beside
        # it whenever use_gui is true (which is the default). Adding either
        # side again here puts two publishers on one topic. That does not fail
        # loudly -- the topic just runs at roughly double rate with two writers
        # interleaving frames of different ages, which reads as jitter and is
        # easy to mistake for a link problem.
        #
        # So with stack:=true this layer contributes nothing itself, and side
        # is only about what full_hardware is asked to bring up.
        want_rover = False
        want_operator = False

    if want_rover:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                bringup, '/launch/camera_downlink.launch.py']),
            launch_arguments={
                'cameras': cameras,
                'downlink_profile': LaunchConfiguration('downlink_profile'),
            }.items(),
        ))

    if want_operator:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                bringup, '/launch/camera_view.launch.py']),
            launch_arguments={
                'cameras': cameras,
                'use_rviz': use_rviz,
            }.items(),
        ))

    return actions


def generate_launch_description():
    if not os.path.exists(CYCLONEDDS_XML):
        raise RuntimeError(f'cyclonedds.xml not found at {CYCLONEDDS_XML}')

    return LaunchDescription([
        # These three must precede every node action below; launch executes
        # actions in order, and a Node that starts first would inherit the
        # shell's environment instead of this one.
        SetEnvironmentVariable('ROS_DOMAIN_ID', ROS_DOMAIN_ID),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', RMW),
        SetEnvironmentVariable('CYCLONEDDS_URI', f'file://{CYCLONEDDS_XML}'),

        LogInfo(msg=f'[comms] domain {ROS_DOMAIN_ID}, {RMW}'),
        LogInfo(msg=f'[comms] {CYCLONEDDS_XML}'),
        LogInfo(msg='[comms] shells need: source '
                    f'{os.path.join(_HERE, "comms_env.sh")}'),

        DeclareLaunchArgument(
            'side', default_value='operator',
            choices=['rover', 'operator', 'both'],
            description='rover compresses, operator decompresses.'),
        DeclareLaunchArgument(
            'cameras', default_value='gripper_camera,rover_camera',
            description='Comma-separated cameras to carry.'),
        DeclareLaunchArgument(
            'downlink_profile', default_value='balanced',
            description='Rover-side bandwidth profile.'),
        DeclareLaunchArgument(
            'stack', default_value='false', choices=['true', 'false'],
            description='Also bring up full_hardware.launch.py on this domain.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Operator side: start RViz on the view topics.'),

        OpaqueFunction(function=_bringup),
    ])
