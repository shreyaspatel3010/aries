#!/usr/bin/env python3
"""Portable operator view. Copy this folder anywhere and run it -- no build.

    ros2 launch <this folder>/operator.launch.py

Deliberately depends on nothing from the aries workspace. Every node it starts
comes from a stock ROS 2 install (image_transport, rviz2), the RViz config sits
beside this file, and the DDS transport config is generated at launch for the
machine it finds itself on. That is what makes it copyable: there is no package
to build, no install space to source, and no path to fix up afterwards.

    cameras:=gripper_camera,rover_camera   which streams to pull
    use_rviz:=false                        decompress only, bring your own viewer
    rviz_config:=/path/to.rviz             use a different layout

On the target machine you need ROS 2 Jazzy plus:

    sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp \\
                        ros-jazzy-image-transport-plugins ros-jazzy-rviz2

WHAT THIS SHOWS, AND WHAT IT DOES NOT
    Camera images and depth clouds, which is what the link carries. It does not
    show the robot model: RViz loads meshes from the local filesystem via
    package:// paths, and those live in the aries packages. /robot_description
    and /tf do cross the link, so a machine that has the aries packages
    installed can display the arm; a bare laptop cannot, and would just log
    resource errors. Rather than half-work, this config leaves the model out.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, LogInfo, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)

import dds_config  # noqa: E402  (needs _HERE on the path first)

# Deliberately NOT os.environ['ROS_DOMAIN_ID']. That variable is the stale
# value this launch exists to override: a terminal opened before the operator
# environment was set still carries the old domain, and reading it here would
# faithfully copy the very setting that breaks discovery. Override with
# ARIES_DOMAIN_ID, or the domain: launch argument, both of which are explicit.
DEFAULT_DOMAIN_ID = os.environ.get('ARIES_DOMAIN_ID', '30')


def _decompressors(camera):
    """Pull one camera's compressed streams and expand them locally.

    Decompressing here rather than in each display is the whole point of the
    exercise. RViz's Image display has no transport selection -- it subscribes
    raw -- so pointing it at a rover topic would drag uncompressed frames over
    the antenna. And two displays subscribing to the same compressed topic
    would pull two copies, because each process is its own DDS participant.
    One republisher per stream, everything downstream reading its local output.
    """
    src = f'/downlink/{camera}'
    dst = f'/{camera}/view'
    common = {
        # Must match the rover's best_effort publisher. A RELIABLE subscriber
        # against a BEST_EFFORT publisher is an incompatible QoS pair: DDS
        # makes no match, so the topic lists fine and never delivers a frame.
        'reliability': 'best_effort',
        'history': 'keep_last',
        'depth': 1,
    }

    def qos(topic, kind):
        return {f'qos_overrides.{topic}.subscription.{k}': v
                for k, v in common.items()} if kind else {}

    return [
        Node(
            package='image_transport', executable='republish',
            name=f'{camera}_view_color_decompress', output='screen',
            # Transports are parameters, not positional arguments: passed
            # positionally, out_transport is silently ignored and this node
            # sits waiting on a topic nothing publishes.
            parameters=[dict({'in_transport': 'compressed',
                              'out_transport': 'raw'},
                             **qos(f'{src}/color/compressed', True))],
            remappings=[('in/compressed', f'{src}/color/compressed'),
                        ('out', f'{dst}/color')],
        ),
        Node(
            package='image_transport', executable='republish',
            name=f'{camera}_view_depth_decompress', output='screen',
            parameters=[dict({'in_transport': 'compressedDepth',
                              'out_transport': 'raw'},
                             **qos(f'{src}/depth/compressedDepth', True))],
            remappings=[('in/compressedDepth', f'{src}/depth/compressedDepth'),
                        ('out', f'{dst}/depth')],
        ),
    ]


def _setup(context, *args, **kwargs):
    cameras = [c.strip().strip('/') for c
               in LaunchConfiguration('cameras').perform(context).split(',')
               if c.strip()]
    actions = []
    for camera in cameras:
        actions += _decompressors(camera)

    actions.append(Node(
        package='rviz2', executable='rviz2', name='aries_operator_rviz',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config').perform(context)],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    ))
    return actions


def generate_launch_description():
    xml_path, local = dds_config.write_config()
    domain = DEFAULT_DOMAIN_ID

    return LaunchDescription([
        # Applied here rather than left to the shell, because a terminal opened
        # before these were exported keeps its old domain for as long as it
        # lives, and every node launched from it inherits that. The symptom is
        # indistinguishable from a dead link: ping fine, topic list empty.
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
        SetEnvironmentVariable('CYCLONEDDS_URI', f'file://{xml_path}'),

        LogInfo(msg=f'[operator] domain {domain}, cyclonedds, '
                    f'interface {local}'),
        LogInfo(msg=f'[operator] transport config: {xml_path}'),
        LogInfo(msg='[operator] other shells: source '
                    f'{os.path.join(_HERE, "operator_env.sh")}'),

        DeclareLaunchArgument('cameras',
                              default_value='gripper_camera,rover_camera'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              choices=['true', 'false']),
        DeclareLaunchArgument('rviz_config',
                              default_value=os.path.join(_HERE, 'operator.rviz')),

        OpaqueFunction(function=_setup),
    ])
