#!/usr/bin/env python3
"""Operator side of the camera downlink: decompress once, locally.

Run this wherever RViz runs. Per camera it starts two republishers that pull the
compressed streams across the link and expand them into machine-local raw
topics:

    /downlink/<camera>/color/compressed      -> /<camera>/view/color  (rgb8)
    /downlink/<camera>/depth/compressedDepth -> /<camera>/view/depth  (16UC1)

Decompressing here rather than letting each display do it is not a detail, it is
the point. RViz's Image display in ROS 2 has no transport selection at all -- it
subscribes raw, full stop -- so pointing it at a rover topic puts uncompressed
frames on the link no matter what the DepthCloud beside it is set to. And two
processes each subscribing to the same compressed topic would pull two copies
across, because they are separate DDS participants. One republisher per stream,
everything downstream reading its local output, means exactly one compressed
copy crosses the antenna.

CameraInfo is not republished: it is a few hundred bytes, latched, and the
displays read /downlink/<camera>/camera_info directly.

Nothing here needs the rover's TF or robot description; it is purely image
plumbing. Point RViz at aries_moveit/launch/moveit.rviz, which is already wired
to these topic names.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _view_for(camera):
    # Everything the antenna carries lives under one top-level prefix.
    src = f'/downlink/{camera}'
    dst = f'/{camera}/view'
    return [
        Node(
            package='image_transport',
            executable='republish',
            name=f'{camera}_view_color_decompress',
            # Transports are parameters, not positional arguments: passing them
            # positionally leaves in_transport at 'raw' and this node would sit
            # waiting on an uncompressed topic that nothing publishes.
            # The subscriber QoS must match the rover's best_effort publisher.
            # A RELIABLE subscriber against a BEST_EFFORT publisher is an
            # incompatible pair: DDS makes no match at all, so the topic lists
            # fine and simply never delivers a frame.
            parameters=[{
                'in_transport': 'compressed',
                'out_transport': 'raw',
                f'qos_overrides.{src}/color/compressed.subscription.reliability':
                    'best_effort',
                f'qos_overrides.{src}/color/compressed.subscription.history':
                    'keep_last',
                f'qos_overrides.{src}/color/compressed.subscription.depth': 1,
            }],
            remappings=[('in/compressed', f'{src}/color/compressed'),
                        ('out', f'{dst}/color')],
            output='screen',
        ),
        Node(
            package='image_transport',
            executable='republish',
            name=f'{camera}_view_depth_decompress',
            parameters=[{
                'in_transport': 'compressedDepth',
                'out_transport': 'raw',
                f'qos_overrides.{src}/depth/compressedDepth.subscription.reliability':
                    'best_effort',
                f'qos_overrides.{src}/depth/compressedDepth.subscription.history':
                    'keep_last',
                f'qos_overrides.{src}/depth/compressedDepth.subscription.depth': 1,
            }],
            remappings=[('in/compressedDepth', f'{src}/depth/compressedDepth'),
                        ('out', f'{dst}/depth')],
            output='screen',
        ),
    ]


def launch_setup(context, *args, **kwargs):
    cameras = [c.strip().strip('/')
               for c in LaunchConfiguration('cameras').perform(context).split(',')
               if c.strip()]
    actions = []
    for camera in cameras:
        actions += _view_for(camera)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'cameras', default_value='gripper_camera,rover_camera',
            description='Comma-separated camera names, matching the rover side.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Also start RViz with the MoveIt config. Leave false when '
                        'RViz comes from another launch file.'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution(
                [FindPackageShare('aries_moveit'), 'launch', 'moveit.rviz']),
            description='RViz config to open. The shipped one is already pointed '
                        'at the /<camera>/view/* topics this file publishes.'),
        OpaqueFunction(function=launch_setup),
        Node(
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            output='screen',
        ),
    ])
