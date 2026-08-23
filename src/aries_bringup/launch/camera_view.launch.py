#!/usr/bin/env python3
"""Operator side of the camera downlink: decompress once, locally.

Run this wherever RViz runs. Per camera it starts two republishers that pull the
compressed streams across the link and expand them into machine-local raw
topics:

    /downlink/<camera>/color/compressed      -> /<camera>/view/color  (rgb8)
    /downlink/<camera>/depth/compressedDepth -> /<camera>/view/depth  (16UC1)

A camera named in `color_only` gets the colour one only: it has no depth sensor,
so there is no second stream to expand. That list has to match the rover side's
argument of the same name in camera_downlink.launch.py.

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

NO VIEWER IS STARTED HERE, and that is deliberate.

This file used to carry a `use_rviz` argument that started an rviz2 of its own.
Every caller already runs a viewer, so nobody set it -- but an argument is not
private to the file that declares it. An included launch description inherits
the parent's launch configurations, and DeclareLaunchArgument does not overwrite
a configuration that is already set. So a parent that declared its own
`use_rviz` (base_station.launch.py, default true) silently switched this one on
too, and got a SECOND RViz -- opened with the parent's `rviz_config`, which for
the base station was the empty string meaning "use my default", so the extra
window came up blank and unconfigured.

The rule this leaves behind: a launch file that is included by others does not
declare a common name for a node it is not the owner of. Whoever wants a viewer
starts it themselves.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _view_for(camera, color_only=False):
    """The decompressors for one camera: two, or one without depth.

    color_only skips the depth decompressor. Left running for a camera that has
    no depth sensor it is not merely idle -- it is a node subscribed to a topic
    nothing will ever publish, which is indistinguishable at a glance from the
    depth stream having failed. The rear camera is the colour-only one: a UVC
    webcam under the tail aimed at the drill.
    """
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
    ] + ([] if color_only else [
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
    ])


def launch_setup(context, *args, **kwargs):
    cameras = [c.strip().strip('/')
               for c in LaunchConfiguration('cameras').perform(context).split(',')
               if c.strip()]
    color_only = {c.strip().strip('/')
                  for c in LaunchConfiguration('color_only').perform(context).split(',')
                  if c.strip()}
    actions = []
    for camera in cameras:
        actions += _view_for(camera, color_only=camera in color_only)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'cameras', default_value='gripper_camera,rover_camera',
            description='Comma-separated camera names, matching the rover side.'),
        DeclareLaunchArgument(
            'color_only', default_value='rear_camera',
            description='Of those, the ones with no depth stream. Must match '
                        'the rover side\'s argument of the same name. Naming a '
                        'camera that is not in `cameras` is harmless.'),
        # No use_rviz here on purpose -- see the NO VIEWER note in the module
        # docstring. This file is image plumbing; the viewer belongs to whoever
        # included it.
        OpaqueFunction(function=launch_setup),
    ])
