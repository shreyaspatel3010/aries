#!/usr/bin/env python3
"""Cameras and the downlink, and nothing else.

    ros2 launch aries_bringup cameras.launch.py

For when the point is to look through the rover, not drive it: both RealSense
drivers, the compression chain, and enough TF for a depth cloud to have
somewhere to sit. No arm, no controllers, no MoveIt, no joystick. On this
machine that is the difference between ~30 nodes and ~8, and it leaves the arm
hardware alone.

The operator side is unchanged and stays where it was: communication/
aries_operator, which needs nothing from this workspace.

    cameras:=rover_camera        just one of them
    downlink_profile:=lean       half resolution, for a weak link
    tf:=driver                   see below
    use_rviz:=true               local viewer, for checking the rover end

TF, WHICH IS THE PART THAT BITES
    An Image display needs no TF at all. A DepthCloud does: it has to place the
    points somewhere, so the camera's optical frame must connect to whatever
    RViz is using as its fixed frame.

    tf:=robot (default) runs robot_state_publisher over the rover URDF, so both
    cameras land in one connected tree at their real positions relative to
    base_link. The arm is not running here, so joint_state_publisher supplies
    zeros: every fixed joint is exact, and the gripper camera -- which hangs
    off the arm -- sits at the arm's NOMINAL pose, not wherever it actually is.
    For the body-mounted front camera that distinction does not exist. For the
    wrist camera it does, so do not measure anything off that cloud unless the
    arm stack is up and publishing real joint states, in which case
    robot_state_publisher picks those up instead of the zeros.

    tf:=driver asks each camera to publish its own little tree instead. Those
    trees are internally correct but mutually disconnected, and RViz has only
    one fixed frame, so exactly one camera's cloud will render. Use it when the
    URDF is not available.

    tf:=none publishes nothing. Images work, depth clouds do not.
"""

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

from aries_common.devices import device_str


def _hardware_module():
    """Borrow the device-pinning logic from aries_hardware.launch.py.

    Deliberately imported rather than copied. Which serial belongs to which end
    of the robot, and the fact that the D435i's sysfs serial is a DIFFERENT
    number from the one the driver matches on, is exactly the kind of detail
    that goes stale in a second copy. That file has no module-level side
    effects, so importing it costs nothing.
    """
    path = os.path.join(
        FindPackageShare('aries_bringup').find('aries_bringup'),
        'launch', 'aries_hardware.launch.py')
    spec = importlib.util.spec_from_file_location('aries_hardware_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(context, *args, **kwargs):
    hw = _hardware_module()

    cameras = [c.strip().strip('/') for c
               in LaunchConfiguration('cameras').perform(context).split(',')
               if c.strip()]
    tf_mode = LaunchConfiguration('tf').perform(context).lower()
    driver_tf = 'true' if tf_mode == 'driver' else 'false'

    # Two different questions, and conflating them is a trap the hardware
    # launch documents at length: identified is "librealsense enumerated the
    # cameras and these are their serials", detected may instead be a sysfs
    # COUNT, where the entries are empty strings because the serial visible
    # there is the ASIC serial and not the one the driver matches on.
    #
    # A pinned serial missing from a list that never had serials in it is not
    # evidence of an absent camera. Skipping on that would refuse to start a
    # camera that is sitting right there on the bus.
    identified = hw._serials_from_librealsense()
    detected = hw._find_realsense_devices()
    serial_for = {
        'gripper_camera': LaunchConfiguration('gripper_serial').perform(context),
        'rover_camera': LaunchConfiguration('front_serial').perform(context),
    }

    actions = []
    started = []
    for camera in cameras:
        serial = serial_for.get(camera, '')
        # Only an identified enumeration can prove a camera absent. A driver
        # told to wait for a serial that is not there does not fail: it retries
        # "device ... is NOT found" forever while the downlink beside it warns
        # about a stream that will never arrive. Skipping says so once and
        # leaves the other camera working.
        if serial and identified and serial not in identified:
            print(f'[cameras] {camera} (serial {serial}) not on USB '
                  f'-- enumerated: {", ".join(identified)}. Skipping.')
            continue
        if not identified:
            print(f'[cameras] {camera}: could not enumerate serials '
                  f'({len(detected)} RealSense-like device(s) in sysfs); '
                  'starting it and letting the driver match.')
        actions.append(_driver_with_tf(hw, camera, serial, driver_tf))
        started.append(camera)

    if not started:
        raise RuntimeError(
            'None of the requested cameras is on USB. Enumerated: '
            f'{", ".join(identified) if identified else "nothing"}. '
            'A camera that was working and vanished usually means it '
            're-enumerated: check `lsusb` and replug it.')

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('aries_bringup'),
            '/launch/camera_downlink.launch.py']),
        launch_arguments={
            'cameras': ','.join(started),
            'downlink_profile': LaunchConfiguration('downlink_profile'),
            'downlink_rate_hz': LaunchConfiguration('downlink_rate_hz'),
            'downlink_depth_rate_hz': LaunchConfiguration('downlink_depth_rate_hz'),
        }.items(),
    ))
    return actions


def _driver_with_tf(hw, camera, serial, publish_tf):
    """aries_hardware's driver include, with publish_tf overridden.

    Rebuilt from the original's own source and arguments rather than reaching
    into it, so the profile and reset behaviour stay defined in one place: this
    file should not be the reason the two ends drift apart.
    """
    base = hw._realsense_driver(camera, serial)
    args = dict(base.launch_arguments)
    args['publish_tf'] = publish_tf
    return IncludeLaunchDescription(base.launch_description_source,
                                    launch_arguments=args.items())


def generate_launch_description():
    # my_robot.urdf.xacro, not rover.urdf.xacro. The rover one describes the
    # chassis only: no arm, and therefore no gripper camera, so TF came out
    # with every rover_camera_* frame present and not one gripper_camera_*
    # frame. That failure is quiet in exactly the wrong way -- the front
    # camera's cloud renders perfectly while the wrist camera's silently never
    # appears, which reads as a camera fault rather than a missing transform.
    # This is the same description MoveIt and full_hardware load.
    urdf = PathJoinSubstitution([
        FindPackageShare('aries'), 'urdf', 'my_robot.urdf.xacro'])
    rviz_cfg = PathJoinSubstitution([
        FindPackageShare('aries_bringup'), 'launch', 'cameras.rviz'])
    robot_tf = IfCondition(PythonExpression(
        ["'", LaunchConfiguration('tf'), "' == 'robot'"]))

    return LaunchDescription([
        DeclareLaunchArgument('cameras',
                              default_value='gripper_camera,rover_camera'),
        DeclareLaunchArgument('tf', default_value='robot',
                              choices=['robot', 'driver', 'none'],
                              description='Where camera TF comes from.'),
        DeclareLaunchArgument('downlink_profile', default_value='balanced'),
        DeclareLaunchArgument('downlink_rate_hz', default_value='15.0'),
        DeclareLaunchArgument('downlink_depth_rate_hz', default_value='5.0'),
        DeclareLaunchArgument('gripper_serial',
                              default_value=device_str('cameras.gripper_serial')),
        DeclareLaunchArgument('front_serial',
                              default_value=device_str('cameras.front_serial')),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              choices=['true', 'false']),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='robot_state_publisher', output='screen',
             condition=robot_tf,
             parameters=[{'robot_description': ParameterValue(
                 Command(['xacro ', urdf]), value_type=str)}]),
        # Zeros for the arm joints. Without this robot_state_publisher emits
        # only the fixed joints, and the wrist camera -- which is behind three
        # revolute ones -- never gets a transform at all, so its depth cloud
        # silently never renders.
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             name='joint_state_publisher', output='screen',
             condition=robot_tf),

        Node(package='rviz2', executable='rviz2', name='cameras_rviz',
             output='screen', arguments=['-d', rviz_cfg],
             condition=IfCondition(LaunchConfiguration('use_rviz'))),

        OpaqueFunction(function=_setup),
    ])
