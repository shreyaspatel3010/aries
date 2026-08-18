#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _text(argument):
    """A launch argument forced to STRING.

    These three take "auto" | "true" | "false". Left unwrapped, launch infers
    the type from the text, so passing true/false lands a BOOL on a parameter
    the node declared as a string and the checker dies on startup with
    InvalidParameterTypeException instead of reporting anything at all.
    """
    return ParameterValue(LaunchConfiguration(argument), value_type=str)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("checker_interval", default_value="3.0"),
        DeclareLaunchArgument("timeout", default_value="5.0"),
        DeclareLaunchArgument("serial_port", default_value="/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00"),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("use_imu", default_value="auto"),
        DeclareLaunchArgument("imu_port", default_value="/dev/microstrain_main"),
        DeclareLaunchArgument("imu_frame", default_value="imu_frame"),
        DeclareLaunchArgument("imu_topic", default_value="/microstrain/imu/data"),
        DeclareLaunchArgument("check_imu", default_value="true"),
        DeclareLaunchArgument("require_all_rover_axes", default_value="true"),
        DeclareLaunchArgument(
            "require_closed_loop",
            default_value="false",
            description=(
                "Require closed-loop ODrive state. False is the safe preflight "
                "default because the drive starts disarmed."
            ),
        ),
        DeclareLaunchArgument("check_odrive_status", default_value="true"),
        DeclareLaunchArgument("expected_odrive_axes", default_value="6"),
        # Both RealSense cameras aries_hardware.launch.py can start. The modes
        # mirror its enable_depth_sensor / enable_front_camera arguments, so a
        # camera explicitly asked for is reported as an error when it is absent
        # while an "auto" one is only noted.
        DeclareLaunchArgument(
            "gripper_camera_mode", default_value="auto",
            choices=["auto", "true", "false"],
        ),
        DeclareLaunchArgument(
            "front_camera_mode", default_value="auto",
            choices=["auto", "true", "false"],
        ),
        DeclareLaunchArgument(
            "gripper_camera_color_topic",
            default_value="/gripper_camera/color/image_raw",
        ),
        DeclareLaunchArgument(
            "front_camera_color_topic",
            default_value="/camera/color/image_raw",
        ),

        Node(
            package="aries_bringup",
            executable="full_hardware_checker.py",
            name="full_hardware_checker",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "check_interval": LaunchConfiguration("checker_interval"),
                "timeout": LaunchConfiguration("timeout"),
                "gripper_serial_port": LaunchConfiguration("serial_port"),
                "can_interface": LaunchConfiguration("can_interface"),
                "use_imu": _text("use_imu"),
                "imu_port": LaunchConfiguration("imu_port"),
                "imu_frame": LaunchConfiguration("imu_frame"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "check_imu": LaunchConfiguration("check_imu"),
                "require_all_rover_axes": LaunchConfiguration("require_all_rover_axes"),
                "require_closed_loop": LaunchConfiguration("require_closed_loop"),
                "check_odrive_status": LaunchConfiguration("check_odrive_status"),
                "expected_odrive_axes": LaunchConfiguration("expected_odrive_axes"),
                "gripper_camera_color_topic": LaunchConfiguration("gripper_camera_color_topic"),
                "front_camera_color_topic": LaunchConfiguration("front_camera_color_topic"),
                "gripper_camera_mode": _text("gripper_camera_mode"),
                "front_camera_mode": _text("front_camera_mode"),
                "print_only_on_change": True,
            }],
        ),
    ])
