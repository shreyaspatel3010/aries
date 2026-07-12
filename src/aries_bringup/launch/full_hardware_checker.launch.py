#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("checker_interval", default_value="3.0"),
        DeclareLaunchArgument("timeout", default_value="5.0"),
        DeclareLaunchArgument("serial_port", default_value="/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00"),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("use_imu", default_value="auto"),
        DeclareLaunchArgument("imu_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("imu_frame", default_value="bno055"),
        DeclareLaunchArgument("imu_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("picoscan_imu_topic", default_value="/picoscan/imu"),
        DeclareLaunchArgument("lidar_topic", default_value="/scan"),
        DeclareLaunchArgument("check_imu", default_value="true"),
        DeclareLaunchArgument("require_all_rover_axes", default_value="true"),
        DeclareLaunchArgument("require_closed_loop", default_value="true"),
        DeclareLaunchArgument("check_odrive_status", default_value="true"),
        DeclareLaunchArgument("expected_odrive_axes", default_value="6"),

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
                "use_imu": LaunchConfiguration("use_imu"),
                "imu_port": LaunchConfiguration("imu_port"),
                "imu_frame": LaunchConfiguration("imu_frame"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "picoscan_imu_topic": LaunchConfiguration("picoscan_imu_topic"),
                "lidar_topic": LaunchConfiguration("lidar_topic"),
                "check_imu": LaunchConfiguration("check_imu"),
                "require_all_rover_axes": LaunchConfiguration("require_all_rover_axes"),
                "require_closed_loop": LaunchConfiguration("require_closed_loop"),
                "check_odrive_status": LaunchConfiguration("check_odrive_status"),
                "expected_odrive_axes": LaunchConfiguration("expected_odrive_axes"),
                "print_only_on_change": True,
            }],
        ),
    ])
