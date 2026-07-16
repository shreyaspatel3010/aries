#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"]),

        DeclareLaunchArgument("gripper_type", default_value="new", choices=["old", "new"]),
        DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("serial_port", default_value="/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00"),
        DeclareLaunchArgument("suppress_rebel_logs", default_value="true"),
        DeclareLaunchArgument("suppress_moveit_execution_logs", default_value="true"),
        DeclareLaunchArgument("enable_depth_sensor", default_value="auto", choices=["auto", "true", "false"]),

        DeclareLaunchArgument("start_rover", default_value="true"),
        DeclareLaunchArgument("rover_hardware_protocol", default_value="auto", choices=["auto", "odrive", "mock_hardware"]),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("setup_rover_can", default_value="true"),
        DeclareLaunchArgument("use_rover_imu", default_value="auto"),
        DeclareLaunchArgument("rover_imu_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("rover_imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("rover_imu_frame", default_value="bno055"),
        DeclareLaunchArgument("rover_imu_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("use_rover_lidar", default_value="auto", choices=["auto", "true", "false"]),
        DeclareLaunchArgument("rover_lidar_sensor_ip", default_value="169.254.136.69"),
        DeclareLaunchArgument("rover_lidar_host_ip", default_value="169.254.180.121"),
        DeclareLaunchArgument("rover_lidar_frame", default_value="Lidar_Scan_Link"),
        DeclareLaunchArgument("rover_lidar_topic", default_value="/scan"),
        DeclareLaunchArgument("rover_picoscan_raw_imu_topic", default_value="/picoscan/imu_raw"),
        DeclareLaunchArgument("rover_picoscan_imu_topic", default_value="/picoscan/imu"),
        DeclareLaunchArgument("use_rover_joy_node", default_value="false"),

        DeclareLaunchArgument("start_checker", default_value="true"),
        DeclareLaunchArgument("checker_interval", default_value="4.0"),

        # Arm + gripper + MoveIt/RViz + shared joy_node.
        # Existing aries_hardware.launch.py handles real-or-mock arm/gripper auto behavior.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "aries_hardware.launch.py",
                ])
            ),
            launch_arguments={
                "use_gui": LaunchConfiguration("use_gui"),
                "use_joystick": LaunchConfiguration("use_joystick"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "joystick_control_mode": LaunchConfiguration("joystick_control_mode"),
                "gripper_type": LaunchConfiguration("gripper_type"),
                "hardware_protocol": LaunchConfiguration("hardware_protocol"),
                "arm_hardware_protocol": LaunchConfiguration("arm_hardware_protocol"),
                "gripper_hardware_protocol": LaunchConfiguration("gripper_hardware_protocol"),
                "serial_port": LaunchConfiguration("serial_port"),
                "suppress_rebel_logs": LaunchConfiguration("suppress_rebel_logs"),
                "suppress_moveit_execution_logs": LaunchConfiguration("suppress_moveit_execution_logs"),
                "enable_depth_sensor": LaunchConfiguration("enable_depth_sensor"),
            }.items(),
        ),

        # Rover real-or-mock backend.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "rover_drive_auto.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("start_rover")),
            launch_arguments={
                "rover_hardware_protocol": LaunchConfiguration("rover_hardware_protocol"),
                "can_interface": LaunchConfiguration("can_interface"),
                "setup_can": LaunchConfiguration("setup_rover_can"),
                "use_joystick": LaunchConfiguration("use_joystick"),
                # joy_node is already started by aries_hardware when use_joystick:=true.
                "use_joy_node": LaunchConfiguration("use_rover_joy_node"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "use_imu": LaunchConfiguration("use_rover_imu"),
                "imu_port": LaunchConfiguration("rover_imu_port"),
                "imu_baudrate": LaunchConfiguration("rover_imu_baudrate"),
                "imu_frame": LaunchConfiguration("rover_imu_frame"),
                "imu_topic": LaunchConfiguration("rover_imu_topic"),
                "use_lidar": LaunchConfiguration("use_rover_lidar"),
                "lidar_sensor_ip": LaunchConfiguration("rover_lidar_sensor_ip"),
                "lidar_host_ip": LaunchConfiguration("rover_lidar_host_ip"),
                "lidar_frame": LaunchConfiguration("rover_lidar_frame"),
                "lidar_scan_topic": LaunchConfiguration("rover_lidar_topic"),
                "picoscan_raw_imu_topic": LaunchConfiguration("rover_picoscan_raw_imu_topic"),
                "picoscan_imu_topic": LaunchConfiguration("rover_picoscan_imu_topic"),
            }.items(),
        ),

        # Separate checker.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "full_hardware_checker.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("start_checker")),
            launch_arguments={
                "checker_interval": LaunchConfiguration("checker_interval"),
                "serial_port": LaunchConfiguration("serial_port"),
                "can_interface": LaunchConfiguration("can_interface"),
                "use_imu": LaunchConfiguration("use_rover_imu"),
                "imu_port": LaunchConfiguration("rover_imu_port"),
                "imu_frame": LaunchConfiguration("rover_imu_frame"),
                "imu_topic": LaunchConfiguration("rover_imu_topic"),
                "picoscan_imu_topic": LaunchConfiguration("rover_picoscan_imu_topic"),
                "lidar_topic": LaunchConfiguration("rover_lidar_topic"),
            }.items(),
        ),
    ])
