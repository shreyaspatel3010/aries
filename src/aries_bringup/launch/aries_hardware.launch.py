import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _is_realsense_connected():
    for dev_path in glob.glob("/sys/bus/usb/devices/*/"):
        vendor_file = os.path.join(dev_path, "idVendor")
        product_file = os.path.join(dev_path, "idProduct")
        try:
            if open(vendor_file).read().strip() != "8086":
                continue
            product = int(open(product_file).read().strip(), 16)
            if 0x0AD1 <= product <= 0x0B64:
                return True
        except (OSError, ValueError):
            continue
    return False


def launch_setup(context, *args, **kwargs):
    actions = []

    if _is_realsense_connected():
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("realsense2_camera"),
                        "launch",
                        "rs_launch.py",
                    ])
                ),
                launch_arguments={
                    "camera_name": "gripper_camera",
                    "camera_namespace": "",
                    "enable_color": "true",
                    "enable_depth": "true",
                    "enable_infra": "false",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "align_depth.enable": "true",
                    "publish_tf": "false",
                    "initial_reset": "true",
                    "output": "screen",
                }.items(),
            )
        )

        actions.append(
            Node(
                package="aries_vision_grasp",
                executable="yolo_detection_node.py",
                name="yolo_detection_node",
                output="screen",
                parameters=[{
                    "confidence_threshold": 0.50,
                }],
            )
        )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_moveit"),
                    "launch",
                    "aries_hardware.launch.py",
                ])
            ),
            launch_arguments={
                "use_gui": LaunchConfiguration("use_gui"),
                "gripper_type": LaunchConfiguration("gripper_type"),
                "arm_hardware_protocol": LaunchConfiguration("arm_hardware_protocol"),
                "hardware_protocol": LaunchConfiguration("hardware_protocol"),
                "gripper_hardware_protocol": LaunchConfiguration("gripper_hardware_protocol"),
                "use_joystick": LaunchConfiguration("use_joystick"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "joystick_control_mode": LaunchConfiguration("joystick_control_mode"),
                "serial_port": LaunchConfiguration("serial_port"),
                "suppress_rebel_logs": LaunchConfiguration("suppress_rebel_logs"),
                "suppress_moveit_execution_logs": LaunchConfiguration("suppress_moveit_execution_logs"),
            }.items(),
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "rover_drive_core.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("use_rover_drive")),
            launch_arguments={
                "use_imu": LaunchConfiguration("use_rover_imu"),
                "use_joystick": LaunchConfiguration("use_rover_joystick"),
                "use_joy_node": LaunchConfiguration("use_rover_joy_node"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "can_interface": LaunchConfiguration("can_interface"),
                "setup_can": LaunchConfiguration("setup_rover_can"),
                "imu_port": LaunchConfiguration("rover_imu_port"),
                "imu_baudrate": LaunchConfiguration("rover_imu_baudrate"),
                "imu_frame": LaunchConfiguration("rover_imu_frame"),
                "imu_topic": LaunchConfiguration("rover_imu_topic"),
            }.items(),
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("gripper_type", default_value="new", choices=["old", "new"]),
        DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"]),
        DeclareLaunchArgument("serial_port", default_value="/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00"),
        DeclareLaunchArgument("suppress_rebel_logs", default_value="false"),
        DeclareLaunchArgument("suppress_moveit_execution_logs", default_value="false"),

        DeclareLaunchArgument("use_rover_drive", default_value="false"),
        DeclareLaunchArgument("use_rover_joystick", default_value="true"),
        DeclareLaunchArgument("use_rover_joy_node", default_value="false"),
        DeclareLaunchArgument("use_rover_imu", default_value="auto"),
        DeclareLaunchArgument("rover_imu_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("rover_imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("rover_imu_frame", default_value="bno055"),
        DeclareLaunchArgument("rover_imu_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("setup_rover_can", default_value="true"),

        OpaqueFunction(function=launch_setup),
    ])
