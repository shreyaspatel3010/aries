#!/usr/bin/env python3
"""
Standalone manual bringup launch for the real Aries rover.

This file only declares arguments, starts the robot model and the hardware
checker, and forwards everything else to the subsystem packages. Each of them
has one launch file and runs standalone:

  aries_imu          imu.launch.py           MicroStrain 3DM-GX5-AHRS
  aries_localization localization.launch.py  Odom.py + EKF
  aries_drive        drive.launch.py         ODrive cmd_vel bridge, or mock drive
  aries_teleop       joystick.launch.py      joy driver, layout normalizer, rover controller
                                             (included by drive.launch.py)

Every hardware subsystem auto-detects, using the shared probes in aries_common:
if the real device is present it is used, otherwise the rover falls back to
mock/degraded operation. Defaults are 'auto' everywhere, so
`ros2 launch aries_bringup rover_drive.launch.py` does the right thing on the
bench and on the real rover without extra arguments.

  - drive   : ODrive over CAN if the CAN interface exists, else mock_rover_drive
              (rover_hardware_protocol=auto)
  - IMU     : MicroStrain 3DM-GX5-AHRS if /dev/microstrain_main exists, else
              wheel-odom only (use_imu=auto)
  - localization: Odom.py + EKF on the real drive; skipped in mock (mock owns
              /odom and the odom->base_footprint TF)

This launch deliberately does not include a waypoint follower or cmd_vel
arbiter. On the physical ODrive backend it starts disarmed and connects the
LB-gated joystick stream directly to the fail-safe ODrive bridge:

  /joy -> /cmd_vel/teleop -> /cmd_vel -> cmd_vel_odrive_bridge

Enable the drive explicitly through ``/aries_drive/enable`` after checking the
rover and keeping the joystick centred. Pressing LB+Y on the pad calls the same
service, which is also how a faulted axis is recovered in the field.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _include(package, filename, forward=(), literals=None, **renamed):
    """Include a subsystem launch file.

    `forward` lists arguments passed straight through; `renamed` maps a
    subsystem's argument name to the top-level argument that feeds it;
    `literals` maps a subsystem's argument name to an already-resolved value.
    """
    launch_arguments = {name: LaunchConfiguration(name) for name in forward}
    launch_arguments.update(
        {child: LaunchConfiguration(parent) for child, parent in renamed.items()}
    )
    launch_arguments.update(literals or {})
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", filename])
        ),
        launch_arguments=launch_arguments.items(),
    )


def _robot_model_nodes():
    urdf_path = PathJoinSubstitution([FindPackageShare("aries"), "urdf", "rover.urdf.xacro"])
    rviz_config_path = PathJoinSubstitution([FindPackageShare("aries"), "rviz", "urdf_config.rviz"])

    return [
        Node(
            condition=IfCondition(
                LaunchConfiguration("use_robot_description")
            ),
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": ParameterValue(
                    Command(["xacro ", urdf_path]),
                    value_type=str,
                ),
            }],
        ),
        Node(
            condition=IfCondition(
                LaunchConfiguration("use_wheel_joint_publisher")
            ),
            package="aries_bringup",
            executable="publish_wheel_joints.py",
            name="wheel_joint_publisher",
            output="screen",
        ),
        Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config_path],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        # --- drive backend -------------------------------------------------
        DeclareLaunchArgument(
            "rover_hardware_protocol",
            default_value="auto",
            choices=["auto", "odrive", "mock_hardware"],
            description="auto picks odrive when the CAN interface exists, otherwise mock.",
        ),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("can_bitrate", default_value="250000"),
        DeclareLaunchArgument(
            "setup_can",
            default_value="true",
            description="Bring the CAN interface up with sudo -n before starting ODrive nodes.",
        ),

        # --- joystick ------------------------------------------------------
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("use_joy_node", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node",
                              choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto",
                              choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("use_cmd_vel_bridge", default_value="true"),
        DeclareLaunchArgument("drive_cmd_vel_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument(
            "drive_auto_arm",
            default_value="false",
            description=(
                "Automatically arm every ODrive axis. Keep false until the "
                "physical rover has passed its pre-drive checks."
            ),
        ),
        DeclareLaunchArgument("drive_command_timeout_s", default_value="0.25"),
        DeclareLaunchArgument("drive_max_linear_mps", default_value="0.45"),
        DeclareLaunchArgument("drive_max_angular_rps", default_value="2.10"),
        DeclareLaunchArgument("drive_max_wheel_rps", default_value="1.50"),
        DeclareLaunchArgument("drive_wheel_accel_rps2", default_value="3.0"),
        DeclareLaunchArgument(
            "use_cmd_vel_relay",
            default_value="true",
            description=(
                "Route LB-gated /cmd_vel/teleop directly to /cmd_vel. "
                "Waypoint launches override this to false and start an arbiter."
            ),
        ),

        # --- visualization -------------------------------------------------
        DeclareLaunchArgument("use_robot_description", default_value="true"),
        DeclareLaunchArgument(
            "use_wheel_joint_publisher", default_value="true"
        ),
        DeclareLaunchArgument("use_rviz", default_value="true"),

        # --- hardware checker ----------------------------------------------
        DeclareLaunchArgument(
            "start_checker",
            default_value="true",
            description="Run the rover hardware checker alongside the ODrive backend.",
        ),
        DeclareLaunchArgument("checker_interval", default_value="3.0"),
        DeclareLaunchArgument("checker_timeout", default_value="5.0"),
        DeclareLaunchArgument(
            "checker_require_closed_loop",
            default_value="false",
            description=(
                "Require every responding ODrive axis to be in closed-loop. "
                "The safe preflight default is false because the drive starts disarmed."
            ),
        ),

        # --- localization IMU ----------------------------------------------
        DeclareLaunchArgument("start_imu_driver", default_value="auto",
                              choices=["auto", "true", "false"],
                              description="auto starts the driver for whichever IMU hardware is "
                                          "present, on any drive backend; false disables it."),
        DeclareLaunchArgument("use_imu", default_value="auto",
                              choices=["auto", "true", "false", "microstrain"]),
        DeclareLaunchArgument("imu_port", default_value="/dev/microstrain_main",
                              description="3DM-GX5-AHRS serial port."),
        DeclareLaunchArgument("imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("imu_frame", default_value="imu_frame"),
        DeclareLaunchArgument("imu_topic", default_value="/microstrain/imu/data"),

        *_robot_model_nodes(),

        Node(
            condition=IfCondition(LaunchConfiguration("start_checker")),
            package="aries_bringup",
            executable="hardware_checker.py",
            name="rover_hardware_checker",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "check_interval": LaunchConfiguration("checker_interval"),
                "timeout": LaunchConfiguration("checker_timeout"),
                "require_closed_loop": LaunchConfiguration(
                    "checker_require_closed_loop"
                ),
            }],
        ),

        _include(
            "aries_imu",
            "imu.launch.py",
            [
                "start_imu_driver",
                "use_imu",
                "imu_port",
                "imu_baudrate",
                "imu_frame",
                "imu_topic",
            ],
        ),
        _include(
            "aries_localization",
            "localization.launch.py",
            [
                "rover_hardware_protocol",
                "can_interface",
                "use_imu",
                "imu_port",
                "imu_frame",
                "imu_topic",
            ],
        ),
        _include("aries_drive", "drive.launch.py", [
            "rover_hardware_protocol",
            "can_interface",
            "can_bitrate",
            "setup_can",
            "use_joystick",
            "use_joy_node",
            "joy_driver",
            "joy_layout",
            "joy_dev",
            "use_cmd_vel_bridge",
            "drive_cmd_vel_topic",
            "drive_auto_arm",
            "drive_command_timeout_s",
            "drive_max_linear_mps",
            "drive_max_angular_rps",
            "drive_max_wheel_rps",
            "drive_wheel_accel_rps2",
            "use_cmd_vel_relay",
            # Lets the mock backend work out whether the EKF will claim
            # odom -> base_footprint; no IMU driver is started there.
            "use_imu",
            "imu_port",
        ]),
    ])
