#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    aries_share = get_package_share_directory("aries")

    urdf_path = PathJoinSubstitution([
        FindPackageShare("aries"),
        "urdf",
        "rover.urdf.xacro",
    ])
    gazebo_config_path = PathJoinSubstitution([
        FindPackageShare("aries"),
        "config",
        "rover_gazebo_bridge.yaml",
    ])
    gz_gui_config_path = PathJoinSubstitution([
        FindPackageShare("aries"),
        "config",
        "gz_gui_config.config",
    ])
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare("aries"),
        "rviz",
        "urdf_config.rviz",
    ])
    world_path = PathJoinSubstitution([
        FindPackageShare("aries"),
        "worlds",
        "test_world.sdf",
    ])
    virtual_diff_config_path = PathJoinSubstitution([
        FindPackageShare("aries"),
        "config",
        "virtual_differential.yaml",
    ])
    rover_joystick_config_path = os.path.join(
        get_package_share_directory("aries_bringup"),
        "config",
        "rover_cmd_vel_joystick.yaml",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    use_rover_joystick = LaunchConfiguration("use_rover_joystick")
    use_rover_joy_node = LaunchConfiguration("use_rover_joy_node")
    joy_driver = LaunchConfiguration("joy_driver")
    joy_layout = LaunchConfiguration("joy_layout")
    joy_dev = LaunchConfiguration("joy_dev")
    use_static_wheel_joint_publisher = LaunchConfiguration(
        "use_static_wheel_joint_publisher"
    )
    gz_version = LaunchConfiguration("gz_version")
    gz_config_path = LaunchConfiguration("gz_config_path")
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_z = LaunchConfiguration("spawn_z")

    set_gz_sim_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        [
            os.path.dirname(aries_share),
            ":",
            aries_share,
            "/models:",
            aries_share,
            "/worlds:",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )
    set_gz_config_path = SetEnvironmentVariable("GZ_CONFIG_PATH", gz_config_path)

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            ])
        ),
        launch_arguments={
            "gz_args": [world_path, " -r"],
            "gz_version": gz_version,
        }.items(),
    )

    robot_description_content = ParameterValue(
        Command(["xacro ", urdf_path]),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description_content,
            "use_sim_time": use_sim_time,
        }],
    )

    spawn_robot_node = Node(
        package="ros_gz_sim",
        executable="create",
        name="create",
        output="screen",
        arguments=[
            "-name", "aries",
            "--gui-config", gz_gui_config_path,
            "-topic", "robot_description",
            "-allow_renaming", "true",
            "-x", spawn_x,
            "-y", spawn_y,
            "-z", spawn_z,
        ],
    )

    parameter_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{
            "config_file": gazebo_config_path,
            "use_sim_time": use_sim_time,
        }],
    )

    virtual_differential_node = Node(
        package="aries",
        executable="virtual_differential",
        name="virtual_differential",
        output="screen",
        parameters=[
            virtual_diff_config_path,
            {"use_sim_time": use_sim_time},
        ],
    )

    wheel_joint_publisher_node = Node(
        condition=IfCondition(use_static_wheel_joint_publisher),
        package="aries_moveit",
        executable="publish_wheel_joints.py",
        name="wheel_joint_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    rover_joystick_node = Node(
        condition=IfCondition(use_rover_joystick),
        package="aries_bringup",
        executable="rover_cmd_vel_joystick.py",
        name="rover_cmd_vel_joystick",
        output="screen",
        parameters=[rover_joystick_config_path],
    )

    rover_joy_node = Node(
        condition=IfCondition(use_rover_joy_node),
        package="joy",
        executable=joy_driver,
        name="rover_joy_node",
        parameters=[{"dev": joy_dev}],
        remappings=[("joy", "joy/raw")],
        output="screen",
    )

    rover_joy_layout_normalizer_node = Node(
        condition=IfCondition(use_rover_joy_node),
        package="aries_moveit",
        executable="joy_layout_normalizer.py",
        name="rover_joy_layout_normalizer",
        parameters=[{
            "input_topic": "joy/raw",
            "output_topic": "joy",
            "layout": joy_layout,
            "device": joy_dev,
        }],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz with aries/rviz/urdf_config.rviz",
        ),
        DeclareLaunchArgument(
            "gz_version",
            default_value="8",
            description="Gazebo Sim major version: 8=Harmonic, 10=Jetty",
        ),
        DeclareLaunchArgument(
            "gz_config_path",
            default_value=EnvironmentVariable("GZ_CONFIG_PATH", default_value=""),
            description=(
                "Gazebo tool config path. Use /usr/share/gz with gz_version:=10 "
                "to launch system Jetty instead of ROS Jazzy's vendored Harmonic."
            ),
        ),
        DeclareLaunchArgument(
            "spawn_x",
            default_value="0.0",
            description="Spawn position X coordinate",
        ),
        DeclareLaunchArgument(
            "spawn_y",
            default_value="0.0",
            description="Spawn position Y coordinate",
        ),
        DeclareLaunchArgument(
            "spawn_z",
            default_value="0.1",
            description="Spawn position Z coordinate",
        ),
        DeclareLaunchArgument(
            "use_rover_joystick",
            default_value="true",
            description="Start joystick-to-cmd_vel rover teleop",
        ),
        DeclareLaunchArgument(
            "use_rover_joy_node",
            default_value="false",
            description="Start a joy_node for rover joystick input",
        ),
        DeclareLaunchArgument(
            "joy_layout",
            default_value="auto",
            choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"],
            description="Normalize joystick layout before rover teleop consumes /joy",
        ),
        DeclareLaunchArgument(
            "joy_driver",
            default_value="game_controller_node",
            choices=["game_controller_node", "joy_node"],
            description="Joystick driver executable from the joy package",
        ),
        DeclareLaunchArgument(
            "joy_dev",
            default_value="/dev/input/js0",
            description="Joystick device used by joy_node and the layout normalizer",
        ),
        DeclareLaunchArgument(
            "use_static_wheel_joint_publisher",
            default_value="false",
            description=(
                "Publish zero wheel joint states when Gazebo joint states are not bridged"
            ),
        ),

        set_gz_sim_resource_path,
        set_gz_config_path,
        gazebo_launch,
        robot_state_publisher_node,
        spawn_robot_node,
        parameter_bridge_node,
        virtual_differential_node,
        wheel_joint_publisher_node,
        rviz_node,
        rover_joystick_node,
        rover_joy_node,
        rover_joy_layout_normalizer_node,
    ])
