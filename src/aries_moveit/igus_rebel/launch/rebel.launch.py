from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
  
    hardware_protocol_arg = DeclareLaunchArgument(
        "hardware_protocol",
        default_value="rebel",
        choices=["mock_hardware", "gazebo", "rebel"],
        description="Which hardware protocol or mock hardware should be used",
    )
    
    igus_rebel_pkg = get_package_share_directory("igus_rebel")
    aries_pkg = get_package_share_directory("aries")
    
    controller_config_path = os.path.join(igus_rebel_pkg, "config", "controller.yaml")

    robot_description_file = os.path.join(
        aries_pkg,
        'urdf',
        'igus_rebel_robot2.urdf.xacro'
    )

    robot_description = {"robot_description" : Command(
        [
            FindExecutable(name="xacro"),
            " ",
            robot_description_file,
            " hardware_protocol:=",
            LaunchConfiguration("hardware_protocol"),
        ]
    )}


    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            controller_config_path],
        remappings=[
        ],
        output="both",
    )
    
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        remappings=[],
        output="both",
    )

    print("robot state publisher started")

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="both",
    )
    
    robot_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_arm_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="both",
    )

    robot_velocity_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_arm_velocity_controller", "--controller-manager", "/controller_manager"],
        output="both",
    )

    hand_guiding_node = Node(
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("hardware_protocol"), "' == 'rebel'"
        ])),
        package="igus_rebel",
        executable="rebel_hand_guiding.py",
        name="rebel_hand_guiding",
        output="screen",
    )
    
    return LaunchDescription([
        hardware_protocol_arg,
        ros2_control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        robot_trajectory_controller_spawner,
        hand_guiding_node,
        # robot_velocity_controller_spawner,
    ])
    
