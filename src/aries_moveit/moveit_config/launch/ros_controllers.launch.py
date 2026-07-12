from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Declare use_sim_time argument
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Get the controller configuration file
    ros2_controllers_path = os.path.join(
        get_package_share_directory('aries_moveit'),
        'config',
        'ros2_controllers.yaml'
    )
    
    robot_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_arm_trajectory_controller", 
                   "--controller-manager", "/controller_manager",
                   "--param-file", ros2_controllers_path],
        output="both",
        parameters=[{'use_sim_time': use_sim_time}]
    )

    robot_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_arm_position_controller", 
                   "--controller-manager", "/controller_manager",
                   "--param-file", ros2_controllers_path],
        output="both",
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager",
                   "--param-file", ros2_controllers_path,
                   "--switch-timeout", "30"],
        output="both",
        parameters=[{'use_sim_time': use_sim_time}]
    )

    gripper_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_gripper_controller", 
                   "--controller-manager", "/controller_manager",
                   "--param-file", ros2_controllers_path],
        output="both",
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Make arm controller wait for joint_state_broadcaster to finish
    arm_controller_spawner_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_trajectory_controller_spawner],
        )
    )
    
    # Make gripper controller wait for arm controller to finish
    gripper_controller_spawner_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_trajectory_controller_spawner,
            on_exit=[gripper_trajectory_controller_spawner],
        )
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        joint_state_broadcaster_spawner,
        arm_controller_spawner_event,
        gripper_controller_spawner_event,
    ])