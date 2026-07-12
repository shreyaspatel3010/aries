import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    default_rviz_file = os.path.join(
        get_package_share_directory('aries_moveit'),
        'launch',
        'moveit.rviz'
    )
    return LaunchDescription([
        DeclareLaunchArgument('debug', default_value='false', description='Enable debugging'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz_file, description='Path to RViz config file'),
        DeclareLaunchArgument(
            'use_sim_time', 
            default_value='false', 
            description='Use sim time if true'),

        # Determine launch prefix based on debug flag
        DeclareLaunchArgument(
            'launch_prefix', 
            default_value='',
            condition=launch.conditions.LaunchConfigurationEquals('debug', 'false'),
            description='Prefix for launching nodes (default is empty)'),

        DeclareLaunchArgument(
            'launch_prefix', 
            default_value='gdb --ex run --args',
            condition=launch.conditions.LaunchConfigurationEquals('debug', 'true'),
            description='Prefix for launching nodes with gdb'),

        # Set arguments for RViz node
        DeclareLaunchArgument(
            'command_args',
            default_value='',
            condition=launch.conditions.LaunchConfigurationEquals('rviz_config', ''),
            description='Command arguments for RViz'),

        # Start RViz node with the arguments set above
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output='screen',
            parameters=[{
                "use_sim_time": LaunchConfiguration('use_sim_time'),
            }],
            arguments=["--display-config", LaunchConfiguration('rviz_config')]
        ),
    ])