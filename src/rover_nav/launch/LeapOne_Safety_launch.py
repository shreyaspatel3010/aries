from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to the odrive launch file
    
    # Joy node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen'
    )

    rover_control_pure_pursuit = Node(
        package='rover_nav',
        executable='rover_controller_pure_pursuit.py',
        name='rover_control',
        output='screen'
    )

    # Your rover control node
    joy_safety = Node(
        package='rover_nav',
        executable='joy_safety.py',
        name='joy_safety',
        output='screen'
    )


    return LaunchDescription([
        
        rover_control_pure_pursuit,
        joy_node,
        joy_safety
    ])