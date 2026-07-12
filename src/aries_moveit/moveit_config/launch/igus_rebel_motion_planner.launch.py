from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    debug_arg = DeclareLaunchArgument(
        'debug', default_value='false', description='')
    load_robot_description_arg = DeclareLaunchArgument(
        'load_robot_description', default_value='false', description='')
    use_gui_arg = DeclareLaunchArgument(
        'use_gui', default_value='false', description='')
    
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('aries_moveit'), 'launch', 'move_group.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'use_gui': LaunchConfiguration('use_gui'),
        }.items(),
    )

    
    return LaunchDescription([
        debug_arg,
        load_robot_description_arg,
        use_gui_arg,
        moveit_launch,
    ])
    