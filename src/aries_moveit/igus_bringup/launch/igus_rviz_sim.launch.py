from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Declare launch arguments
    hardware_protocol_arg = DeclareLaunchArgument(
        'hardware_protocol',
        default_value='mock_hardware',
        choices=['mock_hardware', 'gazebo', 'rebel'],
        description='Hardware protocol to use (mock_hardware for simulation)'
    )
    
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Start RViz GUI'
    )
    
    # Include rebel.launch.py from igus_rebel package
    rebel_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('igus_rebel'),
                'launch',
                'rebel.launch.py'
            ])
        ]),
        launch_arguments={'hardware_protocol': LaunchConfiguration('hardware_protocol')}.items()
    )

    # Include igus_rebel_motion_planner.launch.py from aries_moveit package
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('aries_moveit'),
                'launch',
                'igus_rebel_motion_planner.launch.py'
            ])
        ]),
        launch_arguments={'use_gui': LaunchConfiguration('use_gui')}.items()
    )

    return LaunchDescription([
        hardware_protocol_arg,
        use_gui_arg,
        rebel_launch,
        moveit_launch
    ])
