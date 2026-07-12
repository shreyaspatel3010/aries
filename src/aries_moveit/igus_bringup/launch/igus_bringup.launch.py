from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # Include rebel.launch.py from igus_rebel package
    rebel_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('igus_rebel'),
                'launch',
                'rebel.launch.py'
            ])
        ])
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
        launch_arguments={'use_gui': 'true'}.items()
    )

    return LaunchDescription([
        rebel_launch,
        moveit_launch
    ])
