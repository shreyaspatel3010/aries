from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _share_path(relative: str) -> str:
    return get_package_share_directory('aries_soil_sample') + '/' + relative


def generate_launch_description():
    launch_args = [
        # Same reasoning as aries_vision_grasp: standalone use means the
        # physical rover, and a true default with no /clock publisher freezes
        # every timer, leaving the node "ready" while producing nothing. Pass
        # use_sim_time:=true explicitly against Gazebo.
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'params_file',
            default_value=_share_path('config/soil_sample_params.yaml'),
            description='YAML with the full terrain/scoop parameter set.',
        ),
        DeclareLaunchArgument(
            'finger_type',
            default_value='bucket',
            choices=['bucket'],
            description='Only the bucket fingertip can scoop. Listed as an '
                        'argument so a mismatch with the URDF is an explicit '
                        'choice rather than a silent default -- the four-bar '
                        'contact point differs by up to 23 mm between jaws.',
        ),
        DeclareLaunchArgument(
            'auto_start',
            default_value='false',
            description='When false the node surveys and waits, and a scoop '
                        'runs only when triggered. Default off deliberately: '
                        'this drives a gripper into the ground.',
        ),
    ]

    return LaunchDescription(launch_args + [
        Node(
            package='aries_soil_sample',
            executable='soil_sample_node.py',
            name='soil_sample_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'finger_type': LaunchConfiguration('finger_type'),
                    'auto_start': LaunchConfiguration('auto_start'),
                },
            ],
        ),
    ])
