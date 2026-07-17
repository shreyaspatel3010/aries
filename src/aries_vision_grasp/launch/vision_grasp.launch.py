from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _share_path(relative: str) -> str:
    return get_package_share_directory('aries_vision_grasp') + '/' + relative


def generate_launch_description():
    # Only the settings that are commonly overridden per run are launch
    # arguments. All other tuning lives in config/vision_grasp_params.yaml;
    # postures and gripper-completion gating live in config/pick_place.yaml
    # (loaded last, so it stays the authoritative file for those values).
    launch_args = [
        # aries_bringup/my_robot.launch.py defaults to Gazebo/use_sim_time=true.
        # Keep action durations and watchdogs on that same clock. Override with
        # use_sim_time:=false when running the physical rover.
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('model_path', default_value=_share_path('models/grasp.pt')),
        DeclareLaunchArgument('target_class', default_value='probe'),
        DeclareLaunchArgument(
            'params_file',
            default_value=_share_path('config/vision_grasp_params.yaml'),
            description='YAML with the full vision/grasp tuning parameter set.',
        ),
        DeclareLaunchArgument(
            'pick_place_config',
            default_value=_share_path('config/pick_place.yaml'),
            description='YAML containing home and base-box placement postures.',
        ),
    ]

    return LaunchDescription(launch_args + [
        Node(
            package='aries_vision_grasp',
            executable='vision_grasp_node.py',
            name='vision_grasp_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'model_path': LaunchConfiguration('model_path'),
                    'target_class': LaunchConfiguration('target_class'),
                },
                # Last so posture values live in one authoritative file.
                LaunchConfiguration('pick_place_config'),
            ],
        ),
    ])
