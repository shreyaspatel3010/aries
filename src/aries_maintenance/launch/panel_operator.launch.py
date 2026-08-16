"""Bring up the maintenance-panel operator against an already-running arm.

Assumes MoveIt is up (aries_moveit) and the rover/gripper cameras are
publishing; this launch adds only the panel node, so it can be started and
stopped against a live sim without disturbing anything else.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    config = PathJoinSubstitution([
        get_package_share_directory('aries_maintenance'),
        'config', 'panel_tasks.yaml'])
    table = PathJoinSubstitution([
        get_package_share_directory('aries'),
        'models', 'maintenance_panel', 'panel_task.json'])
    args = [
        DeclareLaunchArgument('config_file', default_value=config),
        DeclareLaunchArgument('task_table', default_value=table),
        DeclareLaunchArgument('planning_group', default_value='igus_rebel_arm'),
        DeclareLaunchArgument('tool_frame', default_value='gripper_tcp'),
        # Two tags, because a single tag's pose carries the planar-PnP
        # ambiguity and the arm must not move on an ambiguous pose.
        DeclareLaunchArgument('min_markers', default_value='2'),
    ]
    return LaunchDescription(args + [
        Node(
            package='aries_maintenance',
            executable='panel_operator_node.py',
            name='panel_operator',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    # Also expose the exact file to the node so every startup
                    # can report which fresh YAML snapshot supplied its flags.
                    'config_file': LaunchConfiguration('config_file'),
                    'task_table': LaunchConfiguration('task_table'),
                    'planning_group': LaunchConfiguration('planning_group'),
                    'tool_frame': LaunchConfiguration('tool_frame'),
                    'min_markers': LaunchConfiguration('min_markers'),
                },
            ],
        ),
    ])
