import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ...existing code...
    pkg_share = get_package_share_directory('aries')
    default_model = os.path.join(pkg_share, 'urdf', 'igus_rebel_robot2.urdf.xacro')
    default_rviz = os.path.join(pkg_share, 'rviz', 'urdf_config.rviz')

    declare_model = DeclareLaunchArgument('model', default_value=default_model, description='Path to robot xacro/urdf')
    declare_rviz = DeclareLaunchArgument('rviz_config', default_value=default_rviz, description='Path to RViz config')

    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': Command([FindExecutable(name='xacro'), ' ', LaunchConfiguration('model')])}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')]
    )

    return LaunchDescription([
        declare_model,
        declare_rviz,
        joint_state_publisher,
        robot_state_publisher,
        rviz_node
    ])
    # ...existing code...
