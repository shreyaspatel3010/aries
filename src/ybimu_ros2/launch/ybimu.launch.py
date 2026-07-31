"""Launch the standalone YaBoom 10-axis IMU driver."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Create a configurable standalone driver launch description."""
    package_share = get_package_share_directory('ybimu_ros2')
    default_config = os.path.join(package_share, 'config', 'ybimu.yaml')

    arguments = [
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='YaBoom driver parameter YAML'),
        DeclareLaunchArgument(
            'port', default_value='/dev/imu_ybimu',
            description='Serial device or persistent udev symlink'),
        DeclareLaunchArgument(
            'frame_id', default_value='imu_link',
            description='Frame assigned to published sensor messages'),
        DeclareLaunchArgument(
            'imu_topic', default_value='/imu',
            description='sensor_msgs/Imu output topic'),
        DeclareLaunchArgument(
            'report_rate_hz', default_value='100',
            description='IMU report rate, clamped to 10-100 Hz'),
        DeclareLaunchArgument(
            'fusion_axes', default_value='9',
            description='Fusion mode: 6 ignores magnetometer; 9 uses it'),
        DeclareLaunchArgument(
            'orientation_mode', default_value='planar_gyro_mag',
            description='planar_gyro_mag or vendor_fused'),
        DeclareLaunchArgument(
            'publish_linear_acceleration', default_value='false',
            description='Publish acceleration in sensor_msgs/Imu'),
    ]

    driver = Node(
        package='ybimu_ros2',
        executable='ybimu_driver',
        name='ybimu',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'port': LaunchConfiguration('port'),
                'frame_id': LaunchConfiguration('frame_id'),
                'imu_topic': LaunchConfiguration('imu_topic'),
                'report_rate_hz': ParameterValue(
                    LaunchConfiguration('report_rate_hz'), value_type=int),
                'fusion_axes': ParameterValue(
                    LaunchConfiguration('fusion_axes'), value_type=int),
                'orientation_mode': LaunchConfiguration('orientation_mode'),
                'publish_linear_acceleration': ParameterValue(
                    LaunchConfiguration('publish_linear_acceleration'),
                    value_type=bool),
            },
        ],
        respawn=True,
        respawn_delay=5.0,
    )

    return LaunchDescription(arguments + [driver])
