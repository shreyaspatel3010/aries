from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the config file
    pkg_share = get_package_share_directory('rover_nav')
    ekf_config = os.path.join(pkg_share, 'config', 'ekf_config.yaml')
    
    return LaunchDescription([
        # Encoder odometry node
        Node(
            package='rover_nav',
            executable='Odom.py',
            name='Odom',
            output='screen'
        ),
        
        # BNO055 IMU node
        Node(
            package='bno055',
            executable='bno055',
            name='bno055',
            output='screen',
            parameters=[{
                'connection_type': 'uart',
                'uart_port': '/dev/ttyUSB0',
                'uart_baudrate': 115200,
                'frame_id': 'bno055'
            }]
        ),
        
        # Static transform: base_link → bno055
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'bno055']
        ),
        
        # EKF node (publishes odom → base_link with fused IMU yaw)
     Node(
           package='robot_localization',
           executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
           parameters=[ekf_config]
       ),
    ])
