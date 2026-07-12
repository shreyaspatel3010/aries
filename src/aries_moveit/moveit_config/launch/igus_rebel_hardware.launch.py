from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    use_gui_arg = DeclareLaunchArgument(
        'use_gui', 
        default_value='true', 
        description='Launch RViz with MoveIt interface'
    )
    use_joystick_arg = DeclareLaunchArgument(
        'use_joystick',
        default_value='true',
        description='Start joystick arm teleop'
    )
    joy_layout_arg = DeclareLaunchArgument(
        'joy_layout',
        default_value='auto',
        choices=['auto', 'dongle', 'bluetooth', 'game_controller', 'passthrough'],
        description='Normalize joystick layout before teleop nodes consume /joy'
    )
    joy_driver_arg = DeclareLaunchArgument(
        'joy_driver',
        default_value='game_controller_node',
        choices=['game_controller_node', 'joy_node'],
        description='Joystick driver executable from the joy package'
    )
    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev',
        default_value='/dev/input/js0',
        description='Joystick device used by joy_node and the layout normalizer'
    )
    joystick_control_mode_arg = DeclareLaunchArgument(
        'joystick_control_mode',
        default_value='servo',
        choices=['move_group', 'servo'],
        description='servo uses smooth Cartesian MoveIt Servo teleop with collision guard; move_group uses planned steps'
    )
    
    # Get the path to igus_rebel launch file (relative to this launch file)
    # This launch file is at: src/aries_moveit/launch/igus_rebel_hardware.launch.py
    # We need: src/aries_moveit/igus_rebel/launch/rebel.launch.py
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    igus_rebel_launch_path = os.path.join(
        os.path.dirname(current_file_dir),  # Go up to aries_moveit
        'igus_rebel', 'launch', 'rebel.launch.py'
    )
    
    # Launch the robot hardware interface
    robot_hardware_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(igus_rebel_launch_path),
        launch_arguments={
            'hardware_protocol': 'rebel',
        }.items(),
    )
    
    # Launch MoveIt move_group
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('aries_moveit'), 'launch', 'move_group.launch.py')
        ),
        launch_arguments={
            'hardware_protocol': 'rebel',
            'use_sim_time': 'false',
            'use_gui': LaunchConfiguration('use_gui'),
            'use_joystick': LaunchConfiguration('use_joystick'),
            'joy_driver': LaunchConfiguration('joy_driver'),
            'joy_layout': LaunchConfiguration('joy_layout'),
            'joy_dev': LaunchConfiguration('joy_dev'),
            'joystick_control_mode': LaunchConfiguration('joystick_control_mode'),
        }.items(),
    )
    
    return LaunchDescription([
        use_gui_arg,
        use_joystick_arg,
        joy_driver_arg,
        joy_layout_arg,
        joy_dev_arg,
        joystick_control_mode_arg,
        robot_hardware_launch,
        moveit_launch,
    ])
