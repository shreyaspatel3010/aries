from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch_ros.parameter_descriptions import ParameterValue

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    debug_arg = DeclareLaunchArgument(
        'debug', default_value='false', description='')
    load_robot_description_arg = DeclareLaunchArgument(
        'load_robot_description', default_value='false', description='')
    use_gui_arg = DeclareLaunchArgument(
        'use_gui', default_value='true', description='')
    use_joystick_arg = DeclareLaunchArgument(
        'use_joystick',
        default_value='true',
        description='Start joystick arm teleop')
    joy_layout_arg = DeclareLaunchArgument(
        'joy_layout',
        default_value='auto',
        choices=['auto', 'dongle', 'bluetooth', 'game_controller', 'passthrough'],
        description='Normalize joystick layout before teleop nodes consume /joy')
    joy_driver_arg = DeclareLaunchArgument(
        'joy_driver',
        default_value='game_controller_node',
        choices=['game_controller_node', 'joy_node'],
        description='Joystick driver executable from the joy package')
    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev',
        default_value='/dev/input/js0',
        description='Joystick device used by joy_node and the layout normalizer')
    joystick_control_mode_arg = DeclareLaunchArgument(
        'joystick_control_mode',
        default_value='servo',
        choices=['move_group', 'servo'],
        description='servo uses smooth Cartesian MoveIt Servo teleop with collision guard; move_group uses planned steps')
    gazebo_gui_arg = DeclareLaunchArgument(
        'gazebo_gui', default_value='true', description='Start Gazebo with GUI')
    paused_arg = DeclareLaunchArgument(
        'paused', default_value='false', description='')
    hardware_protocol_arg = DeclareLaunchArgument(
        "hardware_protocol",
        default_value="gazebo",
        choices=["mock_hardware", "gazebo", "rebel"],
        description="Which hardware protocol or mock hardware should be used",)
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use sim time if true')
    # Same omission that made `finger_type:=bucket` show probe fingers from
    # my_robot.launch.py: without these the xacro falls back to its own defaults
    # (finger_type=probe) and MoveIt plans against a fingertip the robot does not
    # have. The contact point differs by up to 23 mm between the three jaws.
    gripper_type_arg = DeclareLaunchArgument(
        'gripper_type',
        default_value='v2',
        choices=['old', 'new', 'v2'],
        description="Gripper type: 'v2', 'new' or 'old'")
    finger_type_arg = DeclareLaunchArgument(
        'finger_type',
        default_value='bucket',
        choices=['bucket', 'maintenance', 'probe'],
        description='Swappable fingertip; must match the mounted jaw')
    hardware_protocol = LaunchConfiguration('hardware_protocol')
    gripper_type = LaunchConfiguration('gripper_type')
    finger_type = LaunchConfiguration('finger_type')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')
    use_joystick = LaunchConfiguration('use_joystick')

    robot_description_file = os.path.join(
        get_package_share_directory('aries'),
        'urdf',
        'my_robot.urdf.xacro'
    )

    robot_description = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            robot_description_file,
            " hardware_protocol:=",
            hardware_protocol,
            " gripper_type:=",
            gripper_type,
            " finger_type:=",
            finger_type,
        ]
    )
    # Launch gazebo simulator and spwan the robot
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('aries_moveit'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'gazebo_gui': LaunchConfiguration('gazebo_gui'),
        }.items(),
    )
    
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
                'robot_description': ParameterValue(robot_description, value_type=str),
                'use_sim_time': use_sim_time
        }],
        remappings=[],
        output="both",
    )
    
    # Publish wheel joint states (not controlled by ros2_control)
    wheel_joint_publisher_node = Node(
        package='aries_moveit',
        executable='publish_wheel_joints.py',
        name='wheel_joint_publisher',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        output='screen',
    )
    
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('aries_moveit'), 'launch', 'move_group.launch.py')
        ),
        launch_arguments={
            'hardware_protocol': hardware_protocol,
            'use_sim_time': use_sim_time,
            'use_gui': use_gui,
            'use_joystick': use_joystick,
            'joy_driver': LaunchConfiguration('joy_driver'),
            'joy_layout': LaunchConfiguration('joy_layout'),
            'joy_dev': LaunchConfiguration('joy_dev'),
            'joystick_control_mode': LaunchConfiguration('joystick_control_mode'),
        }.items(),
    )
    
    return LaunchDescription([
        debug_arg,
        load_robot_description_arg,
        use_gui_arg,
        use_joystick_arg,
        joy_driver_arg,
        joy_layout_arg,
        joy_dev_arg,
        joystick_control_mode_arg,
        gazebo_gui_arg,
        paused_arg,
        hardware_protocol_arg,
        gripper_type_arg,
        finger_type_arg,
        use_sim_time_arg,
        gazebo_launch,
        robot_state_pub_node,
        wheel_joint_publisher_node,
        moveit_launch,
    ])
