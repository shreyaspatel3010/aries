import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package directories
    aries_share = get_package_share_directory('aries')
    # Paths
    urdf_path = PathJoinSubstitution(
        [FindPackageShare('aries'), 'urdf', 'my_robot.urdf.xacro']
    )
    gazebo_config_path = PathJoinSubstitution(
        [FindPackageShare('aries'), 'config', 'gazebo_bridge.yaml']
    )
    gz_gui_config_path = PathJoinSubstitution(
        [FindPackageShare('aries'), 'config', 'gz_gui_config.config']
    )
    # Selectable so the soil-sampling world can be launched without editing this
    # file: world:=soil_world.sdf. Both worlds are named 'empty' internally
    # because gazebo_bridge.yaml hardcodes /world/empty/* topics.
    world_path = PathJoinSubstitution(
        [FindPackageShare('aries'), 'worlds', LaunchConfiguration('world')]
    )
    virtual_diff_config_path = PathJoinSubstitution(
        [FindPackageShare('aries'), 'config', 'virtual_differential.yaml']
    )
    rover_joystick_config_path = PathJoinSubstitution(
        [
            FindPackageShare('aries_teleop'),
            'config',
            'rover_cmd_vel_joystick.yaml',
        ]
    )

    # Declare arguments
    gripper_type_arg = DeclareLaunchArgument(
        'gripper_type',
        default_value='v2',
        choices=['old', 'new', 'v2'],
        description='Gripper type: "v2", "new" or "old"'
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value='maintenance_world.sdf',
        choices=['sandbox_world.sdf', 'soil_world.sdf', 'marsyard2026.sdf', 'maintenance_world.sdf'],
        description='World file in aries/worlds. sandbox_world.sdf has the '
                    'planted probe; soil_world.sdf replaces it with a bed of '
                    'loose soil grains and a deposit box for bucket sampling.'
    )

    finger_type_arg = DeclareLaunchArgument(
        'finger_type',
        default_value='bucket',
        choices=['bucket', 'maintenance', 'probe'],
        description='Swappable fingertip: "bucket", "maintenance", or "probe"'
    )

    hardware_protocol_arg = DeclareLaunchArgument(
        'hardware_protocol',
        default_value='gazebo',
        description='Hardware protocol for ros2_control'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run only the Gazebo server (useful for automated world tests)'
    )

    use_joystick_arg = DeclareLaunchArgument(
        'use_joystick',
        default_value='true',
        description='Start the shared joystick driver and arm/gripper teleop'
    )

    use_rover_joystick_arg = DeclareLaunchArgument(
        'use_rover_joystick',
        default_value='true',
        description='Use the hardware joystick mapping for simulated rover drive'
    )

    use_cmd_vel_relay_arg = DeclareLaunchArgument(
        'use_cmd_vel_relay',
        default_value='true',
        description='Relay joystick /cmd_vel/teleop commands to Gazebo /cmd_vel'
    )

    use_sim_ekf_arg = DeclareLaunchArgument(
        'use_sim_ekf',
        default_value='true',
        description=(
            'Fuse Gazebo ground-truth odometry and IMU and publish the '
            'odom to base_footprint transform'
        )
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
        description=(
            'servo uses smooth Cartesian MoveIt Servo teleop with collision guard; '
            'move_group uses planned steps'
        )
    )

    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x',
        default_value='0.0',
        description='Spawn position X coordinate'
    )

    spawn_y_arg = DeclareLaunchArgument(
        'spawn_y',
        default_value='0.0',
        description='Spawn position Y coordinate'
    )

    spawn_z_arg = DeclareLaunchArgument(
        'spawn_z',
        default_value='0.1',
        description='Spawn position Z coordinate above the world origin'
    )

    spawn_yaw_arg = DeclareLaunchArgument(
        'spawn_yaw',
        default_value='0.0',
        description='Spawn heading in radians (use 1.5708 to face into MarsYard from S1)'
    )

    # Launch configurations
    gripper_type = LaunchConfiguration('gripper_type')
    finger_type = LaunchConfiguration('finger_type')
    hardware_protocol = LaunchConfiguration('hardware_protocol')
    use_sim_time = LaunchConfiguration('use_sim_time')
    headless = LaunchConfiguration('headless')
    use_joystick = LaunchConfiguration('use_joystick')
    use_rover_joystick = LaunchConfiguration('use_rover_joystick')
    use_cmd_vel_relay = LaunchConfiguration('use_cmd_vel_relay')
    use_sim_ekf = LaunchConfiguration('use_sim_ekf')
    joy_driver = LaunchConfiguration('joy_driver')
    joy_layout = LaunchConfiguration('joy_layout')
    joy_dev = LaunchConfiguration('joy_dev')
    joystick_control_mode = LaunchConfiguration('joystick_control_mode')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    # Set environment variable for Gazebo Sim resource path
    # Add aries package, models, worlds, and existing path
    gz_resource_path = EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value='')
    set_gz_sim_resource_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        [
            os.path.dirname(aries_share),
            ':',
            aries_share,
            '/models:',
            aries_share,
            '/worlds:',
            gz_resource_path,
        ]
    )

    # Start Gazebo (server + GUI) with your world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': [
            world_path,
            ' -r',
            PythonExpression(["' -s' if '", headless, "' == 'true' else ''"]),
        ]}.items()
    )

    # Robot state publisher
    robot_description_content = ParameterValue(
        Command([
            'xacro ', urdf_path,
            ' gripper_type:=', gripper_type,
            ' finger_type:=', finger_type,
            ' hardware_protocol:=', hardware_protocol
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time
        }]
    )

    # Spawn robot from robot_description
    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        name='create',
        output='screen',
        arguments=[
            '-name', 'aries',
            '--gui-config', gz_gui_config_path,
            '-topic', 'robot_description',
            '-allow_renaming', 'true',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-Y', spawn_yaw
        ]
    )

    # Bridge topics using YAML config
    parameter_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file': gazebo_config_path,
            'use_sim_time': use_sim_time
        }]
    )

    # Virtual differential node
    virtual_differential_node = Node(
        package='aries',
        executable='virtual_differential',
        name='virtual_differential',
        output='screen',
        parameters=[
            virtual_diff_config_path,
            {'use_sim_time': use_sim_time}
        ]
    )

    # Reuse the real robot's rover teleop node and canonical mapping. The joy
    # driver/layout normalizer is already owned by move_group_launch below, so
    # starting another joy node here would duplicate /joy publishers.
    rover_joystick_node = Node(
        condition=IfCondition(use_rover_joystick),
        package='aries_teleop',
        executable='rover_cmd_vel_joystick.py',
        name='rover_cmd_vel_joystick',
        output='screen',
        parameters=[
            rover_joystick_config_path,
            {'use_sim_time': use_sim_time},
        ],
    )

    cmd_vel_relay_node = Node(
        condition=IfCondition(use_cmd_vel_relay),
        package='aries_teleop',
        executable='cmd_vel_teleop_relay.py',
        name='cmd_vel_teleop_relay',
        output='screen',
        parameters=[{
            'input_topic': '/cmd_vel/teleop',
            'output_topic': '/cmd_vel',
            'use_sim_time': use_sim_time,
        }],
    )

    # Gazebo publishes both odometry measurements, but its wheel-integrated TF
    # is deliberately not bridged. The simulation EKF is therefore the single
    # owner of odom -> base_footprint, matching the localization wrapper and
    # preventing RViz from reporting that the odom frame does not exist.
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('aries_localization'),
                'launch',
                'localization.launch.py',
            ])
        ),
        condition=IfCondition(use_sim_ekf),
        launch_arguments={
            'use_sim_ekf': 'true',
            'use_sim_time': use_sim_time,
            'sim_odom_topic': '/ground_truth/odom',
            'sim_imu_topic': '/imu',
            'filtered_odom_topic': '/odometry/filtered',
        }.items(),
    )

    # Rover wheel/rocker joint states come from the Gazebo JointStatePublisher
    # plugin via the gz bridge (real physics values, sim-time stamps). Do not
    # run publish_wheel_joints.py here: its zero-value, potentially wall-clock
    # stamped messages corrupt TF for the moving wrist camera.

    # MoveIt move_group for arm control
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('aries_moveit'), 'launch', 'move_group.launch.py']
            )
        ),
        launch_arguments={
            'hardware_protocol': hardware_protocol,
            # Forwarded so MoveIt builds the SAME robot as Gazebo and
            # robot_state_publisher. Omitting these let move_group fall back to
            # the xacro default (finger_type=probe) and, because it publishes its
            # own robot_description, overwrite the correct model on that topic.
            'gripper_type': gripper_type,
            'finger_type': finger_type,
            'use_sim_time': use_sim_time,
            'use_gui': 'true',
            'use_joystick': use_joystick,
            'joy_driver': joy_driver,
            'joy_layout': joy_layout,
            'joy_dev': joy_dev,
            'joystick_control_mode': joystick_control_mode,
        }.items()
    )

    # Spawn ros2_control controllers (delayed until robot is spawned)
    ros_controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('aries_moveit'), 'launch', 'ros_controllers.launch.py']
            )
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Delay controller spawning until robot is spawned in Gazebo
    delay_controllers_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot_node,
            on_exit=[ros_controllers_launch],
        )
    )

    return LaunchDescription([
        # Arguments
        gripper_type_arg,
        world_arg,
        finger_type_arg,
        hardware_protocol_arg,
        use_sim_time_arg,
        headless_arg,
        use_joystick_arg,
        use_rover_joystick_arg,
        use_cmd_vel_relay_arg,
        use_sim_ekf_arg,
        joy_driver_arg,
        joy_layout_arg,
        joy_dev_arg,
        joystick_control_mode_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,

        # Environment
        set_gz_sim_resource_path,

        # Launches and nodes
        gazebo_launch,
        robot_state_publisher_node,
        spawn_robot_node,
        parameter_bridge_node,
        virtual_differential_node,
        rover_joystick_node,
        cmd_vel_relay_node,
        localization_launch,
        move_group_launch,
        delay_controllers_after_spawn,  # Controllers spawn after robot spawns
    ])
