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
        default_value='st3215',
        choices=['st3215'],
        description='Gripper type. Only "st3215" exists - the ST3215 rack-and-'
                    'pinion. v2 and the older four-bars are retired to '
                    'aries/urdf/legacy/ and are not built any more.'
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
        choices=['bucket', 'maintenance'],
        description='Swappable fingertip: "bucket" (scoops) or "maintenance" (flat jaws); must match what is bolted on'
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

    use_drill_teleop_arg = DeclareLaunchArgument(
        'use_drill_teleop',
        default_value='true',
        description='LT-gated drill teleop: feed carriage, sample bin and auger'
    )

    use_cmd_vel_relay_arg = DeclareLaunchArgument(
        'use_cmd_vel_relay',
        default_value='true',
        description='Relay joystick /cmd_vel/teleop commands to Gazebo /cmd_vel'
    )

    use_stacklight_arg = DeclareLaunchArgument(
        'use_stacklight', default_value='true',
        description=(
            'Run the stack light: the state node that decides the colour, and '
            'the viewer that lights the three tiers on the model to match. '
            'Sources are simulation ones - see aries/config/stacklight_sim.yaml.'
        ),
    )

    use_camera_downlink_arg = DeclareLaunchArgument(
        'use_camera_downlink', default_value='true',
        description=(
            'Run the operator camera downlink and its decompressors, the same '
            'chain the field stack runs. On by default because RViz reads the '
            'decompressed /<camera>/view/* topics and nothing else publishes '
            'them, so with this off the image panels are simply blank. Turn it '
            'off to give the JPEG/PNG codecs back to physics if the sim is '
            'running short of real time.'
        ),
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

    use_sim_teleop_speeds_arg = DeclareLaunchArgument(
        'use_sim_teleop_speeds',
        default_value='true',
        choices=['true', 'false'],
        description=(
            'Scale the joystick arm speeds up by ~1.85 to cancel Gazebo\'s '
            '~0.54 real-time factor, so a stick deflection moves the arm at the '
            'same wall-clock speed as on the rover. See '
            'aries_moveit/config/teleop_speeds_sim.yaml. Set false to run the '
            'sim on the shared hardware numbers in teleop_speeds.yaml.'
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
    use_stacklight = LaunchConfiguration('use_stacklight')
    use_camera_downlink = LaunchConfiguration('use_camera_downlink')
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

    # The drill's three axes reach gz through the JointController plugins in
    # aries_gazebo.xacro, bridged by config/gazebo_bridge.yaml - not through
    # ros2_control. All three take a rate, because all three are DC motors.
    # This node is the only thing that commands them, and only while LT is
    # held; it also owns their limit switches.
    drill_joystick_node = Node(
        condition=IfCondition(LaunchConfiguration('use_drill_teleop')),
        package='aries_teleop',
        executable='drill_joystick.py',
        name='drill_joystick',
        output='screen',
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare('aries_teleop'), 'config', 'joystick.yaml']),
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

    # Stack light. Two nodes, and the split is the point: stacklight.py decides
    # the colour from the rover's state and publishes the same UInt8 the Teensy
    # firmware subscribes to on hardware, and stacklight_gz_visual.py only
    # WATCHES that topic and turns the model's three cylinders up or down to
    # match. The simulated light is therefore driven by exactly the message the
    # real one is, and the two cannot disagree about what the rover is doing.
    #
    # Sources differ from the rover's, because the drive bridge does not run
    # here - /cmd_vel and /joint_states stand in for /aries_drive/status. That
    # is all stacklight_sim.yaml is.
    stacklight_config = PathJoinSubstitution(
        [FindPackageShare('aries'), 'config', 'stacklight_sim.yaml'])

    stacklight_node = Node(
        condition=IfCondition(use_stacklight),
        package='aries_bringup',
        executable='stacklight.py',
        name='stacklight',
        output='screen',
        parameters=[stacklight_config, {'use_sim_time': use_sim_time}],
    )

    stacklight_gz_visual_node = Node(
        condition=IfCondition(use_stacklight),
        package='aries_bringup',
        executable='stacklight_gz_visual.py',
        name='stacklight_gz_visual',
        output='screen',
        parameters=[stacklight_config, {'use_sim_time': use_sim_time}],
    )

    # Operator camera downlink, exactly as the field stack runs it:
    #
    #   /<cam>/color/image_raw                    (bridged from gz)
    #     -> camera_downlink.py                   reduce
    #     -> /downlink/<cam>/color/compressed     JPEG
    #     -> republish                            decompress
    #     -> /<cam>/view/color                    what RViz displays
    #
    # aries_moveit's moveit.rviz - the config this sim's RViz loads - has its
    # two Image displays wired to /<cam>/view/color, which only the last step
    # publishes. Without this chain those panels have no publisher at all,
    # which is why the simulation showed nothing on the camera side while the
    # bridged topics underneath were perfectly healthy.
    #
    # Running the real path rather than pointing RViz at the raw topics also
    # means the thing that carries every image in the field is exercised in
    # simulation instead of only on competition day.
    #
    # Included by share path, not declared as a package dependency: aries_bringup
    # already exec_depends on aries, and declaring the reverse would make a
    # dependency cycle colcon refuses to order. Same reason the aries_moveit and
    # aries_localization includes below are undeclared.
    camera_downlink_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('aries_bringup'),
                'launch',
                'camera_downlink.launch.py',
            ])
        ),
        condition=IfCondition(use_camera_downlink),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    camera_view_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('aries_bringup'),
                'launch',
                'camera_view.launch.py',
            ])
        ),
        condition=IfCondition(use_camera_downlink),
        # use_rviz stays false: move_group.launch.py owns the one RViz here,
        # and a second one would load a second copy of every display.
        launch_arguments={'use_rviz': 'false'}.items(),
    )

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
            # the xacro defaults and, because it publishes its own
            # robot_description, overwrite the correct model on that topic.
            'gripper_type': gripper_type,
            'finger_type': finger_type,
            'use_sim_time': use_sim_time,
            'use_gui': 'true',
            'use_joystick': use_joystick,
            'joy_driver': joy_driver,
            'joy_layout': joy_layout,
            'joy_dev': joy_dev,
            'joystick_control_mode': joystick_control_mode,
            'use_sim_teleop_speeds': LaunchConfiguration('use_sim_teleop_speeds'),
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
        use_drill_teleop_arg,
        use_cmd_vel_relay_arg,
        use_stacklight_arg,
        use_camera_downlink_arg,
        use_sim_ekf_arg,
        joy_driver_arg,
        joy_layout_arg,
        joy_dev_arg,
        joystick_control_mode_arg,
        use_sim_teleop_speeds_arg,
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
        drill_joystick_node,
        cmd_vel_relay_node,
        localization_launch,
        stacklight_node,
        stacklight_gz_visual_node,
        camera_downlink_launch,
        camera_view_launch,
        move_group_launch,
        delay_controllers_after_spawn,  # Controllers spawn after robot spawns
    ])
