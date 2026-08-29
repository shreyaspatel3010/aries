import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def opaque_func(context, *args, **kwargs):
    
    namespace = LaunchConfiguration("namespace")
    hardware_protocol = LaunchConfiguration('hardware_protocol')
    # The fingertip and gripper MUST reach the xacro below. Without them this
    # launch built its robot_description from the xacro DEFAULTS
    # (finger_type=probe), while my_robot.launch.py built Gazebo's from the
    # requested finger. Because move_group runs with publish_robot_description,
    # its model then overwrote the correct one on /robot_description -- so
    # `finger_type:=bucket` showed probe fingers in RViz and, worse, MoveIt
    # planned and collision-checked against a fingertip the robot did not have
    # (the contact point differs by up to 23 mm between the three jaws).
    gripper_type = LaunchConfiguration('gripper_type')
    finger_type = LaunchConfiguration('finger_type')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_joystick = LaunchConfiguration("use_joystick")
    joy_driver = LaunchConfiguration("joy_driver")
    joy_layout = LaunchConfiguration("joy_layout")
    joy_dev = LaunchConfiguration("joy_dev")
    joystick_control_mode = LaunchConfiguration("joystick_control_mode")
    servo_joystick_condition = IfCondition(PythonExpression([
        "'", use_joystick, "' == 'true' and '", joystick_control_mode, "' == 'servo'"
    ]))
    move_group_joystick_condition = IfCondition(PythonExpression([
        "'", use_joystick, "' == 'true' and '", joystick_control_mode, "' == 'move_group'"
    ]))

    joint_limits_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "joint_limits.yaml",
        ]
    )

    robot_description_file = PathJoinSubstitution(
        [
            FindPackageShare("aries"),
            "urdf",
            "my_robot.urdf.xacro",
        ]
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
    
    robot_description_semantic_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "aries.srdf",
        ]
    )
    
    # aries.srdf is XACRO, not plain text: it carries both grippers' link sets
    # and gates them on gripper_type. `cat` here yields a semantic model with
    # the other gripper's links in it, which srdfdom reports as Errors, and
    # with the xacro tags left in as unknown elements.
    robot_description_semantic = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            robot_description_semantic_file,
            " gripper_type:=",
            gripper_type,
        ]
    )

    controllers_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "moveit_controllers.yaml",   
        ]
    )

    controllers_dict = load_yaml(Path(controllers_file.perform(context)))

    ompl_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "ompl_planning.yaml",
        ]
    )
    ompl_config = load_yaml(Path(ompl_file.perform(context)))
    # Handle optional 'move_group:' top-level wrapper in ompl_planning.yaml
    if 'move_group' in ompl_config:
        move_group_block = ompl_config.pop('move_group')
        ompl_config.update(move_group_block)
    ompl_planning_yaml = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_config,
    }

    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_dict,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    robot_description_kinematics_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "kinematics.yaml",
        ]
    )

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    
    kinematics_config = load_yaml(Path(robot_description_kinematics_file.perform(context)))
    joint_limits_config = load_yaml(Path(joint_limits_file.perform(context)))

    moveit_args_not_concatenated = [
        {"robot_description": ParameterValue(robot_description.perform(context), value_type=str)},
        {"robot_description_semantic": ParameterValue(robot_description_semantic.perform(context), value_type=str)},
        {"robot_description_kinematics": kinematics_config},
        {"robot_description_planning": joint_limits_config},
        moveit_controllers,
        planning_scene_monitor_parameters,
        {
            "publish_robot_description": True,
            "publish_robot_description_semantic": True,
            "publish_geometry_updates": True,
            "publish_state_updates": True,
            "publish_transforms_updates": True,
        },
        ompl_planning_yaml,
    ]

    # Concatenate all dictionaries together, else moveitpy won't read all parameters
    moveit_args = dict()
    for d in moveit_args_not_concatenated:
        moveit_args.update(d)
                        
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            moveit_args,
        ],
    )

    # Get parameters for the Servo node
    servo_params_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "servo.yaml",
        ]
    )
    servo_context = load_yaml(Path(servo_params_file.perform(context)))
    servo_params = {
        "moveit_servo": servo_context
    }
    # This sets the update rate and planning group name for the acceleration limiting filter.
    planning_group_name = {"planning_group_name": "igus_rebel_arm"}

    servo_node = Node(
        condition=servo_joystick_condition,
        package="moveit_servo",
        executable="servo_node",
        namespace=namespace,
        parameters=[
            {'use_sim_time': use_sim_time},
            servo_params,
            planning_group_name,
            moveit_args,
        ],
        output="screen",
    )

    servo_collision_guard_node = Node(
        condition=servo_joystick_condition,
        package="aries_moveit",
        executable="servo_collision_guard",
        namespace=namespace,
        name="servo_collision_guard",
        parameters=[
            {'use_sim_time': use_sim_time},
            moveit_args,
            {
                "input_topic": "servo_guard/input_joint_trajectory",
                "output_topic": "rebel_arm_trajectory_controller/joint_trajectory",
                "joint_state_topic": "joint_states",
                "status_topic": "/arm_joystick/status",
                "group_name": "arm_with_gripper",
                "min_self_distance": 0.015,
                "distance_tolerance": 0.001,
                # Keep joystick collision response identical to the real-arm
                # launch. These values affect how quickly a command is passed
                # or held near a collision.
                "interpolation_steps": 1,
                "hold_time": 0.02,
            },
        ],
        output="screen",
    )

    # Launch gamepad
    joy_node = Node(
        condition=IfCondition(use_joystick),
        package="joy",
        executable=joy_driver,
        namespace=namespace,
        name="joy_node",
        parameters=[
            {'use_sim_time': use_sim_time},
            {
                # Match aries_hardware.launch.py exactly so button edges,
                # trigger thresholds and commanded update timing behave the
                # same in simulation and on the real robot.
                "dev": joy_dev,
                "autorepeat_rate": 80.0,
                "deadzone": 0.0,
                "coalesce_interval_ms": 1,
            },
        ],
        remappings=[("joy", "joy/raw")],
        output="screen",
    )

    joy_layout_normalizer_node = Node(
        condition=IfCondition(use_joystick),
        package="aries_moveit",
        executable="joy_layout_normalizer.py",
        namespace=namespace,
        name="joy_layout_normalizer",
        parameters=[{
            'use_sim_time': use_sim_time,
            "input_topic": "joy/raw",
            "output_topic": "joy",
            "layout": joy_layout,
            "device": joy_dev,
        }],
        output="screen",
    )
    
    teleop_joy_twist_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "gamepad.yaml",
        ]
    )
    # Speeds live in teleop_speeds.yaml and are loaded last so they win over the
    # copies still in gamepad.yaml. gamepad.yaml keeps the button/axis mapping.
    teleop_speeds_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "teleop_speeds.yaml",
        ]
    )
    # Simulation-only overlay, loaded after teleop_speeds.yaml so its keys win.
    # Gazebo runs at RTF ~0.54 here, so identical commands look ~1.85x slower on
    # screen than on the rover; this multiplies the scales back up so a given
    # stick deflection moves the arm at real-arm WALL-CLOCK speed.
    #
    # The condition is what keeps the real arm out of it. This launch file is
    # shared: igus_rebel_hardware.launch.py includes it with
    # hardware_protocol:='rebel', which fails the check below, and the rover's
    # own aries_hardware.launch.py does not include this file at all. So the
    # overlay can only reach a Gazebo or mock-hardware session.
    teleop_speeds_sim_file = PathJoinSubstitution(
        [
            FindPackageShare("aries_moveit"),
            "config",
            "teleop_speeds_sim.yaml",
        ]
    )

    # We are inside an OpaqueFunction, so the launch configurations already have
    # concrete values and the overlay can simply be appended (or not) to the
    # parameter list. This is what keeps the real arm out of it: this launch file
    # is shared, and igus_rebel_hardware.launch.py includes it with
    # hardware_protocol:='rebel', which fails the check below. The rover's own
    # aries_hardware.launch.py does not include this file at all.
    use_sim_speeds = (
        LaunchConfiguration("use_sim_teleop_speeds").perform(context) == "true"
        and hardware_protocol.perform(context) in ("gazebo", "mock_hardware")
    )
    teleop_speed_params = [teleop_speeds_file]
    if use_sim_speeds:
        teleop_speed_params.append(teleop_speeds_sim_file)
    # Per-gripper overlay, appended last so its keys win over both of the
    # above. See teleop_speeds_st3215.yaml for why only that gripper has one.
    if gripper_type.perform(context) == "st3215":
        teleop_speed_params.append(PathJoinSubstitution(
            [FindPackageShare("aries_moveit"), "config", "teleop_speeds_st3215.yaml"]))
    teleop_twist_joy_node = Node(
        condition=servo_joystick_condition,
        package="aries_moveit",
        executable="rebel_servo_teleop_gamepad",
        namespace=namespace,
        name="rebel_servo_teleop_gamepad",
        parameters=[{'use_sim_time': use_sim_time}, teleop_joy_twist_file,
                    *teleop_speed_params],
        output="screen",
    )

    move_group_joystick_node = Node(
        condition=move_group_joystick_condition,
        package="aries_moveit",
        executable="rebel_movegroup_joystick.py",
        namespace=namespace,
        name="rebel_movegroup_joystick",
        parameters=[{'use_sim_time': use_sim_time}, teleop_joy_twist_file,
                    *teleop_speed_params],
        output="screen",
    )

    # Same collision-planned, direct-controller presets as full_hardware.launch.py:
    # LT+Y -> pick_home, LT+A -> probe_drop, LT+B -> soil_drop.
    arm_preset_pose_node = Node(
        condition=IfCondition(use_joystick),
        package="aries_moveit",
        executable="arm_preset_pose_joystick.py",
        namespace=namespace,
        name="arm_preset_pose_joystick",
        parameters=[
            {'use_sim_time': use_sim_time},
            teleop_joy_twist_file,
            *teleop_speed_params,
        ],
        output="screen",
    )

    # Gripper arc overlay for RViz: jaw open/close sweep + point of closing.
    # It carries its own geometry, so it has to be told which gripper is loaded
    # or it draws the four-bar's curve over a mechanism whose jaws travel in a
    # straight line.
    gripper_arc_visualizer_node = Node(
        package="aries_moveit",
        executable="gripper_arc_visualizer.py",
        namespace=namespace,
        name="gripper_arc_visualizer",
        parameters=[{'use_sim_time': use_sim_time,
                     'gripper_type': gripper_type}],
        output="screen",
    )

    default_rviz_file = os.path.join(
        get_package_share_directory('aries_moveit'),
        'launch',
        'moveit.rviz'
    )
    
    rviz_parameters = [
        {
            'use_sim_time': use_sim_time,
            'robot_description_kinematics': kinematics_config,
            'robot_description_planning': joint_limits_config,
            'robot_description': ParameterValue(robot_description, value_type=str),
            'robot_description_semantic': ParameterValue(robot_description_semantic, value_type=str),
        },
    ]

    launch_rviz = Node(
        condition=IfCondition(LaunchConfiguration("use_gui")),
        package="rviz2",
        executable="rviz2",
        output={"both": "log"},
        arguments=["-d", default_rviz_file],
        parameters=rviz_parameters,
    )
    
    entities = [
        move_group_node,
        servo_node,
        servo_collision_guard_node,
        joy_node,
        joy_layout_normalizer_node,
        teleop_twist_joy_node,
        move_group_joystick_node,
        arm_preset_pose_node,
        gripper_arc_visualizer_node,
        launch_rviz
    ]
    return entities


def generate_launch_description():
    namespace_arg = DeclareLaunchArgument("namespace", default_value="")
    prefix_arg = DeclareLaunchArgument("prefix", default_value="")
    use_gui_arg = DeclareLaunchArgument("use_gui", default_value="true")
    use_joystick_arg = DeclareLaunchArgument(
        "use_joystick",
        default_value="true",
        description="Start joy_node and joystick arm teleop",
    )
    joy_layout_arg = DeclareLaunchArgument(
        "joy_layout",
        default_value="auto",
        choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"],
        description="Normalize joystick layout before teleop nodes consume /joy",
    )
    joy_driver_arg = DeclareLaunchArgument(
        "joy_driver",
        default_value="game_controller_node",
        choices=["game_controller_node", "joy_node"],
        description="Joystick driver executable from the joy package",
    )
    joy_dev_arg = DeclareLaunchArgument(
        "joy_dev",
        default_value="/dev/input/js0",
        description="Joystick device used by joy_node and the layout normalizer",
    )
    joystick_control_mode_arg = DeclareLaunchArgument(
        "joystick_control_mode",
        default_value="servo",
        choices=["move_group", "servo"],
        description="servo uses smooth Cartesian MoveIt Servo teleop with collision guard; move_group uses planned steps",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use sim time if true')

    use_sim_teleop_speeds_arg = DeclareLaunchArgument(
        "use_sim_teleop_speeds",
        default_value="true",
        choices=["true", "false"],
        description="Apply config/teleop_speeds_sim.yaml on top of "
                    "config/teleop_speeds.yaml. Ignored unless hardware_protocol "
                    "is gazebo or mock_hardware, so it can never change the real "
                    "arm. The overlay multiplies the joystick scales by ~1.85 to "
                    "cancel Gazebo's ~0.54 real-time factor, so a stick "
                    "deflection moves the arm at real-arm wall-clock speed. Set "
                    "false to run the sim on the shared hardware numbers.",
    )

    hardware_protocol_arg = DeclareLaunchArgument(
        "hardware_protocol",
        default_value="rebel",
        choices=["mock_hardware", "gazebo", "rebel"],
        description="Which hardware protocol or mock hardware should be used",
    )

    # Defaults match my_robot.urdf.xacro's own gripper default and the fingertip
    # the rover actually carries, so a bare `ros2 launch move_group.launch.py`
    # builds the bucket rather than silently falling back to the probe.
    gripper_type_arg = DeclareLaunchArgument(
        "gripper_type",
        default_value="v2",
        choices=["v2", "st3215"],
        description="Gripper type. Only 'v2' exists; 'new' and 'old' are retired "
                    "Gazebo and robot_state_publisher were launched with.",
    )
    finger_type_arg = DeclareLaunchArgument(
        "finger_type",
        default_value="bucket",
        choices=["bucket", "maintenance", "probe"],
        description="Swappable fingertip. MUST match the finger the rest of the "
                    "stack was launched with: MoveIt plans and collision-checks "
                    "with this, and the four-bar contact point differs by up to "
                    "23 mm between the three jaws.",
    )

    ld = LaunchDescription()
    ld.add_action(use_sim_time_arg)
    ld.add_action(use_sim_teleop_speeds_arg)
    ld.add_action(gripper_type_arg)
    ld.add_action(finger_type_arg)
    ld.add_action(namespace_arg)
    ld.add_action(use_gui_arg)
    ld.add_action(use_joystick_arg)
    ld.add_action(joy_driver_arg)
    ld.add_action(joy_layout_arg)
    ld.add_action(joy_dev_arg)
    ld.add_action(joystick_control_mode_arg)
    ld.add_action(hardware_protocol_arg)

    ld.add_action(OpaqueFunction(function=opaque_func))

    return ld
