"""
aries_hardware.launch.py

Mixed real/fake hardware launch for Aries:
- arm: auto-detect igus ReBeL, otherwise mock_hardware
- gripper: auto-detect Teensy serial device, otherwise mock_hardware

Important detail:
ros2_control controller definitions are written to a temporary YAML file,
because controller_manager expects the multi-node YAML layout
(controller_manager + per-controller sections).
"""

import os
import socket
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["gripper_gear_left_joint"]


def build_ros2_control_yaml(arm_protocol: str, gripper_protocol: str) -> str:
    arm_command_interface = "velocity" if arm_protocol == "rebel" else "position"

    data = {
        "controller_manager": {
            "ros__parameters": {
                # 50 Hz for real rebel arm: each velocity command is held for 20 ms,
                # so the 100 Hz ALIVEJOG sends it twice before the next one arrives.
                # This naturally smooths the commanded velocity and reduces jitter.
                "update_rate": 80,
                "joint_state_broadcaster": {
                    "type": "joint_state_broadcaster/JointStateBroadcaster"
                },
                "rebel_arm_trajectory_controller": {
                    "type": "joint_trajectory_controller/JointTrajectoryController"
                },
            }
        },
        "joint_state_broadcaster": {
            "ros__parameters": {}
        },
        "rebel_arm_trajectory_controller": {
            "ros__parameters": {
                "joints": ARM_JOINTS,
                "command_interfaces": [arm_command_interface],
                "state_interfaces": ["position", "velocity"],
                "state_publish_rate": 80.0,
                "action_monitor_rate": 40.0,
                "allow_nonzero_velocity_at_trajectory_end": True,
            }
        },
    }

    if arm_protocol == "rebel":
        # JOINT_VELOCITY_SCALE=2.0 means effective max vel_cmd ≈ 0.87 rad/s.
        # Old p=10 saturated at only ~5° error → overshoot → jitter.
        # Halving P and D prevents saturation while ff_velocity_scale=1.0
        # keeps trajectory feed-forward as the primary motion driver.
        data["rebel_arm_trajectory_controller"]["ros__parameters"]["gains"] = {
            # d=1.0 damps position-error corrections so the arm does not oscillate
            # when the Servo velocity stream stops (joystick centered).  ff_velocity_scale=1.0
            # keeps trajectory feed-forward as the primary motion driver so planned
            # RViz trajectories are unaffected.
            "joint1": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint2": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint3": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint4": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint5": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
            "joint6": {"p": 1.0, "d": 0.1, "i": 0.0, "i_clamp": 0.0, "ff_velocity_scale": 1.0},
        }

    if gripper_protocol in ("rebel", "mock_hardware", "gazebo"):
        data["controller_manager"]["ros__parameters"]["rebel_gripper_controller"] = {
            "type": "joint_trajectory_controller/JointTrajectoryController"
        }
        data["rebel_gripper_controller"] = {
            "ros__parameters": {
                "joints": GRIPPER_JOINTS,
                "command_interfaces": ["position"],
                "state_interfaces": ["position", "velocity"],
                "state_publish_rate": 80.0,
                "action_monitor_rate": 40.0,
                # The Teensy uses echo-mode feedback (state = cmd, no real sensor).
                # open_loop_control makes the JTC use its own last command as the
                # reference state rather than the echoed measurement.  Without this
                # the JTC can enter a correction loop that fights the servo: it sees
                # state ≈ cmd(t-1) and computes an extra correction that arrives at
                # the Teensy BEFORE the servo has physically moved, causing the servo
                # to overshoot, reverse, and overshoot again — the visible
                # close → open → close symptom.
                "open_loop_control": True,
                # The vision grasp node owns a bounded feedback/contact watchdog
                # and cancels explicitly. Keep the JTC deadline outside that
                # window so rigid contact cannot abort first.
                "constraints": {
                    "stopped_velocity_tolerance": 0.01,
                    "goal_time": 30.0,
                    "gripper_gear_left_joint": {
                        "trajectory": 0.05,
                        "goal": 0.01,
                    },
                },
            }
        }

    tmp = tempfile.NamedTemporaryFile(mode="w", prefix="aries_ros2_control_", suffix=".yaml", delete=False)
    yaml.safe_dump(data, tmp, sort_keys=False)
    tmp.flush()
    tmp.close()
    return tmp.name


def build_moveit_controller_config(include_gripper: bool):
    controllers = {
        # NOTE: trajectory_execution settings must be top-level move_group params,
        # NOT inside this dict.  See moveit_args in launch_setup() below.
        "controller_names": ["rebel_arm_trajectory_controller"],
        "rebel_arm_trajectory_controller": {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": True,
            "joints": ARM_JOINTS,
        },
    }

    if include_gripper:
        controllers["controller_names"].append("rebel_gripper_controller")
        controllers["rebel_gripper_controller"] = {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": False,
            "joints": GRIPPER_JOINTS,
        }

    return controllers


def build_ros_log_arguments(log_levels):
    if not log_levels:
        return []

    arguments = ["--ros-args"]
    for logger_name, level in log_levels:
        arguments.extend(["--log-level", f"{logger_name}:={level}"])
    return arguments


def launch_setup(context, *args, **kwargs):
    use_sim_time = False
    hardware_protocol = LaunchConfiguration("hardware_protocol").perform(context)
    arm_hardware_protocol = LaunchConfiguration("arm_hardware_protocol").perform(context)
    gripper_hardware_protocol = LaunchConfiguration("gripper_hardware_protocol").perform(context)
    gripper_type = LaunchConfiguration("gripper_type").perform(context)
    finger_type = LaunchConfiguration("finger_type").perform(context)
    serial_port = LaunchConfiguration("serial_port").perform(context)
    use_gui = LaunchConfiguration("use_gui").perform(context)
    suppress_rebel_logs = LaunchConfiguration("suppress_rebel_logs").perform(context).lower() in ("1", "true", "yes", "on")
    suppress_moveit_execution_logs = LaunchConfiguration("suppress_moveit_execution_logs").perform(context).lower() in ("1", "true", "yes", "on")
    enable_depth_sensor = LaunchConfiguration("enable_depth_sensor").perform(context).lower() in ("1", "true", "yes", "on")

    if arm_hardware_protocol == "auto":
        try:
            with socket.create_connection(("192.168.3.11", 3920), timeout=0.25):
                arm_hardware_protocol = "rebel"
        except OSError:
            arm_hardware_protocol = "mock_hardware"

    if gripper_hardware_protocol == "auto":
        gripper_hardware_protocol = "rebel" if Path(serial_port).exists() else "mock_hardware"

    urdf_file = PathJoinSubstitution([FindPackageShare("aries"), "urdf", "my_robot.urdf.xacro"])
    robot_description_raw = Command(
        [
            FindExecutable(name="xacro"),
            " ", urdf_file,
            " hardware_protocol:=", hardware_protocol,
            " arm_hardware_protocol:=", arm_hardware_protocol,
            " gripper_hardware_protocol:=", gripper_hardware_protocol,
            " gripper_type:=", gripper_type,
            " finger_type:=", finger_type,
            " serial_port:=", serial_port,
        ]
    ).perform(context)
    robot_description = ParameterValue(robot_description_raw, value_type=str)

    srdf_file = os.path.join(get_package_share_directory("aries_moveit"), "config", "aries.srdf")
    robot_description_semantic = ParameterValue(Command(["cat ", srdf_file]), value_type=str)

    kinematics_file = PathJoinSubstitution([FindPackageShare("aries_moveit"), "config", "kinematics.yaml"])
    joint_limits_file = PathJoinSubstitution([FindPackageShare("aries_moveit"), "config", "joint_limits.yaml"])
    ompl_file = PathJoinSubstitution([FindPackageShare("aries_moveit"), "config", "ompl_planning.yaml"])

    kinematics_config = load_yaml(Path(kinematics_file.perform(context)))
    joint_limits_config = load_yaml(Path(joint_limits_file.perform(context)))
    ompl_config = load_yaml(Path(ompl_file.perform(context)))

    if enable_depth_sensor:
        sensors_3d_file = PathJoinSubstitution(
            [FindPackageShare("aries_moveit"), "config", "sensors_3d.yaml"]
        )
        sensor_config = load_yaml(Path(sensors_3d_file.perform(context)))
    else:
        # Do not pass an empty array through a launch parameter dictionary.
        # launch_ros normalizes [] to (), whose element type cannot be inferred.
        # Omitting the sensor plugin parameter disables 3D sensor integration.
        sensor_config = {}

    if "move_group" in ompl_config:
        ompl_config.update(ompl_config.pop("move_group"))

    ompl_planning_yaml = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_config,
    }

    ros2_control_yaml = build_ros2_control_yaml(arm_hardware_protocol, gripper_hardware_protocol)
    controllers_dict = build_moveit_controller_config(
        gripper_hardware_protocol in ("rebel", "mock_hardware", "gazebo")
    )
    ros2_control_log_levels = []
    if suppress_rebel_logs and arm_hardware_protocol == "rebel":
        ros2_control_log_levels.append(("igus_rebel", "fatal"))
    if suppress_moveit_execution_logs:
        ros2_control_log_levels.extend([
            ("rebel_arm_trajectory_controller", "warn"),
            ("rebel_gripper_controller", "warn"),
        ])
    ros2_control_arguments = build_ros_log_arguments(ros2_control_log_levels)

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        arguments=ros2_control_arguments,
        parameters=[
            {"robot_description": robot_description_raw},
            ros2_control_yaml,
            {"use_sim_time": use_sim_time},
        ],
        output="both",
    )

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
        ],
        output="both",
    )

    wheel_joint_publisher_node = Node(
        package="aries_moveit",
        executable="publish_wheel_joints.py",
        name="wheel_joint_publisher",
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--param-file", ros2_control_yaml,
            "--switch-timeout", "30",
        ],
        output="both",
    )

    arm_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "rebel_arm_trajectory_controller",
            "--controller-manager", "/controller_manager",
            "--param-file", ros2_control_yaml,
            "--switch-timeout", "30",
        ],
        output="both",
    )

    hand_guiding_node = None
    if arm_hardware_protocol == "rebel":
        hand_guiding_node = Node(
            package="igus_rebel",
            executable="rebel_hand_guiding.py",
            name="rebel_hand_guiding",
            output="screen",
        )

    micro_ros_agent = None
    if gripper_hardware_protocol == "rebel":
        _fastdds_xml = os.path.join(
            get_package_share_directory("aries_moveit"), "config", "fastdds_low_latency.xml"
        )
        micro_ros_agent = ExecuteProcess(
            cmd=[
                "ros2", "run", "micro_ros_agent", "micro_ros_agent",
                "serial", "--dev", serial_port, "-b", "6000000",
            ],
            additional_env={
                # Unicast-only participant + synchronous publish mode + zero latency budget.
                # Eliminates multicast discovery overhead and DDS write-thread buffering,
                # cutting the agent ↔ ROS2 round-trip by several milliseconds.
                "FASTRTPS_DEFAULT_PROFILES_FILE": _fastdds_xml,
            },
            output="screen",
        )

    gripper_controller_spawner = None
    if gripper_hardware_protocol in ("rebel", "mock_hardware", "gazebo"):
        gripper_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "rebel_gripper_controller",
                "--controller-manager", "/controller_manager",
                "--param-file", ros2_control_yaml,
                "--switch-timeout", "30",
            ],
            output="both",
        )

    arm_spawner_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_trajectory_controller_spawner],
        )
    )

    gripper_spawner_event = None
    if gripper_controller_spawner is not None:
        gripper_spawner_event = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=arm_trajectory_controller_spawner,
                on_exit=[gripper_controller_spawner],
            )
        )

    moveit_args = {
        "robot_description": robot_description_raw,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": kinematics_config,
        "robot_description_planning": joint_limits_config,
        "moveit_simple_controller_manager": controllers_dict,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
        # trajectory_execution MUST be at move_group top-level (dot-notation keys).
        # Putting them inside moveit_simple_controller_manager (above) puts them in
        # the wrong namespace and MoveIt ignores them — causing execution_duration
        # monitoring to default ON (aborts mid-stroke) and start_tolerance to
        # default 0 (executes wrong-start trajectories that snap the servo open).
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.03,
        "trajectory_execution.execution_duration_monitoring": False,
        # Use full joint velocity/acceleration limits by default (MoveIt default is 0.1).
        # The gripper only travels 89 mm; at 10% speed it takes ~1 s to close.
        # At 100% it closes in ~0.2 s (limited by physical servo speed ~0.45 m/s).
        "default_velocity_scaling_factor": 1.0,
        "default_acceleration_scaling_factor": 1.0,
        **sensor_config,
        **ompl_planning_yaml,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        arguments=build_ros_log_arguments([
            ("move_group", "warn"),
            ("move_group.moveit.moveit.ros.planning_pipeline", "error"),
            ("moveit.simple_controller_manager.follow_joint_trajectory_controller_handle", "warn"),
        ] if suppress_moveit_execution_logs else []),
        parameters=[{"use_sim_time": use_sim_time}, moveit_args],
        output="screen",
    )

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
    servo_params_file = os.path.join(get_package_share_directory("aries_moveit"), "config", "servo.yaml")
    servo_context = load_yaml(Path(servo_params_file))
    servo_params = {"moveit_servo": servo_context}
    planning_group_name = {"planning_group_name": "igus_rebel_arm"}

    servo_node = Node(
        condition=servo_joystick_condition,
        package="moveit_servo",
        executable="servo_node",
        parameters=[
            {"use_sim_time": use_sim_time},
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
        name="servo_collision_guard",
        parameters=[
            {"use_sim_time": use_sim_time},
            moveit_args,
            {
                "input_topic": "servo_guard/input_joint_trajectory",
                "output_topic": "rebel_arm_trajectory_controller/joint_trajectory",
                "joint_state_topic": "joint_states",
                "status_topic": "/arm_joystick/status",
                "group_name": "arm_with_gripper",
                "min_self_distance": 0.015,
                "distance_tolerance": 0.001,
                "interpolation_steps": 1,
                "hold_time": 0.02,
            },
        ],
        output="screen",
    )

    joy_node = Node(
        condition=IfCondition(use_joystick),
        package="joy",
        executable=joy_driver,
        name="joy_node",
        parameters=[{
            "dev": joy_dev,
            "autorepeat_rate": 80.0,
            "deadzone": 0.0,
            "coalesce_interval_ms": 1,
        }],
        remappings=[("joy", "joy/raw")],
        output="screen",
    )

    joy_layout_normalizer_node = Node(
        condition=IfCondition(use_joystick),
        package="aries_moveit",
        executable="joy_layout_normalizer.py",
        name="joy_layout_normalizer",
        parameters=[{
            "input_topic": "joy/raw",
            "output_topic": "joy",
            "layout": joy_layout,
            "device": joy_dev,
        }],
        output="screen",
    )

    gamepad_file = os.path.join(get_package_share_directory("aries_moveit"), "config", "gamepad.yaml")
    gamepad_node = Node(
        condition=servo_joystick_condition,
        package="aries_moveit",
        executable="rebel_servo_teleop_gamepad",
        name="rebel_servo_teleop_gamepad",
        parameters=[gamepad_file],
        output="screen",
    )

    move_group_joystick_node = Node(
        condition=move_group_joystick_condition,
        package="aries_moveit",
        executable="rebel_movegroup_joystick.py",
        name="rebel_movegroup_joystick",
        parameters=[gamepad_file],
        output="screen",
    )

    # Gripper arc overlay for RViz: jaw open/close sweep + point of closing.
    # Geometry tables model the new four-bar gripper only.
    gripper_arc_visualizer_node = None
    if gripper_type == "new":
        gripper_arc_visualizer_node = Node(
            package="aries_moveit",
            executable="gripper_arc_visualizer.py",
            name="gripper_arc_visualizer",
            parameters=[{"use_sim_time": use_sim_time}],
            output="screen",
        )

    rviz_config = os.path.join(get_package_share_directory("aries_moveit"), "launch", "moveit.rviz")
    rviz_node = Node(
        condition=IfCondition(use_gui),
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        parameters=[
            {
                "robot_description": robot_description_raw,
                "robot_description_semantic": robot_description_semantic,
                "robot_description_kinematics": kinematics_config,
                "robot_description_planning": joint_limits_config,
            }
        ],
        output={"both": "log"},
    )

    nodes = [
        ros2_control_node,
        robot_state_pub,
        wheel_joint_publisher_node,
        joint_state_broadcaster_spawner,
        arm_spawner_event,
        move_group_node,
        servo_node,
        servo_collision_guard_node,
        joy_node,
        joy_layout_normalizer_node,
        gamepad_node,
        move_group_joystick_node,
        rviz_node,
    ]
    if gripper_spawner_event:
        nodes.append(gripper_spawner_event)
    if hand_guiding_node:
        nodes.append(hand_guiding_node)
    if micro_ros_agent:
        nodes.append(micro_ros_agent)
    if gripper_arc_visualizer_node:
        nodes.append(gripper_arc_visualizer_node)
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_gui", default_value="true", description="Launch RViz with MoveIt interface"),
            DeclareLaunchArgument("gripper_type", default_value="new", choices=["old", "new"], description="Which gripper URDF to load"),
            DeclareLaunchArgument("finger_type", default_value="bucket", choices=["bucket", "maintenance", "probe"], description="Swappable fingertip mesh (new gripper)"),
            DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Hardware protocol for arm backend"),
            DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Global hardware protocol passed to xacro (arm+gripper)"),
            DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Hardware protocol for gripper backend"),
            DeclareLaunchArgument("use_joystick", default_value="false", description="Start joystick arm teleop"),
            DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"], description="Joystick driver executable from the joy package"),
            DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"], description="Normalize joystick layout before teleop nodes consume /joy"),
            DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0", description="Joystick device used by joy_node and the layout normalizer"),
            DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"], description="servo uses smooth Cartesian MoveIt Servo teleop with collision guard; move_group uses planned steps"),
            DeclareLaunchArgument("serial_port", default_value="/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00", description="USB-serial port for the Teensy gripper controller"),
            DeclareLaunchArgument("suppress_rebel_logs", default_value="false", description="Suppress chatty igus_rebel logger output from ros2_control_node"),
            DeclareLaunchArgument("suppress_moveit_execution_logs", default_value="false", description="Suppress routine MoveIt execution chatter from move_group and ros2_control_node"),
            DeclareLaunchArgument("enable_depth_sensor", default_value="true", description="Populate MoveIt's Octomap from the gripper depth camera"),
            OpaqueFunction(function=launch_setup),
        ]
    )
