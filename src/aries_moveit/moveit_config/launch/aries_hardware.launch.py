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

import glob
import os
import socket
import tempfile
import time
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from aries_common.devices import device, device_str

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["gripper_gear_left_joint"]

# The board ID is baked into the by-id path, so swapping the Teensy changes it
# (16739090 -> 20379650 on 2026-08-12) and an exact-path check then resolves to
# mock_hardware against a perfectly healthy board. Treat the configured path as
# a preference and accept any Teensy. full_hardware_checker globs the same way,
# which is why it kept reporting "Gripper serial connected" while this probe
# fell back to mock and no command reached the servo.
TEENSY_BY_ID_GLOB = "/dev/serial/by-id/*Teensy*-if00"


def resolve_gripper_serial(configured: str, detect_timeout: float):
    """Find the Teensy to talk to. Returns (port_or_None, note_for_the_log).

    Waits up to detect_timeout for the device: a Teensy reset re-enumerates over
    USB, which takes 1-2 s, so relaunching straight after a reset loses the race.
    Observed one probe at 17:10:58 with the by-id link appearing at 17:10:59.2 --
    a one-shot check ran the whole session on a simulated gripper.
    """
    deadline = time.monotonic() + detect_timeout
    while True:
        if Path(configured).exists():
            return configured, ""
        found = sorted(glob.glob(TEENSY_BY_ID_GLOB))
        if found:
            note = f"  -- {configured} is absent, using the Teensy that IS present: {found[0]}"
            if len(found) > 1:
                note += f" ({len(found)} Teensys connected: {', '.join(found)})"
            return found[0], note
        if time.monotonic() >= deadline:
            return None, ""
        time.sleep(0.1)


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
        # JOINT_VELOCITY_SCALE=2.0 means effective max vel_cmd ≈ 0.87 rad/s,
        # so gain p saturates at a position error of 0.87/p rad.
        #   p=10 -> saturates at  5.0 deg   (the old jitter: measured release
        #                                    overshoot is 4-9 deg, i.e. right
        #                                    inside the saturating range)
        #   p= 3 -> saturates at 16.6 deg   (clear of it)
        #   p= 1 -> saturates at 49.8 deg   but brakes with only 0.07 rad/s at
        #                                   a 4 deg error, i.e. barely at all
        #
        # Measured with scripts/measure_teleop_tracking.py on the real arm
        # (27 clean releases): tracking lag 85 ms, release overshoot mean
        # 4.03 deg / worst 8.94 deg, against a designed lead of 1.38 deg. That
        # decomposes as v*lookahead (1.38) + v*lag (2.34) + the arm's own decel
        # (0.32) = 4.04 deg, which is the whole of the measured value.
        #
        # DO NOT RAISE p. Measured twice on hardware with
        # scripts/measure_teleop_tracking.py, the second time with the hold
        # target confirmed latched (holddrift 0.00 deg), so the JTC genuinely
        # had a growing position error to brake against:
        #
        #   p=1.0  stopping time 0.244 s, release overshoot 3.15 deg
        #   p=3.0  stopping time 0.339 s, release overshoot 5.78 deg   WORSE
        #
        # Raising the gain makes it worse because the loop has ~75 ms of
        # transport delay: the braking command is computed from an error that
        # is already 75 ms stale and lands 75 ms later still, so more gain
        # simply drives the loop under-damped and the peak displacement grows.
        # The d term compounds it, because igus_rebel/src/Rebel.cpp fills the
        # velocity state with an unfiltered first difference of a quantised
        # position (Rebel.cpp:683) and p=3 amplifies that noise.
        #
        # Saturation is NOT the limit here and was a red herring:
        # JOINT_VELOCITY_SCALE is 1.0 (Rebel.hpp:22 — the "2.0" that used to be
        # claimed in this comment was stale), so against the 1.5 rad/s rating
        # p=3 would not saturate until a 28 deg error, far past the ~6 deg
        # worst case observed. Dead time, not authority, is the constraint.
        #
        # The lever that does work is max_joint_velocity in
        # config/teleop_speeds.yaml: overshoot = release speed * ~0.25 s.
        data["rebel_arm_trajectory_controller"]["ros__parameters"]["gains"] = {
            # d damps the position-error correction so the arm does not
            # oscillate when the velocity stream stops (joystick centered).
            # ff_velocity_scale=1.0 keeps trajectory feed-forward as the primary
            # motion driver, so planned RViz trajectories are unaffected.
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
                        # trajectory (path) tolerance MUST stay 0.0 = disabled on
                        # this backend, because this joint has no encoder and so
                        # the "error" it measures is never a physical quantity.
                        #
                        # Two ways it fires spuriously:
                        #  1. Teensy present. read() sets state_pos_ = servo_pos_
                        #     and write() sets servo_pos_ = cmd_pos_ afterwards in
                        #     the same cycle, so measured trails desired by
                        #     exactly one control cycle by construction. At
                        #     update_rate 80 Hz that is 12.5 ms, and
                        #     joint_limits.yaml lets MoveIt plan this joint at
                        #     10 rad/s, so the structural lag alone reads as up
                        #     to 10 * 0.0125 = 0.125 rad, well past 0.05.
                        #  2. Teensy absent. read() only assigns state_pos_ when
                        #     state_received_ is true, so with no agent session
                        #     the reported position stays pinned at the
                        #     on_activate() default (min_pos, -1.57) while the
                        #     command moves. The error is then just "how far the
                        #     command has left fully-open" and grows without
                        #     bound - observed 0.053 rad at the first abort and
                        #     0.846 rad later in the same run.
                        #
                        # Case 2 is a disconnected gripper, NOT a tolerance
                        # problem; it must be diagnosed from the TeensyGripperSystem
                        # "Teensy not connected" warning and micro_ros_agent, not
                        # from a JTC abort. Contact and stall belong to the grasp
                        # node's watchdog.
                        "trajectory": 0.0,
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

    if arm_hardware_protocol == "auto":
        try:
            arm_endpoint = (device("arm.host"), int(device("arm.port")))
            with socket.create_connection(arm_endpoint, timeout=0.25):
                arm_hardware_protocol = "rebel"
        except OSError:
            arm_hardware_protocol = "mock_hardware"

    # Resolve the device before choosing the backend, so an explicit
    # gripper_hardware_protocol:=rebel also survives a board swap. Only spend the
    # detect timeout when we actually intend to drive the Teensy.
    detect_timeout = float(
        LaunchConfiguration("gripper_detect_timeout").perform(context)
    ) if gripper_hardware_protocol in ("auto", "rebel") else 0.0
    teensy_port, serial_note = resolve_gripper_serial(serial_port, detect_timeout)
    if teensy_port:
        serial_port = teensy_port

    if gripper_hardware_protocol == "auto":
        gripper_hardware_protocol = "rebel" if teensy_port else "mock_hardware"

    # Always say which backend won. Silent fallback to mock is indistinguishable
    # from a dead gripper from the outside.
    gripper_detect_note = LogInfo(
        msg=f"[gripper auto] serial_port={serial_port} resolved={gripper_hardware_protocol}"
        + serial_note
        + ("" if gripper_hardware_protocol == "rebel"
           else "  -- SIMULATED gripper: no command will reach the servo")
    )

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
        condition=IfCondition(
            LaunchConfiguration("use_wheel_joint_publisher")
        ),
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
                # 115200, NOT 6000000. Linux speed_t values are encodings, not
                # literal bit rates: the largest valid one is B4000000 == 4111.
                # 6000000 is not in the agent's baud table
                # (xrceagent/.../baud_rate_table_linux.h), so it falls through to
                # a raw (speed_t)atoi() cast, and cfsetispeed/cfsetospeed then
                # reject it with EINVAL and leave c_ispeed/c_ospeed at 0. The
                # agent does not check either return value, so it proceeds to
                # tcsetattr with the line speed unset -- which is why the agent
                # came up most of the time and intermittently did not.
                # The Teensy is USB CDC (Tools > USB Type > Serial), where baud
                # is ignored by the device entirely, so nothing is lost here.
                "serial", "--dev", serial_port, "-b", "115200",
            ],
            additional_env={
                # Unicast-only participant + synchronous publish mode + zero latency budget.
                # Eliminates multicast discovery overhead and DDS write-thread buffering,
                # cutting the agent ↔ ROS2 round-trip by several milliseconds.
                "FASTRTPS_DEFAULT_PROFILES_FILE": _fastdds_xml,
            },
            output="screen",
            # The agent is the ONLY path between /gripper/cmd and the servo, and
            # it does die on its own: observed exit code 254 roughly 6 minutes
            # into a session, with /dev/ttyACM0 never re-enumerating (so not a
            # USB drop). Without respawn nothing restarts it, the Teensy falls
            # back to WAITING_AGENT and pings a closed port forever, and the
            # gripper stops responding mid-run with no error anywhere: the
            # hardware plugin keeps state_received_ latched true, so read()
            # still echoes servo_pos_ and the JTC still sees perfect tracking
            # while no command reaches the servo. The firmware already handles
            # its side of the reconnect, so restarting the agent is enough to
            # re-establish the session.
            respawn=True,
            respawn_delay=2.0,
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
    # Speeds live in teleop_speeds.yaml and are loaded last so they win over the
    # copies still in gamepad.yaml. gamepad.yaml keeps the button/axis mapping.
    teleop_speeds_file = os.path.join(
        get_package_share_directory("aries_moveit"), "config", "teleop_speeds.yaml"
    )
    gamepad_node = Node(
        condition=servo_joystick_condition,
        package="aries_moveit",
        executable="rebel_servo_teleop_gamepad",
        name="rebel_servo_teleop_gamepad",
        parameters=[gamepad_file, teleop_speeds_file],
        output="screen",
    )

    move_group_joystick_node = Node(
        condition=move_group_joystick_condition,
        package="aries_moveit",
        executable="rebel_movegroup_joystick.py",
        name="rebel_movegroup_joystick",
        parameters=[gamepad_file, teleop_speeds_file],
        output="screen",
    )

    # LT + Y -> pick_home, LT + A -> probe_drop, LT + B -> soil_drop. MoveIt
    # collision-plans each preset; the idle arm controller executes it directly.
    # Refusing while RB/RT/LB are held prevents competition with manual teleop.
    arm_preset_pose_node = Node(
        condition=IfCondition(use_joystick),
        package="aries_moveit",
        executable="arm_preset_pose_joystick.py",
        name="arm_preset_pose_joystick",
        parameters=[gamepad_file, teleop_speeds_file],
        output="screen",
    )

    # Gripper arc overlay for RViz (aries_moveit/scripts/gripper_arc_visualizer.py)
    # is not launched. Its geometry tables model the 85.563 mm four-bar of the
    # retired gripper_new only; on v2 (50 mm parallelogram, 83 mm stroke) the
    # sweep it draws is wrong by 100 mm. It used to be gated on
    # gripper_type == "new", which no longer exists. Re-fit the tables to v2
    # before adding the node back.

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
        gripper_detect_note,
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
        arm_preset_pose_node,
        rviz_node,
    ]
    if gripper_spawner_event:
        nodes.append(gripper_spawner_event)
    if hand_guiding_node:
        nodes.append(hand_guiding_node)
    if micro_ros_agent:
        nodes.append(micro_ros_agent)
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_gui", default_value="true", description="Launch RViz with MoveIt interface"),
            DeclareLaunchArgument("gripper_type", default_value="v2", choices=["v2"], description="Which gripper URDF to load. Only 'v2' exists; 'new' and 'old' are retired to aries/urdf/legacy/."),
            DeclareLaunchArgument("finger_type", default_value="bucket", choices=["bucket", "maintenance", "probe"], description="Swappable fingertip mesh (new/v2 gripper)"),
            DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Hardware protocol for arm backend"),
            DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Global hardware protocol passed to xacro (arm+gripper)"),
            DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"], description="Hardware protocol for gripper backend"),
            DeclareLaunchArgument("use_joystick", default_value="false", description="Start joystick arm teleop"),
            DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"], description="Joystick driver executable from the joy package"),
            DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"], description="Normalize joystick layout before teleop nodes consume /joy"),
            DeclareLaunchArgument("joy_dev", default_value=device_str("joystick.device"), description="Joystick device used by joy_node and the layout normalizer"),
            DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"], description="servo uses smooth Cartesian MoveIt Servo teleop with collision guard; move_group uses planned steps"),
            DeclareLaunchArgument("serial_port", default_value=device_str("gripper.serial_port"), description="USB-serial port for the Teensy gripper controller"),
            DeclareLaunchArgument("gripper_detect_timeout", default_value="8.0", description="Seconds to wait for the Teensy serial device before falling back to mock_hardware. Covers USB re-enumeration after a board reset."),
            DeclareLaunchArgument("suppress_rebel_logs", default_value="false", description="Suppress chatty igus_rebel logger output from ros2_control_node"),
            DeclareLaunchArgument("suppress_moveit_execution_logs", default_value="false", description="Suppress routine MoveIt execution chatter from move_group and ros2_control_node"),
            DeclareLaunchArgument(
                "use_wheel_joint_publisher",
                default_value="true",
                description=(
                    "Publish zero-valued rover wheel joints when no real "
                    "encoder-backed publisher is running"
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
