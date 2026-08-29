#!/usr/bin/env python3
"""The one command to run on the BASE STATION.

    ros2 launch aries_comms base_station.launch.py

Partner of rover_field.launch.py, beside it in this package. Between them
nothing else has to be started by hand:

    rover                                    base station
    -----                                    ------------
    rover_field.launch.py                    base_station.launch.py
      DDS on the field link                    DDS on the field link
      arm + gripper + MoveIt                   joy driver     -> /joy
      rover drive + IMU                        decompressors  -> /<cam>/view/*
      cameras -> /downlink/<cam>/*             RViz  (3 cams: wrist, front, rear)
      teleop consumers <- /joy                 base_station_checker
      full_hardware_checker

WHAT RUNS HERE, AND WHY EACH ONE IS HERE AND NOT THERE

  joy driver + layout normalizer
      The pad is plugged in HERE. /joy crosses the antenna to the teleop
      consumers on the rover, which is a few hundred kbit/s -- the joystick is
      the cheapest thing on the link by three orders of magnitude.

      Exactly one machine may run this. The rover must be launched with
      use_joy_node:=false (rover_field.launch.py already defaults to that).
      Two joy drivers means two publishers on /joy and the consumers see the
      two pads interleaved at double rate: buttons appear to chatter and
      nothing is reproducible.

      A dropped link now means a silent /joy, which every teleop node treats
      as "stop" after 0.35 s. That is a behaviour change from a locally
      plugged pad and it is the point: the alternative is a rover that keeps
      driving on its last command.

  decompressors (aries_bringup/camera_view.launch.py)
      They turn /downlink/<cam>/... into machine-local /<cam>/view/... . This
      is what keeps the link at ~28 Mbit/s instead of ~740. RViz's ROS 2 Image
      display has no transport selection -- it subscribes RAW, always -- so an
      Image display pointed at a rover topic pulls uncompressed frames across
      the antenna no matter what the DepthCloud beside it is set to. Never
      point anything here at a rover camera topic; read /<camera>/view/*.

      Two processes subscribing to the same compressed topic would also pull
      two copies across, being separate DDS participants. One republisher per
      stream, everything else reading its local output.

  RViz -- exactly one, the same one full_hardware.launch.py opens
      Not on the rover. There it costs CPU and GPU that two RealSense
      pipelines need, and fails outright over SSH with no display.

      The robot model renders here only because the base station runs the same
      workspace as the rover: RViz resolves package:// mesh paths against the
      local filesystem, so a machine without the aries packages would show an
      empty scene and a list of resource errors.

      This used to open TWO windows. camera_view.launch.py, included below,
      declared a `use_rviz` of its own -- and an included launch description
      inherits the parent's launch configurations, with DeclareLaunchArgument
      leaving an already-set one alone. So the `use_rviz` declared HERE, default
      true, switched on a viewer over THERE as well, and it came up blank
      because it also inherited this file's `rviz_config` of "" (which only
      means "use my default" to the code below). That node is gone from
      camera_view, and the include is scoped so nothing else can leak into it.

  base_station_checker
      The operator-side counterpart of full_hardware_checker.py, which runs on
      the rover and cannot answer any question that matters here: it probes
      serial ports, CAN and USB, all of which are on the robot, and its output
      goes to the rover's console. What can be wrong at this end is the link,
      the pad, and whether the downlink is actually arriving -- so that is what
      this one reports. It never subscribes to /downlink/*; see the node.

  NOT robot_state_publisher.
      /tf and /joint_states come from the rover. A second publisher here would
      fight it with a model built from a possibly different xacro.

MATCH gripper_type AND finger_type TO THE ROVER
      The description below is built locally, so it is only as correct as the
      arguments given to it. A base station showing bucket fingers while the
      rover carries the probe tips is a plausible-looking display that is
      quietly wrong, and MoveIt would plan against the wrong collision model.

SHELLS
      This launch sets its environment for the nodes it starts, but it cannot
      reach into your terminal. For rqt_image_view, a second rviz2, or
      `ros2 topic list`:

          source "$(ros2 pkg prefix aries_common)/share/aries_common/aries_dds_env.sh"
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_param_builder import load_yaml
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from aries_common.comms import dds_launch_actions
from aries_common.devices import device_str


def _rviz_parameters(context):
    """The robot description RViz needs, built from this machine's workspace.

    Taken locally rather than from the rover's /robot_description topic because
    the MoveIt RViz plugins read these as node parameters. The kinematics are
    identical either way -- only the ros2_control block differs, and RViz does
    not look at it -- so the description is built with mock_hardware here and
    no serial port is probed.
    """
    gripper_type = LaunchConfiguration("gripper_type").perform(context)
    finger_type = LaunchConfiguration("finger_type").perform(context)

    urdf = PathJoinSubstitution(
        [FindPackageShare("aries"), "urdf", "my_robot.urdf.xacro"]
    )
    robot_description = Command([
        FindExecutable(name="xacro"), " ", urdf,
        " hardware_protocol:=mock_hardware",
        " arm_hardware_protocol:=mock_hardware",
        " gripper_hardware_protocol:=mock_hardware",
        " gripper_type:=", gripper_type,
        " finger_type:=", finger_type,
        " serial_port:=", device_str("gripper.serial_port"),
    ]).perform(context)

    moveit_share = get_package_share_directory("aries_moveit")
    srdf = os.path.join(moveit_share, "config", "aries.srdf")

    return {
        "robot_description": robot_description,
        "robot_description_semantic": ParameterValue(
            Command(["cat ", srdf]), value_type=str
        ),
        "robot_description_kinematics": load_yaml(
            Path(os.path.join(moveit_share, "config", "kinematics.yaml"))
        ),
        "robot_description_planning": load_yaml(
            Path(os.path.join(moveit_share, "config", "joint_limits.yaml"))
        ),
    }


def _setup(context, *args, **kwargs):
    cameras = LaunchConfiguration("cameras").perform(context)
    actions = []

    # Decompress once, locally. Everything downstream reads /<camera>/view/*.
    #
    # forwarding=False is the fix for the two-RViz bug and is worth keeping even
    # though camera_view no longer has a viewer to switch on. An include
    # normally inherits EVERY launch configuration set above it, so any
    # argument this file declares is silently visible to the included file
    # under the same name -- `use_rviz` and `rviz_config` were, and started a
    # second, unconfigured RViz. Scoped like this the child sees exactly the
    # two values listed and nothing else, so the interface between the files is
    # what is written here rather than whatever names happen to coincide.
    actions.append(
        GroupAction(
            condition=IfCondition(LaunchConfiguration("use_camera_view")),
            scoped=True,
            forwarding=False,
            launch_configurations={
                "cameras": cameras,
                "color_only": LaunchConfiguration("color_only"),
            },
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare("aries_bringup"),
                            "launch",
                            "camera_view.launch.py",
                        ])
                    )
                )
            ],
        )
    )

    # Above the early return below: use_rviz:=false is the "decompress only"
    # case, and that is exactly when a status line is the only thing telling
    # the operator whether the link is alive.
    actions.append(
        GroupAction(
            condition=IfCondition(LaunchConfiguration("start_checker")),
            # Scoped for the same reason the include above is: what this file
            # hands the checker is the list below, not every name that happens
            # to be declared here.
            scoped=True,
            forwarding=False,
            launch_configurations={
                "checker_interval": LaunchConfiguration("checker_interval"),
                # The same two lists the decompressors were started with. A
                # camera the checker knows about and camera_view does not would
                # be reported as a permanently dead stream, which reads as a
                # link fault.
                "cameras": cameras,
                "color_only": LaunchConfiguration("color_only"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                # Whether a local /joy publisher is expected, or is the fault.
                "expect_local_joy": LaunchConfiguration("use_joy_node"),
                # The rover's own subsystems, read from the topics that cross
                # the link. On by default: the operator cannot see the robot,
                # and this is the only report they get. `checker_imu:=false`
                # is the one worth turning off on a marginal link -- it is
                # 100 Hz of sensor_msgs/Imu and everything else together is
                # under 0.1 Mbit/s.
                "check_arm": LaunchConfiguration("checker_arm"),
                "check_gripper": LaunchConfiguration("checker_gripper"),
                "check_drive": LaunchConfiguration("checker_drive"),
                "check_imu": LaunchConfiguration("checker_imu"),
            },
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare("aries_comms"),
                            "launch",
                            "base_station_checker.launch.py",
                        ])
                    )
                )
            ],
        )
    )

    # Built only when it is going to be used: this runs xacro over the whole
    # robot, which is a second or two, and use_rviz:=false is the "decompress
    # only, I will start my own viewer" case.
    if LaunchConfiguration("use_rviz").perform(context).lower() != "true":
        return actions

    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    if not rviz_config:
        # aries_moveit's own config, not a copy: its four camera displays are
        # already wired to /<camera>/view/* and its MotionPlanning panel is the
        # one the arm was set up with. A second copy here would drift.
        rviz_config = os.path.join(
            get_package_share_directory("aries_moveit"), "launch", "moveit.rviz"
        )

    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="base_station_rviz",
            arguments=["-d", rviz_config],
            parameters=[_rviz_parameters(context)],
            output={"both": "log"},
        )
    )
    return actions


def generate_launch_description():
    joy_dev = LaunchConfiguration("joy_dev")

    return LaunchDescription([
        # Before every node: launch runs actions in order and a node started
        # above these would keep the calling shell's environment, which is the
        # single most common cause of "the link is fine but there are no
        # topics". See aries_common/comms.py.
        *dds_launch_actions(),

        DeclareLaunchArgument(
            "use_joy_node", default_value="true",
            description="Read the pad here. The rover must then run with "
                        "use_joy_node:=false -- only one /joy publisher.",
        ),
        DeclareLaunchArgument(
            "joy_driver", default_value="game_controller_node",
            choices=["game_controller_node", "joy_node"],
        ),
        DeclareLaunchArgument(
            "joy_layout", default_value="auto",
            choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"],
            description="Maps whatever pad is plugged in back onto the "
                        "Xbox-style layout the rover's teleop nodes expect.",
        ),
        DeclareLaunchArgument("joy_dev", default_value=device_str("joystick.device")),
        DeclareLaunchArgument(
            "use_camera_view", default_value="true",
            description="Decompress the downlink here. Without it RViz has "
                        "nothing local to read and the /<cam>/view topics "
                        "never exist.",
        ),
        DeclareLaunchArgument(
            "cameras", default_value="gripper_camera,rover_camera,rear_camera",
            description="Comma-separated cameras to decompress. Drop one to "
                        "cut the link load when it is tight -- the two D435is "
                        "are ~14 Mbit/s each, the rear camera ~3.9.",
        ),
        DeclareLaunchArgument(
            "color_only", default_value="rear_camera",
            description="Of those, the ones with no depth stream, so no depth "
                        "decompressor is started for them. Must match the "
                        "rover's argument of the same name.",
        ),
        DeclareLaunchArgument(
            "start_checker", default_value="true",
            description="The operator-side status line: link, pad, downlink, "
                        "and the rover's own subsystems as its topics show "
                        "them. full_hardware_checker stays on the rover, sees "
                        "the link from neither end, and prints where nobody "
                        "here is looking.",
        ),
        DeclareLaunchArgument("checker_interval", default_value="4.0"),

        # Which of the rover's subsystems the checker reports on. On by
        # default: the operator cannot see the robot, so this is the only
        # report they get. Presence comes from the graph and costs nothing;
        # the subscriptions behind these are the topics that were sized for a
        # 250 kbit/s CAN bus, together under 0.1 Mbit/s.
        DeclareLaunchArgument("checker_arm", default_value="true"),
        DeclareLaunchArgument("checker_gripper", default_value="true"),
        DeclareLaunchArgument(
            "checker_drive", default_value="true",
            description="The drive bridge's own 2 Hz report -- armed, "
                        "pending_axes, and the CAN link as the ROVER sees it "
                        "-- plus /cmd_vel and the six ODrive axes.",
        ),
        DeclareLaunchArgument(
            "checker_imu", default_value="true",
            description="The one row that costs real bandwidth: 100 Hz of "
                        "sensor_msgs/Imu, ~0.3 Mbit/s. Still on by default, "
                        "because the checker already pays about that for "
                        "/joint_states and a silently dead IMU means the "
                        "heading drifts with nothing saying so. Turn it off "
                        "when the link itself is what is marginal.",
        ),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config", default_value="",
            description="Empty means aries_moveit/launch/moveit.rviz.",
        ),
        DeclareLaunchArgument(
            "gripper_type", default_value="v2", choices=["v2", "st3215"],
            description="MUST match the rover, or the model shown here is not "
                        "the robot that is moving.",
        ),
        DeclareLaunchArgument(
            "finger_type", default_value="bucket",
            choices=["bucket", "maintenance", "probe"],
            description="MUST match the rover. Wrong tips look plausible and "
                        "plan against the wrong collision model.",
        ),

        LogInfo(msg="[base_station] operator side up. The rover must be running "
                    "rover_field.launch.py (or full_hardware with "
                    "use_joy_node:=false)."),

        # The joy driver and the normalizer, with the parameters the arm teleop
        # was tuned against. autorepeat_rate is the one that matters over a
        # radio link: the driver resends the current pad state at 80 Hz, so a
        # lost packet is corrected 12 ms later instead of leaving the rover
        # holding a stale command until the next physical stick movement. It is
        # also what keeps the 0.35 s staleness guards on the rover from tripping
        # while the operator holds a stick perfectly still.
        Node(
            condition=IfCondition(LaunchConfiguration("use_joy_node")),
            package="joy",
            executable=LaunchConfiguration("joy_driver"),
            name="joy_node",
            parameters=[{
                "dev": joy_dev,
                "autorepeat_rate": 80.0,
                # Zero here on purpose: the deadzone belongs to each consumer,
                # which applies its own and rescales what is left. Two
                # deadzones in series leave a dead band you cannot tune out.
                "deadzone": 0.0,
                "coalesce_interval_ms": 1,
            }],
            remappings=[("joy", "joy/raw")],
            output="screen",
        ),
        Node(
            condition=IfCondition(LaunchConfiguration("use_joy_node")),
            package="aries_moveit",
            executable="joy_layout_normalizer.py",
            name="joy_layout_normalizer",
            parameters=[{
                "input_topic": "joy/raw",
                "output_topic": "joy",
                "layout": LaunchConfiguration("joy_layout"),
                "device": joy_dev,
            }],
            output="screen",
        ),

        OpaqueFunction(function=_setup),
    ])
