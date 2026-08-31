#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from aries_common.comms import dds_launch_actions, local_address
from aries_common.devices import device_str
from aries_common.gripper_cal import cal_str


def generate_launch_description():
    # This machine's field-link address, or None if it has no antenna. Read
    # once here only so the launch can SAY which of the two configurations it
    # ended up with; dds_environment() detects it again for itself.
    link_address = local_address()

    return LaunchDescription([
        # MUST stay first: launch executes actions in order, and a node started
        # above these keeps the calling terminal's environment. That is not
        # cosmetic -- a terminal that never sourced aries_dds_env.sh puts this
        # whole stack on domain 0 with rmw_fastrtps_cpp, where every driver
        # runs perfectly and nothing else on the robot can see a single topic.
        # Nothing logs an error; the symptom is an empty `ros2 topic list` on a
        # link that pings fine. See aries_common/comms.py.
        #
        # require_link=False, unlike rover_field.launch.py's require_link=True.
        # This file is also the bench and single-machine entry point, and a
        # developer laptop with no antenna must still be able to bring the
        # stack up -- it gets a loopback-only config instead of an exception.
        # rover_field keeps the hard failure, because out there a missing cable
        # IS the bug and should stop the launch rather than degrade it.
        #
        # To override: ARIES_DOMAIN_ID for the domain, ARIES_KEEP_CYCLONEDDS_URI=1
        # to keep a hand-written Cyclone config. Plain ROS_DOMAIN_ID in the
        # shell is deliberately NOT honoured -- that is the whole point.
        *dds_launch_actions(require_link=False),
        LogInfo(msg=(
            f"[full_hardware] field link: this machine is {link_address}"
            if link_address else
            "[full_hardware] field link: NOT on it -- loopback-only DDS. "
            "Nothing off this machine will see these topics. Check the antenna "
            "cable and `ip -4 -br addr` if that is not what you wanted."
        )),

        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        # Whether the pad is READ on this machine, separately from whether the
        # teleop nodes run here. Set false on the rover when the operator holds
        # the pad at the base station: every consumer stays up and takes /joy
        # over the antenna. rover_field.launch.py does exactly that.
        DeclareLaunchArgument("use_joy_node", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"]),

        DeclareLaunchArgument("gripper_type", default_value="st3215", choices=["st3215"]),
        DeclareLaunchArgument("finger_type", default_value="bucket", choices=["bucket", "maintenance"]),
        DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("serial_port", default_value=device_str("gripper.serial_port")),
        # The secondary gripper's wire. Overridable here for the same reason
        # serial_port is: a bench adapter does not always land on the udev
        # symlink, and without these the only way to point the stack at a
        # different port is to edit devices.yaml.
        DeclareLaunchArgument("servo_bus_port", default_value=device_str("servo_bus.port")),
        DeclareLaunchArgument("servo_bus_baud", default_value=device_str("servo_bus.baud")),
        DeclareLaunchArgument("servo_id", default_value=device_str("servo_bus.gripper_servo_id")),
        DeclareLaunchArgument("gripper_closed_steps", default_value=cal_str("closed_steps")),
        DeclareLaunchArgument("gripper_servo_invert", default_value=cal_str("invert"), choices=["true", "false"]),
        DeclareLaunchArgument("suppress_rebel_logs", default_value="true"),
        DeclareLaunchArgument("suppress_moveit_execution_logs", default_value="true"),
        DeclareLaunchArgument("enable_depth_sensor", default_value="auto", choices=["auto", "true", "false"]),
        # Pinned by default: with two identical D435is connected, anything that
        # assigns them by detection order can hand the grasp stack the front
        # camera. See aries_common.detect for which serial is on which end.
        DeclareLaunchArgument("gripper_camera_serial", default_value=device_str("cameras.gripper_serial")),
        DeclareLaunchArgument("enable_front_camera", default_value="auto", choices=["auto", "true", "false"]),
        # The third camera: a Logitech Brio under the tail aimed at the drill.
        # A UVC webcam, not a RealSense, so it is addressed by device path and
        # carries colour only -- no depth, no serial, nothing on the enumeration
        # the two D435is are pinned from.
        DeclareLaunchArgument("enable_rear_camera", default_value="auto", choices=["auto", "true", "false"]),
        DeclareLaunchArgument("rear_camera_device", default_value=device_str("cameras.rear_device")),
        # Operator camera downlink: a reduced, JPEG/PNG-compressed copy of every
        # camera for anything watching over the antenna. The driver topics the
        # on-board pipelines read are untouched and stay at 640x480@15. The rear
        # camera is colour only and runs at its own, lower rate -- see
        # downlink_color_only_rate_hz in camera_downlink.launch.py.
        DeclareLaunchArgument(
            "enable_camera_downlink", default_value="true", choices=["true", "false"],
            description="Publish the compressed operator camera streams.",
        ),
        DeclareLaunchArgument(
            "downlink_rate_hz", default_value="15.0",
            description="Downlink colour frames per second; 15 matches the camera.",
        ),
        DeclareLaunchArgument(
            "downlink_profile", default_value="balanced",
            choices=["quality", "balanced", "lean"],
            description="Measured operating points for both cameras at 15 Hz "
                        "colour / 5 Hz depth: quality 42.3 Mbit/s (640x480 q90), "
                        "balanced 28.3 (640x480 q75), lean 10.9 (320x240 q90).",
        ),
        DeclareLaunchArgument(
            "downlink_depth_rate_hz", default_value="5.0",
            description="Downlink depth frames per second. Depth costs ~6x a colour "
                        "frame, so cut this first when the link is tight.",
        ),
        DeclareLaunchArgument(
            "downlink_decimation", default_value="profile",
            description="Integer spatial divisor: 1 -> 640x480, 2 -> 320x240.",
        ),
        DeclareLaunchArgument(
            "downlink_jpeg_quality", default_value="profile",
            description="Colour JPEG quality 1-100, or \"profile\".",
        ),
        DeclareLaunchArgument(
            "downlink_depth_max_m", default_value="6.0",
            description="Depth beyond this is dropped before compression.",
        ),
        DeclareLaunchArgument(
            "downlink_depth_quantization_mm", default_value="10",
            description="Depth rounding step. View-only stream; nothing plans on it.",
        ),
        DeclareLaunchArgument("front_camera_serial", default_value=device_str("cameras.front_serial")),
        DeclareLaunchArgument(
            "use_static_wheel_joint_publisher",
            default_value="false",
            description=(
                "Publish zero-valued wheel joints from the arm stack. Keep "
                "false when the rover encoder-backed publisher is active."
            ),
        ),

        DeclareLaunchArgument("start_rover", default_value="true"),
        DeclareLaunchArgument("rover_hardware_protocol", default_value="auto", choices=["auto", "odrive", "mock_hardware"]),
        DeclareLaunchArgument("can_interface", default_value=device_str("rover.can_interface")),
        DeclareLaunchArgument("setup_rover_can", default_value="true"),
        DeclareLaunchArgument("drive_auto_arm", default_value="true"),
        DeclareLaunchArgument(
            "use_rover_imu",
            default_value="auto",
            choices=[
                "auto",
                "true",
                "false",
                "microstrain",
            ],
        ),
        DeclareLaunchArgument(
            "rover_imu_port", default_value=device_str("imu.port")
        ),
        DeclareLaunchArgument("rover_imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("rover_imu_frame", default_value="imu_frame"),
        DeclareLaunchArgument(
            "rover_imu_topic", default_value="/microstrain/imu/data"
        ),
        DeclareLaunchArgument("use_rover_joy_node", default_value="false"),

        DeclareLaunchArgument("use_stacklight", default_value="true"),
        DeclareLaunchArgument("use_load_cells", default_value="true"),
        DeclareLaunchArgument("use_science", default_value="true"),
        DeclareLaunchArgument("use_drill_driver", default_value="true"),
        DeclareLaunchArgument("load_cell_source", default_value="auto",
                              choices=["auto", "microros", "mock"]),
        DeclareLaunchArgument("start_checker", default_value="true"),
        # Preserve the top-level choice before rover_drive_auto.launch.py sets
        # its own nested start_checker argument to false. Included launch
        # configurations share context and would otherwise disable this checker.
        SetLaunchConfiguration(
            "_start_full_hardware_checker",
            LaunchConfiguration("start_checker"),
        ),
        DeclareLaunchArgument("checker_interval", default_value="4.0"),

        # Arm + gripper + MoveIt/RViz + shared joy_node.
        # Existing aries_hardware.launch.py handles real-or-mock arm/gripper auto behavior.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "aries_hardware.launch.py",
                ])
            ),
            launch_arguments={
                "use_gui": LaunchConfiguration("use_gui"),
                "use_joystick": LaunchConfiguration("use_joystick"),
                "use_joy_node": LaunchConfiguration("use_joy_node"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "joystick_control_mode": LaunchConfiguration("joystick_control_mode"),
                "gripper_type": LaunchConfiguration("gripper_type"),
                "finger_type": LaunchConfiguration("finger_type"),
                # One flag gates both halves of the science module: the agent,
                # which is started down in aries_moveit's hardware launch, and
                # the host node included below.
                "use_science": LaunchConfiguration("use_science"),
                "hardware_protocol": LaunchConfiguration("hardware_protocol"),
                "arm_hardware_protocol": LaunchConfiguration("arm_hardware_protocol"),
                "gripper_hardware_protocol": LaunchConfiguration("gripper_hardware_protocol"),
                "serial_port": LaunchConfiguration("serial_port"),
                "servo_bus_port": LaunchConfiguration("servo_bus_port"),
                "servo_bus_baud": LaunchConfiguration("servo_bus_baud"),
                "servo_id": LaunchConfiguration("servo_id"),
                "gripper_closed_steps": LaunchConfiguration("gripper_closed_steps"),
                "gripper_servo_invert": LaunchConfiguration("gripper_servo_invert"),
                "suppress_rebel_logs": LaunchConfiguration("suppress_rebel_logs"),
                "suppress_moveit_execution_logs": LaunchConfiguration("suppress_moveit_execution_logs"),
                "enable_depth_sensor": LaunchConfiguration("enable_depth_sensor"),
                "gripper_camera_serial": LaunchConfiguration("gripper_camera_serial"),
                "enable_front_camera": LaunchConfiguration("enable_front_camera"),
                "front_camera_serial": LaunchConfiguration("front_camera_serial"),
                "enable_rear_camera": LaunchConfiguration("enable_rear_camera"),
                "rear_camera_device": LaunchConfiguration("rear_camera_device"),
                "enable_camera_downlink": LaunchConfiguration("enable_camera_downlink"),
                "downlink_profile": LaunchConfiguration("downlink_profile"),
                "downlink_rate_hz": LaunchConfiguration("downlink_rate_hz"),
                "downlink_depth_rate_hz": LaunchConfiguration("downlink_depth_rate_hz"),
                "downlink_decimation": LaunchConfiguration("downlink_decimation"),
                "downlink_jpeg_quality": LaunchConfiguration("downlink_jpeg_quality"),
                "downlink_depth_max_m": LaunchConfiguration("downlink_depth_max_m"),
                "downlink_depth_quantization_mm": LaunchConfiguration(
                    "downlink_depth_quantization_mm"
                ),
                "use_wheel_joint_publisher": LaunchConfiguration(
                    "use_static_wheel_joint_publisher"
                ),
            }.items(),
        ),

        # Rover real-or-mock backend.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "rover_drive_auto.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("start_rover")),
            launch_arguments={
                "rover_hardware_protocol": LaunchConfiguration("rover_hardware_protocol"),
                "can_interface": LaunchConfiguration("can_interface"),
                "setup_can": LaunchConfiguration("setup_rover_can"),
                "drive_auto_arm": LaunchConfiguration("drive_auto_arm"),
                "use_joystick": LaunchConfiguration("use_joystick"),
                # joy_node is already started by aries_hardware when use_joystick:=true.
                "use_joy_node": LaunchConfiguration("use_rover_joy_node"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "use_imu": LaunchConfiguration("use_rover_imu"),
                "imu_port": LaunchConfiguration("rover_imu_port"),
                "imu_baudrate": LaunchConfiguration("rover_imu_baudrate"),
                "imu_frame": LaunchConfiguration("rover_imu_frame"),
                "imu_topic": LaunchConfiguration("rover_imu_topic"),
            }.items(),
        ),

        # Mast stack light: red on e-stop or halt, yellow operating, green
        # ready. The publisher for the topic the drill Teensy's firmware has
        # always subscribed to -- without it the light stays dark whatever the
        # rover does. Reads the drive bringup's status, so it belongs on the
        # rover side even though the Teensy is on the arm's.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "stacklight.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("use_stacklight")),
        ),

        # The drill driver: turns drill_joystick.py's rate commands into the
        # duty cycle the drill Teensy takes. Same board and same agent as the
        # stack light and the gripper -- one Teensy runs all three since the
        # firmware moved to firmware/teensy_drill_sys.
        #
        # Until this existed the drill's three command topics reached nothing on
        # the real rover; drill_joystick.py's own docstring said as much. NOTE
        # that its calibration is not measured yet: the drill moves, but the
        # rates on those topics are not the rates the mechanism is doing. See
        # config/drill_driver.yaml.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "drill_driver.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("use_drill_driver")),
        ),

        # The three load cells (aries_load_cells): the sand and stone boxes on
        # the left of the deck, and the drill's sample bin. Same Teensy as the
        # stack light, so the micro-ROS agent started above already carries
        # them -- this only starts the node that turns counts into kilograms.
        #
        # load_cell_source:=mock makes up counts, for exercising the topics
        # with no board attached. The firmware is still being written; `auto`
        # does NOT fall back to mock, because a fabricated weight that looks
        # exactly like a measured one is not something the rover should emit.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_load_cells"),
                    "launch",
                    "load_cells.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("use_load_cells")),
            launch_arguments={
                "load_cell_source": LaunchConfiguration("load_cell_source"),
            }.items(),
        ),

        # THE SCIENCE MODULE'S HOST HALF (aries_science). The sensors live on a
        # SECOND Teensy -- firmware/teensy_science_sys, its own USB port, its
        # own micro-ROS agent, started by aries_hardware.launch.py alongside
        # the drill board's. This starts only the node that splits the board's
        # ten-value telemetry array into named topics.
        #
        # It is started whether or not the board is present, deliberately: with
        # no board it publishes "no telemetry yet" on /science/status once a
        # second, which is how an operator tells a missing board from a missing
        # node. The same use_science argument gates the agent over in
        # aries_hardware.launch.py, so turning it off here turns off both.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_science"),
                    "launch",
                    "science.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("use_science")),
        ),

        # Separate checker.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "full_hardware_checker.launch.py",
                ])
            ),
            condition=IfCondition(
                LaunchConfiguration("_start_full_hardware_checker")
            ),
            launch_arguments={
                "checker_interval": LaunchConfiguration("checker_interval"),
                "serial_port": LaunchConfiguration("serial_port"),
                "gripper_type": LaunchConfiguration("gripper_type"),
                "servo_bus_port": LaunchConfiguration("servo_bus_port"),
                "can_interface": LaunchConfiguration("can_interface"),
                "use_imu": LaunchConfiguration("use_rover_imu"),
                "imu_port": LaunchConfiguration("rover_imu_port"),
                "imu_frame": LaunchConfiguration("rover_imu_frame"),
                "imu_topic": LaunchConfiguration("rover_imu_topic"),
                # The rover has three cameras; the checker has to be told about
                # all of them or one that never started reads as healthy.
                # Same flags aries_hardware.launch.py resolves the drivers with.
                "gripper_camera_mode": LaunchConfiguration("enable_depth_sensor"),
                "front_camera_mode": LaunchConfiguration("enable_front_camera"),
                "rear_camera_mode": LaunchConfiguration("enable_rear_camera"),
            }.items(),
        ),
    ])
