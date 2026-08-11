import glob
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _find_realsense_devices():
    """Serial numbers of every Intel RealSense D4xx currently on USB, sorted.

    Sorting matters: it is the only thing that makes auto-assignment repeatable
    across reboots. USB enumeration order is not stable, so without a serial the
    driver binds to whichever device librealsense happens to find first -- with
    two cameras plugged in, the front camera could come up publishing
    /gripper_camera/* and the whole grasp stack would be looking at the wrong end
    of the robot.

    A device whose sysfs node has no readable serial is still reported, as an
    empty string, so it is never silently dropped from detection -- it just
    cannot be pinned, and its driver runs the way it did before serials were
    read at all.
    """
    serials = []
    for dev_path in glob.glob("/sys/bus/usb/devices/*/"):
        try:
            if open(os.path.join(dev_path, "idVendor")).read().strip() != "8086":
                continue
            product = int(open(os.path.join(dev_path, "idProduct")).read().strip(), 16)
            if not 0x0AD1 <= product <= 0x0B64:
                continue
        except (OSError, ValueError):
            continue
        try:
            serial = open(os.path.join(dev_path, "serial")).read().strip()
        except OSError:
            serial = ""
        if serial and serial in serials:
            continue
        serials.append(serial)
    return sorted(serials)


def _realsense_driver(camera_name, serial):
    """rs_launch.py configured to match the simulated sensor of the same name.

    Colour and depth are forced to the same 640x480x15 profile the URDF renders,
    so a node that indexes the depth image with colour pixel coordinates behaves
    identically in simulation and on hardware. TF comes from the robot
    description, never from the driver. No IMU: gyro/accel are unused stack-wide
    and the URDF no longer carries the frames for them.
    """
    launch_arguments = {
        "camera_name": camera_name,
        "camera_namespace": "",
        "enable_color": "true",
        "enable_depth": "true",
        "enable_infra": "false",
        "enable_infra1": "false",
        "enable_infra2": "false",
        "enable_gyro": "false",
        "enable_accel": "false",
        "rgb_camera.color_profile": "640,480,15",
        "depth_module.depth_profile": "640,480,15",
        "align_depth.enable": "true",
        "publish_tf": "false",
        "initial_reset": "true",
        "output": "screen",
    }
    if serial:
        # The wrapper expects the serial prefixed with an underscore so launch
        # cannot coerce an all-digit serial into a number.
        launch_arguments["serial_no"] = "_" + serial
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
        ),
        launch_arguments=launch_arguments.items(),
    )


def launch_setup(context, *args, **kwargs):
    actions = []
    depth_sensor_mode = LaunchConfiguration("enable_depth_sensor").perform(context).lower()
    front_camera_mode = LaunchConfiguration("enable_front_camera").perform(context).lower()
    gripper_camera_serial = LaunchConfiguration("gripper_camera_serial").perform(context).strip()
    front_camera_serial = LaunchConfiguration("front_camera_serial").perform(context).strip()

    detected_serials = _find_realsense_devices()

    # Explicit serials always win. Anything left over is assigned gripper first,
    # because the gripper camera is the one the grasp stack cannot run without.
    unclaimed = [
        s for s in detected_serials
        if s not in (gripper_camera_serial, front_camera_serial)
    ]
    if not gripper_camera_serial and unclaimed:
        gripper_camera_serial = unclaimed.pop(0)
    if not front_camera_serial and unclaimed:
        front_camera_serial = unclaimed.pop(0)

    if len(detected_serials) > 1:
        actions.append(
            LogInfo(msg=(
                "[aries_hardware] {} RealSense devices found: {}. Using {} as the "
                "gripper camera and {} as the front camera. Pin them with "
                "gripper_camera_serial:=<serial> front_camera_serial:=<serial> if "
                "that is the wrong way round.".format(
                    len(detected_serials),
                    ", ".join(detected_serials),
                    gripper_camera_serial or "<none>",
                    front_camera_serial or "<none>",
                )
            ))
        )

    # "auto" follows what is actually plugged in; "true" starts the driver anyway
    # so a camera behind a hub that hides its sysfs serial can still be used.
    if depth_sensor_mode == "auto":
        enable_depth_sensor = gripper_camera_serial in detected_serials
    else:
        enable_depth_sensor = depth_sensor_mode == "true"

    if front_camera_mode == "auto":
        enable_front_camera = front_camera_serial in detected_serials
    else:
        enable_front_camera = front_camera_mode == "true"

    # Two unpinned drivers would race for the same device, and the loser exits
    # after locking up the camera the grasp stack needs. Never risk that.
    if enable_depth_sensor and enable_front_camera and not front_camera_serial:
        actions.append(
            LogInfo(msg=(
                "[aries_hardware] Front camera requested but no serial to bind it to, "
                "and the gripper camera driver is already running. Skipping it -- pass "
                "front_camera_serial:=<serial> to run both."
            ))
        )
        enable_front_camera = False

    if enable_depth_sensor:
        actions.append(_realsense_driver("gripper_camera", gripper_camera_serial))

    if enable_front_camera:
        # Front/rover camera supplies its own independent colored DepthCloud.
        actions.append(_realsense_driver("camera", front_camera_serial))

    if not enable_depth_sensor:
        actions.append(
            LogInfo(msg=(
                "[aries_hardware] No gripper RealSense{}. The gripper DepthCloud and "
                "vision grasp pipeline have no wrist input; "
                "everything else comes up "
                "normally.".format(
                    " detected on USB" if depth_sensor_mode == "auto" else " (disabled)"
                )
            ))
        )

    if enable_depth_sensor:
        # Debug-only annotated detection stream. Off by default: vision_grasp_node
        # runs the same YOLO model on the same camera feed (and publishes its own
        # /vision_grasp/detection_image), so running this node too doubles the
        # GPU/CPU inference cost and model memory on the rover.
        enable_yolo_debug = (
            LaunchConfiguration("enable_yolo_debug").perform(context).lower() == "true"
        )
        if enable_yolo_debug:
            actions.append(
                Node(
                    package="aries_vision_grasp",
                    executable="yolo_detection_node.py",
                    name="yolo_detection_node",
                    output="screen",
                    parameters=[{
                        "confidence_threshold": 0.50,
                    }],
                )
            )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_moveit"),
                    "launch",
                    "aries_hardware.launch.py",
                ])
            ),
            launch_arguments={
                "use_gui": LaunchConfiguration("use_gui"),
                "gripper_type": LaunchConfiguration("gripper_type"),
                "finger_type": LaunchConfiguration("finger_type"),
                "arm_hardware_protocol": LaunchConfiguration("arm_hardware_protocol"),
                "hardware_protocol": LaunchConfiguration("hardware_protocol"),
                "gripper_hardware_protocol": LaunchConfiguration("gripper_hardware_protocol"),
                "use_joystick": LaunchConfiguration("use_joystick"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "joystick_control_mode": LaunchConfiguration("joystick_control_mode"),
                "serial_port": LaunchConfiguration("serial_port"),
                "suppress_rebel_logs": LaunchConfiguration("suppress_rebel_logs"),
                "suppress_moveit_execution_logs": LaunchConfiguration("suppress_moveit_execution_logs"),
                "use_wheel_joint_publisher": LaunchConfiguration(
                    "use_wheel_joint_publisher"
                ),
            }.items(),
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("aries_bringup"),
                    "launch",
                    "rover_drive_core.launch.py",
                ])
            ),
            condition=IfCondition(LaunchConfiguration("use_rover_drive")),
            launch_arguments={
                "use_imu": LaunchConfiguration("use_rover_imu"),
                "use_joystick": LaunchConfiguration("use_rover_joystick"),
                "use_joy_node": LaunchConfiguration("use_rover_joy_node"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "can_interface": LaunchConfiguration("can_interface"),
                "setup_can": LaunchConfiguration("setup_rover_can"),
                "imu_port": LaunchConfiguration("rover_imu_port"),
                "imu_baudrate": LaunchConfiguration("rover_imu_baudrate"),
                "imu_frame": LaunchConfiguration("rover_imu_frame"),
                "imu_topic": LaunchConfiguration("rover_imu_topic"),
            }.items(),
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("gripper_type", default_value="new", choices=["old", "new"]),
        DeclareLaunchArgument("finger_type", default_value="bucket", choices=["bucket", "maintenance", "probe"]),
        DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"]),
        DeclareLaunchArgument("serial_port", default_value="/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00"),
        DeclareLaunchArgument("suppress_rebel_logs", default_value="false"),
        DeclareLaunchArgument("suppress_moveit_execution_logs", default_value="false"),
        DeclareLaunchArgument(
            "enable_depth_sensor",
            default_value="auto",
            choices=["auto", "true", "false"],
            description="Auto-detect, force-enable, or disable the gripper RealSense used by vision and its DepthCloud",
        ),
        DeclareLaunchArgument(
            "gripper_camera_serial",
            default_value="",
            description=(
                "Serial number of the gripper-mounted D435i (see rs-enumerate-devices "
                "or lsusb -v). Empty auto-assigns the lowest detected serial, which is "
                "only unambiguous with a single camera connected."
            ),
        ),
        DeclareLaunchArgument(
            "enable_front_camera",
            default_value="auto",
            choices=["auto", "true", "false"],
            description="Start a driver for the rover front camera (publishes /camera/*)",
        ),
        DeclareLaunchArgument(
            "front_camera_serial",
            default_value="",
            description="Serial number of the rover front D435i; empty takes whichever detected camera the gripper did not claim",
        ),
        DeclareLaunchArgument(
            "use_wheel_joint_publisher",
            default_value="true",
            description=(
                "Publish static zero wheel joints. Disable when the rover "
                "ODrive encoder publisher owns the wheel joint states."
            ),
        ),
        DeclareLaunchArgument(
            "enable_yolo_debug",
            default_value="false",
            choices=["true", "false"],
            description="Run the standalone annotated-detection node; duplicates the "
                        "inference vision_grasp_node already performs, debug only",
        ),

        DeclareLaunchArgument("use_rover_drive", default_value="false"),
        DeclareLaunchArgument("use_rover_joystick", default_value="true"),
        DeclareLaunchArgument("use_rover_joy_node", default_value="false"),
        DeclareLaunchArgument("use_rover_imu", default_value="auto"),
        DeclareLaunchArgument("rover_imu_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("rover_imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("rover_imu_frame", default_value="bno055"),
        DeclareLaunchArgument("rover_imu_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("setup_rover_can", default_value="true"),

        OpaqueFunction(function=launch_setup),
    ])
