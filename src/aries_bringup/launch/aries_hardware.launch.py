import glob
import os
import subprocess

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

from aries_common.devices import device_str
from aries_common.gripper_cal import cal_str


def _serials_from_librealsense():
    """Camera serials as librealsense reports them, or None if it could not be asked.

    This has to come from librealsense, NOT from the USB descriptor in sysfs. On
    the D435i the sysfs `serial` is the *ASIC* serial, which is a different
    number from the camera serial the driver matches on -- the wrist camera
    fitted on 2026-08-12 reads 221123061847 in sysfs and 216322070216 in
    librealsense. Pinning the sysfs value makes rs_launch retry
    "The requested device with serial number ... is NOT found" forever against a
    camera that is plugged in and healthy. Older units happened to report the
    same number in both places, which is why sysfs looked correct for so long.

    Enumeration runs in a subprocess so the launch process never holds a handle
    on a device the driver is about to open.
    """
    try:
        out = subprocess.run(
            ["rs-enumerate-devices", "-s"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = None

    if out:
        serials = []
        for line in out.splitlines()[1:]:  # first line is the column header
            # "Intel RealSense D435I   216322070216   5.17.3.10" -- the serial is
            # the only all-digit field; firmware versions carry dots.
            for field in line.split():
                if field.isdigit() and len(field) >= 8 and field not in serials:
                    serials.append(field)
                    break
        if serials:
            return sorted(serials)

    try:
        import pyrealsense2 as rs
    except ImportError:
        return None
    try:
        found = [
            dev.get_info(rs.camera_info.serial_number)
            for dev in rs.context().query_devices()
        ]
    except RuntimeError:
        return None
    return sorted(set(found)) if found else None


def _find_realsense_devices():
    """Serial numbers of every Intel RealSense D4xx currently on USB, sorted.

    Sorting matters: it is the only thing that makes auto-assignment repeatable
    across reboots. USB enumeration order is not stable, so without a serial the
    driver binds to whichever device librealsense happens to find first -- with
    two cameras plugged in, the front camera could come up publishing
    /gripper_camera/* and the whole grasp stack would be looking at the wrong end
    of the robot.

    Falls back to counting devices in sysfs when librealsense cannot be reached.
    Those serials are reported as empty strings rather than the ASIC serial that
    lives there: an unpinnable camera still runs, a mis-pinned one never starts.
    """
    serials = _serials_from_librealsense()
    if serials is not None:
        return serials

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
        serials.append("")
    return serials


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
        # 30 fps, not 15. A frame cannot be delivered before the sensor has
        # finished producing it, so the capture rate sets the floor on latency
        # for everything downstream: at 15 fps a frame reaches the first ROS
        # topic ~67 ms old (measured 69.6 ms), which was 91% of the whole
        # glass-to-operator budget. At 30 fps that floor halves.
        #
        # This costs no link bandwidth. The downlink's rate gate still runs at
        # downlink_rate_hz and simply picks every other frame; what changes is
        # that the frame it picks is half a period fresher.
        "rgb_camera.color_profile": "640,480,30",
        "depth_module.depth_profile": "640,480,30",
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


def _usb_cam_driver(camera_name, device, framerate="15.0"):
    """usb_cam on a plain UVC webcam. The rear camera, and only the rear camera.

    NOT a RealSense, and nothing about the realsense2_camera path applies: no
    depth, no align_depth, no serial to match on, and no enumeration to check
    it against. It is addressed by DEVICE PATH, and that path must be the
    /dev/v4l/by-id/ one -- see devices.yaml for why a pinned /dev/videoN opens
    the wrong camera rather than failing.

    The node runs in its own namespace so it publishes <camera_name>/image_raw
    and <camera_name>/camera_info. Note the missing /color/ segment: that is
    usb_cam's naming, it is deliberate, and rear_camera.xacro's gz sensor uses
    the same names so the two ends agree.

    pixel_format is mjpeg2rgb, not yuyv2rgb. Both are offered by the Brio, but
    YUYV at 640x480x15 is 74 Mbit/s of USB for a camera sharing a controller
    with two D435is; MJPEG is roughly a tenth of that, at the cost of a JPEG
    decode per frame (~1-2 ms at this size). Bandwidth is the scarcer resource
    here.

    15 fps, matching the gz sensor in rear_camera.xacro. The 30 fps argument
    made for the D435is -- that capture rate sets the latency floor -- carries
    much less weight for this camera: the downlink gate below it runs at 5 Hz,
    so 30 fps would buy ~33 ms of freshness on a view of a carriage that moves
    at 0.05 m/s, and cost twice the decode.

    THE by-id PATH IS RESOLVED HERE, and it has to be. usb_cam does not accept
    a /dev/v4l/by-id/ symlink: it dereferences the link itself and joins the
    result against /dev/ rather than against the link's own directory, so
    /dev/v4l/by-id/usb-046d_Brio_100_...-video-index0 -> ../../video0 comes out
    as the nonexistent `/dev/../../video0` and the node exits with

        Device specified is not available or is not a vaild V4L2 device

    which reads exactly like an unplugged camera. os.path.realpath does the
    dereference correctly, so devices.yaml keeps pinning the STABLE identity
    (which is the point -- this camera was observed moving from /dev/video4 to
    /dev/video0 across a replug) and the launch turns it into whatever node
    that camera is right now. Never put a bare /dev/videoN in devices.yaml to
    work around this: a stale one opens a different camera instead of failing.

    NOT CALIBRATED, but the path is left open. camera_info_url is unset, and an
    empty URL is not "no calibration file" -- camera_info_manager expands it to
    its default, file://${ROS_HOME}/camera_info/${NAME}.yaml, i.e.
    ~/.ros/camera_info/rear_camera.yaml. Nothing has written that file, so the
    CameraInfo published today carries the frame size and zeros in K, which is
    all a view stream needs. Run the stock camera_calibration node against this
    camera and it writes exactly that path, and the intrinsics appear on the
    next start with nothing here to change.

    Until then, do not measure anything off this image. rear_camera.xacro's
    simulated FOV is a catalogue figure for the same reason.
    """
    # realpath, not the by-id path itself -- see above. Resolved at launch, so a
    # camera that re-enumerated between runs is still found.
    return Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="rear_camera_driver",
        namespace=camera_name,
        output="screen",
        parameters=[{
            "video_device": os.path.realpath(device) if device else device,
            "camera_name": camera_name,
            "frame_id": f"{camera_name}_optical_frame",
            "pixel_format": "mjpeg2rgb",
            "image_width": 640,
            "image_height": 480,
            "framerate": float(framerate),
            "io_method": "mmap",
        }],
    )


def launch_setup(context, *args, **kwargs):
    actions = []
    depth_sensor_mode = LaunchConfiguration("enable_depth_sensor").perform(context).lower()
    front_camera_mode = LaunchConfiguration("enable_front_camera").perform(context).lower()
    gripper_camera_serial = LaunchConfiguration("gripper_camera_serial").perform(context).strip()
    front_camera_serial = LaunchConfiguration("front_camera_serial").perform(context).strip()

    rear_camera_mode = LaunchConfiguration("enable_rear_camera").perform(context).lower()
    rear_camera_device = LaunchConfiguration("rear_camera_device").perform(context).strip()

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

    # Empty entries are devices counted in sysfs when librealsense could not be
    # asked; they say a camera is there but not which one it is.
    identified_serials = [s for s in detected_serials if s]

    if len(detected_serials) > 1:
        actions.append(
            LogInfo(msg=(
                "[aries_hardware] {} RealSense devices found: {}. Using {} as the "
                "gripper camera and {} as the front camera. Pin them with "
                "gripper_camera_serial:=<serial> front_camera_serial:=<serial> if "
                "that is the wrong way round.".format(
                    len(detected_serials),
                    ", ".join(s or "<unidentified>" for s in detected_serials),
                    gripper_camera_serial or "<none>",
                    front_camera_serial or "<none>",
                )
            ))
        )

    # "auto" follows what is actually plugged in; "true" starts the driver anyway
    # so a camera behind a hub that hides its sysfs serial can still be used.
    #
    # A pinned serial that no *identified* device matches is not evidence the
    # camera is absent when nothing could be identified in the first place: the
    # sysfs fallback only ever counts. Fall back to that count there, and let the
    # driver do the matching itself -- it enumerates the cameras properly.
    def _present(serial, needed_devices):
        if identified_serials:
            return serial in detected_serials
        return len(detected_serials) >= needed_devices

    if depth_sensor_mode == "auto":
        enable_depth_sensor = _present(gripper_camera_serial, 1)
    else:
        enable_depth_sensor = depth_sensor_mode == "true"

    if front_camera_mode == "auto":
        enable_front_camera = _present(front_camera_serial, 2)
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
        actions.append(_realsense_driver("rover_camera", front_camera_serial))

    # "auto" follows whether the device node is actually there. os.path.exists
    # follows the symlink, so a by-id path left behind by an unplugged camera
    # reads as absent -- which is the answer wanted. Starting usb_cam on a
    # missing device is not a clean failure: the node comes up, fails to open
    # it, and either exits or sits there publishing nothing.
    if rear_camera_mode == "auto":
        enable_rear_camera = bool(rear_camera_device) and os.path.exists(rear_camera_device)
    else:
        enable_rear_camera = rear_camera_mode == "true"

    if enable_rear_camera and not rear_camera_device:
        actions.append(LogInfo(msg=(
            "[aries_hardware] Rear camera requested but rear_camera_device is empty. "
            "Skipping it -- pass rear_camera_device:=/dev/v4l/by-id/<path>.")))
        enable_rear_camera = False

    if enable_rear_camera:
        actions.append(_usb_cam_driver("rear_camera", rear_camera_device,
                                       LaunchConfiguration("rear_camera_framerate")
                                       .perform(context)))
    elif rear_camera_mode == "auto":
        actions.append(LogInfo(msg=(
            "[aries_hardware] No rear camera at {}. The drill view is not "
            "available; everything else comes up normally.".format(
                rear_camera_device or "<no device set>"))))

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
        #
        # DEAD while aries_vision_grasp is isolated (COLCON_IGNORE, 2026-08-22):
        # the package is not built, so enable_yolo_debug:=true fails to find the
        # executable. Left in place because removing the COLCON_IGNORE is all it
        # takes to bring both back.
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

    # Operator downlink. RViz keeps four camera displays live, and subscribed
    # raw that is 369 Mbit/s of pixels on Reliable QoS -- far past what the
    # antenna carries, so the link congestion-collapses and everything on it
    # goes laggy, not just the images. These nodes publish a second, reduced and
    # compressed copy for the operator (~3 Mbit/s for both cameras) and leave
    # the driver topics the grasp pipeline reads completely alone.
    #
    # Only cameras that actually came up get one: a downlink for an absent
    # camera would sit there warning about a stream that is never going to
    # arrive.
    downlink_cameras = []
    if enable_depth_sensor:
        downlink_cameras.append("gripper_camera")
    if enable_front_camera:
        downlink_cameras.append("rover_camera")
    # Colour only, so it gets a two-node chain and its own frame rate. The name
    # is passed to both ends as `color_only`; the rover side would otherwise
    # start a depth republisher on a stream that does not exist, and the
    # operator side a decompressor waiting on it forever.
    if enable_rear_camera:
        downlink_cameras.append("rear_camera")

    enable_downlink = LaunchConfiguration("enable_camera_downlink").perform(context).lower()
    if enable_downlink == "true" and downlink_cameras:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("aries_bringup"),
                        "launch",
                        "camera_downlink.launch.py",
                    ])
                ),
                launch_arguments={
                    "cameras": ",".join(downlink_cameras),
                    "color_only": "rear_camera",
                    "downlink_profile": LaunchConfiguration("downlink_profile"),
                    "downlink_rate_hz": LaunchConfiguration("downlink_rate_hz"),
                    "downlink_depth_rate_hz": LaunchConfiguration("downlink_depth_rate_hz"),
                    "downlink_decimation": LaunchConfiguration("downlink_decimation"),
                    "downlink_jpeg_quality": LaunchConfiguration("downlink_jpeg_quality"),
                    "downlink_depth_max_m": LaunchConfiguration("downlink_depth_max_m"),
                    "downlink_depth_quantization_mm": LaunchConfiguration(
                        "downlink_depth_quantization_mm"
                    ),
                }.items(),
            )
        )

        # moveit.rviz reads /<camera>/view/*, which only exists where the
        # decompressors run. When RViz comes up here (use_gui:=true) start them
        # here too, otherwise the displays sit empty. On an operator laptop this
        # is the one thing that has to be launched: see camera_view.launch.py.
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("aries_bringup"),
                        "launch",
                        "camera_view.launch.py",
                    ])
                ),
                condition=IfCondition(LaunchConfiguration("use_gui")),
                launch_arguments={
                    "cameras": ",".join(downlink_cameras),
                    "color_only": "rear_camera",
                }.items(),
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
                "use_joy_node": LaunchConfiguration("use_joy_node"),
                "joy_driver": LaunchConfiguration("joy_driver"),
                "joy_layout": LaunchConfiguration("joy_layout"),
                "joy_dev": LaunchConfiguration("joy_dev"),
                "joystick_control_mode": LaunchConfiguration("joystick_control_mode"),
                "serial_port": LaunchConfiguration("serial_port"),
                # The SECOND Teensy. Forwarded explicitly, like everything else
                # here -- this include passes a named dict, so an argument that
                # is not in it does not reach the moveit launch reliably.
                "use_science": LaunchConfiguration("use_science"),
                "science_serial_port": LaunchConfiguration("science_serial_port"),
                "servo_bus_port": LaunchConfiguration("servo_bus_port"),
                "servo_bus_baud": LaunchConfiguration("servo_bus_baud"),
                "servo_id": LaunchConfiguration("servo_id"),
                "gripper_closed_steps": LaunchConfiguration("gripper_closed_steps"),
                "gripper_servo_invert": LaunchConfiguration("gripper_servo_invert"),
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
        DeclareLaunchArgument("gripper_type", default_value="st3215", choices=["st3215"]),
        DeclareLaunchArgument("finger_type", default_value="bucket", choices=["bucket", "maintenance"]),
        DeclareLaunchArgument("arm_hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("hardware_protocol", default_value="auto", choices=["auto", "rebel", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("gripper_hardware_protocol", default_value="auto", choices=["auto", "st3215", "mock_hardware", "gazebo"]),
        DeclareLaunchArgument("use_joystick", default_value="true"),
        # Whether the pad is READ here, as opposed to whether teleop runs here.
        # false when the joystick is on the base station: the consumers below
        # stay up and take /joy over the link. See aries_comms.
        DeclareLaunchArgument("use_joy_node", default_value="true"),
        DeclareLaunchArgument("joy_driver", default_value="game_controller_node", choices=["game_controller_node", "joy_node"]),
        DeclareLaunchArgument("joy_layout", default_value="auto", choices=["auto", "dongle", "bluetooth", "game_controller", "passthrough"]),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("joystick_control_mode", default_value="servo", choices=["move_group", "servo"]),
        DeclareLaunchArgument("serial_port", default_value=device_str("gripper.serial_port")),
        DeclareLaunchArgument("use_science", default_value="true",
                              description="Start the micro-ROS agent for the science board (the second Teensy)."),
        DeclareLaunchArgument("science_serial_port", default_value=device_str("science.serial_port"),
                              description="USB-serial port for the science Teensy. Empty until science.serial_port is set in devices.yaml."),
        DeclareLaunchArgument("servo_bus_port", default_value=device_str("servo_bus.port")),
        DeclareLaunchArgument("servo_bus_baud", default_value=device_str("servo_bus.baud")),
        DeclareLaunchArgument("servo_id", default_value=device_str("servo_bus.gripper_servo_id")),
        DeclareLaunchArgument("gripper_closed_steps", default_value=cal_str("closed_steps")),
        DeclareLaunchArgument("gripper_servo_invert", default_value=cal_str("invert"), choices=["true", "false"]),
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
            default_value=device_str("cameras.gripper_serial"),
            description=(
                "Camera serial of the gripper-mounted D435i, as reported by "
                "rs-enumerate-devices -s (NOT the sysfs serial). Defaults to the "
                "fitted wrist camera so the two cameras cannot trade places "
                "between launches. Empty auto-assigns the lowest detected serial, "
                "which is only unambiguous with a single camera connected."
            ),
        ),
        DeclareLaunchArgument(
            "enable_front_camera",
            default_value="auto",
            choices=["auto", "true", "false"],
            description="Start a driver for the rover front camera (publishes /rover_camera/*)",
        ),
        DeclareLaunchArgument(
            "front_camera_serial",
            default_value=device_str("cameras.front_serial"),
            description=(
                "Camera serial of the rover front D435i, as reported by "
                "rs-enumerate-devices -s. Empty takes whichever detected camera "
                "the gripper did not claim."
            ),
        ),
        DeclareLaunchArgument(
            "enable_rear_camera",
            default_value="auto",
            choices=["auto", "true", "false"],
            description="Start the rear Logitech Brio (publishes /rear_camera/*). "
                        "auto starts it when rear_camera_device exists. This is a "
                        "colour-only view camera aimed at the drill -- there is no "
                        "depth and nothing but the operator reads it.",
        ),
        DeclareLaunchArgument(
            "rear_camera_device",
            default_value=device_str("cameras.rear_device"),
            description=(
                "V4L2 device for the rear camera. Always the /dev/v4l/by-id/ "
                "path, never /dev/videoN: the numbering moves and a stale one "
                "opens a different camera instead of failing. Must be the "
                "-video-index0 node; index1 is metadata and never delivers a "
                "frame."
            ),
        ),
        DeclareLaunchArgument(
            "rear_camera_framerate",
            default_value="15.0",
            description="Rear camera capture rate. The downlink gates it to "
                        "downlink_color_only_rate_hz (5 Hz) regardless; raising "
                        "this only buys latency, at a JPEG decode per frame.",
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
            "enable_camera_downlink",
            default_value="true",
            choices=["true", "false"],
            description="Publish the reduced+compressed operator camera streams. "
                        "Turn off only when nothing views the rover remotely -- the "
                        "encoders are lazy and idle until something subscribes.",
        ),
        DeclareLaunchArgument("downlink_profile", default_value="balanced",
                              choices=["quality", "balanced", "lean"]),
        DeclareLaunchArgument("downlink_rate_hz", default_value="15.0"),
        DeclareLaunchArgument("downlink_depth_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("downlink_decimation", default_value="profile"),
        DeclareLaunchArgument("downlink_jpeg_quality", default_value="profile"),
        DeclareLaunchArgument("downlink_depth_max_m", default_value="6.0"),
        DeclareLaunchArgument("downlink_depth_quantization_mm", default_value="10"),
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
        DeclareLaunchArgument("rover_imu_port", default_value=device_str("imu.port")),
        DeclareLaunchArgument("rover_imu_baudrate", default_value="115200"),
        DeclareLaunchArgument("rover_imu_frame", default_value="imu_frame"),
        DeclareLaunchArgument(
            "rover_imu_topic", default_value="/microstrain/imu/data"
        ),
        DeclareLaunchArgument("can_interface", default_value=device_str("rover.can_interface")),
        DeclareLaunchArgument("setup_rover_can", default_value="true"),

        OpaqueFunction(function=launch_setup),
    ])
