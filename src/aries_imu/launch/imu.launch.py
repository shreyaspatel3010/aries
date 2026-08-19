#!/usr/bin/env python3
"""Start the MicroStrain by HBK 3DM-GX5-AHRS, or nothing if it is absent.

The rover carries exactly one IMU. There is no source selection any more: the
driver starts when the device node exists and the driver package is installed,
and otherwise the stack falls back to wheel odometry only, which
aries_localization decides on its own using the same aries_common probe.

Topics land under the ``microstrain`` namespace, so the physical IMU publishes
``/microstrain/imu/data`` and never collides with Gazebo's simulated ``/imu``.

The GX5-AHRS is an attitude and heading reference, not a bare gyro pack: its
onboard estimation filter supplies absolute roll, pitch and heading. That is
why the EKF fuses yaw rather than yaw rate alone.

No static transform is published here. base_link -> imu_frame is a fixed joint
in aries_base.xacro, so robot_state_publisher already owns it; the old BNO055
path published a competing one, and the driver would too if
publish_mount_to_frame_id_transform were left at its default.

The driver's own params.yml is loaded first, the same way its
microstrain_launch.py does it, because the node expects the full set of
several hundred declared parameters. config/microstrain.yaml then overrides
only what differs, and launch arguments override port/baudrate/frame on top.
Running the node directly rather than including microstrain_launch.py is what
lets those last overrides through: that launch file exposes only namespace,
node_name, params_file and debug.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aries_common.detect import AUTO_VALUES, as_bool, as_int, resolve_imu_source

from aries_common.devices import device_str

# Must match the key inside config/microstrain.yaml.
DRIVER_NAMESPACE = "microstrain"
DRIVER_NODE_NAME = "microstrain_inertial_driver"


def _driver_defaults():
    """The driver's shipped params.yml, as a flat dict of parameter values.

    microstrain_launch.py loads this itself instead of passing the path,
    because the file has no node-name key and so is not a valid ROS parameter
    file on its own.
    """
    path = os.path.join(
        get_package_share_directory("microstrain_inertial_driver"),
        "microstrain_inertial_driver_common",
        "config",
        "params.yml",
    )
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _start_imu(context, *args, **kwargs):
    start_mode = LaunchConfiguration("start_imu_driver").perform(context)
    driver_enabled = (
        str(start_mode).strip().lower() in AUTO_VALUES or as_bool(start_mode)
    )
    use_imu = LaunchConfiguration("use_imu").perform(context)
    imu_port = LaunchConfiguration("imu_port").perform(context)

    source, imu_present = resolve_imu_source(use_imu, imu_port)
    if not driver_enabled:
        source = "none"

    actions = [
        LogInfo(
            msg=(
                f"[rover imu] use_imu={use_imu} selected={source}; "
                f"microstrain_available={imu_present} "
                f"port={imu_port} driver_enabled={driver_enabled}"
            )
        )
    ]

    if source != "microstrain":
        return actions

    microstrain_config = os.path.join(
        get_package_share_directory("aries_imu"), "config", "microstrain.yaml"
    )
    actions.append(
        Node(
            package="microstrain_inertial_driver",
            executable="microstrain_inertial_driver_node",
            name=DRIVER_NODE_NAME,
            namespace=DRIVER_NAMESPACE,
            output="screen",
            parameters=[
                _driver_defaults(),
                microstrain_config,
                {
                    "port": imu_port,
                    "baudrate": as_int(
                        LaunchConfiguration("imu_baudrate").perform(context),
                        115200,
                    ),
                    # 4.8.0 has a single frame_id; there is no imu_frame_id or
                    # filter_frame_id despite what older docs suggest.
                    "frame_id": LaunchConfiguration("imu_frame"),
                },
            ],
            # The IMU is the heading reference for the EKF; a USB hiccup should
            # not silently downgrade the rover to wheel odometry for the rest
            # of the run.
            respawn=True,
            respawn_delay=5.0,
        )
    )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_imu_driver",
                default_value="auto",
                choices=["auto", "true", "false"],
            ),
            DeclareLaunchArgument(
                "use_imu",
                default_value="auto",
                choices=["auto", "true", "false", "microstrain"],
            ),
            DeclareLaunchArgument(
                "imu_port",
                default_value=device_str("imu.port"),
                description=(
                    "Persistent symlink created by the driver package's own "
                    "60-ros-jazzy-microstrain-inertial-driver.rules. Not "
                    "/dev/microstrain: that plainer symlink is commented out "
                    "upstream. Prefer this over a bare /dev/ttyACM index, "
                    "which the Teensy gripper also claims."
                ),
            ),
            DeclareLaunchArgument(
                "imu_baudrate",
                default_value="115200",
                description="3DM-GX5 factory default.",
            ),
            DeclareLaunchArgument(
                "imu_frame",
                default_value="imu_frame",
                description="Matches the fixed imu_frame link in aries_base.xacro.",
            ),
            DeclareLaunchArgument(
                "imu_topic",
                default_value="/microstrain/imu/data",
                description=(
                    "Topic the EKF and hardware checker consume. Declared here "
                    "so callers forward one consistent name; if your installed "
                    "driver version publishes orientation on the estimation "
                    "filter topic instead, point this at "
                    "/microstrain/ekf/imu/data."
                ),
            ),
            OpaqueFunction(function=_start_imu),
        ]
    )
