#!/usr/bin/env python3
"""The rover's three load cells, as weights in kilograms.

    sand_box         front-left deck box, the sand sample
    stone_box        the box behind it, also on the left, the stone sample
    drill_container  the drill's sample bin

All three hang off the drill/science Teensy, so they arrive over the micro-ROS link
aries_hardware.launch.py already brings up -- this launch starts only the node
that turns the firmware's raw counts into kilograms. It does not start an agent
and does not care which machine one is running on.

    ros2 topic echo /load_cells/status
    ros2 topic echo /load_cells/sand_box/weight

THE BIN'S CELL IS ONLY UNDER THE BIN WHEN IT IS PARKED, so
/load_cells/drill_container/valid says whether that weight means anything and
/load_cells/drill_container/weight_held carries the last one that did. See
config/load_cells.yaml.

The firmware is still being written. To see the topic layout without it:

    ros2 launch aries_load_cells load_cells.launch.py load_cell_source:=mock

`auto` never falls back to that on its own -- a fabricated weight indis-
tinguishable from a real one is not something this should ever emit by itself.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "load_cells_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("aries_load_cells"), "config", "load_cells.yaml",
            ]),
            description="Per-cell calibration and the drill bin's gate.",
        ),
        DeclareLaunchArgument(
            "load_cell_source", default_value="auto",
            choices=["auto", "microros", "mock"],
            description="auto/microros read the Teensy; mock makes up counts "
                        "so the topics can be exercised with no hardware.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false",
            description="Sim clock. The staleness and settle windows are in "
                        "seconds, so this has to match the rest of the stack.",
        ),
        Node(
            package="aries_load_cells",
            executable="load_cells.py",
            name="load_cells",
            output="screen",
            parameters=[
                LaunchConfiguration("load_cells_config"),
                # After the config file, so the argument wins over it.
                {"source": LaunchConfiguration("load_cell_source")},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
