from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch_param_builder import load_yaml
from moveit_configs_utils import MoveItConfigsBuilder
import os


def generate_launch_description():
    # Declare launch arguments
    arm_hardware_protocol_arg = DeclareLaunchArgument(
        "arm_hardware_protocol",
        default_value="mock_hardware",
        description="Hardware protocol for the arm in this launch"
    )

    gripper_type_arg = DeclareLaunchArgument(
        'gripper_type',
        default_value='new',
        description='Which gripper URDF to load'
    )
    
    use_gui_arg = DeclareLaunchArgument(
        'use_gui', 
        default_value='true', 
        description='Launch RViz with MoveIt interface'
    )

    micro_ros_device_arg = DeclareLaunchArgument(
        'micro_ros_device',
        default_value='/dev/serial/by-id/usb-Teensyduino_USB_Serial_16739090-if00',
        description='USB-serial device for the micro-ROS agent (Teensy)'
    )
    
    # Get paths
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    aries_moveit_dir = os.path.dirname(current_file_dir)
    moveit_config_dir = os.path.join(aries_moveit_dir, "config")
    controller_config_path = os.path.join(moveit_config_dir, "ros2_controllers.yaml")
    
    # URDF files are in the aries package
    aries_pkg = get_package_share_directory("aries")
    arm_hardware_protocol = LaunchConfiguration('arm_hardware_protocol')
    gripper_type = LaunchConfiguration('gripper_type')
    micro_ros_device = LaunchConfiguration('micro_ros_device')

    robot_description_content = ParameterValue(
        Command([
            FindExecutable(name='xacro'), ' ',
            PathJoinSubstitution([
                FindPackageShare("aries"), "urdf", "my_robot.urdf.xacro"
            ]),
            ' ',
            'hardware_protocol:=mock_hardware',
            ' ',
            'arm_hardware_protocol:=', arm_hardware_protocol,
            ' ',
            'gripper_hardware_protocol:=rebel',
            ' ',
            'gripper_type:=', gripper_type,
        ]),
        value_type=str
    )

    robot_description = {'robot_description': robot_description_content}
    
    # Robot semantic description (SRDF)
    robot_description_semantic_content = Command([
        'cat ', os.path.join(moveit_config_dir, 'aries.srdf')
    ])
    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(robot_description_semantic_content, value_type=str)
    }
    
    kinematics_yaml = {
        'robot_description_kinematics': load_yaml(Path(os.path.join(moveit_config_dir, 'kinematics.yaml')))
    }
    
    # OMPL Planning configuration - load as dict, not as params-file
    import yaml as _yaml
    with open(os.path.join(moveit_config_dir, 'ompl_planning.yaml')) as f:
        ompl_config = _yaml.safe_load(f)
    
    # If the yaml has a 'move_group' top-level wrapper, unwrap it and merge with sibling keys
    if 'move_group' in ompl_config:
        move_group_block = ompl_config.pop('move_group')
        ompl_config.update(move_group_block)

    # MoveIt2 Jazzy expects: planning_pipelines=['ompl'], default_planning_pipeline='ompl',
    # and pipeline config nested under 'ompl' key
    ompl_planning_yaml = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_config,
    }
    
    # Sensor manager config - disable octomap (no depth camera)
    sensor_manager_yaml = {
        'moveit_sensor_manager': 'moveit_msgs/MoveItSensorManager',
        'sensor_manager': '',
        'octomap_resolution': 0.0,
    }

    joint_limits_yaml = {
        'robot_description_planning': load_yaml(Path(os.path.join(moveit_config_dir, 'joint_limits.yaml')))
    }
    
    moveit_controllers = {
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': load_yaml(Path(os.path.join(moveit_config_dir, 'moveit_controllers.yaml'))),
    }

    # ros2_control node
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            controller_config_path
        ],
        output="both",
    )
    
    # Robot state publisher
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="both",
    )

    # Publish static rover wheel/rocker joints so MoveIt has a complete state.
    wheel_joint_publisher_node = Node(
        package='aries_moveit',
        executable='publish_wheel_joints.py',
        name='wheel_joint_publisher',
        output='screen',
    )

    # Joint state broadcaster spawner
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="both",
    )
    
    # Arm and gripper controller spawners
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_arm_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="both",
    )

    # Gripper controller spawner (spawned after arm controller)
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_gripper_controller", "--controller-manager", "/controller_manager"],
        output="both",
    )
    
    # Spawn arm controller after joint_state_broadcaster
    delay_arm_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    # Spawn gripper controller after arm controller
    delay_gripper_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner],
        )
    )
    
    # MoveGroup node for planning
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_yaml,
            moveit_controllers,
            sensor_manager_yaml,
            joint_limits_yaml,
            {
                'use_sim_time': False,
                'publish_robot_description_semantic': True,
                'publish_planning_scene': True,
                'publish_geometry_updates': True,
                'publish_state_updates': True,
                'publish_transforms_updates': True,
                'planning_scene_monitor.publish_planning_scene': True,
                'planning_scene_monitor.publish_geometry_updates': True,
                'planning_scene_monitor.publish_state_updates': True,
                'planning_scene_monitor.publish_transforms_updates': True,
                'trajectory_execution.allowed_execution_duration_scaling': 1.2,
                'trajectory_execution.allowed_goal_duration_margin': 0.5,
                'trajectory_execution.allowed_start_tolerance': 0.03,
                # Disable octomap/3D sensor monitor (no depth camera attached)
                'octomap_resolution': 0.0,
            }
        ],
    )
    
    # RViz node with MoveIt interface
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare("aries_moveit"), "config", "gripper.rviz"
    ])
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
        ],
        condition=IfCondition(LaunchConfiguration('use_gui'))
    )
    
    # micro-ROS agent: bridges Teensy USB serial to ROS 2 topics
    micro_ros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['serial', '--dev', micro_ros_device, '-b', '6000000'],
        output='screen',
    )

    return LaunchDescription([
        arm_hardware_protocol_arg,
        gripper_type_arg,
        use_gui_arg,
        micro_ros_device_arg,
        micro_ros_agent_node,
        ros2_control_node,
        robot_state_pub_node,
        wheel_joint_publisher_node,
        joint_state_broadcaster_spawner,
        delay_arm_spawner,
        delay_gripper_spawner,
        move_group_node,
        rviz_node,
    ])
