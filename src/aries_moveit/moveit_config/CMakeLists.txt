cmake_minimum_required(VERSION 3.22)
project(aries_moveit)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(control_msgs)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(rebel_servo_teleop_keyboard src/rebel_servo_teleop_keyboard.cpp)
ament_target_dependencies(rebel_servo_teleop_keyboard 
  rclcpp 
  control_msgs
  geometry_msgs
  moveit_msgs
)

add_executable(rebel_servo_teleop_gamepad src/rebel_servo_teleop_gamepad.cpp)
ament_target_dependencies(rebel_servo_teleop_gamepad 
  rclcpp 
  control_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  std_msgs
  moveit_msgs
  std_srvs
  moveit_core
  moveit_ros_planning
)

add_executable(servo_collision_guard src/servo_collision_guard.cpp)
ament_target_dependencies(servo_collision_guard
  rclcpp
  sensor_msgs
  std_msgs
  trajectory_msgs
  moveit_core
  moveit_ros_planning
)

install(TARGETS
  rebel_servo_teleop_keyboard
  rebel_servo_teleop_gamepad
  servo_collision_guard
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/joy_layout_normalizer.py
  scripts/publish_wheel_joints.py
  scripts/rebel_movegroup_joystick.py
  scripts/gripper_arc_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Declare ament package
ament_package()
