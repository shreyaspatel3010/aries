cmake_minimum_required(VERSION 3.22)
project(aries_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python executable nodes
install(
  PROGRAMS
    nodes/full_hardware_checker.py
    nodes/mock_rover_drive.py
    nodes/custom_joystick_controller.py
    nodes/hardware_checker.py
    nodes/odrive_can_poller.py
    nodes/rover_cmd_vel_joystick.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Optional CAN setup reference files
install(
  DIRECTORY setup
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
