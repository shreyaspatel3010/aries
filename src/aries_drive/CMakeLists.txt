cmake_minimum_required(VERSION 3.22)
project(aries_drive)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(
  PROGRAMS
    nodes/cmd_vel_odrive_bridge.py
    nodes/odrive_can_poller.py
    nodes/mock_rover_drive.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# Reference sudoers rule for bringing up the CAN interface without a prompt.
install(
  DIRECTORY setup
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(cmd_vel_odrive_bridge test/test_cmd_vel_odrive_bridge.py)
endif()

ament_package()
