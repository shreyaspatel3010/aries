cmake_minimum_required(VERSION 3.22)
project(aries_teleop)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(
  PROGRAMS
    nodes/joy_layout_normalizer.py
    nodes/custom_joystick_controller.py
    nodes/rover_cmd_vel_joystick.py
    nodes/cmd_vel_teleop_relay.py
    nodes/drill_joystick.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# joystick.yaml is also read by aries_drive's mock_rover_drive.
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # The rover must stop when /joy goes quiet. Both drive nodes publish on a
  # timer carrying the last joystick state, which over the antenna makes a
  # dropout look exactly like a held stick.
  ament_add_pytest_test(
    joy_watchdog
    test/test_joy_watchdog.py
    TIMEOUT 120
  )
endif()

ament_package()
