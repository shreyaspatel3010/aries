cmake_minimum_required(VERSION 3.22)
project(aries_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python executable nodes
install(
  PROGRAMS
    nodes/camera_downlink.py
    nodes/downlink_report.py
    nodes/full_hardware_checker.py
    nodes/mock_rover_drive.py
    nodes/hardware_checker.py
    nodes/odrive_can_poller.py
    nodes/publish_wheel_joints.py
    nodes/stacklight.py
  DESTINATION lib/${PROJECT_NAME}
)

# stacklight.yaml is the state -> colour table the stacklight node reads.
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    rover_drive_launch_contract
    test/test_rover_drive_launch_contract.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    wheel_encoder_joint_states
    test/test_wheel_encoder_joint_states.py
    TIMEOUT 60
  )
  # The stack light must agree with the Teensy firmware's colour codes, and it
  # must never show green on no evidence.
  ament_add_pytest_test(
    stacklight
    test/test_stacklight.py
    TIMEOUT 60
  )
endif()

ament_package()
