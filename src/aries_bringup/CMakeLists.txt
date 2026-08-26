cmake_minimum_required(VERSION 3.22)
project(aries_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python executable nodes
install(
  PROGRAMS
    nodes/camera_downlink.py
    nodes/downlink_report.py
    nodes/drill_driver.py
    nodes/full_hardware_checker.py
    nodes/mock_rover_drive.py
    nodes/hardware_checker.py
    nodes/odrive_can_poller.py
    nodes/publish_wheel_joints.py
    nodes/stacklight.py
    nodes/stacklight_gz_visual.py
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
  # must never show green on no evidence. The drill firmware arrived with red
  # and green transposed, which would have shown GREEN for an e-stop.
  ament_add_pytest_test(
    stacklight
    test/test_stacklight.py
    TIMEOUT 60
  )
  # The drill driver is the only thing between the joystick's physical rates and
  # the Teensy's duty cycle, and every way it can disagree with the firmware --
  # the actuator's free speed, the output topic names -- is silent at runtime.
  ament_add_pytest_test(
    drill_driver
    test/test_drill_driver.py
    TIMEOUT 60
  )
  # Gazebo depth is 32FC1 metres, not the D435i's 16UC1 millimetres. Rejecting
  # it dropped the whole synchronised pair and blanked the operator view in sim.
  ament_add_pytest_test(
    camera_downlink_depth
    test/test_camera_downlink_depth.py
    TIMEOUT 60
  )
  # The rear camera has no depth sensor. Fed to the paired path, its single
  # stream would sit in a two-input synchroniser that never fires -- silently,
  # with both topics looking healthy.
  ament_add_pytest_test(
    camera_downlink_color_only
    test/test_camera_downlink_color_only.py
    TIMEOUT 60
  )
endif()

ament_package()
