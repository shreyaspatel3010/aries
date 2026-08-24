cmake_minimum_required(VERSION 3.22)
project(aries_common)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# config/devices.yaml: the device table every launch file reads its defaults from.
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# aries_dds_env.sh is SOURCED, never executed, so it goes in share/ rather than
# lib/ -- an executable copy would invite `ros2 run`, which sets the variables
# in a subshell that exits immediately.
install(FILES scripts/aries_dds_env.sh DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    hardware_detection
    test/test_detect.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    comms_config
    test/test_comms.py
    TIMEOUT 60
  )
  # Pins devices.yaml's arm endpoint to the one compiled into the driver.
  ament_add_pytest_test(
    device_table
    test/test_device_table.py
    TIMEOUT 60
  )
  # Pins the ODriveError bit values the hardware checkers decode against.
  ament_add_pytest_test(
    odrive_errors
    test/test_odrive_errors.py
    TIMEOUT 60
  )
endif()

ament_package()
