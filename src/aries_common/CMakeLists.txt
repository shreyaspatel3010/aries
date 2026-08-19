cmake_minimum_required(VERSION 3.22)
project(aries_common)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# config/devices.yaml: the device table every launch file reads its defaults from.
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    hardware_detection
    test/test_detect.py
    TIMEOUT 60
  )
endif()

ament_package()
