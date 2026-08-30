cmake_minimum_required(VERSION 3.22)
project(aries_science)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    nodes/science_telemetry.py
  DESTINATION lib/${PROJECT_NAME}
)

# science.yaml is the field table: the array order the firmware sends, and the
# name and unit each index carries.
install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # The array carries no names, so the ONLY thing tying an index to a sensor is
  # this package's YAML agreeing with the firmware's TelemetryIndex enum. That
  # agreement is what this pins -- it reads the actual firmware source.
  ament_add_pytest_test(
    science_telemetry
    test/test_science_telemetry.py
    TIMEOUT 60
  )
endif()

ament_package()
