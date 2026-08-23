cmake_minimum_required(VERSION 3.22)
project(aries_load_cells)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    nodes/load_cells.py
  DESTINATION lib/${PROJECT_NAME}
)

# load_cells.yaml is the three cells' calibration and the drill bin's gate.
install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # The bin's load cell only carries the bin at one end of its stroke, and an
  # unloaded cell reads ZERO rather than "no data" - the same number an empty
  # bin reads. The gate that separates them is what this pins.
  ament_add_pytest_test(
    load_cells
    test/test_load_cells.py
    TIMEOUT 60
  )
endif()

ament_package()
