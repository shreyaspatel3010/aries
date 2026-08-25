cmake_minimum_required(VERSION 3.22)
project(aries_comms)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    nodes/base_station_checker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # The two launch files are one contract. Nothing checks at runtime that
  # exactly one machine reads the pad or that both set the DDS environment
  # before their first node, so it is pinned here.
  ament_add_pytest_test(
    field_link_contract
    test/test_field_link_contract.py
    TIMEOUT 60
  )
  # The operator's ODrive table and the rover's are deliberate duplicates, so
  # that neither end depends on a shared module changing under it. This is what
  # stops them drifting into naming the same axis two different wheels.
  ament_add_pytest_test(
    axis_tables
    test/test_axis_tables.py
    TIMEOUT 60
  )
endif()

ament_package()
