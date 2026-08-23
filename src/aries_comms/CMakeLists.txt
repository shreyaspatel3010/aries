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
endif()

ament_package()
