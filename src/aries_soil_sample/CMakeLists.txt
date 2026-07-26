cmake_minimum_required(VERSION 3.8)
project(aries_soil_sample)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Shared python library (terrain analysis, scoop geometry).
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/soil_sample_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Run from the package root so the tests import the source tree package.
  ament_add_pytest_test(test_terrain test/test_terrain.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  ament_add_pytest_test(test_scoop test/test_scoop.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  ament_add_pytest_test(test_deposit test/test_deposit.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_package()
