cmake_minimum_required(VERSION 3.8)
project(aries_maintenance)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Shared python library (marker geometry, panel pose, control waypoints).
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/panel_operator_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Run from the package root so the test imports the source tree package.
  ament_add_pytest_test(test_panel_alignment test/test_panel_alignment.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  ament_add_pytest_test(test_panel_operator_actions test/test_panel_operator_actions.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_package()
