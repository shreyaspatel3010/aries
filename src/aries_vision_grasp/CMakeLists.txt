cmake_minimum_required(VERSION 3.8)
project(aries_vision_grasp)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Shared python library (geometry, fourbar model, stages, inference worker).
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/camera_viewer.py
    scripts/vision_grasp_node.py
    scripts/yolo_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Run from the package root so the tests import the source tree package.
  ament_add_pytest_test(test_fourbar test/test_fourbar.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  ament_add_pytest_test(test_geometry test/test_geometry.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_package()
