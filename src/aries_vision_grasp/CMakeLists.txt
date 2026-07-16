cmake_minimum_required(VERSION 3.8)
project(aries_vision_grasp)

find_package(ament_cmake REQUIRED)

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

ament_package()
