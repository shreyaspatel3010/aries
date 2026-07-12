cmake_minimum_required(VERSION 3.22)
project(aries_lidar)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS nodes/lidar_scan_frame_relay.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
