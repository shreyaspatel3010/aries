cmake_minimum_required(VERSION 3.22)
project(aries_imu)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS nodes/picoscan_imu_relay.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
