cmake_minimum_required(VERSION 3.22)
project(aries_imu)

find_package(ament_cmake REQUIRED)

# No udev rule ships here: ros-jazzy-microstrain-inertial-driver already
# installs 60-ros-jazzy-microstrain-inertial-driver.rules, which creates
# /dev/microstrain_main.
install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
