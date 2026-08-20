cmake_minimum_required(VERSION 3.8)
project(aries)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes models rviz urdf worlds
  DESTINATION share/${PROJECT_NAME}/
  # meshes/unused holds geometry nothing in the description loads any more
  # (superseded arm/rover parts, the source STLs the coloured arm .glb files
  # were baked from, and since 2026-08-20 the retired gripper meshes). Kept in
  # the repo, but not worth shipping.
  PATTERN "unused" EXCLUDE
  # urdf/legacy is the same deal for the two retired grippers: the 85.563 mm
  # four-bar (gripper_new) and the original (gripper). v2 is the only gripper
  # the robot is built with, so these are not installed - and the meshes they
  # point at live under meshes/unused, which is not installed either.
  #
  # Both EXCLUDEs only bite in a copy build: `colcon build --symlink-install`
  # builds the directory tree and symlinks every file into it without
  # consulting these patterns, so a dev workspace does carry unused/ and
  # legacy/ in install/. That is cosmetic - nothing loads them - so do not read
  # their presence there as the exclusion being broken.
  PATTERN "legacy" EXCLUDE
)

install(
  PROGRAMS
    scripts/teleop_keyboard.py
    scripts/waypoint_follower.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/virtual_differential.py
  RENAME virtual_differential
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
