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
  # (superseded arm/rover parts, plus the source STLs the coloured arm .glb
  # files were baked from). Kept in the repo, but not worth shipping.
  PATTERN "unused" EXCLUDE
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
