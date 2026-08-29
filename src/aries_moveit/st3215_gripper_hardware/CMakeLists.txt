cmake_minimum_required(VERSION 3.8)
project(st3215_gripper_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

add_library(st3215_gripper_hardware SHARED
  src/sts_bus.cpp
  src/st3215_gripper_system.cpp
)
target_include_directories(st3215_gripper_hardware PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(st3215_gripper_hardware
  rclcpp hardware_interface pluginlib rclcpp_lifecycle
)

pluginlib_export_plugin_description_file(hardware_interface st3215_gripper_hardware.xml)

install(TARGETS st3215_gripper_hardware
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES st3215_gripper_hardware.xml DESTINATION share/${PROJECT_NAME})

ament_package()
