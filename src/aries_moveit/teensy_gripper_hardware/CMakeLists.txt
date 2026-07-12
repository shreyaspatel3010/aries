cmake_minimum_required(VERSION 3.8)
project(teensy_gripper_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)

add_library(teensy_gripper_hardware SHARED
  src/teensy_gripper_system.cpp
)
target_include_directories(teensy_gripper_hardware PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(teensy_gripper_hardware
  rclcpp hardware_interface pluginlib rclcpp_lifecycle std_msgs
)

pluginlib_export_plugin_description_file(hardware_interface teensy_gripper_hardware.xml)

install(TARGETS teensy_gripper_hardware
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES teensy_gripper_hardware.xml DESTINATION share/${PROJECT_NAME})

ament_package()
