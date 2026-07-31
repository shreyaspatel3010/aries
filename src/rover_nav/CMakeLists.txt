cmake_minimum_required(VERSION 3.5)
project(rover_nav)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate message interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
)

# Install Python executable nodes (scripts)
install(
  PROGRAMS
    scripts/Rover_control_Joy.py
    scripts/LeapOne_test_encoders.py
    scripts/Odom.py
    scripts/rover_controller_pure_pursuit.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)


install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  scripts/Sounds
  DESTINATION share/${PROJECT_NAME}/sounds/
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    odometry_stationary
    test/test_odometry_stationary.py
    TIMEOUT 60
  )
endif()

ament_package()
