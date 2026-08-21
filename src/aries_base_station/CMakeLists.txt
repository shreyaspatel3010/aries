cmake_minimum_required(VERSION 3.22)
project(aries_base_station)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    base_station_contract
    test/test_base_station_contract.py
    TIMEOUT 60
  )
endif()

ament_package()
