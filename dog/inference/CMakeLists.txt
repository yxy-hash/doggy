cmake_minimum_required(VERSION 3.5)

project(inference)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(dog REQUIRED)

# find_package(std_msgs REQUIRED)

include_directories(
  include/
  ./
  ../
)

add_executable(inference_node
  ${SRC}
)

ament_target_dependencies(inference_node
  rclcpp
  dog_interface
  sensor_msgs
  dog
)

install(
  TARGETS inference_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
