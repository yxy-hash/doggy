cmake_minimum_required(VERSION 3.8)
project(dog)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)

rosidl_generate_interfaces(dog
  "../interface/Depthimage.msg"
  "../interface/InferRKNN.srv"

  DEPENDENCIES std_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(
  DIRECTORY ../launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY ../config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
# ament_target_dependencies(my_node rclcpp sensor_msgs)
