cmake_minimum_required(VERSION 3.5)
project(concat_sensor_to_map)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(concat_sensor_to_map SHARED
  src/concat_sensor_to_map.cpp
)

rclcpp_components_register_node(concat_sensor_to_map
  PLUGIN "ConcatSensorToMap"
  EXECUTABLE concat_sensor_to_map_node
)

target_link_libraries(concat_sensor_to_map ${PCL_LIBRARIES})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    launch
)

install(FILES DESTINATION share/${PROJECT_NAME})
