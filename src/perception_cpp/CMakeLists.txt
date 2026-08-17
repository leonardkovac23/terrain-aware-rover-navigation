cmake_minimum_required(VERSION 3.8)
project(perception_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

#PointCloud Filter Node
add_executable(pointcloud_filter_node 
	src/pointcloud_filter_node.cpp
)

ament_target_dependencies(pointcloud_filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(pointcloud_filter_node
  ${PCL_LIBRARIES}
)

target_include_directories(pointcloud_filter_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)


#Obstacle Costmap Node
add_executable(obstacle_costmap_node
	src/obstacle_costmap_node.cpp
)

ament_target_dependencies(obstacle_costmap_node
	rclcpp
	sensor_msgs
	nav_msgs
	std_msgs
	geometry_msgs
	pcl_conversions
	tf2
	tf2_ros
	tf2_geometry_msgs
)

target_link_libraries(obstacle_costmap_node
  ${PCL_LIBRARIES}
)

target_include_directories(obstacle_costmap_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

#Slope Costmap Node
add_executable(slope_costmap_node
	src/slope_costmap_node.cpp
)

ament_target_dependencies(slope_costmap_node
	rclcpp
	sensor_msgs
	nav_msgs
	std_msgs
	geometry_msgs
	pcl_conversions
	tf2
	tf2_ros
	tf2_geometry_msgs
)

target_link_libraries(slope_costmap_node
  ${PCL_LIBRARIES}
)

target_include_directories(slope_costmap_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)

install(TARGETS
  pointcloud_filter_node
  obstacle_costmap_node
  slope_costmap_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

ament_package()
