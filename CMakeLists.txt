cmake_minimum_required(VERSION 3.16)
project(surround_vision)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(Threads REQUIRED)
find_package(ament_index_cpp REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

# CUDA support permanently disabled to avoid version conflicts
message(STATUS "CUDA support disabled - using CPU-only processing")

# Find libjpeg-turbo
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(TURBOJPEG QUIET libturbojpeg)
endif()

if(NOT TURBOJPEG_FOUND)
    find_path(TURBOJPEG_INCLUDE_DIRS NAMES turbojpeg.h)
    find_library(TURBOJPEG_LIBRARIES NAMES turbojpeg)
    if(TURBOJPEG_INCLUDE_DIRS AND TURBOJPEG_LIBRARIES)
        set(TURBOJPEG_FOUND TRUE)
    endif()
endif()

if(TURBOJPEG_FOUND)
    message(STATUS "libjpeg-turbo found")
    add_definitions(-DUSE_LIBJPEG_TURBO)
    include_directories(${TURBOJPEG_INCLUDE_DIRS})
else()
    message(STATUS "libjpeg-turbo not found, using OpenCV fallback")
endif()


# find_package(PkgConfig REQUIRED)
# pkg_check_modules(NCURSES REQUIRED ncurses)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIRS}
  # ${NCURSES_INCLUDE_DIRS}
)

# Add executables
add_executable(surround_vision_node 
  src/surround_vision_node.cpp
  src/image_decoder.cpp
)

add_executable(fast_image_republisher 
  src/fast_image_republisher_optimized.cpp
)

add_executable(obj_viewer_node src/obj_viewer_node.cpp)

# Set target properties
set_target_properties(surround_vision_node PROPERTIES
  COMPILE_FLAGS "-O3 -march=native"
)


set_target_properties(fast_image_republisher PROPERTIES
  COMPILE_FLAGS "-O3 -march=native"
)

# Link libraries
target_link_libraries(surround_vision_node
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  # ${NCURSES_LIBRARIES}
  glfw
  assimp
  Threads::Threads
  yaml-cpp
)

target_link_libraries(fast_image_republisher
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
  Threads::Threads
)

# CUDA libraries permanently disabled

# Add libjpeg-turbo libraries if available
if(TURBOJPEG_FOUND)
    target_link_libraries(fast_image_republisher
        ${TURBOJPEG_LIBRARIES}
    )
    target_link_libraries(surround_vision_node
        ${TURBOJPEG_LIBRARIES}
    )
endif()

# ROS 2 dependencies
ament_target_dependencies(surround_vision_node
  rclcpp
  sensor_msgs
  cv_bridge
  ament_index_cpp
)

ament_target_dependencies(fast_image_republisher
  rclcpp
  sensor_msgs
  cv_bridge
)

ament_target_dependencies(obj_viewer_node
  rclcpp
  ament_index_cpp
)

target_link_libraries(obj_viewer_node
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  ${glm_LIBRARIES}
)

# Install executables
install(TARGETS
  surround_vision_node  fast_image_republisher obj_viewer_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files if any
install(DIRECTORY
  launch config models icons rviz urdf camera_calibration
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

# Install config files if any
# install(DIRECTORY
#   config
#   DESTINATION share/${PROJECT_NAME}/
#   OPTIONAL
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()