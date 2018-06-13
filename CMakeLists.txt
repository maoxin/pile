cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pile)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(normal_estimate normal_estimate.cpp)
add_executable(region_growing_segmentation region_growing_segmentation.cpp)
target_link_libraries(normal_estimate ${PCL_LIBRARIES})
target_link_libraries(region_growing_segmentation ${PCL_LIBRARIES})