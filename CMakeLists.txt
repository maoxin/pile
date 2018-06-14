cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pile)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pile pile.cpp)
target_link_libraries(pile ${PCL_LIBRARIES})