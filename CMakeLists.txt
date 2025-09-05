cmake_minimum_required(VERSION 3.20)
project(numerical_methods_course)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(linalg)

add_subdirectory(labs)