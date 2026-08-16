cmake_minimum_required(VERSION 3.16)
project(xorAlg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_executable(xorAlg src/main.cpp)
target_include_directories(xorAlg PRIVATE include)
target_compile_options(xorAlg PRIVATE -Wall -Wextra)