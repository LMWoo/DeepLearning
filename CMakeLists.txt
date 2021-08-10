cmake_minimum_required(VERSION 3.14)
project(MyTorch)

set(CMAKE_CXX_STANDARD 11)

find_package(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

add_library(MyTorch SHARED module.cpp)

target_link_libraries(MyTorch ${PYTHON_LIBRARIES})