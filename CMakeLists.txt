cmake_minimum_required(VERSION 3.14)
project(myTorch)

set(CMAKE_CXX_STANDARD 11)

find_package(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

add_executable(myTorch main.cpp)

target_link_libraries(myTorch ${PYTHON_LIBRARIES})