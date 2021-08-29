cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(MyTorch_cpp)

set(MYTORCH_CPP_DIR "${CMAKE_SOURCE_DIR}")
set(MYTORCH_CPP_SOURCES
  ${MYTORCH_CPP_DIR}/main.cpp
)

add_executable(MyTorch_cpp ${MYTORCH_CPP_SOURCES})
target_include_directories(MyTorch_cpp PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(MyTorch_cpp PRIVATE torch)
