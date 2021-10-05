cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(DeepLearning_cpp)

set(DEEPLEARNING_CPP_DIR "${CMAKE_SOURCE_DIR}")
set(DEEPLEARNING_CPP_SOURCES
/media/lee/ESD-ISO/study/DeepLearning/cpp/main.cpp
)

add_executable(DeepLearning_cpp ${DEEPLEARNING_CPP_SOURCES})
target_include_directories(DeepLearning_cpp PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(DeepLearning_cpp PRIVATE torch)
