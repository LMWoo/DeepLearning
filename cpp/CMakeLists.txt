cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(DeepLearning_cpp)

set(DEEPLEARNING_CPP_DIR "${CMAKE_SOURCE_DIR}")
set(DEEPLEARNING_CPP_SOURCES
/media/lee/ESD-ISO/study/DeepLearning/cpp/main.cpp
)
set(THIRD_PARTY_INCLUDE "/media/lee/ESD-ISO/study/DeepLearning/cpp/third_party/")

add_executable(DeepLearning_cpp ${DEEPLEARNING_CPP_SOURCES})
target_include_directories(DeepLearning_cpp PRIVATE ${ATen_CPU_INCLUDE} ${THIRD_PARTY_INCLUDE})
target_link_libraries(DeepLearning_cpp PRIVATE torch)
