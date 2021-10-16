cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(DeepLearning_cpp)

set(DEEPLEARNING_CPP_SOURCES ${PROJECT_SOURCE_DIR}/main.cpp)

add_executable(${PROJECT_NAME} ${DEEPLEARNING_CPP_SOURCES})

set(DEEPLEARNING_CPP_INCLUDE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${ATen_CPU_INCLUDE} ${DEEPLEARNING_CPP_INCLUDE})
target_link_libraries(${PROJECT_NAME} PRIVATE torch_cuda)