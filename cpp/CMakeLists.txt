cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(DeepLearning_cpp)

set(DEEPLEARNING_CPP_SOURCES ${PROJECT_SOURCE_DIR}/main.cpp)

add_executable(${PROJECT_NAME} ${DEEPLEARNING_CPP_SOURCES})

set(THIRD_PARTY_INCLUDE ${PROJECT_SOURCE_DIR}/third_party)
target_include_directories(${PROJECT_NAME} PRIVATE ${ATen_CPU_INCLUDE} ${THIRD_PARTY_INCLUDE})
target_link_libraries(${PROJECT_NAME} PRIVATE torch)
