cmake_minimum_required(VERSION 3.14)

project(CppTensorEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(tensor_core 
    src/tensor/tensor.cpp
)

target_include_directories(tensor_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(engine_run 
    examples/main.cpp
)

target_link_libraries(engine_run PRIVATE tensor_core)