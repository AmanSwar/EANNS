cmake_minimum_required(VERSION 3.12)
project(TestProgram LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


include_directories(../include/similarity)
include_directories(/usr/local/cuda/include)



set(CUDA_SOURCES ../../../src/similarity/cosine_sim.cu)
set(CPP_SOURCES ./similarity/cosine.cpp)
set(SOURCES ${CUDA_SOURCES} ${CPP_SOURCES})

# Create the executable target
add_executable(test_program ${SOURCES})

# Link against cuBLAS
target_link_libraries(test_program cublas)

# Optionally, enable separable compilation for CUDA if needed
set_target_properties(test_program PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
