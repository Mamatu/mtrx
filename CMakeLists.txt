cmake_minimum_required(VERSION 3.17)

project(mtrx CXX)
option(BUILD_CUBLAS "Enable cublas compilation" ON)

include_directories(".")

add_subdirectory(mtrxCore)

if(BUILD_CUBLAS)
  set(CUBLAS_BUILD_CUDA_PART ON CACHE CACHE BOOL "Compile cuda part of module")
  set(CUBLAS_BUILD_CUDA_PART ON CACHE CACHE BOOL "Compile host part of module")
  add_subdirectory(mtrxCublas)
endif(BUILD_CUBLAS)
