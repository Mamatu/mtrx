cmake_minimum_required(VERSION 3.17)

project(mtrx CXX CUDA)
option(BUILD_CUBLAS "Enable cublas compilation" ON)
option(BUILD_CLBLAS "Enable clblas compilation" ON)

include_directories(".")

add_subdirectory(mtrxCore)

if(BUILD_CUBLAS)
  set(CUBLAS_BUILD_CUDA_PART ON CACHE CACHE BOOL "Compile cuda part of module")
  set(CUBLAS_BUILD_CUDA_PART ON CACHE CACHE BOOL "Compile host part of module")
  set(${option} ${value} CACHE INTERNAL "" FORCE)
  add_subdirectory(mtrxCublas)
endif(BUILD_CUBLAS)

if(BUILD_CLBLAS)
  add_subdirectory(mtrxClblas)
endif(BUILD_CLBLAS)
