cmake_minimum_required(VERSION 3.17)

project(mtrx CXX)
option(BUILD_CUBLAS "Enable cublas compilation" ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

include_directories(".")

add_subdirectory(mtrxCore)

if(BUILD_CUBLAS)
  set(CUBLAS_BUILD_DEVICE_PART ON CACHE BOOL "Compile device part of module")
  set(CUBLAS_BUILD_HOST_PART ON CACHE BOOL "Compile host part of module")
  add_subdirectory(mtrxCublas)
endif()
