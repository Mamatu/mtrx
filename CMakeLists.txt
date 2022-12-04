cmake_minimum_required(VERSION 3.17)

project(mtrx CXX)
option(MTRX_BUILD_CUBLAS "Enable cublas compilation" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

include_directories(mtrxCore/include mtrxCublas/include mtrxIram/include)

add_subdirectory(mtrxCore)
add_subdirectory(mtrxIram)

if(MTRX_BUILD_CUBLAS)
  add_subdirectory(mtrxCublas)
  add_subdirectory(mtrxIramCublas)
endif()
