cmake_minimum_required(VERSION 3.17)

project(mtrx CXX)
option(MTRX_BUILD_CUBLAS "Enable cublas compilation" ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

include_directories(mtrxCore/include mtrxCublas/include)

add_subdirectory(mtrxCore)

if(MTRX_BUILD_CUBLAS)
  add_subdirectory(mtrxCublas)
endif()

#add_subdirectory(mtrxIram)
