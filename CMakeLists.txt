cmake_minimum_required(VERSION 3.17)

project(mtrx)
option(BUILD_CUBLAS "Enable cublas compilation" ON)

include_directories(".")

add_subdirectory(mtrxCore)

if(BUILD_CUBLAS)
add_subdirectory(mtrxCublas)
endif(BUILD_CUBLAS)
