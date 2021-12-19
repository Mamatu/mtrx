cmake_minimum_required(VERSION 3.17)

find_package(CUDA REQUIRED)

set_target_properties( PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

link_directories(/usr/local/cuda/lib64)

add_executable(blas main.cpp)
target_link_libraries(blas mtrxCublas mtrxCore)
target_include_directories(blas PUBLIC mtrxCore mtrxCublas)
