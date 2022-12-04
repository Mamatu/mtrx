cmake_minimum_required(VERSION 3.17)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

link_directories(/usr/local/cuda/lib64 /usr/local/math_libs/lib64)

file(GLOB TESTS_SRC "tests/*.cpp")

set (TEST_TARGET "mtrxIramCublasTests")
include(GoogleTest)
add_executable(${TEST_TARGET} ${SRC} ${TESTS_SRC})
target_link_libraries(${TEST_TARGET} mtrxCore mtrxCublas cuda cudart cublas cublasLt stdc++fs gtest_main gtest gmock pthread spdlog)
target_include_directories(${TEST_TARGET} PUBLIC mtrxIram mtrxCublas)
