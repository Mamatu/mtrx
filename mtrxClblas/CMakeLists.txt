cmake_minimum_required(VERSION 3.17)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

link_directories(/usr/local/lib64 /usr/lib/x86_64-linux-gnu)

file(GLOB SRC "src/*.cpp")
file(GLOB TESTS_SRC "tests/*.cpp")
#file(GLOB HEADERS "include/mtrxClblas/*.hpp")

add_library(mtrxClblas STATIC ${SRC})
target_link_libraries(mtrxClblas mtrxCore clBLAS OpenCL)
target_include_directories(mtrxClblas PUBLIC include /usr/local/include)

include(GoogleTest)
add_executable(mtrxClblasTests ${SRC} ${TESTS_SRC})
target_link_libraries(mtrxClblasTests mtrxCore clBLAS OpenCL gtest_main gtest gmock pthread)
target_include_directories(mtrxClblasTests PUBLIC include /usr/local/include)
gtest_add_tests(TARGET mtrxClblasTests)
