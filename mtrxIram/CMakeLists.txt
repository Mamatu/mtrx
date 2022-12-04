cmake_minimum_required(VERSION 3.17)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB SRC "src/*.cpp")
file(GLOB TESTS_SRC "tests/*.cpp")
file(GLOB HEADERS "include/mtrxIram/*.hpp" "include/mtrxIram/impl/*.hpp")

add_library(mtrxIram STATIC ${SRC})
set_target_properties(mtrxIram PROPERTIES PUBLIC ${HEADERS} "include/mtrxIram/impl/*.hpp" "src/*.cpp")
target_link_libraries(mtrxIram spdlog mtrxCore)
target_include_directories(mtrxIram PUBLIC include)

set (TEST_TARGET "mtrxTests")
include(GoogleTest)
add_executable(${TEST_TARGET} ${SRC} ${TESTS_SRC})
target_link_libraries(${TEST_TARGET} mtrxCore stdc++fs gtest_main gtest gmock pthread spdlog)
target_include_directories(${TEST_TARGET} PUBLIC include src)
