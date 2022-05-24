cmake_minimum_required(VERSION 3.17)

project(mtrxIram CXX)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB SRC "src/*.cpp")
file(GLOB HEADERS "include/mtrxIram/*.hpp")

add_library(mtrxIram STATIC ${SRC})
set_target_properties(mtrxIram PROPERTIES PUBLIC ${HEADERS} "src/*.hpp")
target_link_libraries(mtrxIram spdlog mtrxCore mtrxCublas)
target_include_directories(mtrxIram PUBLIC include)
