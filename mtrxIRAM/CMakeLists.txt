cmake_minimum_required(VERSION 3.17)

project(mtrxIRAM CXX)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB SRC "src/*.cpp")
file(GLOB HEADERS "include/mtrxIRAM/*.hpp")

add_library(mtrxIRAM STATIC ${SRC})
set_target_properties(mtrxIRAM PROPERTIES PUBLIC ${HEADERS})
target_link_libraries(mtrxIRAM spdlog)
target_include_directories(mtrxIRAM PUBLIC include)
