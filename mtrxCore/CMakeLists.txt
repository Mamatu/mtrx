cmake_minimum_required(VERSION 3.17)

project(mtrxCore CXX)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB SRC "src/*.cpp")
file(GLOB HEADERS "include/mtrxCore/*.hpp")

add_library(mtrxCore STATIC ${SRC})
set_target_properties(mtrxCore PROPERTIES ${HEADERS})
target_link_libraries(mtrxCore spdlog)
target_include_directories(mtrxCore PUBLIC include)
