cmake_minimum_required(VERSION 3.17)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB SRC "src/*.cpp")
file(GLOB DEVICE_TESTS_SRC "tests/device/*.cpp")
file(GLOB HOST_SRC "src/host/*.cpp")
file(GLOB HOST_TESTS_SRC "tests/host/*.cpp")
file(GLOB CU_SRC "src/*.cu")
file(GLOB CUBLAS_SRC "src/cublas/*.cpp")

option(MTRX_CUBLAS_BUILD_DEVICE "Compile device part of module" ON)
option(MTRX_CUBLAS_BUILD_HOST "Compile host part of module" ON)
option(MTRX_CUBLAS_NVPROF_TESTS "Compile device tests with profiler support" OFF)
option(MTRX_CUBLAS_NVPROF_KERNELS "Compile device kernels with profiler support" OFF)
option(MTRX_CUBLAS_ASAN "Compile in asan mode" ON)

link_directories(/usr/local/cuda/lib64 /usr/local/math_libs/lib64)

set(LogLevels CRITICAL ERROR INFO DEBUG TRACE)
set(MTRX_CUBLAS_TEST_LOG_LEVEL ERROR CACHE STRING "log level for tests")
set_property(CACHE MTRX_CUBLAS_TEST_LOG_LEVEL PROPERTY STRINGS ${LogLevels})

if(NOT MTRX_CUBLAS_TEST_LOG_LEVEL IN_LIST LogLevels)
  message(FATAL_ERROR "Log level can be on of ${LogLevels}")
endif()
message("Test log level is ${MTRX_CUBLAS_TEST_LOG_LEVEL}")
list(FIND LogLevels ${MTRX_CUBLAS_TEST_LOG_LEVEL} logLevelIndex)

if(${MTRX_CUBLAS_BUILD_DEVICE})
  enable_language(CUDA)
  # cubin mtrxCublas
  set (CUBIN_TARGET "libmtrxCuda.cubin")
  add_custom_command(OUTPUT ${CUBIN_TARGET} COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILATION_OPTIONS} ${CUDA_DEFINITIONS} ${CUDA_INCLUDE_DIRS} -I ${PROJECT_SOURCE_DIR}/mtrxCore/include --cubin ${CU_SRC} -o ${CUBIN_TARGET} DEPENDS ${CU_SRC})
  add_custom_target(CU_MTRX_CUDA ALL DEPENDS ${CUBIN_TARGET})

  # mtrxCublas library
  add_library(mtrxCublas STATIC ${SRC})
  target_link_libraries(mtrxCublas mtrxCore stdc++fs cuda cudart cublas cublasLt spdlog)
  target_include_directories(mtrxCublas PUBLIC include /usr/local/cuda/include)
if(${MTRX_CUBLAS_NVPROF_KERNELS})
  set_target_properties(mtrxCublas PROPERTIES COMPILE_FLAGS -DMTRX_CUBLAS_NVPROF_KERNELS)
endif()
  # device tests
  set (TEST_DEVICE_TARGET "mtrxDeviceCublasTests")
  include(GoogleTest)
  add_executable(${TEST_DEVICE_TARGET} ${SRC} ${DEVICE_TESTS_SRC})
  target_link_libraries(${TEST_DEVICE_TARGET} mtrxCore stdc++fs cuda cudart cublas cublasLt gtest_main gtest gmock pthread spdlog)
  target_include_directories(${TEST_DEVICE_TARGET} PUBLIC include tests/include src)
if(${MTRX_CUBLAS_NVPROF_TESTS})
  set_target_properties(${TEST_DEVICE_TARGET} PROPERTIES COMPILE_FLAGS -DMTRX_CUBLAS_NVPROF_TESTS)
endif()
  set_property(TARGET ${TEST_DEVICE_TARGET} APPEND PROPERTY COMPILE_OPTIONS -DMTRX_CUBLAS_TEST_LOG_LEVEL=${logLevelIndex})
  gtest_add_tests(TARGET ${TEST_DEVICE_TARGET})
endif()

if(${MTRX_CUBLAS_BUILD_HOST})
  # host tests
  set(REQUIRED_CPP "src/kernels.cpp" "src/ikernel_executor.cpp" "src/host/host_kernel_executor.cpp" "src/sys_pathes_parser.cpp" "src/calc_dim.cpp" "src/host_alloc.cpp")
  set(TEST_HOST_TARGET "mtrxHostCublasTests")
  include(GoogleTest)
  add_executable(${TEST_HOST_TARGET} ${HOST_TESTS_SRC} ${HOST_SRC} ${REQUIRED_CPP})
if(${MTRX_CUBLAS_ASAN})
  target_compile_options (${TEST_HOST_TARGET} PRIVATE -fsanitize=address)
  target_link_libraries(${TEST_HOST_TARGET} asan mtrxCore gtest_main gtest gmock pthread spdlog)
else()
  target_compile_options (${TEST_HOST_TARGET} PRIVATE)
  target_link_libraries(${TEST_HOST_TARGET} mtrxCore gtest_main gtest gmock pthread spdlog)
endif()
  set_target_properties(${TEST_HOST_TARGET} PROPERTIES COMPILE_FLAGS -DMTRX_HOST_CUDA_BUILD)
  target_link_libraries(${TEST_HOST_TARGET} mtrxCore gtest_main gtest gmock pthread spdlog)
  target_include_directories(${TEST_HOST_TARGET} PUBLIC include tests/include)
  set_property(TARGET ${TEST_HOST_TARGET} APPEND PROPERTY COMPILE_OPTIONS -DMTRX_CUBLAS_TEST_LOG_LEVEL=${logLevelIndex})
  gtest_add_tests(TARGET ${TEST_HOST_TARGET})
endif()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/run_device_tests_profiling.sh.in
"#!/bin/bash
sudo env \"LD_LIBRARY_PATH=$LD_LIBRARY_PATH\" nvprof ${CMAKE_CURRENT_BINARY_DIR}/mtrxDeviceCublasTests
")

file(
    COPY ${CMAKE_CURRENT_BINARY_DIR}/run_device_tests_profiling.sh.in
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

file(RENAME
    ${CMAKE_CURRENT_BINARY_DIR}/run_device_tests_profiling.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/run_device_tests_profiling.sh
)
