cmake_minimum_required(VERSION 3.17)

set_target_properties(PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

link_directories(/usr/local/cuda/lib64)

file(GLOB SRC "src/*.cpp")
file(GLOB DEVICE_TESTS_SRC "tests/device/*.cpp")
file(GLOB HOST_SRC "src/host/*.cpp")
file(GLOB HOST_TESTS_SRC "tests/host/*.cpp")
file(GLOB CU_SRC "src/*.cu")

option(CUBLAS_BUILD_CUDA_PART "Compile cuda part of module" ON)
option(CUBLAS_BUILD_HOST_PART "Compile host part of module" ON)

if(${CUBLAS_BUILD_CUDA_PART})
  project(mtrxCublas CXX CUDA)
  # cubin mtrxCublas
  set (CUBIN_TARGET "libmtrxCuda.cubin")
  add_custom_command(OUTPUT ${CUBIN_TARGET} COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILATION_OPTIONS} ${CUDA_DEFINITIONS} ${CUDA_INCLUDE_DIRS} --compile ${CU_SRC} -o ${CUBIN_TARGET} DEPENDS ${CU_SRC})
  add_custom_target(CU_MTRX_CUDA ALL DEPENDS ${CUBIN_TARGET})

  # executable mtrxCublas
  add_library(mtrxCublas STATIC ${SRC})
  target_link_libraries(mtrxCublas mtrxCore cuda cudart cublas mtrxCuda spdlog)
  target_include_directories(mtrxCublas PUBLIC include /usr/local/cuda/include /usr/local/include)

  # device tests
  set (TEST_DEVICE_TARGET "mtrxDeviceCublasTests")
  include(GoogleTest)
  add_executable(${TEST_DEVICE_TARGET} ${SRC} ${DEVICE_TESTS_SRC})
  target_link_libraries(${TEST_DEVICE_TARGET} mtrxCore cuda cudart cublas gtest_main gtest gmock pthread spdlog)
  target_include_directories(${TEST_DEVICE_TARGET} PUBLIC include src /usr/local/cuda/include /usr/local/include)
  gtest_add_tests(TARGET ${TEST_DEVICE_TARGET})
endif()

if(${CUBLAS_BUILD_HOST_PART})
  project(mtrxCublasHostTests CXX)
  # host tests
  set (TEST_HOST_TARGET "mtrxHostCublasTests")
  include(GoogleTest)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
  add_executable(${TEST_HOST_TARGET} ${HOST_TESTS_SRC} ${HOST_SRC} "src/kernels.cpp" "src/ikernel_executor.cpp" "src/host/host_kernel_executor.cpp")
  set_target_properties(${TEST_HOST_TARGET} PROPERTIES COMPILE_FLAGS -DMTRX_HOST_CUDA_BUILD)
  target_link_libraries(${TEST_HOST_TARGET} mtrxCore cublas gtest_main gtest gmock pthread spdlog)
  target_include_directories(${TEST_HOST_TARGET} PUBLIC include /usr/local/cuda/include /usr/local/include)
  gtest_add_tests(TARGET ${TEST_HOST_TARGET})
endif()
