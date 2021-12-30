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

option(CUBLAS_BUILD_DEVICE_PART "Compile device part of module" ON)
option(CUBLAS_BUILD_HOST_PART "Compile host part of module" ON)

link_directories(/opt/nvidia/hpc_sdk/Linux_x86_64/20.7/cuda/10.1/lib64 /opt/nvidia/hpc_sdk/Linux_x86_64/20.7/math_libs/10.1/lib64)

if(${CUBLAS_BUILD_DEVICE_PART})
  enable_language(CUDA)
  # cubin mtrxCublas
  set (CUBIN_TARGET "libmtrxCuda.cubin")
  add_custom_command(OUTPUT ${CUBIN_TARGET} COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILATION_OPTIONS} ${CUDA_DEFINITIONS} ${CUDA_INCLUDE_DIRS} --compile ${CU_SRC} -o ${CUBIN_TARGET} DEPENDS ${CU_SRC})
  add_custom_target(CU_MTRX_CUDA ALL DEPENDS ${CUBIN_TARGET})

  # mtrxCublas library
  add_library(mtrxCublas STATIC ${SRC})
  target_link_libraries(mtrxCublas mtrxCore cuda cudart cublas cublasLt spdlog)
  target_include_directories(mtrxCublas PUBLIC include /opt/nvidia/hpc_sdk/Linux_x86_64/20.7/cuda/10.1/include)

  # device tests
  set (TEST_DEVICE_TARGET "mtrxDeviceCublasTests")
  include(GoogleTest)
  add_executable(${TEST_DEVICE_TARGET} ${SRC} ${DEVICE_TESTS_SRC})
  target_link_libraries(${TEST_DEVICE_TARGET} mtrxCore cuda cudart cublas cublasLt gtest_main gtest gmock pthread spdlog)
  target_include_directories(${TEST_DEVICE_TARGET} PUBLIC include src)
  gtest_add_tests(TARGET ${TEST_DEVICE_TARGET})
endif()

if(${CUBLAS_BUILD_HOST_PART})
  # host tests
  set (TEST_HOST_TARGET "mtrxHostCublasTests")
  include(GoogleTest)
  add_executable(${TEST_HOST_TARGET} ${HOST_TESTS_SRC} ${HOST_SRC} "src/kernels.cpp" "src/ikernel_executor.cpp" "src/host/host_kernel_executor.cpp")
  target_compile_options (${TEST_HOST_TARGET} PRIVATE -fsanitize=address)
  set_target_properties(${TEST_HOST_TARGET} PROPERTIES COMPILE_FLAGS -DMTRX_HOST_CUDA_BUILD)
  target_link_libraries(${TEST_HOST_TARGET} asan mtrxCore gtest_main gtest gmock pthread spdlog)
  target_include_directories(${TEST_HOST_TARGET} PUBLIC include)
  gtest_add_tests(TARGET ${TEST_HOST_TARGET})
endif()
