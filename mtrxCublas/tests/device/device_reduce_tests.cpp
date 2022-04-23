#include <mtrxCublas/test.hpp>
#include <math.h>
#include <mtrxCore/size_of.hpp>
#include <mtrxCore/types.hpp>
#include <mtrxCublas/cublas.hpp>
#include <mtrxCublas/matchers.hpp>

#include <array>
#include <cuda.h>
#include <numeric>

#ifdef CUBLAS_NVPROF_TESTS
#include "cuda_profiler.hpp"
#endif

#include "../src/cuda_alloc.hpp"
#include "../src/kernels.hpp"

namespace mtrx {
class DeviceReduceTests : public Test {
};

TEST_F(DeviceReduceTests, reduce_size_1x1) {
  CudaAlloc cudaAlloc;
  Kernels kernels(0, &cudaAlloc);

  std::array<int, 1> h_array = {5};

  int *d_array = nullptr;
  cudaMalloc(&d_array, SizeOf<int>(h_array.size()));
  cudaMemcpy(d_array, h_array.data(), SizeOf<int>(h_array.size()),
             cudaMemcpyHostToDevice);

  auto reduction = kernels.reduceShm(1, 1, d_array, 1);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}

TEST_F(DeviceReduceTests, reduce_size_2x1) {
  CudaAlloc cudaAlloc;
  Kernels kernels(0, &cudaAlloc);

  std::array<int, 2> h_array = {5, 6};

  int *d_array = nullptr;
  cudaMalloc(&d_array, SizeOf<int>(h_array.size()));
  cudaMemcpy(d_array, h_array.data(), SizeOf<int>(h_array.size()),
             cudaMemcpyHostToDevice);

  auto reduction = kernels.reduceShm(2, 1, d_array, 2);
  EXPECT_EQ(11, reduction);
}

TEST_F(DeviceReduceTests, reduce_size_1x2) {
  CudaAlloc cudaAlloc;
  Kernels kernels(0, &cudaAlloc);

  std::array<int, 2> h_array = {5, 6};

  int *d_array = nullptr;
  cudaMalloc(&d_array, SizeOf<int>(h_array.size()));
  cudaMemcpy(d_array, h_array.data(), SizeOf<int>(h_array.size()),
             cudaMemcpyHostToDevice);

  auto reduction = kernels.reduceShm(1, 2, d_array, 1);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}

TEST_F(DeviceReduceTests, reduce_size_2x2) {
  CudaAlloc cudaAlloc;
  Kernels kernels(0, &cudaAlloc);

  std::array<int, 4> h_array = {5, 6, 7, 8};

  int *d_array = nullptr;
  cudaMalloc(&d_array, SizeOf<int>(h_array.size()));
  cudaMemcpy(d_array, h_array.data(), SizeOf<int>(h_array.size()),
             cudaMemcpyHostToDevice);

  auto reduction = kernels.reduceShm(2, 2, d_array, 2);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}

TEST_F(DeviceReduceTests, reduce_size_3x3) {
  CudaAlloc cudaAlloc;
  Kernels kernels(0, &cudaAlloc);

  std::array<int, 9> h_array = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  int *d_array = nullptr;
  cudaMalloc(&d_array, SizeOf<int>(h_array.size()));
  cudaMemcpy(d_array, h_array.data(), SizeOf<int>(h_array.size()),
             cudaMemcpyHostToDevice);

  auto reduction = kernels.reduceShm(3, 3, d_array, 3);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}
} // namespace mtrx
