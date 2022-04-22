#include <array>
#include <gtest/gtest.h>

#include "../src/calc_dim.hpp"
#include "../src/device_properties.hpp"
#include "../src/host/device_properties_provider.hpp"
#include "../src/host_alloc.hpp"
#include "../src/kernels.hpp"
#include <numeric>
#include <spdlog/spdlog.h>

namespace mtrx {
class HostReduceTests : public testing::Test {
public:
  void SetUp() override { spdlog::set_level(spdlog::level::debug); }
};

TEST_F(HostReduceTests, constant) {
  DeviceProperties dp;
  dp.blockDim = {1, 1, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 1;
  dp.maxThreadsPerBlock = 1;
  dp.sharedMemPerBlock = 4;

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<float, 1> matrix = {1};

  int sum = kernels.reduceShm(1, 1, matrix.data(), 1);
  EXPECT_EQ(1, sum);
}

TEST_F(HostReduceTests, multi_blocks) {
  DeviceProperties dp;
  dp.blockDim = {1, 1, 1};
  dp.gridDim = {2, 2, 1};
  dp.maxRegistersPerBlock = 1;
  dp.maxThreadsPerBlock = 1;
  dp.sharedMemPerBlock = sizeof(float);

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<float, 4> matrix = {1, 2, 3, 4};

  int sum = kernels.reduceShm(2, 2, matrix.data(), 2);
  EXPECT_EQ(10, sum);
}

TEST_F(HostReduceTests, reduce_size_1x1) {
  DeviceProperties dp;
  dp.blockDim = {10, 10, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 1;
  dp.maxThreadsPerBlock = 100;
  dp.sharedMemPerBlock = sizeof(float) * 100;

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<int, 1> h_array = {5};

  auto reduction = kernels.reduceShm(1, 1, h_array.data(), 1);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}

TEST_F(HostReduceTests, reduce_size_2x1) {
  DeviceProperties dp;
  dp.blockDim = {10, 10, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 1;
  dp.maxThreadsPerBlock = 100;
  dp.sharedMemPerBlock = sizeof(float) * 100;

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);
  std::array<int, 2> h_array = {5, 6};

  auto reduction = kernels.reduceShm(2, 1, h_array.data(), 2);
  EXPECT_EQ(11, reduction);
}

TEST_F(HostReduceTests, reduce_size_1x2) {
  DeviceProperties dp;
  dp.blockDim = {10, 10, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 1;
  dp.maxThreadsPerBlock = 100;
  dp.sharedMemPerBlock = sizeof(float) * 100;

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);
  std::array<int, 2> h_array = {5, 6};

  auto reduction = kernels.reduceShm(1, 2, h_array.data(), 1);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}

TEST_F(HostReduceTests, reduce_size_2x2) {
  DeviceProperties dp;
  dp.blockDim = {2, 2, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 1;
  dp.maxThreadsPerBlock = 4;
  dp.sharedMemPerBlock = sizeof(float) * 100;

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);
  std::array<int, 4> h_array = {5, 6, 7, 8};

  auto reduction = kernels.reduceShm(2, 2, h_array.data(), 2);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}

TEST_F(HostReduceTests, reduce_size_3x3) {
  DeviceProperties dp;
  dp.blockDim = {10, 10, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 10;
  dp.maxThreadsPerBlock = 100;
  dp.sharedMemPerBlock = sizeof(float) * 100;

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);
  std::array<int, 9> h_array = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto reduction = kernels.reduceShm(3, 3, h_array.data(), 3);

  int expected =
      std::accumulate(h_array.begin(), h_array.end(), static_cast<int>(0));
  EXPECT_EQ(expected, reduction);
}

} // namespace mtrx
