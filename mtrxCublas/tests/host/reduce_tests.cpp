#include <array>
#include <gtest/gtest.h>

#include "../src/calc_dim.hpp"
#include "../src/device_properties.hpp"
#include "../src/host/device_properties_provider.hpp"
#include "../src/host_alloc.hpp"
#include "../src/kernels.hpp"
#include <spdlog/spdlog.h>

namespace mtrx {
class ReduceTests : public testing::Test {
public:
  void SetUp() override { spdlog::set_level(spdlog::level::debug); }
};

TEST_F(ReduceTests, constant) {
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

TEST_F(ReduceTests, multi_blocks) {
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

} // namespace mtrx
