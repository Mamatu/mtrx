#include <array>
#include <gtest/gtest.h>

#include "../src/calc_dim.hpp"
#include "../src/device_properties.hpp"
#include "../src/host/device_properties_provider.hpp"
#include "../src/host_alloc.hpp"
#include "../src/kernels.hpp"
#include <spdlog/spdlog.h>

namespace mtrx {
class IsULTriangularTests : public testing::Test {
public:
  void SetUp() override {
    spdlog::set_level(spdlog::level::debug);

  }
};

TEST_F(IsULTriangularTests, is_upper_triangular) {
  DeviceProperties dp;
  dp.blockDim = {32, 32, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 1024;
  dp.maxThreadsPerBlock = 1024;
  dp.sharedMemPerBlock = 16000;

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<float, 4> matrix = {1, 0, 1, 1};

  auto is = kernels.isUpperTriangular(2, 2, matrix.data(), 2, 0);
  EXPECT_TRUE(is);
}

} // namespace mtrx
