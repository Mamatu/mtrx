#include <array>
#include <mtrxCublas/test.hpp>

#include "../src/calc_dim.hpp"
#include "../src/device_properties.hpp"
#include "../src/host/device_properties_provider.hpp"
#include "../src/host_alloc.hpp"
#include "../src/kernels.hpp"

namespace mtrx {
class IsULTriangularTests : public Test {};

TEST_F(IsULTriangularTests,
       is_upper_triangular_matrixDim_2x2_blockDim_2x2_gridDim_1x1) {
  DeviceProperties dp;
  dp.blockDim = {2, 2, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 4;
  dp.maxThreadsPerBlock = 4;
  dp.sharedMemPerBlock = 4 * sizeof(float);

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<float, 4> matrix = {1, 0, 1, 1};

  auto is = kernels.isUpperTriangular(2, 2, matrix.data(), 2, 0);
  EXPECT_TRUE(is);
}

TEST_F(IsULTriangularTests,
       is_upper_triangular_matrixDim_2x2_blockDim_1x1_gridDim_2x2) {
  DeviceProperties dp;
  dp.blockDim = {1, 1, 1};
  dp.gridDim = {2, 2, 1};
  dp.maxRegistersPerBlock = 4;
  dp.maxThreadsPerBlock = 1;
  dp.sharedMemPerBlock = 4 * sizeof(float);

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<float, 4> matrix = {1, 0, 1, 1};

  auto is = kernels.isUpperTriangular(2, 2, matrix.data(), 2, 0);
  EXPECT_TRUE(is);
}

TEST_F(IsULTriangularTests,
       is_lower_triangular_matrixDim_2x2_blockDim_2x2_gridDim_1x1) {
  DeviceProperties dp;
  dp.blockDim = {2, 2, 1};
  dp.gridDim = {1, 1, 1};
  dp.maxRegistersPerBlock = 4;
  dp.maxThreadsPerBlock = 4;
  dp.sharedMemPerBlock = 4 * sizeof(float);

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<float, 4> matrix = {1, 1, 0, 1};

  auto is = kernels.isLowerTriangular(2, 2, matrix.data(), 2, 0);
  EXPECT_TRUE(is);
}

TEST_F(IsULTriangularTests,
       is_lower_triangular_matrixDim_2x2_blockDim_1x1_gridDim_2x2) {
  DeviceProperties dp;
  dp.blockDim = {1, 1, 1};
  dp.gridDim = {2, 2, 1};
  dp.maxRegistersPerBlock = 4;
  dp.maxThreadsPerBlock = 1;
  dp.sharedMemPerBlock = 4 * sizeof(float);

  DevicePropertiesProvider::set(0, dp);

  HostAlloc hostAlloc;
  Kernels kernels(0, &hostAlloc);

  std::array<float, 4> matrix = {1, 1, 0, 1};

  auto is = kernels.isLowerTriangular(2, 2, matrix.data(), 2, 0);
  EXPECT_TRUE(is);
}

} // namespace mtrx
