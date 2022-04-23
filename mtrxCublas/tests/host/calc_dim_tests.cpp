#include <array>

#include <mtrxCublas/test.hpp>

#include "../src/calc_dim.hpp"

namespace mtrx {
class CalcDimTests : public Test {
public:
};

TEST_F(CalcDimTests, 1x1) {
  dim3 threads = {0, 0};
  dim3 blocks = {0, 0};
  std::array<int, 3> blockDim = {1, 1, 1};
  std::array<int, 3> gridDim = {1, 1, 1};
  calculateDim(threads, blocks, 1, 1, blockDim, gridDim, 1);
  EXPECT_EQ(1, threads.x);
  EXPECT_EQ(1, threads.y);
  EXPECT_EQ(1, blocks.x);
  EXPECT_EQ(1, blocks.y);
}

TEST_F(CalcDimTests, blockDim_1x1_gridDim_2x2) {
  dim3 threads = {0, 0};
  dim3 blocks = {0, 0};
  std::array<int, 3> blockDim = {1, 1, 1};
  std::array<int, 3> gridDim = {2, 2, 1};
  calculateDim(threads, blocks, 2, 2, blockDim, gridDim, 1);
  EXPECT_EQ(1, threads.x);
  EXPECT_EQ(1, threads.y);
  EXPECT_EQ(2, blocks.x);
  EXPECT_EQ(2, blocks.y);
}

TEST_F(CalcDimTests, 32x32) {
  dim3 threads = {0, 0};
  dim3 blocks = {0, 0};
  std::array<int, 3> blockDim = {32, 32, 1};
  std::array<int, 3> gridDim = {1, 1, 1};
  calculateDim(threads, blocks, 32, 32, blockDim, gridDim, 1024);
  EXPECT_EQ(32, threads.x);
  EXPECT_EQ(32, threads.y);
  EXPECT_EQ(1, blocks.x);
  EXPECT_EQ(1, blocks.y);
}

TEST_F(CalcDimTests, 64x64) {
  dim3 threads = {0, 0};
  dim3 blocks = {0, 0};
  std::array<int, 3> blockDim = {32, 32, 1};
  std::array<int, 3> gridDim = {32, 32, 1};
  calculateDim(threads, blocks, 64, 64, blockDim, gridDim, 1024);
  EXPECT_EQ(32, threads.x);
  EXPECT_EQ(32, threads.y);
  EXPECT_EQ(2, blocks.x);
  EXPECT_EQ(2, blocks.y);
}
} // namespace mtrx
