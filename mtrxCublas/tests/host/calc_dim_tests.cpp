#include <array>
#include <gtest/gtest.h>

#include "../src/calc_dim.hpp"
#include <spdlog/spdlog.h>

namespace mtrx {
class CalcDimTests : public testing::Test {
public:
  void SetUp() override { spdlog::set_level(spdlog::level::debug); }
};

TEST_F(CalcDimTests, 1x1) {
  std::array<int, 2> threads = {0, 0};
  std::array<int, 2> blocks = {0, 0};
  std::array<int, 3> blockDim = {1, 1, 1};
  std::array<int, 3> gridDim = {1, 1, 1};
  calculateDim(threads, blocks, 1, 1, blockDim, gridDim, 1);
  EXPECT_EQ(1, threads[0]);
  EXPECT_EQ(1, threads[1]);
  EXPECT_EQ(1, blocks[0]);
  EXPECT_EQ(1, blocks[1]);
}

TEST_F(CalcDimTests, 32x32) {
  std::array<int, 2> threads = {0, 0};
  std::array<int, 2> blocks = {0, 0};
  std::array<int, 3> blockDim = {32, 32, 1};
  std::array<int, 3> gridDim = {1, 1, 1};
  calculateDim(threads, blocks, 32, 32, blockDim, gridDim, 1024);
  EXPECT_EQ(32, threads[0]);
  EXPECT_EQ(32, threads[1]);
  EXPECT_EQ(1, blocks[0]);
  EXPECT_EQ(1, blocks[1]);
}

TEST_F(CalcDimTests, 64x64) {
  std::array<int, 2> threads = {0, 0};
  std::array<int, 2> blocks = {0, 0};
  std::array<int, 3> blockDim = {32, 32, 1};
  std::array<int, 3> gridDim = {32, 32, 1};
  calculateDim(threads, blocks, 64, 64, blockDim, gridDim, 1024);
  EXPECT_EQ(32, threads[0]);
  EXPECT_EQ(32, threads[1]);
  EXPECT_EQ(2, blocks[0]);
  EXPECT_EQ(2, blocks[1]);
}
} // namespace mtrx
