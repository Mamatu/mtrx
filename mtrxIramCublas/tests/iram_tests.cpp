#include <gtest/gtest.h>
#include <mtrxCublas/cublas.hpp>
#include <mtrxIram/iram.hpp>

namespace mtrx {

class IramCublasTests : public testing::Test {
public:
  virtual void setUp() {}
  virtual void tearDown() {}
};

TEST_F(IramCublasTests, StartWithoutInitVector) {
  auto cublas = std::make_shared<mtrx::Cublas<float>>();
  mtrx::Iram<float> iram(cublas);
  EXPECT_THROW(iram.start(), std::runtime_error);
}

TEST_F(IramCublasTests, StartWithRandomInitVector) {
  auto cublas = std::make_shared<mtrx::Cublas<float>>();
  mtrx::Iram<float> iram(cublas);
  iram.setRandomVectorToInit(8);
  EXPECT_NO_THROW(iram.start());
}

TEST_F(IramCublasTests, StartWithUnitInitVector) {
  auto cublas = std::make_shared<mtrx::Cublas<float>>();
  mtrx::Iram<float> iram(cublas);
  iram.setUnitVectorToInit(8);
  EXPECT_NO_THROW(iram.start());
}

} // namespace mtrx
