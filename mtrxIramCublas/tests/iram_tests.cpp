#include <exception>
#include <gtest/gtest.h>
#include <mtrxCublas/cublas.hpp>
#include <mtrxIram/iram.hpp>
#include <stdexcept>

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
  EXPECT_NO_THROW(
      try {
        auto cublas = std::make_shared<mtrx::Cublas<float>>();
        mtrx::Iram<float> iram(cublas);
        iram.setRandomVectorToInit(8);
        iram.start();
      } catch (const std::exception &e) {
        spdlog::error(e.what());
        throw e;
      });
}

TEST_F(IramCublasTests, StartWithUnitInitVector) {
  EXPECT_NO_THROW(
      try {
        auto cublas = std::make_shared<mtrx::Cublas<float>>();
        mtrx::Iram<float> iram(cublas);
        iram.setUnitVectorToInit(8);
        iram.start();
      } catch (const std::exception &e) {
        spdlog::error(e.what());
        throw e;
      });
}

} // namespace mtrx
