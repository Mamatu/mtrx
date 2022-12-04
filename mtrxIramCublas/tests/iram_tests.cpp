#include <gtest/gtest.h>
#include <mtrxCublas/cublas.hpp>
#include <mtrxIram/iram.hpp>

namespace mtrx {

class IramCublasTests : public testing::Test {
public:
  virtual void setUp() {}
  virtual void tearDown() {}
};

TEST_F(IramCublasTests, Init) {
  auto cublas = std::make_shared<mtrx::Cublas<float>>();
  mtrx::Iram<float> iram(cublas);
  iram.start();
}

} // namespace mtrx
