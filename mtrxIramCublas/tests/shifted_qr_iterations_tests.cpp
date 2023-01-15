#include <exception>
#include <gtest/gtest.h>
#include <mtrxCublas/cublas.hpp>
#include <mtrxIram/iram.hpp>
#include <mtrxIram/shifted_qr_iteration.hpp>
#include <stdexcept>

namespace mtrx {

class ShiftedQRIterationCublasTests : public testing::Test {
public:
  virtual void setUp() {}
  virtual void tearDown() {}
};

TEST_F(ShiftedQRIterationCublasTests, Test_1) {
  mtrx::Cublas<float> cublas;
  float *A = nullptr;
  float *V = nullptr;
  float *H = nullptr;
  mtrx::ShiftedQRIteration sqri(cublas, A, V, H);
  sqri();
}

} // namespace mtrx
