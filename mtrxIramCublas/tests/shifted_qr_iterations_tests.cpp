#include <exception>
#include <gtest/gtest.h>
#include <mtrxCublas/cublas.hpp>
#include <mtrxIram/iram.hpp>
#include <stdexcept>

namespace mtrx {

class ShiftedQRIterationCublasTests : public testing::Test {
public:
  virtual void setUp() {}
  virtual void tearDown() {}
};

TEST_F(IramCublasTests, StartWithoutInitVector) {
  

}

} // namespace mtrx
