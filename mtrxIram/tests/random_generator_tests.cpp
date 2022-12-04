#include <gtest/gtest.h>
#include <mtrxIram/impl/random_generator.hpp>

namespace mtrx {

class RandomGeneratorTests : public testing::Test {
public:
  virtual void setUp() {}
  virtual void tearDown() {}
};

TEST_F(RandomGeneratorTests, Init) { RandomGenerator<float> rg(0., 1.); }

} // namespace mtrx
