#include <array>
#include <gtest/gtest.h>

#include "../src/kernels.hpp"
#include <spdlog/spdlog.h>

namespace mtrx {
class HostCublasTests : public testing::Test {
public:
  void SetUp() override {
    spdlog::set_level(spdlog::level::debug);
  }

  template<typename T>
  void initDiagonalMatrix(T* matrix, int dim, T diagonalValue, T otherValue)
  {
    for (int x = 0; x < dim; ++x)
    {
      for (int y = 0; y < dim; y++)
      {
        if (x == y)
        {
          matrix[y+dim*x] = diagonalValue;
        }
        else
        {
          matrix[y+dim*x] = otherValue;
        }
      }
    }
  }
};

TEST_F(HostCublasTests, Kernel_SF_scaleTrace) {
  try {
    int dim = 10;

    float* matrix = new float[dim * dim];
    std::unique_ptr<float, std::function<void(float*)>> matrixUnqiue(matrix, [](float* matrix){ delete[] matrix; });

    initDiagonalMatrix(matrix, dim, 2.f, 10.f);

    int lda = 10;
    float factor = 0.5f;

    Kernel_SF_scaleTrace(dim, matrix, lda, factor);

    for (int x = 0; x < dim; ++x)
    {
      for (int y = 0; y < dim; y++)
      {
        if (x == y)
        {
          EXPECT_EQ(1.f, matrix[y+dim*x]) << "(" << x << ", " << y << ")";
        }
        else
        {
          EXPECT_EQ(10.f, matrix[y+dim*x]) << "(" << x << ", " << y << ")";
        }
      }
    }
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}
} // namespace mtrx
