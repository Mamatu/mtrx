#include <mtrxCublas/test.hpp>
#include <math.h>
#include <mtrxCore/types.hpp>
#include <mtrxCublas/cublas.hpp>
#include <mtrxCublas/matchers.hpp>

#include <array>
#include <cuda.h>

namespace mtrx {
class DeviceCublasTests : public Test {
public:
  void SetUp() override {
#ifdef CUBLAS_NVPROF_TESTS
    startProfiler();
#endif
  }

  void TearDown() override {
#ifdef CUBLAS_NVPROF_TESTS
    stopProfiler();
#endif
  }
};

TEST_F(DeviceCublasTests, create_destroy) {
  try {
    mtrx::Cublas cublas;
    auto *mem = cublas.create(4 * 5, ValueType::FLOAT);
    cublas.destroy(mem);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, create_checkvalues_destroy) {
  try {
    mtrx::Cublas cublas;
    std::vector<float> h_mem(4 * 5, -1.f);

    auto *mem = cublas.create(4 * 5, ValueType::FLOAT);

    for (size_t idx = 0; idx < h_mem.size(); ++idx) {
      EXPECT_EQ(-1.f, h_mem[idx]);
    }

    cublas.copyKernelToHost(h_mem.data(), mem);

    for (size_t idx = 0; idx < h_mem.size(); ++idx) {
      EXPECT_EQ(0.f, h_mem[idx]);
    }

    cublas.destroy(mem);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, create_copyHostToKernel_copyKernelToHost_destroy) {
  try {
    mtrx::Cublas cublas;
    constexpr auto count = 20;
    auto *mem = cublas.create(count, ValueType::FLOAT);
    std::array<float, count> array;
    std::array<float, count> array1;
    std::vector<float> vec1;
    vec1.resize(count);
    for (int idx = 0; idx < count; ++idx) {
      array[idx] = idx;
      array1[idx] = 0;
      vec1[idx] = -1;
    }
    cublas.copyHostToKernel(mem, array.data());
    cublas.copyKernelToHost(array1.data(), mem);
    cublas.copyKernelToHost(vec1.data(), mem);

    for (int idx = 0; idx < count; ++idx) {
      EXPECT_EQ(idx, array1[idx]) << idx;
      EXPECT_EQ(idx, vec1[idx]) << idx;
    }
    cublas.destroy(mem);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, createIdentityMatrix) {
  try {
    mtrx::Cublas cublas;

    int columns = 5;
    int rows = 5;
    auto *mem = cublas.createIdentityMatrix(columns, rows, ValueType::FLOAT);

    std::vector<float> h_matrix(columns * rows, -1.f);
    cublas.copyKernelToHost(h_matrix.data(), mem);

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < columns; ++c) {
        if (c == r) {
          EXPECT_EQ(1.f, h_matrix[r * columns + c])
              << "row: " << r << ", column: " << c;
        } else {
          EXPECT_EQ(0.f, h_matrix[r * columns + c])
              << "row: " << r << ", column: " << c;
        }
      }
    }

    cublas.destroy(mem);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, create_getCount_destroy) {
  try {
    mtrx::Cublas cublas;

    auto *mem = cublas.create(4 * 5, ValueType::FLOAT);
    auto count = cublas.getCount(mem);

    EXPECT_EQ(4 * 5, count);

    cublas.destroy(mem);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, create_copyHostToKernel_amax_amin_destroy) {
  try {
    mtrx::Cublas cublas;

    auto *mem = cublas.create(4 * 5, ValueType::FLOAT);

    auto count = cublas.getCount(mem);

    EXPECT_EQ(4 * 5, count);

    std::vector<float> array;
    array.reserve(count);

    for (size_t idx = 0; idx < count; ++idx) {
      array.push_back(idx + 2);
    }

    uintt maxIdx = count / 2;
    uintt minIdx = maxIdx - 1;
    array[maxIdx] = 100.f;
    array[minIdx] = 1.f;

    cublas.copyHostToKernel(mem, array.data());
    EXPECT_EQ(maxIdx, cublas.amax(mem));
    EXPECT_EQ(minIdx, cublas.amin(mem));

    cublas.destroy(mem);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, create_copyHostToKernel_rot_destroy) {
  try {
    mtrx::Cublas cublas;

    auto *x = cublas.create(2, ValueType::FLOAT);
    auto *y = cublas.create(2, ValueType::FLOAT);

    auto countx = cublas.getCount(x);
    auto county = cublas.getCount(y);

    EXPECT_EQ(2, countx);
    EXPECT_EQ(2, county);

    std::vector<float> array(2, 1.f);

    cublas.copyHostToKernel(x, array.data());
    cublas.copyHostToKernel(y, array.data());

    cublas.rot(x, y, 1.f, 1.f);

    std::vector<float> hostx(2, 0.f);
    std::vector<float> hosty(2, 0.f);

    cublas.copyKernelToHost(hostx.data(), x);
    cublas.copyKernelToHost(hosty.data(), y);

    std::vector<float> expectedx({2.f, 2.f});
    std::vector<float> expectedy({0.f, 0.f});

    for (size_t idx = 0; idx < array.size(); ++idx) {
      EXPECT_EQ(expectedx[idx], hostx[idx])
          << "Mismatch in hostx at index: " << idx;
      EXPECT_EQ(expectedy[idx], hosty[idx])
          << "Mismatch in hosty at index: " << idx;
    }

    cublas.destroy(x);
    cublas.destroy(y);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests,
       create_copyHostToKernel_add_copyKernelToHost_destroy) {
  try {
    mtrx::Cublas cublas;

    auto *a = cublas.createMatrix(3, 2, ValueType::FLOAT);
    auto *b = cublas.createMatrix(3, 2, ValueType::FLOAT);
    auto *output = cublas.createMatrix(3, 2, ValueType::FLOAT);

    auto counta = cublas.getCount(a);
    auto countb = cublas.getCount(b);

    EXPECT_EQ(6, counta);
    EXPECT_EQ(6, countb);

    std::vector<float> arraya({0, 4, -1, 11, 2, 2});
    std::vector<float> arrayb({3, 1, 6, -1, 2, 1});

    cublas.copyHostToKernel(a, arraya.data());
    cublas.copyHostToKernel(b, arrayb.data());
    cublas.add(output, a, b);

    std::vector<float> h_output;
    h_output.resize(6);
    cublas.copyKernelToHost(h_output.data(), output);

    EXPECT_EQ(3, h_output[0]);
    EXPECT_EQ(5, h_output[1]);
    EXPECT_EQ(5, h_output[2]);
    EXPECT_EQ(10, h_output[3]);
    EXPECT_EQ(4, h_output[4]);
    EXPECT_EQ(3, h_output[5]);

    cublas.destroy(a);
    cublas.destroy(b);
    cublas.destroy(output);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests,
       create_copyHostToKernel_subtract_copyKernelToHost_destroy) {
  try {
    Cublas cublas;
    auto *a = cublas.createMatrix(3, 2, ValueType::FLOAT);
    auto *b = cublas.createMatrix(3, 2, ValueType::FLOAT);
    auto *output = cublas.createMatrix(3, 2, ValueType::FLOAT);

    auto counta = cublas.getCount(a);
    auto countb = cublas.getCount(b);

    EXPECT_EQ(6, counta);
    EXPECT_EQ(6, countb);

    std::vector<float> arraya({0, 4, -1, 11, 2, 2});
    std::vector<float> arrayb({3, 1, 6, -1, 2, 1});

    cublas.copyHostToKernel(a, arraya.data());
    cublas.copyHostToKernel(b, arrayb.data());
    cublas.subtract(output, a, b);

    std::vector<float> h_output;
    h_output.resize(6);
    cublas.copyKernelToHost(h_output.data(), output);

    EXPECT_EQ(-3, h_output[0]);
    EXPECT_EQ(3, h_output[1]);
    EXPECT_EQ(-7, h_output[2]);
    EXPECT_EQ(12, h_output[3]);
    EXPECT_EQ(0, h_output[4]);
    EXPECT_EQ(1, h_output[5]);

    cublas.destroy(a);
    cublas.destroy(b);
    cublas.destroy(output);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests,
       create_copyHostToKernel_matrixMul_copyKernelToHost_destroy) {
  try {
    mtrx::Cublas cublas;

    auto *a = cublas.createMatrix(2, 3, ValueType::FLOAT);
    auto *b = cublas.createMatrix(3, 2, ValueType::FLOAT);
    auto *output = cublas.createMatrix(2, 2, ValueType::FLOAT);

    auto counta = cublas.getCount(a);
    auto countb = cublas.getCount(b);

    EXPECT_EQ(6, counta);
    EXPECT_EQ(6, countb);

    std::vector<float> arraya({0, 4, -1, 11, 2, 2});
    std::vector<float> arrayb({3, 1, 6, -1, 2, 1});

    cublas.copyHostToKernel(a, arraya.data());
    cublas.copyHostToKernel(b, arrayb.data());
    cublas.matrixMul(output, a, b);

    std::vector<float> h_output;
    h_output.resize(4);
    cublas.copyKernelToHost(h_output.data(), output);

    EXPECT_EQ(11, h_output[0]);
    EXPECT_EQ(35, h_output[1]);
    EXPECT_EQ(0, h_output[2]);
    EXPECT_EQ(20, h_output[3]);

    cublas.destroy(a);
    cublas.destroy(b);
    cublas.destroy(output);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests,
       create_copyHostToKernel_syr_copyKernelToHost_destroy) {
  try {
    mtrx::Cublas cublas;

    auto *v = cublas.create(3, ValueType::FLOAT);
    auto *outputUpper = cublas.createMatrix(3, 3, ValueType::FLOAT);
    auto *outputLower = cublas.createMatrix(3, 3, ValueType::FLOAT);
    auto *outputFull = cublas.createMatrix(3, 3, ValueType::FLOAT);
    float alpha = 1.f;

    std::vector<float> h_v = {1, 2, 3};
    std::vector<float> h_outputUpper(9, 0.f);
    std::vector<float> h_outputLower(9, 0.f);
    std::vector<float> h_outputFull(9, 0.f);

    //   1 2 3
    // 1 1 2 3
    // 2 2 4 6
    // 3 3 6 9
    std::vector<float> eh_outputFull = {1, 2, 3, 2, 4, 6, 3, 6, 9};

    cublas.copyHostToKernel(v, h_v.data());

    cublas.copyHostToKernel(outputUpper, h_outputUpper.data());
    cublas.copyHostToKernel(outputLower, h_outputLower.data());
    cublas.copyHostToKernel(outputFull, h_outputFull.data());

    cublas.syr(FillMode::UPPER, outputUpper, &alpha, ValueType::FLOAT, v);
    cublas.syr(FillMode::LOWER, outputLower, &alpha, ValueType::FLOAT, v);

    cublas.syr(FillMode::FULL, outputFull, &alpha, ValueType::FLOAT, v);

    cublas.copyKernelToHost(h_outputUpper.data(), outputUpper);
    cublas.copyKernelToHost(h_outputLower.data(), outputLower);
    cublas.copyKernelToHost(h_outputFull.data(), outputFull);

    for (int c = 0; c < 3; ++c) {
      for (int r = 0; r < 3; ++r) {
        EXPECT_EQ(eh_outputFull[c * 3 + r], h_outputFull[c * 3 + r])
            << "r: " << r << " c: " << c;
      }
    }

    cublas.destroy(v);
    cublas.destroy(outputUpper);
    cublas.destroy(outputLower);
    cublas.destroy(outputFull);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, qrDecomposition2x2) {
  std::array<float, 4> h_a = {-1, 1, 3, 5};
  std::array<float, 4> h_a_1 = {0, 0, 0, 0};
  const float sqrt2 = sqrt(2.);
  std::array<float, 4> expected_r = {sqrt2, 0, sqrt2, 4 * sqrt2};
  std::array<float, 4> h_r = {0, 0, 0, 0};
  std::array<float, 4> expected_q = {-1.f / sqrt2, 1.f / sqrt2, 1.f / sqrt2,
                                     1.f / sqrt2};
  std::array<float, 4> h_q = {0, 0, 0, 0};
  std::array<float, 4> h_I = {-1, -1, -1, -1};
  try {
    constexpr auto delta = 0.000001f;
    mtrx::Cublas cublas;

    Mem *a = cublas.createMatrix(2, 2, ValueType::FLOAT);
    Mem *a1 = cublas.createMatrix(2, 2, ValueType::FLOAT);
    Mem *q = cublas.createMatrix(2, 2, ValueType::FLOAT);
    Mem *r = cublas.createMatrix(2, 2, ValueType::FLOAT);
    Mem *I = cublas.createMatrix(2, 2, ValueType::FLOAT);

    cublas.copyHostToKernel(a, h_a.data());
    cublas.copyHostToKernel(r, h_r.data());
    cublas.copyHostToKernel(q, h_q.data());
    cublas.qrDecomposition(q, r, a);
    cublas.copyKernelToHost(h_q.data(), q);
    cublas.copyKernelToHost(h_r.data(), r);

    float alpha = 1.f;
    float beta = 0.f;
    cublas.gemm(I, &alpha, ValueType::FLOAT, Operation::OP_T, q,
                Operation::OP_N, q, &beta, ValueType::FLOAT);

    cublas.matrixMul(a1, q, r);
    cublas.copyKernelToHost(h_a_1.data(), a1);
    cublas.copyKernelToHost(h_I.data(), I);

    EXPECT_THAT(a, MemIsEqualTo(a1, &cublas, delta));
    EXPECT_THAT(r, MemIsEqualTo(expected_r, &cublas, delta));
    EXPECT_THAT(q, MemIsEqualTo(expected_q, &cublas, delta));

    cublas.destroy(a);
    cublas.destroy(a1);
    cublas.destroy(q);
    cublas.destroy(r);
    cublas.destroy(I);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, qrDecomposition3x3) {
  std::array<float, 9> h_a = {12, 6, -4, -51, 167, 24, 4, -68, -41};

  std::array<float, 9> h_r
      __attribute__((unused)) = {14, 0, 0, 21, 175, 0, -14, -70, 35};
  std::array<float, 9> h_q __attribute__((unused)) = {
      6.f / 7.f,  3.f / 7.f,     -2.f / 7.f,  -69.f / 175.f, 158.f / 175.f,
      6.f / 35.f, -58.f / 175.f, 6.f / 175.f, -33.f / 35.f};

  std::array<float, 9> he_r = {-14, 0, 0, -21, -175, 0, 14, 70, -35};
  std::array<float, 9> he_q = {-6.f / 7.f,   -3.f / 7.f,     2.f / 7.f,
                               69.f / 175.f, -158.f / 175.f, -6.f / 35.f,
                               58.f / 175.f, -6.f / 175.f,   33.f / 35.f};
  try {
    constexpr auto delta = 0.0001f;
    mtrx::Cublas cublas;

    Mem *a = cublas.createMatrix(3, 3, ValueType::FLOAT);
    Mem *a1 = cublas.createMatrix(3, 3, ValueType::FLOAT);
    Mem *q = cublas.createMatrix(3, 3, ValueType::FLOAT);
    Mem *r = cublas.createMatrix(3, 3, ValueType::FLOAT);

    cublas.copyHostToKernel(a, h_a.data());
    cublas.qrDecomposition(q, r, a);
    cublas.matrixMul(a1, q, r);

    EXPECT_THAT(r, MemIsEqualTo(he_r, &cublas, delta));
    EXPECT_THAT(q, MemIsEqualTo(he_q, &cublas, delta));
    EXPECT_THAT(a1, MemIsEqualTo(a, &cublas, delta));

    cublas.destroy(a);
    cublas.destroy(a1);
    cublas.destroy(q);
    cublas.destroy(r);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}

TEST_F(DeviceCublasTests, copyKernelToKernel) {
  try {
    std::array<float, 9> h_a = {12, 6, -4, -51, 167, 24, 4, -68, -41};
    std::array<float, 9> h_b = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<float, 9> h_c = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    mtrx::Cublas cublas;

    Mem *a = cublas.createMatrix(3, 3, ValueType::FLOAT);
    Mem *b = cublas.createMatrix(3, 3, ValueType::FLOAT);
    Mem *c = cublas.createMatrix(3, 3, ValueType::FLOAT);

    auto *I = cublas.createIdentityMatrix(3, 3, ValueType::FLOAT);

    cublas.copyHostToKernel(a, h_a.data());

    cublas.matrixMul(b, a, I);
    cublas.matrixMul(c, I, a);

    cublas.copyKernelToHost(h_b.data(), b);
    cublas.copyKernelToHost(h_c.data(), c);
    for (size_t idx = 0; idx < h_a.size(); ++idx) {
      EXPECT_EQ(h_a[idx], h_b[idx]) << "idx: " << idx;
      EXPECT_EQ(h_a[idx], h_c[idx]) << "idx: " << idx;
    }

    cublas.destroy(a);
    cublas.destroy(b);
    cublas.destroy(c);
  } catch (const std::exception &ex) {
    FAIL() << ex.what();
  }
}
} // namespace mtrx
