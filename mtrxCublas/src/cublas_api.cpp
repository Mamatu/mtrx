#include "cublas_api.hpp"
#include <sstream>

namespace mtrx {
namespace {
template <typename T, typename Cublas_Gemm>
void cublas_api_gemm(T *output, T *alpha, Operation transa, T *a,
                     Operation transb, T *b, T *beta,
                     Cublas_Gemm &&cublas_gemm) {

  const auto m = getRows(a);
  const auto m1 = getRows(output);

  if (m != m1) {
    std::stringstream sstream;
    sstream
        << "Rows of 'a' matrix is not equal to rows of 'output'. Rows('a'): "
        << m << " Rows('output'): " << m1;
    throw std::runtime_error(sstream.str());
  }

  const auto n = getColumns(b);
  const auto n1 = getColumns(output);

  if (n != n1) {
    std::stringstream sstream;
    sstream << "Columns of 'b' matrix is not equal to columns of 'output'. "
               "Columns('b'): "
            << n << " Columns('output'): " << n1;
    throw std::runtime_error(sstream.str());
  }

  const auto k = getColumns(a);
  const auto k1 = getRows(b);

  if (k != k1) {
    std::stringstream sstream;
    sstream << "Columns of 'a' matrix is not equal to rows of 'b' matrix. "
               "Columns('a'): "
            << k << " Rows('b'): " << k1;
    throw std::runtime_error(sstream.str());
  }

  auto cutransa = convert(transa);
  auto cutransb = convert(transb);

  auto status = cublas_gemm(cutransa, cutransb, m, n, k, alpha, a, m, b, k,
                            beta, output, m);
  handleStatus(status);
}
} // namespace
void CublasApi::gemm(cublasHandle_t handle, float *output, float *alpha,
                     Operation transa, float *a, Operation transb, float *b,
                     float *beta) {
  auto gemm = [](auto cutransa, auto cutransb, auto m, auto n, auto k,
                 auto *alpha, auto *A, auto lda, auto *B, auto ldb, auto *beta,
                 auto *C, auto ldc) {
    return cublasSgemm(cutrasna, cutransb, m, n, k, alpha, A, lda, B, ldb, beta,
                       C, ldc);
  };
  cublas_api_gemm(output, alpha, transa, a, transb, b, beta, std::move(gemm));
}

void CublasApi::gemm(cublasHandle_t handle, double *output, double *alpha,
                     Operation transa, double *a, Operation transb, double *b,
                     double *beta) {}

void CublasApi::gemm(cublasHandle_t handle, cuComplex *output, cuComplex *alpha,
                     Operation transa, cuComplex *a, Operation transb,
                     cuComplex *b, cuComplex *beta) {}

void CublasApi::gemm(cublasHandle_t handle, cuDoubleComplex *output,
                     cuDoubleComplex *alpha, Operation transa,
                     cuDoubleComplex *a, Operation transb, cuDoubleComplex *b,
                     cuDoubleComplex *beta) {}
} // namespace mtrx
