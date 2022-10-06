/*
 * Copyright 2021 - 2022 Marcin Matula
 *
 * This file is part of mtrx.
 *
 * mtrx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * mtrx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with mtrx.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MTRX_CUBLAS_HPP
#define MTRX_CUBLAS_HPP

#include "mtrxCublas/status_handler.hpp"
#include <cstdint>
#include <cublas_v2.h>
#include <sstream>

#include <mtrxCore/blas.hpp>
#include <mtrxCore/size_of.hpp>
#include <mtrxCore/types.hpp>

#include <mtrxCublas/impl/cu_convert.hpp>
#include <mtrxCublas/impl/cublas_kernels.hpp>
#include <mtrxCublas/impl/cuda_alloc.hpp>
#include <mtrxCublas/impl/cuda_kernels.hpp>
#include <type_traits>

namespace mtrx {

namespace {
template <typename T> void swap(T **a, T **b) {
  T *temp = *a;
  *a = *b;
  *b = temp;
}

template <typename T> T *cublas_getOffset(T *ptr, int idx) { return &ptr[idx]; }

template <typename T, typename Vec>
void cublas_setVec(Vec &&vec, int rows, int columns) {
  vec.reserve(rows * columns);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < columns; ++c) {
      if (r == c) {
        vec.push_back(cu_convert<T>(1));
      } else {
        vec.push_back(cu_convert<T>(0));
      }
    }
  }
}

} // namespace

template <typename T> class Cublas : public Blas<T> {
public:
  using Vec = typename Blas<T>::Vec;

  Cublas() {}
  ~Cublas() override {}

protected:
  std::vector<int> _getDevices() const override;
  void _setDevice(int device) override;

  T *_create(int size) override;
  void _destroy(const T *mem) override;

  T *_createIdentityMatrix(int rows, int columns) override;

  int _getCount(const T *mem) const override;
  int _getSizeInBytes(const T *mem) const override;

  void _copyHostToKernel(T *mem, const T *array, int count) override;
  void _copyKernelToHost(T *array, const T *mem, int count) override;

  uintt _amax(const T *mem) override;
  uintt _amin(const T *mem) override;

  void _rot(T *x, T *y, T &&c, T &&s) override;

  void _syr(FillMode fillMode, T *output, T *alpha, T *x) override;

  void _gemm(T *output, T *alpha, Operation transa, T *a, Operation transb,
             T *b, T *beta) override;

  void _gemm(T *output, T alpha, Operation transa, T *a, Operation transb, T *b,
             T beta);

  void _symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha, T *a,
             T *b, T *beta) override;

  void _matrixMul(T *output, T *a, T *b) override;

  void _geqrf(T *a, T *tau) override;
  void _geqrf(Vec &a, Vec &tau) override;

  void _qrDecomposition(T *q, T *r, T *a) override;
  void _qrDecomposition(const Vec &q, const Vec &r, const Vec &a) override;

  void _shiftQRIteration(T *H, T *Q) override;

  bool _isUpperTriangular(T *m) override;
  bool _isLowerTriangular(T *m) override;

  void _geam(T *output, T *alpha, Operation transa, T *a, T *beta,
             Operation transb, T *b) override;

  void _add(T *output, T *a, T *b) override;
  void _subtract(T *output, T *a, T *b) override;

  void _scaleDiagonal(T *matrix, T *factor) override;
  void _scaleDiagonal(T *matrix, T factor);

  bool _isComplex() const override;

  void _tpttr(FillMode uplo, int n, T *AP, T *A, int lda) override;
  void _trttp(FillMode uplo, int n, T *A, int lda, T *AP) override;

  bool _isUnit(T *mem, T *delta) override;

  std::string _toStr(T *mem) override;

private:
  CublasKernels m_cublasKernels;
  std::unordered_map<uintptr_t, uint> m_counts;

  T *alloc(int count);
  void dealloc(T *mem);

  std::pair<int, int> checkForTriangular(T *matrix) {
    auto rows = this->getRows(matrix);
    auto columns = this->getColumns(matrix);

    if (rows != columns) {
      std::stringstream sstream;
      sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
              << columns;
      throw std::runtime_error(sstream.str());
    }
    return std::make_pair(rows, columns);
  }
};
} // namespace mtrx

namespace mtrx {

namespace {

template <typename T> T *cublas_allocCudaArrayCopy(const T *array, int count) {
  T *d_array = nullptr;
  auto error = cudaMalloc(&d_array, sizeof(T) * count);
  handleStatus(error);

  auto error1 =
      cudaMemcpy(d_array, array, sizeof(T) * count, cudaMemcpyHostToDevice);
  handleStatus(error1);

  return d_array;
}

template <typename T> T *cublas_allocCudaArrayCopy(const std::vector<T> &vec) {
  return cublas_allocCudaArrayCopy(vec.data(), vec.size());
}

template <typename T> void cublas_deallocCudaArray(T *array) {
  auto error = cudaFree(array);
  handleStatus(error);
}

inline cublasOperation_t convert(Operation operation) {
  switch (operation) {
  case Operation::OP_N:
    return CUBLAS_OP_N;
  case Operation::OP_T:
    return CUBLAS_OP_T;
  case Operation::OP_C:
    return CUBLAS_OP_C;
  };

  throw std::runtime_error("Not defined side mode");
  return CUBLAS_OP_N;
}

inline cublasSideMode_t convert(SideMode sideMode) {
  switch (sideMode) {
  case SideMode::LEFT:
    return CUBLAS_SIDE_LEFT;
  case SideMode::RIGHT:
    return CUBLAS_SIDE_RIGHT;
  };

  throw std::runtime_error("Not defined side mode");
  return CUBLAS_SIDE_LEFT;
}

inline cublasFillMode_t convert(FillMode fillMode) {
  switch (fillMode) {
  case FillMode::FULL:
    return CUBLAS_FILL_MODE_FULL;
  case FillMode::LOWER:
    return CUBLAS_FILL_MODE_LOWER;
  case FillMode::UPPER:
    return CUBLAS_FILL_MODE_UPPER;
  }

  throw std::runtime_error("Not defined fill mode");
  return CUBLAS_FILL_MODE_UPPER;
}

} // namespace

template <typename T> std::vector<int> Cublas<T>::_getDevices() const {
  int count = 0;
  handleStatus(cudaGetDeviceCount(&count));

  std::vector<int> devices;
  devices.reserve(count);

  for (int i = 0; i < count; ++i) {
    devices.push_back(i);
  }
  return devices;
}

template <typename T> void Cublas<T>::_setDevice(int device) {
  handleStatus(cudaSetDevice(device));
}

template <typename T> T *Cublas<T>::_create(int count) {
  T *d_mem = nullptr;
  const auto sizeInBytes = count * sizeof(T);
  auto status = cudaMalloc(&d_mem, sizeInBytes);
  cudaMemset(d_mem, 0, sizeInBytes);
  m_counts[reinterpret_cast<uintptr_t>(d_mem)] = count;
  handleStatus(status);
  return d_mem;
}

template <typename T> void Cublas<T>::_destroy(const T *mem) {
  auto error = cudaFree(const_cast<T *>(mem));
  handleStatus(error);
}

template <typename T>
T *Cublas<T>::_createIdentityMatrix(int rows, int columns) {
  T *matrix = Blas<T>::createMatrix(rows, columns);
  std::vector<T> vec;
  cublas_setVec<T>(vec, rows, columns);
  this->copyHostToKernel(matrix, vec.data());
  return matrix;
}

template <typename T> int Cublas<T>::_getCount(const T *mem) const {
  uintptr_t handler = reinterpret_cast<uintptr_t>(mem);
  auto it = m_counts.find(handler);
  if (it == m_counts.end()) {
    std::stringstream sstream;
    sstream << "FATAL: Not found count for " << mem;
    throw std::runtime_error(sstream.str());
  }
  return it->second;
}

template <typename T> int Cublas<T>::_getSizeInBytes(const T *mem) const {
  auto count = this->getCount(mem);
  return SizeOf<T>(count);
}

template <typename T>
void Cublas<T>::_copyHostToKernel(T *dst, const T *src, int count) {
  auto status = cublasSetVector(count, SizeOf<T>(), src, 1, dst, 1);
  handleStatus(status);
}

template <typename T>
void Cublas<T>::_copyKernelToHost(T *dst, const T *src, int count) {
  auto status = cublasGetVector(count, SizeOf<T>(), src, 1, dst, 1);
  handleStatus(status);
}

template <typename T> uintt Cublas<T>::_amax(const T *mem) {
  const auto n = _getCount(mem);
  return m_cublasKernels.amax(mem, n);
}

template <typename T> uintt Cublas<T>::_amin(const T *mem) {
  const auto n = _getCount(mem);
  return m_cublasKernels.amin(mem, n);
}

template <typename T> void Cublas<T>::_rot(T *x, T *y, T &&c, T &&s) {
  auto x_count = this->getCount(x);
  auto y_count = this->getCount(y);
  if (x_count != y_count) {
    std::stringstream sstream;
    sstream << "Different count for x and y. x: " << x_count
            << " y: " << y_count;
    throw std::runtime_error(sstream.str());
  }

  const auto &n = x_count;
  m_cublasKernels.rot(n, x, y, std::forward<T>(c), std::forward<T>(s));
}

template <typename T>
void Cublas<T>::_syr(FillMode fillMode, T *output, T *alpha, T *x) {
  int n = this->getRows(output);
  int m = this->getColumns(output);
  const int lda = n;

  if (n != m) {
    std::stringstream sstream;
    sstream << "Matrix 'output' must be square not " << n << "x" << m;
    throw std::runtime_error(sstream.str());
  }

  auto handle = m_cublasKernels;

  auto call = [this, &handle, n, lda, output, alpha, x](FillMode fillMode) {
    m_cublasKernels.syr(lda, fillMode, n, output, alpha, x);
  };

  if (fillMode != FillMode::FULL) {
    call(fillMode);
  } else {
    call(FillMode::LOWER);
    call(FillMode::UPPER);
    this->_scaleDiagonal(output, cu_convert<T>(0.5));
  }
}

template <typename T>
void Cublas<T>::_gemm(T *output, T *alpha, Operation transa, T *a,
                      Operation transb, T *b, T *beta) {
  const auto m = this->getRows(a);
  const auto m1 = this->getRows(output);

  if (m != m1) {
    std::stringstream sstream;
    sstream
        << "Rows of 'a' matrix is not equal to rows of 'output'. Rows('a'): "
        << m << " Rows('output'): " << m1;
    throw std::runtime_error(sstream.str());
  }

  const auto n = this->getColumns(b);
  const auto n1 = this->getColumns(output);

  if (n != n1) {
    std::stringstream sstream;
    sstream << "Columns of 'b' matrix is not equal to columns of 'output'. "
               "Columns('b'): "
            << n << " Columns('output'): " << n1;
    throw std::runtime_error(sstream.str());
  }

  const auto k = this->getColumns(a);
  const auto k1 = this->getRows(b);

  if (k != k1) {
    std::stringstream sstream;
    sstream << "Columns of 'a' matrix is not equal to rows of 'b' matrix. "
               "Columns('a'): "
            << k << " Rows('b'): " << k1;
    throw std::runtime_error(sstream.str());
  }

  m_cublasKernels.gemm(output, m, n, k, alpha, transa, a, transb, b, beta);
}

template <typename T>
void Cublas<T>::_gemm(T *output, T alpha, Operation transa, T *a,
                      Operation transb, T *b, T beta) {
  _gemm(output, &alpha, transa, a, transb, b, &beta);
}

template <typename T>
void Cublas<T>::_symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha,
                      T *a, T *b, T *beta) {

  int m = this->getRows(output);
  int n = this->getColumns(output);

  m_cublasKernels.symm(output, sideMode, fillMode, m, n, alpha, a, b, beta);
}

template <typename T> void Cublas<T>::_matrixMul(T *output, T *a, T *b) {
  _gemm(output, cu_convert<T>(1), Operation::OP_N, a, Operation::OP_N, b,
        cu_convert<T>(0));
}

template <typename T> void Cublas<T>::_geqrf(T *a, T *tau) {
  auto as = std::vector<T *>{a};
  auto taus = std::vector<T *>{tau};
  _geqrf(as, taus);
}

template <typename T>
void Cublas<T>::_geqrf(Cublas<T>::Vec &a, Cublas<T>::Vec &tau) {
  if (a.size() != tau.size()) {
    std::stringstream sstream;
    sstream << "Invalid count of element between 'a' and 'tau'." << a.size()
            << " != " << tau.size();
    throw std::runtime_error(sstream.str());
  }

  if (a.size() == 0) {
    throw std::runtime_error("Array of martices 'a' is empty");
  }

  if (tau.size() == 0) {
    throw std::runtime_error("Array of martices 'tau' is empty");
  }

  struct Spec {
    cublasHandle_t handle;
    int m, n, lda;
    int *info;
    int batchSize;

    cublasStatus_t geqrfBatched(float **a, float **tau) {
      return cublasSgeqrfBatched(handle, m, n, a, lda, tau, info, batchSize);
    }

    cublasStatus_t geqrfBatched(double **a, double **tau) {
      return cublasDgeqrfBatched(handle, m, n, a, lda, tau, info, batchSize);
    }
  };

  auto checkDim = [](int _rows, int _columns, int rows, int columns) {
    if (_rows != -1 && (rows != _rows || columns != _columns)) {
      std::stringstream sstream;
      sstream << "Invalid dim. ";
      sstream << "Expected: ";
      sstream << "(" << _rows << ", " << _columns << ") ";
      sstream << "Actual: ";
      sstream << "(" << rows << ", " << columns << ")";
      throw std::runtime_error(sstream.str());
    }
  };

  auto getDim = [this, &checkDim](const auto &input) {
    int m = -1;
    int n = -1;

    int _rows = -1;
    int _columns = -1;

    for (const auto *elem : input) {

      auto rows = this->getRows(elem);
      auto columns = this->getColumns(elem);

      checkDim(_rows, _columns, rows, columns);

      _rows = rows;
      _columns = columns;
    }
    m = _rows;
    n = _columns;
    return std::make_pair(m, n);
  };

  auto pair = getDim(a);
  getDim(tau);

  int m = pair.first;
  int n = pair.second;

  const int lda = m;
  int batchSize = a.size();

  int info = 0;

  static_assert(sizeof(float *) == sizeof(void *),
                "Mismatch beetwen size of float* and void*");

  auto **a_t = static_cast<T **>(a.data());
  auto **tau_t = static_cast<T **>(tau.data());

  auto **d_a = cublas_allocCudaArrayCopy<T *>(a_t, a.size());
  auto **d_tau = cublas_allocCudaArrayCopy<T *>(tau_t, tau.size());

  m_cublasKernels.geqrfBatched(m, n, d_a, lda, d_tau, &info, batchSize);

  cublas_deallocCudaArray(d_a);
  cublas_deallocCudaArray(d_tau);

  if (info < 0) {
    std::stringstream sstream;
    sstream << "The parameters passed at " << -info << " is invalid";
    throw std::runtime_error(sstream.str());
  }
}

template <typename T> void Cublas<T>::_qrDecomposition(T *q, T *r, T *a) {
  _qrDecomposition(Cublas<T>::Vec({q}), Cublas<T>::Vec({r}),
                   Cublas<T>::Vec({a}));
}

template <typename T>
void Cublas<T>::_qrDecomposition(const Cublas<T>::Vec &q,
                                 const Cublas<T>::Vec &r,
                                 const Cublas<T>::Vec &a) {
  auto m = this->getRows(a[0]);
  auto n = this->getColumns(a[0]);
  auto k = std::min(m, n);

  if (m != n) {
    std::stringstream sstream;
    sstream << "Matrix is not square! It is " << m << "x" << n;
    throw std::runtime_error(sstream.str());
  }

  auto m2 = m * n;
  std::vector<T> h_zeros(m2, cu_convert<T>(0));

  T *I = this->createIdentityMatrix(m, m);
  T *H = this->createMatrix(m, m);
  T *Aux1 = this->createMatrix(m, m);
  T *Aux2 = this->createMatrix(m, m);
  T *v = this->create(m);

  Cublas<T>::Vec aux3;
  for (auto *mema : a) {
    T *Aux3 = this->createMatrix(m, m);
    this->matrixMul(Aux3, I, mema);
    aux3.push_back(Aux3);
  }

  auto dim = std::max(static_cast<int>(1), std::min(m, n));

  Cublas<T>::Vec taus;
  taus.reserve(a.size());
  for (size_t idx = 0; idx < a.size(); ++idx) {
    taus.push_back(this->create(dim));
  }

  this->geqrf(aux3, taus);
  std::vector<T> h_taus;
  h_taus.resize(dim);

  for (size_t j = 0; j < aux3.size(); ++j) {
    this->copyKernelToHost(h_taus.data(), taus[j]);
    for (int i = 0; i < k; ++i) {

      auto *d_array = aux3[j];
      auto rows = this->getRows(d_array);

      int idx1 = (i + 1) + rows * i;
      int idx2 = (m) + rows * i;
      int offset = idx2 - idx1;
      T *Aux4 = cublas_getOffset(d_array, idx1);
      auto Aux4_size = offset;

      std::vector<T> h_v(m, cu_convert<T>(0));
      h_v[i] = cu_convert<T>(1);
      auto status = cudaMemcpy(h_v.data() + i + 1, Aux4, SizeOf<T>(Aux4_size),
                               cudaMemcpyDeviceToHost);
      handleStatus(status);

      this->copyHostToKernel(v, h_v.data());

      T tau = h_taus[i];
      this->copyHostToKernel(Aux1, h_zeros.data());
      this->syr(FillMode::FULL, Aux1, &tau, v);

      this->subtract(Aux2, I, Aux1);
      if (i > 0) {
        this->matrixMul(Aux1, I, H);
        this->matrixMul(H, Aux1, Aux2);
      } else {
        this->matrixMul(H, I, Aux2);
      }
    }

    this->matrixMul(q[j], I, H);
#if 0
    auto status = cudaMemcpy(q[j]->ptr, H->ptr, SizeOf<T>(H->count),
                             cudaMemcpyDeviceToDevice);
    handleStatus(status);
#endif

    T alpha = cu_convert<T>(1);
    T beta = cu_convert<T>(0);
    this->gemm(r[j], &alpha, Operation::OP_T, q[j], Operation::OP_N, a[j],
               &beta);
  }

  for (const auto *tau : taus) {
    this->destroy(tau);
  }

  this->destroy(v);
  this->destroy(Aux2);
  this->destroy(Aux1);
  this->destroy(H);
  this->destroy(I);
}

template <typename T> void Cublas<T>::_shiftQRIteration(T *H, T *Q) {
  bool status = false;

  const auto dims = this->getDims(H);

  auto rows = dims.first;
  auto columns = dims.second;

  T *aux_Q = this->createIdentityMatrix(rows, columns);
  T *aux_Q1 = this->createMatrix(rows, columns);

  T *aux_R = this->createMatrix(rows, columns);

  T *ioQ = Q;

  status = this->isUpperTriangular(H);

  while (status == false) {
    this->qrDecomposition(ioQ, aux_R, H);

    this->matrixMul(H, aux_R, ioQ);
    this->matrixMul(aux_Q1, ioQ, aux_Q);
    swap(&aux_Q1, &aux_Q);
    status = this->isUpperTriangular(H);
  }
}

template <typename T> bool Cublas<T>::_isUpperTriangular(T *m) {

  const auto &dim = checkForTriangular(m);

  auto lda = dim.first;

  CudaAlloc alloc;
  CudaKernels kernels(0, &alloc);

  return kernels.isUpperTriangular(dim.first, dim.second, m, lda,
                                   cu_convert<T>(0.));
}

template <typename T> bool Cublas<T>::_isLowerTriangular(T *m) {

  const auto &dim = checkForTriangular(m);

  auto lda = dim.first;

  CudaAlloc alloc;
  CudaKernels kernels(0, &alloc);

  return kernels.isLowerTriangular(dim.first, dim.second, m, lda,
                                   cu_convert<T>(0.));
}

template <typename T>
void Cublas<T>::_geam(T *output, T *alpha, Operation transa, T *a, T *beta,
                      Operation transb, T *b) {
  auto m = this->getRows(a);
  auto n = this->getColumns(b);

  auto lda = m;
  auto ldb = this->getRows(b);
  auto ldc = this->getRows(output);

  m_cublasKernels.geam(output, ldc, m, n, alpha, transa, a, lda, beta, transb,
                       b, ldb);
}

template <typename T> void Cublas<T>::_add(T *output, T *a, T *b) {
  this->geam(output, cu_convert<T>(1), Operation::OP_N, a, cu_convert<T>(1),
             Operation::OP_N, b);
}

template <typename T> void Cublas<T>::_subtract(T *output, T *a, T *b) {
  this->geam(output, cu_convert<T>(1), Operation::OP_N, a, cu_convert<T>(-1),
             Operation::OP_N, b);
}

template <typename T> bool Cublas<T>::_isComplex() const {
  struct True {
    bool get() const { return true; }
  };
  struct False {
    bool get() const { return false; }
  };
  using isCuComplex =
      std::conditional<std::is_same<T, cuComplex>::value, True, False>;
  using isCuDoubleComplex =
      std::conditional<std::is_same<T, cuDoubleComplex>::value, True,
                       typename isCuComplex::type>;
  using isFloat = std::conditional<std::is_same<T, float>::value, False,
                                   typename isCuDoubleComplex::type>;
  typename std::conditional<std::is_same<T, double>::value, False,
                            typename isFloat::type>::type obj;
  return obj.get();
}

template <typename T> void Cublas<T>::_scaleDiagonal(T *matrix, T *factor) {
  auto rows = this->getRows(matrix);
  auto columns = this->getColumns(matrix);
  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  CudaAlloc cudaAlloc;
  CudaKernels kernels(0, &cudaAlloc);

  kernels.scaleDiagonal(rows, matrix, rows, factor);
}

template <typename T> void Cublas<T>::_scaleDiagonal(T *matrix, T factor) {
  _scaleDiagonal(matrix, &factor);
}

template <typename T>
void Cublas<T>::_tpttr(FillMode uplo, int n, T *AP, T *A, int lda) {
  struct Spec {
    cublasHandle_t handle;
    FillMode uplo;
    int n;
    int lda;

    cublasStatus_t tpttr(float *AP, float *A) {
      return cublasStpttr(handle, convert(uplo), n, AP, A, lda);
    };
    cublasStatus_t tpttr(double *AP, double *A) {
      return cublasDtpttr(handle, convert(uplo), n, AP, A, lda);
    };

    cublasStatus_t tpttr(cuComplex *AP, cuComplex *A) {
      return cublasCtpttr(handle, convert(uplo), n, AP, A, lda);
    };

    cublasStatus_t tpttr(cuDoubleComplex *AP, cuDoubleComplex *A) {
      return cublasZtpttr(handle, convert(uplo), n, AP, A, lda);
    };
  };

  auto rows = this->getRows(A);
  auto columns = this->getColumns(A);
  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  auto handle = m_cublasKernels;
  Spec spec = {handle, uplo, n, lda};
  auto status = spec.tpttr(AP, A);
  handleStatus(status);
}

template <typename T>
void Cublas<T>::_trttp(FillMode uplo, int n, T *A, int lda, T *AP) {

  struct Spec {
    cublasHandle_t handle;
    FillMode uplo;
    int n;
    int lda;

    cublasStatus_t trttp(float *AP, float *A) {
      return cublasStrttp(handle, convert(uplo), n, A, lda, AP);
    };
    cublasStatus_t trttp(double *AP, double *A) {
      return cublasDtrttp(handle, convert(uplo), n, A, lda, AP);
    };

    cublasStatus_t trttp(cuComplex *AP, cuComplex *A) {
      return cublasCtrttp(handle, convert(uplo), n, A, lda, AP);
    };

    cublasStatus_t trttp(cuDoubleComplex *AP, cuDoubleComplex *A) {
      return cublasZtrttp(handle, convert(uplo), n, A, lda, AP);
    };
  };

  auto rows = this->getRows(A);
  auto columns = this->getColumns(A);
  if (rows != columns) {
    std::stringstream sstream;
    sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
            << columns;
    throw std::runtime_error(sstream.str());
  }

  auto handle = m_cublasKernels;
  Spec spec = {handle, uplo, n, lda};
  auto status = spec.trttp(AP, A);
  handleStatus(status);
}

template <typename T> bool Cublas<T>::_isUnit(T *mem, T *delta) {
  CudaAlloc alloc;
  CudaKernels kernels(0, &alloc);

  auto m = this->getRows(mem);
  auto n = this->getColumns(mem);
  auto lda = m;

  return kernels.isUnit(m, n, mem, lda, delta);
}

inline std::ostream &operator<<(std::ostream &os, const cuComplex &ccomplex) {
  os << "(" << ccomplex.x << ", " << ccomplex.y << ")";
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const cuDoubleComplex &ccomplex) {
  os << "(" << ccomplex.x << ", " << ccomplex.y << ")";
  return os;
}

template <typename T> std::string Cublas<T>::_toStr(T *m) {
  auto count = _getCount(m);

  std::vector<T> vec;
  vec.resize(count);

  this->copyKernelToHost(vec.data(), m);
  std::stringstream sstream;

  sstream << "[";
  for (int idx = 0; idx < count; ++idx) {
    sstream << vec[idx];
    if (idx < count - 1) {
      sstream << ", ";
    }
  }

  sstream << "]";
  return sstream.str();
}

template <typename T> T Cublas<T>::*alloc(int count) {
  if (count <= 0) {
    std::stringstream sstream;
    sstream << "Cannot created mem with count: " << count;
    throw std::runtime_error(sstream.str());
  }

  T *ptr = nullptr;
  try {
    auto error = cudaMalloc(&ptr, SizeOf<T>(count));
    handleStatus(error);
    auto error1 = cudaMemset(ptr, 0, SizeOf<T>(count));
    handleStatus(error1);
  } catch (const std::exception &ex) {
    cudaFree(ptr);
    throw std::runtime_error(ex.what());
  }
  return ptr;
}

template <typename T> void Cublas<T>::dealloc(T *mem) {
  auto status = cudaFree(mem);
  handleStatus(status);
}
} // namespace mtrx

#endif
