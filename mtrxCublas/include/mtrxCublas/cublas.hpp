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

#include <cublas_v2.h>
#include <sstream>

#include <mtrxCore/blas.hpp>
#include <mtrxCore/size_of.hpp>
#include <mtrxCore/types.hpp>

#include <mtrxCublas/impl/cuda_alloc.hpp>
#include <mtrxCublas/impl/kernels.hpp>

namespace mtrx {

template <typename T> class Cublas : public Blas<T> {
public:
  using Vec = typename Blas<T>::Vec;

  Cublas() {}
  ~Cublas() override {}

protected:
  std::vector<int> _getDevices() const override;
  void _setDevice(int device) override;

  T *_createMem(size_t size) override;
  void _destroy(const T *mem) override;

  T *_createIdentityMatrix(size_t rows, size_t columns) override;

  uintt _getCount(const T *mem) const override;
  uintt _getSizeInBytes(const T *mem) const override;

  void _copyHostToKernel(T *mem, T *array) override;
  void _copyHostToKernel(T *mem, const T *array) override;
  void _copyKernelToHost(T *array, T *mem) override;

  uintt _amax(const T *mem) override;
  uintt _amin(const T *mem) override;

  void _rot(T *x, T *y, T *c, T *s) override;

  void _syr(FillMode fillMode, T *output, T *alpha, T *x) override;

  void _gemm(T *output, T *alpha, T *a, Operation transb, T *b,
             T *beta) override;

  void _symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha, T *a,
             T *b, T *beta) override;

  void _matrixMul(T *output, T *a, T *b) override;

  void _geqrf(T *a, T *tau) override;
  void _geqrf(Vec &a, Vec &tau) override;

  void _qrDecomposition(T *q, T *r, T *a) override;
  void _qrDecomposition(Vec &q, Vec &r, Vec &a) override;

  void _shiftQRIteration(T *H, T *Q) override;

  bool _isUpperTriangular(T *m) override;
  bool _isLowerTriangular(T *m) override;

  void _geam(T *output, T *alpha, Operation transa, T *a, T *beta,
             Operation transb, T *b) override;
  void _geam(T *output, T *alpha, T *a, T *beta, T *b) override;

  void _add(T *output, T *a, T *b) override;
  void _subtract(T *output, T *a, T *b) override;

  void _scaleDiagonal(T *matrix, T *factor) override;

  void _tpttr(FillMode uplo, int n, T *AP, T *A, int lda) override;
  void _trttp(FillMode uplo, int n, T *A, int lda, T *AP) override;

  bool _isUnit(T *mem, T *delta) override;

  std::string _toStr(T *mem) override;

private:
  T *alloc(size_t count);
  void dealloc(T *mem);

  template <typename Vec> void setVec(Vec &&vec, size_t rows, size_t columns) {
    vec.reserve(rows * columns);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < columns; ++c) {
        if (r == c) {
          vec.push_back(static_cast<T>(1));
        } else {
          vec.push_back(static_cast<T>(0));
        }
      }
    }
  }

  std::pair<size_t, size_t> checkForTriangular(T *matrix) {
    auto rows = getRows(matrix);
    auto columns = getColumns(matrix);

    if (rows != columns) {
      std::stringstream sstream;
      sstream << __func__ << ": Matrix is not square matrix " << rows << " x "
              << columns;
      throw std::runtime_error(sstream.str());
    }
    return std::make_pair(rows, columns);
  }

  std::unordered_map<uintptr_t, uint> m_counts;
};
} // namespace mtrx

namespace mtrx {

template <typename T> std::vector<int> Cublas<T>::_getDevices() const {}

template <typename T> void Cublas<T>::_setDevice(int device) {}

template <typename T> T *_createMem(size_t size) {}

template <typename T> void Cublas<T>::_destroy(const T *mem) {}

template <typename T> T *_createIdentityMatrix(size_t rows, size_t columns) {
  Mem *matrix = Blas<T>::createMatrix(rows, columns);
  std::vector<T> vec;
  setVec(vec, rows, columns);
  copyHostToKernel(matrix, vec.data());
  return matrix;
}

template <typename T> uintt Cublas<T>::_getCount(const T *mem) const {
  uintptr_t handler = reinterpret_cast<uintptr_t>(mem);
  auto it = m_counts.find(handler);
  if (it == m_counts.end()) {
    std::stringstream sstream;
    sstream << "FATAL: Not found count for " << mem;
    throw std::runtime_error(sstream.str());
  }
  return it->second;
}

template <typename T> uintt Cublas<T>::_getSizeInBytes(const T *mem) const {}

template <typename T> void Cublas<T>::_copyHostToKernel(T *mem, T *array) {}

template <typename T>
void Cublas<T>::_copyHostToKernel(T *mem, const T *array) {}

template <typename T> void Cublas<T>::_copyKernelToHost(T *array, T *mem) {}

template <typename T> uintt Cublas<T>::_amax(const T *mem) {}

template <typename T> uintt Cublas<T>::_amin(const T *mem) {}

template <typename T> void Cublas<T>::_rot(T *x, T *y, T *c, T *s) {}

template <typename T>
void Cublas<T>::_syr(FillMode fillMode, T *output, T *alpha, T *x) {}

template <typename T>
void Cublas<T>::_gemm(T *output, T *alpha, T *a, Operation transb, T *b,
                      T *beta) {}

template <typename T>
void Cublas<T>::_symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha,
                      T *a, T *b, T *beta) {}

template <typename T> void Cublas<T>::_matrixMul(T *output, T *a, T *b) {
  auto alpha = static_cast<T>(1.);
  auto beta = static_cast<T>(0.);

  _gemm(output, alpha, Operation::OP_N, a, Operation::OP_N, b, beta);
}

template <typename T> void Cublas<T>::_geqrf(T *a, T *tau) {
    auto as = std::vector<T *>{a};
    auto taus = std::vector<T *>{tau};
    _geqrf(as, taus);
}

template <typename T> void Cublas<T>::_geqrf(Bias<T>::Vec &a, Bias<T>::Vec &tau) {}

template <typename T> void Cublas<T>::_qrDecomposition(T *q, T *r, T *a) {}

template <typename T>
void Cublas<T>::_qrDecomposition(Bias<T>::Vec &q, Bias<T>::Vec &r, Bias<T>::Vec &a) {}

template <typename T> void Cublas<T>::_shiftQRIteration(T *H, T *Q) {}

template <typename T> bool Cublas<T>::_isUpperTriangular(T *m) {

  const auto &dim = checkForTriangular(matrix);

  auto lda = dim.first;

  CudaAlloc alloc;
  Kernels kernels(0, &alloc);

  return kernels.isUpperTriangular(dim.first, dim.second, m, lda, 0.);
}

template <typename T> bool Cublas<T>::_isLowerTriangular(T *m) {

  const auto &dim = checkForTriangular(matrix);

  auto lda = dim.first;

  CudaAlloc alloc;
  Kernels kernels(0, &alloc);

  return kernels.isLowerTriangular(dim.first, dim.second, matrix, lda, 0.);
}

template <typename T>
void Cublas<T>::_geam(T *output, T *alpha, Operation transa, T *a, T *beta,
                      Operation transb, T *b) {}
template <typename T>
void Cublas<T>::_geam(T *output, T *alpha, T *a, T *beta, T *b) {}

template <typename T> void Cublas<T>::_add(T *output, T *a, T *b) {}

template <typename T> void Cublas<T>::_subtract(T *output, T *a, T *b) {}

template <typename T> void Cublas<T>::_scaleDiagonal(T *matrix, T *factor) {}

template <typename T>
void Cublas<T>::_tpttr(FillMode uplo, int n, T *AP, T *A, int lda) {}

template <typename T>
void Cublas<T>::_trttp(FillMode uplo, int n, T *A, int lda, T *AP) {}

template <typename T> bool Cublas<T>::_isUnit(T *mem, T *delta) {}

template <typename T> std::string Cublas<T>::_toStr(T *mem) {}

template <typename T> T Cublas<T>::*alloc(size_t count) {
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
