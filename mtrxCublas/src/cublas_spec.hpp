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

#ifndef MTRX_CUBLAS_SPEC_HPP
#define MTRX_CUBLAS_SPEC_HPP

#include "cuda_alloc.hpp"
#include "kernels.hpp"
#include "mtrxCore/size_of.hpp"
#include <cstdint>
#include <cublas_v2.h>
#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mtrx {
template <typename T> using Vec = std::vector<T>;

class CublasSpec final {
public:
  CublasSpec();
  ~CublasSpec();

public:
  std::vector<int> getDevices() const;
  void setDevice(int device);

  template <typename T> void dealloc(T *mem) {
    auto status = cudaFree(mem);
    handleStatus(status);
  }

  template <typename T> uintt _getSizeInBytes(const T *mem) const {
    return SizeOf(mem->valueType, mem->count);
  }

  template <typename T> void _copyHostToKernel(T *k_ptr, T *h_ptr) {
    auto count = getCount(k_ptr);
    auto status = cublasSetVector(count, SizeOf<T>(), h_ptr, 1, k_ptr, 1);
    handleStatus(status);
  }

  template <typename T> void _copyKernelToHost(T *h_ptr, T *k_ptr) {
    auto count = getCount(k_ptr);
    auto status = cublasGetVector(count, SizeOf<T>(), k_ptr, 1, h_ptr, 1);
    handleStatus(status);
  }

  uintt _amax(const float *mem);
  uintt _amax(const double *mem);
  uintt _amax(const cuComplex *mem);
  uintt _amax(const cuDoubleComplex *mem);

  uintt _amin(const float *mem);
  uintt _amin(const double *mem);
  uintt _amin(const cuComplex *mem);
  uintt _amin(const cuDoubleComplex *mem);

  void _rot(float *x, float *y, float *c, float *s);
  void _rot(double *x, double *y, double *c, double *s);

  void _syr(FillMode fillMode, float *output, float *alpha, float *x);
  void _syr(FillMode fillMode, double *output, double *alpha, double *x);

  void _gemm(float *output, float *alpha, Operation transa, float *a,
             Operation transb, float *b, float *beta);
  void _gemm(double *output, double *alpha, Operation transa, double *a,
             Operation transb, double *b, double *beta);

  void _symm(SideMode sideMode, FillMode fillMode, Mem *output, void *alpha,
             ValueType alphaType, Mem *a, Mem *b, void *beta,
             ValueType betaType);
  void _symm(SideMode sideMode, FillMode fillMode, Mem *output, Mem *alpha,
             Mem *a, Mem *b, Mem *beta);

  template <typename T> void _matrixMul(T *output, T *a, T *b) {
    auto alpha = static_cast<T>(1.);
    auto beta = static_cast<T>(0.);

    _gemm(output, alpha, Operation::OP_N, a, Operation::OP_N, b, beta);
  }

  template <typename T> void _geqrf(T *a, T *tau) {
    auto as = std::vector<T *>{a};
    auto taus = std::vector<T *>{tau};
    _geqrf(as, taus);
  }

  void _geqrf(Vec<float> &a, Vec<float> &tau);
  void _geqrf(Vec<double> &a, Vec<double> &tau);

  template <typename T> void _qrDecomposition(T *q, T *r, T *a) {
    Vec<T> qs{q};
    Vec<T> rs{r};
    Vec<T> as{a};

    _qrDecomposition(qs, rs, as);
  }

  void _qrDecomposition(Vec<float> &q, Vec<float> &r, Vec<float> &a);
  void _qrDecomposition(Vec<double> &q, Vec<double> &r, Vec<double> &a);

  void _shiftQRIteration(float *H, float *Q);
  void _shiftQRIteration(double *H, double *Q);

  template <typename T>
  void _isUpperTriangular(bool &result, CublasSpec &cublas, T *matrix,
                          T delta = static_cast<T>(0));

  template <typename T>
  void _isLowerTriangular(bool &result, CublasSpec &cublas, T *matrix,
                          T delta = static_cast<T>(0));

  template <typename T>
  bool _isUpperTriangular(T *matrix, T delta = static_cast<T>(0));

  template <typename T>
  bool _isLowerTriangular(T *matrix, T delta = static_cast<T>(0));

  bool _isUpperTriangular(Mem *m);
  bool _isLowerTriangular(Mem *m);

  void _geam(Mem *output, Mem *alpha, Operation transa, Mem *a, Mem *beta,
             Operation transb, Mem *b);
  void _geam(Mem *output, void *alpha, ValueType alphaType, Operation transa,
             Mem *a, void *beta, ValueType betaType, Operation transb, Mem *b);

  void _add(Mem *output, Mem *a, Mem *b);
  void _subtract(Mem *output, Mem *a, Mem *b);

  void _scaleDiagonal(Mem *matrix, Mem *factor);
  void _scaleDiagonal(Mem *matrix, void *factor, ValueType factorType);

  void _tpttr(FillMode uplo, int n, Mem *AP, Mem *A, int lda);
  void _trttp(FillMode uplo, int n, Mem *A, int lda, Mem *AP);

  bool _isUnit(Mem *mem, void *delta, ValueType deltaType);
  bool _isUnit(Mem *mem, Mem *delta);

  std::string _toStr(Mem *mem);

private:
  cublasHandle_t m_handle;
  void swap(Mem **a, Mem **b);
};

template <typename T> bool CublasSpec::_isUpperTriangular(T *matrix, T delta) {
  bool result = false;
  cublas_isUpperTriangular(result, *this, matrix, delta);
  return result;
}

template <typename T> bool CublasSpec::_isLowerTriangular(T *matrix, T delta) {
  bool result = false;
  cublas_isLowerTriangular(result, *this, matrix, delta);
  return result;
}

} // namespace mtrx

#endif
