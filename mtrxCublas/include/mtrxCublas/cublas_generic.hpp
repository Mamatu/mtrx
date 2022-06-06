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

#ifndef MTRX_CUBLAS_GENERIC_HPP
#define MTRX_CUBLAS_GENERIC_HPP

#include <cublas_v2.h>
#include <mtrxCore/blas_generic.hpp>
#include <mtrxCore/types.hpp>

namespace mtrx {

template <typename T> class CublasG : public BlasGeneric<T> {
public:
  CublasG() {}
  ~CublasG() override {}

protected:
  std::vector<int> _getDevices() const override;
  void _setDevice(int device) override;

  T *_createMem(size_t size) override;
  void _destroy(const T *mem) override;

  T *_createIdentityMatrix(size_t rows, size_t columns, ) override;

  uintt _getCount(const T *mem) const override;
  uintt _getSizeInBytes(const T *mem) const override;

  void _copyHostToKernel(T *mem, T *array) override;
  void _copyHostToKernel(T *mem, const T *array) override;
  void _copyKernelToHost(T *array, T *mem) override;

  uintt _amax(const T *mem) override;
  uintt _amin(const T *mem) override;

  void _rot(T *x, T *y, T *c, ) override;
  void _rot(T *x, T *y, T *c, T *s) override;

  void _syr(FillMode fillMode, T *output, T *alpha, T *x) override;
  void _syr(FillMode fillMode, T *output, T *alpha, T *x) override;

  void _gemm(T *output, T *alpha, T *a, Operation transb, T *b,
             T *beta) override;

  void _symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha, , T *a,
             T *b, T *beta) override;
  void _symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha, T *a,
             T *b, T *beta) override;

  void _matrixMul(T *output, T *a, T *b) override;

  void _geqrf(T *a, T *tau) override;
  void _geqrf(Mems &a, Mems &tau) override;

  void _qrDecomposition(T *q, T *r, T *a) override;
  void _qrDecomposition(Mems &q, Mems &r, Mems &a) override;

  void _shiftQRIteration(T *H, T *Q) override;

  bool _isUpperTriangular(T *m) override;
  bool _isLowerTriangular(T *m) override;

  void _geam(T *output, T *alpha, Operation transa, T *a, T *beta,
             Operation transb, T *b) override;
  void _geam(T *output, T *alpha, T *a, T *beta, T *b) override;

  void _add(T *output, T *a, T *b) override;
  void _subtract(T *output, T *a, T *b) override;

  void _scaleDiagonal(T *matrix, T *factor) override;
  void _scaleDiagonal(T *matrix, T *factor) override;

  void _tpttr(FillMode uplo, int n, T *AP, T *A, int lda) override;
  void _trttp(FillMode uplo, int n, T *A, int lda, T *AP) override;

  bool _isUnit(T *mem, T *delta) override;
  bool _isUnit(T *mem, T *delta) override;

  std::string _toStr(T *mem) override;
};
} // namespace mtrx

#endif
