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
#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>

namespace mtrx {
class Cublas final {
public:
  Cublas();
  ~Cublas();

public:
  std::vector<int> getDevices() const;
  void setDevice(int device);

  float *allocS(size_t size);
  double *allocD(size_t size);
  cuComplex *allocC(size_t size);
  cuDoubleComplex *allocZ(size_t size);

  void dealloc(float *mem);
  void dealloc(double *mem);
  void dealloc(cuComplex *mem);
  void dealloc(cuDoubleComplex *mem);

  Mem *_createIdentityMatrix(size_t rows, size_t columns, ValueType valueType);

  uintt _getCount(const Mem *mem) const;
  uintt _getSizeInBytes(const Mem *mem) const;

  void _copyHostToKernel(Mem *mem, void *array);
  void _copyHostToKernel(Mem *mem, const void *array);
  void _copyKernelToHost(void *array, Mem *mem);

  uintt _amax(const Mem *mem);
  uintt _amin(const Mem *mem);

  void _rot(Mem *x, Mem *y, void *c, ValueType cType, void *s, ValueType sType);
  void _rot(Mem *x, Mem *y, Mem *c, Mem *s);

  void _syr(FillMode fillMode, Mem *output, void *alpha, ValueType alphaType,
            Mem *x);
  void _syr(FillMode fillMode, Mem *output, Mem *alpha, Mem *x);

  void _gemm(Mem *output, void *alpha, ValueType alphaType, Operation transa,
             Mem *a, Operation transb, Mem *b, void *beta, ValueType betaType);
  void _gemm(Mem *output, Mem *alpha, Operation transa, Mem *a,
             Operation transb, Mem *b, Mem *beta);

  void _symm(SideMode sideMode, FillMode fillMode, Mem *output, void *alpha,
             ValueType alphaType, Mem *a, Mem *b, void *beta,
             ValueType betaType);
  void _symm(SideMode sideMode, FillMode fillMode, Mem *output, Mem *alpha,
             Mem *a, Mem *b, Mem *beta);

  void _matrixMul(Mem *output, Mem *a, Mem *b);

  void _geqrf(Mem *a, Mem *tau);
  void _geqrf(Mems &a, Mems &tau);

  void _qrDecomposition(Mem *q, Mem *r, Mem *a);
  void _qrDecomposition(Mems &q, Mems &r, Mems &a);

  void _shiftQRIteration(Mem *H, Mem *Q);

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
} // namespace mtrx

#endif
