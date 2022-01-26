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

#ifndef MTRX_CUBLAS_MATRIX_API_HPP
#define MTRX_CUBLAS_MATRIX_API_HPP

#include <cublas_v2.h>
#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>

namespace mtrx {
class Cublas : public Blas {
public:
  Cublas();
  virtual ~Cublas() override;

protected:
  std::vector<int> _getDevices() const override;
  void _setDevice(int device) override;

  Mem *_createMem(size_t size, ValueType valueType) override;
  void _destroy(const Mem *mem) override;

  Mem *_createIdentityMatrix(size_t rows, size_t columns,
                             ValueType valueType) override;

  uintt _getCount(const Mem *mem) const override;
  uintt _getSizeInBytes(const Mem *mem) const override;

  void _copyHostToKernel(Mem *mem, void *array) override;
  void _copyKernelToHost(void *array, Mem *mem) override;

  uintt _amax(const Mem *mem) override;
  uintt _amin(const Mem *mem) override;

  void _rot(Mem *x, Mem *y, void *c, ValueType cType, void *s,
            ValueType sType) override;
  void _rot(Mem *x, Mem *y, Mem *c, Mem *s) override;

  void _syr(FillMode fillMode, Mem *output, void *alpha, ValueType alphaType,
            Mem *x) override;
  void _syr(FillMode fillMode, Mem *output, Mem *alpha, Mem *x) override;

  void _gemm(Mem *output, void *alpha, ValueType alphaType, Operation transa,
             Mem *a, Operation transb, Mem *b, void *beta,
             ValueType betaType) override;
  void _gemm(Mem *output, Mem *alpha, Operation transa, Mem *a,
             Operation transb, Mem *b, Mem *beta) override;

  void _symm(SideMode sideMode, FillMode fillMode, Mem *output, void *alpha,
             ValueType alphaType, Mem *a, Mem *b, void *beta,
             ValueType betaType) override;
  void _symm(SideMode sideMode, FillMode fillMode, Mem *output, Mem *alpha,
             Mem *a, Mem *b, Mem *beta) override;

  void _matrixMul(Mem *output, Mem *a, Mem *b) override;

  void _geqrf(Mem *a, Mem *tau) override;
  void _geqrf(Mems &a, Mems &tau) override;

  void _qrDecomposition(Mem *q, Mem *r, Mem *a) override;
  void _qrDecomposition(Mems &q, Mems &r, Mems &a) override;

  void _geam(Mem *output, Mem *alpha, Operation transa, Mem *a, Mem *beta,
             Operation transb, Mem *b) override;
  void _geam(Mem *output, void *alpha, ValueType alphaType, Operation transa,
             Mem *a, void *beta, ValueType betaType, Operation transb,
             Mem *b) override;

  void _add(Mem *output, Mem *a, Mem *b) override;
  void _subtract(Mem *output, Mem *a, Mem *b) override;

  void _scaleTrace(Mem *matrix, Mem *factor) override;
  void _scaleTrace(Mem *matrix, void *factor, ValueType factorType) override;

  void _tpttr(FillMode uplo, int n, Mem *AP, Mem *A, int lda) override;
  void _trttp(FillMode uplo, int n, Mem *A, int lda, Mem *AP) override;

  std::string _toStr(Mem *mem) override;

private:
  cublasHandle_t m_handle;
};
} // namespace mtrx

#endif
