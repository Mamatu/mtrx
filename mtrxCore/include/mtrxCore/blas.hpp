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

#ifndef MTRX_CORE_BLAS_API_HPP
#define MTRX_CORE_BLAS_API_HPP

#include <map>
#include <vector>

#include "types.hpp"

namespace mtrx {

class Blas {
public:
  using Mems = std::vector<Mem *>;

  Blas() = default;
  virtual ~Blas() = default;

  std::vector<int> getDevices() const;
  void setDevice(int device);

  Mem *create(size_t count, ValueType valueType);
  Mem *createMatrix(size_t rows, size_t columns, ValueType valueType);
  Mem *createMatrix(size_t rows, size_t columns, Mem *mem);

  Mem *createIdentityMatrix(size_t rows, size_t columns, ValueType valueType);

  void destroy(const Mem *mem);

  bool isAllocator(const Mem *mem) const;

  size_t getCount(const Mem *mem) const;
  size_t getSizeInBytes(const Mem *mem) const;

  size_t getRows(const Mem *mem) const;
  size_t getColumns(const Mem *mem) const;
  ValueType getValueType(const Mem *mem) const;

  std::pair<size_t, size_t> getDims(const Mem *mem) const;

  void copyHostToKernel(Mem *mem, void *array);
  void copyHostToKernel(Mem *mem, const void *array);
  void copyKernelToHost(void *array, Mem *mem);

  uintt amax(const Mem *mem);
  uintt amin(const Mem *mem);

  void rot(Mem *x, Mem *y, void *c, ValueType cType, void *s, ValueType sType);
  void rot(Mem *x, Mem *y, Mem *c, Mem *s);
  void rot(Mem *x, Mem *y, float c, float s);
  void rot(Mem *x, Mem *y, double c, double s);

  void syr(FillMode fillMode, Mem *output, void *alpha, ValueType alphaType,
           Mem *x);
  void syr(FillMode fillMode, Mem *output, Mem *alpha, Mem *x);

  void gemm(Mem *output, void *alpha, ValueType alphaType, Mem *a, Mem *b,
            void *beta, ValueType betaType);
  void gemm(Mem *output, Mem *alpha, Mem *a, Mem *b, Mem *beta);

  void gemm(Mem *output, void *alpha, ValueType alphaType, Operation transa,
            Mem *a, Operation transb, Mem *b, void *beta, ValueType betaType);
  void gemm(Mem *output, Mem *alpha, Operation transa, Mem *a, Operation transb,
            Mem *b, Mem *beta);

  void symm(SideMode sideMode, FillMode fillMode, Mem *output, void *alpha,
            ValueType alphaType, Mem *a, Mem *b, void *beta,
            ValueType betaType);
  void symm(SideMode sideMode, FillMode fillMode, Mem *output, Mem *alpha,
            Mem *a, Mem *b, Mem *beta);

  void matrixMul(Mem *output, Mem *a, Mem *b);

  void geqrf(Mem *a, Mem *tau);
  void geqrf(Mems &a, Mems &tau);

  void qrDecomposition(Mem *q, Mem *r, Mem *a);
  void qrDecomposition(Mems &q, Mems &r, Mems &a);

  void shiftQRIteration(Mem *H, Mem *Q);

  bool isUpperTriangular(Mem *m);
  bool isLowerTriangular(Mem *m);

  /**
   * @brief matrix-matrix addition/transposition
   * output = alpha * oper(a) + beta * oper(b)
   */
  void geam(Mem *output, Mem *alpha, Operation transa, Mem *a, Mem *beta,
            Operation transb, Mem *b);
  void geam(Mem *output, void *alpha, ValueType alphaType, Operation transa,
            Mem *a, void *beta, ValueType betaType, Operation transb, Mem *b);

  void add(Mem *output, Mem *a, Mem *b);
  void subtract(Mem *output, Mem *a, Mem *b);

  void scaleDiagonal(Mem *matrix, Mem *factor);
  void scaleDiagonal(Mem *matrix, void *factor, ValueType factorType);

  void tpttr(FillMode uplo, int n, Mem *AP, Mem *A, int lda);
  void trttp(FillMode uplo, int n, Mem *A, int lda, Mem *AP);

  bool isUnit(Mem *mem, void *delta, ValueType deltaType);
  bool isUnit(Mem *mem, Mem *delta);

  std::string toStr(Mem *mem);

protected:
  virtual std::vector<int> _getDevices() const = 0;
  virtual void _setDevice(int device) = 0;

  virtual Mem *_createMem(size_t count, ValueType valueType) = 0;
  virtual void _destroy(const Mem *mem) = 0;

  virtual Mem *_createIdentityMatrix(size_t rows, size_t columns,
                                     ValueType valueType) = 0;

  virtual size_t _getCount(const Mem *mem) const = 0;
  virtual size_t _getSizeInBytes(const Mem *mem) const = 0;

  virtual void _copyHostToKernel(Mem *mem, void *array) = 0;
  virtual void _copyHostToKernel(Mem *mem, const void *array) = 0;
  virtual void _copyKernelToHost(void *array, Mem *mem) = 0;

  virtual uintt _amax(const Mem *mem) = 0;
  virtual uintt _amin(const Mem *mem) = 0;

  virtual void _rot(Mem *x, Mem *y, void *c, ValueType cType, void *s,
                    ValueType sType) = 0;
  virtual void _rot(Mem *x, Mem *y, Mem *c, Mem *s) = 0;

  virtual void _syr(FillMode fillMode, Mem *output, void *alpha,
                    ValueType alphaType, Mem *x) = 0;
  virtual void _syr(FillMode fillMode, Mem *output, Mem *alpha, Mem *x) = 0;

  virtual void _gemm(Mem *output, void *alpha, ValueType alphaType,
                     Operation transa, Mem *a, Operation transb, Mem *b,
                     void *beta, ValueType betaType) = 0;
  virtual void _gemm(Mem *output, Mem *alpha, Operation transa, Mem *a,
                     Operation transb, Mem *b, Mem *beta) = 0;

  virtual void _matrixMul(Mem *output, Mem *a, Mem *b) = 0;

  virtual void _geqrf(Mem *a, Mem *tau) = 0;
  virtual void _geqrf(Mems &a, Mems &tau) = 0;

  virtual void _qrDecomposition(Mem *q, Mem *r, Mem *a) = 0;
  virtual void _qrDecomposition(Mems &q, Mems &r, Mems &a) = 0;

  virtual void _shiftQRIteration(Mem *H, Mem *Q) = 0;

  virtual bool _isUpperTriangular(Mem *m) = 0;
  virtual bool _isLowerTriangular(Mem *m) = 0;

  virtual void _geam(Mem *output, Mem *alpha, Operation transa, Mem *a,
                     Mem *beta, Operation transb, Mem *b) = 0;
  virtual void _geam(Mem *output, void *alpha, ValueType alphaType,
                     Operation transa, Mem *a, void *beta, ValueType betaType,
                     Operation transb, Mem *b) = 0;

  virtual void _symm(SideMode sideMode, FillMode fillMode, Mem *output,
                     void *alpha, ValueType alphaType, Mem *a, Mem *b,
                     void *beta, ValueType betaType) = 0;
  virtual void _symm(SideMode sideMode, FillMode fillMode, Mem *output,
                     Mem *alpha, Mem *a, Mem *b, Mem *beta) = 0;

  virtual void _add(Mem *output, Mem *a, Mem *b) = 0;
  virtual void _subtract(Mem *output, Mem *a, Mem *b) = 0;

  virtual void _scaleDiagonal(Mem *matrix, Mem *factor) = 0;
  virtual void _scaleDiagonal(Mem *matrix, void *factor,
                              ValueType factorType) = 0;

  virtual void _tpttr(FillMode uplo, int n, Mem *AP, Mem *A, int lda) = 0;
  virtual void _trttp(FillMode uplo, int n, Mem *A, int lda, Mem *AP) = 0;

  virtual bool _isUnit(Mem *mem, void *delta, ValueType deltaType) = 0;
  virtual bool _isUnit(Mem *mem, Mem *delta) = 0;

  virtual std::string _toStr(Mem *mem) = 0;

private:
  std::vector<Mem *> m_mems;
  std::map<const Mem *, std::tuple<size_t, size_t, ValueType>> m_matrices;

  void checkMem(const Mem *mem) const;
  void checkMems(const Mems &mems) const;
};

template <typename Toutput, typename Talpha, typename Ta, typename Tb,
          typename Tbeta>
void gemm(Blas &blas, Toutput *output, Talpha *alpha, Ta *a, Tb *b,
          Tbeta *beta) {
  blas.gemm(output, alpha, a, b, beta);
}

} // namespace mtrx

#endif
