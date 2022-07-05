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

#ifndef MTRX_CORE_BLAS_GENERIC_HPP
#define MTRX_CORE_BLAS_GENERIC_HPP

#include <map>
#include <vector>

#include "types.hpp"

namespace mtrx {

template <typename T> class Blas {
public:
  using Vec = std::vector<T *>;

  Blas() = default;
  virtual ~Blas() = default;

  std::vector<int> getDevices() const;
  void setDevice(int device);

  T *create(size_t count);
  T *createMatrix(size_t rows, size_t columns);
  T *createMatrix(size_t rows, size_t columns, T *mem);

  T *createIdentityMatrix(size_t rows, size_t columns);

  void destroy(const T *mem);

  bool isAllocator(const T *mem) const;

  size_t getCount(const T *mem) const;
  size_t getSizeInBytes(const T *mem) const;

  size_t getRows(const T *mem) const;
  size_t getColumns(const T *mem) const;
  ValueType getValueType(const T *mem) const;

  std::pair<size_t, size_t> getDims(const T *mem) const;

  void copyHostToKernel(T *mem, T *array);
  void copyHostToKernel(T *mem, const T *array);
  void copyKernelToHost(T *array, T *mem);

  uintt amax(const T *mem);
  uintt amin(const T *mem);

  void rot(T *x, T *y, T *c);
  void rot(T *x, T *y, T *c, T *s);
  void rot(T *x, T *y, float c, float s);
  void rot(T *x, T *y, double c, double s);

  void syr(FillMode fillMode, T *output, T *alpha, T *x);

  void gemm(T *output, T *alpha, T *beta);
  void gemm(T *output, T *alpha, T *a, T *b, T *beta);

  void gemm(T *output, T *alpha, T *a, Operation transb, T *b, T *beta);
  void gemm(T *output, T *alpha, Operation transa, T *a, Operation transb, T *b,
            T *beta);

  void symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha);
  void symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha, T *a,
            T *b, T *beta);

  void matrixMul(T *output, T *a, T *b);

  void geqrf(T *a, T *tau);
  void geqrf(typename Blas<T>::Vec &a, typename Blas<T>::Vec &tau);

  void qrDecomposition(T *q, T *r, T *a);
  void qrDecomposition(typename Blas<T>::Vec &q, typename Blas<T>::Vec &r,
                       typename Blas<T>::Vec &a);

  void shiftQRIteration(T *H, T *Q);

  bool isUpperTriangular(T *m);
  bool isLowerTriangular(T *m);

  /**
   * @brief matrix-matrix addition/transposition
   * output = alpha * oper(a) + beta * oper(b)
   */
  void geam(T *output, T *alpha, Operation transa, T *a, T *beta,
            Operation transb, T *b);

  void add(T *output, T *a, T *b);
  void subtract(T *output, T *a, T *b);

  void scaleDiagonal(T *matrix, T factor);

  void tpttr(FillMode uplo, int n, T *AP, T *A, int lda);
  void trttp(FillMode uplo, int n, T *A, int lda, T *AP);

  bool isUnit(T *mem, T delta);

  std::string toStr(T *mem);

protected:
  virtual std::vector<int> _getDevices() const = 0;
  virtual void _setDevice(int device) = 0;

  virtual T *_createMem(size_t count) = 0;
  virtual void _destroy(const T *mem) = 0;

  virtual T *_createIdentityMatrix(size_t rows, size_t columns) = 0;

  virtual size_t _getCount(const T *mem) const = 0;
  virtual size_t _getSizeInBytes(const T *mem) const = 0;

  virtual void _copyHostToKernel(T *mem, T *array) = 0;
  virtual void _copyHostToKernel(T *mem, const T *array) = 0;
  virtual void _copyKernelToHost(T *array, T *mem) = 0;

  virtual uintt _amax(const T *mem) = 0;
  virtual uintt _amin(const T *mem) = 0;

  virtual void _rot(T *x, T *y, T *c, T *s) = 0;

  virtual void _syr(FillMode fillMode, T *output, T *alpha, T *x) = 0;

  virtual void _gemm(T *output, T *alpha, Operation transa, T *a,
                     Operation transb, T *b, T *beta) = 0;

  virtual void _matrixMul(T *output, T *a, T *b) = 0;

  virtual void _geqrf(T *a, T *tau) = 0;
  virtual void _geqrf(typename Blas<T>::Vec &a, typename Blas<T>::Vec &tau) = 0;

  virtual void _qrDecomposition(T *q, T *r, T *a) = 0;
  virtual void _qrDecomposition(typename Blas<T>::Vec &q,
                                typename Blas<T>::Vec &r,
                                typename Blas<T>::Vec &a) = 0;

  virtual void _shiftQRIteration(T *H, T *Q) = 0;

  virtual bool _isUpperTriangular(T *m) = 0;
  virtual bool _isLowerTriangular(T *m) = 0;

  virtual void _geam(T *output, T *alpha, Operation transa, T *a, T *beta,
                     Operation transb, T *b) = 0;

  virtual void _symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha,
                     T *beta) = 0;
  virtual void _symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha,
                     T *a, T *b, T *beta) = 0;

  virtual void _add(T *output, T *a, T *b) = 0;
  virtual void _subtract(T *output, T *a, T *b) = 0;

  virtual void _scaleDiagonal(T *matrix, T *factor) = 0;

  virtual void _tpttr(FillMode uplo, int n, T *AP, T *A, int lda) = 0;
  virtual void _trttp(FillMode uplo, int n, T *A, int lda, T *AP) = 0;

  virtual bool _isUnit(T *mem, T *delta) = 0;

  virtual std::string _toStr(T *mem) = 0;

private:
  std::vector<T *> m_mems;
  std::map<const T *, std::tuple<size_t, size_t, ValueType>> m_matrices;

  void checkMem(const T *mem) const;
};

} // namespace mtrx

#endif
