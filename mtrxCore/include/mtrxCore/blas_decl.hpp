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

#ifndef MTRX_CORE_BLAS_DECL_HPP
#define MTRX_CORE_BLAS_DECL_HPP

#include <map>
#include <utility>
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

  T *create(int count);
  T *createMatrix(int rows, int columns);
  T *createMatrix(int rows, int columns, T *mem);

  T *createIdentityMatrix(int rows, int columns);

  void destroy(const T *mem);

  bool isAllocator(const T *mem) const;

  int getCount(const T *mem) const;
  int getSizeInBytes(const T *mem) const;

  int getRows(const T *mem) const;
  int getColumns(const T *mem) const;

  std::pair<int, int> getDims(const T *mem) const;

  void copyHostToKernel(T *mem, const T *array);
  void copyKernelToHost(T *array, const T *mem);

  void copyHostToKernel(T *mem, const T *array, int count);
  void copyKernelToHost(T *array, const T *mem, int count);

  uintt amax(const T *m);
  uintt amin(const T *m);

  void rot(T *x, T *y, T &&c, T &&s);

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

  void geam(T *output, T alpha, Operation transa, T *a, T beta,
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

  virtual T *_create(int count) = 0;
  virtual void _destroy(const T *mem) = 0;

  virtual T *_createIdentityMatrix(int rows, int columns) = 0;

  virtual int _getCount(const T *mem) const = 0;
  virtual int _getSizeInBytes(const T *mem) const = 0;

  virtual void _copyHostToKernel(T *mem, const T *array, int count) = 0;
  virtual void _copyKernelToHost(T *array, const T *mem, int count) = 0;

  virtual uintt _amax(const T *mem) = 0;
  virtual uintt _amin(const T *mem) = 0;

  virtual void _rot(T *x, T *y, T &&c, T &&s) = 0;

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
  std::map<const T *, std::tuple<int, int>> m_matrices;

  void checkMem(const T *mem) const;
};

} // namespace mtrx

#endif
