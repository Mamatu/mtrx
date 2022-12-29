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

#ifndef MTRX_CORE_BLAS_IMPL_HPP
#define MTRX_CORE_BLAS_IMPL_HPP

#include "mtrxCore/types.hpp"
#include <algorithm>
#include <sstream>
#include <tuple>

#include <mtrxCore/blas_decl.hpp>
#include <mtrxCore/size_of.hpp>

namespace mtrx {

template <typename T> std::vector<int> Blas<T>::getDevices() const {
  return _getDevices();
}

template <typename T> void Blas<T>::setDevice(int device) {
  _setDevice(device);
}

template <typename T> T *Blas<T>::create(int count) {
  T *mem = _create(count);
  m_mems.push_back(mem);
  return mem;
}

template <typename T> T *Blas<T>::createMatrix(int rows, int columns) {
  if (rows == 0 || columns == 0) {
    std::stringstream sstream;
    sstream << "Invalid dim for matrix: (" << rows << ", " << columns << ")";
    throw std::runtime_error(sstream.str());
  }

  auto *mem = create(rows * columns);
  m_matrices[mem] = std::make_tuple(rows, columns);

  return mem;
}

template <typename T> T *Blas<T>::createMatrix(int rows, int columns, T *mem) {
  const int count = getCount(mem);
  if (count != rows * columns) {
    std::stringstream sstream;
    sstream << "Matrix dim is not equal to count. Dim:(";
    sstream << rows << ", ";
    sstream << columns << ") Count: " << count;
    throw std::runtime_error(sstream.str());
  }

  m_matrices[mem] = std::make_tuple(rows, columns);
  return mem;
}

template <typename T> T *Blas<T>::createIdentityMatrix(int rows, int columns) {
  return _createIdentityMatrix(rows, columns);
}

template <typename T> void Blas<T>::destroy(const T *mem) {
  checkMem(mem);
  _destroy(mem);
}

template <typename T> bool Blas<T>::isAllocator(const T *mem) const {
  auto it = std::find(m_mems.begin(), m_mems.end(), mem);
  return it != m_mems.end();
}

template <typename T> bool Blas<T>::isComplex() const { return _isComplex(); }

template <typename T> int Blas<T>::getCount(const T *mem) const {
  checkMem(mem);
  return _getCount(mem);
}

template <typename T> int Blas<T>::getSizeInBytes(const T *mem) const {
  checkMem(mem);
  return _getSizeInBytes(mem);
}

template <typename T> int Blas<T>::getRows(const T *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return getCount(mem);
  }
  return std::get<0>(it->second);
}

template <typename T> int Blas<T>::getColumns(const T *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return 1;
  }
  return std::get<1>(it->second);
}

template <typename T> std::pair<int, int> Blas<T>::getDims(const T *mem) const {
  auto rows = getRows(mem);
  auto columns = getColumns(mem);
  return std::make_pair(rows, columns);
}

template <typename T> void Blas<T>::copyHostToKernel(T *mem, const T *array) {
  checkMem(mem);
  auto count = getCount(mem);
  _copyHostToKernel(mem, array, count);
}

template <typename T> void Blas<T>::copyKernelToHost(T *array, const T *mem) {
  checkMem(mem);
  auto count = getCount(mem);
  _copyKernelToHost(array, mem, count);
}

template <typename T>
void Blas<T>::copyHostToKernel(T *mem, const T *array, int count) {
  checkMem(mem);
  _copyHostToKernel(mem, array, count);
}

template <typename T>
void Blas<T>::copyKernelToHost(T *array, const T *mem, int count) {
  checkMem(mem);
  _copyKernelToHost(array, mem, count);
}

template <typename T> uintt Blas<T>::amax(const T *mem) {
  checkMem(mem);
  return _amax(mem);
}

template <typename T> uintt Blas<T>::amin(const T *mem) {
  checkMem(mem);
  return _amin(mem);
}

template <typename T> void Blas<T>::rot(T *x, T *y, T &&c, T &&s) {
  checkMem(x);
  checkMem(y);
  _rot(x, y, std::forward<T>(c), std::forward<T>(s));
}

template <typename T>
void Blas<T>::syr(FillMode fillMode, T *output, T *alpha, T *x) {
  checkMem(output);
  checkMem(x);

  _syr(fillMode, output, alpha, x);
}

template <typename T>
void Blas<T>::gemm(T *output, T *alpha, T *a, T *b, T *beta) {
  // Checking of matrix is redundant here
  gemm(output, alpha, Operation::OP_N, a, Operation::OP_N, b, beta);
}

template <typename T>
void Blas<T>::gemm(T *output, T *alpha, Operation transa, T *a,
                   Operation transb, T *b, T *beta) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  _gemm(output, alpha, transa, a, transb, b, beta);
}

template <typename T>
void Blas<T>::symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha,
                   T *a, T *b, T *beta) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  checkMem(alpha);
  checkMem(beta);

  _symm(sideMode, fillMode, output, alpha, a, b, beta);
}

template <typename T> void Blas<T>::matrixMul(T *output, T *a, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _matrixMul(output, a, b);
}

template <typename T> void Blas<T>::geqrf(T *a, T *tau) {
  checkMem(a);
  checkMem(tau);
  _geqrf(a, tau);
}

template <typename T> void Blas<T>::geqrf(Vec &a, Vec &tau) { _geqrf(a, tau); }

template <typename T> void Blas<T>::qrDecomposition(T *q, T *r, T *a) {
  checkMem(q);
  checkMem(r);
  checkMem(a);

  _qrDecomposition(q, r, a);
}

template <typename T>
void Blas<T>::qrDecomposition(const Vec &q, const Vec &r, const Vec &a) {
  checkVec(q);
  checkVec(r);
  checkVec(a);

  _qrDecomposition(q, r, a);
}

template <typename T> void Blas<T>::shiftQRIteration(T *H, T *Q) {
  checkMem(H);
  checkMem(Q);

  _shiftQRIteration(H, Q);
}

template <typename T> bool Blas<T>::isUpperTriangular(T *m) {
  checkMem(m);
  return _isUpperTriangular(m);
}

template <typename T> bool Blas<T>::isLowerTriangular(T *m) {
  checkMem(m);
  return _isLowerTriangular(m);
}

template <typename T>
void Blas<T>::geam(T *output, T *alpha, Operation transa, T *a, T *beta,
                   Operation transb, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _geam(output, alpha, transa, a, beta, transb, b);
}
template <typename T>
void Blas<T>::geam(T *output, T alpha, Operation transa, T *a, T beta,
                   Operation transb, T *b) {
  this->geam(output, &alpha, transa, a, &beta, transb, b);
}

template <typename T> void Blas<T>::add(T *output, T *a, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _add(output, a, b);
}

template <typename T> void Blas<T>::subtract(T *output, T *a, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _subtract(output, a, b);
}

template <typename T> void Blas<T>::scaleDiagonal(T *matrix, T factor) {
  checkMem(matrix);
  _scaleDiagonal(matrix, &factor);
}

template <typename T>
void Blas<T>::tpttr(FillMode uplo, int n, T *AP, T *A, int lda) {
  checkMem(AP);
  checkMem(A);
  _tpttr(uplo, n, AP, A, lda);
}

template <typename T>
void Blas<T>::trttp(FillMode uplo, int n, T *A, int lda, T *AP) {
  checkMem(AP);
  checkMem(A);
  _trttp(uplo, n, A, lda, AP);
}

template <typename T> bool Blas<T>::isUnit(T *mem, T delta) {
  checkMem(mem);
  return _isUnit(mem, &delta);
}

template <typename T> bool Blas<T>::eye(T *mem, T delta) {
  return this->isUnit(mem, delta);
}

template <typename T> T Blas<T>::cast(int v) const { return _cast(v); }

template <typename T> T Blas<T>::cast(float v) const { return _cast(v); }

template <typename T> T Blas<T>::cast(double v) const { return _cast(v); }

template <typename T> std::string Blas<T>::toStr(T *mem) {
  checkMem(mem);
  return _toStr(mem);
}

template <typename T> void Blas<T>::checkMem(const T *mem) const {
  if (!isAllocator(mem)) {
    std::stringstream sstream;
    sstream << "Mem " << mem << " is not allocated in " << this;
    throw std::runtime_error(sstream.str());
  }
}

} // namespace mtrx

#endif
