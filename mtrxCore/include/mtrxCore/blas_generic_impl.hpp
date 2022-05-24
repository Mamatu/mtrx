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

#include "mtrxCore/types.hpp"
#include <algorithm>
#include <sstream>

#include <mtrxCore/blas_generic.hpp>
#include <mtrxCore/size_of.hpp>

namespace mtrx {

template<typename T>
std::vector<int> BlasGeneric<T>::getDevices() const { return _getDevices(); }

template<typename T>
void BlasGeneric<T>::setDevice(int device) { _setDevice(device); }

template<typename T>
T *BlasGeneric<T>::create(size_t count) {
  T *mem = _createMem(count);
  m_mems.push_back(mem);
  return mem;
}

template<typename T>
T *BlasGeneric<T>::createMatrix(size_t rows, size_t columns) {
  if (rows == 0 || columns == 0) {
    std::stringstream sstream;
    sstream << "Invalid dim for matrix: (" << rows << ", " << columns << ")";
    throw std::runtime_error(sstream.str());
  }

  auto *mem = create(rows * columns);
  m_matrices[mem] = std::make_tuple(rows, columns);

  return mem;
}

template<typename T>
T *BlasGeneric<T>::createMatrix(size_t rows, size_t columns, T *mem) {
  const size_t count = getCount(mem);
  auto valueType = getValueType(mem);
  if (count != rows * columns) {
    std::stringstream sstream;
    sstream << "Matrix dim is not equal to count. Dim:(";
    sstream << rows << ", ";
    sstream << columns << ") Count: " << count;
    throw std::runtime_error(sstream.str());
  }

  m_matrices[mem] = std::make_tuple(rows, columns, valueType);
  return mem;
}

template<typename T>
T *BlasGeneric<T>::createIdentityMatrix(size_t rows, size_t columns) {
  return _createIdentityMatrix(rows, columns);
}

template<typename T>
void BlasGeneric<T>::destroy(const T *mem) {
  checkMem(mem);
  _destroy(mem);
}

template<typename T>
bool BlasGeneric<T>::isAllocator(const T *mem) const {
  auto it = std::find(m_mems.begin(), m_mems.end(), mem);
  return it != m_mems.end();
}

template<typename T>
size_t BlasGeneric<T>::getCount(const T *mem) const {
  checkMem(mem);
  return _getCount(mem);
}

template<typename T>
size_t BlasGeneric<T>::getSizeInBytes(const T *mem) const {
  checkMem(mem);
  return _getSizeInBytes(mem);
}

template<typename T>
size_t BlasGeneric<T>::getRows(const T *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return getCount(mem);
  }
  return std::get<0>(it->second);
}

template<typename T>
size_t BlasGeneric<T>::getColumns(const T *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return 1;
  }
  return std::get<1>(it->second);
}

template<typename T>
ValueType BlasGeneric<T>::getValueType(const T *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return ValueType::NOT_DEFINED;
  }
  return std::get<2>(it->second);
}

template<typename T>
std::pair<size_t, size_t> BlasGeneric<T>::getDims(const T *mem) const {
  auto rows = getRows(mem);
  auto columns = getColumns(mem);
  return std::make_pair(rows, columns);
}

template<typename T>
void BlasGeneric<T>::copyHostToKernel(T *mem, void *array) {
  checkMem(mem);
  _copyHostToKernel(mem, array);
}

template<typename T>
void BlasGeneric<T>::copyHostToKernel(T *mem, const void *array) {
  checkMem(mem);
  _copyHostToKernel(mem, array);
}

template<typename T>
void BlasGeneric<T>::copyKernelToHost(void *array, T *mem) {
  checkMem(mem);
  _copyKernelToHost(array, mem);
}

template<typename T>
uintt BlasGeneric<T>::amax(const T *mem) {
  checkMem(mem);
  return _amax(mem);
}

template<typename T>
uintt BlasGeneric<T>::amin(const T *mem) {
  checkMem(mem);
  return _amin(mem);
}

template<typename T>
void BlasGeneric<T>::rot(T *x, T *y, T *c, T *s) {
  checkMem(x);
  checkMem(y);
  checkMem(c);
  checkMem(s);
  _rot(x, y, c, s);
}

template<typename T>
void BlasGeneric<T>::rot(T *x, T *y, float c, float s) {
  checkMem(x);
  checkMem(y);
  rot(x, y, &c, ValueType::FLOAT, &s, ValueType::FLOAT);
}

template<typename T>
void BlasGeneric<T>::rot(T *x, T *y, double c, double s) {
  checkMem(x);
  checkMem(y);
  rot(x, y, &c, ValueType::DOUBLE, &s, ValueType::DOUBLE);
}

template<typename T>
void BlasGeneric<T>::syr(FillMode fillMode, T *output, void *alpha, T *x) {
  checkMem(output);
  checkMem(x);

  _syr(fillMode, output, alpha, x);
}

template<typename T>
void BlasGeneric<T>::syr(FillMode fillMode, T *output, T *alpha, T *x) {
  checkMem(output);
  checkMem(x);

  checkMem(alpha);

  _syr(fillMode, output, alpha, x);
}

template<typename T>
void BlasGeneric<T>::gemm(T *output, void *alpha, T *a, T *b, void *beta) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  gemm(output, alpha, Operation::OP_N, a, Operation::OP_N, b, beta);
}

template<typename T>
void BlasGeneric<T>::gemm(T *output, T *alpha, T *a, T *b, T *beta) {
  // Checking of matrix is redundant here
  gemm(output, alpha, Operation::OP_N, a, Operation::OP_N, b, beta);
}

template<typename T>
void BlasGeneric<T>::gemm(T *output, void *alpha, Operation transa,
                T *a, Operation transb, T *b, void *beta) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  _gemm(output, alpha, transa, a, transb, b, beta);
}

template<typename T>
void BlasGeneric<T>::gemm(T *output, T *alpha, Operation transa, T *a,
                Operation transb, T *b, T *beta) {
  checkMem(output);
  checkMem(alpha);
  checkMem(a);
  checkMem(b);
  checkMem(beta);

  _gemm(output, alpha, transa, a, transb, b, beta);
}

template<typename T>
void BlasGeneric<T>::symm(SideMode sideMode, FillMode fillMode, T *output, void *alpha,
                T *a, T *b, void *beta) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  _symm(sideMode, fillMode, output, alpha, a, b, beta);
}

template<typename T>
void BlasGeneric<T>::symm(SideMode sideMode, FillMode fillMode, T *output, T *alpha,
                T *a, T *b, T *beta) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  checkMem(alpha);
  checkMem(beta);

  _symm(sideMode, fillMode, output, alpha, a, b, beta);
}

template<typename T>
void BlasGeneric<T>::matrixMul(T *output, T *a, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _matrixMul(output, a, b);
}

template<typename T>
void BlasGeneric<T>::geqrf(T *a, T *tau) {
  checkMem(a);
  checkMem(tau);
  _geqrf(a, tau);
}

template<typename T>
void BlasGeneric<T>::geqrf(Mems &a, Mems &tau) { _geqrf(a, tau); }

template<typename T>
void BlasGeneric<T>::qrDecomposition(T *q, T *r, T *a) {
  checkMem(q);
  checkMem(r);
  checkMem(a);

  _qrDecomposition(q, r, a);
}

template<typename T>
void BlasGeneric<T>::qrDecomposition(Mems &q, Mems &r, Mems &a) {
  checkMems(q);
  checkMems(r);
  checkMems(a);

  _qrDecomposition(q, r, a);
}

template<typename T>
void BlasGeneric<T>::shiftQRIteration(T *H, T *Q) {
  checkMem(H);
  checkMem(Q);

  _shiftQRIteration(H, Q);
}

template<typename T>
bool BlasGeneric<T>::isUpperTriangular(T *m) {
  checkMem(m);
  return _isUpperTriangular(m);
}

template<typename T>
bool BlasGeneric<T>::isLowerTriangular(T *m) {
  checkMem(m);
  return _isLowerTriangular(m);
}

template<typename T>
void BlasGeneric<T>::geam(T *output, T *alpha, Operation transa, T *a, T *beta,
                Operation transb, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _geam(output, alpha, transa, a, beta, transb, b);
}

template<typename T>
void BlasGeneric<T>::geam(T *output, void *alpha, Operation transa,
                T *a, void *beta, Operation transb,
                T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _geam(output, alpha, transa, a, beta, transb, b);
}

template<typename T>
void BlasGeneric<T>::add(T *output, T *a, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _add(output, a, b);
}

template<typename T>
void BlasGeneric<T>::subtract(T *output, T *a, T *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _subtract(output, a, b);
}

template<typename T>
void BlasGeneric<T>::scaleDiagonal(T *matrix, T *factor) {
  checkMem(matrix);
  _scaleDiagonal(matrix, factor);
}

template<typename T>
void BlasGeneric<T>::scaleDiagonal(T *matrix, void *factor) {
  checkMem(matrix);
  _scaleDiagonal(matrix, factor);
}

template<typename T>
void BlasGeneric<T>::tpttr(FillMode uplo, int n, T *AP, T *A, int lda) {
  checkMem(AP);
  checkMem(A);
  _tpttr(uplo, n, AP, A, lda);
}

template<typename T>
void BlasGeneric<T>::trttp(FillMode uplo, int n, T *A, int lda, T *AP) {
  checkMem(AP);
  checkMem(A);
  _trttp(uplo, n, A, lda, AP);
}

template<typename T>
bool BlasGeneric<T>::isUnit(T *mem, void *delta) {
  checkMem(mem);
  return _isUnit(mem, delta);
}

template<typename T>
bool BlasGeneric<T>::isUnit(T *mem, T *delta) {
  checkMem(mem);
  return _isUnit(mem, delta);
}

template<typename T>
std::string BlasGeneric<T>::toStr(T *mem) {
  checkMem(mem);
  return _toStr(mem);
}

template<typename T>
void BlasGeneric<T>::checkMem(const T *mem) const {
  if (!isAllocator(mem)) {
    std::stringstream sstream;
    sstream << "Mem " << mem << " is not allocated in " << this;
    throw std::runtime_error(sstream.str());
  }
}

template<typename T>
void BlasGeneric<T>::checkMems(const Mems &mems) const {
  for (const auto *mem : mems) {
    checkMem(mem);
  }
}
} // namespace mtrx
