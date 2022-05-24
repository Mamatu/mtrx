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

#include <mtrxCore/blas.hpp>
#include <mtrxCore/size_of.hpp>

namespace mtrx {

std::vector<int> Blas::getDevices() const { return _getDevices(); }

void Blas::setDevice(int device) { _setDevice(device); }

Mem *Blas::create(size_t count, ValueType valueType) {
  Mem *mem = _createMem(count, valueType);
  m_mems.push_back(mem);
  return mem;
}

Mem *Blas::createMatrix(size_t rows, size_t columns, ValueType valueType) {
  if (rows == 0 || columns == 0) {
    std::stringstream sstream;
    sstream << "Invalid dim for matrix: (" << rows << ", " << columns << ")";
    throw std::runtime_error(sstream.str());
  }

  auto *mem = create(rows * columns, valueType);
  m_matrices[mem] = std::make_tuple(rows, columns, valueType);

  return mem;
}

Mem *Blas::createMatrix(size_t rows, size_t columns, Mem *mem) {
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

Mem *Blas::createIdentityMatrix(size_t rows, size_t columns,
                                ValueType valueType) {
  return _createIdentityMatrix(rows, columns, valueType);
}

void Blas::destroy(const Mem *mem) {
  checkMem(mem);
  _destroy(mem);
}

bool Blas::isAllocator(const Mem *mem) const {
  auto it = std::find(m_mems.begin(), m_mems.end(), mem);
  return it != m_mems.end();
}

size_t Blas::getCount(const Mem *mem) const {
  checkMem(mem);
  return _getCount(mem);
}

size_t Blas::getSizeInBytes(const Mem *mem) const {
  checkMem(mem);
  return _getSizeInBytes(mem);
}

size_t Blas::getRows(const Mem *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return getCount(mem);
  }
  return std::get<0>(it->second);
}

size_t Blas::getColumns(const Mem *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return 1;
  }
  return std::get<1>(it->second);
}

ValueType Blas::getValueType(const Mem *mem) const {
  const auto it = m_matrices.find(mem);
  if (it == m_matrices.end()) {
    return ValueType::NOT_DEFINED;
  }
  return std::get<2>(it->second);
}

std::pair<size_t, size_t> Blas::getDims(const Mem *mem) const {
  auto rows = getRows(mem);
  auto columns = getColumns(mem);
  return std::make_pair(rows, columns);
}

void Blas::copyHostToKernel(Mem *mem, void *array) {
  checkMem(mem);
  _copyHostToKernel(mem, array);
}

void Blas::copyHostToKernel(Mem *mem, const void *array) {
  checkMem(mem);
  _copyHostToKernel(mem, array);
}

void Blas::copyKernelToHost(void *array, Mem *mem) {
  checkMem(mem);
  _copyKernelToHost(array, mem);
}

uintt Blas::amax(const Mem *mem) {
  checkMem(mem);
  return _amax(mem);
}

uintt Blas::amin(const Mem *mem) {
  checkMem(mem);
  return _amin(mem);
}

void Blas::rot(Mem *x, Mem *y, void *c, ValueType cType, void *s,
               ValueType sType) {
  checkMem(x);
  checkMem(y);
  _rot(x, y, c, cType, s, sType);
}

void Blas::rot(Mem *x, Mem *y, Mem *c, Mem *s) {
  checkMem(x);
  checkMem(y);
  checkMem(c);
  checkMem(s);
  _rot(x, y, c, s);
}

void Blas::rot(Mem *x, Mem *y, float c, float s) {
  checkMem(x);
  checkMem(y);
  rot(x, y, &c, ValueType::FLOAT, &s, ValueType::FLOAT);
}

void Blas::rot(Mem *x, Mem *y, double c, double s) {
  checkMem(x);
  checkMem(y);
  rot(x, y, &c, ValueType::DOUBLE, &s, ValueType::DOUBLE);
}

void Blas::syr(FillMode fillMode, Mem *output, void *alpha, ValueType alphaType,
               Mem *x) {
  checkMem(output);
  checkMem(x);

  _syr(fillMode, output, alpha, alphaType, x);
}

void Blas::syr(FillMode fillMode, Mem *output, Mem *alpha, Mem *x) {
  checkMem(output);
  checkMem(x);

  checkMem(alpha);

  _syr(fillMode, output, alpha, x);
}

void Blas::gemm(Mem *output, void *alpha, ValueType alphaType, Mem *a, Mem *b,
                void *beta, ValueType betaType) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  gemm(output, alpha, alphaType, Operation::OP_N, a, Operation::OP_N, b, beta,
       betaType);
}

void Blas::gemm(Mem *output, Mem *alpha, Mem *a, Mem *b, Mem *beta) {
  // Checking of matrix is redundant here
  gemm(output, alpha, Operation::OP_N, a, Operation::OP_N, b, beta);
}

void Blas::gemm(Mem *output, void *alpha, ValueType alphaType, Operation transa,
                Mem *a, Operation transb, Mem *b, void *beta,
                ValueType betaType) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  _gemm(output, alpha, alphaType, transa, a, transb, b, beta, betaType);
}

void Blas::gemm(Mem *output, Mem *alpha, Operation transa, Mem *a,
                Operation transb, Mem *b, Mem *beta) {
  checkMem(output);
  checkMem(alpha);
  checkMem(a);
  checkMem(b);
  checkMem(beta);

  _gemm(output, alpha, transa, a, transb, b, beta);
}

void Blas::symm(SideMode sideMode, FillMode fillMode, Mem *output, void *alpha,
                ValueType alphaType, Mem *a, Mem *b, void *beta,
                ValueType betaType) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  _symm(sideMode, fillMode, output, alpha, alphaType, a, b, beta, betaType);
}

void Blas::symm(SideMode sideMode, FillMode fillMode, Mem *output, Mem *alpha,
                Mem *a, Mem *b, Mem *beta) {
  checkMem(output);
  checkMem(a);
  checkMem(b);

  checkMem(alpha);
  checkMem(beta);

  _symm(sideMode, fillMode, output, alpha, a, b, beta);
}

void Blas::matrixMul(Mem *output, Mem *a, Mem *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _matrixMul(output, a, b);
}

void Blas::geqrf(Mem *a, Mem *tau) {
  checkMem(a);
  checkMem(tau);
  _geqrf(a, tau);
}

void Blas::geqrf(Mems &a, Mems &tau) { _geqrf(a, tau); }

void Blas::qrDecomposition(Mem *q, Mem *r, Mem *a) {
  checkMem(q);
  checkMem(r);
  checkMem(a);

  _qrDecomposition(q, r, a);
}

void Blas::qrDecomposition(Mems &q, Mems &r, Mems &a) {
  checkMems(q);
  checkMems(r);
  checkMems(a);

  _qrDecomposition(q, r, a);
}

void Blas::shiftQRIteration(Mem *H, Mem *Q) {
  checkMem(H);
  checkMem(Q);

  _shiftQRIteration(H, Q);
}

bool Blas::isUpperTriangular(Mem *m) {
  checkMem(m);
  return _isUpperTriangular(m);
}

bool Blas::isLowerTriangular(Mem *m) {
  checkMem(m);
  return _isLowerTriangular(m);
}

void Blas::geam(Mem *output, Mem *alpha, Operation transa, Mem *a, Mem *beta,
                Operation transb, Mem *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _geam(output, alpha, transa, a, beta, transb, b);
}

void Blas::geam(Mem *output, void *alpha, ValueType alphaType, Operation transa,
                Mem *a, void *beta, ValueType betaType, Operation transb,
                Mem *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _geam(output, alpha, alphaType, transa, a, beta, betaType, transb, b);
}

void Blas::add(Mem *output, Mem *a, Mem *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _add(output, a, b);
}

void Blas::subtract(Mem *output, Mem *a, Mem *b) {
  checkMem(output);
  checkMem(a);
  checkMem(b);
  _subtract(output, a, b);
}

void Blas::scaleDiagonal(Mem *matrix, Mem *factor) {
  checkMem(matrix);
  _scaleDiagonal(matrix, factor);
}

void Blas::scaleDiagonal(Mem *matrix, void *factor, ValueType factorType) {
  checkMem(matrix);
  _scaleDiagonal(matrix, factor, factorType);
}

void Blas::tpttr(FillMode uplo, int n, Mem *AP, Mem *A, int lda) {
  checkMem(AP);
  checkMem(A);
  _tpttr(uplo, n, AP, A, lda);
}

void Blas::trttp(FillMode uplo, int n, Mem *A, int lda, Mem *AP) {
  checkMem(AP);
  checkMem(A);
  _trttp(uplo, n, A, lda, AP);
}

bool Blas::isUnit(Mem *mem, void *delta, ValueType deltaType) {
  checkMem(mem);
  return _isUnit(mem, delta, deltaType);
}

bool Blas::isUnit(Mem *mem, Mem *delta) {
  checkMem(mem);
  return _isUnit(mem, delta);
}

std::string Blas::toStr(Mem *mem) {
  checkMem(mem);
  return _toStr(mem);
}

void Blas::checkMem(const Mem *mem) const {
  if (!isAllocator(mem)) {
    std::stringstream sstream;
    sstream << "Mem " << mem << " is not allocated in " << this;
    throw std::runtime_error(sstream.str());
  }
}

void Blas::checkMems(const Mems &mems) const {
  for (const auto *mem : mems) {
    checkMem(mem);
  }
}
} // namespace mtrx
