
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

#ifndef MTRX_IRAM_IRAM_ROUTINES_HPP
#define MTRX_IRAM_IRAM_ROUTINES_HPP

#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>
#include <mtrxCublas/cublas_types.hpp>
#include <mtrxIram/iram_types.hpp>

#include <memory>

#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace mtrx {

template <typename T> using BlasPtr = typename std::shared_ptr<mtrx::Blas<T>>;

template <typename T, typename T1>
void createRandomUnitVector(const BlasPtr<T> &blasPtr, int length);

template <typename T>
void createUnitVector(const BlasPtr<T> &blasPtr, int length,
                      ValueType valueType);

template <typename T>
void checkCustomInitVector(const BlasPtr<T> &blasPtr, T *mem);

} // namespace mtrx

namespace mtrx {
namespace {
template <typename HVT>
void initRandom(std::vector<HVT> &values, size_t length) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  values.resize(length);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    values[idx] = static_cast<HVT>(dis(gen));
  }
}

template <typename T> void convertToUnit(std::vector<T> &values) {
  T factor = static_cast<T>(0.);
  for (auto it = values.begin(); it != values.end(); ++it) {
    factor = factor + (*it) * (*it);
  }
  factor = sqrt(factor);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    values[idx] = values[idx] / factor;
  }
}

template <typename T>
T *_createFromVector(const BlasPtr<T> &blasPtr, unsigned int length,
                     const std::vector<T> &values) {
  auto *d_vector = blasPtr->createMatrix(length, 1);
  blasPtr->copyHostToKernel(d_vector, values.data());
  return d_vector;
}

class Factor1 {
public:
  Factor1(unsigned int length) : m_length(length) {}

  unsigned int getLength() const { return m_length; }

private:
  unsigned int m_length;
};

class Factor2 {
public:
  Factor2(unsigned int length) : m_length(length * 2) {}

  unsigned int getLength() const { return m_length; }

private:
  unsigned int m_length;
};

template <typename T, typename T1>
T *_createRandomUnitVector(const BlasPtr<T> &blasPtr, unsigned int length) {
  auto _length = length;
  if (mtrx::isComplex(valueType)) {
    _length = _length * 2;
  }
  std::vector<T1> h_vector;
  initRandom<T1>(h_vector, _length);
  convertToUnit<T1>(h_vector);
  return _createFromVector<T, T1>(blasPtr, length, h_vector);
}

template <typename T, typename T1>
T *_createUnitVector(const BlasPtr<T> &blasPtr, unsigned int length) {
  auto _length = length;
  if (mtrx::isComplex(valueType)) {
    _length = _length * 2;
  }
  std::vector<T1> vector(_length, static_cast<T1>(0));
  vector[0] = static_cast<T1>(1);

  auto *d_vector = blasPtr->createMatrix(length, 1);
  blasPtr->copyHostToKernel(d_vector, vector.data());

  return d_vector;
}

} // namespace

template <typename T>
T *createRandomUnitVector(const BlasPtr<T> &blasPtr, unsigned int length) {

  switch (valueType) {
  case ValueType::FLOAT:
    return _createRandomUnitVector<float, float>(blasPtr, length, valueType);
  case ValueType::DOUBLE:
    return _createRandomUnitVector<double, double>(blasPtr, length, valueType);
  case ValueType::FLOAT_COMPLEX:
    return _createRandomUnitVector<cuComplex, float>(blasPtr, length,
                                                     valueType);
  case ValueType::DOUBLE_COMPLEX:
    return _createRandomUnitVector<cuDoubleComplex, double>(blasPtr, length,
                                                            valueType);
  default:
    std::runtime_error("Not supported");
  }
  return nullptr;
}

template <typename T>
T *createUnitVector(const BlasPtr &blasPtr, unsigned int length) {
  switch (valueType) {
  case ValueType::FLOAT:
    return _createUnitVector<float, float>(blasPtr, length, valueType);
  case ValueType::DOUBLE:
    return _createUnitVector<double, double>(blasPtr, length, valueType);
  case ValueType::FLOAT_COMPLEX:
    return _createUnitVector<cuComplex, float>(blasPtr, length, valueType);
  case ValueType::DOUBLE_COMPLEX:
    return _createUnitVector<cuDoubleComplex, double>(blasPtr, length,
                                                      valueType);
  default:
    std::runtime_error("Not supported");
  }
  return nullptr;
}

void checkCustomInitVector(const BlasPtr &blasPtr, Mem *mem) {
  if (!blasPtr->isAllocator(mem)) {
    std::stringstream sstream;
    throw std::runtime_error(
        "Custom vector is only supported if allocated by specific blas api");
  }

  if (!blasPtr->isUnit(mem, )) {
  }
}

} // namespace mtrx

#endif
