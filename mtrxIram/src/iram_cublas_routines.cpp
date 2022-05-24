
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

#include "iram_cublas_routines.hpp"
#include "mtrxCore/types.hpp"
#include "mtrxCublas/cublas_types.hpp"

#include <numeric>
#include <random>
#include <sstream>
#include <vector>

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

template <typename HVT> void convertToUnit(std::vector<HVT> &values) {
  HVT factor = static_cast<HVT>(0.);
  for (auto it = values.begin(); it != values.end(); ++it) {
    factor = factor + (*it) * (*it);
  }
  factor = sqrt(factor);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    values[idx] = values[idx] / factor;
  }
}

template <typename VT, typename HVT>
Mem *_createFromVector(const BlasPtr &blasPtr, unsigned int length,
                       ValueType valueType, const std::vector<HVT> &values) {
  auto *d_vector = blasPtr->createMatrix(length, 1, valueType);
  blasPtr->copyHostToKernel(d_vector, values.data());
  return d_vector;
}

template <typename VT, typename HVT>
Mem *_createRandomUnitVector(const BlasPtr &blasPtr, unsigned int length,
                             ValueType valueType) {
  auto _length = length;
  if (mtrx::isComplex(valueType)) {
    _length = _length * 2;
  }
  std::vector<HVT> h_vector;
  initRandom<HVT>(h_vector, _length);
  convertToUnit<HVT>(h_vector);
  return _createFromVector<VT, HVT>(blasPtr, length, valueType, h_vector);
}

template <typename VT, typename HVT>
Mem *_createUnitVector(const BlasPtr &blasPtr, unsigned int length,
                       ValueType valueType) {
  auto _length = length;
  if (mtrx::isComplex(valueType)) {
    _length = _length * 2;
  }
  std::vector<HVT> vector(_length, static_cast<HVT>(0));
  vector[0] = static_cast<HVT>(1);

  auto *d_vector = blasPtr->createMatrix(length, 1, valueType);
  blasPtr->copyHostToKernel(d_vector, vector.data());

  return d_vector;
}

} // namespace

Mem *createRandomUnitVector(const BlasPtr &blasPtr, unsigned int length,
                            ValueType valueType) {

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

Mem *createUnitVector(const BlasPtr &blasPtr, unsigned int length,
                      ValueType valueType) {
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
