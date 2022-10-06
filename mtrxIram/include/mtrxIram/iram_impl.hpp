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

#ifndef MTRX_IRAM_IRAM_IMPL_HPP
#define MTRX_IRAM_IRAM_IMPL_HPP

#include <mtrxIram/impl/checkers.hpp>
#include <mtrxIram/iram_decl.hpp>

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>

#include "iram_types.hpp"

namespace mtrx {

template <typename T>
Iram<T>::Iram(CalculationDevice calculationDevice)
    : m_calculationDevice(calculationDevice) {}

template <typename T>
void Iram<T>::setArnoldiFactorizationLength(int afLength) {
  m_afLength = afLength;
}

template <typename T>
void Iram<T>::setVectorToInit(T *vector, MemoryType type) {
  m_initVector = vector;
  m_initVecType = std::make_pair(InitVectorType::CUSTOM_VECTOR, type);
}

template <typename T>
void Iram<T>::setUnitVectorToInit(unsigned int length, MemoryType type) {
  m_initVecType = std::make_pair(InitVectorType::UNIT_VECTOR, type);
  m_length = length;
}

template <typename T>
void Iram<T>::setRandomVectorToInit(unsigned int length, MemoryType type) {
  m_initVecType = std::make_pair(InitVectorType::RANDOM_UNIT_VECTOR, type);
  m_length = length;
}

template <typename T> void Iram<T>::setEigenValuesSort(const Sort &sort) {
  m_sort = sort;
}

template <typename T> void Iram<T>::setEigenValuesSort(Sort &&sort) {
  m_sort = std::move(sort);
}

template <typename T> void Iram<T>::check() {
  mtrx::checkAFLength(m_afLength);
  checkSortFunction(m_sort);
}

template <typename T> void Iram<T>::checkInitVector() {
  if (m_initVecType.first == InitVectorType::RANDOM_UNIT_VECTOR) {
    return;
  }
  if (m_initVecType.first == InitVectorType::UNIT_VECTOR) {
    return;
  }
  mtrx::checkInitVector(m_initVector, m_initVecType.second);
}

template <typename T> void Iram<T>::createInitVector() {

  if (m_initVecType.first == InitVectorType::RANDOM_UNIT_VECTOR) {
  }
  if (m_initVecType.first == InitVectorType::UNIT_VECTOR) {
    // std::vector<mtrx::get_type<m_valueType>> initVector(
    //    m_length, static_cast<mtrx::get_type<m_valueType>>(0));
  }
  if (m_initVecType.first == InitVectorType::CUSTOM_VECTOR &&
      m_initVecType.second == MemoryType::HOST) {
    auto *initVector = m_blas->createMatrix(m_length, 1);
  }
}

template <typename T> void Iram<T>::start() {
  check();
  createInitVector();
  checkInitVector();
}
} // namespace mtrx

#endif
