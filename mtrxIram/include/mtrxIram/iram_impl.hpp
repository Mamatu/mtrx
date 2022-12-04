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

#include <mtrxIram/impl/random_generator.hpp>
#include <mtrxIram/iram_decl.hpp>

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <mtrxCore/blas.hpp>
#include <mtrxCore/thrower.hpp>
#include <mtrxCore/types.hpp>

#include "iram_types.hpp"

namespace mtrx {

template <typename T>
Iram<T>::Iram(const Iram<T>::BlasShared &blas) : m_blas(blas) {}

template <typename T>
void Iram<T>::setArnoldiFactorizationLength(int afLength) {
  m_afLength = afLength;
  checkAFLength(m_afLength);
}

template <typename T>
void Iram<T>::setVectorToInit(T *vector, MemoryType type) {
  mtrx::throw_exception_ifnot(m_blas->isAllocator(vector), [](auto &in) {
    in << "Custom init vector must be allocated by assigned blas api";
  });
  m_initVector = vector;
  m_initVecType =
      std::make_pair(InitVectorType::CUSTOM_VECTOR, MemoryType::DEVICE);
}

template <typename T> void Iram<T>::setUnitVectorToInit(unsigned int length) {
  m_initVecType =
      std::make_pair(InitVectorType::UNIT_VECTOR, MemoryType::DEVICE);
  m_length = length;
}

template <typename T> void Iram<T>::setRandomVectorToInit(unsigned int length) {
  m_initVecType =
      std::make_pair(InitVectorType::RANDOM_UNIT_VECTOR, MemoryType::DEVICE);
  m_length = length;
}

template <typename T> void Iram<T>::setEigenValuesSort(const Sort &sort) {
  m_sort = sort;
  checkSortFunction();
}

template <typename T> void Iram<T>::setEigenValuesSort(Sort &&sort) {
  m_sort = std::move(sort);
  checkSortFunction();
}

template <typename T>
T *Iram<T>::createInitVector(const Iram<T>::BlasShared &blas,
                             T *initCustomVector, InitVectorType initVectorType,
                             int rows) {
  if (initVectorType == InitVectorType::RANDOM_UNIT_VECTOR) {
    RandomGenerator<T> rg(blas->cast(0.), blas->cast(1.));
    std::vector<T> vec(rows, blas->cast(0.));
    T sum = 0;
    for (size_t idx = 0; idx < vec.size(); ++idx) {
      auto v = rg();
      vec[idx] = v;
      sum += v;
    }
    for (size_t idx = 0; idx < vec.size(); ++idx) {
      vec[idx] = vec[idx] / sum;
    }
    return blas->createMatrix(rows, 1);
  }
  if (initVectorType == InitVectorType::UNIT_VECTOR) {
    return blas->createIdentityMatrix(rows, 1);
  }
  if (initVectorType == InitVectorType::CUSTOM_VECTOR &&
      initCustomVector == nullptr) {
    throw_exception([](auto &in) {
      in << "If initVectorType is CUSTOM_VECTOR then initCustomVector "
            "cannot be nullptr";
    });
  }
  return initCustomVector;
}

template <typename T> void Iram<T>::start() {
  checkInitVector();
  m_initVector = Iram<T>::createInitVector(m_blas, m_initVector,
                                           m_initVecType.first, m_length);
}

template <typename T> void Iram<T>::checkInitVector() {
  mtrx::throw_exception_ifnot(m_blas->isUnit(m_initVector, m_blas->cast(0.00001)), [](auto &in) {
    in << "Custom init vector must be unit";
  });
}

template <typename T> void Iram<T>::checkAFLength(int afLength) {
  mtrx::throw_exception_ifnot(afLength == 0, [](auto &in) {
    in << "Arnoldi Factorization Length is not initialized properly (cannot be "
          "0)";
  });
}
} // namespace mtrx

#endif
