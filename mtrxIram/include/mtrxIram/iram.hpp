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

#ifndef MTRX_IRAM_IRAM_HPP
#define MTRX_IRAM_IRAM_HPP

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <mtrxCore/blas.hpp>
#include <mtrxCore/types.hpp>

#include "iram_types.hpp"

namespace mtrx {

/**
 * For refence see:
 * https://www.caam.rice.edu/software/ARPACK/UG/node45.html#SECTION00800000000000000000
 */
class Iram final {
public:
  Iram(ValueType valueType, CalculationDevice calculationDevice);
  ~Iram() = default;

  void setArnoldiFactorizationLength(int afLength);

  using EigenValues = std::vector<float>;
  using Wanted = std::vector<int>;
  using Unwanted = std::vector<int>;

  void setVectorToInit(Mem *vector, MemoryType type);
  void setUnitVectorToInit(unsigned int length, MemoryType type);
  void setRandomVectorToInit(unsigned int length, MemoryType type);

  /**
   * Sort of eigenvalues.
   * As the output it should be wanted and unwanted set which contain indexies
   * of eigenvalues from EigenValues
   */
  using Sort = std::function<void(Wanted &wanted, Unwanted &unwanted,
                                  const EigenValues &eigenValues)>;
  void setEigenValuesSort(const Sort &sort);
  void setEigenValuesSort(Sort &&sort);

  void start();

private:
  ValueType m_valueType = ValueType::NOT_DEFINED;
  CalculationDevice m_calculationDevice;
  std::shared_ptr<mtrx::Blas> m_blas = nullptr;

  int m_afLength = 0;

  std::pair<InitVectorType, MemoryType> m_initVecType =
      std::make_pair(InitVectorType::NONE, MemoryType::NONE);

  Mem *m_initVector = nullptr;
  unsigned int m_length = 0;

  Sort m_sort = nullptr;

  void check();
  void checkInitVector();
  void createInitVector();
};

} // namespace mtrx

#endif
