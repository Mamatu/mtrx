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

#ifndef MTRX_CORE_IRAM_API_HPP
#define MTRX_CORE_IRAM_API_HPP

#include <functional>
#include <map>
#include <vector>

#include <mtrxCore/types.hpp>

namespace mtrx {

/**
 * For refence see:
 * https://www.caam.rice.edu/software/ARPACK/UG/node45.html#SECTION00800000000000000000
 */
class Iram {
public:
  enum class Type { CUDA, HOST };
  void setArnoldiFactorizationLength(int afLength);

  using EigenValues = std::vector<float>;
  using Wanted = std::vector<int>;
  using Unwanted = std::vector<int>;

  void setVectorToInit(Mem* vector, Type type);
  void setUnitVectorToInit(Type type);
  void setRandomVectorToInit(Type type);

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
  int m_afLength = 0;

  enum class InitVecType { CustomVector, UnitVector, RandomVector, NotSet };
  InitVecType m_initVecType = InitVecType::NotSet;
  Mem* m_initVector;

  Sort m_sort = nullptr;

  void check();
};

} // namespace mtrx

#endif
