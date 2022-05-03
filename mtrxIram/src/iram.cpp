#include <mtrxIram/iram.hpp>

#include "checkers.hpp"

namespace mtrx {

void Iram::setArnoldiFactorizationLength(int afLength) {
  m_afLength = afLength;
}

void Iram::setVectorToInit(Mem* vector, Iram::Type type)
{
  m_initVector = vector;
  m_initVecType = InitVecType::CustomVector;
}

void Iram::setUnitVectorToInit(Iram::Type type)
{
  m_initVecType = InitVecType::UnitVector;
  //m_initVector = vector;
}

void Iram::setRandomVectorToInit(Iram::Type type)
{
  m_initVecType = InitVecType::RandomVector;
}

void Iram::setEigenValuesSort(const Sort &sort) { m_sort = sort; }

void Iram::setEigenValuesSort(Sort &&sort) { m_sort = std::move(sort); }

void Iram::check() {
  checkAFLength(m_afLength);
  checkSortFunction(m_sort);
}

void Iram::start() { check(); }
} // namespace mtrx
