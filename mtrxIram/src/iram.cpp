#include <mtrxIram/iram.hpp>

#include <mtrxCore/types.hpp>
#include <mtrxCublas/cublas.hpp>
#include <mtrxCublas/cublas_types.hpp>

#include "checkers.hpp"
#include "iram_cublas_routines.hpp"

namespace mtrx {
Iram::Iram(ValueType valueType, CalculationDevice calculationDevice)
    : m_valueType(valueType), m_calculationDevice(calculationDevice) {}

void Iram::setArnoldiFactorizationLength(int afLength) {
  m_afLength = afLength;
}

void Iram::setVectorToInit(Mem *vector, MemoryType type) {
  m_initVector = vector;
  m_initVecType = std::make_pair(InitVectorType::CUSTOM_VECTOR, type);
}

void Iram::setUnitVectorToInit(unsigned int length, MemoryType type) {
  m_initVecType = std::make_pair(InitVectorType::UNIT_VECTOR, type);
  m_length = length;
}

void Iram::setRandomVectorToInit(unsigned int length, MemoryType type) {
  m_initVecType = std::make_pair(InitVectorType::RANDOM_UNIT_VECTOR, type);
  m_length = length;
}

void Iram::setEigenValuesSort(const Sort &sort) { m_sort = sort; }

void Iram::setEigenValuesSort(Sort &&sort) { m_sort = std::move(sort); }

void Iram::check() {
  checkAFLength(m_afLength);
  checkSortFunction(m_sort);
}

void Iram::checkInitVector() {
  if (m_initVecType.first == InitVectorType::RANDOM_UNIT_VECTOR) {
    return;
  }
  if (m_initVecType.first == InitVectorType::UNIT_VECTOR) {
    return;
  }
  mtrx::checkInitVector(m_initVector, m_initVecType.second);
}

namespace {
template <mtrx::ValueType vt>
using get_type = typename mtrx::get_cublas_value_type<vt>::type;
}

void Iram::createInitVector() {

  if (m_initVecType.first == InitVectorType::RANDOM_UNIT_VECTOR) {
  }
  if (m_initVecType.first == InitVectorType::UNIT_VECTOR) {
    // std::vector<mtrx::get_type<m_valueType>> initVector(
    //    m_length, static_cast<mtrx::get_type<m_valueType>>(0));
  }
  if (m_initVecType.first == InitVectorType::CUSTOM_VECTOR &&
      m_initVecType.second == MemoryType::HOST) {
    // auto *initVector = m_blas->createMatrix(m_length, 1, m_valueType);
  }
}

void Iram::start() {
  check();
  createInitVector();
  checkInitVector();
}
} // namespace mtrx
