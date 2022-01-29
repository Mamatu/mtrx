#ifndef MTRX_COMPARE_HPP
#define MTRX_COMPARE_HPP

#include "cuComplex.h"
#include <sstream>
#include <type_traits>
#include <mtrxCore/blas.hpp>
#include <mtrxCublas/to_string.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mtrx {

template<typename T>
void check(ValueType valueType)
{
  auto throw_exception = [valueType]()
  {
    std::stringstream sstream;
    sstream << "ValueType is " << toString(valueType) << " but T is not " << toString<T>();
    throw std::runtime_error(sstream.str());
  };

  if (ValueType::FLOAT == valueType && !std::is_same<T, float>::value)
  {
    throw_exception();
  }

  if (ValueType::DOUBLE == valueType && !std::is_same<T, double>::value)
  {
    throw_exception();
  }

  if (ValueType::FLOAT_COMPLEX == valueType && !std::is_same<T, cuComplex>::value)
  {
    throw_exception();
  }

  if (ValueType::DOUBLE_COMPLEX == valueType && !std::is_same<T, cuDoubleComplex>::value)
  {
    throw_exception();
  }
}

template<typename Container, typename Blas>
bool compare(Container&& container, Mem* mem, Blas* blas, typename Container::value_type delta)
{
  auto type = blas->getValueType(mem);
  check<typename Container::value_type> (type);

  size_t count = blas->getCount(mem);

  if (count != container.size()) {
    return false;
  }

  std::vector<typename Container::value_type> vec(count, 0);
  blas->copyKernelToHost(vec.data(), mem);

  for (size_t idx = 0; idx < vec.size(); ++idx) {
    auto v1 = vec[idx];
    auto v2 = container[idx];
    if (abs(v1 - v2) > delta) {
      return false;
    }
  }

  return true;
}

template<typename Blas, typename T>
bool compare(Mem* mem1, Mem* mem2, Blas* blas, T delta)
{
  auto type1 = blas->getValueType(mem1);
  auto type2 = blas->getValueType(mem2);

  if (type1 != type2)
  {
    std::stringstream sstream;
    sstream << toString(type1) << " != " << toString(type2);
    throw std::runtime_error(sstream.str());
  }

  size_t count1 = blas->getCount(mem1);
  size_t count2 = blas->getCount(mem2);

  if (count1 != count2) {
    return false;
  }

  std::vector<T> vec1(count1, 0);
  std::vector<T> vec2(count2, 0);
  blas->copyKernelToHost(vec1.data(), mem1);
  blas->copyKernelToHost(vec2.data(), mem2);

  for (size_t idx = 0; idx < vec1.size(); ++idx) {
    auto v1 = vec1[idx];
    auto v2 = vec2[idx];
    if (abs(v1 - v2) > delta) {
      return false;
    }
  }

  return true;
}

} // namespace mtrx
#endif
