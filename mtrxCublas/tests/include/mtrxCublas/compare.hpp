#ifndef MTRX_COMPARE_HPP
#define MTRX_COMPARE_HPP

#include "cuComplex.h"
#include <mtrxCore/blas.hpp>
#include <mtrxCublas/to_string.hpp>
#include <sstream>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mtrx {

template <typename Container, typename T, typename Blas>
bool compare(Container &&container, T *mem, Blas *blas,
             typename Container::value_type delta) {

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

template <typename Blas, typename T>
bool compare(T *mem1, T *mem2, Blas *blas, T delta) {
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
