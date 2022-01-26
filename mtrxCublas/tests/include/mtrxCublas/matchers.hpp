#ifndef MTRX_MATCHERS_HPP
#define MTRX_MATCHERS_HPP

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "mem_is_equal_to.hpp"
#include "mem_values_are_equal_to.hpp"
#include "mtrxCublas/cublas.hpp"

namespace mtrx {
using ::testing::Matcher;

constexpr auto DELTA = 0.0000001f;

template <typename T, typename Blas>
Matcher<Mem *> MemValuesAreEqualTo(T value, Blas *blas, T delta = DELTA) {
  return MakeMatcher(
      new MemValuesAreEqualToMatcher<T, Blas>(value, blas, delta));
}

template <typename T, typename Blas>
Matcher<Mem *> MemIsEqual(Mem *mem, Blas *blas, T delta = DELTA) {
  return MakeMatcher(new MemIsEqualMatcher<Blas, T>(mem, blas, delta));
}
} // namespace mtrx
#endif
