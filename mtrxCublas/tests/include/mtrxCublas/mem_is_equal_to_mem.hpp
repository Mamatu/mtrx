#ifndef MTRX_MEM_IS_EQUAL_TO_MEM_HPP
#define MTRX_MEM_IS_EQUAL_TO_MEM_HPP

#include "compare.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mtrxCore/blas.hpp>

namespace mtrx {

using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

template <typename T, typename Blas>
class MemIsEqualToMemMatcher : public MatcherInterface<T *> {

protected:
  T *m_mem2 = nullptr;
  Blas *m_blas = nullptr;
  T m_delta = 0;

public:
  MemIsEqualToMemMatcher(T *mem2, Blas *blas, T delta)
      : m_mem2(mem2), m_blas(blas), m_delta(delta) {}

  virtual bool MatchAndExplain(T *mem1, MatchResultListener *listener) const {
    std::string mem1_str = m_blas->toStr(mem1);
    std::string mem2_str = m_blas->toStr(m_mem2);

    (*listener) << mem1_str << " against " << mem2_str;

    return compare(mem1, m_mem2, m_blas, m_delta);
  }

  virtual void DescribeTo(::std::ostream *os) const {
    *os << "Mems are equal.";
  }

  virtual void DescribeNegationTo(::std::ostream *os) const {
    *os << "Mems are not equal.";
  }
};

} // namespace mtrx
#endif
