#ifndef MTRX_MEM_IS_EQUAL_TO_MEM_HPP
#define MTRX_MEM_IS_EQUAL_TO_MEM_HPP

#include <mtrxCore/blas.hpp>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "compare.hpp"

namespace mtrx {

using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

template <typename Blas, typename T>
class MemIsEqualToMemMatcher : public MatcherInterface<Mem *> {

protected:
  Mem* m_mem2 = nullptr;
  Blas *m_blas = nullptr;
  T m_delta = 0;

public:
  MemIsEqualToMemMatcher(Mem* mem2, Blas *blas, T delta)
      : m_mem2(mem2), m_blas(blas), m_delta(delta) {}

  virtual bool MatchAndExplain(Mem* mem1, MatchResultListener *listener) const {
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
