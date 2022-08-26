#ifndef MTRX_CONTAINER_IS_EQUAL_TO_MEM_HPP
#define MTRX_CONTAINER_IS_EQUAL_TO_MEM_HPP

#include "compare.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mtrxCore/blas.hpp>

namespace mtrx {

using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

template <typename T, typename Container, typename Blas>
class ContainerIsEqualToMemMatcher : public MatcherInterface<T *> {

protected:
  T *m_mem = nullptr;
  Blas *m_blas = nullptr;
  T m_delta = 0;

public:
  ContainerIsEqualToMemMatcher(T *mem, Blas *blas, T delta)
      : m_mem(mem), m_blas(blas), m_delta(delta) {}

  virtual bool MatchAndExplain(Container &&container,
                               MatchResultListener *listener) const {

    std::string container_str = toString(std::forward<Container>(container));
    std::string mem_str = m_blas->toStr(m_mem);

    (*listener) << container_str << " against " << mem_str;

    return compare(std::forward<Container>(container), m_mem, m_blas, m_delta);
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
