#ifndef MTRX_MEM_IS_EQUAL_TO_CONTAINER_HPP
#define MTRX_MEM_IS_EQUAL_TO_CONTAINER_HPP

#include "compare.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mtrxCore/blas.hpp>

namespace mtrx {

using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

template <typename Container, typename Blas, typename T>
class MemIsEqualToContainerMatcher : public MatcherInterface<Mem *> {

protected:
  Container m_container;
  Blas *m_blas = nullptr;
  T m_delta = 0;

public:
  MemIsEqualToContainerMatcher(Container &&container, Blas *blas, T delta)
      : m_container(std::forward<Container>(container)), m_blas(blas),
        m_delta(delta) {}

  virtual bool MatchAndExplain(Mem *mem, MatchResultListener *listener) const {

    std::string container_str = toString(std::forward<Container>(m_container));
    std::string mem_str = m_blas->toStr(mem);

    (*listener) << container_str << " against " << mem_str;

    return compare(std::move(m_container), mem, m_blas, m_delta);
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
