#ifndef MTRX_MEM_VALUES_ARE_EQUAL_TO_VALUE_HPP
#define MTRX_MEM_VALUES_ARE_EQUAL_TO_VALUE_HPP

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mtrxCore/blas.hpp>

namespace mtrx {

using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

template <typename T, typename Blas>
class MemValuesAreEqualToValueMatcher : public MatcherInterface<T *> {
  T m_value;
  Blas *m_blas;
  T m_delta;

public:
  MemValuesAreEqualToValueMatcher(T value, Blas *blas, T delta)
      : m_value(value), m_blas(blas), m_delta(delta) {}

  virtual bool MatchAndExplain(T *mem, MatchResultListener *listener) const {

    std::string v = m_blas->toStr(mem);

    auto valueType = m_blas->getValueType(mem);

    (*listener) << v;
    return compare(mem);
  }

  bool compare(T *mem) const {
    auto count = m_blas->getCount(mem);
    std::vector<T> vec(count, 0);
    m_blas->copyKernelToHost(vec.data(), mem);
    for (size_t idx = 0; idx < vec.size(); ++idx) {
      T v = vec[idx];
      if (abs(v - m_value) > m_delta) {
        return false;
      }
    }
    return true;
  }

  virtual void DescribeTo(::std::ostream *os) const {
    *os << "Mem values are equal " << m_value;
  }

  virtual void DescribeNegationTo(::std::ostream *os) const {
    *os << "Mem values are not equal " << m_value;
  }
};

} // namespace mtrx
#endif
