#ifndef MTRX_MEM_VALUES_ARE_EQUAL_TO_VALUE_HPP
#define MTRX_MEM_VALUES_ARE_EQUAL_TO_VALUE_HPP

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mtrxCore/blas.hpp>

namespace mtrx {

using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

template <typename T, typename Blas>
class MemValuesAreEqualToValueMatcher : public MatcherInterface<mtrx::Mem *> {
  T m_value;
  Blas *m_blas;
  T m_delta;

public:
  MemValuesAreEqualToValueMatcher(T value, Blas *blas, T delta)
      : m_value(value), m_blas(blas), m_delta(delta) {}

  virtual bool MatchAndExplain(mtrx::Mem *mem,
                               MatchResultListener *listener) const {

    std::string v = m_blas->toStr(mem);

    auto valueType = m_blas->getValueType(mem);

    (*listener) << v;

    std::map<ValueType, std::function<bool(Mem *)>> compares = {
        {ValueType::FLOAT, [this](Mem *mem) { return compare(mem); }},
        {ValueType::DOUBLE, [this](Mem *mem) { return compare(mem); }},
    };

    auto it = compares.find(valueType);
    if (it == compares.end()) {
      return false;
    }

    return it->second(mem);
  }

  bool compare(Mem *mem) const {
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
