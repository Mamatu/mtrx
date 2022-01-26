#ifndef MTRX_MEM_IS_EQUAL_TO_HPP
#define MTRX_MEM_IS_EQUAL_TO_HPP

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mtrxCore/blas.hpp>

namespace mtrx {

using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

template <typename Blas, typename T>
class MemIsEqualMatcher : public MatcherInterface<Mem *> {

protected:
  Mem *m_mem2 = nullptr;
  Blas *m_blas = nullptr;
  T m_delta = 0;

public:
  MemIsEqualMatcher(Mem *mem2, Blas *blas, T delta)
      : m_mem2(mem2), m_blas(blas), m_delta(delta) {}

  virtual bool MatchAndExplain(Mem *mem1, MatchResultListener *listener) const {
    std::string v = m_blas->toStr(mem1);
    auto valueType = m_blas->getValueType(mem1);

    (*listener) << v;

    std::map<ValueType, std::function<bool(Mem *)>> compares = {
        {ValueType::FLOAT, [this](Mem *mem1) { return compare(mem1); }},
        {ValueType::DOUBLE, [this](Mem *mem1) { return compare(mem1); }},
    };

    auto it = compares.find(valueType);
    if (it == compares.end()) {
      return false;
    }

    return it->second(mem1);
  }

  virtual void DescribeTo(::std::ostream *os) const {
    *os << "Mems are equal.";
  }

  virtual void DescribeNegationTo(::std::ostream *os) const {
    *os << "Mems are not equal.";
  }

  bool compare(Mem *mem1) const {
    size_t count1 = m_blas->getCount(mem1);
    size_t count2 = m_blas->getCount(m_mem2);
    if (count1 != count2) {
      return false;
    }
    std::vector<T> vec1(count1, 0);
    std::vector<T> vec2(count2, 0);
    m_blas->copyKernelToHost(vec1.data(), mem1);
    m_blas->copyKernelToHost(vec2.data(), m_mem2);
    for (size_t idx = 0; idx < vec1.size(); ++idx) {
      T v1 = vec1[idx];
      T v2 = vec2[idx];
      if (abs(v1 - v2) > m_delta) {
        return false;
      }
    }
    return true;
  }
};

} // namespace mtrx
#endif
