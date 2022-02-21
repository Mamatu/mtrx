#ifndef MTRX_MATCHERS_HPP
#define MTRX_MATCHERS_HPP

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <type_traits>

#include "container_is_equal_to_mem.hpp"
#include "mem_is_equal_to_container.hpp"
#include "mem_is_equal_to_mem.hpp"
#include "mem_values_are_equal_to_value.hpp"
#include "mtrxCublas/cublas.hpp"

namespace mtrx {
using ::testing::Matcher;

template <typename T, typename Blas>
Matcher<Mem *> MemValuesAreEqualTo(T value, Blas *blas, T delta) {
  return MakeMatcher(
      new MemValuesAreEqualToValueMatcher<T, Blas>(value, blas, delta));
}

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <typename Test, template <typename T, long unsigned int N> class Ref>
struct is_array_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template <template <typename T, long unsigned int N> class Ref, typename T,
          long unsigned int N>
struct is_array_specialization<Ref<T, N>, Ref> : std::true_type {};

class MemIsEqualToMemFactory {
public:
  template <typename Blas, typename T>
  MemIsEqualToMemMatcher<Blas, T> *create(Mem *mem, Blas *blas, T delta) {
    return new MemIsEqualToMemMatcher<Blas, T>(mem, blas, delta);
  }
};

class ContainerIsEqualToMemFactory {
public:
  template <typename Container, typename Blas, typename T>
  ContainerIsEqualToMemMatcher<Container, Blas, T> *
  create(Container &&container, Blas *blas, T delta) {
    return new ContainerIsEqualToMemMatcher<Container, Blas, T>(
        std::forward<Container>(container), blas, delta);
  }
};

class MemIsEqualToContainerFactory {
public:
  template <typename Container, typename Blas, typename T>
  MemIsEqualToContainerMatcher<Container, Blas, T> *
  create(Container &&container, Blas *blas, T delta) {
    return new MemIsEqualToContainerMatcher<Container, Blas, T>(
        std::forward<Container>(container), blas, delta);
  }
};

template <typename CT2, typename Blas, typename T>
Matcher<Mem *> MemIsEqualTo(CT2 &&ct2, Blas *blas, T delta) {

  using CT2_is_Mem_Ptr =
      std::is_same<typename std::remove_reference<CT2>::type, Mem *>;
  using CT2_is_vector =
      is_specialization<typename std::remove_reference<CT2>::type, std::vector>;
  using CT2_is_array =
      is_array_specialization<typename std::remove_reference<CT2>::type,
                              std::array>;

  constexpr bool isMemIsEqualToMemFactory = CT2_is_Mem_Ptr::value;
  constexpr bool isMemIsEqualToContainerFactory =
      CT2_is_vector::value || CT2_is_array::value;

  using typeMemIsEqualToContainerFactory =
      typename std::conditional<isMemIsEqualToContainerFactory,
                                MemIsEqualToContainerFactory,
                                std::false_type>::type;
  typename std::conditional<isMemIsEqualToMemFactory, MemIsEqualToMemFactory,
                            typeMemIsEqualToContainerFactory>::type factory;

  static_assert(not std::is_same<decltype(factory), std::false_type>::value);

  return MakeMatcher(factory.create(std::forward<CT2>(ct2), blas, delta));
}
} // namespace mtrx
#endif
