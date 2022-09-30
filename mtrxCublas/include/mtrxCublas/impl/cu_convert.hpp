#ifndef CUBLAS_CAST_HPP
#define CUBLAS_CAST_HPP

#include <cuComplex.h>
#include <stdexcept>
#include <type_traits>

namespace mtrx {
namespace {
template <typename T> class _CuFail {
public:
  _CuFail(T) { throw std::runtime_error("Not supported type"); }
};

template <typename T> class _CuFloat {
public:
  _CuFloat(T v) : m_value(v) {}

  float m_value;
};

template <typename T> class _CuDouble {
public:
  _CuDouble(T v) : m_value(v) {}

  double m_value;
};

template <typename T> class _CuComplex {
public:
  _CuComplex(T v) : m_value(make_cuComplex(v, 0)) {}

  cuComplex m_value;
};

template <typename T> class _CuDoubleComplex {
public:
  _CuDoubleComplex(T v) : m_value(make_cuDoubleComplex(v, 0)) {}

  cuDoubleComplex m_value;
};
} // namespace

template <typename T, typename T1> T cu_convert(T1 value) {
  using CCuComplex = std::conditional<std::is_same<T, cuComplex>::value,
                                      _CuComplex<T>, _CuFail<T>>;
  using CCuDoubleComplex =
      std::conditional<std::is_same<T, cuDoubleComplex>::value,
                       _CuDoubleComplex<T>, typename CCuComplex::type>;
  using CCuDouble =
      std::conditional<std::is_same<T, double>::value, _CuDouble<T>,
                       typename CCuDoubleComplex::type>;

  typename std::conditional<std::is_same<T, float>::value, _CuFloat<T>,
                            typename CCuDouble::type>::type obj(value);
  return obj.m_value;
}
} // namespace mtrx
#endif
