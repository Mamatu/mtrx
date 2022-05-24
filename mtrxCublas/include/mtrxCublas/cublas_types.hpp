/*
 * Copyright 2021 - 2022 Marcin Matula
 *
 * This file is part of mtrx.
 *
 * mtrx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * mtrx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with mtrx.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MTRX_CUBLAS_TYPES_HPP
#define MTRX_CUBLAS_TYPES_HPP

#include <cublas_v2.h>
#include <mtrxCore/types.hpp>

#include <type_traits>

namespace mtrx {

template <ValueType vt>
using _get_cublas_value_type_dc =
    std::conditional<(vt == ValueType::DOUBLE_COMPLEX), cuDoubleComplex,
                     cuDoubleComplex>;

template <ValueType vt>
using _get_cublas_value_type_fc =
    std::conditional<(vt == ValueType::FLOAT_COMPLEX), cuComplex,
                     typename _get_cublas_value_type_dc<vt>::type>;

template <ValueType vt>
using _get_cublas_value_type_d =
    std::conditional<(vt == ValueType::DOUBLE), double,
                     typename _get_cublas_value_type_fc<vt>::type>;

template <ValueType vt>
using get_cublas_value_type =
    typename std::conditional<(vt == ValueType::FLOAT), float,
                              typename _get_cublas_value_type_d<vt>::type>;

class InvalidType {
public:
  inline ValueType get() const {
    static_assert("Invalid type");
    return ValueType::NOT_DEFINED;
  }
};

class GetFloat {
public:
  inline ValueType get() const { return ValueType::FLOAT; }
};

class GetDouble {
public:
  inline ValueType get() const { return ValueType::DOUBLE; }
};

class GetFloatComplex {
public:
  inline ValueType get() const { return ValueType::FLOAT_COMPLEX; }
};

class GetDoubleComplex {
public:
  inline ValueType get() const { return ValueType::DOUBLE_COMPLEX; }
};

template <typename T> ValueType get_value_type() {
  using get_double_complex =
      typename std::conditional<std::is_same<T, cuDoubleComplex>::value,
                                GetDoubleComplex, InvalidType>::type;
  using get_float_complex =
      typename std::conditional<std::is_same<T, cuComplex>::value,
                                GetFloatComplex, get_double_complex>::type;
  using get_double =
      typename std::conditional<std::is_same<T, double>::value, GetDouble,
                                get_float_complex>::type;
  typename std::conditional<std::is_same<T, float>::value, GetFloat,
                            get_double>::type obj;
  return obj.get();
}

} // namespace mtrx

#endif
