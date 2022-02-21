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

#ifndef MTRX_CORE_TYPES_HPP
#define MTRX_CORE_TYPES_HPP

#include <cstdint>

namespace mtrx {
using intt = int;
using uintt = uint64_t;

enum class ValueType {
  FLOAT,
  DOUBLE,
  FLOAT_COMPLEX,
  DOUBLE_COMPLEX,
  NOT_DEFINED
};

enum class Operation { OP_N, OP_T, OP_C };

using Oper = Operation;

enum class FillMode { LOWER, UPPER, FULL };

enum class SideMode { LEFT, RIGHT };

struct Mem;
} // namespace mtrx

#endif
