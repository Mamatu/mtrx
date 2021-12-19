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

#ifndef MTRX_CORE_SIZE_OF_HPP
#define MTRX_CORE_SIZE_OF_HPP

#include <type_traits>

#include "status_handler.hpp"
#include "types.hpp"

namespace mtrx {
template <typename A> constexpr int SizeOf(int n = 1) {
  return sizeof(typename std::remove_pointer<A>::type) * n;
}
} // namespace mtrx
#endif
