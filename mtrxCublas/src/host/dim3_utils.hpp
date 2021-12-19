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

#ifndef MTRX_DIM3_UTILS_HPP
#define MTRX_DIM3_UTILS_HPP

#include <vector_types.h>

inline void set(dim3 &vec, int x, int y, int z) {
  vec.x = x;
  vec.y = y;
  vec.z = z;
}

inline void clear(dim3 &vec) { set(vec, 0, 0, 0); }

inline void set(uint3 &vec, int x, int y, int z) {
  vec.x = x;
  vec.y = y;
  vec.z = z;
}

inline void clear(uint3 &vec) { set(vec, 0, 0, 0); }

#endif
