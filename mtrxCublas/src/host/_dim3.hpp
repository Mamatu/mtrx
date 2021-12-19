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

#ifndef MTRX_DIM3_HPP
#define MTRX_DIM3_HPP

#ifndef MTRX_HOST_CUDA_BUILD

#include <cuda.h>

#else

#include <mtrxCore/types.hpp>

namespace mtrx {
class dim3 {
public:
  dim3() {
    x = 0;
    y = 0;
    z = 1;
  }

  dim3(const int *const array) {
    x = array[0];
    y = array[1];
    z = array[2];
  }

  dim3(int x, int y, int z = 1) { set(x, y, z); }

  void clear() {
    x = 0;
    y = 0;
  }

  void set(int _x, int _y, int _z = 1) {
    x = _x;
    y = _y;
    z = _z;
  }

  int x;
  int y;
  int z;
};

void ResetCudaCtx();
} // namespace mtrx

using uint3 = mtrx::dim3;
using dim3 = mtrx::dim3;

#endif
#endif /* MTRX_DIM3_HPP */
