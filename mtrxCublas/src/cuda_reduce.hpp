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

#ifndef MTRX_CUBLAS_CUDA_REDUCE_HPP
#define MTRX_CUBLAS_CUDA_REDUCE_HPP

#include "cuda_core.hpp"
#include <cuComplex.h>

/**
 * @brief reduction function which use shared memory buffer for calculations
 */
template <typename T> __device__ void cuda_reduce_shm(int m, int n, T *array) {
  HOST_INIT();

  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;
  const int rows = blockDim.y * gridDim.y;

  int idx = y + rows * x;
  int dim = m * n;
  while (dim > 1) {
    int next_dim = dim / 2;
    array[idx] += array[idx + next_dim];
    if (next_dim * 2 < dim && idx == next_dim - 1) {
      array[idx] += array[idx + next_dim + 1];
    }
    dim = next_dim;
    __syncthreads();
  }
}

template <typename T> __device__ void cuda_reduce(int m, int n, T *array) {
  cuda_reduce_shm(m, n, array);
}

#endif
