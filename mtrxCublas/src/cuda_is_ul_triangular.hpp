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

#ifndef MTRX_CUBLAS_CUDA_IS_UL_TRIANGULAR_HPP
#define MTRX_CUBLAS_CUDA_IS_UL_TRIANGULAR_HPP

#include "cuda_core.hpp"
#include "cuda_math_utils.hpp"
#include "cuda_reduce.hpp"

template <typename T>
__device__ void cuda_isUpperTriangular(int rows, int columns, T *matrix,
                                       int lda, T delta,
                                       int *reductionResults) {
  HOST_INIT();

  int *reduceBuffer = nullptr;
  GENERIC_INIT_SHARED(int, reduceBuffer);

  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;

  const int shmIdx = threadIdx.x * blockDim.y + threadIdx.y;
  reduceBuffer[shmIdx] = 0;
  if (y < rows && x < columns) {
    const T v = matrix[x * lda + y];
    int is = (x >= y && cuda_isAbsHigher(v, delta)) ||
             (x < y && !cuda_isAbsHigher(v, delta));

    reduceBuffer[x * blockDim.y + y] = is;
    const int reduceBufferLen = blockDim.x * blockDim.y;
    cuda_reduce_shm_multi_blocks<int>(reduceBuffer, reduceBufferLen,
                                      reductionResults);
  }
}

template <typename T>
__device__ void cuda_isLowerTriangular(int rows, int columns, T *matrix,
                                       int lda, T delta,
                                       int *reductionResults) {
  HOST_INIT();

  int *reduceBuffer = nullptr;
  GENERIC_INIT_SHARED(int, reduceBuffer);

  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;

  const int shmIdx = threadIdx.x * blockDim.y + threadIdx.y;
  reduceBuffer[shmIdx] = 0;
  if (y < rows && x < columns) {
    const T v = matrix[x * lda + y];
    int is = (x > y && !cuda_isAbsHigher(v, delta)) ||
             (x <= y && cuda_isAbsHigher(v, delta));

    reduceBuffer[x * blockDim.y + y] = is;
    const int reduceBufferLen = blockDim.x * blockDim.y;
    cuda_reduce_shm_multi_blocks<int>(reduceBuffer, reduceBufferLen,
                                      reductionResults);
  }
}

#endif
