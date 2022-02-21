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

template <typename T>
__device__ void oper_plus_equal(T *array, int idx, int idx1) {
  array[idx] += array[idx1];
}

__device__ __inline__ void oper_plus_equal(cuComplex *array, int idx,
                                           int idx1) {
  array[idx] = cuCaddf(array[idx], array[idx1]);
}

__device__ __inline__ void oper_plus_equal(cuDoubleComplex *array, int idx,
                                           int idx1) {
  array[idx] = cuCadd(array[idx], array[idx1]);
}

template <typename T> __device__ void init_with_zeros(T *array) {
  HOST_INIT();
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int trow = blockDim.y;
  array[ty + trow * tx] = 0;
}

__device__ __inline__ void init_with_zeros(cuComplex *array) {
  HOST_INIT();
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int trow = blockDim.y;
  array[ty + trow * tx] = cuComplex{0, 0};
}

__device__ __inline__ void init_with_zeros(cuDoubleComplex *array) {
  HOST_INIT();
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int trow = blockDim.y;
  array[ty + trow * tx] = cuDoubleComplex{0, 0};
}

/**
 * @brief Calculates reduction of array for single threads block.
 * WARNING! It doesn't sync threads before!
 * @param array_shm - array of shared memory type. It means that it will be
 * processed as array in one threads block.
 * @param length - length of array_shm
 * @return reduced value of one threads block.
 */
template <typename T>
__device__ T cuda_reduce_shm_single_block(T *array_shm, int length) {
  HOST_INIT();

  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int rows = blockDim.y;

  int idx = y + rows * x;
  int dim = length;

  if (idx < length) {
    while (dim > 1) {
      int next_dim = dim / 2;
      if (idx < next_dim) {
        oper_plus_equal(array_shm, idx, idx + next_dim);
        if (next_dim * 2 < dim && idx == next_dim - 1) {
          oper_plus_equal(array_shm, idx, idx + next_dim + 1);
        }
      }
      dim = next_dim;
      __syncthreads();
    }
  }
  __syncthreads();
  return array_shm[0];
}

/**
 * @brief Calculates reduction of array for single threads block. It syncs
 * threads at beginning
 * @param array_shm - array of shared memory type. It means that it will be
 * processed as array in one threads block.
 * @param length - length of array_shm
 * @return reduced value of one threads block.
 */
template <typename T>
__device__ T cuda_reduce_shm_single_block_sync(T *array_shm, int length) {
  HOST_INIT();
  __syncthreads();
  return cuda_reduce_shm_single_block(array_shm, length);
}

/**
 * @brief Calculates reduction of array with support for all blocks.
 * WARNING! It doesn't sync threads before!
 * @param reductionResults result of reduction. It should be array with size of
 * equal to number of blocks.
 * @param array_shm - array of shared memory type. It means that it will be
 * processed as array in one threads block
 * @param length - length of array_shm
 */
template <typename T>
__device__ void cuda_reduce_shm_multi_blocks(T *array_shm, int length,
                                             T *reductionResults) {
  HOST_INIT();

  T v = cuda_reduce_shm_single_block(array_shm, length);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    reductionResults[blockIdx.x * gridDim.y + blockIdx.y] = v;
  }
}

/**
 * @brief Calculates reduction of array with support for all blocks. It sync
 * threas at beginning. WARNING! It doesn't sync threads before!
 * @param reductionResults result of reduction. It should be array with size of
 * equal to number of blocks.
 * @param array_shm - array of shared memory type. It means that it will be
 * processed as array in one threads block
 * @param length - length of array_shm
 */
template <typename T>
__device__ void cuda_reduce_shm_multi_blocks_sync(T *array_shm, int length,
                                                  T *reductionResults) {
  HOST_INIT();

  __syncthreads();
  cuda_reduce_shm_multi_blocks(array_shm, length, reductionResults);
}

/**
 * @brief Calculate reduction of array.
 * WARNING! It requires shared memory!
 */
template <typename T>
__device__ void cuda_reduce_shm(int m, int n, T *array, int lda,
                                T *reductionResults) {
  HOST_INIT();

  const int tbx = threadIdx.x + blockDim.x * blockIdx.x;
  const int tby = threadIdx.y + blockDim.y * blockIdx.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int trow = blockDim.y;

  T *reduceBuffer = nullptr;
  GENERIC_INIT_SHARED(T, reduceBuffer);
  init_with_zeros(reduceBuffer);

  if (ty < m && tx < n) {
    reduceBuffer[ty + trow * tx] = array[tby + lda * tbx];
    const int reduceBufferLen = blockDim.x * blockDim.y;
    cuda_reduce_shm_multi_blocks(reduceBuffer, reduceBufferLen,
                                 reductionResults);
  }
}

#endif
