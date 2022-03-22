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

#ifndef MTRX_CUBLAS_CUDA_CORE_HPP
#define MTRX_CUBLAS_CUDA_CORE_HPP

#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define UNIQUE_NAME(base) CONCAT(base, __COUNTER__)

#ifndef MTRX_HOST_CUDA_BUILD

#include <cuda.h>
#include <cuda_runtime.h>

#define HOST_INIT()

#define SHARED_BUFFER_INIT_SET(type, buffer, mtrx_shared_buffer) extern __shared__ char mtrx_shared_buffer[]; buffer = reinterpret_cast<type*>(mtrx_shared_buffer);
#define GENERIC_INIT_SHARED(type, buffer) SHARED_BUFFER_INIT_SET(type, buffer, UNIQUE_NAME(mtrx_device_shared_buffer))

#define HOST_CODE(code)

#else

#include "host/thread_idx.hpp"
#include <pthread.h>
#include <vector_types.h>

// setting sequence is for suppress warning
#define HOST_INIT()                                                            \
  mtrx::ThreadIdx &ti = mtrx::ThreadIdx::GetThreadIdx();                       \
  uint3 threadIdx = ti.getThreadIdx();                                         \
  dim3 blockIdx = ti.getBlockIdx();                                            \
  dim3 blockDim = ti.getBlockDim();                                            \
  dim3 gridDim = ti.getGridDim();                                              \
  threadIdx = threadIdx; /*for suppress warning*/                              \
  blockIdx = blockIdx;   /*for suppress warning*/                              \
  blockDim = blockDim;   /*for suppress warning*/                              \
  gridDim = gridDim;     /*for suppress warning*/

#define SHARED_BUFFER_INIT_SET(type, buffer, mtrx_shared_buffer) mtrx::ThreadIdx &mtrx_shared_buffer = mtrx::ThreadIdx::GetThreadIdx(); buffer = static_cast<type *>(mtrx_shared_buffer.getSharedBuffer());
#define GENERIC_INIT_SHARED(type, buffer) SHARED_BUFFER_INIT_SET(type, buffer, UNIQUE_NAME(mtrx_host_shared_buffer))

#define HOST_CODE(code) code

#define __syncthreads() mtrx::ThreadIdx::wait();

#endif
#endif
