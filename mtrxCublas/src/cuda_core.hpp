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

#ifndef MTRX_HOST_CUDA_BUILD

#include <cuda.h>
#include <cuda_runtime.h>

#define HOST_INIT()

#define HOST_INIT_SHARED(type, buffer)                                         \
  extern __shared__ type mtrx_shared_buffer[];                                 \
  buffer = mtrx_shared_buffer;

#define HOST_CODE(code)

#else

#include <vector_types.h>
//#include "host/dim3.hpp"
#include "host/thread_idx.hpp"
#include <pthread.h>

//#define __global__ inline
//#define __host__
//#define __device__
//#define __shared__
//#define __inline__ inline

#define HOST_INIT()                                                            \
  mtrx::ThreadIdx &ti = mtrx::ThreadIdx::GetThreadIdx();               \
  uint3 threadIdx = ti.getThreadIdx();                                         \
  dim3 blockIdx = ti.getBlockIdx();                                            \
  dim3 blockDim = ti.getBlockDim();                                            \
  dim3 gridDim = ti.getGridDim();

#define HOST_INIT_SHARED(type, buffer)                                         \
  buffer = static_cast<type*>(mtrx::ThreadIdx::GetThreadIdx());                     \

#define HOST_CODE(code) code

#define __syncthreads() mtrx::ThreadIdx::wait();

#endif
#endif
