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

#ifndef MTRX_CUBLAS_CUDA_TRACE_SCALE_HPP
#define MTRX_CUBLAS_CUDA_TRACE_SCALE_HPP

#include "cuda_core.hpp"
#include <cuComplex.h>

#ifdef MTRX_HOST_CUDA_BUILD
#include <spdlog/spdlog.h>

#include <string>

template <typename T>
void logMulResult(const std::string &file, const std::string &func, int line,
                  T *matrix, int idx, T a, T b, int x, int lda) {
  spdlog::debug("{} {} : {}\n {} = {} * {} ({}, {}) (x: {}, lda: {})", file,
                func, line, matrix[idx], a, b, idx, idx, x, lda);
}

template <typename T>
void logMulResultComplex(const std::string &file, const std::string &func,
                         int line, T *matrix, int idx, T a, T b, int x,
                         int lda) {
  spdlog::debug(
      "{} {} : {}\n {}+{}i = {}+{}i * {}+{}i ({}, {}) (x: {}, lda: {})", file,
      func, line, matrix[idx].x, matrix[idx].y, a.x, a.y, b.x, b.y, idx, idx, x,
      lda);
}
#endif

__device__ __inline__ void
cuda_SF_scaleTrace(int /*m*/, int /*n*/, float *matrix, int lda, float factor) {
  HOST_CODE(spdlog::set_level(spdlog::level::debug);)
  HOST_INIT();
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int idx = x * lda + x;
  HOST_CODE(float host_matrixValue = matrix[idx];);
  matrix[idx] = matrix[idx] * factor;
  HOST_CODE(logMulResult(__FILE__, __func__, __LINE__, matrix, idx,
                         host_matrixValue, factor, x, lda);)
}

__device__ __inline__ void cuda_SD_scaleTrace(int /*m*/, int /*n*/,
                                              double *matrix, int lda,
                                              double factor) {
  HOST_INIT();
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int idx = x * lda + x;
  HOST_CODE(double host_matrixValue = matrix[idx];);
  matrix[idx] = matrix[idx] * factor;
  HOST_CODE(logMulResult(__FILE__, __func__, __LINE__, matrix, idx,
                         host_matrixValue, factor, x, lda);)
}

__device__ __inline__ void cuda_CF_scaleTrace(int /*m*/, int /*n*/,
                                              cuComplex *matrix, int lda,
                                              cuComplex factor) {
  HOST_INIT();
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int idx = x * lda + x;
  HOST_CODE(cuComplex host_matrixValue = matrix[idx];);
  matrix[idx] = cuCmulf(matrix[idx], factor);
  HOST_CODE(logMulResultComplex(__FILE__, __func__, __LINE__, matrix, idx,
                                host_matrixValue, factor, x, lda);)
}

__device__ __inline__ void cuda_CD_scaleTrace(int /*m*/, int /*n*/,
                                              cuDoubleComplex *matrix, int lda,
                                              cuDoubleComplex factor) {
  HOST_INIT();
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int idx = x * lda + x;
  HOST_CODE(cuDoubleComplex host_matrixValue = matrix[idx];);
  matrix[idx] = cuCmul(matrix[idx], factor);
  HOST_CODE(logMulResultComplex(__FILE__, __func__, __LINE__, matrix, idx,
                                host_matrixValue, factor, x, lda);)
}

#endif
