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

#ifdef MTRX_HOST_CUDA_BUILD
#include <spdlog/spdlog.h>

#include <string>

template <typename T>
void logMulResult(const std::string &file, const std::string &func, int line,
                  T *array, int idx, T a, T b, int x, int lda) {
  spdlog::debug("{} {} : {}\n {} = {} * {} ({}, {}) (x: {}, lda: {})", file,
                func, line, array[idx], a, b, idx, idx, x, lda);
}

template <typename T>
void logMulResultComplex(const std::string &file, const std::string &func,
                         int line, T *array, int idx, T a, T b, int x,
                         int lda) {
  spdlog::debug(
      "{} {} : {}\n {}+{}i = {}+{}i * {}+{}i ({}, {}) (x: {}, lda: {})", file,
      func, line, array[idx].x, array[idx].y, a.x, a.y, b.x, b.y, idx, idx, x,
      lda);
}
#endif

__device__ __inline__ void cuda_SF_reduce(int m, int n, float *array, int lda,
                                          float delta) {
  HOST_CODE(spdlog::set_level(spdlog::level::debug);)
  HOST_INIT();
}

__device__ __inline__ void cuda_SD_reduce(int m, int n, double *array, int lda,
                                          double delta) {}

__device__ __inline__ void cuda_CF_reduce(int m, int n, cuComplex *array,
                                          int lda, cuComplex delta) {}

__device__ __inline__ void cuda_CD_reduce(int m, int n, cuDoubleComplex *array,
                                          int lda, cuDoubleComplex delta) {}

#endif
