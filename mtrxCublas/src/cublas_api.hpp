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

#ifndef MTRX_CUBLAS_CUBLAS_API_H
#define MTRX_CUBLAS_CUBLAS_API_H

#include "mtrxCore/types.hpp"
#include <cuComplex.h>

namespace mtrx {
class CublasApi {
public:
  void gemm(cublasHandle_t handle, float *output, float *alpha,
            Operation transa, float *a, Operation transb, float *b,
            float *beta);

  void gemm(cublasHandle_t handle, fdouble *output, double *alpha,
            Operation transa, double *a, Operation transb, double *b,
            double *beta);

  void gemm(cublasHandle_t handle, cuComplex *output, cuComplex *alpha,
            Operation transa, cuComplex *a, Operation transb, cuComplex *b,
            cuComplex *beta);

  void gemm(cublasHandle_t handle, cuDoubleComplex *output,
            cuDoubleComplex *alpha, Operation transa, cuDoubleComplex *a,
            Operation transb, cuDoubleComplex *b, cuDoubleComplex *beta);
};
} // namespace mtrx

#endif
