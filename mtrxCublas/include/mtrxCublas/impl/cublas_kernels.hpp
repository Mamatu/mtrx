/*
 * Copyright 2016 - 2021 Marcin Matula
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

#ifndef MTRX_CUBLAS_CUBLAS_KERNELS_H
#define MTRX_CUBLAS_CUBLAS_KERNELS_H

#include <mtrxCore/types.hpp>

#include "alloc.hpp"
#include <cuComplex.h>
#include <cublas_v2.h>

namespace mtrx {

class CublasKernels final {
public:
  CublasKernels();
  CublasKernels(cublasHandle_t handle);
  ~CublasKernels() = default;

  operator cublasHandle_t() const { return m_handle; }

  uintt amax(const float *mem, int n);
  uintt amax(const double *mem, int n);
  uintt amax(const cuComplex *mem, int n);
  uintt amax(const cuDoubleComplex *mem, int n);

  uintt amin(const float *mem, int n);
  uintt amin(const double *mem, int n);
  uintt amin(const cuComplex *mem, int n);
  uintt amin(const cuDoubleComplex *mem, int n);

private:
  cublasHandle_t m_handle;
};
} // namespace mtrx

#endif
