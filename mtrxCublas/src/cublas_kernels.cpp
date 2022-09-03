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

#include "cublas_v2.h"
#include "mtrxCublas/status_handler.hpp"
#include <mtrxCublas/impl/cublas_kernels.hpp>

#include <cstdint>
#include <stdexcept>

namespace mtrx {
namespace {
template <typename T, typename CublasIamam>
uintt cublas_amam(cublasHandle_t handler, const T *mem, int n,
                  CublasIamam &&cublasIamam) {
  int resultIdx = -1;
  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  status = cublasIamam(handler, n, mem, 1, &resultIdx);
  handleStatus(status);

  if (resultIdx == -1) {
    throw std::runtime_error("resultIdx is -1");
  }

  return resultIdx - 1;
}

} // namespace

CublasKernels::CublasKernels() { handleStatus(cublasCreate(&m_handle)); }
CublasKernels::CublasKernels(cublasHandle_t handle) : m_handle(handle) {}

uintt CublasKernels::amax(const float *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIsamax);
}

uintt CublasKernels::amax(const double *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIdamax);
}

uintt CublasKernels::amax(const cuComplex *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIcamax);
}

uintt CublasKernels::amax(const cuDoubleComplex *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIzamax);
}

uintt CublasKernels::amin(const float *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIsamin);
}

uintt CublasKernels::amin(const double *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIdamin);
}

uintt CublasKernels::amin(const cuComplex *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIcamin);
}

uintt CublasKernels::amin(const cuDoubleComplex *mem, int n) {
  return cublas_amam(m_handle, mem, n, cublasIzamin);
}

} // namespace mtrx
