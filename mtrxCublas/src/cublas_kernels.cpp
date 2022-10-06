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
uintt cublas_amam(cublasHandle_t m_handler, const T *mem, int n,
                  CublasIamam &&cublasIamam) {
  int resultIdx = -1;
  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  status = cublasIamam(m_handler, n, mem, 1, &resultIdx);
  handleStatus(status);

  if (resultIdx == -1) {
    throw std::runtime_error("resultIdx is -1");
  }

  return resultIdx - 1;
}

inline cublasOperation_t convert(Operation operation) {
  switch (operation) {
  case Operation::OP_N:
    return CUBLAS_OP_N;
  case Operation::OP_T:
    return CUBLAS_OP_T;
  case Operation::OP_C:
    return CUBLAS_OP_C;
  };

  throw std::runtime_error("Not defined side mode");
  return CUBLAS_OP_N;
}

inline cublasSideMode_t convert(SideMode sideMode) {
  switch (sideMode) {
  case SideMode::LEFT:
    return CUBLAS_SIDE_LEFT;
  case SideMode::RIGHT:
    return CUBLAS_SIDE_RIGHT;
  };

  throw std::runtime_error("Not defined side mode");
  return CUBLAS_SIDE_LEFT;
}

inline cublasFillMode_t convert(FillMode fillMode) {
  switch (fillMode) {
  case FillMode::FULL:
    return CUBLAS_FILL_MODE_FULL;
  case FillMode::LOWER:
    return CUBLAS_FILL_MODE_LOWER;
  case FillMode::UPPER:
    return CUBLAS_FILL_MODE_UPPER;
  }

  throw std::runtime_error("Not defined fill mode");
  return CUBLAS_FILL_MODE_UPPER;
}

} // namespace

CublasKernels::CublasKernels() { handleStatus(cublasCreate(&m_handle)); }
CublasKernels::CublasKernels(cublasHandle_t m_handle) : m_handle(m_handle) {}

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

namespace {
template <typename T, typename T1, typename CublasRot>
void cublas_rot(cublasHandle_t handle, int n, T *x, T *y, T1 *c, T *s,
                CublasRot &&cublasRot) {
  auto status = cublasRot(handle, n, x, 1, y, 1, c, s);
  handleStatus(status);
}
} // namespace

void CublasKernels::rot(int n, float *x, float *y, float c, float s) {
  cublas_rot(m_handle, n, x, y, &c, &s, cublasSrot);
}

void CublasKernels::rot(int n, float *x, float *y, float *c, float *s) {
  cublas_rot(m_handle, n, x, y, c, s, cublasSrot);
}

void CublasKernels::rot(int n, double *x, double *y, double c, double s) {
  cublas_rot(m_handle, n, x, y, &c, &s, cublasDrot);
}

void CublasKernels::rot(int n, double *x, double *y, double *c, double *s) {
  cublas_rot(m_handle, n, x, y, c, s, cublasDrot);
}

void CublasKernels::rot(int n, cuComplex *x, cuComplex *y, cuComplex c,
                        cuComplex s) {
  if (c.y != 0) {
    throw std::runtime_error("Imag part of c parametr must be 0");
  }
  cublas_rot(m_handle, n, x, y, &c.x, &s, cublasCrot);
}

void CublasKernels::rot(int n, cuComplex *x, cuComplex *y, cuComplex *c,
                        cuComplex *s) {
  cublas_rot(m_handle, n, x, y, &c->x, s, cublasCrot);
}

void CublasKernels::rot(int n, cuDoubleComplex *x, cuDoubleComplex *y,
                        cuDoubleComplex c, cuDoubleComplex s) {
  cublas_rot(m_handle, n, x, y, &c.x, &s, cublasZrot);
}

void CublasKernels::rot(int n, cuDoubleComplex *x, cuDoubleComplex *y,
                        cuDoubleComplex *c, cuDoubleComplex *s) {
  if (c->y != 0) {
    throw std::runtime_error("Imag part of c parametr must be 0");
  }
  cublas_rot(m_handle, n, x, y, &c->x, s, cublasZrot);
}

void CublasKernels::syr(uint lda, FillMode fillMode, int n, float *output,
                        float *alpha, float *x) {
  auto status =
      cublasSsyr(m_handle, convert(fillMode), n, alpha, x, 1, output, lda);
  handleStatus(status);
}

void CublasKernels::syr(uint lda, FillMode fillMode, int n, double *output,
                        double *alpha, double *x) {
  auto status =
      cublasDsyr(m_handle, convert(fillMode), n, alpha, x, 1, output, lda);
  handleStatus(status);
}

void CublasKernels::syr(uint lda, FillMode fillMode, int n, cuComplex *output,
                        cuComplex *alpha, cuComplex *x) {
  auto status =
      cublasCsyr(m_handle, convert(fillMode), n, alpha, x, 1, output, lda);
  handleStatus(status);
}

void CublasKernels::syr(uint lda, FillMode fillMode, int n,
                        cuDoubleComplex *output, cuDoubleComplex *alpha,
                        cuDoubleComplex *x) {
  auto status =
      cublasZsyr(m_handle, convert(fillMode), n, alpha, x, 1, output, lda);
  handleStatus(status);
}

void CublasKernels::gemm(float *output, int m, int n, int k, float *alpha,
                         Operation transa, float *a, Operation transb, float *b,
                         float *beta) {
  auto status = cublasSgemm(m_handle, convert(transa), convert(transb), m, n, k,
                            alpha, a, m, b, k, beta, output, m);
  handleStatus(status);
}

void CublasKernels::gemm(float *output, int m, int n, int k, float alpha,
                         Operation transa, float *a, Operation transb, float *b,
                         float beta) {
  gemm(output, m, n, k, &alpha, transa, a, transb, b, &beta);
}

void CublasKernels::gemm(double *output, int m, int n, int k, double *alpha,
                         Operation transa, double *a, Operation transb,
                         double *b, double *beta) {
  auto status = cublasDgemm(m_handle, convert(transa), convert(transb), m, n, k,
                            alpha, a, m, b, k, beta, output, m);
  handleStatus(status);
}

void CublasKernels::gemm(double *output, int m, int n, int k, double alpha,
                         Operation transa, double *a, Operation transb,
                         double *b, double beta) {
  gemm(output, m, n, k, &alpha, transa, a, transb, b, &beta);
}

void CublasKernels::gemm(cuComplex *output, int m, int n, int k,
                         cuComplex *alpha, Operation transa, cuComplex *a,
                         Operation transb, cuComplex *b, cuComplex *beta) {
  auto status = cublasCgemm(m_handle, convert(transa), convert(transb), m, n, k,
                            alpha, a, m, b, k, beta, output, m);
  handleStatus(status);
}

void CublasKernels::gemm(cuComplex *output, int m, int n, int k,
                         cuComplex alpha, Operation transa, cuComplex *a,
                         Operation transb, cuComplex *b, cuComplex beta) {
  gemm(output, m, n, k, &alpha, transa, a, transb, b, &beta);
}

void CublasKernels::gemm(cuDoubleComplex *output, int m, int n, int k,
                         cuDoubleComplex *alpha, Operation transa,
                         cuDoubleComplex *a, Operation transb,
                         cuDoubleComplex *b, cuDoubleComplex *beta) {

  auto status = cublasZgemm(m_handle, convert(transa), convert(transb), m, n, k,
                            alpha, a, m, b, k, beta, output, m);
  handleStatus(status);
}

void CublasKernels::gemm(cuDoubleComplex *output, int m, int n, int k,
                         cuDoubleComplex alpha, Operation transa,
                         cuDoubleComplex *a, Operation transb,
                         cuDoubleComplex *b, cuDoubleComplex beta) {
  gemm(output, m, n, k, &alpha, transa, a, transb, b, &beta);
}

void CublasKernels::symm(float *output, SideMode sideMode, FillMode fillMode,
                         int m, int n, float *alpha, float *a, float *b,
                         float *beta) {
  auto status = cublasSsymm(m_handle, convert(sideMode), convert(fillMode), m,
                            n, alpha, a, m, b, m, beta, output, m);
  handleStatus(status);
}

void CublasKernels::symm(float *output, SideMode sideMode, FillMode fillMode,
                         int m, int n, float alpha, float *a, float *b,
                         float beta) {
  symm(output, sideMode, fillMode, m, n, &alpha, a, b, &beta);
}

void CublasKernels::symm(double *output, SideMode sideMode, FillMode fillMode,
                         int m, int n, double *alpha, double *a, double *b,
                         double *beta) {

  auto status = cublasDsymm(m_handle, convert(sideMode), convert(fillMode), m,
                            n, alpha, a, m, b, m, beta, output, m);
  handleStatus(status);
}

void CublasKernels::symm(double *output, SideMode sideMode, FillMode fillMode,
                         int m, int n, double alpha, double *a, double *b,
                         double beta) {
  symm(output, sideMode, fillMode, m, n, &alpha, a, b, &beta);
}

void CublasKernels::symm(cuComplex *output, SideMode sideMode,
                         FillMode fillMode, int m, int n, cuComplex *alpha,
                         cuComplex *a, cuComplex *b, cuComplex *beta) {

  auto status = cublasCsymm(m_handle, convert(sideMode), convert(fillMode), m,
                            n, alpha, a, m, b, m, beta, output, m);
  handleStatus(status);
}

void CublasKernels::symm(cuComplex *output, SideMode sideMode,
                         FillMode fillMode, int m, int n, cuComplex alpha,
                         cuComplex *a, cuComplex *b, cuComplex beta) {
  symm(output, sideMode, fillMode, m, n, &alpha, a, b, &beta);
}

void CublasKernels::symm(cuDoubleComplex *output, SideMode sideMode,
                         FillMode fillMode, int m, int n,
                         cuDoubleComplex *alpha, cuDoubleComplex *a,
                         cuDoubleComplex *b, cuDoubleComplex *beta) {
  auto status = cublasZsymm(m_handle, convert(sideMode), convert(fillMode), m,
                            n, alpha, a, m, b, m, beta, output, m);
  handleStatus(status);
}

void CublasKernels::symm(cuDoubleComplex *output, SideMode sideMode,
                         FillMode fillMode, int m, int n, cuDoubleComplex alpha,
                         cuDoubleComplex *a, cuDoubleComplex *b,
                         cuDoubleComplex beta) {
  symm(output, sideMode, fillMode, m, n, &alpha, a, b, &beta);
}
void CublasKernels::geqrfBatched(int m, int n, float **a, int lda, float **tau,
                                 int *info, int batchSize) {
  auto handle =
      cublasSgeqrfBatched(m_handle, m, n, a, lda, tau, info, batchSize);
  handleStatus(handle);
}

void CublasKernels::geqrfBatched(int m, int n, double **a, int lda,
                                 double **tau, int *info, int batchSize) {
  auto handle =
      cublasDgeqrfBatched(m_handle, m, n, a, lda, tau, info, batchSize);
  handleStatus(handle);
}
void CublasKernels::geqrfBatched(int m, int n, cuComplex **a, int lda,
                                 cuComplex **tau, int *info, int batchSize) {
  auto handle =
      cublasCgeqrfBatched(m_handle, m, n, a, lda, tau, info, batchSize);
  handleStatus(handle);
}

void CublasKernels::geqrfBatched(int m, int n, cuDoubleComplex **a, int lda,
                                 cuDoubleComplex **tau, int *info,
                                 int batchSize) {
  auto handle =
      cublasZgeqrfBatched(m_handle, m, n, a, lda, tau, info, batchSize);
  handleStatus(handle);
}

void CublasKernels::geam(float *output, int ldo, int m, int n, float *alpha,
                         Operation transa, float *a, int lda, float *beta,
                         Operation transb, float *b, int ldb) {
  auto status = cublasSgeam(m_handle, convert(transa), convert(transb), m, n,
                            alpha, a, lda, beta, b, ldb, output, ldo);
  handleStatus(status);
}

void CublasKernels::geam(float *output, int ldo, int m, int n, float alpha,
                         Operation transa, float *a, int lda, float beta,
                         Operation transb, float *b, int ldb) {
  geam(output, ldo, m, n, &alpha, transa, a, lda, &beta, transb, b, ldb);
}

void CublasKernels::geam(double *output, int ldo, int m, int n, double *alpha,
                         Operation transa, double *a, int lda, double *beta,
                         Operation transb, double *b, int ldb) {
  auto status = cublasDgeam(m_handle, convert(transa), convert(transb), m, n,
                            alpha, a, lda, beta, b, ldb, output, ldo);
  handleStatus(status);
}

void CublasKernels::geam(double *output, int ldo, int m, int n, double alpha,
                         Operation transa, double *a, int lda, double beta,
                         Operation transb, double *b, int ldb) {
  geam(output, ldo, m, n, &alpha, transa, a, lda, &beta, transb, b, ldb);
}

void CublasKernels::geam(cuComplex *output, int ldo, int m, int n,
                         cuComplex *alpha, Operation transa, cuComplex *a,
                         int lda, cuComplex *beta, Operation transb,
                         cuComplex *b, int ldb) {
  auto status = cublasCgeam(m_handle, convert(transa), convert(transb), m, n,
                            alpha, a, lda, beta, b, ldb, output, ldo);
  handleStatus(status);
}

void CublasKernels::geam(cuComplex *output, int ldo, int m, int n,
                         cuComplex alpha, Operation transa, cuComplex *a,
                         int lda, cuComplex beta, Operation transb,
                         cuComplex *b, int ldb) {
  geam(output, ldo, m, n, &alpha, transa, a, lda, &beta, transb, b, ldb);
}

void CublasKernels::geam(cuDoubleComplex *output, int ldo, int m, int n,
                         cuDoubleComplex *alpha, Operation transa,
                         cuDoubleComplex *a, int lda, cuDoubleComplex *beta,
                         Operation transb, cuDoubleComplex *b, int ldb) {
  auto status = cublasZgeam(m_handle, convert(transa), convert(transb), m, n,
                            alpha, a, lda, beta, b, ldb, output, ldo);
  handleStatus(status);
}

void CublasKernels::geam(cuDoubleComplex *output, int ldo, int m, int n,
                         cuDoubleComplex alpha, Operation transa,
                         cuDoubleComplex *a, int lda, cuDoubleComplex beta,
                         Operation transb, cuDoubleComplex *b, int ldb) {
  geam(output, ldo, m, n, &alpha, transa, a, lda, &beta, transb, b, ldb);
}

} // namespace mtrx
