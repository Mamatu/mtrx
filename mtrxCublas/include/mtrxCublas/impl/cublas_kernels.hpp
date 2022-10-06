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

  void rot(int n, float *x, float *y, float c, float s);
  void rot(int n, float *x, float *y, float *c, float *s);
  void rot(int n, double *x, double *y, double c, double s);
  void rot(int n, double *x, double *y, double *c, double *s);
  void rot(int n, cuComplex *x, cuComplex *y, cuComplex c, cuComplex s);
  void rot(int n, cuComplex *x, cuComplex *y, cuComplex *c, cuComplex *s);
  void rot(int n, cuDoubleComplex *x, cuDoubleComplex *y, cuDoubleComplex c,
           cuDoubleComplex s);
  void rot(int n, cuDoubleComplex *x, cuDoubleComplex *y, cuDoubleComplex *c,
           cuDoubleComplex *s);

  void syr(uint lda, FillMode fillMode, int n, float *output, float *alpha,
           float *x);
  void syr(uint lda, FillMode fillMode, int n, double *output, double *alpha,
           double *x);
  void syr(uint lda, FillMode fillMode, int n, cuComplex *output,
           cuComplex *alpha, cuComplex *x);
  void syr(uint lda, FillMode fillMode, int n, cuDoubleComplex *output,
           cuDoubleComplex *alpha, cuDoubleComplex *x);

  void gemm(float *output, int m, int n, int k, float *alpha, Operation transa,
            float *a, Operation transb, float *b, float *beta);
  void gemm(float *output, int m, int n, int k, float alpha, Operation transa,
            float *a, Operation transb, float *b, float beta);

  void gemm(double *output, int m, int n, int k, double *alpha,
            Operation transa, double *a, Operation transb, double *b,
            double *beta);
  void gemm(double *output, int m, int n, int k, double alpha, Operation transa,
            double *a, Operation transb, double *b, double beta);

  void gemm(cuComplex *output, int m, int n, int k, cuComplex *alpha,
            Operation transa, cuComplex *a, Operation transb, cuComplex *b,
            cuComplex *beta);
  void gemm(cuComplex *output, int m, int n, int k, cuComplex alpha,
            Operation transa, cuComplex *a, Operation transb, cuComplex *b,
            cuComplex beta);

  void gemm(cuDoubleComplex *output, int m, int n, int k,
            cuDoubleComplex *alpha, Operation transa, cuDoubleComplex *a,
            Operation transb, cuDoubleComplex *b, cuDoubleComplex *beta);
  void gemm(cuDoubleComplex *output, int m, int n, int k, cuDoubleComplex alpha,
            Operation transa, cuDoubleComplex *a, Operation transb,
            cuDoubleComplex *b, cuDoubleComplex beta);

  void symm(float *output, SideMode sideMode, FillMode fillMode, int m, int n,
            float *alpha, float *a, float *b, float *beta);
  void symm(float *output, SideMode sideMode, FillMode fillMode, int m, int n,
            float alpha, float *a, float *b, float beta);

  void symm(double *output, SideMode sideMode, FillMode fillMode, int m, int n,
            double *alpha, double *a, double *b, double *beta);
  void symm(double *output, SideMode sideMode, FillMode fillMode, int m, int n,
            double alpha, double *a, double *b, double beta);

  void symm(cuComplex *output, SideMode sideMode, FillMode fillMode, int m,
            int n, cuComplex *alpha, cuComplex *a, cuComplex *b,
            cuComplex *beta);
  void symm(cuComplex *output, SideMode sideMode, FillMode fillMode, int m,
            int n, cuComplex alpha, cuComplex *a, cuComplex *b, cuComplex beta);

  void symm(cuDoubleComplex *output, SideMode sideMode, FillMode fillMode,
            int m, int n, cuDoubleComplex *alpha, cuDoubleComplex *a,
            cuDoubleComplex *b, cuDoubleComplex *beta);
  void symm(cuDoubleComplex *output, SideMode sideMode, FillMode fillMode,
            int m, int n, cuDoubleComplex alpha, cuDoubleComplex *a,
            cuDoubleComplex *b, cuDoubleComplex beta);

  void geqrfBatched(int m, int n, float **a, int lda, float **tau, int *info,
                    int batchSize);
  void geqrfBatched(int m, int n, double **a, int lda, double **tau, int *info,
                    int batchSize);
  void geqrfBatched(int m, int n, cuComplex **a, int lda, cuComplex **tau,
                    int *info, int batchSize);
  void geqrfBatched(int m, int n, cuDoubleComplex **a, int lda,
                    cuDoubleComplex **tau, int *info, int batchSize);

  void geam(float *output, int ldo, int m, int n, float *alpha,
            Operation transa, float *a, int lda, float *beta, Operation transb,
            float *b, int ldb);
  void geam(float *output, int ldo, int m, int n, float alpha, Operation transa,
            float *a, int lda, float beta, Operation transb, float *b, int ldb);

  void geam(double *output, int ldo, int m, int n, double *alpha,
            Operation transa, double *a, int lda, double *beta,
            Operation transb, double *b, int ldb);
  void geam(double *output, int ldo, int m, int n, double alpha,
            Operation transa, double *a, int lda, double beta, Operation transb,
            double *b, int ldb);

  void geam(cuComplex *output, int ldo, int m, int n, cuComplex *alpha,
            Operation transa, cuComplex *a, int lda, cuComplex *beta,
            Operation transb, cuComplex *b, int ldb);
  void geam(cuComplex *output, int ldo, int m, int n, cuComplex alpha,
            Operation transa, cuComplex *a, int lda, cuComplex beta,
            Operation transb, cuComplex *b, int ldb);

  void geam(cuDoubleComplex *output, int ldo, int m, int n,
            cuDoubleComplex *alpha, Operation transa, cuDoubleComplex *a,
            int lda, cuDoubleComplex *beta, Operation transb,
            cuDoubleComplex *b, int ldb);
  void geam(cuDoubleComplex *output, int ldo, int m, int n,
            cuDoubleComplex alpha, Operation transa, cuDoubleComplex *a,
            int lda, cuDoubleComplex beta, Operation transb, cuDoubleComplex *b,
            int ldb);

private:
  cublasHandle_t m_handle;
};
} // namespace mtrx

#endif
