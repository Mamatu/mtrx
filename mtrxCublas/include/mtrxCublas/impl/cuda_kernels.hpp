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

#ifndef MTRX_CUBLAS_CUDA_KERNELS_H
#define MTRX_CUBLAS_CUDA_KERNELS_H

#include <mtrxCore/types.hpp>

#include "alloc.hpp"
#include <cuComplex.h>

namespace mtrx {

class CudaKernels final {
public:
  CudaKernels(int device, Alloc *alloc);
  ~CudaKernels() = default;

  void scaleDiagonal(int dim, float *matrix, int lda, float factor);
  void scaleDiagonal(int dim, double *matrix, int lda, double factor);
  void scaleDiagonal(int dim, cuComplex *matrix, int lda, cuComplex factor);
  void scaleDiagonal(int dim, cuDoubleComplex *matrix, int lda,
                     cuDoubleComplex factor);
  void scaleDiagonal(int dim, cuComplex *matrix, int lda, float factor);
  void scaleDiagonal(int dim, cuDoubleComplex *matrix, int lda, double factor);

  void scaleDiagonal(int dim, float *matrix, int lda, float *factor);
  void scaleDiagonal(int dim, double *matrix, int lda, double *factor);
  void scaleDiagonal(int dim, cuComplex *matrix, int lda, cuComplex *factor);
  void scaleDiagonal(int dim, cuDoubleComplex *matrix, int lda,
                     cuDoubleComplex *factor);

  bool isUpperTriangular(int rows, int columns, float *matrix, int lda,
                         float delta);
  bool isUpperTriangular(int rows, int columns, double *matrix, int lda,
                         double delta);
  bool isUpperTriangular(int rows, int columns, cuComplex *matrix, int lda,
                         cuComplex delta);
  bool isUpperTriangular(int rows, int columns, cuDoubleComplex *matrix,
                         int lda, cuDoubleComplex delta);

  bool isLowerTriangular(int rows, int columns, float *matrix, int lda,
                         float delta);
  bool isLowerTriangular(int rows, int columns, double *matrix, int lda,
                         double delta);
  bool isLowerTriangular(int rows, int columns, cuComplex *matrix, int lda,
                         cuComplex delta);
  bool isLowerTriangular(int rows, int columns, cuDoubleComplex *matrix,
                         int lda, cuDoubleComplex delta);

  /**
   * @brief Reduction by using shared memory.
   */
  int reduceShm(int m, int n, int *array, int lda,
                AccumulationMode mode = AccumulationMode::NORMAL);

  /**
   * @brief Reduction by using shared memory.
   */
  float reduceShm(int m, int n, float *array, int lda,
                  AccumulationMode mode = AccumulationMode::NORMAL);

  /**
   * @brief Reduction by using shared memory.
   */
  double reduceShm(int m, int n, double *array, int lda,
                   AccumulationMode mode = AccumulationMode::NORMAL);

  /**
   * @brief Reduction by using shared memory.
   */
  cuComplex reduceShm(int m, int n, cuComplex *array, int lda,
                      AccumulationMode mode = AccumulationMode::NORMAL);

  /**
   * @brief Reduction by using shared memory.
   */
  cuDoubleComplex reduceShm(int m, int n, cuDoubleComplex *array, int lda,
                            AccumulationMode mode = AccumulationMode::NORMAL);

  bool isUnit(int m, int n, float *matrix, int lda, float delta);

  bool isUnit(int m, int n, double *matrix, int lda, double delta);

  bool isUnit(int m, int n, cuComplex *matrix, int lda, float delta);

  bool isUnit(int m, int n, cuDoubleComplex *matrix, int lda, double delta);

private:
  int m_device;
  Alloc *m_alloc;
};
} // namespace mtrx

#endif
