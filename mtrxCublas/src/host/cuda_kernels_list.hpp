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

#ifndef MTRX_CUBLAS_CUDA_KERNELS_LIST_H
#define MTRX_CUBLAS_CUDA_KERNELS_LIST_H

#include "../cuda_scale_trace.hpp"

template <typename T> T *getParam(const void *param) {
  return *static_cast<T *const *>(param);
}

template <typename T> T *getParam(const void **params, size_t index) {
  return getParam<T>(params[index]);
}

template <typename T> T *getParam(void *param) {
  return *static_cast<T **>(param);
}

template <typename T> T *getParam(void **params, size_t index) {
  return getParam<T>(params[index]);
}

#define DEFINE_1M(arg_hostKernel, arg_cudaKernel)                              \
  void arg_hostKernel(math::ComplexMatrix *output) { arg_cudaKernel(output); } \
                                                                               \
  void proxy_##arg_hostKernel(const void **params) {                           \
    math::ComplexMatrix *output = getParam<math::ComplexMatrix>(params[0]);    \
    arg_hostKernel(output);                                                    \
  }

#define DEFINE_1M_EX(arg_hostKernel, arg_cudaKernel)                           \
  void arg_hostKernel(math::ComplexMatrix *output, uintt *ex) {                \
    arg_cudaKernel(output, ex);                                                \
  }                                                                            \
                                                                               \
  void proxy_##arg_hostKernel(const void **params) {                           \
    math::ComplexMatrix *output = getParam<math::ComplexMatrix>(params[0]);    \
    uintt *ex = getParam<uintt>(params[1]);                                    \
    arg_hostKernel(output, ex);                                                \
  }

#define DEFINE_2M(arg_hostKernel, arg_cudaKernel)                              \
  void arg_hostKernel(math::ComplexMatrix *output,                             \
                      math::ComplexMatrix *param1) {                           \
    arg_cudaKernel(output, param1);                                            \
  }                                                                            \
                                                                               \
  void proxy_##arg_hostKernel(const void **params) {                           \
    math::ComplexMatrix *output = getParam<math::ComplexMatrix>(params[0]);    \
    math::ComplexMatrix *matrix = getParam<math::ComplexMatrix>(params[1]);    \
    arg_hostKernel(output, matrix);                                            \
  }

#define DEFINE_2M_EX(arg_hostKernel, arg_cudaKernel)                           \
  void arg_hostKernel(math::ComplexMatrix *output,                             \
                      math::ComplexMatrix *param1, uintt *ex) {                \
    arg_cudaKernel(output, param1, ex);                                        \
  }                                                                            \
                                                                               \
  void proxy_##arg_hostKernel(const void **params) {                           \
    math::ComplexMatrix *output = getParam<math::ComplexMatrix>(params[0]);    \
    math::ComplexMatrix *matrix = getParam<math::ComplexMatrix>(params[1]);    \
    uintt *ex = getParam<uintt>(params[2]);                                    \
    arg_hostKernel(output, matrix, ex);                                        \
  }

#define DEFINE_3M(arg_hostKernel, arg_cudaKernel)                              \
  void arg_hostKernel(math::ComplexMatrix *output,                             \
                      math::ComplexMatrix *param1,                             \
                      math::ComplexMatrix *param2) {                           \
    arg_cudaKernel(output, param1, param2);                                    \
  }                                                                            \
                                                                               \
  void proxy_##arg_hostKernel(const void **params) {                           \
    math::ComplexMatrix *output = getParam<math::ComplexMatrix>(params[0]);    \
    math::ComplexMatrix *matrix = getParam<math::ComplexMatrix>(params[1]);    \
    math::ComplexMatrix *matrix1 = getParam<math::ComplexMatrix>(params[2]);   \
    arg_hostKernel(output, matrix, matrix1);                                   \
  }

#define DEFINE_3M_EX(arg_hostKernel, arg_cudaKernel)                           \
  void arg_hostKernel(math::ComplexMatrix *output,                             \
                      math::ComplexMatrix *param1,                             \
                      math::ComplexMatrix *param2, uintt *ex) {                \
    arg_cudaKernel(output, param1, param2, ex);                                \
  }                                                                            \
                                                                               \
  void proxy_##arg_hostKernel(const void **params) {                           \
    math::ComplexMatrix *output = getParam<math::ComplexMatrix>(params[0]);    \
    math::ComplexMatrix *matrix = getParam<math::ComplexMatrix>(params[1]);    \
    math::ComplexMatrix *matrix1 = getParam<math::ComplexMatrix>(params[2]);   \
    uintt *ex = getParam<uintt>(params[3]);                                    \
    arg_hostKernel(output, matrix, matrix1, ex);                               \
  }

void HOSTKernel_SF_scaleTrace(int m, int n, float *matrix, int lda,
                              float factor) {
  cuda_SF_scaleTrace(m, n, matrix, lda, factor);
}

template <typename T, typename Callback>
void proxy_HOSTKernel_generic_scaleTrace(const void **params,
                                         Callback &&callback) {
  int m = *static_cast<const int *>(params[0]);
  int n = *static_cast<const int *>(params[1]);
  T *matrix = getParam<T>(params[2]);
  int lda = *static_cast<const int *>(params[3]);
  T factor = *static_cast<const T *>(params[4]);

  callback(m, n, matrix, lda, factor);
}

void proxy_HOSTKernel_SF_scaleTrace(const void **params) {
  proxy_HOSTKernel_generic_scaleTrace<float>(params, HOSTKernel_SF_scaleTrace);
}

void HOSTKernel_SD_scaleTrace(int m, int n, double *matrix, int lda,
                              double factor) {
  cuda_SD_scaleTrace(m, n, matrix, lda, factor);
}

void proxy_HOSTKernel_SD_scaleTrace(const void **params) {
  proxy_HOSTKernel_generic_scaleTrace<double>(params, HOSTKernel_SD_scaleTrace);
}

void HOSTKernel_CF_scaleTrace(int m, int n, cuComplex *matrix, int lda,
                              cuComplex factor) {
  cuda_CF_scaleTrace(m, n, matrix, lda, factor);
}

void proxy_HOSTKernel_CF_scaleTrace(const void **params) {
  proxy_HOSTKernel_generic_scaleTrace<cuComplex>(params,
                                                 HOSTKernel_CF_scaleTrace);
}

void HOSTKernel_CD_scaleTrace(int m, int n, cuDoubleComplex *matrix, int lda,
                              cuDoubleComplex factor) {
  cuda_CD_scaleTrace(m, n, matrix, lda, factor);
}

void proxy_HOSTKernel_CD_scaleTrace(const void **params) {
  proxy_HOSTKernel_generic_scaleTrace<cuDoubleComplex>(
      params, HOSTKernel_CD_scaleTrace);
}

#endif
