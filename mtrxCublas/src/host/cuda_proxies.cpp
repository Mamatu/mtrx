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

#include "cuda_proxies.hpp"
#include "../cuda_is_ul_triangular.hpp"
#include "../cuda_reduce.hpp"
#include "../cuda_scale_diagonal.hpp"

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

template <typename T, typename Callback>
void proxy_HostKernel_generic_scaleDiagonal(const void **params,
                                            Callback &&callback) {
  int m = *static_cast<const int *>(params[0]);
  int n = *static_cast<const int *>(params[1]);
  T *matrix = getParam<T>(params[2]);
  int lda = *static_cast<const int *>(params[3]);
  T factor = *static_cast<const T *>(params[4]);

  callback(m, n, matrix, lda, factor);
}

void HostKernel_SF_scaleDiagonal(int m, int n, float *matrix, int lda,
                                 float factor) {
  cuda_SF_scaleDiagonal(m, n, matrix, lda, factor);
}

void proxy_HostKernel_SF_scaleDiagonal(const void **params) {
  proxy_HostKernel_generic_scaleDiagonal<float>(params,
                                                HostKernel_SF_scaleDiagonal);
}

void HostKernel_SD_scaleDiagonal(int m, int n, double *matrix, int lda,
                                 double factor) {
  cuda_SD_scaleDiagonal(m, n, matrix, lda, factor);
}

void proxy_HostKernel_SD_scaleDiagonal(const void **params) {
  proxy_HostKernel_generic_scaleDiagonal<double>(params,
                                                 HostKernel_SD_scaleDiagonal);
}

void HostKernel_CF_scaleDiagonal(int m, int n, cuComplex *matrix, int lda,
                                 cuComplex factor) {
  cuda_CF_scaleDiagonal(m, n, matrix, lda, factor);
}

void proxy_HostKernel_CF_scaleDiagonal(const void **params) {
  proxy_HostKernel_generic_scaleDiagonal<cuComplex>(
      params, HostKernel_CF_scaleDiagonal);
}

void HostKernel_CD_scaleDiagonal(int m, int n, cuDoubleComplex *matrix, int lda,
                                 cuDoubleComplex factor) {
  cuda_CD_scaleDiagonal(m, n, matrix, lda, factor);
}

void proxy_HostKernel_CD_scaleDiagonal(const void **params) {
  proxy_HostKernel_generic_scaleDiagonal<cuDoubleComplex>(
      params, HostKernel_CD_scaleDiagonal);
}

template <typename T, typename Callback>
void proxy_HostKernel_generic_isULTriangular(const void **params,
                                             Callback &&callback) {
  int m = *static_cast<const int *>(params[0]);
  int n = *static_cast<const int *>(params[1]);
  T *matrix = getParam<T>(params[2]);
  int lda = *static_cast<const int *>(params[3]);
  T delta = *static_cast<const T *>(params[4]);
  int *reductionResults = getParam<int>(params[5]);

  callback(m, n, matrix, lda, delta, reductionResults);
}

void proxy_HostKernel_SF_isUpperTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isUpperTriangular<float>(rows, columns, matrix, lda, delta,
                                  reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<float>(params, std::move(cuda_func));
}

void proxy_HostKernel_SD_isUpperTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isUpperTriangular<double>(rows, columns, matrix, lda, delta,
                                   reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<double>(params, std::move(cuda_func));
}

void proxy_HostKernel_CF_isUpperTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isUpperTriangular<cuComplex>(rows, columns, matrix, lda, delta,
                                      reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<cuComplex>(params,
                                                     std::move(cuda_func));
}

void proxy_HostKernel_CD_isUpperTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isUpperTriangular<cuDoubleComplex>(rows, columns, matrix, lda, delta,
                                            reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<cuDoubleComplex>(
      params, std::move(cuda_func));
}

void proxy_HostKernel_SF_isLowerTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isLowerTriangular<float>(rows, columns, matrix, lda, delta,
                                  reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<float>(params, std::move(cuda_func));
}

void proxy_HostKernel_SD_isLowerTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isLowerTriangular<double>(rows, columns, matrix, lda, delta,
                                   reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<double>(params, std::move(cuda_func));
}

void proxy_HostKernel_CF_isLowerTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isLowerTriangular<cuComplex>(rows, columns, matrix, lda, delta,
                                      reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<cuComplex>(params,
                                                     std::move(cuda_func));
}

void proxy_HostKernel_CD_isLowerTriangular(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *matrix, int lda, auto delta,
                      int *reductionResults) {
    cuda_isLowerTriangular<cuDoubleComplex>(rows, columns, matrix, lda, delta,
                                            reductionResults);
  };
  proxy_HostKernel_generic_isULTriangular<cuDoubleComplex>(
      params, std::move(cuda_func));
}

template <typename T, typename Callback>
void proxy_HostKernel_generic_reduceShm(const void **params,
                                        Callback &&callback) {
  int m = *static_cast<const int *>(params[0]);
  int n = *static_cast<const int *>(params[1]);
  T *array = getParam<T>(params[2]);
  int lda = *static_cast<const int *>(params[3]);
  T *reductionResults = getParam<T>(params[4]);

  callback(m, n, array, lda, reductionResults);
}

void proxy_HostKernel_SI_reduceShm(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *array, int lda,
                      auto *reductionResults) {
    cuda_reduce_shm<int>(rows, columns, array, lda, reductionResults);
  };
  proxy_HostKernel_generic_reduceShm<int>(params, std::move(cuda_func));
}

void proxy_HostKernel_SF_reduceShm(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *array, int lda,
                      auto *reductionResults) {
    cuda_reduce_shm<float>(rows, columns, array, lda, reductionResults);
  };
  proxy_HostKernel_generic_reduceShm<float>(params, std::move(cuda_func));
}

void proxy_HostKernel_SD_reduceShm(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *array, int lda,
                      auto *reductionResults) {
    cuda_reduce_shm<double>(rows, columns, array, lda, reductionResults);
  };
  proxy_HostKernel_generic_reduceShm<double>(params, std::move(cuda_func));
}

void proxy_HostKernel_CF_reduceShm(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *array, int lda,
                      auto *reductionResults) {
    cuda_reduce_shm<cuComplex>(rows, columns, array, lda, reductionResults);
  };
  proxy_HostKernel_generic_reduceShm<cuComplex>(params, std::move(cuda_func));
}

void proxy_HostKernel_CD_reduceShm(const void **params) {
  auto cuda_func = [](int rows, int columns, auto *array, int lda,
                      auto *reductionResults) {
    cuda_reduce_shm<cuDoubleComplex>(rows, columns, array, lda,
                                     reductionResults);
  };
  proxy_HostKernel_generic_reduceShm<cuDoubleComplex>(params,
                                                      std::move(cuda_func));
}
