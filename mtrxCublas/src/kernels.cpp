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

#include "kernels.hpp"

#include "driver_types.h"
#include "ikernel_executor.hpp"
#include "mtrxCublas/status_handler.hpp"
#include <memory>
#include <spdlog/details/os.h>
#include <spdlog/spdlog.h>
#include <sstream>

#ifndef MTRX_HOST_CUDA_BUILD
#include "device_kernel_executor.hpp"
#else
#include "host/host_kernel_executor.hpp"
#endif

namespace mtrx {

std::shared_ptr<mtrx::IKernelExecutor> GetKernelExecutor() {
#ifndef MTRX_HOST_CUDA_BUILD
  auto kernelExecutor = std::make_shared<mtrx::DeviceKernelExecutor>();
#else
  auto kernelExecutor = std::make_shared<mtrx::HostKernelExecutor>();
#endif
  return kernelExecutor;
}

template <typename T>
void Kernel_scaleTrace(const std::string &kernelName, int dim, T *matrix,
                       int lda, T factor) {
  auto ke = GetKernelExecutor();
  ke->setThreadsCount(dim, 1, 1);
  ke->setBlocksCount(1, 1, 1);

  void *params[] = {&dim, &dim, &matrix, &lda, &factor};
  ke->setParams(const_cast<const void **>(params));

  std::stringstream cukernelName;
  cukernelName << "CUDA" << kernelName;
  spdlog::info("Run kernel '{}'", cukernelName.str());
  ke->run(cukernelName.str());
}

void Kernel_SF_scaleTrace(int dim, float *matrix, int lda, float factor) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor);
}

void Kernel_SD_scaleTrace(int dim, double *matrix, int lda, double factor) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor);
}

void Kernel_CF_scaleTrace(int dim, cuComplex *matrix, int lda,
                          cuComplex factor) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor);
}

void Kernel_CD_scaleTrace(int dim, cuDoubleComplex *matrix, int lda,
                          cuDoubleComplex factor) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor);
}

template <typename T>
void Kernel_isULTriangular(const Alloc &alloc, bool &is,
                           const std::string &kernelName, int rows, int columns,
                           T *matrix, int lda, T delta) {
  auto ke = GetKernelExecutor();
  ke->setThreadsCount(rows, columns, 1);
  ke->setBlocksCount(1, 1, 1);

  // ToDo: d_is should be optimized
  bool *d_is = nullptr;
  void *dv_is = static_cast<void *>(d_is);
  alloc.malloc(&dv_is, sizeof(bool));

  void *params[] = {&d_is, &rows, &columns, &matrix, &lda, &delta};
  ke->setParams(const_cast<const void **>(params));

  std::stringstream cukernelName;
  cukernelName << "CUDA" << kernelName;
  spdlog::info("Run kernel '{}'", cukernelName.str());
  ke->run(cukernelName.str());

  void *h_is = static_cast<void *>(&is);
  alloc.memcpyKernelToHost(h_is, dv_is, sizeof(bool));
  alloc.free(d_is);
}

bool Kernel_SF_isUpperTriangular(const Alloc &alloc, int rows, int columns,
                                 float *matrix, int lda, float delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}

bool Kernel_SD_isUpperTriangular(const Alloc &alloc, int rows, int columns,
                                 double *matrix, int lda, double delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}

bool Kernel_CF_isUpperTriangular(const Alloc &alloc, int rows, int columns,
                                 cuComplex *matrix, int lda, cuComplex delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}

bool Kernel_CD_isUpperTriangular(const Alloc &alloc, int rows, int columns,
                                 cuDoubleComplex *matrix, int lda,
                                 cuDoubleComplex delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}

bool Kernel_SF_isLowerTriangular(const Alloc &alloc, int rows, int columns,
                                 float *matrix, int lda, float delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}

bool Kernel_SD_isLowerTriangular(const Alloc &alloc, int rows, int columns,
                                 double *matrix, int lda, double delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}

bool Kernel_CF_isLowerTriangular(const Alloc &alloc, int rows, int columns,
                                 cuComplex *matrix, int lda, cuComplex delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}

bool Kernel_CD_isLowerTriangular(const Alloc &alloc, int rows, int columns,
                                 cuDoubleComplex *matrix, int lda,
                                 cuDoubleComplex delta) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta);
  return is;
}
} // namespace mtrx
