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

#include <mtrxCore/checkers.hpp>
#include <mtrxCore/to_string.hpp>

#include "calc_dim.hpp"
#include "driver_types.h"
#include "ikernel_executor.hpp"
#include "mtrxCublas/status_handler.hpp"
#include <memory>
#include <numeric>
#include <spdlog/details/os.h>
#include <spdlog/spdlog.h>
#include <sstream>

#ifndef MTRX_HOST_CUDA_BUILD
#include "device_kernel_executor.hpp"
#else
#include "host/device_properties_provider.hpp"
#include "host/host_kernel_executor.hpp"
#endif

namespace mtrx {

Kernels::Kernels(CUdevice device, Alloc *alloc)
    : m_device(device), m_alloc(alloc) {}

std::shared_ptr<mtrx::IKernelExecutor> GetKernelExecutor(int device) {
// ToDo: one pointer during whole session
#ifndef MTRX_HOST_CUDA_BUILD
  auto kernelExecutor = std::make_shared<mtrx::DeviceKernelExecutor>(device);
#else
  auto kernelExecutor = std::make_shared<mtrx::HostKernelExecutor>(device);
#endif
  return kernelExecutor;
}

template <typename T>
void Kernel_scaleTrace(const std::string &kernelName, int dim, T *matrix,
                       int lda, T factor, int device) {
  auto ke = GetKernelExecutor(device);
  const auto &dp = ke->getDeviceProperties();

  std::array<int, 2> threads;
  std::array<int, 2> blocks;
  calculateDim(threads, blocks, dim, 1, dp.blockDim, dp.gridDim,
               dp.maxThreadsPerBlock);

  ke->setThreadsCount(threads);
  ke->setBlocksCount(blocks);

  void *params[] = {&dim, &dim, &matrix, &lda, &factor};
  ke->setParams(const_cast<const void **>(params));

  std::stringstream cukernelName;
  cukernelName << "CUDA" << kernelName;
  spdlog::info("Run kernel '{}'", cukernelName.str());
  ke->run(cukernelName.str());
}

void Kernel_SF_scaleTrace(int dim, float *matrix, int lda, float factor,
                          int device) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor, device);
}

void Kernel_SD_scaleTrace(int dim, double *matrix, int lda, double factor,
                          int device) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor, device);
}

void Kernel_CF_scaleTrace(int dim, cuComplex *matrix, int lda, cuComplex factor,
                          int device) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor, device);
}

void Kernel_CD_scaleTrace(int dim, cuDoubleComplex *matrix, int lda,
                          cuDoubleComplex factor, int device) {
  Kernel_scaleTrace(__func__, dim, matrix, lda, factor, device);
}

void Kernels::scaleTrace(int dim, float *matrix, int lda, float factor) {
  Kernel_SF_scaleTrace(dim, matrix, lda, factor, m_device);
}

void Kernels::scaleTrace(int dim, double *matrix, int lda, double factor) {
  Kernel_SD_scaleTrace(dim, matrix, lda, factor, m_device);
}

void Kernels::scaleTrace(int dim, cuComplex *matrix, int lda,
                         cuComplex factor) {
  Kernel_CF_scaleTrace(dim, matrix, lda, factor, m_device);
}

void Kernels::scaleTrace(int dim, cuDoubleComplex *matrix, int lda,
                         cuDoubleComplex factor) {
  Kernel_CD_scaleTrace(dim, matrix, lda, factor, m_device);
}

template <typename T>
void Kernel_isULTriangular(Alloc *alloc, bool &is,
                           const std::string &kernelName, int rows, int columns,
                           T *matrix, int lda, T delta, int device) {

  MTRX_CHECK_IF_NOT_NULL(alloc);

  auto ke = GetKernelExecutor(device);

  const auto &dp = ke->getDeviceProperties();

  std::array<int, 2> threads;
  std::array<int, 2> blocks;
  calculateDim(threads, blocks, rows, columns, dp.blockDim, dp.gridDim,
               dp.maxThreadsPerBlock);

  ke->setThreadsCount(threads);
  ke->setBlocksCount(blocks);
  int sharedMem = rows * columns * sizeof(int);

  if (sharedMem > dp.maxThreadsPerBlock) {
    std::stringstream sstream;
    sstream << "Required shared memory (" << sharedMem
            << ") is higher than max threads per block ("
            << dp.maxThreadsPerBlock << ")";
    throw std::runtime_error(sstream.str());
  }

  ke->setSharedMemory(sharedMem);
  auto blocksCount = blocks[0] * blocks[1];

  int *d_reductionResults = nullptr;
  void *dv_reductionResults = static_cast<void *>(d_reductionResults);
  alloc->malloc(&dv_reductionResults, sizeof(int) * blocksCount);

  void *params[] = {&rows, &columns, &matrix,
                    &lda,  &delta,   &dv_reductionResults};
  ke->setParams(const_cast<const void **>(params));

  std::stringstream cukernelName;
  cukernelName << "CUDA" << kernelName;
  spdlog::info("Run kernel '{}'", cukernelName.str());
  ke->run(cukernelName.str());

  std::vector<int> h_reductionResults(blocksCount, 0);
  alloc->memcpyKernelToHost(h_reductionResults.data(), dv_reductionResults,
                            sizeof(int) * blocksCount);
  alloc->free(dv_reductionResults);
  int rr =
      std::accumulate(h_reductionResults.begin(), h_reductionResults.end(), 0);

  is = (rr == rows * columns);
}

bool Kernel_SF_isUpperTriangular(Alloc *alloc, int rows, int columns,
                                 float *matrix, int lda, float delta,
                                 int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernel_SD_isUpperTriangular(Alloc *alloc, int rows, int columns,
                                 double *matrix, int lda, double delta,
                                 int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernel_CF_isUpperTriangular(Alloc *alloc, int rows, int columns,
                                 cuComplex *matrix, int lda, cuComplex delta,
                                 int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernel_CD_isUpperTriangular(Alloc *alloc, int rows, int columns,
                                 cuDoubleComplex *matrix, int lda,
                                 cuDoubleComplex delta, int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernel_SF_isLowerTriangular(Alloc *alloc, int rows, int columns,
                                 float *matrix, int lda, float delta,
                                 int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernel_SD_isLowerTriangular(Alloc *alloc, int rows, int columns,
                                 double *matrix, int lda, double delta,
                                 int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernel_CF_isLowerTriangular(Alloc *alloc, int rows, int columns,
                                 cuComplex *matrix, int lda, cuComplex delta,
                                 int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernel_CD_isLowerTriangular(Alloc *alloc, int rows, int columns,
                                 cuDoubleComplex *matrix, int lda,
                                 cuDoubleComplex delta, int device) {
  bool is = false;
  Kernel_isULTriangular(alloc, is, __func__, rows, columns, matrix, lda, delta,
                        device);
  return is;
}

bool Kernels::isUpperTriangular(int rows, int columns, float *matrix, int lda,
                                float delta) {
  return Kernel_SF_isUpperTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}

bool Kernels::isUpperTriangular(int rows, int columns, double *matrix, int lda,
                                double delta) {
  return Kernel_SD_isUpperTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}

bool Kernels::isUpperTriangular(int rows, int columns, cuComplex *matrix,
                                int lda, cuComplex delta) {
  return Kernel_CF_isUpperTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}

bool Kernels::isUpperTriangular(int rows, int columns, cuDoubleComplex *matrix,
                                int lda, cuDoubleComplex delta) {
  return Kernel_CD_isUpperTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}

bool Kernels::isLowerTriangular(int rows, int columns, float *matrix, int lda,
                                float delta) {
  return Kernel_SF_isLowerTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}

bool Kernels::isLowerTriangular(int rows, int columns, double *matrix, int lda,
                                double delta) {
  return Kernel_SD_isLowerTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}

bool Kernels::isLowerTriangular(int rows, int columns, cuComplex *matrix,
                                int lda, cuComplex delta) {
  return Kernel_CF_isLowerTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}

bool Kernels::isLowerTriangular(int rows, int columns, cuDoubleComplex *matrix,
                                int lda, cuDoubleComplex delta) {
  return Kernel_CD_isLowerTriangular(m_alloc, rows, columns, matrix, lda, delta,
                                     m_device);
}
} // namespace mtrx
