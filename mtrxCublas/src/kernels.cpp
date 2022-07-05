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

#include <cstdlib>
#include <mtrxCore/checkers.hpp>
#include <mtrxCore/to_string.hpp>

#include "calc_dim.hpp"
#include "driver_types.h"
#include "ikernel_executor.hpp"
#include "mtrxCore/types.hpp"
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

#ifdef CUBLAS_NVPROF_KERNELS
#include "cuda_profiler.hpp"
#define PROFILER() Profiler p;
#else
#define PROFILER()
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
void Kernel_scaleDiagonal(const std::string &kernelName, int dim, T *matrix,
                          int lda, T factor, int device) {
  auto ke = GetKernelExecutor(device);
  const auto &dp = ke->getDeviceProperties();

  dim3 threads;
  dim3 blocks;
  calculateDim(threads, blocks, dim, 1, dp.blockDim, dp.gridDim,
               dp.maxThreadsPerBlock);

  ke->setThreadsCount(threads);
  ke->setBlocksCount(blocks);

  void *params[] = {&dim, &dim, &matrix, &lda, &factor};
  ke->setParams(const_cast<const void **>(params));

  std::stringstream cukernelName;
  cukernelName << "Cuda" << kernelName;
  spdlog::info("Run kernel '{}'", cukernelName.str());
  {
    PROFILER();
    ke->run(cukernelName.str());
  }
}

void Kernel_SF_scaleDiagonal(int dim, float *matrix, int lda, float factor,
                             int device) {
  Kernel_scaleDiagonal(__func__, dim, matrix, lda, factor, device);
}

void Kernel_SD_scaleDiagonal(int dim, double *matrix, int lda, double factor,
                             int device) {
  Kernel_scaleDiagonal(__func__, dim, matrix, lda, factor, device);
}

void Kernel_CF_scaleDiagonal(int dim, cuComplex *matrix, int lda,
                             cuComplex factor, int device) {
  Kernel_scaleDiagonal(__func__, dim, matrix, lda, factor, device);
}

void Kernel_CD_scaleDiagonal(int dim, cuDoubleComplex *matrix, int lda,
                             cuDoubleComplex factor, int device) {
  Kernel_scaleDiagonal(__func__, dim, matrix, lda, factor, device);
}

void Kernels::scaleDiagonal(int dim, float *matrix, int lda, float factor) {
  Kernel_SF_scaleDiagonal(dim, matrix, lda, factor, m_device);
}

void Kernels::scaleDiagonal(int dim, double *matrix, int lda, double factor) {
  Kernel_SD_scaleDiagonal(dim, matrix, lda, factor, m_device);
}

void Kernels::scaleDiagonal(int dim, cuComplex *matrix, int lda,
                            cuComplex factor) {
  Kernel_CF_scaleDiagonal(dim, matrix, lda, factor, m_device);
}

void Kernels::scaleDiagonal(int dim, cuDoubleComplex *matrix, int lda,
                            cuDoubleComplex factor) {
  Kernel_CD_scaleDiagonal(dim, matrix, lda, factor, m_device);
}

template <typename T>
void Kernel_isULTriangular(Alloc *alloc, bool &is,
                           const std::string &kernelName, int rows, int columns,
                           T *matrix, int lda, T delta, int device) {

  MTRX_CHECK_IF_NOT_NULL(alloc);

  auto ke = GetKernelExecutor(device);

  const auto &dp = ke->getDeviceProperties();

  dim3 threads;
  dim3 blocks;
  calculateDim(threads, blocks, rows, columns, dp.blockDim, dp.gridDim,
               dp.maxThreadsPerBlock);

  ke->setThreadsCount(threads);
  ke->setBlocksCount(blocks);
  int sharedMem = rows * columns * sizeof(int);

  if (sharedMem > dp.sharedMemPerBlock) {
    std::stringstream sstream;
    sstream << "Required shared memory (" << sharedMem
            << ") is higher than shared memory per block ("
            << dp.maxThreadsPerBlock << ")";
    throw std::runtime_error(sstream.str());
  }

  ke->setSharedMemory(sharedMem);
  auto blocksCount = blocks.x * blocks.y;

  int *d_reductionResults = nullptr;
  alloc->malloc(reinterpret_cast<void **>(&d_reductionResults),
                sizeof(int) * blocksCount);

  void *params[] = {&rows, &columns, &matrix,
                    &lda,  &delta,   &d_reductionResults};
  ke->setParams(const_cast<const void **>(params));

  std::stringstream cukernelName;
  cukernelName << "Cuda" << kernelName;
  spdlog::info("Run kernel '{}'", cukernelName.str());
  {
    PROFILER()
    ke->run(cukernelName.str());
  }
  std::vector<int> h_reductionResults(blocksCount, 0);
  alloc->memcpyKernelToHost(h_reductionResults.data(), d_reductionResults,
                            sizeof(int) * blocksCount);
  alloc->free(d_reductionResults);
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

template <typename T,
          typename BinaryOperation = std::function<T(const T &, const T &)>>
T Kernel_reduceShm(
    Alloc *alloc, const std::string &kernelName, int m, int n, T *array,
    int lda, AccumulationMode mode, CUdevice device,
    BinaryOperation &&boper = [](const T &t1, const T &t2) {
      return t1 + t2;
    }) {
  auto ke = GetKernelExecutor(device);
  const auto &dp = ke->getDeviceProperties();

  dim3 threads;
  dim3 blocks;
  calculateDim(threads, blocks, m, n, dp.blockDim, dp.gridDim,
               dp.maxThreadsPerBlock);

  auto threadsCount = threads.x * threads.y;
  int sharedMem = threadsCount * sizeof(T);

  if (sharedMem > dp.sharedMemPerBlock) {
    std::stringstream sstream;
    sstream << "Required shared memory (" << sharedMem
            << ") is higher than shared memory per block ("
            << dp.maxThreadsPerBlock << ")";
    throw std::runtime_error(sstream.str());
  }

  auto blocksCount = blocks.x * blocks.y;
  std::vector<T> h_reductionResults;
  h_reductionResults.resize(blocksCount);

  T *d_reductionResults = nullptr;
  alloc->malloc(reinterpret_cast<void **>(&d_reductionResults),
                blocksCount * sizeof(T));

  ke->setThreadsCount(threads);
  ke->setBlocksCount(blocks);
  ke->setSharedMemory(sharedMem);

  void *params[] = {&m, &n, &array, &lda, &d_reductionResults, &mode};
  ke->setParams(const_cast<const void **>(params));

  std::stringstream cukernelName;
  cukernelName << "Cuda" << kernelName;
  spdlog::info("Run kernel '{}'", cukernelName.str());
  {
    PROFILER()
    ke->run(cukernelName.str());
  }
  alloc->memcpyKernelToHost(h_reductionResults.data(), d_reductionResults,
                            sizeof(T) * blocksCount);
  alloc->free(d_reductionResults);

  return std::accumulate(h_reductionResults.begin(), h_reductionResults.end(),
                         T(), std::forward<BinaryOperation>(boper));
}

int Kernels::reduceShm(int m, int n, int *array, int lda,
                       AccumulationMode mode) {
  return Kernel_reduceShm<int>(m_alloc, "Kernel_SI_reduceShm", m, n, array, lda,
                               mode, m_device);
}

float Kernels::reduceShm(int m, int n, float *array, int lda,
                         AccumulationMode mode) {
  return Kernel_reduceShm<float>(m_alloc, "Kernel_SF_reduceShm", m, n, array,
                                 lda, mode, m_device);
}

double Kernels::reduceShm(int m, int n, double *array, int lda,
                          AccumulationMode mode) {
  return Kernel_reduceShm<double>(m_alloc, "Kernel_SD_reduceShm", m, n, array,
                                  lda, mode, m_device);
}

cuComplex Kernels::reduceShm(int m, int n, cuComplex *array, int lda,
                             AccumulationMode mode) {
  return Kernel_reduceShm<cuComplex>(m_alloc, "Kernel_CF_reduceShm", m, n,
                                     array, lda, mode, m_device, cuCaddf);
}

cuDoubleComplex Kernels::reduceShm(int m, int n, cuDoubleComplex *array,
                                   int lda, AccumulationMode mode) {
  return Kernel_reduceShm<cuDoubleComplex>(m_alloc, "Kernel_CD_reduceShm", m, n,
                                           array, lda, mode, m_device, cuCadd);
}

bool Kernels::isUnit(int m, int n, float *matrix, int lda, float delta) {
  auto sum = reduceShm(m, n, matrix, lda, AccumulationMode::POWER_OF_2);
  return abs(sum - 1) < delta;
}

bool Kernels::isUnit(int m, int n, double *matrix, int lda, double delta) {
  auto sum = reduceShm(m, n, matrix, lda, AccumulationMode::POWER_OF_2);
  return abs(sum - 1) < delta;
}

bool Kernels::isUnit(int m, int n, cuComplex *matrix, int lda, float delta) {
  auto sum = reduceShm(m, n, matrix, lda, AccumulationMode::POWER_OF_2);
  return abs(sum.x - sum.y - 1) < delta;
}

bool Kernels::isUnit(int m, int n, cuDoubleComplex *matrix, int lda,
                     double delta) {
  auto sum = reduceShm(m, n, matrix, lda, AccumulationMode::POWER_OF_2);
  return abs(sum.x - sum.y - 1) < delta;
}

} // namespace mtrx
