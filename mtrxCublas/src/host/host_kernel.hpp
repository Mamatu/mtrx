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

#ifndef MTRX_CUBLAS_HOST_KERNEL_H
#define MTRX_CUBLAS_HOST_KERNEL_H

#include <map>
#include <mtrxCore/types.hpp>
#include <mutex>

//#include "dim3.hpp"
//#include "IKernelExecutor.hpp"
#include "host_kernel_thread.hpp"

namespace mtrx {
class ThreadImpl;
class HostKernel;

class HostKernel {
public:
  HostKernel();
  HostKernel(void *ctx, bool releaseThreads);

  virtual ~HostKernel();
  static void ReleaseThreads(void *ctx);

  void setDims(const dim3 &gridDim, const dim3 &blockDim);

  void calculateDims(uintt columns, uintt rows);

  void setSharedMemory(size_t sizeInBytes);

  void executeKernelAsync();

  void executeKernelSync();

protected:
  virtual void execute(const dim3 &threadIdx, const dim3 &blockIdx) = 0;

  enum ContextChange { CUDA_THREAD, CUDA_BLOCK };

  virtual void onChange(ContextChange /*contextChnage*/,
                        const dim3 & /*threadIdx*/, const dim3 & /*blockIdx*/) {
    // empty
  }

  virtual void onSetDims(const dim3 & /*gridDim*/, const dim3 & /*blockDim*/) {
    // empty
  }

  dim3 gridDim;
  dim3 blockDim;
  size_t m_sharedMemorySize = 0;

private:
  using ThreadsPool =
      std::map<std::pair<uintt, uintt>, mtrx::HostKernelThread *>;
  static std::map<void *, ThreadsPool> s_threads;
  static std::mutex s_mutex;
  void *m_ctx;
  bool m_release = false;
};
} // namespace mtrx
#endif
