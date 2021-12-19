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

#ifndef MTRX_CUBLAS_HOST_KERNEL_THREAD_H
#define MTRX_CUBLAS_HOST_KERNEL_THREAD_H

#include "barrier.hpp"
#include <vector_types.h>
//#include "dim3.hpp"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mtrx {
class HostKernelThread {
public:
  using ThreadFunc = std::function<void(dim3, dim3)>;

  HostKernelThread();

  void setThreadFunc(ThreadFunc &&threadFunc);
  void setThreadFunc(const ThreadFunc &threadFunc);

  void setBlockDim(const dim3 &blockDim);

  void setGridDim(const dim3 &gridDim);

  void setThreadIdx(const dim3 &threadIdx);

  void setBlockIdx(const dim3 &blockIdx);

  dim3 getThreadIdx() const;

  dim3 getBlockIdx() const;

  void setPthreads(std::vector<std::thread::id> *pthreads);

  void setBarrier(mtrx::Barrier *barrier);

  void setSharedBuffer(void *buffer);

  virtual ~HostKernelThread();

  void waitOn();

  void run();

  std::thread::id get_id() const;

  // void stop();
protected:
  static void Execute(HostKernelThread *hkt);
  virtual void onRun(std::thread::id threadId);

private:
  dim3 m_threadIdx;
  dim3 m_blockIdx;
  std::vector<std::thread::id> *m_pthreads;
  mtrx::Barrier *m_barrier;
  std::condition_variable m_cond;
  std::mutex m_mutex;
  bool m_cancontinue;
  void *m_sharedBuffer;

  dim3 m_gridDim;
  dim3 m_blockDim;

  std::shared_ptr<std::thread> m_thread;
  ThreadFunc m_threadFunc;
};
} // namespace mtrx
#endif
